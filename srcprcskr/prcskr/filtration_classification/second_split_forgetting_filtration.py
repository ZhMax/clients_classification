from pathlib import Path


from typing import Any, Dict, List, Union, Tuple

import pandas as pd
import numpy as np

from prcskr.data_loaders.datautils import create_datasets
from prcskr.models.due.model_due import start_training, get_predictions 

from prcskr.forgetting.forgettingutils import (
    get_forget_forever_epochs_for_examples
)

def _make_key_extractor(key):
    """Return max key containing in name of file"""
    def key_extractor(p: Path) -> float:
        metrics = {}
        for it in p.stem.split(" - "):
            kv = it.split(": ")
            assert len(kv) == 2, f"Failed to parse filename: {p.name}"
            k = kv[0]
            v = -float(kv[1]) if "loss" in k else float(kv[1])
            metrics[k] = v
        return metrics[key]

    return key_extractor

def filtration_of_dataset_by_second_split_forgetting(
    log_dir: str,
    ckpt_dir: str,
    features_path: str,
    targets_path: str,
    random_state: int,
    total_epochs_per_step: int,
    lr: float,
    dataconf: Dict[str, Any],
    modelconf: Dict[str, Any],
    example_forgetting_dir: str = None,
    threshold_val: int = None,
    ckpt_resume: str = None,
    ckpt_resume_step_idx: int = None,
    path_to_file_names_to_be_excluded: str = None
):
    """
    Find examples of a dataset, that were forgotten after continuing to train 
    a model on the new half part of the dataset.

    Args:
        log_dir: str
            Path to directory for saving logs
        
        ckpt_dir: str
            Path to directory for saving checkpoints

        features_path: str
            Path to a directory which contains files with features.

        targets_path: str
            Path to a directory which contains files with true labels.
            It is supposed that the files containing features and true label 
            for an example from the dataset have the same name.

        random_state: int
            To provide reproducibility of computations.
        
        total_epochs_per_step: int
            A number of epochs for one step of model training.

        lr: float = None
            Learning rate in an optimizer.

        dataconf: Dict[str, Any]
            Data config

        modelconf: Dict[str, Any]
            Model config

        example_forgetting_dir: str = None
            Path to a directory which will be used to save array with file names 
            containing noisy labels. If it is `None`, then the directory with name
            `f"{data_name}_second_forgetting"` will be created in the parent of the directory
            `data_filepath`. The field `data_name` is included in the data configuration
            file.
        
        threshold_val: int = None
            Threshold value for `epoch_forget_forever`, which can be used to filtrate examples.
            If it is `None`, only examples which are forgotten after one epoch of the next 
            training step will be proposed for excluding from the dataset. 

        ckpt_resume: str = None
            Path to a checkpoint file `*.ckpt` which is used to load the model.
            It should be `None` to train a model from an initial state.

        ckpt_resume_step_idx: int = None
            Index of a training step for which the model is loaded from `*.ckpt`
            file.

        path_to_file_names_to_be_excluded: str
            Path to a `.txt` file which contains names of files 
            to be excluded from the original dataset for training.

        Returns:
            df_examples: pd.DataFrame
                pd.DataFrame containing the number of epoch when each examples of 
                the dataset were forgotten forever and predictions given by the trained 
                model at the second and fourth training steps.
    """


    if ckpt_resume is None:
        #If the method starts from the initial state, a training step is equal 1
        # and the total number of epochs for model training is equal to
        # the number of epochs of training step
        step_idx_beginning = 1 
        total_epochs = total_epochs_per_step
    else:
        #If the method resumes from a checkpoint, a training step is equal to the step,
        # when the model have been saved. The total number of epochs for model trainig is
        # depended on the number of the training step
        step_idx_beginning = ckpt_resume_step_idx 
        total_epochs = total_epochs_per_step
        total_epochs += total_epochs_per_step * ((step_idx_beginning - 1) % 2)

    #run_names for saving checkpoints and logs 
    run_names = ['second_forgetting_step' + str(i) for i in range(1, 5)]

    #Create two train dataset by splitting the original dataset into two equal parts 
    train_dataset_pt1, train_dataset_pt2 = create_datasets(
        features_path=features_path,
        targets_path=targets_path,
        random_state=random_state,
        features_dim=dataconf['features_dim'],
        path_to_file_names_to_be_excluded=path_to_file_names_to_be_excluded,
        split_fraction=None,
        mode='second-split-forgetting'
    )

    #Train datasets for the training steps
    train_datasets = [
        train_dataset_pt1, train_dataset_pt2, 
        train_dataset_pt2, train_dataset_pt1
    ]


    #Train the model and collect masks to compute  
    #forgetting values of examples
    for step_idx in range(step_idx_beginning, 5):
        run_name = run_names[step_idx - 1]       
        trainer_pt = start_training(
            run_name=run_name,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
            ckpt_resume=ckpt_resume,
            train_dataset=train_datasets[step_idx - 1],
            val_dataset=train_datasets[step_idx - 1],
            random_state=random_state,
            total_epochs=total_epochs,
            lr=lr,
            metrics_on_train=False,
            is_forgetting=True,
            dataconf=dataconf,
            modelconf=modelconf,
        )

        if (step_idx - 1) % 2 == 0:
            #If idx of a training step is 1 or 3, at the next step 
            #the model training will be continued 
            ckpt_resume = trainer_pt.get_best_or_last_ckpt('last')
            total_epochs += total_epochs_per_step
        else:
            #If idx of a training step is 2, at the next step
            #the model will be trained from the initial state
            ckpt_resume = None
            total_epochs = total_epochs_per_step


    #Compute predictions of the model saved at the 4 and 2 training steps on
    #train_dataset_pt1, train_dataset_pt2
    preds_labels = []
    preds_proba = []
    uncertainties = []
    dataset_inds_loader = []
    dataset_file_names_loader = []
    gts = []

    #Find examples from train_dataset_pt1, train_dataset_pt2,
    #which were forgotten after one epoch of 4 and 2 training steps 
    dataset_file_names = []
    examples_forget_epochs = []



    for step_idx in [4, 2]:

        ckpt_resume_dir = Path(ckpt_dir) / run_names[step_idx - 1]
        ckpt_resume_dir = Path(ckpt_resume_dir)

        all_ckpt = ckpt_resume_dir.glob('*.ckpt')
        ckpt_resume = max(all_ckpt, key=_make_key_extractor("epoch"))

        preds_pt, uncertainties_pt, dataset_inds_pt, file_names_pt, gts_pt, trainer_pt = \
            get_predictions(    
                run_name=run_names[step_idx - 1],
                ckpt_resume=ckpt_resume,
                random_state=random_state,
                dataset=train_datasets[step_idx - 1],
                dataconf=dataconf,
                modelconf=modelconf,
                is_forgetting=True
            )
        
        
        preds_pt_label = np.argmax(preds_pt, axis=1)
        preds_proba_max_pt_label = np.max(preds_pt, axis=1)

        ret_ft_pt = trainer_pt._forgetting_dict[run_names[step_idx-1]]
        forget_epochs_pt, file_names_pt_forget = \
            get_forget_forever_epochs_for_examples(ret_ft_pt)


        preds_labels.append(preds_pt_label)
        preds_proba.append(preds_proba_max_pt_label)
        uncertainties.append(uncertainties_pt)
        dataset_inds_loader.append(dataset_inds_pt)
        dataset_file_names_loader.append(file_names_pt)
        gts.append(gts_pt)

        examples_forget_epochs.append(forget_epochs_pt)
        dataset_file_names.append(file_names_pt_forget)

    #Combine arrays from the training steps in one pd.DataFrame
    preds_labels = np.hstack(preds_labels)
    preds_proba = np.hstack(preds_proba)
    uncertainties = np.hstack(uncertainties)
    dataset_inds_loader = np.hstack(dataset_inds_loader)
    dataset_file_names_loader = np.hstack(dataset_file_names_loader)  
    gts = np.hstack(gts)   

    ar_predicted = np.vstack([
        dataset_inds_loader, 
        dataset_file_names_loader,
        preds_labels, preds_proba, 
        uncertainties, gts
    ]).T

    df_predicted = pd.DataFrame(
        data=ar_predicted, 
        columns=['example_idx', 'file_name_loader', 
                    'pred_label', 'pred_proba', 
                    'uncertainty', 'true_label'])

    examples_forget_epochs = np.hstack(examples_forget_epochs)
    dataset_file_names = np.hstack(dataset_file_names)
    
    ar_forget = np.vstack([
        dataset_file_names, examples_forget_epochs]).T
    
    df_forget = pd.DataFrame(
        data=ar_forget, 
        columns=['dataset_file_name', 'epoch_forget_forever'])
    
    df_examples = pd.merge(
        df_predicted, 
        df_forget,
        left_on='file_name_loader',
        right_on='dataset_file_name',
        how='left')

    df_examples.drop(columns='file_name_loader', inplace=True)

    float_cols = ['pred_proba', 'pred_label', 'uncertainty', 
                    'epoch_forget_forever', 'true_label']
    df_examples[float_cols] = df_examples[float_cols].apply(lambda x: x.astype('float'))

    int_cols = ['example_idx', 'pred_label', 'epoch_forget_forever', 'true_label']
    df_examples[int_cols] = df_examples[int_cols].apply(lambda x: x.astype('int'))

    cols = [
        'example_idx', 'dataset_file_name', 
        'pred_proba', 'uncertainty', 
        'pred_label', 'true_label',		
        'epoch_forget_forever'
    ]
    df_examples = df_examples[cols]
    
    #Specify examples to excluded from the dataset
    if threshold_val is None:
        filtr = df_examples['epoch_forget_forever'] == 1
    else:
        filtr = df_examples['epoch_forget_forever'] <= threshold_val
    df_examples['is_filtered'] = 0
    df_examples.loc[filtr, 'is_filtered'] = 1

    #Specify folder to save files with results of the method
    if example_forgetting_dir is None:
        example_forgetting_dir = Path(features_path).parent
    else:
        example_forgetting_dir = Path(features_path)

    example_forgetting_dir = \
        example_forgetting_dir / f"{dataconf['data_name']}_second_forgetting" 
    example_forgetting_dir = Path(example_forgetting_dir)

    #Save forgetting info and clients ids for filtering  
    if not example_forgetting_dir.is_dir():
        example_forgetting_dir.mkdir(parents=True, exist_ok=True)

    file_name_to_save = \
        example_forgetting_dir / f"{dataconf['data_name']}_second_forgetting_stats.csv"
    file_name_to_save_df = Path(file_name_to_save)
    df_examples.to_csv(file_name_to_save_df, index=False)

    file_name_to_save_ser = \
        example_forgetting_dir / f"{dataconf['data_name']}_files_to_be_excluded.txt"
    filtr = (df_examples['is_filtered'] == 1)
    ser_client_ids_excluded = df_examples.loc[filtr, 'dataset_file_name'].values
    np.savetxt(file_name_to_save_ser, ser_client_ids_excluded, delimiter=" ", fmt="%s") 

    print(f"Number of examples to exclude: {ser_client_ids_excluded.size}")
    print(f"Examples second forgetting stats are saved in {file_name_to_save_df}")
    print(f"Files for excluding are saved in {file_name_to_save_ser}")

    return df_examples
