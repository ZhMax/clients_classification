from pathlib import Path


from typing import Any, Dict, List, Union, Tuple

import pandas as pd
import numpy as np

from prcskr.data_loaders.datautils import save_parquetfile, create_datasets
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
    path_to_examples_to_be_excluded: str = None
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
            Path to a directory which will be used to save array with noisy exmaples 
            If it is `None`, then the directory with name `f"{data_name}_second_forgetting"`
             will be created in the parent of the directory `data_filepath`. 
            The field `data_name` is included in the data configuration file.
            
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

        path_to_examples_to_be_excluded: str
            Path to a `.txt` file which contains names of examples 
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
        path_to_examples_to_be_excluded=path_to_examples_to_be_excluded,
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
        preds_proba_pt_pos = preds_pt[:, 1]

        ret_ft_pt = trainer_pt._forgetting_dict[run_names[step_idx-1]]
        forget_epochs_pt, file_names_pt_forget = \
            get_forget_forever_epochs_for_examples(ret_ft_pt)


        preds_labels.append(preds_pt_label)
        preds_proba.append(preds_proba_pt_pos)
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
        columns=['example_idx', 'example_name_loader', 
                    'pred_label', 'pred_proba', 
                    'uncertainty', 'true_label'])

    examples_forget_epochs = np.hstack(examples_forget_epochs)
    dataset_file_names = np.hstack(dataset_file_names)
    
    ar_forget = np.vstack([
        dataset_file_names, examples_forget_epochs]).T
    
    df_forget = pd.DataFrame(
        data=ar_forget, 
        columns=['example_name', 'epoch_forget_forever'])
    
    df_examples = pd.merge(
        df_predicted, 
        df_forget,
        left_on='example_name_loader',
        right_on='example_name',
        how='left')

    df_examples.drop(columns='example_name_loader', inplace=True)

    float_cols = ['pred_proba', 'pred_label', 'uncertainty', 
                    'epoch_forget_forever', 'true_label']
    df_examples[float_cols] = df_examples[float_cols].apply(lambda x: x.astype('float'))

    int_cols = ['example_idx', 'pred_label', 'epoch_forget_forever', 'true_label']
    df_examples[int_cols] = df_examples[int_cols].apply(lambda x: x.astype('int'))

    cols = [
        'example_idx', 'example_name', 
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
        example_forgetting_dir / f"{dataconf['data_name']}_secondsplit_forgetting" 
    example_forgetting_dir = Path(example_forgetting_dir)

    #Save forgetting info and clients ids for filtering  
    if not example_forgetting_dir.is_dir():
        example_forgetting_dir.mkdir(parents=True, exist_ok=True)

    filtr = (df_examples['is_filtered'] == 1)
    examples_to_be_corrected = df_examples.loc[filtr, 'example_name'].unique()

    file_name_to_save = \
        example_forgetting_dir / f"{dataconf['data_name']}_secondsplit_forgetting_stats.csv"
    file_name_to_save_df = Path(file_name_to_save)
    df_examples.to_csv(file_name_to_save_df, index=False)

    print(f"Number of examples to exclude: {len(examples_to_be_corrected)}")
    print(f"Examples forgetting stats are saved in {file_name_to_save_df}")

    if len(examples_to_be_corrected) > 0:

        #Save array with file names to excluded from the dataset in .txt file
        file_name_to_save_ar = \
            example_forgetting_dir / f"{dataconf['data_name']}_examples_to_be_excluded.txt"
        np.savetxt(
            file_name_to_save_ar, 
            examples_to_be_corrected, 
            delimiter=" ", 
            fmt="%s"
        )

        #Save pd.DataFrame with corrected labels of examples
        df_examples['corrected_label'] = df_examples['true_label']
        filtr = df_examples['example_name'].isin(examples_to_be_corrected)
        df_examples.loc[filtr, 'corrected_label'] = \
            df_examples.loc[filtr, 'corrected_label']\
                       .apply(lambda x: 1 if x==0 else 0)
        
        labels = df_examples['corrected_label'].values
        objects = df_examples['example_name'].values
        objects = [str(item) for item in objects]

        targets_to_save = pd.DataFrame(data=labels, index=objects)

        examples_to_be_corrected = [str(item) for item in examples_to_be_corrected]
        targets_to_save['is_corrected'] = 0
        targets_to_save.loc[examples_to_be_corrected, 'is_corrected'] = 1
        path_to_save = Path(targets_path).parent
        file_name = Path(targets_path).stem
        path_to_save = path_to_save / f'{file_name}_corrected_by_secondsplit_forgetting.parquet'
        path_to_save = str(path_to_save)

        save_parquetfile(path_to_save, targets_to_save)

        print(f"Examples for excluding are saved in {file_name_to_save_ar}")
        print(f"Corrected labels are saved in {path_to_save}")

    return df_examples
