
from pathlib import Path


from typing import Any, Dict, List, Union, Tuple

import pandas as pd
import numpy as np

from prcskr.data_loaders.datautils import save_parquetfile, create_datasets
from prcskr.models.due.model_due import start_training, get_predictions 

from prcskr.forgetting.forgettingutils import (
    get_learn_epoch_forgetting_counts_for_examples
)


def filtration_of_dataset_by_forgetting(
    run_name: str,
    log_dir: str,
    ckpt_dir: str,
    features_path: str,
    targets_path: str,
    random_state: int,
    total_epochs: int,
    lr: float,
    dataconf: Dict[str, Any],
    modelconf: Dict[str, Any],
    example_forgetting_dir: str = None,
    threshold_val: int = None,
    ckpt_resume: str = None,
    path_to_examples_to_be_excluded: str = None
): 

    """
    Find examples of dataset, that were not learned by a model during 
    a training process, or were frequently forgotten by the model in 
    the training process.

    Args:
        run_name: str
            To distinguish runs of filtration

        log_dir: str
            Path to directory for saving logs
        
        ckpt_dir: str
            Path to directory for saving checkpoints

        features_path: str
            Path to a file with features.

        targets_path: str
            Path a file with true labels. It is supposed that features 
            and true label related to one example from the dataset have 
            the same index (name).

        random_state: int = None
            To provide reproducibility of computations. If it is `None`, a value  
            from the field `random_state` from the data configuration file 
            will be used.
        
        total_epochs: int = None
            A number of epochs for model training. If it is `None`, a value  
            from the field `total_epochs` of the model configuration file 
            will be used.

        lr: float = None
            Learning rate in an optimizer. If it is `None`, a value from
            the field `lr` of the model configuration file 
            will be used.

        dataconf: Dict[str, Any]
            Data config

        modelconf: Dict[str, Any]
            Model config

        example_forgetting_dir: str = None
            Path to a directory which will be used to save array with names of 
            noisy examples. If it is `None`, then the directory with name
            `f"{data_name}_forgetting"` will be created in the parent of the directory
            `data_filepath`. The field `data_name` is provided by the data configuration
            file.
        
        threshold_val: int = None
            Threshold value for `forgetting_counts`, which can be used to filtrate examples.
            If it is `None`, only unlearned examples will be proposed for excluding 
            from the dataset. 
         
        ckpt_resume: str = None
            Path to a checkpoint file `*.ckpt` which is used to load the model.
            It should be `None` to train a model from an initial state.

        path_to_examples_to_be_excluded: str
            Path to a `.txt` file which contains names of examples 
            to be excluded from the original dataset for training.

        Returns:
            df_examples: pd.DataFrame
                pd.DataFrame containing forgetting counts for examples of 
                the dataset and predictions given by the trained model.

            trainer: DueTrainerForgetting
                An object is used to train, validate and load model.
    """

    #Create training dataset from files given by data_filepath
    train_dataset = create_datasets(
        features_path=features_path,
        targets_path=targets_path,
        random_state=random_state,
        features_dim=dataconf['features_dim'],
        path_to_examples_to_be_excluded=path_to_examples_to_be_excluded,
        split_fraction=None,
        mode='forgetting'
    )

    #Train model and collect masks to compute 
    #Forgetting values for each example
    trainer = start_training(
        run_name=run_name,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        ckpt_resume=ckpt_resume,
        train_dataset=train_dataset,
        val_dataset=train_dataset,
        random_state=random_state,
        total_epochs=total_epochs,
        lr=lr,
        metrics_on_train=False,
        is_forgetting=True,
        dataconf=dataconf,
        modelconf=modelconf,
    )

    ckpt_resume = trainer.get_best_or_last_ckpt('last')
    
    #Get predictions on the train dataset and combine them in dataframe
    preds, uncertainties, dataset_inds, file_names, gts, trainer = \
        get_predictions(    
            run_name=run_name,
            ckpt_resume=ckpt_resume,
            random_state=random_state,
            dataset=train_dataset,
            dataconf=dataconf,
            modelconf=modelconf,
            is_forgetting=True
        )
    
    preds_label = np.argmax(preds, axis=1)
    preds_proba = preds[:, 1]

    ar_predicted = np.vstack([
        dataset_inds, file_names,
        preds_label, preds_proba, uncertainties, gts
    ]).T

    df_predicted = pd.DataFrame(
        data=ar_predicted, 
        columns=['example_idx', 'file_name_loader', 
                 'pred_label', 'pred_proba', 
                 'uncertainty', 'true_label'])

    #For each example find epochs when each it was learned forever and
    #compute a quantatity of forgeetings
    ret_pre = trainer._forgetting_dict[run_name]
    learn_epochs, forgetting_counts, file_names = \
        get_learn_epoch_forgetting_counts_for_examples(ret_pre)

    ar_forget = np.vstack([
        file_names, learn_epochs, forgetting_counts
    ]).T

    #Combine found values in a pd.DataFrame
    df_forget = pd.DataFrame(
        data=ar_forget, 
        columns=['example_name', 'learn_epoch', 'forgetting_counts'])
    
    #Merge pd.DataFrames based on file_name
    df_examples = pd.merge(
        df_predicted, 
        df_forget,
        left_on='file_name_loader',
        right_on='example_name',
        how='left')
    df_examples.drop(columns='file_name_loader', inplace=True)   

    float_cols = [
        'pred_proba', 'pred_label', 'uncertainty', 
        'learn_epoch', 'forgetting_counts', 'true_label'
    ]
    df_examples[float_cols] = df_examples[float_cols].apply(lambda x: x.astype('float'))

    int_cols = [
        'example_idx', 'pred_label', 'learn_epoch', 
        'forgetting_counts', 'true_label'
    ]
    df_examples[int_cols] = df_examples[int_cols].apply(lambda x: x.astype('int'))

    cols = [
        'example_idx', 'example_name', 
        'pred_proba', 'uncertainty', 
        'pred_label', 'true_label',		
        'learn_epoch', 'forgetting_counts'
    ]
    df_examples = df_examples[cols]

    #Find unlearned examples
    #If an example is unlearned, its forgetting_counts is given equal to
    #a number of the last epoch
    filtr = (df_examples['pred_label'] != df_examples['true_label']) &\
            (df_examples['learn_epoch'] == df_examples['learn_epoch'].max()) &\
            (df_examples['forgetting_counts'] == 0)

    df_examples.loc[filtr, 'forgetting_counts'] = trainer._last_epoch

    #Specify examples to excluded from the dataset
    if threshold_val is None:
        filtr = df_examples['forgetting_counts'] == df_examples['forgetting_counts'].max()
    else:
        filtr = df_examples['forgetting_counts'] >= threshold_val

    df_examples['is_filtered'] = 0
    df_examples.loc[filtr, 'is_filtered'] = 1    

    #Specify directory to save results
    if example_forgetting_dir is None:
        example_forgetting_dir = Path(features_path).parent
    else:
        example_forgetting_dir = Path(features_path)

    example_forgetting_dir = \
        example_forgetting_dir / f"{dataconf['data_name']}_forgetting" 
    example_forgetting_dir = Path(example_forgetting_dir)

    if not example_forgetting_dir.is_dir():
        example_forgetting_dir.mkdir(parents=True, exist_ok=True)

    filtr = (df_examples['is_filtered'] == 1)
    examples_to_be_corrected = df_examples.loc[filtr, 'example_name'].unique()

    #Save pd.DataFrame with the forgeeting values and the model predictions in .csv file
    file_name_to_save = \
        example_forgetting_dir / f"{dataconf['data_name']}_forgetting_stats.csv"
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
        path_to_save = path_to_save / f'{file_name}_corrected_by_forgetting.parquet'
        path_to_save = str(path_to_save)

        save_parquetfile(path_to_save, targets_to_save)

        print(f"Examples for excluding are saved in {file_name_to_save_ar}")
        print(f"Corrected labels are saved in {path_to_save}")

    return df_examples, trainer
