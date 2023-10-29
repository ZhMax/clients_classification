from typing import Any, Dict, List, Union, Tuple


from prcskr.data_loaders.datautils import create_datasets
from prcskr.models.due.model_due import start_training

def fit_classifier(
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
    split_frac_train_val: float = 1.0,
    path_to_file_names_to_be_excluded: str = None,
    is_forgetting: bool = False,
    metrics_on_train: bool = False,
    ckpt_resume: str = None
):

    """
    Create training and validation datasets, train model with saving
    checkpoints in files

    Args:
        run_name: str
            To distinguish runs of trainings

        log_dir: str
            Path to directory for saving logs
        
        ckpt_dir: str
            Path to directory for saving checkpoints

        features_path: str
            Path to a directory which contains files with features.

        targets_path: str
            Path to a directory which contains files with true labels.
            It is supposed that the files containing features and true 
            label related to one example from the dataset have the same name.

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

        split_frac_train_val: float = 1.0
            Fraction of training part of the original dataset. The value 1.0 spicifies
            that the overall dataset will be used for training.

        path_to_file_names_to_be_excluded: str
            Path to a `.txt` file which contains names of files 
            to be excluded from the original dataset for training.

        is_forgetting: bool = False
            If the indicator is true, the masks required for computing 
            forgetting values of examples will be collected during training and 
            saved in checkpoint files.  
        
        metrics_on_train: bool = False
            Is it necessary to compute metrics on training set

        ckpt_resume: str = None
            Path to a checkpoint file `*.ckpt` which is used to load the model.
            It should be `None` to train a model from an initial state.

        Returns:
            val_metrics: Dict[str, Any]
                Dict containing name of metrics and values

            trainer: DueTrainerForgetting
                An object is used to train, validate and load model
    """

    #Create training and validation datasets
    train_dataset, val_dataset = create_datasets(
        features_path=features_path,
        random_state=random_state,
        features_dim=dataconf['features_dim'],
        split_fraction=split_frac_train_val,
        targets_path=targets_path,
        path_to_file_names_to_be_excluded=path_to_file_names_to_be_excluded,
        mode='fit'
    )

    #Start model training
    trainer = start_training(
        run_name=run_name,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        ckpt_resume=ckpt_resume,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        random_state=random_state,
        total_epochs=total_epochs,
        lr=lr,
        metrics_on_train=metrics_on_train,
        is_forgetting=is_forgetting,
        dataconf=dataconf,
        modelconf=modelconf
    )

    #Print path to the checkpoint associated with the last epoch
    ckpt_resume = trainer.get_best_or_last_ckpt('last')
    print(f'Checkpoint for the last epoch: {ckpt_resume}')
    
    #Load the model and compute metrics on the validation datase
    trainer.load_last_model()
    val_metrics = trainer.test(trainer._val_loader)
    
    return val_metrics, trainer
