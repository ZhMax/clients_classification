from typing import Any, Dict, List, Union, Tuple

from prcskr.data_loaders.datautils import create_datasets
from prcskr.models.due.model_due import get_predictions 




def load_classifier_and_predict(
    run_name: str,
    features_path: str,
    ckpt_resume: str,
    random_state: int,
    dataconf: Dict[str, Any],
    modelconf: Dict[str, Any],
    targets_path: str = None,
    path_to_file_names_to_be_excluded: str = None,       
):
    """
        Load model from a checkpoint and get predictions on dataset

        Args:
            run_name: str
                To distinguish runs of predictions
            
            features_path: str
                Path to a directory which contains files with features.

            ckpt_resume: str = None
                Path to a checkpoint file `*.ckpt` which is used to load the model.

            random_state: int = None
                To provide reproducibility of computations.

            dataconf: Dict[str, Any]
                Data config

            modelconf: Dict[str, Any]
                Model config

            targets_path: str
                Path to a directory which contains files with true labels.
                If it is `None`, target variable will not be returned.
                It is supposed that the files containing features and true label 
                for the same examples from the dataset have the same name.

            path_to_file_names_to_be_excluded: str = None
                Path to a `.txt` file which contains names of files 
                from the original dataset to be excluded from prediction.
        
        Returns:
            preds: np.array
                Array with probabilities of classes for each example.
                
            uncertainties: np.array
                Array with uncertainty of a prediction for each example.

            file_name: List
                List of file names for which predictions are made.

            gts: np.array 
                Array of true labels. 
    """
    
    dataset = create_datasets(
        features_path=features_path,
        random_state=random_state,
        path_to_file_names_to_be_excluded=path_to_file_names_to_be_excluded,
        targets_path=targets_path,
        features_dim=dataconf['features_dim'],
        split_fraction=None,
        mode='predict'
    )

    preds, uncertainties, dataset_inds, file_names, gts, trainer = \
        get_predictions(    
            run_name=run_name,
            ckpt_resume=ckpt_resume,
            random_state=random_state,
            dataset=dataset,
            dataconf=dataconf,
            modelconf=modelconf
        )
    
    return preds, uncertainties, file_names, gts, trainer