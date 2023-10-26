import yaml
import shutil
from pathlib import Path
from datetime import datetime

from typing import Any, Dict, List, Union, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from prcskr.filtration_classification.fit import fit_classifier
from prcskr.filtration_classification.forgetting_filtration import (
    filtration_of_dataset_by_forgetting
)
from prcskr.filtration_classification.second_split_forgetting_filtration import (
    filtration_of_dataset_by_second_split_forgetting
)
from prcskr.filtration_classification.predict import load_classifier_and_predict


class FilteredClassifier():
    """
    Main class to create model to solve classification problem and 
    conduct filtration dataset by excluding examples with noisy (incorrect)
    labels. 
    
    Parameters
    ----------
        run_name: str
            To distinguish classifier runs

        path_to_dataconf: str
            Path to a `.yml` data configuration file
        
        path_to_modelconf: str
            Path to a `.yml` model configuration file
    
        log_dir: str
            Path to a directory where logs of model training will be saved. 
            The logs will be saved in a directory which will be created inside the log_dir

        ckpt_dir: str
            Path to a directory where checkpoints of state dictionaries (e.g model) will be saved 
            during training. The checkpoints will be saved in a directory which will be created 
            inside the ckpt_dir
    
    Methods
    -------
        fit(
            data_filepath: str,
            split_frac_train_val: float = 1.0,
            random_state: int = None,
            total_epochs: int = None,
            lr: float = None,
            path_to_file_names_to_be_excluded: str = None,
            is_forgetting: bool = False,
            metrics_on_train: bool = False,
            ckpt_resume: str = None
        )
            A method to train a model for solving classification problem. The implemetation is
            performed for the Deterministic Uncertainty Estimation (DUE) model, which was proposed in 

            https://arxiv.org/abs/2102.11409, https://github.com/y0ast/DUE/tree/main

            The model includes the fully connected ResNet combined with the Gaussian process.
            The fully connected ResNet consists of residual feedforward layers with the relu activation 
            functions. As a regularization, spectral normalization and dropout are applied to each layer.
            Combination of residual connections with spectral normalization enables to provide smoothness 
            and sensitivity for fully connected neural network. Smothness (or stability) implies small 
            changes in the input cannot cause massive shifts in the output. Sensitivity implies that
            when the input changes the feature representation also changes. This properties are capable to 
            prevent feature collapse, when different input features are mapped by neural network into close
            vectors or the same features are mapped into far-away vectors.
            The Gaussian process is used for classification of vectors provided by the fully connected ResNet
            by given labels. The main advantage of the Gaussian process is uncertainty estimation of made
            predictions. Therefore, the predictions with high uncertainty can be excluded to achive more 
            reiliable results.

        filtration_by_forgetting(
            data_filepath: str,
            example_forgetting_dir: str = None,
            threshold_val: int = None,
            random_state: int = None,
            total_epochs: int = None,
            lr: float = None,
            verbose: str = True,
            ckpt_resume: str = None,
            path_to_file_names_to_be_excluded: str = None
        )
            An implementation of a method to find noisy examples (mislabeled examples) in dataset by
            counting the forgetting of examples during training. The method is named as the forgetting 
            method. Firstly it was proposed in

            https://arxiv.org/abs/1812.05159, https://github.com/mtoneva/example_forgetting/tree/master

            According to the paper, noisy examples are frequently forgotten by a model during training
            or are stayed be unlearned. Therefore, to find noisy examples, the following algorithm was
            implemented.
            1) A model is trained on the full dataset, and forgetting masks of examples are saved. The
            masks are formed by comparison of model predictions on each epoch with true labels.
            2) After training, the quantity of the epochs, when each example was forgotten, is counted.
            The forgetting count for unlearned examples is assigned to be equal to `total_epochs`.
            3) Array of file names of the dataset containing unlearned examples are saved in `.txt` file
            and can be used for excluding from the dataset in the next model trainings. By varying 
            `threshold_val`, examples with high value of the forgetting counts also can be excluded. 

        
        filtration_by_second_split_forgetting(
                data_filepath: str,
                example_forgetting_dir: str = None,
                threshold_val: int = None,
                random_state: int = None,
                total_epochs_per_step: int = None,
                lr: float = None,
                verbose: str = True,
                ckpt_resume: str = None,
                path_to_file_names_to_be_excluded: str = None
        )
            An implementation of a method to find noisy examples (mislabeled examples) in dataset by 
            sequential model training on its parts and counting forgetting of the examples. The method 
            is named as the second-spit forgetting method. It was proposed in

            https://arxiv.org/abs/2210.15031, https://github.com/pratyushmaini/ssft


            One of the disadvantage of the forgetting method for filtration of dataset is that the set 
            of unlearned and frequently forgetting examples can include complex examples. The complex 
            examples are placed close to the boundary between different classes. Therefore, such examples
            contribute to improve model training. To separate noisy examples from the complex ones, the
            following algorithm was implemented.
            1) The full dataset is divided into two halves, which we will name as the first part and
            the second part.
            2) The model is trained on the first part of the dataset until the values of the loss functions 
            or the tracked metric stabilize (the first training). Then the model training continues 
            on the second part of the dataset until the values of the loss functions or the tracked metric 
            stabilize (the second training).
            3) Examples from the second part of the dataset, which were forgotten after one epoch of 
            the second training, are marked as noisy. 
            4) From the initial state, model trainings are perfomed for the second and the first part of the
            dataset (the third and fourth training).
            5) Examples from the first part of the dataset, which were forgotten after one epoch of 
            the fourth training, are marked as noisy.
            6) Array of file names of the dataset containing noisy examples are saved in `.txt` file
            and can be used for excluding from the dataset in the next model trainings. By varying 
            `threshold_val`, examples, which were forgotten after larger number of epoch of the subsequent
            training, can be excluded.
 
        predict(
            data_filepath: str,
            ckpt_resume: str,
            random_state: int = None,
            is_target_col: bool = True,
            path_to_file_names_to_be_excluded: str = None
        )
            A method to load model from chekpoint file given by `ckpt_resume` and get predictions for
            a dataset given by `data_filepath`.
    """

    def __init__(
        self,
        run_name: str,
        path_to_dataconf: str,
        path_to_modelconf: str,
        log_dir: str,
        ckpt_dir: str
    ):

        self._run_name = (
            run_name if run_name is not None else datetime.now().strftime('%F_%T')
        )

        self._log_dir = log_dir
        self._ckpt_dir = ckpt_dir

        self._path_to_dataconf = path_to_dataconf
        self._path_to_modelconf = path_to_modelconf

        with open(self._path_to_dataconf, 'r') as f:
            self._dataconf = yaml.load(f, Loader=yaml.FullLoader)

        with open(self._path_to_modelconf, 'r') as f:
            self._modelconf = yaml.load(f, Loader=yaml.FullLoader)

        #Attribute which contains `trainer` object returned after the model training
        self._trainer = None  

    @property
    def data_config(self) -> Dict[str, Any]:
        return self._dataconf
    
    @property
    def model_config(self) -> Dict[str, Any]:
        return self._modelconf


    def _clear_dir(
        self,
        dir_path: str
    ):
        """Remove all checkpoint from the directory"""
        dir_path = Path(dir_path)
        if dir_path.is_dir():
            all_ckpt = list(dir_path.glob('*.ckpt'))
            for ckpt in all_ckpt:
                ckpt.unlink()

    def _remove_logs(
        self,
        log_dir: str
    ):
        """Remove all logs from the directory"""
        log_dir = Path(log_dir)
        if log_dir.is_dir():
            all_logs = list(log_dir.glob('*.log'))
            for log_file in all_logs:
                log_file.unlink()

    def fit(
        self,
        data_filepath: str,
        split_frac_train_val: float = 1.0,
        random_state: int = None,
        total_epochs: int = None,
        lr: float = None,
        path_to_file_names_to_be_excluded: str = None,
        is_forgetting: bool = False,
        metrics_on_train: bool = False,
        ckpt_resume: str = None
    ):  
        """
        Method to train model for solving classification problem.

        Args:
            data_filepath: str
                Path to a directory which contains files with features and labels. 
                It is supposed that each file contains vector consisting of
                features and a label of an example (in the last component).

            split_frac_train_val: float = 1.0
                Fraction of training part size from the size of the full dataset. 
                The value 1.0 spicifies that the full dataset will be used for training.

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

            path_to_file_names_to_be_excluded: str
                Path to a `.txt` file which contains names of files 
                to be excluded from the original dataset for training.

            is_forgetting: bool = False
                If the indicator is true, the masks required for computing 
                forgetting counts of examples will be collected during training and 
                saved in checkpoint files.  
            
            metrics_on_train: bool = False
                Is it necessary to compute metrics on training dataset

            ckpt_resume: str = None
                Path to a checkpoint file `*.ckpt` which is used to load the model.
                It should be `None` to train a model from an initial state.
        """
        
        #Specify parameters for model training and 
        #names of directories to save logs and checkpoints
        run_name = self._run_name + f'_fit'

        if random_state is None:
            random_state = self._dataconf['random_state']
            print(f'random state is specified from config as {random_state}')

        if total_epochs is None:
            total_epochs = self._modelconf['total_epochs']
            print(f'total epochs is specified from config as {total_epochs}')

        if lr is None:
            lr = self._modelconf['lr'] 
            print(f'lr is specified from config as {lr}')

        if self._ckpt_dir is not None:
            ckpt_dir = Path(self._ckpt_dir)

        if self._log_dir is not None:
            log_dir = Path(self._log_dir) / run_name  


        if ckpt_resume is None:
            #Remove checkpoints and logs saved from the previous fitting
            if ckpt_dir.is_dir():
                ckpt_path = Path(ckpt_dir) / run_name
                ckpt_path = Path(ckpt_path)
                self._clear_dir(ckpt_path)

            if log_dir.is_dir():
                self._remove_logs(log_dir)

        #Train model
        val_metrics, trainer = fit_classifier(
            run_name=run_name, 
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
            data_filepath=data_filepath,
            split_frac_train_val=split_frac_train_val,
            random_state=random_state,
            total_epochs=total_epochs,
            lr=lr,
            path_to_file_names_to_be_excluded=path_to_file_names_to_be_excluded,
            is_forgetting=is_forgetting,
            metrics_on_train=metrics_on_train,
            ckpt_resume=ckpt_resume,
            dataconf=self._dataconf,
            modelconf=self._modelconf
        )

        self._trainer = trainer

        print('Metrics for the validation dataset:')
        print(val_metrics)

    def filtration_by_forgetting(
        self,
        data_filepath: str,
        example_forgetting_dir: str = None,
        threshold_val: int = None,
        random_state: int = None,
        total_epochs: int = None,
        lr: float = None,
        verbose: str = True,
        ckpt_resume: str = None,
        path_to_file_names_to_be_excluded: str = None           
    ):
        """
        Method for filtering examples with noisy labels from the original dataset by 
        the forgetting method. The file names containig found examples are saved in 
        `.txt` file. 

        Args:
            data_filepath: str
                Path to a directory which contains files with features and labels. 
                It is supposed that each file contains vector consisting of
                an embedding and a label of an example (in the last component).

            example_forgetting_dir: str = None
                Path to a directory which will be used to save array with file names 
                containing noisy labels. If it is `None`, then the directory with name
                `f"{data_name}_forgetting"` will be created in the parent of the directory
                `data_filepath`. The field `data_name` is provided by the data configuration
                file.
            
            threshold_val: int = None
                Threshold value for `forgetting_counts`, which can be used to filtrate examples.
                If it is `None`, only unlearned examples will be proposed for excluding 
                from the dataset. 

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

            verbose: bool = True
                Is it necessary to return pd.DataFrame with counts of 
                forgetting for examples.

            ckpt_resume: str = None
                Path to a checkpoint file `*.ckpt` which is used to load the model
                and masks collected during training.
                It should be `None` to train a model from an initial state.     

            path_to_file_names_to_be_excluded: str
                Path to a `.txt` file which contains names of files 
                to be excluded from the original dataset for training.

        Return:
            df_examples: pd.DataFrame
                pd.DataFrame containing forgetting counts for examples of 
                the dataset and predictions given by the trained model 
        """
         
        #Specify parameters from configs
        if random_state is None:
            random_state = self._dataconf['random_state']
            print(f'random state is specified from config as {random_state}')

        if total_epochs is None:
            total_epochs = self._modelconf['total_epochs']
            print(f'total epochs is specified from config as {total_epochs}')

        if lr is None:
            lr = self._modelconf['lr'] 
            print(f'lr is specified from config as {lr}')

        #Specify names of directories for saving checkpoints and logs
        run_name = self._run_name + '_forgetting'

        if self._ckpt_dir is not None:
            ckpt_dir = Path(self._ckpt_dir)

        if self._log_dir is not None:
            log_dir = Path(self._log_dir) / run_name  


        if ckpt_resume is None:
            #Remove checkpoints and logs saved from the previous method launch 
            if ckpt_dir.is_dir():
                ckpt_path = Path(ckpt_dir) / run_name
                ckpt_path = Path(ckpt_path)
                self._clear_dir(ckpt_path)

            if log_dir.is_dir():
                self._remove_logs(log_dir)

        df_examples, trainer = filtration_of_dataset_by_forgetting(
            run_name=run_name,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
            data_filepath=data_filepath,
            random_state=random_state,
            total_epochs=total_epochs,
            lr=lr,
            dataconf=self._dataconf,
            modelconf=self._modelconf,
            example_forgetting_dir=example_forgetting_dir,
            threshold_val=threshold_val,
            ckpt_resume=ckpt_resume,
            path_to_file_names_to_be_excluded=path_to_file_names_to_be_excluded
        )

        self._trainer = trainer

        if verbose:
            return df_examples



    def filtration_by_second_split_forgetting(
        self,
        data_filepath: str,
        example_forgetting_dir: str = None,
        threshold_val: int = None,
        random_state: int = None,
        total_epochs_per_step: int = None,
        lr: float = None,
        verbose: str = True,
        ckpt_resume: str = None,
        path_to_file_names_to_be_excluded: str = None
    ):

        """
        Method for filtering examples with noisy labels from the original dataset by 
        the second split forgetting method. The file names containing found examples are 
        saved in `.txt` file. 

        Args:
            data_filepath: str
                Path to a directory which contains files with features and labels. 
                It is supposed that each file contains vector consisting of
                an embedding and a label of an example (in the last component).

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

            random_state: int = None
                To provide reproducibility of computations. If it is `None`, a value  
                from the field `random_state` from the data configuration file 
                will be used.
            
            total_epochs_per_step: int = None
                A number of epochs for one step of model training. If it is `None`, a value  
                from the field `total_epochs` of the model configuration file 
                will be used.

            lr: float = None
                Learning rate in an optimizer. If it is `None`, a value from
                the field `lr` of the model configuration file 
                will be used.

            verbose: bool = True
                Is it necessary to return pd.DataFrame with counts of 
                forgetting for examples.

            ckpt_resume: str = None
                Path to a checkpoint file `*.ckpt` which is used to load the model
                and masks collected during training.
                It should be `None` to train a model from an initial state.     

            path_to_file_names_to_be_excluded: str
                Path to a `.txt` file which contains names of files 
                to be excluded from the original dataset for training.

        Return:
            df_examples: pd.DataFrame 
                pd.DataFrame containing the number of the epoch when each examples of 
                the dataset were forgotten forever and predictions given by the trained 
                model at the second and fourth training steps
        """

        #Specify parameters from configs
        if random_state is None:
            random_state = self._dataconf['random_state']
            print(f'random state is specified from config as {random_state}')

        if total_epochs_per_step is None:
            total_epochs_per_step = self._modelconf['total_epochs']
            print(f'total epochs is specified from config as {total_epochs_per_step}')

        if lr is None:
            lr = self._modelconf['lr'] 
            print(f'lr is specified from config as {lr}')

        #Specify names of directories for saving checkpoints and logs
        #checkpoints and logs from each step of the method will be saved in 
        #separate directory inside of ckpt_dir and log_dir
        if self._ckpt_dir is not None:
            ckpt_dir = Path(self._ckpt_dir) / f'{self._run_name}_second_forgetting'
            ckpt_dir = Path(ckpt_dir)

        if self._log_dir is not None:
            log_dir = Path(self._log_dir) / f'{self._run_name}_second_forgetting'
            log_dir = Path(log_dir)


        if ckpt_resume is None:
            #Remove directories from the previous method launch
            ckpt_resume_step_idx = None
            if ckpt_dir.is_dir():
                shutil.rmtree(ckpt_dir)

            if log_dir.is_dir():
                shutil.rmtree(log_dir)
        else:
            ckpt_resume = Path(ckpt_resume)
            ckpt_resume_step_idx = int(str(ckpt_resume.parent.stem)[-1])

        df_examples = filtration_of_dataset_by_second_split_forgetting(
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
            data_filepath=data_filepath,
            random_state=random_state,
            total_epochs_per_step=total_epochs_per_step,
            lr=lr,
            dataconf=self._dataconf,
            modelconf=self._modelconf,
            example_forgetting_dir=example_forgetting_dir,
            threshold_val=threshold_val,
            ckpt_resume=ckpt_resume,
            ckpt_resume_step_idx=ckpt_resume_step_idx,
            path_to_file_names_to_be_excluded=path_to_file_names_to_be_excluded
        )

        if verbose:
            return df_examples


    def predict(
        self,
        data_filepath: str,
        ckpt_resume: str,
        random_state: int = None,
        is_target_col: bool = True,
        path_to_file_names_to_be_excluded: str = None
    ):
        """
        Method to get predictions using a model loaded from a checkpoint file

        Args:
            data_filepath: str
                Path to a directory which contains files with features.
                If the files also contain labels, they should be in
                the last component of the vectors.

            ckpt_resume: str = None
                Path to a checkpoint file `*.ckpt` which is used to load the model.

            random_state: int = None
                To provide reproducibility of computations. If it is `None`, a value  
                from the field `random_state` from the data configuration file 
                will be used.

            is_target_col: bool = True
            Is a value of a target variable included into to the input vector?
            If it is `False`, target variable will be returned as `None`.

            path_to_file_names_to_be_excluded: str = None
                Path to a `.txt` file which contains names of files 
                from the original dataset to be excluded from prediction.
        
        Return:
            preds: np.array
                Array with probabilities of classes for each example.

            uncertainties: np.array
                Array with uncertainty of a prediction for each example.

            file_name: List
                List of file names for which predictions are made.

            gts: np.array 
                Array of true labels. 
        """

        if random_state is None:
            random_state = self._dataconf['random_state']
            print(f'random state is specified from config as {random_state}')

        preds, uncertainties, file_names, gts, trainer = load_classifier_and_predict(
            run_name=self._run_name,
            data_filepath=data_filepath,
            ckpt_resume=ckpt_resume,
            random_state=random_state,
            dataconf=self._dataconf,
            modelconf=self._modelconf,
            is_target_col=is_target_col,
            path_to_file_names_to_be_excluded=path_to_file_names_to_be_excluded, 
        )

        self._trainer = trainer

        if gts is not None:
            return preds, uncertainties, file_names, gts
        else:
            return preds, uncertainties, file_names
        