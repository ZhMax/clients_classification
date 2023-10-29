from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from typing import Any, Dict, List, Union, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class IndexedDatasetFromFiles(Dataset):
    """Dataset for loading data vectors from `.npy` files."""

    def __init__(
        self,
        features_path: str,
        features_dim: int,
        targets_path: str = None,
        data_files: List[str] = None,
    ):
        """
        Parameters
        ----------
            features_path: str
                The directory where files with feature vectors are placed.

            features_dim: int
                Dimension (the number of components) of the feature vector

            targets_path: str
                The directory where files with true labels are placed.
                It is supposed that the files containing features and true label 
                related to one example from the dataset have the same name.

            datafile_names: (str) = None
                Names of files are included in the dataset. If it is `None`,
                all `.npy` files containing in the directory will be chosen.
        """

        super(IndexedDatasetFromFiles, self).__init__()

        self._features_path = Path(features_path)
        
        self._features_dim = features_dim
        self._targets_path = targets_path

        #files included in the dataset
        if data_files is None:
            self._all_data_files = list(self._features_path.glob('*.npy'))
        else:
            all_data_files = list(self._features_path.glob('*.npy'))
            self._all_data_files = [file_name for file_name in all_data_files 
                                              if file_name.stem in data_files]


    def __getitem__(self, idx: int):
        """
        Get vectors with features and label by idx of `file_name` 
        in `all_data_files`
        """

        path_to_feature_vec = Path(self._all_data_files[idx])
        input_vec = np.load(path_to_feature_vec)
        x = input_vec[0:self._features_dim]
        x = torch.from_numpy(x).float()

        if self._targets_path is not None:
            target_file = path_to_feature_vec.name
            path_to_target = Path(self._targets_path) / target_file
            path_to_target = Path(path_to_target)
            target = np.load(path_to_target)
            y = np.array(target)
            y = torch.from_numpy(y).int()
            
            return x, y, idx

        else:
            y = None

            return x, idx
            

    def __len__(self):
        return len(self._all_data_files)

    
    def get_file_name(self, idx: int):
        """Get file name without extension by idx"""
        file_name = self._all_data_files[idx]
        return file_name.stem


def split_array_into_twoparts_by_inds(
    ar: np.array,
    random_state: int,
    split_fraction: float,
):
    """
    Divide input array by two parts and return indices 
    for each part

    Args:
        ar: np.array
            Input array

        random_state: int
            To provide reproducibility

        split_fraction: float
        Relation between sizes of the first part and the input array.
        If it is equal 1.0, the first part size is equal to 
        the input array size.
    """
    original_ids = np.array(range(len(ar)))
    
    inds_pt1, inds_pt2 = train_test_split(
        original_ids, 
        train_size=split_fraction, 
        random_state=random_state, 
        shuffle=False)

    return inds_pt1, inds_pt2


def create_datasets(
    features_path: str,
    random_state: int,
    features_dim: int,
    mode: Literal['predict', 'fit', 'forgetting', 'second-split-forgetting'],
    targets_path: str = None,
    path_to_file_names_to_be_excluded: str = None,
    split_fraction: float = None
):
    """
    Create datasets from files containing in the directory `data_filepath` 

    Args:
        data_filepath: str
            Path to a directory which contains all files of the dataset.

        random_state: int
            To provide reproducibility.

        features_dim: int
            Dimension (the number of components) of the feature vector

        mode: str
            It takes one of the values 'predict', 'fit', 'forgetting' or 'second-split-forgetting'. 
            Depending on the value of the argument, datasets will be created to train 
            the model, to get predictions or to find noisy examples by forgetting methods
        
        targets_path: str = None
            Path to a directory which contains files with true labels.
            It is supposed that the files containing features and true label 
            related to one example from the dataset have the same name. If it is `None`, 
            target variable will not be returned.

        path_to_file_names_to_be_excluded: str
            Path to a `.txt` file which contains names of files 
            to be excluded from the original dataset.

        split_fraction: float
        Relation between sizes of the first part and the input array.
    """

    #Find `*.npy` files to include into datasets 
    features_path = Path(features_path)
    all_data_files = list(features_path.glob('*.npy'))
    datafiles_names = [file_id.stem for file_id in all_data_files]

    #Exclude files with names containing in .txt file
    #given by path_to_file_names_to_be_excluded
    if path_to_file_names_to_be_excluded is not None:
        file_name = path_to_file_names_to_be_excluded
        excluded_names = np.loadtxt(file_name, delimiter=' ', dtype='str')
        datafiles_names = [item for item in datafiles_names if item not in excluded_names]
        print(f"From the dataset {len(excluded_names)} files are excluded.")

    #Create datasets
    if mode == 'predict':
        dataset_pt1 = IndexedDatasetFromFiles(
            features_path=features_path, 
            features_dim=features_dim,
            targets_path=targets_path,
            data_files=datafiles_names)
        dataset_pt2 = None

        return dataset_pt1

    elif mode == 'fit':
        if split_fraction is None:
            split_fraction = 1.0

        if split_fraction < 1.0:
            inds_pt1, inds_pt2 = split_array_into_twoparts_by_inds(
                datafiles_names, 
                random_state,
                split_fraction
            )

            datafiles_names_pt1 = [datafiles_names[i] for i in inds_pt1]
            datafiles_names_pt2 = [datafiles_names[i] for i in inds_pt2]

            dataset_pt1 = IndexedDatasetFromFiles(
                features_path=features_path, 
                features_dim=features_dim,
                targets_path=targets_path,
                data_files=datafiles_names_pt1)
            dataset_pt2 = IndexedDatasetFromFiles(
                features_path=features_path, 
                features_dim=features_dim,
                targets_path=targets_path, 
                data_files=datafiles_names_pt2)
        
        else:
            dataset_pt1 = IndexedDatasetFromFiles(
                features_path=features_path, 
                features_dim=features_dim,
                targets_path=targets_path, 
                data_files=datafiles_names)
            dataset_pt2 = IndexedDatasetFromFiles(
                features_path=features_path, 
                features_dim=features_dim,
                targets_path=targets_path,
                data_files=datafiles_names)

        return dataset_pt1, dataset_pt2
    
    elif mode == 'forgetting':
        dataset_pt1 = IndexedDatasetFromFiles(
            features_path=features_path, 
            features_dim=features_dim,
            targets_path=targets_path,
            data_files=datafiles_names)
        dataset_pt2 = None

        return dataset_pt1
    
    elif mode == 'second-split-forgetting':
        if split_fraction is None:
            split_fraction = 0.5

        inds_pt1, inds_pt2 = split_array_into_twoparts_by_inds(
            datafiles_names, 
            random_state,
            split_fraction
        )


        datafiles_names_pt1 = [datafiles_names[i] for i in inds_pt1]
        datafiles_names_pt2 = [datafiles_names[i] for i in inds_pt2]

        dataset_pt1 = IndexedDatasetFromFiles(
            features_path=features_path, 
            features_dim=features_dim,
            targets_path=targets_path,
            data_files=datafiles_names_pt1)
        dataset_pt2 = IndexedDatasetFromFiles(
            features_path=features_path, 
            features_dim=features_dim,
            targets_path=targets_path,
            data_files=datafiles_names_pt2)

        return dataset_pt1, dataset_pt2
    else:
        raise ValueError('That mode is unknown')


def create_dataloader(
    dataset: Dataset,
    random_state: int,
    batch_size: int,
    is_shuffle: bool,
    num_workers: int,
    is_pin_memory: bool
):
    """Create a torch dataloader from a dataset"""

    g = torch.Generator()
    g.manual_seed(random_state)

    torch_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        num_workers=num_workers,
        pin_memory=is_pin_memory,
        generator=g
    )

    return torch_loader
