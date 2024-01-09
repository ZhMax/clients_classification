from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pa_dataset

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from typing import Any, Dict, List, Union, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def save_parquetfile(path_to_save, df):
    """Save parquet file"""
    pa_tab = pa.Table.from_pandas(df)
    pq.write_table(pa_tab, path_to_save)


class IndexedDataset(Dataset):
    """Dataset for loading data vectors from `.npy` files."""

    def __init__(
        self,
        example_names: list,
        features: Dict[str, np.array],
        targets: Dict[str, np.array] = None,
    ):
        """
        Parameters
        ----------
            example_names: list
                List contains names of examples from a dataset

            features: Dict[str, float]
                Dictionary contains for each example its name and 
                vector with features

            targets: Dict[str, np.array] = None
                Dictionary contains for each example its name and 
                true label
        """

        super(IndexedDataset, self).__init__()

        self._example_names = example_names
        
        self._features = features
        self._targets = targets

    def __getitem__(self, idx: int):
        """
        Get vectors with features and label by idx of `file_name` 
        in `all_data_files`
        """

        example_name = self._example_names[idx]
        x = self._features[example_name]
        x = np.asarray(x)
        x = torch.from_numpy(x).float()

        if self._targets is not None:
            target = self._targets[example_name]
            y = np.asarray(target)
            y = torch.from_numpy(y).int()
            return x, y, idx

        else:
            y = None
            return x, idx
            

    def __len__(self):
        return len(self._example_names)

    
    def get_file_name(self, idx: int):
        """Get example by idx"""
        example_name = self._example_names[idx]
        return example_name


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
    path_to_examples_to_be_excluded: str = None,
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

        path_to_examples_to_be_excluded: str
            Path to a `.txt` file which contains names of files 
            to be excluded from the original dataset.

        split_fraction: float
        Relation between sizes of the first part and the input array.
    """

    #Load pyarrow datasets
    ds_features = pa_dataset.dataset(features_path)
    
    if targets_path is not None:
        ds_targets = pa_dataset.dataset(targets_path)

    #get np arrays
    if path_to_examples_to_be_excluded is None:
        example_names = np.array(
            ds_features.scanner(
                columns=['__index_level_0__']
            ).to_table()
        )[0]

        features = np.array(
            ds_features.scanner(
                columns=[str(item) for item in range(0, features_dim)]
            ).to_table()
        )

        if targets_path is not None:
            targets = ds_targets.to_table()
            labels = np.array(targets[0])
            target_names = np.array(targets['__index_level_0__'])

    else:
        file_name = path_to_examples_to_be_excluded
        excluded_names = np.loadtxt(file_name, delimiter=' ', dtype='str')

        example_names = np.array(
            ds_features.scanner(
                columns=['__index_level_0__'],
                filter=(~pa_dataset.field('__index_level_0__').isin(excluded_names))
            ).to_table()
        )[0]

        features = np.array(
            ds_features.scanner(
                columns=[str(item) for item in range(0, features_dim)],
                filter=(~pa_dataset.field('__index_level_0__').isin(excluded_names))
            ).to_table()
        )

        if targets_path is not None:
            targets = ds_targets.scanner(
                filter=(~pa_dataset.field('__index_level_0__').isin(excluded_names))
            ).to_table()
            labels = np.array(targets[0])
            target_names = np.array(targets['__index_level_0__'])

        print(f"From the dataset {len(excluded_names)} files are excluded.")

    #create dictionaries with features and targets
    features = features.T
    features = {k: v for k, v in zip(example_names, features)}

    if targets_path is not None:
        corrected_target_names = None
        if 'is_corrected' in targets.column_names:
            mask = (np.array(targets['is_corrected']) == 1)
            if mask.sum() > 0:
                corrected_target_names = target_names[mask]
    
        targets = {k: int(v) for k, v in zip(target_names, labels)}

    else:
        targets = None


    #create datasets
    if mode == 'predict':
        dataset_pt1 = IndexedDataset(
            example_names, 
            features, 
            targets
        )
        dataset_pt2 = None

        return dataset_pt1

    elif mode == 'fit':
        if split_fraction is None:
            split_fraction = 1.0

        if split_fraction < 1.0:

            inds_pt1, inds_pt2 = split_array_into_twoparts_by_inds(
                example_names, 
                random_state,
                split_fraction
            )

            example_names_pt1 = [item for item in example_names[inds_pt1]]
            example_names_pt2 = [item for item in example_names[inds_pt2]]

            #add examples with corrected label to train_dataset
            if corrected_target_names is not None:
                examples_to_add = np.setdiff1d(
                    corrected_target_names, 
                    example_names_pt1
                )

                num_examples = examples_to_add.size
                if num_examples > 0:
                    example_names_pt1 = np.concatenate([
                        example_names_pt1, examples_to_add
                    ])

                    example_names_pt2 = np.setdiff1d(
                        example_names_pt2,
                        examples_to_add
                    )

                    example_names_pt2 = np.concatenate([
                        example_names_pt2,
                        example_names_pt1[0:num_examples]
                    ])

                    example_names_pt1 = np.setdiff1d(
                        example_names_pt1,
                        example_names_pt2
                    )

                    example_names_pt1 = list(example_names_pt1)
                    example_names_pt2 = list(example_names_pt2)

            #create datasets
            dataset_pt1 = IndexedDataset(
                example_names_pt1, 
                features, 
                targets
            )
            dataset_pt2 = IndexedDataset(
                example_names_pt2, 
                features, 
                targets
            )
        
        else:
            dataset_pt1 = IndexedDataset(
                example_names, 
                features, 
                targets
            )
            dataset_pt2 = IndexedDataset(
                example_names, 
                features, 
                targets
            )

        return dataset_pt1, dataset_pt2
    
    elif mode == 'forgetting':
        dataset_pt1 = IndexedDataset(
            example_names, 
            features, 
            targets
        )
        dataset_pt2 = None

        return dataset_pt1
    
    elif mode == 'second-split-forgetting':
        if split_fraction is None:
            split_fraction = 0.5

            inds_pt1, inds_pt2 = split_array_into_twoparts_by_inds(
                example_names, 
                random_state,
                split_fraction
            )

            example_names_pt1 = [item for item in example_names[inds_pt1]]
            example_names_pt2 = [item for item in example_names[inds_pt2]]

            dataset_pt1 = IndexedDataset(
                example_names_pt1, 
                features, 
                targets
            )
            dataset_pt2 = IndexedDataset(
                example_names_pt2, 
                features, 
                targets
            )

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
