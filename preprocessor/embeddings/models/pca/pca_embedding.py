import sys
import yaml
from pathlib import Path
import shutil
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from  sklearn.decomposition import PCA

from typing import Any, Dict, List, Union, Tuple

def get_data_from_file(
    path: str
):
    """
    Load pd.DataFrame from '.parquet' or 
    '.csv' file

    Args:
        path: str
            Path to the file with data
    """
    
    path = Path(path)
    if path.suffixes[0] == '.parquet':
        df = pd.read_parquet(path)
    elif path.suffixes[0] == '.csv':
        df = pd.read_csv(path)

    return df

def extract_features_target_from_df(
    df: pd.DataFrame, 
    target_col: str
):
    """
    Extract np.arrays with features and target
    from pd.DataFrame

    Args:
        df: pd.DataFrame
            pd.DataFrame with features and target column 

        target_col: str
            name of the column with target variable

    Returns:
        X: np.array
            Array of feature vectors

        y: np.array
            Array with values of target variable
    """

    X = df.drop(columns=target_col).values
    y = df[target_col].values

    return X, y


def compute_pca_embedding(
    X: np.array,
    n_components: Union[int, float] = 0.95,
    is_std_scaler: bool = True
):
    """
    Transform features through StandardScaler and apply
    the PCA method.

    Args:
        X: np.array
            Array of feature vectors

        n_components: int or float
            as in the PCA method from sklearn

        is_std_scaler: bool = True
            Is it necessary to standardize features?

    Returns:
        X: np.array
           Array of embeddings given by the PCA method 
    """

    if is_std_scaler:
        std_scaler = StandardScaler()
        std_scaler.fit(X)
        X = std_scaler.transform(X)

    model_PCA = PCA(n_components=n_components)
    model_PCA.fit(X)
    X = model_PCA.transform(X)

    return X


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument(
        '--path-to-config',
        help='path to config for embedding computing',
        default=None,
    )

    args = parser.parse_args()

    #Config reading
    path_to_config = args.path_to_config
    with open(path_to_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    path_to_df = config['data_load']['path_to_dataframe']
    target_col = config['data_load']['target_col']

    pca_n_components = config['pca']['n_components']
    pca_is_std_scaler = config['pca']['is_standard_scaler']

    dataset_name = config['data_save']['data_name']
    path_to_save = config['data_save']['path_to_save']


    #Load data and apply PCA
    df = get_data_from_file(path_to_df)

    index_list = list(df.index)
    X, y = extract_features_target_from_df(df, target_col)
    X = compute_pca_embedding(
        X, 
        n_components=pca_n_components,
        is_std_scaler=pca_is_std_scaler)
    # labeled_embeddings = np.hstack([X, np.expand_dims(y, axis=1)])


    #Save embeddings
    path_to_save_embeddings = Path(path_to_save) / f'{dataset_name}_embeddings'
    path_to_save_embeddings = Path(path_to_save_embeddings)

    path_to_save_targets = Path(path_to_save) / f'{dataset_name}_targets'
    path_to_save_targets = Path(path_to_save_targets)

    if path_to_save_embeddings.is_dir():
        shutil.rmtree(path_to_save_embeddings)
        path_to_save_embeddings.mkdir(parents=True, exist_ok=True)
    else:
        path_to_save_embeddings.mkdir(parents=True, exist_ok=True)

    if path_to_save_targets.is_dir():
        shutil.rmtree(path_to_save_targets)
        path_to_save_targets.mkdir(parents=True, exist_ok=True)
    else:
        path_to_save_targets.mkdir(parents=True, exist_ok=True)    


    for idx, vec, target in zip(index_list, X, y):
        file_name = str(idx)

        file_path_embedding = Path(path_to_save_embeddings) / file_name
        file_path_embedding = Path(file_path_embedding)
        np.save(file_path_embedding, vec)

        file_path_target = Path(path_to_save_targets) / file_name
        file_path_target = Path(file_path_target)
        np.save(file_path_target, target)


    embeddings_config ={
        'data_name': dataset_name,
        'vectors_num': X.shape[0],
        'features_len': X.shape[1],
        'path_to_embeddings': str(path_to_save_embeddings),
        'path_to_targets': str(path_to_save_targets)
    }
    
    embeddings_config_name = f'{dataset_name}_config.yml'
    embeddings_config_path = Path(path_to_save) / embeddings_config_name
    embeddings_config_path = Path(embeddings_config_path)
    with open(embeddings_config_path, 'w') as f:
        yaml.dump(embeddings_config, f)
