import sys
import yaml
from pathlib import Path
import shutil
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

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
            Name of the column with target variable

    Returns:
        X: np.array
            Array of feature vectors

        y: np.array
            Array with values of target variable
    """

    X = df.drop(columns=target_col).values
    y = df[target_col].values

    return X, y

def get_pca_embeddings(
    path_to_config: str
):
    """
    Method to fit PCA transformation, compute embeddings,
    and save embeddings and targets into` files 
    f'{dataset_name}_pca_embeddings.parquet' and 
    f'{dataset_name}_targets.parquet', where `dataset_name` 
    is given in the configuration file.

    Args:
        path_to_config: str
            Path to configuration file
    """

    with open(path_to_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    path_to_df_fit = config['data_load']['path_to_fit_dataframe']
    path_to_df_transform = config['data_load']['path_to_transform_dataframe']
    target_col = config['data_load']['target_col']

    pca_n_components = config['pca']['n_components']
    pca_is_std_scaler = config['pca']['is_standard_scaler']
    pca_random_state = config['pca']['random_state']

    dataset_name = config['data_save']['data_name']
    path_to_save = config['data_save']['path_to_save']


    #Load data and fit std_scaler and PCA
    df = pq.read_table(path_to_df_fit).to_pandas()
    X, y = extract_features_target_from_df(df, target_col)

    if pca_is_std_scaler:
        std_scaler = StandardScaler()
        std_scaler.fit(X)
        X = std_scaler.transform(X)

    model_PCA = PCA(
        n_components=pca_n_components,
        random_state=pca_random_state,
        svd_solver='full'
    )
    model_PCA.fit(X)

    #Load data and create embeddings by the PCA method
    df = pq.read_table(path_to_df_transform).to_pandas()
    index_list = list(df.index)
    X, y = extract_features_target_from_df(df, target_col)

    if pca_is_std_scaler:
        X = std_scaler.transform(X)

    X = model_PCA.transform(X)

    #Save embeddings and targets
    path_to_save_embeddings = Path(path_to_save)
    path_to_save_embeddings.mkdir(parents=True, exist_ok=True)
    file_path_embeddings = Path(path_to_save_embeddings) / f'{dataset_name}_pca_embeddings.parquet'

    path_to_save_targets = Path(path_to_save)
    path_to_save_targets.mkdir(parents=True, exist_ok=True)
    file_path_targets = Path(path_to_save_embeddings) / f'{dataset_name}_targets.parquet'

    df_to_save = pd.DataFrame(data=X, index=index_list)
    df_to_save = pa.Table.from_pandas(df_to_save)
    pq.write_table(df_to_save, file_path_embeddings)

    df_to_save = pd.DataFrame(data=y, index=index_list)
    df_to_save = pa.Table.from_pandas(df_to_save)
    pq.write_table(df_to_save, file_path_targets)

    embeddings_config ={
        'data_name': dataset_name,
        'vectors_num': X.shape[0],
        'features_dim': X.shape[1],
        'path_to_embeddings': str(path_to_save_embeddings),
        'path_to_targets': str(path_to_save_targets)
    }
    
    embeddings_config_name = f'{dataset_name}_config.yml'
    embeddings_config_path = Path(path_to_save) / embeddings_config_name
    embeddings_config_path = Path(embeddings_config_path)
    with open(embeddings_config_path, 'w') as f:
        yaml.dump(embeddings_config, f)


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

    #compute PCA embeddings
    get_pca_embeddings(path_to_config)
