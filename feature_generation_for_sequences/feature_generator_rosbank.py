import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np


from typing import Any, Dict, List, Union, Tuple
from pathlib import Path

PATH_TO_TRAIN_DATA = '/home/storage/priceseekers/data/rosbank/rosbank_dataset/train_part'
PATH_TO_TRAIN_DATA_SAVE = '/home/storage/priceseekers/data/rosbank/rosbank_dataset/train_part'

PATH_TO_TEST_DATA = '/home/storage/priceseekers/data/rosbank/rosbank_dataset/test_part'
PATH_TO_TEST_DATA_SAVE = '/home/storage/priceseekers/data/rosbank/rosbank_dataset/test_part' 

class DataFrameAggregator(object):
    """
    Class to create pipeline for feature generation by aggregation of 
    transaction history of clients from the Rosbank dataset
    """

    def __init__(
        self, 
        df_in_features: pd.DataFrame, 
        df_in_targets: pd.DataFrame, 
        idx_col: str,
        time_col: str,
        target_col: str,
        numerical_cols: List[str],
        cat_cols: List[str]
    ):
        """
        Initialize Aggregator.

        Parameters:
        ----------
            df_in_features: pd.DataFrame
                pd.DataFrame with clients transactions

            df_in_targets: pd.DataFrame
                pd.DataFrame with target class 

            idx_col: str
                Name of the column which contains client IDs

            time_col: str
                Name of the column which contains time variable

            target_col: str
                Name of the column which contains value of the target
                variable

            numerical_cols: (str)
                Names of the columns with numerical values
            
            cat_cols: (str)
                Names of the columns with categorical values
        """

        self.df_in_features = df_in_features.copy()
        self.df_in_targets = df_in_targets.copy()
        self.idx_col = idx_col
        self.time_col = time_col
        self.target_col = target_col

        self.df_in_features = self.df_in_features\
            .sort_values(by=[self.idx_col, self.time_col]) 

        self.df_in_targets = self.df_in_targets\
            .sort_values(self.idx_col)

        self.all_cols = list(self.df_in_features.columns)
        self.numerical_cols = numerical_cols
        self.binary_cols = None
        self.int_cols = None
        self.float_cols = None

        self.date_cols = None
        self.cat_cols = cat_cols

        #pd.DataFrame with aggregated features
        self.df_out_agg = None

    def transform(self):
        """
        Transform dataframe to generate aggregated 
        features
        """

        self._determine_columns_type()
        self._float_cols_agg()
        self._cat_cols_agg()
        self._join_out_agg_with_targets()


    def _determine_columns_type(self):
        """
        Divide the columns with numerical values into 
        columns with booleans, integer and float values  
        """

        df_in = self.df_in_features

        ser_by_numcols_nunique = df_in[self.numerical_cols]\
            .apply(lambda x: x.nunique())
        filtr = (ser_by_numcols_nunique == 2)
        self.binary_cols = list(ser_by_numcols_nunique.loc[filtr].index)

        ser_by_numcols_round = df_in[self.numerical_cols]\
            .apply(lambda x: (np.round(x) == x).sum() == \
                             x.notna().sum())
        self.int_cols = ser_by_numcols_round.loc[ser_by_numcols_round].index
        self.int_cols = [item for item in self.int_cols
                              if item not in self.binary_cols]
        
        self.float_cols = [item for item in self.numerical_cols
                                if (item not in self.int_cols) and \
                                   (item not in self.binary_cols)]

    def _float_cols_agg(self):
        """
        Calculate aggregated features from columns with float values
        """

        #Compute various aggregation values 
        df_in = self.df_in_features
        df_out = df_in.groupby(by=self.idx_col, as_index=False)[self.float_cols]\
            .agg(['min', 'max', 'mean', 'median', 'std', 'var'])

        df_out.fillna(0, inplace=True)
        df_out.columns = [f'{item[0]}_{item[1]}' for item in df_out.columns]

        #Compute mean of difference between neighboring values for each column
        df_in_diff = df_in[self.idx_col].to_frame()
        float_cols_diff_abs_mean = [f'{item}_diff_abs_mean' for item in self.float_cols]

        df_in_diff[float_cols_diff_abs_mean] = df_in\
            .groupby(by=self.idx_col)[self.float_cols]\
            .apply(lambda x: self._compute_diff_for_ser(x))\
            .fillna(0)

        df_out[float_cols_diff_abs_mean] = df_in_diff\
            .groupby(by=self.idx_col)[float_cols_diff_abs_mean]\
            .agg(lambda x: np.mean(np.abs(x)))

        #Save results of the calculations into the variable
        if self.df_out_agg is None:
            self.df_out_agg = df_out
        else:
            self.df_out_agg[df_out.columns] = df_out

    def _int_cols_agg(self):
        """
        Calculate aggregated features from columns with integer values
        """

        #Compute various aggregation values
        df_in = self.df_in_features
        int_cols_without_idx = [item for item in self.int_cols \
                                    if item != self.idx_col]

        df_out = df_in.groupby(by=self.idx_col,
                            as_index=False)[int_cols_without_idx]\
                    .agg([lambda x: pd.Series.mode(x)[0], 'max', 'min', 'nunique'])
        df_out.fillna(0, inplace=True)
        df_out.columns = [f'{item[0]}_{item[1]}' for item in df_out.columns]

        #Compute mean of difference between neighboring values for each column
        df_in_diff = df_in[self.idx_col].to_frame()
        int_cols_diff_abs_mean = [f'{item}_diff_abs_mean' for item in int_cols_without_idx]

        df_in_diff[int_cols_diff_abs_mean] = df_in\
            .groupby(by=self.idx_col)[int_cols_without_idx]\
            .apply(lambda x: self._compute_diff_for_ser(x))\
            .fillna(0)  

        df_out[int_cols_diff_abs_mean] = df_in_diff\
            .groupby(by=self.idx_col)[int_cols_diff_abs_mean]\
            .apply(lambda x: np.mean(np.abs(x)))

        #Save results of the calculations into the variable
        if self.df_out_agg is None:
            self.df_out_agg = df_out
        else:
            self.df_out_agg[df_out.columns] = df_out

    def _cat_cols_agg(self):
        """
        Calculate aggregated features from columns with categorical values
        """

        #Compute mode of values for each column
        df_in = self.df_in_features

        df_out = df_in.groupby(by=self.idx_col)[self.cat_cols]\
                    .agg([lambda x: pd.Series.mode(x)[0], 'nunique'])
        df_out.fillna(0, inplace=True)
        df_out.columns = [f'{item[0]}_{item[1]}' for item in df_out.columns]

        #Transform categorical features using One-Hot-Encoding
        cat_cols_lambda = [item for item in df_out.columns if item.find('lambda') != -1]
        cat_cols_unique = [item for item in df_out.columns if item not in cat_cols_lambda]

        df_dummies_list = [pd.get_dummies(df_out[cat_cols_lambda[i]], prefix=f'{cat_cols_lambda[i]}_') \
                            for i in range(len(cat_cols_lambda))]
        df_dummies_list.append(df_out[cat_cols_unique])

        df_out = pd.concat(df_dummies_list, axis=1)

        self.df_out_agg[df_out.columns] = df_out

        #Save results of the calculations into the variable
        if self.df_out_agg is None:
            self.df_out_agg = df_out
        else:
            self.df_out_agg[df_out.columns] = df_out

    def _join_out_agg_with_targets(self):
        """
        Join aggregated features with target variable by
        client ID
        """

        ser_target = self.df_in_targets.groupby(by=self.idx_col)[self.target_col]\
                                       .apply(lambda x: pd.Series.mode(x)[0])

        self.df_out_agg = self.df_out_agg.join(ser_target, on=self.idx_col)

    def _compute_diff_for_ser(self, x):
        """
        Compute difference between neighboring values of pd.Series
        """

        num_cols = x.shape[1]
        for i in range(0, num_cols):
            x.iloc[:, i] = x.iloc[:, i].diff()

        return x

if __name__ == '__main__':

    path_to_train_data = f'{PATH_TO_TRAIN_DATA}/rosbank_churn_train.parquet'
    df_train = pq.read_table(path_to_train_data).to_pandas()

    path_to_test_data = f'{PATH_TO_TEST_DATA}/rosbank_churn_test.parquet'
    df_test = pq.read_table(path_to_test_data).to_pandas()


    #aggregate features for train dataset
    df_train = df_train.drop_duplicates().reset_index(drop=True)

    obj_df_agg = DataFrameAggregator(
        df_train.drop(columns='target_target_flag'),
        df_train[['cl_id', 'target_target_flag']],
        idx_col='cl_id',
        time_col ='event_time',
        target_col = 'target_target_flag',
        numerical_cols=['amount'],
        cat_cols = ['mcc', 'channel_type', 'currency', 'trx_category']
    )

    obj_df_agg.transform()
    df_train_agg = obj_df_agg.df_out_agg

    #aggregate features for test dataset
    df_test = df_test.drop_duplicates().reset_index(drop=True)

    obj_df_agg = DataFrameAggregator(
        df_test.drop(columns='target_target_flag'),
        df_test[['cl_id', 'target_target_flag']],
        idx_col='cl_id',
        time_col ='event_time',
        target_col = 'target_target_flag',
        numerical_cols=['amount'],
        cat_cols = ['mcc', 'channel_type', 'currency', 'trx_category']
    )

    obj_df_agg.transform()
    df_test_agg = obj_df_agg.df_out_agg

    #add missing columns to test dataframe
    columns_to_add = list(set(df_train_agg.columns).difference(set(df_test_agg.columns)))

    if len(columns_to_add) > 0:
        df_test_agg[columns_to_add] = 0

    #equate columns for train and test datasets
    df_test_agg = df_test_agg[df_train_agg.columns]

    #save datasets with aggregated features
    path_to_save = Path(PATH_TO_TRAIN_DATA_SAVE)
    path_to_save.mkdir(parents=True, exist_ok=True)
    df_train_agg = pa.Table.from_pandas(df_train_agg)
    pq.write_table(df_train_agg, f'{path_to_save}/rosbank_train_features.parquet')
    
    path_to_save = Path(PATH_TO_TEST_DATA_SAVE)
    path_to_save.mkdir(parents=True, exist_ok=True)
    df_test_agg = pa.Table.from_pandas(df_test_agg)
    pq.write_table(df_test_agg, f'{path_to_save}/rosbank_test_features.parquet')
