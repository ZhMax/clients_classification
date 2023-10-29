import pandas as pd
import numpy as np

from typing import Any, Dict, List, Union, Tuple
from pathlib import Path

PATH_TO_DATA = '/home/rosbank_data'
PATH_TO_SAVE = '/home/storage/projects/sber_project_priceseekers/data/rosbank'  

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

    def determine_columns_type(self):
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

    def float_cols_agg(self):
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

    def int_cols_agg(self):
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

    def cat_cols_agg(self):
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

    def join_out_agg_with_targets(self):
        """
        Join aggregated features with target variable by
        client ID
        """

        ser_target = self.df_in_targets.groupby(by=self.idx_col)['target']\
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

    path_to_train_data = f'{PATH_TO_DATA}/train.csv.zip'
    df_train = pd.read_csv(path_to_train_data)

    path_to_test_data = f'{PATH_TO_DATA}/test.csv.zip'
    df_test = pd.read_csv(path_to_test_data)

    path_to_save = Path(PATH_TO_SAVE)
    path_to_save.mkdir(parents=True, exist_ok=True)

    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_full = df_full.drop_duplicates().reset_index(drop=True)

    obj_df_agg = DataFrameAggregator(
        df_full.drop(columns='target'),
        df_full[['index', 'target']],
        idx_col='index',
        time_col ='time',
        target_col = 'target',
        numerical_cols=['amount'],
        cat_cols = ['mcc', 'channel_type', 'currency', 'trx_category']
    )

    obj_df_agg.determine_columns_type()
    obj_df_agg.float_cols_agg()
    obj_df_agg.cat_cols_agg()
    obj_df_agg.join_out_agg_with_targets()

    obj_df_agg.df_out_agg.to_parquet(f'{path_to_save}/rosbank_features.parquet')
