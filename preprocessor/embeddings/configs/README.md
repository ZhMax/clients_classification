## Configuration files for models used to create embeddings

### PCA method

Parameters are used to load dataset
- **data_load**
    - **path_to_fit_dataframe** (*str*) is path to load dataframe which is used to fit the PCA method.
    - **path_to_transform_dataframe** (*str*) is #path to load dataframe for which embeddings by the PCA method will be obtained.
    - **target_col** (*str*) is name of the column with the target variable.

Parameters are used to save embeddings obtained by the PCA method 
- **data_save** 
    - **data_name** (*str*) is user defined name of the dataset which is used for saving.
    - **is_save_in_files** (*str*) if it is `True`, then each embedding will be saved in separate file.
    - **path_to_save** (*str*) is path to a directory where directories for files with embeddings and targets will be created.

Parameters are used for the PCA method
- **pca** 
    - **is_standard_scaler** (*str*) if it is `True`, then values of features will be standardize before apply the PCA method.
    - **random_state** (*int*) provides reproducibility for different runs.
    - **n_components** (*int*, *float*) is the number of components in embeddings. If the *0 < n_components < 1* the number of components. is selected such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.
