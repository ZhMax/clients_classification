## Configuration files for models used to create embeddings

### PCA method

Parameters are used to load dataset
- **data_load**
    - **path_to_dataframe** (*str*) is path to '.csv' or '.parquet' file containing dataset
    - **target_col** (*str*) is name of the column with the target variable

Parameters are used to save embeddings obtained by the PCA method 
- **data_save** 
    - **data_name** (*str*) is user defined name of the dataset which is used for saving
    - **is_save_in_files** (*str*) if it is `True`, then each embedding will be saved in separate file
    - **path_to_embeddings_dir** (*str*) is path to a directory where files with embeddings will be saved

Parameters are used for the PCA method
- **pca**
    - **is_standard_scaler** (*str*) if it is `True`, then values of features will be standardize before apply the PCA method
    - **n_components** (*int*, *float*) is the number of components in embeddings. If the *0 < n_components < 1* the number of components is selected such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.
