## Data Configuration 

- **data_name** (*str*) is name which will be used to save checkpoints, logs, and files for filtration dataset.

- **data_filepath** (*str*) is path to dataset.

- **random_state** is used to provide reproducibility of computations

- **num_classes** is the number of classes in the target variable

- **features_dim** is number of features in input feature vectors. This number of features will be extracted from dataset files 
and returned by dataloader for model using.

Parameters to specify dataloaders.
[ https://pytorch.org/docs/1.10/data.html?highlight=torch%20utils%20data%20dataloader#torch.utils.data.DataLoader ]

- **num_workers** how many subprocesses to use for data loading.
0 means that the data will be loaded in the main process.

#pin_memory for dataloader
- **pin_memory** if `True`, the data loader will copy Tensors into CUDA pinned memory before returning them.

#size of batch for dataloader of training dataset
- **train_batchsize** how many samples per batch to load for training.

#size of batch for dataloader of dataset used for evaluation
- **eval_batchsize** how many samples per batch to load for validation and predictions.
