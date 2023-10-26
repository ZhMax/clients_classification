# Model for clients classification
This repository contains code to conduct classification of clients based on their demographic features, history of interaction with a company, purchases, and transactions. <br> The classification problem is solved using a model which combines fully connected ResNet and Gaussian process. The model is capable to estimate uncertainty of its predictions. Therefore, the predictions with high uncertainty can be excluded to achive more reiliable results. Also methods for filtration noisy (mislabeled) examples from dataset are proposed.  

`feature_generation` contains code to create aggregated features from historical data of clients.

`preprocessor` contains code to create embeddings from aggreagated features through the PCA method.

`srcprcskr` contains code to create classification model and perform filtration of dataset. 

`notebooks` contains jupyter notebook for demonstration purpose.

## INSTALLATION GUIDE

```bash
git clone https://gitlab.appliedai.tech/priceseekers/core/priceseekers

cd priceseekers

pip3 install -r requirements.txt 
pip3 install .
```

## DOCUMENTATION

### FilteredClassifier
Main class to solve classification problems and filtrate noisy examples from dataset.

```python
classifier = FilteredClassifier(
    run_name="user_run_name",
    log_dir="./logs", 
    ckpt_dir="./ckpt_dir",
    path_to_dataconf="sber_ps/configs/data/data_config.yml",
    path_to_modelconf="sber_ps/configs/models/modeldue_config.yml"
)
```

- **run_name** (*str*) is used to distinguish classifier runs

- **path_to_dataconf** (*str*) is path to a `.yml` data configuration file.

- **path_to_modelconf** (*str*) is path to a `.yml` model configuration file.

- **log_dir** (*str*) is path to a directory where logs of model training will be saved. 
The logs will be saved in a directory which will be created inside the log_dir.

- **ckpt_dir** (*str*) is path to a directory where checkpoints of state dictionaries (e.g model) 
will be saved during training. The checkpoints will be saved in a directory which will be created 
inside the ckpt_dir.

#### Fit
A method to train a model for solving classification problem. The implemetation is
performed for the Deterministic Uncertainty Estimation (DUE) model, which was proposed in

[https://arxiv.org/abs/2102.11409], [https://github.com/y0ast/DUE/tree/main]

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

```python
classifier.fit(
    data_filepath='data/embeddings/',
    split_frac_train_val=0.8,
    random_state=None,
    total_epochs=None,
    lr=None,
    path_to_file_names_to_be_excluded=None,
    is_forgetting=False,
    metrics_on_train=False,
    ckpt_resume=None
)
```

- **data_filepath** (*str*) is path to a directory which contains files with features and labels. 
It is supposed that each file contains vector consisting of features and a label of 
an example (in the last component).

- **split_frac_train_val** (*float*) is fraction of training part size from the size of the full dataset. 
The value 1.0 spicifies that the full dataset will be used for training.

- **random_state** (*int*) is used to provide reproducibility of computations. 
If it is `None`, a value  from the field `random_state` from the data configuration file 
will be used.

- **total_epochs** (*int*) is a number of epochs for model training. 
If it is `None`, a value from the field `total_epochs` of the model configuration file 
will be used.

- **lr** (*float*) is learning rate in an optimizer. If it is `None`, a value from
the field `lr` of the model configuration file will be used.

- **path_to_file_names_to_be_excluded** (*str*) is path to a `.txt` file which contains names of files 
to be excluded from the original dataset for training.

- **is_forgetting** (*bool*) inidicates that the masks required for computing 
forgetting counts of examples will be collected during training and saved in checkpoint files.  

- **metrics_on_train** (*bool*) inidicates that the metrics will be computed metrics on training dataset.

- **ckpt_resume** (*str*) is a path to a checkpoint file `*.ckpt` which is used to load the model.
It should be `None` to train a model from an initial state


#### Filtration dataset by the forgetting method
An implementation of a method to find noisy examples (mislabeled examples) in dataset by
counting the forgetting of examples during training. The method is named as the forgetting 
method. Firstly it was proposed in

[https://arxiv.org/abs/1812.05159], [https://github.com/mtoneva/example_forgetting/tree/master]

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

```python
df_examples = classifier.filtration_by_forgetting(
    data_filepath='data/embeddings/',
    example_forgetting_dir=None,
    threshold_val=None,
    random_state=None,
    total_epochs=None,
    lr=None,
    verbose=True,
    ckpt_resume=None,
    path_to_file_names_to_be_excluded=None
)
```

- **data_filepath** (*str*) is path to a directory which contains files with features and labels. 
It is supposed that each file contains vector consisting of an embedding and a label of an example (in the last component).

- **example_forgetting_dir** (*str*) is path to a directory which will be used to save array with file names 
containing noisy labels. If it is `None`, then the directory with name `f"{data_name}_forgetting"` will be created in the parent of the directory `data_filepath`. The field `data_name` is provided by the data configuration file.

- **threshold_val** (*int*) is the threshold value for `forgetting_counts`, which can be used to filtrate examples.
If it is `None`, only unlearned examples will be proposed for excluding from the dataset. 

- **random_state** (*int*) is used to provide reproducibility of computations. If it is `None`, a value  
from the field `random_state` from the data configuration file will be used.

- **total_epochs** (*int*) is a number of epochs for model training. If it is `None`, a value  
from the field `total_epochs` of the model configuration file will be used.

- **lr** (*float*) is learning rate in an optimizer. If it is `None`, a value from
the field `lr` of the model configuration file will be used.

- **verbose** (*bool*) indicates that pd.DataFrame with forgetting counts for examples will be returned.

- **ckpt_resume** (*str*) is path to a checkpoint file `*.ckpt` which is used to load the model
and masks collected during training. It should be `None` to train a model from an initial state.     

- **path_to_file_names_to_be_excluded** (*str*) is path to a `.txt` file which contains names of files 
to be excluded from the original dataset for training.

*Returns:*
- **df_examples** (*pd.DataFrame*) contains forgetting counts for examples of 
the dataset and predictions given by the trained model 


#### Filtration dataset by the second-split forgetting method
An implementation of a method to find noisy examples (mislabeled examples) in dataset by 
sequential model training on its parts and counting forgetting of the examples. The method 
is named as the second-spit forgetting method. It was proposed in

[https://arxiv.org/abs/2210.15031], [https://github.com/pratyushmaini/ssft]


One of the disadvantage of the forgetting method for filtration of dataset is that the set 
of unlearned and frequently forgetting examples can include complex examples. The complex 
examples are placed close to the boundary between different classes. Therefore, such examples
contribute to improve model training. To separate noisy examples from the complex ones, the
following algorithm was implemented.
1) The full dataset is divided into two halves, which we will name as the first part and
the second part.

2) The model is trained on the first part of the dataset until the values of the loss functions 
or the tracked metric stabilize (the first training). Then the model training continues 
on the second part of the dataset (the second training).

3) Examples from the second part of the dataset, which were forgotten after one epoch of 
the second training, are marked as noisy. 

4) Then the model is trained from the initial state on the second and the first half parts of the dataset. 
The examples forgotten after one epoch of the model training on the first half of the dataset are marked as noisy.

5) Array of file names of the dataset containing noisy examples are saved in `.txt` file
and can be used for excluding from the dataset in the next model trainings. By varying 
`threshold_val`, examples, which were forgotten after larger number of epoch of the subsequent
training, can be excluded.

```python
df_examples = classifier.filtration_by_second_split_forgetting(
    data_filepath='data/embeddings/',
    example_forgetting_dir=None,
    threshold_val=None,
    random_state=None,
    total_epochs_per_step=None,
    lr=None,
    verbose=True,
    ckpt_resume=None,
    path_to_file_names_to_be_excluded=None
)
```

- **data_filepath** (*str*) is path to a directory which contains files with features and labels. 
It is supposed that each file contains vector consisting of an embedding and a label of an example (in the last component).

- **example_forgetting_dir** (*str*) is path to a directory which will be used to save array with file names 
containing noisy labels. If it is `None`, then the directory with name `f"{data_name}_forgetting"` will be created in the parent of the directory `data_filepath`. The field `data_name` is provided by the data configuration file.

- **threshold_val** (*int*) is the threshold value for `epoch_forget_forever`, which can be used to filtrate examples.
If it is `None`, only examples which are forgotten after one epoch of the next 
training step will be proposed for excluding from the dataset.

- **random_state** (*int*) is used to provide reproducibility of computations. If it is `None`, a value  
from the field `random_state` from the data configuration file will be used.

- **total_epochs** (*int*) is a number of epochs for model training. If it is `None`, a value  
from the field `total_epochs` of the model configuration file will be used.

- **lr** (*float*) is learning rate in an optimizer. If it is `None`, a value from
the field `lr` of the model configuration file will be used.

- **verbose** (*bool*) indicates that pd.DataFrame with forgetting counts for examples will be returned.

- **ckpt_resume** (*str*) is path to a checkpoint file `*.ckpt` which is used to load the model
and masks collected during training. It should be `None` to train a model from an initial state.     

- **path_to_file_names_to_be_excluded** (*str*) is path to a `.txt` file which contains names of files 
to be excluded from the original dataset for training.

*Returns:*
- **df_examples** (*pd.DataFrame*) contains the number of the epoch when each examples of 
the dataset were forgotten forever and predictions given by the trained 
model at the second and fourth training steps.


#### Predict
Method to get predictions using a model loaded from a checkpoint file.

```python
preds_proba, uncertainties, file_names, true_labels = classifier.predict(
    data_filepath='data/embeddings/',
    ckpt_resume='ckpt_dir/data_name_fit/epoch: 0120 - acc_score: 0.7514 - roc_auc_score: 0.749 - loss: 0.3942.ckpt',
    random_state=None,
    is_target_col=True,
    path_to_file_names_to_be_excluded='data/data_name_second_forgetting/data_name_files_to_be_excluded.txt'
)
```

- **data_filepath** (*str*) is path to a directory which contains files with features. 
If the files also contain labels, they should be in the last component of the vectors.

- **ckpt_resume** (*str*) is path to a checkpoint file `*.ckpt` which is used to load the model.

- **random_state** (*int*) is uded to provide reproducibility of computations. If it is `None`, a value  
from the field `random_state` from the data configuration file will be used.

- **is_target_col** (*bool*) indicates that a value of a target variable is included into to the input vector.

- **path_to_file_names_to_be_excluded** (*str*) is path to a `.txt` file which contains names of files 
to be excluded from the original dataset for training.

