import logging
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F

from gpytorch.likelihoods import SoftmaxLikelihood

from prcskr.data_loaders.datautils import IndexedDataset, create_dataloader
from prcskr.models.due.src import dkl
from prcskr.models.due.src.fc_resnet import FCResNet
from prcskr.trainers.loss_fn.dueloss_elbo import ELBOLoss
from prcskr.trainers.trainer_dueforgetting import DueTrainerForgetting

from typing import Any, Dict, List, Union, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def model_due(
    train_dataset: IndexedDataset,
    fc_resnet_input_dim: int,
    fc_resnet_output_dim: int,
    fc_resnet_depth: int,
    is_fc_resnet_spectral_norm: bool, 
    fc_resnet_spectral_norm_coeff: float,
    fc_resnet_spectral_norm_n_power_iters: int,
    fc_resnet_dropout: float,
    gp_n_inducing_points: int,
    gp_num_classes: int,
    gp_kernel: Literal["RBF", "Matern12", "Matern32", "Matern52", "RQ"],
    likelihood_mixing_weights: bool
):
    """
    Initialization of the DUE model which includes the fully connected ResNet 
    to transform input feature vectors into embeddings and the Gaussian process 
    for estimation of predictions uncertainty.

    https://arxiv.org/abs/2102.11409, https://github.com/y0ast/DUE/tree/main
    
    Args:
        train_dataset: IndexedDataset
            Dataset which will be used to train the model.

        fc_resnet_input_dim: int
            The number of an input feature vector.
  
        fc_resnet_output_dim: int
            The dimension of embedding made by the 
            Fully Connected Residual Neural Net (fc_resnet).

        fc_resnet_depth: int
            The number of layers of the fc_resnet.

        is_fc_resnet_spectral_norm: bool
            Spectral normalization in the fc_resnet.

        fc_resnet_spectral_norm_coeff: float
            Coefficient of spectral normalization in the fc_resnet.

        fc_resnet_spectral_norm_n_power_iters: int
            The number of power iterations in the 
            spectral normalization.

        fc_resnet_dropout: float
            The dropout rate for a dropout layer in the fc_resnet.

        gp_n_inducing_points: int
            The number of inducing points for Gaussian 
            Process (gp).

        gp_num_classes: int
            The number of classes to be predicted by gp.
 
        gp_kernel: str
            Type of kernel in gp which can be 
            "RBF", "Matern12", "Matern32", "Matern52", or "RQ".

        likelihood_mixing_weights: bool
            Mixing weights in likelihood function for gp.

    Returns:
        model: nn.Module
            Model Due
    """

    #Initialization of Fully Connected Residual Neural Net
    feature_extractor = FCResNet(
        input_dim=fc_resnet_input_dim,
        features=fc_resnet_output_dim,
        depth=fc_resnet_depth,
        spectral_normalization=is_fc_resnet_spectral_norm,
        coeff=fc_resnet_spectral_norm_coeff,
        n_power_iterations=fc_resnet_spectral_norm_n_power_iters,
        dropout_rate=fc_resnet_dropout
    )
    
    #Initialization of Gaussian Process
    initial_inducing_points, initial_lengthscale = dkl.initial_values(
        train_dataset, feature_extractor, gp_n_inducing_points
    )
    gp = dkl.GP(
        num_outputs=gp_num_classes,
        initial_inducing_points=initial_inducing_points,
        initial_lengthscale=initial_lengthscale,
        kernel=gp_kernel
    )

    #Create Due model from fc_resnet and gp
    model = dkl.DKL(feature_extractor, gp)

    #Initialization of likelihood function
    likelihood = SoftmaxLikelihood(
        num_classes=gp_num_classes, 
        mixing_weights=likelihood_mixing_weights
    )

    return model, likelihood


def start_training(
    run_name: str,
    log_dir: str,
    ckpt_dir: str,
    ckpt_resume: str,
    train_dataset: IndexedDataset,
    val_dataset: IndexedDataset,
    random_state: int,
    total_epochs: int,
    lr: float,
    metrics_on_train: bool,
    is_forgetting: bool,
    dataconf: Dict[str, Any],
    modelconf: Dict[str, Any]
):
    """
    Initialization, training and validation the Due model

    If `is_forgetting = True`, forgetting masks are computed
    in the training and validation procedure.

    Args:
        run_name: str
            To distinguish trainer runs.

        log_dir: str = None
            Path to a directory where logs of model training will be saved. 
            The logs will be saved in a directory which will be created inside the log_dir.

        ckpt_dir: str = None
            Path to the directory, where checkpoints are saved.

        ckpt_resume: str = None
            Path to the checkpoint to resume model training. 
            If ckpt_resume is not `None`, a previous training process is
            continued by loading of a checkpoint. If a value of the `last_epoch` loaded 
            from the checkpoint is equal or larger to `total_epochs`, then
            all state dictionaries are loaded, but the training and validation procedures
            are not conducted.

        train_dataset: IndexedDataset
            Dataset for training the model

        val_dataset: IndexedDataset
            Dataset for validation the model

        random_state: int
            To provide reproducibility of computations.

        total_epochs: int
            A number of epochs for model training.

        lr: float
            Learning rate in an optimizer

        metrics_on_train: bool
            Is it neccessary to compute metrics on train set?

        is_forgetting: bool
            If the indicator is true, the masks required for computing 
            forgetting values of examples will be collected during training and 
            saved in checkpoint files.

        dataconf: Dict[str, Any]
            Data config

        modelconf: Dict[str, Any]
            Model config

    Returns:
        trainer: DueTrainerForgetting
            An object is used to train, validate and load model
    """

    #Setup logging
    ch = logging.StreamHandler()
    cons_lvl = getattr(logging, "warning".upper())
    ch.setLevel(cons_lvl)
    cfmt = logging.Formatter("{levelname:8} - {asctime} - {message}", style="{")
    ch.setFormatter(cfmt)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{run_name}.log"

    fh = logging.FileHandler(log_file)
    file_lvl = getattr(logging, "info".upper())
    fh.setLevel(file_lvl)
    ffmt = logging.Formatter(
        "{levelname:8} - {process: ^6} - {name: ^16} - {asctime} - {message}",
        style="{",
    )
    fh.setFormatter(ffmt)

    logger = logging.getLogger("event_seq")
    logger.setLevel(min(file_lvl, cons_lvl))
    logger.addHandler(ch)
    logger.addHandler(fh)


    #Determine Reproducibility parameters
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

    #Create dataloader for training and validation datasets
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=dataconf['train_batchsize'],
        num_workers=dataconf['num_workers'],
        is_pin_memory=dataconf['pin_memory'],
        random_state=random_state,
        is_shuffle=True)
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=dataconf['train_batchsize'],
        num_workers=dataconf['num_workers'],
        is_pin_memory=dataconf['pin_memory'],
        random_state=random_state,
        is_shuffle=False)   

    #Initialization the Due model
    model, likelihood = model_due(
        train_dataset,
        modelconf['fc_resnet']['input_dim'],
        modelconf['fc_resnet']['output_dim'],
        modelconf['fc_resnet']['depth'],
        modelconf['fc_resnet']['is_spectral_norm'], 
        modelconf['fc_resnet']['spectral_norm_coeff'],
        modelconf['fc_resnet']['spectral_norm_n_power_iters'],
        modelconf['fc_resnet']['dropout'],
        modelconf['gp']['n_inducing_points'],
        modelconf['gp']['num_classes'],
        modelconf['gp']['kernel'],
        modelconf['softmax_likelihood']['mixing_weights']
    )

    #Initialization Loss Function
    loss_fn = ELBOLoss(
        likelihood, 
        model.gp, 
        len(train_dataset)
    )

    #Initialization optimizer
    opt_params = [
        {'params': model.parameters(), 'lr': lr}
    ]
    opt_params.append(
        {'params': likelihood.parameters(), 'lr': lr}
    )
    optimizer = torch.optim.Adam(opt_params)

    # Define trainer and start train procedure
    trainer = DueTrainerForgetting(
        model=model,
        likelihood=likelihood,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        run_name=run_name,
        ckpt_dir=ckpt_dir,
        ckpt_replace=True,
        ckpt_resume=ckpt_resume,
        ckpt_track_metric=modelconf['ckpt_track_metric'],
        metrics_on_train=metrics_on_train,
        total_epochs=total_epochs,
        device=modelconf['device'],
        model_conf=modelconf,
        is_forgetting_during_training=is_forgetting
    )

    trainer.run()

    logger.removeHandler(fh)
    fh.close()   

    return trainer


def get_predictions(
    run_name: str,
    ckpt_resume: str,
    random_state: int, 
    dataset: IndexedDataset,
    dataconf: Dict[str, Any],
    modelconf: Dict[str, Any],
    is_forgetting: bool = False
):
    """
    Get predictions using the Due model loaded from a `*.ckpt` file

    Args:
        run_name: str
            To distinguish trainer runs.

        ckpt_resume: str
            Path to the checkpoint to load model for making predictions.

        dataset: IndexedDataset
            Dataset to load files for predicting. 
            True labels may be not included in the dataset. 

        random_state: int
            To provide reproducibility of computations.

        dataconf: Dict[str, Any]
            Data config to specify dataloaders

        modelconf: Dict[str, Any]
            Model config to specify model
            
        is_forgetting: bool = True
            If the indicator is true, the forgetting masks collected during training
            will be loaded from a checkpoint file.

    Returns: 
        preds: np.array
            Array with probabilities of classes for each example.

        uncertainties: np.array
            Array with uncertainty of a prediction for each example.

        dataset_inds: np.array
            Array with indices of examples in the dataset.

        file_name: List
            List of file names which were used to load 
            feature vectors of examples.

        gts: np.array 
            Array of true labels.
    """

    # Determine Reproducibility parameters
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

    #Create dataloader
    loader = create_dataloader(
        dataset,
        batch_size=dataconf['train_batchsize'],
        num_workers=dataconf['num_workers'],
        is_pin_memory=dataconf['pin_memory'],
        random_state=random_state,
        is_shuffle=False)

    #Initialization of the Due model
    model, likelihood = model_due(
        dataset,
        modelconf['fc_resnet']['input_dim'],
        modelconf['fc_resnet']['output_dim'],
        modelconf['fc_resnet']['depth'],
        modelconf['fc_resnet']['is_spectral_norm'], 
        modelconf['fc_resnet']['spectral_norm_coeff'],
        modelconf['fc_resnet']['spectral_norm_n_power_iters'],
        modelconf['fc_resnet']['dropout'],
        modelconf['gp']['n_inducing_points'],
        modelconf['gp']['num_classes'],
        modelconf['gp']['kernel'],
        modelconf['softmax_likelihood']['mixing_weights']
    )

    #Initialization of trainer
    trainer = DueTrainerForgetting(
        model=model,
        likelihood=likelihood,
        loss_fn=None,
        optimizer=None,
        train_loader=None,
        val_loader=loader,
        run_name=run_name,
        ckpt_dir=None,
        ckpt_replace=True,
        ckpt_resume=None,
        ckpt_track_metric=None,
        metrics_on_train=False,
        total_epochs=None,
        device=modelconf['device'],
        model_conf=modelconf,
        is_forgetting_during_training=is_forgetting
    )

    #Load model and other state dicts
    trainer.load_ckpt(ckpt_resume)

    #Make predictions
    preds_proba, uncertainties, dataset_inds, gts = trainer.predict(loader)

    preds_proba = torch.cat(preds_proba).cpu().numpy()
    uncertainties = torch.cat(uncertainties).cpu().numpy()
    dataset_inds = torch.cat(dataset_inds).cpu().numpy()

    file_names = [loader.dataset.get_file_name(idx) for idx in dataset_inds]
    file_names = np.asarray(file_names)

    if gts is not None:
        gts = torch.cat(gts).cpu().numpy()

    return preds_proba, uncertainties, dataset_inds, file_names, gts, trainer
