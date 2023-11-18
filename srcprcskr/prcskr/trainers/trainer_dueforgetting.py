import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import gpytorch
import gpytorch.likelihoods
import gpytorch.distributions

from tqdm.autonotebook import tqdm

from typing import Any, Dict, List, Union, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_recall_curve, 
    auc
)

logger = logging.getLogger("event_seq")


class _CyclicalLoader:
    """Cycles through pytorch dataloader specified number of steps."""

    def __init__(self, base_dataloader):
        self.base_loader = base_dataloader
        self._len = None
        self._iter = iter(self.base_loader)

    def set_iters_per_epoch(self, iters_per_epoch: int):
        self._len = iters_per_epoch

    def __len__(self):
        return self._len

    def __iter__(self):
        self._total_iters = 0
        return self

    def __next__(self):
        assert self._len, "call `set_iters_per_epoch` before use"

        if self._total_iters >= self._len:
            raise StopIteration

        try:
            item = next(self._iter)
        except StopIteration:
            self._iter = iter(self.base_loader)
            item = next(self._iter)
        self._total_iters += 1
        return item
    
def _grad_norm(params):
    """
    Calculate gradient norm for logger.debug
    """
    
    total_sq_norm = 0.0
    for p in params:
        param_norm = p.grad.detach().data.norm(2)
        total_sq_norm += param_norm.item() ** 2
    return total_sq_norm**0.5

class DueTrainerForgetting():
    """Class with Trainer for Due Model"""

    def __init__(        
        self,
        *,
        model: nn.Module,
        likelihood: gpytorch.likelihoods,
        loss_fn: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        run_name: str = None,
        total_epochs: int = None,
        iters_per_epoch: int = None,
        ckpt_dir: str = None,
        ckpt_replace: bool = False,
        ckpt_track_metric: str = "epoch",
        ckpt_resume: str = None,
        device: str = "cpu",
        metrics_on_train: bool = False,
        model_conf: Dict[str, Any] = None,
        data_conf: Dict[str, Any] = None,
        is_forgetting_during_training: bool = False
    ):
        
        """
        Initialize trainer.

        Parameters:
        ----------
            model: nn.Module
                Due model to train or validate.

            likelihood: gpytorch.likelihoods
                Liklelihood function for Gaussian Process.

            loss_fn: nn.Module = None
                Loss function.

            optimizer: torch.optim.Optimizer = None
                Torch optimizer for training.

            lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None
                Torch learning rate scheduler.

            train_loader: DataLoader = None
                Dataloader for training dataset.

            val_loader: DataLoader = None
                Dataloader for validation dataset.

            run_name: str = None
                To distinguish trainer runs.

            total_epochs: int = None
                Total number of epoch to train a model.

            total_iters: int = None
                Total number of iterations to train a model.

            iters_per_epoch: int = None
                Validation and checkpointing are performed every
                `iters_per_epoch` iterations.

            ckpt_dir: str = None
                Path to the directory, where checkpoints are saved.

            ckpt_replace: bool = False
                If `replace` is `True`, only the last and the best checkpoint
                are kept in `ckpt_dir`.

            ckpt_track_metric: str = None
                If `ckpt_replace` is `True`, the best checkpoint is
                determined based on `track_metric`. All metrcs except loss are assumed
                to be better if the value is higher.

            ckpt_resume: str = None
                Path to the checkpoint to resume model training.

            device: str = 'cpu'
                Device to be used for model training and validation.

            metrics_on_train: bool = False
                Is it neccessary to compute metrics on train set?

            model_conf: Dict[str, Any] = None
                Model config

            data_conf: Dict[str, Any] = None
                Data config

            is_forgetting_during_training:
                Is it neccessary to compute masks for computing forgetting of 
                examples during `model` training?
        """

        self._run_name = (
            run_name if run_name is not None else datetime.now().strftime("%F_%T")
        )

        self._total_epochs = total_epochs
        self._iters_per_epoch = iters_per_epoch
        self._total_iters = None
        self._ckpt_dir = ckpt_dir
        self._ckpt_replace = ckpt_replace
        self._ckpt_track_metric = ckpt_track_metric
        self._ckpt_resume = ckpt_resume
        self._device = device
        self._metrics_on_train = metrics_on_train
        self._model_conf = model_conf
        self._data_conf = data_conf

        self._model = model
        self._model.to(device)
        self._likelihood = likelihood
        self._likelihood.to(device)

        self._loss_fn = loss_fn
        self._opt = optimizer
        self._sched = lr_scheduler

        self._train_loader = train_loader
        if train_loader is not None:
            self._cyc_train_loader = _CyclicalLoader(train_loader)
        self._val_loader = val_loader

        self._metric_values = None
        self._loss_values = None
        self._last_iter = 0
        self._last_epoch = 0

        self._is_forgetting_during_training = is_forgetting_during_training
        if self._is_forgetting_during_training:            
            #Dict for saving masks of examples in different runs 
            # of the trainer
            self._forgetting_dict = {}

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def likelihood(self) -> gpytorch.likelihoods:
        return self._likelihood
    
    @property
    def loss_fn(self) -> nn.Module:
        return self._loss_fn

    @property
    def train_loader(self) -> DataLoader:
        return self._train_loader

    @property
    def val_loader(self) -> DataLoader:
        return self._val_loader

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._opt

    @property
    def lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self._sched
    
    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def device(self) -> str:
        return self._device
    
    @property
    def forgetting_dict(self) -> Dict[str, Any]:
        return self._forgetting_dict
    

    def save_ckpt(self, ckpt_path: str = None):
        """
        Save model, optimizer and scheduler states.

        Args:
            ckpt_path: str = None
                Path to checkpoints. If `ckpt_path` is a directory, the
                checkpoint will be saved there with epoch, loss an metrics in the
                filename. All scalar metrics returned from `compute_metrics` are used to
                construct a filename. If full path is specified, the checkpoint will be
                saved exactly there. If `None` `ckpt_dir` from construct is used with
                subfolder named `run_name` from Trainer's constructor.
        """

        #Create a directory for saving checkpoints
        if ckpt_path is None and self._ckpt_dir is None:
            logger.warning(
                "`ckpt_path` was not passned to `save_ckpt` and `ckpt_dir` "
                "was not set in Trainer. Checkpoints will not be saved."
            )
            return

        if ckpt_path is None:
            assert self._ckpt_dir is not None
            ckpt_path = Path(self._ckpt_dir) / self._run_name

        ckpt_path = Path(ckpt_path)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        #Saving weights and forgetting masks of examples
        # in a checkpoint
        ckpt = {
            "last_iter": self._last_iter,
            "last_epoch": self._last_epoch,
        }
        if self._model:
            ckpt["model"] = self._model.state_dict()
        if self._opt:
            ckpt["opt"] = self._opt.state_dict()
        if self._sched:
            ckpt["sched"] = self._sched.state_dict()
        if self._is_forgetting_during_training:
            ckpt["forgetting_dict"] = self._forgetting_dict

        if not ckpt_path.is_dir():
            torch.save(ckpt, ckpt_path)

        assert self._metric_values
        assert self._loss_values

        metrics = {k: v for k, v in self._metric_values.items() if np.isscalar(v)}
        metrics["loss"] = np.mean(self._loss_values)

        fname = f"epoch: {self._last_epoch:04d}"
        metrics_str = " - ".join(f"{k}: {v:.4g}" for k, v in metrics.items())
        if len(metrics_str) > 0:
            fname = " - ".join((fname, metrics_str))
        fname += ".ckpt"

        torch.save(ckpt, ckpt_path / Path(fname))

        if not self._ckpt_replace:
            return

        #Remove all checkpoints with the exception of
        # checkpoints at the last epoch and with the high
        # `self._ckpt_track_metric` on the validation dataset
        def make_key_extractor(key):
            def key_extractor(p: Path) -> float:
                metrics = {}
                for it in p.stem.split(" - "):
                    kv = it.split(": ")
                    assert len(kv) == 2, f"Failed to parse filename: {p.name}"
                    k = kv[0]
                    v = -float(kv[1]) if "loss" in k else float(kv[1])
                    metrics[k] = v
                return metrics[key]

            return key_extractor

        all_ckpt = list(ckpt_path.glob("*.ckpt"))
        last_ckpt = max(all_ckpt, key=make_key_extractor("epoch"))
        best_ckpt = max(all_ckpt, key=make_key_extractor(self._ckpt_track_metric))
        for p in all_ckpt:
            if p != last_ckpt and p != best_ckpt:
                p.unlink()



    def load_ckpt(self, ckpt_fname: str):
        """
        Load model, optimizer and scheduler states.

        Args:
            ckpt_fname: str
                Path to checkpoint.
        """

        ckpt = torch.load(ckpt_fname)

        if "model" in ckpt:
            self._model.load_state_dict(ckpt["model"])

        if "opt" in ckpt:
            if self._opt is not None:
                self._opt.load_state_dict(ckpt["opt"])

        if "sched" in ckpt:
            if self._sched is not None:
                self._sched.load_state_dict(ckpt["sched"])
        if "forgetting_dict" in ckpt:
            if self._is_forgetting_during_training:
               self._forgetting_dict = ckpt["forgetting_dict"]           

        self._last_iter = ckpt["last_iter"]
        self._last_epoch = ckpt["last_epoch"]


    def output_transform(
        self, 
        output: gpytorch.distributions
    ):
        """
        Compute probabilities from MultivariateNormal distribution
        """
        
        output_tr = output.to_data_independent_dist()
        output_tr = self._likelihood(output_tr).probs.mean(0)
        return output_tr

    def predict(
        self, 
        loader: DataLoader
    ):
        """
        Get predictions of Due model

        Args:
            loader: DataLoader
                Dataloader of a dataset.
        
        Returns:
            preds: List[torch.tensor]
                Probabilities of classes for examples from a dataset.

            uncertainties: List[torch.tensor]
                Uncertainties of predictions given by a model.

            dataset_inds: List[torch.tensor]
                Indices of examples in a dataset.

            gts: List[torch.tensor]
                True labels of examples from a dataset. If true labels are
                absent in the dataset, `None` is returned.
        """

        self._model.eval()
        self._likelihood.eval()
        preds, uncertainties, dataset_inds, gts = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(loader):
                if len(batch) == 3:
                    inp, gt, inds = batch
                elif len(batch) == 2:
                    inp, inds = batch
                    gt = None
                else:
                    raise ValueError('A number of arrays in batch is unexpected')
                
                inp = inp.to(self._device)
                
                #Use the Monte-Carlo method to compute model prediction
                with gpytorch.settings.num_likelihood_samples(128):
                    #Compute a vector of classes probabilites
                    output = self._model(inp)
                    pred = self.output_transform(output)

                    #Compute entropy as an uncertainty of the model prediction
                    uncertainty = -(pred * pred.log()).sum(axis=1)
                
                preds.append(pred)
                uncertainties.append(uncertainty)
                dataset_inds.append(inds)


                if gt is not None:
                    gt = gt.to(self._device)
                    gts.append(gt)

        if len(gts) == 0:
            gts = None

        return preds, uncertainties, dataset_inds, gts
    

    def compute_metrics(
        self,
        model_preds: List[torch.tensor],
        ground_truths: List[torch.tensor]
    ):
        """
        Compute metrics based on model output.

        The method is used to compute model metrics for further logging
        and checkpoint tracking. Any metrics could be logged, but only scalar metrics
        can be used to track checkpoints.

        Args:
            model_outputs: List[torch.tensor]
                Model outputs obtained in training or validation stage.

            ground_truths: List[torch.tensor]
                True labels of examples from a dataset.

        Returns: 
            metrics_dict: Dict[str, float]
                A dict combines metric names and values.
        """

        metrics_dict = {}
        
        #Concatenation of predictions and true labels obtained
        # in different batches
        model_preds = torch.cat(model_preds)
        model_preds_proba = model_preds[:, 1]
        model_preds_label = torch.argmax(model_preds, axis=1)
        model_preds_proba = model_preds_proba.cpu().numpy()
        model_preds_label = model_preds_label.cpu().numpy()

        ground_truths = torch.cat(ground_truths)
        ground_truths = ground_truths.cpu().numpy()

        #Compute metrics using sklearn functions
        metrics_dict['acc_score'] = accuracy_score(ground_truths, model_preds_label)
        metrics_dict['roc_auc_score'] = roc_auc_score(ground_truths, model_preds_proba)

        precision, recall, _ = precision_recall_curve(
            ground_truths, model_preds_proba
        )
        pr_recall_auc = auc(recall, precision)
        metrics_dict['pr_auc_score'] = pr_recall_auc

        return metrics_dict
    

    def log_metrics(
        self,
        phase: Literal['train', 'val'],
        metrics: Dict[str, Any] = None,
        epoch: int = None,
    ):
        """
        Log metrics.

        The metrics are computed based on the whole epoch data, so the granularity of
        metrics is epoch. When the metrics are not None, the epoch is not None either.
        The metrics can be computed in training and validation phase. The loss is computed
        only during train phase to report the validation loss, compute it in the 
        `compute_metrics` function.

        Args:
            phase: str
                It indicates, the metrics were collected during training phase (value 'train') or
                validation phase (value 'val').

            metrics: Dict['str', Any]
                A Dict is returned by `compute_metrics` every epoch.
            
            epoch: int 
                The number of an epoch when the metrics were computed.
        """

        if metrics is not None:
            logger.info(f"Epoch: {epoch}; metrics on {phase}: {metrics}")


    def compute_batch_masks(
        self, 
        batch_inds: torch.tensor,
        pred_proba: torch.tensor,
        gt: torch.tensor,
        mask: torch.tensor,
        conf: torch.tensor
    ):
        """
        Compute masks from examples from a batch of the training or validation dataset.

        The size of the `mask` and `conf` array is equal to the size of a dataset. 
        If the class predicted by the model for an example is equal to the true class, 
        then the element of the `mask` array, which index is the same as the example 
        in the dataset, is equal to 1. In the other case, the element is equal to 0.
        The `conf` array contains probabilities predicted by the model for the true
        class of an example.

        Args:
            batch_inds: torch.tensor
            Indices in the dataset of examples from the batch
            
            pred_proba: torch.tensor
            Probabilities of classes of examples from the batch

            gt: torch.tensor
            True labels of examples from the batch

            mask: torch.tensor
            An array whose elements indicate whether the class pridected 
            by the model is equal to the true class of an example

            conf: torch.tensor
            An array whose elements are equal to probabilites given by the model
            for true class of an example
        """

        pred_label = torch.argmax(pred_proba, axis=1)

        correct_mask = pred_label.eq(gt)
        conf_mask = pred_proba[torch.arange(gt.shape[0]), gt.long()]
        mask[batch_inds.squeeze(-1)] = correct_mask.float().cpu()
        conf[batch_inds.squeeze(-1)] = conf_mask.float().cpu().clone().detach()

        return mask, conf


    def predict_on_val(
        self, 
        loader: DataLoader
    ):  
        """
        Get predictions for the validation dataset and compute forgetting masks 
        for examples if `self._is_forgetting_during_training = True`

        Args:
            loader: DataLoader
                Dataloader for the validation dataset

        Returns:
            preds: List[torch.tensor]
                Probabilities of classes for examples from a dataset

            uncertainties: List[torch.tensor]
                Uncertainties of predictions given by a model

            gts: List[torch.tensor]
                True labels of examples from a dataset.
        """
        
        self._model.eval()
        self._likelihood.eval()
        preds, uncertainties, gts = [], [], []

        #forgetting masks for examples of the validation dataset
        if self._is_forgetting_during_training:
            mask = torch.zeros(len(self._val_loader.dataset))
            conf = torch.zeros(len(self._val_loader.dataset))

        with torch.no_grad():
            for inp, gt, inds in tqdm(loader):
                inp = inp.to(self._device)
                gt = gt.to(self._device)
                
                #Use Monte-Carlo method to compute model prediction
                with gpytorch.settings.num_likelihood_samples(128):
                    #compute a vector of classes probabilites for
                    # examples from the batch
                    output = self._model(inp)
                    pred = self.output_transform(output)

                    #compute entropy as uncertainty of the model prediction
                    uncertainty = -(pred * pred.log()).sum(axis=1)
                
                #Compute forgetting masks
                if self._is_forgetting_during_training:
                    mask, conf = self.compute_batch_masks(
                        inds, pred, gt, mask, conf)
                
                preds.append(pred)
                uncertainties.append(uncertainty)
                gts.append(gt)
            
            #Append forgetting masks obtained in the epoch to
            # lists containing the masks for all training process
            if self._is_forgetting_during_training:
                self._mask_list_tr.append(mask.unsqueeze(0))
                self._conf_list_tr.append(conf.unsqueeze(0))

        return preds, uncertainties, gts


    def train(
        self, 
        iters: int
    ):
        """
        Train the model for one epoch. 
        If `self._is_forgetting_during_training = True`, forgetting masks 
        are collected. 

        Args:
            iters: int
                The number of iterations per an epoch
        """
        
        assert self._opt is not None, "Set an optimizer first"
        assert self._train_loader is not None, "Set a train loader first"

        logger.info("Epoch %04d: train started", self._last_epoch + 1)
        
        self._model.train()
        self._likelihood.train()

        loss_ema = 0.0
        losses = []
        preds, gts = [], []

        #forgetting masks for examples of the training dataset
        if self._is_forgetting_during_training:
            mask_after_opt = torch.zeros(len(self.train_loader.dataset))
            conf_after_opt = torch.zeros(len(self.train_loader.dataset))

        #training procedure
        pbar = tqdm(zip(range(iters), self._cyc_train_loader), total=iters)
        for i, (inp, gt, inds) in pbar:
            inp, gt = inp.to(self._device), gt.to(self._device)

            #compute loss
            pred = self._model(inp)
            loss = self._loss_fn(pred, gt)
            loss.backward()

            loss_np = loss.item()

            #compute predictions for a batch of the training dataset
            if self._metrics_on_train or self._is_forgetting_during_training:
                pred_proba = self.output_transform(pred)
                preds.append(pred_proba.detach())
                gts.append(gt)

            #compute forgetting masks for a batch of the training dataset
            if self._is_forgetting_during_training:
                mask_after_opt, conf_after_opt = self.compute_batch_masks(
                    inds, pred_proba, gt, mask_after_opt, conf_after_opt
                )

            losses.append(loss_np)
            loss_ema = loss_np if i == 0 else 0.9 * loss_ema + 0.1 * loss_np
            pbar.set_postfix_str(f"Loss: {loss_ema:.4g}")

            #update weights
            self._opt.step()

            self._last_iter += 1
            logger.debug(
                "iter: %d,\tloss value: %4g,\tgrad norm: %4g",
                self._last_iter,
                loss.item(),
                _grad_norm(self._model.parameters()),
            )

            self._opt.zero_grad()

        self._loss_values = losses
        logger.info(
            "Epoch %04d: avg train loss = %.4g", self._last_epoch + 1, np.mean(losses)
        )

        #compute metrics collected for one epoch
        if self._metrics_on_train:
            self._metric_values = self.compute_metrics(preds, gts)
            logger.info(
                "Epoch %04d: train metrics: %s",
                self._last_epoch + 1,
                str(self._metric_values),
            )

        logger.info("Epoch %04d: train finished", self._last_epoch + 1)

        #Append forgetting masks obtained in the training epoch to
        # lists containing the masks for all epochs of the training process
        if self._is_forgetting_during_training:
            self._mask_after_opt_list.append(mask_after_opt.unsqueeze(0))
            self._conf_after_opt_list.append(conf_after_opt.unsqueeze(0))


    def validate(self):
        """
        Validation the model after training in one epoch.
        If `self._is_forgetting_during_training = True`, forgetting masks 
        are collected for the validation step in the `self.predict_on_val`
        method.     
        """

        assert self._val_loader is not None, "Set a val loader first"

        logger.info("Epoch %04d: validation started", self._last_epoch + 1)
        preds, uncertainties, gts = self.predict_on_val(self._val_loader)

        self._metric_values = self.compute_metrics(preds, gts)
        logger.info(
            "Epoch %04d: validation metrics: %s",
            self._last_epoch + 1,
            str(self._metric_values),
        )
        logger.info("Epoch %04d: validation finished", self._last_epoch + 1)


    def run(self):
        """
        Run to train and validate the model for the total number of epochs.
        
        If `self._ckpt_resume` is not `None`, a previous training process is
        continued by loading of a checkpoint. If a value of the `last_epoch` loaded 
        from the checkpoint is equal or larger to `self._total_epochs`, then
        all state dictionaries are loaded, but the training and validation procedures
        are not conducted. 
        If `self._is_forgetting_during_training = True`, forgetting masks are computed
        in the training and validation procedure.
        """

        assert self._opt, "Set an optimizer to run full cycle"
        assert self._train_loader is not None, "Set a train loader to run full cycle"

        logger.info("run %s started", self._run_name)
        logger.info("using following model configs: \n%s", str(self._model_conf))

        #Create dictionary and lists for collecting forgetting masks for `self._run_name`
        if self._is_forgetting_during_training:            
            self._return_masks_dict = {}
            self._return_masks_dict['acc_mask'] = []
            self._return_masks_dict['conf_mask'] = []
            self._return_masks_dict['acc_mask_after_opt'] = []
            self._return_masks_dict['conf_mask_after_opt'] = []

            self._mask_list_tr = []
            self._conf_list_tr = []
            self._mask_after_opt_list = []
            self._conf_after_opt_list = []

            file_names_in_train_dataset = [self._train_loader.dataset.get_file_name(item) \
                        for item in range(len(self._train_loader.dataset))]
            self._return_masks_dict["file_names"] = file_names_in_train_dataset

            self._forgetting_dict[self._run_name] = self._return_masks_dict

        #Loading model and other state dictionaries from checkpoint         
        if self._ckpt_resume is not None:
            logger.info("Resuming from checkpoint '%s'", str(self._ckpt_resume))
            self.load_ckpt(self._ckpt_resume)
            
            #continue to collect forgetting masks, if run_name was not changed
            if self._is_forgetting_during_training:
                if self._run_name in self._forgetting_dict.keys():
                    self._return_masks_dict = self._forgetting_dict[self._run_name]



        #Compute the number of iterations for one epoch
        if self._iters_per_epoch is None:
            self._iters_per_epoch = len(self._cyc_train_loader.base_loader)

        #Compute the number of iterations for the overall training process
        if self._total_iters is None:
            assert self._total_epochs is not None
            self._total_iters = self._total_epochs * self._iters_per_epoch

        self._cyc_train_loader.set_iters_per_epoch(self._iters_per_epoch)

        #If self._return_masks_dict is not empty, collection of forgetting masks 
        # will be continued
        if self._is_forgetting_during_training:
            self._mask_list_tr = self._return_masks_dict["acc_mask"].copy()
            self._conf_list_tr = self._return_masks_dict["conf_mask"].copy()

            self._mask_after_opt_list = self._return_masks_dict["acc_mask_after_opt"].copy()
            self._conf_after_opt_list = self._return_masks_dict["conf_mask_after_opt"].copy()

        #Start to train and validate the model
        self._model.to(self._device)
        self._likelihood.to(self._device)

        while self._last_iter < self._total_iters:
            train_iters = min(
                self._total_iters - self._last_iter,
                self._iters_per_epoch,
            )

            #Training and validation steps for one epoch
            self.train(train_iters)
            if self._sched:
                self._sched.step()

            self.validate()
            self.log_metrics(
                "val",
                self._metric_values,
                self._last_epoch + 1,
            )

            #Filling of dicts of forgetting masks obtained in 
            # training and validation steps for one epoch
            if self._is_forgetting_during_training:
                self._return_masks_dict["acc_mask"] = self._mask_list_tr
                self._return_masks_dict["conf_mask"] = self._conf_list_tr

                self._return_masks_dict["acc_mask_after_opt"] = self._mask_after_opt_list
                self._return_masks_dict["conf_mask_after_opt"] = self._conf_after_opt_list

                self._forgetting_dict[self._run_name] = self._return_masks_dict

            #Save model and other state dicts
            self._last_epoch += 1
            self.save_ckpt()
            self._metric_values = None
            self._loss_values = None

        logger.info("run '%s' finished successfully", self._run_name)


    def get_best_or_last_ckpt(
        self,
        ckpt_type: Literal['best', 'last'] 
    ):
        """
        Find a checkpoint file `*.ckpt` associated with the highest value 
        of the track metric or created at the last epoch.

        Args:
            ckpt_type: str
                It takes 'best' or 'last' value depending on a checkpoint file 
                needs to be found.
        """
        
        #Create path to directory containing checkpoints files
        assert self._ckpt_dir is not None
        ckpt_path = Path(self._ckpt_dir) / self._run_name

        ckpt_path = Path(ckpt_path)

        #Find all checkpoints
        all_ckpt = list(ckpt_path.glob("*.ckpt"))

        #Find the best or last checkpoints
        def make_key_extractor(key):
            def key_extractor(p: Path) -> float:
                metrics = {}
                for it in p.stem.split(" - "):
                    kv = it.split(": ")
                    assert len(kv) == 2, f"Failed to parse filename: {p.name}"
                    k = kv[0]
                    v = -float(kv[1]) if "loss" in k else float(kv[1])
                    metrics[k] = v
                return metrics[key]

            return key_extractor

        if ckpt_type == 'best':
            best_ckpt = max(all_ckpt, key=make_key_extractor(self._ckpt_track_metric))
            return best_ckpt
        elif ckpt_type == 'last':
            last_ckpt = max(all_ckpt, key=make_key_extractor("epoch"))
            return last_ckpt


    def load_best_model(self) -> None:
        """
        Loads self._model and other state dicts from a checkpoint file 
        with the highest value of the track metric.
        """

        best_ckpt = self.get_best_or_last_ckpt('best')
        print(best_ckpt)
        self.load_ckpt(best_ckpt)


    def load_last_model(self) -> None:
        """
        Loads self._model and other state dicts from a checkpoint file 
        created at the last epoch
        """

        last_ckpt = self.get_best_or_last_ckpt('last')
        print(last_ckpt)
        self.load_ckpt(last_ckpt)


    def test(self, loader: DataLoader) -> None:
        """
        Compute metrics for a dataloader of a test dataset
        """

        preds, uncertainties, dataset_inds, gts = self.predict(loader)

        self._metric_values = self.compute_metrics(preds, gts)
        return self._metric_values
