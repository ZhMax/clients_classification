import torch.nn as nn

import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood


class ELBOLoss(nn.Module):
    """ELBO loss function to train the DUE model"""
    def __init__(self, likelihood, gp, num_data):
        super(ELBOLoss, self).__init__()
        self.gp = gp
        self.num_data = num_data
        self.likelihood = likelihood

        self.elbo_fn = VariationalELBO(
            self.likelihood,
            self.gp,
            self.num_data
        )

    def forward(self, outputs, targets):
        loss_fn = -self.elbo_fn(outputs, targets)

        return loss_fn
