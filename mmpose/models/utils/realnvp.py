# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import distributions


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model

    `Density estimation using Real NVP
    arXiv: <https://arxiv.org/abs/1605.08803>`_.

    Code is modified from `the official implementation of RLE
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    See also `real-nvp-pytorch
    <https://github.com/senya-ashukha/real-nvp-pytorch>`_.
    """

    @staticmethod
    def get_scale_net(dim):
        """Get the scale model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(dim, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, dim), nn.Tanh())

    @staticmethod
    def get_trans_net(dim):
        """Get the translation model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(dim, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, dim))

    @property
    def prior(self):
        """The prior distribution."""
        return distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self, dim):
        super(RealNVP, self).__init__()
        self.dim = dim

        self.register_buffer('loc', torch.zeros(dim))
        self.register_buffer('cov', torch.eye(dim))

        if dim == 2:
            self.register_buffer(
                'mask',
                torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))
        elif dim == 3:
            self.register_buffer(
                'mask',
                torch.tensor([[1, 1, 0], [0, 0, 1]] * 3, dtype=torch.float32))
        else:
            raise NotImplementedError

        self.s = torch.nn.ModuleList(
            [self.get_scale_net(self.dim) for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList(
            [self.get_trans_net(self.dim) for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialization model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping form the data space to the latent space and calculate
        the log determinant of the Jacobian matrix."""

        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])  # torch.exp(s): betas
            t = self.t[i](z_) * (1 - self.mask[i])  # gammas
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""

        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det
