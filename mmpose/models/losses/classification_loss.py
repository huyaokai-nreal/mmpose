# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS


@MODELS.register_module()
class BCELoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
        ignore_label (int): label which will be ignored
        with_logits (bool): Whether to use BCEWithLogitsLoss. Default: False.
    """

    def __init__(self,
                 use_target_weight=False,
                 loss_weight=1.,
                 ignore_label=-1,
                 with_logits=False):
        super().__init__()
        self.criterion = F.binary_cross_entropy if not with_logits\
            else F.binary_cross_entropy_with_logits
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        loss = self.criterion(output, target, reduction='none')
        if self.use_target_weight:
            assert target_weight is not None
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = loss * target_weight
        loss = (loss[target != self.ignore_label]).mean()
        return loss * self.loss_weight


@MODELS.register_module()
class FocalLoss(nn.Module):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
    """

    def __init__(self,
                 use_target_weight=False,
                 loss_weight=1.,
                 ignore_label=-1,
                 gamma=2.0,
                 alpha=0.25):
        super().__init__()
        self.criterion = F.binary_cross_entropy_with_logits
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        pred_sigmoid = output.sigmoid()
        target = target.type_as(pred_sigmoid)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = self.criterion(output, target, reduction='none') * focal_weight
        if self.use_target_weight:
            assert target_weight is not None
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = loss * target_weight
        loss = (loss[target != self.ignore_label]).mean()
        return loss * self.loss_weight


@MODELS.register_module()
class JSDiscretLoss(nn.Module):
    """Discrete JS Divergence loss for DSNT with Gaussian Heatmap.

    Modified from `the official implementation
    <https://github.com/anibali/dsntnn/blob/master/dsntnn/__init__.py>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
    """

    def __init__(
        self,
        use_target_weight=True,
        size_average: bool = True,
    ):
        super(JSDiscretLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.size_average = size_average
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def kl(self, p, q):
        """Kullback-Leibler Divergence."""

        eps = 1e-24
        kl_values = self.kl_loss((p + eps).log(), q)
        return kl_values

    def js(self, pred_hm, gt_hm):
        m = 0.5 * (pred_hm + gt_hm).detach()
        js_values = 0.5 * (self.kl(pred_hm, m) + self.kl(gt_hm, m))
        return js_values

    def forward(self, pred_hm, gt_hm, target_weight=None):
        """Forward function.

        Args:
            pred_hm (torch.Tensor[N, K, H, W]): Predicted heatmaps.
            gt_hm (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.

        Returns:
            torch.Tensor: Loss value.
        """

        if self.use_target_weight:
            assert target_weight is not None
            assert pred_hm.ndim >= target_weight.ndim

            for i in range(pred_hm.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)

            loss = self.js(pred_hm * target_weight, gt_hm * target_weight)
        else:
            loss = self.js(pred_hm, gt_hm)

        if self.size_average:
            loss /= len(gt_hm)
        return loss


@MODELS.register_module()
class KLDiscretLoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.
    Modified from `the official implementation.

    <https://github.com/leeyegy/SimCC>`_.
    Args:
        beta (float): Temperature factor of Softmax.
        label_softmax (bool): Whether to use Softmax on labels.
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
    """

    def __init__(self, beta=1.0, label_softmax=False, use_target_weight=True):
        super(KLDiscretLoss, self).__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.use_target_weight = use_target_weight

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        """Criterion function."""

        scores = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.beta, dim=1)
        loss = torch.mean(self.kl_loss(scores, labels), dim=1)
        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        num_joints = pred_simcc[0].size(1)
        loss = 0

        if self.use_target_weight:
            weight = target_weight.reshape(-1)
        else:
            weight = 1.

        for pred, target in zip(pred_simcc, gt_simcc):
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1, target.size(-1))

            loss += self.criterion(pred, target).mul(weight).sum()

        return loss / num_joints


@MODELS.register_module()
class KLDiscretLoss3D(KLDiscretLoss):

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        output_x, output_y, output_z = pred_simcc
        target_x, target_y, target_z = gt_simcc
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx].squeeze()
            coord_y_pred = output_y[:, idx].squeeze()
            coord_z_pred = output_z[:, idx].squeeze()
            coord_x_gt = target_x[:, idx].squeeze()
            coord_y_gt = target_y[:, idx].squeeze()
            coord_z_gt = target_z[:, idx].squeeze()

            if self.use_target_weight:
                weight = target_weight[:, idx].squeeze()
            else:
                weight = 1.

            loss += (
                self.criterion(coord_x_pred, coord_x_gt).mul(weight).sum())
            loss += (
                self.criterion(coord_y_pred, coord_y_gt).mul(weight).sum())
            loss += (
                self.criterion(coord_z_pred, coord_z_gt).mul(weight).sum())

        return loss / num_joints


@MODELS.register_module()
class InfoNCELoss(nn.Module):
    """InfoNCE loss for training a discriminative representation space with a
    contrastive manner.

    `Representation Learning with Contrastive Predictive Coding
    arXiv: <https://arxiv.org/abs/1611.05424>`_.

    Args:
        temperature (float, optional): The temperature to use in the softmax
            function. Higher temperatures lead to softer probability
            distributions. Defaults to 1.0.
        loss_weight (float, optional): The weight to apply to the loss.
            Defaults to 1.0.
    """

    def __init__(self, temperature: float = 1.0, loss_weight=1.0) -> None:
        super(InfoNCELoss, self).__init__()
        assert temperature > 0, f'the argument `temperature` must be ' \
                                f'positive, but got {temperature}'
        self.temp = temperature
        self.loss_weight = loss_weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Computes the InfoNCE loss.

        Args:
            features (Tensor): A tensor containing the feature
                representations of different samples.

        Returns:
            Tensor: A tensor of shape (1,) containing the InfoNCE loss.
        """
        n = features.size(0)
        features_norm = F.normalize(features, dim=1)
        logits = features_norm.mm(features_norm.t()) / self.temp
        targets = torch.arange(n, dtype=torch.long, device=features.device)
        loss = F.cross_entropy(logits, targets, reduction='sum')
        return loss * self.loss_weight
