# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmpose.registry import MODELS


@MODELS.register_module()
class MPJPALoss(nn.Module):
    """mean joint acc loss.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, loss_weight: float = 1.0, seq_length: int = 4) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.seq_length = seq_length

    def forward(self, output, target, target_weight=None):
        B, N, K = output.shape
        output = output.reshape(-1, self.seq_length, N, K)
        output_t1 = output[:, :self.seq_length - 1, :, :]
        output_t2 = output[:, 1:self.seq_length, :, :]
        v = output_t2 - output_t1
        v_length = self.seq_length - 1
        v_t1 = v[:, :v_length - 1, :, :]
        v_t2 = v[:, 1:v_length, :, :]
        acc = v_t2 - v_t1
        return acc.abs().mean() * self.loss_weight


@MODELS.register_module()
class MPJPAELoss(nn.Module):
    """mean joint acc error loss
    Args:
        nn (_type_): _description_
    """

    def __init__(self, loss_weight: float = 1.0, seq_length: int = 4) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.seq_length = seq_length

    def forward(self, output, target, target_weight=None):
        B, N, K = output.shape
        output = output.reshape(-1, self.seq_length, N, K)
        output_t1 = output[:, :self.seq_length - 1, :, :]
        output_t2 = output[:, 1:self.seq_length, :, :]
        v = output_t2 - output_t1
        v_length = self.seq_length - 1
        v_t1 = v[:, :v_length - 1, :, :]
        v_t2 = v[:, 1:v_length, :, :]
        acc = v_t2 - v_t1
        acc_length = v_length - 1
        acc_t1 = acc[:, :acc_length - 1, :, :]
        acc_t2 = acc[:, 1:acc_length, :, :]
        acc_error = acc_t2 - acc_t1
        return acc_error.abs().mean() * self.loss_weight
