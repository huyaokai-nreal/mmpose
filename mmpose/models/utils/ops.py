# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Tuple, Union

import torch
from mmengine.registry import MODELS
from torch.nn import functional as F


class BFBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 use_bias=False,
                 affine=True):
        super(BFBatchNorm2d, self).__init__(num_features, eps, momentum)
        self.affine = affine
        self.use_bias = use_bias

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        if self.use_bias:
            mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)

        if self.training is not True:
            if self.use_bias:
                y = y - self.running_mean.view(-1, 1)
            y = y / (self.running_var.view(-1, 1)**0.5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    if self.use_bias:
                        self.running_mean = (
                            1 - self.momentum
                        ) * self.running_mean + self.momentum * mu
                    self.running_var = (
                        1 - self.momentum
                    ) * self.running_var + self.momentum * sigma2
            if self.use_bias:
                y = y - mu.view(-1, 1)
            y = y / (sigma2.view(-1, 1)**.5 + self.eps)

        if self.affine:
            y = self.weight.view(-1, 1) * y
            if self.use_bias:
                y += self.bias.view(-1, 1)

        return y.view(return_shape).transpose(0, 1)


MODELS.register_module('BFBN', module=BFBatchNorm2d)


def resize(input: torch.Tensor,
           size: Optional[Union[Tuple[int, int], torch.Size]] = None,
           scale_factor: Optional[float] = None,
           mode: str = 'nearest',
           align_corners: Optional[bool] = None,
           warning: bool = True) -> torch.Tensor:
    """Resize a given input tensor using specified size or scale_factor.

    Args:
        input (torch.Tensor): The input tensor to be resized.
        size (Optional[Union[Tuple[int, int], torch.Size]]): The desired
            output size. Defaults to None.
        scale_factor (Optional[float]): The scaling factor for resizing.
            Defaults to None.
        mode (str): The interpolation mode. Defaults to 'nearest'.
        align_corners (Optional[bool]): Determines whether to align the
            corners when using certain interpolation modes. Defaults to None.
        warning (bool): Whether to display a warning when the input and
            output sizes are not ideal for alignment. Defaults to True.

    Returns:
        torch.Tensor: The resized tensor.
    """
    # Check if a warning should be displayed regarding input and output sizes
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would be more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')

    # Convert torch.Size to tuple if necessary
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)

    # Perform the resizing operation
    return F.interpolate(input, size, scale_factor, mode, align_corners)
