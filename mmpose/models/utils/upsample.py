# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Sequence

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_upsample_layer


class ResizeConvModule(nn.Module):

    def __init__(self,
                 scale=2,
                 unit_channels=256,
                 upsample_cfg={'mode': 'nearest'}) -> None:
        super().__init__()
        self.upsample_cfg = copy.deepcopy(upsample_cfg)
        self.scale = scale
        self.up_conv = ConvModule(
            unit_channels,
            unit_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=dict(type='BN'),
            act_cfg=None,
            inplace=True)

    def forward(self, x):
        up_x = F.interpolate(x, scale_factor=self.scale, **self.upsample_cfg)
        up_x = self.up_conv(up_x)
        return up_x


class ResizeModule(nn.Module):

    def __init__(self,
                 scale=2,
                 unit_channels=256,
                 upsample_cfg={'mode': 'nearest'}) -> None:
        super().__init__()
        self.scale = scale
        self.upsample_cfg = copy.deepcopy(upsample_cfg)

    def forward(self, x):
        up_x = F.interpolate(x, scale_factor=self.scale, **self.upsample_cfg)
        return up_x


def _make_deconv_layers(in_channels: int,
                        layer_out_channels: Sequence[int],
                        layer_kernel_sizes: Sequence[int],
                        with_relu: bool = False) -> nn.Module:
    """Create deconvolutional layers by given parameters."""

    layers = []
    for out_channels, kernel_size in zip(layer_out_channels,
                                         layer_kernel_sizes):
        if kernel_size == 4:
            padding = 1
            output_padding = 0
        elif kernel_size == 3:
            padding = 1
            output_padding = 1
        elif kernel_size == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Unsupported kernel size {kernel_size} for'
                             'deconvlutional layers')
        cfg = dict(
            type='deconv',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=output_padding,
            bias=False)
        layers.append(build_upsample_layer(cfg))
        layers.append(nn.BatchNorm2d(num_features=out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels

    return nn.Sequential(*layers)


class DeconvModule(nn.Module):

    def __init__(self,
                 scale=2,
                 unit_channels=256,
                 upsample_cfg={'mode': 'nearest'}) -> None:
        super().__init__()
        self.scale = scale
        self.up_deconv = _make_deconv_layers(unit_channels, [unit_channels],
                                             [4])

    def forward(self, x):
        up_x = self.up_deconv(x)
        return up_x


UPSAMPLE_METHODS = dict(
    fpn=ResizeModule, rsn=ResizeConvModule, deconv=DeconvModule)
