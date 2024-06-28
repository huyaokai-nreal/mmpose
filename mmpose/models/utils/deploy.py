# Copyright (c) OpenMMLab. All rights reserved.
import torch
from loguru import logger


def _get_conv_layer(submodule):
    for name, child in submodule.named_children():
        if isinstance(child, torch.nn.Conv2d):
            return name, child
        else:
            return _get_conv_layer(child)


def fuse_preprocess(module):
    mean = module.cfg.mean
    std = module.cfg.std
    mean = torch.as_tensor([mean])
    std = torch.as_tensor([std])
    for name, child in module.named_children():
        if name == 'backbone':
            name, conv_layer = _get_conv_layer(child)
            logger.info(
                f'fuse preprocess with mean {mean}, std {std} to conv {name}')
            w = conv_layer.weight.data
            b = conv_layer.bias.data
            mean = mean.to(w.device)
            std = std.to(w.device)
            fuse_w = w / std
            fuse_b = -(w * mean / std).view((b.size(0), -1)).sum(dim=-1) + b
            conv_layer.weight.data = fuse_w
            conv_layer.bias.data = fuse_b
            return module
    logger.info('can not find first conv in backbone to fuse preprocess')
    return module
