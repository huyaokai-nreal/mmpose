# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from mmpose.evaluation.functional import multilabel_classification_accuracy
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                 Predictions)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class AttrMlpHead(BaseHead):

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 loss: ConfigType = dict(
                     type='BCELoss', use_target_weight=False),
                 init_cfg: OptConfigType = None,
                 deploy: bool = False):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_module = MODELS.build(loss)
        # Define fully-connected layers
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_channels, out_channels)
        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the coordinates.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output coordinates(and sigmas[optional]).
        """
        x = feats[-1]
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        if self.deploy:
            x = x.sigmoid()
        return x

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from outputs."""
        preds = self.forward(feats)  # (B, K, D)
        outputs = preds.sigmoid()
        return outputs

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_outputs = self.forward(inputs)
        keypoint_labels = np.concatenate(
            [d.gt_instance_labels.attr_labels for d in batch_data_samples])
        keypoint_labels = torch.tensor(
            keypoint_labels, requires_grad=False).cuda()
        loss = self.loss_module(pred_outputs, keypoint_labels)
        score = pred_outputs.sigmoid()
        acc_attr = multilabel_classification_accuracy(
            to_numpy(score).reshape((-1, 1)),
            to_numpy(keypoint_labels).reshape((-1, 1)))
        acc_attr = torch.tensor(acc_attr, device=keypoint_labels.device)

        losses = dict(attr_loss=loss, acc_attr=acc_attr)
        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        The hook will be automatically registered during initialization.
        """

        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            v = state_dict.pop(_k)
            k = _k.lstrip(prefix)
            # In old version, "loss" includes the instances of loss,
            # now it should be renamed "loss_module"
            k_parts = k.split('.')
            if k_parts[0] == 'loss':
                # loss.xxx -> loss_module.xxx
                k_new = prefix + 'loss_module.' + '.'.join(k_parts[1:])
            else:
                k_new = _k

            state_dict[k_new] = v

    @property
    def default_init_cfg(self):
        init_cfg = [dict(type='Normal', layer=['Linear'], std=0.01, bias=0)]
        return init_cfg
