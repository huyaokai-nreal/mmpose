# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union
import torch.nn as nn
from torch import Tensor
import torch
from mmengine.structures import PixelData
import numpy as np
from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_coordinates, flip_heatmaps
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, OptConfigType, OptSampleList)
from ...utils.siamcc_to_kpt import SimCCToKeypoint
from ..heatmap_heads import RTMHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RTMIPRHead(RTMHead):
    """Top-down head introduced in RTMPose (2023). The head is composed of a
    large-kernel convolutional layer, a fully-connected layer and a Gated
    Attention Unit to generate 1d representation from low-resolution feature
    maps.

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map.
        out_channels (int): Number of channels in the output heatmap.
        input_size (tuple): Size of input image in shape [w, h].
        in_featuremap_size (int | sequence[int]): Size of input feature map.
        simcc_split_ratio (float): Split ratio of pixels.
            Default: 2.0.
        final_layer_kernel_size (int): Kernel size of the convolutional layer.
            Default: 1.
        gau_cfg (Config): Config dict for the Gated Attention Unit.
            Default: dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False).
        input_transform (str): Transformation of input features which should
            be one of the following options:

                - ``'resize_concat'``: Resize multiple feature maps specified
                    by ``input_index`` to the same size as the first one and
                    concat these feature maps
                - ``'select'``: Select feature map(s) specified by
                    ``input_index``. Multiple selected features will be
                    bundled into a tuple

            Defaults to ``'select'``
        input_index (int | sequence[int]): The feature map index used in the
            input transformation. See also ``input_transform``. Defaults to -1
        align_corners (bool): `align_corners` argument of
            :func:`torch.nn.functional.interpolate` used in the input
            transformation. Defaults to ``False``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 input_size: Tuple[int, int],
                 in_featuremap_size: Tuple[int, int],
                 simcc_split_ratio: float = 2.0,
                 final_layer_kernel_size: int = 1,
                 gau_cfg: ConfigType = dict(
                     hidden_dims=256,
                     s=128,
                     expansion_factor=2,
                     dropout_rate=0.,
                     drop_path=0.,
                     act_fn='ReLU',
                     use_rel_bias=False,
                     pos_enc=False),
                 input_transform: str = 'select',
                 input_index: Union[int, Sequence[int]] = -1,
                 align_corners: bool = False,
                 loss: ConfigType = dict(
                     type='KLDiscretLoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 output_sigma: bool = False,
                 deploy: bool = False,
                 lambda_t: int = -1):
        super().__init__(in_channels, out_channels, input_size,
                         in_featuremap_size, simcc_split_ratio,
                         final_layer_kernel_size, gau_cfg, input_transform,
                         input_index, align_corners, loss, decoder, init_cfg)
        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)
        self.ipr_module = SimCCToKeypoint(feat_w=W, feat_h=H)
        self.output_sigma = output_sigma
        self.deploy = deploy
        self.lambda_t = lambda_t
        if self.output_sigma:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.sigma_fc = nn.Linear(self.in_channels, self.out_channels * 2)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network.

        The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        pred_x, pred_y = super().forward(feats)
        heatmaps = torch.cat([pred_x, pred_y], dim=1)
        raw_feats = self._transform_inputs(feats)
        pred_x, pred_y = self.ipr_module(pred_x, pred_y)
        if self.output_sigma:
            x = self.gap(raw_feats).reshape(raw_feats.size(0), -1)
            pred_sigma = self.sigma_fc(x)
            pred_sigma = pred_sigma.reshape(
                pred_sigma.size(0), self.out_channels, 2)
        if self.deploy:
            return torch.cat([pred_x, pred_y], dim=-1)
        else:
            output = torch.cat([pred_x, pred_y, pred_sigma], dim=-1)
            return output, heatmaps

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}):
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']
            _feats, _feats_flip = feats

            _batch_coords, _batch_heatmaps = self.forward(_feats)

            _batch_coords_flip, _batch_heatmaps_flip = self.forward(
                _feats_flip)
            _batch_coords_flip = flip_coordinates(
                _batch_coords_flip,
                flip_indices=flip_indices,
                shift_coords=test_cfg.get('shift_coords', True),
                input_size=input_size)
            _batch_heatmaps_flip = flip_heatmaps(
                _batch_heatmaps_flip,
                flip_mode='heatmap',
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))

            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_coords, batch_heatmaps = self.forward(feats)  # (B, K, D)

        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        preds = self.decode(batch_coords)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_outputs, _ = self.forward(inputs)

        keypoint_labels = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_outputs, keypoint_labels,
                                keypoint_weights.unsqueeze(-1))

        losses.update(loss_kpt=loss)

        # calculate accuracy
        pred_coords = pred_outputs[:, :, :2]
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(pred_coords),
            gt=to_numpy(keypoint_labels),
            mask=to_numpy(keypoint_weights) > 0,
            thr=0.05,
            norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32))

        acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
        losses.update(acc_pose=acc_pose)

        return losses
