# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_coordinates, flip_heatmaps
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import ConfigType, OptConfigType, OptSampleList
from ...utils.siamcc_to_kpt import SimCCToKeypoint3D
from ..coord_cls_heads import RTMCCHead3D

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RTMCCIPRHead3D(RTMCCHead3D):
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
                 loss: ConfigType = dict(
                     type='KLDiscretLoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 output_sigma: bool = False,
                 deploy: bool = False,
                 with_gau: bool = False,
                 deploy_output='kpt',
                 mlp_with_conv: bool = False,
                 map_type='softmax'):
        super().__init__(
            in_channels,
            out_channels,
            input_size,
            in_featuremap_size,
            simcc_split_ratio,
            final_layer_kernel_size,
            gau_cfg,
            loss,
            decoder,
            init_cfg,
            with_gau=with_gau,
            mlp_with_conv=mlp_with_conv)
        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)
        D = int(self.input_size[2] * self.simcc_split_ratio)
        self.ipr_module = SimCCToKeypoint3D(
            feat_w=W, feat_h=H, feat_d=D, map_type=map_type)
        self.with_gau = with_gau
        self.deploy_output = deploy_output
        self.output_sigma = output_sigma
        self.deploy = deploy

        if self.output_sigma:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.sigma_conv = nn.Conv2d(
                self.in_channels, self.out_channels * 3, kernel_size=1)

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
        feat_x, feat_y, feat_z = super().forward(feats)
        heatmaps = torch.cat([feat_x, feat_y, feat_z], dim=1)
        raw_feats = feats[-1]
        pred_x, pred_y, pred_z = self.ipr_module(feat_x.squeeze(),
                                                 feat_y.squeeze(),
                                                 feat_z.squeeze())
        output = torch.cat([pred_x, pred_y, pred_z], dim=-1)
        if self.output_sigma:
            x = self.gap(raw_feats)
            pred_sigma = self.sigma_conv(x)
            pred_sigma_reshape = pred_sigma.reshape(
                pred_sigma.size(0), self.out_channels, 3)
            output = torch.cat([output, pred_sigma_reshape], dim=-1)
        if self.deploy:
            if self.deploy_output == 'kpt':
                batch_coords = torch.cat([pred_x, pred_y, pred_z], dim=-1)
                return batch_coords
            elif self.deploy_output == 'feat':
                return feat_x, feat_y, feat_z
        else:
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
        # label_depth_list = []
        # label_2d_list = []
        # for i, data in enumerate(batch_data_samples):
        #     keypoint_label = data.gt_instance_labels.keypoint_labels
        #     label_depth_list.append(keypoint_label[..., 2:3])
        #     label_2d_list.append(keypoint_label[..., :2])
        # label_2d = torch.cat(label_2d_list)
        # label_depth = torch.cat(label_depth_list)
        if self.deploy:
            if self.deploy_output == 'feat':
                pred_x, pred_y, pred_z = self.ipr_module(
                    feats[0], feats[1], feats[2])
                batch_coords = torch.cat([pred_x, pred_y, pred_z], dim=-1)
            else:
                batch_coords = feats
        else:
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

        if self.output_sigma:
            batch_coords[..., 2:] = batch_coords[..., 2:].sigmoid()
        # batch_coords[..., 2:3] = label_depth
        # batch_coords[..., :2] = label_2d
        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        preds = self.decode(batch_coords)
        return preds

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        if self.deploy:
            if self.deploy_output == 'feat':
                pred_x, pred_y, pred_z = self.ipr_module(
                    inputs[0], inputs[1], inputs[2])
                pred_outputs = torch.cat([pred_x, pred_y, pred_z], dim=-1)
            else:
                pred_outputs = inputs
        else:
            pred_outputs, _ = self.forward(inputs)
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ]).cuda()
        label_2d_list = []
        label_depth_list = []
        label_depth_id_list = []
        for i, data in enumerate(batch_data_samples):
            keypoint_label = data.gt_instance_labels.keypoint_labels
            label_2d_list.append(keypoint_label[..., :2])
            if keypoint_label.shape[-1] == 3:
                label_depth_list.append(keypoint_label[..., 2:3])
                label_depth_id_list.append(i)
        label_2d = torch.cat(label_2d_list)
        label_depth = torch.cat(label_depth_list)
        label_depth_id = torch.tensor(
            label_depth_id_list, dtype=torch.int32).cuda()
        valid_depth_pred = torch.index_select(pred_outputs, 0,
                                              label_depth_id)[..., 2:3]
        valid_depth_weights = torch.index_select(keypoint_weights, 0,
                                                 label_depth_id)

        losses = dict()
        input_list = [pred_outputs[..., :2], valid_depth_pred]
        target_list = [label_2d, label_depth]
        loss = self.loss_module(input_list, target_list,
                                [keypoint_weights, valid_depth_weights])

        losses.update(loss_kpt2d=loss[0])
        losses.update(loss_depth=loss[1])
        keypoint3d_ratio = len(label_depth_id_list) / float(
            len(batch_data_samples))
        losses.update(kpt3d_ratio=torch.Tensor([keypoint3d_ratio]).cuda())

        # calculate accuracy
        pred_coords = pred_outputs[:, :, :3]
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(pred_outputs[..., :2]),
            gt=to_numpy(label_2d),
            mask=to_numpy(keypoint_weights) > 0,
            thr=0.05,
            norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32))

        acc_pose = torch.tensor(avg_acc).cuda()
        losses.update(acc_pose=acc_pose)

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to load weights of deconv layers from
        :class:`HeatmapHead` into `simplebaseline_head`.

        The hook will be automatically registered during initialization.
        """

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            # convert fc to conv
            if _k == 'head.sigma_fc.weight':
                state_dict['head.sigma_conv.weight'] = torch.unsqueeze(
                    torch.unsqueeze(v, -1), -1)
            if _k == 'head.sigma_fc.bias':
                state_dict['head.sigma_conv.bias'] = v
