# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from .rtmcc_head import RTMCCHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RTMCCHead3D(RTMCCHead):
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
                 with_root_net: bool = False,
                 with_gau: bool = False,
                 mlp_with_conv: bool = False):
        if init_cfg is None:
            init_cfg = self.default_init_cfg
        super().__init__(in_channels, out_channels, input_size,
                         in_featuremap_size, simcc_split_ratio,
                         final_layer_kernel_size, gau_cfg, loss, decoder,
                         init_cfg, with_gau)
        if mlp_with_conv:
            flatten_dims = self.in_featuremap_size[
                0] * self.in_featuremap_size[1]
            self.mlp = nn.Sequential(
                nn.Linear(flatten_dims, 128), nn.ReLU(), nn.Conv2d(21, 21, 1),
                nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                nn.Conv2d(21, 21, 1))
        D = int(self.input_size[2] * self.simcc_split_ratio)
        self.cls_z = nn.Linear(gau_cfg['hidden_dims'], D, bias=False)
        self.with_root_net = with_root_net
        if with_root_net:
            self.depth_fc1 = nn.Linear(in_channels, 128)
            self.depth_fc2 = nn.Linear(128, 1)

    def forward(self,
                feats: Tuple[Tensor],
                k_value: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward the network.

        The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        raw_feats = feats[-1]

        feats = self.final_layer(raw_feats)  # -> B, K, H, W

        # flatten the output heatmap
        feats = feats.reshape((feats.shape[0], feats.shape[1], 1, -1))
        feats = self.mlp(feats)  # -> B, K, hidden
        if self.with_gau:
            feats = self.gau(feats)
        feats = feats.squeeze(2)
        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)
        pred_z = self.cls_z(feats)
        output = [pred_x, pred_y, pred_z]
        if self.with_root_net:
            x = torch.nn.functional.adaptive_avg_pool2d(raw_feats, (1, 1))
            x = torch.flatten(x, 1)
            x = self.depth_fc1(x)
            x = F.relu(x)
            root_depth = self.depth_fc2(x).view(-1)
            root_depth = root_depth * k_value
            output.append(root_depth)
        return tuple(output)

    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            List[InstanceData]: The pose predictions, each contains
            the following fields:
                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
                - keypoint_x_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the x direction
                - keypoint_y_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the y direction
        """
        if self.with_root_net:
            k_value_list = []
            for data_sample in batch_data_samples:
                gt_instances = data_sample.gt_instances
                bbox_scales = gt_instances.bbox_scales[0]
                camera = data_sample.meta['ori_camera']
                fx, fy = camera.f
                real_hand_shape = [200, 200]
                k_value_list.append(
                    np.sqrt(fx * fy * real_hand_shape[0] * real_hand_shape[1] /
                            (bbox_scales[0] * bbox_scales[1])))
            k_values = torch.from_numpy(
                np.array(k_value_list, dtype=np.float32)).cuda()
            pred_x, pred_y, pred_z, root_depth = self.forward(feats, k_values)
        else:
            pred_x, pred_y, pred_z = self.forward(feats)
        preds = self.decode((pred_x, pred_y, pred_z))
        if self.with_root_net:
            for i, pred in enumerate(preds):
                pred.set_field(root_depth[i:i + 1].cpu().numpy(), 'root_depth')
        return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        if self.with_root_net:
            k_value_list = []
            root_depth_list = []
            for data_sample in batch_data_samples:
                gt_instances = data_sample.gt_instances
                bbox_scales = gt_instances.bbox_scales[0]
                camera = data_sample.meta['ori_camera']
                fx, fy = camera.f
                real_hand_shape = [200, 200]
                k_value_list.append(
                    np.sqrt(fx * fy * real_hand_shape[0] * real_hand_shape[1] /
                            (bbox_scales[0] * bbox_scales[1])))
                root_depth_list.append(data_sample.meta['root_depth'])
            k_values = torch.from_numpy(
                np.array(k_value_list, dtype=np.float32)).cuda()
            root_depth_gt = torch.from_numpy(
                np.array(root_depth_list, dtype=np.float32)).cuda()
            pred_x, pred_y, pred_z, root_depth = self.forward(feats, k_values)
        else:
            pred_x, pred_y, pred_z = self.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
                         dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )
        label_depth_list = []
        label_depth_id_list = []
        for i, data in enumerate(batch_data_samples):
            if hasattr(data.gt_instance_labels, 'keypoint_z_labels'):
                label_depth_list.append(
                    data.gt_instance_labels.keypoint_z_labels)
                label_depth_id_list.append(i)
        gt_z = torch.cat(label_depth_list, dim=0)
        label_depth_id = torch.tensor(
            label_depth_id_list, dtype=torch.int32).cuda()
        valid_depth_pred = torch.index_select(pred_z, 0, label_depth_id)
        valid_depth_weights = torch.index_select(keypoint_weights, 0,
                                                 label_depth_id)
        pred_simcc_2d = (pred_x, pred_y)
        gt_simcc_2d = (gt_x, gt_y)
        pred_simcc_depth = (valid_depth_pred, )
        gt_simcc_depth = (gt_z, )
        # calculate losses
        losses = dict()
        loss = self.loss_module([pred_simcc_2d, pred_simcc_depth],
                                [gt_simcc_2d, gt_simcc_depth],
                                [keypoint_weights, valid_depth_weights])
        losses.update(loss_kpt2d=loss[0])
        losses.update(loss_depth=loss[1])
        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc_2d),
            target=to_numpy(gt_simcc_2d),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )
        keypoint3d_ratio = len(label_depth_id_list) / float(
            len(batch_data_samples))
        losses.update(kpt3d_ratio=torch.Tensor([keypoint3d_ratio]).cuda())

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)
        if self.with_root_net:
            loss_root_depth = F.l1_loss(root_depth, root_depth_gt)
            losses.update(loss_root_depth=loss_root_depth)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg
