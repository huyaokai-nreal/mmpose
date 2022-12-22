# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.logging import MessageHub
from torch import Tensor, nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import ConfigType, OptConfigType, OptSampleList
from .dsnt_head import DSNTHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class DSNTAttrHead(DSNTHead):
    """
    Args:
        in_channels (int | sequence[int]): Number of input channels
        in_featuremap_size (int | sequence[int]): Size of input feature map
        num_joints (int): Number of joints
        lambda_t (int): Discard heatmap-based loss when current
            epoch > lambda_t. Defaults to -1.
        debias (bool): Whether to remove the bias of Integral Pose Regression.
            see `Removing the Bias of Integral Pose Regression`_ by Gu et al
            (2021). Defaults to ``False``.
        beta (float): A smoothing parameter in softmax. Defaults to ``1.0``.
        deconv_out_channels (sequence[int]): The output channel number of each
            deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        conv_out_channels (sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        has_final_layer (bool): 1x1 conv layer to produce keypoint outputs.
            Defaults to ``True``
        output_sigma (bool): generate sigma for coords, Defaults to ``False``
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
        loss (Config): Config for keypoint loss. Defaults to use
            :class:`DSNTLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
        deploy (bool, optional): inferece in deploy mode, Defaults to ``False``

    .. _`DSNT`: https://arxiv.org/abs/1801.07372
    """

    _version = 1

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 in_featuremap_size: Tuple[int, int],
                 num_joints: int,
                 lambda_t: int = -1,
                 debias: bool = False,
                 beta: float = 1.0,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 has_final_layer: bool = True,
                 input_transform: str = 'select',
                 input_index: Union[int, Sequence[int]] = -1,
                 align_corners: bool = False,
                 output_sigma: bool = False,
                 loss: ConfigType = dict(
                     type='MultipleLossWrapper',
                     losses=[
                         dict(type='SmoothL1Loss', use_target_weight=True),
                         dict(type='JSDiscretLoss', use_target_weight=True)
                     ]),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 deploy: bool = False,
                 output_fuse_coord=False):

        super().__init__(
            in_channels=in_channels,
            in_featuremap_size=in_featuremap_size,
            num_joints=num_joints,
            debias=debias,
            beta=beta,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            has_final_layer=has_final_layer,
            input_transform=input_transform,
            input_index=input_index,
            align_corners=align_corners,
            loss=loss,
            output_sigma=output_sigma,
            decoder=decoder,
            init_cfg=init_cfg,
            deploy=deploy,
            output_fuse_coord=output_fuse_coord)

        self.lambda_t = lambda_t
        self.attr_fc_1 = nn.Linear(self.in_channels, 128)
        self.attr_fc_out = nn.Linear(128, 1)

    def forward(self, feats: Tuple[Tensor]) -> Union[Tensor, Tuple[Tensor]]:
        """Forward the network. The input is multi scale feature maps and the
        output is the coordinates.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output coordinates(and sigmas[optional]).
        """
        if self.simplebaseline_head is None:
            last_feat = feats[0]
            feats = self._transform_inputs(feats)
            raw_feats = feats
            if self.final_layer is not None:
                feats = self.final_layer(feats)
        else:
            feats = self.simplebaseline_head(feats)
        if self.beta > 1.0:
            heatmaps = self._flat_softmax(feats * self.beta)
        else:
            heatmaps = self._flat_softmax(feats)
        x_fea = heatmaps.sum(dim=2)
        y_fea = heatmaps.sum(dim=3)
        pred_x = x_fea.mul(self.linspace_x)
        pred_y = y_fea.mul(self.linspace_y)
        pred_x = pred_x.sum(dim=-1, keepdim=True)
        pred_y = pred_y.sum(dim=-1, keepdim=True)

        if self.debias:
            B, N, H, W = feats.shape
            C = feats.reshape(B, N, H * W).exp().sum(dim=2).reshape(B, N, 1)
            pred_x = C / (C - 1) * (pred_x - 1 / (2 * C))
            pred_y = C / (C - 1) * (pred_y - 1 / (2 * C))

        coords = torch.cat([pred_x, pred_y], dim=-1)
        if self.output_sigma:
            x = self.gap(raw_feats).reshape(raw_feats.size(0), -1)
            pred_sigma = self.sigma_fc(x)
            pred_sigma = pred_sigma.reshape(
                pred_sigma.size(0), self.num_joints, 2)
            if self.output_fuse_coord:
                last_x = self.gap(last_feat).reshape(raw_feats.size(0), -1)
                global_coords = self.coord_fc1(last_x)
                global_coords = F.relu(global_coords)
                global_coords = self.coord_fc2(global_coords)
                coords = torch.cat([
                    coords,
                    global_coords.reshape(
                        pred_sigma.size(0), self.num_joints, 2)
                ],
                                   dim=-1)
                coords = self.coord_fc(coords.reshape(pred_sigma.size(0), -1))
                attr = self.attr_fc_1(last_x)
                attr = self.attr_fc_out(F.relu(attr))
                coords = coords.reshape(pred_sigma.size(0), self.num_joints, 2)
            if self.deploy:
                return coords, attr
            coords = torch.cat([coords, pred_sigma], dim=-1)
        if self.deploy:
            return pred_x, pred_y
        return coords, heatmaps, attr

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

        batch_coords, batch_heatmaps, batch_attrs = self.forward(
            feats)  # (B, K, D)
        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        preds = self.decode(batch_coords)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, batch_attrs, pred_fields
        else:
            return preds, batch_attrs

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_coords, pred_heatmaps, pred_attrs = self.forward(inputs)
        keypoint_labels = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        attr_labels = torch.cat(
            [d.gt_instance_labels.attr_labels for d in batch_data_samples])
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        pred_attrs = torch.sigmoid(pred_attrs)
        attr_labels = attr_labels.unsqueeze(-1).to(torch.float32)
        input_list = [pred_coords, pred_heatmaps, pred_attrs]
        target_list = [keypoint_labels, gt_heatmaps, attr_labels]
        # calculate losses
        losses = dict()

        loss_list = self.loss_module(input_list, target_list, keypoint_weights)

        loss = loss_list[0] + loss_list[1] + loss_list[2]

        if self.lambda_t > 0:
            mh = MessageHub.get_current_instance()
            cur_epoch = mh.get_info('epoch')
            if cur_epoch >= self.lambda_t:
                loss = loss_list[0]

        losses.update(loss_kpt=loss)
        losses.update(loss_attr=loss_list[2])

        if pred_coords.size(-1) == 4:
            pred_coords = pred_coords[:, :, :2]
        # calculate accuracy
        preds = torch.round(pred_attrs).detach()
        correct = torch.eq(preds, attr_labels).float()
        acc_attr = correct.mean()
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(pred_coords),
            gt=to_numpy(keypoint_labels),
            mask=to_numpy(keypoint_weights) > 0,
            thr=0.05,
            norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32))

        acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
        losses.update(acc_pose=acc_pose)
        losses.update(acc_attr=acc_attr)
        attr_pos_rate = attr_labels.mean()
        losses.update(attr_pos_rate=attr_pos_rate)

        return losses
