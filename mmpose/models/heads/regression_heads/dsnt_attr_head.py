# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.logging import MessageHub
from torch import Tensor, nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmengine import is_seq_of
from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import ConfigType, OptConfigType, OptSampleList
from .integral_regression_head import IntegralRegressionHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class DSNTAttrHead(IntegralRegressionHead):
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
                 input_shape: OptIntSeq = (128, 128),
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
                 attr_dim=0,
                 output_oks=False):

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
            output_heatmap=output_oks)

        self.lambda_t = lambda_t
        self.input_shape = input_shape
        self.output_attr = False
        if attr_dim > 0:
            self.output_attr = True
            self.attr_fc_1 = nn.Linear(self.in_channels, 128)
            self.attr_fc_out = nn.Linear(128, attr_dim)
        self.output_oks = output_oks
        if self.output_oks:
            self.oks_fc_1 = nn.Linear(self.in_channels, 64)
            self.osk_fc_out = nn.Linear(64, 1)
            sigmas = torch.Tensor([
                .87, .62, .35, .25, .25, .39, .25, .25, .25, .39, .25, .25,
                .25, .25, .25, .25, .25, .39, .25, .25, .25
            ]) / 10.0
            self.vars = nn.Parameter((sigmas * 2)**2, requires_grad=False)
            self.gt_area = self.input_shape[0] * self.input_shape[1]

    def forward(self, feats: Tuple[Tensor]) -> Union[Tensor, Tuple[Tensor]]:
        """Forward the network. The input is multi scale feature maps and the
        output is the coordinates.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output coordinates(and sigmas[optional]).
        """
        output_list = []
        if is_seq_of(feats, torch.Tensor):
            last_x = feats[0]
        else:
            last_x = feats[0][0]
        global_feat = self.gap(last_x).reshape(last_x.size(0), -1)
        if self.output_oks:
            if not self.deploy:
                coords, heatmaps = super().forward(feats)
            else:
                kpt_x, kpt_y, heatmaps = super().forward(feats)
            oks = self.oks_fc_1(global_feat)
            oks = self.osk_fc_out(F.relu(oks)).sigmoid()
            if self.deploy:
                output_list += [kpt_x, kpt_y, oks]
            else:
                output_list += [coords, heatmaps, oks]
        else:
            if not self.deploy:
                coords, heatmaps = super().forward(feats)
                output_list += [coords, heatmaps]
            else:
                kpt_x, kpt_y = super().forward(feats)
                output_list += [kpt_x, kpt_y]
        if self.output_attr:
            attr = self.attr_fc_1(global_feat)
            attr = self.attr_fc_out(F.relu(attr))
            output_list.append(attr)
        return tuple(output_list)

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

        outputs = self.forward(feats)  # (B, K, D)
        batch_coords = outputs[0]
        batch_coords[..., 2:] = batch_coords[..., 2:].sigmoid()
        batch_heatmaps = outputs[1]
        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        preds = self.decode(batch_coords)
        pred_oks = outputs[2]
        pred_oks[..., 2:] = 1 - pred_oks.unsqueeze_(dim=-1)
        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds

    def __calculate_oks(self, output, target):
        dx = output[:, :, 0] - target[:, :, 0]
        dy = output[:, :, 1] - target[:, :, 1]
        dx *= self.input_shape[0]
        dy *= self.input_shape[1]
        errors = (dx**2 + dy**2) / self.vars / (
            self.gt_area + torch.finfo(torch.float32).eps) / 2
        oks = torch.sum(torch.exp(-errors), dim=1) / errors.shape[1]
        return oks

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        outputs = self.forward(inputs)
        pred_coords = outputs[0]
        pred_heatmaps = outputs[1]
        if self.output_oks:
            pred_oks = outputs[2]
        if self.output_attr:
            pred_attrs = outputs[3]
        keypoint_labels = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        input_list = [pred_coords, pred_heatmaps]
        target_list = [keypoint_labels, gt_heatmaps]
        if self.output_attr:
            attr_labels = torch.cat(
                [d.gt_instance_labels.attr_labels for d in batch_data_samples])
            pred_attrs = torch.sigmoid(pred_attrs)
            attr_labels = attr_labels.unsqueeze(-1).to(torch.float32)
            input_list.append(pred_attrs)
            target_list.append(attr_labels)
        if self.output_oks:
            target_oks = self.__calculate_oks(pred_coords[:, :, :2],
                                              keypoint_labels)
            input_list.append(pred_oks)
            target_list.append(target_oks.detach())
        # calculate losses
        losses = dict()
        loss_list = self.loss_module(input_list, target_list, keypoint_weights)
        loss_kpt = loss_list[0] + loss_list[1]
        if self.lambda_t > 0:
            mh = MessageHub.get_current_instance()
            cur_epoch = mh.get_info('epoch')
            if cur_epoch >= self.lambda_t:
                loss_kpt = loss_list[0]

        losses.update(loss_kpt=loss_kpt)
        if self.output_oks:
            losses.update(oks_mean=target_oks.mean())
            losses.update(oks_min=target_oks.min())
        loss_idx = 2
        if self.output_attr:
            losses.update(loss_attr=loss_list[loss_idx])
            loss_idx += 1
            preds = torch.round(pred_attrs).detach()
            correct = torch.eq(preds, attr_labels).float()
            acc_attr = correct.mean()
            losses.update(acc_attr=acc_attr)
            attr_pos_rate = attr_labels.mean()
            losses.update(attr_pos_rate=attr_pos_rate)
        if self.output_oks:
            losses.update(loss_oks=loss_list[loss_idx])
        if pred_coords.size(-1) == 4:
            pred_coords = pred_coords[:, :, :2]
        # calculate accuracy
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(pred_coords),
            gt=to_numpy(keypoint_labels),
            mask=to_numpy(keypoint_weights) > 0,
            thr=0.05,
            norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32))

        acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
        losses.update(acc_pose=acc_pose)

        return losses
