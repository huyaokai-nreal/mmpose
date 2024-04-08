# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from mmengine import MMLogger
from mmengine.logging import MessageHub
from torch import Tensor

from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import ConfigType, OptConfigType, OptSampleList
from .integral_regression_head import IntegralRegressionHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class DSNTHead(IntegralRegressionHead):
    """Top-down integral regression head introduced in `DSNT`_ by Nibali et
    al(2018). The head contains a differentiable spatial to numerical transform
    (DSNT) layer that do soft-argmax operation on the predicted heatmaps to
    regress the coordinates.

    This head is used for algorithms that require supervision of heatmaps
    in `DSNT` approach.

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
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        output_sigma (bool): generate sigma for coords, Defaults to ``False``
        loss (Config): Config for keypoint loss. Defaults to use
            :class:`DSNTLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
        deploy (bool, optional): inferece in deploy mode, Defaults to ``False``

    .. _`DSNT`: https://arxiv.org/abs/1801.07372
    """

    _version = 2

    def __init__(
            self,
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
            final_layer: dict = dict(kernel_size=1),
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
            feat_norm_type='softmax',
            deploy_output: List[str] = ['feat', 'score'],
            output_fuse_coord: bool = False,
            symmetry_ipr=False,
            consistency_loss=False,
            heatmap_loss=True,
            output_depth=False,
            depth_channel=256,
            depth_encode_type='direct',  # 'heatmap' or 'direct'
            input_size: Optional[Tuple] = None,
            distill_feat: bool = False):

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
            final_layer=final_layer,
            loss=loss,
            output_sigma=output_sigma,
            decoder=decoder,
            init_cfg=init_cfg,
            deploy=deploy,
            deploy_output=deploy_output,
            feat_norm_type=feat_norm_type,
            symmetry_ipr=symmetry_ipr,
            output_fuse_coord=output_fuse_coord,
            output_depth=output_depth,
            depth_channel=depth_channel,
            depth_encode_type=depth_encode_type)

        self.lambda_t = lambda_t
        self.consistency_loss = consistency_loss
        self.input_size = input_size
        self.heatmap_loss = heatmap_loss
        self.output_depth = output_depth
        self.distill_feat = distill_feat

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        label_2d_list = []
        label_depth_list = []
        label_depth_id_list = []
        for i, data in enumerate(batch_data_samples):
            if self.depth_encode_type == 'direct':
                keypoint_label = data.gt_instance_labels.keypoint_labels
                label_2d_list.append(keypoint_label[..., :2])
                if keypoint_label.shape[-1] == 3:
                    label_depth_list.append(keypoint_label[..., 2:3])
                    label_depth_id_list.append(i)
            elif self.depth_encode_type == 'heatmap':
                keypoint_label = data.gt_instance_labels.keypoint_labels
                label_2d_list.append(keypoint_label[..., :2])
                if hasattr(data.gt_instance_labels, 'keypoint_z_labels'):
                    label_depth_list.append(
                        data.gt_instance_labels.keypoint_z_labels)
                    label_depth_id_list.append(i)
            else:
                logger = MMLogger.get_current_instance()
                logger.error(f'{self.depth_encode_type} is not supported')

        label_2d = torch.cat(label_2d_list)
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        outputs = self.forward(inputs)
        pred_coords, pred_heatmaps = outputs[:2]
        if self.distill_feat:
            noise_pred_coords = pred_coords[1::2]
            clear_pred_coords = pred_coords[::2]
            input_list = [clear_pred_coords, noise_pred_coords]
            target_list = [label_2d, label_2d]
        else:
            input_list = [pred_coords]
            target_list = [label_2d]
        if self.output_depth:
            label_depth = torch.cat(label_depth_list)
            label_depth_id = torch.tensor(
                label_depth_id_list, dtype=torch.int32).cuda()
            pred_depth = outputs[2]
            if self.distill_feat:
                noise_pred_depth = pred_depth[1::2]
                clear_pred_depth = pred_depth[::2]
                valid_clear_pred_depth = torch.index_select(
                    clear_pred_depth, 0, label_depth_id)
                valid_noise_pred_depth = torch.index_select(
                    noise_pred_depth, 0, label_depth_id)
                input_list.append(valid_clear_pred_depth)
                input_list.append(valid_noise_pred_depth)
                target_list.append(label_depth)
                target_list.append(label_depth)
            else:
                valid_depth_pred = torch.index_select(pred_depth, 0,
                                                      label_depth_id)

                input_list.append(valid_depth_pred)
                target_list.append(label_depth)
        if self.heatmap_loss:
            gt_heatmaps = torch.stack(
                [d.gt_fields.heatmaps for d in batch_data_samples])
            input_list.append(pred_heatmaps)
            target_list.append(gt_heatmaps)
        if self.consistency_loss:
            B, N, K = pred_coords[..., :2].shape
            pred_coords_pos = torch.cat([
                pred_coords[..., :2] * self.input_size,
                torch.ones(B, N, 1, device='cuda')
            ],
                                        dim=-1)
            global_pred_coords = torch.zeros((B, N, K)).cuda()
            all_inv_warp_mat = torch.zeros(B, 3, 2).cuda()
            all_inv_warp_mat.requires_grad = False
            for i, d in enumerate(batch_data_samples):
                warp_mat = d.metainfo['warp_mat']
                inv_warp_mat = cv2.invertAffineTransform(warp_mat).astype(
                    np.float32)
                inv_warp_mat = torch.from_numpy(inv_warp_mat).cuda()
                all_inv_warp_mat[i] = inv_warp_mat.transpose(0, 1)
            global_pred_coords = torch.bmm(pred_coords_pos, all_inv_warp_mat)
            pred_coords_1 = global_pred_coords.view(B // 2, 2, N, K)[:,
                                                                     0, :, :]
            pred_coords_2 = global_pred_coords.view(B // 2, 2, N, K)[:,
                                                                     1, :, :]
            input_list.append(pred_coords_1)
            target_list.append(pred_coords_2)
        losses = dict()

        loss_list = self.loss_module(input_list, target_list, keypoint_weights)
        if pred_coords.size(-1) == 4:
            pred_coords = pred_coords[:, :, :2]

        # calculate accuracy
        if self.distill_feat:
            loss_clear_kpt2d = loss_list[0]
            loss_noise_kpt2d = loss_list[1]
            losses.update(loss_reg_clear=loss_clear_kpt2d)
            losses.update(loss_reg_noise=loss_noise_kpt2d)
            if self.output_depth:
                losses.update(loss_depth_clear=loss_list[2])
                losses.update(loss_depth_noise=loss_list[3])
            _, clear_avg_acc, _ = keypoint_pck_accuracy(
                pred=to_numpy(pred_coords[::2]),
                gt=to_numpy(label_2d),
                mask=np.abs(to_numpy(keypoint_weights)) > 0,
                thr=0.05,
                norm_factor=np.ones((label_2d.size(0), 2), dtype=np.float32))
            _, noise_avg_acc, _ = keypoint_pck_accuracy(
                pred=to_numpy(pred_coords[1::2]),
                gt=to_numpy(label_2d),
                mask=np.abs(to_numpy(keypoint_weights)) > 0,
                thr=0.05,
                norm_factor=np.ones((label_2d.size(0), 2), dtype=np.float32))
            clear_acc_pose = torch.tensor(clear_avg_acc).cuda()
            losses.update(clear_acc_pose=clear_acc_pose)
            noise_acc_pose = torch.tensor(noise_avg_acc).cuda()
            losses.update(noise_acc_pose=noise_acc_pose)
            losses.update(acc_gap=clear_acc_pose - noise_acc_pose)
        else:
            loss_kpt2d = loss_list[0]
            losses.update(loss_reg=loss_kpt2d)
            if self.output_depth:
                losses.update(loss_depth=loss_list[1])
            if self.heatmap_loss:
                loss_ht = loss_list[1]
                if self.lambda_t > 0:
                    mh = MessageHub.get_current_instance()
                    cur_epoch = mh.get_info('epoch')
                    if cur_epoch >= self.lambda_t:
                        loss_ht = 0
                losses.update(loss_ht=loss_ht)
            if self.consistency_loss:
                losses.update(loss_const=loss_list[2])
            _, avg_acc, _ = keypoint_pck_accuracy(
                pred=to_numpy(pred_coords),
                gt=to_numpy(label_2d),
                mask=np.abs(to_numpy(keypoint_weights)) > 0,
                thr=0.05,
                norm_factor=np.ones((pred_coords.size(0), 2),
                                    dtype=np.float32))
            acc_pose = torch.tensor(avg_acc).cuda()
            losses.update(acc_pose=acc_pose)

        return losses
