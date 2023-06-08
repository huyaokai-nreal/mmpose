# Copyright (c) OpenMMLab. All rights reserved
from itertools import zip_longest
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.data import format_data
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)

from .topdown import TopdownPoseEstimator

# from IPython import embed


# GMLP CGU 2d_conv
class ChannelGatingUnit(nn.Module):

    def __init__(self, d_ffn):
        super().__init__()
        self.norm = nn.LayerNorm([d_ffn, 1, 1])
        # self.norm = nn.BatchNorm2d(d_ffn)
        self.channel_proj = nn.Conv2d(d_ffn, d_ffn, kernel_size=1)
        nn.init.constant_(self.channel_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=1)
        v = self.norm(v)
        v = self.channel_proj(v)
        out = u * v
        return out


class gMLPBlock(nn.Module):

    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.norm = nn.LayerNorm([d_model, 1, 1])
        # self.norm = nn.BatchNorm2d(d_model)
        self.channel_proj1 = nn.Conv2d(d_model, d_ffn * 2, kernel_size=1)
        self.channel_proj2 = nn.Conv2d(d_ffn, d_model, kernel_size=1)
        self.cgu = ChannelGatingUnit(d_ffn)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        # x = F.gelu(self.channel_proj1(x))
        x = F.relu(self.channel_proj1(x))
        x = self.cgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class gMLP(nn.Module):

    def __init__(self, d_model=128, d_ffn=256, num_layers=6):
        super().__init__()
        self.model_gmlp = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn) for _ in range(num_layers)])

    def forward(self, x):
        return self.model_gmlp(x)


class IprHead(nn.Module):

    def __init__(self, cfg=None):
        super(IprHead, self).__init__()
        W, H = cfg['MODEL.IMAGE_BACKBONE.OUTPUT_SHAPE']
        self.linspace_x = torch.arange(0.0, 1.0 * W, 1) / W
        self.linspace_y = torch.arange(0.0, 1.0 * H, 1) / H
        self.linspace_x = nn.Parameter(self.linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(self.linspace_y, requires_grad=False)

        from mmcv.cnn import build_conv_layer
        conv_cfg = dict(
            type='Conv2d', in_channels=192, out_channels=21, kernel_size=1)
        self.final_layer = build_conv_layer(conv_cfg)

    def _linear_expectation(self, heatmaps: torch.Tensor,
                            linspace: torch.Tensor) -> torch.Tensor:
        """Calculate linear expectation."""

        B, N, _, _ = heatmaps.shape
        heatmaps = heatmaps.mul(linspace).reshape(B, N, -1)
        expectation = torch.sum(heatmaps, dim=2, keepdim=True)

        return expectation

    def _flat_softmax(self, featmaps: torch.Tensor) -> torch.Tensor:
        """Use Softmax to normalize the featmaps in depthwise."""

        _, N, H, W = featmaps.shape

        featmaps = featmaps.reshape(-1, N, H * W)
        heatmaps = F.softmax(featmaps, dim=2)
        return heatmaps.reshape(-1, N, H, W)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = self.final_layer(feats)
        heatmaps = self._flat_softmax(feats)
        x_fea = heatmaps.sum(dim=2)
        y_fea = heatmaps.sum(dim=3)
        pred_x = x_fea.mul(self.linspace_x)
        pred_y = y_fea.mul(self.linspace_y)
        pred_x = pred_x.sum(dim=-1, keepdim=True)
        pred_y = pred_y.sum(dim=-1, keepdim=True)
        coords = torch.cat([pred_x, pred_y], dim=-1)
        return coords


def quaternion_to_rotation_matrix(q, JPL_flag=False):
    # x, y ,z ,w
    if JPL_flag:
        # JPL
        q[0:3] = -q[0:3]
    # Hamilton
    rot_matrix = np.array([[
        1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 *
        (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2])
    ],
                           [
                               2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 *
                               (q[0] * q[0] + q[2] * q[2]), 2 *
                               (q[1] * q[2] - q[3] * q[0])
                           ],
                           [
                               2 * (q[0] * q[2] - q[3] * q[1]), 2 *
                               (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 *
                               (q[0] * q[0] + q[1] * q[1])
                           ]],
                          dtype=np.float32)
    return rot_matrix


@MODELS.register_module()
class TopdownPoseLiftingEstimator(TopdownPoseEstimator):

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 lifting_loss: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone,
            neck,
            head,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg=init_cfg)
        # cfg = {
        #     'MODEL.IMAGE_BACKBONE.OUTPUT_SHAPE': (32, 32),
        #     'MODEL.IMAGE_BACKBONE.OUTPUT_CHANNEL_NUM': 192,
        #     'MODEL.IMAGE_BACKBONE.KEYPOINT_NUM': 21
        # }
        # self.iprhead = IprHead(cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_num = 55
        self.liftnet = gMLP(
            d_model=2 * self.channel_num,
            d_ffn=4 * self.channel_num,
            num_layers=3)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                2 * self.channel_num, 2 * self.channel_num, kernel_size=1),
            nn.SyncBatchNorm(2 * self.channel_num), nn.ReLU(),
            nn.Conv2d(self.channel_num * 2, 2 * 21, kernel_size=1))

        self.lifting_loss = MODELS.build(lifting_loss)

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward(self, inputs: Tensor):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """

        x = self.extract_feat(inputs)
        if self.with_head:
            x = self.head.forward(x)
            if isinstance(x, list):
                x = x[-1]
        return x

    def lifting_head(self, inputs: Tensor, feats_pyramid: Tensor,
                     data_samples: SampleList) -> dict:
        feats = feats_pyramid[-1]  # biggest size featmap
        device = feats.device
        xy_sigma, heatmap = self.head.forward(feats_pyramid)
        xy_sigma = xy_sigma.detach()  # fix 2d model params

        # from IPython import embed; embed()

        B = int(inputs.shape[0] / 2)
        N = 2
        H, W = data_samples[0].input_size
        K = xy_sigma.shape[1]  # (B,21, 2)

        # kpt2d output to crop wh
        uv_coord_im_pred_crop_right = xy_sigma[..., :2] * torch.tensor(
            [W, H]).to(device)
        uv_coord_im_pred_crop_right = uv_coord_im_pred_crop_right.view(
            B, N, K, 2)

        leftcam_cam_matrix = []
        rightcam_cam_matrix = []
        lr_p = []
        lr_rot_matrix = []
        hand3d_gt = []
        is_left_hands = []

        uv_coord_im_gt_global = []

        all_inv_warp_mat = torch.zeros(B * 2, 3, 2).to(device)
        all_inv_warp_mat.requires_grad = False

        for i, data_sample in enumerate(data_samples):
            if i % 2 == 0:
                leftcam_cam_matrix.append(
                    np.array(data_sample.meta['cam_matrix_left']).T)
                lr_p.append(data_sample.meta['leftcam_p_rightcam'])
                lr_rot_matrix.append(
                    quaternion_to_rotation_matrix(
                        data_sample.meta['leftcam_q_rightcam']))
                hand3d_gt.append(data_sample.meta['kp3d_spline'])
                if data_sample.meta['category_id'] == 1:  # 1: left, 2: right
                    is_left_hands.append(1)
                else:
                    is_left_hands.append(0)
            else:
                rightcam_cam_matrix.append(
                    np.array(data_sample.meta['cam_matrix_right']).T)

            warp_mat = data_sample.metainfo['warp_mat']
            inv_warp_mat = cv2.invertAffineTransform(warp_mat).astype(
                np.float32)
            inv_warp_mat = torch.from_numpy(inv_warp_mat).to(device)  # (2,3)
            all_inv_warp_mat[i] = inv_warp_mat.transpose(0, 1)  # (3,2)

            uv_coord_im_gt_global.append(data_sample.gt_instances.keypoints)

        leftcam_cam_matrix = torch.tensor(
            np.array(leftcam_cam_matrix)).to(device).float()
        rightcam_cam_matrix = torch.tensor(
            np.array(rightcam_cam_matrix)).to(device).float()
        lr_p = torch.tensor(np.array(lr_p)).to(device).float()
        lr_rot_matrix = torch.tensor(
            np.array(lr_rot_matrix)).to(device).float()
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).to(device).float()
        left_hand = torch.tensor(np.array(is_left_hands)).to(device).float()
        uv_coord_im_gt_global = torch.tensor(
            np.array(uv_coord_im_gt_global)).to(device).float()
        uv_coord_im_gt_global = uv_coord_im_gt_global.view(B, N, K, 2)

        def recover_hand(uv_coord_im_pred, left_hand, device, w):
            recover_uv_coord_im_pred = (
                1 - left_hand.view(size=(-1, 1, 1, 1))
            ) * uv_coord_im_pred + left_hand.view(size=(-1, 1, 1, 1)) * (
                torch.tensor([w - 1, 0]).view(size=(1, 1, 1, 2)).to(device) +
                torch.tensor([-1, 1]).view(size=(1, 1, 1, 2)).to(device) *
                uv_coord_im_pred)
            return recover_uv_coord_im_pred

        uv_coord_im_pred_crop_leftright = uv_coord_im_pred_crop_right
        # uv_coord_im_pred_crop_leftright = recover_hand(
        #     uv_coord_im_pred_crop_right, left_hand, device, W)

        # embed()

        uv_coord_im_pred_crop_leftright = uv_coord_im_pred_crop_leftright.view(
            B * N, K, 2)

        # from crop uv to global uv
        uv_coord_im_pred = torch.cat([
            uv_coord_im_pred_crop_leftright,
            torch.ones(B * 2, K, 1, device=device)
        ],
                                     dim=-1)
        uv_coord_im_pred_global = torch.bmm(uv_coord_im_pred, all_inv_warp_mat)
        uv_coord_im_pred_global = uv_coord_im_pred_global.view(B, N, K, 2)

        uv_coord_im_pred_global = recover_hand(uv_coord_im_pred_global,
                                               left_hand, device, 640)

        uv_coord_im_gt_global = recover_hand(uv_coord_im_gt_global, left_hand,
                                             device, 640)

        # embed()

        # joint_seq = torch.zeros((B, 21, 3)).to(device)

        # feature = self.avg_pool(feats_final).view((B * N, -1))

        # x=(u-cx)/fx, y=(v-cy)/fy
        leftcam_uv = uv_coord_im_pred_global[:, 0]  # (B, 21, 2)
        leftcam_x = (leftcam_uv[:, :, 0] - leftcam_cam_matrix[:, 0, 2].view(
            (B, 1))) / leftcam_cam_matrix[:, 0, 0].view((B, 1))
        leftcam_y = (leftcam_uv[:, :, 1] - leftcam_cam_matrix[:, 1, 2].view(
            (B, 1))) / leftcam_cam_matrix[:, 1, 1].view((B, 1))
        leftcam_xy = torch.cat(
            (leftcam_x.unsqueeze(-1), leftcam_y.unsqueeze(-1)),
            dim=2)  # (B, 21, 2)
        rightcam_uv = uv_coord_im_pred_global[:, 1]  # (B, 21, 2)
        rightcam_x = (rightcam_uv[:, :, 0] - rightcam_cam_matrix[:, 0, 2].view(
            (B, 1))) / rightcam_cam_matrix[:, 0, 0].view((B, 1))
        rightcam_y = (rightcam_uv[:, :, 1] - rightcam_cam_matrix[:, 1, 2].view(
            (B, 1))) / rightcam_cam_matrix[:, 1, 1].view((B, 1))
        rightcam_xy = torch.cat(
            (rightcam_x.unsqueeze(-1), rightcam_y.unsqueeze(-1)),
            dim=2)  # (B, 21, 2)

        Tmatrix_leftcam = torch.tensor(
            (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)).view((1, -1)).to(device)

        # ZeroPad = torch.zeros((1, 10)).to(device)

        # 21*2 + 3+9 + 1 + 21*3 + 10 =   128
        feature1 = torch.cat((leftcam_xy.view(
            (B, -1)), Tmatrix_leftcam.repeat(B, 1), left_hand.view((B, -1))),
                             dim=1).view((B, self.channel_num, 1, 1))
        feature2 = torch.cat((rightcam_xy.view((B, -1)), lr_p.view(
            (B, -1)), lr_rot_matrix.view((B, -1)), left_hand.view((B, -1))),
                             dim=1).view((B, self.channel_num, 1, 1))
        output = self.liftnet(torch.cat((feature1, feature2), dim=1).float())
        output = self.last_layer(output).view((B, -1, 1, 1))  # [64, 42, 1, 1]

        leftcam_Z = output[:, :21].view((B, 21, 1))
        leftcam_XYZ = torch.cat((leftcam_xy * leftcam_Z, leftcam_Z),
                                dim=2).view((B, 21, 3, 1))
        rightcam_Z = output[:, 21:21 * 2].reshape((B, 21, 1))
        rightcam_XYZ = torch.cat((rightcam_xy * rightcam_Z, rightcam_Z),
                                 dim=2).view((B * 21, 3, 1))

        rightcam_XYZ = (torch.bmm(
            lr_rot_matrix.view((B, 1, 3, 3)).repeat(1, 21, 1, 1).view(
                (B * 21, 3, 3)), rightcam_XYZ) + lr_p.view(
                    (B, 1, 3, 1)).repeat(1, 21, 1, 1).view(
                        (B * 21, 3, 1))).view((B, 21, 3, 1))

        corruption_cam = torch.tensor(0.5).to(device)
        hand3d_pred = (corruption_cam * leftcam_XYZ +
                       (1 - corruption_cam) * rightcam_XYZ).view(B, 21, 3)

        # hand3d_pred to 2d keypoints
        leftcam_uv_reproj = torch.matmul(hand3d_pred,
                                         leftcam_cam_matrix.permute(0, 2, 1))
        leftcam_uv_reproj = leftcam_uv_reproj[..., :2] / leftcam_uv_reproj[...,
                                                                           2:]

        rightcam_uv_reproj = torch.matmul(
            hand3d_pred, lr_rot_matrix) - torch.matmul(
                lr_rot_matrix.permute(0, 2, 1), lr_p.unsqueeze(-1)).reshape(
                    (-1, 1, 3))
        rightcam_uv_reproj = torch.matmul(rightcam_uv_reproj,
                                          rightcam_cam_matrix.permute(0, 2, 1))
        rightcam_uv_reproj = rightcam_uv_reproj[..., :2] / rightcam_uv_reproj[
            ..., 2:]

        leftcam_uv_gt = uv_coord_im_gt_global[:, 0]
        rightcam_uv_gt = uv_coord_im_gt_global[:, 1]

        # embed()

        ret = {
            'hand3d_pred': hand3d_pred,
            'hand3d_gt': hand3d_gt,
            'leftcam_uv_reproj': leftcam_uv_reproj,
            'leftcam_uv_pred': leftcam_uv,
            'leftcam_uv_gt': leftcam_uv_gt,
            'rightcam_uv_reproj': rightcam_uv_reproj,
            'rightcam_uv_pred': rightcam_uv,
            'rightcam_uv_gt': rightcam_uv_gt
        }
        return ret

    @format_data
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        feats_pyramid = self.extract_feat(inputs)
        ret = self.lifting_head(inputs, feats_pyramid, data_samples)

        # from IPython import embed; embed()

        pred_for_loss = [
            ret['hand3d_pred'], ret['leftcam_uv_reproj'],
            ret['rightcam_uv_reproj']
        ]
        targ_for_loss = [
            ret['hand3d_gt'], ret['leftcam_uv_gt'], ret['rightcam_uv_gt']
        ]

        losses = self.lifting_loss(pred_for_loss, targ_for_loss)
        loss_mse_3d, loss_mse_2d_leftcam, loss_mse_2d_rightcam = losses

        # loss_mse_3d = self.l2_loss(ret['hand3d_pred'],
        #                            ret['hand3d_gt'])  # 单位 m^2

        # loss_mse_2d_leftcam = self.l2_loss(
        #     ret['leftcam_uv_pred'],
        #     ret['leftcam_uv']) / 4e6  # 单位转换 pix^2 => m^2

        # loss_mse_2d_rightcam = self.l2_loss(ret['rightcam_uv_pred'],
        #                                     ret['rightcam_uv']) / 4e6

        # from IPython import embed
        # embed()

        losses_dict = dict(
            loss_mse_3d=loss_mse_3d,
            loss_mse_2d_leftcam=loss_mse_2d_leftcam,
            loss_mse_2d_rightcam=loss_mse_2d_rightcam,
        )

        return losses_dict

    @format_data
    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)

        # from IPython import embed; embed()

        ret = self.lifting_head(inputs, feats, data_samples)

        batch_pred_instances = []

        for b in range(ret['hand3d_pred'].shape[0]):
            batch_pred_instances.append(
                InstanceData(
                    keypoints=ret['hand3d_pred'][b],
                    keypoint_scores=torch.ones((21))))

        results = self.add_pred_to_datasample(batch_pred_instances, None,
                                              data_samples)

        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:

        batch_data_samples = batch_data_samples[::2]  # 3d gt信息保存在左目

        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        # output_keypoint_indices = self.test_cfg.get('output_keypoint_indices', None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            pred_instances.keypoints3d = pred_instances.keypoints.cpu().numpy()
            pred_instances.keypoint_scores = np.ones(
                pred_instances.keypoints.shape[0])

            data_sample.pred_instances = pred_instances

            data_sample.gt_instances.keypoints3d = np.array(
                [data_sample.meta['kp3d_spline']])
            data_sample.gt_instances.keypoints_visible = np.ones(
                (1, len(data_sample.meta['kp3d_spline'])))
            # data_sample.

            # data_sample.gt_instances.keypoints3d = data_sample.meta['kp3d_spline']

        return batch_data_samples
