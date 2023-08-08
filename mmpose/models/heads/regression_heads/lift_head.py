# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmpose.models.utils.gmlp import gMLP
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


@MODELS.register_module()
class LiftHead(BaseModule):
    """liftHead for getting 3d keypoints from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 channel_num: int = 55,
                 output_num: int = 42,
                 undistort: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_num = channel_num
        self.liftnet = gMLP(
            d_model=2 * self.channel_num,
            d_ffn=4 * self.channel_num,
            num_layers=3)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                2 * self.channel_num, 2 * self.channel_num, kernel_size=1),
            nn.SyncBatchNorm(2 * self.channel_num), nn.ReLU(),
            nn.Conv2d(self.channel_num * 2, output_num, kernel_size=1))
        self.lift_loss = MODELS.build(lift_loss)
        self.undistort = undistort

    @staticmethod
    def check_cam_matrix(cam_matrix):
        cam_matrix = np.array(cam_matrix)
        assert cam_matrix.shape == (3, 3)
        if cam_matrix[0, 2] * cam_matrix[1, 2] < 1e-2:
            cam_matrix = cam_matrix.T
        return cam_matrix

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        output = self.liftnet(feats)
        output = self.last_layer(output).view(
            (feats.shape[0], -1, 1, 1))  # [64, 42, 1, 1]
        return output

    def preprocess(self, feats, batch_data_samples):
        xy_coord = feats
        B = int(len(batch_data_samples) / 2)
        N = 2
        H, W = batch_data_samples[0].input_size
        K = xy_coord.shape[1]  # (B,21, 2)

        # kpt2d output to crop wh
        uv_coord_im_pred_crop_right = xy_coord[..., :2] * torch.tensor(
            [W, H]).cuda()
        uv_coord_im_pred_crop_right = uv_coord_im_pred_crop_right.view(
            B, N, K, 2)

        leftcam_cam_matrix = []
        rightcam_cam_matrix = []
        lr_p = []
        lr_rot_matrix = []
        hand3d_gt = []
        is_left_hands = []

        uv_coord_im_gt_global = []

        all_inv_warp_mat = torch.zeros(B * 2, 3, 2).cuda()
        all_inv_warp_mat.requires_grad = False
        for i in range(B):
            # left camera
            left_data = batch_data_samples[2 * i]
            left_camera = left_data.meta['ori_camera']
            left_cam_matrix = left_camera.uv_to_window_matrix()
            leftcam_cam_matrix.append(left_cam_matrix)
            hand3d_gt.append(left_data.gt_instances.keypoints3d[0])
            if left_data.meta['category_id'] == 1:  # 1: left, 2: right
                is_left_hands.append(1)
            else:
                is_left_hands.append(0)  # right hand
            left_warp_mat = left_data.metainfo['warp_mat']
            left_inv_warp_mat = cv2.invertAffineTransform(
                left_warp_mat).astype(np.float32)
            left_inv_warp_mat = torch.from_numpy(
                left_inv_warp_mat).cuda()  # (2,3)
            all_inv_warp_mat[2 * i] = left_inv_warp_mat.transpose(0,
                                                                  1)  # (3,2)
            uv_coord_im_gt_global.append(left_data.gt_instances.keypoints)
            # right camera
            right_data = batch_data_samples[2 * i + 1]
            right_camera = right_data.meta['ori_camera']
            right_cam_matrix = right_camera.uv_to_window_matrix()
            rightcam_cam_matrix.append(right_cam_matrix)
            right_warp_mat = right_data.metainfo['warp_mat']
            right_inv_warp_mat = cv2.invertAffineTransform(
                right_warp_mat).astype(np.float32)
            right_inv_warp_mat = torch.from_numpy(
                right_inv_warp_mat).cuda()  # (2,3)
            all_inv_warp_mat[2 * i +
                             1] = right_inv_warp_mat.transpose(0, 1)  # (3,2)
            uv_coord_im_gt_global.append(right_data.gt_instances.keypoints)
            # left right  transform
            left_cam_xf = left_camera.camera_to_world_xf
            right_cam_xf = right_camera.camera_to_world_xf
            lr_t = np.dot(np.linalg.inv(left_cam_xf), right_cam_xf)
            lr_rot_matrix.append(lr_t[:3, :3])
            lr_p.append(lr_t[:3, 3])
        leftcam_cam_matrix = torch.tensor(
            np.array(leftcam_cam_matrix)).cuda().float()
        rightcam_cam_matrix = torch.tensor(
            np.array(rightcam_cam_matrix)).cuda().float()
        lr_p = torch.tensor(np.array(lr_p)).cuda().float()
        lr_rot_matrix = torch.tensor(np.array(lr_rot_matrix)).cuda().float()
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
        left_hand = torch.tensor(np.array(is_left_hands)).cuda().float()
        uv_coord_im_gt_global = torch.tensor(
            np.array(uv_coord_im_gt_global)).cuda().float()
        uv_coord_im_gt_global = uv_coord_im_gt_global.view(B, N, K, 2)

        def recover_hand(uv_coord_im_pred, left_hand, w):
            recover_uv_coord_im_pred = (
                1 - left_hand.view(size=(-1, 1, 1, 1))
            ) * uv_coord_im_pred + left_hand.view(size=(-1, 1, 1, 1)) * (
                torch.tensor([w - 1, 0]).view(size=(1, 1, 1, 2)).cuda() +
                torch.tensor([-1, 1]).view(size=(1, 1, 1, 2)).cuda() *
                uv_coord_im_pred)
            return recover_uv_coord_im_pred

        uv_coord_im_pred_crop_leftright = uv_coord_im_pred_crop_right
        # uv_coord_im_pred_crop_leftright = recover_hand(
        #     uv_coord_im_pred_crop_right, left_hand, device, W)

        uv_coord_im_pred_crop_leftright = uv_coord_im_pred_crop_leftright.view(
            B * N, K, 2)

        # from crop uv to global uv
        uv_coord_im_pred = torch.cat(
            [uv_coord_im_pred_crop_leftright,
             torch.ones(B * 2, K, 1).cuda()],
            dim=-1)
        uv_coord_im_pred_global = torch.bmm(uv_coord_im_pred, all_inv_warp_mat)
        uv_coord_im_pred_global = uv_coord_im_pred_global.view(B, N, K, 2)

        frame_width = batch_data_samples[0].meta['frame_width']
        uv_coord_im_pred_global = recover_hand(uv_coord_im_pred_global,
                                               left_hand, frame_width)

        uv_coord_im_gt_global = recover_hand(uv_coord_im_gt_global, left_hand,
                                             frame_width)

        if self.undistort:
            uv_coord_im_pred_global = uv_coord_im_pred_global.view(-1, K, 2)
            for i, data_sample in enumerate(batch_data_samples):
                camera_model = data_sample.meta['ori_camera']
                kpt2d_u = camera_model.undistort(
                    uv_coord_im_pred_global[i].cpu().numpy())
                uv_coord_im_pred_global[i] = torch.from_numpy(kpt2d_u).cuda()
            uv_coord_im_pred_global = uv_coord_im_pred_global.view(B, N, K, 2)

        uv_coord_im_pred_global = uv_coord_im_pred_global.detach()
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
            (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)).view((1, -1)).cuda()

        feature1 = torch.cat((leftcam_xy.view(
            (B, -1)), Tmatrix_leftcam.repeat(B, 1), left_hand.view((B, -1))),
                             dim=1).view((B, self.channel_num, 1, 1))
        feature2 = torch.cat((rightcam_xy.view((B, -1)), lr_p.view(
            (B, -1)), lr_rot_matrix.view((B, -1)), left_hand.view((B, -1))),
                             dim=1).view((B, self.channel_num, 1, 1))
        feats = torch.torch.cat((feature1, feature2), dim=1).float()
        return feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p, \
            leftcam_cam_matrix, rightcam_cam_matrix, uv_coord_im_gt_global, \
            hand3d_gt

    def postprocess(self, output, leftcam_xy, rightcam_xy, lr_rot_matrix,
                    lr_p):
        B = output.shape[0]
        leftcam_Z = output[:, :21].view((B, 21, 1))
        leftcam_XYZ = torch.cat((leftcam_xy * leftcam_Z, leftcam_Z),
                                dim=2).view((B, 21, 3))
        rightcam_Z = output[:, 21:21 * 2].reshape((B, 21, 1))
        rightcam_XYZ = torch.cat((rightcam_xy * rightcam_Z, rightcam_Z),
                                 dim=2).view((B * 21, 3, 1))

        rightcam_XYZ = (torch.bmm(
            lr_rot_matrix.view((B, 1, 3, 3)).repeat(1, 21, 1, 1).view(
                (B * 21, 3, 3)), rightcam_XYZ) + lr_p.view(
                    (B, 1, 3, 1)).repeat(1, 21, 1, 1).view(
                        (B * 21, 3, 1))).view((B, 21, 3))

        corruption_cam = torch.tensor(0.5).cuda()
        hand3d_pred = (
            corruption_cam * leftcam_XYZ + (1 - corruption_cam) * rightcam_XYZ)

        return hand3d_pred, leftcam_XYZ, rightcam_XYZ

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p = self.preprocess(
            feats, batch_data_samples)[:5]
        output = self.forward(feats)
        hand3d_pred = self.postprocess(output, leftcam_xy, rightcam_xy,
                                       lr_rot_matrix, lr_p)[0]
        return hand3d_pred

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p,\
            leftcam_cam_matrix, rightcam_cam_matrix, uv_coord_im_gt_global, \
            hand3d_gt = self.preprocess(feats, batch_data_samples)
        output = self.forward(feats)
        hand3d_pred, leftcam_XYZ, rightcam_XYZ = self.postprocess(
            output, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p)
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

        major_gt = torch.cat(
            (hand3d_gt[:, 1:10, :], hand3d_gt[:, 13, :].unsqueeze(1)), dim=1)
        major_pred = torch.cat(
            (hand3d_pred[:, 1:10, :], hand3d_pred[:, 13, :].unsqueeze(1)),
            dim=1)

        # origin distance, no norm
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(hand3d_gt[:, 4, :] - hand3d_gt[:, 8, :], dim=-1)
        pred_for_loss = [
            hand3d_pred, leftcam_XYZ, rightcam_XYZ, leftcam_uv_reproj,
            rightcam_uv_reproj, dist_pred, major_pred
        ]
        targ_for_loss = [
            hand3d_gt, hand3d_gt, hand3d_gt, leftcam_uv_gt, rightcam_uv_gt,
            dist_gt, major_gt
        ]
        losses = self.lift_loss(pred_for_loss, targ_for_loss)
        (loss_mse_3d, loss_mse_3d_leftcam, loss_mse_3d_rightcam,
         loss_mse_2d_leftcam, loss_mse_2d_rightcam) = losses

        losses_dict = dict(
            loss_mse_3d=loss_mse_3d,
            loss_mse_3d_leftcam=loss_mse_3d_leftcam,
            loss_mse_3d_rightcam=loss_mse_3d_rightcam,
            loss_mse_2d_leftcam=loss_mse_2d_leftcam,
            loss_mse_2d_rightcam=loss_mse_2d_rightcam,
        )

        return losses_dict
