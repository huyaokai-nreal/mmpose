# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


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


@MODELS.register_module()
class LiftHead(BaseModule):
    """liftHead for getting 3d keypoints from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 channel_num: int = 55,
                 output_num: int = 42,
                 rm_distort: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None,
                 loss_pinch: bool = False):
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
        self.rm_distort = rm_distort
        self.loss_pinch = loss_pinch

    @staticmethod
    def check_cam_matrix(cam_matrix):
        cam_matrix = np.array(cam_matrix)
        assert cam_matrix.shape == (3, 3)
        if cam_matrix[0, 2] * cam_matrix[1, 2] < 1e-2:
            cam_matrix = cam_matrix.T
        return cam_matrix

    def forward(self, feats: Tuple[Tensor],
                batch_data_samples: OptSampleList) -> Tensor:
        xy_coord = feats
        device = xy_coord.device

        B = int(len(batch_data_samples) / 2)
        N = 2
        H, W = batch_data_samples[0].input_size
        K = xy_coord.shape[1]  # (B,21, 2)

        # kpt2d output to crop wh
        uv_coord_im_pred_crop_right = xy_coord[..., :2] * torch.tensor(
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

        for i, data_sample in enumerate(batch_data_samples):
            if i % 2 == 0:
                leftcam_cam_matrix.append(
                    self.check_cam_matrix(data_sample.meta['cam_matrix_left']))
                lr_p.append(data_sample.meta['leftcam_p_rightcam'])
                lr_rot_matrix.append(
                    quaternion_to_rotation_matrix(
                        data_sample.meta['leftcam_q_rightcam']))
                hand3d_gt.append(data_sample.meta['kp3d_spline'])
                if data_sample.meta['category_id'] == 1:  # 1: left, 2: right
                    is_left_hands.append(1)
                else:
                    is_left_hands.append(0)  # right hand
            else:
                rightcam_cam_matrix.append(
                    self.check_cam_matrix(
                        data_sample.meta['cam_matrix_right']))

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

        frame_width = batch_data_samples[0].meta['frame_width']
        # print(f"left_hand: {left_hand}, frame_width: {frame_width}")
        uv_coord_im_pred_global = recover_hand(uv_coord_im_pred_global,
                                               left_hand, device, frame_width)

        uv_coord_im_gt_global = recover_hand(uv_coord_im_gt_global, left_hand,
                                             device, frame_width)

        # fisheye undistort points
        Flora8_cam0 = {
            'K': [[240.47993898902308, 0.0, 238.24292414176563],
                  [0.0, 240.45010798807022, 318.9205573206751],
                  [0.0, 0.0, 1.0]],
            'D': [[
                0.012542161517124128, 0.04662863296034774, 0.0, 0.0,
                -0.04361866666639336, 0.009913181928564089
            ]],
            'T': [[
                -0.015358318543055027, 0.9884598092632204, 0.15070277874959054,
                -0.23011137230439574
            ],
                  [
                      0.13877556838512833, -0.14715455605391148,
                      0.9793298107644641, -0.31722350296789503
                  ],
                  [
                      0.9902047584570148, 0.035954722970622965,
                      -0.13491402530942131, 0.4421284041140596
                  ], [0.0, 0.0, 0.0, 1.0]]
        }

        Flora8_cam1 = {
            'K': [[239.93047131623948, 0.0, 242.78505964570425],
                  [0.0, 240.27271933775592, 320.50564674849875],
                  [0.0, 0.0, 1.0]],
            'D': [[
                0.01485201999344762, 0.03768701104219142, 0.0, 0.0,
                -0.034698759423003406, 0.007490907841159389
            ]],
            'T': [[
                -0.04024976662196289, 0.9905019207128727, 0.13147585843411586,
                -0.23391570709066306
            ],
                  [
                      -0.12629745136489084, -0.13557044230024617,
                      0.9826848980997158, -0.3178980489113908
                  ],
                  [
                      0.9911755193030196, 0.022947771975200926,
                      0.13055454682148604, 0.5785861700059319
                  ], [0.0, 0.0, 0.0, 1.0]]
        }

        # from IPython import embed; embed()

        def get_undistort_kp2d(cam_info, kp2d, B, K, device):
            # kpts shape: (N,K,2)
            kp2d = kp2d.reshape(-1, 2)
            kp2d = kp2d.detach().cpu().numpy()
            cam_k = np.array(cam_info['K'])
            cam_d = np.array(cam_info['D'])[:, (0, 1, 4, 5)]

            # fx, fy = cam_k[0, 0], cam_k[1, 1]
            # cx, cy = cam_k[0, 2], cam_k[1, 2]
            # kp2d[:, 0] = (kp2d[:, 0] - cx) / fx
            # kp2d[:, 1] = (kp2d[:, 1] - cy) / fy
            # newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(  # noqa
            #     cam_k, cam_d, (480, 640), None, balance=1)

            kp2d_undistort = cv2.fisheye.undistortPoints(
                kp2d.reshape(1, -1, 2), cam_k, cam_d, None,
                cam_k).reshape(B, K, 2)
            kp2d_undistort = torch.from_numpy(kp2d_undistort).to(device)

            return kp2d_undistort

        if self.rm_distort:
            uv_coord_im_pred_global[:, 0, :, :] = get_undistort_kp2d(
                Flora8_cam0, uv_coord_im_pred_global[:, 0, :, :], B, K, device)
            uv_coord_im_pred_global[:, 1, :, :] = get_undistort_kp2d(
                Flora8_cam1, uv_coord_im_pred_global[:, 1, :, :], B, K, device)
        uv_coord_im_pred_global = uv_coord_im_pred_global.detach()

        # from IPython import embed
        # embed()
        # use kp2d gt
        # uv_coord_im_pred_global = uv_coord_im_gt_global

        # print kp2d l2 error
        kp2d_l2 = torch.norm(
            uv_coord_im_pred_global.view(-1, 2) -
            uv_coord_im_gt_global.view(-1, 2),
            p=2,
            dim=1).mean()
        # print('kp2d_l2', kp2d_l2)

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

        # print(f"lr_rot_matrix: {lr_rot_matrix[0]}")
        # print(f"lr_p: {lr_p[0]}")
        # print(f"leftcam_xy: {leftcam_xy[0]}")
        # print(f"left_hand: {left_hand[0]}")
        # print(f"rightcam_xy: {rightcam_xy[0]}")
        # print(f"hand3d_gt: {hand3d_gt[0]}")
        # print(f"leftcam_cam_matrix: {leftcam_cam_matrix[0]}")
        # print(f"rightcam_cam_matrix: {rightcam_cam_matrix[0]}")
        # print(f"frame_width: {frame_width}")
        # print(f"uv_coord_im_gt_global: {uv_coord_im_gt_global[0]}")

        # 21*2 + 3+9 + 1 + 21*3 + 10 =   128
        feature1 = torch.cat((leftcam_xy.view(
            (B, -1)), Tmatrix_leftcam.repeat(B, 1), left_hand.view((B, -1))),
                             dim=1).view((B, self.channel_num, 1, 1))
        feature2 = torch.cat((rightcam_xy.view((B, -1)), lr_p.view(
            (B, -1)), lr_rot_matrix.view((B, -1)), left_hand.view((B, -1))),
                             dim=1).view((B, self.channel_num, 1, 1))
        output = self.liftnet(torch.cat((feature1, feature2), dim=1).float())
        output = self.last_layer(output).view((B, -1, 1, 1))  # [64, 42, 1, 1]

        # from IPython import embed; embed()

        # rle_depth_sigma = output[:, 21 * 2:21 * 3].view((B, 21, 1))

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

        corruption_cam = torch.tensor(0.5).to(device)
        hand3d_pred = (
            corruption_cam * leftcam_XYZ + (1 - corruption_cam) * rightcam_XYZ)

        # hand3d_pred = hand3d_gt
        # hand3d_pred = leftcam_XYZ
        # hand3d_pred = rightcam_XYZ

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

        kp2d_left_reproj_l2 = torch.norm(
            leftcam_uv_reproj.reshape(-1, 2) - leftcam_uv_gt.reshape(-1, 2),
            p=2,
            dim=1).mean()

        kp2d_right_reproj_l2 = torch.norm(
            rightcam_uv_reproj.reshape(-1, 2) - rightcam_uv_gt.reshape(-1, 2),
            p=2,
            dim=1).mean()

        # check dataset
        # print(f'kp2d_left_reproj_l2: {kp2d_left_reproj_l2}')
        # print(f'kp2d_right_reproj_l2: {kp2d_right_reproj_l2}')

        # from IPython import embed; embed()
        # # rle about depth
        # hand3d_pred_z = hand3d_pred[:, :, 2].view(B, 21, 1)  # get depth
        # hand3d_pred_z = torch.concat((hand3d_pred_z, hand3d_pred_z), dim=-1)
        # rle_depth_sigma = torch.concat((rle_depth_sigma, rle_depth_sigma), dim=-1) # noqa

        # rle_pred = torch.concat((hand3d_pred_z, rle_depth_sigma), dim=2)
        # rle_gt = hand3d_gt[:, :, 2].view(B, 21, 1)
        # rle_gt = torch.concat([rle_gt, rle_gt], dim=-1)

        # hand3d_pred = output.view(B, -1, 3)

        # print kp3d l2 error
        kp3d_l2 = torch.norm(
            hand3d_pred.reshape(-1, 3) * 1000 -
            hand3d_gt.reshape(-1, 3) * 1000,
            p=2,
            dim=1).mean()
        kp3d_l2_left = torch.norm(
            leftcam_XYZ.reshape(-1, 3) * 1000 -
            hand3d_gt.reshape(-1, 3) * 1000,
            p=2,
            dim=1).mean()
        kp3d_l2_right = torch.norm(
            rightcam_XYZ.reshape(-1, 3) * 1000 -
            hand3d_gt.reshape(-1, 3) * 1000,
            p=2,
            dim=1).mean()
        # print(f'kp3d_l2: {kp3d_l2}')
        # print(f'kp3d_l2_left: {kp3d_l2_left}')
        # print(f'kp3d_l2_right: {kp3d_l2_right}')

        # leftcam as predict
        # hand3d_pred = leftcam_XYZ.view(B, 21, 3)
        # hand3d_pred = rightcam_XYZ.view(B, 21, 3)

        # 加一个拇指、食指距离pred_dist, gt_dist
        # normalization_3d
        pred_root, gt_root = hand3d_pred[:, 9], hand3d_gt[:, 9]
        pred_hand_length = torch.norm(
            hand3d_pred[:, 9] - hand3d_pred[:, 0], dim=-1)
        gt_hand_length = torch.norm(
            hand3d_pred[:, 9] - hand3d_pred[:, 0], dim=-1)

        pred_hand_length = pred_hand_length.repeat_interleave(3).reshape(-1, 3)
        gt_hand_length = gt_hand_length.repeat_interleave(3).reshape(-1, 3)

        pred_norm = hand3d_pred - pred_root.unsqueeze(dim=1)
        gt_norm = hand3d_gt - gt_root.unsqueeze(dim=1)

        pred_norm = pred_norm * 0.08 / pred_hand_length.unsqueeze(dim=1)
        gt_norm = gt_norm * 0.08 / gt_hand_length.unsqueeze(dim=1)

        # thumb index distance
        pred_dist = torch.norm(pred_norm[:, 4, :] - pred_norm[:, 8, :], dim=-1)
        gt_dist = torch.norm(gt_norm[:, 4, :] - gt_norm[:, 8, :], dim=-1)

        ret = {
            'hand3d_pred': hand3d_pred,
            'leftcam_XYZ': leftcam_XYZ,
            'rightcam_XYZ': rightcam_XYZ,
            'hand3d_gt': hand3d_gt,
            'leftcam_uv_reproj': leftcam_uv_reproj,
            'leftcam_uv_pred': leftcam_uv,
            'leftcam_uv_gt': leftcam_uv_gt,
            'rightcam_uv_reproj': rightcam_uv_reproj,
            'rightcam_uv_pred': rightcam_uv,
            'rightcam_uv_gt': rightcam_uv_gt,
            # 'rle_pred': rle_pred,
            # 'rle_gt': rle_gt
            'kp2d_l2': kp2d_l2,
            'kp3d_l2': kp3d_l2,
            'kp3d_l2_left': kp3d_l2_left,
            'kp3d_l2_right': kp3d_l2_right,
            'kp2d_left_reproj_l2': kp2d_left_reproj_l2,
            'kp2d_right_reproj_l2': kp2d_right_reproj_l2,
            'pred_dist': pred_dist,
            'gt_dist': gt_dist
        }
        return ret

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        ret = self.forward(feats, batch_data_samples)
        return ret['hand3d_pred']

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        ret = self.forward(feats, batch_data_samples)

        pred_for_loss = [
            ret['hand3d_pred'], ret['leftcam_XYZ'], ret['rightcam_XYZ'],
            ret['leftcam_uv_reproj'], ret['rightcam_uv_reproj']
        ]
        targ_for_loss = [
            ret['hand3d_gt'], ret['hand3d_gt'], ret['hand3d_gt'],
            ret['leftcam_uv_gt'], ret['rightcam_uv_gt']
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss)
        (loss_mse_3d, loss_mse_3d_leftcam, loss_mse_3d_rightcam,
         loss_mse_2d_leftcam, loss_mse_2d_rightcam) = losses
        loss_pinch = torch.tensor(0.0)
        if self.loss_pinch:
            low_thre, high_thre = 0.02, 0.04
            gt_gesture = [
                batch_data_samples[i].meta['gesture']
                for i in range(len(batch_data_samples)) if i % 2 == 0
            ]
            pinch_index = [
                i for i in range(len(gt_gesture))
                if gt_gesture[i] == 'Pinch' and ret['gt_dist'][i] < low_thre
            ]  # need add gesture tag in train json data
            no_pinch_index = [
                i for i in range(len(gt_gesture))
                if gt_gesture[i] == 'Pinch' and ret['gt_dist'][i] > high_thre
            ]
            # from IPython import embed;embed()
            loss_pinch = 0.7 * (
                torch.sum(
                    torch.minimum(ret['pred_dist'][pinch_index] - low_thre,
                                  torch.tensor(0.0))) +
                torch.sum(
                    torch.minimum(high_thre - ret['pred_dist'][no_pinch_index],
                                  torch.tensor(0.0))))  # 负值loss越小越好，即绝对值越大越好

        losses_dict = dict(
            loss_mse_3d=loss_mse_3d,
            loss_mse_3d_leftcam=loss_mse_3d_leftcam,
            loss_mse_3d_rightcam=loss_mse_3d_rightcam,
            loss_mse_2d_leftcam=loss_mse_2d_leftcam,
            loss_mse_2d_rightcam=loss_mse_2d_rightcam,
            loss_pinch=loss_pinch)

        return losses_dict
