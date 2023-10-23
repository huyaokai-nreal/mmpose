# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from mmengine.logging import MessageHub
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmpose.models.utils.gmlp import gMLP
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


@MODELS.register_module()
class LiftHeadNano(BaseModule):
    """liftHead for getting 3d keypoints from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 channel_num: int = 55,
                 output_num: int = 42,   # 左目21 右目21，共42个点的深度
                 undistort: bool = False,
                 use_kp2d_gt=False,
                 kpt2d_with_depth: bool = False,
                 noRt=False,
                 lambda_t: int = -1,
                 corruption_cam: float = 0.5,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_num = channel_num
        self.lambda_t = lambda_t
        self.kpt2d_with_depth = kpt2d_with_depth
        feat_dim = 2 * self.channel_num
        if self.kpt2d_with_depth:
            feat_dim = feat_dim + 21
        self.corruption_cam = corruption_cam
        self.liftnet = gMLP(
            d_model=2 * self.channel_num + 21,
            d_ffn=4 * self.channel_num,
            num_layers=3)
        # self.liftnet = gMLP(d_model=feat_dim, d_ffn=feat_dim * 2, num_layers=3)
        self.last_layer = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.SyncBatchNorm(feat_dim), nn.ReLU(),
            nn.Conv2d(feat_dim, output_num, kernel_size=1))
        self.lift_loss = MODELS.build(lift_loss)
        self.undistort = undistort
        self.use_kp2d_gt = use_kp2d_gt
        self.noRt = noRt

    def forward(self, feats: Tuple[Tensor]) -> Tensor:   # feats[128, 131, 1, 1]
        # import ipdb;ipdb.set_trace()
        output = self.liftnet(feats)            # [128, 131, 1, 1]
        output = self.last_layer(output).view(
            (feats.shape[0], -1, 1, 1))         # [128, 42, 1, 1]
        return output

    def preprocess(self, feats, batch_data_samples):
        xy_coord = feats[..., :2]
        if self.kpt2d_with_depth:
            depth = feats[..., -1:][::2]
        B = int(len(batch_data_samples) / 2)
        N = 2
        H, W = batch_data_samples[0].input_size  # 128 128
        K = xy_coord.shape[1]  # 21

        uv_coord_im_pred_crop_right = xy_coord.view(
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
        for i, data_sample in enumerate(batch_data_samples):
            if i % 2 == 0:
                left_camera = data_sample.meta['ori_camera']
                left_cam_matrix = left_camera.uv_to_window_matrix()   # 左目内参
                leftcam_cam_matrix.append(left_cam_matrix)
                hand3d_gt.append(data_sample.gt_instances.keypoints3d[0])
                if data_sample.meta['category_id'] == 1:  # 1: left, 2: right
                    is_left_hands.append(1)
                else:
                    is_left_hands.append(0)
            else:
                right_camera = data_sample.meta['ori_camera']
                right_cam_matrix = right_camera.uv_to_window_matrix()  # 右目内参
                rightcam_cam_matrix.append(right_cam_matrix)
                left_cam_xf = left_camera.camera_to_world_xf   # 左目外参
                right_cam_xf = data_sample.meta['ori_xf'] @ right_camera.camera_to_world_xf   # 右目外参
                lr_t = np.dot(np.linalg.inv(left_cam_xf), right_cam_xf).astype(np.float32)   # 计算两个相机之间的坐标变换矩阵 lr_t 参考电子版十四讲P44；外参矩阵的逆：R->R^T  t->-t
                lr_rot_matrix.append(lr_t[:3, :3])   # 旋转矩阵
                lr_p.append(lr_t[:3, 3])  # 平移向量

            warp_mat = data_sample.metainfo['warp_mat']  #(2,3) ?
            inv_warp_mat = cv2.invertAffineTransform(warp_mat).astype(  # ?仿射变换
                np.float32)
            inv_warp_mat = torch.from_numpy(inv_warp_mat).cuda()  # (2,3)
            all_inv_warp_mat[i] = inv_warp_mat.transpose(0, 1)  # (3,2)

            uv_coord_im_gt_global.append(data_sample.gt_instances.keypoints)
        leftcam_cam_matrix = torch.tensor(
            np.array(leftcam_cam_matrix)).cuda().float()   # 左目内参集 [128, 3, 3]：相当于一份内参[3,3]重复了128遍
        rightcam_cam_matrix = torch.tensor(
            np.array(rightcam_cam_matrix)).cuda().float()   # 右目内参 [128, 3, 3]
        lr_p = torch.tensor(np.array(lr_p)).cuda().float()  # 左 -> 右 平移集
        lr_rot_matrix = torch.tensor(np.array(lr_rot_matrix)).cuda().float()  # # 左 -> 右 旋转集
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
        left_rel_depth = hand3d_gt[..., 2:3] - hand3d_gt[:, :1, 2:3]  # kpt3d变为相对根节点深度
        left_hand = torch.tensor(np.array(is_left_hands)).cuda().float()   # [128]  0（右手）或1（左手）
        uv_coord_im_gt_global = torch.tensor(
            np.array(uv_coord_im_gt_global)).cuda().float()
        # import ipdb;ipdb.set_trace()
        uv_coord_im_gt_global = uv_coord_im_gt_global[...,:2].view(B, N, K, 2)   # gt kpt2d (128, 2 , 21, 2) 

        def recover_hand(uv_coord_im_pred, left_hand, w):
            recover_uv_coord_im_pred = (
                1 - left_hand.view(size=(-1, 1, 1, 1))
            ) * uv_coord_im_pred + left_hand.view(size=(-1, 1, 1, 1)) * (
                torch.tensor([w - 1, 0]).view(size=(1, 1, 1, 2)).cuda() +
                torch.tensor([-1, 1]).view(size=(1, 1, 1, 2)).cuda() *
                uv_coord_im_pred)
            return recover_uv_coord_im_pred

        uv_coord_im_pred_crop_leftright = uv_coord_im_pred_crop_right  # [128, 2, 21, 2]
        # uv_coord_im_pred_crop_leftright = recover_hand(
        #     uv_coord_im_pred_crop_right, left_hand, device, W)

        uv_coord_im_pred_crop_leftright = uv_coord_im_pred_crop_leftright.view(   # [256, 21, 2]
            B * N, K, 2)

        # from crop uv to global uv
        uv_coord_im_pred = torch.cat(
            [uv_coord_im_pred_crop_leftright,
             torch.ones(B * 2, K, 1).cuda()],
            dim=-1)
        # import ipdb;ipdb.set_trace()
        uv_coord_im_pred_global_distort = torch.bmm(uv_coord_im_pred,   # (256, 21, 3)
                                                    all_inv_warp_mat)   # (256, 3, 2) 批量矩阵乘法      # 像素坐标还原width=480尺寸
        
        uv_coord_im_pred_global_distort = uv_coord_im_pred_global_distort.view(
            B, N, K, 2)
        
        # import ipdb;ipdb.set_trace()
        frame_width = batch_data_samples[0].meta['frame_width']
        uv_coord_im_pred_global_distort_noflip = recover_hand(    # kpt 2d 翻转为原图像 用于预测3d点
            uv_coord_im_pred_global_distort, left_hand, frame_width)

        # uv_coord_im_gt_global = recover_hand(uv_coord_im_gt_global, left_hand,    # 如果不用use_kp2d_gt不影响结果
        #                                      frame_width)

        if self.use_kp2d_gt:
            uv_coord_im_pred_global = uv_coord_im_gt_global
            depth = left_rel_depth

        if self.undistort:
            uv_coord_im_pred_global = \
                uv_coord_im_pred_global_distort_noflip.clone().view(-1, K, 2)
            for i, data_sample in enumerate(batch_data_samples):
                camera_model = data_sample.meta['ori_camera']
                kpt2d_u = camera_model.undistort(
                    uv_coord_im_pred_global[i].cpu().numpy())
                uv_coord_im_pred_global[i] = torch.from_numpy(kpt2d_u).cuda()
            uv_coord_im_pred_global = uv_coord_im_pred_global.view(B, N, K, 2)
        else:
            uv_coord_im_pred_global = \
                uv_coord_im_pred_global_distort_noflip.clone()

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

        # 显式的使用Rt
        if self.noRt:
            rightcam_xy_normplane = torch.cat(
                (rightcam_xy, torch.ones(B, K, 1).cuda()), dim=2)
            rightcam_xyz1_inworld = (torch.bmm(
                lr_rot_matrix.view((B, 1, 3, 3)).repeat(1, 21, 1, 1).view(
                    (B * 21, 3, 3)), rightcam_xy_normplane.view(
                        (B * K, 3, 1))) + lr_p.view(
                            (B, 1, 3, 1)).repeat(1, 21, 1, 1).view(
                                (B * 21, 3, 1))).view((B, 21, 3))
            feats = torch.cat(
                (leftcam_xy.view(B, -1), rightcam_xyz1_inworld.view(
                    (B, -1)), left_hand.view((B, -1))),
                dim=1).view(B, self.channel_num * 2, 1,
                            1).float()  # 21*2+21*3+1
        # 隐式的使用Rt
        else:
            Tmatrix_leftcam = torch.tensor(
                (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)).view((1, -1)).cuda()
            feature1 = torch.cat((leftcam_xy.view(
                (B, -1)), Tmatrix_leftcam.repeat(B, 1), left_hand.view(
                    (B, -1))),
                                 dim=1).view((B, self.channel_num, 1, 1))
            feature2 = torch.cat((rightcam_xy.view((B, -1)), lr_p.view(
                (B, -1)), lr_rot_matrix.view((B, -1)), left_hand.view(
                    (B, -1))),
                                 dim=1).view((B, self.channel_num, 1, 1))
            if self.kpt2d_with_depth:
                feats = torch.torch.cat(
                    (feature1, feature2, depth.reshape(
                        (B, 21, 1, 1))), dim=1).float()
            else:
                feats = torch.cat((feature1, feature2), dim=1).float()

        return (feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p,
                leftcam_cam_matrix, rightcam_cam_matrix,
                uv_coord_im_pred_global, uv_coord_im_pred_global_distort,
                hand3d_gt)

    def postprocess(self, output, leftcam_xy, rightcam_xy, lr_rot_matrix,
                    lr_p):
        B = output.shape[0]    # output:[128, 42, 1, 1] 
        # 左右目2d坐标和模型输出左右目的深度结合为左右目3d坐标
        leftcam_Z = output[:, :21].view((B, 21, 1))   # 左目深度128,21,1
        leftcam_XYZ = torch.cat((leftcam_xy * leftcam_Z, leftcam_Z),
                                dim=2).view((B, 21, 3))
        rightcam_Z = output[:, 21:21 * 2].reshape((B, 21, 1))
        rightcam_XYZ = torch.cat((rightcam_xy * rightcam_Z, rightcam_Z),
                                 dim=2).view((B * 21, 3, 1))
        # 世界坐标系就是左目系（右下前）。右目系需要转为左目系
        rightcam_XYZ = (torch.bmm(
            lr_rot_matrix.view((B, 1, 3, 3)).repeat(1, 21, 1, 1).view(
                (B * 21, 3, 3)), rightcam_XYZ) + lr_p.view(
                    (B, 1, 3, 1)).repeat(1, 21, 1, 1).view(
                        (B * 21, 3, 1))).view((B, 21, 3))
        # hand_gt是左右目的3d合并出来的
        hand3d_pred = (
            self.corruption_cam * leftcam_XYZ +
            (1 - self.corruption_cam) * rightcam_XYZ)

        return hand3d_pred, leftcam_XYZ, rightcam_XYZ  # 所以三个kpt3d都是左目系下的，可以作相互比较。

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        feats = torch.tensor(feats,dtype=torch.float32).cuda()
        with torch.no_grad():
            (feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p,
             leftcam_cam_matrix, rightcam_cam_matrix, uv_coord_im_pred_global,
             uv_coord_im_pred_global_distort,
             hand3d_gt) = self.preprocess(feats, batch_data_samples)
        
        output = self.forward(feats)
        hand3d_pred = self.postprocess(output, leftcam_xy, rightcam_xy,
                                       lr_rot_matrix, lr_p)[0]
        return hand3d_pred, uv_coord_im_pred_global_distort

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        with torch.no_grad():
            (feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p,
             leftcam_cam_matrix, rightcam_cam_matrix, uv_coord_im_pred_global,
             uv_coord_im_pred_global_distort,
             hand3d_gt) = self.preprocess(feats, batch_data_samples)   # feats:[256, 21, 3]
        output = self.forward(feats)      # output:[128, 42, 1, 1]     feats:[128, 131, 1, 1]
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

        leftcam_uv_gt = uv_coord_im_pred_global[:, 0]
        rightcam_uv_gt = uv_coord_im_pred_global[:, 1]

        major_gt = torch.cat(
            (hand3d_gt[:, 1:10, :], hand3d_gt[:, 13, :].unsqueeze(1)), dim=1)
        major_pred = torch.cat(
            (hand3d_pred[:, 1:10, :], hand3d_pred[:, 13, :].unsqueeze(1)),
            dim=1)

        left_major_pred = torch.cat(
            (leftcam_XYZ[:, 1:10, :], leftcam_XYZ[:, 13, :].unsqueeze(1)),
            dim=1)
        right_major_pred = torch.cat(
            (rightcam_XYZ[:, 1:10, :], rightcam_XYZ[:, 13, :].unsqueeze(1)),
            dim=1)

        thumb_index_3d_gt = torch.cat(
            (hand3d_gt[:, 4:5, :], hand3d_gt[:, 8:9, :]), dim=1)
        thumb_index_3d_pred = torch.cat(
            (hand3d_pred[:, 4:5, :], hand3d_pred[:, 8:9, :]), dim=1)

        # origin distance, no norm
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(hand3d_gt[:, 4, :] - hand3d_gt[:, 8, :], dim=-1)
        pred_for_loss = [
            hand3d_pred, leftcam_XYZ, rightcam_XYZ, leftcam_uv_reproj,
            rightcam_uv_reproj, dist_pred
        ]
        targ_for_loss = [
            hand3d_gt, hand3d_gt, hand3d_gt, leftcam_uv_gt, rightcam_uv_gt,
            dist_gt
        ]
        losses = self.lift_loss(pred_for_loss, targ_for_loss)
        (loss_mse_3d, loss_mse_3d_leftcam, loss_mse_3d_rightcam,
         loss_mse_2d_leftcam, loss_mse_2d_rightcam, loss_pinch) = losses
        if self.lambda_t > 0:
            mh = MessageHub.get_current_instance()
            cur_epoch = mh.get_info('epoch')
            if cur_epoch <= self.lambda_t:
                loss_mse_2d_leftcam *= 0
                loss_mse_2d_rightcam *= 0
        losses_dict = dict(
            loss_mse_3d=loss_mse_3d,
            loss_mse_3d_leftcam=loss_mse_3d_leftcam,
            loss_mse_3d_rightcam=loss_mse_3d_rightcam,
            loss_mse_2d_leftcam=loss_mse_2d_leftcam,
            loss_mse_2d_rightcam=loss_mse_2d_rightcam,
            loss_pinch=loss_pinch
        )

        return losses_dict
