# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from mmengine.logging import MessageHub
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmpose.models.heads.nimble.nimble_utils import (
    SkeletonEncoder, _gen_rigid_features, adjust_predicted_angles,
    batch_rodrigues, cal_proportion, decode_svd, matrix_to_euler_angles,
    matrix_to_quaternion, trans_3d_2_2d)
from mmpose.models.heads.nimble.simple_NIMBLELayer import sim_NIMBLELayer
from mmpose.models.utils.gmlp import gMLP
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions
from mmpose.models.heads.regression_heads.lift_head import LiftHead


@MODELS.register_module()
class LiftNimbleHead(LiftHead):
    """liftHead for getting 3d rotation from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 channel_num: int = 55,
                 undistort: bool = False,
                 use_kp2d_gt=False,
                 kpt2d_with_depth: bool = False,
                 noRt=False,
                 lambda_t: int = -1,
                 corruption_cam: float = 0.5,
                 use_svd: bool = True,
                 use_nimble_part_para: bool = False,
                 shape_ncomp: int = 20,
                 pose_ncomp: int = 60,
                 reg_shape_type: int = 1,
                 skeleton_feature_dim: int = 64,
                 euler_or_quaternion: str = 'euler',
                 use_pose_pca: bool = True,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(lift_loss = lift_loss,
                         channel_num = channel_num,
                         undistort = undistort,
                         use_kp2d_gt = use_kp2d_gt,
                         kpt2d_with_depth = kpt2d_with_depth,
                         noRt = noRt, 
                         lambda_t = lambda_t,
                         init_cfg = init_cfg)

        # define the fix parameters
        self.used_nimble_para = [
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 21, 27, 28, 30, 33,
            39, 40, 42, 45, 51, 52, 54, 57
        ]
        self.used_nimble_para = [x - 3 for x in self.used_nimble_para]
        self.kp_index = [
            0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22,
            23, 24
        ]

        # define the liftnet model
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_num = channel_num
        self.lambda_t = lambda_t
        self.kpt2d_with_depth = kpt2d_with_depth
        feat_dim = 2 * self.channel_num
        if self.kpt2d_with_depth:
            feat_dim = feat_dim + 21
        if reg_shape_type > 1:
            feat_dim = feat_dim + skeleton_feature_dim
        self.liftnet = gMLP(d_model=feat_dim, d_ffn=feat_dim * 2, num_layers=3)
        self.rigid_samples = _gen_rigid_features()

        # define the full connection layer
        if reg_shape_type == 0:
            self.shape_ncomp = shape_ncomp
        elif reg_shape_type == 1 or reg_shape_type == 2:
            self.shape_ncomp = 1
        elif reg_shape_type == 3:
            self.shape_ncomp = 24
        self.pose_ncomp = pose_ncomp
        if use_nimble_part_para:
            self.pose_ncomp = len(self.used_nimble_para)
        output_num = self.shape_ncomp + self.pose_ncomp + 3
        if use_svd:
            self.output_num = output_num + 18  # 21 - 3
        else:
            self.output_num = output_num
        self.last_layer = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.SyncBatchNorm(feat_dim), nn.ReLU(),
            nn.Conv2d(feat_dim, self.output_num, kernel_size=1))
        self.lift_loss = MODELS.build(lift_loss)

        # define the fllow parameters
        self.use_svd = use_svd
        self.scale_parameter = 1000
        self.corruption_cam = corruption_cam
        self.undistort = undistort
        self.use_kp2d_gt = use_kp2d_gt
        self.noRt = noRt
        self.use_nimble_part_para = use_nimble_part_para
        self.euler_or_quaternion = euler_or_quaternion
        if self.euler_or_quaternion not in ['euler', 'quaternion']:
            raise ValueError('must in two pose way')

        # define nimble layer
        self.nimble_layer = sim_NIMBLELayer(
            device='cuda',
            shape_ncomp=self.shape_ncomp,
            pose_ncomp=self.pose_ncomp,
            use_pose_pca=use_pose_pca,
            reg_shape_type=reg_shape_type)

        # define the shape regression type
        self.reg_shape_type = reg_shape_type
        if self.reg_shape_type > 1:
            self.skeleton_feature_dim = skeleton_feature_dim
            self.skeleton_encoder = SkeletonEncoder(skeleton_feature_dim)

    def get_full_pose_with_part_pars(self, pose_reg):
        used_nimble_para = torch.tensor(self.used_nimble_para)
        pose_out = torch.zeros((pose_reg.shape[0], 57),
                               device=pose_reg.device,
                               dtype=torch.float32)
        pose_out[:, used_nimble_para] = pose_reg.to(torch.float32)
        return pose_out

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        output = self.liftnet(feats)
        output = self.last_layer(output).view((feats.shape[0], -1, 1, 1))
        return output


    def postprocess(self,
                    output,
                    left_hand,
                    leftcam_xy,
                    nimble_info,
                    hand3d_gt,
                    only_pre=False):

        B = output.shape[0]
        cuda_device = output.device

        pose_len = self.pose_ncomp
        rot_vector_t = output[:, :pose_len, 0, 0]
        if self.use_nimble_part_para:
            rot_vector_t = self.get_full_pose_with_part_pars(rot_vector_t)
        svd_begin = self.pose_ncomp + self.shape_ncomp
        shape_v = output[:, pose_len:svd_begin, 0, 0]
        pre_pt_features = output[:, svd_begin:, 0, 0]

        matrix_svd = decode_svd(
            pre_pt_features,
            self.rigid_samples,
        )

        pre_root_xyz = matrix_svd[:, 0:3, 3]
        pre_root_matrix = matrix_svd[:, 0:3, 0:3]
        pre_rot_vector = self.nimble_layer.generate_full_pose(
            rot_vector_t, normalized=True, with_root=False).view(-1, 20, 3)
        pre_shape_vector = shape_v

        if not only_pre:
            with torch.no_grad():
                gt_root_xyz = nimble_info['nimble_trans']
                gt_root_matrix = batch_rodrigues(
                    nimble_info['nimble_pose'][:, 0, :]).reshape(-1, 3, 3)
                init_root_rot = torch.zeros((B, 1, 3),
                                            requires_grad=True,
                                            device=cuda_device)
                gt_rot_vector = torch.cat(
                    (init_root_rot, nimble_info['nimble_pose'][:, 1:, :]),
                    dim=1)
                # gt_shape_vector = nimble_info['nimble_shape']

        def get_nimble_3d(root_xyz, root_matrix, rot_vector, shape_vector):

            _, bone_joints = self.nimble_layer.forward_simple(
                rot_vector, shape_vector)
            rebuild_joints = bone_joints[:, self.kp_index, :]
            root_rebuild_joints = rebuild_joints[:, 0:1, :]
            rebuild_joints_temp = rebuild_joints - root_rebuild_joints

            mask = left_hand == 1
            add_matrix = torch.eye(3).unsqueeze(0).expand(B, -1,
                                                          -1).to(cuda_device)
            add_matrix[mask, 0, 0] = -add_matrix[mask, 0, 0]
            root_matrix = torch.matmul(add_matrix, root_matrix)
            rebuild_joints_temp = torch.matmul(rebuild_joints_temp,
                                               root_matrix.transpose(1, 2))
            rebuild_joints_with_scale = \
                rebuild_joints_temp / self.scale_parameter

            xyz_point = rebuild_joints_with_scale + root_xyz.unsqueeze(1)
            return xyz_point

        if only_pre:
            pre_nimble_pre_root_pre_shape__xyz = get_nimble_3d(
                pre_root_xyz, pre_root_matrix, pre_rot_vector,
                pre_shape_vector)
            return pre_nimble_pre_root_pre_shape__xyz, \
                pre_root_xyz, pre_root_matrix, pre_rot_vector
        else:
            pre_root__xyz = get_nimble_3d(pre_root_xyz, pre_root_matrix,
                                          gt_rot_vector, pre_shape_vector)
            pre_nimble__xyz = get_nimble_3d(gt_root_xyz, gt_root_matrix,
                                            pre_rot_vector, pre_shape_vector)
            pre_all__xyz = get_nimble_3d(pre_root_xyz, pre_root_matrix,
                                         pre_rot_vector, pre_shape_vector)
            # gt_nimble_gt_root_gt_shape__xyz = get_nimble_3d(
            # gt_root_xyz, gt_root_matrix, gt_rot_vector, gt_shape_vector)
            return (pre_root__xyz, pre_nimble__xyz, pre_all__xyz, pre_root_xyz,
                    pre_root_matrix, pre_rot_vector)

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            (feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p, left_to_right_rt,
             leftcam_cam_matrix, rightcam_cam_matrix, uv_coord_im_pred_global,
             uv_coord_im_pred_global_distort, hand3d_gt, left_hand,
             nimble_info) = self.preprocess(feats, batch_data_samples)

            if self.reg_shape_type > 1:
                batch_num = feats.shape[0]
                skeleton_joints_info = self.skeleton_joints_info.repeat(
                    batch_num, 1).to(feats.device)
                skeleton_feature = self.skeleton_encoder(
                    skeleton_joints_info).view(batch_num,
                                               self.skeleton_feature_dim, 1,
                                               -1)
                feats = torch.cat((feats, skeleton_feature), dim=1)
            output = self.forward(feats)

        (hand3d_pred, pre_trans_xyz, pre_root_matrix,
         pre_rot_vector) = self.postprocess(
             output,
             left_hand,
             leftcam_xy,
             nimble_info,
             hand3d_gt,
             only_pre=True)

        return hand3d_pred, uv_coord_im_pred_global_distort, \
            pre_root_matrix, pre_rot_vector[:, 1:, :]

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        with torch.no_grad():
            (feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p, left_to_right_rt,
             leftcam_cam_matrix, rightcam_cam_matrix, uv_coord_im_pred_global,
             uv_coord_im_pred_global_distort, hand3d_gt, left_hand,
             nimble_info) = self.preprocess(feats, batch_data_samples)

        if self.reg_shape_type > 1:
            batch_num = feats.shape[0]
            skeleton_joints_info = self.skeleton_joints_info.repeat(
                batch_num, 1).to(feats.device)
            skeleton_feature = self.skeleton_encoder(
                skeleton_joints_info).view(batch_num,
                                           self.skeleton_feature_dim, 1, -1)
            feats = torch.cat((feats, skeleton_feature), dim=1)

        output = self.forward(feats)

        B = output.shape[0]
        # 3d 损失
        (pred_3d_way1, pred_3d_way2, hand3d_pred, pre_trans_xyz,
         pre_root_matrix,
         pre_rot_vector) = self.postprocess(output, left_hand, leftcam_xy,
                                            nimble_info, hand3d_gt, False)

        # 直接监督rot和trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz
        pre_child_vector = pre_rot_vector[:, 1:, :].reshape(-1, 3)
        pre_child_matrix = batch_rodrigues(pre_child_vector).reshape(
            B, -1, 3, 3)
        pre_matrix = torch.cat(
            (pre_root_matrix.unsqueeze(1), pre_child_matrix),
            dim=1).reshape(-1, 3, 3)
        if self.euler_or_quaternion == 'euler':
            pre_euler = matrix_to_euler_angles(pre_matrix, 'XYZ')
        elif self.euler_or_quaternion == 'quaternion':
            pre_nimble_pose = matrix_to_quaternion(pre_matrix).reshape(B, -1)

        if 'nimble_pose' in nimble_info.keys(
        ) and 'nimble_trans' in nimble_info.keys():
            gt_nimble_trans = nimble_info['nimble_trans']
            gt_nimble_pose_roctor = nimble_info['nimble_pose'].reshape(-1, 3)
            gt_nimble_pose_matirx = batch_rodrigues(
                gt_nimble_pose_roctor).reshape(-1, 3, 3)
            if self.euler_or_quaternion == 'euler':
                gt_euler = matrix_to_euler_angles(gt_nimble_pose_matirx, 'XYZ')
                pre_nimble_pose = adjust_predicted_angles(pre_euler,
                                                          gt_euler).reshape(
                                                              B, -1)
                gt_nimble_pose = gt_euler.reshape(B, -1)
            elif self.euler_or_quaternion == 'quaternion':
                gt_nimble_pose = matrix_to_quaternion(
                    gt_nimble_pose_matirx).reshape(B, -1)

        # 2d重投影损失 这里把pre设置为gt
        leftcam_uv_pre, rightcam_uv_pre = trans_3d_2_2d(
            hand3d_pred, leftcam_cam_matrix, rightcam_cam_matrix,
            left_to_right_rt)
        leftcam_uv_gt, rightcam_uv_gt = trans_3d_2_2d(hand3d_gt,
                                                      leftcam_cam_matrix,
                                                      rightcam_cam_matrix,
                                                      left_to_right_rt)

        # xyz比例约束
        proportion_xyz_pre = cal_proportion(leftcam_uv_pre, leftcam_cam_matrix)
        proportion_xyz_gt = cal_proportion(leftcam_uv_gt, leftcam_cam_matrix)

        # 数据归一化
        leftcam_uv_pre = leftcam_uv_pre / 500
        rightcam_uv_pre = rightcam_uv_pre / 500
        leftcam_uv_gt = leftcam_uv_gt / 500
        rightcam_uv_gt = rightcam_uv_gt / 500

        # pinch 损失
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(hand3d_gt[:, 4, :] - hand3d_gt[:, 8, :], dim=-1)

        pred_for_loss = [
            pred_3d_way1, pred_3d_way2, hand3d_pred, leftcam_uv_pre,
            rightcam_uv_pre, dist_pred, proportion_xyz_pre, pre_nimble_pose,
            pre_nimble_trans
        ]
        targ_for_loss = [
            hand3d_gt, hand3d_gt, hand3d_gt, leftcam_uv_gt, rightcam_uv_gt,
            dist_gt, proportion_xyz_gt, gt_nimble_pose, gt_nimble_trans
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss)
        (loss_pre_root, loss_pre_nimble, loss_pre_all, loss_mse_2d_leftcam,
         loss_mse_2d_rightcam, loss_pinch, loss_scale, loss_nimble_pose,
         loss_nimble_trans) = losses

        if self.lambda_t > 0:
            mh = MessageHub.get_current_instance()
            cur_epoch = mh.get_info('epoch')
            if cur_epoch <= self.lambda_t:
                loss_mse_2d_leftcam = torch.tensor(
                    0.0,
                    device=loss_mse_2d_leftcam.device,
                    requires_grad=False)
                loss_mse_2d_rightcam = torch.tensor(
                    0.0,
                    device=loss_mse_2d_rightcam.device,
                    requires_grad=False)
                loss_scale = torch.tensor(
                    0.0, device=loss_scale.device, requires_grad=False)
                loss_nimble_pose = torch.tensor(
                    0.0, device=loss_nimble_pose.device, requires_grad=False)

        losses_dict = dict(
            loss_pre_root=loss_pre_root,
            loss_pre_nimble=loss_pre_nimble,
            loss_pre_all=loss_pre_all,
            loss_mse_2d_leftcam=loss_mse_2d_leftcam,
            loss_mse_2d_rightcam=loss_mse_2d_rightcam,
            loss_pinch=loss_pinch,
            loss_proportion=loss_scale,
            loss_nimble_pose=loss_nimble_pose,
            loss_nimble_trans=loss_nimble_trans)

        return losses_dict
