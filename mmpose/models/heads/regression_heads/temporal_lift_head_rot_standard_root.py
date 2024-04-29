# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.logging import MessageHub
from torch import Tensor, nn

from mmpose.models.heads.nimble.nimble_utils import (adjust_predicted_angles,
                                                     batch_rodrigues,
                                                     cal_proportion,
                                                     matrix_to_euler_angles,
                                                     matrix_to_quaternion,
                                                     trans_3d_2_2d)
from mmpose.models.heads.regression_heads.lift_head_rot_standard import \
    LiftNimbleHeadStandard
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


@MODELS.register_module()
class TemporalLiftNimbleHeadStandardRoot(LiftNimbleHeadStandard):
    """liftHead for getting 3d rotation from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 d_ffn: int = 220,
                 undistort: bool = False,
                 kpt2d_with_depth: bool = False,
                 use_svd: bool = True,
                 use_nimble_part_para: bool = False,
                 shape_ncomp: int = 20,
                 pose_ncomp: int = 60,
                 reg_shape_type: int = 1,
                 skeleton_feature_dim: int = 64,
                 euler_or_quaternion: str = 'euler',
                 use_pose_pca: bool = True,
                 reproj: bool = False,
                 baseline=0.13,
                 reproj_thre=0,
                 iou_thre=0,
                 pad_2d=0,
                 lambda_t: int = -1,
                 corruption_cam: float = 0.5,
                 use_bone_loss: bool = True,
                 use_shape_smooth=False,
                 use_6d_pose_reg: bool = False,
                 all_use_kp2d_gt: bool = False,
                 seq_len: int = 4,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(
            lift_loss=lift_loss,
            d_ffn=d_ffn,
            undistort=undistort,
            reproj=reproj,
            pose_ncomp=pose_ncomp,
            euler_or_quaternion=euler_or_quaternion,
            baseline=baseline,
            use_svd=use_svd,
            reproj_thre=reproj_thre,
            iou_thre=iou_thre,
            pad_2d=pad_2d,
            use_bone_loss=use_bone_loss,
            use_6d_pose_reg=use_6d_pose_reg,
            lambda_t=lambda_t,
            all_use_kp2d_gt=all_use_kp2d_gt,
            init_cfg=init_cfg,
        )
        self.seq_len = seq_len

        self.last_layer = nn.Sequential(
            nn.Conv2d(self.feat_dim * 2, self.feat_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.feat_dim, self.output_num, kernel_size=1))
        self.temporal = nn.Sequential(
            nn.Conv2d(
                2 * self.channel_num * 2, 2 * self.channel_num, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                self.channel_num * 2, self.channel_num * 2, kernel_size=1))
        self.use_shape_smooth = use_shape_smooth
        if use_shape_smooth:
            self.shape_loss_func = F.l1_loss

        self.all_sigma_conv = nn.Conv2d(
            self.feat_dim * 2, 21 * 3, kernel_size=1)
        self.major_sigma_conv = nn.Conv2d(
            self.feat_dim * 2, 11 * 3, kernel_size=1)

    def _forward(
        self,
        feats: Tuple[Tensor],
        mems=None,
    ) -> Tensor:
        devices_cuda = feats.device
        B = feats.shape[0]
        out_feats = self.liftnet(feats)
        if mems is None:
            mems = torch.zeros(B, 2 * self.channel_num, 1, 1).to(devices_cuda)
        feat_mix = torch.cat([out_feats, mems], dim=1)
        mems = self.temporal(feat_mix)
        output = self.last_layer(feat_mix)
        kpt, rot, svd_pt = self.simple_feature_layer(output, feats[:, -1, 0,
                                                                   0])
        return kpt, rot, svd_pt, mems

    def forward(self,
                feats: Tuple[Tensor],
                mems=None,
                seq_len: int = 1) -> Tensor:
        feats = self.liftnet(feats)
        B = int(feats.shape[0] / seq_len)
        if mems is None:
            mems = torch.zeros(B, 2 * self.channel_num, 1, 1).cuda()
        feats = feats.view(B, seq_len, -1)
        outputs = torch.zeros((B, seq_len, self.output_num, 1, 1)).cuda()
        all_sigmas = torch.zeros((B, seq_len, 21 * 3, 1, 1)).cuda()
        major_sigmas = torch.zeros((B, seq_len, 11 * 3, 1, 1)).cuda()
        for i in range(seq_len):
            feat = feats[:, i:i + 1, :].reshape(B, -1, 1, 1)
            feat_mix = torch.cat([feat, mems], dim=1)
            mems = self.temporal(feat_mix)
            output = self.last_layer(feat_mix)
            all_sigma = self.all_sigma_conv(feat_mix)
            major_sigma = self.major_sigma_conv(feat_mix)
            all_sigmas[:, i, ...] = all_sigma
            major_sigmas[:, i, ...] = major_sigma
            outputs[:, i, ...] = output
        outputs = outputs.reshape(B * seq_len, -1, 1, 1)
        all_sigmas = all_sigmas.reshape(B * seq_len, 21, 3)
        major_sigmas = major_sigmas.reshape(B * seq_len, 11, 3)
        return outputs, mems, all_sigmas, major_sigmas

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'predict')

        output, mems, all_sigma, major_sigmas = self.forward(
            data['feats'], mems, 1)

        hand3d_pred = self.postprocess(
            output,
            data['left_hand'],
            data['leftcam_xy'],
            data['left_R'],
            data['nimble_info'],
            data['hand3d_gt'],
            data['baseline_scale'],
            only_pre=True)[0]
        if self.reproj:
            camera_model = batch_data_samples[0].meta['ori_camera']
            leftcam_uv_reproj_distort = camera_model.eye_to_window(
                hand3d_pred.cpu().numpy())
            leftcam_uv_reproj_distort = torch.tensor(
                leftcam_uv_reproj_distort).cuda()
            return hand3d_pred, leftcam_uv_reproj_distort[:, None,
                                                          ...], mems, major_sigmas
        else:
            return hand3d_pred, data[
                'uv_coord_im_pred_global_distort'], mems, major_sigmas

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'loss')

        output, mems, all_sigmas, major_sigmas = self.forward(
            data['feats'], None, self.seq_len)

        B = output.shape[0]
        hand3d_gt = data['hand3d_gt']
        # 3d 损失
        (pred_3d_way1, pred_3d_way2, hand3d_pred, hand3d_part_gt,
         pre_trans_xyz, pre_root_matrix, pre_local_matrix,
         pre_shape) = self.postprocess(output, data['left_hand'],
                                       data['leftcam_xy'], data['left_R'],
                                       data['nimble_info'], hand3d_gt,
                                       data['baseline_scale'], False)

        # 直接监督rot和trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz
        # pre_child_vector = pre_rot_vector[:, 1:, :].reshape(-1, 3)
        pre_child_matrix = pre_local_matrix.reshape(B, -1, 3, 3)
        pre_matrix = torch.cat(
            (pre_root_matrix.unsqueeze(1), pre_child_matrix),
            dim=1).reshape(-1, 3, 3)
        if self.euler_or_quaternion == 'euler':
            pre_euler = matrix_to_euler_angles(pre_matrix, 'XYZ')
        elif self.euler_or_quaternion == 'quaternion':
            pre_nimble_pose = matrix_to_quaternion(pre_matrix).reshape(B, -1)

        if 'nimble_pose' in data['nimble_info'].keys(
        ) and 'nimble_trans' in data['nimble_info'].keys():
            gt_nimble_trans = data['nimble_info']['nimble_trans']
            gt_nimble_pose_roctor = data['nimble_info']['nimble_pose'].reshape(
                -1, 3)
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
        leftcam_cam_matrix = data['leftcam_cam_matrix']
        rightcam_cam_matrix = data['rightcam_cam_matrix']
        left_to_right_rt = data['left_to_right_rt']
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

        re_all_sigmas = torch.cat((hand3d_pred, all_sigmas), dim=-1)
        re_major_sigmas = torch.cat(
            (torch.cat((hand3d_pred[:, :10, :], hand3d_pred[:, 13:14, :]),
                       dim=1), major_sigmas),
            dim=-1)
        pinch_gt = torch.cat((hand3d_gt[:, :10, :], hand3d_gt[:, 13:14, :]),
                             dim=1)

        pred_for_loss = [
            pred_3d_way1, pred_3d_way2, hand3d_pred, leftcam_uv_pre,
            rightcam_uv_pre, dist_pred, proportion_xyz_pre, pre_nimble_pose,
            pre_nimble_trans, hand3d_pred, re_all_sigmas, re_major_sigmas
        ]
        targ_for_loss = [
            hand3d_gt, hand3d_gt, hand3d_gt, leftcam_uv_gt, rightcam_uv_gt,
            dist_gt, proportion_xyz_gt, gt_nimble_pose, gt_nimble_trans,
            hand3d_gt, hand3d_gt, pinch_gt
        ]

        weight_ini = torch.ones((1, 21, 3))
        weight_ini[0, :9, :] = 2
        weight_ini[0, 4, :], weight_ini[0, 8, :] = 4, 4
        weight_ini = weight_ini.repeat(hand3d_gt.shape[0], 1,
                                       1).to(hand3d_gt.device)

        weight_ini_for_pre_nimble = weight_ini.clone().to(hand3d_gt.device)
        weight_ini_for_pre_nimble[:, :9, :] = 4
        weight_ini_for_pre_nimble[:,
                                  4, :], weight_ini_for_pre_nimble[:,
                                                                   8, :] = 8, 8

        weight_for_loss = [
            weight_ini, weight_ini_for_pre_nimble, weight_ini, None, None,
            None, None, None, None, None, None, None
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss, weight_for_loss)
        (loss_pre_root, loss_pre_nimble, loss_pre_all, loss_mse_2d_leftcam,
         loss_mse_2d_rightcam, loss_pinch, loss_scale, loss_nimble_pose,
         loss_nimble_trans, loss_smooth, loss_rle_all, loss_rle_pinch) = losses

        if self.use_shape_smooth:
            pre_shape_reshape = pre_shape.reshape(-1, self.seq_len)
            mean_shape = torch.mean(
                pre_shape_reshape,
                dim=-1).unsqueeze(-1).repeat(1, self.seq_len)
            smooth_shape_loss = self.shape_loss_func(pre_shape_reshape,
                                                     mean_shape)
        else:
            smooth_shape_loss = torch.tensor(0.0, device=loss_pre_root.device)

        # # 子骨骼向量监督
        if self.use_bone_loss:
            bone_loss_weight = 0.15
            bone_3d_pre = (hand3d_pred - hand3d_pred[:, self.joint_parents, :]
                           )[:, self.non_root_indices].reshape(-1, 3)
            bone_3d_gt = (hand3d_gt - hand3d_gt[:, self.joint_parents, :]
                          )[:, self.non_root_indices].reshape(-1, 3)

            bone_3d_pre_vector = self.cal_normalize_vector(bone_3d_pre)
            bone_3d_gt_vector = self.cal_normalize_vector(bone_3d_gt)

            squared_diff = (bone_3d_pre_vector - bone_3d_gt_vector)**2
            bone_loss = torch.mean(torch.sum(squared_diff,
                                             dim=1)) * bone_loss_weight

            # 局部子骨骼监督
            major_bone_loss_weight = 0.5
            local_bone_3d_pre = (
                pred_3d_way2 -
                pred_3d_way2[:, self.joint_parents, :])[:,
                                                        self.non_root_indices]
            local_bone_3d_pre = local_bone_3d_pre[:, :8, :].reshape(-1, 3)
            local_bone_3d_gt = (hand3d_part_gt -
                                hand3d_part_gt[:, self.joint_parents, :]
                                )[:, self.non_root_indices]
            local_bone_3d_gt = local_bone_3d_gt[:, :8, :].reshape(-1, 3)

            local_bone_3d_pre_vector = self.cal_normalize_vector(
                local_bone_3d_pre)
            local_bone_3d_gt_vector = self.cal_normalize_vector(
                local_bone_3d_gt)

            local_squared_diff = (local_bone_3d_pre_vector -
                                  local_bone_3d_gt_vector)**2
            major_bone_loss = torch.mean(torch.sum(
                local_squared_diff, dim=1)) * major_bone_loss_weight

        else:
            bone_loss = torch.tensor(0.0, device=loss_pre_root.device)
            major_bone_loss = torch.tensor(0.0, device=loss_pre_root.device)

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
            bone_loss=bone_loss,
            major_bone_loss=major_bone_loss,
            loss_pinch=loss_pinch,
            loss_proportion=loss_scale,
            loss_nimble_pose=loss_nimble_pose,
            loss_nimble_trans=loss_nimble_trans,
            smooth_shape_loss=smooth_shape_loss,
            loss_smooth=loss_smooth,
            loss_rle_all=loss_rle_all,
            loss_rle_pinch=loss_rle_pinch)

        return losses_dict

    def cal_normalize_vector(self, vector):
        vector_norms = torch.sqrt(
            torch.sum(vector**2, dim=1, keepdim=True) + 1e-8)
        normalized_vector = vector / vector_norms
        return normalized_vector
