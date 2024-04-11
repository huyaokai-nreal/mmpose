# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import torch
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
class TemporalLiftNimbleHeadStandard(LiftNimbleHeadStandard):
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
                 use_plane_coord=True,
                 baseline=0.13,
                 disparity_input=False,
                 reproj_thre=0,
                 iou_thre=0,
                 pad_2d=False,
                 edge_to_center=False,
                 lambda_t: int = -1,
                 corruption_cam: float = 0.5,
                 use_bone_loss: bool = True,
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
            use_plane_coord=use_plane_coord,
            baseline=baseline,
            use_svd=use_svd,
            disparity_input=disparity_input,
            reproj_thre=reproj_thre,
            iou_thre=iou_thre,
            pad_2d=pad_2d,
            edge_to_center=edge_to_center,
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

    def _forward(
        self,
        feats: Tuple[Tensor],
        mems=None,
    ) -> Tensor:
        devices_cuda = feats.device
        feats = self.liftnet(feats)
        B = feats.shape[0]
        if mems is None:
            mems = torch.zeros(B, 2 * self.channel_num, 1, 1).to(devices_cuda)
        feat_mix = torch.cat([feats, mems], dim=1)
        mems = self.temporal(feat_mix)
        output = self.last_layer(feat_mix)
        kpt, rot, svd_pt = self.simple_feature_layer(output)
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
        for i in range(seq_len):
            feat = feats[:, i:i + 1, :].reshape(B, -1, 1, 1)
            feat_mix = torch.cat([feat, mems], dim=1)
            mems = self.temporal(feat_mix)
            output = self.last_layer(feat_mix)
            outputs[:, i, ...] = output
        outputs = outputs.reshape(B * seq_len, -1, 1, 1)
        return outputs, mems

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            (feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p,
             left_to_right_rt, leftcam_cam_matrix, rightcam_cam_matrix,
             uv_coord_im_pred_global, uv_coord_im_gt_global,
             uv_coord_im_pred_global_distort,
             uv_coord_im_pred_global_distort_noflip, hand3d_gt, left_hand,
             nimble_info, left_R, right_R,
             baseline_scale) = self.preprocess(feats, batch_data_samples,
                                               'predict')

        output, mems = self.forward(feats, mems, 1)

        hand3d_pred = self.postprocess(
            output,
            left_hand,
            leftcam_xy,
            left_R,
            nimble_info,
            hand3d_gt,
            baseline_scale,
            only_pre=True)[0]
        if self.reproj:
            camera_model = batch_data_samples[0].meta['ori_camera']
            leftcam_uv_reproj_distort = camera_model.eye_to_window(
                hand3d_pred.cpu().numpy())
            leftcam_uv_reproj_distort = torch.tensor(
                leftcam_uv_reproj_distort).cuda()
            return hand3d_pred, leftcam_uv_reproj_distort[:, None, ...], mems
        else:
            return hand3d_pred, uv_coord_im_pred_global_distort, mems

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        with torch.no_grad():
            (feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p,
             left_to_right_rt, leftcam_cam_matrix, rightcam_cam_matrix,
             uv_coord_im_pred_global, uv_coord_im_gt_global,
             uv_coord_im_pred_global_distort,
             uv_coord_im_pred_global_distort_noflip, hand3d_gt, left_hand,
             nimble_info, left_R, right_R,
             baseline_scale) = self.preprocess(feats, batch_data_samples,
                                               'loss')

        output, _ = self.forward(feats, None, self.seq_len)

        B = output.shape[0]
        # 3d 损失
        (pred_3d_way1, pred_3d_way2, hand3d_pred, pre_trans_xyz,
         pre_root_matrix,
         pre_rot_vector) = self.postprocess(output, left_hand, leftcam_xy,
                                            left_R, nimble_info, hand3d_gt,
                                            baseline_scale, False)

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
            pre_nimble_trans, hand3d_pred
        ]
        targ_for_loss = [
            hand3d_gt, hand3d_gt, hand3d_gt, leftcam_uv_gt, rightcam_uv_gt,
            dist_gt, proportion_xyz_gt, gt_nimble_pose, gt_nimble_trans,
            hand3d_gt
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss)
        (loss_pre_root, loss_pre_nimble, loss_pre_all, loss_mse_2d_leftcam,
         loss_mse_2d_rightcam, loss_pinch, loss_scale, loss_nimble_pose,
         loss_nimble_trans, loss_smooth) = losses

        # # 子骨骼向量监督
        if self.use_bone_loss:
            bone_loss_weight = 0.1
            bone_3d_pre = (hand3d_pred - hand3d_pred[:, self.joint_parents, :]
                           )[:, self.non_root_indices].reshape(-1, 3)
            bone_3d_gt = (hand3d_gt - hand3d_gt[:, self.joint_parents, :]
                          )[:, self.non_root_indices].reshape(-1, 3)

            bone_3d_pre_vector = bone_3d_pre / torch.norm(
                bone_3d_pre, dim=1, keepdim=True)
            bone_3d_gt_vector = bone_3d_gt / torch.norm(
                bone_3d_gt, dim=1, keepdim=True)

            squared_diff = (bone_3d_pre_vector - bone_3d_gt_vector)**2
            bone_loss = torch.mean(torch.sum(squared_diff,
                                             dim=1)) * bone_loss_weight
        else:
            bone_loss = torch.tensor(0.0, device=loss_pre_root.device)

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
            loss_pinch=loss_pinch,
            loss_proportion=loss_scale,
            loss_nimble_pose=loss_nimble_pose,
            loss_nimble_trans=loss_nimble_trans,
            loss_smooth=loss_smooth)

        return losses_dict
