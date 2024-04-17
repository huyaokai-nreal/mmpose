# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import torch
from mmengine.logging import MessageHub
from torch import Tensor, nn

from mmpose.models.heads.nimble.nimble_utils import (
    SkeletonEncoder, _gen_rigid_features, adjust_predicted_angles,
    batch_rodrigues, cal_proportion, convert_vector2matrix, decode_svd,
    euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion,
    rot6D_to_matirx, trans_3d_2_2d)
from mmpose.models.heads.nimble.simple_NIMBLELayer import sim_NIMBLELayer
from mmpose.models.heads.regression_heads.lift_head_standard import \
    LiftHeadStandard
from mmpose.models.utils.gmlp import gMLP
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


@MODELS.register_module()
class LiftNimbleHeadStandard(LiftHeadStandard):
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
                 use_6d_pose_reg: bool = False,
                 direct_pose_reg: bool = False,
                 all_use_kp2d_gt: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(
            lift_loss=lift_loss,
            d_ffn=d_ffn,
            reproj=reproj,
            baseline=baseline,
            reproj_thre=reproj_thre,
            iou_thre=iou_thre,
            pad_2d=pad_2d,
            lambda_t=lambda_t,
            all_use_kp2d_gt=all_use_kp2d_gt,
            init_cfg=init_cfg,
        )

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
        self.channel_num = 43
        self.lambda_t = lambda_t
        self.kpt2d_with_depth = kpt2d_with_depth
        feat_dim = 2 * self.channel_num
        if self.kpt2d_with_depth:
            feat_dim = feat_dim + 21
        if reg_shape_type > 1:
            feat_dim = feat_dim + skeleton_feature_dim
        self.liftnet = gMLP(d_model=feat_dim, d_ffn=d_ffn, num_layers=3)
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
        if use_6d_pose_reg:
            self.pose_ncomp = 19 * 6
        if direct_pose_reg:
            self.pose_ncomp = 21
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
        self.use_bone_loss = use_bone_loss
        self.feat_dim = feat_dim
        self.reproj = reproj
        self.baseline = baseline
        self.reproj_thre = reproj_thre
        self.iou_thre = iou_thre
        self.pad_2d = pad_2d
        self.center_rot = None
        self.use_svd = use_svd
        self.scale_parameter = 1000
        self.corruption_cam = corruption_cam
        self.undistort = undistort
        self.all_use_kp2d_gt = all_use_kp2d_gt
        self.use_nimble_part_para = use_nimble_part_para
        self.euler_or_quaternion = euler_or_quaternion
        self.use_6d_pose_reg = use_6d_pose_reg
        self.direct_pose_reg = direct_pose_reg
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

        self.direct_pose_reg_index = [
            3, 4, 6, 7, 9, 15, 16, 18, 21, 27, 28, 30, 33, 39, 40, 42, 45, 51,
            52, 54, 57
        ]

        self.joint_parents = [
            0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18,
            19
        ]
        self.non_root_indices = []
        for i in range(len(self.joint_parents)):
            if i != self.joint_parents[i]:
                self.non_root_indices.append(i)

    def get_full_pose_with_part_pars(self, pose_reg):
        used_nimble_para = torch.tensor(self.used_nimble_para)
        pose_out = torch.zeros((pose_reg.shape[0], 57),
                               device=pose_reg.device,
                               dtype=torch.float32)
        pose_out[:, used_nimble_para] = pose_reg.to(torch.float32)
        return pose_out

    def simple_feature_layer(self, output, left_hand):
        B = output.shape[0]
        pose_len = self.pose_ncomp
        rot_vector_t = output[:, :pose_len, 0, 0]
        svd_begin = self.pose_ncomp + self.shape_ncomp
        shape_v = output[:, pose_len:svd_begin, 0, 0]
        pre_pt_features = output[:, svd_begin:, 0, 0]
        if self.use_6d_pose_reg:
            pre_local_matrix = rot6D_to_matirx(rot_vector_t.reshape(
                -1, 6)).reshape(B, 19, -1)
        else:
            pre_local_matrix = self.nimble_layer.generate_pose_matrix(
                rot_vector_t, normalized=True, with_root=False)

        _, bone_joints = self.nimble_layer.forward_simple(
            pre_local_matrix, shape_v)
        rebuild_joints = bone_joints[:, self.kp_index, :]
        root_rebuild_joints = rebuild_joints[:, 0:1, :]
        rebuild_joints_temp = rebuild_joints - root_rebuild_joints

        rebuild_joints_temp = rebuild_joints_temp / self.scale_parameter
        return rebuild_joints_temp, pre_local_matrix, pre_pt_features

    def _forward(self, feats: Tuple[Tensor]) -> Tensor:
        output = self.liftnet(feats)
        output = self.last_layer(output).view((feats.shape[0], -1, 1, 1))
        kpt, rot, svd_pt = self.simple_feature_layer(output, feats[:, -1, 0,
                                                                   0])
        return kpt, rot, svd_pt

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        output = self.liftnet(feats)
        output = self.last_layer(output).view((feats.shape[0], -1, 1, 1))
        return output

    def postprocess(self,
                    output,
                    left_hand,
                    leftcam_xy,
                    left_R,
                    nimble_info,
                    hand3d_gt,
                    baseline_scale,
                    only_pre=False):

        B = output.shape[0]
        cuda_device = output.device
        baseline_scale = baseline_scale.view(B, 1, 1)

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
        if self.use_6d_pose_reg:
            pre_local_matrix = rot6D_to_matirx(rot_vector_t.reshape(
                -1, 6)).reshape(B, 19, -1)
        elif self.direct_pose_reg:
            rot_vector_t = torch.mul(rot_vector_t, torch.pi)
            pre_euler_value = torch.zeros((B, 60), device=cuda_device)
            pre_euler_value[:, self.direct_pose_reg_index] = rot_vector_t.to(
                torch.float32)
            pre_euler_value = pre_euler_value.reshape(-1, 3)
            pre_local_matrix = euler_angles_to_matrix(pre_euler_value).reshape(
                B, 20, -1)[:, 1:, :]
        else:
            pre_local_matrix = self.nimble_layer.generate_pose_matrix(
                rot_vector_t, normalized=True, with_root=False)
        pre_shape_vector = shape_v

        if not only_pre:
            with torch.no_grad():
                gt_root_xyz = torch.bmm(
                    left_R, nimble_info['nimble_trans'].unsqueeze(
                        -1))[:, :, 0] / baseline_scale[0]
                gt_root_matrix = batch_rodrigues(
                    nimble_info['nimble_pose'][:, 0, :]).reshape(-1, 3, 3)
                gt_root_matrix = torch.matmul(left_R, gt_root_matrix)

                init_root_rot = torch.zeros((B, 1, 3),
                                            requires_grad=True,
                                            device=cuda_device)
                gt_rot_vector = torch.cat(
                    (init_root_rot, nimble_info['nimble_pose'][:, 1:, :]),
                    dim=1)
                gt_local_matrix = convert_vector2matrix(
                    gt_rot_vector.view(B, -1))
                # gt_shape_vector = nimble_info['nimble_shape']

        def get_nimble_3d(root_xyz, root_matrix, local_matrix, shape_vector,
                          baseline_scale, left_R):

            _, bone_joints = self.nimble_layer.forward_simple(
                local_matrix, shape_vector)
            rebuild_joints = bone_joints[:, self.kp_index, :]
            root_rebuild_joints = rebuild_joints[:, 0:1, :]
            rebuild_joints_temp = rebuild_joints - root_rebuild_joints

            mask = left_hand == 1
            add_matrix = torch.eye(3).unsqueeze(0).expand(B, -1,
                                                          -1).to(cuda_device)
            add_matrix[mask, 0, 0] = -add_matrix[mask, 0, 0]
            root_matrix = torch.matmul(torch.inverse(left_R), root_matrix)
            root_matrix = torch.matmul(root_matrix, add_matrix)
            rebuild_joints_temp = torch.matmul(rebuild_joints_temp,
                                               root_matrix.transpose(1, 2))
            rebuild_joints_with_scale = \
                rebuild_joints_temp / self.scale_parameter

            new_root_xyz = torch.bmm(
                root_xyz.unsqueeze(1),
                torch.inverse(left_R).permute(0, 2, 1))
            xyz_point = rebuild_joints_with_scale + new_root_xyz
            xyz_point *= baseline_scale
            return xyz_point

        if only_pre:
            pre_nimble_pre_root_pre_shape__xyz = get_nimble_3d(
                pre_root_xyz, pre_root_matrix, pre_local_matrix,
                pre_shape_vector, baseline_scale, left_R)

            return pre_nimble_pre_root_pre_shape__xyz, \
                pre_root_xyz, pre_root_matrix, pre_local_matrix
        else:
            pre_root__xyz = get_nimble_3d(pre_root_xyz, pre_root_matrix,
                                          gt_local_matrix, pre_shape_vector,
                                          baseline_scale, left_R)
            pre_nimble__xyz = get_nimble_3d(gt_root_xyz, gt_root_matrix,
                                            pre_local_matrix, pre_shape_vector,
                                            baseline_scale, left_R)
            pre_all__xyz = get_nimble_3d(pre_root_xyz, pre_root_matrix,
                                         pre_local_matrix, pre_shape_vector,
                                         baseline_scale, left_R)
            gt_all__xyz = get_nimble_3d(gt_root_xyz, gt_root_matrix,
                                        gt_local_matrix, pre_shape_vector,
                                        baseline_scale, left_R)

            pre_root_xyz = torch.bmm(
                pre_root_xyz.unsqueeze(1),
                torch.inverse(left_R).permute(0, 2, 1)) * baseline_scale
            pre_root_matrix = torch.matmul(
                torch.inverse(left_R), pre_root_matrix)

            return (pre_root__xyz, pre_nimble__xyz, pre_all__xyz, gt_all__xyz,
                    pre_root_xyz[:, 0, :], pre_root_matrix, pre_local_matrix,
                    pre_shape_vector)

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'predict')
        if self.reg_shape_type > 1:
            batch_num = data['feats'].shape[0]
            skeleton_joints_info = self.skeleton_joints_info.repeat(
                batch_num, 1).to(data['feats'].device)
            skeleton_feature = self.skeleton_encoder(
                skeleton_joints_info).view(batch_num,
                                           self.skeleton_feature_dim, 1, -1)
            data['feats'] = torch.cat((data['feats'], skeleton_feature), dim=1)
        output = self.forward(data['feats'])

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
            return hand3d_pred, leftcam_uv_reproj_distort[:, None, ...]
        else:
            return hand3d_pred, data['uv_coord_im_pred_global_distort']

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'loss')

        if self.reg_shape_type > 1:
            batch_num = data['feats'].shape[0]
            skeleton_joints_info = self.skeleton_joints_info.repeat(
                batch_num, 1).to(data['feats'].device)
            skeleton_feature = self.skeleton_encoder(
                skeleton_joints_info).view(batch_num,
                                           self.skeleton_feature_dim, 1, -1)
            data['feats'] = torch.cat((data['feats'], skeleton_feature), dim=1)

        output = self.forward(data['feats'])

        B = output.shape[0]
        # 3d 损失
        (pred_3d_way1, pred_3d_way2, hand3d_pred, hand3d_part_gt,
         pre_trans_xyz, pre_root_matrix, pre_local_matrix,
         pre_shape) = self.postprocess(output, data['left_hand'],
                                       data['leftcam_xy'], data['left_R'],
                                       data['nimble_info'], data['hand3d_gt'],
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
        leftcam_uv_pre, rightcam_uv_pre = trans_3d_2_2d(
            hand3d_pred, data['leftcam_cam_matrix'],
            data['rightcam_cam_matrix'], data['left_to_right_rt'])
        leftcam_uv_gt, rightcam_uv_gt = trans_3d_2_2d(
            data['hand3d_gt'], data['leftcam_cam_matrix'],
            data['rightcam_cam_matrix'], data['left_to_right_rt'])

        # xyz比例约束
        proportion_xyz_pre = cal_proportion(leftcam_uv_pre,
                                            data['leftcam_cam_matrix'])
        proportion_xyz_gt = cal_proportion(leftcam_uv_gt,
                                           data['leftcam_cam_matrix'])

        # 数据归一化
        leftcam_uv_pre = leftcam_uv_pre / 500
        rightcam_uv_pre = rightcam_uv_pre / 500
        leftcam_uv_gt = leftcam_uv_gt / 500
        rightcam_uv_gt = rightcam_uv_gt / 500

        # pinch 损失
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(
            data['hand3d_gt'][:, 4, :] - data['hand3d_gt'][:, 8, :], dim=-1)

        pred_for_loss = [
            pred_3d_way1, pred_3d_way2, hand3d_pred, leftcam_uv_pre,
            rightcam_uv_pre, dist_pred, proportion_xyz_pre, pre_nimble_pose,
            pre_nimble_trans
        ]
        targ_for_loss = [
            data['hand3d_gt'], data['hand3d_gt'], data['hand3d_gt'],
            leftcam_uv_gt, rightcam_uv_gt, dist_gt, proportion_xyz_gt,
            gt_nimble_pose, gt_nimble_trans
        ]

        weight_ini = torch.ones((1, 21, 3))
        weight_ini[0, :9, :] = 2
        weight_ini[0, 4, :], weight_ini[0, 8, :] = 4, 4
        weight_ini = weight_ini.repeat(data['hand3d_gt'].shape[0], 1,
                                       1).to(data['hand3d_gt'].device)
        weight_for_loss = [
            weight_ini,
            weight_ini,
            weight_ini,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss, weight_for_loss)
        (loss_pre_root, loss_pre_nimble, loss_pre_all, loss_mse_2d_leftcam,
         loss_mse_2d_rightcam, loss_pinch, loss_scale, loss_nimble_pose,
         loss_nimble_trans) = losses

        # # 子骨骼向量监督
        if self.use_bone_loss:
            bone_loss_weight = 0.1
            bone_3d_pre = (hand3d_pred - hand3d_pred[:, self.joint_parents, :]
                           )[:, self.non_root_indices].reshape(-1, 3)
            bone_3d_gt = (data['hand3d_gt'] -
                          data['hand3d_gt'][:, self.joint_parents, :]
                          )[:, self.non_root_indices].reshape(-1, 3)

            bone_3d_pre_vector = self.cal_normalize_vector(bone_3d_pre)
            bone_3d_gt_vector = self.cal_normalize_vector(bone_3d_gt)

            squared_diff = (bone_3d_pre_vector - bone_3d_gt_vector)**2
            bone_loss = torch.mean(torch.sum(squared_diff,
                                             dim=1)) * bone_loss_weight

            # 局部子骨骼监督
            major_bone_loss_weight = 0.3
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
            loss_nimble_trans=loss_nimble_trans)

        return losses_dict

    def cal_normalize_vector(self, vector):
        vector_norms = torch.sqrt(
            torch.sum(vector**2, dim=1, keepdim=True) + 1e-8)
        normalized_vector = vector / vector_norms
        return normalized_vector
