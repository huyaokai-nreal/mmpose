# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import numpy as np
import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmpose.models.heads.nimble.nimble_utils import (_gen_rigid_features,
                                                     batch_rodrigues,
                                                     convert_vector2matrix,
                                                     decode_svd,
                                                     euler_angles_to_matrix,
                                                     rot6D_to_matirx,
                                                     rot9D_to_matirx)
from mmpose.models.heads.nimble.simple_NIMBLELayer import sim_NIMBLELayer
from mmpose.models.utils.gmlp import gMLP
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


def create_multi_view_fusion_layers(
    nc_in: int,
    nc_out: int,
    n_blocks: int,
) -> nn.Module:
    """Linearly increase/reduce channels per block."""
    n_channels_list = np.linspace(nc_in, nc_out, n_blocks + 1)
    fusion_layers = nn.ModuleList()

    for i in range(n_blocks):
        nc_in_cur = int(n_channels_list[i])
        nc_out_cur = int(n_channels_list[i + 1])

        fusion_layers.append(
            nn.Conv2d(nc_in_cur, nc_out_cur, kernel_size=1, padding=0))
        fusion_layers.append(nn.BatchNorm2d(nc_out_cur, momentum=0.1))
        fusion_layers.append(nn.ReLU())

    # Add an extra convolution to avoid all positive features from ReLU
    fusion_layers.append(nn.Conv2d(nc_out, nc_out, kernel_size=1, padding=0))

    return nn.Sequential(*fusion_layers)


def apply_ftl_to_feature_maps(
    xfs: torch.Tensor,
    feature_maps: torch.Tensor,
    ftl_ratio: float = 1,
) -> torch.Tensor:
    assert ftl_ratio >= 0 and ftl_ratio <= 1

    if ftl_ratio == 0:
        return feature_maps

    n_images = feature_maps.shape[0]
    n_channels = int(feature_maps.shape[1])

    # number of ftl channels
    nc_ftl = int(round(n_channels * ftl_ratio))
    assert nc_ftl % 3 == 0
    ftl_feature_maps = feature_maps[:, 0:nc_ftl].clone()

    # Apply feature transformations to the point features
    point_features_xfed = ftl_feature_maps.reshape(n_images, 3, -1)

    r_in = xfs[:, 0:3, 0:3].clone()
    t_in = xfs[:, 0:3, 3].clone()
    point_features_xfed = torch.matmul(
        r_in, point_features_xfed) + t_in.unsqueeze(-1)

    # Reshape back to feature maps
    ftl_feature_maps_xfed = point_features_xfed.reshape(ftl_feature_maps.shape)
    if nc_ftl != n_channels:
        cat_maps = torch.cat((ftl_feature_maps_xfed, feature_maps[:, nc_ftl:]),
                             dim=1)
        return cat_maps
    else:
        return ftl_feature_maps_xfed


@MODELS.register_module()
class UmeHead(BaseModule):

    def __init__(self,
                 ume_loss: ConfigType,
                 feat_channel: int = 6,
                 d_ffn: int = 220,
                 feature_map_shape: tuple = (8, 8),
                 shape_ncomp: int = 20,
                 pose_ncomp: int = 60,
                 use_pose_pca: bool = True,
                 reg_shape_type: int = 1,
                 use_nimble_part_para: bool = False,
                 use_6d_pose_reg: bool = True,
                 use_9d_pose_reg: bool = False,
                 direct_pose_reg: bool = False,
                 use_bone_loss: bool = True,
                 enhance_lefthand=True,
                 enhance_static=True,
                 use_svd: bool = True,
                 use_gmlp: bool = True,
                 baseline: float = 0.135,
                 use_scaled_as_canonical: bool = True,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)

        # model_opts = ModelOpts()
        self.ume_loss = MODELS.build(ume_loss)
        self.feat_channel = feat_channel
        self.use_nimble_part_para = use_nimble_part_para
        self.use_6d_pose_reg = use_6d_pose_reg
        self.use_9d_pose_reg = use_9d_pose_reg
        self.direct_pose_reg = direct_pose_reg
        self.use_bone_loss = use_bone_loss
        self.enhance_lefthand = enhance_lefthand
        self.enhance_static = enhance_static
        self.shape_ncomp = shape_ncomp
        self.pose_ncomp = pose_ncomp
        self.use_scaled_as_canonical = use_scaled_as_canonical
        self.use_gmlp = use_gmlp
        self.scale_parameter = 1000
        self.rigid_samples = _gen_rigid_features()
        self.baseline = baseline
        self.proj_layer = nn.Conv2d(
            256, feat_channel, kernel_size=1, padding=0)
        self._multi_view_fusion = create_multi_view_fusion_layers(
            feat_channel * 2, feat_channel, 2)
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
        self.static_data_date_list = ['20240516', '20240517', '20240522']
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
        if use_9d_pose_reg:
            self.pose_ncomp = 19 * 9
        if direct_pose_reg:
            self.pose_ncomp = 21
        output_num = self.shape_ncomp + self.pose_ncomp + 3
        if use_svd:
            self.output_num = output_num + 18  # 21 - 3
        else:
            self.output_num = output_num
        feat_dim = feat_channel * feature_map_shape[0] * feature_map_shape[1]
        if self.use_gmlp:
            self.gmlp = gMLP(d_model=feat_dim, d_ffn=d_ffn, num_layers=3)
        self.last_layer = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1), nn.ReLU(),
            nn.Conv2d(feat_dim, self.output_num, kernel_size=1))
        self.sigma_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.Conv2d(feat_dim, 21 * 3, kernel_size=1))
        # define nimble layer
        self.nimble_layer = sim_NIMBLELayer(
            device='cuda',
            shape_ncomp=self.shape_ncomp,
            pose_ncomp=self.pose_ncomp,
            use_pose_pca=use_pose_pca,
            reg_shape_type=reg_shape_type)
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

    def preprocess(self, batch_data_samples):
        left_cam_f = []
        right_cam_f = []
        left_cam_xf = []
        right_cam_xf = []
        hand3d_gt = []
        is_left_hands = []
        nimble_pose = []
        nimble_trans = []
        nimble_shape = []
        nimble_info = dict()

        for i, data_sample in enumerate(batch_data_samples):
            if i % 2 == 0:
                left_vir_camera = data_sample.meta['virtual_camera']
                left_cam_f.append(left_vir_camera.f[0])
                if data_sample.meta['flipped']:
                    mirror_x_matrix = np.eye(4)
                    mirror_x_matrix[0][0] = -1
                    left_cam_xf.append(
                        data_sample.meta['ori_xf'] @ mirror_x_matrix
                        @ left_vir_camera.camera_to_world_xf)
                    data_sample.gt_instances.keypoints3d[..., 0] *= -1
                else:
                    left_cam_xf.append(data_sample.meta['ori_xf']
                                       @ left_vir_camera.camera_to_world_xf)
                hand3d_gt.append(data_sample.gt_instances.keypoints3d[0])
                if 'nimble_pose' in data_sample.meta.keys() and not np.equal(
                        data_sample.meta['nimble_pose'].any(), None):
                    nimble_pose.append(data_sample.meta['nimble_pose'])
                    nimble_trans.append(data_sample.meta['nimble_translation'])
                    nimble_shape.append(data_sample.meta['nimble_shape'])
                if data_sample.meta['category_id'] == 1:
                    is_left_hands.append(1)
                else:
                    is_left_hands.append(0)
            else:
                right_vir_camera = data_sample.meta['virtual_camera']
                right_cam_f.append(right_vir_camera.f[0])

                if data_sample.meta['flipped']:
                    mirror_x_matrix = np.eye(4)
                    mirror_x_matrix[0][0] = -1
                    right_cam_xf.append(
                        data_sample.meta['ori_xf'] @ mirror_x_matrix
                        @ right_vir_camera.camera_to_world_xf)
                else:
                    right_cam_xf.append(data_sample.meta['ori_xf']
                                        @ right_vir_camera.camera_to_world_xf)
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
        left_cam_f = torch.tensor(np.array(left_cam_f)).cuda().float()
        right_cam_f = torch.tensor(np.array(right_cam_f)).cuda().float()
        left_cam_xf = torch.tensor(np.array(left_cam_xf)).cuda().float()
        right_cam_xf = torch.tensor(np.array(right_cam_xf)).cuda().float()
        cam_f = torch.stack((left_cam_f, right_cam_f), axis=1)
        cam_xf = torch.stack((left_cam_xf, right_cam_xf), axis=1)
        # 求虚拟双目的baseline
        baseline_vector = torch.norm(
            torch.bmm(torch.inverse(left_cam_xf), right_cam_xf)[:, :3, 3],
            dim=1)
        baseline_scale = baseline_vector / self.baseline
        if len(nimble_pose) > 0:
            nimble_info = {
                'nimble_pose':
                torch.tensor(np.array(nimble_pose)).cuda().float(),
                'nimble_trans':
                torch.tensor(np.array(nimble_trans)).cuda().float(),
                'nimble_shape':
                torch.tensor(np.array(nimble_shape)).cuda().float()
            }
        left_hand = torch.tensor(np.array(is_left_hands)).cuda().float()
        return {
            'cam_f': cam_f,
            'cam_xf': cam_xf,
            'hand3d_gt': hand3d_gt,
            'left_hand': left_hand,
            'nimble_info': nimble_info,
            'baseline_scale': baseline_scale
        }

    def postprocess(
        self,
        output,
        nimble_info,
        left_hand,
        cam_xf,
        baseline_scale,
        only_pre=False,
    ):
        B = output.shape[0]
        cuda_device = output.device
        baseline_scale = baseline_scale.view(B, 1, 1)

        pose_len = self.pose_ncomp
        rot_vector_t = output[:, :pose_len, 0, 0].float()
        if self.use_nimble_part_para:
            rot_vector_t = self.get_full_pose_with_part_pars(rot_vector_t)
        svd_begin = self.pose_ncomp + self.shape_ncomp
        shape_v = output[:, pose_len:svd_begin, 0, 0]
        pre_pt_features = output[:, svd_begin:, 0, 0]
        matrix_svd = decode_svd(
            pre_pt_features,
            self.rigid_samples,
        )
        pre_root = torch.bmm(cam_xf[:, 0], matrix_svd)
        pre_root_xyz = pre_root[:, :3, 3:].squeeze(-1)
        pre_root_matrix = pre_root[:, :3, :3]
        if self.use_6d_pose_reg:
            pre_local_matrix = rot6D_to_matirx(rot_vector_t.reshape(
                -1, 6)).reshape(B, 19, -1)
        elif self.use_9d_pose_reg:
            pre_local_matrix = rot9D_to_matirx(rot_vector_t.reshape(
                -1, 9)).reshape(B, 19, -1)
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

        def get_nimble_3d(root_xyz, root_matrix, local_matrix, shape_vector):
            _, bone_joints = self.nimble_layer.forward_simple(
                local_matrix, shape_vector)
            rebuild_joints = bone_joints[:, self.kp_index, :]
            root_rebuild_joints = rebuild_joints[:, 0:1, :]
            rebuild_joints_temp = rebuild_joints - root_rebuild_joints

            mask = left_hand == 1
            add_matrix = torch.eye(3).unsqueeze(0).expand(B, -1,
                                                          -1).to(cuda_device)
            add_matrix[mask, 0, 0] = -add_matrix[mask, 0, 0]  # nimble只有右手手膜
            root_matrix = torch.matmul(root_matrix, add_matrix)
            rebuild_joints_temp = torch.matmul(rebuild_joints_temp,
                                               root_matrix.transpose(1, 2))
            rebuild_joints_with_scale = \
                rebuild_joints_temp / self.scale_parameter
            xyz_point = rebuild_joints_with_scale + root_xyz.unsqueeze(1)
            # xyz_point *= baseline_scale
            return xyz_point

        if not only_pre:
            with torch.no_grad():
                gt_root_xyz = nimble_info['nimble_trans'].unsqueeze(-1)[:, :,
                                                                        0]
                gt_root_matrix = batch_rodrigues(
                    nimble_info['nimble_pose'][:, 0, :]).reshape(-1, 3, 3)

                init_root_rot = torch.zeros((B, 1, 3),
                                            requires_grad=True,
                                            device=cuda_device)
                gt_rot_vector = torch.cat(
                    (init_root_rot, nimble_info['nimble_pose'][:, 1:, :]),
                    dim=1)
                gt_local_matrix = convert_vector2matrix(
                    gt_rot_vector.view(B, -1)).reshape(B, -1, 9)

        if only_pre:
            pre_nimble_pre_root_pre_shape__xyz = get_nimble_3d(
                pre_root_xyz, pre_root_matrix, pre_local_matrix,
                pre_shape_vector)

            return pre_nimble_pre_root_pre_shape__xyz, \
                pre_root_xyz, pre_root_matrix, pre_local_matrix
        else:
            pre_root__xyz = get_nimble_3d(pre_root_xyz, pre_root_matrix,
                                          gt_local_matrix, pre_shape_vector)
            pre_nimble__xyz = get_nimble_3d(gt_root_xyz, gt_root_matrix,
                                            pre_local_matrix, pre_shape_vector)
            pre_all__xyz = get_nimble_3d(pre_root_xyz, pre_root_matrix,
                                         pre_local_matrix, pre_shape_vector)
            gt_all__xyz = get_nimble_3d(gt_root_xyz, gt_root_matrix,
                                        gt_local_matrix, pre_shape_vector)

            gt_matrix = torch.cat(
                (gt_root_matrix.reshape(B, 1, 9), gt_local_matrix), dim=1)
            pre_matrix = torch.cat(
                (pre_root_matrix.reshape(B, 1, 9), pre_local_matrix), dim=1)
            return (pre_root__xyz, pre_nimble__xyz, pre_all__xyz, gt_all__xyz,
                    pre_root_xyz, pre_shape_vector, gt_matrix, pre_matrix)

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(batch_data_samples)

        # forward
        output, sigma = self.forward(feats, data['cam_f'], data['cam_xf'])
        hand3d_pred = self.postprocess(
            output,
            data['nimble_info'],
            data['left_hand'],
            data['cam_xf'],
            data['baseline_scale'],
            only_pre=True)[0]
        hand3d_pred = hand3d_pred.cpu().numpy()
        leftcam_uv_distort = []
        for i, left_sample in enumerate(batch_data_samples[::2]):
            ori_cam = left_sample.meta['ori_camera']
            ori_cam.camera_to_world_xf = left_sample.meta['ori_xf']
            _leftcam_uv_distort = ori_cam.eye_to_window(
                ori_cam.world_to_eye(hand3d_pred[i]))
            leftcam_uv_distort.append(_leftcam_uv_distort)
        leftcam_uv_distort = np.stack(leftcam_uv_distort, axis=0)
        return hand3d_pred, leftcam_uv_distort, sigma

    def loss(self, feats, batch_data_samples) -> dict:
        with torch.no_grad():
            data = self.preprocess(batch_data_samples)
        hand3d_gt = data['hand3d_gt']

        output, sigma = self.forward(feats, data['cam_f'], data['cam_xf'])

        (pred_3d_way1, pred_3d_way2, hand3d_pred, hand3d_part_gt,
         pre_trans_xyz, pre_shape, gt_all_matrix,
         pre_all_matrix) = self.postprocess(output, data['nimble_info'],
                                            data['left_hand'], data['cam_xf'],
                                            data['baseline_scale'])

        # 直接监督rot和trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz
        gt_nimble_trans = data['nimble_info']['nimble_trans']

        # pinch 损失
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(hand3d_gt[:, 4, :] - hand3d_gt[:, 8, :], dim=-1)

        if self.enhance_lefthand:
            mask = data['left_hand'] == 1
            left_weight = 1.2
            enhanced_left_hand3d_gt = self.enhanced_fun(
                hand3d_gt, mask, left_weight)
            enhanced_left_pred_3d_way1 = self.enhanced_fun(
                pred_3d_way1, mask, left_weight)
            enhanced_left_pred_3d_way2 = self.enhanced_fun(
                pred_3d_way2, mask, left_weight)
            enhanced_left_hand3d_pred = self.enhanced_fun(
                hand3d_pred, mask, left_weight)
        else:
            enhanced_left_hand3d_gt = hand3d_gt
            enhanced_left_pred_3d_way1 = pred_3d_way1
            enhanced_left_pred_3d_way2 = pred_3d_way2
            enhanced_left_hand3d_pred = hand3d_pred

        # sigma for RLELoss
        re_all_sigmas = torch.cat((enhanced_left_hand3d_pred, sigma), dim=-1)

        # 归一化平面2d 重投影
        norm_2d_pred = enhanced_left_hand3d_pred / enhanced_left_hand3d_pred[
            ..., 2:]
        norm_2d_gt = enhanced_left_hand3d_gt / enhanced_left_hand3d_gt[..., 2:]

        pred_for_loss = [
            enhanced_left_pred_3d_way1, enhanced_left_pred_3d_way2,
            enhanced_left_hand3d_pred, dist_pred, pre_nimble_trans,
            re_all_sigmas, norm_2d_pred
        ]
        targ_for_loss = [
            enhanced_left_hand3d_gt, enhanced_left_hand3d_gt,
            enhanced_left_hand3d_gt, dist_gt, gt_nimble_trans, hand3d_gt,
            norm_2d_gt
        ]

        weight_ini = torch.ones((hand3d_gt.shape[0], 21, 3),
                                device=hand3d_gt.device)
        weight_ini[:, :9, :] = 2
        weight_ini[:, 4, :], weight_ini[:, 8, :] = 4, 4
        weight_for_loss = [
            weight_ini, weight_ini, weight_ini, None, None, None, weight_ini
        ]
        losses = self.ume_loss(pred_for_loss, targ_for_loss, weight_for_loss)
        (loss_pre_root, loss_pre_nimble, loss_pre_all, loss_pinch,
         loss_nimble_trans, loss_rle_all, loss_2d) = losses
        # # 子骨骼向量监督
        if self.use_bone_loss:
            loss_bone_weight = 0.1
            bone_3d_pre = (hand3d_pred - hand3d_pred[:, self.joint_parents, :]
                           )[:, self.non_root_indices].reshape(-1, 3)
            bone_3d_gt = (hand3d_gt - hand3d_gt[:, self.joint_parents, :]
                          )[:, self.non_root_indices].reshape(-1, 3)

            bone_3d_pre_vector = self.cal_normalize_vector(bone_3d_pre)
            bone_3d_gt_vector = self.cal_normalize_vector(bone_3d_gt)

            squared_diff = (bone_3d_pre_vector - bone_3d_gt_vector)**2
            loss_bone = torch.mean(torch.sum(squared_diff,
                                             dim=1)) * loss_bone_weight

            # 局部子骨骼监督
            loss_major_bone_weight = 0.3
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
            loss_major_bone = torch.mean(torch.sum(
                local_squared_diff, dim=1)) * loss_major_bone_weight

        else:
            loss_bone = torch.tensor(0.0, device=loss_pre_root.device)
            loss_major_bone = torch.tensor(0.0, device=loss_pre_root.device)

        losses_dict = dict(
            loss_pre_root=loss_pre_root,
            loss_pre_nimble=loss_pre_nimble,
            loss_pre_all=loss_pre_all,
            loss_bone=loss_bone,
            loss_major_bone=loss_major_bone,
            loss_pinch=loss_pinch,
            loss_nimble_trans=loss_nimble_trans,
            loss_rle_all=loss_rle_all,
            loss_2d=loss_2d)
        return losses_dict

    def forward(self, feats, cam_f, cam_xf) -> dict:
        B = feats.shape[0] // 2
        feats = self.proj_layer(feats)
        # fuse multiv feat
        cur_fused_features = self.forward_feature_fuse(feats, cam_f, cam_xf)
        cur_fused_features = cur_fused_features.view(B, -1, 1, 1)
        if self.use_gmlp:
            cur_fused_features = self.gmlp(cur_fused_features)
        sigma = self.sigma_conv(cur_fused_features).reshape(B, 21, 3)
        output = self.last_layer(cur_fused_features)
        return output, sigma

    def forward_feature_fuse(self, feats, cam_f, cam_xf):
        # Per-view img features
        singlev_scaled_to_orig_xf = self.compute_singlev_xfs(cam_f).to(
            feats.device)
        extrinsics_xf = torch.inverse(cam_xf.view(-1, 4, 4)).view(
            cam_xf.shape)  # B,2,4,4
        feats = feats.reshape((-1, 2) + feats.shape[1:])  # B,2,6,8,8
        img_features = self.compute_multiv_features(feats,
                                                    singlev_scaled_to_orig_xf,
                                                    extrinsics_xf)
        return img_features

    def compute_singlev_xfs(self, cam_f, canonical_focal_length: float = 200):
        singlev_scaled_to_orig_xf = torch.eye(4).repeat(cam_f.shape +
                                                        (1, 1)).to(
                                                            cam_f.device)
        if self.use_scaled_as_canonical:
            singlev_scaled_to_orig_xf[
                ..., 2, 2] = cam_f / canonical_focal_length  # B,2,4,4
        return singlev_scaled_to_orig_xf

    def compute_multiv_features(
        self,
        img_features: torch.Tensor,
        singlev_scaled_to_orig_xf: torch.Tensor,
        extrinsics_xf: torch.Tensor,
    ) -> torch.Tensor:
        assert img_features.shape[1] == 2, 'Only 2 views supported'

        (
            multiv_scaled_to_canonical_xf,
            multiv_canonical_to_cam0_xf,
        ) = self._compute_multiv_xfs(singlev_scaled_to_orig_xf, extrinsics_xf)
        # multiv_scaled_to_canonical_xf[..., :3, 3] = 0.
        # Transform all the features to the canonical space
        multiv_canonical_features = apply_ftl_to_feature_maps(
            multiv_scaled_to_canonical_xf.reshape(-1, 4, 4),
            torch.flatten(img_features, start_dim=0,
                          end_dim=1)).reshape(img_features.shape)
        # Flatten the view and channel dimensions then apply multi-view fusion.
        multiv_fused_img_features = self._multi_view_fusion(
            torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2))
        # Apply ftl so that the maps are transformed from
        # the canonical space to cam0 space.
        cam0_maps = apply_ftl_to_feature_maps(multiv_canonical_to_cam0_xf,
                                              multiv_fused_img_features)

        return cam0_maps

    def _compute_multiv_xfs(self, singlev_scaled_to_orig_xf: torch.Tensor,
                            extrinsics_xf: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given per-view data for one frame of 2 views, compute transformation
        from its view to the canonical space.

        The canonical space is related to cam0 by a transformation
        `canonical_to_cam0`, which is identity if
        `self.use_scaled_as_canonical` is `True`, or the scaling transform
        """
        # Use the first camera space as the reference camera.
        xf_0 = extrinsics_xf[:, 0:1].clone()
        xf_inv = torch.inverse(extrinsics_xf.view(-1, 4, 4)).reshape(
            extrinsics_xf.shape)
        xf_to_world = xf_inv @ singlev_scaled_to_orig_xf
        canonical_to_cam0_xf = singlev_scaled_to_orig_xf[:, 0].clone()
        s_0 = torch.inverse(singlev_scaled_to_orig_xf[:, 0:1].clone())
        scaled_to_canonical_xf = s_0 @ xf_0 @ xf_to_world
        return scaled_to_canonical_xf, canonical_to_cam0_xf

    def get_full_pose_with_part_pars(self, pose_reg):
        used_nimble_para = torch.tensor(self.used_nimble_para)
        pose_out = torch.zeros((pose_reg.shape[0], 57),
                               device=pose_reg.device,
                               dtype=torch.float32)
        pose_out[:, used_nimble_para] = pose_reg.to(torch.float32)
        return pose_out

    def cal_normalize_vector(self, vector):
        vector_norms = torch.sqrt(
            torch.sum(vector**2, dim=1, keepdim=True) + 1e-8)
        normalized_vector = vector / vector_norms
        return normalized_vector

    def generate_static_mask(self, batch_data_samples):
        mask = []
        for batch_sample in batch_data_samples[::2]:
            data_info = batch_sample.img_path.split('/')[-1].split(
                '__')[1].split('_')[0]
            if data_info in self.static_data_date_list:
                mask.append(True)
            else:
                mask.append(False)
        mask = torch.tensor(mask)
        return mask

    def enhanced_fun(self, kpt, mask, weight):
        enhanced_kpt = kpt.clone()
        enhanced_kpt[mask] = enhanced_kpt[mask] * weight
        return enhanced_kpt
