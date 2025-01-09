# Copyright (c) XREAL. All rights reserved.
import copy
import random
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mmpose.models.heads.nimble.nimble_utils import (batch_rodrigues,
                                                     convert_vector2matrix,
                                                     decode_svd,
                                                     euler_angles_to_matrix,
                                                     rot6D_to_matirx,
                                                     rot9D_to_matirx)
from mmpose.models.heads.regression_heads.temporal_lift_head_rot_standard_e2e_2d import \
    TemporalLiftNimbleHeadStandardE2e2D
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


@MODELS.register_module()
class TemporalLiftNimbleHeadStandardE2e2DMono(
        TemporalLiftNimbleHeadStandardE2e2D):
    """liftHead for getting 3d rotation from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 d_ffn: int = 220,
                 kpt2d_with_depth: bool = False,
                 use_svd: bool = True,
                 use_nimble_part_para: bool = False,
                 shape_ncomp: int = 20,
                 pose_ncomp: int = 60,
                 reg_shape_type: int = 1,
                 skeleton_feature_dim: int = 64,
                 use_pose_pca: bool = True,
                 reproj: bool = False,
                 reproj_thre=0,
                 iou_thre=0,
                 pad_2d=0,
                 lambda_t: int = -1,
                 corruption_cam: float = 0.5,
                 use_bone_loss: bool = True,
                 use_shape_smooth=True,
                 use_9d_pose_reg: bool = False,
                 use_6d_pose_reg: bool = False,
                 all_use_kp2d_gt: bool = False,
                 mono: bool = False,
                 random_camera: int = 0,
                 seq_len: int = 4,
                 enhance_lefthand=True,
                 enhance_static=True,
                 fix_sigma_pars=False,
                 data_flip_aug: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(
            lift_loss=lift_loss,
            d_ffn=d_ffn,
            reproj=reproj,
            pose_ncomp=pose_ncomp,
            use_svd=use_svd,
            reproj_thre=reproj_thre,
            iou_thre=iou_thre,
            pad_2d=pad_2d,
            use_bone_loss=use_bone_loss,
            use_6d_pose_reg=use_6d_pose_reg,
            use_9d_pose_reg=use_9d_pose_reg,
            lambda_t=lambda_t,
            all_use_kp2d_gt=all_use_kp2d_gt,
            data_flip_aug=data_flip_aug,
            init_cfg=init_cfg,
            enhance_static=enhance_static,
            enhance_lefthand=enhance_lefthand,
            use_shape_smooth=use_shape_smooth,
            seq_len=seq_len,
            fix_sigma_pars=fix_sigma_pars)
        self.temporal_mono = copy.deepcopy(self.temporal)
        self.sigma_conv_mono = copy.deepcopy(self.sigma_conv)
        self.last_layer_mono = copy.deepcopy(self.last_layer)
        self.liftnet_mono = copy.deepcopy(self.liftnet)
        self.load_mono = False
        self.random_camera = random_camera
        if mono:
            for param in self.liftnet.parameters():
                param.requires_grad = False
            for param in self.sigma_conv.parameters():
                param.requires_grad = False
            for param in self.last_layer.parameters():
                param.requires_grad = False
            for param in self.temporal.parameters():
                param.requires_grad = False

    def _forward(self, feats: Tuple[Tensor], mems=None, mono=False) -> Tensor:
        devices_cuda = feats.device
        B = feats.shape[0]
        if not mono:
            out_feats = self.liftnet(feats.reshape(B, 21, -1)).reshape(B, -1)
            if mems is None:
                mems = torch.zeros(B, out_feats.shape[-1], 1,
                                   1).to(devices_cuda)
            feat_mix = torch.cat(
                [out_feats.reshape(B, -1),
                 mems.reshape(B, -1)], dim=1)
            score = self.sigma_conv(out_feats.reshape(B,
                                                      -1)).sigmoid().mean(-1)
            mems = self.temporal(feat_mix)
            output = self.last_layer(feat_mix)
        else:
            out_feats = self.liftnet_mono(feats.reshape(B, 21,
                                                        -1)).reshape(B, -1)
            if mems is None:
                mems = torch.zeros(B, out_feats.shape[-1], 1,
                                   1).to(devices_cuda)
            feat_mix = torch.cat(
                [out_feats.reshape(B, -1),
                 mems.reshape(B, -1)], dim=1)
            score = self.sigma_conv_mono(out_feats.reshape(
                B, -1)).sigmoid().mean(-1)
            mems = self.temporal_mono(feat_mix)
            output = self.last_layer_mono(feat_mix)
        shape, rot, svd_pt = self.simple_feature_layer(output[..., None, None])
        score = score.reshape(shape.shape)
        return shape, rot, svd_pt, mems, score

    def forward(self,
                feats: Tuple[Tensor],
                mems=None,
                mono=False,
                seq_len: int = 1) -> Tensor:
        if not mono:
            feats = self.liftnet(feats).reshape(feats.shape[0], -1)
            sigma = self.sigma_conv(feats)
            sigmas = sigma.reshape(feats.shape[0], 21, 3)
            B = int(feats.shape[0] / seq_len)
            if mems is None:
                mems = torch.zeros(B, feats.shape[-1]).cuda()
            feats = feats.view(B, seq_len, -1)
            outputs = torch.zeros((B, seq_len, self.output_num)).cuda()
            for i in range(seq_len):
                feat = feats[:, i:i + 1, :].reshape(B, -1)
                feat_mix = torch.cat([feat, mems], dim=1)
                mems = self.temporal(feat_mix)
                output = self.last_layer(feat_mix)
                outputs[:, i, ...] = output
        else:
            feats = self.liftnet_mono(feats).reshape(feats.shape[0], -1)
            sigma = self.sigma_conv_mono(feats)
            sigmas = sigma.reshape(feats.shape[0], 21, 3)
            B = int(feats.shape[0] / seq_len)
            if mems is None:
                mems = torch.zeros(B, feats.shape[-1]).cuda()
            feats = feats.view(B, seq_len, -1)
            outputs = torch.zeros((B, seq_len, self.output_num)).cuda()
            for i in range(seq_len):
                feat = feats[:, i:i + 1, :].reshape(B, -1)
                feat_mix = torch.cat([feat, mems], dim=1)
                mems = self.temporal_mono(feat_mix)
                output = self.last_layer_mono(feat_mix)
                outputs[:, i, ...] = output
        outputs = outputs.reshape(B * seq_len, -1, 1, 1)
        return outputs, mems, sigmas

    def preprocess(self, feats, batch_data_samples, mode):
        xy_coord = feats[..., :2]
        B = int(len(batch_data_samples) / 2)
        N = 2
        H, W = batch_data_samples[0].input_size
        K = xy_coord.shape[1]
        # kpt2d output to crop wh
        uv_coord_im_pred_crop_right = xy_coord * torch.tensor([W, H]).cuda()
        uv_coord_im_pred_crop = uv_coord_im_pred_crop_right.view(B, N, K, 2)
        leftcam_cam_matrix = []
        rightcam_cam_matrix = []
        left_vir_cam_matrix = []
        right_vir_cam_matrix = []
        left_vircam_xf = []
        right_vircam_xf = []
        lr_p = []
        lr_rot_matrix = []
        hand3d_gt = []
        is_left_hands = []
        edge_able = [[], []]
        nimble_pose = []
        nimble_trans = []
        nimble_shape = []
        nimble_info = dict()
        uv_coord_im_gt_global = []

        for i, data_sample in enumerate(batch_data_samples):
            if i % 2 == 0:
                left_vir_camera = data_sample.meta['virtual_camera']
                left_camera = data_sample.meta['ori_camera']
                leftcam_cam_matrix.append(left_camera.uv_to_window_matrix())
                left_vir_cam_matrix.append(
                    left_vir_camera.uv_to_window_matrix())
                hand3d_gt.append(data_sample.gt_instances.keypoints3d[0])
                edge_able[0].append(data_sample.meta.get('edge_able', False))
                if 'nimble_pose' in data_sample.meta.keys() and not np.equal(
                        data_sample.meta['nimble_pose'].any(), None):
                    nimble_pose.append(data_sample.meta['nimble_pose'])
                    nimble_trans.append(data_sample.meta['nimble_translation'])
                    nimble_shape.append(data_sample.meta['nimble_shape'])
                if data_sample.meta['category_id'] == 1:
                    is_left_hands.append(1)
                    if data_sample.meta['flipped']:
                        uv_coord_im_pred_crop[
                            i // 2, :, :,
                            0] = W - 1 - uv_coord_im_pred_crop[i // 2, :, :, 0]
                else:
                    is_left_hands.append(0)
            else:
                right_camera = data_sample.meta['ori_camera']
                rightcam_cam_matrix.append(right_camera.uv_to_window_matrix())
                right_vir_camera = data_sample.meta['virtual_camera']
                right_vir_cam_matrix.append(
                    right_vir_camera.uv_to_window_matrix())
                left_vircam_xf.append(
                    left_vir_camera.camera_to_world_xf[:3, :3])
                right_vircam_xf.append(
                    right_vir_camera.camera_to_world_xf[:3, :3])
                left_cam_xf = left_camera.camera_to_world_xf
                right_cam_xf = data_sample.meta['ori_xf']
                lr_t = np.dot(np.linalg.inv(left_cam_xf),
                              right_cam_xf).astype(np.float32)
                left_to_right_rt = np.linalg.inv(right_cam_xf)
                lr_rot_matrix.append(lr_t[:3, :3])
                lr_p.append(lr_t[:3, 3])
                edge_able[1].append(data_sample.meta.get('edge_able', False))
            uv_coord_im_gt_global.append(data_sample.gt_instances.keypoints)
        leftcam_cam_matrix = torch.tensor(
            np.array(leftcam_cam_matrix)).cuda().float()
        rightcam_cam_matrix = torch.tensor(
            np.array(rightcam_cam_matrix)).cuda().float()
        left_vir_cam_matrix = torch.tensor(
            np.array(left_vir_cam_matrix)).cuda().float()
        right_vir_cam_matrix = torch.tensor(
            np.array(right_vir_cam_matrix)).cuda().float()
        left_vircam_xf = torch.tensor(np.array(left_vircam_xf)).cuda().float()
        right_vircam_xf = torch.tensor(
            np.array(right_vircam_xf)).cuda().float()
        lr_p = torch.tensor(np.array(lr_p)).cuda().float()
        lr_rot_matrix = torch.tensor(np.array(lr_rot_matrix)).cuda().float()
        left_to_right_rt = torch.tensor(
            np.array(left_to_right_rt)).cuda().float()
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
        edge_able = torch.tensor(np.array(edge_able)).cuda().float()
        if len(nimble_pose) > 0:
            nimble_pose = torch.tensor(np.array(nimble_pose)).cuda().float()
            nimble_trans = torch.tensor(np.array(nimble_trans)).cuda().float()
            nimble_shape = torch.tensor(np.array(nimble_shape)).cuda().float()
        left_hand = torch.tensor(np.array(is_left_hands)).cuda().float()
        uv_coord_im_gt_global = torch.tensor(
            np.array(uv_coord_im_gt_global)).cuda().float()
        uv_coord_im_gt_global = uv_coord_im_gt_global[..., :2]
        uv_coord_im_gt_global = uv_coord_im_gt_global.view(-1, K, 2)

        uv_coord_im_pred_global = uv_coord_im_pred_crop.view(B, N, K, 2)

        def exchange_value(value):
            tmp = value.clone()
            tmp[::2, ...] = value[1::2, ...]
            tmp[1::2, ...] = value[::2, ...]
            value = tmp
            return value

        origin_W = batch_data_samples[0].meta['frame_width']
        if self.data_flip_aug:
            right_hand = torch.ones_like(
                left_hand).cuda().float() - left_hand.clone().cuda().float()
            new_uv_coord_im_pred_global = uv_coord_im_pred_global.clone()
            new_uv_coord_im_pred_global[
                ..., 0] = origin_W - 1 - uv_coord_im_pred_global[..., 0]
            new_uv_coord_im_pred_global = exchange_value(
                new_uv_coord_im_pred_global)
            left_hand = torch.concat([left_hand, right_hand])
            uv_coord_im_pred_global = torch.concat(
                [uv_coord_im_pred_global, new_uv_coord_im_pred_global], dim=0)

            B *= 2
            left_vir_cam_matrix = torch.concat(
                [left_vir_cam_matrix, left_vir_cam_matrix])
            right_vir_cam_matrix = torch.concat(
                [right_vir_cam_matrix, right_vir_cam_matrix])
            left_vircam_xf = torch.concat([left_vircam_xf, left_vircam_xf])
            right_vircam_xf = torch.concat([right_vircam_xf, right_vircam_xf])
            hand3d_gt = torch.concat([hand3d_gt, hand3d_gt])
            edge_able = torch.concat([edge_able, edge_able])
            valid_mask = torch.concat(
                [torch.ones(B // 2), torch.zeros(B // 2)]).cuda().float()
            if len(nimble_pose) > 0:
                nimble_pose = torch.concat([nimble_pose, nimble_pose])
                nimble_trans = torch.concat([nimble_trans, nimble_trans])
                nimble_shape = torch.concat([nimble_shape, nimble_shape])
        else:
            valid_mask = torch.ones_like(left_hand).cuda().float()

        try:
            nimble_info = {
                'nimble_pose': nimble_pose,
                'nimble_trans': nimble_trans,
                'nimble_shape': nimble_shape
            }
        except Exception as e:
            nimble_info = {
                'nimble_pose': None,
                'nimble_trans': None,
                'nimble_shape': None,
            }
            print(f'An error occurred: {e}')

        leftcam_uv = uv_coord_im_pred_global[:, 0].clone()
        leftcam_x = (leftcam_uv[:, :, 0] - left_vir_cam_matrix[:, 0, 2].view(
            (B, 1))) / left_vir_cam_matrix[:, 0, 0].view(B, 1)
        leftcam_y = (leftcam_uv[:, :, 1] - left_vir_cam_matrix[:, 1, 2].view(
            (B, 1))) / left_vir_cam_matrix[:, 1, 1].view(B, 1)
        leftcam_xy = torch.cat(
            (leftcam_x.unsqueeze(-1), leftcam_y.unsqueeze(-1)), dim=2)
        rightcam_uv = uv_coord_im_pred_global[:, 1].clone()
        rightcam_x = (rightcam_uv[:, :, 0] -
                      right_vir_cam_matrix[:, 0, 2].view(
                          (B, 1))) / right_vir_cam_matrix[:, 0, 0].view(B, 1)
        rightcam_y = (rightcam_uv[:, :, 1] -
                      right_vir_cam_matrix[:, 1, 2].view(
                          (B, 1))) / right_vir_cam_matrix[:, 1, 1].view(B, 1)
        rightcam_xy = torch.cat(
            (rightcam_x.unsqueeze(-1), rightcam_y.unsqueeze(-1)), dim=2)

        uv_coord_im_pred_global = uv_coord_im_pred_global.view(-1, K, 2)

        # 2D GT 转归一化平面坐标：先去畸变，再转系
        B //= 2
        for i, data_sample in enumerate(batch_data_samples):
            camera_model = data_sample.meta['ori_camera']
            kpt2d_u = camera_model.undistort(
                uv_coord_im_gt_global[i].cpu().numpy())
            uv_coord_im_gt_global[i] = torch.from_numpy(kpt2d_u).cuda()
        uv_coord_im_gt_global = uv_coord_im_gt_global.view(B, N, K, 2)
        leftcam_uv_gt = uv_coord_im_gt_global[:, 0]
        leftcam_x_gt = (leftcam_uv_gt[:, :, 0] -
                        leftcam_cam_matrix[:, 0, 2].view(
                            (B, 1))) / leftcam_cam_matrix[:, 0, 0].view(B, 1)
        leftcam_y_gt = (leftcam_uv_gt[:, :, 1] -
                        leftcam_cam_matrix[:, 1, 2].view(
                            (B, 1))) / leftcam_cam_matrix[:, 1, 1].view(B, 1)
        leftcam_xyz_gt = torch.cat(
            (leftcam_x_gt.unsqueeze(-1), leftcam_y_gt.unsqueeze(-1)), dim=2)
        rightcam_uv_gt = uv_coord_im_gt_global[:, 1]
        rightcam_x_gt = (
            rightcam_uv_gt[:, :, 0] - rightcam_cam_matrix[:, 0, 2].view(
                (B, 1))) / rightcam_cam_matrix[:, 0, 0].view(B, 1)
        rightcam_y_gt = (
            rightcam_uv_gt[:, :, 1] - rightcam_cam_matrix[:, 1, 2].view(
                (B, 1))) / rightcam_cam_matrix[:, 1, 1].view(B, 1)
        rightcam_xyz_gt = torch.cat(
            (rightcam_x_gt.unsqueeze(-1), rightcam_y_gt.unsqueeze(-1)), dim=2)

        # 2D 模型推理的2D点，用于指标测试
        if mode == 'predict':
            for i, data_sample in enumerate(batch_data_samples):
                virtual_cam = batch_data_samples[i].meta['virtual_camera']
                ori_cam = batch_data_samples[i].meta['ori_camera']
                kpt_norm_eye = virtual_cam.window_to_eye(
                    uv_coord_im_pred_global[i].clone().detach().cpu())
                kpt_norm_world = virtual_cam.eye_to_world(kpt_norm_eye)
                kpt2d_ori = ori_cam.eye_to_window(kpt_norm_world)
                uv_coord_im_pred_global[i] = torch.tensor(
                    kpt2d_ori).cuda().float()

        # 相机坐标转标准双目
        oricam_left_xyz, oricam_right_xyz = self.standardize_stereo(
            leftcam_xy, rightcam_xy, left_vircam_xf, right_vircam_xf)

        B *= 2
        random_camera = self.random_camera  # 预测时只用其中一个单目
        if mode == 'loss':
            random_camera = random.choice([0, 1])
        # 当前batch被随机取单目：0为左单目，1为右单目
        ori_mono_xyz = oricam_left_xyz if random_camera == 0 else oricam_right_xyz
        feats = torch.cat((ori_mono_xyz[:, :, :2], ori_mono_xyz[:, :, :2]),
                          dim=-1)
        feats_bino = torch.cat(
            (oricam_left_xyz[:, :, :2], oricam_right_xyz[:, :, :2]), dim=-1)
        hand_feat = left_hand[:, None, None].repeat(1, 21, 1)
        feats = torch.cat((feats, hand_feat), dim=-1)
        feats_bino = torch.cat((feats_bino, hand_feat), dim=-1)
        return {
            'feats': feats,
            'feats_bino': feats_bino,
            'oricam_left_xyz': oricam_left_xyz,
            'oricam_right_xyz': oricam_right_xyz,
            'leftcam_xyz_gt': leftcam_xyz_gt,
            'rightcam_xyz_gt': rightcam_xyz_gt,
            'lr_rot_matrix': lr_rot_matrix,
            'lr_p': lr_p,
            'left_to_right_rt': left_to_right_rt,
            'left_vir_cam_matrix': left_vir_cam_matrix,
            'right_vir_cam_matrix': right_vir_cam_matrix,
            'uv_coord_im_pred_global': uv_coord_im_pred_global,
            'hand3d_gt': hand3d_gt,
            'leftcam_xy': leftcam_xy,
            'left_hand': left_hand,
            'nimble_info': nimble_info,
            'valid_mask': valid_mask,
            'random_camera': random_camera
        }

    def postprocess(self, output, left_hand, nimble_info, hand3d_gt, only_pre,
                    shape_v):
        B = output.shape[0]
        cuda_device = output.device

        pose_len = self.pose_ncomp
        rot_vector_t = output[:, :pose_len, 0, 0].float()
        if self.use_nimble_part_para:
            rot_vector_t = self.get_full_pose_with_part_pars(rot_vector_t)
        svd_begin = self.pose_ncomp + self.shape_ncomp
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
        pre_shape_vector = shape_v.clone()
        if self.data_flip_aug:
            pre_shape_vector[(B // 2):] = pre_shape_vector[:(B // 2)]

        if not only_pre:
            with torch.no_grad():
                gt_root_xyz = nimble_info['nimble_trans'].unsqueeze(-1)[:, :,
                                                                        0]
                gt_root_matrix = batch_rodrigues(  # 轴角 -> rot
                    nimble_info['nimble_pose'][:, 0, :]).reshape(-1, 3, 3)

                init_root_rot = torch.zeros((B, 1, 3),
                                            requires_grad=True,
                                            device=cuda_device)
                gt_rot_vector = torch.cat(
                    (init_root_rot, nimble_info['nimble_pose'][:, 1:, :]),
                    dim=1)
                gt_local_matrix = convert_vector2matrix(
                    gt_rot_vector.view(B, -1)).reshape(B, -1, 9)
                # gt_shape_vector = nimble_info['nimble_shape']

        def get_nimble_3d(root_xyz, root_matrix, local_matrix, shape_vector):

            _, bone_joints = self.nimble_layer.forward_simple(
                local_matrix, shape_vector)  # 通过局部点旋转，scale，将默认局部手型得到实际局部手型
            rebuild_joints = bone_joints[:, self.kp_index, :]
            root_rebuild_joints = rebuild_joints[:, 0:1, :]
            rebuild_joints_temp = rebuild_joints - root_rebuild_joints

            mask = left_hand == 1
            add_matrix = torch.eye(3).unsqueeze(0).expand(B, -1,
                                                          -1).to(cuda_device)
            add_matrix[mask, 0, 0] = -add_matrix[mask, 0, 0]
            root_matrix = torch.matmul(root_matrix, add_matrix)
            rebuild_joints_temp = torch.matmul(rebuild_joints_temp,
                                               root_matrix.transpose(1, 2))
            rebuild_joints_with_scale = rebuild_joints_temp / self.scale_parameter

            xyz_point = rebuild_joints_with_scale + root_xyz.unsqueeze(1)
            return xyz_point

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

            return (pre_root__xyz, pre_nimble__xyz, pre_all__xyz, gt_all__xyz,
                    pre_root_xyz)

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                mems_bino=None,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'predict')
        valid_mask = data['valid_mask'] == 1
        output_bino, mems_bino, _ = self.forward(data['feats_bino'], mems_bino,
                                                 False, 1)  # bino
        output, mems, all_sigmas = self.forward(data['feats'], mems, True,
                                                1)  # mono

        shape_bino = output_bino[:, self.pose_ncomp:self.pose_ncomp +
                                 self.shape_ncomp, 0, 0]
        hand3d_pred = self.postprocess(output, data['left_hand'],
                                       data['nimble_info'], data['hand3d_gt'],
                                       True, shape_bino)[0]
        hand3d_pred = hand3d_pred[valid_mask]
        uv_coord_im_pred_global = data['uv_coord_im_pred_global'][
            valid_mask.repeat_interleave(2)]
        # camera_model = batch_data_samples[0].meta['ori_camera']
        # leftcam_uv_reproj_distort = camera_model.eye_to_window(
        #     hand3d_pred.cpu().numpy())
        # leftcam_uv_reproj_distort = torch.tensor(
        #     leftcam_uv_reproj_distort).cuda()
        # return hand3d_pred, leftcam_uv_reproj_distort, mems, all_sigmas
        return hand3d_pred, uv_coord_im_pred_global, mems, mems_bino, all_sigmas

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        # 单目模型未加载权重时，只加载一次双目模型权重
        if not self.load_mono:
            self.temporal_mono.load_state_dict(self.temporal.state_dict())
            self.sigma_conv_mono.load_state_dict(self.sigma_conv.state_dict())
            self.last_layer_mono.load_state_dict(self.last_layer.state_dict())
            self.liftnet_mono.load_state_dict(self.liftnet.state_dict())
            self.load_mono = True
        data = self.preprocess(feats, batch_data_samples, 'loss')
        hand3d_gt = data['hand3d_gt']
        valid_mask = data['valid_mask'] == 1
        output_bino = self.forward(data['feats_bino'], None, False,
                                   self.seq_len)[0]
        shape_bino = output_bino[:, self.pose_ncomp:self.pose_ncomp +
                                 self.shape_ncomp, 0, 0]
        output, mems, all_sigmas = self.forward(data['feats'], None, True,
                                                self.seq_len)

        # 3d 损失
        (pred_3d_way1, pred_3d_way2, hand3d_pred, hand3d_part_gt,
         pre_trans_xyz) = self.postprocess(output, data['left_hand'],
                                           data['nimble_info'],
                                           data['hand3d_gt'], False,
                                           shape_bino)
        pred_3d_way1 = pred_3d_way1[valid_mask]
        hand3d_pred = hand3d_pred[valid_mask]
        pre_trans_xyz = pre_trans_xyz[valid_mask]
        all_sigmas = all_sigmas[valid_mask]
        hand3d_gt = data['hand3d_gt'][valid_mask]

        # 如果是右单目，以上3d点结果都需要转为世界系下3d
        # 如果是左单目，data_flip_aug的另一半是右单目需要转世界系，但用的gt_root所以不需要
        if data['random_camera'] == 1:
            B, K = hand3d_pred.shape[:2]
            lr_rot_matrix = data['lr_rot_matrix'].view(B, 1, 3, 3).repeat(
                1, K, 1, 1).view(B * K, 3, 3)
            lr_p = data['lr_p'].view(B, 1, 3, 1).repeat(1, K, 1,
                                                        1).view(B * K, 3, 1)
            pred_3d_way1 = (torch.bmm(lr_rot_matrix,
                                      pred_3d_way1.view(B * K, 3, 1) +
                                      lr_p)).view(B, K, 3)
            hand3d_pred = (torch.bmm(lr_rot_matrix,
                                     hand3d_pred.view(B * K, 3, 1) +
                                     lr_p)).view(B, K, 3)
            lr_rot_matrix = data['lr_rot_matrix'].view(B, 3, 3)
            lr_p = data['lr_p'].view(B, 3, 1)
            pre_trans_xyz = (torch.bmm(lr_rot_matrix,
                                       pre_trans_xyz.view(B, 3, 1) +
                                       lr_p)).view(B, 3)

        # 以下都是在左目下的3d点
        # 直接监督rot和trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz
        gt_nimble_trans = data['nimble_info']['nimble_trans'][valid_mask]

        # 监督中间2d输出和最终2d
        norm_left_pred_reproj = self.reproj_norm_2d(hand3d_pred)
        norm_left_pred = data['oricam_left_xyz'][valid_mask][..., :2]
        norm_left_gt, norm_left_gt_reproj = data['leftcam_xyz_gt'], data[
            'leftcam_xyz_gt']
        # 2d转3d的数据不计算3d相关loss
        convert_2d_mask = self.generate_2d_maskv1(batch_data_samples).to(
            hand3d_pred.device)  # 所有2d，都不计算3d loss
        convert_2d_maskv2 = self.generate_2d_maskv2(batch_data_samples).to(
            hand3d_pred.device)  # 仅时序2d，可以计算重投影
        pred_3d_way1 = pred_3d_way1 * (1 - convert_2d_mask.float())
        all_sigmas = all_sigmas * (1 - convert_2d_mask.float())
        hand3d_pred = hand3d_pred * (1 - convert_2d_mask.float())
        pre_nimble_trans = pre_nimble_trans * (1 -
                                               convert_2d_mask[..., 0].float())
        norm_left_pred_reproj = norm_left_pred_reproj * (
            1 - convert_2d_maskv2.float())
        norm_left_gt_reproj = norm_left_gt_reproj * (1 -
                                                     convert_2d_maskv2.float())
        if self.data_flip_aug:
            convert_2d_mask = torch.concat([convert_2d_mask, convert_2d_mask],
                                           dim=0)
        hand3d_part_gt = hand3d_part_gt * (1 - convert_2d_mask.float())
        pred_3d_way2 = pred_3d_way2 * (1 - convert_2d_mask.float())
        # pinch 损失
        dist_pred = torch.norm(
            pred_3d_way2[:, 4, :] - pred_3d_way2[:, 8, :], dim=-1)
        dist_gt = torch.norm(
            hand3d_part_gt[:, 4, :] - hand3d_part_gt[:, 8, :], dim=-1)

        if self.enhance_lefthand:
            mask = data['left_hand'][valid_mask] == 1
            mask_rel = data['left_hand'] == 1
            left_weight = 1.2
            enhanced_left_hand3d_gt = self.enhanced_fun(
                hand3d_gt, mask, left_weight)
            enhanced_left_pred_3d_way1 = self.enhanced_fun(
                pred_3d_way1, mask, left_weight)
            enhanced_left_pred_3d_way2 = self.enhanced_fun(
                pred_3d_way2, mask_rel, left_weight)
            enhanced_left_hand3d_pred = self.enhanced_fun(
                hand3d_pred, mask, left_weight)
            enhanced_left_hand3d_part_gt = self.enhanced_fun(
                hand3d_part_gt, mask_rel, left_weight)
        else:
            enhanced_left_hand3d_gt = hand3d_gt
            enhanced_left_pred_3d_way1 = pred_3d_way1
            enhanced_left_pred_3d_way2 = pred_3d_way2
            enhanced_left_hand3d_pred = hand3d_pred
            enhanced_left_hand3d_part_gt = hand3d_part_gt

        if self.enhance_static:
            static_weight = 25
            static_mask = self.generate_mask(batch_data_samples,
                                             self.static_data_date_list)
            enhanced_static_hand3d_pred = self.enhanced_fun(
                hand3d_pred, static_mask, static_weight)
            enhanced_static_pred_3d_way1 = self.enhanced_fun(
                pred_3d_way1, static_mask, static_weight)
            enhanced_static_hand3d_gt = self.enhanced_fun(
                hand3d_gt, static_mask, static_weight)
        else:
            enhanced_static_hand3d_pred = hand3d_pred
            enhanced_static_hand3d_gt = hand3d_gt

        re_all_sigmas = torch.cat((hand3d_pred, all_sigmas), dim=-1)

        pred_for_loss = [
            enhanced_left_pred_3d_way1, enhanced_left_pred_3d_way2,
            enhanced_left_hand3d_pred, dist_pred, pre_nimble_trans,
            enhanced_static_hand3d_pred, enhanced_static_pred_3d_way1,
            re_all_sigmas, norm_left_pred, norm_left_pred_reproj
        ]
        targ_for_loss = [
            enhanced_left_hand3d_gt,
            enhanced_left_hand3d_part_gt,
            enhanced_left_hand3d_gt,
            dist_gt,
            gt_nimble_trans,
            enhanced_static_hand3d_gt,
            enhanced_static_hand3d_gt,
            hand3d_gt,
            norm_left_gt,
            norm_left_gt_reproj,
        ]
        if self.data_flip_aug:
            convert_2d_mask = convert_2d_mask[convert_2d_mask.shape[0] // 2:]
        weight_ini = torch.ones((1, 21, 3))
        weight_ini[0, :9, :] = 2
        weight_ini[0, 4, :], weight_ini[0, 8, :] = 4, 4
        weight_ini_ori = weight_ini.repeat(hand3d_gt.shape[0], 1,
                                           1).to(hand3d_gt.device)

        weight_ini_for_pre_nimble = weight_ini.repeat(
            hand3d_part_gt.shape[0], 1, 1).to(hand3d_part_gt.device)
        weight_ini_for_pre_nimble[:, :9, :] = 4
        weight_ini_for_pre_nimble[:,
                                  4, :], weight_ini_for_pre_nimble[:,
                                                                   8, :] = 8, 8
        weight_ini_ori_rle = weight_ini_ori * (1 - convert_2d_mask.float())
        weight_for_loss = [
            weight_ini_ori, weight_ini_for_pre_nimble, weight_ini_ori, None,
            None, None, None, weight_ini_ori_rle, None, None
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss, weight_for_loss)
        (loss_pre_root, loss_pre_nimble, loss_pre_all, loss_pinch,
         loss_nimble_trans, loss_smooth, loss_smooth_root, loss_rle,
         loss_left_2d, loss_left_2d_reproj) = losses

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

        pinch_mask = (dist_pred > dist_gt - 0.001) & (dist_gt < 0.02)

        reverse_mask = self.generate_mask(batch_data_samples,
                                          self.reverse_pinch_date_list).to(
                                              pinch_mask.cuda())
        if self.data_flip_aug:
            reverse_mask = torch.concat([reverse_mask, reverse_mask])
        pinch_reverse_mask = pinch_mask & reverse_mask

        if sum(pinch_mask) > 0:
            valid_num = len(dist_gt[pinch_mask])
            softmax_weight = F.softmax(-dist_gt[pinch_mask], dim=0) * valid_num
            margin_value = torch.ones_like(dist_gt) * 0.003
            margin_value[pinch_reverse_mask] = 0.005
            pinch_loss_add = self.pinch_loss_func(
                dist_pred[pinch_mask] * softmax_weight,
                (dist_gt[pinch_mask] - margin_value[pinch_mask]) *
                softmax_weight) * 3
        else:
            pinch_loss_add = torch.tensor(0.0, device=loss_pre_root.device)

        plam_ratio = torch.norm(
            hand3d_gt[:, 9, :] - hand3d_gt[:, 0, :], dim=-1) / 0.08
        standard_hand3d_gt = hand3d_gt / plam_ratio[:, None, None]
        dis = (standard_hand3d_gt[:, 8, :] +
               standard_hand3d_gt[:, 5, :]) / 2 - standard_hand3d_gt[:, 6, :]
        dis_norm = torch.norm(dis, dim=1)
        poke_dist_mask = dis_norm < 0.012

        poke_date_mask = self.generate_mask(
            batch_data_samples, self.poke_date_list).to(pinch_mask.cuda())
        poke_mask = poke_dist_mask & poke_date_mask
        if self.data_flip_aug:
            poke_mask = torch.concat([poke_mask, poke_mask])
        if sum(poke_mask) > 0:
            direction_vector1 = pred_3d_way2[poke_mask,
                                             5, :] - pred_3d_way2[poke_mask,
                                                                  6, :]
            direction_vector2 = pred_3d_way2[poke_mask,
                                             6, :] - pred_3d_way2[poke_mask,
                                                                  8, :]
            vector1_norm = F.normalize(direction_vector1, dim=1)
            vector2_norm = F.normalize(direction_vector2, dim=1)
            cosine_similarity = (vector1_norm * vector2_norm).sum(dim=1)
            loss_poke = (1 - torch.abs(cosine_similarity)).mean() / 5
            loss_poke += torch.norm(
                pred_3d_way2[valid_mask][:, 8, :] - hand3d_gt[:, 8, :],
                dim=-1).mean()
        else:
            loss_poke = torch.tensor(0.0, device=hand3d_gt.device)

        if self.fix_sigma_pars:
            loss_rle = torch.tensor(0.0, device=loss_pre_root.device)

        hand_constraint_loss = self.hand_constraint(
            hand3d_pred, self.hand_constraint_index_list) * 0.02
        losses_dict = dict(
            loss_pre_root=loss_pre_root,
            loss_pre_nimble=loss_pre_nimble,
            loss_pre_all=loss_pre_all,
            bone_loss=bone_loss,
            major_bone_loss=major_bone_loss,
            loss_pinch=loss_pinch,
            loss_nimble_trans=loss_nimble_trans,
            loss_smooth=loss_smooth,
            loss_smooth_root=loss_smooth_root,
            loss_rle=loss_rle,
            hand_constraint_loss=hand_constraint_loss,
            loss_poke=loss_poke,
            pinch_loss_add=pinch_loss_add,
            loss_left_2d=loss_left_2d,
            loss_left_2d_reproj=loss_left_2d_reproj,
        )

        return losses_dict

    def cal_normalize_vector(self, vector):
        vector_norms = torch.sqrt(
            torch.sum(vector**2, dim=1, keepdim=True) + 1e-8)
        normalized_vector = vector / vector_norms
        return normalized_vector

    def standardize_stereo(self, leftcam_xy, rightcam_xy, left_vircam_xf,
                           right_vircam_xf):
        """transform to standard stereo system."""
        oricam_left_xyz = self.align_monocular_to_parallel_stereo(
            leftcam_xy, left_vircam_xf)
        oricam_right_xyz = self.align_monocular_to_parallel_stereo(
            rightcam_xy, right_vircam_xf)
        oricam_left_xyz = oricam_left_xyz / oricam_left_xyz[:, :, 2:]
        oricam_right_xyz = oricam_right_xyz / oricam_right_xyz[:, :, 2:]
        return oricam_left_xyz, oricam_right_xyz

    @staticmethod
    def align_monocular_to_parallel_stereo(cam_xy, vircam_xf):
        B, K = cam_xy.shape[:2]
        cam_xyz = torch.cat((cam_xy, torch.ones(B, K, 1).cuda()),
                            dim=-1).view(B * K, 3, 1)
        vircam_xf = vircam_xf.view(B, 1, 3, 3).repeat(1, K, 1,
                                                      1).view(B * K, 3, 3)

        oricam_cam_xyz = torch.matmul(vircam_xf, cam_xyz).view(B, K, 3)
        return oricam_cam_xyz.view(B, K, 3)

    def generate_mask(self, batch_data_samples, date_list):
        mask = []
        for batch_sample in batch_data_samples[::2]:
            if 'XS__' in batch_sample.img_path:
                data_info = batch_sample.img_path.split('/')[-1].split(
                    '__')[1].split('_')[0]
                if data_info in date_list:
                    mask.append(True)
                else:
                    mask.append(False)
            else:
                mask.append(False)
        mask = torch.tensor(mask)
        return mask

    def enhanced_fun(self, kpt, mask, weight):
        enhanced_kpt = kpt.clone()
        enhanced_kpt[mask] = enhanced_kpt[mask] * weight
        return enhanced_kpt

    def reproj_norm_2d(self, hand3d):
        B, K = hand3d.shape[:2]
        norm_left = hand3d[..., :2] / hand3d[..., 2:]
        return norm_left

    @staticmethod
    def generate_2d_maskv1(batch_data_samples):
        mask = []
        for batch_sample in batch_data_samples[::2]:
            if 'hand_train_flora' in batch_sample.img_path:
                mask.append(True)
            else:
                mask.append(False)
        mask = torch.tensor(mask).unsqueeze(1).unsqueeze(2)
        return mask

    @staticmethod
    def generate_2d_maskv2(batch_data_samples):
        mask = []
        for batch_sample in batch_data_samples[::2]:
            if 'hand_train_flora' in batch_sample.img_path:
                if 'hand_train_flora_e2e' in batch_sample.img_path:
                    mask.append(False)
                else:
                    mask.append(True)
            else:
                mask.append(False)
        mask = torch.tensor(mask).unsqueeze(1).unsqueeze(2)
        return mask

    def hand_constraint(self, kpt3d, index_list):
        hand_constraint_loss = 0
        for index_sin in index_list:
            vector_1 = kpt3d[:, index_sin[0], :] - kpt3d[:, index_sin[1], :]
            vector_2 = kpt3d[:, index_sin[1], :] - kpt3d[:, index_sin[2], :]
            vector_3 = kpt3d[:, index_sin[2], :] - kpt3d[:, index_sin[3], :]
            out_vector_1 = torch.cross(vector_1, vector_2, dim=1)
            out_vector_2 = torch.cross(vector_2, vector_3, dim=1)

            vector1_norm = F.normalize(out_vector_1, dim=1)
            vector2_norm = F.normalize(out_vector_2, dim=1)
            cosine_similarity = (vector1_norm * vector2_norm).sum(dim=1)
            loss_part = (1 - torch.abs(cosine_similarity)).mean()
            hand_constraint_loss += loss_part
        return hand_constraint_loss
