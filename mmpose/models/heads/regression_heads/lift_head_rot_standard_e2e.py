# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mmpose.models.heads.regression_heads.lift_head_rot_standard import \
    LiftNimbleHeadStandard
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


@MODELS.register_module()
class LiftNimbleHeadStandardE2e(LiftNimbleHeadStandard):
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
                 baseline=0.13,
                 reproj_thre=0,
                 iou_thre=0,
                 pad_2d=0,
                 lambda_t: int = -1,
                 corruption_cam: float = 0.5,
                 use_bone_loss: bool = True,
                 use_9d_pose_reg: bool = False,
                 use_6d_pose_reg: bool = False,
                 all_use_kp2d_gt: bool = False,
                 fix_sigma_pars=False,
                 data_flip_aug: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(
            lift_loss=lift_loss,
            d_ffn=d_ffn,
            reproj=reproj,
            pose_ncomp=pose_ncomp,
            baseline=baseline,
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
            init_cfg=init_cfg)

        self.fix_sigma_pars = fix_sigma_pars
        self.pinch_loss_func = F.l1_loss

        if self.fix_sigma_pars:
            for param in self.liftnet.parameters():
                param.requires_grad = False
            for param in self.sigma_conv.parameters():
                param.requires_grad = False

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        output = self.liftnet(feats).reshape(feats.shape[0], -1)
        mems = torch.zeros_like(output).to(output.device)
        sigma = self.sigma_conv(output).reshape(feats.shape[0], 21, 3)
        feat_mix = torch.cat([output, mems], dim=1)
        output = self.last_layer(feat_mix).view((feats.shape[0], -1, 1, 1))
        return output, sigma

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
        left_R = []
        right_R = []
        baseline_scale = []
        lr_p = []
        lr_rot_matrix = []
        hand3d_gt = []
        is_left_hands = []
        nimble_pose = []
        nimble_trans = []
        nimble_shape = []
        nimble_info = dict()
        uv_coord_im_gt_global = []

        for i, data_sample in enumerate(batch_data_samples):
            if i % 2 == 0:
                left_vir_camera = data_sample.meta['virtual_camera']
                left_camera = data_sample.meta['ori_camera']
                left_vir_cam_matrix.append(
                    left_vir_camera.uv_to_window_matrix())
                leftcam_cam_matrix.append(left_camera.uv_to_window_matrix())
                left_R.append(data_sample.meta['cam_to_virtual_R'])
                hand3d_gt.append(data_sample.gt_instances.keypoints3d[0])
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
                right_R.append(data_sample.meta['cam_to_virtual_R'])
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
                baseline_scale.append(data_sample.meta['virtual_baseline'] /
                                      self.baseline)

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
        left_R = torch.tensor(np.array(left_R)).cuda().float()
        right_R = torch.tensor(np.array(right_R)).cuda().float()
        baseline_scale = torch.tensor(np.array(baseline_scale)).cuda().float()
        lr_p = torch.tensor(np.array(lr_p)).cuda().float()
        lr_rot_matrix = torch.tensor(np.array(lr_rot_matrix)).cuda().float()
        left_to_right_rt = torch.tensor(
            np.array(left_to_right_rt)).cuda().float()
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
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
            tmp[:, 0, ...] = value[:, 1, ...]
            tmp[:, 1, ...] = value[:, 0, ...]
            value = tmp
            return value

        if self.data_flip_aug:
            right_hand = torch.ones_like(
                left_hand).cuda().float() - left_hand.clone().cuda().float()
            new_uv_coord_im_pred_global = uv_coord_im_pred_global.clone()
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
            left_R = torch.concat([left_R, left_R])
            right_R = torch.concat([right_R, right_R])
            baseline_scale = torch.concat([baseline_scale, baseline_scale])
            hand3d_gt = torch.concat([hand3d_gt, hand3d_gt])
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
        norm_leftcam_xyz_gt = torch.cat(
            (leftcam_x_gt.unsqueeze(-1), leftcam_y_gt.unsqueeze(-1)), dim=2)
        rightcam_uv_gt = uv_coord_im_gt_global[:, 1]
        rightcam_x_gt = (
            rightcam_uv_gt[:, :, 0] - rightcam_cam_matrix[:, 0, 2].view(
                (B, 1))) / rightcam_cam_matrix[:, 0, 0].view(B, 1)
        rightcam_y_gt = (
            rightcam_uv_gt[:, :, 1] - rightcam_cam_matrix[:, 1, 2].view(
                (B, 1))) / rightcam_cam_matrix[:, 1, 1].view(B, 1)
        norm_rightcam_xyz_gt = torch.cat(
            (rightcam_x_gt.unsqueeze(-1), rightcam_y_gt.unsqueeze(-1)), dim=2)

        # 2D 模型推理的2D点，用于指标测试
        # for i, data_sample in enumerate(batch_data_samples):
        #     virtual_cam = batch_data_samples[i].meta['virtual_camera']
        #     ori_cam = batch_data_samples[i].meta['ori_camera']
        #     kpt_norm_eye = virtual_cam.window_to_eye(uv_coord_im_pred_global[i].clone().detach().cpu())
        #     kpt_norm_world = virtual_cam.eye_to_world(kpt_norm_eye)
        #     kpt2d_ori = ori_cam.eye_to_window(kpt_norm_world)
        #     uv_coord_im_pred_global[i] = torch.tensor(kpt2d_ori).cuda().float()

        # 相机坐标转标准双目
        norm_leftcam_xyz, norm_rightcam_xyz = self.standardize_stereo(
            leftcam_xy, rightcam_xy, left_R, right_R, left_vircam_xf,
            right_vircam_xf)

        feats = torch.cat(
            (norm_leftcam_xyz[:, :, :2], norm_rightcam_xyz[:, :, :2]), dim=-1)
        hand_feat = left_hand[:, None, None].repeat(1, 21, 1)
        feats = torch.cat((feats, hand_feat), dim=-1)
        return {
            'feats': feats,
            'norm_leftcam_xyz': norm_leftcam_xyz,
            'norm_rightcam_xyz': norm_rightcam_xyz,
            'norm_leftcam_xyz_gt': norm_leftcam_xyz_gt,
            'norm_rightcam_xyz_gt': norm_rightcam_xyz_gt,
            'lr_rot_matrix': lr_rot_matrix,
            'lr_p': lr_p,
            'left_to_right_rt': left_to_right_rt,
            'left_vir_cam_matrix': left_vir_cam_matrix,
            'right_vir_cam_matrix': right_vir_cam_matrix,
            'uv_coord_im_pred_global': uv_coord_im_pred_global,
            'hand3d_gt': hand3d_gt,
            'left_R': left_R,
            'right_R': right_R,
            'leftcam_xy': leftcam_xy,
            'left_hand': left_hand,
            'baseline_scale': baseline_scale,
            'nimble_info': nimble_info,
            'valid_mask': valid_mask
        }

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'predict')
        valid_mask = data['valid_mask'] == 1
        output, all_sigmas = self.forward(data['feats'])

        hand3d_pred = self.postprocess(
            output,
            data['left_hand'],
            data['leftcam_xy'],
            data['left_R'],
            data['nimble_info'],
            data['hand3d_gt'],
            data['baseline_scale'],
            only_pre=True)[0]
        hand3d_pred = hand3d_pred[valid_mask]
        camera_model = batch_data_samples[0].meta['ori_camera']
        leftcam_uv_reproj_distort = camera_model.eye_to_window(
            hand3d_pred.cpu().numpy())
        leftcam_uv_reproj_distort = torch.tensor(
            leftcam_uv_reproj_distort).cuda()
        return hand3d_pred, leftcam_uv_reproj_distort, all_sigmas

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        data = self.preprocess(feats, batch_data_samples, 'loss')

        valid_mask = data['valid_mask'] == 1
        output, all_sigmas = self.forward(data['feats'])

        hand3d_gt = data['hand3d_gt']
        # 3d 损失
        (pred_3d_way1, pred_3d_way2, hand3d_pred, hand3d_part_gt,
         pre_trans_xyz, pre_shape, gt_all_matrix,
         pre_all_matrix) = self.postprocess(output, data['left_hand'],
                                            data['leftcam_xy'], data['left_R'],
                                            data['nimble_info'],
                                            data['hand3d_gt'],
                                            data['baseline_scale'], False)
        pred_3d_way1 = pred_3d_way1[valid_mask]
        hand3d_pred = hand3d_pred[valid_mask]
        pre_trans_xyz = pre_trans_xyz[valid_mask]
        all_sigmas = all_sigmas[valid_mask]
        hand3d_gt = data['hand3d_gt'][valid_mask]

        # 直接监督rot和trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz
        gt_nimble_trans = data['nimble_info']['nimble_trans'][valid_mask]

        # oricam norm 2d loss
        # oricam norm reproj 2d loss
        norm_left_pred, norm_right_pred = self.reproj_norm_2d(
            hand3d_pred.clone(), data['lr_rot_matrix'].clone(),
            data['lr_p'].clone())
        norm_left_gt, norm_right_gt = data['norm_leftcam_xyz_gt'], data[
            'norm_rightcam_xyz_gt']
        # 2d转3d的数据不计算3d相关loss
        convert_2d_mask = self.generate_2d_mask(batch_data_samples)
        pred_3d_way1[convert_2d_mask] *= 0.
        pred_3d_way2[convert_2d_mask] *= 0.
        pred_3d_way2[convert_2d_mask] *= 0.
        hand3d_part_gt[convert_2d_mask] *= 0.
        all_sigmas[convert_2d_mask] *= 0.
        hand3d_pred[convert_2d_mask] *= 0.
        pre_trans_xyz[convert_2d_mask] *= 0.
        # pinch 损失
        dist_pred = torch.norm(
            pred_3d_way2[:, 4, :] - pred_3d_way2[:, 8, :], dim=-1)
        dist_gt = torch.norm(
            hand3d_part_gt[:, 4, :] - hand3d_part_gt[:, 8, :], dim=-1)

        re_all_sigmas = torch.cat((hand3d_pred, all_sigmas), dim=-1)

        pred_for_loss = [
            pred_3d_way1, pred_3d_way2, hand3d_pred, dist_pred,
            pre_nimble_trans, re_all_sigmas, norm_left_pred, norm_right_pred
        ]
        targ_for_loss = [
            hand3d_gt, hand3d_part_gt, hand3d_gt, dist_gt, gt_nimble_trans,
            hand3d_gt, norm_left_gt, norm_right_gt
        ]

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
        weight_ini_ori_ = weight_ini_ori.clone()
        weight_ini_ori_[convert_2d_mask] *= 0
        weight_for_loss = [
            weight_ini_ori, weight_ini_for_pre_nimble, weight_ini_ori, None,
            None, weight_ini_ori_, None, None
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss, weight_for_loss)
        (loss_pre_root, loss_pre_nimble, loss_pre_all, loss_pinch,
         loss_nimble_trans, loss_rle, loss_left_2d, loss_right_2d) = losses

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

        if self.fix_sigma_pars:
            loss_rle = torch.tensor(0.0, device=loss_pre_root.device)

        losses_dict = dict(
            loss_pre_root=loss_pre_root,
            loss_pre_nimble=loss_pre_nimble,
            loss_pre_all=loss_pre_all,
            bone_loss=bone_loss,
            major_bone_loss=major_bone_loss,
            loss_pinch=loss_pinch,
            loss_nimble_trans=loss_nimble_trans,
            loss_rle=loss_rle,
            loss_left_2d=loss_left_2d,
            loss_right_2d=loss_right_2d)
        return losses_dict

    def cal_normalize_vector(self, vector):
        vector_norms = torch.sqrt(
            torch.sum(vector**2, dim=1, keepdim=True) + 1e-8)
        normalized_vector = vector / vector_norms
        return normalized_vector

    def standardize_stereo(self, leftcam_xy, rightcam_xy, left_R, right_R,
                           left_vircam_xf, right_vircam_xf):
        """transform to standard stereo system."""
        standard_left_xyz = self.align_monocular_to_parallel_stereo(
            leftcam_xy, left_R, left_vircam_xf)
        standard_right_xyz = self.align_monocular_to_parallel_stereo(
            rightcam_xy, right_R, right_vircam_xf)
        norm_left_xyz = standard_left_xyz / standard_left_xyz[:, :, 2:]
        norm_right_xyz = standard_right_xyz / standard_right_xyz[:, :, 2:]
        return norm_left_xyz, norm_right_xyz

    @staticmethod
    def align_monocular_to_parallel_stereo(cam_xy, R, vircam_xf):
        """Aligns a monocular camera to a parallel stereo setup using the given
        rotation matrix."""
        B, K = cam_xy.shape[:2]
        cam_xyz = torch.cat((cam_xy, torch.ones(B, K, 1).cuda()),
                            dim=-1).view(B * K, 3, 1)
        vircam_xf = vircam_xf.view(B, 1, 3, 3).repeat(1, K, 1,
                                                      1).view(B * K, 3, 3)
        R = R.view(B, 1, 3, 3).repeat(1, K, 1, 1).view(B * K, 3, 3)

        oricam_cam_xyz = torch.matmul(vircam_xf, cam_xyz)
        standard_cam_xyz = torch.matmul(R, oricam_cam_xyz).view(B, K, 3)
        return standard_cam_xyz

    @staticmethod
    def generate_2d_mask(batch_data_samples):
        mask = []
        for batch_sample in batch_data_samples[::2]:
            if 'hand_train_flora' in batch_sample.img_path:
                mask.append(True)
            else:
                mask.append(False)
        mask = torch.tensor(mask)
        return mask

    def reproj_norm_2d(self, hand3d, lr_rot_matrix, lr_p):
        B, K = hand3d.shape[:2]
        norm_left = hand3d[..., :2] / hand3d[..., 2:]
        lr_rot_matrix = torch.inverse(lr_rot_matrix).view(B, 1, 3, 3).repeat(
            1, K, 1, 1).view(B * K, 3, 3)
        lr_p = lr_p.view(B, 1, 3, 1).repeat(1, K, 1, 1).view(B * K, 3, 1)
        norm_right = (torch.bmm(lr_rot_matrix,
                                hand3d.view(B * K, 3, 1) -
                                lr_p)).view(B, K, 3)
        norm_right = norm_right[..., :2] / norm_right[..., 2:]
        return norm_left, norm_right
