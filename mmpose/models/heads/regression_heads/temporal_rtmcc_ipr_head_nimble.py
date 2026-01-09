# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MessageHub
from torch import Tensor

from mmpose.models.heads.nimble.nimble_utils import (batch_rodrigues,
                                                     decode_svd,
                                                     rot9D_to_matirx)
from mmpose.models.heads.regression_heads.rtmcc_ipr_head_nimble import \
    RTMCCIPRHeadNimble
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptConfigType, OptSampleList
from ..coord_cls_heads import RTMCCHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class TemporalRTMCCIPRHeadNimble(RTMCCIPRHeadNimble):

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        use_DLT: bool,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 1,
        gau_cfg: ConfigType = dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='ReLU',
            use_rel_bias=False,
            pos_enc=False),
        loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
        output_sigma: bool = True,
        deploy: bool = False,
        with_gau: bool = False,
        deploy_output='kpt',
        feat_channel=6,
        map_type='softmax',
        seq_len=4,
        enhance_static=True,
    ):
        super().__init__(in_channels, out_channels, use_DLT, input_size,
                         in_featuremap_size, simcc_split_ratio,
                         final_layer_kernel_size, gau_cfg, loss, decoder,
                         init_cfg, output_sigma, deploy, with_gau,
                         deploy_output, feat_channel, map_type)

        self.seq_len = seq_len
        self.temporal = nn.Sequential(
            nn.Conv2d(
                2 * self.input_dim, self.nimble_hidden_num, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.nimble_hidden_num, self.input_dim, kernel_size=1))
        self.static_data_date_list = ['20240516', '20240517', '20240522']
        self.reverse_pinch_date_list = ['20240220', '20240229', '20240926']
        self.hand_constraint_index_list = [[5, 6, 7, 8], [9, 10, 11, 12],
                                           [13, 14, 15, 16], [17, 18, 19, 20]]
        self.enhance_static = enhance_static

    def independent_part1(self, feat_x, feat_y, left_hand):
        pred_x, pred_y = self.ipr_module(feat_x, feat_y)
        pred_x *= 128
        pred_y *= 128
        mask = left_hand == 1
        pred_x[mask] = 127 - pred_x[mask]
        return pred_x, pred_y

    def independent_part2(self, sigma):
        weight_num = 42
        kpt_weight = torch.eye(weight_num).unsqueeze(0).to(sigma.device)

        sigma_kpt = torch.mean(sigma, dim=-1)
        sigma_kpt_softmax = torch.softmax(sigma_kpt, dim=1)
        sigma_kpt_softmax = sigma_kpt_softmax.unsqueeze(2).repeat(
            1, 1, 2).view(sigma_kpt.shape[0], -1)
        indices = torch.arange(weight_num)
        kpt_weight[:, indices, indices] = sigma_kpt_softmax * 21
        return kpt_weight

    def indepentent_part3(self, feat_x, feat_y, rot, svd_pt, mems, score,
                          left_hand):
        matrix_svd = decode_svd(svd_pt, self.rigid_samples)
        pre_root_matrix = matrix_svd[:, 0:3, 0:3]

        mask = left_hand == 1
        add_matrix = torch.eye(3).unsqueeze(0).expand(1, -1, -1).to('cuda')
        add_matrix[mask, 0, 0] = -add_matrix[mask, 0, 0]
        shape_vector = torch.zeros((1, 1)).to('cuda')

        pre_local_matrix = rot9D_to_matirx(rot.reshape(-1,
                                                       9)).reshape(1, 19, -1)
        pre_root_matrix = torch.matmul(add_matrix, pre_root_matrix)

        _, bone_joints = self.nimble_layer_predict.forward_simple(
            pre_local_matrix, shape_vector)
        rebuild_joints = bone_joints[:, self.kp_index, :]
        root_rebuild_joints = rebuild_joints[:, 0:1, :]
        rebuild_joints_temp = rebuild_joints - root_rebuild_joints
        rebuild_joints_temp = torch.matmul(rebuild_joints_temp,
                                           pre_root_matrix.transpose(1, 2))
        hand3d_wo_root = rebuild_joints_temp / self.scale_parameter
        return hand3d_wo_root

    def indepentent_part4(self, hand3d_rel, cood_2d, intrix_matrix, W):

        batch_size, K = hand3d_rel.shape[0], hand3d_rel.shape[1]
        cuda_device = cood_2d.device
        cood_2d = torch.concat(
            (cood_2d, torch.ones(batch_size, K, 1).to(cuda_device)), dim=-1)
        uv_cood_leftmatrix = torch.matmul(
            torch.inverse(intrix_matrix),
            cood_2d.permute(0, 2, 1)).permute(0, 2, 1)[..., :2].to(cuda_device)

        A = torch.zeros((batch_size, 2 * K, 3), device=cuda_device)
        A[:, ::2, 0] = -1
        A[:, 1::2, 1] = -1
        A[:, ::2, 2] = uv_cood_leftmatrix[:, :, 0].view(batch_size, K)
        A[:, 1::2, 2] = uv_cood_leftmatrix[:, :, 1].view(batch_size, K)

        B = torch.zeros((batch_size, 2 * K, 1), device=cuda_device)
        B[:, ::2,
          0] = hand3d_rel[:, :,
                          0] - hand3d_rel[:, :, 2] * uv_cood_leftmatrix[:, :,
                                                                        0]
        B[:, 1::2,
          0] = hand3d_rel[:, :,
                          1] - hand3d_rel[:, :, 2] * uv_cood_leftmatrix[:, :,
                                                                        1]

        part_1 = torch.inverse(
            torch.matmul(
                torch.matmul(torch.matmul(A.permute(0, 2, 1), W), W), A))
        part_2 = torch.matmul(
            torch.matmul(torch.matmul(A.permute(0, 2, 1), W), W), B)
        result = torch.matmul(part_1, part_2).permute(0, 2, 1)

        hand3d = hand3d_rel + result
        return hand3d

    def _forward(self,
                 feats: Tuple[Tensor],
                 f_scale: Tensor,
                 mems=None) -> Tuple[Tensor, Tensor]:
        feat_x, feat_y = RTMCCHead.forward(self, feats)
        pose_len = self.pose_num

        raw_feats = feats[-1]
        image_fea = self.proj_layer(raw_feats)
        ftl_image_fea = self.trans_feat(image_fea, f_scale[:, 0, :, :])
        feature_fuszion = self.liftnet(ftl_image_fea.reshape(1, -1, 1, 1))
        if mems is None:
            mems = torch.zeros_like(feature_fuszion).to(feature_fuszion.device)
        feat_mix = torch.cat([feature_fuszion, mems], dim=1)
        mems = self.temporal(feat_mix)
        output = self.nimble_last_layer(feat_mix)
        rot = output[:, :pose_len, 0, 0].reshape(1, 19, -1)
        svd_pt = output[:, pose_len:, 0, 0]

        pred_sigma = self.sigma_conv(image_fea.reshape(1, -1, 1, 1))
        pred_sigma_reshape = pred_sigma.reshape(
            pred_sigma.size(0), self.out_channels, 3)
        score = pred_sigma.sigmoid().mean().reshape(1, 1)
        return feat_x, feat_y, rot, svd_pt, mems, score, pred_sigma_reshape

    def _forward_1(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        feat_x, feat_y = RTMCCHead.forward(self, feats)
        return feat_x, feat_y, feats[-1]

    def _forward_2(self,
                   feats: Tuple[Tensor],
                   f_scale: Tensor,
                   mems=None) -> Tuple[Tensor, Tensor]:
        raw_feats = feats
        pose_len = self.pose_num
        image_fea = self.proj_layer(raw_feats)
        ftl_image_fea = self.trans_feat(image_fea, f_scale[:, 0, 0, 0])
        feature_fuszion = self.liftnet(ftl_image_fea.reshape(1, -1, 1, 1))
        if mems is None:
            mems = torch.zeros_like(feature_fuszion).to(feature_fuszion.device)
        feat_mix = torch.cat([feature_fuszion, mems], dim=1)
        mems = self.temporal(feat_mix)
        output = self.nimble_last_layer(feat_mix)
        rot = output[:, :pose_len, 0, 0].reshape(1, 19, -1)
        svd_pt = output[:, pose_len:, 0, 0]

        pred_sigma = self.sigma_conv(image_fea.reshape(1, -1, 1, 1))
        sigma = pred_sigma.reshape(pred_sigma.size(0), self.out_channels, 3)
        score = pred_sigma.sigmoid().mean().reshape(1, 1)
        return rot, svd_pt, mems, score, sigma

    def forward(self,
                feats: Tuple[Tensor],
                f_scale: Tensor,
                mems=None,
                seq_len: int = 1) -> Tuple[Tensor, Tensor]:
        feat_x, feat_y = RTMCCHead.forward(self, feats)
        # heatmaps = torch.cat([feat_x, feat_y], dim=1)
        raw_feats = feats[-1]
        pred_x, pred_y = self.ipr_module(feat_x, feat_y)
        output_2d = torch.cat([pred_x, pred_y], dim=-1) * 128

        B_ori = raw_feats.shape[0]
        B = int(B_ori / seq_len)
        image_fea = self.proj_layer(raw_feats)
        ftl_image_fea = self.trans_feat(image_fea, f_scale[:, 0, 0, 0])
        feature_fuszion = self.liftnet(ftl_image_fea).reshape(B_ori, -1, 1, 1)

        if mems is None:
            mems = torch.zeros(B, feature_fuszion.shape[1], 1, 1).cuda()
        feature_fuszion = feature_fuszion.view(B, seq_len, -1)
        outputs = torch.zeros(
            (B, seq_len, self.nimble_output_num, 1, 1)).cuda()
        for i in range(seq_len):
            feat = feature_fuszion[:, i:i + 1, :].reshape(B, -1, 1, 1)
            feat_mix = torch.cat([feat, mems], dim=1)
            mems = self.temporal(feat_mix)
            output = self.nimble_last_layer(feat_mix)
            outputs[:, i, ...] = output
        outputs = outputs.reshape(B * seq_len, -1, 1, 1)

        if self.output_sigma:
            # x = self.gap(raw_feats)
            pred_sigma = self.sigma_conv(image_fea.reshape(B_ori, -1, 1, 1))
            pred_sigma_reshape = pred_sigma.reshape(
                pred_sigma.size(0), self.out_channels, 3)

            return outputs, output_2d, mems, pred_sigma_reshape

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                test_cfg: ConfigType = {}):
        left_R = []
        is_left_hands = []
        hand3d_gt = []
        hand2d_gt = []
        intrix_matrix = []
        f_scale = []

        for i, data in enumerate(batch_data_samples):
            keypoint_label = data.gt_instance_labels.keypoint_labels
            camera_model = data.meta['virtual_camera']
            keypoint_2d_lable = data.gt_instances.keypoints[:, :, :2]
            # keypoint_2d_lable = camera_model.undistort(keypoint_2d_lable)
            f_scale.append(camera_model.f[0] / self.f_standard)
            intrix_m = np.array([[camera_model.f[0], 0, camera_model.c[0]],
                                 [0, camera_model.f[1], camera_model.c[1]],
                                 [0, 0, 1]])

            if keypoint_label.shape[-1] == 3:
                if data.meta['category_id'] == 1:
                    is_left_hands.append(1)
                else:
                    is_left_hands.append(0)
            if 'virtual_camera' in data.meta:
                virtual_cam = data.meta['virtual_camera']
                left_R.append(
                    np.linalg.inv(virtual_cam.camera_to_world_xf[:3, :3]))
            hand3d_gt.append(data.gt_instances.keypoints3d[0])
            hand2d_gt.append(keypoint_2d_lable)
            intrix_matrix.append(intrix_m)

        left_R = torch.tensor(np.array(left_R)).cuda().float()
        left_hand = torch.tensor(np.array(is_left_hands)).cuda().float()
        intrix_matrix = torch.tensor(np.array(intrix_matrix)).cuda().float()
        # intrix_fea = intrix_matrix.reshape(left_R.shape[0], -1)
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
        hand2d_gt = torch.tensor(np.array(hand2d_gt)).cuda().float()
        f_scale = torch.tensor(np.array(f_scale)).cuda().float()[:, None, None,
                                                                 None]

        nimble_output, output_2d, mems, sigma = self.forward(
            feats, f_scale, mems, 1)  # (B, K, D)

        batch_size = sigma.shape[0]
        weight_num = sigma.shape[1] * 2
        kpt_weight = torch.eye(weight_num).unsqueeze(0).repeat(
            batch_size, 1, 1).to(sigma.device)

        sigma_kpt = torch.mean(sigma, dim=-1)
        sigma_kpt_softmax = torch.softmax(sigma_kpt, dim=1)
        sigma_kpt_softmax = sigma_kpt_softmax.unsqueeze(2).repeat(
            1, 1, 2).view(sigma_kpt.shape[0], -1)
        indices = torch.arange(weight_num)
        kpt_weight[:, indices, indices] = sigma_kpt_softmax * 21

        hand3d_pred = self.decode_nimble_fun(nimble_output, left_R, None,
                                             left_hand, None,
                                             f_scale[:, 0, 0, 0], output_2d,
                                             intrix_matrix, kpt_weight, True)

        #  修改为对应的_forward结果
        # feat_x, feat_y, feats = self._forward_1(feats)
        # rot, svd_pt, mems, score, sigma = self._forward_2(feats, f_scale, mems)

        # feat_x, feat_y, rot, svd_pt, mems, score, sigma = self._forward(
        #     feats, f_scale, mems)
        # pred_x, pred_y = self.independent_part1(feat_x, feat_y, left_hand)
        # cood_2d = torch.cat([pred_x, pred_y], dim=-1)
        # kpt_weight = self.independent_part2(sigma)
        # hand3d_wo_root = self.indepentent_part3(feat_x, feat_y, rot, svd_pt, mems, score, left_hand)
        # hand3d_pred = self.indepentent_part4(hand3d_wo_root, cood_2d, intrix_matrix, kpt_weight)
        # hand3d_pred = torch.matmul(
        #     torch.inverse(left_R),
        #     hand3d_pred.permute(0, 2, 1)).permute(0, 2, 1)

        # 模型直出的2d结果
        # result = []
        # for output_2d_sin, batch_data_sample in zip(output_2d,
        #                                          batch_data_samples):
        #     ori_cam = batch_data_sample.meta['ori_camera']
        #     virtual_cam = batch_data_sample.meta['virtual_camera']
        #     virtual_3d = virtual_cam.window_to_eye(output_2d_sin.detach().cpu().numpy())
        #     world_3d = virtual_cam.eye_to_world(virtual_3d)
        #     result.append(ori_cam.eye_to_window(world_3d))

        # 重投影后的2d结果
        result = []
        for hand3d_sin, batch_data_sample in zip(hand3d_pred,
                                                 batch_data_samples):
            ori_cam = batch_data_sample.meta['ori_camera']
            result.append(ori_cam.eye_to_window(hand3d_sin.cpu().numpy()))

        uv_reproj = torch.tensor(np.array(result)).cuda()

        return hand3d_pred, uv_reproj, mems, sigma

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        label_2d_list = []
        label_depth_list = []
        label_depth_id_list = []
        nimble_pose = []
        nimble_trans = []
        nimble_shape = []
        hand3d_gt = []
        hand2d_gt = []
        hand3d_gt_pcl = []
        hand2d_gt_pcl = []
        intrix_matrix = []
        left_R = []
        is_left_hands = []
        f_scale = []
        external_matrix = []
        nimble_info = dict()
        nimble_lable_exist = []
        use_3d_supervise_mask = []
        
        for i, data in enumerate(batch_data_samples):
            if 'ume' in data.img_path or 'hand_train_flora' in data.img_path:
                use_3d_supervise_mask.append(False)
            else:
                use_3d_supervise_mask.append(True)

            keypoint_label = data.gt_instance_labels.keypoint_labels
            label_2d_list.append(keypoint_label[..., :2])

            label_depth_list.append(keypoint_label[..., 2:3])
            label_depth_id_list.append(i)
            hand3d_gt.append(data.gt_instances.keypoints3d[0])
            if data.meta['category_id'] == 1:
                is_left_hands.append(1)
            else:
                is_left_hands.append(0)
            if data.meta['camera_name'] == 'right':
                external_matrix.append(data.meta['external'])
            else:
                external_matrix.append(np.eye(4))

            keypoint_2d_lable = data.gt_instances.keypoints[:, :, :2]
            camera_model = data.meta['ori_camera']
            vritual_camera = data.meta['virtual_camera']
            keypoint_2d_lable = camera_model.undistort(keypoint_2d_lable)
            intrix_m = np.array([[vritual_camera.f[0], 0, vritual_camera.c[0]],
                                 [0, vritual_camera.f[1], vritual_camera.c[1]],
                                 [0, 0, 1]])
            f_scale.append(vritual_camera.f[0] / self.f_standard)
            hand2d_gt.append(keypoint_2d_lable)  # 仅使用pcl2d，此项暂未使用
            intrix_matrix.append(intrix_m)

            if 'nimble_pose' not in data.meta or data.meta[
                    'nimble_pose'].shape == ():
                nimble_lable_exist.append(False)
                nimble_pose.append(np.zeros((20, 3)))
                nimble_trans.append(np.zeros(3))
                nimble_shape.append(np.zeros(20))
            else:
                nimble_lable_exist.append(True)
                nimble_pose.append(data.meta['nimble_pose'])
                nimble_trans.append(data.meta['nimble_translation'])
                nimble_shape.append(data.meta['nimble_shape'])

            virtual_cam = data.meta['virtual_camera']
            left_R.append(
                np.linalg.inv(virtual_cam.camera_to_world_xf[:3, :3]))
            hand3d_pcl_sin = virtual_cam.world_to_eye(
                data.gt_instances.keypoints3d[0])
            hand3d_gt_pcl.append(hand3d_pcl_sin)
            kpt3d_tmp = camera_model.window_to_eye(
                data.gt_instances.keypoints[0, :, :2])
            hand2d_gt_pcl.append(vritual_camera.world_to_window(kpt3d_tmp))

        left_R = torch.tensor(np.array(left_R)).cuda().float()
        left_hand = torch.tensor(np.array(is_left_hands)).cuda().float()
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
        hand2d_gt = torch.tensor(np.array(hand2d_gt)).cuda().float()
        hand2d_gt_pcl = torch.tensor(np.array(hand2d_gt_pcl)).cuda().float()
        hand3d_gt_pcl = torch.tensor(np.array(hand3d_gt_pcl)).cuda().float()
        f_scale = torch.tensor(np.array(f_scale)).cuda().float()[:, None, None,
                                                                 None]
        intrix_matrix = torch.tensor(np.array(intrix_matrix)).cuda().float()
        external_matrix = torch.tensor(
            np.array(external_matrix)).cuda().float()
        nimble_pose = torch.tensor(np.array(nimble_pose)).cuda().float()
        nibmle_root_matrix = batch_rodrigues(nimble_pose[:, 0, :]).reshape(
            -1, 3, 3)
        nibmle_root_matrix = torch.matmul(external_matrix[:, :3, :3],
                                          nibmle_root_matrix)

        nimble_trans = torch.tensor(np.array(nimble_trans)).cuda().float()
        nimble_trans = torch.matmul(
            external_matrix,
            torch.concat((nimble_trans, torch.ones(nimble_trans.shape[0],
                                                   1).to(nimble_trans.device)),
                         dim=1).unsqueeze(-1))[:, :3, 0]
        use_3d_supervise_mask = torch.tensor(use_3d_supervise_mask).cuda()
        nimble_lable_exist = torch.tensor(nimble_lable_exist).cuda()
        nimble_lable_exist = use_3d_supervise_mask & nimble_lable_exist

        nimble_info = {
            'nibmle_root_matrix': nibmle_root_matrix,
            'nimble_pose': nimble_pose,
            'nimble_trans': nimble_trans,
            'nimble_shape':
            torch.tensor(np.array(nimble_shape)).cuda().float(),
            'nimble_lable_exist': nimble_lable_exist
        }

        nimble_output, output_2d, mems, sigma = self.forward(
            inputs, f_scale, None, self.seq_len)

        mh = MessageHub.get_current_instance()
        cur_epoch = mh.get_info('epoch')

        batch_size = sigma.shape[0]
        weight_num = sigma.shape[1] * 2
        kpt_weight = torch.eye(weight_num).unsqueeze(0).repeat(
            batch_size, 1, 1).to(sigma.device)
        if cur_epoch > 100:
            sigma_kpt = torch.mean(sigma, dim=-1)
            sigma_kpt_softmax = torch.softmax(sigma_kpt, dim=1)
            sigma_kpt_softmax = sigma_kpt_softmax.unsqueeze(2).repeat(
                1, 1, 2).view(sigma_kpt.shape[0], -1)
            indices = torch.arange(weight_num)
            kpt_weight[:, indices, indices] = sigma_kpt_softmax * 21

        hand2d_gt_pcl = hand2d_gt_pcl / 128
        mask = left_hand == 1
        hand2d_pred_pcl = output_2d.clone()
        hand2d_pred_pcl[mask, :, 0] = (127 - hand2d_pred_pcl[mask, :, 0])
        hand2d_pred_pcl_direct = hand2d_pred_pcl / 128

        (pred_3d_way1, pred_3d_way2, hand3d_pred_total, hand3d_part_gt,
         pre_trans_xyz,
         gt_trans_xyz) = self.decode_nimble_fun(nimble_output, left_R,
                                                nimble_info, left_hand,
                                                hand3d_gt, f_scale[:, 0, 0, 0],
                                                output_2d, intrix_matrix,
                                                kpt_weight, False)

        hand3d_gt_xreal = hand3d_gt[nimble_lable_exist]
        hand3d_gt = hand3d_gt[use_3d_supervise_mask]
        pred_3d_way1 = pred_3d_way1[nimble_lable_exist]
        pred_3d_way2 = pred_3d_way2[nimble_lable_exist]
        hand3d_pred = hand3d_pred_total[use_3d_supervise_mask]
        hand3d_part_gt = hand3d_part_gt[nimble_lable_exist]
        sigma = sigma[use_3d_supervise_mask]

        hand3d_pred_pcl = torch.matmul(left_R,
                                       hand3d_pred_total.permute(0, 2,
                                                                 1)).permute(
                                                                     0, 2, 1)
        hand3d_pred_rep = torch.matmul(intrix_matrix,
                                       hand3d_pred_pcl.permute(0, 2,
                                                               1)).permute(
                                                                   0, 2, 1)
        hand2d_pred_pcl_reproject = (hand3d_pred_rep /
                                     hand3d_pred_rep[:, :, -1:])[:, :, :2]
        hand2d_pred_pcl_reproject = hand2d_pred_pcl_reproject / 128

        # 直接监督rot和trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz[use_3d_supervise_mask]
        gt_nimble_trans = hand3d_gt[:, 0, :]

        # pinch 损失
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(hand3d_gt[:, 4, :] - hand3d_gt[:, 8, :], dim=-1)

        re_all_sigmas = torch.cat((hand3d_pred, sigma), dim=-1)

        if self.enhance_static:
            static_weight = 25
            static_mask = self.generate_mask(
                batch_data_samples,
                self.static_data_date_list).to('cuda')[use_3d_supervise_mask]
            enhanced_static_hand3d_pred = self.enhanced_fun(
                hand3d_pred, static_mask, static_weight)
            enhanced_static_hand3d_gt = self.enhanced_fun(
                hand3d_gt, static_mask, static_weight)
        else:
            enhanced_static_hand3d_pred = hand3d_pred
            enhanced_static_hand3d_gt = hand3d_gt

        pred_for_loss = [
            pred_3d_way1, pred_3d_way2, hand3d_pred, dist_pred,
            pre_nimble_trans, re_all_sigmas, hand2d_pred_pcl_direct,
            enhanced_static_hand3d_pred
        ]
        targ_for_loss = [
            hand3d_gt_xreal, hand3d_gt_xreal, hand3d_gt, dist_gt,
            gt_nimble_trans, hand3d_gt, hand2d_gt_pcl,
            enhanced_static_hand3d_gt
        ]

        weight_ini = torch.ones((1, 21, 3))
        weight_ini[0, :9, :] = 2
        weight_ini[0, 4, :], weight_ini[0, 8, :] = 4, 4
        weight_ini_ori = weight_ini.repeat(hand3d_gt.shape[0], 1,
                                           1).to(hand3d_gt.device)
        weight_ini_part = weight_ini.repeat(pred_3d_way1.shape[0], 1,
                                            1).to(pred_3d_way1.device)
        weight_for_loss = [
            weight_ini_part, weight_ini_part, weight_ini_ori, None, None, None,
            None, None
        ]

        losses = self.loss_module(pred_for_loss, targ_for_loss,
                                  weight_for_loss)
        (loss_pre_root, loss_pre_nimble, loss_pre_all, loss_pinch,
         loss_nimble_trans, loss_rle_all, loss_2d_direct, loss_smooth) = losses
        # # 子骨骼向量监督
        bone_loss_weight = 0.1
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
        major_bone_loss_weight = 0.6
        local_bone_3d_pre = (
            pred_3d_way2 -
            pred_3d_way2[:, self.joint_parents, :])[:, self.non_root_indices]
        local_bone_3d_pre = local_bone_3d_pre[:, :8, :].reshape(-1, 3)
        local_bone_3d_gt = (
            hand3d_part_gt -
            hand3d_part_gt[:, self.joint_parents, :])[:, self.non_root_indices]
        local_bone_3d_gt = local_bone_3d_gt[:, :8, :].reshape(-1, 3)

        local_bone_3d_pre_vector = self.cal_normalize_vector(local_bone_3d_pre)
        local_bone_3d_gt_vector = self.cal_normalize_vector(local_bone_3d_gt)

        local_squared_diff = (local_bone_3d_pre_vector -
                              local_bone_3d_gt_vector)**2
        major_bone_loss = torch.mean(torch.sum(local_squared_diff,
                                               dim=1)) * major_bone_loss_weight

        pinch_mask = (dist_pred > dist_gt - 0.001) & (dist_gt < 0.03)
        mh = MessageHub.get_current_instance()
        cur_epoch = mh.get_info('epoch')
        reverse_mask = self.generate_mask(
            batch_data_samples, self.reverse_pinch_date_list).to(
                pinch_mask.cuda())[use_3d_supervise_mask]
        pinch_reverse_mask = pinch_mask & reverse_mask
        if cur_epoch > 30 and sum(pinch_mask) > 0:
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

        if cur_epoch > 35:
            loss_2d_reproject = self.reproject_2d_loss(
                hand2d_pred_pcl_reproject, hand2d_pred_pcl_direct.detach())
        else:
            loss_2d_reproject = torch.tensor(0.0, device=hand2d_gt_pcl.device)

        # 手型约束
        hand_constraint_loss = self.hand_constraint(
            hand3d_pred_total, self.hand_constraint_index_list) * 0.02

        plam_ratio = torch.norm(
            hand3d_gt[:, 9, :] - hand3d_gt[:, 0, :], dim=-1) / 0.08
        standard_hand3d_gt = hand3d_gt / plam_ratio[:, None, None]
        dis = (standard_hand3d_gt[:, 8, :] +
               standard_hand3d_gt[:, 5, :]) / 2 - standard_hand3d_gt[:, 6, :]
        dis_norm = torch.norm(dis, dim=1)
        poke_mask = dis_norm < 0.012
        if cur_epoch > 30 and sum(poke_mask) > 0:
            direction_vector1 = hand3d_pred[poke_mask,
                                            5, :] - hand3d_pred[poke_mask,
                                                                6, :]
            direction_vector2 = hand3d_pred[poke_mask,
                                            6, :] - hand3d_pred[poke_mask,
                                                                8, :]
            vector1_norm = F.normalize(direction_vector1, dim=1)
            vector2_norm = F.normalize(direction_vector2, dim=1)
            cosine_similarity = (vector1_norm * vector2_norm).sum(dim=1)
            loss_poke = (1 - torch.abs(cosine_similarity)).mean() / 5
        else:
            loss_poke = torch.tensor(0.0, device=hand3d_gt.device)

        losses = dict(
            # loss_kpt2d=loss_kpt2d,
            # label_depth=loss_depth,
            loss_pre_root=loss_pre_root,
            loss_pre_nimble=loss_pre_nimble,
            loss_pre_all=loss_pre_all,
            bone_loss=bone_loss,
            major_bone_loss=major_bone_loss,
            loss_pinch=loss_pinch,
            loss_nimble_trans=loss_nimble_trans,
            loss_rle_all=loss_rle_all,
            loss_2d_direct=loss_2d_direct,
            loss_2d_reproject=loss_2d_reproject,
            loss_smooth=loss_smooth,
            pinch_loss_add=pinch_loss_add,
            hand_constraint_loss=hand_constraint_loss,
            loss_poke=loss_poke)

        # 如有nan loss 则置为0
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    losses[key] = torch.tensor(0.0, device=value.device)
            else:
                if value != value or value == float('inf') or value == float(
                        '-inf'):
                    losses[key] = 0.0

        # calculate 3d metric
        def cal_mpjpe(pre_kpt, gt_kpt):
            error = np.linalg.norm(
                pre_kpt - gt_kpt, ord=2, axis=-1).mean() * 1000
            return error

        mpjpe_value = cal_mpjpe(hand3d_gt.cpu().numpy(),
                                hand3d_pred.detach().cpu().numpy())
        mpjpe_value = torch.tensor(mpjpe_value).cuda()
        losses.update(mpjpe_value=mpjpe_value)

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to load weights of deconv layers from
        :class:`HeatmapHead` into `simplebaseline_head`.

        The hook will be automatically registered during initialization.
        """

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            # convert fc to conv
            if _k == 'head.sigma_fc.weight':
                state_dict['head.sigma_conv.weight'] = torch.unsqueeze(
                    torch.unsqueeze(v, -1), -1)
            if _k == 'head.sigma_fc.bias':
                state_dict['head.sigma_conv.bias'] = v

    def cal_normalize_vector(self, vector):
        vector_norms = torch.sqrt(
            torch.sum(vector**2, dim=1, keepdim=True) + 1e-8)
        normalized_vector = vector / vector_norms
        return normalized_vector

    def enhanced_fun(self, kpt, mask, weight):
        enhanced_kpt = kpt.clone()
        enhanced_kpt[mask] = enhanced_kpt[mask] * weight
        return enhanced_kpt

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
