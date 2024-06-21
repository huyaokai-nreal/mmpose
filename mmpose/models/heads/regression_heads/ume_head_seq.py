# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from mmpose.models.heads.regression_heads.ume_head import (
    UmeHead, apply_ftl_to_feature_maps)
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


class SimpleConvRNN(nn.Module):

    def __init__(
        self,
        nTemporalBlocks: int,
        nTemporalMemoryChannels: int,
        nImageFeatureChannels: int,
        temporalFTLRatio: float,
        featureMapShape: Tuple[int, int],
    ) -> None:
        super(SimpleConvRNN, self).__init__()
        self._nc_memory = nTemporalMemoryChannels
        n_temporal_channels = nImageFeatureChannels + self._nc_memory

        temporal_module = nn.ModuleList()

        for i in range(nTemporalBlocks):
            nc = n_temporal_channels
            temporal_module.append(nn.Conv2d(nc, nc, kernel_size=1, padding=0))
            if i != nTemporalBlocks - 1:
                temporal_module.append(nn.ReLU())

        self._temporal_module = nn.Sequential(*temporal_module)
        self._temporal_ftl_ratio = float(temporalFTLRatio)
        self._input_shape = (n_temporal_channels, *featureMapShape)

    def input_shape(self) -> Tuple[int, int, int]:
        """
        Return: input shape to self._temporal_module
            [channels, feature_size[0], feature_size[1]]
        """
        return self._input_shape

    def transform_memory_features(self, prev_mem_features, cur_extrinsics,
                                  prev_extrinsics):
        if prev_extrinsics is not None:
            prev_cam0_to_world_xf = torch.inverse(prev_extrinsics)
            # import ipdb;ipdb.set_trace()
            prev_cam0_to_cur_cam0_xf = (cur_extrinsics @ prev_cam0_to_world_xf)
            prev_mem_features = apply_ftl_to_feature_maps(
                prev_cam0_to_cur_cam0_xf,
                prev_mem_features,
                self._temporal_ftl_ratio,
            )
        # Update prev_extrinsics with cur_extrinsics
        prev_extrinsics = cur_extrinsics
        return prev_mem_features

    def forward_one_step(self, mem_features, cur_fused_features):
        # import ipdb;ipdb.set_trace()
        temporal_out = torch.cat((mem_features, cur_fused_features), dim=1)
        temporal_out = self._temporal_module(temporal_out)

        mem_features_out = temporal_out[:, 0:self._nc_memory].clone()
        fused_features = temporal_out[:, self._nc_memory:].clone()

        return mem_features_out, fused_features

    def forward_tem_feat(self, prev_mem_features, cur_fused_features,
                         cur_extrinsics, prev_extrinsics):
        # 通过前后帧的外参，转换前一帧3D特征到当前帧的位姿下
        mem_features = self.transform_memory_features(prev_mem_features,
                                                      cur_extrinsics,
                                                      prev_extrinsics)

        # 模型融合前后帧3D特征
        mem_features_out, temporal_features = self.forward_one_step(
            mem_features, cur_fused_features)

        return temporal_features, mem_features_out

    def reset_mem_features(self):
        self._mem_features = torch.empty(0)
        self._mem_features_multiv = torch.empty(0)
        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics = torch.empty(0)
        self._prev_extrinsics_multiv = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)


@MODELS.register_module()
class UmeHeadSeq(UmeHead):

    def __init__(self,
                 ume_loss: ConfigType,
                 feat_channel: int = 6,
                 mem_channel: int = 18,
                 feature_map_shape: tuple = (8, 8),
                 shape_ncomp: int = 20,
                 pose_ncomp: int = 60,
                 seq_len: int = 4,
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
                 use_scaled_as_canonical: bool = True,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(
            ume_loss=ume_loss,
            feat_channel=feat_channel,
            feature_map_shape=feature_map_shape,
            shape_ncomp=shape_ncomp,
            pose_ncomp=pose_ncomp,
            use_pose_pca=use_pose_pca,
            reg_shape_type=reg_shape_type,
            use_nimble_part_para=use_nimble_part_para,
            use_6d_pose_reg=use_6d_pose_reg,
            use_9d_pose_reg=use_9d_pose_reg,
            direct_pose_reg=direct_pose_reg,
            use_bone_loss=use_bone_loss,
            enhance_lefthand=enhance_lefthand,
            enhance_static=enhance_static,
            use_svd=use_svd,
            use_gmlp=use_gmlp,
            use_scaled_as_canonical=use_scaled_as_canonical,
            init_cfg=init_cfg)
        self.seq_len = seq_len
        self.use_gmlp = use_gmlp
        self.temporal = SimpleConvRNN(
            nTemporalBlocks=3,
            nTemporalMemoryChannels=mem_channel,
            nImageFeatureChannels=feat_channel,
            temporalFTLRatio=1.0,
            featureMapShape=feature_map_shape,
        )

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(batch_data_samples)

        # forward
        output, sigma, mems = self.forward(
            feats, data['cam_f'], data['cam_xf'], mems, seq_len=1)
        hand3d_pred = self.postprocess(
            output,
            data['nimble_info'],
            data['left_hand'],
            data['cam_xf'],
            only_pre=True)[0]

        hand3d_pred = hand3d_pred.cpu().numpy()
        leftcam_uv_distort = []
        for i, left_sample in enumerate(batch_data_samples[::2]):
            _leftcam_uv_distort = left_sample.meta['ori_camera'].eye_to_window(
                hand3d_pred[i])
            leftcam_uv_distort.append(_leftcam_uv_distort)
        leftcam_uv_distort = np.stack(leftcam_uv_distort, axis=0)
        return hand3d_pred, leftcam_uv_distort, mems, sigma

    def forward(self, feats, cam_f, cam_xf, mem, seq_len) -> dict:
        B = int(feats.shape[0] / seq_len) // 2
        if mem is None:
            mem = torch.zeros(
                B,
                self.temporal._nc_memory,
                self.temporal._input_shape[1],
                self.temporal._input_shape[2],
                dtype=feats.dtype,
                device=feats.device,
            )
        prev_extrinsics = None
        feats = self.proj_layer(feats)
        feats = feats.view((2 * B, seq_len) + feats.shape[-3:])
        outputs = torch.zeros(
            (B, seq_len, self.output_num, 1, 1)).to(feats.device)
        sigmas = torch.zeros((B, seq_len, 21, 3)).to(feats.device)
        for i in range(seq_len):
            feat = feats[:, i, :, :, :]
            _cam_f = cam_f[i * B:(i + 1) * B].clone()
            _cam_xf = cam_xf[i * B:(i + 1) * B].clone()
            # fuse multiv feat
            _cur_fused_features = self.forward_feature_fuse(
                feat, _cam_f, _cam_xf)
            # temporal module
            temporal_features, mem_features = self.temporal.forward_tem_feat(
                mem, _cur_fused_features, _cam_xf[:, 0], prev_extrinsics)
            # record the previous frame information and results
            mem = mem_features
            prev_extrinsics = _cam_xf[:, 0]
            # import ipdb;ipdb.set_trace()
            temporal_features = temporal_features.view(B, -1, 1, 1)
            if self.use_gmlp:
                temporal_features = self.gmlp(temporal_features)
            sigma = self.sigma_conv(temporal_features).reshape(B, 21, 3)
            output = self.last_layer(temporal_features)
            outputs[:, i, ...] = output
            sigmas[:, i, ...] = sigma

        outputs = outputs.reshape(B * seq_len, -1, 1, 1)
        sigmas = sigmas.reshape(B * seq_len, 21, 3)
        return outputs, sigmas, mem

    def loss(self, feats, batch_data_samples) -> dict:
        with torch.no_grad():
            data = self.preprocess(batch_data_samples)
        # forward
        output, sigma, _ = self.forward(
            feats, data['cam_f'], data['cam_xf'], None, seq_len=self.seq_len)

        (pred_3d_way1, pred_3d_way2, hand3d_pred, hand3d_part_gt,
         pre_trans_xyz, pre_shape, gt_all_matrix,
         pre_all_matrix) = self.postprocess(output, data['nimble_info'],
                                            data['left_hand'], data['cam_xf'])

        # 直接监督rot和trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz
        gt_nimble_trans = data['nimble_info']['nimble_trans']

        # pinch 损失
        hand3d_gt = data['hand3d_gt']
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

        if self.enhance_static:
            static_weight = 25
            static_mask = self.generate_static_mask(batch_data_samples)
            enhanced_static_hand3d_pred = self.enhanced_fun(
                hand3d_pred, static_mask, static_weight)
            enhanced_static_hand3d_gt = self.enhanced_fun(
                hand3d_gt, static_mask, static_weight)
        else:
            enhanced_static_hand3d_pred = hand3d_pred
            enhanced_static_hand3d_gt = hand3d_gt

        # sigma for RLELoss
        re_all_sigmas = torch.cat((enhanced_left_hand3d_pred, sigma), dim=-1)

        # 归一化平面2d 重投影
        norm_2d_pred = enhanced_left_hand3d_pred / enhanced_left_hand3d_pred[
            ..., 2:]
        norm_2d_gt = enhanced_left_hand3d_gt / enhanced_left_hand3d_gt[..., 2:]

        pred_for_loss = [
            enhanced_left_pred_3d_way1, enhanced_left_pred_3d_way2,
            enhanced_left_hand3d_pred, dist_pred, pre_nimble_trans,
            enhanced_static_hand3d_pred, re_all_sigmas, norm_2d_pred
        ]
        targ_for_loss = [
            enhanced_left_hand3d_gt, enhanced_left_hand3d_gt,
            enhanced_left_hand3d_gt, dist_gt, gt_nimble_trans,
            enhanced_static_hand3d_gt, hand3d_gt, norm_2d_gt
        ]

        weight_ini = torch.ones((hand3d_gt.shape[0], 21, 3),
                                device=hand3d_gt.device)
        weight_ini[:, :9, :] = 2
        weight_ini[:, 4, :], weight_ini[:, 8, :] = 4, 4
        weight_for_loss = [
            weight_ini,
            weight_ini,
            weight_ini,
            None,
            None,
            None,
            None,
            weight_ini,
        ]
        losses = self.ume_loss(pred_for_loss, targ_for_loss, weight_for_loss)
        (loss_pre_root, loss_pre_nimble, loss_pre_all, loss_pinch,
         loss_nimble_trans, loss_smooth, loss_rle_all, loss_2d) = losses

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
            loss_smooth=loss_smooth,
            loss_rle_all=loss_rle_all,
            loss_2d=loss_2d)
        return losses_dict
