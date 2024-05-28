# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmpose.models.heads.regression_heads.temporal_lift_head_rot_standard import \
    TemporalLiftNimbleHeadStandard
from mmpose.models.losses.regression_loss import L1Loss, RLELoss
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions
from mmpose.models.heads.nimble.smoothnet import MotionSmoothNet, SmoothNet


@MODELS.register_module()
class TemporalLiftNimbleHeadStandardPredict(TemporalLiftNimbleHeadStandard):
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
                 use_pose_pca: bool = True,
                 reproj: bool = False,
                 baseline=0.13,
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
                 seq_len: int = 4,
                 enhance_lefthand=True,
                 enhance_static=True,
                 predict_frame: int = 1,
                 predict_way = "SmoothNet",
                 smooth_window_len=8,
                 fix_sigma_pars=False,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(
            lift_loss=lift_loss,
            d_ffn=d_ffn,
            undistort=undistort,
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
            init_cfg=init_cfg,
            seq_len=seq_len,
            use_shape_smooth=use_shape_smooth,
            enhance_lefthand=enhance_lefthand,
            enhance_static=enhance_static,
            fix_sigma_pars=fix_sigma_pars
            )
        self.predict_frame = predict_frame
        self.smooth_window_len = smooth_window_len
        self.predict_way = predict_way
        if predict_way not in ['SmoothNet', 'RNN']:
            raise ValueError('the predict way must in SmoothNet or RNN')
        
        if smooth_window_len > seq_len:
            raise ValueError('smooth_window_len size must bigger than seq_len')

        # self.out_list = dict()
        self.smoothnet_predict = False
        if self.predict_way == "SmoothNet":
            self.smoothnet_predict = True
            self.predict_root_layer = MotionSmoothNet(self.smooth_window_len,
                                                    self.predict_frame)
            self.predict_local_layer = SmoothNet(self.smooth_window_len,
                                                self.predict_frame)
        self.l1_loss_func = L1Loss(
            use_target_weight=True,
            loss_weight=1.,
        )

    def _forward(
        self,
        feats: Tuple[Tensor],
        history_feats: Tuple[Tensor],
        mems=None,
    ) -> Tensor:
        devices_cuda = feats.device
        B = feats.shape[0]
        out_feats = self.liftnet(feats)
        history_feats = history_feats.reshape(B, self.output_num, -1, 1)
        if mems is None:
            mems = torch.zeros(B, 2 * self.channel_num, 1, 1).to(devices_cuda)
        feat_mix = torch.cat([out_feats, mems], dim=1)
        mems = self.temporal(feat_mix)
        output = self.last_layer(feat_mix)
        raw_output = output[:, :, 0].clone()
        if self.smoothnet_predict == True:
            output = self.smoothnet_predict_forward(output.unsqueeze(0), history_feats[:,:,:,0], 1)[0]
        shape, rot, svd_pt = self.simple_feature_layer(output, feats[:, -1, 0,
                                                                     0])
        score = self.sigma_conv(out_feats).sigmoid().mean().reshape(shape.shape)
        return shape, rot, svd_pt, mems, score

    def forward(self,
                feats: Tuple[Tensor],
                history_feats: Tuple[Tensor],
                mems=None,
                seq_len: int = 1) -> Tensor:
        feats = self.liftnet(feats)
        sigma = self.sigma_conv(feats)
        
        B = int(feats.shape[0] / seq_len)
        history_feats = history_feats.reshape(B, self.output_num, -1, 1)
        sigmas = sigma.reshape(feats.shape[0], 21, 3)
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
        # outputs = outputs.reshape(B * seq_len, -1, 1, 1)
        raw_output = outputs[:, :, :, 0].clone()
        if self.smoothnet_predict == True:
            outputs = self.smoothnet_predict_forward(outputs,
                                                    history_feats[:, :, :,
                                                                0], seq_len)
        return raw_output, outputs, mems, sigmas

    def smoothnet_predict_forward(self,
                                  feats: Tuple[Tensor],
                                  feats_history: Tuple[Tensor],
                                  seq_len: int = 1) -> Tensor:
        devices_cuda = feats.device
        outputs = feats.clone()
        feats_reshape = feats[:, :, :, 0, 0].permute(0, 2, 1)
        B = feats_reshape.shape[0]
        svd_begin = self.pose_ncomp + self.shape_ncomp

        for i in range(seq_len):
            root_info_history = torch.zeros(
                (B, 21, self.smooth_window_len)).to(devices_cuda)
            local_info_history = torch.zeros(
                (B, self.pose_ncomp, self.smooth_window_len)).to(devices_cuda)
            feed_len = self.smooth_window_len - 1 - i
            if feed_len > 0:
                root_info_history[:, :, :feed_len] = feats_history[:,
                                                                   svd_begin:,
                                                                   -1 *
                                                                   feed_len:]
                local_info_history[:, :, :feed_len] = feats_history[:, :self.
                                                                    pose_ncomp,
                                                                    -1 *
                                                                    feed_len:]

            for j in range(i + 1):
                if self.smooth_window_len - 1 - i + j >= 0:
                    root_info_history[:, :, self.smooth_window_len - 1 - i +
                                      j] = feats_reshape[:, svd_begin:, j]
                    local_info_history[:, :, self.smooth_window_len - 1 - i +
                                       j] = feats_reshape[:, :self.pose_ncomp,
                                                          j]
            predicted_root_value = self.predict_root_layer(root_info_history)
            predicted_local_value = self.predict_local_layer(
                local_info_history)

            outputs[:, i:i + 1, svd_begin:, 0,
                    0] = predicted_root_value.permute(0, 2, 1)
            outputs[:, i:i + 1, :self.pose_ncomp, 0,
                    0] = predicted_local_value.permute(0, 2, 1)
        return outputs

    def predict(self,
                feats: Tuple[Tensor],
                feats_history: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'predict')

        raw_feats, output, mems, sigma = self.forward(data['feats'],
                                                      feats_history, mems, 1)

        B = output.shape[0]
        output = output.reshape(B, -1, 1, 1)
        sigma = sigma.reshape(-1, 21, 3)
        
        hand3d_pred = self.postprocess(
            output,
            data['left_hand'],
            data['leftcam_xy'],
            data['left_R'],
            data['nimble_info'],
            data['hand3d_gt'],
            data['baseline_scale'],
            only_pre=True)[0]
        
        # import numpy as np
        # for (batch_data_sample,hand3d_pred_sin) in zip(batch_data_samples[::2], hand3d_pred):
        #     id_name = int(batch_data_sample.img_path.split("__")[-1])
        #     self.out_list[id_name] = {
        #         "kpt3d": hand3d_pred_sin.detach().cpu().numpy(),
        #     }
        # np.save("pre_predict_0516.npy", self.out_list)
        
        if self.reproj:
            camera_model = batch_data_samples[0].meta['ori_camera']
            leftcam_uv_reproj_distort = camera_model.eye_to_window(
                hand3d_pred.cpu().numpy())
            leftcam_uv_reproj_distort = torch.tensor(
                leftcam_uv_reproj_distort).cuda()
            return hand3d_pred, leftcam_uv_reproj_distort[:, None,
                                                          ...], mems, sigma, raw_feats
        else:
            return hand3d_pred, data[
                'uv_coord_im_pred_global_distort'], mems, sigma, raw_feats

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'loss')
        B = int(data['feats'].shape[0] / self.seq_len)
        history_dim = self.output_num
        history_output = torch.zeros((B, history_dim, self.seq_len, 1))
        raw_output, output, mems, all_sigmas = self.forward(data['feats'],
                                                       history_output, None,
                                                       self.seq_len)
        valid_frame = self.seq_len - self.predict_frame
        output = output[:,:valid_frame,...].reshape(B * valid_frame, -1, 1, 1)
        
        predict_used_index = []
        predict_drop_t = [j for j in range(self.predict_frame)]
        for i in range(B * self.seq_len):
            if i % self.seq_len not in predict_drop_t:
                predict_used_index.append(i)

        left_hand = data['left_hand'][predict_used_index]
        leftcam_xy = data['leftcam_xy'][predict_used_index]
        left_R = data['left_R'][predict_used_index]
        hand3d_gt_predict = data['hand3d_gt'][predict_used_index]
        baseline_scale = data['baseline_scale'][predict_used_index]
        nimble_info_trans = data['nimble_info']['nimble_trans'][
            predict_used_index]

        hand3d_gt_curent = data['hand3d_gt'][predict_used_index]
        nimble_info_shape = data['nimble_info']['nimble_shape'][
            predict_used_index]
        nimble_info_pose = data['nimble_info']['nimble_pose'][
            predict_used_index]

        new_nimble_info = {
            'nimble_pose': nimble_info_pose,
            'nimble_shape': nimble_info_shape,
            'nimble_trans': nimble_info_trans
        }
        
        
        # 3d 损失
        (pre_root__xyz, pre_local__xyz, hand3d_pred, hand3d_part_gt,
         pre_trans_xyz, pre_shape, gt_all_matrix,
         pre_all_matrix) = self.postprocess(output, left_hand, leftcam_xy,
                                            left_R, new_nimble_info,
                                            hand3d_gt_curent, baseline_scale,
                                            False)

        # 直接监督rot和trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz
        gt_nimble_trans = new_nimble_info['nimble_trans']

        # pinch 损失
        dist_pred = torch.norm(
            pre_local__xyz[:, 4, :] - pre_local__xyz[:, 8, :], dim=-1)
        dist_gt = torch.norm(
            hand3d_gt_curent[:, 4, :] - hand3d_gt_curent[:, 8, :], dim=-1)

        if self.enhance_lefthand:
            mask = left_hand == 1
            left_weight = 1.2
            enhanced_left_hand3d_gt_curent = self.enhanced_fun(
                hand3d_gt_curent, mask, left_weight)
            enhanced_left_hand3d_gt_predict = self.enhanced_fun(
                hand3d_gt_predict, mask, left_weight)
            enhanced_left_pre_root__xyz = self.enhanced_fun(
                pre_root__xyz, mask, left_weight)
            enhanced_left_pre_local__xyz = self.enhanced_fun(
                pre_local__xyz, mask, left_weight)
            enhanced_left_pre_all_xyz = self.enhanced_fun(
                hand3d_pred, mask, left_weight)
        else:
            enhanced_left_hand3d_gt_curent = hand3d_gt_curent
            enhanced_left_hand3d_gt_predict = hand3d_gt_predict
            enhanced_left_pre_root__xyz = pre_root__xyz
            enhanced_left_pre_local__xyz = pre_local__xyz
            enhanced_left_pre_all_xyz = hand3d_pred

        if self.enhance_static:
            static_weight = 25
            static_mask = self.generate_static_mask(batch_data_samples)
            static_mask = static_mask[predict_used_index]
            enhanced_static_pre_root__xyz = self.enhanced_fun(
                pre_root__xyz, static_mask, static_weight)
            enhanced_static_pre_local__xyz = self.enhanced_fun(
                pre_local__xyz, static_mask, static_weight)
            enhanced_static_pre_all_xyz = self.enhanced_fun(
                hand3d_pred, static_mask, static_weight)
            enhanced_static_hand3d_gt_predict = self.enhanced_fun(
                hand3d_gt_predict, static_mask, static_weight)
            enhanced_static_hand3d_gt_curent = self.enhanced_fun(
                hand3d_gt_curent, static_mask, static_weight)
        else:
            enhanced_static_pre_root__xyz = pre_root__xyz
            enhanced_static_pre_local__xyz = pre_local__xyz
            enhanced_static_pre_all_xyz = hand3d_pred
            enhanced_static_hand3d_gt_predict = hand3d_gt_predict
            enhanced_static_hand3d_gt_curent = hand3d_gt_curent

        enhanced_static_pre_root__xyz = enhanced_static_pre_all_xyz
        enhanced_static_pre_local__xyz = enhanced_static_pre_all_xyz
        
        re_all_sigmas = torch.cat((hand3d_pred, all_sigmas), dim=-1)

        pred_for_loss = [
            enhanced_left_pre_root__xyz, enhanced_left_pre_local__xyz,
            dist_pred, pre_nimble_trans, enhanced_static_pre_root__xyz,
            enhanced_static_pre_local__xyz, re_all_sigmas
        ]
        targ_for_loss = [
            enhanced_left_hand3d_gt_predict, enhanced_left_hand3d_gt_curent,
            dist_gt, gt_nimble_trans, enhanced_static_hand3d_gt_predict,
            enhanced_static_hand3d_gt_curent, hand3d_gt
        ]

        weight_ini = torch.ones((1, 21, 3))
        weight_ini[0, :9, :] = 2
        weight_ini[0, 4, :], weight_ini[0, 8, :] = 4, 4
        weight_ini = weight_ini.repeat(hand3d_gt_curent.shape[0], 1,
                                       1).to(hand3d_gt_curent.device)

        # weight_ini_for_pre_nimble = weight_ini.clone().to(hand3d_gt_curent.device)
        # weight_ini_for_pre_nimble[:, :9, :] = 4
        # weight_ini_for_pre_nimble[:,
        #                           4, :], weight_ini_for_pre_nimble[:,
        #                                                            8, :] = 8, 8

        weight_for_loss = [
            weight_ini,
            weight_ini,
            None,
            None,
            None,
            None,
            None
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss, weight_for_loss)
        (loss_root_predict, loss_local_curent, loss_pinch, loss_nimble_trans,
         loss_root_smooth, loss_local_smooth, loss_rle) = losses

        loss_all_predcit = self.l1_loss_func(enhanced_left_pre_all_xyz,
                                             enhanced_left_hand3d_gt_predict,
                                             weight_ini) * 2

        if self.use_shape_smooth:
            pre_shape_reshape = pre_shape.reshape(-1, valid_frame)
            mean_shape = torch.mean(
                pre_shape_reshape,
                dim=-1).unsqueeze(-1).repeat(1, valid_frame)
            smooth_shape_loss = self.shape_loss_func(pre_shape_reshape,
                                                     mean_shape)
        else:
            smooth_shape_loss = torch.tensor(
                0.0, device=loss_root_predict.device)

        # # 子骨骼向量监督
        if self.use_bone_loss:
            bone_loss_weight = 0.03
            bone_3d_pre = (hand3d_pred - hand3d_pred[:, self.joint_parents, :]
                           )[:, self.non_root_indices].reshape(-1, 3)
            bone_3d_gt = (hand3d_gt_predict -
                          hand3d_gt_predict[:, self.joint_parents, :]
                          )[:, self.non_root_indices].reshape(-1, 3)

            bone_3d_pre_vector = self.cal_normalize_vector(bone_3d_pre)
            bone_3d_gt_vector = self.cal_normalize_vector(bone_3d_gt)

            squared_diff = (bone_3d_pre_vector - bone_3d_gt_vector)**2
            bone_loss = torch.mean(torch.sum(squared_diff,
                                             dim=1)) * bone_loss_weight

            # 局部子骨骼监督
            major_bone_loss_weight = 0.1
            local_bone_3d_pre = (pre_local__xyz -
                                 pre_local__xyz[:, self.joint_parents, :]
                                 )[:, self.non_root_indices]
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
            major_bone_loss = torch.tensor(
                0.0, device=loss_root_predict.device)

        if self.fix_sigma_pars:
            loss_rle = torch.tensor(0.0, device=loss_root_predict.device)
        
        losses_dict = dict(
            loss_pre_root=loss_root_predict,
            loss_pre_nimble=loss_local_curent,
            loss_pre_all=loss_all_predcit,
            bone_loss=bone_loss,
            major_bone_loss=major_bone_loss,
            loss_pinch=loss_pinch,
            loss_nimble_trans=loss_nimble_trans,
            smooth_shape_loss=smooth_shape_loss,
            loss_root_smooth=loss_root_smooth,
            loss_local_smooth=loss_local_smooth,
            loss_rle=loss_rle)

        return losses_dict

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