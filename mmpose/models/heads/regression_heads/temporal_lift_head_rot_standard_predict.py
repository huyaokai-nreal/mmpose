# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import roma
import torch
import torch.nn.functional as F
from mmengine.logging import MessageHub
from torch import Tensor, nn

from mmpose.models.heads.nimble.nimble_utils import (adjust_predicted_angles,
                                                     cal_proportion,
                                                     matrix_to_euler_angles,
                                                     matrix_to_quaternion,
                                                     trans_3d_2_2d)
from mmpose.models.heads.regression_heads.lift_head_rot_standard import \
    LiftNimbleHeadStandard
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions
from mmpose.models.losses.regression_loss import RLELoss


@MODELS.register_module()
class TemporalLiftNimbleHeadStandardPredict(LiftNimbleHeadStandard):
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
                 use_pose_loss: bool = False,
                 use_rle_loss: bool = False,
                 use_shape_smooth=False,
                 use_9d_pose_reg: bool = False,
                 use_6d_pose_reg: bool = False,
                 all_use_kp2d_gt: bool = False,
                 seq_len: int = 4,
                 predict_frame: int=1,
                 flow_model_pretrain = "",
                 enhance_lefthand = True,
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
            use_rle_loss=use_rle_loss,
            use_6d_pose_reg=use_6d_pose_reg,
            use_9d_pose_reg=use_9d_pose_reg,
            lambda_t=lambda_t,
            all_use_kp2d_gt=all_use_kp2d_gt,
            init_cfg=init_cfg,
            use_pose_loss=use_pose_loss)
        self.seq_len = seq_len
        self.predict_frame = predict_frame

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
        if self.use_rle_loss:
            self.sigma_conv = nn.Conv2d(self.feat_dim * 2, 21 * 3, kernel_size=1)
            self.rle_loss_func = RLELoss(dim=3,flow_model_pretrain_path=flow_model_pretrain)
        self.enhance_lefthand = enhance_lefthand
        self.out_list = dict()

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
        shape, rot, svd_pt = self.simple_feature_layer(output, feats[:, -1, 0,
                                                                     0])
        score = self.sigma_conv(feat_mix).sigmoid().mean().reshape(shape.shape)
        return shape, rot, svd_pt, mems, score

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
        sigmas = torch.zeros((B, seq_len, 21 * 3, 1, 1)).cuda()
        for i in range(seq_len):
            feat = feats[:, i:i + 1, :].reshape(B, -1, 1, 1)
            feat_mix = torch.cat([feat, mems], dim=1)
            mems = self.temporal(feat_mix)
            output = self.last_layer(feat_mix)
            if self.use_rle_loss:
                sigma = self.sigma_conv(feat_mix)
                sigmas[:, i, ...] = sigma
            outputs[:, i, ...] = output
        # outputs = outputs.reshape(B * seq_len, -1, 1, 1)
        # sigmas = sigmas.reshape(B * seq_len, 21, 3)
        return outputs, mems, sigmas

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'predict')

        output, mems, sigma = self.forward(data['feats'], mems, 1)

        B = output.shape[0]
        output = output.reshape(B, -1, 1, 1)
        
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
                                                          ...], mems, sigma
        else:
            return hand3d_pred, data[
                'uv_coord_im_pred_global_distort'], mems, sigma

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'loss')
        
        output, mems, sigma = self.forward(data['feats'], None, self.seq_len)

        B = output.shape[0]
        valid_frame = self.seq_len - self.predict_frame
        output = output[:,:valid_frame,...].reshape(B * valid_frame, -1, 1, 1)
        sigma = sigma[:,:valid_frame,...].reshape(B * valid_frame, 21, 3)
        
        predict_used_index = []
        curent_used_index = []
        predict_drop_t = [j for j in range(self.predict_frame)]
        curent_drop_t = [j for j in range(valid_frame, self.seq_len)]
        for i in range(B * self.seq_len):
            if i % self.seq_len not in predict_drop_t:
                predict_used_index.append(i) 
            if i % self.seq_len not in curent_drop_t:
                curent_used_index.append(i)

        left_hand = data['left_hand'][predict_used_index]
        leftcam_xy = data['leftcam_xy'][predict_used_index]
        left_R = data['left_R'][predict_used_index]
        hand3d_gt_predict = data['hand3d_gt'][predict_used_index]
        hand3d_gt_curent = data['hand3d_gt'][curent_used_index]
        baseline_scale = data['baseline_scale'][predict_used_index]
        
        nimble_info_pose = torch.cat((data['nimble_info']['nimble_pose'][predict_used_index][:,:1,:], data['nimble_info']['nimble_pose'][curent_used_index][:,1:,:]), dim= 1)
        nimble_info_shape = data['nimble_info']['nimble_shape'][curent_used_index]
        nimble_info_trans = data['nimble_info']['nimble_trans'][predict_used_index]
        new_nimble_info = {
            "nimble_pose":nimble_info_pose, 
            "nimble_shape":nimble_info_shape,
            "nimble_trans":nimble_info_trans
        }
        
        
        # 3d 损失
        (pre_root__xyz, pre_local__xyz, hand3d_pred, hand3d_part_gt,
         pre_trans_xyz, pre_shape, gt_all_matrix,
         pre_all_matrix) = self.postprocess(output, left_hand,
                                            leftcam_xy, left_R,
                                            new_nimble_info,
                                            hand3d_gt_curent,
                                            baseline_scale, False)        


        # 直接监督rot和trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz
        gt_nimble_trans = new_nimble_info['nimble_trans']

        # pinch 损失
        dist_pred = torch.norm(
            pre_local__xyz[:, 4, :] - pre_local__xyz[:, 8, :], dim=-1)
        dist_gt = torch.norm(hand3d_gt_curent[:, 4, :] - hand3d_gt_curent[:, 8, :], dim=-1)

        def left_enhanced_fun(kpt, mask):
            enhanced_kpt = kpt.clone()
            enhanced_kpt[mask] = enhanced_kpt[mask] * 1.2
            return enhanced_kpt
        if self.enhance_lefthand:
            mask = left_hand == 1
            enhanced_hand3d_gt_curent = left_enhanced_fun(hand3d_gt_curent, mask)
            enhanced_hand3d_gt_predict = left_enhanced_fun(hand3d_gt_predict, mask)
            
            enhanced_pre_root__xyz = left_enhanced_fun(pre_root__xyz, mask)
            enhanced_pre_local__xyz = left_enhanced_fun(pre_local__xyz, mask)

            pred_for_loss = [
                enhanced_pre_root__xyz, enhanced_pre_local__xyz, dist_pred, 
                pre_nimble_trans, pre_root__xyz, pre_local__xyz
            ]
            targ_for_loss = [
                enhanced_hand3d_gt_predict, enhanced_hand3d_gt_curent, 
                dist_gt,  gt_nimble_trans, hand3d_gt_predict, hand3d_gt_curent
            ]

        else:
            pred_for_loss = [
                pre_root__xyz, pre_local__xyz, dist_pred, 
                pre_nimble_trans, pre_root__xyz, pre_local__xyz
            ]
            targ_for_loss = [
                hand3d_gt_predict, hand3d_gt_curent, dist_gt, 
                gt_nimble_trans, hand3d_gt_predict, hand3d_gt_curent
            ]

        weight_ini = torch.ones((1, 21, 3))
        weight_ini[0, :9, :] = 2
        weight_ini[0, 4, :], weight_ini[0, 8, :] = 4, 4
        weight_ini = weight_ini.repeat(hand3d_gt_curent.shape[0], 1,
                                       1).to(hand3d_gt_curent.device)

        weight_ini_for_pre_nimble = weight_ini.clone().to(hand3d_gt_curent.device)
        weight_ini_for_pre_nimble[:, :9, :] = 4
        weight_ini_for_pre_nimble[:,
                                  4, :], weight_ini_for_pre_nimble[:,
                                                                   8, :] = 8, 8

        weight_for_loss = [
            weight_ini, weight_ini_for_pre_nimble, None, None, None,
            None, 
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss, weight_for_loss)
        (loss_root_predict, loss_local_curent, loss_pinch, 
         loss_nimble_trans, loss_root_smooth, loss_local_smooth) = losses

        if self.use_shape_smooth:
            pre_shape_reshape = pre_shape.reshape(-1, self.seq_len)
            mean_shape = torch.mean(
                pre_shape_reshape,
                dim=-1).unsqueeze(-1).repeat(1, self.seq_len)
            smooth_shape_loss = self.shape_loss_func(pre_shape_reshape,
                                                     mean_shape)
        else:
            smooth_shape_loss = torch.tensor(0.0, device=loss_root_predict.device)

        # # 子骨骼向量监督
        if self.use_bone_loss:
            # 局部子骨骼监督
            major_bone_loss_weight = 0.5
            local_bone_3d_pre = (
                pre_local__xyz -
                pre_local__xyz[:, self.joint_parents, :])[:,
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
            major_bone_loss = torch.tensor(0.0, device=loss_root_predict.device)

        
        mh = MessageHub.get_current_instance()
        cur_epoch = mh.get_info('epoch')

        if self.use_rle_loss and cur_epoch > 10:
            re_sigma = torch.cat((hand3d_pred, sigma), dim=-1)
            loss_rle = self.rle_loss_func(re_sigma, hand3d_gt_predict)
        else:
            loss_rle = torch.tensor(
                0.0, device=loss_root_predict.device)

        losses_dict = dict(
            loss_pre_root=loss_root_predict,
            loss_pre_nimble=loss_local_curent,
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
