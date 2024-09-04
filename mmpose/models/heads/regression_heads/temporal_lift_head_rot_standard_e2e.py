# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmpose.models.heads.regression_heads.lift_head_rot_standard_e2e import \
    LiftNimbleHeadStandardE2e
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


class CFA_Module(nn.Module):

    def __init__(self, embed_dim):
        super(CFA_Module, self).__init__()
        self.W_Q = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.W_K = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.W_V = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, v_t, v_t_1):
        # v_t, v_t_1 are expected to be [B, C, H, W] tensors
        B, C, H, W = v_t.shape

        # Generate Q, K, V and ensure they are contiguous
        Q = self.W_Q(v_t).view(B, C, -1).permute(0, 2,
                                                 1).contiguous()  # [B, H*W, C]
        K = self.W_K(v_t_1).view(B, C, -1).contiguous()  # [B, C, H*W]
        V = self.W_V(v_t_1).view(B, C,
                                 -1).permute(0, 2,
                                             1).contiguous()  # [B, H*W, C]

        # Attention computation
        attn = self.softmax(torch.bmm(Q, K) / (C**0.5))  # [B, H*W, H*W]
        v_i_prime = torch.bmm(attn, V).permute(0, 2, 1).view(B, C, H,
                                                             W).contiguous()

        return v_i_prime


@MODELS.register_module()
class TemporalLiftNimbleHeadStandardE2e(LiftNimbleHeadStandardE2e):
    """liftHead for getting 3d rotation from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 d_ffn: int = 220,
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
                 e2e=False,
                 lambda_t: int = -1,
                 corruption_cam: float = 0.5,
                 use_bone_loss: bool = True,
                 use_shape_smooth=False,
                 use_9d_pose_reg: bool = False,
                 use_6d_pose_reg: bool = False,
                 all_use_kp2d_gt: bool = False,
                 seq_len: int = 4,
                 enhance_lefthand=False,
                 enhance_static=False,
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
            init_cfg=init_cfg,
            e2e=e2e)
        self.seq_len = seq_len
        self.data_flip_aug = data_flip_aug

        self.cfa = CFA_Module(self.channel_num * 2)  # CFA模块
        self.last_layer = nn.Sequential(
            nn.Conv2d(self.feat_dim * 2, self.feat_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.feat_dim, self.output_num, kernel_size=1))
        # self.temporal = nn.Sequential(
        #     nn.Conv2d(
        #         2 * self.channel_num * 2, 2 * self.channel_num, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(
        #         self.channel_num * 2, self.channel_num * 2, kernel_size=1))
        self.use_shape_smooth = use_shape_smooth
        if use_shape_smooth:
            self.shape_loss_func = F.l1_loss

        self.enhance_lefthand = enhance_lefthand
        self.enhance_static = enhance_static
        self.static_data_date_list = ['20240516', '20240517', '20240522']

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
        mems = self.cfa(out_feats, mems)
        feat_mix = torch.cat([out_feats, mems], dim=1)
        # mems = self.temporal(feat_mix)
        output = self.last_layer(feat_mix)
        shape, rot, svd_pt = self.simple_feature_layer(output, feats[:, -1, 0,
                                                                     0])
        score = self.sigma_conv(out_feats).sigmoid().mean().reshape(
            shape.shape)
        return shape, rot, svd_pt, mems, score

    def forward(self,
                feats: Tuple[Tensor],
                mems=None,
                seq_len: int = 1) -> Tensor:

        feats = self.liftnet(feats)
        sigma = self.sigma_conv(feats)
        sigmas = sigma.reshape(feats.shape[0], 21, 3)

        B = int(feats.shape[0] / seq_len)
        if mems is None:
            mems = torch.zeros(B, 2 * self.channel_num, 1, 1).cuda()
        feats = feats.view(B, seq_len, -1)
        outputs = torch.zeros((B, seq_len, self.output_num, 1, 1)).cuda()
        # import ipdb;ipdb.set_trace()
        for i in range(seq_len):
            feat = feats[:, i:i + 1, :].reshape(B, -1, 1, 1)
            mems = self.cfa(feat, mems)
            feat_mix = torch.cat([feat, mems], dim=1)
            # mems = self.temporal(feat_mix)
            output = self.last_layer(feat_mix)
            outputs[:, i, ...] = output
        outputs = outputs.reshape(B * seq_len, -1, 1, 1)
        return outputs, mems, sigmas

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples)

        output, mems, all_sigmas = self.forward(data['feats'], mems, 1)

        hand3d_pred = self.postprocess(
            output,
            data['left_hand'],
            data['leftcam_xy'],
            data['left_R'],
            data['nimble_info'],
            data['hand3d_gt'],
            data['baseline_scale'],
            only_pre=True)[0]
        camera_model = batch_data_samples[0].meta['ori_camera']
        leftcam_uv_reproj_distort = camera_model.eye_to_window(
            hand3d_pred.cpu().numpy())
        leftcam_uv_reproj_distort = torch.tensor(
            leftcam_uv_reproj_distort).cuda()
        return hand3d_pred, leftcam_uv_reproj_distort, mems, all_sigmas

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        if self.e2e:
            data = self.preprocess(
                feats,
                batch_data_samples,
            )
        else:
            with torch.no_grad():
                data = self.preprocess(feats, batch_data_samples)

        output, mems, all_sigmas = self.forward(data['feats'], None,
                                                self.seq_len)

        hand3d_gt = data['hand3d_gt']
        # 3d 损失
        (pred_3d_way1, pred_3d_way2, hand3d_pred, hand3d_part_gt,
         pre_trans_xyz, pre_shape, gt_all_matrix,
         pre_all_matrix) = self.postprocess(output, data['left_hand'],
                                            data['leftcam_xy'], data['left_R'],
                                            data['nimble_info'],
                                            data['hand3d_gt'],
                                            data['baseline_scale'], False)
        pred_3d_way1 = pred_3d_way1
        hand3d_pred = hand3d_pred
        pre_trans_xyz = pre_trans_xyz
        all_sigmas = all_sigmas
        hand3d_gt = data['hand3d_gt']

        # 直接监督rot和trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz
        gt_nimble_trans = data['nimble_info']['nimble_trans']

        # pinch 损失
        dist_pred = torch.norm(
            pred_3d_way2[:, 4, :] - pred_3d_way2[:, 8, :], dim=-1)
        dist_gt = torch.norm(
            hand3d_part_gt[:, 4, :] - hand3d_part_gt[:, 8, :], dim=-1)

        if self.enhance_lefthand:
            mask = data['left_hand'] == 1
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
            static_mask = self.generate_static_mask(batch_data_samples)
            enhanced_static_hand3d_pred = self.enhanced_fun(
                hand3d_pred, static_mask, static_weight)
            enhanced_static_hand3d_gt = self.enhanced_fun(
                hand3d_gt, static_mask, static_weight)
        else:
            enhanced_static_hand3d_pred = hand3d_pred
            enhanced_static_hand3d_gt = hand3d_gt

        re_all_sigmas = torch.cat((hand3d_pred, all_sigmas), dim=-1)

        pred_for_loss = [
            enhanced_left_pred_3d_way1, enhanced_left_pred_3d_way2,
            enhanced_left_hand3d_pred, dist_pred, pre_nimble_trans,
            enhanced_static_hand3d_pred, re_all_sigmas
        ]
        targ_for_loss = [
            enhanced_left_hand3d_gt, enhanced_left_hand3d_part_gt,
            enhanced_left_hand3d_gt, dist_gt, gt_nimble_trans,
            enhanced_static_hand3d_gt, hand3d_gt
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

        weight_for_loss = [
            weight_ini_ori, weight_ini_for_pre_nimble, weight_ini_ori, None,
            None, None, None
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss, weight_for_loss)
        (loss_pre_root, loss_pre_nimble, loss_pre_all, loss_pinch,
         loss_nimble_trans, loss_smooth, loss_rle) = losses

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

        if self.data_flip_aug:
            B = pre_shape.shape[0] // 2
            origin_shape, flip_shape = pre_shape[:B], pre_shape[B:]
            shape_loss_cons = torch.mean(torch.abs(origin_shape - flip_shape))
        else:
            shape_loss_cons = torch.tensor(0.0, device=loss_pre_root.device)

        losses_dict = dict(
            loss_pre_root=loss_pre_root,
            loss_pre_nimble=loss_pre_nimble,
            loss_pre_all=loss_pre_all,
            bone_loss=bone_loss,
            major_bone_loss=major_bone_loss,
            loss_pinch=loss_pinch,
            loss_nimble_trans=loss_nimble_trans,
            smooth_shape_loss=smooth_shape_loss,
            loss_smooth=loss_smooth,
            loss_rle=loss_rle,
            shape_loss_cons=shape_loss_cons)

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
