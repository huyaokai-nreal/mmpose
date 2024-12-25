# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.logging import MessageHub
from torch import Tensor, nn

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
                 direct_pose_reg: bool = False,
                 seq_len: int = 4,
                 enhance_lefthand=True,
                 enhance_static=True,
                 fix_sigma_pars=False,
                 data_flip_aug: bool = False,
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
            direct_pose_reg=direct_pose_reg,
            lambda_t=lambda_t,
            all_use_kp2d_gt=all_use_kp2d_gt,
            data_flip_aug=data_flip_aug,
            init_cfg=init_cfg)
        self.seq_len = seq_len

        self.use_shape_smooth = use_shape_smooth
        if use_shape_smooth:
            self.shape_loss_func = F.l1_loss

        self.temporal = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim * 2), nn.ReLU(),
            nn.Linear(self.feat_dim * 2, self.feat_dim))
        self.enhance_lefthand = enhance_lefthand
        self.enhance_static = enhance_static
        self.static_data_date_list = [
            '20240516', '20240517', '20240522', '20241114'
        ]
        self.reverse_pinch_date_list = [
            '20240220', '20240229', '20240926', '20241030'
        ]
        self.fix_sigma_pars = fix_sigma_pars
        self.pinch_loss_func = F.l1_loss
        self.hand_constraint_index_list = [[5, 6, 7, 8], [9, 10, 11, 12],
                                           [13, 14, 15, 16], [17, 18, 19, 20]]

        if self.fix_sigma_pars:
            for param in self.liftnet.parameters():
                param.requires_grad = False
            for param in self.sigma_conv.parameters():
                param.requires_grad = False
        # self.out_list = dict()

    def _forward(
        self,
        feats: Tuple[Tensor],
        mems=None,
    ) -> Tensor:
        devices_cuda = feats.device
        B = feats.shape[0]
        out_feats = self.liftnet(feats.reshape(B, 21, -1)).reshape(B, -1)
        if mems is None:
            mems = torch.zeros(B, out_feats.shape[-1], 1, 1).to(devices_cuda)
        feat_mix = torch.cat([out_feats.reshape(B, -1),
                              mems.reshape(B, -1)],
                             dim=1)
        mems = self.temporal(feat_mix)
        output = self.last_layer(feat_mix)
        shape, rot, svd_pt = self.simple_feature_layer(output[..., None, None])
        score = self.sigma_conv(out_feats.reshape(
            B, -1)).sigmoid().mean(-1).reshape(shape.shape)
        return shape, rot, svd_pt, mems, score

    def forward(self,
                feats: Tuple[Tensor],
                mems=None,
                seq_len: int = 1) -> Tensor:
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
        outputs = outputs.reshape(B * seq_len, -1, 1, 1)
        return outputs, mems, sigmas

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'predict')
        valid_mask = data['valid_mask'] == 1
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
        hand3d_pred = hand3d_pred[valid_mask]
        # for (batch_data_sample,hand3d_pred_sin) in zip(batch_data_samples[::2], hand3d_pred):
        #     id_name = int(batch_data_sample.img_path.split("__")[-1])
        #     self.out_list[id_name] = {
        #         "kpt3d": hand3d_pred_sin.detach().cpu().numpy(),
        #     }
        # np.save("pre_predict_0517_static_raw.npy", self.out_list)

        if self.reproj:
            camera_model = batch_data_samples[0].meta['ori_camera']
            leftcam_uv_reproj_distort = camera_model.eye_to_window(
                hand3d_pred.cpu().numpy())
            leftcam_uv_reproj_distort = torch.tensor(
                leftcam_uv_reproj_distort).cuda()
            return (hand3d_pred, leftcam_uv_reproj_distort[:, None, ...], mems,
                    all_sigmas)
        else:
            return (hand3d_pred, data['uv_coord_im_pred_global_distort'], mems,
                    all_sigmas)

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'loss')

        valid_mask = data['valid_mask'] == 1

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
        pred_3d_way1 = pred_3d_way1[valid_mask]
        hand3d_pred = hand3d_pred[valid_mask]
        pre_trans_xyz = pre_trans_xyz[valid_mask]
        all_sigmas = all_sigmas[valid_mask]
        hand3d_gt = data['hand3d_gt'][valid_mask]

        # 直接监督rot和trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz
        gt_nimble_trans = data['nimble_info']['nimble_trans'][valid_mask]

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
            re_all_sigmas
        ]
        targ_for_loss = [
            enhanced_left_hand3d_gt, enhanced_left_hand3d_part_gt,
            enhanced_left_hand3d_gt, dist_gt, gt_nimble_trans,
            enhanced_static_hand3d_gt, enhanced_static_hand3d_gt, hand3d_gt
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
            None, None, None, None
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss, weight_for_loss)
        (loss_pre_root, loss_pre_nimble, loss_pre_all, loss_pinch,
         loss_nimble_trans, loss_smooth, loss_smooth_root, loss_rle) = losses

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

        mh = MessageHub.get_current_instance()
        cur_epoch = mh.get_info('epoch')
        pinch_mask = (dist_pred > dist_gt - 0.002) & (dist_gt < 0.02)
        reverse_mask = self.generate_mask(batch_data_samples,
                                          self.reverse_pinch_date_list).to(
                                              pinch_mask.cuda())
        reverse_mask = torch.concat([reverse_mask, reverse_mask])
        pinch_reverse_mask = pinch_mask & reverse_mask

        if cur_epoch > 30 and sum(pinch_mask) > 0:
            valid_num = len(dist_gt[pinch_mask])
            softmax_weight = F.softmax(-dist_gt[pinch_mask], dim=0) * valid_num
            margin_value = torch.ones_like(dist_gt) * 0.003
            margin_value[pinch_reverse_mask] = 0.006
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
        poke_mask = dis_norm < 0.012
        poke_mask = torch.concat([poke_mask, poke_mask])
        if cur_epoch > 30 and sum(poke_mask) > 0:
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
        else:
            loss_poke = torch.tensor(0.0, device=hand3d_gt.device)

        if self.fix_sigma_pars:
            loss_rle = torch.tensor(0.0, device=loss_pre_root.device)

        hand_constraint_loss = self.hand_constraint(
            hand3d_pred, self.hand_constraint_index_list) * 1.5

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
            loss_smooth_root=loss_smooth_root,
            loss_poke=loss_poke,
            loss_hand_constraint=hand_constraint_loss,
            loss_rle=loss_rle,
            pinch_loss_add=pinch_loss_add)

        return losses_dict

    def cal_normalize_vector(self, vector):
        vector_norms = torch.sqrt(
            torch.sum(vector**2, dim=1, keepdim=True) + 1e-8)
        normalized_vector = vector / vector_norms
        return normalized_vector

    def generate_mask(self, batch_data_samples, date_list):
        mask = []
        for batch_sample in batch_data_samples[::2]:
            data_info = batch_sample.img_path.split('/')[-1].split(
                '__')[1].split('_')[0]
            if data_info in date_list:
                mask.append(True)
            else:
                mask.append(False)
        mask = torch.tensor(mask)
        return mask

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
