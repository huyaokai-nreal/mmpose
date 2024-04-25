# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import torch
from torch import Tensor, nn

# from mmpose.post_process.temporal_filters import build_filter
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions
from .lift_head_standard import LiftHeadStandard


@MODELS.register_module()
class TemporalLiftHeadStandard(LiftHeadStandard):
    """liftHead for getting 3d keypoints from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 num_layers: int = 3,
                 d_ffn: int = 220,
                 output_num: int = 42,
                 reproj: bool = False,
                 reproj_thre=0,
                 iou_thre=0,
                 pad_2d=0,
                 score_dim=0,
                 baseline=0.13,
                 corruption_cam: float = 0.5,
                 all_use_kp2d_gt: bool = False,
                 seq_len: int = 4,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(
            lift_loss,
            num_layers,
            d_ffn,
            output_num,
            reproj,
            baseline,
            all_use_kp2d_gt=all_use_kp2d_gt,
            corruption_cam=corruption_cam,
            init_cfg=init_cfg,
            reproj_thre=reproj_thre,
            iou_thre=iou_thre,
            pad_2d=pad_2d,
            score_dim=score_dim)
        self.seq_len = seq_len
        self.last_layer = nn.Sequential(
            nn.Conv2d(self.feat_dim * 2, self.feat_dim, kernel_size=1),
            nn.ReLU(), nn.Conv2d(self.feat_dim, output_num, kernel_size=1))
        self.temporal = nn.Sequential(
            nn.Conv2d(
                2 * self.channel_num * 2, 2 * self.channel_num, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                self.channel_num * 2, self.channel_num * 2, kernel_size=1))
        if score_dim:
            for param in self.liftnet.parameters():
                param.requires_grad = False
            for param in self.last_layer.parameters():
                param.requires_grad = False
            for param in self.temporal.parameters():
                param.requires_grad = False

    def forward(self, feats, mems, seq_len):
        B = int(feats.shape[0] / seq_len)
        K = feats.shape[1] // 2 - 1
        # 标准双目归一化平面2d
        norm_leftcam_xyz = torch.cat(
            (feats[:, :K, 0, 0].reshape(B * seq_len, K // 2, 2),
             torch.ones(B * seq_len, K // 2, 1).cuda()),
            dim=-1)
        norm_rightcam_xyz = torch.cat(
            (feats[:, K + 1:-1, 0, 0].reshape(B * seq_len, K // 2, 2),
             torch.ones(B * seq_len, K // 2, 1).cuda()),
            dim=-1)

        # depth/mems output
        feats = self.liftnet(feats)
        liftnet_output = feats.clone()
        if mems is None:
            mems = torch.zeros(B, 2 * self.channel_num, 1, 1).cuda()
        feats = feats.view(B, seq_len, -1)
        outputs = torch.zeros((B, seq_len, 42, 1, 1)).cuda()
        for i in range(seq_len):
            feat = feats[:, i:i + 1, :].reshape(B, -1, 1, 1)
            feat_mix = torch.concatenate([feat, mems], dim=1)
            mems = self.temporal(feat_mix)
            output = self.last_layer(feat_mix)
            outputs[:, i, ...] = output
        outputs = outputs.reshape(B * seq_len, -1, 1, 1)
        # mems = torch.zeros(B, 2 * self.channel_num, 1, 1).cuda()

        # 标准双目3d点输出
        hand3d_standard = self.get_standard_kpt3d(outputs, norm_leftcam_xyz,
                                                  norm_rightcam_xyz)
        # score output
        if self.score_dim:
            left_reproj, right_reproj = self.trans_3d_2_2d(hand3d_standard)
            left_reproj_error = (left_reproj - norm_leftcam_xyz[..., :2]).view(
                hand3d_standard.shape[0], -1, 1, 1)
            right_reproj_error = (right_reproj -
                                  norm_rightcam_xyz[..., :2]).view(
                                      hand3d_standard.shape[0], -1, 1, 1)
            reproj_error = torch.cat((left_reproj_error, right_reproj_error),
                                     dim=1)
            reproj_feats = self.reproj_layer(reproj_error)
            score_feats = torch.cat((reproj_feats, liftnet_output), axis=1)
            score = self.major_score_layer(score_feats).view(B * seq_len, -1)
            # pinch_score = self.pinch_score_layer(score_feats).view(
            #     B * seq_len, -1)
            # score = torch.cat((major_score, pinch_score), dim=-1)
        else:
            score = torch.ones(B, self.score_dim)
        return hand3d_standard, mems, score

    def _forward(self, feats, mems=None):
        feats = self.liftnet(feats)
        liftnet_output = feats.clone()
        B = feats.shape[0]
        K = feats.shape[1] // 2 - 1
        # 标准双目归一化平面2d
        norm_leftcam_xyz = torch.cat(
            (feats[:, :K, 0, 0].reshape(B, K // 2, 2), torch.ones(
                B, K // 2, 1).cuda()),
            dim=-1)
        norm_rightcam_xyz = torch.cat((feats[:, K + 1:-1, 0, 0].reshape(
            B, K // 2, 2), torch.ones(B, K // 2, 1).cuda()),
                                      dim=-1)

        if mems is None:
            mems = torch.zeros(B, 2 * self.channel_num, 1, 1)
        feat_mix = torch.cat([feats, mems], dim=1)
        mems = self.temporal(feat_mix)
        output = self.last_layer(feat_mix)
        output = output.reshape(B, -1, 1, 1) / self.baseline

        # 标准双目3d点输出
        hand3d_standard = self.get_standard_kpt3d(output, norm_leftcam_xyz,
                                                  norm_rightcam_xyz)
        if self.score_dim:
            left_reproj, right_reproj = self.trans_3d_2_2d(hand3d_standard)
            left_reproj_error = (left_reproj - norm_leftcam_xyz[..., :2]).view(
                hand3d_standard.shape[0], -1, 1, 1)
            right_reproj_error = (right_reproj -
                                  norm_rightcam_xyz[..., :2]).view(
                                      hand3d_standard.shape[0], -1, 1, 1)
            reproj_error = torch.cat((left_reproj_error, right_reproj_error),
                                     dim=1)
            reproj_feats = self.reproj_layer(reproj_error)
            score_feats = torch.cat((reproj_feats, liftnet_output), axis=1)
            score = self.major_score_layer(score_feats).view(B, -1)
            # pinch_score = self.pinch_score_layer(score_feats).view(B, -1)
            # score = torch.cat((major_score, pinch_score), dim=-1)
        else:
            score = torch.ones(B, self.score_dim)
        return hand3d_standard, mems, score

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'predict')
        hand3d_standard, mems, score = self.forward(data['feats'], mems, 1)
        hand3d_pred, leftcam_XYZ, rightcam_XYZ = self.postprocess(
            hand3d_standard, data['left_to_right_rt'], data['left_R'],
            data['baseline_scale'], data['hand_scale'])
        if self.reproj:
            camera_model = batch_data_samples[0].meta['ori_camera']
            leftcam_uv_reproj_distort = camera_model.eye_to_window(
                hand3d_pred.cpu().numpy())
            leftcam_uv_reproj_distort = torch.tensor(
                leftcam_uv_reproj_distort).cuda()
            return (hand3d_pred, leftcam_uv_reproj_distort[:, None, ...], mems,
                    score) if self.score_dim else (
                        hand3d_pred, leftcam_uv_reproj_distort[:, None,
                                                               ...], mems)
        else:
            return (hand3d_pred, data['uv_coord_im_pred_global_distort'], mems,
                    score) if self.score_dim else (
                        hand3d_pred, data['uv_coord_im_pred_global_distort'],
                        mems)

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'loss')
        hand3d_standard, _, score = self.forward(data['feats'], None,
                                                 self.seq_len)
        hand3d_pred, leftcam_XYZ, rightcam_XYZ = self.postprocess(
            hand3d_standard, data['left_to_right_rt'], data['left_R'],
            data['baseline_scale'], data['hand_scale'])

        left_reproj, right_reproj = self.trans_3d_2_2d(hand3d_standard)
        major_gt = torch.cat((data['hand3d_gt'][:, 1:10, :],
                              data['hand3d_gt'][:, 13, :].unsqueeze(1)),
                             dim=1)
        major_pred = torch.cat(
            (hand3d_pred[:, 1:10, :], hand3d_pred[:, 13, :].unsqueeze(1)),
            dim=1)

        # pinch distance, no norm
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(
            data['hand3d_gt'][:, 4, :] - data['hand3d_gt'][:, 8, :], dim=-1)

        pred_for_loss = [
            hand3d_pred, leftcam_XYZ, rightcam_XYZ, left_reproj, right_reproj,
            dist_pred, hand3d_pred, major_pred
        ]
        targ_for_loss = [
            data['hand3d_gt'], data['hand3d_gt'], data['hand3d_gt'],
            data['norm_leftcam_xyz'][..., :2],
            data['norm_rightcam_xyz'][..., :2], dist_gt, data['hand3d_gt'],
            major_gt
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss)
        (loss_mse_3d, loss_mse_3d_leftcam, loss_mse_3d_rightcam,
         loss_mse_2d_leftcam, loss_mse_2d_rightcam, loss_pinch,
         loss_smooth) = losses
        losses_dict = dict(
            loss_mse_3d=loss_mse_3d,
            loss_mse_3d_leftcam=loss_mse_3d_leftcam,
            loss_mse_3d_rightcam=loss_mse_3d_rightcam,
            loss_mse_2d_leftcam=loss_mse_2d_leftcam,
            loss_mse_2d_rightcam=loss_mse_2d_rightcam,
            loss_pinch=loss_pinch,
            loss_smooth=loss_smooth)

        if self.score_dim:
            losses_dict['loss_major_score'], losses_dict[
                'loss_pinch_score_dist'], losses_dict[
                    'loss_pinch_score_mpjpe'] = self.compute_score_loss(
                        major_pred, major_gt, dist_pred, dist_gt, score)
            # losses_dict['loss_major_score'] = self.compute_score_loss(
            #             major_pred, major_gt, dist_pred, dist_gt, score)
        return losses_dict
