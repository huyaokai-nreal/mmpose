# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import torch
from torch import Tensor, nn

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
                 use_plane_coord=True,
                 baseline=0.13,
                 corruption_cam: float = 0.5,
                 perturb_right_use_2d_gt: bool = False,
                 seq_len: int = 4,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(
            lift_loss,
            num_layers,
            d_ffn,
            output_num,
            reproj,
            use_plane_coord,
            baseline,
            perturb_right_use_2d_gt=perturb_right_use_2d_gt,
            all_use_kp2d_gt=False,
            corruption_cam=corruption_cam,
            init_cfg=init_cfg)
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

    def forward(self,
                feats: Tuple[Tensor],
                mems=None,
                seq_len: int = 1) -> Tensor:
        feats = self.liftnet(feats)
        B = int(feats.shape[0] / seq_len)
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
        return outputs, mems

    def _forward(self, feats, mems):
        feats = self.liftnet(feats)
        B = feats.shape[0]
        if mems is None:
            mems = torch.zeros(B, 2 * self.channel_num, 1, 1).cuda()
        feats = feats.view(B, 1, -1)
        outputs = torch.zeros((B, 1, 42, 1, 1)).cuda()
        for i in range(1):
            feat = feats[:, i:i + 1, :].reshape(B, -1, 1, 1)
            feat_mix = torch.concatenate([feat, mems], dim=1)
            mems = self.temporal(feat_mix)
            output = self.last_layer(feat_mix)
            outputs[:, i, ...] = output
        outputs = outputs.reshape(B, -1, 1, 1)
        return outputs, mems

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            (feats, norm_leftcam_xyz, norm_rightcam_xyz, lr_rot_matrix, lr_p,
             leftcam_cam_matrix, rightcam_cam_matrix, uv_coord_im_pred_global,
             uv_coord_im_pred_global_distort, hand3d_gt, left_R, right_R,
             baseline_scale) = self.preprocess(feats, batch_data_samples)
        output, mems = self.forward(feats, mems, 1)
        hand3d_pred = self.postprocess(output, norm_leftcam_xyz,
                                       norm_rightcam_xyz, left_R, right_R,
                                       lr_rot_matrix, lr_p, baseline_scale)[0]
        if self.reproj:
            camera_model = batch_data_samples[0].meta['ori_camera']
            leftcam_uv_reproj_distort = camera_model.eye_to_window(
                hand3d_pred.cpu().numpy())
            leftcam_uv_reproj_distort = torch.tensor(
                leftcam_uv_reproj_distort).cuda()
            return hand3d_pred, leftcam_uv_reproj_distort[:, None, ...], mems
        else:
            return hand3d_pred, uv_coord_im_pred_global_distort, mems

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        with torch.no_grad():
            (feats, norm_leftcam_xyz, norm_rightcam_xyz, lr_rot_matrix, lr_p,
             leftcam_cam_matrix, rightcam_cam_matrix, uv_coord_im_pred_global,
             uv_coord_im_pred_global_distort, hand3d_gt, left_R, right_R,
             baseline_scale) = self.preprocess(feats, batch_data_samples)
        output, _ = self.forward(feats, None, self.seq_len)
        hand3d_pred, leftcam_XYZ, rightcam_XYZ = self.postprocess(
            output, norm_leftcam_xyz, norm_rightcam_xyz, left_R, right_R,
            lr_rot_matrix, lr_p, baseline_scale)
        leftcam_uv_reproj = torch.matmul(hand3d_pred,
                                         leftcam_cam_matrix.permute(0, 2, 1))
        leftcam_uv_reproj = leftcam_uv_reproj[..., :2] / leftcam_uv_reproj[...,
                                                                           2:]

        rightcam_uv_reproj = torch.matmul(
            hand3d_pred, lr_rot_matrix) - torch.matmul(
                lr_rot_matrix.permute(0, 2, 1), lr_p.unsqueeze(-1)).reshape(
                    (-1, 1, 3))
        rightcam_uv_reproj = torch.matmul(rightcam_uv_reproj,
                                          rightcam_cam_matrix.permute(0, 2, 1))
        rightcam_uv_reproj = rightcam_uv_reproj[..., :2] / rightcam_uv_reproj[
            ..., 2:]

        leftcam_uv_gt = uv_coord_im_pred_global[:, 0]
        rightcam_uv_gt = uv_coord_im_pred_global[:, 1]

        major_gt = torch.cat(
            (hand3d_gt[:, 1:10, :], hand3d_gt[:, 13, :].unsqueeze(1)), dim=1)
        major_pred = torch.cat(
            (hand3d_pred[:, 1:10, :], hand3d_pred[:, 13, :].unsqueeze(1)),
            dim=1)

        # origin distance, no norm
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(hand3d_gt[:, 4, :] - hand3d_gt[:, 8, :], dim=-1)
        pred_for_loss = [
            hand3d_pred, leftcam_XYZ, rightcam_XYZ, leftcam_uv_reproj,
            rightcam_uv_reproj, dist_pred, hand3d_pred, major_pred
        ]
        targ_for_loss = [
            hand3d_gt, hand3d_gt, hand3d_gt, leftcam_uv_gt, rightcam_uv_gt,
            dist_gt, hand3d_gt, major_gt
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

        return losses_dict
