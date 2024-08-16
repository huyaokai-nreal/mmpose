# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import torch
from torch import Tensor, nn

# from mmpose.post_process.temporal_filters import build_filter
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions
from .lift_head_standard_ori import LiftHeadStandardOri


@MODELS.register_module()
class TemporalLiftHeadStandardOri(LiftHeadStandardOri):
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
                 enhance_static=False,
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
            pad_2d=pad_2d)
        self.score_dim=score_dim
        self.enhance_static = enhance_static
        self.static_data_date_list = ['20240516', '20240517', '20240522']
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
            feat_mix = torch.cat([feat, mems], dim=1)
            mems = self.temporal(feat_mix)
            output = self.last_layer(feat_mix)
            outputs[:, i, ...] = output
        outputs = outputs.reshape(B * seq_len, -1, 1, 1)
        score = torch.ones(B, self.score_dim)
        return outputs, mems, score

    def _forward(self, feats, mems):
        feats = self.liftnet(feats)
        B = feats.shape[0]
        if mems is None:
            mems = torch.zeros(B, 2 * self.channel_num, 1, 1).cuda()

        feat_mix = torch.cat([feats, mems], dim=1)
        mems = self.temporal(feat_mix)
        output = self.last_layer(feat_mix)
        output = output.reshape(B, -1, 1, 1) / self.baseline
        return output, mems

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'predict')
        output, mems, score = self.forward(data['feats'], mems, 1)
        hand3d_pred, leftcam_XYZ, rightcam_XYZ = self.postprocess(output, data['norm_leftcam_xyz'],
                                data['norm_rightcam_xyz'], data['left_R'], data['right_R'],
                                data['lr_rot_matrix'], data['lr_p'], data['baseline_scale'])
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
        output, _, score = self.forward(data['feats'], None,
                                                 self.seq_len)
        hand3d_pred, leftcam_XYZ, rightcam_XYZ = self.postprocess(output, data['norm_leftcam_xyz'],
                                data['norm_rightcam_xyz'], data['left_R'], data['right_R'],
                                data['lr_rot_matrix'], data['lr_p'], data['baseline_scale'])
        hand3d_gt = data['hand3d_gt']
        # pinch distance, no norm
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(
            hand3d_gt[:, 4, :] - hand3d_gt[:, 8, :], dim=-1)

        if self.enhance_static:
            static_weight = 25
            static_mask = self.generate_static_mask(batch_data_samples)
            _hand3d_pred = self.enhanced_fun(
                hand3d_pred, static_mask, static_weight)
            _hand3d_gt = self.enhanced_fun(
                hand3d_gt, static_mask, static_weight)
        else:
            _hand3d_gt = hand3d_gt
            _hand3d_pred = hand3d_pred
        pred_for_loss = [
            hand3d_pred, leftcam_XYZ, rightcam_XYZ,
            dist_pred, _hand3d_pred
        ]
        targ_for_loss = [
            hand3d_gt, hand3d_gt, hand3d_gt, dist_gt, _hand3d_gt
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss)
        (loss_mse_3d, loss_mse_3d_leftcam, loss_mse_3d_rightcam, loss_pinch,
         loss_smooth) = losses
        losses_dict = dict(
            loss_mse_3d=loss_mse_3d,
            loss_mse_3d_leftcam=loss_mse_3d_leftcam,
            loss_mse_3d_rightcam=loss_mse_3d_rightcam,
            loss_pinch=loss_pinch,
            loss_smooth=loss_smooth)
        return losses_dict

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