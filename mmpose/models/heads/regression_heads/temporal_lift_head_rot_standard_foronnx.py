# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import torch
from heads.regression_heads.lift_head_rot_standard_foronnx import \
    LiftNimbleHeadStandardONNX
from torch import Tensor, nn

from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


@MODELS.register_module()
class TemporalLiftNimbleHeadStandardONNX(LiftNimbleHeadStandardONNX):
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
                 use_plane_coord=True,
                 baseline=0.13,
                 disparity_input=False,
                 plane_arctan=False,
                 reproj_thre=0,
                 iou_thre=0,
                 pad_2d=False,
                 edge_to_center=False,
                 lambda_t: int = -1,
                 corruption_cam: float = 0.5,
                 use_bone_loss: bool = True,
                 use_6d_pose_reg: bool = False,
                 all_use_kp2d_gt: bool = False,
                 seq_len: int = 4,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(
            lift_loss=lift_loss,
            d_ffn=d_ffn,
            undistort=undistort,
            reproj=reproj,
            pose_ncomp=pose_ncomp,
            euler_or_quaternion=euler_or_quaternion,
            use_plane_coord=use_plane_coord,
            baseline=baseline,
            use_svd=use_svd,
            disparity_input=disparity_input,
            plane_arctan=plane_arctan,
            reproj_thre=reproj_thre,
            iou_thre=iou_thre,
            pad_2d=pad_2d,
            edge_to_center=edge_to_center,
            use_bone_loss=use_bone_loss,
            use_6d_pose_reg=use_6d_pose_reg,
            lambda_t=lambda_t,
            all_use_kp2d_gt=all_use_kp2d_gt,
            init_cfg=init_cfg,
        )
        self.seq_len = seq_len

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

    def forward(
        self,
        feats: Tuple[Tensor],
        mems=None,
    ) -> Tensor:
        devices_cuda = feats.device
        feats = self.liftnet(feats)
        B = feats.shape[0]
        if mems is None:
            mems = torch.zeros(B, 2 * self.channel_num, 1, 1).to(devices_cuda)
        feat_mix = torch.cat([feats, mems], dim=1)
        mems = self.temporal(feat_mix)
        output = self.last_layer(feat_mix)
        result = self.simple_feature_layer(output)
        return result, mems

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                mems=None,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            (feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p,
             left_to_right_rt, leftcam_cam_matrix, rightcam_cam_matrix,
             uv_coord_im_pred_global, uv_coord_im_gt_global,
             uv_coord_im_pred_global_distort,
             uv_coord_im_pred_global_distort_noflip, hand3d_gt, left_hand,
             nimble_info, left_R, right_R,
             baseline_scale) = self.preprocess(feats, batch_data_samples,
                                               'predict')

            output, mems = self.forward(feats, mems)

        hand3d_pred = self.simple_postprocess(output, left_hand, left_R,
                                              baseline_scale)
        return hand3d_pred, uv_coord_im_pred_global_distort, mems
