# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmpose.models.heads.nimble.nimble_utils import decode_svd
from mmpose.models.heads.regression_heads.lift_head_rot_standard import \
    LiftNimbleHeadStandard
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


@MODELS.register_module()
class LiftNimbleHeadStandardONNX(LiftNimbleHeadStandard):
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
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(
            lift_loss=lift_loss,
            d_ffn=d_ffn,
            undistort=undistort,
            kpt2d_with_depth=kpt2d_with_depth,
            shape_ncomp=shape_ncomp,
            pose_ncomp=pose_ncomp,
            reg_shape_type=reg_shape_type,
            use_svd=use_svd,
            use_nimble_part_para=use_nimble_part_para,
            use_pose_pca=use_pose_pca,
            euler_or_quaternion=euler_or_quaternion,
            reproj=reproj,
            use_plane_coord=use_plane_coord,
            baseline=baseline,
            disparity_input=disparity_input,
            plane_arctan=plane_arctan,
            reproj_thre=reproj_thre,
            iou_thre=iou_thre,
            pad_2d=pad_2d,
            edge_to_center=edge_to_center,
            lambda_t=lambda_t,
            use_bone_loss=use_bone_loss,
            use_6d_pose_reg=use_6d_pose_reg,
            all_use_kp2d_gt=all_use_kp2d_gt,
            init_cfg=init_cfg,
        )

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        output = self.liftnet(feats)
        output = self.last_layer(output).view((feats.shape[0], -1, 1, 1))
        output = self.simple_feature_layer(output)
        return output

    def simple_feature_layer(self, output):

        pose_len = self.pose_ncomp
        rot_vector_t = output[:, :pose_len, 0, 0]
        svd_begin = self.pose_ncomp + self.shape_ncomp
        shape_v = output[:, pose_len:svd_begin, 0, 0]
        pre_pt_features = output[:, svd_begin:, 0, 0]
        pre_rot_vector = self.nimble_layer.generate_full_pose_foronnx(
            rot_vector_t).view(-1, 20, 3)

        _, bone_joints = self.nimble_layer.forward_simple_foronnx(
            pre_rot_vector, shape_v)
        rebuild_joints = bone_joints[:, self.kp_index, :]
        root_rebuild_joints = rebuild_joints[:, 0:1, :]
        rebuild_joints_temp = rebuild_joints - root_rebuild_joints
        rebuild_joints_temp = rebuild_joints_temp / self.scale_parameter
        out_fea = torch.cat(
            (rebuild_joints_temp, pre_pt_features.unsqueeze(-1)), dim=-1)
        return out_fea

    def simple_postprocess(self, output, left_hand, left_R, baseline_scale):

        B = output.shape[0]
        cuda_device = output.device

        rebuild_joints_temp = output[:, :, :3]
        pre_pt_features = output[:, :, -1]
        matrix_svd = decode_svd(
            pre_pt_features,
            self.rigid_samples,
        )
        root_xyz = matrix_svd[:, 0:3, 3]
        root_matrix = matrix_svd[:, 0:3, 0:3]

        mask = left_hand == 1
        add_matrix = torch.eye(3).unsqueeze(0).expand(B, -1,
                                                      -1).to(cuda_device)
        add_matrix[mask, 0, 0] = -add_matrix[mask, 0, 0]
        root_matrix = torch.matmul(torch.inverse(left_R), root_matrix)
        root_matrix = torch.matmul(add_matrix, root_matrix)
        rebuild_joints_temp = torch.matmul(rebuild_joints_temp,
                                           root_matrix.transpose(1, 2))

        new_root_xyz = torch.bmm(
            root_xyz.unsqueeze(1),
            torch.inverse(left_R).permute(0, 2, 1))
        xyz_point = rebuild_joints_temp + new_root_xyz
        xyz_point *= baseline_scale.unsqueeze(-1).unsqueeze(-1)
        return xyz_point

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
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
            output = self.forward(feats)

        hand3d_pred = self.simple_postprocess(output, left_hand, left_R,
                                              baseline_scale)
        return hand3d_pred, uv_coord_im_pred_global_distort
