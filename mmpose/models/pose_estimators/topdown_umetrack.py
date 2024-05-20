# Copyright (c) OpenMMLab. All rights reserved.
# import argparse
from itertools import zip_longest
from typing import List, Optional, Tuple

# import numpy as np
# from dataclasses import dataclass
# from typing import Dict, List, Tuple
# import torch
# import torch.nn as nn
# from mmpose.models.models_umetrack import model_utils
# from mmpose.models.models_umetrack.feature_extractor import (
#     FeatureExtractor,FeatureExtractor_New,FeatureExtractor_TM,
#     FeatureExtractor_TM_Res,FeatureExtractor_TM_Res_New,
#     FeatureExtractor_TM_Res_Sub)
# from mmpose.models.models_umetrack.regressor import PoseRegressor
# from mmpose.models.models_umetrack.skeleton_encoder import SkeletonEncoder
# from mmpose.models.models_umetrack.texture_to_coord import Texture_to_Coord
# from mmpose.models.models_umetrack.temporal import (SimpleConvRNN,
#                                                     SimpleConvRNN_New,
#                                                     SimpleConvRNN_CrossView)
# from mmpose.models.models_umetrack.ccf_modules import CrossCueFusion
# import cv2
# import numpy as np
import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.umelib.batched_dataset.data_transform import (ModelInput,
                                                          ModelTarget,
                                                          PerBranchOutput,
                                                          PoseData)
from mmpose.umelib.common.hand import (HandModel, mirrored_hand_model,
                                       scaled_hand_model)
from mmpose.umelib.common.hand_skinning import skin_landmarks
from mmpose.umelib.data_utils import bundles
# from mmpose.umelib.loss.loss import Loss, Loss_Pose, Loss_Pose_xv
# from mmpose.umelib.loss.loss import Loss_Pose_xv
# from mmpose.models.models_umetrack.model_loader import load_pretrained_model
# from mmpose.umelib.models.model_loader import (create_model_coord,
#                                                load_pretrained_model)
# from mmpose.umelib.models.model_loader import (create_model,
#                                                create_model_coord)
from mmpose.umelib.models.umetrack_model import (InputFrameData,
                                                 InputFrameDesc,
                                                 InputSkeletonData)
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 维度扩展
def expand_data_samples_fun(data_samples):
    # O_hand
    expanded_O_joint_rotation_axes_all = []  # 用于存储扩展后的张量
    expanded_O_joint_rest_positions_all = []
    expanded_O_landmark_rest_positions_all = []
    # expanded_O_mesh_vertices_all = []
    expanded_O_joint_frame_index_all = []
    expanded_O_joint_parent_all = []
    expanded_O_joint_first_child_all = []
    expanded_O_joint_next_sibling_all = []
    expanded_O_landmark_rest_bone_weights_all = []
    expanded_O_landmark_rest_bone_indices_all = []
    expanded_O_hand_scale_all = []
    # expanded_O_mesh_triangles_all = []
    # expanded_O_dense_bone_weights_all = []
    # expanded_O_joint_limits_all = []

    # S_hand
    expanded_S_joint_rotation_axes_all = []  # 用于存储扩展后的张量
    expanded_S_joint_rest_positions_all = []
    expanded_S_landmark_rest_positions_all = []
    # expanded_S_mesh_vertices_all = []
    expanded_S_joint_frame_index_all = []
    expanded_S_joint_parent_all = []
    expanded_S_joint_first_child_all = []
    expanded_S_joint_next_sibling_all = []
    expanded_S_landmark_rest_bone_weights_all = []
    expanded_S_landmark_rest_bone_indices_all = []
    expanded_S_hand_scale_all = []
    # expanded_S_mesh_triangles_all = []
    # expanded_S_dense_bone_weights_all = []
    # expanded_S_joint_limits_all = []

    expanded_intrinsics_all = []
    expanded_extrinsics_xf_all = []
    expanded_hand_idx_all = []
    # O_PoseData
    expanded_O_joint_angles_all = []
    expanded_O_wrist_xfs_all = []
    # S_PoseData
    expanded_S_joint_angles_all = []
    expanded_S_wrist_xfs_all = []

    # G_perBranchOutput
    expanded_G_joint_angles_all = []
    expanded_G_wrist_xfs_all = []
    expanded_G_skel_scales_all = []
    expanded_G_pinch_prediction_all = []
    # P_PerBranchOutput
    expanded_P_joint_angles_all = []
    expanded_P_wrist_xfs_all = []
    expanded_P_skel_scales_all = []
    expanded_P_pinch_prediction_all = []

    for data in data_samples:
        # O_hand
        expanded_O_joint_rotation_axes = data.orig_pose_data \
            .left_hand_model.joint_rotation_axes.unsqueeze(0)
        expanded_O_joint_rotation_axes_all.append(
            expanded_O_joint_rotation_axes)

        expanded_O_joint_rest_positions = data.orig_pose_data \
            .left_hand_model.joint_rest_positions.unsqueeze(0)
        expanded_O_joint_rest_positions_all.append(
            expanded_O_joint_rest_positions)

        expanded_O_landmark_rest_positions = data.orig_pose_data \
            .left_hand_model.landmark_rest_positions.unsqueeze(0)
        expanded_O_landmark_rest_positions_all.append(
            expanded_O_landmark_rest_positions)
        # expanded_O_mesh_vertices = data.orig_pose_data \
        #     .left_hand_model.mesh_vertices.unsqueeze(0)
        # expanded_O_mesh_vertices_all.append(
        #     expanded_O_mesh_vertices)
        expanded_O_joint_frame_index = data.orig_pose_data \
            .left_hand_model.joint_frame_index.unsqueeze(0)
        expanded_O_joint_frame_index_all.append(expanded_O_joint_frame_index)

        expanded_O_joint_parent = data.orig_pose_data \
            .left_hand_model.joint_parent.unsqueeze(0)
        expanded_O_joint_parent_all.append(expanded_O_joint_parent)

        expanded_O_joint_first_child = data.orig_pose_data \
            .left_hand_model.joint_first_child.unsqueeze(0)
        expanded_O_joint_first_child_all.append(expanded_O_joint_first_child)

        expanded_O_joint_next_sibling = data.orig_pose_data \
            .left_hand_model.joint_next_sibling.unsqueeze(0)
        expanded_O_joint_next_sibling_all.append(expanded_O_joint_next_sibling)

        expanded_O_landmark_rest_bone_weights = data.orig_pose_data \
            .left_hand_model.landmark_rest_bone_weights.unsqueeze(0)
        expanded_O_landmark_rest_bone_weights_all.append(
            expanded_O_landmark_rest_bone_weights)

        expanded_O_landmark_rest_bone_indices = data.orig_pose_data \
            .left_hand_model.landmark_rest_bone_indices.unsqueeze(0)
        expanded_O_landmark_rest_bone_indices_all.append(
            expanded_O_landmark_rest_bone_indices)

        expanded_O_hand_scale = data.orig_pose_data \
            .left_hand_model.hand_scale.unsqueeze(0)
        expanded_O_hand_scale_all.append(expanded_O_hand_scale)
        # expanded_O_mesh_triangles = data.orig_pose_data \
        #     .left_hand_model.mesh_triangles.unsqueeze(0)
        # expanded_O_mesh_triangles_all.append(
        #     expanded_O_mesh_triangles)
        # expanded_O_dense_bone_weights = data.orig_pose_data \
        #     .left_hand_model.dense_bone_weights.unsqueeze(0)
        # expanded_O_dense_bone_weights_all.append(
        #     expanded_O_dense_bone_weights)
        # expanded_O_joint_limits = data.orig_pose_data \
        #     .left_hand_model.joint_limits.unsqueeze(0)
        # expanded_O_joint_limits_all.append(
        #     expanded_O_joint_limits)

        # S_hand
        expanded_S_joint_rotation_axes = data.s_solved_pose_data \
            .left_hand_model.joint_rotation_axes.unsqueeze(0)
        expanded_S_joint_rotation_axes_all.append(
            expanded_S_joint_rotation_axes)

        expanded_S_joint_rest_positions = data.s_solved_pose_data \
            .left_hand_model.joint_rest_positions.unsqueeze(0)
        expanded_S_joint_rest_positions_all.append(
            expanded_S_joint_rest_positions)
        expanded_S_landmark_rest_positions = data.s_solved_pose_data \
            .left_hand_model.landmark_rest_positions.unsqueeze(0)
        expanded_S_landmark_rest_positions_all.append(
            expanded_S_landmark_rest_positions)

        # expanded_S_mesh_vertices = data.s_solved_pose_data
        # .left_hand_model.mesh_vertices.unsqueeze(0)
        # expanded_S_mesh_vertices_all.append(expanded_S_mesh_vertices)

        expanded_S_joint_frame_index = data.s_solved_pose_data \
            .left_hand_model.joint_frame_index.unsqueeze(0)
        expanded_S_joint_frame_index_all.append(expanded_S_joint_frame_index)
        expanded_S_joint_parent = data.s_solved_pose_data \
            .left_hand_model.joint_parent.unsqueeze(0)
        expanded_S_joint_parent_all.append(expanded_S_joint_parent)
        expanded_S_joint_first_child = data.s_solved_pose_data \
            .left_hand_model.joint_first_child.unsqueeze(0)
        expanded_S_joint_first_child_all.append(expanded_S_joint_first_child)
        expanded_S_joint_next_sibling = data.s_solved_pose_data \
            .left_hand_model.joint_next_sibling.unsqueeze(0)
        expanded_S_joint_next_sibling_all.append(expanded_S_joint_next_sibling)
        expanded_S_landmark_rest_bone_weights = data.s_solved_pose_data \
            .left_hand_model.landmark_rest_bone_weights.unsqueeze(0)
        expanded_S_landmark_rest_bone_weights_all.append(
            expanded_S_landmark_rest_bone_weights)
        expanded_S_landmark_rest_bone_indices = data.s_solved_pose_data \
            .left_hand_model.landmark_rest_bone_indices.unsqueeze(0)
        expanded_S_landmark_rest_bone_indices_all.append(
            expanded_S_landmark_rest_bone_indices)
        expanded_S_hand_scale = data.s_solved_pose_data \
            .left_hand_model.hand_scale.unsqueeze(0)
        expanded_S_hand_scale_all.append(expanded_S_hand_scale)

        # expanded_S_mesh_triangles = data.s_solved_pose_data \
        #     .left_hand_model.mesh_triangles.unsqueeze(0)
        # expanded_S_mesh_triangles_all.append(expanded_S_mesh_triangles)
        # expanded_S_dense_bone_weights = data.s_solved_pose_data \
        #     .left_hand_model.dense_bone_weights.unsqueeze(0)
        # expanded_S_dense_bone_weights_all.append(
        #     expanded_S_dense_bone_weights)
        # expanded_S_joint_limits = data.s_solved_pose_data
        # .left_hand_model.joint_limits.unsqueeze(0)
        # expanded_S_joint_limits_all.append(expanded_S_joint_limits)

        expanded_intrinsics = data.intrinsics.unsqueeze(0)
        expanded_intrinsics_all.append(expanded_intrinsics)
        expanded_extrinsics_xf = data.extrinsics_xf.unsqueeze(0)
        expanded_extrinsics_xf_all.append(expanded_extrinsics_xf)
        expanded_hand_idx = data.hand_idx.unsqueeze(0)
        expanded_hand_idx_all.append(expanded_hand_idx)
        # O_PoseData
        expanded_O_joint_angles = data.orig_pose_data.joint_angles.unsqueeze(0)
        expanded_O_joint_angles_all.append(expanded_O_joint_angles)
        expanded_O_wrist_xfs = data.orig_pose_data.wrist_xfs.unsqueeze(0)
        expanded_O_wrist_xfs_all.append(expanded_O_wrist_xfs)
        # S_PoseData
        expanded_S_joint_angles = data.gt_skel_targets.joint_angles.unsqueeze(
            0)
        expanded_S_joint_angles_all.append(expanded_S_joint_angles)
        expanded_S_wrist_xfs = data.gt_skel_targets.wrist_xfs.unsqueeze(0)
        expanded_S_wrist_xfs_all.append(expanded_S_wrist_xfs)

        # G_perBranchOutput
        expanded_G_joint_angles = data.gt_skel_targets.joint_angles.unsqueeze(
            0)
        expanded_G_joint_angles_all.append(expanded_G_joint_angles)
        expanded_G_wrist_xfs = data.gt_skel_targets.wrist_xfs.unsqueeze(0)
        expanded_G_wrist_xfs_all.append(expanded_G_wrist_xfs)
        expanded_G_skel_scales = data.gt_skel_targets.skel_scales.unsqueeze(0)
        expanded_G_skel_scales_all.append(expanded_G_skel_scales)
        expanded_G_pinch_prediction = data.gt_skel_targets \
            .pinch_prediction.unsqueeze(0)
        expanded_G_pinch_prediction_all.append(expanded_G_pinch_prediction)

        # P_PerBranchOutput
        expanded_P_joint_angles = data.preds_targets.joint_angles.unsqueeze(0)
        expanded_P_joint_angles_all.append(expanded_P_joint_angles)
        expanded_P_wrist_xfs = data.preds_targets.wrist_xfs.unsqueeze(0)
        expanded_P_wrist_xfs_all.append(expanded_P_wrist_xfs)
        expanded_P_skel_scales = data.preds_targets.skel_scales.unsqueeze(0)
        expanded_P_skel_scales_all.append(expanded_P_skel_scales)
        expanded_P_pinch_prediction = data.preds_targets \
            .pinch_prediction.unsqueeze(0)
        expanded_P_pinch_prediction_all.append(expanded_P_pinch_prediction)

    # O_hand
    expand_O_joint_rotation_axes = torch.cat(
        expanded_O_joint_rotation_axes_all, dim=0)
    expand_O_joint_rest_positions = torch.cat(
        expanded_O_joint_rest_positions_all, dim=0)
    expand_O_landmark_rest_positions = torch.cat(
        expanded_O_landmark_rest_positions_all, dim=0)
    # expand_O_mesh_vertices = torch.cat(
    #     expanded_O_joint_mesh_vertices_all, dim=0)
    expand_O_joint_frame_index = torch.cat(
        expanded_O_joint_frame_index_all, dim=0)
    expand_O_joint_parent = torch.cat(expanded_O_joint_parent_all, dim=0)
    expand_O_joint_first_child = torch.cat(
        expanded_O_joint_first_child_all, dim=0)
    expand_O_joint_next_sibling = torch.cat(
        expanded_O_joint_next_sibling_all, dim=0)
    expand_O_landmark_rest_bone_weights = torch.cat(
        expanded_O_landmark_rest_bone_weights_all, dim=0)
    expand_O_landmark_rest_bone_indices = torch.cat(
        expanded_O_landmark_rest_bone_indices_all, dim=0)
    expand_O_hand_scale = torch.cat(expanded_O_hand_scale_all, dim=0)
    # expand_O_mesh_triangles = torch.cat(expanded_O_mesh_triangles_all, dim=0)
    # expand_O_dense_bone_weights = torch.cat(
    #     expanded_O_dense_bone_weights_all, dim=0)
    # expand_O_joint_limits = torch.cat(
    #     expanded_O_joint_limits_all, dim=0)

    # S_hand
    expand_S_joint_rotation_axes = torch.cat(
        expanded_S_joint_rotation_axes_all, dim=0)
    expand_S_joint_rest_positions = torch.cat(
        expanded_S_joint_rest_positions_all, dim=0)
    expand_S_landmark_rest_positions = torch.cat(
        expanded_S_landmark_rest_positions_all, dim=0)
    # expand_S_mesh_vertices = torch.cat(
    #     expanded_S_joint_mesh_vertices_all, dim=0)
    expand_S_joint_frame_index = torch.cat(
        expanded_S_joint_frame_index_all, dim=0)
    expand_S_joint_parent = torch.cat(expanded_S_joint_parent_all, dim=0)
    expand_S_joint_first_child = torch.cat(
        expanded_S_joint_first_child_all, dim=0)
    expand_S_joint_next_sibling = torch.cat(
        expanded_S_joint_next_sibling_all, dim=0)
    expand_S_landmark_rest_bone_weights = torch.cat(
        expanded_S_landmark_rest_bone_weights_all, dim=0)
    expand_S_landmark_rest_bone_indices = torch.cat(
        expanded_S_landmark_rest_bone_indices_all, dim=0)
    expand_S_hand_scale = torch.cat(expanded_S_hand_scale_all, dim=0)
    # expand_S_mesh_triangles = torch.cat(
    #     expanded_S_mesh_triangles_all, dim=0)
    # expand_S_dense_bone_weights = torch.cat(
    #     expanded_S_dense_bone_weights_all, dim=0)
    # expand_S_joint_limits = torch.cat(
    #     expanded_S_joint_limits_all, dim=0)

    expand_intrinsics = torch.cat(expanded_intrinsics_all, dim=0)
    expand_extrinsics_xf = torch.cat(expanded_extrinsics_xf_all, dim=0)
    expand_hand_idx = torch.cat(expanded_hand_idx_all, dim=0)
    # O_PoseData
    expand_O_joint_angles = torch.cat(expanded_O_joint_angles_all, dim=0)
    expand_O_wrist_xfs = torch.cat(expanded_O_wrist_xfs_all, dim=0)
    # S_PoseData
    expand_S_joint_angles = torch.cat(expanded_S_joint_angles_all, dim=0)
    expand_S_wrist_xfs = torch.cat(expanded_S_wrist_xfs_all, dim=0)

    # G_perBranchOutput
    expand_G_joint_angles = torch.cat(expanded_G_joint_angles_all, dim=0)
    expand_G_wrist_xfs = torch.cat(expanded_G_wrist_xfs_all, dim=0)
    expand_G_skel_scales = torch.cat(expanded_G_skel_scales_all, dim=0)
    expand_G_pinch_prediction = torch.cat(
        expanded_G_pinch_prediction_all, dim=0)

    # P_PerBranchOutput

    expand_P_joint_angles = torch.cat(expanded_P_joint_angles_all, dim=0)
    expand_P_wrist_xfs = torch.cat(expanded_P_wrist_xfs_all, dim=0)
    expand_P_skel_scales = torch.cat(expanded_P_skel_scales_all, dim=0)
    expand_P_pinch_prediction = torch.cat(
        expanded_P_pinch_prediction_all, dim=0)

    return (
        expand_O_joint_rotation_axes,
        expand_O_joint_rest_positions,
        expand_O_landmark_rest_positions,
        # expand_O_mesh_vertices,
        expand_O_joint_frame_index,
        expand_O_joint_parent,
        expand_O_joint_first_child,
        expand_O_joint_next_sibling,
        expand_O_landmark_rest_bone_weights,
        expand_O_landmark_rest_bone_indices,
        expand_O_hand_scale,
        # expand_O_mesh_triangles,
        # expand_O_dense_bone_weights,
        # expand_O_joint_limits,
        expand_S_joint_rotation_axes,
        expand_S_joint_rest_positions,
        expand_S_landmark_rest_positions,
        # expand_S_mesh_vertices,
        expand_S_joint_frame_index,
        expand_S_joint_parent,
        expand_S_joint_first_child,
        expand_S_joint_next_sibling,
        expand_S_landmark_rest_bone_weights,
        expand_S_landmark_rest_bone_indices,
        expand_S_hand_scale,
        # expand_S_mesh_triangles,
        # expand_S_dense_bone_weights,
        # expand_S_joint_limits,
        expand_intrinsics,
        expand_extrinsics_xf,
        expand_hand_idx,
        expand_O_joint_angles,
        expand_O_wrist_xfs,
        expand_S_joint_angles,
        expand_S_wrist_xfs,
        expand_G_joint_angles,
        expand_G_wrist_xfs,
        expand_G_skel_scales,
        expand_G_pinch_prediction,
        expand_P_joint_angles,
        expand_P_wrist_xfs,
        expand_P_skel_scales,
        expand_P_pinch_prediction)


# model_input, model_target扩展维度重建
def NewClass(inputs: Tensor, data_samples: SampleList):
    (O_joint_rotation_axes, O_joint_rest_positions, O_landmark_rest_positions,
     O_joint_frame_index, O_joint_parent, O_joint_first_child,
     O_joint_next_sibling, O_landmark_rest_bone_weights,
     O_landmark_rest_bone_indices, O_hand_scale, S_joint_rotation_axes,
     S_joint_rest_positions, S_landmark_rest_positions, S_joint_frame_index,
     S_joint_parent, S_joint_first_child, S_joint_next_sibling,
     S_landmark_rest_bone_weights, S_landmark_rest_bone_indices, S_hand_scale,
     intrinsics, extrinsics_xf, hand_idx, O_joint_angles, O_wrist_xfs,
     S_joint_angles, S_wrist_xfs, G_joint_angles, G_wrist_xfs, G_skel_scales,
     G_pinch_prediction, P_joint_angles, P_wrist_xfs, P_skel_scales,
     P_pinch_prediction) = expand_data_samples_fun(data_samples)

    O_hand = HandModel(
        joint_rotation_axes=O_joint_rotation_axes,
        joint_rest_positions=O_joint_rest_positions,
        joint_frame_index=O_joint_frame_index,
        joint_parent=O_joint_parent,
        joint_first_child=O_joint_first_child,
        joint_next_sibling=O_joint_next_sibling,
        landmark_rest_positions=O_landmark_rest_positions,
        landmark_rest_bone_weights=O_landmark_rest_bone_weights,
        landmark_rest_bone_indices=O_landmark_rest_bone_indices,
        hand_scale=O_hand_scale,
    )
    S_hand = HandModel(
        joint_rotation_axes=S_joint_rotation_axes,
        joint_rest_positions=S_joint_rest_positions,
        joint_frame_index=S_joint_frame_index,
        joint_parent=S_joint_parent,
        joint_first_child=S_joint_first_child,
        joint_next_sibling=S_joint_next_sibling,
        landmark_rest_positions=S_landmark_rest_positions,
        landmark_rest_bone_weights=S_landmark_rest_bone_weights,
        landmark_rest_bone_indices=S_landmark_rest_bone_indices,
        hand_scale=S_hand_scale,
    )

    O_PoseData = PoseData(
        joint_angles=O_joint_angles,  # data.orig_pose_data.joint_angles
        wrist_xfs=O_wrist_xfs,
        left_hand_model=O_hand,
    )
    S_PoseData = PoseData(
        joint_angles=S_joint_angles,  # data.s_solved_pose_data.joint_angles
        wrist_xfs=S_wrist_xfs,
        left_hand_model=S_hand,
    )

    model_input = ModelInput(
        orig_pose_data=O_PoseData,
        s_solved_pose_data=S_PoseData,
        left_images=inputs,
        intrinsics=intrinsics,  # data.intrinsics
        extrinsics_xf=extrinsics_xf,
        hand_idx=hand_idx,  # data.intrinsics
    )

    G_PerBranchOutput = PerBranchOutput(
        # data_samples[0].gt_skel_targets.joint_angles
        joint_angles=G_joint_angles,
        # data_samples[0].gt_skel_targets.wrist_xfs
        wrist_xfs=G_wrist_xfs,
        # data_samples[0].gt_skel_targets.skel_scales
        skel_scales=G_skel_scales,
        # data_samples[0].gt_skel_targets.pinch_prediction
        pinch_prediction=G_pinch_prediction,
    )

    P_PerBranchOutput = PerBranchOutput(
        # data_samples[0].preds_targets.joint_angles
        joint_angles=P_joint_angles,
        # data_samples[0].preds_targets.wrist_xfs
        wrist_xfs=P_wrist_xfs,
        # data_samples[0].preds_targets.skel_scales
        skel_scales=P_skel_scales,
        # data_samples[0].preds_targets.pinch_prediction
        pinch_prediction=P_pinch_prediction,
    )

    model_target = ModelTarget(
        gt_skel_targets=G_PerBranchOutput,
        preds_targets=P_PerBranchOutput,
        intrinsics=intrinsics,  # data.intrinsics
        extrinsics_xf=extrinsics_xf,  # data.intrinsics
    )
    # import ipdb;ipdb.set_trace()
    return model_input, model_target


def _unpack_batched_data(
    training_input: ModelInput, seq_mode: str
) -> List[Tuple[InputFrameData, InputFrameDesc, InputSkeletonData]]:
    # Construct the left hand input images, skeletons and skinned landmarks
    bs = training_input.left_images.shape[0]  # batchsize n
    seq_len = training_input.left_images.shape[1]  # 时间序列长度
    left_images = training_input.left_images.clone()
    # 原始手部姿态数据
    left_hand_model = training_input.orig_pose_data.left_hand_model

    inference_inputs = []
    # 将每帧数据提取并处理->append
    for i_frame in range(seq_len):
        # memory_idx = torch.arange(0, bs, device=left_images.device)
        # use_memory = torch.ones(
        #   bs, device=left_images.device, dtype=torch.bool)
        memory_idx = torch.arange(0, bs)
        use_memory = torch.ones(bs, dtype=torch.bool)
        if i_frame == 0:
            use_memory[:] = False
        if seq_mode == 'multiv':
            nv = 2
        elif seq_mode == 'singlev':
            nv = 1
        else:
            raise ValueError(f'Unknown sequence mode: {seq_mode}')

        sample_range = torch.tensor(
            # [(i * nv, (i + 1) * nv) for i in range(bs)],
            # device=left_images.device
            [(i * nv, (i + 1) * nv) for i in range(bs)])
        # eg:n=3 nv=2,sample_range=torch.tensor([(0, 2), (2, 4), (4, 6)])

        frame_data = InputFrameData(
            left_images=torch.flatten(left_images[:, i_frame, 0:nv], 0, 1),
            # [n,16,2,96,96]->选择第i_frame帧，前nv=2个通道数据
            # [n, 2，96, 96] -> [n*2,96,96]
            intrinsics=torch.flatten(
                training_input.intrinsics[:, i_frame, 0:nv], 0,
                1),  # [n,16,2,3,3]-> [n,2,3,3,] -> [n*2,3,3]
            extrinsics_xf=torch.flatten(
                training_input.extrinsics_xf[:, i_frame, 0:nv], 0,
                1),  # [n,16,2,4,4]-> [n,2,4,4,] -> [n*2,4,4]
        )
        frame_desc = InputFrameDesc(
            hand_idx=training_input.hand_idx[:,
                                             i_frame].long(),  # [n,16]-> [n]
            sample_range=sample_range.long(),  # [n,2]
            memory_idx=memory_idx.long(),  # [n]  tensor([0, 1, 2, 3, 4, n-1])
            use_memory=use_memory,
            # [n]  tensor([False, False, False, False, False, False])  n=6时
        )
        skel_data = InputSkeletonData(
            joint_rotation_axes=left_hand_model.
            joint_rotation_axes[:, i_frame],  # [n,16,22,3]->[n,22,3]
            joint_rest_positions=left_hand_model.
            joint_rest_positions[:, i_frame],  # [n,16,22,3]->[n,22,3]
        )
        inference_inputs.append((frame_data, frame_desc, skel_data))
    # import ipdb;ipdb.set_trace()
    return inference_inputs


def acc(unknown_output,
        known_output_mv,
        known_output_l,
        known_output_r,
        generic_hand_model,
        gt_hand_model,
        target,
        mask=None):

    gt_target = target.gt_skel_targets
    # gt_target = bundles.to device(gt_target, 'cuda')
    # preds_target = target.preds_targets
    gt_keypoints = skin_landmarks(gt_hand_model, gt_target.joint_angles,
                                  gt_target.wrist_xfs)
    # preds_keypoints = skin_landmarks(
    #     generic_hand_model,
    #     preds_target.joint_angles,
    #     preds_target.wrist_xfs
    # )

    known_keypoints_mv = skin_landmarks(
        gt_hand_model,
        known_output_mv.joint_angles,
        known_output_mv.wrist_xfs,
    )
    known_keypoints_l = skin_landmarks(
        gt_hand_model,
        known_output_l.joint_angles,
        known_output_l.wrist_xfs,
    )
    known_keypoints_r = skin_landmarks(
        gt_hand_model,
        known_output_r.joint_angles,
        known_output_r.wrist_xfs,
    )
    unknown_keypoints = skin_landmarks(
        generic_hand_model,
        unknown_output.joint_angles,
        unknown_output.wrist_xfs,
    )

    errors = {}
    total_errors = 0

    # keypoints_diff = preds_keypoints - unknown_keypoints
    keypoints_diff = gt_keypoints - unknown_keypoints
    if mask is not None:
        mask = mask.unsqueeze(2).unsqueeze(2)
        keypoints_diff = keypoints_diff * mask
    keypoint_errors = keypoints_diff.norm(dim=-1).mean()
    keypoint_errors = keypoint_errors * 1000
    errors.update({'unknown_keypoints_errors': keypoint_errors})
    total_errors += keypoint_errors

    keypoints_diff = gt_keypoints - known_keypoints_mv
    keypoint_errors = keypoints_diff.norm(dim=-1).mean()
    keypoint_errors = keypoint_errors * 1000
    errors.update({'known_keypoints_errors_mv': keypoint_errors})
    total_errors += keypoint_errors

    keypoints_diff = gt_keypoints - known_keypoints_l
    keypoint_errors = keypoints_diff.norm(dim=-1).mean()
    keypoint_errors = keypoint_errors * 1000
    errors.update({'known_keypoints_errors_l': keypoint_errors})
    total_errors += keypoint_errors

    keypoints_diff = gt_keypoints - known_keypoints_r
    keypoint_errors = keypoints_diff.norm(dim=-1).mean()
    keypoint_errors = keypoint_errors * 1000
    errors.update({'known_keypoints_errors_r': keypoint_errors})
    total_errors += keypoint_errors

    errors.update({'total_keypoints_errors': total_errors / 4})
    # import ipdb;ipdb.set_trace()
    return errors


# cur_mode = 'multiv' 多视角模式
def _train_batch(model, model_input, model_target, cur_mode, criterion):

    hand_model = mirrored_hand_model(
        model_input.orig_pose_data.left_hand_model,
        model_input.hand_idx == 1,  # right hand is index 1
    )
    hand_model = bundles.to_device(hand_model, 'cuda')

    generic_hand_model = mirrored_hand_model(
        model_input.s_solved_pose_data.left_hand_model,
        model_input.hand_idx == 1,  # right hand is index 1
    )
    generic_hand_model = bundles.to_device(generic_hand_model, 'cuda')

    train_inputs = _unpack_batched_data(model_input, cur_mode)
    model_target = bundles.to_device(model_target, 'cuda')

    unknown_outputs = []
    model.reset_temporal()  # 重置
    for i_step, step_input in enumerate(train_inputs):
        frame_data, frame_desc, skel_input = bundles.to_device(
            step_input, 'cuda')
        # forward
        cur_output = model(frame_data, frame_desc)
        unknown_outputs.append(cur_output)
    unknown_outputs_batched = bundles.collate(unknown_outputs)  # 同一类型p拼为batch
    unknown_outputs_batched = bundles.map_fields(
        lambda t: t.transpose(0, 1) if t is not None else None,
        unknown_outputs_batched,
    )

    scale = unknown_outputs_batched.skel_scales
    generic_hand_model = scaled_hand_model(generic_hand_model, scale)
    loss = criterion(unknown_outputs_batched, generic_hand_model, hand_model,
                     model_target)

    return loss


def _eval_batch(model, model_input, model_target, cur_mode: str, device: str):
    hand_model = mirrored_hand_model(
        model_input.orig_pose_data.left_hand_model,
        model_input.hand_idx == 1,  # right hand is index 1
    )
    hand_model = bundles.to_device(hand_model, device)

    generic_hand_model = mirrored_hand_model(
        model_input.s_solved_pose_data.left_hand_model,
        model_input.hand_idx == 1,  # right hand is index 1
    )
    generic_hand_model = bundles.to_device(generic_hand_model, device)

    inference_inputs = _unpack_batched_data(model_input, cur_mode)
    unknown_outputs = []
    for i_step, step_input in enumerate(inference_inputs):
        frame_data, frame_desc, skel_input = bundles.to_device(
            step_input, device)
        cur_output = model(frame_data, frame_desc)
        unknown_outputs.append(cur_output)
    unknown_outputs_batched = bundles.collate(unknown_outputs)
    unknown_outputs_batched = bundles.map_fields(
        lambda t: t.transpose(0, 1) if t is not None else None,
        unknown_outputs_batched,
    )
    scale = unknown_outputs_batched.skel_scales
    generic_hand_model = scaled_hand_model(generic_hand_model, scale)
    return (
        unknown_outputs_batched,
        generic_hand_model,
        hand_model,
        model_target)


@MODELS.register_module()
class TopdownUmetrack(BasePoseEstimator):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super().__init__(backbone=backbone, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        cur_mode = 'multiv'
        ModelInput, ModelTarget = NewClass(inputs, data_samples)
        loss = _train_batch(
            self.backbone,
            ModelInput,
            ModelTarget,
            cur_mode,
            criterion=self.backbone.loss,
        )
        loss_dict = dict(loss_pose=loss)
        return loss_dict

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        model_input, model_target = NewClass(inputs, data_samples)
        cur_mode = 'multiv'

        (unknown_outputs_batched, generic_hand_model, hand_model,
         model_target) = _eval_batch(
             self.backbone,
             model_input,
             model_target,
             cur_mode,
             device='cuda',
         )

        regression_target = model_target.gt_skel_targets
        regression_target = bundles.to_device(regression_target, 'cuda')

        gt_keypoints = skin_landmarks(hand_model,
                                      regression_target.joint_angles,
                                      regression_target.wrist_xfs)

        batch_gt_instances = []
        for i in range(gt_keypoints.size(0)):
            gt_instances = InstanceData(keypoints_cam=gt_keypoints[i], )
            batch_gt_instances.append(gt_instances)

        for gt_instances, data_sample in zip_longest(batch_gt_instances,
                                                     data_samples):
            data_sample.gt_instances = gt_instances
        unknown_output_keypoints = skin_landmarks(
            generic_hand_model,
            unknown_outputs_batched.joint_angles,
            unknown_outputs_batched.wrist_xfs,
        )

        # keypoints_diff = gt_keypoints - unknown_output_keypoints
        # keypoint_errors = keypoints_diff.norm(dim=-1).mean(dim=(1, 2))
        # keypoint_errors_mm = keypoint_errors * 1000
        # mean_error = keypoint_errors_mm.mean()
        batch_pred_instances = []
        for i in range(unknown_output_keypoints.size(0)):
            pred_instances = InstanceData(
                unknown_keypoints=unknown_output_keypoints[i], )
            batch_pred_instances.append(pred_instances)
        batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)
        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        # output_keypoint_indices = self.test_cfg.get(
        #     'output_keypoint_indices',
        #     None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            # gt_instances = data_sample.gt_instances

            # import ipdb;ipdb.set_trace()
            # # convert keypoint coordinates from input space to image space
            # input_center = data_sample.metainfo['input_center']
            # input_scale = data_sample.metainfo['input_scale']
            # input_size = data_sample.metainfo['input_size']

            # pred_instances.keypoints[..., :2] = \
            #     pred_instances.keypoints[..., :2] / input_size * \
            #         input_scale + input_center - 0.5 * input_scale
            # if 'keypoints_visible' not in pred_instances:
            #     pred_instances.keypoints_visible = \
            #         pred_instances.keypoint_scores

            # if output_keypoint_indices is not None:
            #     # select output keypoints with given indices
            #     num_keypoints = pred_instances.keypoints.shape[1]
            #     for key, value in pred_instances.all_items():
            #         if key.startswith('keypoint'):
            #             pred_instances.set_field(
            #                 value[:, output_keypoint_indices], key)

            # # add bbox information into pred_instances
            # pred_instances.bboxes = gt_instances.bboxes
            # pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances
            # import ipdb;ipdb.set_trace()
            # import ipdb;ipdb.set_trace()
            # if pred_fields is not None:
            #     if output_keypoint_indices is not None:
            #         # select output heatmap channels with keypoint indices
            #         # when the number of heatmap channel
            #         # matches num_keypoints
            #         for key, value in pred_fields.all_items():
            #             if value.shape[0] != num_keypoints:
            #                 continue
            #             pred_fields.set_field(value[output_keypoint_indices],
            #                                   key)
            #     data_sample.pred_fields = pred_fields
        # import ipdb;ipdb.set_trace()
        return batch_data_samples
