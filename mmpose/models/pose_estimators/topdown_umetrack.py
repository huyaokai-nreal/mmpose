# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import Optional

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
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.umelib.common.hand import (HandModel, mirrored_hand_model,
                                       scaled_hand_model)
from mmpose.umelib.common.hand_skinning import skin_landmarks
from mmpose.umelib.data_utils import bundles
# from mmpose.models.models_umetrack.model_loader import load_pretrained_model
from mmpose.umelib.models.model_loader import load_pretrained_model
from mmpose.umelib.models.umetrack_model import (InputFrameData,
                                                 InputFrameDesc,
                                                 InputSkeletonData)
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator


def expand_data_samples_fun(data_samples):
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
    expanded_joint_angles_all = []
    expanded_wrist_xfs_all = []

    for data in data_samples:
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
        expanded_joint_angles = data.gt_skel_targets.joint_angles.unsqueeze(0)
        expanded_joint_angles_all.append(expanded_joint_angles)
        expanded_wrist_xfs = data.gt_skel_targets.wrist_xfs.unsqueeze(0)
        expanded_wrist_xfs_all.append(expanded_wrist_xfs)

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
    expand_joint_angles = torch.cat(expanded_joint_angles_all, dim=0)
    expand_wrist_xfs = torch.cat(expanded_wrist_xfs_all, dim=0)

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
        expand_joint_angles,
        expand_wrist_xfs)


# 略
def expand_fun(data_samples, path):
    expanded_all = []
    for data in data_samples:
        # path_all =  getattr(data, path)
        expanded = data.path.unsqueeze(0)
        expanded_all.append(expanded)
    expand = torch.cat(expanded_all, dim=0)
    return expand


# left_images, orig_pose_data,intrinsics,extrinsics_xf,hand_idx, seq_mode: str


def _unpack_batched_data(left_images, joint_rotation_axes,
                         joint_rest_positions, intrinsics, extrinsics_xf,
                         hand_idx, seq_mode: str):
    # Construct the left hand input images, skeletons and skinned landmarks
    bs = left_images.shape[0]
    seq_len = left_images.shape[1]
    left_images = left_images.clone()
    # left_hand_model = orig_pose_data.left_hand_model
    # print(left_images.shape)
    inference_inputs = []
    for i_frame in range(seq_len):
        memory_idx = torch.arange(0, bs, device=left_images.device)
        use_memory = torch.ones(
            bs, device=left_images.device, dtype=torch.bool)
        if i_frame == 0:
            use_memory[:] = False

        if seq_mode == 'multiv':
            nv = 2
        elif seq_mode == 'singlev':
            nv = 1
        else:
            raise ValueError(f'Unknown sequence mode: {seq_mode}')

        sample_range = torch.tensor([(i * nv, (i + 1) * nv)
                                     for i in range(bs)],
                                    device=left_images.device)

        frame_data = InputFrameData(
            left_images=torch.flatten(left_images[:, i_frame, 0:nv], 0, 1),
            intrinsics=torch.flatten(intrinsics[:, i_frame, 0:nv], 0, 1),
            extrinsics_xf=torch.flatten(extrinsics_xf[:, i_frame, 0:nv], 0, 1),
        )
        # frame_data = InputFrameData(
        #     left_images=torch.flatten(left_images[:, i_frame, 1:2], 0, 1),
        #     intrinsics=torch.flatten(intrinsics[:, i_frame, 1:2], 0, 1),
        #     extrinsics_xf=torch.flatten(
        #         extrinsics_xf[:, i_frame, 1:2], 0, 1
        #     ),
        # )

        # print(frame_data.left_images.shape)
        # print(frame_data.intrinsics.shape)
        # print(frame_data.extrinsics_xf.shape)
        # assert 1==2
        # import ipdb;ipdb.set_trace()
        frame_desc = InputFrameDesc(
            # hand_idx=hand_idx[:, i_frame].long(),
            hand_idx=hand_idx[:, i_frame].long(),
            sample_range=sample_range.long(),
            memory_idx=memory_idx.long(),
            use_memory=use_memory,
        )

        skel_data = InputSkeletonData(
            # joint_rotation_axes=left_hand_model \
            #     .joint_rotation_axes[:, i_frame],
            # joint_rest_positions=left_hand_model \
            #     .joint_rest_positions[:, i_frame],
            joint_rotation_axes=joint_rotation_axes[:, i_frame],
            joint_rest_positions=joint_rest_positions[:, i_frame],
        )
        inference_inputs.append((frame_data, frame_desc, skel_data))

    return inference_inputs


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
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)

        model_name = ('/home/liyilin/workspace/UmeTrack/pretrained_models/'
                      'pretrained_weights.torch')
        self.model = load_pretrained_model(model_name)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        feats = self.extract_feat(inputs)

        losses = dict()

        if self.with_head:
            losses.update(
                self.head.loss(feats, data_samples, train_cfg=self.train_cfg))

        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        device: str = 'cuda' if torch.cuda.device_count() else 'cpu'
        device = 'cpu'
        self.model.eval()
        self.model.to(device)

        use_skel = True

        (O_joint_rotation_axes, O_joint_rest_positions,
         O_landmark_rest_positions, O_joint_frame_index, O_joint_parent,
         O_joint_first_child, O_joint_next_sibling,
         O_landmark_rest_bone_weights, O_landmark_rest_bone_indices,
         O_hand_scale, S_joint_rotation_axes, S_joint_rest_positions,
         S_landmark_rest_positions, S_joint_frame_index, S_joint_parent,
         S_joint_first_child, S_joint_next_sibling,
         S_landmark_rest_bone_weights, S_landmark_rest_bone_indices,
         S_hand_scale, intrinsics, extrinsics_xf, hand_idx, joint_angles,
         wrist_xfs) = expand_data_samples_fun(data_samples)

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

        hand_model = mirrored_hand_model(
            O_hand,
            hand_idx == 1,
        )
        generic_hand_model = mirrored_hand_model(
            S_hand,
            hand_idx == 1,  # right hand is index 1
        )

        inference_inputs = _unpack_batched_data(inputs, O_joint_rotation_axes,
                                                O_joint_rest_positions,
                                                intrinsics, extrinsics_xf,
                                                hand_idx, 'multiv')
        inference_outputs = []

        # model.reset_temporal()
        for i_step, step_input in enumerate(inference_inputs):
            frame_data, frame_desc, skel_input = bundles.to_device(
                step_input, device)
            # frame_data, frame_desc, skel_input = step_input
            if use_skel:
                cur_output = self.model.regress_pose_use_skeleton(
                    frame_data,
                    frame_desc,
                    skel_input,
                )
            else:
                # assert (cur_mode == 'multiv'
                #         ), 'Skeleton scale prediction requires multiv data'
                cur_output = self.model.regress_pose_pred_skel_scale(
                    frame_data, frame_desc)

            inference_outputs.append(cur_output)

        # import ipdb;ipdb.set_trace()
        inference_outputs_batched = bundles.collate(inference_outputs)

        inference_outputs_batched = bundles.map_fields(
            lambda t: t.transpose(0, 1) if t is not None else None,
            inference_outputs_batched,
        )

        if not use_skel:
            scale = inference_outputs_batched.skel_scales
            generic_hand_model = scaled_hand_model(generic_hand_model, scale)

        # gt_keypoints = skin_landmarks(hand_model, joint_angles, wrist_xfs)

        if use_skel:
            output_keypoints = skin_landmarks(
                hand_model,
                inference_outputs_batched.joint_angles,
                inference_outputs_batched.wrist_xfs,
            )
        else:
            # preds_target = model_target.preds_targets
            # gt_keypoints = skin_landmarks(
            #     generic_hand_model, preds_target.joint_angles,
            #     preds_target.wrist_xfs
            # )
            output_keypoints = skin_landmarks(
                generic_hand_model,
                inference_outputs_batched.joint_angles,
                inference_outputs_batched.wrist_xfs,
            )
        import ipdb
        ipdb.set_trace()
        return output_keypoints

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
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            input_center = data_sample.metainfo['input_center']
            input_scale = data_sample.metainfo['input_scale']
            input_size = data_sample.metainfo['input_size']

            pred_instances.keypoints[..., :2] = \
                pred_instances.keypoints[..., :2] / input_size * input_scale \
                + input_center - 0.5 * input_scale
            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = \
                    pred_instances.keypoint_scores

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
