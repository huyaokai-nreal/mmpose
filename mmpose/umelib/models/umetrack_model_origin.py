# flake8: noqa
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import torch
import torch.nn as nn

from . import model_utils
from .ccf_modules import CrossCueFusion
from .feature_extractor import (FeatureExtractor, FeatureExtractor_New,
                                FeatureExtractor_TM, FeatureExtractor_TM_Res,
                                FeatureExtractor_TM_Res_New,
                                FeatureExtractor_TM_Res_Sub)
from .regressor import PoseRegressor
from .skeleton_encoder import SkeletonEncoder
from .temporal import SimpleConvRNN, SimpleConvRNN_CrossView, SimpleConvRNN_New
from .texture_to_coord import Texture_to_Coord


@dataclass
class InputFrameData:
    """Each entry corresponds to data from a camera. The data here doesn't
    contain which images are observing the same hand. InputFrameDesc is used to
    assemble features observing the same hands.

    * left_images (shape [n_images, h, w])
    * intrinsics (shape [n_images, 3, 3])
    * extrinsics_xf (shape [n_images, 4, 4])
    """

    left_images: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics_xf: torch.Tensor


# per-frame data descriptions, could potentially
# create another struct InputFrameDescription
@dataclass
class InputFrameDesc:
    """
    Descriptions for InputFrameData. Each tensor should
    bs: batch_size

    * sample_range (shape [bs, 2]): the 2 columns are the starting and
        ending indices. Example: a tensor [[0, 2], [2, 3]] means first sample
        corresponds to left_images[0:2] which is a multi-view sample and second
        sample corresponds to lefts_images[2:3] which is a single-view sample
    * memory_idx (shape [bs]): only applicable with a valid _temporal field. In
        rum-time if we have tracking for 2 hands, this tensor could be [0, 1].
        If the next frame left hand loses track, this memory_idx could become [1] tensor
    * use_memory (shape [bs]): a boolean tensor indicating whether to use the memory
        features for this sample
    * hand_idx (shape [bs]): hand index for each sample. There is a chance to factor this out.
    """

    sample_range: torch.Tensor
    memory_idx: torch.Tensor
    use_memory: torch.Tensor
    hand_idx: torch.Tensor


@dataclass
class InputSkeletonData:
    """Descriptions for InputFrameData.

    * joint_rotation_axes (shape [bs, 22, 3]): 22 joint axes
    * joint_rest_positions (shape [bs, 22, 3]): 22 joint positions in rest pose
    """

    joint_rotation_axes: torch.Tensor
    joint_rest_positions: torch.Tensor


def _recover_wrist_xfs_in_world(
    hand_idx: torch.Tensor,
    cam0_extrinsics: torch.Tensor,
    left_wrist_xfs_in_cam0: torch.Tensor,
) -> torch.Tensor:
    left_wrist_xf_world = torch.inverse(
        cam0_extrinsics) @ left_wrist_xfs_in_cam0

    # The model only makes predictions for the left hands. In order to recover
    # the right hand transform, the x component needs to be mirrored in the final
    # transformation matrix.
    right_hand_masks = hand_idx == 1
    wrist_xf_world = left_wrist_xf_world.clone()
    wrist_xf_world[right_hand_masks, :,
                   0] = -1 * wrist_xf_world[right_hand_masks, :, 0].clone()
    return wrist_xf_world


def _get_cam0_extrinsics(frame_data: InputFrameData,
                         frame_desc: InputFrameDesc) -> torch.Tensor:
    # Extracting reference cam extrinsics
    return frame_data.extrinsics_xf[
        frame_desc.sample_range[:, 0].clone()].clone()


def _get_cam1_extrinsics(frame_data: InputFrameData,
                         frame_desc: InputFrameDesc) -> torch.Tensor:
    # Extracting reference cam extrinsics
    return frame_data.extrinsics_xf[frame_desc.sample_range[:, 1] - 1].clone()


class UmeTrackModel(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        temporal: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_u: PoseRegressor,
    ):
        super().__init__()

        self._feature_extractor: FeatureExtractor = feature_extractor
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_u = regressor_u

        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   sample_range: torch.Tensor) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        extrinsics_xf = frame_data.extrinsics_xf

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )

        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    # singlev_features = self._feature_extractor.compute_multiv_features(
                    #     torch.cat([per_view_img_features[r0:r1].clone(),per_view_img_features[r0:r1].clone()],dim=0).unsqueeze(0),
                    #     torch.cat([singlev_scaled_to_orig_xf[r0:r1].clone(),singlev_scaled_to_orig_xf[r0:r1].clone()],dim=0).unsqueeze(0),
                    #     torch.cat([extrinsics_xf[r0:r1].clone(),extrinsics_xf[r0:r1].clone()],dim=0).unsqueeze(0),
                    # )
                    # img_features_list.append(singlev_features)
                    singlev_features = self._feature_extractor.compute_singlev_features(
                        per_view_img_features[r0:r1].clone(),
                        singlev_scaled_to_orig_xf[r0:r1].clone())
                    img_features_list.append(singlev_features)
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_all(self, frame_data: InputFrameData
                                       ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        extrinsics_xf = frame_data.extrinsics_xf

        img_features_multiv = self._feature_extractor.compute_multiv_features(
            per_view_img_features.reshape((-1, 2) +
                                          per_view_img_features.shape[1:]),
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )

        img_features_l = self._feature_extractor.compute_singlev_features(
            per_view_img_features[0::2].clone(),
            singlev_scaled_to_orig_xf[0::2].clone())

        img_features_r = self._feature_extractor.compute_singlev_features(
            per_view_img_features[1::2].clone(),
            singlev_scaled_to_orig_xf[1::2].clone())
        # img_features = torch.cat([img_features_multiv,img_features_l,img_features_r])

        return img_features_multiv, img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data,
                                                       frame_desc.sample_range)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_multiv = self._temporal.forward_temporal_features_multiv(
            feats_multiv,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        temporal_features_l = self._temporal.forward_temporal_features_l(
            feats_l,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        temporal_features_r = self._temporal.forward_temporal_features_r(
            feats_r,
            extrinsics_1,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features_multiv, temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)
        # temporal_features = self._forward_feature_extractor(
        #     frame_data, frame_desc.sample_range
        # )

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and feats_multiv.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_multiv.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)

        img_skel_features_multiv = torch.cat(
            [feats_multiv.clone(), skel_features.clone()], dim=1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features = torch.cat([
            img_skel_features_multiv, img_skel_features_l, img_skel_features_r
        ],
                                      dim=0)

        unknown_output = self._regressor_u.regress_poses(feats_multiv)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses_smv(img_skel_features)
        known_output_multiv = known_output[0]
        known_output_l = known_output[1]
        known_output_r = known_output[2]

        known_output_multiv.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_multiv.wrist_xfs,
        )
        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output_multiv': known_output_multiv,
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_coord(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        texture_to_coord: Texture_to_Coord,
        temporal: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_u: PoseRegressor,
    ):
        super().__init__()
        print('UmeTrackModel_coord')
        self._feature_extractor: FeatureExtractor = feature_extractor
        self._textture_to_coord = texture_to_coord
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_u = regressor_u

        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   sample_range: torch.Tensor) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        extrinsics_xf = frame_data.extrinsics_xf

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )

        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    # singlev_features = self._feature_extractor.compute_multiv_features(
                    #     torch.cat([per_view_img_features[r0:r1].clone(),per_view_img_features[r0:r1].clone()],dim=0).unsqueeze(0),
                    #     torch.cat([singlev_scaled_to_orig_xf[r0:r1].clone(),singlev_scaled_to_orig_xf[r0:r1].clone()],dim=0).unsqueeze(0),
                    #     torch.cat([extrinsics_xf[r0:r1].clone(),extrinsics_xf[r0:r1].clone()],dim=0).unsqueeze(0),
                    # )
                    # img_features_list.append(singlev_features)
                    singlev_features = self._feature_extractor.compute_singlev_features(
                        per_view_img_features[r0:r1].clone(),
                        singlev_scaled_to_orig_xf[r0:r1].clone())
                    img_features_list.append(singlev_features)
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_all(self, frame_data: InputFrameData
                                       ) -> torch.Tensor:
        # Per-view img features
        img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))
        per_view_img_features = self._textture_to_coord(img_features)

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        extrinsics_xf = frame_data.extrinsics_xf

        img_features_multiv = self._feature_extractor.compute_multiv_features(
            per_view_img_features.reshape((-1, 2) +
                                          per_view_img_features.shape[1:]),
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )

        img_features_l = self._feature_extractor.compute_singlev_features(
            per_view_img_features[0::2].clone(),
            singlev_scaled_to_orig_xf[0::2].clone())

        img_features_r = self._feature_extractor.compute_singlev_features(
            per_view_img_features[1::2].clone(),
            singlev_scaled_to_orig_xf[1::2].clone())
        # img_features = torch.cat([img_features_multiv,img_features_l,img_features_r])

        return img_features_multiv, img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data,
                                                       frame_desc.sample_range)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_multiv = self._temporal.forward_temporal_features_multiv(
            feats_multiv,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        temporal_features_l = self._temporal.forward_temporal_features_l(
            feats_l,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        temporal_features_r = self._temporal.forward_temporal_features_r(
            feats_r,
            extrinsics_1,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features_multiv, temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)
        # temporal_features = self._forward_feature_extractor(
        #     frame_data, frame_desc.sample_range
        # )

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and feats_multiv.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_multiv.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)

        img_skel_features_multiv = torch.cat(
            [feats_multiv.clone(), skel_features.clone()], dim=1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features = torch.cat([
            img_skel_features_multiv, img_skel_features_l, img_skel_features_r
        ],
                                      dim=0)

        unknown_output = self._regressor_u.regress_poses(feats_multiv)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses_smv(img_skel_features)
        known_output_multiv = known_output[0]
        known_output_l = known_output[1]
        known_output_r = known_output[2]

        known_output_multiv.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_multiv.wrist_xfs,
        )
        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output_multiv': known_output_multiv,
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_Fuse(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        temporal: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_u: PoseRegressor,
    ):
        super().__init__()

        self._feature_extractor: FeatureExtractor = feature_extractor
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_u = regressor_u

        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   sample_range: torch.Tensor) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        extrinsics_xf = frame_data.extrinsics_xf

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )

        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    singlev_features = self._feature_extractor.compute_singlev_features(
                        per_view_img_features[r0:r1].clone(),
                        singlev_scaled_to_orig_xf[r0:r1].clone())
                    img_features_list.append(singlev_features)
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_all(self, frame_data: InputFrameData
                                       ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        # extrinsics_xf = frame_data.extrinsics_xf

        # img_features_multiv = self._feature_extractor.compute_multiv_features(
        #         per_view_img_features.reshape(
        #             (-1, 2) + per_view_img_features.shape[1:]
        #         ),
        #         singlev_scaled_to_orig_xf.reshape(
        #             (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]
        #         ),
        #         extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        #     )

        img_features_l = self._feature_extractor.compute_singlev_features(
            per_view_img_features[0::2].clone(),
            singlev_scaled_to_orig_xf[0::2].clone())

        img_features_r = self._feature_extractor.compute_singlev_features(
            per_view_img_features[1::2].clone(),
            singlev_scaled_to_orig_xf[1::2].clone())

        return img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data,
                                                       frame_desc.sample_range)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_l, feats_r = self._forward_feature_extractor_all(frame_data)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)
        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)
        extrinsics_xf = frame_data.extrinsics_xf
        # Temporal features

        temporal_features_l = self._temporal.forward_temporal_features_l(
            feats_l,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        temporal_features_r = self._temporal.forward_temporal_features_r(
            feats_r,
            extrinsics_1,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        per_view_features = torch.stack(
            [temporal_features_l.clone(),
             temporal_features_r.clone()], dim=1)
        temporal_features_multiv = self._temporal.compute_multiv_features(
            per_view_features,
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )

        return temporal_features_multiv, temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and feats_multiv.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_multiv.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)

        img_skel_features_multiv = torch.cat(
            [feats_multiv.clone(), skel_features.clone()], dim=1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features = torch.cat([
            img_skel_features_multiv, img_skel_features_l, img_skel_features_r
        ],
                                      dim=0)

        unknown_output = self._regressor_u.regress_poses(feats_multiv)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses_smv(img_skel_features)
        known_output_multiv = known_output[0]
        known_output_l = known_output[1]
        known_output_r = known_output[2]

        known_output_multiv.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_multiv.wrist_xfs,
        )
        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output_multiv': known_output_multiv,
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_CrossView(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        temporal: SimpleConvRNN_CrossView,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_u: PoseRegressor,
    ):
        super().__init__()

        self._feature_extractor: FeatureExtractor = feature_extractor
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_u = regressor_u

        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   sample_range: torch.Tensor) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        extrinsics_xf = frame_data.extrinsics_xf

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )

        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    singlev_features = self._feature_extractor.compute_singlev_features(
                        per_view_img_features[r0:r1].clone(),
                        singlev_scaled_to_orig_xf[r0:r1].clone())
                    img_features_list.append(singlev_features)
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_all(self, frame_data: InputFrameData
                                       ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        extrinsics_xf = frame_data.extrinsics_xf

        img_features_multiv = self._feature_extractor.compute_multiv_features(
            per_view_img_features.reshape((-1, 2) +
                                          per_view_img_features.shape[1:]),
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )

        img_features_l = self._feature_extractor.compute_singlev_features(
            per_view_img_features[0::2].clone(),
            singlev_scaled_to_orig_xf[0::2].clone())

        img_features_r = self._feature_extractor.compute_singlev_features(
            per_view_img_features[1::2].clone(),
            singlev_scaled_to_orig_xf[1::2].clone())
        # img_features = torch.cat([img_features_multiv,img_features_l,img_features_r])

        return img_features_multiv, img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data,
                                                       frame_desc.sample_range)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_multiv, temporal_features_l, temporal_features_r = self._temporal.forward_temporal_features_all(
            feats_multiv,
            feats_l,
            feats_r,
            extrinsics_0,
            extrinsics_1,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features_multiv, temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and feats_multiv.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_multiv.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)

        img_skel_features_multiv = torch.cat(
            [feats_multiv.clone(), skel_features.clone()], dim=1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features = torch.cat([
            img_skel_features_multiv, img_skel_features_l, img_skel_features_r
        ],
                                      dim=0)

        unknown_output = self._regressor_u.regress_poses(feats_multiv)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses_smv(img_skel_features)
        known_output_multiv = known_output[0]
        known_output_l = known_output[1]
        known_output_r = known_output[2]

        known_output_multiv.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_multiv.wrist_xfs,
        )
        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output_multiv': known_output_multiv,
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_CCF(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        temporal: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_u: PoseRegressor,
    ):
        super().__init__()

        self._feature_extractor: FeatureExtractor = feature_extractor
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_u = regressor_u

        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   sample_range: torch.Tensor) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        extrinsics_xf = frame_data.extrinsics_xf

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )

        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    singlev_features = self._feature_extractor.compute_singlev_features(
                        per_view_img_features[r0:r1].clone(),
                        singlev_scaled_to_orig_xf[r0:r1].clone())
                    img_features_list.append(singlev_features)
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_all(self, frame_data: InputFrameData
                                       ) -> torch.Tensor:
        # print(frame_data.left_images.device)
        # if frame_data.left_images.device == torch.device('cuda:0'):
        #     cv2.imwrite('test_1.png',frame_data.left_images[0].cpu().numpy()*255)
        #     cv2.imwrite('test_2.png',frame_data.left_images[32].cpu().numpy()*255)
        # assert 1==2
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))
        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        extrinsics_xf = frame_data.extrinsics_xf
        # img_features_l = per_view_img_features[0::2].clone()
        # img_features_r = per_view_img_features[1::2].clone()
        # singlev_scaled_to_orig_xf_l = singlev_scaled_to_orig_xf[0::2].clone()
        # singlev_scaled_to_orig_xf_r = singlev_scaled_to_orig_xf[1::2].clone()
        # extrinsics_xf_l = extrinsics_xf[0::2].clone()
        # extrinsics_xf_r = extrinsics_xf[1::2].clone()

        img_features_multiv = self._feature_extractor.compute_multiv_features(
            per_view_img_features.reshape((-1, 2) +
                                          per_view_img_features.shape[1:]),
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )

        img_features_l = self._feature_extractor.compute_singlev_features(
            per_view_img_features[0::2].clone(),
            singlev_scaled_to_orig_xf[0::2].clone())

        img_features_r = self._feature_extractor.compute_singlev_features(
            per_view_img_features[1::2].clone(),
            singlev_scaled_to_orig_xf[1::2].clone())
        # img_features = torch.cat([img_features_multiv,img_features_l,img_features_r])

        return img_features_multiv, img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data,
                                                       frame_desc.sample_range)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_multiv = self._temporal.forward_temporal_features_multiv(
            feats_multiv,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        temporal_features_l = self._temporal.forward_temporal_features_l(
            feats_l,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        temporal_features_r = self._temporal.forward_temporal_features_r(
            feats_r,
            extrinsics_1,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features_multiv, temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and feats_multiv.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_multiv.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)

        img_skel_features_multiv = torch.cat(
            [feats_multiv.clone(), skel_features.clone()], dim=1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features = torch.cat([
            img_skel_features_multiv, img_skel_features_l, img_skel_features_r
        ],
                                      dim=0)

        unknown_output = self._regressor_u.regress_poses(feats_multiv)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses_smv(img_skel_features)
        known_output_multiv = known_output[0]
        known_output_l = known_output[1]
        known_output_r = known_output[2]

        known_output_multiv.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_multiv.wrist_xfs,
        )
        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output_multiv': known_output_multiv,
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_TM(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor_TM,
        temporal: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_u: PoseRegressor,
    ):
        super().__init__()
        print('UmetrackModel_TM')
        self._feature_extractor: FeatureExtractor_TM = feature_extractor
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_u = regressor_u

        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)
        self._prev_s2o_l = torch.empty(0)
        self._prev_s2o_r = torch.empty(0)
        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   frame_desc: InputFrameDesc) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)
        extrinsics_xf = frame_data.extrinsics_xf
        sample_range = frame_desc.sample_range
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory

        required_memory_len = int(torch.max(memory_idx)) + 1

        if not use_memory.all():
            # print('***********')
            self._mem_features_l = per_view_img_features.clone()
            self._prev_extrinsics_l = extrinsics_xf.clone()
            self._prev_s2o_l = singlev_scaled_to_orig_xf.clone()
            # mem_features_l = feats_l.clone()
            # prev_extrinsics_l = extrinsics_xf_l.clone()
            # prev_s2o_l = s2o_l.clone()
            # self._mem_features_l = torch.zeros(
            #     required_memory_len,
            #     72,
            #     6,
            #     6,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
            # mem_features_r = torch.zeros(
            #     required_memory_len,
            #     72,
            #     6,
            #     6,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
            # self._prev_extrinsics_l = torch.zeros(
            #     required_memory_len,
            #     4,
            #     4,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # ) + \
            #   torch.eye(
            #       4,
            #       dtype=per_view_img_features.dtype,
            #       device=per_view_img_features.device,)
            # prev_extrinsics_r = torch.zeros(
            #     required_memory_len,
            #     4,
            #     4,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
            # self._prev_s2o_l = torch.zeros(
            #     required_memory_len,
            #     4,
            #     4,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
            # prev_s2o_r = torch.zeros(
            #     required_memory_len,
            #     4,
            #     4,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
        mem_features_l = self._mem_features_l
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_s2o_l = self._prev_s2o_l
        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )
        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    # singlev_features = self._feature_extractor.compute_singlev_features(
                    #     per_view_img_features[r0:r1].clone(), singlev_scaled_to_orig_xf[r0:r1].clone()
                    # )
                    # img_features_list.append(singlev_features)
                    feats_l = per_view_img_features[r0:r1].clone()
                    s2o_l = singlev_scaled_to_orig_xf[r0:r1].clone()
                    extrinsics_xf_l = extrinsics_xf[r0:r1].clone()
                    mem_feats = mem_features_l[r0:r1].clone()
                    prev_s2o = prev_s2o_l[r0:r1].clone()
                    prev_extrinsics_xf = prev_extrinsics_l[r0:r1].clone()
                    feats_t = torch.cat([feats_l, mem_feats], dim=0)
                    s2o_t = torch.cat([s2o_l, prev_s2o], dim=0)
                    extrinsics_t = torch.cat(
                        [extrinsics_xf_l, prev_extrinsics_xf], dim=0)
                    singlev_features = self._feature_extractor.compute_multiv_features_t(
                        feats_t.unsqueeze(0), s2o_t.unsqueeze(0),
                        extrinsics_t.unsqueeze(0))
                    img_features_list.append(singlev_features)
                    # print(self._mem_features_l.shape)
                    # print(feats_l.shape)
                    self._mem_features_l[r0:r1] = feats_l
                    self._prev_s2o_l[r0:r1] = s2o_l
                    self._prev_extrinsics_l[r0:r1] = extrinsics_xf_l
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_all(self, frame_data: InputFrameData,
                                       frame_desc: InputFrameDesc
                                       ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory
        required_memory_len = int(torch.max(memory_idx)) + 1
        feats_l = per_view_img_features[0::2].clone()
        feats_r = per_view_img_features[1::2].clone()

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        extrinsics_xf = frame_data.extrinsics_xf
        extrinsics_xf_l = extrinsics_xf[0::2].clone()
        s2o_l = singlev_scaled_to_orig_xf[0::2].clone()
        s2o_r = singlev_scaled_to_orig_xf[1::2].clone()

        mem_features_l = self._mem_features_l
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_s2o_l = self._prev_s2o_l
        if not use_memory.all():
            mem_features_l = feats_l.clone()
            prev_extrinsics_l = extrinsics_xf_l.clone()
            prev_s2o_l = s2o_l.clone()
            # mem_features_l = torch.zeros(
            #     required_memory_len,
            #     72,
            #     6,
            #     6,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
            # mem_features_r = torch.zeros(
            #     required_memory_len,
            #     72,
            #     6,
            #     6,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
            # prev_extrinsics_l = torch.zeros(
            #     required_memory_len,
            #     4,
            #     4,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # ) + \
            #   torch.eye(
            #       4,
            #       dtype=per_view_img_features.dtype,
            #       device=per_view_img_features.device,)
            # prev_extrinsics_r = torch.zeros(
            #     required_memory_len,
            #     4,
            #     4,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
            # prev_s2o_l = torch.zeros(
            #     required_memory_len,
            #     4,
            #     4,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
            # prev_s2o_r = torch.zeros(
            #     required_memory_len,
            #     4,
            #     4,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )

        feats_l_tm = torch.stack(
            [x for y in zip(feats_l, mem_features_l) for x in y], dim=0)
        # feats_r_tm = torch.stack([x for y in zip(feats_r,mem_features_r) for x in y],dim=0)
        extrinsics_xf_l_tm = torch.stack(
            [x for y in zip(extrinsics_xf_l, prev_extrinsics_l) for x in y],
            dim=0)
        # extrinsics_xf_r_tm = torch.stack([x for y in zip(extrinsics_xf[1::2].clone(),
        #  prev_extrinsics_r) for x in y],dim=0)
        s2o_l_tm = torch.stack([x for y in zip(s2o_l, prev_s2o_l) for x in y],
                               dim=0)
        # s2o_r_tm = torch.stack([x for y in zip(s2o_r,prev_s2o_r) for x in y],dim=0)
        img_features_multiv = self._feature_extractor.compute_multiv_features(
            per_view_img_features.reshape((-1, 2) +
                                          per_view_img_features.shape[1:]),
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )
        img_features_l = self._feature_extractor.compute_multiv_features_t(
            feats_l_tm.reshape((-1, 2) + feats_l_tm.shape[1:]),
            s2o_l_tm.reshape((-1, 2) + s2o_l_tm.shape[1:]),
            extrinsics_xf_l_tm.reshape((-1, 2) + extrinsics_xf_l_tm.shape[1:]),
        )
        # img_features_r = self._feature_extractor.compute_multiv_features(
        #         feats_r_tm.reshape(
        #             (-1, 2) + feats_r_tm.shape[1:]
        #         ),
        #         s2o_r_tm.reshape(
        #             (-1, 2) + s2o_r_tm.shape[1:]
        #         ),
        #         extrinsics_xf_r_tm.reshape((-1, 2) + extrinsics_xf_r_tm.shape[1:]),
        #     )
        # img_features_l = self._feature_extractor.compute_singlev_features(
        #     per_view_img_features[0::2].clone(), singlev_scaled_to_orig_xf[0::2].clone())
        img_features_r = self._feature_extractor.compute_singlev_features(
            feats_r, s2o_r)
        self._mem_features_l = feats_l.clone()
        self._prev_extrinsics_l = extrinsics_xf_l.clone()
        self._prev_s2o_l = s2o_l.clone()

        return img_features_multiv, img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data, frame_desc)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data, frame_desc)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_multiv = self._temporal.forward_temporal_features_multiv(
            feats_multiv,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        temporal_features_l = self._temporal.forward_temporal_features_l(
            feats_l,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        temporal_features_r = self._temporal.forward_temporal_features_r(
            feats_r,
            extrinsics_1,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features_multiv, temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and feats_multiv.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_multiv.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)

        img_skel_features_multiv = torch.cat(
            [feats_multiv.clone(), skel_features.clone()], dim=1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features = torch.cat([
            img_skel_features_multiv, img_skel_features_l, img_skel_features_r
        ],
                                      dim=0)

        unknown_output = self._regressor_u.regress_poses(feats_multiv)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses_smv(img_skel_features)
        known_output_multiv = known_output[0]
        known_output_l = known_output[1]
        known_output_r = known_output[2]

        known_output_multiv.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_multiv.wrist_xfs,
        )
        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output_multiv': known_output_multiv,
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_TM_New(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor_TM,
        temporal: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_u: PoseRegressor,
    ):
        super().__init__()

        self._feature_extractor: FeatureExtractor_TM = feature_extractor
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_u = regressor_u

        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)
        self._prev_s2o_l = torch.empty(0)
        self._prev_s2o_r = torch.empty(0)
        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   sample_range: torch.Tensor) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        extrinsics_xf = frame_data.extrinsics_xf

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )

        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    singlev_features = self._feature_extractor.compute_singlev_features(
                        per_view_img_features[r0:r1].clone(),
                        singlev_scaled_to_orig_xf[r0:r1].clone())
                    img_features_list.append(singlev_features)
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_all(self, frame_data: InputFrameData,
                                       frame_desc: InputFrameDesc
                                       ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))
        bs = per_view_img_features.shape[0]
        # print(bs)
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory
        required_memory_len = int(torch.max(memory_idx)) + 1
        feats_l = per_view_img_features[0::2].clone()
        feats_r = per_view_img_features[1::2].clone()

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        extrinsics_xf = frame_data.extrinsics_xf
        extrinsics_xf_l = extrinsics_xf[0::2].clone()
        extrinsics_xf_r = extrinsics_xf[1::2].clone()
        s2o_l = singlev_scaled_to_orig_xf[0::2].clone()
        s2o_r = singlev_scaled_to_orig_xf[1::2].clone()
        mem_features_l = self._mem_features_l
        mem_features_r = self._mem_features_r
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_extrinsics_r = self._prev_extrinsics_r
        prev_s2o_l = self._prev_s2o_l
        prev_s2o_r = self._prev_s2o_r
        if not use_memory.all():
            mem_features_l = feats_l.clone()
            mem_features_r = feats_r.clone()
            prev_extrinsics_l = extrinsics_xf_l.clone()
            prev_extrinsics_r = extrinsics_xf_r.clone()
            prev_s2o_l = s2o_l.clone()
            prev_s2o_r = s2o_r.clone()

        feats_l_tm = torch.stack(
            [x for y in zip(feats_l, mem_features_l) for x in y], dim=0)
        feats_r_tm = torch.stack(
            [x for y in zip(feats_r, mem_features_r) for x in y], dim=0)

        extrinsics_xf_l_tm = torch.stack(
            [x for y in zip(extrinsics_xf_l, prev_extrinsics_l) for x in y],
            dim=0)
        extrinsics_xf_r_tm = torch.stack(
            [x for y in zip(extrinsics_xf_r, prev_extrinsics_r) for x in y],
            dim=0)
        s2o_l_tm = torch.stack([x for y in zip(s2o_l, prev_s2o_l) for x in y],
                               dim=0)
        # if s2o_l.device == torch.device('cuda:0'):
        #     print(prev_s2o_r)
        s2o_r_tm = torch.stack([x for y in zip(s2o_r, prev_s2o_r) for x in y],
                               dim=0)
        feats_tm = torch.cat([feats_l_tm, feats_r_tm], dim=0)
        extrinsics_xf_tm = torch.cat([extrinsics_xf_l_tm, extrinsics_xf_r_tm],
                                     dim=0)
        s2o_tm = torch.cat([s2o_l_tm, s2o_r_tm], dim=0)

        img_features_multiv = self._feature_extractor.compute_multiv_features(
            per_view_img_features.reshape((-1, 2) +
                                          per_view_img_features.shape[1:]),
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )
        img_features_tm = self._feature_extractor.compute_multiv_features_t(
            feats_tm.reshape((-1, 2) + feats_tm.shape[1:]),
            s2o_tm.reshape((-1, 2) + s2o_tm.shape[1:]),
            extrinsics_xf_tm.reshape((-1, 2) + extrinsics_xf_tm.shape[1:]),
        )
        # print(img_features_tm.shape)
        img_features_l = img_features_tm[0:bs // 2].clone()
        img_features_r = img_features_tm[bs // 2:].clone()
        # img_features_r = self._feature_extractor.compute_multiv_features(
        #         feats_r_tm.reshape(
        #             (-1, 2) + feats_r_tm.shape[1:]
        #         ),
        #         s2o_r_tm.reshape(
        #             (-1, 2) + s2o_r_tm.shape[1:]
        #         ),
        #         extrinsics_xf_r_tm.reshape((-1, 2) + extrinsics_xf_r_tm.shape[1:]),
        #     )
        # img_features_l = self._feature_extractor.compute_singlev_features(
        #     per_view_img_features[0::2].clone(), singlev_scaled_to_orig_xf[0::2].clone())
        # img_features_r = self._feature_extractor.compute_singlev_features(
        #     per_view_img_features[1::2].clone(), singlev_scaled_to_orig_xf[1::2].clone())
        self._mem_features_l = feats_l
        self._mem_features_r = feats_r
        self._prev_extrinsics_l = extrinsics_xf_l
        self._prev_extrinsics_r = extrinsics_xf_r
        self._prev_s2o_l = s2o_l
        self._prev_s2o_r = s2o_r

        return img_features_multiv, img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data,
                                                       frame_desc.sample_range)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data, frame_desc)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_multiv = self._temporal.forward_temporal_features_multiv(
            feats_multiv,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        temporal_features_l = self._temporal.forward_temporal_features_l(
            feats_l,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        temporal_features_r = self._temporal.forward_temporal_features_r(
            feats_r,
            extrinsics_1,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features_multiv, temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and feats_multiv.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_multiv.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)

        img_skel_features_multiv = torch.cat(
            [feats_multiv.clone(), skel_features.clone()], dim=1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features = torch.cat([
            img_skel_features_multiv, img_skel_features_l, img_skel_features_r
        ],
                                      dim=0)

        unknown_output = self._regressor_u.regress_poses(feats_multiv)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses_smv(img_skel_features)
        known_output_multiv = known_output[0]
        known_output_l = known_output[1]
        known_output_r = known_output[2]

        known_output_multiv.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_multiv.wrist_xfs,
        )
        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output_multiv': known_output_multiv,
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_TM_sv(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor_TM,
        temporal: SimpleConvRNN,
        temporal_sv: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_k_sv: PoseRegressor,
        regressor_u: PoseRegressor,
    ):
        super().__init__()
        print('UmetrackModel_TM_sv')
        self._feature_extractor: FeatureExtractor_TM = feature_extractor
        self._temporal = temporal
        self._temporal_sv = temporal_sv
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_k_sv = regressor_k_sv
        self._regressor_u = regressor_u

        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)
        self._prev_s2o_l = torch.empty(0)
        self._prev_s2o_r = torch.empty(0)
        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   frame_desc: InputFrameDesc) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)
        extrinsics_xf = frame_data.extrinsics_xf
        sample_range = frame_desc.sample_range
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory

        required_memory_len = int(torch.max(memory_idx)) + 1

        if not use_memory.all():
            # print('***********')
            self._mem_features_l = per_view_img_features.clone()
            self._prev_extrinsics_l = extrinsics_xf.clone()
            self._prev_s2o_l = singlev_scaled_to_orig_xf.clone()
            # mem_features_l = feats_l.clone()
            # prev_extrinsics_l = extrinsics_xf_l.clone()
            # prev_s2o_l = s2o_l.clone()
            # self._mem_features_l = torch.zeros(
            #     required_memory_len,
            #     72,
            #     6,
            #     6,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
            # mem_features_r = torch.zeros(
            #     required_memory_len,
            #     72,
            #     6,
            #     6,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
            # self._prev_extrinsics_l = torch.zeros(
            #     required_memory_len,
            #     4,
            #     4,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # ) + \
            #   torch.eye(
            #       4,
            #       dtype=per_view_img_features.dtype,
            #       device=per_view_img_features.device,)
            # prev_extrinsics_r = torch.zeros(
            #     required_memory_len,
            #     4,
            #     4,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
            # self._prev_s2o_l = torch.zeros(
            #     required_memory_len,
            #     4,
            #     4,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
            # prev_s2o_r = torch.zeros(
            #     required_memory_len,
            #     4,
            #     4,
            #     dtype=per_view_img_features.dtype,
            #     device=per_view_img_features.device,
            # )
        mem_features_l = self._mem_features_l
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_s2o_l = self._prev_s2o_l
        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )
        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    # singlev_features = self._feature_extractor.compute_singlev_features(
                    #     per_view_img_features[r0:r1].clone(), singlev_scaled_to_orig_xf[r0:r1].clone()
                    # )
                    # img_features_list.append(singlev_features)
                    feats_l = per_view_img_features[r0:r1].clone()
                    s2o_l = singlev_scaled_to_orig_xf[r0:r1].clone()
                    extrinsics_xf_l = extrinsics_xf[r0:r1].clone()
                    mem_feats = mem_features_l[r0:r1].clone()
                    prev_s2o = prev_s2o_l[r0:r1].clone()
                    prev_extrinsics_xf = prev_extrinsics_l[r0:r1].clone()
                    feats_t = torch.cat([feats_l, mem_feats], dim=0)
                    s2o_t = torch.cat([s2o_l, prev_s2o], dim=0)
                    extrinsics_t = torch.cat(
                        [extrinsics_xf_l, prev_extrinsics_xf], dim=0)
                    singlev_features = self._feature_extractor.compute_multiv_features_t(
                        feats_t.unsqueeze(0), s2o_t.unsqueeze(0),
                        extrinsics_t.unsqueeze(0))
                    img_features_list.append(singlev_features)
                    # print(self._mem_features_l.shape)
                    # print(feats_l.shape)
                    self._mem_features_l[r0:r1] = feats_l
                    self._prev_s2o_l[r0:r1] = s2o_l
                    self._prev_extrinsics_l[r0:r1] = extrinsics_xf_l
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_all(self, frame_data: InputFrameData,
                                       frame_desc: InputFrameDesc
                                       ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))
        bs = per_view_img_features.shape[0]
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory
        required_memory_len = int(torch.max(memory_idx)) + 1
        feats_l = per_view_img_features[0::2].clone()
        feats_r = per_view_img_features[1::2].clone()

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        extrinsics_xf = frame_data.extrinsics_xf
        extrinsics_xf_l = extrinsics_xf[0::2].clone()
        extrinsics_xf_r = extrinsics_xf[1::2].clone()
        s2o_l = singlev_scaled_to_orig_xf[0::2].clone()
        s2o_r = singlev_scaled_to_orig_xf[1::2].clone()

        mem_features_l = self._mem_features_l
        mem_features_r = self._mem_features_r
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_extrinsics_r = self._prev_extrinsics_r
        prev_s2o_l = self._prev_s2o_l
        prev_s2o_r = self._prev_s2o_r
        if not use_memory.all():
            mem_features_l = feats_l.clone()
            mem_features_r = feats_r.clone()
            prev_extrinsics_l = extrinsics_xf_l.clone()
            prev_extrinsics_r = extrinsics_xf_r.clone()
            prev_s2o_l = s2o_l.clone()
            prev_s2o_r = s2o_r.clone()

        feats_l_tm = torch.stack(
            [x for y in zip(feats_l, mem_features_l) for x in y], dim=0)
        feats_r_tm = torch.stack(
            [x for y in zip(feats_r, mem_features_r) for x in y], dim=0)
        extrinsics_xf_l_tm = torch.stack(
            [x for y in zip(extrinsics_xf_l, prev_extrinsics_l) for x in y],
            dim=0)
        extrinsics_xf_r_tm = torch.stack(
            [x for y in zip(extrinsics_xf_r, prev_extrinsics_r) for x in y],
            dim=0)
        s2o_l_tm = torch.stack([x for y in zip(s2o_l, prev_s2o_l) for x in y],
                               dim=0)
        s2o_r_tm = torch.stack([x for y in zip(s2o_r, prev_s2o_r) for x in y],
                               dim=0)
        feats_tm = torch.cat([feats_l_tm, feats_r_tm], dim=0)
        extrinsics_xf_tm = torch.cat([extrinsics_xf_l_tm, extrinsics_xf_r_tm],
                                     dim=0)
        s2o_tm = torch.cat([s2o_l_tm, s2o_r_tm], dim=0)

        img_features_multiv = self._feature_extractor.compute_multiv_features(
            per_view_img_features.reshape((-1, 2) +
                                          per_view_img_features.shape[1:]),
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )
        img_features_tm = self._feature_extractor.compute_multiv_features_t(
            feats_tm.reshape((-1, 2) + feats_tm.shape[1:]),
            s2o_tm.reshape((-1, 2) + s2o_tm.shape[1:]),
            extrinsics_xf_tm.reshape((-1, 2) + extrinsics_xf_tm.shape[1:]),
        )
        img_features_l = img_features_tm[0:bs // 2].clone()
        img_features_r = img_features_tm[bs // 2:].clone()
        # img_features_r = self._feature_extractor.compute_multiv_features(
        #         feats_r_tm.reshape(
        #             (-1, 2) + feats_r_tm.shape[1:]
        #         ),
        #         s2o_r_tm.reshape(
        #             (-1, 2) + s2o_r_tm.shape[1:]
        #         ),
        #         extrinsics_xf_r_tm.reshape((-1, 2) + extrinsics_xf_r_tm.shape[1:]),
        #     )
        # img_features_l = self._feature_extractor.compute_singlev_features(
        #     per_view_img_features[0::2].clone(), singlev_scaled_to_orig_xf[0::2].clone())
        # img_features_r = self._feature_extractor.compute_singlev_features(
        #     feats_r, s2o_r)
        self._mem_features_l = feats_l.clone()
        self._mem_features_r = feats_r.clone()
        self._prev_extrinsics_l = extrinsics_xf_l.clone()
        self._prev_extrinsics_r = extrinsics_xf_r.clone()
        self._prev_s2o_l = s2o_l.clone()
        self._prev_s2o_r = s2o_r.clone()

        return img_features_multiv, img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data, frame_desc)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        # temporal_features = self._temporal_sv.forward_temporal_features(
        #     img_features,
        #     extrinsics,
        #     frame_desc.memory_idx,
        #     frame_desc.use_memory,
        # )

        return temporal_features

    def _forward_feature_extractor_temporal_sv(self,
                                               frame_data: InputFrameData,
                                               frame_desc: InputFrameDesc
                                               ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data, frame_desc)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal_sv.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data, frame_desc)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_multiv = self._temporal.forward_temporal_features_multiv(
            feats_multiv,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        temporal_features_l = self._temporal_sv.forward_temporal_features_l(
            feats_l,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        temporal_features_r = self._temporal_sv.forward_temporal_features_r(
            feats_r,
            extrinsics_1,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features_multiv, temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)
        # regression_output = self._regressor_k_sv.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_use_skeleton_sv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal_sv(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        # regression_output = self._regressor_k.regress_poses(img_skel_features)
        regression_output = self._regressor_k_sv.regress_poses(
            img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()
        self._temporal_sv.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and feats_multiv.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_multiv.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)

        img_skel_features_multiv = torch.cat(
            [feats_multiv.clone(), skel_features.clone()], dim=1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features_sv = torch.cat(
            [img_skel_features_l, img_skel_features_r], dim=0)

        unknown_output = self._regressor_u.regress_poses(feats_multiv)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output_multiv = self._regressor_k.regress_poses(
            img_skel_features_multiv)

        known_output_sv = self._regressor_k_sv.regress_poses_sv(
            img_skel_features_sv)
        known_output_l = known_output_sv[0]
        known_output_r = known_output_sv[1]

        known_output_multiv.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_multiv.wrist_xfs,
        )
        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output_multiv': known_output_multiv,
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_TM_Res(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor_TM_Res,
        temporal: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_u: PoseRegressor,
        use_sv_temp=True,
    ):
        super().__init__()
        print('UmeTrackModel_TM_residual')
        self._feature_extractor: FeatureExtractor_TM_Res = feature_extractor
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_u = regressor_u
        self.use_sv_temp = use_sv_temp

        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)
        self._prev_s2o_l = torch.empty(0)
        self._prev_s2o_r = torch.empty(0)
        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   frame_desc: InputFrameDesc) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)
        extrinsics_xf = frame_data.extrinsics_xf
        sample_range = frame_desc.sample_range
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory

        required_memory_len = int(torch.max(memory_idx)) + 1

        if not use_memory.all():
            # print('***********')
            self._mem_features_l = per_view_img_features.clone()
            self._prev_extrinsics_l = extrinsics_xf.clone()
            self._prev_s2o_l = singlev_scaled_to_orig_xf.clone()

        mem_features_l = self._mem_features_l
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_s2o_l = self._prev_s2o_l
        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )
        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    # singlev_features = self._feature_extractor.compute_singlev_features(
                    #     per_view_img_features[r0:r1].clone(), singlev_scaled_to_orig_xf[r0:r1].clone()
                    # )
                    # img_features_list.append(singlev_features)
                    feats_l = per_view_img_features[r0:r1].clone()
                    s2o_l = singlev_scaled_to_orig_xf[r0:r1].clone()
                    extrinsics_xf_l = extrinsics_xf[r0:r1].clone()
                    mem_feats = mem_features_l[r0:r1].clone()
                    prev_s2o = prev_s2o_l[r0:r1].clone()
                    prev_extrinsics_xf = prev_extrinsics_l[r0:r1].clone()
                    feats_t = torch.cat([feats_l, mem_feats], dim=0)
                    s2o_t = torch.cat([s2o_l, prev_s2o], dim=0)
                    extrinsics_t = torch.cat(
                        [extrinsics_xf_l, prev_extrinsics_xf], dim=0)
                    singlev_features = self._feature_extractor.compute_multiv_features_t(
                        feats_l, feats_t.unsqueeze(0), s2o_t.unsqueeze(0),
                        extrinsics_t.unsqueeze(0))
                    img_features_list.append(singlev_features)
                    # print(self._mem_features_l.shape)
                    # print(feats_l.shape)
                    self._mem_features_l[r0:r1] = feats_l
                    self._prev_s2o_l[r0:r1] = s2o_l
                    self._prev_extrinsics_l[r0:r1] = extrinsics_xf_l
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_all(self, frame_data: InputFrameData,
                                       frame_desc: InputFrameDesc
                                       ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))
        bs = per_view_img_features.shape[0]
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory
        required_memory_len = int(torch.max(memory_idx)) + 1
        feats_l = per_view_img_features[0::2].clone()
        feats_r = per_view_img_features[1::2].clone()

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        extrinsics_xf = frame_data.extrinsics_xf
        extrinsics_xf_l = extrinsics_xf[0::2].clone()
        extrinsics_xf_r = extrinsics_xf[1::2].clone()
        s2o_l = singlev_scaled_to_orig_xf[0::2].clone()
        s2o_r = singlev_scaled_to_orig_xf[1::2].clone()

        mem_features_l = self._mem_features_l
        mem_features_r = self._mem_features_r
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_extrinsics_r = self._prev_extrinsics_r
        prev_s2o_l = self._prev_s2o_l
        prev_s2o_r = self._prev_s2o_r
        if not use_memory.all():
            mem_features_l = feats_l.clone()
            mem_features_r = feats_r.clone()
            prev_extrinsics_l = extrinsics_xf_l.clone()
            prev_extrinsics_r = extrinsics_xf_r.clone()
            prev_s2o_l = s2o_l.clone()
            prev_s2o_r = s2o_r.clone()

        feats_l_tm = torch.stack(
            [x for y in zip(feats_l, mem_features_l) for x in y], dim=0)
        feats_r_tm = torch.stack(
            [x for y in zip(feats_r, mem_features_r) for x in y], dim=0)
        extrinsics_xf_l_tm = torch.stack(
            [x for y in zip(extrinsics_xf_l, prev_extrinsics_l) for x in y],
            dim=0)
        extrinsics_xf_r_tm = torch.stack([
            x for y in zip(extrinsics_xf[1::2].clone(), prev_extrinsics_r)
            for x in y
        ],
                                         dim=0)
        s2o_l_tm = torch.stack([x for y in zip(s2o_l, prev_s2o_l) for x in y],
                               dim=0)
        s2o_r_tm = torch.stack([x for y in zip(s2o_r, prev_s2o_r) for x in y],
                               dim=0)
        feats_0 = torch.cat([feats_l, feats_r], dim=0)
        feats_tm = torch.cat([feats_l_tm, feats_r_tm], dim=0)
        extrinsics_xf_tm = torch.cat([extrinsics_xf_l_tm, extrinsics_xf_r_tm],
                                     dim=0)
        s2o_tm = torch.cat([s2o_l_tm, s2o_r_tm], dim=0)

        img_features_multiv = self._feature_extractor.compute_multiv_features(
            per_view_img_features.reshape((-1, 2) +
                                          per_view_img_features.shape[1:]),
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )
        img_features_tm = self._feature_extractor.compute_multiv_features_t(
            feats_0,
            feats_tm.reshape((-1, 2) + feats_tm.shape[1:]),
            s2o_tm.reshape((-1, 2) + s2o_tm.shape[1:]),
            extrinsics_xf_tm.reshape((-1, 2) + extrinsics_xf_tm.shape[1:]),
        )
        img_features_l = img_features_tm[0:bs // 2].clone()
        img_features_r = img_features_tm[bs // 2:].clone()

        self._mem_features_l = feats_l.clone()
        self._mem_features_r = feats_r.clone()
        self._prev_extrinsics_l = extrinsics_xf_l.clone()
        self._prev_extrinsics_r = extrinsics_xf_r.clone()
        self._prev_s2o_l = s2o_l.clone()
        self._prev_s2o_r = s2o_r.clone()

        return img_features_multiv, img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data, frame_desc)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data, frame_desc)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_multiv = self._temporal.forward_temporal_features_multiv(
            feats_multiv,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        if self.use_sv_temp:
            temporal_features_l = self._temporal.forward_temporal_features_l(
                feats_l,
                extrinsics_0,
                frame_desc.memory_idx,
                frame_desc.use_memory,
            )
            temporal_features_r = self._temporal.forward_temporal_features_r(
                feats_r,
                extrinsics_1,
                frame_desc.memory_idx,
                frame_desc.use_memory,
            )
        else:
            temporal_features_l = feats_l
            temporal_features_r = feats_r

        return temporal_features_multiv, temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and feats_multiv.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_multiv.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)

        img_skel_features_multiv = torch.cat(
            [feats_multiv.clone(), skel_features.clone()], dim=1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features = torch.cat([
            img_skel_features_multiv, img_skel_features_l, img_skel_features_r
        ],
                                      dim=0)

        unknown_output = self._regressor_u.regress_poses(feats_multiv)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses_smv(img_skel_features)
        known_output_multiv = known_output[0]
        known_output_l = known_output[1]
        known_output_r = known_output[2]

        known_output_multiv.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_multiv.wrist_xfs,
        )
        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output_multiv': known_output_multiv,
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_TM_Res_Sub(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor_TM_Res_Sub,
        temporal: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_u: PoseRegressor,
        use_sv_temp=True,
    ):
        super().__init__()
        print('UmeTrackModel_TM_residual_sub')
        self._feature_extractor: FeatureExtractor_TM_Res_Sub = feature_extractor
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_u = regressor_u
        self.use_sv_temp = use_sv_temp

        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)
        self._prev_s2o_l = torch.empty(0)
        self._prev_s2o_r = torch.empty(0)
        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   frame_desc: InputFrameDesc) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)
        extrinsics_xf = frame_data.extrinsics_xf
        sample_range = frame_desc.sample_range
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory

        required_memory_len = int(torch.max(memory_idx)) + 1

        if not use_memory.all():
            # print('***********')
            self._mem_features_l = per_view_img_features.clone()
            self._prev_extrinsics_l = extrinsics_xf.clone()
            self._prev_s2o_l = singlev_scaled_to_orig_xf.clone()

        mem_features_l = self._mem_features_l
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_s2o_l = self._prev_s2o_l
        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )
        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    # singlev_features = self._feature_extractor.compute_singlev_features(
                    #     per_view_img_features[r0:r1].clone(), singlev_scaled_to_orig_xf[r0:r1].clone()
                    # )
                    # img_features_list.append(singlev_features)
                    feats_l = per_view_img_features[r0:r1].clone()
                    s2o_l = singlev_scaled_to_orig_xf[r0:r1].clone()
                    extrinsics_xf_l = extrinsics_xf[r0:r1].clone()
                    mem_feats = mem_features_l[r0:r1].clone()
                    prev_s2o = prev_s2o_l[r0:r1].clone()
                    prev_extrinsics_xf = prev_extrinsics_l[r0:r1].clone()
                    feats_t = torch.cat([feats_l, mem_feats], dim=0)
                    s2o_t = torch.cat([s2o_l, prev_s2o], dim=0)
                    extrinsics_t = torch.cat(
                        [extrinsics_xf_l, prev_extrinsics_xf], dim=0)
                    singlev_features = self._feature_extractor.compute_multiv_features_t(
                        feats_l, feats_t.unsqueeze(0), s2o_t.unsqueeze(0),
                        extrinsics_t.unsqueeze(0))
                    img_features_list.append(singlev_features)
                    # print(self._mem_features_l.shape)
                    # print(feats_l.shape)
                    self._mem_features_l[r0:r1] = feats_l
                    self._prev_s2o_l[r0:r1] = s2o_l
                    self._prev_extrinsics_l[r0:r1] = extrinsics_xf_l
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_all(self, frame_data: InputFrameData,
                                       frame_desc: InputFrameDesc
                                       ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))
        bs = per_view_img_features.shape[0]
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory
        required_memory_len = int(torch.max(memory_idx)) + 1
        feats_l = per_view_img_features[0::2].clone()
        feats_r = per_view_img_features[1::2].clone()

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        extrinsics_xf = frame_data.extrinsics_xf
        extrinsics_xf_l = extrinsics_xf[0::2].clone()
        extrinsics_xf_r = extrinsics_xf[1::2].clone()
        s2o_l = singlev_scaled_to_orig_xf[0::2].clone()
        s2o_r = singlev_scaled_to_orig_xf[1::2].clone()

        mem_features_l = self._mem_features_l
        mem_features_r = self._mem_features_r
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_extrinsics_r = self._prev_extrinsics_r
        prev_s2o_l = self._prev_s2o_l
        prev_s2o_r = self._prev_s2o_r
        if not use_memory.all():
            mem_features_l = feats_l.clone()
            mem_features_r = feats_r.clone()
            prev_extrinsics_l = extrinsics_xf_l.clone()
            prev_extrinsics_r = extrinsics_xf_r.clone()
            prev_s2o_l = s2o_l.clone()
            prev_s2o_r = s2o_r.clone()

        feats_l_tm = torch.stack(
            [x for y in zip(feats_l, mem_features_l) for x in y], dim=0)
        feats_r_tm = torch.stack(
            [x for y in zip(feats_r, mem_features_r) for x in y], dim=0)
        extrinsics_xf_l_tm = torch.stack(
            [x for y in zip(extrinsics_xf_l, prev_extrinsics_l) for x in y],
            dim=0)
        extrinsics_xf_r_tm = torch.stack([
            x for y in zip(extrinsics_xf[1::2].clone(), prev_extrinsics_r)
            for x in y
        ],
                                         dim=0)
        s2o_l_tm = torch.stack([x for y in zip(s2o_l, prev_s2o_l) for x in y],
                               dim=0)
        s2o_r_tm = torch.stack([x for y in zip(s2o_r, prev_s2o_r) for x in y],
                               dim=0)
        feats_0 = torch.cat([feats_l, feats_r], dim=0)
        feats_tm = torch.cat([feats_l_tm, feats_r_tm], dim=0)
        extrinsics_xf_tm = torch.cat([extrinsics_xf_l_tm, extrinsics_xf_r_tm],
                                     dim=0)
        s2o_tm = torch.cat([s2o_l_tm, s2o_r_tm], dim=0)

        img_features_multiv = self._feature_extractor.compute_multiv_features(
            per_view_img_features.reshape((-1, 2) +
                                          per_view_img_features.shape[1:]),
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )
        img_features_tm = self._feature_extractor.compute_multiv_features_t(
            feats_0,
            feats_tm.reshape((-1, 2) + feats_tm.shape[1:]),
            s2o_tm.reshape((-1, 2) + s2o_tm.shape[1:]),
            extrinsics_xf_tm.reshape((-1, 2) + extrinsics_xf_tm.shape[1:]),
        )
        img_features_l = img_features_tm[0:bs // 2].clone()
        img_features_r = img_features_tm[bs // 2:].clone()

        self._mem_features_l = feats_l.clone()
        self._mem_features_r = feats_r.clone()
        self._prev_extrinsics_l = extrinsics_xf_l.clone()
        self._prev_extrinsics_r = extrinsics_xf_r.clone()
        self._prev_s2o_l = s2o_l.clone()
        self._prev_s2o_r = s2o_r.clone()

        return img_features_multiv, img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data, frame_desc)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data, frame_desc)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_multiv = self._temporal.forward_temporal_features_multiv(
            feats_multiv,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        if self.use_sv_temp:
            temporal_features_l = self._temporal.forward_temporal_features_l(
                feats_l,
                extrinsics_0,
                frame_desc.memory_idx,
                frame_desc.use_memory,
            )
            temporal_features_r = self._temporal.forward_temporal_features_r(
                feats_r,
                extrinsics_1,
                frame_desc.memory_idx,
                frame_desc.use_memory,
            )
        else:
            temporal_features_l = feats_l
            temporal_features_r = feats_r

        return temporal_features_multiv, temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and feats_multiv.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_multiv.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)

        img_skel_features_multiv = torch.cat(
            [feats_multiv.clone(), skel_features.clone()], dim=1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features = torch.cat([
            img_skel_features_multiv, img_skel_features_l, img_skel_features_r
        ],
                                      dim=0)

        unknown_output = self._regressor_u.regress_poses(feats_multiv)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses_smv(img_skel_features)
        known_output_multiv = known_output[0]
        known_output_l = known_output[1]
        known_output_r = known_output[2]

        known_output_multiv.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_multiv.wrist_xfs,
        )
        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output_multiv': known_output_multiv,
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_TM_Res_sv(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor_TM_Res,
        temporal: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_k_sv: PoseRegressor,
        regressor_u: PoseRegressor,
    ):
        super().__init__()
        print('UmeTrackModel_TM_residual_sv')
        self._feature_extractor: FeatureExtractor_TM_Res = feature_extractor
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_k_sv = regressor_k_sv
        self._regressor_u = regressor_u

        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)
        self._prev_s2o_l = torch.empty(0)
        self._prev_s2o_r = torch.empty(0)
        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   frame_desc: InputFrameDesc) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)
        extrinsics_xf = frame_data.extrinsics_xf
        sample_range = frame_desc.sample_range
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory

        required_memory_len = int(torch.max(memory_idx)) + 1

        if not use_memory.all():
            # print('***********')
            self._mem_features_l = per_view_img_features.clone()
            self._prev_extrinsics_l = extrinsics_xf.clone()
            self._prev_s2o_l = singlev_scaled_to_orig_xf.clone()

        mem_features_l = self._mem_features_l
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_s2o_l = self._prev_s2o_l
        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )
        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    # singlev_features = self._feature_extractor.compute_singlev_features(
                    #     per_view_img_features[r0:r1].clone(), singlev_scaled_to_orig_xf[r0:r1].clone()
                    # )
                    # img_features_list.append(singlev_features)
                    feats_l = per_view_img_features[r0:r1].clone()
                    s2o_l = singlev_scaled_to_orig_xf[r0:r1].clone()
                    extrinsics_xf_l = extrinsics_xf[r0:r1].clone()
                    mem_feats = mem_features_l[r0:r1].clone()
                    prev_s2o = prev_s2o_l[r0:r1].clone()
                    prev_extrinsics_xf = prev_extrinsics_l[r0:r1].clone()
                    feats_t = torch.cat([feats_l, mem_feats], dim=0)
                    s2o_t = torch.cat([s2o_l, prev_s2o], dim=0)
                    extrinsics_t = torch.cat(
                        [extrinsics_xf_l, prev_extrinsics_xf], dim=0)
                    singlev_features = self._feature_extractor.compute_multiv_features_t(
                        feats_l, feats_t.unsqueeze(0), s2o_t.unsqueeze(0),
                        extrinsics_t.unsqueeze(0))
                    img_features_list.append(singlev_features)
                    # print(self._mem_features_l.shape)
                    # print(feats_l.shape)
                    self._mem_features_l[r0:r1] = feats_l
                    self._prev_s2o_l[r0:r1] = s2o_l
                    self._prev_extrinsics_l[r0:r1] = extrinsics_xf_l
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_all(self, frame_data: InputFrameData,
                                       frame_desc: InputFrameDesc
                                       ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))
        bs = per_view_img_features.shape[0]
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory
        required_memory_len = int(torch.max(memory_idx)) + 1
        feats_l = per_view_img_features[0::2].clone()
        feats_r = per_view_img_features[1::2].clone()

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        extrinsics_xf = frame_data.extrinsics_xf
        extrinsics_xf_l = extrinsics_xf[0::2].clone()
        extrinsics_xf_r = extrinsics_xf[1::2].clone()
        s2o_l = singlev_scaled_to_orig_xf[0::2].clone()
        s2o_r = singlev_scaled_to_orig_xf[1::2].clone()

        mem_features_l = self._mem_features_l
        mem_features_r = self._mem_features_r
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_extrinsics_r = self._prev_extrinsics_r
        prev_s2o_l = self._prev_s2o_l
        prev_s2o_r = self._prev_s2o_r
        if not use_memory.all():
            mem_features_l = feats_l.clone()
            mem_features_r = feats_r.clone()
            prev_extrinsics_l = extrinsics_xf_l.clone()
            prev_extrinsics_r = extrinsics_xf_r.clone()
            prev_s2o_l = s2o_l.clone()
            prev_s2o_r = s2o_r.clone()

        feats_l_tm = torch.stack(
            [x for y in zip(feats_l, mem_features_l) for x in y], dim=0)
        feats_r_tm = torch.stack(
            [x for y in zip(feats_r, mem_features_r) for x in y], dim=0)
        extrinsics_xf_l_tm = torch.stack(
            [x for y in zip(extrinsics_xf_l, prev_extrinsics_l) for x in y],
            dim=0)
        extrinsics_xf_r_tm = torch.stack(
            [x for y in zip(extrinsics_xf_r, prev_extrinsics_r) for x in y],
            dim=0)
        s2o_l_tm = torch.stack([x for y in zip(s2o_l, prev_s2o_l) for x in y],
                               dim=0)
        s2o_r_tm = torch.stack([x for y in zip(s2o_r, prev_s2o_r) for x in y],
                               dim=0)
        feats_0 = torch.cat([feats_l, feats_r], dim=0)
        feats_tm = torch.cat([feats_l_tm, feats_r_tm], dim=0)
        extrinsics_xf_tm = torch.cat([extrinsics_xf_l_tm, extrinsics_xf_r_tm],
                                     dim=0)
        s2o_tm = torch.cat([s2o_l_tm, s2o_r_tm], dim=0)

        img_features_multiv = self._feature_extractor.compute_multiv_features(
            per_view_img_features.reshape((-1, 2) +
                                          per_view_img_features.shape[1:]),
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )
        img_features_tm = self._feature_extractor.compute_multiv_features_t(
            feats_0,
            feats_tm.reshape((-1, 2) + feats_tm.shape[1:]),
            s2o_tm.reshape((-1, 2) + s2o_tm.shape[1:]),
            extrinsics_xf_tm.reshape((-1, 2) + extrinsics_xf_tm.shape[1:]),
        )
        img_features_l = img_features_tm[0:bs // 2].clone()
        img_features_r = img_features_tm[bs // 2:].clone()

        self._mem_features_l = feats_l.clone()
        self._mem_features_r = feats_r.clone()
        self._prev_extrinsics_l = extrinsics_xf_l.clone()
        self._prev_extrinsics_r = extrinsics_xf_r.clone()
        self._prev_s2o_l = s2o_l.clone()
        self._prev_s2o_r = s2o_r.clone()

        return img_features_multiv, img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data, frame_desc)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data, frame_desc)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        # extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_multiv = self._temporal.forward_temporal_features_multiv(
            feats_multiv,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        temporal_features_l = feats_l
        temporal_features_r = feats_r

        return temporal_features_multiv, temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_use_skeleton_sv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k_sv.regress_poses(
            img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and feats_multiv.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_multiv.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)

        img_skel_features_multiv = torch.cat(
            [feats_multiv.clone(), skel_features.clone()], dim=1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features_sv = torch.cat(
            [img_skel_features_l, img_skel_features_r], dim=0)

        unknown_output = self._regressor_u.regress_poses(feats_multiv)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output_multiv = self._regressor_k.regress_poses(
            img_skel_features_multiv)
        known_output = self._regressor_k_sv.regress_poses_sv(
            img_skel_features_sv)

        known_output_l = known_output[0]
        known_output_r = known_output[1]

        known_output_multiv.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_multiv.wrist_xfs,
        )
        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output_multiv': known_output_multiv,
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_TM_Res_New(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor_TM_Res,
        temporal: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_u: PoseRegressor,
        use_sv_temp=True,
    ):
        super().__init__()
        print('UmeTrackModel_TM_residual_New')
        self._feature_extractor: FeatureExtractor_TM_Res = feature_extractor
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_u = regressor_u
        self.use_sv_temp = use_sv_temp

        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)
        self._prev_s2o_l = torch.empty(0)
        self._prev_s2o_r = torch.empty(0)
        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   frame_desc: InputFrameDesc) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)
        extrinsics_xf = frame_data.extrinsics_xf
        sample_range = frame_desc.sample_range
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory

        required_memory_len = int(torch.max(memory_idx)) + 1

        if not use_memory.all():
            # print('***********')
            self._mem_features_l = per_view_img_features.clone()
            self._prev_extrinsics_l = extrinsics_xf.clone()
            self._prev_s2o_l = singlev_scaled_to_orig_xf.clone()

        mem_features_l = self._mem_features_l
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_s2o_l = self._prev_s2o_l
        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )
        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    # singlev_features = self._feature_extractor.compute_singlev_features(
                    #     per_view_img_features[r0:r1].clone(), singlev_scaled_to_orig_xf[r0:r1].clone()
                    # )
                    # img_features_list.append(singlev_features)
                    feats_l = per_view_img_features[r0:r1].clone()
                    s2o_l = singlev_scaled_to_orig_xf[r0:r1].clone()
                    extrinsics_xf_l = extrinsics_xf[r0:r1].clone()
                    mem_feats = mem_features_l[r0:r1].clone()
                    prev_s2o = prev_s2o_l[r0:r1].clone()
                    prev_extrinsics_xf = prev_extrinsics_l[r0:r1].clone()
                    feats_t = torch.cat([feats_l, mem_feats], dim=0)
                    s2o_t = torch.cat([s2o_l, prev_s2o], dim=0)
                    extrinsics_t = torch.cat(
                        [extrinsics_xf_l, prev_extrinsics_xf], dim=0)
                    singlev_features = self._feature_extractor.compute_multiv_features_t(
                        feats_l, feats_t.unsqueeze(0), s2o_t.unsqueeze(0),
                        extrinsics_t.unsqueeze(0))
                    img_features_list.append(singlev_features)
                    # print(self._mem_features_l.shape)
                    # print(feats_l.shape)
                    self._mem_features_l[r0:r1] = feats_l
                    self._prev_s2o_l[r0:r1] = s2o_l
                    self._prev_extrinsics_l[r0:r1] = extrinsics_xf_l
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_all(self, frame_data: InputFrameData,
                                       frame_desc: InputFrameDesc
                                       ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))
        bs = per_view_img_features.shape[0]
        use_memory = frame_desc.use_memory

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        extrinsics_xf = frame_data.extrinsics_xf
        extrinsics_xf_l = extrinsics_xf[0::2].clone()
        extrinsics_xf_r = extrinsics_xf[1::2].clone()

        img_features_multiv = self._feature_extractor.compute_multiv_features(
            per_view_img_features.reshape((-1, 2) +
                                          per_view_img_features.shape[1:]),
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )
        feats_l = self._feature_extractor.compute_singlev_features(
            per_view_img_features[0::2].clone(),
            singlev_scaled_to_orig_xf[0::2].clone())
        feats_r = self._feature_extractor.compute_singlev_features(
            per_view_img_features[1::2].clone(),
            singlev_scaled_to_orig_xf[1::2].clone())
        mem_features_l = self._mem_features_l
        mem_features_r = self._mem_features_r
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_extrinsics_r = self._prev_extrinsics_r

        if not use_memory.all():
            mem_features_l = feats_l.clone()
            mem_features_r = feats_r.clone()
            prev_extrinsics_l = extrinsics_xf_l.clone()
            prev_extrinsics_r = extrinsics_xf_r.clone()
        prev_cam0_to_world_xf_l = torch.inverse(prev_extrinsics_l)
        prev_cam0_to_world_xf_r = torch.inverse(prev_extrinsics_r)
        prev_cam0_to_cur_cam0_xf_l = extrinsics_xf_l.bmm(
            prev_cam0_to_world_xf_l)
        prev_cam0_to_cur_cam0_xf_r = extrinsics_xf_r.bmm(
            prev_cam0_to_world_xf_r)
        prev_mem_features_l = model_utils.apply_ftl_to_feature_maps(
            prev_cam0_to_cur_cam0_xf_l,
            mem_features_l,
            1.0,
        )
        prev_mem_features_r = model_utils.apply_ftl_to_feature_maps(
            prev_cam0_to_cur_cam0_xf_r,
            mem_features_r,
            1.0,
        )
        feats_l_tm = torch.cat([feats_l, prev_mem_features_l], dim=1)
        feats_r_tm = torch.cat([feats_r, prev_mem_features_r], dim=1)
        feats_tm = torch.cat([feats_l_tm, feats_r_tm], dim=0)
        feats_0 = torch.cat([feats_l, feats_r], dim=0)
        img_features_tm = self._feature_extractor.compute_multiv_features_t_new(
            feats_0, feats_tm)
        img_features_l = img_features_tm[0:bs // 2].clone()
        img_features_r = img_features_tm[bs // 2:].clone()

        self._mem_features_l = feats_l.clone()
        self._mem_features_r = feats_r.clone()
        self._prev_extrinsics_l = extrinsics_xf_l.clone()
        self._prev_extrinsics_r = extrinsics_xf_r.clone()

        return img_features_multiv, img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data, frame_desc)
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data, frame_desc)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_multiv = self._temporal.forward_temporal_features_multiv(
            feats_multiv,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        if self.use_sv_temp:
            temporal_features_l = self._temporal.forward_temporal_features_l(
                feats_l,
                extrinsics_0,
                frame_desc.memory_idx,
                frame_desc.use_memory,
            )
            temporal_features_r = self._temporal.forward_temporal_features_r(
                feats_r,
                extrinsics_1,
                frame_desc.memory_idx,
                frame_desc.use_memory,
            )
        else:
            temporal_features_l = feats_l
            temporal_features_r = feats_r

        return temporal_features_multiv, temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_multiv, feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and feats_multiv.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_multiv.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)

        img_skel_features_multiv = torch.cat(
            [feats_multiv.clone(), skel_features.clone()], dim=1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features = torch.cat([
            img_skel_features_multiv, img_skel_features_l, img_skel_features_r
        ],
                                      dim=0)

        unknown_output = self._regressor_u.regress_poses(feats_multiv)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses_smv(img_skel_features)
        known_output_multiv = known_output[0]
        known_output_l = known_output[1]
        known_output_r = known_output[2]

        known_output_multiv.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_multiv.wrist_xfs,
        )
        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output_multiv': known_output_multiv,
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_TM_Res_New_Fine_Tune(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor_TM_Res_New,
        temporal: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_k_sv: PoseRegressor,
        regressor_u: PoseRegressor,
    ):
        super().__init__()
        print('UmeTrackModel_TM_residual_New_Fine_Tune')
        self._feature_extractor: FeatureExtractor_TM_Res_New = feature_extractor
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_k_sv = regressor_k_sv
        self._regressor_u = regressor_u

        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)
        self._prev_s2o_l = torch.empty(0)
        self._prev_s2o_r = torch.empty(0)
        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   frame_desc: InputFrameDesc) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))
        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)
        extrinsics_xf = frame_data.extrinsics_xf
        sample_range = frame_desc.sample_range
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory

        required_memory_len = int(torch.max(memory_idx)) + 1

        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )
            return img_features
        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    singlev_features = self._feature_extractor.compute_singlev_features(
                        per_view_img_features[r0:r1].clone(),
                        singlev_scaled_to_orig_xf[r0:r1].clone())
                    img_features_list.append(singlev_features)
                    # feats_l = per_view_img_features[r0:r1].clone()
                    # s2o_l = singlev_scaled_to_orig_xf[r0:r1].clone()
                    # extrinsics_xf_l = extrinsics_xf[r0:r1].clone()
                    # mem_feats = mem_features_l[r0:r1].clone()
                    # prev_s2o = prev_s2o_l[r0:r1].clone()
                    # prev_extrinsics_xf = prev_extrinsics_l[r0:r1].clone()
                    # feats_t = torch.cat([feats_l,mem_feats],dim=0)
                    # s2o_t = torch.cat([s2o_l,prev_s2o],dim=0)
                    # extrinsics_t = torch.cat([extrinsics_xf_l,prev_extrinsics_xf],dim=0)
                    # singlev_features = self._feature_extractor.compute_multiv_features_t(
                    #     feats_l,
                    #     feats_t.unsqueeze(0),
                    #     s2o_t.unsqueeze(0),
                    #     extrinsics_t.unsqueeze(0)
                    # )
                    # img_features_list.append(singlev_features)
                    # print(self._mem_features_l.shape)
                    # print(feats_l.shape)
                    # self._mem_features_l[r0:r1] = feats_l

                    # self._prev_extrinsics_l[r0:r1] = extrinsics_xf_l
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        if not use_memory.all():
            # print('***********')
            self._mem_features_l = img_features.clone()

            self._prev_extrinsics_l = extrinsics_xf.clone()
        mem_features_l = self._mem_features_l
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_cam0_to_world_xf_l = torch.inverse(prev_extrinsics_l)
        prev_cam0_to_cur_cam0_xf_l = extrinsics_xf.bmm(prev_cam0_to_world_xf_l)
        prev_mem_features_l = model_utils.apply_ftl_to_feature_maps(
            prev_cam0_to_cur_cam0_xf_l,
            mem_features_l,
            1.0,
        )
        feats_l_tm = torch.cat([img_features, prev_mem_features_l], dim=1)
        img_feats = self._feature_extractor.compute_multiv_features_t_new(
            img_features, feats_l_tm)

        self._mem_features_l = img_features
        self._prev_extrinsics_l = extrinsics_xf

        return img_feats

    def _forward_feature_extractor_all(self, frame_data: InputFrameData,
                                       frame_desc: InputFrameDesc
                                       ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))

        bs = per_view_img_features.shape[0]
        use_memory = frame_desc.use_memory
        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)
        extrinsics_xf = frame_data.extrinsics_xf
        extrinsics_xf_l = extrinsics_xf[0::2].clone()
        extrinsics_xf_r = extrinsics_xf[1::2].clone()
        feats_l = self._feature_extractor.compute_singlev_features(
            per_view_img_features[0::2].clone(),
            singlev_scaled_to_orig_xf[0::2].clone())
        feats_r = self._feature_extractor.compute_singlev_features(
            per_view_img_features[1::2].clone(),
            singlev_scaled_to_orig_xf[1::2].clone())
        mem_features_l = self._mem_features_l
        mem_features_r = self._mem_features_r
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_extrinsics_r = self._prev_extrinsics_r
        if not use_memory.all():
            mem_features_l = feats_l.clone()
            mem_features_r = feats_r.clone()
            prev_extrinsics_l = extrinsics_xf_l.clone()
            prev_extrinsics_r = extrinsics_xf_r.clone()
        prev_cam0_to_world_xf_l = torch.inverse(prev_extrinsics_l)
        prev_cam0_to_world_xf_r = torch.inverse(prev_extrinsics_r)
        prev_cam0_to_cur_cam0_xf_l = extrinsics_xf_l.bmm(
            prev_cam0_to_world_xf_l)
        prev_cam0_to_cur_cam0_xf_r = extrinsics_xf_r.bmm(
            prev_cam0_to_world_xf_r)
        prev_mem_features_l = model_utils.apply_ftl_to_feature_maps(
            prev_cam0_to_cur_cam0_xf_l,
            mem_features_l,
            1.0,
        )
        prev_mem_features_r = model_utils.apply_ftl_to_feature_maps(
            prev_cam0_to_cur_cam0_xf_r,
            mem_features_r,
            1.0,
        )
        feats_l_tm = torch.cat([feats_l, prev_mem_features_l], dim=1)
        feats_r_tm = torch.cat([feats_r, prev_mem_features_r], dim=1)
        feats_tm = torch.cat([feats_l_tm, feats_r_tm], dim=0)
        feats_0 = torch.cat([feats_l, feats_r], dim=0)
        img_features_tm = self._feature_extractor.compute_multiv_features_t_new(
            feats_0, feats_tm)
        img_features_l = img_features_tm[0:bs // 2].clone()
        img_features_r = img_features_tm[bs // 2:].clone()

        self._mem_features_l = feats_l.clone()
        self._mem_features_r = feats_r.clone()
        self._prev_extrinsics_l = extrinsics_xf_l.clone()
        self._prev_extrinsics_r = extrinsics_xf_r.clone()

        return img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data, frame_desc)

        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data, frame_desc)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_l = self._temporal.forward_temporal_features_l(
            feats_l,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        temporal_features_r = self._temporal.forward_temporal_features_r(
            feats_r,
            extrinsics_1,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_use_skeleton_sv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k_sv.regress_poses(
            img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data, frame_desc)
        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and feats_l.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_l.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features = torch.cat(
            [img_skel_features_l, img_skel_features_r], dim=0)

        known_output = self._regressor_k_sv.regress_poses_sv(img_skel_features)
        known_output_l = known_output[0]
        known_output_r = known_output[1]

        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )
        regress_output = {
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output


class UmeTrackModel_TM_Res_New_Fine_Tune_wT(nn.Module):

    def __init__(
        self,
        feature_extractor: FeatureExtractor_TM_Res_New,
        temporal: SimpleConvRNN,
        temporal_sv: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_k_sv: PoseRegressor,
        regressor_u: PoseRegressor,
    ):
        super().__init__()
        print('UmeTrackModel_TM_residual_New_Fine_Tune_wT')
        self._feature_extractor: FeatureExtractor_TM_Res_New = feature_extractor
        self._temporal = temporal
        self._temporal_sv = temporal_sv
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_k_sv = regressor_k_sv
        self._regressor_u = regressor_u

        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)
        self._prev_s2o_l = torch.empty(0)
        self._prev_s2o_r = torch.empty(0)
        # self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(self, frame_data: InputFrameData,
                                   frame_desc: InputFrameDesc) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))
        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)
        extrinsics_xf = frame_data.extrinsics_xf
        sample_range = frame_desc.sample_range
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory

        required_memory_len = int(torch.max(memory_idx)) + 1

        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[
            0] * num_views == per_view_img_features.shape[0]

        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape((-1, num_views) +
                                              per_view_img_features.shape[1:]),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]),
                extrinsics_xf.reshape((-1, num_views) +
                                      extrinsics_xf.shape[1:]),
            )
            return img_features
        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    singlev_features = self._feature_extractor.compute_singlev_features(
                        per_view_img_features[r0:r1].clone(),
                        singlev_scaled_to_orig_xf[r0:r1].clone())
                    img_features_list.append(singlev_features)
                    # feats_l = per_view_img_features[r0:r1].clone()
                    # s2o_l = singlev_scaled_to_orig_xf[r0:r1].clone()
                    # extrinsics_xf_l = extrinsics_xf[r0:r1].clone()
                    # mem_feats = mem_features_l[r0:r1].clone()
                    # prev_s2o = prev_s2o_l[r0:r1].clone()
                    # prev_extrinsics_xf = prev_extrinsics_l[r0:r1].clone()
                    # feats_t = torch.cat([feats_l,mem_feats],dim=0)
                    # s2o_t = torch.cat([s2o_l,prev_s2o],dim=0)
                    # extrinsics_t = torch.cat([extrinsics_xf_l,prev_extrinsics_xf],dim=0)
                    # singlev_features = self._feature_extractor.compute_multiv_features_t(
                    #     feats_l,
                    #     feats_t.unsqueeze(0),
                    #     s2o_t.unsqueeze(0),
                    #     extrinsics_t.unsqueeze(0)
                    # )
                    # img_features_list.append(singlev_features)
                    # print(self._mem_features_l.shape)
                    # print(feats_l.shape)
                    # self._mem_features_l[r0:r1] = feats_l

                    # self._prev_extrinsics_l[r0:r1] = extrinsics_xf_l
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].clone().unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].clone().unsqueeze(0),
                        extrinsics_xf[r0:r1].clone().unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        if not use_memory.all():
            # print('***********')
            self._mem_features_l = img_features.clone()

            self._prev_extrinsics_l = extrinsics_xf.clone()
        mem_features_l = self._mem_features_l
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_cam0_to_world_xf_l = torch.inverse(prev_extrinsics_l)
        prev_cam0_to_cur_cam0_xf_l = extrinsics_xf.bmm(prev_cam0_to_world_xf_l)
        prev_mem_features_l = model_utils.apply_ftl_to_feature_maps(
            prev_cam0_to_cur_cam0_xf_l,
            mem_features_l,
            1.0,
        )
        feats_l_tm = torch.cat([img_features, prev_mem_features_l], dim=1)
        img_feats = self._feature_extractor.compute_multiv_features_t_new(
            img_features, feats_l_tm)

        self._mem_features_l = img_features
        self._prev_extrinsics_l = extrinsics_xf

        return img_feats

    def _forward_feature_extractor_all(self, frame_data: InputFrameData,
                                       frame_desc: InputFrameDesc
                                       ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1))
        bs = per_view_img_features.shape[0]
        memory_idx = frame_desc.memory_idx
        use_memory = frame_desc.use_memory
        required_memory_len = int(torch.max(memory_idx)) + 1
        feats_l = per_view_img_features[0::2].clone()
        feats_r = per_view_img_features[1::2].clone()

        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics)

        extrinsics_xf = frame_data.extrinsics_xf
        extrinsics_xf_l = extrinsics_xf[0::2].clone()
        extrinsics_xf_r = extrinsics_xf[1::2].clone()
        s2o_l = singlev_scaled_to_orig_xf[0::2].clone()
        s2o_r = singlev_scaled_to_orig_xf[1::2].clone()

        mem_features_l = self._mem_features_l
        mem_features_r = self._mem_features_r
        prev_extrinsics_l = self._prev_extrinsics_l
        prev_extrinsics_r = self._prev_extrinsics_r
        prev_s2o_l = self._prev_s2o_l
        prev_s2o_r = self._prev_s2o_r
        if not use_memory.all():
            mem_features_l = feats_l.clone()
            mem_features_r = feats_r.clone()
            prev_extrinsics_l = extrinsics_xf_l.clone()
            prev_extrinsics_r = extrinsics_xf_r.clone()
            prev_s2o_l = s2o_l.clone()
            prev_s2o_r = s2o_r.clone()

        feats_l_tm = torch.stack(
            [x for y in zip(feats_l, mem_features_l) for x in y], dim=0)
        feats_r_tm = torch.stack(
            [x for y in zip(feats_r, mem_features_r) for x in y], dim=0)
        extrinsics_xf_l_tm = torch.stack(
            [x for y in zip(extrinsics_xf_l, prev_extrinsics_l) for x in y],
            dim=0)
        extrinsics_xf_r_tm = torch.stack([
            x for y in zip(extrinsics_xf[1::2].clone(), prev_extrinsics_r)
            for x in y
        ],
                                         dim=0)
        s2o_l_tm = torch.stack([x for y in zip(s2o_l, prev_s2o_l) for x in y],
                               dim=0)
        s2o_r_tm = torch.stack([x for y in zip(s2o_r, prev_s2o_r) for x in y],
                               dim=0)
        feats_0 = torch.cat([feats_l, feats_r], dim=0)
        feats_tm = torch.cat([feats_l_tm, feats_r_tm], dim=0)
        extrinsics_xf_tm = torch.cat([extrinsics_xf_l_tm, extrinsics_xf_r_tm],
                                     dim=0)
        s2o_tm = torch.cat([s2o_l_tm, s2o_r_tm], dim=0)

        img_features_multiv = self._feature_extractor.compute_multiv_features(
            per_view_img_features.reshape((-1, 2) +
                                          per_view_img_features.shape[1:]),
            singlev_scaled_to_orig_xf.reshape(
                (-1, 2) + singlev_scaled_to_orig_xf.shape[1:]),
            extrinsics_xf.reshape((-1, 2) + extrinsics_xf.shape[1:]),
        )
        img_features_tm = self._feature_extractor.compute_multiv_features_t(
            feats_0,
            feats_tm.reshape((-1, 2) + feats_tm.shape[1:]),
            s2o_tm.reshape((-1, 2) + s2o_tm.shape[1:]),
            extrinsics_xf_tm.reshape((-1, 2) + extrinsics_xf_tm.shape[1:]),
        )
        img_features_l = img_features_tm[0:bs // 2].clone()
        img_features_r = img_features_tm[bs // 2:].clone()

        self._mem_features_l = feats_l.clone()
        self._mem_features_r = feats_r.clone()
        self._prev_extrinsics_l = extrinsics_xf_l.clone()
        self._prev_extrinsics_r = extrinsics_xf_r.clone()
        self._prev_s2o_l = s2o_l.clone()
        self._prev_s2o_r = s2o_r.clone()

        return img_features_l, img_features_r

    def _forward_feature_extractor_temporal(self, frame_data: InputFrameData,
                                            frame_desc: InputFrameDesc
                                            ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(frame_data, frame_desc)

        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features

    def _forward_feature_extractor_temporal_all(self,
                                                frame_data: InputFrameData,
                                                frame_desc: InputFrameDesc
                                                ) -> torch.Tensor:
        # Fused img features
        feats_l, feats_r = self._forward_feature_extractor_all(
            frame_data, frame_desc)

        extrinsics_0 = _get_cam0_extrinsics(frame_data, frame_desc)
        extrinsics_1 = _get_cam1_extrinsics(frame_data, frame_desc)

        # Temporal features

        temporal_features_l = self._temporal_sv.forward_temporal_features_l(
            feats_l,
            extrinsics_0,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        temporal_features_r = self._temporal_sv.forward_temporal_features_r(
            feats_r,
            extrinsics_1,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )

        return temporal_features_l, temporal_features_r

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_use_skeleton_sv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features],
                                      dim=1)

        regression_output = self._regressor_k_sv.regress_poses(
            img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(self, frame_data: InputFrameData,
                                     frame_desc: InputFrameDesc
                                     ) -> Dict[str, torch.Tensor]:
        singlev_masks = (frame_desc.sample_range[:, 1] -
                         frame_desc.sample_range[:, 0]) != 1
        assert (singlev_masks.all(
        )), 'Unsupported: found single-view samples when calibration scale'
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output

    def reset_temporal(self):
        self._temporal.reset_mem_features()
        self._temporal_sv.reset_mem_features()

    def forward_multiv(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc)

        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )

        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(temporal_features.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat(
            [temporal_features.clone(),
             skel_features.clone()], dim=1)

        unknown_output = self._regressor_u.regress_poses(temporal_features)
        unknown_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            unknown_output.wrist_xfs,
        )

        known_output = self._regressor_k.regress_poses(img_skel_features)
        known_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output.wrist_xfs,
        )

        regress_output = {
            'unknown_output': unknown_output,
            'known_output': known_output
        }
        return regress_output

    def forward(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ):
        feats_l, feats_r = self._forward_feature_extractor_temporal_all(
            frame_data, frame_desc)
        # print('temporal')
        # feats_l,feats_r = self._forward_feature_extractor_all(
        #     frame_data, frame_desc
        # )
        skel_features = self._skeleton_enc(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and feats_l.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(feats_l.shape[0],
                                                 *skel_features.shape[1:])

        # Concatenate along the channel dimension (1)
        img_skel_features_l = torch.cat(
            [feats_l.clone(), skel_features.clone()], dim=1)
        img_skel_features_r = torch.cat(
            [feats_r.clone(), skel_features.clone()], dim=1)
        img_skel_features = torch.cat(
            [img_skel_features_l, img_skel_features_r], dim=0)

        known_output = self._regressor_k_sv.regress_poses_sv(img_skel_features)
        known_output_l = known_output[0]
        known_output_r = known_output[1]

        known_output_l.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            known_output_l.wrist_xfs,
        )
        known_output_r.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam1_extrinsics(frame_data, frame_desc),
            known_output_r.wrist_xfs,
        )
        regress_output = {
            'known_output_l': known_output_l,
            'known_output_r': known_output_r
        }
        return regress_output
