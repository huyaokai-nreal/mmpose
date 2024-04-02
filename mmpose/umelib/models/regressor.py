# flake8: noqa
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import model_utils
from .model_opts import ModelOpts


def _gen_rigid_features():
    rigid_samples = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        # xy plane
        [-1, -1, 0],
        # xz plane
        [-1, 0, -1],
        # yz plane
        [0, -1, -1],
    ])

    rigid_samples_rescaled = np.empty(rigid_samples.shape)
    expected_norm = 0.1

    for i in range(len(rigid_samples)):
        norm = np.linalg.norm(rigid_samples[i])
        if norm == 0:
            rigid_samples_rescaled[i] = rigid_samples[i]
        else:
            rigid_samples_rescaled[i] = rigid_samples[i] / norm * expected_norm

    rigid_samples_rescaled = torch.from_numpy(rigid_samples_rescaled).float()

    return rigid_samples_rescaled


def get_output_index_ranges(
    mo: ModelOpts,
    predict_skel_scale: bool,
) -> Tuple[Dict[str, Tuple[int, int]], int]:
    rigid_samples = _gen_rigid_features()

    assert mo.nWristRigidPts <= len(rigid_samples), (
        'Max supported n_wrist_rigid_pts is '
        f'{len(rigid_samples)}, got {mo.nWristRigidPts}')

    output_dims = {
        'joint_angles': 20,
        'wrist_xfs': mo.nWristRigidPts * 3,
        'skel_scales': 1 if predict_skel_scale else 0,
        'landmark_uncertainty_sigmas': 21,
    }
    n_output_dims = 0
    output_index_range = {}
    for k, v in output_dims.items():
        if v != 0:
            output_index_range[k] = (n_output_dims, n_output_dims + v)
            n_output_dims = n_output_dims + v
    return output_index_range, n_output_dims


def decode_joint_angles(finger_angles: torch.Tensor):
    wrist_angles = torch.zeros(
        finger_angles.shape[0],
        2,
        device=finger_angles.device,
        dtype=finger_angles.dtype,
    )
    joint_angles = torch.cat([finger_angles, wrist_angles], dim=1)

    return joint_angles


def decode_wrist_xfs_svd(
    pred_pts_features: torch.Tensor,
    rigid_pts_src: torch.Tensor,
) -> torch.Tensor:
    batch_size = pred_pts_features.shape[0]
    rigid_points = pred_pts_features.reshape(pred_pts_features.shape[0], -1, 3)

    from_points = rigid_pts_src.to(rigid_points.device)
    from_points = (
        from_points.unsqueeze(0).expand(batch_size, from_points.shape[0],
                                        from_points.shape[1]).clone())

    wrist_xfs = model_utils.procrustes_align(from_points, rigid_points)

    return wrist_xfs


def decode_skel_scales(raw_features: torch.Tensor, ) -> torch.Tensor:
    log_scales = raw_features.reshape(-1)
    # In general, The calibrated skeleton scale values are 0.8~1.2.
    # The log scale values predicted by the network will be -0.22~0.18
    skel_scales = torch.exp(log_scales)
    return skel_scales


def decode_landmark_unc_sigmas(raw_features: torch.Tensor, ) -> torch.Tensor:
    unc_sigmas = torch.clamp(nn.functional.softplus(raw_features), min=1e-5)
    return unc_sigmas


@dataclass
class RegressorOutput:
    joint_angles: torch.Tensor
    wrist_xfs: torch.Tensor
    skel_scales: Optional[torch.Tensor] = None
    landmark_uncertainty_sigmas: Optional[torch.Tensor] = None


class PoseRegressor(nn.Module):

    def __init__(
        self,
        n_channels_in: int,
        n_output_dims: int,
        output_index_ranges: Dict[str, Tuple[int, int]],
        n_blocks: int,
        n_wrist_rigid_pts: int,
        feature_map_sizes: Tuple[int, int],
    ):
        super().__init__()

        rigid_samples = _gen_rigid_features()
        assert n_wrist_rigid_pts <= len(rigid_samples)
        self._left_wrist_sample_points = rigid_samples[:
                                                       n_wrist_rigid_pts].clone(
                                                       )

        self._pose_regression_layers = model_utils.create_pose_regression_layers(
            n_in_channels=n_channels_in,
            n_blocks=n_blocks,
            n_out_channels=n_output_dims,
        )
        self._output_index_ranges = output_index_ranges
        self._input_shape = (n_channels_in, *feature_map_sizes)

    def input_shape(self) -> Tuple[int, int, int]:
        """
        Return: input shape to self._pose_regression_layers
            [channels, feature_size[0], feature_size[1]]
        """
        return self._input_shape

    def regress_poses(self,
                      img_skel_features: torch.Tensor,
                      left_hand: bool = True) -> Dict[str, torch.Tensor]:
        pose_features = self._pose_regression_layers(img_skel_features)
        pose_features = torch.flatten(pose_features, 1)

        output_dict = {}
        for key, f_range in self._output_index_ranges.items():
            raw_features = pose_features[:, f_range[0]:f_range[1]].clone()
            if key == 'joint_angles':
                output_dict[key] = decode_joint_angles(raw_features.clone())
            elif key == 'wrist_xfs':
                output_dict[key] = decode_wrist_xfs_svd(
                    raw_features.clone(),
                    self._left_wrist_sample_points,
                )
            elif key == 'skel_scales':
                output_dict[key] = decode_skel_scales(raw_features.clone())
            elif key == 'landmark_uncertainty_sigmas':
                output_dict[key] = decode_landmark_unc_sigmas(
                    raw_features.clone())
            else:
                raise ValueError(f'Unknown output key: {key}')

        return RegressorOutput(**output_dict)

    def regress_poses_smv(self,
                          img_skel_features: torch.Tensor,
                          left_hand: bool = True) -> Dict[str, torch.Tensor]:
        pose_features = self._pose_regression_layers(img_skel_features)
        pose_features = torch.flatten(pose_features, 1)
        bs = pose_features.shape[0] // 3

        output_dict = {}
        outputs = []
        for i in range(3):
            for key, f_range in self._output_index_ranges.items():
                raw_features = pose_features[i * bs:(i + 1) * bs,
                                             f_range[0]:f_range[1]].clone()
                if key == 'joint_angles':
                    output_dict[key] = decode_joint_angles(
                        raw_features.clone())
                elif key == 'wrist_xfs':
                    output_dict[key] = decode_wrist_xfs_svd(
                        raw_features.clone(),
                        self._left_wrist_sample_points,
                    )
                elif key == 'skel_scales':
                    output_dict[key] = decode_skel_scales(raw_features.clone())
                elif key == 'landmark_uncertainty_sigmas':
                    output_dict[key] = decode_landmark_unc_sigmas(
                        raw_features.clone())
                else:
                    raise ValueError(f'Unknown output key: {key}')
            outputs.append(RegressorOutput(**output_dict))
        return outputs

    def regress_poses_smv_2(self,
                            img_skel_features: torch.Tensor,
                            left_hand: bool = True) -> Dict[str, torch.Tensor]:
        pose_features = self._pose_regression_layers(img_skel_features)
        pose_features = torch.flatten(pose_features, 1)
        bs = pose_features.shape[0] // 2

        output_dict = {}
        outputs = []
        for i in range(2):
            for key, f_range in self._output_index_ranges.items():
                raw_features = pose_features[i * bs:(i + 1) * bs,
                                             f_range[0]:f_range[1]].clone()
                if key == 'joint_angles':
                    output_dict[key] = decode_joint_angles(
                        raw_features.clone())
                elif key == 'wrist_xfs':
                    output_dict[key] = decode_wrist_xfs_svd(
                        raw_features.clone(),
                        self._left_wrist_sample_points,
                    )
                elif key == 'skel_scales':
                    output_dict[key] = decode_skel_scales(raw_features.clone())
                elif key == 'landmark_uncertainty_sigmas':
                    output_dict[key] = decode_landmark_unc_sigmas(
                        raw_features.clone())
                else:
                    raise ValueError(f'Unknown output key: {key}')
            outputs.append(RegressorOutput(**output_dict))
        return outputs

    def regress_poses_sv(self,
                         img_skel_features: torch.Tensor,
                         left_hand: bool = True) -> Dict[str, torch.Tensor]:
        pose_features = self._pose_regression_layers(img_skel_features)
        pose_features = torch.flatten(pose_features, 1)
        bs = pose_features.shape[0] // 2

        output_dict = {}
        outputs = []
        for i in range(2):
            for key, f_range in self._output_index_ranges.items():
                raw_features = pose_features[i * bs:(i + 1) * bs,
                                             f_range[0]:f_range[1]].clone()
                if key == 'joint_angles':
                    output_dict[key] = decode_joint_angles(
                        raw_features.clone())
                elif key == 'wrist_xfs':
                    output_dict[key] = decode_wrist_xfs_svd(
                        raw_features.clone(),
                        self._left_wrist_sample_points,
                    )
                elif key == 'skel_scales':
                    output_dict[key] = decode_skel_scales(raw_features.clone())
                elif key == 'landmark_uncertainty_sigmas':
                    output_dict[key] = decode_landmark_unc_sigmas(
                        raw_features.clone())
                else:
                    raise ValueError(f'Unknown output key: {key}')
            outputs.append(RegressorOutput(**output_dict))
        return outputs


class PoseRegressor_Deconv(nn.Module):

    def __init__(
        self,
        n_channels_in: int,
        n_output_dims: int,
        output_index_ranges: Dict[str, Tuple[int, int]],
        n_blocks: int,
        n_wrist_rigid_pts: int,
        feature_map_sizes: Tuple[int, int],
    ):
        super().__init__()
        print('deconv')
        rigid_samples = _gen_rigid_features()
        assert n_wrist_rigid_pts <= len(rigid_samples)
        self._left_wrist_sample_points = rigid_samples[:
                                                       n_wrist_rigid_pts].clone(
                                                       )
        self.deconv_layers = self._make_deconv_layer(
            1,  # 1
            [n_channels_in],  # [d_model]
            [4],  # [4]
        )
        self._pose_regression_layers = model_utils.create_pose_regression_layers(
            n_in_channels=n_channels_in,
            n_blocks=n_blocks,
            n_out_channels=n_output_dims,
        )
        self._output_index_ranges = output_index_ranges
        self._input_shape = (n_channels_in, *feature_map_sizes)

    def input_shape(self) -> Tuple[int, int, int]:
        """
        Return: input shape to self._pose_regression_layers
            [channels, feature_size[0], feature_size[1]]
        """
        return self._input_shape

    def regress_poses(self,
                      img_skel_features: torch.Tensor,
                      left_hand: bool = True) -> Dict[str, torch.Tensor]:
        pose_features = self._pose_regression_layers(img_skel_features)
        pose_features = torch.flatten(pose_features, 1)

        output_dict = {}
        for key, f_range in self._output_index_ranges.items():
            raw_features = pose_features[:, f_range[0]:f_range[1]].clone()
            if key == 'joint_angles':
                output_dict[key] = decode_joint_angles(raw_features.clone())
            elif key == 'wrist_xfs':
                output_dict[key] = decode_wrist_xfs_svd(
                    raw_features.clone(),
                    self._left_wrist_sample_points,
                )
            elif key == 'skel_scales':
                output_dict[key] = decode_skel_scales(raw_features.clone())
            elif key == 'landmark_uncertainty_sigmas':
                output_dict[key] = decode_landmark_unc_sigmas(
                    raw_features.clone())
            else:
                raise ValueError(f'Unknown output key: {key}')

        return RegressorOutput(**output_dict)

    def regress_poses_smv(self,
                          img_skel_features: torch.Tensor,
                          left_hand: bool = True) -> Dict[str, torch.Tensor]:
        feats_deconv = self.deconv_layers(img_skel_features)
        pose_features = self._pose_regression_layers(feats_deconv)
        pose_features = torch.flatten(pose_features, 1)
        bs = pose_features.shape[0] // 3

        output_dict = {}
        outputs = []
        for i in range(3):
            for key, f_range in self._output_index_ranges.items():
                raw_features = pose_features[i * bs:(i + 1) * bs,
                                             f_range[0]:f_range[1]].clone()
                if key == 'joint_angles':
                    output_dict[key] = decode_joint_angles(
                        raw_features.clone())
                elif key == 'wrist_xfs':
                    output_dict[key] = decode_wrist_xfs_svd(
                        raw_features.clone(),
                        self._left_wrist_sample_points,
                    )
                elif key == 'skel_scales':
                    output_dict[key] = decode_skel_scales(raw_features.clone())
                elif key == 'landmark_uncertainty_sigmas':
                    output_dict[key] = decode_landmark_unc_sigmas(
                        raw_features.clone())
                else:
                    raise ValueError(f'Unknown output key: {key}')
            outputs.append(RegressorOutput(**output_dict))
        return outputs

    def regress_poses_smv_2(self,
                            img_skel_features: torch.Tensor,
                            left_hand: bool = True) -> Dict[str, torch.Tensor]:
        pose_features = self._pose_regression_layers(img_skel_features)
        pose_features = torch.flatten(pose_features, 1)
        bs = pose_features.shape[0] // 2

        output_dict = {}
        outputs = []
        for i in range(2):
            for key, f_range in self._output_index_ranges.items():
                raw_features = pose_features[i * bs:(i + 1) * bs,
                                             f_range[0]:f_range[1]].clone()
                if key == 'joint_angles':
                    output_dict[key] = decode_joint_angles(
                        raw_features.clone())
                elif key == 'wrist_xfs':
                    output_dict[key] = decode_wrist_xfs_svd(
                        raw_features.clone(),
                        self._left_wrist_sample_points,
                    )
                elif key == 'skel_scales':
                    output_dict[key] = decode_skel_scales(raw_features.clone())
                elif key == 'landmark_uncertainty_sigmas':
                    output_dict[key] = decode_landmark_unc_sigmas(
                        raw_features.clone())
                else:
                    raise ValueError(f'Unknown output key: {key}')
            outputs.append(RegressorOutput(**output_dict))
        return outputs

    def regress_poses_sv(self,
                         img_skel_features: torch.Tensor,
                         left_hand: bool = True) -> Dict[str, torch.Tensor]:
        pose_features = self._pose_regression_layers(img_skel_features)
        pose_features = torch.flatten(pose_features, 1)
        bs = pose_features.shape[0] // 2

        output_dict = {}
        outputs = []
        for i in range(2):
            for key, f_range in self._output_index_ranges.items():
                raw_features = pose_features[i * bs:(i + 1) * bs,
                                             f_range[0]:f_range[1]].clone()
                if key == 'joint_angles':
                    output_dict[key] = decode_joint_angles(
                        raw_features.clone())
                elif key == 'wrist_xfs':
                    output_dict[key] = decode_wrist_xfs_svd(
                        raw_features.clone(),
                        self._left_wrist_sample_points,
                    )
                elif key == 'skel_scales':
                    output_dict[key] = decode_skel_scales(raw_features.clone())
                elif key == 'landmark_uncertainty_sigmas':
                    output_dict[key] = decode_landmark_unc_sigmas(
                        raw_features.clone())
                else:
                    raise ValueError(f'Unknown output key: {key}')
            outputs.append(RegressorOutput(**output_dict))
        return outputs

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
