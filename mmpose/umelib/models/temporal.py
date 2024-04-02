# flake8: noqa
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Tuple

import torch
import torch.nn as nn

from . import model_utils
from .model_opts import ModelOpts


class SimpleConvRNN(nn.Module):

    def __init__(
        self,
        nTemporalBlocks: int,
        nTemporalMemoryChannels: int,
        nImageFeatureChannels: int,
        temporalFTLRatio: float,
        featureMapShape: Tuple[int, int],
    ) -> None:
        super(SimpleConvRNN, self).__init__()
        self._nc_memory = nTemporalMemoryChannels
        n_temporal_channels = nImageFeatureChannels + self._nc_memory

        temporal_module = nn.ModuleList()

        for i in range(nTemporalBlocks):
            nc = n_temporal_channels
            temporal_module.append(nn.Conv2d(nc, nc, kernel_size=1, padding=0))
            # Don't add ReLU in the last block since it makes all features positives
            if i != nTemporalBlocks - 1:
                temporal_module.append(nn.ReLU())

        self._temporal_module = nn.Sequential(*temporal_module)
        self._temporal_ftl_ratio = float(temporalFTLRatio)
        self._input_shape = (n_temporal_channels, *featureMapShape)
        self._mem_features = torch.empty(0)
        self._mem_features_multiv = torch.empty(0)
        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)

        self._prev_extrinsics = torch.empty(0)
        self._prev_extrinsics_multiv = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)

    def input_shape(self) -> Tuple[int, int, int]:
        """
        Return: input shape to self._temporal_module
            [channels, feature_size[0], feature_size[1]]
        """
        return self._input_shape

    def transform_memory_features(
        self,
        prev_extrinsics: torch.Tensor,
        prev_mem_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        use_mem_idx = memory_idx[use_memory]
        # print(use_mem_idx)
        if len(use_mem_idx) != len(use_memory):
            zero_mem_idx = memory_idx[torch.logical_not(use_memory)]
            prev_mem_features[zero_mem_idx] = 0
            prev_extrinsics[zero_mem_idx] = 0

        if len(use_mem_idx) != 0:
            prev_cam0_to_world_xf = torch.inverse(prev_extrinsics[use_mem_idx])
            prev_cam0_to_cur_cam0_xf = cur_extrinsics[use_memory].bmm(
                prev_cam0_to_world_xf)
            prev_mem_features[
                use_mem_idx] = model_utils.apply_ftl_to_feature_maps(
                    prev_cam0_to_cur_cam0_xf,
                    prev_mem_features[use_mem_idx],
                    self._temporal_ftl_ratio,
                )

        # Update prev_extrinsics with cur_extrinsics
        prev_extrinsics[memory_idx] = cur_extrinsics
        # print(prev_mem_features)
        return prev_extrinsics, prev_mem_features

    def forward_one_step(self, prev_mem_features_xfed: torch.Tensor,
                         cur_img_features: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_input = [prev_mem_features_xfed, cur_img_features]

        temporal_out = torch.cat(temporal_input, dim=1)
        temporal_out = self._temporal_module(temporal_out)

        mem_features_out = temporal_out[:, 0:self._nc_memory].clone()
        fused_features = temporal_out[:, self._nc_memory:].clone()

        return mem_features_out, fused_features

    def forward_temporal_features(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features
        prev_extrinsics = self._prev_extrinsics
        if len(self._mem_features) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features) != 0:
                mem_features[0:self._mem_features.
                             shape[0]] = self._mem_features
                prev_extrinsics[0:self._prev_extrinsics.
                                shape[0]] = self._prev_extrinsics
        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx,
            use_memory)
        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features)
        # Update memory features
        mem_features[memory_idx] = mem_features_out
        self._prev_extrinsics = prev_extrinsics
        self._mem_features = mem_features

        return fused_features

    def forward_temporal_features_multiv(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features_multiv
        prev_extrinsics = self._prev_extrinsics_multiv
        if len(self._mem_features_multiv) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features_multiv) != 0:
                mem_features[0:self._mem_features_multiv.
                             shape[0]] = self._mem_features_multiv
                prev_extrinsics[0:self._prev_extrinsics_multiv.
                                shape[0]] = self._prev_extrinsics_multiv

        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx,
            use_memory)

        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features)
        # Update memory features
        mem_features[memory_idx] = mem_features_out

        self._prev_extrinsics_multiv = prev_extrinsics
        self._mem_features_multiv = mem_features

        return fused_features

    def forward_temporal_features_l(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features_l
        prev_extrinsics = self._prev_extrinsics_l
        if len(self._mem_features_l) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features_l) != 0:
                mem_features[0:self._mem_features_l.
                             shape[0]] = self._mem_features_l
                prev_extrinsics[0:self._prev_extrinsics_l.
                                shape[0]] = self._prev_extrinsics_l

        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx,
            use_memory)
        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features)
        # Update memory features
        mem_features[memory_idx] = mem_features_out

        self._prev_extrinsics_l = prev_extrinsics.clone()
        self._mem_features_l = mem_features.clone()

        return fused_features

    def forward_temporal_features_r(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features_r
        prev_extrinsics = self._prev_extrinsics_r
        if len(self._mem_features_r) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features_r) != 0:
                mem_features[0:self._mem_features_r.
                             shape[0]] = self._mem_features_r
                prev_extrinsics[0:self._prev_extrinsics_r.
                                shape[0]] = self._prev_extrinsics_r

        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx,
            use_memory)
        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features)
        # Update memory features
        mem_features[memory_idx] = mem_features_out

        self._prev_extrinsics_r = prev_extrinsics.clone()
        self._mem_features_r = mem_features.clone()

        return fused_features

    def _compute_multiv_xfs(self, singlev_scaled_to_orig_xf: torch.Tensor,
                            extrinsics_xf: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given per-view data for one frame of 2 views, compute transformation
        from its view to the canonical space.

        The canonical space is related to cam0 by a transformation
        `canonical_to_cam0`, which is identity if
        `self._use_unscaled_as_canonical` is `True`, or the scaling transform
        """
        # Use the first camera space as the reference camera.
        xf_0 = extrinsics_xf[:, 0:1].clone()
        xf_inv = torch.inverse(extrinsics_xf)
        xf_to_world = xf_inv @ singlev_scaled_to_orig_xf
        # print(singlev_scaled_to_orig_xf)
        # assert 1==2
        if self._use_unscaled_as_canonical:
            bs = singlev_scaled_to_orig_xf.shape[0]
            dtype = singlev_scaled_to_orig_xf.dtype
            device = singlev_scaled_to_orig_xf.device
            canonical_to_cam0_xf = (
                torch.eye(4, dtype=dtype,
                          device=device).unsqueeze(0).repeat(bs, 1, 1))
            scaled_to_canonical_xf = xf_0 @ xf_to_world
        else:
            canonical_to_cam0_xf = singlev_scaled_to_orig_xf[:, 0].clone()
            s_0 = torch.inverse(singlev_scaled_to_orig_xf[:, 0:1].clone())
            scaled_to_canonical_xf = s_0 @ xf_0 @ xf_to_world

        return scaled_to_canonical_xf, canonical_to_cam0_xf

    def compute_multiv_features(
        self,
        img_features: torch.Tensor,
        singlev_scaled_to_orig_xf: torch.Tensor,
        extrinsics_xf: torch.Tensor,
    ) -> torch.Tensor:
        """
          Fuse image features in the canonical space and transform to cam0 space.
          Note the n_views dimension no longer exists after fusion.
          Output shape should be
          [
              batch_size,
              n_output_feature_channels,
              feature_shape0,
              feature_shape1,
          ]
          """

        batch_size = img_features.shape[0]
        n_views = img_features.shape[1]
        assert img_features.shape[1] == 2, 'Only 2 views supported'

        (
            multiv_scaled_to_canonical_xf,
            multiv_canonical_to_cam0_xf,
        ) = self._compute_multiv_xfs(singlev_scaled_to_orig_xf, extrinsics_xf)

        # Transform all the features to the canonical space
        multiv_canonical_features = model_utils.apply_ftl_to_feature_maps(
            multiv_scaled_to_canonical_xf.reshape(-1, 4, 4),
            torch.flatten(img_features, start_dim=0, end_dim=1),
            self._ftl_ratio,
        ).reshape(img_features.shape)

        # print(torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2).shape)
        # Flatten the view dimension with the channel dimension and apply multi-view fusion
        multiv_fused_img_features = self._multi_view_fusion(
            torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2))
        # print(multiv_fused_img_features)
        # Apply ftl so that the maps are transformed from
        # the canonical space to cam0 space.
        cam0_maps = model_utils.apply_ftl_to_feature_maps(
            multiv_canonical_to_cam0_xf, multiv_fused_img_features,
            self._ftl_ratio)

        return cam0_maps

    def reset_mem_features(self):
        self._mem_features = torch.empty(0)
        self._mem_features_multiv = torch.empty(0)
        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics = torch.empty(0)
        self._prev_extrinsics_multiv = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)


class SimpleConvRNN_CrossView(nn.Module):

    def __init__(
        self,
        nTemporalBlocks: int,
        nTemporalMemoryChannels: int,
        nImageFeatureChannels: int,
        temporalFTLRatio: float,
        featureMapShape: Tuple[int, int],
    ) -> None:
        super(SimpleConvRNN_CrossView, self).__init__()
        self._nc_memory = nTemporalMemoryChannels
        n_temporal_channels = nImageFeatureChannels + self._nc_memory

        temporal_module = nn.ModuleList()

        for i in range(nTemporalBlocks):
            nc = n_temporal_channels
            temporal_module.append(nn.Conv2d(nc, nc, kernel_size=1, padding=0))
            # Don't add ReLU in the last block since it makes all features positives
            if i != nTemporalBlocks - 1:
                temporal_module.append(nn.ReLU())

        self._temporal_module = nn.Sequential(*temporal_module)
        self._temporal_ftl_ratio = float(temporalFTLRatio)
        self._input_shape = (n_temporal_channels, *featureMapShape)
        self._mem_features = torch.empty(0)
        self._mem_features_multiv = torch.empty(0)
        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)

        self._prev_extrinsics = torch.empty(0)
        self._prev_extrinsics_multiv = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)

    def input_shape(self) -> Tuple[int, int, int]:
        """
        Return: input shape to self._temporal_module
            [channels, feature_size[0], feature_size[1]]
        """
        return self._input_shape

    def transform_memory_features(
        self,
        prev_extrinsics: torch.Tensor,
        prev_mem_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        use_mem_idx = memory_idx[use_memory]
        if len(use_mem_idx) != len(use_memory):
            zero_mem_idx = memory_idx[torch.logical_not(use_memory)]
            prev_mem_features[zero_mem_idx] = 0
            prev_extrinsics[zero_mem_idx] = 0

        if len(use_mem_idx) != 0:
            prev_cam0_to_world_xf = torch.inverse(prev_extrinsics[use_mem_idx])
            prev_cam0_to_cur_cam0_xf = cur_extrinsics[use_memory].bmm(
                prev_cam0_to_world_xf)
            prev_mem_features[
                use_mem_idx] = model_utils.apply_ftl_to_feature_maps(
                    prev_cam0_to_cur_cam0_xf,
                    prev_mem_features[use_mem_idx],
                    self._temporal_ftl_ratio,
                )

        # Update prev_extrinsics with cur_extrinsics
        prev_extrinsics[memory_idx] = cur_extrinsics
        return prev_extrinsics, prev_mem_features

    def forward_one_step(self, prev_mem_features_xfed: torch.Tensor,
                         cur_img_features: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_input = [prev_mem_features_xfed, cur_img_features]

        temporal_out = torch.cat(temporal_input, dim=1)
        temporal_out = self._temporal_module(temporal_out)

        mem_features_out = temporal_out[:, 0:self._nc_memory].clone()
        fused_features = temporal_out[:, self._nc_memory:].clone()

        return mem_features_out, fused_features

    def forward_temporal_features(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features
        prev_extrinsics = self._prev_extrinsics
        if len(self._mem_features) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features) != 0:
                mem_features[0:self._mem_features.
                             shape[0]] = self._mem_features
                prev_extrinsics[0:self._prev_extrinsics.
                                shape[0]] = self._prev_extrinsics
        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx,
            use_memory)
        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features)
        # Update memory features
        mem_features[memory_idx] = mem_features_out
        self._prev_extrinsics = prev_extrinsics
        self._mem_features = mem_features

        return fused_features

    def forward_temporal_features_all(
        self,
        feats_multiv: torch.Tensor,
        feats_l: torch.Tensor,
        feats_r: torch.Tensor,
        cur_extrinsics_0: torch.Tensor,
        cur_extrinsics_1: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (feats_multiv.shape[-2], feats_multiv.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features_multiv = self._mem_features_multiv
        prev_extrinsics_multiv = self._prev_extrinsics_multiv
        if len(self._mem_features_multiv) < required_memory_len:
            mem_features_multiv = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=feats_multiv.dtype,
                device=feats_multiv.device,
            )
            prev_extrinsics_multiv = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=feats_multiv.dtype,
                device=feats_multiv.device,
            )
            if len(self._mem_features_multiv) != 0:
                mem_features_multiv[0:self._mem_features_multiv.
                                    shape[0]] = self._mem_features_multiv
                prev_extrinsics_multiv[0:self._prev_extrinsics_multiv.
                                       shape[0]] = self._prev_extrinsics_multiv
        mem_features_l = self._mem_features_l
        prev_extrinsics_l = self._prev_extrinsics_l
        if len(self._mem_features_l) < required_memory_len:
            mem_features_l = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=feats_l.dtype,
                device=feats_l.device,
            )
            prev_extrinsics_l = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=feats_l.dtype,
                device=feats_l.device,
            )
            if len(self._mem_features_l) != 0:
                mem_features_l[0:self._mem_features_l.
                               shape[0]] = self._mem_features_l
                prev_extrinsics_l[0:self._prev_extrinsics_l.
                                  shape[0]] = self._prev_extrinsics_l
        mem_features_r = self._mem_features_r
        prev_extrinsics_r = self._prev_extrinsics_r
        if len(self._mem_features_r) < required_memory_len:
            mem_features_r = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=feats_r.dtype,
                device=feats_r.device,
            )
            prev_extrinsics_r = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=feats_r.dtype,
                device=feats_r.device,
            )
            if len(self._mem_features_r) != 0:
                mem_features_r[0:self._mem_features_r.
                               shape[0]] = self._mem_features_r
                prev_extrinsics_r[0:self._prev_extrinsics_r.
                                  shape[0]] = self._prev_extrinsics_r

        prev_extrinsics_multiv_tmp = prev_extrinsics_multiv
        mem_features_multiv_tmp = mem_features_multiv
        prev_extrinsics_l_tmp = prev_extrinsics_l
        mem_features_l_tmp = mem_features_l
        prev_extrinsics_r_tmp = prev_extrinsics_r
        mem_features_r_tmp = mem_features_r
        if random.random() > 0.7:
            if random.random() < 0.5:
                prev_extrinsics_multiv_tmp = prev_extrinsics_l.clone()
                mem_features_multiv_tmp = mem_features_l.clone()
            else:
                prev_extrinsics_multiv_tmp = prev_extrinsics_r.clone()
                mem_features_multiv_tmp = mem_features_r.clone()
        if random.random() > 0.7:
            if random.random() < 0.5:
                prev_extrinsics_l_tmp = prev_extrinsics_multiv.clone()
                mem_features_l_tmp = mem_features_multiv.clone()
            else:
                prev_extrinsics_l_tmp = prev_extrinsics_r.clone()
                mem_features_l_tmp = mem_features_r.clone()
        if random.random() > 0.7:
            if random.random() < 0.5:
                prev_extrinsics_r_tmp = prev_extrinsics_multiv.clone()
                mem_features_r_tmp = mem_features_multiv.clone()
            else:
                prev_extrinsics_r_tmp = prev_extrinsics_l.clone()
                mem_features_r_tmp = mem_features_l.clone()
        prev_extrinsics_multiv = prev_extrinsics_multiv_tmp
        mem_features_multiv = mem_features_multiv_tmp
        prev_extrinsics_l = prev_extrinsics_l_tmp
        mem_features_l = mem_features_l_tmp
        prev_extrinsics_r = prev_extrinsics_r_tmp
        mem_features_r = mem_features_r_tmp

        prev_extrinsics_multiv, mem_features_multiv = self.transform_memory_features(
            prev_extrinsics_multiv, mem_features_multiv, cur_extrinsics_0,
            memory_idx, use_memory)
        prev_extrinsics_r, mem_features_r = self.transform_memory_features(
            prev_extrinsics_r, mem_features_r, cur_extrinsics_1, memory_idx,
            use_memory)
        prev_extrinsics_l, mem_features_l = self.transform_memory_features(
            prev_extrinsics_l, mem_features_l, cur_extrinsics_0, memory_idx,
            use_memory)

        mem_features_out_multiv, fused_features_multiv = self.forward_one_step(
            mem_features_multiv[memory_idx], feats_multiv)
        mem_features_out_l, fused_features_l = self.forward_one_step(
            mem_features_l[memory_idx], feats_l)
        mem_features_out_r, fused_features_r = self.forward_one_step(
            mem_features_r[memory_idx], feats_r)
        # Update memory features
        mem_features_multiv[memory_idx] = mem_features_out_multiv
        mem_features_l[memory_idx] = mem_features_out_l
        mem_features_r[memory_idx] = mem_features_out_r

        self._prev_extrinsics_multiv = prev_extrinsics_multiv
        self._mem_features_multiv = mem_features_multiv
        self._prev_extrinsics_l = prev_extrinsics_l
        self._mem_features_l = mem_features_l
        self._prev_extrinsics_r = prev_extrinsics_r
        self._mem_features_r = mem_features_r

        return fused_features_multiv, fused_features_l, fused_features_r

    def reset_mem_features(self):
        self._mem_features = torch.empty(0)
        self._mem_features_multiv = torch.empty(0)
        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics = torch.empty(0)
        self._prev_extrinsics_multiv = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)


class SimpleConvRNN_New(nn.Module):

    def __init__(
        self,
        nTemporalBlocks: int,
        nTemporalMemoryChannels: int,
        nImageFeatureChannels: int,
        temporalFTLRatio: float,
        featureMapShape: Tuple[int, int],
        model_opts: ModelOpts,
    ) -> None:
        super(SimpleConvRNN_New, self).__init__()
        self._nc_memory = nTemporalMemoryChannels
        n_temporal_channels = nImageFeatureChannels + self._nc_memory

        temporal_module = nn.ModuleList()

        for i in range(nTemporalBlocks):
            nc = n_temporal_channels
            temporal_module.append(nn.Conv2d(nc, nc, kernel_size=1, padding=0))
            # Don't add ReLU in the last block since it makes all features positives
            if i != nTemporalBlocks - 1:
                temporal_module.append(nn.ReLU())

        self._multi_view_fusion = model_utils.create_multi_view_fusion_layers(
            nImageFeatureChannels * 2,
            nImageFeatureChannels,
            model_opts.nMultiViewFusionBlocks,
        )
        self._use_unscaled_as_canonical = model_opts.useUnscaledAsCanonical

        self._temporal_module = nn.Sequential(*temporal_module)
        self._temporal_ftl_ratio = float(temporalFTLRatio)
        self._input_shape = (n_temporal_channels, *featureMapShape)
        self._mem_features = torch.empty(0)

        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)

        self._prev_extrinsics = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)

    def input_shape(self) -> Tuple[int, int, int]:
        """
        Return: input shape to self._temporal_module
            [channels, feature_size[0], feature_size[1]]
        """
        return self._input_shape

    def transform_memory_features(
        self,
        prev_extrinsics: torch.Tensor,
        prev_mem_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        use_mem_idx = memory_idx[use_memory]
        if len(use_mem_idx) != len(use_memory):
            zero_mem_idx = memory_idx[torch.logical_not(use_memory)]
            prev_mem_features[zero_mem_idx] = 0
            prev_extrinsics[zero_mem_idx] = 0

        if len(use_mem_idx) != 0:
            prev_cam0_to_world_xf = torch.inverse(prev_extrinsics[use_mem_idx])
            prev_cam0_to_cur_cam0_xf = cur_extrinsics[use_memory].bmm(
                prev_cam0_to_world_xf)
            prev_mem_features[
                use_mem_idx] = model_utils.apply_ftl_to_feature_maps(
                    prev_cam0_to_cur_cam0_xf,
                    prev_mem_features[use_mem_idx],
                    self._temporal_ftl_ratio,
                )

        # Update prev_extrinsics with cur_extrinsics
        prev_extrinsics[memory_idx] = cur_extrinsics
        return prev_extrinsics, prev_mem_features

    def forward_one_step(self, prev_mem_features_xfed: torch.Tensor,
                         cur_img_features: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_input = [prev_mem_features_xfed, cur_img_features]

        temporal_out = torch.cat(temporal_input, dim=1)
        temporal_out = self._temporal_module(temporal_out)

        mem_features_out = temporal_out[:, 0:self._nc_memory].clone()
        fused_features = temporal_out[:, self._nc_memory:].clone()

        return mem_features_out, fused_features

    def forward_temporal_features(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features
        prev_extrinsics = self._prev_extrinsics
        if len(self._mem_features) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features) != 0:
                mem_features[0:self._mem_features.
                             shape[0]] = self._mem_features
                prev_extrinsics[0:self._prev_extrinsics.
                                shape[0]] = self._prev_extrinsics
        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx,
            use_memory)
        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features)
        # Update memory features
        mem_features[memory_idx] = mem_features_out
        self._prev_extrinsics = prev_extrinsics
        self._mem_features = mem_features

        return fused_features

    def forward_temporal_features_l(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features_l
        prev_extrinsics = self._prev_extrinsics_l
        if len(self._mem_features_l) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features_l) != 0:
                mem_features[0:self._mem_features_l.
                             shape[0]] = self._mem_features_l
                prev_extrinsics[0:self._prev_extrinsics_l.
                                shape[0]] = self._prev_extrinsics_l

        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx,
            use_memory)
        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features)
        # Update memory features
        mem_features[memory_idx] = mem_features_out

        self._prev_extrinsics_l = prev_extrinsics
        self._mem_features_l = mem_features

        return fused_features

    def forward_temporal_features_r(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features_r
        prev_extrinsics = self._prev_extrinsics_r
        if len(self._mem_features_r) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features_r) != 0:
                mem_features[0:self._mem_features_r.
                             shape[0]] = self._mem_features_r
                prev_extrinsics[0:self._prev_extrinsics_r.
                                shape[0]] = self._prev_extrinsics_r

        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx,
            use_memory)
        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features)
        # Update memory features
        mem_features[memory_idx] = mem_features_out

        self._prev_extrinsics_r = prev_extrinsics
        self._mem_features_r = mem_features

        return fused_features

    def _compute_multiv_xfs(self, singlev_scaled_to_orig_xf: torch.Tensor,
                            extrinsics_xf: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given per-view data for one frame of 2 views, compute transformation
        from its view to the canonical space.

        The canonical space is related to cam0 by a transformation
        `canonical_to_cam0`, which is identity if
        `self._use_unscaled_as_canonical` is `True`, or the scaling transform
        """
        # Use the first camera space as the reference camera.
        xf_0 = extrinsics_xf[:, 0:1].clone()
        xf_inv = torch.inverse(extrinsics_xf)
        xf_to_world = xf_inv @ singlev_scaled_to_orig_xf
        # print(singlev_scaled_to_orig_xf)
        # assert 1==2
        if self._use_unscaled_as_canonical:
            bs = singlev_scaled_to_orig_xf.shape[0]
            dtype = singlev_scaled_to_orig_xf.dtype
            device = singlev_scaled_to_orig_xf.device
            canonical_to_cam0_xf = (
                torch.eye(4, dtype=dtype,
                          device=device).unsqueeze(0).repeat(bs, 1, 1))
            scaled_to_canonical_xf = xf_0 @ xf_to_world
        else:
            canonical_to_cam0_xf = singlev_scaled_to_orig_xf[:, 0].clone()
            s_0 = torch.inverse(singlev_scaled_to_orig_xf[:, 0:1].clone())
            scaled_to_canonical_xf = s_0 @ xf_0 @ xf_to_world

        return scaled_to_canonical_xf, canonical_to_cam0_xf

    def compute_multiv_features(
        self,
        img_features: torch.Tensor,
        singlev_scaled_to_orig_xf: torch.Tensor,
        extrinsics_xf: torch.Tensor,
    ) -> torch.Tensor:
        """
          Fuse image features in the canonical space and transform to cam0 space.
          Note the n_views dimension no longer exists after fusion.
          Output shape should be
          [
              batch_size,
              n_output_feature_channels,
              feature_shape0,
              feature_shape1,
          ]
          """

        batch_size = img_features.shape[0]
        n_views = img_features.shape[1]
        assert img_features.shape[1] == 2, 'Only 2 views supported'

        (
            multiv_scaled_to_canonical_xf,
            multiv_canonical_to_cam0_xf,
        ) = self._compute_multiv_xfs(singlev_scaled_to_orig_xf, extrinsics_xf)

        # print(img_features.shape)
        # assert 1==2
        # Transform all the features to the canonical space
        multiv_canonical_features = model_utils.apply_ftl_to_feature_maps(
            multiv_scaled_to_canonical_xf.reshape(-1, 4, 4),
            torch.flatten(img_features, start_dim=0, end_dim=1),
            self._temporal_ftl_ratio,
        ).reshape(img_features.shape)

        # print(torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2).shape)
        # Flatten the view dimension with the channel dimension and apply multi-view fusion
        multiv_fused_img_features = self._multi_view_fusion(
            torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2))
        # print(multiv_fused_img_features)
        # Apply ftl so that the maps are transformed from
        # the canonical space to cam0 space.
        cam0_maps = model_utils.apply_ftl_to_feature_maps(
            multiv_canonical_to_cam0_xf, multiv_fused_img_features,
            self._temporal_ftl_ratio)

        return cam0_maps

    def reset_mem_features(self):
        self._mem_features = torch.empty(0)
        self._mem_features_multiv = torch.empty(0)
        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics = torch.empty(0)
        self._prev_extrinsics_multiv = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)


class SimpleConvRNN_CCF(nn.Module):

    def __init__(
        self,
        nTemporalBlocks: int,
        nTemporalMemoryChannels: int,
        nImageFeatureChannels: int,
        temporalFTLRatio: float,
        featureMapShape: Tuple[int, int],
    ) -> None:
        super(SimpleConvRNN_CCF, self).__init__()
        # nImageFeatureChannels = nImageFeatureChannels*2
        self._nc_memory = nTemporalMemoryChannels
        n_temporal_channels = nImageFeatureChannels + self._nc_memory

        temporal_module = nn.ModuleList()

        for i in range(nTemporalBlocks):
            nc = n_temporal_channels
            temporal_module.append(nn.Conv2d(nc, nc, kernel_size=1, padding=0))
            # Don't add ReLU in the last block since it makes all features positives
            if i != nTemporalBlocks - 1:
                temporal_module.append(nn.ReLU())

        self._temporal_module = nn.Sequential(*temporal_module)
        self._temporal_ftl_ratio = float(temporalFTLRatio)
        self._input_shape = (n_temporal_channels, *featureMapShape)
        self._mem_features = torch.empty(0)
        self._mem_features_multiv = torch.empty(0)
        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)

        self._prev_extrinsics = torch.empty(0)
        self._prev_extrinsics_multiv = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)

    def input_shape(self) -> Tuple[int, int, int]:
        """
        Return: input shape to self._temporal_module
            [channels, feature_size[0], feature_size[1]]
        """
        return self._input_shape

    def transform_memory_features(
        self,
        prev_extrinsics: torch.Tensor,
        prev_mem_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        use_mem_idx = memory_idx[use_memory]
        if len(use_mem_idx) != len(use_memory):
            zero_mem_idx = memory_idx[torch.logical_not(use_memory)]
            prev_mem_features[zero_mem_idx] = 0
            prev_extrinsics[zero_mem_idx] = 0

        if len(use_mem_idx) != 0:
            prev_cam0_to_world_xf = torch.inverse(prev_extrinsics[use_mem_idx])
            prev_cam0_to_cur_cam0_xf = cur_extrinsics[use_memory].bmm(
                prev_cam0_to_world_xf)
            prev_mem_features[
                use_mem_idx] = model_utils.apply_ftl_to_feature_maps(
                    prev_cam0_to_cur_cam0_xf,
                    prev_mem_features[use_mem_idx],
                    self._temporal_ftl_ratio,
                )

        # Update prev_extrinsics with cur_extrinsics
        prev_extrinsics[memory_idx] = cur_extrinsics
        return prev_extrinsics, prev_mem_features

    def forward_one_step(self, prev_mem_features_xfed: torch.Tensor,
                         cur_img_features: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_input = [prev_mem_features_xfed, cur_img_features]

        temporal_out = torch.cat(temporal_input, dim=1)
        temporal_out = self._temporal_module(temporal_out)

        mem_features_out = temporal_out[:, 0:self._nc_memory].clone()
        fused_features = temporal_out[:, self._nc_memory:].clone()

        return mem_features_out, fused_features

    def forward_temporal_features(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features
        prev_extrinsics = self._prev_extrinsics
        if len(self._mem_features) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features) != 0:
                mem_features[0:self._mem_features.
                             shape[0]] = self._mem_features
                prev_extrinsics[0:self._prev_extrinsics.
                                shape[0]] = self._prev_extrinsics
        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx,
            use_memory)
        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features)
        # Update memory features
        mem_features[memory_idx] = mem_features_out
        self._prev_extrinsics = prev_extrinsics
        self._mem_features = mem_features

        return fused_features

    def forward_temporal_features_multiv(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features_multiv
        prev_extrinsics = self._prev_extrinsics_multiv
        if len(self._mem_features_multiv) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features_multiv) != 0:
                mem_features[0:self._mem_features_multiv.
                             shape[0]] = self._mem_features_multiv
                prev_extrinsics[0:self._prev_extrinsics_multiv.
                                shape[0]] = self._prev_extrinsics_multiv

        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx,
            use_memory)

        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features)
        # Update memory features
        mem_features[memory_idx] = mem_features_out

        self._prev_extrinsics_multiv = prev_extrinsics
        self._mem_features_multiv = mem_features

        return fused_features

    def forward_temporal_features_l(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features_l
        prev_extrinsics = self._prev_extrinsics_l
        if len(self._mem_features_l) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features_l) != 0:
                mem_features[0:self._mem_features_l.
                             shape[0]] = self._mem_features_l
                prev_extrinsics[0:self._prev_extrinsics_l.
                                shape[0]] = self._prev_extrinsics_l

        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx,
            use_memory)
        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features)
        # Update memory features
        mem_features[memory_idx] = mem_features_out

        self._prev_extrinsics_l = prev_extrinsics
        self._mem_features_l = mem_features

        return fused_features

    def forward_temporal_features_r(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features_r
        prev_extrinsics = self._prev_extrinsics_r
        if len(self._mem_features_r) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features_r) != 0:
                mem_features[0:self._mem_features_r.
                             shape[0]] = self._mem_features_r
                prev_extrinsics[0:self._prev_extrinsics_r.
                                shape[0]] = self._prev_extrinsics_r

        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx,
            use_memory)
        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features)
        # Update memory features
        mem_features[memory_idx] = mem_features_out

        self._prev_extrinsics_r = prev_extrinsics
        self._mem_features_r = mem_features

        return fused_features

    def _compute_multiv_xfs(self, singlev_scaled_to_orig_xf: torch.Tensor,
                            extrinsics_xf: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given per-view data for one frame of 2 views, compute transformation
        from its view to the canonical space.

        The canonical space is related to cam0 by a transformation
        `canonical_to_cam0`, which is identity if
        `self._use_unscaled_as_canonical` is `True`, or the scaling transform
        """
        # Use the first camera space as the reference camera.
        xf_0 = extrinsics_xf[:, 0:1].clone()
        xf_inv = torch.inverse(extrinsics_xf)
        xf_to_world = xf_inv @ singlev_scaled_to_orig_xf
        # print(singlev_scaled_to_orig_xf)
        # assert 1==2
        if self._use_unscaled_as_canonical:
            bs = singlev_scaled_to_orig_xf.shape[0]
            dtype = singlev_scaled_to_orig_xf.dtype
            device = singlev_scaled_to_orig_xf.device
            canonical_to_cam0_xf = (
                torch.eye(4, dtype=dtype,
                          device=device).unsqueeze(0).repeat(bs, 1, 1))
            scaled_to_canonical_xf = xf_0 @ xf_to_world
        else:
            canonical_to_cam0_xf = singlev_scaled_to_orig_xf[:, 0].clone()
            s_0 = torch.inverse(singlev_scaled_to_orig_xf[:, 0:1].clone())
            scaled_to_canonical_xf = s_0 @ xf_0 @ xf_to_world

        return scaled_to_canonical_xf, canonical_to_cam0_xf

    def compute_multiv_features(
        self,
        img_features: torch.Tensor,
        singlev_scaled_to_orig_xf: torch.Tensor,
        extrinsics_xf: torch.Tensor,
    ) -> torch.Tensor:
        """
          Fuse image features in the canonical space and transform to cam0 space.
          Note the n_views dimension no longer exists after fusion.
          Output shape should be
          [
              batch_size,
              n_output_feature_channels,
              feature_shape0,
              feature_shape1,
          ]
          """

        batch_size = img_features.shape[0]
        n_views = img_features.shape[1]
        assert img_features.shape[1] == 2, 'Only 2 views supported'

        (
            multiv_scaled_to_canonical_xf,
            multiv_canonical_to_cam0_xf,
        ) = self._compute_multiv_xfs(singlev_scaled_to_orig_xf, extrinsics_xf)

        # Transform all the features to the canonical space
        multiv_canonical_features = model_utils.apply_ftl_to_feature_maps(
            multiv_scaled_to_canonical_xf.reshape(-1, 4, 4),
            torch.flatten(img_features, start_dim=0, end_dim=1),
            self._ftl_ratio,
        ).reshape(img_features.shape)

        # print(torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2).shape)
        # Flatten the view dimension with the channel dimension and apply multi-view fusion
        multiv_fused_img_features = self._multi_view_fusion(
            torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2))
        # print(multiv_fused_img_features)
        # Apply ftl so that the maps are transformed from
        # the canonical space to cam0 space.
        cam0_maps = model_utils.apply_ftl_to_feature_maps(
            multiv_canonical_to_cam0_xf, multiv_fused_img_features,
            self._ftl_ratio)

        return cam0_maps

    def reset_mem_features(self):
        self._mem_features = torch.empty(0)
        self._mem_features_multiv = torch.empty(0)
        self._mem_features_l = torch.empty(0)
        self._mem_features_r = torch.empty(0)
        self._prev_extrinsics = torch.empty(0)
        self._prev_extrinsics_multiv = torch.empty(0)
        self._prev_extrinsics_l = torch.empty(0)
        self._prev_extrinsics_r = torch.empty(0)


def create_temporal_model(model_opts: ModelOpts,
                          feature_map_shape: Tuple[int, int]) -> SimpleConvRNN:
    return SimpleConvRNN(
        nTemporalBlocks=model_opts.nTemporalBlocks,
        nTemporalMemoryChannels=model_opts.nTemporalMemoryChannels,
        nImageFeatureChannels=model_opts.nImageFeatureChannels,
        temporalFTLRatio=model_opts.temporalFTLRatio,
        featureMapShape=feature_map_shape,
    )


def create_temporal_model_new(model_opts: ModelOpts,
                              feature_map_shape: Tuple[int,
                                                       int]) -> SimpleConvRNN:
    return SimpleConvRNN_New(
        nTemporalBlocks=model_opts.nTemporalBlocks,
        nTemporalMemoryChannels=model_opts.nTemporalMemoryChannels,
        nImageFeatureChannels=model_opts.nImageFeatureChannels,
        temporalFTLRatio=model_opts.temporalFTLRatio,
        featureMapShape=feature_map_shape,
        model_opts=model_opts,
    )


def create_temporal_model_crossview(model_opts: ModelOpts,
                                    feature_map_shape: Tuple[int, int]
                                    ) -> SimpleConvRNN:
    return SimpleConvRNN_CrossView(
        nTemporalBlocks=model_opts.nTemporalBlocks,
        nTemporalMemoryChannels=model_opts.nTemporalMemoryChannels,
        nImageFeatureChannels=model_opts.nImageFeatureChannels,
        temporalFTLRatio=model_opts.temporalFTLRatio,
        featureMapShape=feature_map_shape,
    )


def create_temporal_model_ccf(model_opts: ModelOpts,
                              feature_map_shape: Tuple[int,
                                                       int]) -> SimpleConvRNN:
    return SimpleConvRNN_CCF(
        nTemporalBlocks=model_opts.nTemporalBlocks,
        nTemporalMemoryChannels=model_opts.nTemporalMemoryChannels,
        nImageFeatureChannels=model_opts.nImageFeatureChannels,
        temporalFTLRatio=model_opts.temporalFTLRatio,
        featureMapShape=feature_map_shape,
    )
