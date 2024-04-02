# flake8: noqa
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
import torch.nn as nn

from . import model_utils
from .ccf_modules import CrossCueFusion
from .model_opts import ModelOpts


class FeatureExtractor(nn.Module):
    """While this is a nn.Module, it doesn't provide any forward function.

    Instead it's a bundle of utility functions that produce features for
    different stages.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        model_opts: ModelOpts,
    ) -> None:
        super(FeatureExtractor, self).__init__()
        nc_img_features = model_opts.nImageFeatureChannels
        num_views = 2
        self._input_img_sizes = input_size
        self._ftl_ratio = float(model_opts.spatialFTLRatio)
        self._use_unscaled_as_canonical = model_opts.useUnscaledAsCanonical
        # Create the image backbone
        self._image_backbone, backbone_outshape = model_utils.create_backbone(
            model_opts.network,
            input_size,
            nc_img_features,
        )
        self._backbone_out_feature_sizes: List[int] = backbone_outshape[-2:]
        self._multi_view_fusion = model_utils.create_multi_view_fusion_layers(
            nc_img_features * num_views,
            nc_img_features,
            model_opts.nMultiViewFusionBlocks,
        )

    @property
    def input_image_sizes(self) -> Tuple[int, int]:
        return self._input_img_sizes

    @property
    def output_feature_sizes(self) -> Tuple[int, int]:
        """
        Return: feature map resolution from feature extractor
            [feature_size[0], feature_size[1]]
        """
        return (*self._backbone_out_feature_sizes, )

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

    def compute_singlev_features(self, img_features: torch.Tensor,
                                 singlev_scaled_to_orig_xf: torch.Tensor
                                 ) -> torch.Tensor:
        return model_utils.apply_ftl_to_feature_maps(singlev_scaled_to_orig_xf,
                                                     img_features,
                                                     self._ftl_ratio)

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


class FeatureExtractor_New(nn.Module):
    """While this is a nn.Module, it doesn't provide any forward function.

    Instead it's a bundle of utility functions that produce features for
    different stages.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        model_opts: ModelOpts,
    ) -> None:
        super(FeatureExtractor_New, self).__init__()
        nc_img_features = model_opts.nImageFeatureChannels
        self._input_img_sizes = input_size
        self._ftl_ratio = float(model_opts.spatialFTLRatio)
        self._use_unscaled_as_canonical = model_opts.useUnscaledAsCanonical
        # Create the image backbone
        self._image_backbone, backbone_outshape = model_utils.create_backbone(
            model_opts.network,
            input_size,
            nc_img_features,
        )
        self._backbone_out_feature_sizes: List[int] = backbone_outshape[-2:]

    @property
    def input_image_sizes(self) -> Tuple[int, int]:
        return self._input_img_sizes

    @property
    def output_feature_sizes(self) -> Tuple[int, int]:
        """
        Return: feature map resolution from feature extractor
            [feature_size[0], feature_size[1]]
        """
        return (*self._backbone_out_feature_sizes, )

    def compute_singlev_features(self, img_features: torch.Tensor,
                                 singlev_scaled_to_orig_xf: torch.Tensor
                                 ) -> torch.Tensor:
        return model_utils.apply_ftl_to_feature_maps(singlev_scaled_to_orig_xf,
                                                     img_features,
                                                     self._ftl_ratio)


class FeatureExtractor_CCF(nn.Module):
    """While this is a nn.Module, it doesn't provide any forward function.

    Instead it's a bundle of utility functions that produce features for
    different stages.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        model_opts: ModelOpts,
    ) -> None:
        super(FeatureExtractor_CCF, self).__init__()
        nc_img_features = model_opts.nImageFeatureChannels
        num_views = 2

        self._input_img_sizes = input_size
        self._ftl_ratio = float(model_opts.spatialFTLRatio)
        self._use_unscaled_as_canonical = model_opts.useUnscaledAsCanonical

        # Create the image backbone
        self._image_backbone, backbone_outshape = model_utils.create_backbone(
            model_opts.network,
            input_size,
            nc_img_features,
        )

        self._backbone_out_feature_sizes: List[int] = backbone_outshape[-2:]
        self._multi_view_fusion = CrossCueFusion(cv_hypo_num=72, mid_dim=72)
        self._compress = nn.Conv2d(144, 72, kernel_size=1, padding=0)

    @property
    def input_image_sizes(self) -> Tuple[int, int]:
        return self._input_img_sizes

    @property
    def output_feature_sizes(self) -> Tuple[int, int]:
        """
        Return: feature map resolution from feature extractor
            [feature_size[0], feature_size[1]]
        """
        return (*self._backbone_out_feature_sizes, )

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

    def compute_singlev_features(self, img_features: torch.Tensor,
                                 singlev_scaled_to_orig_xf: torch.Tensor
                                 ) -> torch.Tensor:
        return model_utils.apply_ftl_to_feature_maps(singlev_scaled_to_orig_xf,
                                                     img_features,
                                                     self._ftl_ratio)

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
        c = img_features.shape[2]
        h = img_features.shape[3]
        w = img_features.shape[4]
        assert img_features.shape[1] == 2, 'Only 2 views supported'

        (
            multiv_scaled_to_canonical_xf,
            multiv_canonical_to_cam0_xf,
        ) = self._compute_multiv_xfs(singlev_scaled_to_orig_xf, extrinsics_xf)
        # print(multiv_canonical_to_cam0_xf.shape)

        # Transform all the features to the canonical space
        multiv_canonical_features = model_utils.apply_ftl_to_feature_maps(
            multiv_scaled_to_canonical_xf.reshape(-1, 4, 4),
            torch.flatten(img_features, start_dim=0, end_dim=1),
            self._ftl_ratio,
        ).reshape(batch_size * n_views, c, h, w)

        l_canonical_features = multiv_canonical_features[0::2].clone()
        r_canonical_features = multiv_canonical_features[1::2].clone()

        # print(torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2).shape)
        # Flatten the view dimension with the channel dimension and apply multi-view fusion
        multiv_fused_img_features = self._multi_view_fusion(
            l_canonical_features, r_canonical_features)
        multiv_fused_img_features = self._compress(multiv_fused_img_features)
        # print(multiv_fused_img_features)
        # Apply ftl so that the maps are transformed from
        # the canonical space to cam0 space.
        cam0_maps = model_utils.apply_ftl_to_feature_maps(
            multiv_canonical_to_cam0_xf, multiv_fused_img_features,
            self._ftl_ratio)

        return cam0_maps


class FeatureExtractor_TM(nn.Module):
    """While this is a nn.Module, it doesn't provide any forward function.

    Instead it's a bundle of utility functions that produce features for
    different stages.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        model_opts: ModelOpts,
    ) -> None:
        super(FeatureExtractor_TM, self).__init__()
        nc_img_features = model_opts.nImageFeatureChannels
        num_views = 2

        self._input_img_sizes = input_size
        self._ftl_ratio = float(model_opts.spatialFTLRatio)
        self._use_unscaled_as_canonical = model_opts.useUnscaledAsCanonical

        # Create the image backbone
        self._image_backbone, backbone_outshape = model_utils.create_backbone(
            model_opts.network,
            input_size,
            nc_img_features,
        )

        self._backbone_out_feature_sizes: List[int] = backbone_outshape[-2:]
        self._multi_view_fusion = model_utils.create_multi_view_fusion_layers(
            nc_img_features * num_views,
            nc_img_features,
            model_opts.nMultiViewFusionBlocks,
        )
        self._multi_view_fusion_t = model_utils.create_multi_view_fusion_layers(
            nc_img_features * num_views,
            nc_img_features,
            model_opts.nMultiViewFusionBlocks,
        )

        # self._mem_features_l = torch.empty(0)
        # self._mem_features_r = torch.empty(0)

        # self._prev_extrinsics_l = torch.empty(0)
        # self._prev_extrinsics_r = torch.empty(0)

    @property
    def input_image_sizes(self) -> Tuple[int, int]:
        return self._input_img_sizes

    @property
    def output_feature_sizes(self) -> Tuple[int, int]:
        """
        Return: feature map resolution from feature extractor
            [feature_size[0], feature_size[1]]
        """
        return (*self._backbone_out_feature_sizes, )

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

    def compute_singlev_features(self, img_features: torch.Tensor,
                                 singlev_scaled_to_orig_xf: torch.Tensor
                                 ) -> torch.Tensor:
        return model_utils.apply_ftl_to_feature_maps(singlev_scaled_to_orig_xf,
                                                     img_features,
                                                     self._ftl_ratio)

    def compute_multiv_features_t(
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
        # required_memory_len = int(torch.max(memory_idx)) + 1
        # # if len(self._mem_features_l) < required_memory_len:

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
        multiv_fused_img_features = self._multi_view_fusion_t(
            torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2))
        # print(multiv_fused_img_features)
        # Apply ftl so that the maps are transformed from
        # the canonical space to cam0 space.
        cam0_maps = model_utils.apply_ftl_to_feature_maps(
            multiv_canonical_to_cam0_xf, multiv_fused_img_features,
            self._ftl_ratio)

        return cam0_maps

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


class FeatureExtractor_TM_Res(nn.Module):
    """While this is a nn.Module, it doesn't provide any forward function.

    Instead it's a bundle of utility functions that produce features for
    different stages.
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 model_opts: ModelOpts,
                 use_bn=True) -> None:
        super(FeatureExtractor_TM_Res, self).__init__()
        nc_img_features = model_opts.nImageFeatureChannels
        num_views = 2

        self._input_img_sizes = input_size
        self._ftl_ratio = float(model_opts.spatialFTLRatio)
        self._use_unscaled_as_canonical = model_opts.useUnscaledAsCanonical

        # Create the image backbone
        self._image_backbone, backbone_outshape = model_utils.create_backbone(
            model_opts.network,
            input_size,
            nc_img_features,
        )

        self._backbone_out_feature_sizes: List[int] = backbone_outshape[-2:]
        self._multi_view_fusion = model_utils.create_multi_view_fusion_layers(
            nc_img_features * num_views,
            nc_img_features,
            model_opts.nMultiViewFusionBlocks,
        )
        if use_bn:
            self._multi_view_fusion_t = model_utils.create_multi_view_fusion_layers(
                nc_img_features * num_views,
                nc_img_features,
                model_opts.nMultiViewFusionBlocks,
            )
        else:
            print('no bn')
            self._multi_view_fusion_t = model_utils.create_multi_view_fusion_layers_t(
                nc_img_features * num_views,
                nc_img_features,
                model_opts.nMultiViewFusionBlocks,
            )
        self.gamma = nn.Parameter(torch.zeros(1))

    @property
    def input_image_sizes(self) -> Tuple[int, int]:
        return self._input_img_sizes

    @property
    def output_feature_sizes(self) -> Tuple[int, int]:
        """
        Return: feature map resolution from feature extractor
            [feature_size[0], feature_size[1]]
        """
        return (*self._backbone_out_feature_sizes, )

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

    def compute_singlev_features(self, img_features: torch.Tensor,
                                 singlev_scaled_to_orig_xf: torch.Tensor
                                 ) -> torch.Tensor:
        return model_utils.apply_ftl_to_feature_maps(singlev_scaled_to_orig_xf,
                                                     img_features,
                                                     self._ftl_ratio)

    def compute_multiv_features_t(
        self,
        feats_0: torch.Tensor,
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
        # required_memory_len = int(torch.max(memory_idx)) + 1
        # # if len(self._mem_features_l) < required_memory_len:

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

        # feats_t =  multiv_canonical_features[:,0]
        # feats_t_1 =  multiv_canonical_features[:,1]
        # feats = feats_t-feats_t_1
        # print(torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2).shape)
        # Flatten the view dimension with the channel dimension and apply multi-view fusion
        multiv_fused_img_features = self._multi_view_fusion_t(
            torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2))
        # print(multiv_fused_img_features)
        # Apply ftl so that the maps are transformed from
        # the canonical space to cam0 space.
        cam0_maps = model_utils.apply_ftl_to_feature_maps(
            multiv_canonical_to_cam0_xf, multiv_fused_img_features,
            self._ftl_ratio)
        feats = feats_0 + self.gamma * cam0_maps
        return feats

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


class FeatureExtractor_TM_Res_Sub(nn.Module):
    """While this is a nn.Module, it doesn't provide any forward function.

    Instead it's a bundle of utility functions that produce features for
    different stages.
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 model_opts: ModelOpts,
                 use_bn=True) -> None:
        super(FeatureExtractor_TM_Res_Sub, self).__init__()
        nc_img_features = model_opts.nImageFeatureChannels
        num_views = 2

        self._input_img_sizes = input_size
        self._ftl_ratio = float(model_opts.spatialFTLRatio)
        self._use_unscaled_as_canonical = model_opts.useUnscaledAsCanonical

        # Create the image backbone
        self._image_backbone, backbone_outshape = model_utils.create_backbone(
            model_opts.network,
            input_size,
            nc_img_features,
        )

        self._backbone_out_feature_sizes: List[int] = backbone_outshape[-2:]
        self._multi_view_fusion = model_utils.create_multi_view_fusion_layers(
            nc_img_features * num_views,
            nc_img_features,
            model_opts.nMultiViewFusionBlocks,
        )
        if use_bn:
            self._multi_view_fusion_t = model_utils.create_multi_view_fusion_layers(
                nc_img_features,
                nc_img_features,
                model_opts.nMultiViewFusionBlocks,
            )
        else:
            print('no bn')
            self._multi_view_fusion_t = model_utils.create_multi_view_fusion_layers_t(
                nc_img_features,
                nc_img_features,
                model_opts.nMultiViewFusionBlocks,
            )
        self.gamma = nn.Parameter(torch.zeros(1))

    @property
    def input_image_sizes(self) -> Tuple[int, int]:
        return self._input_img_sizes

    @property
    def output_feature_sizes(self) -> Tuple[int, int]:
        """
        Return: feature map resolution from feature extractor
            [feature_size[0], feature_size[1]]
        """
        return (*self._backbone_out_feature_sizes, )

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

    def compute_singlev_features(self, img_features: torch.Tensor,
                                 singlev_scaled_to_orig_xf: torch.Tensor
                                 ) -> torch.Tensor:
        return model_utils.apply_ftl_to_feature_maps(singlev_scaled_to_orig_xf,
                                                     img_features,
                                                     self._ftl_ratio)

    def compute_multiv_features_t_new(
        self,
        feats_0: torch.Tensor,
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
        # required_memory_len = int(torch.max(memory_idx)) + 1
        # # if len(self._mem_features_l) < required_memory_len:

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

        feats_t = multiv_canonical_features[:, 0]
        feats_t_1 = multiv_canonical_features[:, 1]
        feats = feats_t - feats_t_1
        # print(torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2).shape)
        # Flatten the view dimension with the channel dimension and apply multi-view fusion
        multiv_fused_img_features = self._multi_view_fusion_t(feats)
        # print(multiv_fused_img_features)
        # Apply ftl so that the maps are transformed from
        # the canonical space to cam0 space.
        cam0_maps = model_utils.apply_ftl_to_feature_maps(
            multiv_canonical_to_cam0_xf, multiv_fused_img_features,
            self._ftl_ratio)
        feats = feats_0 + self.gamma * cam0_maps
        return feats

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


class FeatureExtractor_TM_Res_New(nn.Module):
    """While this is a nn.Module, it doesn't provide any forward function.

    Instead it's a bundle of utility functions that produce features for
    different stages.
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 model_opts: ModelOpts,
                 use_bn=True) -> None:
        super(FeatureExtractor_TM_Res_New, self).__init__()
        nc_img_features = model_opts.nImageFeatureChannels
        num_views = 2

        self._input_img_sizes = input_size
        self._ftl_ratio = float(model_opts.spatialFTLRatio)
        self._use_unscaled_as_canonical = model_opts.useUnscaledAsCanonical

        # Create the image backbone
        self._image_backbone, backbone_outshape = model_utils.create_backbone(
            model_opts.network,
            input_size,
            nc_img_features,
        )

        self._backbone_out_feature_sizes: List[int] = backbone_outshape[-2:]
        self._multi_view_fusion = model_utils.create_multi_view_fusion_layers(
            nc_img_features * num_views,
            nc_img_features,
            model_opts.nMultiViewFusionBlocks,
        )
        if use_bn:
            self._multi_view_fusion_t = model_utils.create_multi_view_fusion_layers(
                nc_img_features * num_views,
                nc_img_features,
                model_opts.nMultiViewFusionBlocks,
            )
        else:
            print('no bn')
            self._multi_view_fusion_t = model_utils.create_multi_view_fusion_layers_t(
                nc_img_features * num_views,
                nc_img_features,
                model_opts.nMultiViewFusionBlocks,
            )
        self.gamma = nn.Parameter(torch.zeros(1))

    @property
    def input_image_sizes(self) -> Tuple[int, int]:
        return self._input_img_sizes

    @property
    def output_feature_sizes(self) -> Tuple[int, int]:
        """
        Return: feature map resolution from feature extractor
            [feature_size[0], feature_size[1]]
        """
        return (*self._backbone_out_feature_sizes, )

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

    def compute_singlev_features(self, img_features: torch.Tensor,
                                 singlev_scaled_to_orig_xf: torch.Tensor
                                 ) -> torch.Tensor:
        return model_utils.apply_ftl_to_feature_maps(singlev_scaled_to_orig_xf,
                                                     img_features,
                                                     self._ftl_ratio)

    def compute_multiv_features_t(
        self,
        feats_0: torch.Tensor,
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
        # required_memory_len = int(torch.max(memory_idx)) + 1
        # # if len(self._mem_features_l) < required_memory_len:

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
        multiv_fused_img_features = self._multi_view_fusion_t(
            torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2))
        # print(multiv_fused_img_features)
        # Apply ftl so that the maps are transformed from
        # the canonical space to cam0 space.
        cam0_maps = model_utils.apply_ftl_to_feature_maps(
            multiv_canonical_to_cam0_xf, multiv_fused_img_features,
            self._ftl_ratio)
        feats = feats_0 + self.gamma * cam0_maps
        return feats

    def compute_multiv_features_t(
        self,
        feats_0: torch.Tensor,
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
        # required_memory_len = int(torch.max(memory_idx)) + 1
        # # if len(self._mem_features_l) < required_memory_len:

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
        multiv_fused_img_features = self._multi_view_fusion_t(
            torch.flatten(multiv_canonical_features, start_dim=1, end_dim=2))
        # print(multiv_fused_img_features)
        # Apply ftl so that the maps are transformed from
        # the canonical space to cam0 space.
        cam0_maps = model_utils.apply_ftl_to_feature_maps(
            multiv_canonical_to_cam0_xf, multiv_fused_img_features,
            self._ftl_ratio)
        feats = feats_0 + self.gamma * cam0_maps
        return feats

    def compute_multiv_features_t_new(
        self,
        feats_0: torch.Tensor,
        feats_mem: torch.Tensor,
    ) -> torch.Tensor:

        cam0_maps = self._multi_view_fusion_t(feats_mem)

        feats = feats_0 + self.gamma * cam0_maps

        return feats

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
