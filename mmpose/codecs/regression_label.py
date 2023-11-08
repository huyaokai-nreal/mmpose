# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils.post_processing import get_simcc_1d_maximum


@KEYPOINT_CODECS.register_module()
class RegressionLabel(BaseKeypointCodec):
    r"""Generate keypoint coordinates.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]

    Encoded:

        - keypoint_labels (np.ndarray): The normalized regression labels in
            shape (N, K, D) where D is 2 for 2d coordinates
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Input image size in [w, h]

    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 with_depth: bool = False,
                 depth_bound: float = 0.4,
                 sigma=6,
                 depth_channel=256,
                 depth_type: str = 'direct') -> None:
        super().__init__()
        self.with_depth = with_depth
        self.depth_bound = depth_bound
        self.input_size = input_size
        self.depth_type = depth_type
        self.depth_channel = depth_channel
        self.sigma = sigma
        if self.with_depth:
            assert self.depth_bound > 0, \
                f'depth bound should be positive vs {self.depth_bound}'
            assert len(self.input_size) == 3, \
                f'input size should be 3 param while having \
                    {len(self.input_size)}'

    def __encode_depth_to_heatmap(
        self,
        depth,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encoding keypoints into SimCC labels with Gaussian Label Smoothing
        strategy."""
        D = self.depth_channel
        N, K = depth.shape
        target_z = np.zeros((N, K, D), dtype=np.float32)
        # 3-sigma rule
        radius = self.sigma * 3
        z = np.arange(0, D, 1, dtype=np.float32)
        keypoint_weights = keypoints_visible.copy()
        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if np.abs(keypoints_visible[n, k]) < 0.5:
                continue
            mu = depth[n, k]
            # check that the gaussian has in-bounds part
            near = mu - radius
            far = mu + radius + 1
            if near >= D or far < 0:
                keypoint_weights[n, k] = 0
                continue
            target_z[n, k] = np.exp(-((z - mu)**2) / (2 * self.sigma**2))
        return target_z, keypoint_weights

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encoding keypoints from input image space to normalized space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - keypoint_labels (np.ndarray): The normalized regression labels in
                shape (N, K, D) where D is 2 for 2d coordinates
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        w, h = self.input_size[:2]
        valid = ((keypoints[..., :2] >= 0) &
                 (keypoints[..., :2] <= [w - 1, h - 1])).all(axis=-1)
        keypoint_weights = np.where(valid, 1., 0.).astype(np.float32)
        keypoint_weights = keypoint_weights * keypoints_visible
        if not self.with_depth:
            keypoint_labels = (keypoints / np.array([w, h])).astype(np.float32)
        else:
            keypoint_labels = (keypoints /
                               np.array([w, h, 1])).astype(np.float32)
            keypoint_labels[..., 2] = (
                keypoints[..., 2] / self.depth_bound + 0.5)
        if self.depth_type == 'direct':
            encoded = dict(
                keypoint_labels=keypoint_labels,
                keypoint_weights=keypoint_weights)
        if self.depth_type == 'heatmap':
            keypoints_split = np.around(keypoint_labels[..., 2] *
                                        self.depth_channel)
            keypoint_z_labels, _ = self.__encode_depth_to_heatmap(
                keypoints_split, keypoints_visible)
            encoded = dict(
                keypoint_labels=keypoint_labels[..., :2],
                keypoint_z_labels=keypoint_z_labels,
                keypoint_weights=keypoint_weights)
        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (N, K, D)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """

        N, K, _ = encoded.shape
        if encoded.shape[-1] in [2, 3]:
            normalized_coords = encoded.copy()
            scores = np.ones((N, K), dtype=np.float32)
        elif encoded.shape[-1] in [4, 6]:
            # split coords and sigma if outputs contain output_sigma
            key_dim = encoded.shape[-1] // 2
            normalized_coords = encoded[..., :key_dim].copy()
            output_sigma = encoded[..., key_dim:key_dim * 2].copy()
            scores = (1 - output_sigma).mean(axis=-1)
        elif encoded.shape[-1] == 258:
            normalized_coords = encoded[..., :2].copy()
            depth_heatmap = encoded[..., 2:].copy()
            scores = np.ones((N, K), dtype=np.float32)
        else:
            raise ValueError(
                'Keypoint dimension should be 2 or 4 (with sigma), '
                f'but got {encoded.shape[-1]}')

        w, h = self.input_size[:2]
        if not self.with_depth:
            keypoints = normalized_coords[:, :, :2] * np.array([w, h])
        else:
            if self.depth_type == 'heatmap':
                keypoints = normalized_coords * np.array([w, h])
                depth = get_simcc_1d_maximum(
                    depth_heatmap) / self.depth_channel
                keypoints = np.concatenate(
                    [keypoints[..., :2],
                     depth.reshape((N, K, 1))], axis=-1)
            else:
                keypoints = normalized_coords * np.array([w, h, 1])
            keypoints[..., 2] = (keypoints[..., 2] - 0.5) * self.depth_bound
        return keypoints, scores
