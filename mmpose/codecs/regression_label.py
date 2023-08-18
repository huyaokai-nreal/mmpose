# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


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
                 depth_bound: float = 0.4) -> None:
        super().__init__()
        self.with_depth = with_depth
        self.depth_bound = depth_bound
        self.input_size = input_size
        if self.with_depth:
            assert self.depth_bound > 0, \
                f'depth bound should be positive vs {self.depth_bound}'
            assert len(self.input_size) == 3, \
                f'input size should be 3 param while having \
                    {len(self.input_size)}'

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
        encoded = dict(
            keypoint_labels=keypoint_labels, keypoint_weights=keypoint_weights)

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

        if encoded.shape[-1] in [2, 3]:
            N, K, _ = encoded.shape
            normalized_coords = encoded.copy()
            scores = np.ones((N, K), dtype=np.float32)
        elif encoded.shape[-1] in [4, 6]:
            # split coords and sigma if outputs contain output_sigma
            key_dim = encoded.shape[-1] // 2
            normalized_coords = encoded[..., :key_dim].copy()
            output_sigma = encoded[..., key_dim:key_dim * 2].copy()
            scores = (1 - output_sigma).mean(axis=-1)
        else:
            raise ValueError(
                'Keypoint dimension should be 2 or 4 (with sigma), '
                f'but got {encoded.shape[-1]}')

        w, h = self.input_size[:2]
        if not self.with_depth:
            keypoints = normalized_coords * np.array([w, h])
        else:
            keypoints = normalized_coords * np.array([w, h, 1])
            keypoints[..., 2] = (keypoints[..., 2] - 0.5) * self.depth_bound
        return keypoints, scores
