from typing import Tuple
import numpy as np
from mmpose.registry import KEYPOINT_CODECS
from .utils import gaussian_blur, get_heatmap_maximum
from .megvii_heatmap import MegviiHeatmap


def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
        dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
        dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
        dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
        dxy = 0.25 * (
            hm[py + 1][px + 1] - hm[py - 1][px + 1] - hm[py + 1][px - 1] +
            hm[py - 1][px - 1])
        dyy = 0.25 * (hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
        derivative = np.matrix([[dx], [dy]])
        hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


@KEYPOINT_CODECS.register_module()
class NrealHeatmap(MegviiHeatmap):

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.kernel_size = 9
        heatmaps = gaussian_blur(encoded.copy(), self.kernel_size)
        K = heatmaps.shape[0]

        keypoints, scores = get_heatmap_maximum(heatmaps)
        for k in range(K):
            heatmap = heatmaps[k]
            px = int(keypoints[k, 0])
            py = int(keypoints[k, 1])
            coord = [px, py]
            coord = taylor(heatmap, coord)
            px, py = coord[0], coord[1]
            keypoints[k] = np.array([px, py])
        scores = scores / 255.0 + 0.5
        keypoints = keypoints[None] * self.scale_factor + 2
        scores = scores[None]
        return keypoints, scores
