# Copyright (c) OpenMMLab. All rights reserved.
from .associative_embedding import AssociativeEmbedding
from .decoupled_heatmap import DecoupledHeatmap
from .image_pose_lifting import ImagePoseLifting
from .integral_regression_label import IntegralRegressionLabel
from .megvii_heatmap import MegviiHeatmap
from .msra_heatmap import MSRAHeatmap
from .nreal_heatmap import NrealHeatmap
from .regression_label import RegressionLabel
from .simcc_label import SimCCLabel
from .simcc_label_3d import SimCCLabel3D
from .spr import SPR
from .udp_heatmap import UDPHeatmap
from .video_pose_lifting import VideoPoseLifting

__all__ = [
    'MSRAHeatmap', 'MegviiHeatmap', 'UDPHeatmap', 'RegressionLabel',
    'SimCCLabel', 'IntegralRegressionLabel', 'AssociativeEmbedding', 'SPR',
    'DecoupledHeatmap', 'NrealHeatmap', 'SimCCLabel3D', 'VideoPoseLifting',
    'ImagePoseLifting'
]
