# Copyright (c) OpenMMLab. All rights reserved.
from .ae_loss import AssociativeEmbeddingLoss
from .classification_loss import BCELoss, JSDiscretLoss, KLDiscretLoss
from .loss_wrappers import CombinedLoss, MultipleLossWrapper
from .heatmap_loss import AdaptiveWingLoss
from .mse_loss import CombinedTargetMSELoss, JointsL2Loss, KeypointMSELoss, KeypointOHKMMSELoss  # noqa
from .multi_loss_factory import AELoss, HeatmapLoss, MultiLossFactory
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, RLELoss,
                              SemiSupervisionLoss, SmoothL1Loss,
                              SoftWeightSmoothL1Loss, SoftWingLoss, WingLoss)

__all__ = [
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'KLDiscretLoss', 'MultipleLossWrapper', 'JSDiscretLoss', 'CombinedLoss',
    'AssociativeEmbeddingLoss', 'SoftWeightSmoothL1Loss', 'JointsL2Loss'
]
