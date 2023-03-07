# Copyright (c) OpenMMLab. All rights reserved.
from .ae_loss import AssociativeEmbeddingLoss
from .classification_loss import BCELoss, JSDiscretLoss, KLDiscretLoss
from .loss_wrappers import CombinedLoss, MultipleLossWrapper
from .heatmap_loss import (AdaptiveWingLoss, KeypointMSELoss,
                           KeypointOHKMMSELoss, JointsL2Loss, FocalHeatmapLoss,
                           CombinedTargetMSELoss)
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, RLELoss,
                              SemiSupervisionLoss, SmoothL1Loss,
                              SoftWeightSmoothL1Loss, SoftWingLoss, WingLoss)
from .pruning_loss import FilterL2Loss

__all__ = [
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'KLDiscretLoss', 'MultipleLossWrapper', 'JSDiscretLoss', 'CombinedLoss',
    'AssociativeEmbeddingLoss', 'SoftWeightSmoothL1Loss', 'JointsL2Loss',
    'FilterL2Loss', 'FocalHeatmapLoss', 'CombinedTargetMSELoss'
]
