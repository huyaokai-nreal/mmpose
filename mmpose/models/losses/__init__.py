# Copyright (c) OpenMMLab. All rights reserved.
from .ae_loss import AssociativeEmbeddingLoss
from .classification_loss import (BCELoss, FocalLoss, JSDiscretLoss,
                                  KLDiscretLoss, KLDiscretLoss3D)
from .heatmap_loss import (AdaptiveWingLoss, CombinedTargetMSELoss,
                           FocalHeatmapLoss, JointsL2Loss, KeypointMSELoss,KeypointOHKMMSELoss)
from .fea_dis_loss import FeaLoss
from .logit_dis_loss import KDLoss
from .loss_wrappers import CombinedLoss, MultipleLossWrapper
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, PinchLoss,
                              RLELoss, SemiSupervisionLoss, SmoothL1Loss,
                              SoftWeightSmoothL1Loss, SoftWingLoss, WingLoss)

__all__ = [
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'KLDiscretLoss', 'MultipleLossWrapper', 'JSDiscretLoss', 'CombinedLoss',
    'AssociativeEmbeddingLoss', 'SoftWeightSmoothL1Loss', 'JointsL2Loss',
    'FocalHeatmapLoss', 'CombinedTargetMSELoss', 'KLDiscretLoss3D',
    'FocalLoss', 'PinchLoss',
    'AssociativeEmbeddingLoss', 'SoftWeightSmoothL1Loss',
    'MPJPEVelocityJointLoss', 'FeaLoss', 'KDLoss'
]
