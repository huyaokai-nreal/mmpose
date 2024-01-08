# Copyright (c) OpenMMLab. All rights reserved.
from .ae_loss import AssociativeEmbeddingLoss
from .classification_loss import (BCELoss, FocalLoss, JSDiscretLoss,
                                  KLDiscretLoss)
from .fea_dis_loss import FeaLoss
from .heatmap_loss import (AdaptiveWingLoss, CombinedTargetMSELoss,
                           FocalHeatmapLoss, JointsL2Loss, KeypointMSELoss,
                           KeypointOHKMMSELoss)
from .logit_dis_loss import KDLoss, KDLoss3D
from .loss_wrappers import CombinedLoss, MultipleLossWrapper
from .pct_loss import Classifer_loss, JointS1Loss, Tokenizer_loss
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, PinchLoss,
                              RLELoss, SemiSupervisionLoss, SmoothL1Loss,
                              SoftWeightSmoothL1Loss, SoftWingLoss, WingLoss)
from .seq_smooth_loss import MPJPAELoss, MPJPALoss

__all__ = [
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'KLDiscretLoss', 'MultipleLossWrapper', 'JSDiscretLoss', 'CombinedLoss',
    'AssociativeEmbeddingLoss', 'SoftWeightSmoothL1Loss', 'JointsL2Loss',
    'FocalHeatmapLoss', 'CombinedTargetMSELoss', 'FocalLoss', 'PinchLoss',
    'AssociativeEmbeddingLoss', 'SoftWeightSmoothL1Loss', 'FeaLoss', 'KDLoss',
    'KDLoss3D', 'MPJPAELoss', 'MPJPALoss', 'Classifer_loss', 'JointS1Loss',
    'Tokenizer_loss'
]
