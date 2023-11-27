# Copyright (c) OpenMMLab. All rights reserved.
from .attr_mlp_head import AttrMlpHead
from .dsnt_attr_head import DSNTAttrHead
from .dsnt_head import DSNTHead
from .integral_regression_head import IntegralRegressionHead
from .lift_head import LiftHead
from .lift_head_rot import LiftNimbleHead
from .lift_head_standard import LiftHeadStandard
from .regression_head import RegressionHead
from .rle_head import RLEHead
from .rtmcc_ipr_head import RTMCCIPRHead
from .rtmcc_ipr_head_3d import RTMCCIPRHead3D
from .temporal_lift_head import TemporalLiftHead
from .temporal_lift_head_standard import TemporalLiftHeadStandard
from .temporal_regression_head import TemporalRegressionHead
from .trajectory_regression_head import TrajectoryRegressionHead

__all__ = [
    'RegressionHead', 'IntegralRegressionHead', 'DSNTHead', 'RLEHead',
    'TemporalRegressionHead', 'TrajectoryRegressionHead', 'DSNTAttrHead',
    'RTMCCIPRHead', 'AttrMlpHead', 'LiftHead', 'RTMCCIPRHead3D',
    'LiftHeadStandard', 'LiftNimbleHead', 'TemporalLiftHead',
    'TemporalLiftHeadStandard'
]
