# Copyright (c) OpenMMLab. All rights reserved.
from .attr_mlp_head import AttrMlpHead
from .dsnt_attr_head import DSNTAttrHead
from .dsnt_head import DSNTHead
from .integral_regression_head import IntegralRegressionHead
from .lift_head import LiftHead
from .lift_head_rot_standard import LiftNimbleHeadStandard
from .lift_head_rot_standard_e2e import LiftNimbleHeadStandardE2e
from .lift_head_standard import LiftHeadStandard
from .lift_head_standard_ori_e2e import LiftHeadStandardOriE2e
from .regression_head import RegressionHead
from .rle_head import RLEHead
from .rtmcc_ipr_head import RTMCCIPRHead
from .rtmcc_ipr_head_3d import RTMCCIPRHead3D
from .temporal_lift_head import TemporalLiftHead
from .temporal_lift_head_rot_standard import TemporalLiftNimbleHeadStandard
from .temporal_lift_head_rot_standard_e2e import \
    TemporalLiftNimbleHeadStandardE2e
from .temporal_lift_head_rot_standard_predict import \
    TemporalLiftNimbleHeadStandardPredict
from .temporal_lift_head_standard import TemporalLiftHeadStandard
from .temporal_lift_head_standard_ori import TemporalLiftHeadStandardOri
from .temporal_regression_head import TemporalRegressionHead
from .trajectory_regression_head import TrajectoryRegressionHead
from .rtmcc_ipr_head_nimble import RTMCCIPRHeadNimble
from .ume_head import UmeHead

__all__ = [
    'RegressionHead',
    'IntegralRegressionHead',
    'DSNTHead',
    'RLEHead',
    'TemporalRegressionHead',
    'TrajectoryRegressionHead',
    'DSNTAttrHead',
    'RTMCCIPRHead',
    'AttrMlpHead',
    'LiftHead',
    'RTMCCIPRHead3D',
    'LiftHeadStandard',
    'LiftNimbleHead',
    'TemporalLiftHead',
    'TemporalLiftNimbleHeadStandard',
    'TemporalLiftNimbleHeadStandardPredict',
    'TemporalLiftHeadStandard',
    'LiftNimbleHeadStandard',
    'RTMCCIPRHeadNimble',
    # 'PCT_Head', 'PCT_Tokenizer',
    # 'LiftClassifierHead', 'SwinV2TransformerRPE2FC'
]
