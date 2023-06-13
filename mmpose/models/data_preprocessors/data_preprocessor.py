# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import ImgDataPreprocessor

from mmpose.registry import MODELS
from mmpose.utils.data import format_data


@MODELS.register_module()
class PoseDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for pose estimation tasks."""

    @format_data
    def forward(self, data: dict, training: bool = False):
        return super().forward(data, training)
