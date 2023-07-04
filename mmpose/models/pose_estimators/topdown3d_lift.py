# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import Optional, Tuple

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.models.utils import check_and_update_config
from mmpose.registry import MODELS
from mmpose.utils.data import format_data
from mmpose.utils.typing import (ConfigType, ForwardResults, InstanceList,
                                 OptConfigType, OptMultiConfig, OptSampleList,
                                 PixelDataList, SampleList)


@MODELS.register_module()
class TopdownPoseLiftEstimator(BaseModel):
    """Base class for 3d hand kpts estimators."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 kpt3d_lift: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super().__init__(data_preprocessor, init_cfg=init_cfg)
        self.metainfo = self._load_metainfo(metainfo)
        self.backbone = MODELS.build(backbone)

        neck, head = check_and_update_config(neck, head)
        neck, kpt3d_lift = check_and_update_config(neck, kpt3d_lift)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if (head is not None) and (kpt3d_lift is not None):
            self.head = MODELS.build(head)  # adapt 2d kpts model
            self.kpt3d_lift = MODELS.build(kpt3d_lift)

        self.train_cfg = train_cfg if train_cfg else {}
        self.test_cfg = test_cfg if test_cfg else {}

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    @property
    def with_neck(self) -> bool:
        """bool: whether the pose estimator has a neck."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """bool: whether the pose estimator has a head"""
        return hasattr(self, 'head') and self.head is not None

    @staticmethod
    def _load_metainfo(metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """

        if metainfo is None:
            return None

        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[OptSampleList] = None,
                mode: str = 'tensor') -> ForwardResults:
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # use customed metainfo to override the default metainfo
            if self.metainfo is not None:
                for data_sample in data_samples:
                    data_sample.set_metainfo(self.metainfo)
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode.')

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward(self, inputs: Tensor, data_samples: SampleList):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """

        feats_pyramid = self.extract_feat(inputs)
        xy_sigma, heatmap = self.head.forward(feats_pyramid)
        xy_sigma = xy_sigma.detach()  # fix 2d model params

        ret = self.kpt3d_lift.forward(xy_sigma, data_samples)

        x = ret['hand3d_pred']

        return x

    @format_data
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        feats_pyramid = self.extract_feat(inputs)
        xy_sigma, heatmap = self.head.forward(feats_pyramid)
        xy_sigma = xy_sigma.detach()  # fix 2d model params

        losses = dict()

        losses.update(self.kpt3d_lift.loss(xy_sigma, data_samples))
        return losses

    @format_data
    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)

        xy_sigma, heatmap = self.head.forward(feats)
        xy_sigma = xy_sigma.detach()  # fix 2d model params

        pred = self.kpt3d_lift.predict(
            xy_sigma, data_samples, test_cfg=self.test_cfg)

        batch_pred_instances = []

        for b in range(pred.shape[0]):
            batch_pred_instances.append(
                InstanceData(
                    keypoints=pred[b], keypoint_scores=torch.ones((21))))

        results = self.add_pred_to_datasample(batch_pred_instances, None,
                                              data_samples)

        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:

        batch_data_samples = batch_data_samples[::2]  # 3d gt信息保存在左目

        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        # output_keypoint_indices = self.test_cfg.get('output_keypoint_indices', None) # noqa

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            pred_instances.keypoints3d = pred_instances.keypoints.cpu().numpy()
            pred_instances.keypoint_scores = np.ones(
                pred_instances.keypoints.shape[0])

            data_sample.pred_instances = pred_instances

            data_sample.gt_instances.keypoints3d = np.array(
                [data_sample.meta['kp3d_spline']])
            data_sample.gt_instances.keypoints_visible = np.ones(
                (1, len(data_sample.meta['kp3d_spline'])))
            # data_sample.

            # data_sample.gt_instances.keypoints3d = data_sample.meta['kp3d_spline']  # noqa

        return batch_data_samples

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for k in keys:
            if 'keypoint_head' in k:
                v = state_dict.pop(k)
                k = k.replace('keypoint_head', 'head')
                state_dict[k] = v
