# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import Optional, Tuple

import cv2
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.structures.bbox import bbox_cs2xyxy
from mmpose.utils.data import format_data
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator


@MODELS.register_module()
class TopdownPoseEstimator(BasePoseEstimator):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone,
            neck,
            head,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg=init_cfg)
        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

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

    def _forward(self, inputs: Tensor):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """

        x = self.extract_feat(inputs)
        if self.with_head:
            x = self.head.forward(x)
            if isinstance(x, list):
                x = x[-1]
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
        feats = self.extract_feat(inputs)

        losses = dict()
        if self.with_head:
            losses.update(
                self.head.loss(feats, data_samples, train_cfg=self.train_cfg))
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
        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)
        # import ipdb;ipdb.set_trace()
        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)
        
        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            bbox_centers = gt_instances.bbox_centers
            bbox_scales = gt_instances.bbox_scales
            warp_mat = data_sample.metainfo['warp_mat']
            inv_warp_mat = cv2.invertAffineTransform(warp_mat)
            pred_instances.keypoints[..., :2] = cv2.transform(
                pred_instances.keypoints[..., :2], inv_warp_mat)

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            # pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bboxes = bbox_cs2xyxy(bbox_centers, bbox_scales)
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples

import torch
import numpy as np
from mmengine.structures import InstanceData
from mmpose.models.utils import check_and_update_config
@MODELS.register_module()
class TopdownPoseLiftEstimatorNano(BasePoseEstimator):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 kpt3d_lift: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 kpt2d_with_depth: bool = False,
                 kpt3d_lift_model_path: str = '',
                 distill_model_path: str = '',
                 with_hand_scale:bool = False):
        super().__init__(
            backbone,
            neck,
            head,
            train_cfg,
            test_cfg,
            data_preprocessor,
            init_cfg=init_cfg)
        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        self.kpt2d_with_depth = kpt2d_with_depth
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.with_hand_scale = with_hand_scale
        neck, head = check_and_update_config(neck, head)
        neck, kpt3d_lift = check_and_update_config(neck, kpt3d_lift)
        self.distill_model_path= distill_model_path

        if (head is not None) and (kpt3d_lift is not None):
            self.head = MODELS.build(head)  # adapt 2d kpts model
            self.kpt3d_lift = MODELS.build(kpt3d_lift)
            pretrained_dict = torch.load(kpt3d_lift_model_path)
            liftnet_dict = {k.replace('kpt3d_lift.', ''): v for k, v in pretrained_dict['state_dict'].items() if k.startswith("kpt3d_lift.")}
            self.kpt3d_lift.load_state_dict(liftnet_dict) 

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

    def _forward(self, inputs: Tensor):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """

        x = self.extract_feat(inputs)
        if self.with_head:
            x = self.head.forward(x)
            if isinstance(x, list):
                x = x[-1]
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
        with torch.no_grad():
            feats_pyramid = self.extract_feat(inputs)
            outputs = self.head.forward(feats_pyramid)
            xy_sigma = outputs[:2]  # 2d
            if self.kpt2d_with_depth:
                depth = outputs[2]
                xy_sigma = torch.cat([xy_sigma, depth], dim=-1)
        losses = self.kpt3d_lift.loss(xy_sigma, data_samples)
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
        
        if self.distill_model_path:
            distill_model = torch.load(self.distill_model_path)
            state_dict = {k.replace('student.', ''): v for k, v in distill_model['state_dict'].items() if not k.startswith('teacher')}
            backbone_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone')}
            head_state_dict = {k.replace('head.', ''): v for k, v in state_dict.items() if k.startswith('head')}
            self.backbone.load_state_dict(backbone_state_dict)
            self.head.load_state_dict(head_state_dict)
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)
        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)
        
        xy_sigma = []
        for i in range(len(preds)):
            xy_sigma.append(preds[i].keypoints)
        xy_sigma = np.concatenate(xy_sigma,axis=0)
        pred, pred_bino_kp2d = self.kpt3d_lift.predict(
            xy_sigma, data_samples, test_cfg=self.test_cfg)
        
        batch_pred_instances = []
        for b in range(pred.shape[0]):
            keypoints = pred_bino_kp2d[b:b + 1, 0, ...]  # 左目信息
            batch_pred_instances.append(
                InstanceData(
                    keypoints3d=pred[b:b + 1, ...],
                    keypoints3d_scores=torch.ones((1, 21)),
                    keypoints=keypoints,
                    keypoint_scores=torch.ones((1, 21)),
                ))

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

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):
            if data_sample.meta['flipped']:
                data_sample.gt_instances.keypoints3d[...,0] *= -1
            pred_instances.keypoints3d = pred_instances.keypoints3d.cpu(
            ).numpy()
            pred_instances.keypoints3d_scores = np.ones(
                (1, pred_instances.keypoints3d.shape[1]))
            pred_instances.keypoints = pred_instances.keypoints.cpu().numpy()
            pred_instances.keypoints = np.concatenate((pred_instances.keypoints, pred_instances.keypoints3d[...,2:]), axis=-1)
            data_sample.gt_instances.keypoints = np.concatenate((data_sample.gt_instances.keypoints[...,:2], data_sample.gt_instances.keypoints3d[...,2:]), axis=-1)
            pred_instances.keypoint_scores = np.ones(
                (1, pred_instances.keypoints.shape[1]))

            data_sample.pred_instances = pred_instances
        return batch_data_samples