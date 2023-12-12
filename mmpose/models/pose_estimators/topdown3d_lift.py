# Copyright (c) OpenMMLab. All rights reserved.
from itertools import chain, zip_longest
from typing import Optional, Tuple

import numpy as np
import torch
from mmengine import MMLogger
from mmengine.model import BaseModel
from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.models.heads.nimble.nimble_utils import (adjust_predicted_angles,
                                                     batch_rodrigues,
                                                     matrix_to_euler_angles)
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
                 kpt2d_with_depth: bool = False,
                 metainfo: Optional[dict] = None,
                 nano_2d=False):
        super().__init__(data_preprocessor, init_cfg=init_cfg)
        self.metainfo = self._load_metainfo(metainfo)
        self.backbone = MODELS.build(backbone)
        self.kpt2d_with_depth = kpt2d_with_depth
        neck, head = check_and_update_config(neck, head)
        neck, kpt3d_lift = check_and_update_config(neck, kpt3d_lift)

        self.nano_2d = nano_2d
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
            logger: MMLogger = MMLogger.get_current_instance()
            logger.warning(
                'topdown3d lift only support output 2d kpt for tensor mode')
            return self._forward(inputs)
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

    def _forward(self, inputs: Tensor):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        # TODO: only support export 2d result
        x = self.extract_feat(inputs)
        if self.with_head:
            x = self.head.forward(x)
            if isinstance(x, list):
                x = x[-1]

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
            xy_sigma, heatmap = outputs[:2]
            if self.kpt2d_with_depth:
                depth = outputs[2]
                depth = (depth - 0.5) * 0.4
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
        assert self.with_head, (
            'The model must have head to perform prediction.')
        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)
            if self.nano_2d:
                preds = self.head.predict(
                    feats, data_samples, test_cfg=self.test_cfg)
                xy_sigma = []
                for i in range(len(preds)):
                    preds[i].keypoints[..., :2] /= data_samples[0].input_size
                    xy_sigma.append(
                        torch.tensor(preds[i].keypoints,
                                     dtype=torch.float32).cuda())
                xy_sigma = torch.cat(xy_sigma, dim=0)
            else:
                outputs = self.head.forward(feats)
                xy_sigma, heatmap = outputs[:2]
                if self.kpt2d_with_depth:
                    depth = outputs[2]
                    depth = (depth - 0.5) * 0.4  # 0.4 is the depth bound
                    xy_sigma = torch.cat([xy_sigma, depth], dim=-1)

        batch_pred_instances = []
        pre_info = self.kpt3d_lift.predict(
            xy_sigma, data_samples, test_cfg=self.test_cfg)

        if len(pre_info) == 4:
            pred, pred_bino_kp2d, parent_matrix, child_vector = pre_info
            for b in range(pred.shape[0]):
                keypoints = pred_bino_kp2d[b:b + 1, 0, ...]  # gt为左目信息
                child_matrix = batch_rodrigues(child_vector[b, :, :]).reshape(
                    -1, 3, 3)
                pre_matrix = torch.cat(
                    (parent_matrix[b:b + 1, :, :], child_matrix), dim=0)
                pre_euler = matrix_to_euler_angles(pre_matrix)

                batch_pred_instances.append(
                    InstanceData(
                        keypoints3d=pred[b:b + 1, ...],
                        keypoints3d_scores=torch.ones((1, 21)),
                        keypoints=keypoints,
                        keypoint_scores=torch.ones((1, 21)),
                        keypoint_euler=pre_euler.unsqueeze(0),
                        gt_keypoint_euler=torch.ones_like(pre_euler).unsqueeze(
                            0)))
        else:
            pred, pred_bino_kp2d = pre_info
            for b in range(pred.shape[0]):
                keypoints = pred_bino_kp2d[b:b + 1, 0, ...]  # gt为左目信息
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

            pred_instances.keypoints3d = pred_instances.keypoints3d.cpu(
            ).numpy()
            pred_instances.keypoints3d_scores = np.ones(
                (1, pred_instances.keypoints3d.shape[1]))
            pred_instances.keypoints = pred_instances.keypoints.cpu().numpy()
            pred_instances.keypoints = np.concatenate(
                (pred_instances.keypoints, pred_instances.keypoints3d[...,
                                                                      2:]),
                axis=-1)

            if ('nimble_pose' in data_sample.meta
                    and 'keypoint_euler' in pred_instances.keys()):
                pre_euler = pred_instances.keypoint_euler[0]
                gt_nimble_pose_roctor = torch.tensor(
                    data_sample.meta['nimble_pose'][:, :3]).to(
                        pre_euler.device)
                gt_nimble_pose_matirx = batch_rodrigues(
                    gt_nimble_pose_roctor).reshape(-1, 3, 3)
                gt_euler = matrix_to_euler_angles(gt_nimble_pose_matirx)

                pre_nimble_pose = adjust_predicted_angles(
                    pre_euler, gt_euler).unsqueeze(0).cpu().numpy()
                gt_nimble_pose = gt_euler.unsqueeze(0).cpu().numpy()

                pred_instances.keypoint_euler = pre_nimble_pose
                pred_instances.gt_keypoint_euler = gt_nimble_pose

            if self.nano_2d:
                data_sample.gt_instances.keypoints = np.concatenate(
                    (data_sample.gt_instances.keypoints[..., :2],
                     data_sample.gt_instances.keypoints3d[..., 2:]),
                    axis=-1)
            pred_instances.keypoint_scores = np.ones(
                (1, pred_instances.keypoints.shape[1]))
            if data_sample.meta['flipped']:
                pred_kpt = pred_instances.keypoints[0]
                gt_kpt = data_sample.gt_instances.keypoints[0]
                pred_kpt[..., 0] = (
                    data_sample.meta['frame_width'] - 1 - pred_kpt[..., 0])
                gt_kpt[..., 0] = (
                    data_sample.meta['frame_width'] - 1 - gt_kpt[..., 0])
            data_sample.pred_instances = pred_instances
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

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        if mode:
            self.backbone.eval()
            self.neck.eval()
            self.head.eval()


@MODELS.register_module()
class TopdownPoseLiftEstimatorSeq(TopdownPoseLiftEstimator):

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 kpt3d_lift: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None,
                 seq_len: int = 32):
        super().__init__(backbone, neck, head, kpt3d_lift, train_cfg, test_cfg,
                         data_preprocessor, init_cfg, metainfo)
        self.seq_len = seq_len

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
            logger: MMLogger = MMLogger.get_current_instance()
            logger.warning(
                'topdown3d lift only support output 2d kpt for tensor mode')
            return self._forward(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode.')

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
            outputs = self.head.forward(feats)
            xy_sigma, heatmap = outputs[:2]
            if self.kpt2d_with_depth:
                depth = outputs[2]
                depth = (depth - 0.5) * 0.4  # 0.4 is the depth bound
                xy_sigma = torch.cat([xy_sigma, depth], dim=-1)
        batch_pred_instances = []
        mem = None
        assert inputs.shape[
            0] // 2 % self.seq_len == 0, \
            f'batch size {inputs.shape[0]//2} can be divided by {self.seq_len}'
        clip_len = self.seq_len * 2
        clip_num = inputs.shape[0] // clip_len
        N = xy_sigma.shape[-2]
        K = xy_sigma.shape[-1]
        xy_sigma_input = xy_sigma.reshape(clip_num, clip_len, N, K)
        for b in range(self.seq_len):
            sub_xy_input = xy_sigma_input[:, 2 * b:2 * b +
                                          2, :].reshape(-1, N, K)
            data_sample_id_list = [[
                2 * b + clip_id * clip_len, 2 * b + 1 + clip_id * clip_len
            ] for clip_id in range(clip_num)]
            data_sample_id_list = list(
                chain.from_iterable(data_sample_id_list))
            pred, pred_bino_kp2d, mem = self.kpt3d_lift.predict(
                sub_xy_input, [data_samples[i] for i in data_sample_id_list],
                mem,
                test_cfg=self.test_cfg)
            for b in range(pred.shape[0]):
                keypoints = pred_bino_kp2d[b:b + 1, 0, ...]  # gt为左目信息
                batch_pred_instances.append(
                    InstanceData(
                        keypoints3d=pred[b:b + 1, ...],
                        keypoints3d_scores=torch.ones((1, 21)),
                        keypoints=keypoints,
                        keypoint_scores=torch.ones((1, 21)),
                    ))
        final_pred_instances = []
        for i in range(clip_num):
            final_pred_instances += batch_pred_instances[i::clip_num]

        results = self.add_pred_to_datasample(final_pred_instances, None,
                                              data_samples)
        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples_seq: SampleList
                               ) -> SampleList:

        batch_data_samples = batch_data_samples_seq[::2]
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            pred_instances.keypoints3d = pred_instances.keypoints3d.cpu(
            ).numpy()
            pred_instances.keypoint3d_scores = np.ones(
                (1, pred_instances.keypoints3d.shape[1]))
            pred_instances.keypoints = pred_instances.keypoints.cpu().numpy()
            pred_instances.keypoint_scores = np.ones(
                (1, pred_instances.keypoints.shape[1]))

            data_sample.pred_instances = pred_instances
        return batch_data_samples
