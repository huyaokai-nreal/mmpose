# Copyright (c) OpenMMLab. All rights reserved.
import copy
from itertools import zip_longest
from typing import Optional

import cv2
import numpy as np
from mmengine import MMLogger
from torch import Tensor
import torch
import torch.nn as nn

from nreal_data_tool.utils.camera import NoDistortion, PinholePlaneCameraModel

from mmpose.models.utils.pose_solver import (get_kpt_depth,
                                             get_kpt_depth_binocular,
                                             get_kpt_depth_binocular63,
                                             get_root_depth)
from mmpose.registry import MODELS
from mmpose.structures.bbox import bbox_cs2xyxy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .topdown import TopdownPoseEstimator
from mmengine.structures import InstanceData
from itertools import chain, zip_longest

@MODELS.register_module()
class TopdownPose3DEstimator(TopdownPoseEstimator):

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 camera_layout: str = 'monocular',
                 root_mode: str = 'gt',
                 refine_kpt: bool = False,
                 root_id: int = 0):
        super().__init__(backbone, neck, head, train_cfg, test_cfg,
                         data_preprocessor, init_cfg)
        self.root_mode = root_mode
        self.camera_layout = camera_layout
        self.last_kpt3d = None
        self.root_id = root_id
        self.refine_kpt = refine_kpt
        
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
            x = self.head._forward(x)
            if isinstance(x, list):
                x = x[-1]
        return x

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:

        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)
        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)
        
        if self.camera_layout == "nimble":
            results = self.add_pred_to_datasample_nimble(
                preds, None,  data_samples)
        else:
            if isinstance(preds, tuple):
                batch_pred_instances, batch_pred_fields = preds
            else:
                batch_pred_instances = preds
                batch_pred_fields = None
            results = self.add_pred_to_datasample(
                batch_pred_instances, batch_pred_fields, data_samples)

        return results

    def add_pred_to_datasample(self, batch_pred_instances,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        if self.camera_layout == 'monocular':
            return self.add_pred_to_datasample_monocular(
                batch_pred_instances, batch_pred_fields, batch_data_samples)
        elif self.camera_layout == 'ori_binocular':
            return self.add_pred_to_datasample_binocular(
                batch_pred_instances, batch_pred_fields, batch_data_samples)
        elif self.camera_layout == 'ori_binocular_depth':
            return self.add_pred_to_datasample_binocular_depth(
                batch_pred_instances, batch_pred_fields, batch_data_samples)
        elif self.camera_layout == 'virtual_binocular':
            return self.add_pred_to_datasample_binocular_virtual(
                batch_pred_instances, batch_pred_fields, batch_data_samples)
        else:
            logger = MMLogger.get_current_instance()
            logger.error(f'layout { self.camera_layout} is not supported')

    # warpaffine抠图时进行三角化
    def add_pred_to_datasample_binocular_depth(
            self, batch_pred_instances: InstanceList,
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
        N = len(batch_data_samples) // 2
        new_batch_data_samples = []
        for i in range(N):
            left_pred_instance = batch_pred_instances[2 * i]
            left_data_sample = batch_data_samples[2 * i]
            right_pred_instance = batch_pred_instances[2 * i + 1]
            right_data_sample = batch_data_samples[2 * i + 1]
            left_camera = left_data_sample.meta['ori_camera']
            left_kpt = left_pred_instance.keypoints[0].copy()
            left_gt_instances = left_data_sample.gt_instances
            input_size = left_data_sample.metainfo['input_size']
            left_bbox_centers = left_gt_instances.bbox_centers
            left_bbox_scales = left_gt_instances.bbox_scales
            left_kpt[..., :2] = left_kpt[
                ..., :
                2] / input_size * left_bbox_scales + left_bbox_centers - 0.5 * left_bbox_scales  # noqa
            right_camera = right_data_sample.meta['ori_camera']
            right_camera.camera_to_world_xf = right_data_sample.meta['ori_xf']
            right_kpt = right_pred_instance.keypoints[0].copy()
            right_gt_instances = right_data_sample.gt_instances
            right_bbox_centers = right_gt_instances.bbox_centers
            right_bbox_scales = right_gt_instances.bbox_scales
            right_kpt[..., :2] = right_kpt[
                ..., :
                2] / input_size * right_bbox_scales + right_bbox_centers - 0.5 * right_bbox_scales  # noqa
            if left_data_sample.meta['flipped']:
                image_width = left_data_sample.meta['frame_width']
                left_kpt[..., 0] = image_width - 1 - left_kpt[..., 0]
                right_kpt[..., 0] = image_width - 1 - right_kpt[..., 0]
                left_gt_instances.keypoints3d[..., 0] *= -1
            left_pred_instance.keypoints = left_kpt[None, ..., :2].copy()
            left_kpt_depth = get_kpt_depth_binocular(left_kpt, left_camera,
                                                     right_kpt, right_camera,
                                                     None)
            left_kpt[..., -1] = left_kpt_depth
            left_kpt3d = left_camera.window_to_eye(left_kpt)
            left_pred_instance.keypoints3d = left_kpt3d[None, ...]
            left_data_sample.pred_instances = left_pred_instance
            new_batch_data_samples.append(left_data_sample)
        return new_batch_data_samples

    # warpaffine抠图时进行三角化
    def add_pred_to_datasample_binocular(
            self, batch_pred_instances: InstanceList,
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
        N = len(batch_data_samples) // 2
        new_batch_data_samples = []
        for i in range(N):
            left_pred_instance = batch_pred_instances[2 * i]
            left_data_sample = batch_data_samples[2 * i]
            right_pred_instance = batch_pred_instances[2 * i + 1]
            right_data_sample = batch_data_samples[2 * i + 1]
            left_camera = left_data_sample.meta['ori_camera']
            left_kpt = left_pred_instance.keypoints[0].copy()
            left_gt_instances = left_data_sample.gt_instances
            input_size = left_data_sample.metainfo['input_size']
            left_bbox_centers = left_gt_instances.bbox_centers
            left_bbox_scales = left_gt_instances.bbox_scales
            left_kpt[..., :2] = left_kpt[
                ..., :
                2] / input_size * left_bbox_scales + left_bbox_centers - 0.5 * left_bbox_scales  # noqa
            right_camera = right_data_sample.meta['ori_camera']
            right_kpt = right_pred_instance.keypoints[0].copy()
            right_gt_instances = right_data_sample.gt_instances
            right_bbox_centers = right_gt_instances.bbox_centers
            right_bbox_scales = right_gt_instances.bbox_scales
            right_kpt[..., :2] = right_kpt[
                ..., :
                2] / input_size * right_bbox_scales + right_bbox_centers - 0.5 * right_bbox_scales  # noqa
            T1 = left_data_sample.meta['ori_camera'].camera_to_world_xf[:3]
            T = right_data_sample.meta['ori_xf'] @ right_data_sample.meta[
                'ori_camera'].camera_to_world_xf
            T2 = np.linalg.inv(T)[:3]
            left_f, left_c = left_camera.f, left_camera.c
            right_f, right_c = right_camera.f, right_camera.c
            if left_data_sample.meta.get('norm_depth', False):
                hand_scale = left_data_sample.meta.get('hand_scale', 1.0)
                left_gt_instances.keypoints[..., -1] *= hand_scale
                left_kpt[..., -1] *= hand_scale
            left_pred_instance.keypoints = left_kpt[None, ...].copy()
            if left_data_sample.meta['flipped']:
                image_width = left_data_sample.meta['frame_width']
                left_kpt[..., 0] = image_width - 1 - left_kpt[..., 0]
                right_kpt[..., 0] = image_width - 1 - right_kpt[..., 0]
                left_gt_instances.keypoints3d[..., 0] *= -1
            left_kpt_u = left_kpt.copy()
            right_kpt_u = right_kpt.copy()
            left_kpt_u[..., :2] = left_camera.undistort(left_kpt_u[..., :2])
            right_kpt_u[..., :2] = right_camera.undistort(right_kpt_u[..., :2])
            left_kpt_norm = (left_kpt_u[..., :2] -
                             np.array([left_c], dtype=np.float32)) / np.array(
                                 [left_f], dtype=np.float32)
            right_kpt_norm = (right_kpt_u[..., :2] - np.array(
                [right_c], dtype=np.float32)) / np.array([right_f],
                                                         dtype=np.float32)
            X = cv2.triangulatePoints(T1, T2, left_kpt_norm.transpose(),
                                      right_kpt_norm.transpose())
            new_pred_kpt3d = X[:3] / X[3:]
            new_pred_kpt3d = new_pred_kpt3d.T
            right_camera.camera_to_world_xf = right_data_sample.meta['ori_xf']
            # dont consider distortion in the refine stage
            if self.refine_kpt:
                left_camera.distort = NoDistortion()
                right_camera.distort = NoDistortion()
                refined_kpt3d = get_kpt_depth_binocular63(
                    left_kpt_u, left_camera, right_kpt_u, right_camera,
                    new_pred_kpt3d, self.last_kpt3d)
                self.last_kpt3d = refined_kpt3d
                new_pred_kpt3d = refined_kpt3d
            left_pred_instance.keypoints3d = new_pred_kpt3d[None, ...]
            left_data_sample.pred_instances = left_pred_instance
            new_batch_data_samples.append(left_data_sample)
        return new_batch_data_samples

    # pcl处理后利用虚拟相机进行三角化
    def add_pred_to_datasample_binocular_virtual(
            self, batch_pred_instances: InstanceList,
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
        N = len(batch_data_samples) // 2
        new_batch_data_samples = []
        for i in range(N):
            left_pred_instance = batch_pred_instances[2 * i]
            left_data_sample = batch_data_samples[2 * i]
            right_pred_instance = batch_pred_instances[2 * i + 1]
            right_data_sample = batch_data_samples[2 * i + 1]
            left_virtual_camera = left_data_sample.meta['virtual_camera']
            left_kpt = left_pred_instance.keypoints[0].copy()
            right_virtual_camera = right_data_sample.meta['virtual_camera']
            right_kpt = right_pred_instance.keypoints[0].copy()
            if left_data_sample.meta['flipped']:
                mirror_x_matrix = np.eye(4)
                mirror_x_matrix[0][0] = -1
                T1 = np.linalg.inv(
                    mirror_x_matrix @ left_data_sample.meta['virtual_camera'].
                    camera_to_world_xf)[:3]
                T = right_data_sample.meta[
                    'ori_xf'] @ mirror_x_matrix @ right_data_sample.meta[
                        'virtual_camera'].camera_to_world_xf
                T2 = np.linalg.inv(T)[:3]
                left_data_sample.gt_instances.keypoints3d[..., 0] *= -1
            else:
                T1 = np.linalg.inv(left_data_sample.meta['virtual_camera'].
                                   camera_to_world_xf)[:3]
                T = right_data_sample.meta['ori_xf'] @ right_data_sample.meta[
                    'virtual_camera'].camera_to_world_xf
                right_virtual_camera.camera_to_world_xf = T
                T2 = np.linalg.inv(T)[:3]
            left_f, left_c = left_virtual_camera.f, left_virtual_camera.c
            right_f, right_c = right_virtual_camera.f, right_virtual_camera.c
            left_kpt_u = left_kpt[..., :2].copy()
            right_kpt_u = right_kpt[..., :2].copy()
            left_kpt_u = (left_kpt_u - np.array([left_c], dtype=np.float32)
                          ) / np.array([left_f], dtype=np.float32)
            right_kpt_u = (right_kpt_u - np.array([right_c], dtype=np.float32)
                           ) / np.array([right_f], dtype=np.float32)
            X = cv2.triangulatePoints(T1, T2, left_kpt_u.transpose(),
                                      right_kpt_u.transpose())
            pred_kpt3d = X[:3] / X[3:]
            pred_kpt3d = pred_kpt3d.T
            left_pred_instance.keypoints3d = pred_kpt3d[None, ...]
            left_camera = left_data_sample.meta['ori_camera']
            # vir_camera_window->vir_camera_eye->ori_camera_eye->ori_camera_windows
            kpt_norm_eye = left_virtual_camera.window_to_eye(left_kpt[..., :2])
            kpt_norm_world = left_virtual_camera.eye_to_world(kpt_norm_eye)
            kpt2d_ori = left_camera.eye_to_window(kpt_norm_world)
            left_pred_instance.keypoints[..., :2] = kpt2d_ori[None, ...]
            left_data_sample.pred_instances = left_pred_instance
            if left_data_sample.meta.get('norm_depth', False):
                hand_scale = left_data_sample.meta.get('hand_scale', 1.0)
                left_data_sample.pred_instances.keypoints[...,
                                                          -1] *= hand_scale
                left_data_sample.gt_instances.keypoints[
                    ...,
                    -1] = left_data_sample.gt_instances.transformed_keypoints[
                        ..., -1] * hand_scale
            new_batch_data_samples.append(left_data_sample)
        return new_batch_data_samples

    # 单目使用模版进行kpt3d求解
    def add_pred_to_datasample_monocular(
            self, batch_pred_instances: InstanceList,
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

        self.last_kpt3d = None
        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            bbox_centers = gt_instances.bbox_centers
            bbox_scales = gt_instances.bbox_scales
            # uv depth to camera coord pose
            ori_cam = data_sample.meta.get(
                'ori_camera',
                PinholePlaneCameraModel(
                    c=(240, 320),
                    f=(200, 200),
                    camera_to_world_xf=np.eye(4),
                    distort_coeffs=[]))
            root_depth = data_sample.meta.get('root_depth', 0.5)
            gt_hand_scale = data_sample.meta.get('hand_scale', 1.0)
            if 'virtual_camera' in data_sample.meta:
                virtual_cam = data_sample.meta['virtual_camera']
                gt_keypoints3d = gt_instances.keypoints3d[0]
                if data_sample.meta.get('norm_depth', False):
                    pred_instances.keypoints[..., 2] *= gt_hand_scale
                virtual_keypoints = pred_instances.keypoints[0].copy()
                if self.root_mode == 'optimize':
                    root_depth, hand_scale = get_root_depth(
                        virtual_keypoints, virtual_cam,
                        data_sample.meta['template_bones'] * gt_hand_scale,
                        pred_instances.keypoint_scores, gt_keypoints3d, False,
                        self.last_kpt3d)
                elif self.root_mode == 'optimizev2':
                    kpt_depth = get_kpt_depth(
                        pred_instances.keypoints[0],
                        virtual_cam,
                        data_sample.meta['template_bones'],
                        last_kpt3d=self.last_kpt3d,
                        root_id=self.root_id)
                    virtual_keypoints[..., 2] = kpt_depth
                    root_depth = 0
                virtual_keypoints[..., 2] += root_depth
                virtual_keypoints3d = virtual_cam.window_to_eye(
                    virtual_keypoints)
                if data_sample.meta['flipped']:
                    virtual_keypoints3d[..., 0] *= -1
                world_keypoints3d = virtual_cam.eye_to_world(
                    virtual_keypoints3d)
                self.last_kpt3d = world_keypoints3d
                # vir_camera_window->vir_camera_eye->ori_camera_eye->ori_camera_windows
                kpt_norm_eye = virtual_cam.window_to_eye(
                    virtual_keypoints[:, :2])
                if data_sample.meta['flipped']:
                    kpt_norm_eye[..., 0] *= -1
                kpt_norm_world = virtual_cam.eye_to_world(kpt_norm_eye)
                kpt2d_ori = ori_cam.eye_to_window(kpt_norm_world)
                pred_instances.keypoints[0][..., :2] = kpt2d_ori
                pred_instances.keypoints3d = pred_instances.keypoints.copy()
                pred_instances.keypoints3d[0] = world_keypoints3d
            else:
                input_size = data_sample.metainfo['input_size']
                if data_sample.meta['flipped']:
                    pred_instances.keypoints[
                        ...,
                        0] = input_size[0] - 1 - pred_instances.keypoints[...,
                                                                          0]
                global_keypoints = copy.deepcopy(pred_instances.keypoints)
                global_keypoints[..., :2] = global_keypoints[
                    ..., :
                    2] / input_size * bbox_scales + bbox_centers - 0.5 * bbox_scales  # noqa
                # for 2d keypoint evaluation
                pred_instances.keypoints[..., :2] = pred_instances.keypoints[
                    ..., :
                    2] / input_size * bbox_scales + bbox_centers - 0.5 * bbox_scales  # noqa
                if self.root_mode == 'optimize':
                    kpt = pred_instances.keypoints[0].copy()
                    root_depth, hand_scale = get_root_depth(
                        kpt, ori_cam, data_sample.meta['template_bones'],
                        pred_instances.keypoint_scores)
                    if data_sample.meta.get('norm_depth', False):
                        global_keypoints[..., 2] *= hand_scale
                elif self.root_mode == 'optimizev2':
                    kpt = global_keypoints[0].copy()
                    kpt[..., :2] = ori_cam.undistort(kpt[..., :2])
                    tmp_cam = copy.deepcopy(ori_cam)
                    tmp_cam.distort = NoDistortion()
                    kpt_depth = get_kpt_depth(
                        kpt,
                        tmp_cam,
                        data_sample.meta['template_bones'],
                        root_id=self.root_id,
                        last_kpt3d=None)
                    global_keypoints[..., 2] = kpt_depth
                    root_depth = 0
                global_keypoints[..., 2] += root_depth
                ori_keypoints3d = ori_cam.window_to_eye(global_keypoints[0])
                pred_instances.keypoints3d = global_keypoints.copy()
                pred_instances.keypoints3d[0] = ori_keypoints3d
            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = bbox_cs2xyxy(bbox_centers, bbox_scales)
            pred_instances.bbox_scores = gt_instances.bbox_scores
            if data_sample.meta.get('norm_depth', False):
                gt_instances.keypoints[..., -1] *= gt_hand_scale
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


    # 单目使用nimble结果保存
    def add_pred_to_datasample_nimble(
            self, pre_info,
            batch_pred_fields: Optional[PixelDataList],
            batch_data_samples: SampleList) -> SampleList:
        
        pred, pred_bino_kp2d, sigmas = pre_info
        batch_pred_instances = []
        for b in range(pred.shape[0]):
            keypoints = pred_bino_kp2d[b:b + 1, ...]  # gt为左目信息
            batch_pred_instances.append(
                InstanceData(
                    keypoints3d=pred[b:b + 1, ...],
                    keypoints3d_scores=sigmas[b:b + 1, ...],
                    keypoints=keypoints,
                    keypoint_scores=torch.ones((1, 21)),
                ))

        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):
            pred_instances.keypoints3d = pred_instances.keypoints3d.cpu(
            ).numpy()
            pred_instances.keypoints3d_scores = \
                pred_instances.keypoints3d_scores.cpu().numpy()
            pred_instances.keypoints = pred_instances.keypoints.cpu().numpy()
            pred_instances.keypoints = np.concatenate(
                (pred_instances.keypoints, pred_instances.keypoints3d[...,
                                                                      2:]),
                axis=-1)
            ori_cam = data_sample.meta['ori_camera']
            hand2d_gt = ori_cam.eye_to_window(data_sample.gt_instances.keypoints3d[0])
            
            # if 'virtual_camera' in data_sample.meta:
            #     ori_cam = data_sample.meta['ori_camera']
            #     virtual_cam = data_sample.meta['virtual_camera']
            #     # vritual_point_world = virtual_cam.world_to_eye(hand3d_gt_sin.cpu().numpy())
            #     # vritual_point = virtual_cam.eye_to_window(vritual_point_world)
            #     kpt_norm_eye = virtual_cam.window_to_eye(
            #         pred_instances.keypoints[0,:, :2])
            #     kpt_norm_world = virtual_cam.eye_to_world(kpt_norm_eye)
            #     kpt2d_ori = ori_cam.eye_to_window(kpt_norm_world)
            #     pred_instances.keypoints[0][..., :2] = kpt2d_ori
                
            # data_sample.gt_instances.keypoints = np.concatenate(
            #     (hand2d_gt, data_sample.gt_instances.keypoints3d[..., 2:]),
            #     axis=-1)
            pred_instances.keypoint_scores = np.ones(
                (1, pred_instances.keypoints.shape[1]))
            # if data_sample.meta['flipped']:
            #     pred_kpt = pred_instances.keypoints[0]
            #     gt_kpt = data_sample.gt_instances.keypoints[0]
            #     pred_kpt[..., 0] = (
            #         data_sample.meta['frame_width'] - 1 - pred_kpt[..., 0])
            #     gt_kpt[..., 0] = (
            #         data_sample.meta['frame_width'] - 1 - gt_kpt[..., 0])
            data_sample.pred_instances = pred_instances
        return batch_data_samples

@MODELS.register_module()
class TopdownPose3DAndHeldLabelEstimator(TopdownPoseEstimator):

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 camera_layout: str = 'monocular',
                 root_mode: str = 'gt',
                 refine_kpt: bool = False,
                 root_id: int = 0):
        super().__init__(backbone, neck, head, train_cfg, test_cfg,
                         data_preprocessor, init_cfg)
        # 冻结backbone权重
        for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 冻结head中非cls的权重
        for name, param in self.head.named_parameters():
            if "cls_module" not in name:
                param.requires_grad = False
        
    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:

        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)
        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)
        
        return self.add_pred_to_datasample(data_samples, preds)
    
    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: InstanceList) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        
        return data_samples
    

@MODELS.register_module()
class TopdownPose3DEstimatorSeq(TopdownPose3DEstimator):

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 camera_layout: str = 'monocular',
                 root_id: int = 0,
                 seq_len: int = 32):
        super().__init__(backbone, neck, head, train_cfg, test_cfg,
                         data_preprocessor, init_cfg, camera_layout, root_id)
        self.seq_len = seq_len

    def inference_getm(self, feats, inputs, data_samples, begin, step):
        mem = None
        pred_instances = []
        clip_len = feats[0].shape[0]
        clip_num = inputs.shape[0] // clip_len
        feats = feats[-1]
        C, H, W = feats.shape[-3], feats.shape[-2], feats.shape[-1]
        feats_input = feats.reshape(clip_num, clip_len, C, H, W)
        for b in range(begin, clip_len, step):
            sub_feat_input = feats_input[:, b:b +1, ...].reshape(1, -1, C, H, W)
            pred_kpt3d, pred_kpt2d, mem, sigma = self.head.predict(
                sub_feat_input, [data_samples[b]],
                mem,
                test_cfg=self.test_cfg)
            for b in range(pred_kpt3d.shape[0]):
                pred_instances.append(
                    InstanceData(
                        keypoints3d=pred_kpt3d[b:b + 1, ...],
                        keypoints3d_scores=sigma[b:b + 1, ...],
                        keypoints=pred_kpt2d[b:b + 1, ...],
                        keypoint_scores=torch.ones((1, 21)),
                    ))
        return pred_instances

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:

        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)
        
        clip_len = feats[0].shape[0]
        clip_num = inputs.shape[0] // clip_len
        batch_pred_instances = []
        assert inputs.shape[
            0]  % self.seq_len == 0, \
            f'batch size {inputs.shape[0]} can be divided by {self.seq_len}'

        if "hot3d" in data_samples[0].img_path:
            left_batch_pred_instances = self.inference_getm(feats, inputs, data_samples, begin=0, step=2)
            right_batch_pred_instances = self.inference_getm(feats, inputs, data_samples, begin=1, step=2)
            for left_instance, right_instance in zip(left_batch_pred_instances, right_batch_pred_instances):
                batch_pred_instances.append(left_instance)
                batch_pred_instances.append(right_instance)
        else:
            batch_pred_instances = self.inference_getm(feats, inputs, data_samples, begin=0, step=1)

        final_pred_instances = []
        for i in range(clip_num):
            final_pred_instances += batch_pred_instances[i::clip_num]
        
        if self.camera_layout == "nimble":
            results = self.add_pred_to_datasample_nimble_seq(
                final_pred_instances, data_samples)
        else:
            if isinstance(preds, tuple):
                batch_pred_instances, batch_pred_fields = preds
            else:
                batch_pred_instances = preds
                batch_pred_fields = None
            results = self.add_pred_to_datasample(
                batch_pred_instances, batch_pred_fields, data_samples)

        return results
    
    def add_pred_to_datasample_nimble_seq(self, batch_pred_instances: InstanceList,
                               batch_data_samples_seq: SampleList
                               ) -> SampleList:


        batch_data_samples = batch_data_samples_seq
        assert len(batch_pred_instances) == len(batch_data_samples)

        for pred_instances, data_sample in zip_longest(
                batch_pred_instances, batch_data_samples):

            pred_instances.keypoints3d = pred_instances.keypoints3d.cpu(
            ).numpy()
            pred_instances.keypoints3d_scores = pred_instances.keypoints3d_scores.cpu(
            ).numpy()
            pred_instances.keypoints = pred_instances.keypoints.cpu().numpy()
            pred_instances.keypoint_scores = np.ones(
                (1, pred_instances.keypoints.shape[1]))

            data_sample.pred_instances = pred_instances
        return batch_data_samples
        