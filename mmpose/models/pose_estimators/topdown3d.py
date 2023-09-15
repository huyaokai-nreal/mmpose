# Copyright (c) OpenMMLab. All rights reserved.
import copy
from itertools import zip_longest
from typing import Optional

import cv2
import numpy as np
from nreal_data_tool.utils.camera import PinholePlaneCameraModel
from scipy.optimize import leastsq

from mmpose.registry import MODELS
from mmpose.structures.bbox import bbox_cs2xyxy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .topdown import TopdownPoseEstimator


def get_root_depthv2(keypoints, camera, template_bones, undistort):
    if undistort:
        keypoints[..., :2] = camera.undistort(keypoints[..., :2])
    f = np.array(camera.f, dtype=np.float32)
    c = np.array(camera.c, dtype=np.float32)
    keypoints[..., :2] = (keypoints[..., :2] - c) / f
    root_kpt = keypoints[:1].reshape((1, 1, 3))
    root_kpt = np.tile(root_kpt, (5, 1, 1))
    kpt = keypoints[1:].reshape((5, 4, 3))
    kpt = np.concatenate([root_kpt, kpt], axis=1)
    root_list = []
    template_bones = template_bones.reshape(-1)
    for i in range(5):
        for j in range(4):
            kpt_m = kpt[i][j]
            kpt_n = kpt[i][j + 1]
            bone = template_bones[i * 4 + j]
            xm = kpt_m[0]
            ym = kpt_m[1]
            zm = kpt_m[2]
            xn = kpt_n[0]
            yn = kpt_n[1]
            zn = kpt_n[2]
            a = (xn - xm)**2 + (yn - ym)**2
            b = zn * (xn**2 + yn**2 - xn * xm - yn * ym) + zm * (
                xm**2 + ym**2 - xn * xm - yn * ym)
            c = (xn * zn - xm * zm)**2 + (yn * zn - ym * zm)**2 + (
                zn - zm)**2 - bone * bone
            root = 0.5 * (-b + np.sqrt(b**2 - 4 * a * c)) / a
            if not np.isnan(root):
                root_list.append(root)
    return np.mean(root_list)


def get_root_depth(keypoints,
                   camera,
                   template_bones,
                   weight,
                   gt: Optional[np.array] = None,
                   undistort: bool = True,
                   estimate_hand_scale: bool = False):
    rel_depth = keypoints[..., 2:]
    kpt2d = keypoints[..., :2]
    if undistort:
        kpt2d = camera.undistort(kpt2d)
    f = np.array(camera.f, dtype=np.float32)
    c = np.array(camera.c, dtype=np.float32)
    norm_kpt2d = np.concatenate([(kpt2d - c) / f, np.ones((21, 1))], axis=-1)

    def get_bones_from_kpt3d(kpt3d):
        root_kpt = kpt3d[:1].reshape((1, 1, 3))
        root_kpt = np.tile(root_kpt, (5, 1, 1))
        kpt = kpt3d[1:].reshape((5, 4, 3))
        kpt = np.concatenate([root_kpt, kpt], axis=1)
        bones = np.linalg.norm(kpt[:, 1:, :] - kpt[:, :-1, :], axis=-1)
        return bones.reshape(-1)

    def error(p, x, y, w, gt):
        w0 = w[0].reshape((1, 1))
        _w = w[1:].reshape((5, 4))
        w0 = np.tile(w0, (5, 1))
        new_w = np.concatenate([w0, _w], axis=-1)
        mean_w = (new_w[:, :4] + new_w[:, 1:]) / 2.0
        mean_w = mean_w.reshape(-1)
        mean_w /= np.max(mean_w)
        kpt3d = x * rel_depth + x * p[0]
        bones = get_bones_from_kpt3d(kpt3d)
        if estimate_hand_scale:
            kpt_error = kpt3d[8] - gt[8]
            result = mean_w * ((y * p[1] - bones).reshape(-1))
            result = np.concatenate([result, kpt_error * 10])
        else:
            result = mean_w * ((y - bones).reshape(-1))
        return result

    p0 = [0.3, 1.0]
    param = leastsq(
        error,
        p0,
        args=(norm_kpt2d, template_bones.reshape(-1), weight.reshape(-1), gt))
    return param[0][0], param[0][1]


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
                 root_mode: str = 'gt'):
        super().__init__(backbone, neck, head, train_cfg, test_cfg,
                         data_preprocessor, init_cfg)
        self.root_mode = root_mode
        self.camera_layout = camera_layout

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        if self.camera_layout == 'monocular':
            return self.add_pred_to_datasample_monocular(
                batch_pred_instances, batch_pred_fields, batch_data_samples)
        elif self.camera_layout == 'ori_binocular':
            return self.add_pred_to_datasample_binocular(
                batch_pred_instances, batch_pred_fields, batch_data_samples)
        elif self.camera_layout == 'virtual_binocular':
            return self.add_pred_to_datasample_binocular_virtual(
                batch_pred_instances, batch_pred_fields, batch_data_samples)

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
            left_kpt = left_pred_instance.keypoints[0].copy()[..., :2]
            left_gt_instances = left_data_sample.gt_instances
            input_size = left_data_sample.metainfo['input_size']
            left_bbox_centers = left_gt_instances.bbox_centers
            left_bbox_scales = left_gt_instances.bbox_scales
            left_kpt[..., :2] = left_kpt[
                ..., :
                2] / input_size * left_bbox_scales + left_bbox_centers - 0.5 * left_bbox_scales  # noqa
            right_camera = right_data_sample.meta['ori_camera']
            right_kpt = right_pred_instance.keypoints[0].copy()[..., :2]
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
            if left_data_sample.meta['flipped']:
                image_width = left_data_sample.meta['frame_width']
                left_kpt[..., 0] = image_width - left_kpt[..., 0]
                right_kpt[..., 0] = image_width - right_kpt[..., 0]
                left_gt_instances.keypoints3d[..., 0] *= -1
            left_kpt_u = left_kpt.copy()
            right_kpt_u = right_kpt.copy()
            left_kpt_u = left_camera.undistort(left_kpt_u)
            right_kpt_u = right_camera.undistort(right_kpt_u)
            left_kpt_u = (left_kpt_u - np.array([left_c], dtype=np.float32)
                          ) / np.array([left_f], dtype=np.float32)
            right_kpt_u = (right_kpt_u - np.array([right_c], dtype=np.float32)
                           ) / np.array([right_f], dtype=np.float32)
            X = cv2.triangulatePoints(T1, T2, left_kpt_u.transpose(),
                                      right_kpt_u.transpose())
            new_pred_kpt3d = X[:3] / X[3:]
            new_pred_kpt3d = new_pred_kpt3d.T
            left_pred_instance.keypoints3d = new_pred_kpt3d[None, ...]
            left_pred_instance.keypoints = left_kpt[None, ...]
            left_data_sample.pred_instances = left_pred_instance
            new_batch_data_samples.append(left_data_sample)
        return new_batch_data_samples

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
            left_kpt = left_pred_instance.keypoints[0].copy()[..., :2]
            left_gt_instances = left_data_sample.gt_instances
            left_kpt = left_gt_instances.transformed_keypoints[0, ..., :2]
            right_virtual_camera = right_data_sample.meta['virtual_camera']
            right_kpt = right_pred_instance.keypoints[0].copy()[..., :2]
            right_gt_instances = right_data_sample.gt_instances
            right_kpt = right_gt_instances.transformed_keypoints[0, ..., :2]
            T1 = np.linalg.inv(
                left_data_sample.meta['virtual_camera'].camera_to_world_xf)[:3]
            T = right_data_sample.meta['ori_xf'] @ right_data_sample.meta[
                'ori_camera'].camera_to_world_xf
            T2 = np.linalg.inv(T)[:3]
            left_f, left_c = left_virtual_camera.f, left_virtual_camera.c
            right_f, right_c = right_virtual_camera.f, right_virtual_camera.c
            left_kpt_u = left_kpt.copy()
            right_kpt_u = right_kpt.copy()
            left_kpt_u = (left_kpt - np.array([left_c], dtype=np.float32)
                          ) / np.array([left_f], dtype=np.float32)
            right_kpt_u = (right_kpt - np.array([right_c], dtype=np.float32)
                           ) / np.array([right_f], dtype=np.float32)
            X = cv2.triangulatePoints(T1, T2, left_kpt_u.transpose(),
                                      right_kpt_u.transpose())
            pred_kpt3d = X[:3] / X[3:]
            pred_kpt3d = -pred_kpt3d.T
            left_pred_instance.keypoints3d = pred_kpt3d[None, ...]
            left_camera = left_data_sample.meta['ori_camera']
            kpt2d = left_camera.eye_to_window(pred_kpt3d)
            left_pred_instance.keypoints = kpt2d[None, ...]
            left_data_sample.pred_instances = left_pred_instance
            new_batch_data_samples.append(left_data_sample)
        return new_batch_data_samples

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
            if 'virtual_camera' in data_sample.meta:
                virtual_cam = data_sample.meta['virtual_camera']
                virtual_keypoints = pred_instances.keypoints[0].copy()
                gt_keypoints3d = gt_instances.keypoints3d[0]
                if self.root_mode == 'optimize':
                    root_depth, hand_scale = get_root_depth(
                        pred_instances.keypoints[0], virtual_cam,
                        data_sample.meta['template_bones'],
                        pred_instances.keypoint_scores, gt_keypoints3d, False)
                virtual_keypoints[..., 2] *= hand_scale
                virtual_keypoints[..., 2] += root_depth
                virtual_keypoints3d = virtual_cam.window_to_eye(
                    virtual_keypoints)
                world_keypoints3d = virtual_cam.eye_to_world(
                    virtual_keypoints3d)
                ori_keypoints3d = ori_cam.world_to_eye(world_keypoints3d)
                pred_instances.keypoints[0][..., :2] = ori_cam.eye_to_window(
                    ori_keypoints3d)
                pred_instances.keypoints3d = pred_instances.keypoints.copy()
                pred_instances.keypoints3d[0] = ori_keypoints3d
            else:
                input_size = data_sample.metainfo['input_size']
                global_keypoints = copy.deepcopy(pred_instances.keypoints)
                global_keypoints[..., :2] = global_keypoints[
                    ..., :
                    2] / input_size * bbox_scales + bbox_centers - 0.5 * bbox_scales  # noqa
                # for 2d keypoint evaluation
                pred_instances.keypoints[..., :2] = pred_instances.keypoints[
                    ..., :
                    2] / input_size * bbox_scales + bbox_centers - 0.5 * bbox_scales  # noqa
                if self.root_mode == 'optimize':
                    root_depth, hand_scale = get_root_depth(
                        pred_instances.keypoints[0], ori_cam,
                        data_sample.meta['template_bones'],
                        pred_instances.keypoint_scores)
                    global_keypoints[..., 2] *= hand_scale
                elif self.root_mode == 'rootnet':
                    root_depth = pred_instances.root_depth
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
