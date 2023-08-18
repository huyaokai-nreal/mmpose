# Copyright (c) OpenMMLab. All rights reserved.
import copy
from itertools import zip_longest
from typing import Optional

import numpy as np
from scipy.optimize import leastsq

from mmpose.registry import MODELS
from mmpose.structures.bbox import bbox_cs2xyxy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .topdown import TopdownPoseEstimator


def get_root_depth(keypoints, camera, template_bones, weight):
    rel_depth = keypoints[..., 2:]
    kpt2d = keypoints[..., :2]
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

    def model(x, p):
        kpt3d = x * rel_depth + x * p[0]
        bones = get_bones_from_kpt3d(kpt3d)
        return bones

    def error(p, x, y, w):
        w0 = w[0].reshape((1, 1))
        _w = w[1:].reshape((5, 4))
        w0 = np.tile(w0, (5, 1))
        new_w = np.concatenate([w0, _w], axis=-1)
        mean_w = (new_w[:, :4] + new_w[:, 1:]) / 2.0
        mean_w = mean_w.reshape(-1)
        mean_w /= np.max(mean_w)
        result = mean_w * ((y - model(x, p)).reshape(-1))
        return result

    p0 = [0.3]
    param = leastsq(
        error,
        p0,
        args=(norm_kpt2d, template_bones.reshape(-1), weight.reshape(-1)))
    return param[0]


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
                 root_mode: str = 'gt'):
        super().__init__(backbone, neck, head, train_cfg, test_cfg,
                         data_preprocessor, init_cfg)
        self.root_mode = root_mode

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
            # uv depth to camera coord pose
            ori_cam = data_sample.meta['ori_camera']
            root_depth = data_sample.meta['root_depth']
            if 'virtual_camera' in data_sample.meta:
                virtual_cam = data_sample.meta['virtual_camera']
                virtual_keypoints = pred_instances.keypoints[0].copy()
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
                if data_sample.meta['flipped']:
                    pred_instances.keypoints[..., 0] = data_sample.meta[
                        'frame_width'] - pred_instances.keypoints[..., 0]
                    gt_instances.keypoints[..., 0] = data_sample.meta[
                        'frame_width'] - gt_instances.keypoints[..., 0]
                if self.root_mode == 'optimize':
                    root_depth = get_root_depth(
                        pred_instances.keypoints[0], ori_cam,
                        data_sample.meta['template_bones'],
                        pred_instances.keypoint_scores)
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
