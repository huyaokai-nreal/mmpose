# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
from itertools import zip_longest
from typing import Optional
from mmpose.registry import MODELS
from mmpose.utils.typing import (InstanceList, PixelDataList, SampleList)
from mmpose.structures.bbox import bbox_cs2xyxy
from .topdown import TopdownPoseEstimator


@MODELS.register_module()
class TopdownPose3DEstimator(TopdownPoseEstimator):

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
            input_size = data_sample.metainfo['input_size']
            # uv depth to camera coord pose
            root_depth = data_sample.meta['root_depth']
            global_keypoints = copy.deepcopy(pred_instances.keypoints)
            if 'virtual_camera' in data_sample.meta:
                camera = data_sample.meta['virtual_camera']
                P_virt2orig = data_sample.meta['P_virt2orig'][0]
                local_keypoints = pred_instances.keypoints[0][
                    ..., :2] / input_size
                local_keypoints = np.concatenate(
                    [local_keypoints,
                     np.ones((local_keypoints.shape[0], 1))],
                    axis=-1)
                origin_keypoints = (P_virt2orig @ local_keypoints.T).T
                pred_instances.keypoints[..., :2] = origin_keypoints[
                    ..., :2] / origin_keypoints[..., 2:]
            else:
                global_keypoints[..., :2] = global_keypoints[
                    ..., :
                    2] / input_size * bbox_scales + bbox_centers - 0.5 * bbox_scales  # noqa
                camera = data_sample.meta['camera']
                # for 2d keypoint evaluation
                pred_instances.keypoints[..., :2] = pred_instances.keypoints[
                    ..., :
                    2] / input_size * bbox_scales + bbox_centers - 0.5 * bbox_scales  # noqa
            pred_instances.keypoints3d = global_keypoints.copy()
            pred_instances.keypoints3d[..., 2] += root_depth
            pred_instances.keypoints3d[0] = camera.pixel_to_camera(
                pred_instances.keypoints3d[0])
            if 'virtual_camera' in data_sample.meta:
                pred_instances.keypoints3d[0] = camera.camera_to_world(
                    pred_instances.keypoints3d[0])
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
