# Copyright (c) OpenMMLab. All rights reserved.
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

            pred_instances.keypoints[..., :2] = pred_instances.keypoints[
                ..., :
                2] / input_size * bbox_scales + bbox_centers - 0.5 * bbox_scales  # noqa

            # uv depth to camera coord pose
            root_depth = data_sample.meta['root_depth']
            camera = data_sample.meta['camera']
            pred_instances.keypoints3d = pred_instances.keypoints.copy()
            pred_instances.keypoints3d[..., 2] += root_depth
            pred_instances.keypoints3d[0] = camera.pixel_to_camera(
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
