# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.logging import MMLogger
import numpy as np
from mmengine.evaluator import BaseMetric
from nreal_data_tool.metric import KeypointOKSMetric
from mmpose.registry import METRICS
from typing import List
from copy import deepcopy
from ..functional.keypoint_eval import keypoint_epe


@METRICS.register_module()
class MPJPEMetric(BaseMetric):
    default_prefix: Optional[str] = ''

    def __init__(self,
                 gesture_list: List[str] = [],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 result_dir=None) -> None:
        super().__init__(collect_device, prefix)
        self.result_dir = result_dir
        self.logger = MMLogger.get_current_instance()
        self.metric = KeypointOKSMetric(
            gesture_list=deepcopy(gesture_list), logger=self.logger)

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_instances' not in data_sample:
                raise ValueError(
                    '`pred_instances` are required to process the '
                    f'predictions results in {self.__class__.__name__}. ')
            keypoints = data_sample['pred_instances']['keypoints']
            gt = data_sample['gt_instances']
            gt_keypoints = gt['keypoints']
            # [N, K], the scores for all keypoints of all instances
            keypoint_scores = data_sample['pred_instances']['keypoint_scores']
            assert keypoint_scores.shape == keypoints.shape[:2]

            result = dict()
            result['id'] = data_sample['id']
            result['img_id'] = data_sample['img_id']
            result['gt_keypoints'] = gt_keypoints
            result['keypoints'] = keypoints
            mask = gt['keypoints_visible'].astype(bool).reshape(1, -1)
            result['mask'] = mask
            result['keypoint_scores'] = keypoint_scores
            result['bbox_scores'] = data_sample['gt_instances']['bbox_scores']
            result['meta'] = data_sample['meta']
            # get area information
            if 'bbox_scales' in data_sample['gt_instances']:
                result['meta']['bbox_scales'] = data_sample['gt_instances'][
                    'bbox_scales'].tolist()
                result['meta']['bbox_centers'] = data_sample['gt_instances'][
                    'bbox_centers'].tolist()
                result['area'] = np.prod(
                    data_sample['gt_instances']['bbox_scales'], axis=1)
            # add converted result to the results list
            self.results.append(result)

    @staticmethod
    def image_to_cam(pixel_coord, f, c):
        """Transform the joints from their pixel coordinates to their camera
        coordinates.

        Note:
            N: number of joints

        Args:
            pixel_coord (ndarray[N, 3]): 3D joints coordinates
                in the pixel coordinate system
            f (ndarray[2]): focal length of x and y axis
            c (ndarray[2]): principal point of x and y axis

        Returns:
            cam_coord (ndarray[N, 3]): 3D joints coordinates
                in the camera coordinate system
        """
        x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
        y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
        z = pixel_coord[:, 2]
        cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
        return cam_coord

    def compute_metrics(self, results: list) -> dict:
        gt_list = []
        dt_list = []
        mask_list = []
        for result in results:
            keypoints = result['keypoints'][0]
            # 2d keypoints to 2.5D keypoints
            focal = result['meta']['focal']
            princpt = result['meta']['princpt']
            root_depth = result['meta']['root_depth']
            keypoints[:, 2] += root_depth
            pred_pt_cam = self.image_to_cam(keypoints, focal, princpt)
            pred_pt_cam = pred_pt_cam - pred_pt_cam[20]
            gt_pt_cam = result['meta']['keypoints_cam']
            gt_pt_cam = gt_pt_cam - gt_pt_cam[20]
            dt_list.append(pred_pt_cam[np.newaxis, ...])
            gt_list.append(gt_pt_cam[np.newaxis, ...])
            mask_list.append(result['mask'])
        gt = np.concatenate(gt_list, axis=0)
        dt = np.concatenate(dt_list, axis=0)
        mask = np.concatenate(mask_list, axis=0)
        mpjpe = keypoint_epe(dt, gt, mask)
        return dict(mpjpe=mpjpe)
