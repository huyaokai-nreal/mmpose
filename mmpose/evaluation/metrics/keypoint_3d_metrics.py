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
                 result_dir=None,
                 root_kpt_id=20) -> None:
        super().__init__(collect_device, prefix)
        self.result_dir = result_dir
        self.root_kpt_id = root_kpt_id
        self.logger = MMLogger.get_current_instance()
        self.metric = KeypointOKSMetric(
            gesture_list=deepcopy(gesture_list), logger=self.logger)

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_instances' not in data_sample:
                raise ValueError(
                    '`pred_instances` are required to process the '
                    f'predictions results in {self.__class__.__name__}. ')
            keypoints3d = data_sample['pred_instances']['keypoints3d']
            gt = data_sample['gt_instances']
            # [N, K], the scores for all keypoints of all instances
            keypoint_scores = data_sample['pred_instances']['keypoint_scores']
            assert keypoint_scores.shape == keypoints3d.shape[:2]

            result = dict()
            result['id'] = data_sample['id']
            result['img_id'] = data_sample['img_id']
            result['gt_keypoints3d'] = gt['keypoints3d']
            result['keypoints3d'] = keypoints3d
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

    def compute_metrics(self, results: list) -> dict:
        gt_list = []
        dt_list = []
        mask_list = []
        for result in results:
            pred_pt_cam = result['keypoints3d']
            pred_pt_cam = pred_pt_cam - pred_pt_cam[:, self.root_kpt_id]
            gt_pt_cam = result['gt_keypoints3d']
            gt_pt_cam = gt_pt_cam - gt_pt_cam[:, self.root_kpt_id]
            dt_list.append(pred_pt_cam)
            gt_list.append(gt_pt_cam)
            mask_list.append(result['mask'])
        gt = np.concatenate(gt_list, axis=0)
        dt = np.concatenate(dt_list, axis=0)
        mask = np.concatenate(mask_list, axis=0)
        all_mpjpe = keypoint_epe(dt, gt, mask)
        x_mpjpe = keypoint_epe(dt[..., :1], gt[..., :1], mask)
        y_mpjpe = keypoint_epe(dt[..., 1:2], gt[..., 1:2], mask)
        z_mpjpe = keypoint_epe(dt[..., 2:], gt[..., 2:], mask)
        return dict(
            all_mpjpe=all_mpjpe,
            x_mpjpe=x_mpjpe,
            y_mpjpe=y_mpjpe,
            z_mpjpe=z_mpjpe)
