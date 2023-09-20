# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import tempfile
from copy import deepcopy
from typing import List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from nreal_data_tool.metric import KeypointOKSMetric
from nreal_data_tool.schema import KeypointEvaluationItem

from mmpose.registry import METRICS


@METRICS.register_module()
class NrealKeypointAP(BaseMetric):
    default_prefix: Optional[str] = ''

    def __init__(self,
                 gesture_list: List[str] = [],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 result_dir=None,
                 with_tag: bool = False) -> None:
        super().__init__(collect_device, prefix)
        self.result_dir = result_dir
        self.logger = MMLogger.get_current_instance()
        self.metric = KeypointOKSMetric(
            gesture_list=deepcopy(gesture_list),
            logger=self.logger,
            with_tag=with_tag)

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_instances' not in data_sample:
                raise ValueError(
                    '`pred_instances` are required to process the '
                    f'predictions results in {self.__class__.__name__}. ')
            keypoints = data_sample['pred_instances']['keypoints']
            # [N, K], the scores for all keypoints of all instances
            keypoint_scores = data_sample['pred_instances']['keypoint_scores']
            assert keypoint_scores.shape == keypoints.shape[:2]
            result = KeypointEvaluationItem(image_id=data_sample['img_id'])
            result.gt_keypoints3d = (
                data_sample['gt_instances']['keypoints3d'][0] *
                1e3).tolist()  # m -> mm
            result.gt_keypoints = data_sample['gt_instances']['keypoints'][
                0].tolist()
            result.keypoints = keypoints[0].tolist()
            result.keypoint_visible = data_sample['gt_instances'][
                'keypoints_visible'].reshape((-1)).tolist()
            result.score = float(np.mean(keypoint_scores))
            if 'tag' in data_sample['meta']:
                result.meta['tag'] = data_sample['meta']['tag']
            if 'gesture' in data_sample['meta']:
                result.meta['gesture'] = data_sample['meta']['gesture']
            # get area information
            if 'bbox_scales' in data_sample['gt_instances']:
                result.meta['bbox_scales'] = data_sample['gt_instances'][
                    'bbox_scales'].tolist()
                result.meta['bbox_centers'] = data_sample['gt_instances'][
                    'bbox_centers'].tolist()
                result.area = float(
                    np.prod(
                        data_sample['gt_instances']['bbox_scales'], axis=1))
            # add converted result to the results list
            self.results.append(result.to_dict())

    def compute_metrics(self, results: list) -> dict:

        if self.result_dir is None:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')
        else:
            res_file = osp.join(self.result_dir, 'result_keypoints.json')
        self.logger.info(f'result file path is {res_file}')
        # from nreal_data_tool.metric.filter_metric import FilterMetric
        # kpt_list = []
        # kpts = np.concatenate(kpt_list, axis=0)
        # filter_result = FilterMetric()(kpts, kpts)
        # print(filter_result)
        # print(filter_result['filtered_acc_noise'].mean())
        with open(res_file, 'w') as f:
            json.dump(self.results, f, sort_keys=True, indent=4)
        return self.metric(res_file)
