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
            keypoints_visible = data_sample['gt_instances'][
                'keypoints_visible']
            assert keypoint_scores.shape == keypoints.shape[:2]

            result = dict()
            result['id'] = data_sample['id']
            result['img_id'] = data_sample['img_id']
            result['gt_keypoints'] = data_sample['gt_instances']['keypoints'][
                0]
            result['keypoints'] = keypoints[0]
            result['keypoint_scores'] = keypoint_scores
            result['keypoints_visible'] = keypoints_visible[0]
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
        if self.result_dir is None:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')
        else:
            res_file = osp.join(self.result_dir, 'result_keypoints.json')
        self.logger.info(f'result file path is {res_file}')
        kpt_result = list()
        # from nreal_data_tool.metric.filter_metric import FilterMetric
        # kpt_list = []
        for result in results:
            image_id = result['img_id']
            keypoints = result['keypoints']
            # kpt_list.append(result['keypoints'])
            item = KeypointEvaluationItem(
                image_id=image_id,
                score=float(np.mean(result['keypoint_scores'][0])),
                area=float(result['area']),
                keypoints=keypoints.tolist(),
                gt_keypoints=result['gt_keypoints'].tolist(),
                keypoint_visible=np.abs(result['keypoints_visible']).tolist(),
                meta=result['meta'],
                keypoint_scores=result['keypoint_scores'][0].tolist())
            kpt_result.append(item.to_dict())
        # kpts = np.concatenate(kpt_list, axis=0)
        # filter_result = FilterMetric()(kpts, kpts)
        # print(filter_result)
        # print(filter_result['filtered_acc_noise'].mean())
        with open(res_file, 'w') as f:
            json.dump(kpt_result, f, sort_keys=True, indent=4)
        return self.metric(res_file)
