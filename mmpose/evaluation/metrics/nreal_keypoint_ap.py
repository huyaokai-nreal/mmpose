# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import tempfile
from typing import Optional, Sequence

from mmengine.logging import MMLogger
import numpy as np
from mmengine.evaluator import BaseMetric
from nreal_data_tool.schema import KeypointEvaluationItem
from nreal_data_tool.metric import KeypointOKSMetric
from mmpose.registry import METRICS


@METRICS.register_module()
class NrealKeypointAP(BaseMetric):

    default_prefix: Optional[str] = ''

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 result_dir=None) -> None:
        super().__init__(collect_device, prefix)
        self.result_dir = result_dir
        self.metric = KeypointOKSMetric()

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

            result = dict()
            result['id'] = data_sample['id']
            result['img_id'] = data_sample['img_id']
            result['gt_keypoints'] = data_sample['gt_instances']['keypoints']
            result['keypoints'] = keypoints
            result['keypoint_scores'] = keypoint_scores
            result['bbox_scores'] = data_sample['gt_instances']['bbox_scores']

            # get area information
            if 'bbox_scales' in data_sample['gt_instances']:
                result['areas'] = np.prod(
                    data_sample['gt_instances']['bbox_scales'], axis=1)
            # add converted result to the results list
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        if self.result_dir is None:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')
        else:
            res_file = osp.join(self.result_dir, 'result_keypoints.json')
        logger = MMLogger.get_current_instance()
        logger.info(f'result file path is {res_file}')
        kpt_result = list()
        for result in results:
            image_id = result['img_id']
            keypoints = result['keypoints'][0]
            item = KeypointEvaluationItem(
                image_id=image_id,
                keypoints=keypoints.tolist(),
                gt_keypoints=result['gt_keypoints'][0].tolist(),
                keypoint_scores=result['keypoint_scores'][0].tolist())
            kpt_result.append(item.to_dict())
        with open(res_file, 'w') as f:
            json.dump(kpt_result, f, sort_keys=True, indent=4)
        return self.metric(res_file)
