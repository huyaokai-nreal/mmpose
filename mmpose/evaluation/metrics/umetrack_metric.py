# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

# from mmpose.codecs.utils import pixel_to_camera
from mmpose.registry import METRICS

# from ..functional import keypoint_epe


@METRICS.register_module()
class UmetrackMetric(BaseMetric):

    METRICS = {'MPJPE', 'MRRPE', 'HandednessAcc'}

    def __init__(self,
                 modes: List[str] = ['MPJPE', 'MRRPE', 'HandednessAcc'],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        for mode in modes:
            if mode not in self.METRICS:
                raise ValueError("`mode` should be 'MPJPE', 'MRRPE', or "
                                 f"'HandednessAcc', but got '{mode}'.")

        self.modes = modes

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            unknown_pred_coords = data_sample['pred_instances'][
                'unknown_keypoints']
            unknown_pred_coords_cam = unknown_pred_coords.clone()  # 16,21,3

            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints_cam']

            keypoints_cam = gt_coords.clone()
            result = {}

            if 'MPJPE' in self.modes:
                result[
                    'unknown_pred_coords'] = unknown_pred_coords_cam  # 16 21 3
                result['gt_coords'] = keypoints_cam
            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        metrics = dict()

        logger.info(f'Evaluating {self.__class__.__name__}...')

        if 'MPJPE' in self.modes:
            # pred_coords: [N, K, D]
            unknown_pred_coords = np.concatenate([
                result['unknown_pred_coords'] for result in results
            ])  # bc*16 21 3

            # gt_coords: [N, K, D]
            gt_coords = np.concatenate(
                [result['gt_coords'] for result in results])
            # 加载进来所有样本数
            unknown_batchsize = unknown_pred_coords.shape[0] // 16
            unknown_re_pred_coords = unknown_pred_coords.reshape(
                unknown_batchsize, 16, 21, 3)

            gt_batchsize = gt_coords.shape[0] // 16
            re_gt_coords = gt_coords.reshape(gt_batchsize, 16, 21, 3)
            unknown_keypoints_diff = (re_gt_coords - unknown_re_pred_coords
                                      )  # 51(所有加载数据) 16 21 3
            unknown_keypoint_errors = np.linalg.norm(
                unknown_keypoints_diff,
                axis=-1).mean(axis=(1, 2))  # 51 16 21 -> 51
            unknown_keypoint_errors_mm = unknown_keypoint_errors * 1000
            metrics['MPJPE_all'] = unknown_keypoint_errors_mm.mean()
        return metrics
