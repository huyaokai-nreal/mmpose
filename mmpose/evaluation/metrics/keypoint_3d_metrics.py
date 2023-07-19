# Copyright (c) OpenMMLab. All rights reserved.
import json
import tempfile
from collections import defaultdict
from os import path as osp
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from nreal_data_tool.metric import MPJPEMetric
from nreal_data_tool.schema import KeypointEvaluationItem

from mmpose.registry import METRICS
from ..functional import keypoint_mpjpe
from ..functional.keypoint_eval import keypoint_epe


@METRICS.register_module()
class MPJPE(BaseMetric):
    """MPJPE evaluation metric.

    Calculate the mean per-joint position error (MPJPE) of keypoints.

    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)

    Args:
        mode (str): Method to align the prediction with the
            ground truth. Supported options are:

                - ``'mpjpe'``: no alignment will be applied
                - ``'p-mpjpe'``: align in the least-square sense in scale
                - ``'n-mpjpe'``: align in the least-square sense in
                    scale, rotation, and translation.

        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.
    """

    ALIGNMENT = {'mpjpe': 'none', 'p-mpjpe': 'procrustes', 'n-mpjpe': 'scale'}

    def __init__(self,
                 mode: str = 'mpjpe',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        allowed_modes = self.ALIGNMENT.keys()
        if mode not in allowed_modes:
            raise KeyError("`mode` should be 'mpjpe', 'p-mpjpe', or "
                           f"'n-mpjpe', but got '{mode}'.")

        self.mode = mode

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
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['lifting_target']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['lifting_target_visible'].astype(bool).reshape(1, -1)
            # instance action
            img_path = data_sample['target_img_path']
            _, rest = osp.basename(img_path).split('_', 1)
            action, _ = rest.split('.', 1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
                'tag': action
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are the corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        if pred_coords.ndim == 4 and pred_coords.shape[1] == 1:
            pred_coords = np.squeeze(pred_coords, axis=1)
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])
        # action_category_indices: Dict[List[int]]
        action_category_indices = defaultdict(list)
        for idx, result in enumerate(results):
            action_category = result['action'].split('_')[0]
            action_category_indices[action_category].append(idx)

        error_name = self.mode.upper()

        logger.info(f'Evaluating {self.mode.upper()}...')
        metrics = dict()

        metrics[error_name] = keypoint_mpjpe(pred_coords, gt_coords, mask,
                                             self.ALIGNMENT[self.mode])

        for action_category, indices in action_category_indices.items():
            metrics[f'{error_name}_{action_category}'] = keypoint_mpjpe(
                pred_coords[indices], gt_coords[indices], mask[indices])

        return metrics


@METRICS.register_module()
class MPJPEV2(MPJPE):
    default_prefix: Optional[str] = ''

    def __init__(self,
                 gesture_list: list=[],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 mode: str = 'mpjpe',
                 result_dir=None,
                 with_tag=False) -> None:
        super().__init__(mode, collect_device, prefix)
        self.result_dir = result_dir
        self.logger = MMLogger.get_current_instance()
        self.metric = MPJPEMetric(gesture_list, mode=mode, with_tag=with_tag)

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
            # assert keypoint_scores.shape == keypoints3d.shape[:2]
            result = KeypointEvaluationItem(image_id=data_sample['img_id'])
            result.gt_keypoints3d = gt['keypoints3d'][0].tolist()
            # result.keypoints3d = keypoints3d[0].tolist()
            result.keypoints3d = keypoints3d.tolist()
            result.keypoint_visible = gt['keypoints_visible'].reshape(
                (-1)).tolist()
            result.score = float(np.mean(keypoint_scores))
            result.meta['tag'] = 'all_tag'
            result.meta['gesture'] = data_sample['meta']['gesture']
            # get area information
            if 'bbox_scales' in data_sample['gt_instances']:
                result.area = float(
                    np.prod(
                        data_sample['gt_instances']['bbox_scales'], axis=1))
            # add converted result to the results list
            self.results.append(result.to_dict())

    def compute_metrics(self, results: list) -> Dict[str, float]:
        if self.result_dir is None:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')
        else:
            res_file = osp.join(self.result_dir, 'result_keypoints.json')
        self.logger.info(f'result file path is {res_file}')
        with open(res_file, 'w') as f:
            json.dump(self.results, f, sort_keys=True, indent=4)
        return self.metric(res_file)


@METRICS.register_module()
class MPJPEMetricLifting(MPJPEV2):
    default_prefix: Optional[str] = ''

    def __init__(self,
                 gesture_list: List[str] = [],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 mode: str = 'mpjpe',
                 result_dir=None,
                 root_kpt_id=0) -> None:
        super().__init__(gesture_list, collect_device, prefix, mode,
                         result_dir, root_kpt_id)

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

            # from IPython import embed; embed()

            result = dict()

            result['gt_keypoints3d'] = gt['keypoints3d'][0]
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
            pred_pt_cam = [result['keypoints3d']]
            # from IPython import embed; embed()
            # pred_pt_cam = pred_pt_cam - pred_pt_cam[:, self.root_kpt_id]
            gt_pt_cam = [result['gt_keypoints3d']]
            # gt_pt_cam = gt_pt_cam - gt_pt_cam[:, self.root_kpt_id]
            dt_list.append(pred_pt_cam)
            gt_list.append(gt_pt_cam)
            mask_list.append(result['mask'])
        gt = np.concatenate(gt_list, axis=0)
        dt = np.concatenate(dt_list, axis=0)
        mask = np.concatenate(mask_list, axis=0)
        # from IPython import embed; embed()
        mpjpe_all = keypoint_epe(dt, gt, mask)
        mpjpe_x = keypoint_epe(dt[..., :1], gt[..., :1], mask)
        mpjpe_y = keypoint_epe(dt[..., 1:2], gt[..., 1:2], mask)
        mpjpe_z = keypoint_epe(dt[..., 2:], gt[..., 2:], mask)

        return dict(
            mpjpe_all=mpjpe_all * 1000,
            mpjpe_x=mpjpe_x * 1000,
            mpjpe_y=mpjpe_y * 1000,
            mpjpe_z=mpjpe_z * 1000)
