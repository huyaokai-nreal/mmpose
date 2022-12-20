# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import tempfile
from typing import Optional, Sequence

from mmengine.logging import MMLogger
import numpy as np
from mmengine.evaluator import BaseMetric
from xtcocotools.coco import COCO

from mmpose.registry import METRICS


@METRICS.register_module()
class NrealKeypointAP(BaseMetric):

    default_prefix: Optional[str] = ''

    def __init__(self,
                 ann_file,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 flip_left_to_right=True,
                 result_dir=None) -> None:
        self.flip_left_to_right = flip_left_to_right
        super().__init__(collect_device, prefix)
        self.coco = COCO(ann_file)
        self.result_dir = result_dir

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_instances' not in data_sample:
                raise ValueError(
                    '`pred_instances` are required to process the '
                    f'predictions results in {self.__class__.__name__}. ')

            # keypoints.shape: [N, K, 2],
            # N: number of instances, K: number of keypoints
            # for topdown-style output, N is usually 1, while for
            # bottomup-style output, N is the number of instances in the image
            keypoints = data_sample['pred_instances']['keypoints']
            # [N, K], the scores for all keypoints of all instances
            keypoint_scores = data_sample['pred_instances']['keypoint_scores']
            assert keypoint_scores.shape == keypoints.shape[:2]

            result = dict()
            result['id'] = data_sample['id']
            result['img_id'] = data_sample['img_id']
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
            ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)
            keypoints = result['keypoints'][0]
            if self.flip_left_to_right:
                cat = objs[0]['category_id']
                if cat == 1:
                    image_width = self.coco.imgs[image_id]['width']
                    keypoints[:, 0] = image_width - 1 - keypoints[:, 0]
            item = dict(
                image_id=image_id,
                keypoints=keypoints.tolist(),
                gt_keypoints=objs[0]['keypoints'],
                keypint_scores=result['keypoint_scores'][0].tolist())
            kpt_result.append(item)
        with open(res_file, 'w') as f:
            json.dump(kpt_result, f, sort_keys=True, indent=4)
        return self._do_evaluate(res_file)

    def _kps_to_bbox(self, kps):
        min_x = kps[:, 0].min()
        min_y = kps[:, 1].min()
        max_x = kps[:, 0].max()
        max_y = kps[:, 1].max()
        cx = (min_x + max_x) * 0.5
        cy = (min_y + max_y) * 0.5
        min_x = (min_x - cx) * 1.5 + cx
        min_y = (min_y - cy) * 1.4 + cy
        max_x = (max_x - cx) * 1.5 + cx
        max_y = (max_y - cy) * 1.4 + cy
        bbox = np.array([min_x, min_y, max_x - min_x, max_y - min_y],
                        dtype=np.float32)
        return bbox

    def _calculate_oks(self, dt, gt):
        sigmas = np.array([
            .87, .62, .35, .25, .25, .39, .25, .25, .25, .39, .25, .25, .25,
            .25, .25, .25, .25, .39, .25, .25, .25
        ]) / 10.0
        vars = (sigmas * 2)**2
        # compute oks between each detection and ground truth object
        xg = gt[:, 0]
        yg = gt[:, 1]
        bb = self._kps_to_bbox(gt)
        area = bb[2] * bb[3]
        xd = dt[:, 0]
        yd = dt[:, 1]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / (area + np.spacing(1)) / 2
        ious = np.sum(np.exp(-e)) / e.shape[0]

        return ious

    def _do_evaluate(self, res_file):
        pr_thresh = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        pr = dict()
        for prt in pr_thresh:
            pr[str(prt)] = 0.0
        sumpr = 0.0
        with open(res_file) as f:
            dts = json.load(f)
        for dt in dts:
            dt_kpt = np.array(dt['keypoints'])
            gt_kpt = np.array(dt['gt_keypoints'])
            oks = self._calculate_oks(dt_kpt, gt_kpt)
            for thr in pr_thresh:
                if oks >= thr:
                    pr[str(thr)] += 1.0 / len(dts)
                    sumpr += 1.0 / len(dts) / 10.0
        pr['mAP'] = sumpr
        return pr
