from typing import Optional, Sequence

from mmengine.logging import MMLogger
import numpy as np
from mmengine.evaluator import BaseMetric
from mmpose.registry import METRICS
from ..functional.keypoint_eval import multilabel_classification_accuracy


@METRICS.register_module()
class AttrClsAccuracy(BaseMetric):
    default_prefix: Optional[str] = ''

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 score_th: float = 0.5) -> None:
        super().__init__(collect_device, prefix)
        self.logger = MMLogger.get_current_instance()
        self.score_th = score_th

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_instances' not in data_sample:
                raise ValueError(
                    '`pred_instances` are required to process the '
                    f'predictions results in {self.__class__.__name__}. ')
            result = dict()
            result['id'] = data_sample['id']
            result['img_id'] = data_sample['img_id']
            result['pred_attr'] = data_sample['pred_instances']['attr']
            result['gt_attr'] = data_sample['gt_instance_labels'][
                'attr_labels']
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        gts = [item['gt_attr'] for item in results]
        preds = [item['pred_attr'] for item in results]
        gts = np.concatenate(gts, axis=0)
        preds = np.concatenate(preds, axis=0)
        visible_gt = gts[:, :21].reshape((-1, 1))
        visible_pred = preds[:, :21].reshape((-1, 1))
        visiable_acc = multilabel_classification_accuracy(
            visible_pred, visible_gt, np.ones((visible_gt.shape[0])),
            self.score_th)
        hand_cls_acc = multilabel_classification_accuracy(
            preds[:, -2:], gts[:, -2:], np.ones((preds.shape[0])),
            self.score_th)
        metrics = dict(visiable_acc=visiable_acc, hand_cls_acc=hand_cls_acc)
        return metrics
