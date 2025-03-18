from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmpose.registry import METRICS

import numpy as np


@METRICS.register_module()  # 将 Accuracy 类注册到 METRICS 注册器
class SimpleAccuracy(BaseMetric):
    """ Accuracy Evaluator

    Default prefix: ACC

    Metrics:
        - accuracy (float): classification accuracy
    """

    default_prefix = 'ACC'  # 设置 default_prefix

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        
        # 取出分类预测结果和类别标签
        for data_sample in data_samples:
            result = {
                'pred': data_sample['pred_instances']['hold_obj'],
                'gt': data_sample['gt_instance_labels']['hold_obj']
            }
            
            # 将当前 batch 的结果存进 self.results
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        # 汇总所有样本的分类预测结果和类别标签
        preds = np.concatenate([res['pred'] for res in results])
        gts = np.concatenate([res['gt'] for res in results])
        
        # 计算分类准确率
        thr_list = [0.5]
        # thr_list = [0.5, 0.55, 0.6]
        acc_list = []
        precision_list = []
        recall_list = []
        
        for thr in thr_list:
            
            preds = np.where(preds >= thr, 1, 0)
            
            total_num = preds.size
            correct_num = (preds == gts).sum()
            TP = (preds * gts).sum()
            TN = correct_num - TP
            FP = preds.sum() - TP
            FN = total_num - TP - TN - FP
            
            acc = correct_num * 1.0 / total_num
            precision = TP * 1.0 / (TP + FP)
            recall = TP * 1.0 / (TP + FN)

            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            # 计算TP, TN, FP, FN, 其中positive为遮挡
        
        # 返回评测指标结果
        return {'accuracy': np.mean(acc_list), 'precision': np.mean(precision_list), 'recall':  np.mean(recall_list)}