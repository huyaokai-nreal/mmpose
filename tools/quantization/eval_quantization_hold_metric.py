import argparse
import os
import os.path as osp
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import onnx
import onnxruntime
import torch
import cv2
from itertools import zip_longest
import copy
from tqdm import tqdm

import mmcv
import mmengine
import numpy as np
from mmengine import Config, DictAction, mkdir_or_exist
from mmengine.registry import build_from_cfg, init_default_scope
from mmengine.structures import InstanceData
from nreal_data_tool import LmdbClient
from nreal_data_tool.utils.camera import NoDistortion, PinholePlaneCameraModel
from mmpose.models.utils.pose_solver import (get_kpt_depth,
                                             get_kpt_depth_binocular,
                                             get_kpt_depth_binocular63,
                                             get_root_depth)
from mmpose.structures.bbox import bbox_cs2xyxy

from typing import Optional
from mmengine.evaluator import Evaluator
from mmpose.registry import DATASETS, VISUALIZERS
from mmpose.structures import PoseDataSample
from mmpose.models.utils.siamcc_to_kpt import SimCCToKeypoint3D
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from mmpose.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--interval', type=int, default=1, help='interval to show the result')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def add_pred_to_datasample(
        data_samples: SampleList,
        results_list: InstanceList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        
        return data_samples

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_all_modules()
    cfg[f'{args.phase}_dataloader'].dataset.pipeline[
        -1].pack_transformed = True
    
    dataset_list = [
        # flora static
        "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_hand_held_occ_250219__1__binocular__lmdb.json",

    ]
    result_dir = "/data/byzhou/keypoint2d/quantization/test_results/1482_Inference_20250520_153353_rtmtiny_res26sw_kpt_0520_freeze_hold_cls_sigmoid_4d9575_2d26ee67_CPU_FP32"
    result_files = ["_".join(file.split("_")[:-2]) for file in list(os.listdir(result_dir))]

    evaluator = Evaluator([dict(type='SimpleAccuracy')])
    for idx, dataset_path in enumerate(dataset_list):
        
        cfg[f'{args.phase}_dataloader']["dataset"]["data_file_list"] = [dataset_path]    

        dataset = build_from_cfg(cfg[f'{args.phase}_dataloader'].dataset, DATASETS)
    
        print("length of dataset ", len(dataset))
        
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]        
        
        valid_num = 0
        idx = 0
        
        while idx < len(dataset):
            item = dataset[idx]
            idx += 1

            img_id = str(item['data_samples'].metainfo['img_id']).zfill(8)
            anno_id = str(item['data_samples'].id).zfill(8)
            if f'{dataset_name}_{img_id}_{anno_id}' not in result_files:
                continue
            
            valid_num += 1
            hold_cls = np.fromfile(os.path.join(result_dir, f'{dataset_name}_{img_id}_{anno_id}_hold_cls.raw'), dtype=np.float32)
            hold_cls = hold_cls.reshape(2, 1)
            # cls_scores = 1 / (1 + np.exp(-hold_cls))
            cls_scores = hold_cls
            # print("hold_cls_scores")
            # print(cls_scores[0])
            
            data_sample = item['data_samples']
            preds = [
                InstanceData(hold_obj=cls_score[np.newaxis]) for cls_score in cls_scores]
            data_sample.pred_instances = preds[0]
            # samplelist_boxtype2tensor(data_sample)
            
            data_samples=add_pred_to_datasample([data_sample], [preds[0]])
            
            evaluator.process(data_samples=data_samples, data_batch=None)

            # vis_img = item['inputs'].permute(1, 2, 0).numpy()
            # print(item['data_samples'].keys())
            # keypoints = item['data_samples'].gt_instances.transformed_keypoints[0]
        
        metrics = evaluator.evaluate(valid_num)
        print(metrics)

        
if __name__ == '__main__':
    main()
