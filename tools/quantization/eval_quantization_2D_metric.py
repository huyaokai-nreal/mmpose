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
        '--result_dir',
        type=str,
        help='path of quantization result')
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

def add_pred_to_datasample_monocular(
        batch_pred_instances: InstanceList,
        batch_pred_fields: Optional[PixelDataList],
        batch_data_samples: SampleList) -> SampleList:
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
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices =  None
        root_mode = 'gt'
        last_kpt3d = None
        
        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            bbox_centers = gt_instances.bbox_centers
            bbox_scales = gt_instances.bbox_scales
            # uv depth to camera coord pose
            ori_cam = data_sample.meta.get(
                'ori_camera',
                PinholePlaneCameraModel(
                    c=(240, 320),
                    f=(200, 200),
                    camera_to_world_xf=np.eye(4),
                    distort_coeffs=[]))
            root_depth = data_sample.meta.get('root_depth', 0.5)
            gt_hand_scale = data_sample.meta.get('hand_scale', 1.0)
            if 'virtual_camera' in data_sample.meta:
                virtual_cam = data_sample.meta['virtual_camera']
                gt_keypoints3d = gt_instances.keypoints3d[0]
                if data_sample.meta.get('norm_depth', False):
                    pred_instances.keypoints[..., 2] *= gt_hand_scale
                virtual_keypoints = pred_instances.keypoints[0].copy()
                if root_mode == 'optimize':
                    root_depth, hand_scale = get_root_depth(
                        virtual_keypoints, virtual_cam,
                        data_sample.meta['template_bones'] * gt_hand_scale,
                        pred_instances.keypoint_scores, gt_keypoints3d, False,
                        last_kpt3d)

                virtual_keypoints[..., 2] += root_depth
                virtual_keypoints3d = virtual_cam.window_to_eye(
                    virtual_keypoints)
                if data_sample.meta['flipped']:
                    virtual_keypoints3d[..., 0] *= -1
                world_keypoints3d = virtual_cam.eye_to_world(
                    virtual_keypoints3d)
                last_kpt3d = world_keypoints3d
                # vir_camera_window->vir_camera_eye->ori_camera_eye->ori_camera_windows
                kpt_norm_eye = virtual_cam.window_to_eye(
                    virtual_keypoints[:, :2])
                if data_sample.meta['flipped']:
                    kpt_norm_eye[..., 0] *= -1
                kpt_norm_world = virtual_cam.eye_to_world(kpt_norm_eye)
                kpt2d_ori = ori_cam.eye_to_window(kpt_norm_world)
                pred_instances.keypoints[0][..., :2] = kpt2d_ori
                pred_instances.keypoints3d = pred_instances.keypoints.copy()
                pred_instances.keypoints3d[0] = world_keypoints3d
            else:
                input_size = data_sample.metainfo['input_size']
                if data_sample.meta['flipped']:
                    pred_instances.keypoints[
                        ...,
                        0] = input_size[0] - 1 - pred_instances.keypoints[...,
                                                                          0]
                global_keypoints = copy.deepcopy(pred_instances.keypoints)
                global_keypoints[..., :2] = global_keypoints[
                    ..., :
                    2] / input_size * bbox_scales + bbox_centers - 0.5 * bbox_scales  # noqa
                # for 2d keypoint evaluation
                pred_instances.keypoints[..., :2] = pred_instances.keypoints[
                    ..., :
                    2] / input_size * bbox_scales + bbox_centers - 0.5 * bbox_scales  # noqa
                if root_mode == 'optimize':
                    kpt = pred_instances.keypoints[0].copy()
                    root_depth, hand_scale = get_root_depth(
                        kpt, ori_cam, data_sample.meta['template_bones'],
                        pred_instances.keypoint_scores)
                    if data_sample.meta.get('norm_depth', False):
                        global_keypoints[..., 2] *= hand_scale

                global_keypoints[..., 2] += root_depth
                ori_keypoints3d = ori_cam.window_to_eye(global_keypoints[0])
                pred_instances.keypoints3d = global_keypoints.copy()
                pred_instances.keypoints3d[0] = ori_keypoints3d
            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = bbox_cs2xyxy(bbox_centers, bbox_scales)
            pred_instances.bbox_scores = gt_instances.bbox_scores
            if data_sample.meta.get('norm_depth', False):
                gt_instances.keypoints[..., -1] *= gt_hand_scale
            data_sample.pred_instances = pred_instances
            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_all_modules()
    cfg[f'{args.phase}_dataloader'].dataset.pipeline[
        -1].pack_transformed = True
    
    dataset_list = [
        
        # 2d, warpAffine preprocess
        # flora static
        '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations/hand_test_flora_static_benchmark_230627_10k_lmdb.json',
        '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations/hand_test_flora_static_benchmark_230703_10k_lmdb.json',  # flora test
        
        # flora_dynamic
        '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations/hand_test_dynamic_keypoint_230907_20k__1__binocular__lmdb.json',
        
        # flora decoration
        '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations/hand_test_flora_keypoint_decoration_1_231208_1k__1__binocular__lmdb.json',

        # wrist_occlusion
        '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations/hand_test_flora_wrist_occlusion_240417_2k__1__binocular__lmdb.json'
                
         # 3d, PCL preprocess
        # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301.json',
        # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301.json',
        # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_080910__pinch__normal__left__1111__0019__undistort_tar__Flora301.json',
        # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230831_082659__pinch__bright__right__1111__0020__undistort_tar__Flora302.json',
        # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230831_060954__all__normal__right__1111__0002__undistort_tar__Flora301.json',

    ]
    input_size = cfg['codec2d']['input_size'][0]
    
    result_files = ["_".join(file.split("_")[:-2]) for file in list(os.listdir(args.result_dir))]
    
    ipr_module = SimCCToKeypoint3D(input_size*2, input_size*2, input_size*2, map_type='softmax')
        
    evaluator = Evaluator([dict(type='EPE'), dict(type='NrealKeypointAP', with_tag=True)])
    valid_num = 0
    for idx, dataset_path in enumerate(dataset_list):
        
        cfg[f'{args.phase}_dataloader']["dataset"]["data_file_list"] = [dataset_path]    

        dataset = build_from_cfg(cfg[f'{args.phase}_dataloader'].dataset, DATASETS)
    
        print("length of dataset ", len(dataset))
        
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]        
    
        idx = 0
        
        while idx < len(dataset):
            item = dataset[idx]
            idx += 1

            img_id = str(item['data_samples'].metainfo['img_id']).zfill(8)
            anno_id = str(item['data_samples'].id).zfill(8)            
            if f'{dataset_name}_{img_id}_{anno_id}' not in result_files:
                continue
            
            valid_num += 1
            feat_x = np.fromfile(os.path.join(args.result_dir, f'{dataset_name}_{img_id}_{anno_id}_feat_x.raw'), dtype=np.float32)
            feat_y = np.fromfile(os.path.join(args.result_dir, f'{dataset_name}_{img_id}_{anno_id}_feat_y.raw'), dtype=np.float32)
            feat_z = np.fromfile(os.path.join(args.result_dir, f'{dataset_name}_{img_id}_{anno_id}_feat_z.raw'), dtype=np.float32)
            
            feat_x = feat_x.reshape(2, 21, input_size*2)
            feat_y = feat_y.reshape(2, 21, input_size*2)
            feat_z = feat_z.reshape(2, 21, input_size*2)
            
            pred_x, pred_y, pred_z = ipr_module(torch.from_numpy(feat_x), torch.from_numpy(feat_y), torch.from_numpy(feat_z))
            pred_x = pred_x.numpy()
            pred_y = pred_y.numpy()
            pred_z = pred_z.numpy()

            keypoints = np.concatenate([pred_x[0], pred_y[0], pred_z[0]], axis=1)
            keypoints = keypoints * np.array([input_size, input_size, 1])
            keypoints[..., 2] = (keypoints[..., 2] - 0.5) * 0.4
            data_sample = item['data_samples']

            preds = [
                InstanceData(keypoints=keypoints[None, ...], keypoint_scores=np.ones((1, 21)))]
            data_sample.pred_instances = preds[0]
            # samplelist_boxtype2tensor(data_sample)
            
            data_samples=add_pred_to_datasample_monocular(preds, None, [data_sample])

            evaluator.process(data_samples=data_samples, data_batch=None)

            # vis_img = item['inputs'].permute(1, 2, 0).numpy()
            # # keypoints = item['data_samples'].gt_instances.transformed_keypoints[0]
            
            # for point in keypoints_tmp:
            #     x, y = point[:2]
            #     print(point)
            #     cv2.circle(vis_img, (int(x), int(y)), 5, (255, 255, 0))

            # cv2.imwrite("./vis_key.jpg", vis_img)
            
            # break

    metrics = evaluator.evaluate(valid_num)
    print(metrics)

    

        
if __name__ == '__main__':
    main()
