# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import tempfile
from argparse import ArgumentParser
import json
import glob
from tqdm import tqdm

import mmcv
import mmengine
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import register_all_modules as register_mmpose_modules

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.utils import register_all_modules as register_mmdet_modules
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def visualize_img(args, img_path, detector, pose_estimator, visualizer,
                  show_interval):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    register_mmdet_modules()
    detect_result = inference_detector(detector, img_path)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr)][:, :4]

    # predict keypoints
    register_mmpose_modules()
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    img = mmcv.imread(img_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')

    out_file = None
    if args.output_root:
        out_file = f'{args.output_root}/{os.path.basename(img_path)}'

    visualizer.add_datasample(
        'result',
        img,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=args.draw_heatmap,
        draw_bbox=False,
        show=args.show,
        wait_time=show_interval,
        out_file=out_file,
        kpt_score_thr=args.kpt_thr)

def pipline_inference(args, img_path, detector, pose_estimator):
    """predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    register_mmdet_modules()
    detect_result = inference_detector(detector, img_path)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr)][:, :4]
    
    if len(bboxes) == 0:
        print("Warning: Empty bboxes", img_path, pred_instance.scores)
    else:
        # predict keypoints
        register_mmpose_modules()
        pose_results = inference_topdown(pose_estimator, img_path, bboxes)
        data_samples = merge_data_samples(pose_results)
        
        # save result json
        save_result_json(args, img_path, data_samples)
    
def save_result_json(args, img_path, data_samples, suffix='.png'):
    file_name = img_path.split("/")[-1].replace(suffix, '.json')
    with open(os.path.join(args.output_root, file_name), "w") as f:
        kp_temp = np.concatenate((data_samples.pred_instances['keypoints'][0], data_samples.pred_instances['keypoint_scores'][0][:, np.newaxis]), axis=1)
        json.dump({'keypoints':kp_temp.tolist()}, f)

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('--det_config', type=str, default='demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py', help='Config file for detection')
    parser.add_argument('--det_checkpoint', type=str, default='https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth', help='Checkpoint file for detection')
    parser.add_argument('--pose_config', type=str, default='configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256.py', help='Config file for pose')
    parser.add_argument('--pose_checkpoint', type=str, default='https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='/data/AI_DATA/data_hand/original_data/hand_keypoint/simulation_data/test1.4/*.png', help='Image path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='/data/AI_DATA/data_hand/original_data/hand_keypoint/simulation_data/test1.4_pre_anno/',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.01,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Whether to draw output heatmap')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)

    # build detector
    register_mmdet_modules()
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)

    # build pose estimator
    register_mmpose_modules()
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    input_lists = sorted(glob.glob(args.input))
    for img_path in tqdm(input_lists):
        pipline_inference(args, img_path, detector, pose_estimator)




if __name__ == '__main__':
    main()
