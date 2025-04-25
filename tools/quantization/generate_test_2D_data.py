# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
from mmengine import Config, DictAction, mkdir_or_exist
from mmengine.registry import build_from_cfg
from tqdm import tqdm

from mmpose.registry import DATASETS
from mmpose.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate data for calibration')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default='calib_data',
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument(
        '--nr-sample',
        default=4096,
        type=int,
        help='number of data samples, -1 means get all data in order')
    parser.add_argument(
        '--data_type',
        default='key',
        type=str,
        choices=['hold_cls', 'keypoint'])
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--type',
        default='raw',
        type=str,
        help='save result type, png for image or npy for numpy ndarray,'
        'raw for snpe in [N, H, W, C]')
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
    parser.add_argument(
        '--quantize_type',
        default='snpe',
        type=str,
        help='choose data channel type, (n,h,w,c) for snpe, (n,c,h,w) for mnn')
    args = parser.parse_args()
    return args


def generate_dup_file_name(out_file):
    """Automatically rename out_file when duplicated file exists.

    This case occurs when there is multiple instances on one image.
    """
    if out_file and osp.exists(out_file):
        img_name, postfix = osp.basename(out_file).rsplit('.', 1)
        exist_files = tuple(
            filter(lambda f: f.startswith(img_name),
                   os.listdir(osp.dirname(out_file))))
        if len(exist_files) > 0:
            img_path = f'{img_name}({len(exist_files)}).{postfix}'
            out_file = osp.join(osp.dirname(out_file), img_path)
    return out_file


def main():
    args = parse_args()
    # 需同时为mnn和snpe生成测试数据
    # mkdir_or_exist(os.path.join(args.output_dir, "NHWC"))
    # mkdir_or_exist(os.path.join(args.output_dir, "NCHW"))

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_all_modules()
    cfg[f'{args.phase}_dataloader'].dataset.pipeline[
        -1].pack_transformed = True
    
    # data_type = "keypoint"  # keypoint, hold_cls
    if args.data_type == "keypoint":
        dataset_list = [
            # # 2d, warpAffine preprocess
            # # flora static
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations/hand_test_flora_static_benchmark_230627_10k_lmdb.json',
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations/hand_test_flora_static_benchmark_230703_10k_lmdb.json',  # flora test
            
            # flora_dynamic
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations/hand_test_dynamic_keypoint_230907_20k__1__binocular__lmdb.json',
            
            # flora decoration
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations/hand_test_flora_keypoint_decoration_1_231208_1k__1__binocular__lmdb.json',

            # wrist_occlusion
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations/hand_test_flora_wrist_occlusion_240417_2k__1__binocular__lmdb.json'
            
        ]
    else:
        dataset_list = [
            
            "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_hand_held_occ_250219__1__binocular__lmdb.json",
        ]
    random.seed(0)
    
    for idx, dataset_path in enumerate(dataset_list):
        cfg[f'{args.phase}_dataloader']["dataset"]["data_file_list"] = [dataset_path]    
        dataset = build_from_cfg(cfg[f'{args.phase}_dataloader'].dataset, DATASETS)
        if args.nr_sample > 0:
            sample_index = random.sample(range(len(dataset)), args.nr_sample)
        else:
            sample_index = range(len(dataset))
        
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        for i, id in enumerate(tqdm(sample_index)):
            item = dataset[id]
            inputs = item['inputs']  # 1x128x128
            
            img_id = str(item['data_samples'].metainfo['img_id']).zfill(8)
            img = inputs.unsqueeze(0).numpy().astype(np.float32)
            # img = (img - 114.495) / 57.63
            if args.type == 'npy':
                np.save(os.path.join(args.output_dir, f'{img_id}.npy'), img)
            if args.type == 'raw':
                # for mnn                
                img.tofile(os.path.join(args.output_dir, "NCHW", f'{dataset_name}_{img_id}.raw'))
                # for snpe
                img = img.transpose((0, 2, 3, 1))  # (N,H,W,C)
                img.tofile(os.path.join(args.output_dir, "NHWC", f'{dataset_name}_{img_id}.raw'))
            if args.type == 'png':
                img = img[0].transpose((1, 2, 0))
                cv2.imwrite(
                    os.path.join(args.output_dir, f'{dataset_name}_{img_id}.png'), img)

if __name__ == '__main__':
    main()
