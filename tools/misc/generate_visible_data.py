# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pickle
import random

import mmcv
import numpy as np
from loguru import logger
from mmengine import Config, DictAction
from mmengine.registry import build_from_cfg
from tqdm import tqdm

from mmpose.registry import DATASETS, VISUALIZERS
from mmpose.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate data for calibration')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output',
        default='test.pkl',
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument(
        '--nr-sample',
        default=4096,
        type=int,
        help='number of data samples, -1 means get all data in order')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
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
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_all_modules()
    cfg[f'{args.phase}_dataloader'].dataset.pipeline[
        -1].pack_transformed = True

    dataset = build_from_cfg(cfg[f'{args.phase}_dataloader'].dataset, DATASETS)
    if args.nr_sample > 0:
        sample_index = random.sample(range(len(dataset)), args.nr_sample)
    else:
        sample_index = range(len(dataset))
    data_samples = []
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.set_dataset_meta(dataset.metainfo)
    for i, id in enumerate(tqdm(sample_index)):
        item = dataset[id]
        if isinstance(item['data_samples'], list):
            img = item['inputs'][0].permute(1, 2, 0).numpy()
            data_sample = item['data_samples'][0]
        else:
            img = item['inputs'].permute(1, 2, 0).numpy()
            data_sample = item['data_samples']

        img_path = data_sample.img_path
        keypoints = item[
            'data_samples'].gt_instances.transformed_keypoints.copy()
        visible = item['data_samples'].gt_instances.keypoints_visible.copy()
        data_sample.gt_instances.keypoints_visible += 1
        img = mmcv.bgr2rgb(img)
        img = np.zeros_like(img)

        show_image = visualizer.add_datasample(
            osp.basename(img_path),
            img,
            data_sample,
            draw_pred=False,
            draw_bbox=False,
            draw_heatmap=False,
            draw_3d=False,
            show=False)
        norm_keypoints = keypoints / 256.0
        if norm_keypoints.max() > 1 or keypoints.min() < 0:
            print(keypoints)
            print(norm_keypoints)
            continue
        data_sample = np.concatenate(
            [norm_keypoints, np.expand_dims(visible, axis=-1)], axis=-1)
        data_samples.append(dict(anno=data_sample, image=show_image))
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'save {len(data_samples)} items to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(data_samples, f)


if __name__ == '__main__':
    main()
