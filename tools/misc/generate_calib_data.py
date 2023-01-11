# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import os
import os.path as osp
import random
from mmengine import Config, DictAction, mkdir_or_exist
from mmengine.registry import build_from_cfg
from mmpose.registry import DATASETS
from mmpose.utils import register_all_modules
import numpy as np
from tqdm import tqdm


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
        '--nr-sample', default=4096, type=int, help='number of data samples')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--type',
        default='png',
        type=str,
        help='save result type, png for image or npy for numpy ndarray')
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
    mkdir_or_exist(args.output_dir)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_all_modules()
    cfg[f'{args.phase}_dataloader'].dataset.pipeline[
        -1].pack_transformed = True

    dataset = build_from_cfg(cfg[f'{args.phase}_dataloader'].dataset, DATASETS)
    sample_index = random.sample(range(len(dataset)), args.nr_sample)
    for i, id in enumerate(tqdm(sample_index)):
        item = dataset[id]
        img = item['inputs'].unsqueeze(0).numpy()
        if args.type == 'npy':
            np.save(os.path.join(args.output_dir, f'{i}.npy'), img)
        if args.type == 'png':
            img = img[0].transpose((1, 2, 0))
            cv2.imwrite(os.path.join(args.output_dir, f'{i}.png'), img)


if __name__ == '__main__':
    main()
