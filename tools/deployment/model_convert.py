# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch


def extrace_from_distilled_model(src_2d_path, save_student_model_path):
    distill_model = torch.load(src_2d_path)
    state_dict = {
        k.replace('student.', ''): v
        for k, v in distill_model['state_dict'].items()
        if not k.startswith('teacher')
    }
    distill_model['state_dict'] = state_dict
    torch.save(distill_model, save_student_model_path)


def merge_model(src_path, liftnet_path, save_model_path):
    src_model = torch.load(src_path)
    merge_model = torch.load(liftnet_path)
    state_dict = {
        k: v
        for k, v in merge_model['state_dict'].items()
        if k.startswith('kpt3d_lift')
    }
    src_model['state_dict'].update(state_dict)
    torch.save(src_model, save_model_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMPose models')
    parser.add_argument('src2d', help='model source path')
    parser.add_argument('save_path', help='model save path')
    parser.add_argument(
        '--liftnet_path',
        default='',
        help='to be merged liftnet model source path')
    parser.add_argument(
        '-d',
        '--distill',
        action='store_true',
        default=False,
        help='extrace student model from distilled model')
    parser.add_argument(
        '-m',
        '--merge',
        action='store_true',
        default=False,
        help='merge 2d and 3d model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.distill:
        extrace_from_distilled_model(args.src2d, args.save_path)

    if args.merge and args.liftnet_path:
        merge_model(args.src2d, args.liftnet_path, args.save_path)
