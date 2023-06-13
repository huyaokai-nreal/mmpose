# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
import argparse
import json
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from mmengine import mkdir_or_exist
from nreal_data_tool.metric import KeypointOKSMetric
from tqdm import tqdm

from mmpose.codecs.nreal_heatmap import NrealHeatmap
from mmpose.codecs.utils import get_simcc_maximum
from mmpose.models.utils.heatmap_to_kpt import HeatmapToKeypoint


def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate hand keypoint with DSP output')
    parser.add_argument(
        'template_json_path', help='template json file for hand evaluation')
    parser.add_argument('dsp_output_dir', help='dsp output path')
    parser.add_argument(
        '--output',
        '-o',
        default='',
        help='json save path for generated result')
    parser.add_argument(
        '--model-type',
        '-m',
        default='reg',
        help='result type reg, hm or siamcc')
    args = parser.parse_args()
    return args


onnx_path = '/mmpose/data/rsntiny_attr_post_rename.onnx'

import onnxruntime

try:
    onnx_model = onnxruntime.InferenceSession(onnx_path)
except:
    print('onnx path is not found')


def feat_to_kpt_attr(kpt_feat, global_feat):
    inputs = {'kpt_feat': kpt_feat, 'global_feat': global_feat}
    output = onnx_model.run(None, inputs)
    kpt = output[0]
    return kpt[0]


def reg_to_kpt(kpt_x, kpt_y):
    kpts = np.concatenate([kpt_x, kpt_y], axis=-1)
    return kpts


hm_codec = NrealHeatmap((128, 128), (32, 32), 3)


def hm_to_kpt(hm):
    return hm_codec.decode(hm)[0]


def simacc_to_kpt(kpt_x, kpt_y, output_size=256):
    kpt_x = kpt_x.reshape(1, 21, output_size)
    kpt_y = kpt_y.reshape(1, 21, output_size)
    keypoints, scores = get_simcc_maximum(kpt_x, kpt_y)
    keypoints = keypoints[0]
    keypoints /= float(output_size - 1)
    return keypoints


post_model = HeatmapToKeypoint()


def feat_to_kpt(feat):
    return post_model.forward(feat)


def main():
    args = parse_args()
    dsp_output_dir = args.dsp_output_dir
    template_json_path = args.template_json_path
    json_path = args.output
    if not json_path:
        tmp_folder = tempfile.TemporaryDirectory()
        res_file = os.path.join(tmp_folder.name, 'dsp_result_keypoints.json')
    else:
        output_dir = os.path.dirname(json_path)
        mkdir_or_exist(output_dir)
        res_file = json_path
    with open(template_json_path, 'r') as f:
        result_list = json.load(f)
    for i, result in enumerate(tqdm(result_list)):
        bbox_centers = np.array(result['meta']['bbox_centers'])
        bbox_scales = np.array(result['meta']['bbox_scales'])
        if args.model_type in ['reg', 'simcc']:
            kpt_x = np.fromfile(
                os.path.join(dsp_output_dir, f'{str(i).zfill(8)}_kpt_x.raw'),
                dtype=np.float32)[:, np.newaxis]
            kpt_y = np.fromfile(
                os.path.join(dsp_output_dir, f'{str(i).zfill(8)}_kpt_y.raw'),
                dtype=np.float32)[:, np.newaxis]
            if args.model_type == 'reg':
                kpts = reg_to_kpt(kpt_x, kpt_y)
            else:
                kpts = simacc_to_kpt(kpt_x, kpt_y)
        elif args.model_type == 'feat':
            feat = np.fromfile(
                os.path.join(dsp_output_dir, f'{str(i).zfill(8)}_feat.raw'),
                dtype=np.float32).reshape((1, 32, 32, 21)).transpose(
                    (0, 3, 1, 2))
            feat = torch.from_numpy(feat)
            kpts = feat_to_kpt(feat=feat)[0].numpy()
        elif args.model_type == 'hm':
            feat = np.fromfile(
                os.path.join(dsp_output_dir, f'{str(i).zfill(8)}_hm.raw'),
                dtype=np.float32).reshape((32, 32, 21)).transpose((2, 0, 1))
            kpts = hm_codec.decode(feat)[0][0] / 128.0
        elif args.model_type == 'direct_reg':
            feat = np.fromfile(
                os.path.join(dsp_output_dir, f'{str(i).zfill(8)}_kpt.raw'),
                dtype=np.float32).reshape((21, 2))
            kpts = feat
        elif args.model_type == 'attr':
            kpt_feat = np.fromfile(
                os.path.join(dsp_output_dir,
                             f'{str(i).zfill(8)}_kpt_feat.raw'),
                dtype=np.float32).reshape((1, 32, 32, 21)).transpose(
                    (0, 3, 1, 2))
            global_feat = np.fromfile(
                os.path.join(dsp_output_dir,
                             f'{str(i).zfill(8)}_global_feat.raw'),
                dtype=np.float32).reshape((1, 4, 4, 192)).transpose(
                    (0, 3, 1, 2))
            kpts = feat_to_kpt_attr(kpt_feat=kpt_feat, global_feat=global_feat)
        else:
            print(f'model type {args.model_type} is not supported')
            return
        kpts = kpts * bbox_scales + bbox_centers - 0.5 * bbox_scales
        result_list[i]['keypoints'] = kpts.tolist()
    with open(res_file, 'w') as f:
        json.dump(result_list, f)
    print(f'save result to {res_file}')
    metric = KeypointOKSMetric()
    result = metric(res_file)
    print(result)


if __name__ == '__main__':
    main()
