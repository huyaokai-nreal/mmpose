# flake8: noqa
import json
import os
import numpy as np
from tqdm import tqdm
from nreal_data_tool.metric import KeypointOKSMetric
import tempfile
import argparse
from mmengine import mkdir_or_exist


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
    args = parser.parse_args()
    return args


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
        kpt_x = np.fromfile(
            os.path.join(dsp_output_dir, f'{str(i).zfill(8)}_kpt_x.raw'),
            dtype=np.float32)[:, np.newaxis]
        kpt_y = np.fromfile(
            os.path.join(dsp_output_dir, f'{str(i).zfill(8)}_kpt_y.raw'),
            dtype=np.float32)[:, np.newaxis]
        kpts = np.concatenate([kpt_x, kpt_y], axis=-1)
        kpts = kpts * bbox_scales + bbox_centers - 0.5 * bbox_scales
        result_list[i]['keypoints'] = kpts.tolist()
    with open(res_file, 'w') as f:
        json.dump(result_list, f)
    metric = KeypointOKSMetric()
    result = metric(res_file)
    print(result)


if __name__ == '__main__':
    main()
