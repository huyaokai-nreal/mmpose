# flake8: noqa
import json
import os
import numpy as np
from tqdm import tqdm
from nreal_data_tool.metric import KeypointOKSMetric

if __name__ == '__main__':
    dsp_output_dir = '/home/zx_li/workspace/mmpose/data/Inference_20230203_160330_DSP/INT8'
    template_json_path = '/home/zx_li/workspace/mmpose/data/result_keypoints.json'
    json_path = 'result.json'
    with open(template_json_path, 'r') as f:
        result_list = json.load(f)
    for i, result in enumerate(tqdm(result_list)):
        gt = result['gt_keypoints']
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
    with open(json_path, 'w') as f:
        json.dump(result_list, f)

    metric = KeypointOKSMetric()
    result = metric(json_path)
    print(result)
