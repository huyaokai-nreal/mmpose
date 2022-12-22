from model_pipe import build_pipeline
from tqdm import tqdm
import json
import os
from collections import OrderedDict
import cv2
import numpy as np
from mmengine.config import Config
from mmpose.structures.bbox import bbox_xywh2xyxy

cfg = Config.fromfile('/home/zx_li/workspace/mmpose/tools/test_config.py')
pipeline = build_pipeline(cfg.pipeline_cfg)


def json_to_dict(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
    data_list = json_data
    data_list.sort(key=lambda x: x['img_path'])
    data_dict = OrderedDict()
    for data in data_list:
        if data['img_path'] not in data_dict:
            data_dict[data['img_path']] = [data]
        else:
            data_dict[data['img_path']].append(data)
    return data_dict


if __name__ == '__main__':
    json_dir = '/home/zx_li/hand_group/model_prelabel/'
    json_list = [os.path.join(json_dir, f'result_{i}.json') for i in range(30)]
    json_path = json_list[0]
    data_dict = json_to_dict(json_path)
    for i, (img_path, data) in enumerate(tqdm(data_dict.items())):
        img_path = img_path.replace(
            '/DATA2/jianxu/dataset/platform',
            '/data/AI_DATA/data_hand/original_data/share/platform_data')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        objects = []
        for item in data:
            bbox = bbox_xywh2xyxy(np.array([item['bbox']]))
            objects.append(dict(bbox=bbox))
        data_info = dict(img=img, objects=objects, image_id=i)
        result = pipeline.run(data_info)
