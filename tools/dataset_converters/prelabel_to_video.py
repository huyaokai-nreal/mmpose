# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from collections import OrderedDict

import cv2
import lmdb
import numpy as np
from mmengine import track_parallel_progress
from mmengine.structures import InstanceData
from tqdm import tqdm

from mmpose.datasets.datasets.hand.nreal_hand import HANDDataset
from mmpose.structures import PoseDataSample
from mmpose.visualization.local_visualizer import PoseLocalVisualizer


def img_to_lmdb(json_path, lmdb_path):
    data_dict = json_to_dict(json_path)
    img_list = list(data_dict.keys())[:600]
    db = lmdb.open(lmdb_path, map_size=1099511627776)
    with db.begin(write=True) as txn:
        for id, img_path in tqdm(enumerate(img_list)):
            img_path = img_path.replace(
                '/DATA2/jianxu/dataset/platform',
                '/data/AI_DATA/data_hand/original_data/share/platform_data')
            with open(img_path, 'rb') as f:
                image_data = f.read()
                key = str(id).zfill(8)
                txn.put(key.encode(), image_data)
            if id > 0 and id % 1000 == 0:
                txn.commit()
    db.close()


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


def anno_to_video(json_path, output_dir='', fps=10):
    data_dict = json_to_dict(json_path)
    pose_visualizer = PoseLocalVisualizer(radius=1)
    dataset_meta = HANDDataset([]).metainfo
    pose_visualizer.set_dataset_meta(dataset_meta)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
    base_name = os.path.basename(json_path).split('.')[0]
    out_video = cv2.VideoWriter(
        os.path.join(output_dir, f'{base_name}.mp4'), fourcc, fps, (640, 480),
        True)
    for img_path in tqdm(list(data_dict.keys())[:600]):
        data_list = data_dict[img_path]
        image_path = img_path.replace(
            '/DATA2/jianxu/dataset/platform',
            '/data/AI_DATA/data_hand/original_data/share/platform_data')
        image = cv2.imread(image_path)
        gt_instances = InstanceData()
        keypoints = list()
        for data in data_list:
            keypoints.append(data['keypoints'])
        keypoints = np.array(keypoints)[:, :, :2]
        gt_instances.keypoints = keypoints
        gt_pose_data_sample = PoseDataSample()
        gt_pose_data_sample.gt_instances = gt_instances
        pose_visualizer.add_datasample(
            'image', image, gt_pose_data_sample, draw_pred=False)
        image = pose_visualizer.get_image()
        out_video.write(image)
    out_video.release()


if __name__ == '__main__':

    json_dir = '/home/zx_li/hand_group/model_prelabel/'
    json_list = [os.path.join(json_dir, f'result_{i}.json') for i in range(30)]
    track_parallel_progress(anno_to_video, json_list, nproc=8)
    img_to_lmdb(json_list[0], 'work_dirs/test_lmdb_result0')
