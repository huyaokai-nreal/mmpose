# Copyright (c) OpenMMLab. All rights reserved.
import json
import os

import cv2
import numpy as np
from mmengine.structures import InstanceData
from nreal_data_tool import LmdbClient
from tqdm import tqdm

from mmpose.datasets.datasets.hand.nreal_hand import HANDDataset
from mmpose.structures import PoseDataSample
from mmpose.visualization.local_visualizer import PoseLocalVisualizer


def anno_to_video(json_path, lmdb_root, output_dir='', fps=20):
    lmdb_client = LmdbClient()
    with open(json_path) as f:
        json_data = json.load(f)
    lmdb_path = os.path.join(lmdb_root, json_data['lmdb_path'])
    data_list = json_data['data']
    pose_visualizer = PoseLocalVisualizer(radius=1)
    dataset_meta = HANDDataset([]).metainfo
    pose_visualizer.set_dataset_meta(dataset_meta)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
    base_name = os.path.basename(json_path).split('.')[0]
    out_video = cv2.VideoWriter(
        os.path.join(output_dir, f'{base_name}.mp4'), fourcc, fps, (640, 480),
        True)

    for data in tqdm(data_list[:1000]):
        if data['cam_info'] == 'right camera':
            continue
        image = lmdb_client.get(f"{lmdb_path}:{data['file_name']}")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gt_instances = InstanceData()
        keypoints = np.array(data['coord_uv'])[np.newaxis, :, :]
        gt_instances.keypoints = keypoints
        gt_pose_data_sample = PoseDataSample()
        gt_pose_data_sample.gt_instances = gt_instances
        pose_visualizer.add_datasample(
            'image', image, gt_pose_data_sample, draw_pred=False)
        image = pose_visualizer.get_image()
        out_video.write(image)
    out_video.release()


if __name__ == '__main__':
    lmdb_root = os.path.join(os.environ['HOME'], 'hand_group/data')
    json_path = '/home/zx_li/hand_group/data/data_hand/hand_keypoint/annotations/test_nreal_gesture_0111_seq_spline3d_clean_lmdb_part0000.json'  # noqa
    anno_to_video(json_path, lmdb_root)
