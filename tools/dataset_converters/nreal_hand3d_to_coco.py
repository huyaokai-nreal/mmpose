# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
import json
import os
from functools import wraps
from pprint import pprint

import cv2
import lmdb
import numpy as np
from mmengine import track_parallel_progress
from tqdm import tqdm


def get_image_from_lmdb(txn, id):
    str_encode = txn.get(id.encode())
    nparr = np.frombuffer(str_encode, np.uint8)
    data_numpy = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(data_numpy, cv2.COLOR_GRAY2BGR)
    return image


def mmcv_track_func(func):

    @wraps(func)
    def wrapped_func(args):
        return func(*args)

    return wrapped_func


def _kps_to_bbox(kps):
    min_x = kps[:, 0].min()
    min_y = kps[:, 1].min()
    max_x = kps[:, 0].max()
    max_y = kps[:, 1].max()
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    min_x = (min_x - cx) * 1.5 + cx
    min_y = (min_y - cy) * 1.4 + cy
    max_x = (max_x - cx) * 1.5 + cx
    max_y = (max_y - cy) * 1.4 + cy
    bbox = np.array([min_x, min_y, max_x - min_x, max_y - min_y],
                    dtype=np.float32)
    return bbox


def pasing_lmdb(txn, str_id, str_value_id):
    label_value = txn.get(str_value_id.encode())
    try:
        label_value = bytes.decode(label_value)
    except:
        print('decode label failed')
        return
    d = json.loads(label_value)
    # from IPython import embed; embed(d)

    keypoints = np.array(d['coord_uv'])
    keypoints = np.concatenate(
        [keypoints[:, :2], np.ones([keypoints.shape[0], 1])], axis=1)

    bbox = _kps_to_bbox(keypoints)
    # bbox = np.array(d['bbox'], dtype=np.float32)

    area = bbox[2] * bbox[3]
    category_id = 1
    if 'hand_info' in d:
        # print(d['hand_info'])
        if d['hand_info'] == 'right hand':
            category_id = 2
    if 'left_or_right' in d:
        if d['left_or_right'] == 'right':
            category_id = 2
    image = get_image_from_lmdb(txn, str_id)
    cam_matrix = d['cam_matrix']
    # print(category_id)
    res = {
        'keypoints': keypoints,
        'bbox': bbox,
        'area': area,
        'category_id': category_id,
        'image': image,
        'cam_matrix': cam_matrix,
        'leftcam_p_rightcam': d['leftcam_p_rightcam'],
        'leftcam_q_rightcam': d['leftcam_q_rightcam'],
        'kp3d_spline': d['kp3d_spline']
    }
    return res


@mmcv_track_func
def lmdb2coco(lmdb_root, lmdb_path, json_path):
    cats = [{
        'id': 1,
        'name': 'left_hand',
    }, {
        'id': 2,
        'name': 'right_hand',
    }]

    env = lmdb.open(
        os.path.join(lmdb_root, lmdb_path),
        max_readers=8,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False)
    txn = env.begin()
    db_size = env.stat()['entries']
    data_num = int(db_size / 4)
    annos = []
    imgs = []

    for i in tqdm(range(data_num)):
        try:
            str_id_left = '{:0>8d}'.format(i * 4 + 0)
            str_value_id_left = '{:0>8d}'.format(i * 4 + 1)
            res_left = pasing_lmdb(txn, str_id_left, str_value_id_left)

            str_id_right = '{:0>8d}'.format(i * 4 + 2)
            str_value_id_right = '{:0>8d}'.format(i * 4 + 3)
            res_right = pasing_lmdb(txn, str_id_right, str_value_id_right)

            anno = dict(
                keypoints_left=res_left['keypoints'].tolist(),
                bbox_left=res_left['bbox'].tolist(),
                area_left=float(res_left['area']),
                keypoints_right=res_right['keypoints'].tolist(),
                bbox_right=res_right['bbox'].tolist(),
                area_right=float(res_right['area']),
                meta=dict(
                    cam_matrix_left=res_left['cam_matrix'],
                    cam_matrix_right=res_right['cam_matrix'],
                    leftcam_p_rightcam=res_left['leftcam_p_rightcam'],
                    leftcam_q_rightcam=res_left['leftcam_q_rightcam'],
                    kp3d_spline=res_left['kp3d_spline'],
                    category_id=res_left['category_id'],
                ),
                category_id=res_left['category_id'],
                num_keypoints=res_left['keypoints'].shape[0],
                image_id='_'.join([str(i * 2), str(i * 2 + 1)]),
                iscrowd=0,
                id=i)

            image_left = res_left['image']  # 只是为了获取图像的(w,h),不保存图片到json
            image_right = res_right['image']

            img_left = dict(
                file_name=str_id_left,
                height=image_left.shape[0],
                width=image_left.shape[1],
                id=i * 2)
            img_right = dict(
                file_name=str_id_right,
                height=image_right.shape[0],
                width=image_right.shape[1],
                id=i * 2 + 1)
            annos.append(anno)
            imgs.append(img_left)
            imgs.append(img_right)

        except:
            continue

    data = dict(
        images=imgs,
        annotations=annos,
        categories=cats,
        lmdb_path=lmdb_path,
    )
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f)
    print(f'save json to {json_path}')


# change lmdb_path_list and json_dir, then run the code
if __name__ == '__main__':
    # root_dir = '/data/AI_DATA'  # shared path
    root_dir = '/data/AI_DATA_LOCAL'  # local path

    lmdb_path_list = []
    dataset_path = 'data_hand/hand_keypoint/seq_data'

    for dataset_part in os.listdir(os.path.join(root_dir, dataset_path)):
        lmdb_path_list.append(os.path.join(dataset_path, dataset_part))
    lmdb_path_list.sort()
    # pprint(lmdb_path_list)

    # lmdb_path_list = [
    #     '',
    # ]

    json_dir = os.path.join(
        root_dir, 'data_hand/hand_keypoint/annotations3d/seq_data_kpts2bbox')

    tasks = [(root_dir, lmdb_path,
              os.path.join(json_dir,
                           os.path.basename(lmdb_path) + '.json'))
             for lmdb_path in lmdb_path_list]
    # print(tasks)
    # lmdb2coco(tasks[0])
    track_parallel_progress(lmdb2coco, tasks, nproc=50)
