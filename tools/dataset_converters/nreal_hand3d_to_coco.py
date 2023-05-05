# flake8: noqa
import json
import os
from functools import wraps

import cv2
import lmdb
from mmengine import track_parallel_progress
import numpy as np
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


def pasing_lmdb(txn, str_id, str_value_id):
    label_value = txn.get(str_value_id.encode())
    try:
        label_value = bytes.decode(label_value)
    except:
        print('decode label failed')
        return
    d = json.loads(label_value)
    keypoints = np.array(d['coord_uv'])
    keypoints = np.concatenate(
        [keypoints[:, :2], np.ones([keypoints.shape[0], 1])], axis=1)
    bbox = np.array(d['bbox'], dtype=np.float32)
    area = bbox[2] * bbox[3]
    category_id = 1
    if 'hand_info' in d:
        if d['hand_info'] == 'right hand':
            category_id = 2
    if 'left_or_right' in d:
        if d['left_or_right'] == 'right':
            category_id = 2
    image = get_image_from_lmdb(txn, str_id)
    res = {
        'keypoints': keypoints,
        'bbox': bbox,
        'area': area,
        'category_id': category_id,
        'image': image
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
    # from IPython import embed; embed()
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
        str_id_left = '{:0>8d}'.format(i * 4 + 0)
        str_value_id_left = '{:0>8d}'.format(i * 4 + 1)
        res_left = pasing_lmdb(txn, str_id_left, str_value_id_left)

        str_id_right = '{:0>8d}'.format(i * 4 + 2)
        str_value_id_right = '{:0>8d}'.format(i * 4 + 3)
        res_right = pasing_lmdb(txn, str_id_right, str_value_id_right)

        keypoints_left = res_left['keypoints']
        keypoints_right = res_right['keypoints']
        keypoints = np.concatenate([
            keypoints_left[np.newaxis, ...], keypoints_right[np.newaxis, ...]
        ],
                                   axis=0)
        bbox_left = res_left['bbox']
        bbox_right = res_right['bbox']
        bbox = np.concatenate(
            [bbox_left[np.newaxis, ...], bbox_right[np.newaxis, ...]], axis=0)

        anno = dict(
            keypoints=keypoints.tolist(),
            num_keypoints=[keypoints_left.shape[0], keypoints_right.shape[0]],
            bbox=bbox.tolist(),
            area=[float(res_left['area']),
                  float(res_right['area'])],
            image_id=[i * 2, i * 2 + 1],
            category_id=[res_left['category_id'], res_right['category_id']],
            iscrowd=[0,0],
            id=i)

        image_left = res_left['image']
        image_right = res_right['image']
        image = np.concatenate(
            [image_left[np.newaxis, ...], image_right[np.newaxis, ...]],
            axis=0)
        img = dict(
            file_name=[str_id_left, str_id_right],
            height=[image_left.shape[0], image_right.shape[0]],
            width=[image_left.shape[1], image_right.shape[1]],
            id=[i * 2, i * 2 + 1])
        annos.append(anno)
        imgs.append(img)

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
    root_dir = '/data/AI_DATA'  # shared path
    # root_dir = '/data/hand_group/data'  # local path
    lmdb_path_list = [
        'data_hand/hand_keypoint/seq_data/train_nreal_gesture_0111_seq_spline3d_clean_lmdb_part0000',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_0111_seq_spline3d_clean_lmdb_part0001',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_0111_seq_spline3d_clean_lmdb_part0002',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_0111_seq_spline3d_clean_lmdb_part0003',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_0111_seq_spline3d_clean_lmdb_part0004',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_0111_seq_spline3d_clean_lmdb_part0005',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_0111_seq_spline3d_clean_lmdb_part0006',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_0111_seq_spline3d_clean_lmdb_part0007',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_0111_seq_spline3d_clean_lmdb_part0008',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_0111_seq_spline3d_clean_lmdb_part0009',

        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_220119_seq_2_spline3d_clean_lmdb_part0000',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_220119_seq_2_spline3d_clean_lmdb_part0001',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_220119_seq_2_spline3d_clean_lmdb_part0002',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_220119_seq_2_spline3d_clean_lmdb_part0003',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_220119_seq_2_spline3d_clean_lmdb_part0004',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_220119_seq_2_spline3d_clean_lmdb_part0005',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_220119_seq_2_spline3d_clean_lmdb_part0006',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_220119_seq_2_spline3d_clean_lmdb_part0007',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_220119_seq_2_spline3d_clean_lmdb_part0008',
        # 'data_hand/hand_keypoint/seq_data/train_nreal_gesture_220119_seq_2_spline3d_clean_lmdb_part0009',
    ]
    json_dir = os.path.join(root_dir,
                            'data_hand/hand_keypoint/annotations3d/seq_data')

    tasks = [(root_dir, lmdb_path,
              os.path.join(json_dir,
                           os.path.basename(lmdb_path) + '.json'))
             for lmdb_path in lmdb_path_list]
    # print(tasks)
    lmdb2coco(tasks[0])
    # track_parallel_progress(lmdb2coco, tasks, nproc=4)