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
    nparr = np.fromstring(str_encode, np.uint8)
    data_numpy = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(data_numpy, cv2.COLOR_GRAY2BGR)
    return image


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


def mmcv_track_func(func):

    @wraps(func)
    def wrapped_func(args):
        return func(*args)

    return wrapped_func


@mmcv_track_func
def lmdb2json(lmdb_root, lmdb_path, json_path):
    env = lmdb.open(
        os.path.join(lmdb_root, lmdb_path),
        max_readers=3,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False)
    txn = env.begin()
    db_size = env.stat()['entries']
    data_num = int(db_size / 2)
    result = dict(lmdb_path=lmdb_path)
    item_list = list()
    for i in tqdm(range(data_num)):
        file_name = '{:0>8d}'.format(i * 2)
        str_value_id = '{:0>8d}'.format(i * 2 + 1)
        label_value = txn.get(str_value_id.encode())
        label_value = bytes.decode(label_value)
        item = json.loads(label_value)
        item['file_name'] = file_name
        item_list.append(item)
    result['data'] = item_list
    with open(json_path, 'w') as f:
        json.dump(result, f)
    print(f'save json file to {json_path}')


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
        max_readers=3,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False)
    txn = env.begin()
    db_size = env.stat()['entries']
    data_num = int(db_size / 2)
    hand_id = 0
    annos = []
    imgs = []
    for i in tqdm(range(data_num)):
        str_id = '{:0>8d}'.format(i * 2)
        str_value_id = '{:0>8d}'.format(i * 2 + 1)
        label_value = txn.get(str_value_id.encode())
        label_value = bytes.decode(label_value)
        d = json.loads(label_value)
        keypoints = np.array(d['coord_uv'])
        keypoints = np.concatenate(
            [keypoints[:, :2],
             np.ones([keypoints.shape[0], 1])], axis=1)
        bbox = _kps_to_bbox(keypoints)
        area = bbox[2] * bbox[3]
        category_id = 1
        if 'left_or_right' in d:
            if d['left_or_right'] == 'right':
                category_id = 2
        if 'hand_info' in d:
            if d['hand_info'] == 'right hand':
                category_id = 2
        anno = dict(
            keypoints=keypoints.tolist(),
            num_keypoints=int(keypoints.shape[0]),
            bbox=bbox.tolist(),
            area=float(area),
            image_id=i,
            category_id=category_id,
            iscrowd=0,
            id=hand_id)
        hand_id += 1
        image = get_image_from_lmdb(txn, str_id)
        img = dict(
            file_name=str_id,
            height=image.shape[0],
            width=image.shape[1],
            id=i)
        annos.append(anno)
        imgs.append(img)
    data = dict(
        images=imgs,
        annotations=annos,
        categories=cats,
        lmdb_path=lmdb_path,
    )
    with open(json_path, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    root_dir = '/data/'
    lmdb_path_list = [
        'data_hand/hand_keypoint/public_data/train_hanco_rgb_gesture_lmdb_refresh',  #84k
        'data_hand/hand_keypoint/baidu_data/train_nreal_baidu1_gesture_right_0930_lmdb',  #13.4k
        'data_hand/hand_keypoint/baidu_data/train_nreal_baidu2_gesture_right_1014_lmdb',  #12k
        'data_hand/hand_keypoint/baidu_data/train_nreal_gesture_baidu_0107_2_1_lmdb',  #16.8k
        'data_hand/hand_keypoint/baidu_data/train_nreal_gesture_baidu_220216_2_2_lmdb',  #32k
        'data_hand/hand_keypoint/baidu_data/train_nreal_gesture_baidu_220216_2_3_lmdb',  #12k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_1111_1_1_twohand_lmdb',  #13.2k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_1118_1_2_twohand_lmdb',  #24k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_1125_1_3_twohand_lmdb',  #29.7k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_1202_1_4_twohand_lmdb',  #31.8k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_1209_1_5_twohand_lmdb',  #24.3k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_1216_1_6_twohand_lmdb',  #25.1k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_1223_1_7_twohand_lmdb',  #27.8k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_1230_1_8_twohand_lmdb',  #17.2k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0113_1_9_twohand_lmdb',  #24k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0127_1_10_twohand_lmdb',  #25k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0218_1_11_twohand_lmdb',  #16k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0304_1_12_twohand_lmdb',  #22k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0318_1_13_twohand_lmdb',  #6k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0318_1_14_bad_data_twohand_lmdb',  #11k
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0401_1_15_bad_data_twohand_lmdb',
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0415_1_16_bad_data_twohand_lmdb',
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0424_1_17_bad_data_twohand_lmdb',
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0510_1_18_bad_data_twohand_lmdb',
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0517_1_19_bad_data_twohand_lmdb',
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0523_1_20_bad_data_twohand_lmdb',
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0616_1_21_bad_data_twohand_lmdb',
        'data_hand/hand_keypoint/platform_data/train_nreal_gesture_0624_1_22_bad_data_twohand_lmdb',
        'data_hand/hand_keypoint/unity_data/train_nreal_synth_gesture_1008_lmdb_3_arbitrary'
    ]  #38k
    lmdb_path_list = [
        'data_hand/hand_keypoint/platform_data/test_nreal_gesture_1111_1_1_twohand_lmdb'
    ]
    lmdb_path_list = [
        'data_hand/hand_keypoint/platform_data/test_nreal_gesture_3_1_221201_fisheye_vertical_binocular_lmdb',
        'data_hand/hand_keypoint/platform_data/test_nreal_gesture_3_2_221201_fisheye_horizontal_binocular_lmdb'
    ]
    root_dir = os.path.join(os.environ['HOME'], 'hand_group/data')
    json_dir = os.path.join(root_dir, 'data_hand/hand_keypoint/annotations')
    lmdb_path_list = [
        'data_hand/hand_keypoint/seq_data/test_nreal_gesture_0111_seq_spline3d_clean_lmdb_part0000'
    ]
    tasks = [(root_dir, lmdb_path,
              os.path.join(json_dir,
                           os.path.basename(lmdb_path) + '.json'))
             for lmdb_path in lmdb_path_list]
    track_parallel_progress(lmdb2json, tasks, nproc=1)
