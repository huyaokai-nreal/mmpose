# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
import os

import lmdb

lmdb_root = os.path.join(os.environ['HOME'], 'hand_group/data')
lmdb_root = '/data'
lmdb_path = 'data_hand/hand_keypoint/public_data/train_hanco_rgb_gesture_lmdb_refresh'
env = lmdb.open(
    os.path.join(lmdb_root, lmdb_path),
    max_readers=3,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False)
db_size = env.stat()['entries']
data_num = int(db_size / 2)
file_name_list = []
for i in range(data_num):
    file_name = '{:0>8d}'.format(i * 2)
    file_name_list.append(file_name)
import time

import numpy as np

start = time.time()
with env.begin(write=False) as txn:
    for file_name in file_name_list:
        label_value = txn.get(file_name.encode())
end = time.time()
time_cost = end - start
print(f'cost {end-start} s')
mean_fps = data_num / float(time_cost)
print(f'speed is {mean_fps} fps')
