# Copyright (c) OpenMMLab. All rights reserved.
import os


def get_quest3_anno_paths(data_root):
    data_path = os.path.join(
        data_root,
        'data_hand/hand_keypoint/public_data/hot3d/train_quest3_anno')
    files = os.listdir(data_path)
    files.sort()
    files = [os.path.join(data_path, file_name) for file_name in files]
    test_num = 100
    train_num = len(files) - test_num
    return files[:train_num], files[train_num:]


def get_aria_anno_paths(data_root):
    data_path = os.path.join(
        data_root, 'data_hand/hand_keypoint/public_data/hot3d/train_aria_anno')
    files = os.listdir(data_path)
    files.sort()
    files = [os.path.join(data_path, file_name) for file_name in files]
    test_num = 100
    train_num = len(files) - test_num
    return files[:train_num], files[train_num:]
