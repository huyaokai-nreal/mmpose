# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import random
from typing import List

import cv2
import numpy as np
import torch
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData
from torch.utils.data import Dataset

from mmpose.datasets.builder import DATASETS
from mmpose.structures import PoseDataSample


@DATASETS.register_module()
class HandAttrDataset(Dataset):

    def __init__(self, data_file_list, test_mode=False) -> None:
        super().__init__()
        self.data_file_list = data_file_list
        self.test_mode = test_mode
        self.load_data_list()

    def load_data_list(self) -> List[dict]:
        full_data_list = []
        partial_data_list = []
        self.full_image_list = []
        self.partial_image_list = []
        for data_file_path in self.data_file_list:
            with open(data_file_path, 'rb') as f:
                data = pickle.load(f)
                anno = [item['anno'] for item in data]
                anno_data = np.concatenate(anno, axis=0)
                visible_anno_data = anno_data[..., -1]
                partial_anno_data = anno_data[visible_anno_data.min(
                    axis=-1) == 0]
                full_anno_data = anno_data[visible_anno_data.min(axis=-1) > 0]
                full_data_list.append(full_anno_data)
                partial_data_list.append(partial_anno_data)
                for item in data:
                    if item['anno'][..., -1].min(axis=-1) == 0:
                        self.partial_image_list.append(item['image'])
                    else:
                        self.full_image_list.append(item['image'])
        self.full_data = np.concatenate(full_data_list, axis=0)
        self.partial_data = np.concatenate(partial_data_list, axis=0)
        logger = MMLogger.get_current_instance()
        logger.info(f'load {self.full_data.shape[0]} full instances')
        logger.info(f'load {self.partial_data.shape[0]} partial instances')
        if self.test_mode:
            self.data_list = np.concatenate(
                [self.full_data, self.partial_data], axis=0)
            self.image_list = self.full_image_list + self.partial_image_list

    def __len__(self):
        return (self.full_data.shape[0] + self.partial_data.shape[0])

    def random_rotate(self, src):
        rows, cols, channel = src.shape
        angle = random.random() * 360
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(src, M, (cols, rows))
        return rotated

    def __getitem__(self, index):
        if self.test_mode:
            data_item = self.data_list[index]
            image = self.image_list[index]
        else:
            random_val = random.random()
            if random_val > 0.7:

                sample_index = np.random.choice(
                    [i for i in range(self.full_data.shape[0])], 1)[0]
                data_item = self.full_data[sample_index]
                image = self.full_image_list[sample_index]
            else:
                sample_index = np.random.choice(
                    [i for i in range(self.partial_data.shape[0])], 1)[0]
                data_item = self.partial_data[sample_index]
                image = self.partial_image_list[sample_index]
            image = self.random_rotate(image)
        keypoints_visible = data_item[..., 2][np.newaxis, :]
        packed_results = dict()
        packed_results['inputs'] = torch.from_numpy(
            image).float().contiguous().permute((2, 0, 1))
        data_sample = PoseDataSample()
        gt_instance_labels = InstanceData()
        gt_instance_labels.set_field(keypoints_visible, 'attr_labels')
        data_sample.gt_instance_labels = gt_instance_labels
        packed_results['data_samples'] = data_sample
        return packed_results
