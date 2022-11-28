# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Sequence, Union

import cv2
import lmdb
import numpy as np
from mmengine.logging import MMLogger
from xtcocotools.coco import COCO
import os.path as osp
from mmpose.datasets.builder import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class HANDDataset(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/nreal_hand.py')

    def __init__(self,
                 data_file_list,
                 data_mode: str = 'topdown',
                 test_mode: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 metainfo: Optional[dict] = None,
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 100,
                 flip_left_to_right: bool = True):
        self.flip_left_to_right = flip_left_to_right
        self.data_file_list = data_file_list
        self.lmdb_server_map = {}
        super().__init__(
            data_root='',
            data_mode=data_mode,
            test_mode=test_mode,
            metainfo=metainfo,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            pipeline=pipeline)

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw COCO annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict | None: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        # filter invalid instance
        if 'bbox' not in ann or 'keypoints' not in ann or max(
                ann['keypoints']) == 0:
            return None

        img_path = osp.join(self.data_prefix['img'], img['file_name'])
        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(
            ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        if 'num_keypoints' in ann:
            num_keypoints = ann['num_keypoints']
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img_path,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'cat_id': ann['category_id']
        }

        return data_info

    def _load_annotations(self):
        data_list = []
        for anno_file in self.data_file_list:
            coco = COCO(anno_file)
            lmdb_path = coco.dataset['lmdb_path']
            lmdb_env = lmdb.open(
                lmdb_path,
                max_readers=3,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)
            lmdb_txn = lmdb_env.begin()
            self.lmdb_server_map[lmdb_path] = dict(env=lmdb_env, txn=lmdb_txn)
            img_ids = coco.getImgIds()
            for img_id in img_ids:
                img = coco.loadImgs(img_id)[0]
                ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
                for ann in coco.loadAnns(ann_ids):
                    data_info = self.parse_data_info(
                        dict(raw_ann_info=ann, raw_img_info=img))
                    # skip invalid instance annotation.
                    if not data_info:
                        continue
                    data_info[
                        'img_path'] = f"{lmdb_path}:{data_info['img_path']}"
                    data_list.append(data_info)
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(f'HandDataset loaded {len(data_list)} images')
        return data_list

    def _get_image(self, img_info):
        lmdb_name = img_info.split(':')[0]
        img_id = img_info.split(':')[1]
        img_array = self.lmdb_server_map[lmdb_name]['txn'].get(img_id.encode())
        img_array = np.fromstring(img_array, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        img = img[:, :, np.newaxis]
        return img

    def __left_2_right_hand(self, results):
        img = results['img']
        width = img.shape[1]
        results['img'] = img[:, ::-1]
        results['keypoints'][:, :,
                             0] = width - 1 - results['keypoints'][:, :, 0]
        results['bbox'][:, 0] = width - 1 - results['bbox'][:, 0]
        results['bbox'][:, 2] = width - 1 - results['bbox'][:, 2]

    def get_data_info(self, idx):

        data_info = super().get_data_info(idx)
        data_info['img'] = self._get_image(data_info['img_path'])
        data_info['img_shape'] = data_info['img'].shape[:2]
        data_info['ori_shape'] = data_info['img'].shape[:2]
        if self.flip_left_to_right and data_info['cat_id'] == 1:
            self.__left_2_right_hand(data_info)
        return data_info
