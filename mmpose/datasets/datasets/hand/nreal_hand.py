# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Sequence, Union

import cv2
import lmdb
import numpy as np
from mmengine.logging import MMLogger
from xtcocotools.coco import COCO

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
                 serialize_data: bool = True,
                 lazy_init: bool = False,
                 max_refetch: int = 100):
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
        return img

    def __left_2_right_hand(self, results):
        img = results['img']
        width = img.shape[1]
        results['img'] = img[:, ::-1]
        results['joints_3d'][:, 0] = width - 1 - results['joints_3d'][:, 0]
        results['bbox'] = self._kps_to_bbox(results['joints_3d'])

    def get_data_info(self, idx):

        data_info = super().get_data_info(idx)
        data_info['img'] = self._get_image(data_info['img_path'])
        data_info['img_shape'] = data_info['img'].shape[:2]
        data_info['ori_shape'] = data_info['img'].shape[:2]
        return data_info
