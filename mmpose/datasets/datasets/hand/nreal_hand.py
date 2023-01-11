# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Sequence, Union
import numpy as np
from mmengine.logging import MMLogger
from xtcocotools.coco import COCO
import os.path as osp
from mmpose.datasets.builder import DATASETS
from ..base import BaseCocoStyleDataset
from nreal_data_tool import LmdbClient


@DATASETS.register_module()
class HANDDataset(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/nreal_hand.py')

    def __init__(self,
                 data_file_list,
                 data_root: str = '/data',
                 data_mode: str = 'topdown',
                 test_mode: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 metainfo: Optional[dict] = None,
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 100,
                 flip_left_to_right: bool = True,
                 dataset_weight_list: List = [],
                 with_mask: bool = False,
                 mask_ext: str = 'mask',
                 sub_data_index=-1):
        self.flip_left_to_right = flip_left_to_right
        self.data_file_list = data_file_list
        self.lmdb_client = LmdbClient()
        self.dataset_info_list = list()
        self.dataset_weight_list = dataset_weight_list
        self.dataset_num = len(self.data_file_list)
        self.lmdb_data_root = data_root
        self.with_mask = with_mask
        self.mask_ext = mask_ext
        self.sub_data_index = int(sub_data_index)
        if dataset_weight_list:
            assert len(dataset_weight_list) == len(data_file_list)
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
            'image_width': img_w,
            'image_height': img_h,
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
        sub_dataset_start_id = 0
        if self.sub_data_index >= 0:
            self.data_file_list = [self.data_file_list[self.sub_data_index]]
        for i, anno_file in enumerate(self.data_file_list):
            coco = COCO(anno_file)
            lmdb_path = osp.join(self.lmdb_data_root,
                                 coco.dataset['lmdb_path'])
            sub_dataset_num = 0
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
                    if self.with_mask:
                        data_info[
                            'mask_path'] = \
                            f"{lmdb_path}_{self.mask_ext}:{data_info['img_path']}" # noqa
                    data_info[
                        'img_path'] = f"{lmdb_path}:{data_info['img_path']}"
                    data_list.append(data_info)
                    sub_dataset_num += 1
            self.dataset_info_list.append(
                (sub_dataset_start_id, sub_dataset_num))
            sub_dataset_start_id += sub_dataset_num
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(f'HandDataset loaded {len(data_list)} images')
        return data_list

    def __left_2_right_hand(self, results):
        img = results['img']
        width = img.shape[1]
        results['img'] = img[:, ::-1]
        results['keypoints'][:, :,
                             0] = width - 1 - results['keypoints'][:, :, 0]
        bbox = results['bbox']
        bbox[:, ::2] = width - 1 - bbox[:, ::2]
        min_x = np.min(bbox[:, ::2])
        max_x = np.max(bbox[:, ::2])
        min_y = np.min(bbox[:, 1::2])
        max_y = np.max(bbox[:, 1::2])
        results['bbox'] = np.array([[min_x, min_y, max_x, max_y]], np.float32)
        if 'mask' in results:
            results['mask'] = results['mask'][:, ::-1]

    def get_data_info(self, idx):
        if self.dataset_weight_list:
            idx = self.__get_weighted_random_image_id()
        data_info = super().get_data_info(idx)
        data_info['img'] = self.lmdb_client.get(data_info['img_path'])
        if self.with_mask:
            data_info['mask'] = self.lmdb_client.get(data_info['mask_path'])
        data_info['img_shape'] = data_info['img'].shape[:2]
        data_info['ori_shape'] = data_info['img'].shape[:2]
        if self.flip_left_to_right and data_info['cat_id'] == 1:
            self.__left_2_right_hand(data_info)
        return data_info

    def __get_weighted_random_image_id(self):
        db_index = np.random.choice([i for i in range(self.dataset_num)],
                                    p=self.dataset_weight_list)
        sample_index = np.random.choice(self.dataset_info_list[db_index][1],
                                        1)[0]
        sample_index = self.dataset_info_list[db_index][0] + sample_index
        return sample_index
