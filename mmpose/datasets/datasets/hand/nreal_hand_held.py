# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.dataset.base_dataset import force_full_init
from mmengine.logging import MMLogger
from nreal_data_tool import LmdbClient
from xtcocotools.coco import COCO

from mmpose.datasets.builder import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class HANDBboxHeldDataset(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/nreal_hand.py')

    def __init__(self,
                 data_file_list,
                 data_root: str = '/data',
                 data_mode: str = 'topdown',
                 test_mode: bool = False,
                 hold_object_types: List = [], 
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
                 sub_data_index=-1,
                 data_ratio=-1,
                 clip_bbox=True,
                 ignore_visible=True,
                 sample_interval=1):
        self.hold_object_types = hold_object_types
        self.flip_left_to_right = flip_left_to_right
        self.data_ratio = data_ratio
        self.clip_bbox = clip_bbox
        self.data_file_list = data_file_list
        self.lmdb_client = LmdbClient()
        self.dataset_info_list = list()
        self.dataset_weight_list = dataset_weight_list
        self.dataset_num = len(self.data_file_list)
        self.lmdb_data_root = data_root
        self.with_mask = with_mask
        self.mask_ext = mask_ext
        self.sample_interval = sample_interval
        self.sub_data_index = int(sub_data_index)
        self.ignore_visible = ignore_visible
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
        self.data_num = super().__len__()

    @force_full_init
    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        if self.data_ratio <= 0:
            return super().__len__()
        else:
            if self.serialize_data:
                return int(len(self.data_address) * self.data_ratio)
            else:
                return int(len(self.data_list) * self.data_ratio)

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
        if 'bbox' not in ann:
            return None
        hold_type = ann.get('hold_obj', 'null')
        if hold_type in self.hold_object_types:
            hold_obj = np.ones((1, 1), dtype=np.float32)
        else:
            hold_obj = np.zeros((1, 1), dtype=np.float32)
        
        img_path = osp.join(self.data_prefix['img'], img['file_name'])
        img_w, img_h = img['width'], img['height']
        # get bbox in shape [1, 4], formatted as xywh
                
        bbox_type = ann.get('bbox_type', 'xywh')
        if bbox_type == 'xywh':
            x1, y1, w, h = ann['bbox']
            x2 = x1 + w
            y2 = y1 + h
            
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            x1 = (x1 - cx) * 1.5 + cx
            x2 = (x2 - cx) * 1.5 + cx
            y1 = (y1 - cy) * 1.4 + cy
            y2 = (y2 - cy) * 1.4 + cy
            
            if self.clip_bbox:
                x1 = np.clip(x1, 0, img_w - 1)
                y1 = np.clip(y1, 0, img_h - 1)
                x2 = np.clip(x2, 0, img_w - 1)
                y2 = np.clip(y2, 0, img_h - 1)
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)
        elif bbox_type == 'xyxy':
            x1, y1, x2, y2 = ann['bbox']
            
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            x1 = (x1 - cx) * 1.5 + cx
            x2 = (x2 - cx) * 1.5 + cx
            y1 = (y1 - cy) * 1.5 + cy
            y2 = (y2 - cy) * 1.5 + cy
            
            if self.clip_bbox:
                x1 = np.clip(x1, 0, img_w - 1)
                y1 = np.clip(y1, 0, img_h - 1)
                x2 = np.clip(x2, 0, img_w - 1)
                y2 = np.clip(y2, 0, img_h - 1)
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)
        else:
            logger = MMLogger.get_current_instance()
            logger.fatal(f'unsupported bbox type: {bbox_type}')

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.ones((1, 21, 3))
        keypoints = _keypoints[..., :2]
        if self.ignore_visible:
            keypoints_visible = np.ones(
                (keypoints.shape[0], keypoints.shape[1]))
        else:
            if 'keypoint_visible' not in ann or not ann['keypoint_visible']:
                keypoints_visible = -np.ones(
                    (keypoints.shape[0], keypoints.shape[1]))
            else:
                keypoints_visible = ann['keypoint_visible']
                keypoints_visible = np.minimum(1, _keypoints[..., 2])
        keypoints_visible[keypoints[..., 1] >= img_h] = 0
        keypoints_visible[keypoints[..., 1] < 0] = 0
        keypoints_visible[keypoints[..., 0] >= img_w] = 0
        keypoints_visible[keypoints[..., 0] < 0] = 0

        if 'num_keypoints' in ann:
            num_keypoints = ann['num_keypoints']
        else:
            num_keypoints = keypoints.shape[1]

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img_path,
            'image_width': img_w,
            'image_height': img_h,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'hold_obj': hold_obj,
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'cat_id': ann['category_id'],
            'meta': ann.get('meta', dict())
        }

        return data_info

    def _get_topdown_data_infos(self, instance_list) -> List:
        return instance_list

    def _load_annotations(self):
        image_list = []
        instance_list = []
        sub_dataset_start_id = 0
        if self.sub_data_index >= 0:
            self.data_file_list = [self.data_file_list[self.sub_data_index]]
        data_tag_dict = dict()
        if isinstance(self.data_file_list, dict):
            all_data_list = []
            for data_name in self.data_file_list:
                for data_file in self.data_file_list[data_name]:
                    data_tag_dict[data_file] = data_name
                    all_data_list.append(data_file)
        else:
            all_data_list = self.data_file_list

        for anno_file in all_data_list:
            coco = COCO(anno_file)
            lmdb_path = osp.join(self.lmdb_data_root,
                                 coco.dataset['lmdb_path'])
            sub_dataset_num = 0
            img_ids = coco.getImgIds()
            for img_id in img_ids[::self.sample_interval]:
                img = coco.loadImgs(img_id)[0]
                image_list.append(img)
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
                    if self.test_mode:
                        if 'tag' not in data_info['meta']:
                            data_info['meta']['tag'] = osp.basename(anno_file)
                        elif data_info['meta']['tag'] is None:
                            data_info['meta']['tag'] = osp.basename(anno_file)
                        else:
                            data_info['meta'][
                                'tag'] += f',{osp.basename(anno_file)}'
                        if isinstance(self.data_file_list, dict):
                            data_info['meta'][
                                'tag'] += f',{data_tag_dict[anno_file]}'
                    instance_list.append(data_info)
                    sub_dataset_num += 1
            self.dataset_info_list.append(
                (sub_dataset_start_id, sub_dataset_num))
            sub_dataset_start_id += sub_dataset_num
        logger: MMLogger = MMLogger.get_current_instance()
        if self.test_mode:
            logger.info(
                f'Test NrealHandDataset loaded {len(image_list)} images, {len(instance_list)} instances'  # noqa
            )
        else:
            logger.info(
                f'Train NrealHandDataset loaded {len(image_list)} images, {len(instance_list)} instances'  # noqa
            )
        return instance_list, image_list

    def get_data_info(self, idx):
        if not self.test_mode:
            idx = random.randint(0, self.data_num - 1)
            if self.dataset_weight_list:
                idx = self.__get_weighted_random_image_id()
        data_info = super().get_data_info(idx)
        data_info['img'] = self.lmdb_client.get(data_info['img_path'])
        if self.with_mask:
            data_info['mask'] = self.lmdb_client.get(data_info['mask_path'])
        data_info['img_shape'] = data_info['img'].shape[:2]
        data_info['ori_shape'] = data_info['img'].shape[:2]
        data_info['meta']['flipped'] = False
        if self.flip_left_to_right and data_info['cat_id'] == 0: # in det json format, 0 is left hand, 1 is right hand
            data_info['meta']['flipped'] = True
        return data_info

    def __get_weighted_random_image_id(self):
        db_index = np.random.choice([i for i in range(self.dataset_num)],
                                    p=self.dataset_weight_list)
        sample_index = np.random.choice(self.dataset_info_list[db_index][1],
                                        1)[0]
        sample_index = self.dataset_info_list[db_index][0] + sample_index
        return sample_index
