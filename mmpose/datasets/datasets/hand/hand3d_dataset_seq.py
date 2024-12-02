# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import random
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.dataset.base_dataset import force_full_init
from mmengine.dataset.utils import default_collate
from mmengine.logging import MessageHub, MMLogger
from nreal_data_tool import LmdbClient, TarClient
from nreal_data_tool.schema.instance import BinocularCameraInstance, CameraInstance
from nreal_data_tool.utils.camera import (build_from_BinocularCameraInstance,
                                          build_from_CameraInstance,
                                          get_virtual_camera_transform)
from xtcocotools.coco import COCO

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.hand.nimble_hand import get_nimble_bones_length
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class Hand3DDatasetSeq(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/nreal_hand.py')
    category_name_list = ['background', 'hand', 'right_hand']

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
                 dataset_weight_list: List = [],
                 with_mask: bool = False,
                 mask_ext: str = 'mask',
                 sub_data_index=-1,
                 data_ratio=-1,
                 point_type='3D',
                 filter_kpt_exceed=False,
                 flip_left_to_right=False,
                 standard_stereo=False,
                 sample_interval=1,
                 seq_len=4,
                 round_num=-1,
                 epochs_per_round=-1):
        self.data_ratio = data_ratio
        self.data_file_list = data_file_list
        self.lmdb_client = LmdbClient()
        self.tar_client = TarClient()
        self.point_type = point_type
        self.dataset_info_list = list()
        self.dataset_weight_list = dataset_weight_list
        self.dataset_num = len(self.data_file_list)
        self.lmdb_data_root = data_root
        self.with_mask = with_mask
        self.mask_ext = mask_ext
        self.sub_data_index = int(sub_data_index)
        self.cams_info = dict()
        self.hand_bones_list = list()
        self.seq_len = seq_len
        self.filter_kpt_exceed = filter_kpt_exceed
        self.sample_interval = sample_interval
        self.standard_stereo = standard_stereo
        self.round_num = round_num
        self.epochs_per_round = epochs_per_round
        self.flip_left_to_right = flip_left_to_right
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
        self.__update_bones_with_mean_template()
        mean_bones = get_nimble_bones_length()
        self.hand_bones_list = [mean_bones] * len(self.hand_bones_list)

    def __update_bones_with_mean_template(self):
        mean_bones = get_nimble_bones_length()
        self.hand_scale_list = []
        for bones in self.hand_bones_list:
            scale = np.mean(bones / mean_bones)
            self.hand_scale_list.append(scale)

    def get_image(self, image_path):
        if '_lmdb' in image_path:
            return self.lmdb_client.get(image_path)
        elif '.tar' in image_path:
            return self.tar_client.get(image_path)
        else:
            raise NotImplementedError

    @staticmethod
    def is_keypoint_within_bounds(keypoint, image_width, image_height):
        x, y = keypoint[:, 0], keypoint[:, 1]
        within_mask = ((0 <= x) & (x < image_width)) & ((0 <= y) &
                                                        (y < image_height))
        return within_mask.sum() >= keypoint.shape[0] * 0.5

    @force_full_init
    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        if self.test_mode and self.point_type == '2.5D':
            self.data_ratio = 1
        if self.data_ratio <= 0:
            return super().__len__()
        else:
            if self.serialize_data:
                return int(len(self.data_address) * self.data_ratio)
            else:
                return int(len(self.data_list) * self.data_ratio)


    def _get_mean_hand_bones(self, keypoints3d_list):
        N = keypoints3d_list.shape[0]
        root_keypoints3d = keypoints3d_list[:, :1, :].reshape((N, 1, 1, 3))
        root_keypoints3d = np.tile(root_keypoints3d, (1, 5, 1, 1))
        other_keypoints3d = keypoints3d_list[:, 1:, :].reshape((N, 5, 4, 3))
        keypoints3d = np.concatenate([root_keypoints3d, other_keypoints3d],
                                     axis=2)
        bones = np.linalg.norm(
            keypoints3d[:, :, 1:, :] - keypoints3d[:, :, :-1, :], axis=-1)
        mean_bones = bones.mean(axis=0)
        return mean_bones

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
        
        img_path = osp.join(self.data_prefix['img'],
                                 img['file_name'])

        # filter invalid instance
        if 'bbox' not in ann or 'keypoints' not in ann:
            return None

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
        keypoints3d = np.array(ann['keypoints3d'])[np.newaxis]  # (1,21,3)
        
        keypoints_visible = np.minimum(1, _keypoints[..., 2])
        if 'hot3d' in img_path:
            keypoints_visible[0][0] = 0
        cam_key = ann['camera_instance_id']
        cam_info = self.cams_info[cam_key]
        cam_model = \
            build_from_CameraInstance(cam_info)

        if 'num_keypoints' in ann:
            num_keypoints = ann['num_keypoints']
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        meta = ann.get('meta', dict())
        if 'camera_angle' not in meta:
            meta['camera_angle'] = 0
        meta['category_id'] = ann['category_id']

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img_path,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'image_width': img_w,
            'image_height': img_h,
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints3d': keypoints3d,
            'keypoints_visible': keypoints_visible,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            'cat_id': ann['category_id'],
            # 'raw_ann_info': copy.deepcopy(ann),
            'cam_model': cam_model,
            'meta': meta
        }

        if 'crowdIndex' in img:
            data_info['crowd_index'] = img['crowdIndex']

        return data_info

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        image_list = []
        instance_list = []
        filter_annotation_num = 0
        if self.sub_data_index >= 0:
            self.data_file_list = [self.data_file_list[self.sub_data_index]]
        instance_idx = 0
        f301, f302, f303, f304 = 0, 0, 0, 0
        left, right = 0, 0
        random.shuffle(self.data_file_list)
        for i, anno_file in enumerate(self.data_file_list):
            coco = COCO(anno_file)
            lmdb_path = osp.join(self.lmdb_data_root,
                                 coco.dataset['lmdb_path'])
            seq_list = []
            if coco.dataset['cameras_info']:  # 真值系统数据
                for k, v in coco.dataset['cameras_info'].items():
                    self.cams_info[k] = CameraInstance.from_dict(v)
            keypoints3d_list = []
            ann_ids = coco.getAnnIds()
            used_ann_ids_num = int(len(ann_ids)*self.sample_interval)
            begin_num = random.randint(0, len(ann_ids)-used_ann_ids_num)
            for ann_id in ann_ids[begin_num:begin_num+used_ann_ids_num]:
                ann = coco.loadAnns(ann_id)[0]
                img_id = int(ann['id'])//2
                img = coco.loadImgs(img_id)[0]

                if ann['category_id'] == 1:
                    left += 1
                else:
                    right += 1
                # img_path = osp.join(self.data_prefix['img'],
                #                         img['file_name'])
                data_info = self.parse_data_info(
                    dict(raw_ann_info=ann, raw_img_info=img))
                if self.test_mode:
                    if 'tag' not in data_info['meta']:
                        data_info['meta']['tag'] = osp.basename(anno_file)
                    elif data_info['meta']['tag'] is None:
                        data_info['meta']['tag'] = osp.basename(anno_file)
                    else:
                        data_info['meta'][
                            'tag'] += f',{osp.basename(anno_file)}'
                keypoints3d_list.append(data_info['keypoints3d'])
                if self.with_mask:
                    category_name = self.category_name_list[int(
                        data_info['cat_id'])]
                data_info['img_path'] = \
                    f"{lmdb_path}:{data_info['img_path']}"

                data_info['meta']['template_bones_id'] = len(
                    self.hand_bones_list)
                instance_list.append(data_info)
                seq_list.append(instance_idx)
                image_list.append(img)
                instance_idx += 1
            
            self.dataset_info_list.append(seq_list)

            if len(keypoints3d_list) > 0:
                keypoints3d_list = np.concatenate(keypoints3d_list, axis=0)
                mean_bones = self._get_mean_hand_bones(keypoints3d_list)
                self.hand_bones_list.append(mean_bones)
        logger: MMLogger = MMLogger.get_current_instance()
        if self.test_mode:
            logger.info(
                f'Test PairHandDataset loaded {len(image_list)} images, {len(instance_list)} pair instances, filter {filter_annotation_num} pair instances'  # noqa
            )
        else:
            logger.info(
                f'Train PairHandDataset loaded {len(image_list)} images, {len(instance_list)} pair instances, filter {filter_annotation_num} pair instances'  # noqa
            )
            logger.info(
                f'flora301: {f301} flora302: {f302} flora303: {f303} flora304: {f304} '  # noqa
            )
        return instance_list, image_list

    def get_data_info(self, idx):
        idx = idx % self.data_num
        data_info = super().get_data_info(idx)
        data_info['meta']['flipped'] = False
        data_info['img'] = self.get_image(data_info['img_path'])
        data_info['meta']['frame_height'] = data_info['img'].shape[0]
        data_info['meta']['frame_width'] = data_info['img'].shape[1]
        return data_info


    def get_seq_idx(self, idx) -> list:
        # 随机一个训练文件
        seq_list = []
        while (len(seq_list) == 0):
            seq_idx = np.random.choice(range(len(self.dataset_info_list)))
            # 数据分批，并根据当前epoch随机从一个批次中抽一份数据
            if self.round_num > 0:
                num_per_round = len(self.dataset_info_list) // self.round_num
                mh = MessageHub.get_current_instance()
                cur_epoch = mh.get_info('epoch')
                round_id = cur_epoch // self.epochs_per_round % self.round_num
                seq_idx = random.randint(
                    round_id * num_per_round,
                    min(
                        len(self.dataset_info_list) - 1,
                        (round_id + 1) * num_per_round - 1))
            seq_list = self.dataset_info_list[seq_idx]
        seq_list_cur_idx = np.random.choice(range(len(seq_list)))
        # import ipdb;ipdb.set_trace()
        if self.test_mode:
            seq_list_cur_idx = idx
        idx_list = []
        for i in range(self.seq_len):
            tmp = max(seq_list_cur_idx - 2*i, 0)
            idx_list.append(tmp)

        idx_list.reverse()

        final_list = []
        for idx in idx_list:
            tmp = seq_list[idx]
            final_list.append(tmp)
        return final_list

    @force_full_init
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        :class:`BaseCocoStyleDataset` overrides this method from
        :class:`mmengine.dataset.BaseDataset` to add the metainfo into
        the ``data_info`` before it is passed to the pipeline.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        
        collate_list = []
        # import ipdb;ipdb.set_trace()
        seq_idx_list = self.get_seq_idx(idx)
        
        for i, idx in enumerate(seq_idx_list):
            data_info_ori = self.get_data_info(idx)
            data_info = copy.deepcopy(data_info_ori)
            meta = copy.deepcopy(data_info['meta'])
            meta['ori_camera'] = copy.deepcopy(data_info['cam_model'])
            meta['template_bones'] = self.hand_bones_list[
                data_info['meta']['template_bones_id']]
            meta['hand_scale'] = self.hand_scale_list[data_info['meta']
                                                        ['template_bones_id']]

            meta['test_mode'] = self.test_mode
            meta['camera_name'] = 'left'

            data_info = {
                'img_id': data_info['img_id'],
                'image_width': data_info['img'].shape[1],
                'image_height': data_info['img'].shape[0],
                'img_path': data_info['img_path'],
                'keypoints': data_info['keypoints'].copy(),
                'img': data_info['img'].copy(),
                'bbox': data_info['bbox'].copy(),
                'keypoints3d': data_info['keypoints3d'].copy(),
                'bbox_score': np.ones(1, dtype=np.float32),
                'iscrowd': data_info['iscrowd'],
                'id': data_info['id'],
                'cat_id': data_info['cat_id'],
                'meta': meta,
                'sample_idx': data_info['sample_idx'],
                'upper_body_ids': data_info['upper_body_ids'],
                'lower_body_ids': data_info['lower_body_ids'],
                'keypoints_visible': data_info['keypoints_visible'].copy(),
                'camera_name': 'left'
            }
            if self.flip_left_to_right and data_info['cat_id'] == 1:
                data_info['meta']['flipped'] = True
            pipeline_results = self.pipeline(data_info)
            collate_list.append(pipeline_results)
        
        all_results = default_collate(collate_list)
        return all_results
