# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.dataset.base_dataset import force_full_init
from mmengine.dataset.utils import default_collate
from mmengine.logging import MMLogger
from nreal_data_tool import LmdbClient
from nreal_data_tool.schema.instance import BinocularCameraInstance
from nreal_data_tool.utils.camera import build_from_BinocularCameraInstance
from xtcocotools.coco import COCO

from mmpose.datasets.builder import DATASETS
from ..base import BaseCocoStyleDataset
from .pair_hand3d_dataset import PairHand3DDataset


@DATASETS.register_module()
class PairHand3DDatasetSeq(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/nreal_hand.py')
    category_name_list = ['background', 'left_hand', 'right_hand']

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
                 sub_data_index=-1,
                 data_ratio=-1,
                 point_type='3D',
                 filter_kpt_exceed=False,
                 seq_len=4):
        self.flip_left_to_right = flip_left_to_right
        self.data_ratio = data_ratio
        self.data_file_list = data_file_list
        self.lmdb_client = LmdbClient()
        self.dataset_info_list = list()
        self.point_type = point_type
        self.dataset_info_list = list()
        self.dataset_weight_list = dataset_weight_list
        self.dataset_num = len(self.data_file_list)
        self.lmdb_data_root = data_root
        self.with_mask = with_mask
        self.mask_ext = mask_ext
        self.sub_data_index = int(sub_data_index)
        self.cams_info = dict()
        self.seq_len = seq_len
        self.test_mode = test_mode
        self.filter_kpt_exceed = filter_kpt_exceed
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
        ann = raw_data_info['raw_ann_info']
        left_img, right_img = raw_data_info['raw_img_info']

        left_img_id = int(ann['image_id'].split('_')[0])
        right_img_id = int(ann['image_id'].split('_')[1])

        left_img_path = osp.join(self.data_prefix['img'],
                                 left_img['file_name'])
        right_img_path = osp.join(self.data_prefix['img'],
                                  right_img['file_name'])

        left_img_w, left_img_h = left_img['width'], left_img['height']
        right_img_w, right_img_h = right_img['width'], right_img['height']

        def convert_bbox(bbox, img_w, img_h):
            x, y, w, h = bbox
            x1 = np.clip(x, 0, img_w - 1)
            y1 = np.clip(y, 0, img_h - 1)
            x2 = np.clip(x + w, 0, img_w - 1)
            y2 = np.clip(y + h, 0, img_h - 1)
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)
            return bbox

        left_bbox = convert_bbox(ann['bbox_left'], left_img_w, left_img_h)
        right_bbox = convert_bbox(ann['bbox_right'], right_img_w, right_img_h)

        left_keypoints = np.array(ann['keypoints_left'])[..., :2].reshape(
            1, -1, 2)
        right_keypoints = np.array(ann['keypoints_right'])[..., :2].reshape(
            1, -1, 2)

        keypoints3d = np.array(ann['keypoints3d'])[np.newaxis]  # (1,21,3)
        num_keypoints = ann['num_keypoints']
        keypoints_visible = np.array(ann['keypoints_left'])[...,
                                                            2].reshape(1, -1)
        cam_key = ann['camera_instance_id']
        cam_info = self.cams_info[cam_key]
        cam_model_left, cam_model_right = build_from_BinocularCameraInstance(
            cam_info)
        meta = ann.get('meta', dict())
        meta['category_id'] = ann['category_id']
        left_R, right_R, virtual_baseline = \
            PairHand3DDataset.get_virtual_cam(cam_model_left, cam_model_right)
        meta['left_R'] = left_R
        meta['right_R'] = right_R
        meta['virtual_baseline'] = virtual_baseline
        meta['gesture'] = ann['gesture']
        meta['tag'] = ann['tag']
        data_info = {
            'left_img_id': left_img_id,
            'right_img_id': right_img_id,
            'left_img_path': left_img_path,
            'right_img_path': right_img_path,
            'left_keypoints': left_keypoints,
            'right_keypoints': right_keypoints,
            'keypoints3d': keypoints3d,
            'cam_info': cam_info,
            'left_bbox': left_bbox,
            'right_bbox': right_bbox,
            'image_width': left_img_w,
            'image_height': left_img_h,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints_visible': keypoints_visible,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'cat_id': ann['category_id'],
            'cam_model_left': cam_model_left,
            'cam_model_right': cam_model_right,
            'meta': meta
        }

        return data_info

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        image_list = []
        instance_list = []
        filter_annotation_num = 0
        if self.sub_data_index >= 0:
            self.data_file_list = [self.data_file_list[self.sub_data_index]]
        instance_idx = 0
        for i, anno_file in enumerate(self.data_file_list):
            coco = COCO(anno_file)
            lmdb_path = osp.join(self.lmdb_data_root,
                                 coco.dataset['lmdb_path'])
            seq_list = []
            for k, v in coco.dataset['cameras_info'].items():
                self.cams_info[k] = BinocularCameraInstance.from_dict(v)
            ann_ids = coco.getAnnIds()
            for ann_id in ann_ids:
                ann = coco.loadAnns(ann_id)[0]
                left_img_id = int(ann['image_id'].split('_')[0])
                right_img_id = int(ann['image_id'].split('_')[1])
                left_img = coco.loadImgs(left_img_id)[0]
                right_img = coco.loadImgs(right_img_id)[0]
                image_list.append(left_img)
                image_list.append(right_img)
                if self.filter_kpt_exceed:
                    left_keypoints = np.array(
                        ann['keypoints_left'])[..., :2].reshape(-1, 2)
                    right_keypoints = np.array(
                        ann['keypoints_right'])[..., :2].reshape(-1, 2)
                    left_within_bounds = PairHand3DDataset \
                        .is_keypoint_within_bounds(
                            left_keypoints, left_img['width'],
                            left_img['height'])
                    right_within_bounds = PairHand3DDataset \
                        .is_keypoint_within_bounds(
                            right_keypoints, right_img['width'],
                            right_img['height'])
                    if not left_within_bounds or not right_within_bounds:
                        filter_annotation_num += 1
                        continue

                data_info = self.parse_data_info(
                    dict(raw_ann_info=ann, raw_img_info=[left_img, right_img]))
                if self.with_mask:
                    category_name = self.category_name_list[int(
                        data_info['cat_id'])]
                    data_info[
                            'left_mask_path'] = \
                            f"{lmdb_path}_{self.mask_ext}:{data_info['left_img_path']}_{category_name}" # noqa
                    data_info[
                            'right_mask_path'] = \
                            f"{lmdb_path}_{self.mask_ext}:{data_info['right_img_path']}_{category_name}" # noqa
                data_info['left_img_path'] = \
                    f"{lmdb_path}:{data_info['left_img_path']}"
                data_info['right_img_path'] = \
                    f"{lmdb_path}:{data_info['right_img_path']}"

                instance_list.append(data_info)
                seq_list.append(instance_idx)
                instance_idx += 1

            self.dataset_info_list.append(seq_list)

        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(
            f'HandDataset loaded {len(image_list)} images, {len(instance_list)} pair instances'  # noqa
        )

        return instance_list, image_list

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
        results['meta']['flipped'] = True

    def get_data_info(self, idx):
        idx = idx % self.data_num
        data_info = super().get_data_info(idx)
        data_info['left_img'] = self.lmdb_client.get(
            data_info['left_img_path'])
        data_info['right_img'] = self.lmdb_client.get(
            data_info['right_img_path'])
        data_info['meta']['frame_height'] = data_info['left_img'].shape[0]
        data_info['meta']['frame_width'] = data_info['left_img'].shape[1]
        data_info['meta']['flipped'] = False
        return data_info

    def prepare_pair_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        meta_left = copy.deepcopy(data_info['meta'])
        meta_left['ori_camera'] = copy.deepcopy(data_info['cam_model_left'])

        meta_right = copy.deepcopy(data_info['meta'])
        meta_right['ori_camera'] = copy.deepcopy(data_info['cam_model_right'])
        meta_left['cam_to_virtual_R'] = copy.deepcopy(
            data_info['meta']['left_R'])
        meta_right['cam_to_virtual_R'] = copy.deepcopy(
            data_info['meta']['right_R'])
        meta_right['virtual_baseline'] = copy.deepcopy(
            data_info['meta']['virtual_baseline'])

        data_info_left = {
            'img_id': data_info['left_img_id'],
            'image_width': data_info['left_img'].shape[1],
            'image_height': data_info['left_img'].shape[0],
            'img_path': data_info['left_img_path'],
            'keypoints': data_info['left_keypoints'],
            'img': data_info['left_img'],
            'bbox': data_info['left_bbox'],
            'keypoints3d': data_info['keypoints3d'],
            'bbox_score': np.ones(1, dtype=np.float32),
            'iscrowd': data_info['iscrowd'],
            'id': data_info['id'],
            'cat_id': data_info['cat_id'],
            'meta': meta_left,
            'sample_idx': data_info['sample_idx'],
            'upper_body_ids': data_info['upper_body_ids'],
            'lower_body_ids': data_info['lower_body_ids'],
            'flip_pairs': data_info['flip_pairs'],
            'flip_indices': data_info['flip_indices'],
            'keypoints_visible': data_info['keypoints_visible'],
            'camera_name': 'left'
        }

        data_info_right = {
            'img_id': data_info['right_img_id'],
            'image_width': data_info['right_img'].shape[1],
            'image_height': data_info['right_img'].shape[0],
            'img_path': data_info['right_img_path'],
            'keypoints': data_info['right_keypoints'],
            'img': data_info['right_img'],
            'bbox': data_info['right_bbox'],
            'keypoints3d': data_info['keypoints3d'],
            'bbox_score': np.ones(1, dtype=np.float32),
            'iscrowd': data_info['iscrowd'],
            'id': data_info['id'],
            'cat_id': data_info['cat_id'],
            'meta': meta_right,
            'sample_idx': data_info['sample_idx'],
            'upper_body_ids': data_info['upper_body_ids'],
            'lower_body_ids': data_info['lower_body_ids'],
            'flip_pairs': data_info['flip_pairs'],
            'flip_indices': data_info['flip_indices'],
            'keypoints_visible': data_info['keypoints_visible'],
            'camera_name': 'right'
        }

        if self.with_mask:
            data_info_left.update(
                dict(mask=self.lmdb_client.get(data_info['left_mask_path'])))
            data_info_right.update(
                dict(mask=self.lmdb_client.get(data_info['right_mask_path'])))

        if data_info['cat_id'] == 1:
            self.__left_2_right_hand(data_info_left)
            self.__left_2_right_hand(data_info_right)

        ppl_left = self.pipeline(data_info_left)
        ppl_right = self.pipeline(data_info_right)

        return ppl_left, ppl_right

    def get_seq_idx(self, idx) -> list:
        seq_idx = np.random.choice(range(len(self.dataset_info_list)))
        seq_list = self.dataset_info_list[seq_idx]

        seq_list_cur_idx = np.random.choice(range(len(seq_list)))
        if self.test_mode:
            seq_list_cur_idx = idx
        idx_list = []
        for i in range(self.seq_len):
            tmp = max(seq_list_cur_idx - i, 0)
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
        seq_idx_list = self.get_seq_idx(idx)

        for idx in seq_idx_list:
            ppl_left, ppl_right = self.prepare_pair_data(idx)
            collate_list.extend([ppl_left, ppl_right])

        all_results = default_collate(collate_list)

        return all_results
