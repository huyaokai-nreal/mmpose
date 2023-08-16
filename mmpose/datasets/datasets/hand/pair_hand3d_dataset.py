# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import random
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.dataset.base_dataset import force_full_init
from mmengine.dataset.utils import default_collate
from mmengine.logging import MMLogger
from nreal_data_tool import LmdbClient
from nreal_data_tool.utils.camera import (OpenCVFisheyeCameraModel,
                                          OpenCVPinholeCameraModel)
from xtcocotools.coco import COCO

from mmpose.datasets.builder import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class PairHand3DDataset(BaseCocoStyleDataset):

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
                 pinch_random=False):
        self.flip_left_to_right = flip_left_to_right
        self.data_ratio = data_ratio
        self.data_file_list = data_file_list
        self.lmdb_client = LmdbClient()
        self.point_type = point_type
        self.pinch_random = pinch_random
        self.dataset_info_list = list()
        self.dataset_weight_list = dataset_weight_list
        self.dataset_num = len(self.data_file_list)
        self.lmdb_data_root = data_root
        self.with_mask = with_mask
        self.mask_ext = mask_ext
        self.sub_data_index = int(sub_data_index)
        self.cams_info = dict()
        self.enter_thre = 0.02
        self.exit_thre = 0.04
        self.pinch_idx_list = []
        self.no_pinch_idx_list = []
        self.media_idx_list = []
        self.hand_bones_list = list()
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

    @staticmethod
    def get_cam_model(cam_info):
        if cam_info['camera_type'] == 'fisheye':
            CMAERA_MODEL = OpenCVFisheyeCameraModel
        elif cam_info['camera_type'] == 'pinhole':
            CMAERA_MODEL = OpenCVPinholeCameraModel
        else:
            raise NotImplementedError
        left_cam_xf = np.array(cam_info['left_T'])
        right_cam_xf = np.array(cam_info['right_T'])
        cam_model_left = CMAERA_MODEL(
            f=(cam_info['left_K'][0][0], cam_info['left_K'][1][1]),
            c=(cam_info['left_K'][0][2], cam_info['left_K'][1][2]),
            camera_to_world_xf=left_cam_xf,
            distort_coeffs=cam_info['left_D'][0]  # k1,k2,k3,k4
        )
        cam_model_right = CMAERA_MODEL(
            f=(cam_info['right_K'][0][0], cam_info['right_K'][1][1]),
            c=(cam_info['right_K'][0][2], cam_info['right_K'][1][2]),
            camera_to_world_xf=right_cam_xf,
            distort_coeffs=cam_info['right_D'][0])  # k1,k2,k3,k4
        return cam_model_left, cam_model_right

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
        cam_model_left, cam_model_right = self.get_cam_model(cam_info)
        meta = ann.get('meta', dict())
        meta.update(cam_info)
        meta['category_id'] = ann['category_id']
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

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        image_list = []
        instance_list = []
        # sub_dataset_start_id = 0
        if self.sub_data_index >= 0:
            self.data_file_list = [self.data_file_list[self.sub_data_index]]
        for i, anno_file in enumerate(self.data_file_list):
            coco = COCO(anno_file)
            lmdb_path = osp.join(self.lmdb_data_root,
                                 coco.dataset['lmdb_path'])
            # sub_dataset_num = 0
            self.cams_info.update(coco.dataset['cameras_info'])
            keypoints3d_list = []
            ann_ids = coco.getAnnIds()
            for ann_id in ann_ids:
                ann = coco.loadAnns(ann_id)[0]
                left_img_id = int(ann['image_id'].split('_')[0])
                right_img_id = int(ann['image_id'].split('_')[1])
                left_img = coco.loadImgs(left_img_id)[0]
                right_img = coco.loadImgs(right_img_id)[0]
                image_list.append(left_img)
                image_list.append(right_img)

                data_info = self.parse_data_info(
                    dict(raw_ann_info=ann, raw_img_info=[left_img, right_img]))
                keypoints3d_list.append(data_info['keypoints3d'])
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

                data_info['meta']['template_bones_id'] = len(
                    self.hand_bones_list)
                instance_list.append(data_info)

            keypoints3d_list = np.concatenate(keypoints3d_list, axis=0)
            mean_bones = self._get_mean_hand_bones(keypoints3d_list)
            self.hand_bones_list.append(mean_bones)
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(
            f'HandDataset loaded {len(image_list)} images, {len(instance_list)} pair instances'  # noqa
        )
        if self.pinch_random:
            for i, instance in enumerate(instance_list):
                kpt3d = instance['keypoints3d']
                dist = np.linalg.norm(
                    kpt3d[:, 4, :] - kpt3d[:, 8, :], axis=-1).item()
                if dist < self.enter_thre:
                    self.pinch_idx_list.append(i)
                elif dist > self.exit_thre:
                    self.no_pinch_idx_list.append(i)
                else:
                    self.media_idx_list.append(i)

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
        if not self.test_mode and self.pinch_random:  # Uniformly allocate three types of pinch data during training
            idx = random.choice(
                random.choice([
                    self.pinch_idx_list, self.no_pinch_idx_list,
                    self.media_idx_list
                ]))
        data_info = super().get_data_info(idx)
        data_info['left_img'] = self.lmdb_client.get(
            data_info['left_img_path'])
        data_info['right_img'] = self.lmdb_client.get(
            data_info['right_img_path'])
        data_info['meta']['frame_height'] = data_info['left_img'].shape[0]
        data_info['meta']['frame_width'] = data_info['left_img'].shape[1]
        return data_info

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
        data_info = self.get_data_info(idx)
        meta_left = copy.deepcopy(data_info['meta'])
        meta_left['ori_camera'] = copy.deepcopy(data_info['cam_model_left'])
        meta_left['template_bones'] = self.hand_bones_list[
            data_info['meta']['template_bones_id']]

        meta_right = copy.deepcopy(data_info['meta'])
        meta_right['ori_camera'] = copy.deepcopy(data_info['cam_model_right'])
        meta_right['template_bones'] = self.hand_bones_list[
            data_info['meta']['template_bones_id']]

        data_info_left = {
            'img_id': data_info['left_img_id'],
            'image_width': data_info['left_img'].shape[1],
            'image_height': data_info['left_img'].shape[0],
            'img_path': data_info['left_img_path'],
            'keypoints': data_info['left_keypoints'].copy(),
            'img': data_info['left_img'].copy(),
            'bbox': data_info['left_bbox'].copy(),
            'keypoints3d': data_info['keypoints3d'].copy(),
            'bbox_score': np.ones(1, dtype=np.float32),
            'iscrowd': data_info['iscrowd'],
            'id': data_info['id'],
            'cat_id': data_info['cat_id'],
            'meta': meta_left,
            'sample_idx': data_info['sample_idx'],
            'upper_body_ids': data_info['upper_body_ids'],
            'lower_body_ids': data_info['lower_body_ids'],
            'flip_pairs': data_info['flip_pairs'],
            # 'keypoint_weights': data_info['dataset_keypoint_weights'],
            'flip_indices': data_info['flip_indices'],
            'keypoints_visible': data_info['keypoints_visible'].copy()
        }

        data_info_right = {
            'img_id': data_info['right_img_id'],
            'image_width': data_info['right_img'].shape[1],
            'image_height': data_info['right_img'].shape[0],
            'img_path': data_info['right_img_path'],
            'keypoints': data_info['right_keypoints'].copy(),
            'img': data_info['right_img'].copy(),
            'bbox': data_info['right_bbox'].copy(),
            'keypoints3d': data_info['keypoints3d'].copy(),
            'bbox_score': np.ones(1, dtype=np.float32),
            'iscrowd': data_info['iscrowd'],
            'id': data_info['id'],
            'cat_id': data_info['cat_id'],
            'meta': meta_right,
            'sample_idx': data_info['sample_idx'],
            'upper_body_ids': data_info['upper_body_ids'],
            'lower_body_ids': data_info['lower_body_ids'],
            'flip_pairs': data_info['flip_pairs'],
            # 'keypoint_weights': data_info['dataset_keypoint_weights'],
            'flip_indices': data_info['flip_indices'],
            'keypoints_visible': data_info['keypoints_visible'].copy()
        }

        if self.with_mask:
            data_info_left.update(
                dict(mask=self.lmdb_client.get(data_info['left_mask_path'])))
            data_info_right.update(
                dict(mask=self.lmdb_client.get(data_info['right_mask_path'])))
        if self.flip_left_to_right:
            if data_info['cat_id'] == 1:
                self.__left_2_right_hand(data_info_left)
                self.__left_2_right_hand(data_info_right)
        if self.point_type == 'leftcam':
            all_results = self.pipeline(data_info_left)
        elif self.test_mode or self.point_type == '3D':
            ppl_left = self.pipeline(data_info_left)
            ppl_right = self.pipeline(data_info_right)
            all_results = default_collate([ppl_left, ppl_right])
        else:
            all_results = self.pipeline(
                random.choice([data_info_left, data_info_right]))
        return all_results
