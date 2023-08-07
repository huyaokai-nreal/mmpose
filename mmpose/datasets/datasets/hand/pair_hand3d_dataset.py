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
from nreal_data_tool.utils.camera import OpenCVFisheyeCameraModel
from scipy.spatial.transform import Rotation as R
from xtcocotools.coco import COCO

from mmpose.datasets.builder import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class PairHand3DDataset(BaseCocoStyleDataset):

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
                 sub_data_index=-1,
                 data_ratio=-1,
                 point_type='3D',
                 camera_type='fisheye'):
        self.flip_left_to_right = flip_left_to_right
        self.data_ratio = data_ratio
        self.data_file_list = data_file_list
        self.lmdb_client = LmdbClient()
        self.point_type = point_type
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

        keypoints3d = ann['keypoints3d']
        num_keypoints = ann['num_keypoints']
        keypoints_visible = np.array(ann['keypoints_left'])[...,
                                                            2].reshape(1, -1)

        data_info = {
            'left_img_id': left_img_id,
            'right_img_id': right_img_id,
            'left_img_path': left_img_path,
            'right_img_path': right_img_path,
            'left_keypoints': left_keypoints,
            'right_keypoints': right_keypoints,
            'keypoints3d': keypoints3d,
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
            'meta': ann.get('meta', dict())
        }

        return data_info

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
            ann_ids = coco.getAnnIds()
            category_name_map = {
                d['id']: d['name']
                for d in coco.dataset['categories']
            }
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
                if self.with_mask:
                    category_name = category_name_map[int(data_info['cat_id'])]
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
        left_camera_matrix = data_info['meta']['cam_matrix_left']
        right_camera_matrix = data_info['meta']['cam_matrix_right']
        # TODO: read from json file
        left_D = [
            0.012542161517124128, 0.04662863296034774, -0.04361866666639336,
            0.009913181928564089
        ]
        right_D = [
            0.01485201999344762, 0.03768701104219142, -0.034698759423003406,
            0.007490907841159389
        ]
        left_camera_to_world = np.eye(4, 4, dtype=np.float32)
        right_camera_to_world = np.eye(4, 4, dtype=np.float32)
        right_camera_to_world[:3, :3] = R.from_quat(
            data_info['meta']['leftcam_q_rightcam']).as_matrix()
        right_camera_to_world[:3, 3] = data_info['meta']['leftcam_p_rightcam']
        left_camera = OpenCVFisheyeCameraModel(
            f=[left_camera_matrix[0][0], left_camera_matrix[1][1]],
            c=[left_camera_matrix[0][2], left_camera_matrix[1][2]],
            distort_coeffs=left_D,
            camera_to_world_xf=left_camera_to_world)
        right_camera = OpenCVFisheyeCameraModel(
            f=[right_camera_matrix[0][0], right_camera_matrix[1][1]],
            c=[right_camera_matrix[0][2], right_camera_matrix[1][2]],
            distort_coeffs=right_D,
            camera_to_world_xf=right_camera_to_world)

        data_info_left = {
            'img_id': data_info['left_img_id'],
            'image_width': data_info['left_img'].shape[1],
            'image_height': data_info['left_img'].shape[0],
            'img_path': data_info['left_img_path'],
            'keypoints': data_info['left_keypoints'],
            'img': data_info['left_img'],
            'bbox': data_info['left_bbox'],
            'cam_matrix_left': data_info['meta']['cam_matrix_left'],
            'cam_matrix_right': data_info['meta']['cam_matrix_right'],
            'leftcam_p_rightcam': data_info['meta']['leftcam_p_rightcam'],
            'leftcam_q_rightcam': data_info['meta']['leftcam_q_rightcam'],
            'keypoints3d': data_info['keypoints3d'],
            'bbox_score': np.ones(1, dtype=np.float32),
            'iscrowd': data_info['iscrowd'],
            'id': data_info['id'],
            'cat_id': data_info['cat_id'],
            'meta': copy.deepcopy(data_info['meta']),
            'sample_idx': data_info['sample_idx'],
            'upper_body_ids': data_info['upper_body_ids'],
            'lower_body_ids': data_info['lower_body_ids'],
            'flip_pairs': data_info['flip_pairs'],
            # 'keypoint_weights': data_info['dataset_keypoint_weights'],
            'flip_indices': data_info['flip_indices'],
            'keypoints_visible': data_info['keypoints_visible']
        }
        data_info_left['meta']['ori_camera'] = left_camera

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
            'meta': copy.deepcopy(data_info['meta']),
            'sample_idx': data_info['sample_idx'],
            'upper_body_ids': data_info['upper_body_ids'],
            'lower_body_ids': data_info['lower_body_ids'],
            'flip_pairs': data_info['flip_pairs'],
            # 'keypoint_weights': data_info['dataset_keypoint_weights'],
            'flip_indices': data_info['flip_indices'],
            'keypoints_visible': data_info['keypoints_visible']
        }
        data_info_right['meta']['ori_camera'] = right_camera

        if self.with_mask:
            data_info_left.update(
                dict(mask=self.lmdb_client.get(data_info['left_mask_path'])))
            data_info_right.update(
                dict(mask=self.lmdb_client.get(data_info['right_mask_path'])))

        if data_info['cat_id'] == 1:
            self.__left_2_right_hand(data_info_left)
            self.__left_2_right_hand(data_info_right)
        if self.test_mode or self.point_type == '3D':
            ppl_left = self.pipeline(data_info_left)
            ppl_right = self.pipeline(data_info_right)
            all_results = default_collate([ppl_left, ppl_right])
        elif self.point_type == 'leftcam':
            all_results = self.pipeline(data_info_left)
        else:
            all_results = self.pipeline(
                random.choice([data_info_left, data_info_right]))
        return all_results
