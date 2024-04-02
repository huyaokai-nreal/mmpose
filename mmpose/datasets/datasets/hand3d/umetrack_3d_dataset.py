# Copyright (c) OpenMMLab. All rights reserved.
import copy
# import json
import os.path as osp
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

# import numpy as np
from mmengine.dataset.base_dataset import force_full_init
# from mmengine.fileio import exists, get_local_path
from mmengine.fileio import exists
from mmengine.utils import is_abs

# from mmpose.codecs.utils import camera_to_pixel
from mmpose.datasets.datasets import BaseCocoStyleDataset
from mmpose.registry import DATASETS
# from mmpose.structures.bbox import bbox_xywh2xyxy
# from mmpose.umelib.batched_dataset.data_transform import ModelInput,
# preprocess
from mmpose.umelib.batched_dataset.data_transform import preprocess
# from mmpose.umelib.data_utils.async_dataset import (AsyncToIterableDataset,
#                                                     Sampler, find_dataset,
#                                                     subsample)
from mmpose.umelib.data_utils.async_dataset import (AsyncToIterableDataset,
                                                    Sampler, find_dataset)
from mmpose.umelib.data_utils.dataset_util import map_dataset
from mmpose.umelib.data_utils.split import Split

# from xtcocotools.coco import COCO


@DATASETS.register_module()
class Umetrack3DDataset(BaseCocoStyleDataset):
    """Umetrack dataset for 3d hand."""

    METAINFO: dict = dict(from_file='configs/_base_/datasets/interhand3d.py')

    def __init__(self,
                 ann_file: str = '',
                 camera_param_file: str = '',
                 joint_file: str = '',
                 use_gt_root_depth: bool = True,
                 rootnet_result_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 data_ratio=0.,
                 max_refetch: int = 1000):

        self.data_ratio = data_ratio
        _ann_file = ann_file
        if not is_abs(_ann_file):
            _ann_file = osp.join(data_root, _ann_file)
        assert exists(_ann_file), 'Annotation file does not exist.'
        self.ann_file = _ann_file

        _camera_param_file = camera_param_file
        if not is_abs(_camera_param_file):
            _camera_param_file = osp.join(data_root, _camera_param_file)
        assert exists(_camera_param_file), 'Camera file does not exist.'

        self.camera_param_file = _camera_param_file

        _joint_file = joint_file
        if not is_abs(_joint_file):
            _joint_file = osp.join(data_root, _joint_file)
        assert exists(_joint_file), 'Joint file does not exist.'
        self.joint_file = _joint_file

        self.use_gt_root_depth = use_gt_root_depth
        if not self.use_gt_root_depth:
            assert rootnet_result_file is not None
            _rootnet_result_file = rootnet_result_file
            if not is_abs(_rootnet_result_file):
                _rootnet_result_file = osp.join(data_root,
                                                _rootnet_result_file)
            assert exists(
                _rootnet_result_file), 'Rootnet result file does not exist.'
            self.rootnet_result_file = _rootnet_result_file

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_mode=data_mode,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

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

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in COCO format."""
        assert exists(self.ann_file), 'Annotation file does not exist'
        dataset_names = ['real', 'synthetic']
        datasets_all = [osp.join(self.ann_file, s) for s in dataset_names]

        fields = ['mono', 'labels']
        datasets = find_dataset(datasets_all, fields)
        # import ipdb;ipdb.set_trace()
        # for k, v in datasets.items():
        # import ipdb;ipdb.set_trace()

        if self.test_mode:
            dataset = datasets[Split['TEST']]
            shuffle = False
        else:
            dataset = datasets[Split['TRAIN']]
            shuffle = True

        # import ipdb;ipdb.set_trace()
        sampler = Sampler(
            dataset, shuffle=shuffle, drop_last=True, distrib_info=(0, 1))
        iterable_dataset = AsyncToIterableDataset(
            dataset,
            sampler,
            max_prefetch=64,
        )

        iterable_dataset = map_dataset(
            partial(preprocess, crop_size=(96, 96)), iterable_dataset)
        image_list = []
        instance_list = []
        index = 0
        for data in iterable_dataset:
            if index > 1:
                break
            # import ipdb;ipdb.set_trace()
            ann = {
                'extrinsics_xf': data[1].extrinsics_xf,
                'intrinsics': data[1].intrinsics,
                'preds_targets': data[1].preds_targets,
                'gt_skel_targets': data[1].gt_skel_targets,
                'hand_idx': data[0].hand_idx,
                'orig_pose_data': data[0].orig_pose_data,
                's_solved_pose_data': data[0].s_solved_pose_data,
            }
            img = data[0].left_images
            # import ipdb;ipdb.set_trace()

            instance_info = self.parse_data_info(
                dict(raw_ann_info=ann, raw_img_info=img))

            # skip invalid instance annotation.
            if not instance_info:
                continue

            instance_list.append(instance_info)
            image_list.append(img)
            index += 1
        return instance_list, image_list

        # with get_local_path(self.ann_file) as local_path:
        #     self.coco = COCO(local_path)
        # # set the metainfo about categories, which is a list of dict
        # # and each dict contains the 'id', 'name', etc. about this category
        # if 'categories' in self.coco.dataset:
        #     self._metainfo['CLASSES'] = self.coco.loadCats(
        #         self.coco.getCatIds())

        # with get_local_path(self.camera_param_file) as local_path:
        #     with open(local_path, 'r') as f:
        #         self.cameras = json.load(f)
        # with get_local_path(self.joint_file) as local_path:
        #     with open(local_path, 'r') as f:
        #         self.joints = json.load(f)

        # instance_list = []
        # image_list = []

        # for idx, img_id in enumerate(self.coco.getImgIds()):
        #     # if idx>10:
        #     #     break
        #     img = self.coco.loadImgs(img_id)[0]
        #     #import ipdb;ipdb.set_trace()
        #     img.update({
        #         'img_id':
        #         img_id,
        #         'img_path':
        #         osp.join(self.data_prefix['img'], img['file_name']),
        #     })
        #     image_list.append(img)

        #     ann_ids = self.coco.getAnnIds(imgIds=img_id)
        #     ann = self.coco.loadAnns(ann_ids)[0]

        #     instance_info = self.parse_data_info(
        #         dict(raw_ann_info=ann, raw_img_info=img))

        #     # skip invalid instance annotation.
        #     if not instance_info:
        #         continue

        #     instance_list.append(instance_info)
        # return instance_list, image_list

    # def load_data_list(self) -> List[dict]:
    #     instance_list = self._load_annotations()
    #     data_list = self._get_topdown_data_infos(instance_list)
    #     return data_list

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

        # if not self.use_gt_root_depth:
        #     rootnet_result = {}
        #     with get_local_path(self.rootnet_result_file) as local_path:
        #         rootnet_annot = json.load(local_path)
        #     for i in range(len(rootnet_annot)):
        #         rootnet_result[str(
        #             rootnet_annot[i]['annot_id'])] = rootnet_annot[i]

        # num_keypoints = self.metainfo['num_keypoints']

        # capture_id = str(img['capture'])
        # camera_name = img['camera']
        # frame_idx = str(img['frame_idx'])
        # camera_pos = np.array(
        #     self.cameras[capture_id]['campos'][camera_name], \
        #         dtype=np.float32)
        # camera_rot = np.array(
        #     self.cameras[capture_id]['camrot'][camera_name], \
        #         dtype=np.float32)
        # focal = np.array(
        #     self.cameras[capture_id]['focal'][camera_name], \
        #         dtype=np.float32)
        # principal_pt = np.array(
        #     self.cameras[capture_id]['princpt'][camera_name], \
        #         dtype=np.float32)
        # joint_world = np.array(
        #     self.joints[capture_id][frame_idx]['world_coord'],
        #     dtype=np.float32)
        # joint_valid = np.array(ann['joint_valid'], \
        #     dtype=np.float32).flatten()

        # keypoints_cam = np.dot(
        #     camera_rot,
        #     joint_world.transpose(1, 0) -
        #     camera_pos.reshape(3, 1)).transpose(1, 0)

        # if self.use_gt_root_depth:
        #     bbox_xywh = np.array(ann['bbox'], dtype=np.float32).reshape(1, 4)
        #     abs_depth = [keypoints_cam[20, 2], keypoints_cam[41, 2]]
        # else:
        #     rootnet_ann_data = rootnet_result[str(ann['id'])]
        #     bbox_xywh = np.array(
        #         rootnet_ann_data['bbox'], dtype=np.float32).reshape(1, 4)
        #     abs_depth = rootnet_ann_data['abs_depth']
        # bbox = bbox_xywh2xyxy(bbox_xywh)

        # # 41: 'l_wrist', left hand root
        # # 20: 'r_wrist', right hand root
        # rel_root_depth = keypoints_cam[41, 2] - keypoints_cam[20, 2]
        # # if root is not valid, root-relative 3D depth is also invalid.
        # rel_root_valid = joint_valid[20] * joint_valid[41]

        # # if root is not valid -> root-relative 3D pose is also not valid.
        # # Therefore, mark all joints as invalid
        # joint_valid[:20] *= joint_valid[20]
        # joint_valid[21:] *= joint_valid[41]

        # joints_3d_visible = np.minimum(1,
        #                                joint_valid.reshape(-1,
        #                                                    1)).reshape(1, -1)
        # keypoints_img = camera_to_pixel(
        #     keypoints_cam,
        #     focal[0],
        #     focal[1],
        #     principal_pt[0],
        #     principal_pt[1],
        #     shift=True)[..., :2]
        # joints_3d = np.zeros((keypoints_cam.shape[-2], 3),
        #                      dtype=np.float32).reshape(1, -1, 3)
        # joints_3d[..., :2] = keypoints_img
        # joints_3d[..., :21,
        #           2] = keypoints_cam[..., :21, 2] - keypoints_cam[..., 20, 2]
        # joints_3d[..., 21:,
        #           2] = keypoints_cam[..., 21:, 2] - keypoints_cam[..., 41, 2]

        data_info = {
            'extrinsics_xf': ann['extrinsics_xf'],
            'intrinsics': ann['intrinsics'],
            'preds_targets': ann['preds_targets'],
            'gt_skel_targets': ann['gt_skel_targets'],
            'hand_idx': ann['hand_idx'],
            'orig_pose_data': ann['orig_pose_data'],
            's_solved_pose_data': ann['s_solved_pose_data'],
            # 'img_id': ann['image_id'],
            # 'img_path': img['img_path'],
            # 'rotation': 0,
            # 'keypoints': joints_3d,
            # 'keypoints_cam': keypoints_cam.reshape(1, -1, 3),
            # 'keypoints_visible': joints_3d_visible,
            # 'hand_type': self.encode_handtype(ann['hand_type']),
            # 'hand_type_valid': np.array([ann['hand_type_valid']]),
            # 'rel_root_depth': rel_root_depth,
            # 'rel_root_valid': rel_root_valid,
            # 'abs_depth': abs_depth,
            # 'focal': focal,
            # 'principal_pt': principal_pt,
            # 'dataset': self.metainfo['dataset_name'],
            # 'bbox': bbox,
            # 'bbox_score': np.ones(1, dtype=np.float32),
            # 'num_keypoints': num_keypoints,
            # 'iscrowd': ann.get('iscrowd', False),
            # 'id': ann['id'],
            'img': img,

            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            'raw_ann_info': copy.deepcopy(ann),
        }
        return data_info
