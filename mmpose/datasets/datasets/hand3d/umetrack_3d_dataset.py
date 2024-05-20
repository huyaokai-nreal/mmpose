# Copyright (c) OpenMMLab. All rights reserved.
import copy
# import os.path as osp
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

from mmengine.dataset.base_dataset import force_full_init
from mmengine.fileio import exists
# from mmengine.utils import is_abs

from mmpose.datasets.datasets import BaseCocoStyleDataset
from mmpose.registry import DATASETS
from mmpose.umelib.batched_dataset.data_transform import preprocess
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
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_mode=data_mode,
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
        # import ipdb;ipdb.set_trace()
        assert exists(self.ann_file), 'Annotation file does not exist'
        fields = ['mono', 'labels']
        datasets = find_dataset(self.ann_file, fields)
        if self.test_mode:
            dataset = datasets[Split['TEST']]
            shuffle = False
        else:
            dataset = datasets[Split['TRAIN']]
            shuffle = False

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
            if index > 10000:
                break
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
            instance_info = self.parse_data_info(
                dict(raw_ann_info=ann, raw_img_info=img))

            # skip invalid instance annotation.
            if not instance_info:
                continue

            instance_list.append(instance_info)
            image_list.append(img)
            index += 1
        return instance_list, image_list

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
        data_info = {
            'extrinsics_xf': ann['extrinsics_xf'],
            'intrinsics': ann['intrinsics'],
            'preds_targets': ann['preds_targets'],
            'gt_skel_targets': ann['gt_skel_targets'],
            'hand_idx': ann['hand_idx'],
            'orig_pose_data': ann['orig_pose_data'],
            's_solved_pose_data': ann['s_solved_pose_data'],
            'img': img,
            'raw_ann_info': copy.deepcopy(ann),
        }
        return data_info
