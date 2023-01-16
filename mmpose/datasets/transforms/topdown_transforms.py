# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union, List
import cv2
import numpy as np
import pickle as pkl
import os
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of
from mmengine.dist import get_world_size, get_rank
from mmengine.logging import MMLogger
from nreal_data_tool import LmdbClient
from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix


@TRANSFORMS.register_module()
class TopdownAffine(BaseTransform):
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints (optional)

    Modified Keys:

        - img
        - bbox_scale

    Added Keys:

        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))
        results['warp_mat'] = warp_mat

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            results['img'] = cv2.warpAffine(
                results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        if results.get('keypoints', None) is not None:
            transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(
                results['keypoints'][..., :2], warp_mat)
            results['transformed_keypoints'] = transformed_keypoints

        results['input_size'] = (w, h)

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str


@TRANSFORMS.register_module()
class RandomBackground(BaseTransform):

    def __init__(self,
                 bg_lmdb_path_list,
                 align_mean=False,
                 bbox_scale=1.0,
                 edge_fuse=False,
                 worker_slice=False) -> None:
        super().__init__()
        self.bg_lmdb_path_list = bg_lmdb_path_list
        self.worker_slice = worker_slice
        self.lmdb_client = LmdbClient()
        self.align_mean = align_mean
        self.bbox_scale = bbox_scale
        self.edge_fuse = edge_fuse
        self.data_list = self.load_data()

    def load_data(self):
        data_list = []
        for lmdb_path in self.bg_lmdb_path_list:
            with open(os.path.join(lmdb_path, 'meta.pkl'), 'rb') as f:
                meta_info = pkl.load(f)
                data_list += [
                    f'{lmdb_path}:{file_name}'
                    for file_name in meta_info['file_name_list']
                ]
        if self.worker_slice:
            world_size = get_world_size()
            batch_num = len(data_list) // world_size
            local_rank = get_rank()
            start_index = batch_num * local_rank
            end_index = min(len(data_list), start_index + batch_num)
            data_list = data_list[start_index:end_index]
            logger: MMLogger = MMLogger.get_current_instance()
            logger.info(
                f'load {len(data_list)} bg images on rank {local_rank}')
        return data_list

    def _apply_gt_with_offset(self, results, offset_x, offset_y):
        kpt = results['keypoints']
        kpt[:, :, 0] = kpt[:, :, 0] + offset_x
        kpt[:, :, 1] = kpt[:, :, 1] + offset_y
        results['keypoints'] = kpt
        bbox = results['bbox']
        bbox[:, ::2] = bbox[:, ::2] + offset_x
        bbox[:, 1::2] = bbox[:, 1::2] + offset_y
        results['bbox'] = bbox
        return results

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        bg_image_path = np.random.choice(self.data_list)
        bg_image = self.lmdb_client.get(bg_image_path)
        mask = results['mask']
        if self.align_mean:
            bg_mean = bg_image.mean()
        x1, y1, x2, y2 = results['bbox'][0].astype(np.int)
        w = x2 - x1
        h = y2 - y1
        img_w = results['image_width']
        img_h = results['image_height']
        if self.bbox_scale > 1:
            x1 = x1 - w * (self.bbox_scale - 1) / 2.0
            y1 = y1 - h * (self.bbox_scale - 1) / 2.0
            w = w * self.bbox_scale
            h = h * self.bbox_scale
            x1 = int(np.clip(x1, 0, img_w - 1))
            y1 = int(np.clip(y1, 0, img_h - 1))
            x2 = int(np.clip(x1 + w, 0, img_w - 1))
            y2 = int(np.clip(y1 + h, 0, img_h - 1))
            w = x2 - x1
            h = y2 - y1
        bg_x1 = np.random.randint(0, img_w - w)
        bg_y1 = np.random.randint(0, img_h - h)
        bg_x2 = bg_x1 + w
        bg_y2 = bg_y1 + h
        offset_x = bg_x1 - x1
        offset_y = bg_y1 - y1

        mask = cv2.resize(mask, (w, h), cv2.INTER_LINEAR)
        if mask.sum() == 0:
            return results
        img = results['img']
        if self.edge_fuse:
            mask[mask > 0] = 255
            hand_weight = mask.copy().astype(np.float32)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(
                hand_weight, contours, -1, (50), 1, lineType=cv2.LINE_AA)
            hand_weight = hand_weight[:, :, np.newaxis]
            hand_weight[hand_weight == 255] = 1.0
            hand_weight[hand_weight == 50] = 0.2
            crop_bg_img = bg_image[bg_y1:bg_y2, bg_x1:bg_x2, :]
            crop_front_img = img[y1:y2, x1:x2, :]
            crop_img = crop_bg_img * (
                1 - hand_weight) + crop_front_img * hand_weight
            bg_image[bg_y1:bg_y2, bg_x1:bg_x2, :] = crop_img
        else:
            bg_image[bg_y1:bg_y2,
                     bg_x1:bg_x2, :][mask > 0] = img[y1:y2, x1:x2, :][mask > 0]
        if self.align_mean:
            hand = bg_image[bg_y1:bg_y2, bg_x1:bg_x2, :][mask > 0]
            bg_image[bg_y1:bg_y2,
                     bg_x1:bg_x2, :][mask > 0] = hand - hand.mean() + bg_mean
        results['img'] = bg_image
        results = self._apply_gt_with_offset(results, offset_x, offset_y)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'bg image number={len(self.data_list)})'
        return repr_str
