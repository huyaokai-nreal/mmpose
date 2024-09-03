# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of
from mmengine.dataset.utils import default_collate
from mmengine.dist import get_rank, get_world_size
from mmengine.logging import MMLogger
from nreal_data_tool import LmdbClient
from nreal_data_tool.utils.camera import PinholePlaneCameraModel

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix
from .pcl import gen_crop_parameters_from_points, warp_image
from .pcl_ume import gen_crop_parameters_from_points as gen_ume_virutal_cam


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
            if results['meta']['flipped']:
                results['img'] = [
                    np.flip(img, axis=1) for img in results['img']
                ]
        else:
            results['img'] = cv2.warpAffine(
                results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)
            if results['meta']['flipped']:
                results['img'] = np.flip(results['img'], axis=1)

        if results.get('keypoints', None) is not None:
            transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(
                results['keypoints'][..., :2], warp_mat)
            if results['meta']['flipped']:
                transformed_keypoints[...,
                                      0] = w - 1 - transformed_keypoints[...,
                                                                         0]
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
    """replace the background with reandom images Required Keys:

        - img
        - bbox
        - mask
        - image_width
        - image_height

    Modified Keys:

        - img
        - bbox (optional)
        - keypoints (optional)
        - image_width (optional)
        - image_height (optional)

    Args:
        bg_lmdb_path_list (List[str]): background image lmdb path list
        prob (float): probability  to apply this transform
        align_mean (bool): whether make the hand mean pixel same
        with background
        bbox_scale (float): scale to expand the bbox for mask,
        same with mask generation parameters
        edge_fuse (bool): whether fuse the edge between hand
        and background
        worker_slice (bool): whether load different background
        images on different workers
        keep_original_pos (bool): if True, the output image will
        have the same shape with original
        image and the bbox, keypoints will not be changed,
        otherwise, the output image will have
        the same shape with background image and keypoints,
        bbox will be changed randomly.
    """

    def __init__(self,
                 bg_lmdb_path_list: List[str],
                 prob: float = 0.5,
                 align_mean: bool = False,
                 bbox_scale: float = 1.5,
                 edge_fuse: bool = False,
                 worker_slice: bool = False,
                 keep_original_pos: bool = False) -> None:
        super().__init__()
        self.bg_lmdb_path_list = bg_lmdb_path_list
        self.prob = prob
        self.worker_slice = worker_slice
        self.lmdb_client = LmdbClient()
        self.align_mean = align_mean
        self.bbox_scale = bbox_scale
        self.edge_fuse = edge_fuse
        self.keep_original_pos = keep_original_pos
        self.data_list = self.load_data()

    def load_data(self) -> List[str]:
        data_list = []
        for lmdb_path in self.bg_lmdb_path_list:
            with open(os.path.join(lmdb_path, 'meta.json'), 'r') as f:
                meta_info = json.load(f)
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

    def _apply_gt_with_offset(self, results: Dict, offset_x: float,
                              offset_y: float) -> Dict:
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
        if np.random.rand() > self.prob:
            return results
        if 'mask' not in results:
            return results
        bg_image_path = np.random.choice(self.data_list)
        bg_image = self.lmdb_client.get(bg_image_path)
        bg_img_w = bg_image.shape[1]
        bg_img_h = bg_image.shape[0]
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
        bg_x1 = np.random.randint(0, bg_img_w - w)
        bg_y1 = np.random.randint(0, bg_img_h - h)
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
        if self.keep_original_pos:
            new_image = np.zeros_like(img)
            new_image[y1:y2, x1:x2] = bg_image[bg_y1:bg_y2, bg_x1:bg_x2]
            results['img'] = new_image
        else:
            results = self._apply_gt_with_offset(results, offset_x, offset_y)
            results['img'] = bg_image
            results['image_width'] = bg_image.shape[1]
            results['image_height'] = bg_image.shape[0]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'bg image number = {len(self.data_list)}\n'
        repr_str += f'prob = {self.prob}\n'
        repr_str += f'worker_slice={self.worker_slice}\n'
        repr_str += f'aligen_mean={self.align_mean}\n'
        repr_str += f'edge_fuse={self.edge_fuse}\n'
        repr_str += f'keep original position={self.keep_original_pos}\n'
        return repr_str


@TRANSFORMS.register_module()
class AffineTransformConsistency(BaseTransform):

    def __init__(self, trans_cfg_list) -> None:
        super().__init__()
        self.trans_list = []
        for cfg in trans_cfg_list:
            self.trans_list.append(TRANSFORMS.build(cfg))

    def _transform(self, results):
        for trans in self.trans_list:
            results = trans(results)
        return results

    def transform(self, results: Dict) -> Dict:
        results_copy = copy.deepcopy(results)
        aug_results_1 = self._transform(results)
        aug_results_2 = self._transform(results_copy)
        all_results = default_collate([aug_results_1, aug_results_2])
        return all_results


@TRANSFORMS.register_module()
class GenerateAttrLabel(BaseTransform):

    def __init__(self,
                 attr_list: List[str],
                 num_class: Optional[int] = None) -> None:
        super().__init__()
        self.attr_list = copy.deepcopy(attr_list)
        self.num_class = num_class

    def transform(self, results: Dict) -> Dict:
        attr_labels = []
        if 'keypoints_visible' in results:
            attr_labels.append(results['keypoints_visible'])
        if 'cat_id' in results:
            attr_label = np.expand_dims(
                np.eye(self.num_class)[results['cat_id']], axis=0)
            attr_labels.append(attr_label)
        attr_labels = np.concatenate(attr_labels, axis=1)
        results['attr_labels'] = attr_labels
        return results


@TRANSFORMS.register_module()
class RandomDownSampleImage(BaseTransform):

    def __init__(self, prob, min_ratio) -> None:
        super().__init__()
        self.prob = prob
        self.min_ratio = min_ratio

    def transform(self, results: Dict) -> Dict:
        if np.random.rand() < self.prob:
            sample_ratio = np.random.uniform(self.min_ratio, 1)
            h, w = results['img'].shape[:2]
            down_image = cv2.resize(results['img'], (int(
                w * sample_ratio), int(h * sample_ratio)), cv2.INTER_NEAREST)
            results['img'] = cv2.resize(down_image, (w, h), cv2.INTER_CUBIC)
        return results


@TRANSFORMS.register_module()
class MixTwoHands(BaseTransform):

    def __init__(self, prob) -> None:
        super().__init__()
        self.prob = prob

    def transform(self, results: Dict) -> Dict:
        if np.random.rand() < self.prob:
            image = results['img'].copy()
            image_h, image_w = image.shape
            x1 = int(np.min(results['transformed_keypoints'][..., 0]))
            y1 = int(np.min(results['transformed_keypoints'][..., 1]))
            x2 = int(np.max(results['transformed_keypoints'][..., 0]))
            y2 = int(np.max(results['transformed_keypoints'][..., 1]))
            crop_image_w = x2 - x1
            crop_image_h = y2 - y1
            hand_crop_image = image[y1:y2, x1:x2]
            flip_image = hand_crop_image[:, ::-1]
            rand_right_up_y = np.random.randint(0, image_h // 2 - 10)
            rand_right_up_x = min(
                np.random.randint(10, image_w // 2), crop_image_w - 10)
            max_y = min(image_h, rand_right_up_y + crop_image_h)
            try:
                image[rand_right_up_y:max_y, 0:rand_right_up_x] = image[
                    rand_right_up_y:max_y,
                    0:rand_right_up_x] * 0.5 + flip_image[:max_y -
                                                          rand_right_up_y,
                                                          crop_image_w -
                                                          rand_right_up_x:
                                                          crop_image_w] * 0.5
            except:  # noqa
                return results
            results['img'] = image
        return results


@TRANSFORMS.register_module()
class TopdownPCL(BaseTransform):

    def __init__(self,
                 input_size: Tuple[int, int],
                 root_id: int = 0,
                 norm_depth: bool = False) -> None:
        self.input_size = input_size
        self.root_id = root_id
        self.norm_depth = norm_depth

    def transform(self, results: Dict) -> Dict:
        w, h = self.input_size
        with_depth = 'root_depth' in results['meta']  # 已过KeypointTo25DLabel
        results['input_size'] = self.input_size
        results['bbox_scale'] = TopdownAffine._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)
        ori_camera = results['meta']['ori_camera']
        if not with_depth:
            results['keypoints3d'][0] = results['meta'][
                'ori_camera'].world_to_eye(results['keypoints3d'][0]).copy()
            results['meta']['ori_xf'] = results['meta'][
                'ori_camera'].camera_to_world_xf
            results['meta']['ori_camera'].camera_to_world_xf = np.eye(4)
        world_points = results['keypoints3d'][0]
        center = results['bbox_center'][0]
        scale = self.input_size[0] / results['bbox_scale'][0][0]
        camera_angle = results['meta'].get('camera_angle', 0)
        try:
            virtual_camera: PinholePlaneCameraModel = \
                gen_crop_parameters_from_points(
                    ori_camera,
                    center,
                    self.input_size,
                    mirror_img_x=False,
                    camera_angle=camera_angle,
                    focal_multiplier=scale)
        except Exception:
            virtual_camera: PinholePlaneCameraModel = gen_ume_virutal_cam(
                ori_camera,
                results['keypoints3d'][0],
                self.input_size,
                mirror_img_x=False,
                camera_angle=camera_angle,
                focal_multiplier=0.8)
        image = results['img']
        crop_img = warp_image(ori_camera, virtual_camera, w, h, image)
        results['img'] = crop_img
        results['meta']['virtual_camera'] = virtual_camera
        kpt3d_in_virutal = virtual_camera.world_to_eye(world_points)
        if results['cat_id'] == 1 and results['meta']['flipped']:
            results['img'] = np.flip(results['img'], axis=1)
            kpt3d_in_virutal[..., 0] *= -1
        warp_keypoints = virtual_camera.eye_to_window(kpt3d_in_virutal)
        results['transformed_keypoints'] = results['keypoints'].copy()
        results['transformed_keypoints'][..., :2] = warp_keypoints
        if self.root_id == 'mean':
            root_depth = kpt3d_in_virutal[:, 2].mean()
        else:
            root_depth = kpt3d_in_virutal[self.root_id, 2]
        if with_depth:
            results['transformed_keypoints'][
                ..., 2] = kpt3d_in_virutal[..., 2] - root_depth
            if self.norm_depth:
                results['transformed_keypoints'][
                    ..., -1] /= results['meta']['hand_scale']
                results['meta']['norm_depth'] = True
            results['keypoints'][...,
                                 2] = results['transformed_keypoints'][..., 2]
            results['meta']['root_depth'] = root_depth
        results['warp_mat'] = np.array([[1, 0, 0], [0, 1, 0]],
                                       dtype=np.float32)
        return results


@TRANSFORMS.register_module()
class UmePCL(BaseTransform):

    def __init__(self, input_size: Tuple[int, int], root_id: int = 0) -> None:
        self.input_size = input_size
        self.root_id = root_id

    def transform(self, results: Dict) -> Dict:
        w, h = self.input_size
        mirror_img_x = results['meta']['ume'] and results['cat_id'] == 1
        if not results['meta']['flipped'] and mirror_img_x:
            results['meta']['flipped'] = True
        if results['meta']['flipped']:
            results['keypoints3d'][..., 0] = -results['keypoints3d'][..., 0]
        results['input_size'] = self.input_size
        results['bbox_scale'] = TopdownAffine._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)
        results['meta']['ori_xf'] = results['meta'][
            'ori_camera'].camera_to_world_xf
        results['meta']['ori_camera'].camera_to_world_xf = np.eye(4)
        ori_camera = results['meta']['ori_camera']
        world_points = results['keypoints3d'][0]
        center = results['bbox_center'][0]
        scale = self.input_size[0] / results['bbox_scale'][0][0]
        camera_angle = 90 if results['meta']['ume'] else 0
        virtual_camera: PinholePlaneCameraModel = \
            gen_crop_parameters_from_points(
                ori_camera,
                center,
                self.input_size,
                mirror_img_x=mirror_img_x,
                camera_angle=camera_angle,
                focal_multiplier=scale)
        image = results['img']
        crop_img = warp_image(ori_camera, virtual_camera, w, h, image)
        # cv2.imwrite('/home/ykhu/workspace/mmpose/zz.png', image)
        # cv2.imwrite('/home/ykhu/workspace/mmpose/zz1.png', crop_img)
        # import ipdb;ipdb.set_trace()
        results['img'] = crop_img
        results['meta']['virtual_camera'] = virtual_camera
        kpt3d_in_virutal = virtual_camera.world_to_eye(world_points)
        warp_keypoints = virtual_camera.eye_to_window(kpt3d_in_virutal)
        results['transformed_keypoints'] = results['keypoints'].copy()
        results['transformed_keypoints'][..., :2] = warp_keypoints
        results['warp_mat'] = np.array([[1, 0, 0], [0, 1, 0]],
                                       dtype=np.float32)
        return results
