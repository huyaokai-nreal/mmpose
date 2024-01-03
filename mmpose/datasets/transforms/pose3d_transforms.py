# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
from mmcv.transforms import BaseTransform
from nreal_data_tool.utils.bbox import kpt_to_bbox
from scipy.spatial.transform import Rotation as R

from mmpose.datasets.datasets.hand.pair_hand3d_dataset import PairHand3DDataset
from mmpose.registry import TRANSFORMS
from mmpose.structures.keypoint import flip_keypoints_custom_center


def convert_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x1 = np.clip(x, 0, img_w - 1)
    y1 = np.clip(y, 0, img_h - 1)
    x2 = np.clip(x + w, 0, img_w - 1)
    y2 = np.clip(y + h, 0, img_h - 1)
    bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)
    return bbox


def left_to_right_hand(keypoints, bbox, width):
    keypoints[:, :, 0] = width - 1 - keypoints[:, :, 0]
    bbox[:, ::2] = width - 1 - bbox[:, ::2]
    min_x = np.min(bbox[:, ::2])
    max_x = np.max(bbox[:, ::2])
    min_y = np.min(bbox[:, 1::2])
    max_y = np.max(bbox[:, 1::2])
    bbox = np.array([[min_x, min_y, max_x, max_y]], np.float32)
    return keypoints, bbox


@TRANSFORMS.register_module()
class RandomStereoParamAug(BaseTransform):

    def __init__(self,
                 prob=0.5,
                 baseline_range=[0, 0],
                 x_angle_range=[0, 0],
                 y_angle_range=[0, 0],
                 z_angle_range=[0, 0]) -> None:
        super().__init__()
        self.prob = prob
        self.baseline_range = deepcopy(baseline_range)
        self.x_angle_range = deepcopy(x_angle_range)
        self.y_angle_range = deepcopy(y_angle_range)
        self.z_angle_range = deepcopy(z_angle_range)

    def transform(self, results):
        """Add disturbance randomly during training and every other data point
        in test mode for the right camera."""
        if results['camera_name'] == 'right' and np.random.rand() < self.prob:
            cam_model_right = deepcopy(results['meta']['ori_camera'])
            keypoints3d = results['keypoints3d']
            delta_baseline = self.choose_baseline()
            x_random_angle = np.random.rand() * (
                self.x_angle_range[1] -
                self.x_angle_range[0]) + self.x_angle_range[0]
            y_random_angle = np.random.rand() * (
                self.y_angle_range[1] -
                self.y_angle_range[0]) + self.y_angle_range[0]
            z_random_angle = np.random.rand() * (
                self.z_angle_range[1] -
                self.z_angle_range[0]) + self.z_angle_range[0]
            delta_R = R.from_euler(
                'ZYX', [z_random_angle, y_random_angle, x_random_angle],
                degrees=True).as_matrix()
            cam_model_right.camera_to_world_xf[:3, :3] = \
                cam_model_right.camera_to_world_xf[:3, :3] @ delta_R
            cam_model_right.camera_to_world_xf[0, 3] += delta_baseline
            right_keypoints = cam_model_right.world_to_eye(keypoints3d[0])
            right_keypoints = cam_model_right.eye_to_window(
                right_keypoints).reshape(1, -1, 2)
            right_keypoints += np.random.normal(0, 1, (right_keypoints.shape))
            right_bbox = kpt_to_bbox(right_keypoints[0])
            right_bbox = convert_bbox(right_bbox, results['image_width'],
                                      results['image_height'])
            cam_model_left = deepcopy(cam_model_right)
            if results['meta']['flipped']:
                width = results['image_width']
                left_to_right_hand(right_keypoints, right_bbox, width)

            cam_model_left.camera_to_world_xf = np.eye(4)
            _, right_R, vir_baseline = PairHand3DDataset.get_virtual_cam(
                cam_model_left, cam_model_right)
            results['meta']['ori_camera'] = cam_model_right
            results['bbox'] = right_bbox
            results['keypoints'] = right_keypoints
            results['meta']['cam_to_virtual_R'] = right_R
            results['meta']['virtual_baseline'] = vir_baseline
            results['meta']['stereo_aug'] = True
        return results


@TRANSFORMS.register_module()
class RandomFlipAroundRoot(BaseTransform):
    """Data augmentation with random horizontal joint flip around a root joint.

    Args:
        keypoints_flip_cfg (dict): Configurations of the
            ``flip_keypoints_custom_center`` function for ``keypoints``. Please
            refer to the docstring of the ``flip_keypoints_custom_center``
            function for more details.
        target_flip_cfg (dict): Configurations of the
            ``flip_keypoints_custom_center`` function for ``lifting_target``.
            Please refer to the docstring of the
            ``flip_keypoints_custom_center`` function for more details.
        flip_prob (float): Probability of flip. Default: 0.5.
        flip_camera (bool): Whether to flip horizontal distortion coefficients.
            Default: ``False``.

    Required keys:
        keypoints
        lifting_target

    Modified keys:
        (keypoints, keypoints_visible, lifting_target, lifting_target_visible,
        camera_param)
    """

    def __init__(self,
                 keypoints_flip_cfg,
                 target_flip_cfg,
                 flip_prob=0.5,
                 flip_camera=False):
        self.keypoints_flip_cfg = keypoints_flip_cfg
        self.target_flip_cfg = target_flip_cfg
        self.flip_prob = flip_prob
        self.flip_camera = flip_camera

    def transform(self, results) -> dict:
        """The transform function of :class:`ZeroCenterPose`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        keypoints = results['keypoints']
        if 'keypoints_visible' in results:
            keypoints_visible = results['keypoints_visible']
        else:
            keypoints_visible = np.ones(keypoints.shape[:-1], dtype=np.float32)
        lifting_target = results['lifting_target']
        if 'lifting_target_visible' in results:
            lifting_target_visible = results['lifting_target_visible']
        else:
            lifting_target_visible = np.ones(
                lifting_target.shape[:-1], dtype=np.float32)

        if np.random.rand() <= self.flip_prob:
            if 'flip_indices' not in results:
                flip_indices = list(range(self.num_keypoints))
            else:
                flip_indices = results['flip_indices']

            # flip joint coordinates
            keypoints, keypoints_visible = flip_keypoints_custom_center(
                keypoints, keypoints_visible, flip_indices,
                **self.keypoints_flip_cfg)
            lifting_target, lifting_target_visible = flip_keypoints_custom_center(  # noqa
                lifting_target, lifting_target_visible, flip_indices,
                **self.target_flip_cfg)

            results['keypoints'] = keypoints
            results['keypoints_visible'] = keypoints_visible
            results['lifting_target'] = lifting_target
            results['lifting_target_visible'] = lifting_target_visible

            # flip horizontal distortion coefficients
            if self.flip_camera:
                assert 'camera_param' in results, \
                    'Camera parameters are missing.'
                _camera_param = deepcopy(results['camera_param'])

                assert 'c' in _camera_param
                _camera_param['c'][0] *= -1

                if 'p' in _camera_param:
                    _camera_param['p'][0] *= -1

                results['camera_param'].update(_camera_param)

        return results
