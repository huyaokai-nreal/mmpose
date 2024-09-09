# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.registry import VISUALIZERS
from mmpose.structures import PoseDataSample
from .opencv_backend_visualizer import OpencvBackendVisualizer
from .simcc_vis import SimCCVisualizer


def _get_adaptive_scales(areas: np.ndarray,
                         min_area: int = 800,
                         max_area: int = 30000) -> np.ndarray:
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``min_area``, the scale is 0.5 while the area is larger than
    ``max_area``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Defaults to 800.
        max_area (int): Upper bound areas for adaptive scales.
            Defaults to 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


@VISUALIZERS.register_module()
class PoseLocalVisualizer(OpencvBackendVisualizer):
    """MMPose Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to ``None``
        vis_backends (list, optional): Visual backend config list. Defaults to
            ``None``
        save_dir (str, optional): Save file dir for all storage backends.
            If it is ``None``, the backend storage will not save any data.
            Defaults to ``None``
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to ``'green'``
        kpt_color (str, tuple(tuple(int)), optional): Color of keypoints.
            The tuple of color should be in BGR order. Defaults to ``'red'``
        link_color (str, tuple(tuple(int)), optional): Color of skeleton.
            The tuple of color should be in BGR order. Defaults to ``None``
        line_width (int, float): The width of lines. Defaults to 1
        radius (int, float): The radius of keypoints. Defaults to 4
        show_keypoint_weight (bool): Whether to adjust the transparency
            of keypoints according to their score. Defaults to ``False``
        alpha (int, float): The transparency of bboxes. Defaults to ``1.0``

    Examples:
        >>> import numpy as np
        >>> from mmengine.structures import InstanceData
        >>> from mmpose.structures import PoseDataSample
        >>> from mmpose.visualization import PoseLocalVisualizer

        >>> pose_local_visualizer = PoseLocalVisualizer(radius=1)
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_instances = InstanceData()
        >>> gt_instances.keypoints = np.array([[[1, 1], [2, 2], [4, 4],
        ...                                          [8, 8]]])
        >>> gt_pose_data_sample = PoseDataSample()
        >>> gt_pose_data_sample.gt_instances = gt_instances
        >>> dataset_meta = {'skeleton_links': [[0, 1], [1, 2], [2, 3]]}
        >>> pose_local_visualizer.set_dataset_meta(dataset_meta)
        >>> pose_local_visualizer.add_datasample('image', image,
        ...                         gt_pose_data_sample)
        >>> pose_local_visualizer.add_datasample(
        ...                       'image', image, gt_pose_data_sample,
        ...                        out_file='out_file.jpg')
        >>> pose_local_visualizer.add_datasample(
        ...                        'image', image, gt_pose_data_sample,
        ...                         show=True)
        >>> pred_instances = InstanceData()
        >>> pred_instances.keypoints = np.array([[[1, 1], [2, 2], [4, 4],
        ...                                       [8, 8]]])
        >>> pred_instances.score = np.array([0.8, 1, 0.9, 1])
        >>> pred_pose_data_sample = PoseDataSample()
        >>> pred_pose_data_sample.pred_instances = pred_instances
        >>> pose_local_visualizer.add_datasample('image', image,
        ...                         gt_pose_data_sample,
        ...                         pred_pose_data_sample)
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = 'green',
                 kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = 'red',
                 link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (255, 255, 255),
                 skeleton: Optional[Union[List, Tuple]] = None,
                 line_width: Union[int, float] = 1,
                 radius: Union[int, float] = 3,
                 show_keypoint_weight: bool = False,
                 backend: str = 'opencv',
                 alpha: float = 1.0):
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            backend=backend)

        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.text_color = text_color
        self.skeleton = skeleton
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight
        # Set default value. When calling
        # `PoseLocalVisualizer().set_dataset_meta(xxx)`,
        # it will override the default value.
        self.dataset_meta = {}

    def set_dataset_meta(self,
                         dataset_meta: Dict,
                         skeleton_style: str = 'mmpose'):
        """Assign dataset_meta to the visualizer. The default visualization
        settings will be overridden.

        Args:
            dataset_meta (dict): meta information of dataset.
        """
        if dataset_meta.get(
                'dataset_name') == 'coco' and skeleton_style == 'openpose':
            dataset_meta = parse_pose_metainfo(
                dict(from_file='configs/_base_/datasets/coco_openpose.py'))

        if isinstance(dataset_meta, dict):
            self.dataset_meta = dataset_meta.copy()
            self.bbox_color = dataset_meta.get('bbox_color', self.bbox_color)
            self.kpt_color = dataset_meta.get('keypoint_colors',
                                              self.kpt_color)
            self.link_color = dataset_meta.get('skeleton_link_colors',
                                               self.link_color)
            self.skeleton = dataset_meta.get('skeleton_links', self.skeleton)
        # sometimes self.dataset_meta is manually set, which might be None.
        # it should be converted to a dict at these times
        if self.dataset_meta is None:
            self.dataset_meta = {}

    def _draw_instances_bbox(self, image: np.ndarray,
                             instances: InstanceData) -> np.ndarray:
        """Draw bounding boxes and corresponding labels of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances:
            bboxes = instances.bboxes
            self.draw_bboxes(
                bboxes,
                edge_colors=self.bbox_color,
                alpha=self.alpha,
                line_widths=self.line_width)
        else:
            return self.get_image()

        if 'labels' in instances and self.text_color is not None:
            classes = self.dataset_meta.get('classes', None)
            labels = instances.labels

            positions = bboxes[:, :2]
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                if isinstance(label, int):
                    label_text = classes[
                        label] if classes is not None else f'class {label}'
                elif isinstance(label, str):
                    label_text = label

                if isinstance(self.bbox_color,
                              tuple) and max(self.bbox_color) > 1:
                    facecolor = [c / 255.0 for c in self.bbox_color]
                else:
                    facecolor = self.bbox_color

                self.draw_texts(
                    label_text,
                    pos,
                    colors=self.text_color,
                    font_sizes=int(13 * scales[i]),
                    vertical_alignments='bottom',
                    bboxes=[{
                        'facecolor': facecolor,
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        return self.get_image()

    def _draw_3d_data_samples(
        self,
        image: np.ndarray,
        pose_samples: PoseDataSample,
        draw_gt: bool = True,
        kpt_thr: float = 0.3,
        num_instances=-1,
        axis_azimuth: float = 70,
        axis_dist: float = 10.0,
        axis_elev: float = 15.0,
    ):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            draw_gt (bool): Whether to draw GT PoseDataSample. Default to
                ``True``
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            num_instances (int): Number of instances to be shown in 3D. If
                smaller than 0, all the instances in the pose_result will be
                shown. Otherwise, pad or truncate the pose_result to a length
                of num_instances.
            axis_azimuth (float): axis azimuth angle for 3D visualizations.
            axis_dist (float): axis distance for 3D visualizations.
            axis_elev (float): axis elevation view angle for 3D visualizations.
            axis_limit (float): The axis limit to visualize 3d pose. The xyz
                range will be set as:
                - x: [x_c - axis_limit/2, x_c + axis_limit/2]
                - y: [y_c - axis_limit/2, y_c + axis_limit/2]
                - z: [0, axis_limit]
                Where x_c, y_c is the mean value of x and y coordinates

        Returns:
            Tuple(np.ndarray): the drawn image which channel is RGB.
        """
        vis_height, vis_width, _ = image.shape

        if 'pred_instances' in pose_samples:
            pred_instances = pose_samples.pred_instances
        else:
            pred_instances = InstanceData()
        if num_instances < 0:
            if 'keypoints' in pred_instances:
                num_instances = len(pred_instances)
            else:
                num_instances = 0
        else:
            if len(pred_instances) > num_instances:
                for k in pred_instances.keys():
                    new_val = pred_instances.k[:num_instances]
                    pose_samples.pred_instances.k = new_val
            elif num_instances < len(pred_instances):
                num_instances = len(pred_instances)

        num_fig = num_instances
        if draw_gt:
            vis_width *= 2
            num_fig *= 2

        plt.ioff()
        fig = plt.figure(
            figsize=(vis_width * num_instances * 0.01, vis_height * 0.01))

        def _draw_3d_instances_kpts(keypoints,
                                    scores,
                                    keypoints_visible,
                                    fig_idx,
                                    title=None):

            for idx, (kpts, score, visible) in enumerate(
                    zip(keypoints, scores, keypoints_visible)):
                ax = fig.add_subplot(
                    1, num_fig, fig_idx * (idx + 1), projection='3d')
                ax.view_init(elev=axis_elev, azim=axis_azimuth)
                ax.set_aspect('auto')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.scatter([0], [0], [0], marker='o', color='red')
                if title:
                    ax.set_title(f'{title} ({idx})')
                ax.dist = axis_dist
                kpts = np.array(kpts, copy=False)
                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f'the length of kpt_color '
                        f'({len(self.kpt_color)}) does not matches '
                        f'that of keypoints ({len(kpts)})')
                x_3d, y_3d, z_3d = np.split(kpts[:, :3], [1, 2], axis=1)
                max_range = np.array([
                    x_3d.max() - x_3d.min(),
                    y_3d.max() - y_3d.min(),
                    z_3d.max() - z_3d.min()
                ]).max() / 2.0

                mid_x = (x_3d.max() + x_3d.min()) * 0.5
                mid_y = (y_3d.max() + y_3d.min()) * 0.5
                mid_z = (z_3d.max() + z_3d.min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

                kpt_color = kpt_color[..., ::-1] / 255.

                ax.scatter(x_3d, y_3d, z_3d, marker='o', color=kpt_color)

                for kpt_idx in range(len(x_3d)):
                    ax.text(x_3d[kpt_idx][0], y_3d[kpt_idx][0],
                            z_3d[kpt_idx][0], str(kpt_idx))

                if self.skeleton is not None and self.link_color is not None:
                    if self.link_color is None or isinstance(
                            self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(self.link_color)}) does not matches '
                            f'that of skeleton ({len(self.skeleton)})')

                    for sk_id, sk in enumerate(self.skeleton):
                        sk_indices = [_i for _i in sk]
                        xs_3d = kpts[sk_indices, 0]
                        ys_3d = kpts[sk_indices, 1]
                        zs_3d = kpts[sk_indices, 2]
                        kpt_score = score[sk_indices]
                        if kpt_score.min() > kpt_thr:
                            # matplotlib uses RGB color in [0, 1] value range
                            _color = link_color[sk_id][::-1] / 255.
                            ax.plot(
                                xs_3d, ys_3d, zs_3d, color=_color, zdir='z')

        if 'keypoints3d' in pred_instances:
            keypoints = pred_instances.get('keypoints3d',
                                           pred_instances.keypoints3d)

            if 'keypoint_scores' in pred_instances:
                scores = pred_instances.keypoint_scores
            else:
                scores = np.ones(keypoints.shape[:-1])

            if 'keypoints_visible' in pred_instances:
                keypoints_visible = pred_instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            _draw_3d_instances_kpts(keypoints, scores, keypoints_visible, 1,
                                    'Prediction')

        if draw_gt and 'gt_instances' in pose_samples:
            gt_instances = pose_samples.gt_instances
            if 'keypoints3d' in gt_instances:
                keypoints = gt_instances.get('keypoints3d',
                                             gt_instances.keypoints3d)
                scores = np.ones(keypoints.shape[:-1])

                if 'lifting_target_visible' in gt_instances:
                    keypoints_visible = gt_instances.lifting_target_visible
                else:
                    keypoints_visible = np.ones(keypoints.shape[:-1])

                _draw_3d_instances_kpts(keypoints, scores, keypoints_visible,
                                        2, 'Ground Truth')

        # convert figure to numpy array
        fig.tight_layout()
        fig.canvas.draw()

        pred_img_data = fig.canvas.tostring_rgb()
        pred_img_data = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8)

        if not pred_img_data.any():
            pred_img_data = np.full((vis_height, vis_width, 3), 255)
        else:
            pred_img_data = pred_img_data.reshape(vis_height,
                                                  vis_width * num_instances,
                                                  -1)

        plt.close(fig)

        return pred_img_data

    def _draw_instances_kpts(self,
                             image: np.ndarray,
                             instances: InstanceData,
                             kpt_thr: float = 0.3,
                             show_kpt_idx: bool = False,
                             skeleton_style: str = 'mmpose',
                             test=False):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        self.set_image(image)
        img_h, img_w, _ = image.shape

        if 'keypoints' in instances:
            if not test:
                keypoints = instances.get('transformed_keypoints',
                                          instances.keypoints)[..., :2]  # noqa
            else:
                keypoints = instances.get('keypoints')[..., :2]  # noqa

            if 'keypoint_scores' in instances:
                scores = instances.keypoint_scores
            else:
                scores = np.ones(keypoints.shape[:-1])

            if 'keypoints_visible' in instances:
                keypoints_visible = instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            if skeleton_style == 'openpose':
                keypoints_info = np.concatenate(
                    (keypoints, scores[..., None], keypoints_visible[...,
                                                                     None]),
                    axis=-1)
                # compute neck joint
                neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
                # neck score when visualizing pred
                neck[:, 2:4] = np.logical_and(
                    keypoints_info[:, 5, 2:4] > kpt_thr,
                    keypoints_info[:, 6, 2:4] > kpt_thr).astype(int)
                new_keypoints_info = np.insert(
                    keypoints_info, 17, neck, axis=1)

                mmpose_idx = [
                    17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
                ]
                openpose_idx = [
                    1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
                ]
                new_keypoints_info[:, openpose_idx] = \
                    new_keypoints_info[:, mmpose_idx]
                keypoints_info = new_keypoints_info

                keypoints, scores, keypoints_visible = keypoints_info[
                    ..., :2], keypoints_info[..., 2], keypoints_info[..., 3]

            for kpts, score, visible in zip(keypoints, scores,
                                            keypoints_visible):
                kpts = np.array(kpts, copy=False)

                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f'the length of kpt_color '
                        f'({len(self.kpt_color)}) does not matches '
                        f'that of keypoints ({len(kpts)})')

                # draw each point on image
                for kid, kpt in enumerate(kpts):
                    if score[kid] < kpt_thr or not visible[
                            kid] or kpt_color[kid] is None:
                        # skip the point that should not be drawn
                        continue

                    color = kpt_color[kid]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(0, min(1, score[kid]))
                    self.draw_circles(
                        kpt,
                        radius=np.array([self.radius]),
                        face_colors=color,
                        edge_colors=color,
                        alpha=transparency,
                        line_widths=self.radius)
                    if show_kpt_idx:
                        self.draw_texts(
                            str(kid),
                            kpt,
                            colors=color,
                            font_sizes=self.radius * 3,
                            vertical_alignments='bottom',
                            horizontal_alignments='center')

                # draw links
                if self.skeleton is not None and self.link_color is not None:
                    if self.link_color is None or isinstance(
                            self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(self.link_color)}) does not matches '
                            f'that of skeleton ({len(self.skeleton)})')

                    for sk_id, sk in enumerate(self.skeleton):
                        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                        if not (visible[sk[0]] and visible[sk[1]]):
                            continue

                        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                                or pos1[1] >= img_h or pos2[0] <= 0
                                or pos2[0] >= img_w or pos2[1] <= 0
                                or pos2[1] >= img_h or score[sk[0]] < kpt_thr
                                or score[sk[1]] < kpt_thr
                                or link_color[sk_id] is None):
                            # skip the link that should not be drawn
                            continue
                        X = np.array((pos1[0], pos2[0]))
                        Y = np.array((pos1[1], pos2[1]))
                        color = link_color[sk_id]
                        if not isinstance(color, str):
                            color = tuple(int(c) for c in color)
                        transparency = self.alpha
                        if self.show_keypoint_weight:
                            transparency *= max(
                                0, min(1, 0.5 * (score[sk[0]] + score[sk[1]])))

                        if skeleton_style == 'openpose':
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                            transparency = 0.6
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            polygons = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(self.line_width)),
                                int(angle), 0, 360, 1)

                            self.draw_polygons(
                                polygons,
                                edge_colors=color,
                                face_colors=color,
                                alpha=transparency)

                        else:
                            self.draw_lines(
                                X, Y, color, line_widths=self.line_width)

                # draw each point on image
                for kid, kpt in enumerate(kpts):
                    if score[kid] < kpt_thr or not visible[
                            kid] or kpt_color[kid] is None:
                        # skip the point that should not be drawn
                        continue

                    color = kpt_color[kid]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(0, min(1, score[kid]))
                    self.draw_circles(
                        kpt,
                        radius=np.array([self.radius]),
                        face_colors=color,
                        edge_colors=color,
                        alpha=transparency,
                        line_widths=self.radius)
                    if show_kpt_idx:
                        kpt[0] += self.radius
                        kpt[1] -= self.radius
                        self.draw_texts(
                            str(kid),
                            kpt,
                            colors=color,
                            font_sizes=self.radius * 3,
                            vertical_alignments='bottom',
                            horizontal_alignments='center')

        return self.get_image()

    def _draw_instance_heatmap(
        self,
        fields: PixelData,
        overlaid_image: Optional[np.ndarray] = None,
    ):
        """Draw heatmaps of GT or prediction.

        Args:
            fields (:obj:`PixelData`): Data structure for
                pixel-level annotations or predictions.
            overlaid_image (np.ndarray): The image to draw.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        if 'heatmaps' not in fields:
            return None
        heatmaps = fields.heatmaps
        if isinstance(heatmaps, list):
            heatmaps = heatmaps[-1]
        if isinstance(heatmaps, np.ndarray):
            heatmaps = torch.from_numpy(heatmaps)
        if heatmaps.dim() == 3:
            heatmaps, _ = heatmaps.max(dim=0)
        heatmaps = heatmaps.unsqueeze(0)
        out_image = self.draw_featmap(heatmaps, overlaid_image)
        return out_image

    def _draw_instance_xy_heatmap(
        self,
        fields: PixelData,
        overlaid_image: Optional[np.ndarray] = None,
        n: int = 20,
    ):
        """Draw heatmaps of GT or prediction.

        Args:
            fields (:obj:`PixelData`): Data structure for
            pixel-level annotations or predictions.
            overlaid_image (np.ndarray): The image to draw.
            n (int): Number of keypoint, up to 20.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        if 'heatmaps' not in fields:
            return None
        heatmaps = fields.heatmaps
        _, h, w = heatmaps.shape
        if isinstance(heatmaps, np.ndarray):
            heatmaps = torch.from_numpy(heatmaps)
        out_image = SimCCVisualizer().draw_instance_xy_heatmap(
            heatmaps, overlaid_image, n)
        out_image = cv2.resize(out_image[:, :, ::-1], (w, h))
        return out_image

    @master_only
    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: PoseDataSample,
                       draw_gt: bool = True,
                       draw_3d: bool = True,
                       draw_pred: bool = True,
                       draw_heatmap: bool = False,
                       draw_bbox: bool = False,
                       show_kpt_idx: bool = False,
                       skeleton_style: str = 'mmpose',
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       kpt_thr: float = 0.3,
                       step: int = 0,
                       test=False) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier
            image (np.ndarray): The image to draw
            data_sample (:obj:`PoseDataSample`, optional): The data sample
                to visualize
            draw_gt (bool): Whether to draw GT PoseDataSample. Default to
                ``True``
            draw_3d (bool): Whether to draw 3d keypoints. Default to
                ``False``
            draw_pred (bool): Whether to draw Prediction PoseDataSample.
                Defaults to ``True``
            draw_bbox (bool): Whether to draw bounding boxes. Default to
                ``False``
            draw_heatmap (bool): Whether to draw heatmaps. Defaults to
                ``False``
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``
            show (bool): Whether to display the drawn image. Default to
                ``False``
            wait_time (float): The interval of show (s). Defaults to 0
            out_file (str): Path to output file. Defaults to ``None``
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            step (int): Global step value to record. Defaults to 0
        """

        gt_img_data = None
        pred_img_data = None
        image = image.astype(np.uint8)
        if draw_gt:
            gt_img_data = image.copy()
            gt_img_heatmap = None

            # draw bboxes & keypoints
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances_kpts(
                    gt_img_data, data_sample.gt_instances, kpt_thr,
                    show_kpt_idx, skeleton_style, test)
                if draw_bbox:
                    gt_img_data = self._draw_instances_bbox(
                        gt_img_data, data_sample.gt_instances)

            # draw heatmaps
            if 'gt_fields' in data_sample and draw_heatmap:
                gt_img_heatmap = self._draw_instance_heatmap(
                    data_sample.gt_fields, image)
                if gt_img_heatmap is not None:
                    gt_img_data = np.concatenate((gt_img_data, gt_img_heatmap),
                                                 axis=0)

        if draw_pred:
            pred_img_data = image.copy()
            pred_img_heatmap = None

            # draw bboxes & keypoints
            if 'pred_instances' in data_sample:
                pred_img_data = self._draw_instances_kpts(
                    pred_img_data, data_sample.pred_instances, kpt_thr,
                    show_kpt_idx, skeleton_style, test)
                if draw_bbox:
                    pred_img_data = self._draw_instances_bbox(
                        pred_img_data, data_sample.pred_instances)

            # draw heatmaps
            if 'pred_fields' in data_sample and draw_heatmap:
                if 'keypoint_x_labels' in data_sample.pred_instances:
                    pred_img_heatmap = self._draw_instance_xy_heatmap(
                        data_sample.pred_fields, image)
                else:
                    pred_img_heatmap = self._draw_instance_heatmap(
                        data_sample.pred_fields, image)
                if pred_img_heatmap is not None:
                    pred_img_data = np.concatenate(
                        (pred_img_data, pred_img_heatmap), axis=0)

        # merge visualization results
        if gt_img_data is not None and pred_img_data is not None:
            if gt_img_heatmap is None and pred_img_heatmap is not None:
                gt_img_data = np.concatenate((gt_img_data, image), axis=0)
            elif gt_img_heatmap is not None and pred_img_heatmap is None:
                pred_img_data = np.concatenate((pred_img_data, image), axis=0)

            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)

        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if draw_3d:
            pred3d_img_data = self._draw_3d_data_samples(
                image.copy(), data_sample, draw_gt=draw_gt)
            drawn_img = np.concatenate((drawn_img, pred3d_img_data), axis=0)

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            if not out_file.endswith('.png') and not out_file.endswith('.jpg'):
                out_file = out_file + '.png'
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            # save drawn_img to backends
            self.add_image(name, drawn_img, step)

        return self.get_image()
