# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional, Sequence, Union
from xtcocotools.coco import COCO
import json_tricks as json
import numpy as np
from mmpose.datasets.builder import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class InterHand3DDataset(BaseCocoStyleDataset):
    """InterHand2.6M 3D dataset for top-down hand pose estimation.

    "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose
    Estimation from a Single RGB Image", ECCV'2020.
    More details can be found in the `paper
    <https://arxiv.org/pdf/2008.09309.pdf>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    InterHand2.6M keypoint indexes::

        0: 'r_thumb4',
        1: 'r_thumb3',
        2: 'r_thumb2',
        3: 'r_thumb1',
        4: 'r_index4',
        5: 'r_index3',
        6: 'r_index2',
        7: 'r_index1',
        8: 'r_middle4',
        9: 'r_middle3',
        10: 'r_middle2',
        11: 'r_middle1',
        12: 'r_ring4',
        13: 'r_ring3',
        14: 'r_ring2',
        15: 'r_ring1',
        16: 'r_pinky4',
        17: 'r_pinky3',
        18: 'r_pinky2',
        19: 'r_pinky1',
        20: 'r_wrist',
        21: 'l_thumb4',
        22: 'l_thumb3',
        23: 'l_thumb2',
        24: 'l_thumb1',
        25: 'l_index4',
        26: 'l_index3',
        27: 'l_index2',
        28: 'l_index1',
        29: 'l_middle4',
        30: 'l_middle3',
        31: 'l_middle2',
        32: 'l_middle1',
        33: 'l_ring4',
        34: 'l_ring3',
        35: 'l_ring2',
        36: 'l_ring1',
        37: 'l_pinky4',
        38: 'l_pinky3',
        39: 'l_pinky2',
        40: 'l_pinky1',
        41: 'l_wrist'

    Args:
        ann_file (str): Path to the annotation file.
        camera_file (str): Path to the camera file.
        joint_file (str): Path to the joint file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        use_gt_root_depth (bool): Using the ground truth depth of the wrist
            or given depth from rootnet_result_file.
        rootnet_result_file (str): Path to the wrist depth file.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (str): Store True when building test or
            validation dataset. Default: False.
    """
    METAINFO: dict = dict(from_file='configs/_base_/datasets/interhand2d.py')

    def __init__(self,
                 ann_file,
                 camera_file,
                 joint_file,
                 img_prefix,
                 pipeline,
                 use_different_joint_weights: bool = False,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 100,
                 use_gt_root_depth=True,
                 rootnet_result_file=None,
                 point_type='3D',
                 test_mode=False):
        self.use_different_joint_weights = use_different_joint_weights
        self.img_prefix = img_prefix
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.point_type = point_type
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)

        if 'categories' in self.coco.dataset:
            cats = [
                cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())
            ]
            self.classes = ['__background__'] + cats
            self.num_classes = len(self.classes)
            self._class_to_ind = dict(
                zip(self.classes, range(self.num_classes)))
            self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
            self._coco_ind_to_class_ind = dict(
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:])
            self.img_ids = self.coco.getImgIds()
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id = self._get_mapping_id_name(
                self.coco.imgs)
        self.camera_file = camera_file
        self.joint_file = joint_file

        self.use_gt_root_depth = use_gt_root_depth
        if not self.use_gt_root_depth:
            assert rootnet_result_file is not None
            self.rootnet_result_file = rootnet_result_file

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

        print(f'=> num_images: {self.num_images}')

    @staticmethod
    def _cam2pixel(cam_coord, f, c):
        """Transform the joints from their camera coordinates to their pixel
        coordinates.

        Note:
            N: number of joints

        Args:
            cam_coord (ndarray[N, 3]): 3D joints coordinates
                in the camera coordinate system
            f (ndarray[2]): focal length of x and y axis
            c (ndarray[2]): principal point of x and y axis

        Returns:
            img_coord (ndarray[N, 3]): the coordinates (x, y, 0)
                in the image plane.
        """
        x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
        y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
        z = np.zeros_like(x)
        img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
        return img_coord

    @staticmethod
    def _world2cam(world_coord, R, T):
        """Transform the joints from their world coordinates to their camera
        coordinates.

        Note:
            N: number of joints

        Args:
            world_coord (ndarray[3, N]): 3D joints coordinates
                in the world coordinate system
            R (ndarray[3, 3]): camera rotation matrix
            T (ndarray[3, 1]): camera position (x, y, z)

        Returns:
            cam_coord (ndarray[3, N]): 3D joints coordinates
                in the camera coordinate system
        """
        cam_coord = np.dot(R, world_coord - T)
        return cam_coord

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _load_annotations(self):
        """Load dataset.

        Adapted from 'https://github.com/facebookresearch/InterHand2.6M/'
            'blob/master/data/InterHand2.6M/dataset.py'
        Copyright (c) FaceBook Research, under CC-BY-NC 4.0 license.
        """
        with open(self.camera_file, 'r') as f:
            cameras = json.load(f)
        with open(self.joint_file, 'r') as f:
            joints = json.load(f)

        if not self.use_gt_root_depth:
            rootnet_result = {}
            with open(self.rootnet_result_file, 'r') as f:
                rootnet_annot = json.load(f)
            for i in range(len(rootnet_annot)):
                rootnet_result[str(
                    rootnet_annot[i]['annot_id'])] = rootnet_annot[i]

        gt_db = []
        bbox_id = 0
        image_list = []
        for img_id in self.img_ids:
            ann_id = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            ann = self.coco.loadAnns(ann_id)[0]
            img = self.coco.loadImgs(img_id)[0]
            image_list.append(img)
            capture_id = str(img['capture'])
            camera_name = img['camera']
            frame_idx = str(img['frame_idx'])
            image_file = osp.join(self.img_prefix, self.id2name[img_id])
            camera_pos = np.array(
                cameras[capture_id]['campos'][camera_name], dtype=np.float32)
            camera_rot = np.array(
                cameras[capture_id]['camrot'][camera_name], dtype=np.float32)
            focal = np.array(
                cameras[capture_id]['focal'][camera_name], dtype=np.float32)
            principal_pt = np.array(
                cameras[capture_id]['princpt'][camera_name], dtype=np.float32)
            joint_world = np.array(
                joints[capture_id][frame_idx]['world_coord'], dtype=np.float32)
            joint_cam = self._world2cam(
                joint_world.transpose(1, 0), camera_rot,
                camera_pos.reshape(3, 1)).transpose(1, 0)
            joint_img = self._cam2pixel(joint_cam, focal, principal_pt)[:, :2]
            joint_valid = np.array(
                ann['joint_valid'], dtype=np.float32).flatten()
            hand_type = ann['hand_type']
            if hand_type not in ['left', 'right']:
                continue
            # only single hand supported now
            # hand_type_valid = ann['hand_type_valid']
            img_w, img_h = img['width'], img['height']
            if self.use_gt_root_depth:
                bbox = np.array(ann['bbox'], dtype=np.float32)
                # get bbox in shape [1, 4], formatted as xywh
                x, y, w, h = bbox
                x1 = np.clip(x, 0, img_w - 1)
                y1 = np.clip(y, 0, img_h - 1)
                x2 = np.clip(x + w, 0, img_w - 1)
                y2 = np.clip(y + h, 0, img_h - 1)
                bbox = np.array([x1, y1, x2, y2],
                                dtype=np.float32).reshape(1, 4)
                abs_depth = [joint_cam[20, 2], joint_cam[41, 2]]
            else:
                rootnet_ann_data = rootnet_result[str(ann_id[0])]
                x, y, w, h = bbox
                x1 = np.clip(x, 0, img_w - 1)
                y1 = np.clip(y, 0, img_h - 1)
                x2 = np.clip(x + w, 0, img_w - 1)
                y2 = np.clip(y + h, 0, img_h - 1)
                bbox = np.array([x1, y1, x2, y2],
                                dtype=np.float32).reshape(1, 4)
                abs_depth = [joint_cam[20, 2], joint_cam[41, 2]]
                bbox = np.array(
                    rootnet_ann_data['bbox'], dtype=np.float32).reshape(1, 4)
                abs_depth = rootnet_ann_data['abs_depth']
            # 41: 'l_wrist', left hand root
            # 20: 'r_wrist', right hand root
            # rel_root_depth = joint_cam[41, 2] - joint_cam[20, 2]
            # if root is not valid, root-relative 3D depth is also invalid.
            # rel_root_valid = joint_valid[20] * joint_valid[41]
            # if root is not valid -> root-relative 3D pose is also not valid.
            # Therefore, mark all joints as invalid
            joint_valid[:20] *= joint_valid[20]
            joint_valid[21:] *= joint_valid[41]
            num_joints = joint_world.shape[0]
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((1, num_joints), dtype=np.float32)
            joints_3d[:, :2] = joint_img
            joints_3d[:21, 2] = joint_cam[:21, 2] - joint_cam[20, 2]
            joints_3d[21:, 2] = joint_cam[21:, 2] - joint_cam[41, 2]
            joints_3d_visible[...] = np.minimum(
                1, joint_valid.reshape(1, num_joints))
            if hand_type == 'right':
                joint_img = joint_img[:21]
                joint_cam = joint_cam[:21]
                joints_3d = joints_3d[:21]
                abs_depth = abs_depth[0]
                joints_3d_visible = joints_3d_visible[:, :21]
            if hand_type == 'left':
                joint_img = joint_img[21:]
                joint_cam = joint_cam[21:]
                joints_3d = joints_3d[21:]
                abs_depth = abs_depth[1]
                joints_3d_visible = joints_3d_visible[:, 21:]
            if self.point_type == '2D':
                keypoints = joint_img[np.newaxis, ...]
            else:
                keypoints = joints_3d[np.newaxis, ...]
            gt_db.append({
                'img_path':
                image_file,
                'img_id':
                img_id,
                'image_width':
                img['width'],
                'image_height':
                img['height'],
                'num_keypoints':
                num_joints,
                'keypoints':
                keypoints,
                'keypoints_visible':
                joints_3d_visible,
                'hand_type':
                hand_type,
                'joint_cam':
                joint_cam,
                'bbox_score':
                np.ones(1, dtype=np.float32),
                'bbox':
                bbox,
                'id':
                bbox_id,
                'meta':
                dict(
                    root_depth=abs_depth,
                    keypoints_cam=joint_cam,
                    focal=focal,
                    princpt=principal_pt)
            })
            bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['id'])
        return gt_db, image_list
