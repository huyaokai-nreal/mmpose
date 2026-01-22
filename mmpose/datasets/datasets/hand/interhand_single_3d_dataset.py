# flake8: noqa
import copy
import gc
import os
import os.path as osp
import pickle
import random
import tempfile
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import cv2
import lmdb
import numpy as np
import json_tricks as json
from mmengine.dataset.base_dataset import force_full_init
from mmengine.logging import MessageHub, MMLogger
from xtcocotools.coco import COCO

from nreal_data_tool.utils.camera import PinholePlaneCameraModel
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.hand.nimble_hand import get_nimble_bones_length
from ..base import BaseCocoStyleDataset


def _norm_key(p: str) -> str:
    p = str(p).strip()
    p = p.replace("\x00", "").replace("\r", "").replace("\n", "")
    p = p.replace("\\", "/").lstrip("/")
    while "//" in p:
        p = p.replace("//", "/")
    return p


def _imdecode_gray(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("cv2.imdecode failed for image bytes")
    return img[..., None]


def _build_pinhole_camera(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_to_world_xf: np.ndarray,
) -> PinholePlaneCameraModel:
    return PinholePlaneCameraModel(
        f=[float(fx), float(fy)],
        c=[float(cx), float(cy)],
        distort_coeffs=[],
        camera_to_world_xf=np.asarray(cam_to_world_xf, dtype=np.float32),
    )


@DATASETS.register_module()
class InterHandSingle3DDataset(BaseCocoStyleDataset):
    
    """
    This dataset builds per-hand instances from InterHand2.6M and outputs 21 keypoints
    in **nreal order**. Images are loaded from an image LMDB, while annotations/camera/joints
    are loaded from an annotation LMDB (COCO json is materialized into a temp file for COCO API).

    Key behaviors:
    - Data length can be downscaled by `data_ratio` via overridden `__len__`.
        In training, sampling is still from the full pool (so `data_ratio` controls
        epoch steps rather than shrinking the sampling pool).
    - Training uses random sampling regardless of incoming `idx`.
    - Optional round-based sampling with `round_num` and `epochs_per_round`.
    - Supports emitting single-hand samples from both single-hand and interacting frames
        (`include_interacting=True/False`).
    - Projects 3D world joints to 2D using per-frame camera intrinsics/extrinsics,
        filters invalid samples by minimum visible keypoints, camera-space z, and (optionally)
        in-frame ratio.
    - Computes a global hand scale from mean bone lengths over valid instances,
        relative to a template bone-length prior; stored in `meta['hand_scale']`.
    - Provides runtime meta for camera and template bones:
        * `meta['ori_camera']`: PinholePlaneCameraModel built from fx/fy/cx/cy + cam_to_world_xf
        * `meta['template_bones']`: Nimble template bone lengths
        * `meta['flipped']`: left->right flip flag (set in `prepare_data`)

    Output per instance:
    - img: uint8 (H, W, 1) grayscale
    - bbox: (1, 4) in xyxy, tight box around visible 2D joints with padding
    - keypoints: (1, 21, 2) pixel coordinates in nreal order
    - keypoints3d: (1, 21, 3) world coordinates in meters in nreal order
    - keypoints_visible: (1, 21) visibility mask (float32, >0 means valid)
    - meta: fx/fy/cx/cy, cam_to_world_xf (float32 4x4), hand_scale, flipped,
            frame_width/height, capture/camera/frame_idx, category_id, etc.
    """


    METAINFO: dict = dict(from_file="configs/_base_/datasets/nreal_hand.py")

    # InterHand(21): [thumb4..thumb1, fore4..fore1, mid4..mid1, ring4..ring1, pinky4..pinky1, wrist]
    # nreal_hand(21): [wrist, thumb1..4, fore1..4, mid1..4, ring1..4, pinky1..4]
    DEFAULT_REORDER = [
        20, 3, 2, 1, 0,
        7, 6, 5, 4,
        11, 10, 9, 8,
        15, 14, 13, 12,
        19, 18, 17, 16
    ]

    def __init__(
        self,
        ann_key: str,
        camera_key: str,
        joint_key: str,
        ann_lmdb_dir: str,
        img_lmdb_dir: str,
        split: str = "train",
        data_root: str = "",
        data_mode: str = "topdown",
        test_mode: bool = False,
        pipeline: Optional[List[Union[dict, Callable]]] = None,
        metainfo: Optional[dict] = None,
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        serialize_data: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 100,

        data_ratio: float = -1.0,
        point_type: str = "2.5D",
        round_num: int = -1,
        epochs_per_round: int = -1,

        sample_interval: int = 1,
        min_valid_kpts: int = 12,
        bbox_padding: float = 1.0,
        filter_kpt_exceed: bool = False,
        include_interacting: bool = True,
        flip_left_to_right: bool = True,
        reorder_indices: Optional[List[int]] = None,
        z_min: float = 1e-4,
        joint_cache: str = "auto",
        release_raw_cache: bool = True,
        **kwargs,
    ):
        if pipeline is None:
            pipeline = []


        self.data_ratio = float(data_ratio)
        self.point_type = str(point_type)
        self.round_num = int(round_num)
        self.epochs_per_round = int(epochs_per_round)

        self.include_interacting = bool(include_interacting)
        self.flip_left_to_right = bool(flip_left_to_right)

        self.sample_interval = max(int(sample_interval), 1)
        self.min_valid_kpts = int(min_valid_kpts)
        self.bbox_padding = float(bbox_padding)
        self.filter_kpt_exceed = bool(filter_kpt_exceed)
        self.z_min = float(z_min)

        self.joint_cache = str(joint_cache).lower()
        if self.joint_cache not in {"auto", "json", "pkl", "pickle"}:
            raise ValueError(
                f"joint_cache must be one of auto/json/pkl/pickle, got: {joint_cache}"
            )
        if self.joint_cache in {"pkl", "pickle"} and (not str(joint_key).endswith((".pkl", ".pickle"))):
            raise ValueError(
                f"joint_cache={self.joint_cache} requires joint_key to be .pkl/.pickle, got: {joint_key}"
            )

        self.release_raw_cache = bool(release_raw_cache)

        reorder = reorder_indices if reorder_indices is not None else self.DEFAULT_REORDER
        self.reorder_indices = np.asarray(reorder, dtype=np.int64)

        # template bones 
        self.template_bones = get_nimble_bones_length().astype(np.float32)

        # computed after loading annotations
        self._bones_mean: Optional[np.ndarray] = None   # (5,4)
        self._hand_scale_global: float = 1.0

        # keys + dirs
        self.ann_key = _norm_key(ann_key)
        self.camera_key = _norm_key(camera_key)
        self.joint_key = _norm_key(joint_key)

        self.ann_lmdb_dir = osp.join(data_root, ann_lmdb_dir) if data_root else ann_lmdb_dir
        self.img_lmdb_dir = osp.join(data_root, img_lmdb_dir) if data_root else img_lmdb_dir
        self.split = str(split)


        self._env_pid = os.getpid()
        self._ann_env: Optional[lmdb.Environment] = None
        self._img_env: Optional[lmdb.Environment] = None


        self._coco: Optional[COCO] = None
        self._cameras: Optional[dict] = None
        self._joints: Optional[dict] = None
        self._id2name: Optional[dict] = None

        # temp annotation file for COCO
        self._tmp_coco_path: Optional[str] = None

        super().__init__(
            data_root="",
            data_mode=data_mode,
            test_mode=test_mode,
            metainfo=metainfo,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            pipeline=pipeline,
            **kwargs,
        )


        self.data_num = super().__len__()  

    @force_full_init
    def __len__(self) -> int:
        if self.test_mode and self.point_type == "2.5D":
            self.data_ratio = 1.0

        if self.data_ratio <= 0:
            return super().__len__()

        if getattr(self, "serialize_data", False) and hasattr(self, "data_address"):
            return int(len(self.data_address) * self.data_ratio)
        if hasattr(self, "data_list"):
            return int(len(self.data_list) * self.data_ratio)

        return int(super().__len__() * self.data_ratio)

    def _open_envs_if_needed(self) -> None:
        pid = os.getpid()
        if self._env_pid != pid:
            self._env_pid = pid
            self._ann_env = None
            self._img_env = None

        if self._ann_env is None:
            if not osp.isdir(self.ann_lmdb_dir):
                raise FileNotFoundError(f"ann_lmdb_dir not found: {self.ann_lmdb_dir}")
            self._ann_env = lmdb.open(
                self.ann_lmdb_dir,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=512,
                subdir=True,
            )

        if self._img_env is None:
            if not osp.isdir(self.img_lmdb_dir):
                raise FileNotFoundError(f"img_lmdb_dir not found: {self.img_lmdb_dir}")
            self._img_env = lmdb.open(
                self.img_lmdb_dir,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=512,
                subdir=True,
            )

    @staticmethod
    def _lmdb_get(env: lmdb.Environment, key: str) -> bytes:
        k = _norm_key(key).encode("utf-8")
        with env.begin(write=False) as txn:
            v = txn.get(k)
        if v is None:
            raise KeyError(f"LMDB key not found: {key}")
        return bytes(v)

    def _read_ann_bytes(self, key: str) -> bytes:
        self._open_envs_if_needed()
        assert self._ann_env is not None
        return self._lmdb_get(self._ann_env, key)

    def _read_img_bytes(self, key: str) -> bytes:
        self._open_envs_if_needed()
        assert self._img_env is not None
        return self._lmdb_get(self._img_env, key)

    def _load_joint_dict(self, key: str) -> dict:
        raw = self._read_ann_bytes(key)
        is_pkl = key.endswith((".pkl", ".pickle"))
        if self.joint_cache in {"pkl", "pickle"} or (self.joint_cache == "auto" and is_pkl):
            return pickle.loads(raw)
        return json.loads(raw.decode("utf-8"))

    @staticmethod
    def _tight_bbox(
        kpt: np.ndarray, vis: np.ndarray, img_w: int, img_h: int, padding: float
    ) -> Optional[np.ndarray]:
        v = (vis > 0).astype(bool)
        if int(v.sum()) == 0:
            return None
        xs, ys = kpt[v, 0], kpt[v, 1]
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        w = max(2.0, (x2 - x1) * padding)
        h = max(2.0, (y2 - y1) * padding)
        x1 = np.clip(cx - 0.5 * w, 0, img_w - 1)
        x2 = np.clip(cx + 0.5 * w, 0, img_w - 1)
        y1 = np.clip(cy - 0.5 * h, 0, img_h - 1)
        y2 = np.clip(cy + 0.5 * h, 0, img_h - 1)
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

    @staticmethod
    def _within_bounds_enough(kpt: np.ndarray, img_w: int, img_h: int, ratio: float = 0.5) -> bool:
        x, y = kpt[:, 0], kpt[:, 1]
        within = ((0 <= x) & (x < img_w) & (0 <= y) & (y < img_h))
        return int(within.sum()) >= int(np.ceil(len(within) * ratio))

    @staticmethod
    def _bones_from_kpt3d_nreal_order(kpt3d: np.ndarray) -> np.ndarray:
        """Compute bone lengths (5,4) for ONE hand in nreal order (21,3)."""
        # nreal order: [wrist, thumb1..4, fore1..4, mid1..4, ring1..4, pinky1..4]
        root = kpt3d[:1].reshape(1, 1, 3)          # (1,1,3)
        root = np.tile(root, (5, 1, 1))            # (5,1,3)
        other = kpt3d[1:].reshape(5, 4, 3)         # (5,4,3)
        kp = np.concatenate([root, other], axis=1) # (5,5,3)
        bones = np.linalg.norm(kp[:, 1:, :] - kp[:, :-1, :], axis=-1)  # (5,4)
        return bones.astype(np.float32)

    def _load_interhand(self) -> None:
        ann_bytes = self._read_ann_bytes(self.ann_key)

        if self._tmp_coco_path is None:
            fd, path = tempfile.mkstemp(prefix="interhand_coco_", suffix=".json")
            os.close(fd)
            with open(path, "wb") as f:
                f.write(ann_bytes)
            self._tmp_coco_path = path

        self._coco = COCO(self._tmp_coco_path)
        self._id2name = {k: v["file_name"] for k, v in self._coco.imgs.items()}

        self._cameras = json.loads(self._read_ann_bytes(self.camera_key).decode("utf-8"))
        self._joints = self._load_joint_dict(self.joint_key)

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        # We already build the final per-instance dicts in _load_annotations
        return raw_data_info

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        self._load_interhand()
        coco, cameras, joints, id2name = self._coco, self._cameras, self._joints, self._id2name
        if coco is None or cameras is None or joints is None or id2name is None:
            raise RuntimeError("InterHand internal cache not initialized.")

        img_ids = sorted(coco.getImgIds())[::self.sample_interval]
        instance_list: List[dict] = []
        image_list: List[dict] = []
        bbox_id = 0

        hand_name = {0: "right", 1: "left"}      # InterHand: 0=right, 1=left
        hand_cat_id = {"left": 1, "right": 2}    # your convention
        wrist_idx_interhand = 20
        reorder = self.reorder_indices

        # Online mean bones 
        bones_sum = np.zeros((5, 4), dtype=np.float64)
        bones_n = 0

        for img_id in img_ids:
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            if not ann_ids:
                continue
            anns = coco.loadAnns(ann_ids)

            # choose the annotation with maximum valid joints for stability
            ann = max(
                anns,
                key=lambda a: int(
                    np.sum(np.asarray(a.get("joint_valid", [0] * 42), dtype=np.float32))
                ),
            )

            hand_type = ann.get("hand_type", "")
            if (not self.include_interacting) and hand_type == "interacting":
                continue

            img = coco.loadImgs(img_id)[0]
            capture_id = str(img["capture"])
            camera_name = str(img["camera"])
            frame_idx = str(img["frame_idx"])
            img_w, img_h = int(img["width"]), int(img["height"])

            img_key = _norm_key(id2name[img_id])

            cap_cam = cameras.get(capture_id, None)
            if cap_cam is None:
                continue

            camrot = np.asarray(cap_cam["camrot"][camera_name], dtype=np.float32)  # world->cam
            focal = np.asarray(cap_cam["focal"][camera_name], dtype=np.float32)
            princpt = np.asarray(cap_cam["princpt"][camera_name], dtype=np.float32)
            campos_mm = np.asarray(cap_cam["campos"][camera_name], dtype=np.float32).reshape(3, 1)

            fx, fy = float(focal[0]), float(focal[1])
            cx, cy = float(princpt[0]), float(princpt[1])

            j = joints.get(capture_id, {}).get(frame_idx, None)
            if j is None or "world_coord" not in j:
                continue
            joint_world_mm = np.asarray(j["world_coord"], dtype=np.float32)
            if joint_world_mm.shape != (42, 3):
                continue

            # mm -> meters
            campos_m = (campos_mm / 1000.0).astype(np.float32)                        # (3,1)
            joint_world_m = (joint_world_mm / 1000.0).astype(np.float32).reshape(2, 21, 3)

            joint_valid = np.asarray(ann.get("joint_valid", [0] * 42), dtype=np.float32).reshape(2, 21)
            # ensure wrist visible gates all joints 
            for hh in range(2):
                joint_valid[hh, :] *= joint_valid[hh, wrist_idx_interhand]

            # decide which hands to emit
            if hand_type in ("right", "left"):
                hands_to_emit = [0] if hand_type == "right" else [1]
            else:
                hands_to_emit = [0, 1]

            # cam_to_world_xf: R_c2w = (world->cam)^T, t = camera position in world
            cam_to_world = np.eye(4, dtype=np.float32)
            cam_to_world[:3, :3] = camrot.T.astype(np.float32)
            cam_to_world[:3, 3] = campos_m.reshape(3).astype(np.float32)

            cam = _build_pinhole_camera(fx, fy, cx, cy, cam_to_world)

            for hh in hands_to_emit:
                vis = joint_valid[hh].astype(np.float32)
                vmask = (vis > 0).astype(bool)
                if int(vmask.sum()) < self.min_valid_kpts:
                    continue

                side = hand_name[hh]
                cat_id = int(hand_cat_id[side])

                kpt3d_world = joint_world_m[hh].astype(np.float32)  # (21,3) world meters

                # validate using camera-space z
                kpt3d_cam = cam.world_to_eye(kpt3d_world)
                z = kpt3d_cam[vmask, 2]
                if (not np.isfinite(kpt3d_cam[vmask]).all()) or np.any(z <= self.z_min):
                    continue

                # project 2D; fill invalid joints with wrist 2D
                kpt2d = np.empty((21, 2), dtype=np.float32)
                wrist_2d = cam.eye_to_window(kpt3d_cam[[wrist_idx_interhand]]).astype(np.float32)[0]
                kpt2d[:] = wrist_2d
                kpt2d[vmask] = cam.eye_to_window(kpt3d_cam[vmask]).astype(np.float32)

                if self.filter_kpt_exceed and (
                    not self._within_bounds_enough(kpt2d[vmask], img_w, img_h, ratio=0.5)
                ):
                    continue

                # reorder to nreal
                kpt2d = kpt2d[reorder]
                vis_r = vis[reorder]
                kpt3d_world_r = kpt3d_world[reorder]

                # fill invisible 3D with root (nreal index 0) to keep downstream stable
                bad3d = (vis_r <= 0)
                if bad3d.any():
                    kpt3d_world_r = kpt3d_world_r.copy()
                    kpt3d_world_r[bad3d] = kpt3d_world_r[0]

                bbox = self._tight_bbox(kpt2d, vis_r, img_w, img_h, padding=self.bbox_padding)
                if bbox is None:
                    continue

                # online bones accumulation
                bones_sum += self._bones_from_kpt3d_nreal_order(kpt3d_world_r).astype(np.float64)
                bones_n += 1

                meta = dict(
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    cam_to_world_xf=cam_to_world.astype(np.float32),
                    camera_angle=0,
                    hand_scale=1.0,  # will be assigned after global computation
                    flipped=False,    
                    frame_height=img_h,
                    frame_width=img_w,
                    dataset_tag="interhand2.6m",
                    tag="interhand2.6m",
                    category_id=cat_id,
                    capture=capture_id,
                    camera=camera_name,
                    frame_idx=frame_idx,
                    bbox_id=bbox_id,
                    split=self.split,
                )

                instance_list.append(dict(
                    img_id=img_id,
                    img_key=img_key,
                    img_path=f"lmdb://{self.img_lmdb_dir}/{img_key}",
                    image_width=img_w,
                    image_height=img_h,
                    bbox=bbox,
                    bbox_score=np.ones(1, dtype=np.float32),
                    keypoints=kpt2d.reshape(1, 21, 2).astype(np.float32),
                    keypoints_visible=vis_r.reshape(1, 21).astype(np.float32),
                    keypoints3d=kpt3d_world_r.reshape(1, 21, 3).astype(np.float32),
                    id=bbox_id,
                    cat_id=cat_id,
                    iscrowd=ann.get("iscrowd", 0),
                    segmentation=ann.get("segmentation", None),
                    meta=meta,
                ))
                bbox_id += 1

            image_list.append(img)

        if bones_n > 0:
            bones_mean = (bones_sum / float(bones_n)).astype(np.float32)  # (5,4)
            self._bones_mean = bones_mean
            ratio = bones_mean / (self.template_bones + 1e-8)
            self._hand_scale_global = float(np.mean(ratio))
            if (not np.isfinite(self._hand_scale_global)) or self._hand_scale_global <= 1e-6:
                self._hand_scale_global = 1.0
        else:
            self._bones_mean = None
            self._hand_scale_global = 1.0

        for inst in instance_list:
            inst_meta = inst.get("meta", {})
            inst_meta["hand_scale"] = float(self._hand_scale_global)
            inst["meta"] = inst_meta

        MMLogger.get_current_instance().info(
            f"InterHand25DSingleHandDatasetLMDB loaded {len(image_list)} images -> {len(instance_list)} hand instances "
            f"(split={self.split}, sample_interval={self.sample_interval}, data_ratio={self.data_ratio}, "
            f"include_interacting={self.include_interacting}, min_valid_kpts={self.min_valid_kpts}, "
            f"z_min={self.z_min}, hand_scale={self._hand_scale_global:.4f})"
        )

        if self.release_raw_cache:
            self._coco = None
            self._cameras = None
            self._joints = None
            self._id2name = None
            gc.collect()

        return instance_list, image_list

    def get_data_info(self, idx: int) -> dict:
        # train-time random sampling from full pool
        if not self.test_mode:
            if getattr(self, "data_num", 0) <= 0:
                self.data_num = super().__len__()
            idx = random.randint(0, self.data_num - 1)

            # Optional: round-based sampling 
            if self.round_num > 0 and self.epochs_per_round > 0:
                num_per_round = max(1, self.data_num // self.round_num)
                mh = MessageHub.get_current_instance()
                try:
                    cur_epoch = int(mh.get_info("epoch"))
                except KeyError:
                    cur_epoch = 0
                round_id = (cur_epoch // self.epochs_per_round) % self.round_num
                lo = round_id * num_per_round
                hi = min(self.data_num - 1, (round_id + 1) * num_per_round - 1)
                if hi >= lo:
                    idx = random.randint(lo, hi)
        else:
            if getattr(self, "data_num", 0) <= 0:
                self.data_num = super().__len__()
            idx = idx % self.data_num

        data_info = super().get_data_info(idx)

        img_key = data_info.get("img_key", None)
        if img_key is None:
            raise KeyError("img_key missing in data_info; LMDB dataset expects img_key")

        data_info["img"] = _imdecode_gray(self._read_img_bytes(img_key))

        h, w = data_info["img"].shape[:2]
        data_info.setdefault("img_shape", (h, w))
        data_info.setdefault("ori_shape", (h, w))

        meta = data_info["meta"]
        meta["test_mode"] = self.test_mode
        meta.setdefault("camera_name", "mono")
        meta["frame_height"] = int(h)
        meta["frame_width"] = int(w)

        # set default flipped flag here; real flipped is decided in prepare_data
        meta["flipped"] = False
        return data_info

    @force_full_init
    def prepare_data(self, idx) -> Any:
        data_info = copy.deepcopy(self.get_data_info(idx))
        meta = copy.deepcopy(data_info["meta"])

        cam = _build_pinhole_camera(
            fx=float(meta["fx"]),
            fy=float(meta["fy"]),
            cx=float(meta["cx"]),
            cy=float(meta["cy"]),
            cam_to_world_xf=np.asarray(meta["cam_to_world_xf"], dtype=np.float32),
        )
        meta["ori_camera"] = copy.deepcopy(cam)
        meta["template_bones"] = self.template_bones

        hs = float(meta.get("hand_scale", self._hand_scale_global))
        meta["hand_scale"] = hs if np.isfinite(hs) and hs > 1e-6 else 1.0

        # flip left->right if cat_id == 1 (left hand)
        meta["flipped"] = bool(self.flip_left_to_right and int(data_info.get("cat_id", 0)) == 1)

        data_info["meta"] = meta
        return self.pipeline(data_info)

    def __del__(self):
        # close envs
        try:
            if getattr(self, "_ann_env", None) is not None:
                self._ann_env.close()
                self._ann_env = None
            if getattr(self, "_img_env", None) is not None:
                self._img_env.close()
                self._img_env = None
        except Exception:
            pass

        # remove tmp coco file
        try:
            p = getattr(self, "_tmp_coco_path", None)
            if p and osp.exists(p):
                os.remove(p)
                self._tmp_coco_path = None
        except Exception:
            pass