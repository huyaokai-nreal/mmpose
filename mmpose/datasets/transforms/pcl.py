# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import cv2
import numpy as np
import torch
from nreal_data_tool.utils.affine import from_two_vectors, transform3
from nreal_data_tool.utils.camera import PinholePlaneCameraModel
from scipy.spatial.transform import Rotation


def warp_image(
    src_camera,
    dst_camera,
    dst_width,
    dst_height,
    src_image: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
) -> np.ndarray:
    px, py = np.meshgrid(np.arange(dst_width), np.arange(dst_height))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))

    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)

    # Mask out points with negative z coordinates
    if depth_check:
        mask = src_eye_pts[:, 2] < 0
        src_win_pts[mask] = -1

    src_win_pts = src_win_pts.astype(np.float32)

    map_x = src_win_pts[:, 0].reshape((dst_height, dst_width))
    map_y = src_win_pts[:, 1].reshape((dst_height, dst_width))

    return cv2.remap(src_image, map_x, map_y, interpolation)


def generate_virtual_K(p_position,
                       K,
                       bbox_size_img,
                       focal_at_image_plane,
                       slant_compensation,
                       maintain_aspect_ratio=True,
                       rectangular_images=False):
    batch_size = bbox_size_img.shape[0]
    p_length = torch.norm(p_position, dim=1, keepdim=True)
    focal_length_factor = 1
    if focal_at_image_plane:
        focal_length_factor *= p_length
    if slant_compensation:
        sx = 1.0 / torch.sqrt(
            p_position[:, 0]**2 + p_position[:, 2]**2)  # this is cos(phi)
        sy = torch.sqrt(p_position[:, 0]**2 + 1) / torch.sqrt(
            p_position[:, 0]**2 + p_position[:, 1]**2 +
            1)  # this is cos(theta)
        bbox_size_img = bbox_size_img * torch.stack([sx, sy], dim=1)

    if not rectangular_images:
        if maintain_aspect_ratio:
            max_width, _ = torch.max(bbox_size_img, dim=-1, keepdims=True)
            bbox_size_img = torch.cat([max_width, max_width], dim=-1)
        f_orig = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1)
        # dividing by the target bbox_size_img will make the coordinates
        # normalized to 0..1, as needed for the perspective grid sample
        # function;
        # an alternative would be to make the grid_sample operate on pixel
        # coordinates
        f_compensated = focal_length_factor * f_orig / bbox_size_img
        K_virt = torch.zeros([batch_size, 3, 3],
                             dtype=torch.float).to(f_compensated.device)
        K_virt[:, 2, 2] = 1
        # Note, in unit image coordinates ranging from 0..1
        K_virt[:, 0, 0] = f_compensated[:, 0]
        K_virt[:, 1, 1] = f_compensated[:, 1]
        K_virt[:, :2, 2] = 0.5
        return K_virt
    else:
        f_orig = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1)
        f_re_scaled = f_orig / bbox_size_img
        if maintain_aspect_ratio:
            min_factor, _ = torch.min(f_re_scaled, dim=-1, keepdims=True)
            f_re_scaled = torch.cat([min_factor, min_factor], dim=-1)
        f_compensated = focal_length_factor * f_re_scaled
        K_virt = torch.zeros([batch_size, 3, 3],
                             dtype=torch.float).to(f_compensated.device)
        K_virt[:, 2, 2] = 1
        K_virt[:, 0, 0] = f_compensated[:, 0]
        K_virt[:, 1, 1] = f_compensated[:, 1]
        K_virt[:, :2, 2] = 0.5
        return K_virt


def make_look_at_matrix(
    orig_world_to_eye: np.ndarray,
    center: np.ndarray,
    camera_angle: float = 0,
) -> np.ndarray:
    center_local = transform3(orig_world_to_eye, center)
    z_dir_local = center_local / np.linalg.norm(center_local)
    delta_r_local = from_two_vectors(
        np.array([0, 0, 1], dtype=center.dtype), z_dir_local)
    orig_eye_to_world = np.linalg.inv(orig_world_to_eye)

    new_eye_to_world = orig_eye_to_world.copy()
    new_eye_to_world[0:3, 0:3] = orig_eye_to_world[0:3, 0:3] @ delta_r_local

    # Locally rotate the z axis to align with the camera angle
    z_local_rot = Rotation.from_euler(
        'z', camera_angle, degrees=True).as_matrix()
    new_eye_to_world[0:3, 0:3] = new_eye_to_world[0:3, 0:3] @ z_local_rot

    return np.linalg.inv(new_eye_to_world)


def gen_intrinsics_from_bounding_pts(pts_eye: np.ndarray,
                                     image_w: int,
                                     image_h: int,
                                     min_focal: float = 5
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    pts_ndc = pts_eye[..., 0:2] / pts_eye[..., 2:]
    img_size = np.array([image_w, image_h], dtype=pts_eye.dtype)
    # Given our convention, we need to shift one pixel before dividing by 2.
    cx_cy = (img_size - 1) / 2
    fx_fy = cx_cy / np.absolute(pts_ndc).max()

    # Some sanity checks
    if np.any(pts_eye[..., 2:] < 0.0001) or np.any(fx_fy < min_focal):
        raise ValueError('Unable to create crop camera', fx_fy)

    return fx_fy, cx_cy


def gen_intrinsics_from_bounding_box(center_eye,
                                     image_w,
                                     image_h,
                                     ori_K,
                                     min_focal: float = 5):
    ori_K = torch.from_numpy(ori_K[np.newaxis, ...])
    pts_eye = torch.from_numpy(center_eye[np.newaxis, ...])
    image_size = torch.tensor([[image_h, image_w]])
    virtual_K = generate_virtual_K(
        pts_eye,
        ori_K,
        image_size,
        focal_at_image_plane=False,
        slant_compensation=False)
    return virtual_K


def gen_crop_parameters_from_points(
    camera_orig,
    crop_center,
    new_image_size: Tuple[int, int],
    mirror_img_x: bool,
    camera_angle: float = 0,
    focal_multiplier: float = 0.9,
):
    """Given the original camera transform and a list of 3D points in the world
    space, compute the new perspective camera that makes sure after projection
    all the points can be projected inside the image.

    Auguments:
    * camera_orig: the original camera used for generating an image.
    The returned camera will have the same position but different
    rotation and intrinsics parameters.
    * pts_world: points in the world space that must be
    projected inside the image by the generated world to eye
    transform and intrinsics.
    * new_image_size: target image size
    * mirror_img_x: whether to flip the image. A typical use case is we
    usually mirror the right hand images so that a model need to handle
    left hand data only
    * camera_angle: how the camera is oriented physically so
    that we can rotate the object of interest to the 'upright' direction
    * focal_multiplier: when less than 1, we are zooming out a little.
    The effect on the image is some margin will be left at the boundary.
    """
    orig_world_to_eye_xf = np.linalg.inv(camera_orig.camera_to_world_xf)
    center_eye = camera_orig.window_to_eye(crop_center)
    new_world_to_eye = make_look_at_matrix(orig_world_to_eye_xf, center_eye,
                                           camera_angle)
    if mirror_img_x:
        mirrorx = np.eye(4, dtype=np.float32)
        mirrorx[0, 0] = -1
        new_world_to_eye = mirrorx @ new_world_to_eye

    ori_K = camera_orig.uv_to_window_matrix()
    homo_center = np.concatenate([crop_center, np.ones((1))], axis=0)
    cam_center = np.linalg.inv(ori_K) @ homo_center
    virtual_K = gen_intrinsics_from_bounding_box(cam_center, new_image_size[0],
                                                 new_image_size[1], ori_K)
    virtual_K = virtual_K[0] * new_image_size[0]
    fx_fy = [
        virtual_K[0][0] * focal_multiplier, virtual_K[1][1] * focal_multiplier
    ]
    cx_cy = [virtual_K[0][2], virtual_K[1][2]]
    return PinholePlaneCameraModel(
        f=fx_fy,
        c=cx_cy,
        distort_coeffs=[],
        camera_to_world_xf=np.linalg.inv(new_world_to_eye),
    )
