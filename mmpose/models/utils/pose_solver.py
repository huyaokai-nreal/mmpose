# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from scipy.optimize import leastsq


# 通过闭式解求解根节点深度
def get_root_depthv2(keypoints, camera, template_bones, undistort):
    if undistort:
        keypoints[..., :2] = camera.undistort(keypoints[..., :2])
    f = np.array(camera.f, dtype=np.float32)
    c = np.array(camera.c, dtype=np.float32)
    keypoints[..., :2] = (keypoints[..., :2] - c) / f
    root_kpt = keypoints[:1].reshape((1, 1, 3))
    root_kpt = np.tile(root_kpt, (5, 1, 1))
    kpt = keypoints[1:].reshape((5, 4, 3))
    kpt = np.concatenate([root_kpt, kpt], axis=1)
    root_list = []
    template_bones = template_bones.reshape(-1)
    for i in range(5):
        for j in range(4):
            kpt_m = kpt[i][j]
            kpt_n = kpt[i][j + 1]
            bone = template_bones[i * 4 + j]
            xm = kpt_m[0]
            ym = kpt_m[1]
            zm = kpt_m[2]
            xn = kpt_n[0]
            yn = kpt_n[1]
            zn = kpt_n[2]
            a = (xn - xm)**2 + (yn - ym)**2
            b = zn * (xn**2 + yn**2 - xn * xm - yn * ym) + zm * (
                xm**2 + ym**2 - xn * xm - yn * ym)
            c = (xn * zn - xm * zm)**2 + (yn * zn - ym * zm)**2 + (
                zn - zm)**2 - bone * bone
            root = 0.5 * (-b + np.sqrt(b**2 - 4 * a * c)) / a
            if not np.isnan(root):
                root_list.append(root)
    return np.mean(root_list)


def get_kpt_depth(keypoints,
                  camera,
                  template_bones,
                  last_kpt3d,
                  undistort: bool = True):
    rel_depth = keypoints[..., 2:]
    kpt2d = keypoints[..., :2]
    if last_kpt3d is not None:
        cur_last_kpt3d = camera.world_to_eye(last_kpt3d)
    if undistort:
        kpt2d = camera.undistort(kpt2d)
    f = np.array(camera.f, dtype=np.float32)
    c = np.array(camera.c, dtype=np.float32)
    norm_kpt2d = np.concatenate([(kpt2d - c) / f, np.ones((21, 1))], axis=-1)

    def get_bones_from_kpt3d(kpt3d):
        root_kpt = kpt3d[:1].reshape((1, 1, 3))
        root_kpt = np.tile(root_kpt, (5, 1, 1))
        kpt = kpt3d[1:].reshape((5, 4, 3))
        kpt = np.concatenate([root_kpt, kpt], axis=1)
        bones = np.linalg.norm(kpt[:, 1:, :] - kpt[:, :-1, :], axis=-1)
        return bones.reshape(-1)

    def error(p, x, y):
        kpt3d = x * p.reshape(-1, 1)
        bones = get_bones_from_kpt3d(kpt3d)
        depth_error = (p - p[-1] - rel_depth.reshape(-1)).reshape(-1)
        bone_error = ((y - bones).reshape(-1))
        reproj_kpt2d = camera.eye_to_window(kpt3d)
        reproj_error = np.linalg.norm(reproj_kpt2d - kpt2d, axis=-1) / 128
        result = np.concatenate([bone_error, depth_error, reproj_error])
        if last_kpt3d is not None:
            time_error = np.linalg.norm(kpt3d - cur_last_kpt3d, axis=-1) * 0.05
            result = np.concatenate([result, time_error])
        return result

    p0 = np.array([0.3] * 21) + rel_depth.reshape(-1)
    param = leastsq(error, p0, args=(norm_kpt2d, template_bones.reshape(-1)))
    return param[0]


# 通过最小二乘迭代优化求解根节点深度
def get_root_depth(keypoints,
                   camera,
                   template_bones,
                   weight,
                   gt: Optional[np.array] = None,
                   undistort: bool = True,
                   estimate_hand_scale: bool = False):
    rel_depth = keypoints[..., 2:]
    kpt2d = keypoints[..., :2]
    if undistort:
        kpt2d = camera.undistort(kpt2d)
    f = np.array(camera.f, dtype=np.float32)
    c = np.array(camera.c, dtype=np.float32)
    norm_kpt2d = np.concatenate([(kpt2d - c) / f, np.ones((21, 1))], axis=-1)

    def get_bones_from_kpt3d(kpt3d):
        root_kpt = kpt3d[:1].reshape((1, 1, 3))
        root_kpt = np.tile(root_kpt, (5, 1, 1))
        kpt = kpt3d[1:].reshape((5, 4, 3))
        kpt = np.concatenate([root_kpt, kpt], axis=1)
        bones = np.linalg.norm(kpt[:, 1:, :] - kpt[:, :-1, :], axis=-1)
        return bones.reshape(-1)

    def error(p, x, y, w, gt):
        w0 = w[0].reshape((1, 1))
        _w = w[1:].reshape((5, 4))
        w0 = np.tile(w0, (5, 1))
        new_w = np.concatenate([w0, _w], axis=-1)
        mean_w = (new_w[:, :4] + new_w[:, 1:]) / 2.0
        mean_w = mean_w.reshape(-1)
        mean_w /= np.max(mean_w)
        kpt3d = x * rel_depth + x * p[0]
        bones = get_bones_from_kpt3d(kpt3d)
        if estimate_hand_scale:
            kpt_error = kpt3d[8] - gt[8]
            result = mean_w * ((y * p[1] - bones).reshape(-1))
            result = np.concatenate([result, kpt_error * 10])
        else:
            result = mean_w * ((y - bones).reshape(-1))
        return result

    p0 = [0.3, 1.0]
    param = leastsq(
        error,
        p0,
        args=(norm_kpt2d, template_bones.reshape(-1), weight.reshape(-1), gt))
    return param[0][0], param[0][1]
