# Copyright (c) OpenMMLab. All rights reserved.
'''
    NIMBLE: A Non-rigid Hand Model with Bones and Muscles[SIGGRAPH-22]
    https://reyuwei.github.io/proj/nimble
'''

from pathlib import Path

# import pytorch3d
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# from pytorch3d.structures.meshes import Meshes
# import pytorch3d.ops

ROOT_JOINT_IDX = 0  # wrist
DOF2_BONES = [1, 2, 4, 5, 8, 9, 12, 13, 16, 17]
DOF1_BONES = [3, 6, 7, 10, 11, 14, 15, 18, 19]
JOINT_PARENT_ID_DICT = {
    0: -1,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 0,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 0,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 0,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 0,
    21: 20,
    22: 21,
    23: 22,
    24: 23
}
JOINT_ID_NAME_DICT = {
    0: 'carpal',
    1: 'met1',
    2: 'pro1',
    3: 'dis1',
    4: 'dis1_end',
    5: 'met2',
    6: 'pro2',
    7: 'int2',
    8: 'dis2',
    9: 'dis2_end',
    10: 'met3',
    11: 'pro3',
    12: 'int3',
    13: 'dis3',
    14: 'dis3_end',
    15: 'met4',
    16: 'pro4',
    17: 'int4',
    18: 'dis4',
    19: 'dis4_end',
    20: 'met5',
    21: 'pro5',
    22: 'int5',
    23: 'dis5',
    24: 'dis5_end'
}
BONE_TO_JOINT_NAME = {
    0: 'carpal',
    1: 'met1',
    2: 'pro1',
    3: 'dis1',
    4: 'met2',
    5: 'pro2',
    6: 'int2',
    7: 'dis2',
    8: 'met3',
    9: 'pro3',
    10: 'int3',
    11: 'dis3',
    12: 'met4',
    13: 'pro4',
    14: 'int4',
    15: 'dis4',
    16: 'met5',
    17: 'pro5',
    18: 'int5',
    19: 'dis5',
}
STATIC_BONE_NUM = 20
STATIC_JOINT_NUM = 25
JOINT_ID_BONE_DICT = {}
JOINT_ID_BONE = np.zeros(STATIC_BONE_NUM)
BONE_ID_JOINT_DICT = {}
for key in JOINT_ID_NAME_DICT:
    value = JOINT_ID_NAME_DICT[key]
    for key_b in BONE_TO_JOINT_NAME:
        if BONE_TO_JOINT_NAME[key_b] == value:
            JOINT_ID_BONE_DICT[key] = key_b
            BONE_ID_JOINT_DICT[key_b] = key
            JOINT_ID_BONE[key_b] = key


def dis_to_weight(dismat, thres_corres, node_sigma):
    dismat[dismat == 0] = 1e5
    dismat[dismat > thres_corres] = 1e5
    node_weight = torch.exp(-dismat / node_sigma)
    norm = torch.norm(node_weight, dim=1)
    norm_node_weight = node_weight / (norm + 1e-6)
    norm_node_weight[norm == 0] = 0
    return norm_node_weight


def batch_to_tensor_device(batch, device):

    def to_tensor(arr):
        if isinstance(arr, int):
            return arr
        if isinstance(arr, torch.Tensor):
            return arr.to(device)
        if arr.dtype == np.int64:
            arr = torch.from_numpy(arr)
        else:
            arr = torch.from_numpy(arr).float()
        return arr

    for key in batch:
        if isinstance(batch[key], np.ndarray):
            batch[key] = to_tensor(batch[key]).to(device)
        elif isinstance(batch[key], list):
            for i in range(len(batch[key])):
                if isinstance(batch[key][i], list):
                    for j in range(len(batch[key][i])):
                        if isinstance(batch[key][i][j], np.ndarray):
                            batch[key][i][j] = to_tensor(
                                batch[key][i][j]).to(device)
                else:
                    batch[key][i] = to_tensor(batch[key][i]).to(device)
        elif isinstance(batch[key], dict):
            batch[key] = batch_to_tensor_device(batch[key], device)
        elif isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    return batch


def quat2aa(quats):
    """Convert wxyz quaternions to angle-axis representation.

    :param quats:
    :return:
    """
    _cos = quats[..., 0]
    xyz = quats[..., 1:]
    _sin = xyz.norm(dim=-1)
    norm = _sin.clone()
    norm[norm < 1e-7] = 1
    axis = xyz / norm.unsqueeze(-1)
    angle = torch.atan2(_sin, _cos) * 2
    return axis * angle.unsqueeze(-1)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion --
        size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_aa2quat(axisang):
    # w, x, y, z
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.sin(torch.pi / 2 - angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    return quat


def batch_rodrigues(axisang):
    # axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    axisang_norm = torch.sqrt(torch.sum((axisang + 1e-8)**2, dim=1))
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.sin(torch.pi / 2 - angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def th_posemap_axisang_2output(pose_vectors):
    rot_nb = int(pose_vectors.shape[1] / 3)
    rot_mats = []
    for joint_idx in range(rot_nb - 1):
        joint_idx_val = joint_idx + 1
        axis_ang = pose_vectors[:, joint_idx_val * 3:(joint_idx_val + 1) * 3]
        rot_mat = batch_rodrigues(axis_ang)
        rot_mats.append(rot_mat)

    # rot_mats = torch.stack(rot_mats, 1).view(-1, 15 *9)
    rot_mats = torch.cat(rot_mats, 1)
    pose_maps = subtract_flat_id(rot_mats)
    return pose_maps, rot_mats


def subtract_flat_id(rot_mats):
    # Subtracts identity as a flattened tensor
    rot_nb = int(rot_mats.shape[1] / 9)
    id_flat = torch.eye(
        3, dtype=rot_mats.dtype,
        device=rot_mats.device).view(1, 9).repeat(rot_mats.shape[0], rot_nb)
    # id_flat.requires_grad = False
    results = rot_mats - id_flat
    return results


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False

    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


def th_scalemat_scale(th_scale_bone):
    batch_size = th_scale_bone.shape[0]
    th_scale_bone_mat = torch.eye(4).repeat(
        [batch_size, th_scale_bone.shape[1], 1, 1])
    th_scale_bone_mat = th_scale_bone_mat.type_as(th_scale_bone).to(
        th_scale_bone.device)
    if len(th_scale_bone.shape) == 3:
        for s in range(th_scale_bone.shape[1]):
            th_scale_bone_mat[:, s, 0, 0] = th_scale_bone[:, s, 0]
            th_scale_bone_mat[:, s, 1, 1] = th_scale_bone[:, s, 1]
            th_scale_bone_mat[:, s, 2, 2] = th_scale_bone[:, s, 2]
    else:
        for s in range(th_scale_bone.shape[1]):
            th_scale_bone_mat[:, s, 0, 0] = th_scale_bone[:, s]
            th_scale_bone_mat[:, s, 1, 1] = th_scale_bone[:, s]
            th_scale_bone_mat[:, s, 2, 2] = th_scale_bone[:, s]
    return th_scale_bone_mat


def th_pack(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new_zeros((batch_size, 4, 3))
    padding.requires_grad = False
    pack_list = [padding, tensor]
    pack_res = torch.cat(pack_list, 2)
    return pack_res


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    '''
        Calculates landmarks by barycentric interpolation
        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks
        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
        Modified from https://github.com/vchoutas/smplx
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    # lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
    # batch_size, -1, 3)
    lmk_faces = torch.index_select(faces, 0,
                                   lmk_faces_idx.view(-1)).view(1, -1, 3)
    lmk_faces = lmk_faces.repeat([batch_size, 1, 1])

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.reshape(-1,
                                    3)[lmk_faces].view(batch_size, -1, 3, 3)
    landmarks = torch.einsum('blfi,lf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def save_textured_nimble(fname, skin_v, tex_img):
    import cv2
    textured_pkl = 'assets/NIMBLE_TEX_FUV.pkl'

    fname = Path(fname)

    obj_name_skin = fname.parent / (fname.stem + '_skin.obj')
    mtl_name = obj_name_skin.with_suffix('.mtl')

    # texture image
    tex_name_diffuse = fname.parent / (fname.stem + '_diffuse.png')
    tex_img = np.uint8(tex_img * 255)

    cv2.imwrite(str(tex_name_diffuse), tex_img[:, :, :3])
    cv2.imwrite(
        str(fname.parent / (fname.stem + '_normal.png')), tex_img[:, :, 3:6])
    cv2.imwrite(
        str(fname.parent / (fname.stem + '_spec.png')), tex_img[:, :, 6:])

    # mtl
    mtl_str = 'newmtl material_0\nKa 0.200000 0.200000 0.200000\nKd 0.800000 0.800000 0.800000\nKs 1.000000 1.000000 1.000000\nTr 1.000000\nillum 2\nNs 0.000000\nmap_Kd '  # noqa
    mtl_str = mtl_str + tex_name_diffuse.name
    with open(mtl_name, 'w') as f:
        f.writelines(mtl_str)

    # obj
    f_uv = np.load(textured_pkl, allow_pickle=True)
    with open(obj_name_skin, 'w') as f:
        f.write('mtllib {:s}\n'.format(mtl_name.name))
        for v in skin_v:
            f.writelines('v {:.5f} {:.5f} {:.5f}\n'.format(v[0], v[1], v[2]))
        f.writelines(f_uv)

    print('save to', fname)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) but with a zero subgradient where x
    is 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def _index_from_letter(letter: str):
    if letter == 'X':
        return 0
    if letter == 'Y':
        return 1
    if letter == 'Z':
        return 2


def matrix_to_euler_angles(matrix, convention='XYZ'):
    """Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError('Convention must have 3 letters.')
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f'Invalid convention {convention}.')
    for letter in convention:
        if letter not in ('X', 'Y', 'Z'):
            raise ValueError(f'Invalid letter {letter} in convention string.')
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(matrix[..., i0, i2] *
                                   (-1.0 if i0 - i2 in [-1, 2] else 1.0))
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(convention[0], convention[1], matrix[..., i2], False,
                        tait_bryan),
        central_angle,
        _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True,
                        tait_bryan),
    )
    return torch.stack(o, -1)


def adjust_predicted_angles(pred_angles, target_angles):
    # 计算欧拉角的差异
    angle_diff = pred_angles - target_angles

    # 将差异映射到 [-pi, pi] 范围内
    angle_diff = (angle_diff + torch.pi) % (2 * torch.pi) - torch.pi

    # 将映射后的差异添加到预测值上
    adjusted_pred_angles = target_angles + angle_diff

    return adjusted_pred_angles


def _angle_from_tan(axis: str, other_axis: str, data, horizontal: bool,
                    tait_bryan: bool):
    """Extract the first or third Euler angle from the two members of the
    matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {'X': (2, 1), 'Y': (0, 2), 'Z': (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ['XY', 'YZ', 'ZX']
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        ))

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0]**2, m21 - m12, m02 - m20, m10 - m01],
                        dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1]**2, m10 + m01, m02 + m20],
                        dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2]**2, m12 + m21],
                        dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3]**2],
                        dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important;
    # if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (
        2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems,
    # quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    result = quat_candidates[F.one_hot(q_abs.argmax(
        dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
                             ].reshape(*batch_dim, 4)

    neg_row_ids = torch.where(result[:, 0] < 0)
    result[neg_row_ids] *= -1

    return result


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        hom_mat = torch.tensor([0, 0, 1]).float()
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        batch_size, device = rot_mat.shape[0], rot_mat.device
        hom_mat = hom_mat.view(1, 3, 1)
        hom_mat = hom_mat.repeat(batch_size, 1, 1).contiguous()
        hom_mat = hom_mat.to(device)
        rotation_matrix = torch.cat([rot_mat, hom_mat], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def rot6d_to_rotmat(x):
    x = x.view(-1, 3, 2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats


def rot6D_to_angular(rot6D):
    batch_size = rot6D.shape[0]
    pred_rotmat = rot6d_to_rotmat(rot6D).view(batch_size, -1, 3, 3)
    pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(
        -1, 3, 3)).reshape(batch_size, -1)
    return pose


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """This function is borrowed from https://github.com/kornia/kornia.

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            'Input must be a tensor of shape Nx4 or 4. Got {}'.format(
                quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """This function is borrowed from https://github.com/kornia/kornia.

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            'Input size must be a three dimensional tensor. Got {}'.format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            'Input size must be a N x 3 x 4  tensor. Got {}'.format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([
        rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0,
        rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]
    ], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([
        rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
        t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]
    ], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([
        rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2
    ], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([
        t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
        rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]
    ], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def _gen_rigid_features():
    rigid_samples = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        # xy plane
        [-1, -1, 0],
        # xz plane
        [-1, 0, -1],
        # yz plane
        [0, -1, -1],
    ])

    rigid_samples_rescaled = np.empty(rigid_samples.shape)
    expected_norm = 0.1

    for i in range(len(rigid_samples)):
        norm = np.linalg.norm(rigid_samples[i])
        if norm == 0:
            rigid_samples_rescaled[i] = rigid_samples[i]
        else:
            rigid_samples_rescaled[i] = rigid_samples[i] / norm * expected_norm

    rigid_samples_rescaled = torch.from_numpy(rigid_samples_rescaled).float()

    return rigid_samples_rescaled


def decode_svd(
    pred_pts_features: torch.Tensor,
    rigid_pts_src: torch.Tensor,
) -> torch.Tensor:
    batch_size = pred_pts_features.shape[0]
    rigid_points = pred_pts_features.reshape(pred_pts_features.shape[0], -1, 3)

    from_points = rigid_pts_src.to(rigid_points.device)
    from_points = (
        from_points.unsqueeze(0).expand(batch_size, from_points.shape[0],
                                        from_points.shape[1]).clone())

    wrist_xfs = procrustes_align(from_points,
                                 rigid_points).to(dtype=torch.float32)
    return wrist_xfs


class SkeletonEncoder(nn.Module):

    def __init__(
        self,
        output_feature_map_dim: int,
    ) -> None:
        super(SkeletonEncoder, self).__init__()
        # We have 16 joints. For each joint,
        # we use joint positions and joint axes as input features.
        n_skel_features = 96
        self._layers = nn.Sequential(
            nn.Linear(n_skel_features, output_feature_map_dim),
            nn.BatchNorm1d(output_feature_map_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, skeleton_features: torch.Tensor) -> torch.Tensor:

        skel_maps = self._layers(skeleton_features)

        return skel_maps


def trans_3d_2_2d(hand3d_point, leftcam_cam_matrix, rightcam_cam_matrix,
                  left_to_right_rt):
    B = hand3d_point.shape[0]
    left_to_right_rt = left_to_right_rt.repeat(B, 1, 1)
    leftcam_uv_reproj = torch.matmul(hand3d_point,
                                     leftcam_cam_matrix.permute(0, 2, 1)).to(
                                         torch.float32)
    leftcam_uv_reproj = leftcam_uv_reproj[..., :2] / leftcam_uv_reproj[..., 2:]

    column_of_ones = torch.ones((B, 21, 1)).to(hand3d_point.device)
    tensor_with_ones = torch.cat((hand3d_point, column_of_ones), dim=2)
    rightcam_uv_reproj = torch.matmul(tensor_with_ones,
                                      left_to_right_rt.permute(0, 2, 1)).to(
                                          torch.float32)
    rightcam_uv_reproj = rightcam_uv_reproj[..., :3] / rightcam_uv_reproj[...,
                                                                          3:]
    rightcam_uv_reproj = torch.matmul(rightcam_uv_reproj,
                                      rightcam_cam_matrix.permute(0, 2, 1)).to(
                                          torch.float32)
    rightcam_uv_reproj = rightcam_uv_reproj[..., :2] / rightcam_uv_reproj[...,
                                                                          2:]
    return leftcam_uv_reproj, rightcam_uv_reproj


def cal_proportion(uv_coor, leftcam_cam_matrix):
    B = uv_coor.shape[0]
    leftcam_x = (uv_coor[:, :, 0] - leftcam_cam_matrix[:, 0, 2].view(
        (B, 1))) / leftcam_cam_matrix[:, 0, 0].view((B, 1))
    leftcam_y = (uv_coor[:, :, 1] - leftcam_cam_matrix[:, 1, 2].view(
        (B, 1))) / leftcam_cam_matrix[:, 1, 1].view((B, 1))
    leftcam_xy = torch.cat((leftcam_x.unsqueeze(-1), leftcam_y.unsqueeze(-1)),
                           dim=2)  # (B, 21, 2)
    return leftcam_xy


def procrustes_align(
    from_points: torch.Tensor,
    to_points: torch.Tensor,
) -> torch.Tensor:
    """Inputs have same shape `(batch_size, n_points, 3)`. Within each sample
    of the batch, `from_points` and `to_points` implicitly correspond to each
    other along dim=1.

    Returns:
    - `rot`, `translation` with shape `(batch_size, 3, 3)`
    representing transformations for each example in batch
    """
    device = from_points.device

    batch_size = from_points.shape[0]
    from_mean = from_points.mean(dim=1)
    to_mean = to_points.mean(dim=1)

    from_centered = from_points - from_mean.reshape(-1, 1, 3)
    to_centered = to_points - to_mean.reshape(-1, 1, 3)

    outer_prod = torch.matmul(
        torch.transpose(from_centered, 1, 2),
        to_centered).to(dtype=torch.float32)

    u, _, v = outer_prod.svd()
    v_m_ut = torch.matmul(v, torch.transpose(u, 1, 2)).to(dtype=torch.float32)
    w = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    det = torch.det(v_m_ut)
    w[:, 2, 2] = det

    xfs = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    xfs[:, 0:3,
        0:3] = torch.matmul(torch.matmul(v, w), torch.transpose(u, 1, 2))
    xfs[:, 0:3, 3] = (
        to_mean -
        torch.matmul(xfs[:, 0:3, 0:3], from_mean.unsqueeze(-1)).squeeze())

    return xfs
