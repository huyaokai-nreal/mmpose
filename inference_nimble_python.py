import numpy as np
import torch

batch_size = 1
STATIC_JOINT_NUM = 25

JOINT_ID_BONE_DICT = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 5, 7: 6, 8: 7,10: 8,
                    11: 9, 12: 10, 13: 11, 15: 12, 16: 13, 17: 14, 18: 15,
                    20: 16, 21: 17, 22: 18, 23: 19}

kintree_parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23])

identity_rot = torch.tensor(np.array([[1,0,0],
                                      [0,1,0],
                                      [0,0,1]]))
kp_index = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22,
            23, 24]
scale_parameter = 1000

template_joints = np.array([[   9.5550,  -22.8580, -102.1520],
         [  35.5118,  -40.8386,  -91.7473],
         [  52.8191,  -66.0090,  -61.6418],
         [  59.9206,  -73.9186,  -34.2990],
         [  61.0230,  -84.2977,  -18.4595],
         [  27.7471,  -27.3750,  -85.3739],
         [  40.6270,  -39.2609,  -24.8645],
         [  45.2326,  -56.6830,   10.4099],
         [  45.3931,  -69.4421,   28.6310],
         [  45.0606,  -78.1369,   41.3210],
         [  17.0778,  -22.4696,  -81.3793],
         [  20.3420,  -33.0670,  -23.0879],
         [  17.9202,  -51.9830,   15.7123],
         [  14.9159,  -68.0406,   36.4155],
         [  12.5946,  -78.9575,   49.5932],
         [   7.4569,  -21.9225,  -79.2171],
         [   2.0990,  -32.3352,  -28.2736],
         [  -3.9275,  -50.1595,    7.2539],
         [  -7.9765,  -66.0658,   26.1811],
         [  -8.7711,  -78.5253,   38.2895],
         [  -2.4532,  -24.6924,  -80.5279],
         [ -15.2679,  -35.1095,  -35.4490],
         [ -24.7150,  -47.0335,   -7.0120],
         [ -29.8117,  -58.5901,    5.6566],
         [ -32.0271,  -69.2831,   15.7944]])

def forward_3d(shape_param, local_pose_matrix):
    # 得到scale后的shape
    shape_param =  torch.tensor(shape_param)
    jreg_joints = torch.tensor(template_joints).unsqueeze(0)
    scale_factor = 1 + shape_param[:, 0]
    root_jreg_joints = jreg_joints[:, 0:1, :]
    jreg_joints_relative = jreg_joints - root_jreg_joints
    jreg_joints = scale_factor.view(
        scale_factor.shape[0], 1,
        1) * jreg_joints_relative + root_jreg_joints
        
    # 得到旋转后的角度
    local_pose_matrix = torch.tensor(local_pose_matrix)
    th_pose_map, th_rot_map = th_posemap_axisang_2output_usematrix(
        local_pose_matrix)
    root_rot = torch.eye(3, 3).unsqueeze(0).repeat(batch_size, 1, 1).to(
        local_pose_matrix.device)
    th_results = []
    root_j = jreg_joints[:, 0, :].contiguous().view(batch_size, 3, 1)
    th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

    # Rotate each part
    for i in range(STATIC_JOINT_NUM - 1):
        i_val_joint = int(i + 1)
        if i_val_joint in JOINT_ID_BONE_DICT:
            i_val_bone = JOINT_ID_BONE_DICT[i_val_joint]
            joint_rot = th_rot_map[:, (i_val_bone - 1) * 9:i_val_bone *
                                    9].contiguous().view(batch_size, 3, 3)
        else:
            joint_rot = identity_rot.repeat(batch_size, 1, 1)

        joint_j = jreg_joints[:,
                        i_val_joint, :].contiguous().view(batch_size, 3, 1)
        parent = kintree_parents[i_val_joint]
        parent_j = jreg_joints[:, parent, :].contiguous().view(batch_size, 3, 1)
        joint_rel_transform = th_with_zeros(
            torch.cat([joint_rot.to(joint_j.device), joint_j - parent_j],
                        2))

        th_results.append(
            torch.matmul(th_results[parent], joint_rel_transform))
    th_results_global = th_results
    th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]
    return th_jtr



def th_posemap_axisang_2output_usematrix(pose_matrix):
    rot_mats = pose_matrix.reshape(pose_matrix.shape[0], -1)
    rot_nb = int(rot_mats.shape[1] / 9)
    id_flat = torch.eye(
        3, dtype=rot_mats.dtype,
        device=rot_mats.device).view(1, 9).repeat(rot_mats.shape[0], rot_nb)
    results = rot_mats - id_flat
    return results, rot_mats

def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False

    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res

def get_rel_kpt3d(bone_joints):
    rebuild_joints = bone_joints[:, kp_index, :]
    root_rebuild_joints = rebuild_joints[:, 0:1, :]
    rebuild_joints_temp = rebuild_joints - root_rebuild_joints
    result = rebuild_joints_temp / scale_parameter
    return result

def rot9D_to_matirx(x):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

    x: should have size [batch_size, 9]

    Output has size [batch_size, 3, 3].
    """
    x = torch.tensor(x)
    batch_size = x.shape[0]
    x_device = x.device
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    device = torch.device('cpu')
    det = torch.det(torch.matmul(u.to(device), vt.to(device)))
    det = det.view(-1, 1, 1).to(x_device)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    pred_matrix = torch.matmul(u, vt).view(batch_size, -1, 9)
    return pred_matrix.numpy()

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

def decode_svd(
    pred_pts_features: torch.Tensor,
) -> torch.Tensor:
    pred_pts_features = torch.tensor(pred_pts_features)
    rigid_pts_src = _gen_rigid_features()
    batch_size = pred_pts_features.shape[0]
    rigid_points = pred_pts_features.reshape(pred_pts_features.shape[0], -1, 3)

    from_points = rigid_pts_src.to(rigid_points.device)
    from_points = (
        from_points.unsqueeze(0).expand(batch_size, from_points.shape[0],
                                        from_points.shape[1]).clone())

    wrist_xfs = procrustes_align(from_points,
                                 rigid_points).to(dtype=torch.float32)
    return wrist_xfs

if __name__ == "__main__":
    shape_param = np.zeros((1,1)).astype(np.float32)
    local_pose_matrix = np.array([[ 0.8374, -0.5457,  0.0316,  0.5100,  0.7593, -0.4042,  0.1966,  0.3545, 0.9141],
        [ 0.3098,  0.7501,  0.5843, -0.9491,  0.2801,  0.1437, -0.0558, -0.5991, 0.7987],
        [ 0.8499,  0.0875,  0.5195, -0.1988,  0.9665,  0.1623, -0.4879, -0.2412, 0.8388],
        [-0.9040,  0.0683,  0.4221, -0.2445, -0.8925, -0.3791,  0.3508, -0.4459, 0.8235],
        [-0.0219, -0.9423, -0.3341,  0.9085,  0.1207, -0.4000,  0.4172, -0.3123, 0.8534],
        [-0.3897,  0.6170,  0.6837, -0.7965,  0.1469, -0.5866, -0.4623, -0.7732, 0.4342],
        [ 0.8696, -0.4269, -0.2482,  0.4386,  0.8986, -0.0088,  0.2268, -0.1012, 0.9687],
        [ 0.9547,  0.2948,  0.0388, -0.2929,  0.9549, -0.0492, -0.0516,  0.0356, 0.9980],
        [-0.9141,  0.3653,  0.1759, -0.4052, -0.8385, -0.3641,  0.0145, -0.4041, 0.9145],
        [-0.9977,  0.0439, -0.0519,  0.0169, -0.5792, -0.8148, -0.0658, -0.8138, 0.5771],
        [ 0.9827,  0.1728,  0.0643, -0.1713,  0.9845, -0.0269, -0.0680,  0.0154, 0.9970],
        [ 0.8534, -0.5098, -0.1086,  0.5120,  0.8589, -0.0081,  0.0974, -0.0487, 0.9940],
        [-0.4882,  0.7792,  0.3928, -0.8667, -0.3805, -0.3222, -0.1016, -0.4978, 0.8611],
        [ 0.0784,  0.7913,  0.6062, -0.7588,  0.4418, -0.4785, -0.6466, -0.4225, 0.6351],
        [-0.2649, -0.4396, -0.8578,  0.8746,  0.2649, -0.4055,  0.4052, -0.8580, 0.3143],
        [ 0.2253, -0.8797, -0.4187,  0.9741,  0.1947,  0.1150, -0.0196, -0.4337, 0.9008],
        [ 0.6307, -0.7751, -0.0298,  0.7096,  0.5610,  0.4259, -0.3136, -0.2899, 0.9038],
        [ 0.9280, -0.1882, -0.3207,  0.3528,  0.7195,  0.5979,  0.1181, -0.6682, 0.7341],
        [-0.8106,  0.3700, -0.4533,  0.4258, -0.1586, -0.8903, -0.4015, -0.9150, -0.0289]])
    
    bone_joints = forward_3d(shape_param, local_pose_matrix[None,:])


    rebuild_joints = bone_joints[:, kp_index, :]
    root_rebuild_joints = rebuild_joints[:, 0:1, :]
    rebuild_joints_temp = rebuild_joints - root_rebuild_joints
    result = rebuild_joints_temp / scale_parameter
    print(result)