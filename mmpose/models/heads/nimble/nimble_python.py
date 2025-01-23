import numpy as np
import torch

batch_size = 1
STATIC_JOINT_NUM = 25

JOINT_ID_BONE_DICT = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 5, 7: 6, 8: 7,10: 8,
                    11: 9, 12: 10, 13: 11, 15: 12, 16: 13, 17: 14, 18: 15,
                    20: 16, 21: 17, 22: 18, 23: 19}

kintree_parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23])
landmark_index = [1143, 3788, 567, 2963, 342, 2382, 2269, 3108, 3444, 3139, 114, 2514, 123, 2380, 4430, 
                               1184, 1859, 3754, 880, 3817, 1906]    #外掌心的
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
    jreg_joints = torch.tensor(template_joints).unsqueeze(0)
    scale_factor = 1 + shape_param[:, 0]
    root_jreg_joints = jreg_joints[:, 0:1, :]
    jreg_joints_relative = jreg_joints - root_jreg_joints
    
    jreg_joints = scale_factor.view(
        scale_factor.shape[0], 1,
        1) * jreg_joints_relative + root_jreg_joints
    
        
    # 得到旋转后的角度
    local_pose_matrix = torch.tensor(local_pose_matrix).unsqueeze(0)
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
    
    # 计算全手的landmark数值
    th_results2 = torch.zeros((batch_size, 4, 4, STATIC_JOINT_NUM),
                                dtype=root_j.dtype,
                                device=root_j.device)
    for i in range(STATIC_JOINT_NUM):
        padd_zero = torch.zeros(1, dtype=jreg_joints.dtype, device=jreg_joints.device)
        joint_j = torch.cat(
            [jreg_joints[:, i],
                padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
        tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
        th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)
    
    pm_dict = np.load("/data/stliu/NIMBLE_model/assets/NIMBLE_DICT_9137.pkl", allow_pickle=True)
    skin_v_index = pm_dict['skin_v_sep']  
    skinning_weight = pm_dict['all_sw'].to(jreg_joints.device).double()[skin_v_index:]
    points_pose_bs = pm_dict['all_pbs'].to(jreg_joints.device).double()[skin_v_index:]
    skin_v = pm_dict['vert'].to(jreg_joints.device).double()[skin_v_index:]

    
    skin_v_relative = skin_v - root_jreg_joints
    skin_v = scale_factor.view(
        scale_factor.shape[0], 1,
        1) * skin_v_relative + root_jreg_joints
    
    points_pose_bs = skin_v + torch.matmul(
        points_pose_bs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)

    skinning_weight = skinning_weight.reshape(1, -1, STATIC_JOINT_NUM)
    th_verts = compute_warp(batch_size, points_pose_bs,
                                    skinning_weight, th_results2)
    return th_jtr, th_verts


def compute_warp(batch_size, points, skinning_weights,
                    full_trans_mat):
    if points.shape[0] != batch_size:
        points = points.repeat(batch_size, 1, 1)
    if skinning_weights.shape[0] != batch_size:
        skinning_weights = skinning_weights.repeat(batch_size, 1, 1)

    th_T = torch.einsum('bijk,bkt->bijt', full_trans_mat,
                        skinning_weights.permute(0, 2, 1))
    th_rest_shape_h = torch.cat([
        points.transpose(2, 1),
        torch.ones((batch_size, 1, points.shape[1]),
                    dtype=skinning_weights.dtype,
                    device=skinning_weights.device),
    ], 1)
    th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
    th_verts = th_verts[:, :, :3]
    return th_verts

def th_pack(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new_zeros((batch_size, 4, 3))
    padding.requires_grad = False
    pack_list = [padding, tensor]
    pack_res = torch.cat(pack_list, 2)
    return pack_res

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

if __name__ == "__main__":
    shape_param = torch.tensor([-0.0146]).unsqueeze(0)
    local_pose_matrix = np.array([[-0.2078, -0.3727,  0.9044, -0.9669, -0.0617, -0.2475,  0.1480, -0.9259,
         -0.3475],
        [ 0.2026, -0.9699,  0.1348,  0.5783,  0.0075, -0.8158,  0.7903,  0.2432,
          0.5625],
        [ 0.7759, -0.2654,  0.5724,  0.0148,  0.9146,  0.4041, -0.6307, -0.3050,
          0.7135],
        [ 0.9998, -0.0068, -0.0199,  0.0057,  0.9985, -0.0549,  0.0203,  0.0548,
          0.9983],
        [-0.8260, -0.1810,  0.5338,  0.0473, -0.9660, -0.2543,  0.5616, -0.1848,
          0.8065],
        [ 0.2340, -0.8936, -0.3829,  0.9323,  0.3180, -0.1724,  0.2759, -0.3167,
          0.9075],
        [-0.2748, -0.8339, -0.4787,  0.7709,  0.1065, -0.6280,  0.5747, -0.5416,
          0.6135],
        [ 0.9816, -0.1860, -0.0432,  0.1865,  0.9824,  0.0091,  0.0408, -0.0169,
          0.9990],
        [ 0.7137,  0.3943,  0.5790, -0.6843,  0.5692,  0.4558, -0.1498, -0.7215,
          0.6760],
        [-0.7356, -0.4106, -0.5388,  0.6361, -0.1450, -0.7579,  0.2331, -0.9002,
          0.3678],
        [-0.8636, -0.1702, -0.4746,  0.5031, -0.2292, -0.8333,  0.0331, -0.9584,
          0.2836],
        [ 0.9439, -0.3179, -0.0890,  0.3226,  0.9455,  0.0437,  0.0702, -0.0700,
          0.9951],
        [-0.9846,  0.1715, -0.0346, -0.1559, -0.9500, -0.2704, -0.0793, -0.2608,
          0.9621],
        [ 0.1308,  0.7591,  0.6377, -0.8935,  0.3689, -0.2559, -0.4296, -0.5363,
          0.7265],
        [ 0.6455,  0.6738,  0.3596, -0.6922,  0.7151, -0.0976, -0.3229, -0.1859,
          0.9280],
        [-0.1974,  0.9733, -0.1169, -0.8709, -0.2289, -0.4349, -0.4501,  0.0160,
          0.8929],
        [-0.7808,  0.4320, -0.4513,  0.1935, -0.5196, -0.8322, -0.5940, -0.7371,
          0.3221],
        [ 0.8361,  0.5443,  0.0683, -0.5388,  0.8383, -0.0835, -0.1027,  0.0330,
          0.9942],
        [-0.6152,  0.7716,  0.1618, -0.1281,  0.1046, -0.9862, -0.7779, -0.6274,
          0.0345]])
        
    bone_joints, skin_landmarks = forward_3d(shape_param, local_pose_matrix)

    from IPython import embed; embed()
    from NIMBLELayer import NIMBLELayer
    from pytorch3d.structures.meshes import Meshes
    import pytorch3d
    import pytorch3d.io

    nlayer = NIMBLELayer("/data/stliu/NIMBLE_model/assets", "cpu", use_pose_pca=False, pose_ncomp=30)
    skin_p3dmesh = Meshes(skin_landmarks, nlayer.skin_f.repeat(1, 1, 1))
    pytorch3d.io.IO().save_mesh(skin_p3dmesh, "rand_bone.obj")
    
    
    

    rebuild_joints = bone_joints[:, kp_index, :]
    root_rebuild_joints = rebuild_joints[:, 0:1, :]
    rebuild_joints_temp = rebuild_joints - root_rebuild_joints
    result = rebuild_joints_temp / scale_parameter
    
    rebuild_skin_landmarks = skin_landmarks
    root_skin_landmarks = rebuild_skin_landmarks[:, 0:1, :]
    rebuild_skin_landmarks_temp = rebuild_skin_landmarks - root_skin_landmarks
    rebuild_skin_landmarks = rebuild_skin_landmarks_temp / scale_parameter
    
    print(result)