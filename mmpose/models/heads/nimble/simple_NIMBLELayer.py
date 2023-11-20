'''
    NIMBLE: A Non-rigid Hand Model with Bones and Muscles[SIGGRAPH-22]
    https://reyuwei.github.io/proj/nimble
'''

import torch
import trimesh
import sys
sys.path.append("./mmpose/models/heads/nimble")
from nimble_utils import *

class sim_NIMBLELayer(torch.nn.Module):
    __constants__ = [
        'use_pose_pca', 'shape_ncomp', 'pose_ncomp', 'pm_dict'
    ]
    def __init__(self, device, shape_ncomp=20, pose_ncomp=30, use_pose_pca=False):
        super(sim_NIMBLELayer, self).__init__()
        self.device = device
        self.base_nimble_path = "/data/AI_DATA_WX/data_hand/nimble_model/nimble_simple.npy"
        nimble_info = np.load(self.base_nimble_path, allow_pickle=True).item()

        identity_rot = torch.eye(3).to(self.device)
        self.register_buffer("identity_rot", identity_rot)
        
        self.shape_ncomp = shape_ncomp
        self.pose_ncomp = pose_ncomp
        self.use_pose_pca = use_pose_pca

        self.register_buffer("th_verts", torch.tensor(nimble_info["th_verts"]))   # shape (2500, 3)
        self.register_buffer("jreg_bone",  torch.tensor(nimble_info["jreg_bone"]))
        self.register_buffer("shape_basis",  torch.tensor(nimble_info["shape_basis"]))
        self.register_buffer("shape_pm_std",  torch.tensor(nimble_info["shape_pm_std"]))
        self.register_buffer("shape_pm_mean",  torch.tensor(nimble_info["shape_pm_mean"]))
        self.register_buffer("pose_basis",  torch.tensor(nimble_info["pose_basis"]))
        self.register_buffer("pose_mean",  torch.tensor(nimble_info["pose_mean"]))
        self.register_buffer("pose_pm_std",  torch.tensor(nimble_info["pose_pm_std"]))
        self.register_buffer("pose_pm_mean",  torch.tensor(nimble_info["pose_pm_mean"]))

        # Kinematic chain params
        kinetree = JOINT_PARENT_ID_DICT
        self.kintree_parents = []
        for i in range(STATIC_JOINT_NUM):
            self.kintree_parents.append(kinetree[i])

    def generate_hand_shape(self, betas, normalized=True):
        # beta : B, N
        
        batch_size, shape_ncomp = betas.shape
        assert self.shape_ncomp == shape_ncomp

        if normalized:
            betas_real = betas * self.shape_pm_std[:shape_ncomp].reshape(1, -1) + self.shape_pm_mean[:shape_ncomp].reshape(1, -1)
        else:
            betas_real = betas
        
        th_v_shaped = (self.shape_basis[:shape_ncomp].T @ betas_real.T).view(-1, 3, batch_size).permute(2, 0, 1) + self.th_verts.unsqueeze(0).repeat(batch_size, 1, 1)
        rebuild_jreg_bone_joints_tmp = self.jreg_bone.unsqueeze(0).repeat(batch_size, 1, 1).unsqueeze(-1) * th_v_shaped.reshape(batch_size, 25, 100, -1)
        rebuild_jreg_bone_joints = torch.sum(rebuild_jreg_bone_joints_tmp, dim=2)
        
        return th_v_shaped, rebuild_jreg_bone_joints     
        
    def generate_full_pose(self, theta, normalized=True, with_root=False):
        # theta : B, N

        batch_size = theta.shape[0]

        if with_root:
            real_theta = theta[:, 3:]
            root_rot = theta[:, :3]
        else:
            real_theta = theta
            root_rot = torch.zeros([batch_size, 3]).to(theta.device)

        pose_ncomp = real_theta.shape[-1]
        if normalized:
            theta_real_denorm = real_theta * self.pose_pm_std[:pose_ncomp].reshape(1, -1) + self.pose_pm_mean[:pose_ncomp].reshape(1, -1)
        else:
            theta_real_denorm = real_theta

        full_pose = (self.pose_basis[:pose_ncomp].T @ theta_real_denorm.T).T + self.pose_mean.unsqueeze(0).repeat(batch_size, 1)
        full_pose = torch.cat([root_rot, full_pose], dim=1).view(batch_size, -1, 3)

        return full_pose

    def convert_rot_to_pca(self, nimble_pose):
        full_pose_de = nimble_pose[:, 3:]
        batch_size = nimble_pose.shape[0]
        
        ##### debug
        # from IPython import embed; embed()
        # randn_pac = torch.rand(batch_size, self.pose_ncomp).cuda()
        # randn_pose = (self.pose_basis[:self.pose_ncomp].T @ randn_pac.T).T + self.pose_mean.unsqueeze(0).repeat(batch_size, 1)
        # full_pose_input = randn_pose
        
        # full_pose_input = nimble_pose[:, 3:]
        # theta_real_denorm_de = (torch.pinverse(self.pose_basis[:self.pose_ncomp].T) @ (full_pose_input - self.pose_mean.unsqueeze(0).repeat(batch_size, 1)).T).T   # 初步定位是这里的问题
        full_pose_demean = full_pose_de - self.pose_mean.unsqueeze(0).repeat(batch_size, 1)
        theta_real_denorm_de = torch.matmul(full_pose_demean, self.pose_basis[:self.pose_ncomp].T)
        # full_pose_out = (self.pose_basis[:self.pose_ncomp].T @ theta_real_denorm_de.T).T + self.pose_mean.unsqueeze(0).repeat(batch_size, 1)
        # torch.abs(full_pose_input - full_pose_out).max()
        
        #####
        
        real_theta_de = (theta_real_denorm_de - self.pose_pm_mean[:self.pose_ncomp].reshape(1, -1)) / (self.pose_pm_std[:self.pose_ncomp].reshape(1, -1))
        return real_theta_de

    def forward(self, pose_param, shape_param):
        """
        Takes points in R^3 and first applies relevant pose and shape blend shapes.
        Then performs skinning.
        """
        if self.use_pose_pca:
            full_pose = self.generate_full_pose(pose_param, normalized=True, with_root=False).view(-1, 20, 3)
        else:
            full_pose = pose_param.view(-1, 20, 3)

        # 得到在某个手型下的所有点的位置和关键骨骼点的位置
        th_v_shaped, jreg_joints = self.generate_hand_shape(shape_param,normalized=True)  

        bone_joints = self.forward_full(full_pose, jreg_joints)
        
        return None, bone_joints


    def forward_full(self, pose, joints,  global_scale=None):
        batch_size = pose.shape[0]

        # Convert axis-angle representation to rotation matrix rep.
        th_pose_map, th_rot_map = th_posemap_axisang_2output(pose.view(batch_size, -1))
        th_full_pose = pose.view(batch_size, -1, 3)
        root_rot = batch_rodrigues(th_full_pose[:, 0]).view(batch_size, 3, 3)

        th_j = joints

        th_results = []
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part
        for i in range(STATIC_JOINT_NUM - 1):
            i_val_joint = int(i + 1)
            if i_val_joint in JOINT_ID_BONE_DICT:
                i_val_bone = JOINT_ID_BONE_DICT[i_val_joint]
                joint_rot = th_rot_map[:, (i_val_bone - 1) * 9:i_val_bone * 9].contiguous().view(batch_size, 3, 3)
            else:
                joint_rot = self.identity_rot.repeat(batch_size, 1, 1)

            joint_j = th_j[:, i_val_joint, :].contiguous().view(batch_size, 3, 1)
            parent = self.kintree_parents[i_val_joint]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(torch.cat([joint_rot, joint_j - parent_j], 2))

            th_results.append(torch.matmul(th_results[parent], joint_rel_transform))

        th_results_global = th_results
        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]

        # global scaling
        if global_scale is not None:
            center_joint = th_jtr[:, ROOT_JOINT_IDX].unsqueeze(1)
            th_jtr = th_jtr - center_joint

            j_scale = global_scale.expand(th_jtr.shape[0], th_jtr.shape[1])
            j_scale = j_scale.unsqueeze(2).repeat(1, 1, 3)
            th_jtr = th_jtr * j_scale
            th_jtr = th_jtr + center_joint

        return th_jtr

        
    # nlayer = sim_NIMBLELayer(device, use_pose_pca=False, pose_ncomp=57, shape_ncomp=20)
    # bone_joints= nlayer.forward(pose_param, shape_param)



        