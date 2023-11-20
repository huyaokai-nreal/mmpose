# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from mmengine.logging import MessageHub
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmpose.models.utils.gmlp import gMLP
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions
import sys
sys.path.append("mmpose/models/heads")
from nimble.NIMBLELayer import NIMBLELayer, procrustes_align
from nimble.simple_NIMBLELayer import sim_NIMBLELayer

def _gen_rigid_features():
    rigid_samples = np.array(
        [
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
        ]
    )

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

@MODELS.register_module()
class LiftHead_Rotation(BaseModule):
    """liftHead for getting 3d rotation from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 channel_num: int = 55,
                #  output_num: int = 71,                  # rot 30 + shape 20 + svd21
                 undistort: bool = False,
                 use_kp2d_gt=False,
                 kpt2d_with_depth: bool = False,
                 noRt=False,
                 lambda_t: int = -1,
                 corruption_cam: float = 0.5,
                 pre_xyz_type: int=0,
                 use_svd: bool = True,
                 use_nimble_part_para: bool = False,
                 use_nimble_pca: bool = False,
                 use_sim_nimble: bool = False,
                 shape_ncomp: int = 20,
                 pose_ncomp: int = 60,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_num = channel_num
        self.lambda_t = lambda_t
        self.kpt2d_with_depth = kpt2d_with_depth
        feat_dim = 2 * self.channel_num
        if self.kpt2d_with_depth:
            feat_dim = feat_dim + 21
        self.corruption_cam = corruption_cam
        self.liftnet = gMLP(d_model=feat_dim, d_ffn=feat_dim * 2, num_layers=3)
        self.rigid_samples = _gen_rigid_features()
        self.use_svd = use_svd
        
        self.shape_ncomp = shape_ncomp
        self.pose_ncomp = pose_ncomp
        if use_nimble_part_para:
            self.pose_ncomp = 60
        
        output_num = self.shape_ncomp + self.pose_ncomp + 3
        if self.use_svd:
            self.output_num = output_num + 18 # 21 - 3
        else:
            self.output_num = output_num
        self.last_layer = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.SyncBatchNorm(feat_dim), nn.ReLU(),
            nn.Conv2d(feat_dim, self.output_num, kernel_size=1))
        self.lift_loss = MODELS.build(lift_loss)
        self.undistort = undistort
        self.use_kp2d_gt = use_kp2d_gt
        self.noRt = noRt
        self.pre_xyz_type = pre_xyz_type
        
        self.use_nimble_part_para = use_nimble_part_para
        self.used_nimble_para = [51, 52, 54, 57, 39, 40, 42, 45, 27, 28, 30, 33, 15, 16, 18, 21, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.use_nimble_pca = use_nimble_pca
        

        # define nimble layer
        if use_sim_nimble:
            self.nimble_layer = sim_NIMBLELayer(device="cuda",  shape_ncomp=self.shape_ncomp, pose_ncomp=self.pose_ncomp, use_pose_pca = self.use_nimble_pca)
        else:
            self.nimble_layer = NIMBLELayer(
                base_path="/data/AI_DATA_WX/data_hand/nimble_model",
                device="cuda",
                shape_ncomp=self.shape_ncomp, 
                pose_ncomp=self.pose_ncomp,
                use_pose_pca = self.use_nimble_pca)    
        self.kp_index = [
            0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22,
            23, 24
        ]
        self.scale_parameter = 1000
        
        if self.use_nimble_pca and self.use_nimble_part_para:
            raise ValueError("Both self.use_nimble_pca and self.use_nimble_part_para cannot be True at the same time.")

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        output = self.liftnet(feats)
        output = self.last_layer(output).view(
            (feats.shape[0], -1, 1, 1))  # [64, 83, 1, 1]
        return output

    def preprocess(self, feats, batch_data_samples):
        
        xy_coord = feats[..., :2]
        if self.kpt2d_with_depth:
            depth = feats[..., -1:][::2]
        B = int(len(batch_data_samples) / 2)
        N = 2
        H, W = batch_data_samples[0].input_size
        K = xy_coord.shape[1]  # (B,21, 2)

        # kpt2d output to crop wh
        uv_coord_im_pred_crop_right = xy_coord * torch.tensor([W, H]).cuda()
        uv_coord_im_pred_crop_right = uv_coord_im_pred_crop_right.view(
            B, N, K, 2)

        leftcam_cam_matrix = []
        rightcam_cam_matrix = []
        lr_p = []
        lr_rot_matrix = []
        hand3d_gt = []
        is_left_hands = []
        nimble_pose = []
        nimble_trans = []
        nimble_shape = []
        nimble_info = dict()

        uv_coord_im_gt_global = []

        all_inv_warp_mat = torch.zeros(B * 2, 3, 2).cuda()
        all_inv_warp_mat.requires_grad = False
        for i, data_sample in enumerate(batch_data_samples):
            if i % 2 == 0:
                left_camera = data_sample.meta['ori_camera']
                left_cam_matrix = left_camera.uv_to_window_matrix()
                leftcam_cam_matrix.append(left_cam_matrix)
                hand3d_gt.append(data_sample.gt_instances.keypoints3d[0])
                if "nimble_pose" in data_sample.meta.keys():
                    nimble_pose.append(data_sample.meta["nimble_pose"])
                    nimble_trans.append(data_sample.meta["nimble_translation"])
                    nimble_shape.append(data_sample.meta["nimble_shape"])
                if data_sample.meta['category_id'] == 1:  # 1: left, 2: right
                    is_left_hands.append(1)
                else:
                    is_left_hands.append(0)
            else:
                right_camera = data_sample.meta['ori_camera']
                right_cam_matrix = right_camera.uv_to_window_matrix()
                rightcam_cam_matrix.append(right_cam_matrix)
                left_cam_xf = left_camera.camera_to_world_xf
                right_cam_xf = right_camera.camera_to_world_xf
                lr_t = np.dot(np.linalg.inv(left_cam_xf),
                              right_cam_xf).astype(np.float32)
                lr_rot_matrix.append(lr_t[:3, :3])                  # lr_rot_matrix 代表的是 右相机向左相机的变换
                lr_p.append(lr_t[:3, 3])

            warp_mat = data_sample.metainfo['warp_mat']            # warp mat 代表的是从原图到crop图像的映射
            inv_warp_mat = cv2.invertAffineTransform(warp_mat).astype(
                np.float32)
            inv_warp_mat = torch.from_numpy(inv_warp_mat).cuda()  # (2,3)
            all_inv_warp_mat[i] = inv_warp_mat.transpose(0, 1)  # (3,2)

            uv_coord_im_gt_global.append(data_sample.gt_instances.keypoints)
        leftcam_cam_matrix = torch.tensor(
            np.array(leftcam_cam_matrix)).cuda().float()
        rightcam_cam_matrix = torch.tensor(
            np.array(rightcam_cam_matrix)).cuda().float()
        lr_p = torch.tensor(np.array(lr_p)).cuda().float()
        lr_rot_matrix = torch.tensor(np.array(lr_rot_matrix)).cuda().float()
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
        nimble_pose = torch.tensor(np.array(nimble_pose)).cuda().float()
        nimble_trans = torch.tensor(np.array(nimble_trans)).cuda().float()
        nimble_shape = torch.tensor(np.array(nimble_shape)).cuda().float()
        if nimble_pose.shape[0]>0:
            nimble_info = {
                "nimble_pose":nimble_pose,
                "nimble_trans":nimble_trans,
                "nimble_shape":nimble_shape
            }
        left_rel_depth = hand3d_gt[..., 2:3] - hand3d_gt[:, :1, 2:3]
        left_hand = torch.tensor(np.array(is_left_hands)).cuda().float()
        uv_coord_im_gt_global = torch.tensor(
            np.array(uv_coord_im_gt_global)).cuda().float()
        uv_coord_im_gt_global = uv_coord_im_gt_global.view(B, N, K, 2)

        def recover_hand(uv_coord_im_pred, left_hand, w):
            recover_uv_coord_im_pred = (
                1 - left_hand.view(size=(-1, 1, 1, 1))
            ) * uv_coord_im_pred + left_hand.view(size=(-1, 1, 1, 1)) * (
                torch.tensor([w - 1, 0]).view(size=(1, 1, 1, 2)).cuda() +
                torch.tensor([-1, 1]).view(size=(1, 1, 1, 2)).cuda() *
                uv_coord_im_pred)
            return recover_uv_coord_im_pred

        uv_coord_im_pred_crop_leftright = uv_coord_im_pred_crop_right
        # uv_coord_im_pred_crop_leftright = recover_hand(
        #     uv_coord_im_pred_crop_right, left_hand, device, W)

        uv_coord_im_pred_crop_leftright = uv_coord_im_pred_crop_leftright.view(
            B * N, K, 2)

        # from crop uv to global uv，也就是加了一个纬度z
        uv_coord_im_pred = torch.cat(
            [uv_coord_im_pred_crop_leftright,
             torch.ones(B * 2, K, 1).cuda()],
            dim=-1)
        # 将crop（仿射变换）的图像的关键点恢复到在原图中位置的点，得到的uv_coord_im_pred（对应shape为（256, 21, 3））* all_inv_warp_mat（对应shape为（256, 3, 2））= shape为（256, 21, 2）
        uv_coord_im_pred_global_distort = torch.bmm(uv_coord_im_pred,    
                                                    all_inv_warp_mat)
        uv_coord_im_pred_global_distort = uv_coord_im_pred_global_distort.view(     # 
            B, N, K, 2)

        # 这里将原来的点（不区分左右手的），现在经过了resize以及左手翻转回对应的坐标，这时候是真正的左右的的在原图的坐标位置了（这里指的是还没有去过畸变的点）
        frame_width = batch_data_samples[0].meta['frame_width']
        uv_coord_im_pred_global_distort_noflip = recover_hand(
            uv_coord_im_pred_global_distort, left_hand, frame_width)
        # 执行了相同的操作，只是这里是对真值来做的
        uv_coord_im_gt_global = recover_hand(uv_coord_im_gt_global, left_hand,
                                             frame_width)

        if self.use_kp2d_gt:
            uv_coord_im_pred_global = uv_coord_im_gt_global
            depth = left_rel_depth

        # 这里指的是去畸变的过程，把预测的点进行去畸变的操作，得到去畸变后的21个点的uv数值
        if self.undistort:
            uv_coord_im_pred_global = \
                uv_coord_im_pred_global_distort_noflip.clone().view(-1, K, 2)
            for i, data_sample in enumerate(batch_data_samples):
                camera_model = data_sample.meta['ori_camera']
                kpt2d_u = camera_model.undistort(
                    uv_coord_im_pred_global[i].cpu().numpy())
                uv_coord_im_pred_global[i] = torch.from_numpy(kpt2d_u).cuda()
            uv_coord_im_pred_global = uv_coord_im_pred_global.view(B, N, K, 2)
        else:
            uv_coord_im_pred_global = \
                uv_coord_im_pred_global_distort_noflip.clone()

        # 这地方可以理解是uv坐标系向相机坐标系的转化，能够得到z方向为1的向量
        leftcam_uv = uv_coord_im_pred_global[:, 0]  # (B, 21, 2)
        leftcam_x = (leftcam_uv[:, :, 0] - leftcam_cam_matrix[:, 0, 2].view(
            (B, 1))) / leftcam_cam_matrix[:, 0, 0].view((B, 1))
        leftcam_y = (leftcam_uv[:, :, 1] - leftcam_cam_matrix[:, 1, 2].view(
            (B, 1))) / leftcam_cam_matrix[:, 1, 1].view((B, 1))
        leftcam_xy = torch.cat(
            (leftcam_x.unsqueeze(-1), leftcam_y.unsqueeze(-1)),
            dim=2)  # (B, 21, 2)
        rightcam_uv = uv_coord_im_pred_global[:, 1]  # (B, 21, 2)
        rightcam_x = (rightcam_uv[:, :, 0] - rightcam_cam_matrix[:, 0, 2].view(
            (B, 1))) / rightcam_cam_matrix[:, 0, 0].view((B, 1))
        rightcam_y = (rightcam_uv[:, :, 1] - rightcam_cam_matrix[:, 1, 2].view(
            (B, 1))) / rightcam_cam_matrix[:, 1, 1].view((B, 1))
        rightcam_xy = torch.cat(
            (rightcam_x.unsqueeze(-1), rightcam_y.unsqueeze(-1)),
            dim=2)  # (B, 21, 2)

        # 显式的使用Rt
        if self.noRt:
            rightcam_xy_normplane = torch.cat(
                (rightcam_xy, torch.ones(B, K, 1).cuda()), dim=2)
            rightcam_xyz1_inworld = (torch.bmm(
                lr_rot_matrix.view((B, 1, 3, 3)).repeat(1, 21, 1, 1).view(
                    (B * 21, 3, 3)), rightcam_xy_normplane.view(
                        (B * K, 3, 1))) + lr_p.view(
                            (B, 1, 3, 1)).repeat(1, 21, 1, 1).view(
                                (B * 21, 3, 1))).view((B, 21, 3))
            # rightcam_xyz1_inworld = (torch.bmm(       
            #     rightcam_xy_normplane.view((B * K, 3, 1)).permute(0, 2, 1),
            #     lr_rot_matrix.view((B, 1, 3, 3)).repeat(1, 21, 1, 1).view(
            #         (B * 21, 3, 3)))  + lr_p.view(
            #                 (B, 1, 3, 1)).repeat(1, 21, 1, 1).view(
            #                     (B * 21, 3, 1))).view((B, 21, 3))
            feats = torch.cat(
                (leftcam_xy.view(B, -1), rightcam_xyz1_inworld.view(
                    (B, -1)), left_hand.view((B, -1))),
                dim=1).view(B, self.channel_num * 2, 1,
                            1).float()  # 21*2+21*3+1
        # 隐式的使用Rt
        else:
            Tmatrix_leftcam = torch.tensor(
                (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)).view((1, -1)).cuda()
            feature1 = torch.cat((leftcam_xy.view(
                (B, -1)), Tmatrix_leftcam.repeat(B, 1), left_hand.view(
                    (B, -1))),
                                 dim=1).view((B, self.channel_num, 1, 1))
            feature2 = torch.cat((rightcam_xy.view((B, -1)), lr_p.view(
                (B, -1)), lr_rot_matrix.view((B, -1)), left_hand.view(
                    (B, -1))),
                                 dim=1).view((B, self.channel_num, 1, 1))
            if self.kpt2d_with_depth:
                feats = torch.torch.cat(
                    (feature1, feature2, depth.reshape(
                        (B, 21, 1, 1))), dim=1).float()
            else:
                feats = torch.cat((feature1, feature2), dim=1).float()

        return (feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p,
                leftcam_cam_matrix, rightcam_cam_matrix,
                uv_coord_im_pred_global, uv_coord_im_pred_global_distort,
                hand3d_gt, left_hand, nimble_info)
    
    def postprocess(self, output, left_hand, leftcam_xy):
        
        B = output.shape[0]
        
        if self.use_nimble_pca:
            pose_len = self.pose_ncomp
        else:
            pose_len = 60
        rot_vector = output[:, :pose_len, 0, 0]
                
        if self.use_svd:
            shape_vector = output[:,pose_len:(pose_len+self.shape_ncomp), 0, 0]
            pre_pt_features = output[:,(pose_len+self.shape_ncomp):, 0, 0]
        else:
            trans_vector = output[:, pose_len:pose_len+3, 0, 0]
            shape_vector = output[:,pose_len+3:, 0, 0]
        
        
        if self.use_svd:
            svd_begin = self.pose_ncomp + self.shape_ncomp
            pre_pt_features = output[:,svd_begin:, 0, 0]
            
        if self.use_nimble_part_para:
            used_nimble_para = torch.tensor(self.used_nimble_para)
            rot_vector_zeros = torch.zeros((rot_vector.shape[0], rot_vector.shape[1]), device=rot_vector.device, dtype = torch.float32)
            rot_vector_zeros[:,used_nimble_para] = rot_vector[:,used_nimble_para].to(torch.float32)
            rot_vector = rot_vector_zeros
        
        if not self.use_svd:
            skin_v, bone_joints = self.nimble_layer.forward(rot_vector, shape_vector)
            rebuild_joints = bone_joints[:, self.kp_index, :]
            root_rebuild_joints = rebuild_joints[:, 0:1, :]
            
            # 对于左手要有额外的操作，x方向偏执
            mask = left_hand == 1 
            rebuild_joints_temp = rebuild_joints - root_rebuild_joints
            rebuild_joints_temp[mask, :, 0] = -rebuild_joints_temp[mask, :, 0]
            rebuild_joints_with_scale = rebuild_joints_temp / self.scale_parameter
            
            # 对于左手要有额外的操作，加入旋转矩阵
            Rot = torch.eye(3).unsqueeze(0).repeat(len(mask), 1, 1)
            special_rotation = torch.tensor([[-0.95179426, 0.27416083, 0.13756282],
                                    [-0.27311822, -0.55333527, -0.78690947],
                                    [-0.13962139, -0.78654683, 0.60153965]])
            Rot[mask] = special_rotation.unsqueeze(0)
            rebuild_joints_with_scale = torch.bmm(Rot.cuda(), rebuild_joints_with_scale.permute(0,2,1)).permute(0,2,1)

            nimble_xyz = rebuild_joints_with_scale + trans_vector.view((B, 1, 3, 1)).repeat(1, 21, 1, 1).view((B * 21, 3, 1)).view((B, 21, 3))
            trans_xyz = output[:, 60:63, 0, 0]
        else:
            init_root_rot = torch.zeros((B, 3), requires_grad=True, device=rot_vector.device)
            new_rot_vector = torch.cat((init_root_rot, rot_vector[:,3:]),dim=1)
            skin_v, bone_joints = self.nimble_layer.forward(new_rot_vector, shape_vector)
            rebuild_joints = bone_joints[:, self.kp_index, :]
            root_rebuild_joints = rebuild_joints[:, 0:1, :]
            
            mask = left_hand == 1 
            rebuild_joints_temp = rebuild_joints - root_rebuild_joints
            rebuild_joints_temp[mask, :, 0] = -rebuild_joints_temp[mask, :, 0]
            rebuild_joints_with_scale = rebuild_joints_temp / self.scale_parameter
            
            matrix_svd = self.decode_svd(
                    pre_pt_features,
                    self.rigid_samples,
                )
            nimble_xyz = torch.matmul(rebuild_joints_with_scale, matrix_svd[:, 0:3, 0:3].transpose(1, 2)) + matrix_svd[:, 0:3, 3].unsqueeze(1)
            trans_xyz = matrix_svd[:, 0:3, 3]
            
            
        if self.pre_xyz_type==0:
            return nimble_xyz, nimble_xyz, nimble_xyz, trans_xyz
        else:
            nimble_Z = nimble_xyz[:,:,2:].view((B, 21, 1))
            rebuild_xyz = torch.cat((leftcam_xy * nimble_Z, nimble_Z), dim=2).view((B, 21, 3))
            if self.pre_xyz_type == 1:
                return rebuild_xyz, rebuild_xyz, rebuild_xyz, trans_xyz
            elif self.pre_xyz_type == 2:
                hand3d_pred = (self.corruption_cam * rebuild_xyz +
                (1 - self.corruption_cam) * nimble_xyz)
                return hand3d_pred, nimble_xyz, rebuild_xyz, trans_xyz

    def decode_svd(self,
        pred_pts_features: torch.Tensor,
        rigid_pts_src: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = pred_pts_features.shape[0]
        rigid_points = pred_pts_features.reshape(pred_pts_features.shape[0], -1, 3)

        from_points = rigid_pts_src.to(rigid_points.device)
        from_points = (
            from_points.unsqueeze(0)
            .expand(batch_size, from_points.shape[0], from_points.shape[1])
            .clone()
        )

        wrist_xfs = procrustes_align(from_points, rigid_points).to(dtype=torch.float32)
        return wrist_xfs

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            (feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p,
             leftcam_cam_matrix, rightcam_cam_matrix, uv_coord_im_pred_global,
             uv_coord_im_pred_global_distort,
             hand3d_gt, left_hand, nimble_info) = self.preprocess(feats, batch_data_samples)
        output = self.forward(feats)
        hand3d_pred  = self.postprocess(output, left_hand, leftcam_xy)[1]
        
        leftcam_uv_reproj = torch.matmul(hand3d_pred,
                                         leftcam_cam_matrix.permute(0, 2, 1))
        leftcam_uv_reproj = \
            leftcam_uv_reproj[..., :2] / leftcam_uv_reproj[..., 2:]
        camera_model = batch_data_samples[0].meta[
            'ori_camera']  # leftcam model
        leftcam_uv_reproj_distort = camera_model.eye_to_window(
            hand3d_pred.cpu().numpy())
        leftcam_uv_reproj_distort = torch.tensor(
            leftcam_uv_reproj_distort).cuda()
        
        return hand3d_pred, uv_coord_im_pred_global_distort
        # return hand3d_pred, leftcam_uv_reproj[:, None, ...]

    def cal_proportion(self, uv_coor, leftcam_cam_matrix):
        B = uv_coor.shape[0]
        leftcam_x = (uv_coor[:, :, 0] - leftcam_cam_matrix[:, 0, 2].view(
            (B, 1))) / leftcam_cam_matrix[:, 0, 0].view((B, 1))
        leftcam_y = (uv_coor[:, :, 1] - leftcam_cam_matrix[:, 1, 2].view(
            (B, 1))) / leftcam_cam_matrix[:, 1, 1].view((B, 1))
        leftcam_xy = torch.cat(
            (leftcam_x.unsqueeze(-1), leftcam_y.unsqueeze(-1)),
            dim=2)  # (B, 21, 2)
        return leftcam_xy

    def trans_3d_2_2d(self, hand3d_point, leftcam_cam_matrix, rightcam_cam_matrix, lr_rot_matrix, lr_p):
        
        # left_point_normal_t = hand3d_point[..., :2] / hand3d_point[...,  2:]
        # left_point_normal = torch.cat([left_point_normal_t,torch.ones(left_point_normal_t.shape[0], left_point_normal_t.shape[1], 1).cuda()], dim=-1)
        # leftcam_uv_reproj = torch.matmul(leftcam_cam_matrix.unsqueeze(1).repeat(1, 21, 1, 1).to(torch.float32), left_point_normal.unsqueeze(-1).to(torch.float32)).to(torch.float32)[:,:,:2,0]

        # right_point = torch.matmul((hand3d_point - lr_p.unsqueeze(1)), torch.inverse(lr_rot_matrix))
        # right_point_normal_t = right_point[..., :2] / right_point[...,  2:]
        # right_point_normal = torch.cat([right_point_normal_t,torch.ones(right_point_normal_t.shape[0], right_point_normal_t.shape[1], 1).cuda()], dim=-1)
        # rightcam_uv_reproj = torch.matmul(rightcam_cam_matrix.unsqueeze(1).repeat(1, 21, 1, 1), right_point_normal.unsqueeze(-1))[:,:,:2,0]
        
        leftcam_uv_reproj = torch.matmul(hand3d_point, leftcam_cam_matrix.permute(0, 2, 1)).to(torch.float32)
        leftcam_uv_reproj = leftcam_uv_reproj[..., :2] / leftcam_uv_reproj[...,  2:]
        
        rightcam_uv_reproj = torch.matmul((hand3d_point - lr_p.unsqueeze(1)), torch.inverse(lr_rot_matrix)).to(torch.float32)
        rightcam_uv_reproj = torch.matmul(rightcam_uv_reproj,
                                          rightcam_cam_matrix.permute(0, 2, 1)).to(torch.float32)
        rightcam_uv_reproj = rightcam_uv_reproj[..., :2] / rightcam_uv_reproj[..., 2:]  

        return leftcam_uv_reproj, rightcam_uv_reproj

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        with torch.no_grad():
            (feats, leftcam_xy, rightcam_xy, lr_rot_matrix, lr_p,
             leftcam_cam_matrix, rightcam_cam_matrix, uv_coord_im_pred_global,
             uv_coord_im_pred_global_distort,
             hand3d_gt, left_hand, nimble_info) = self.preprocess(feats, batch_data_samples)
        

        output = self.forward(feats)
        
        # 3d 损失
        hand3d_pred, hand3d_nimble_xyz, hand3d_rebuild_xyz, trans_xyz = self.postprocess(output, left_hand, leftcam_xy)

    
        # 直接监督rot和trans
        if self.use_nimble_pca:
            pre_nimble_trans = trans_xyz
            if "nimble_pose" in nimble_info.keys() and  "nimble_trans" in nimble_info.keys():
                gt_nimble_trans = nimble_info["nimble_trans"]
                gt_nimble_pose_rot = nimble_info["nimble_pose"].reshape(-1,60)
            pre_nimble_pose = output[:, :self.pose_ncomp, 0, 0]
            gt_nimble_pose = self.nimble_layer.convert_rot_to_pca(gt_nimble_pose_rot)
        else:
            pre_nimble_pose = output[:, :self.pose_ncomp, 0, 0]
            pre_nimble_trans = trans_xyz
            if "nimble_pose" in nimble_info.keys() and  "nimble_trans" in nimble_info.keys():
                gt_nimble_pose = nimble_info["nimble_pose"].reshape(-1,60)
                gt_nimble_trans = nimble_info["nimble_trans"]
            else:
                gt_nimble_pose = pre_nimble_pose
                gt_nimble_trans = pre_nimble_trans
                
            if self.use_nimble_part_para:
                used_nimble_para = torch.tensor(self.used_nimble_para)
                rot_vector_zeros = torch.zeros((pre_nimble_pose.shape[0], pre_nimble_pose.shape[1]), device=pre_nimble_pose.device, dtype = torch.float32)
                rot_vector_zeros[:,used_nimble_para] = rot_vector[:,used_nimble_para].to(torch.float32)
                pre_nimble_pose = rot_vector_zeros

            pre_nimble_pose = pre_nimble_pose[:,3:]
            gt_nimble_pose = gt_nimble_pose[:,3:]
        
        # 2d重投影损失 这里把pre设置为gt
        leftcam_uv_pre, rightcam_uv_pre = self.trans_3d_2_2d(hand3d_pred, leftcam_cam_matrix, rightcam_cam_matrix, lr_rot_matrix, lr_p)
        leftcam_uv_gt, rightcam_uv_gt = self.trans_3d_2_2d(hand3d_gt, leftcam_cam_matrix, rightcam_cam_matrix, lr_rot_matrix, lr_p)
        
        # xyz比例约束
        proportion_xyz_pre = self.cal_proportion(leftcam_uv_pre,leftcam_cam_matrix)
        proportion_xyz_gt = self.cal_proportion(leftcam_uv_gt,leftcam_cam_matrix)
        
        # 数据归一化
        leftcam_uv_pre = leftcam_uv_pre/500
        rightcam_uv_pre = rightcam_uv_pre/500
        leftcam_uv_gt = leftcam_uv_gt/500
        rightcam_uv_gt = rightcam_uv_gt/500        

        # pinch 损失
        dist_pred = torch.norm(
            hand3d_nimble_xyz[:, 4, :] - hand3d_nimble_xyz[:, 8, :], dim=-1)
        dist_gt = torch.norm(hand3d_gt[:, 4, :] - hand3d_gt[:, 8, :], dim=-1)
        
        
        pred_for_loss = [
            hand3d_pred, hand3d_nimble_xyz, hand3d_rebuild_xyz, leftcam_uv_pre,
            rightcam_uv_pre, dist_pred, proportion_xyz_pre, pre_nimble_pose, pre_nimble_trans
        ]
        targ_for_loss = [
            hand3d_gt, hand3d_gt, hand3d_gt, leftcam_uv_gt, rightcam_uv_gt,
            dist_gt, proportion_xyz_gt, gt_nimble_pose, gt_nimble_trans
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss)
        (loss_mse_3d, loss_mse_3d_leftcam, loss_mse_3d_rightcam,
         loss_mse_2d_leftcam, loss_mse_2d_rightcam, loss_pinch, loss_scale, loss_nimble_pose, loss_nimble_trans) = losses
    
        
        if self.lambda_t > 0:
            mh = MessageHub.get_current_instance()
            cur_epoch = mh.get_info('epoch')
            if cur_epoch <= self.lambda_t:
                loss_mse_2d_leftcam = torch.tensor(0.0, device=loss_mse_2d_leftcam.device, requires_grad=False)
                loss_mse_2d_rightcam = torch.tensor(0.0, device=loss_mse_2d_rightcam.device, requires_grad=False)
                loss_scale = torch.tensor(0.0, device=loss_scale.device, requires_grad=False)
                
        losses_dict = dict(
            loss_mse_3d=loss_mse_3d,
            loss_mse_3d_leftcam=loss_mse_3d_leftcam,
            loss_mse_3d_rightcam=loss_mse_3d_rightcam,
            loss_mse_2d_leftcam=loss_mse_2d_leftcam,
            loss_mse_2d_rightcam=loss_mse_2d_rightcam,
            loss_pinch = loss_pinch,
            loss_proportion = loss_scale,
            loss_nimble_pose = loss_nimble_pose, 
            loss_nimble_trans = loss_nimble_trans
        )    

        return losses_dict
