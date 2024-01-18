# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmpose.models.heads.nimble.nimble_utils import (_gen_rigid_features,
                                                     batch_rodrigues,
                                                     decode_svd)
from mmpose.models.heads.nimble.simple_NIMBLELayer import sim_NIMBLELayer
from mmpose.models.utils.gmlp import gMLP
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


class GMLPModel(nn.Module):

    def __init__(self, input_size, hidden_dim=512, output_size=2048):
        super(GMLPModel, self).__init__()
        self.first_layer = nn.Linear(input_size, hidden_dim // 2)
        self.first_relu = nn.ReLU()
        self.second_layer = nn.Linear(hidden_dim // 2, hidden_dim)
        self.second_relu = torch.nn.ReLU()
        self.liftnet = gMLP(
            d_model=hidden_dim, d_ffn=hidden_dim * 2, num_layers=5)
        self.last_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SyncBatchNorm(hidden_dim), nn.ReLU(),
            nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=1),
            nn.SyncBatchNorm(2 * hidden_dim), nn.ReLU(),
            nn.Conv2d(2 * hidden_dim, output_size, kernel_size=1))

    def forward(self, x):
        x = x.view(x.shape[0] * x.shape[1], -1)
        x = self.first_layer(x)
        x = self.first_relu(x)
        x = self.second_layer(x)
        x = self.second_relu(x).unsqueeze(-1).unsqueeze(-1)

        x = self.liftnet(x)
        x = self.last_layer(x)
        return x


class GMLPModel_Large(nn.Module):

    def __init__(self, input_size, hidden_dim=2048, output_size=2048):
        super(GMLPModel_Large, self).__init__()
        self.first_layer = nn.Linear(input_size, hidden_dim // 2)
        self.first_relu = nn.ReLU()
        self.second_layer = nn.Linear(hidden_dim // 2, hidden_dim)
        self.second_relu = nn.ReLU()
        self.liftnet = gMLP(
            d_model=hidden_dim, d_ffn=hidden_dim * 4, num_layers=5)
        self.last_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SyncBatchNorm(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_size, kernel_size=1),
        )

    def forward(self, x):
        x = x.view(x.shape[0] * x.shape[1], -1)
        x = self.first_layer(x)
        x = self.first_relu(x)
        x = self.second_layer(x)
        x = self.second_relu(x).unsqueeze(-1).unsqueeze(-1)

        x = self.liftnet(x)
        x = self.last_layer(x)
        return x


@MODELS.register_module()
class LiftClassifierHead(BaseModule):
    """liftHead for getting 3d rotation from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 keypoint_classifier: ConfigType,
                 channel_num: int = 55,
                 undistort: bool = False,
                 use_kp2d_gt=False,
                 kpt2d_with_depth: bool = False,
                 noRt=False,
                 classifier_num=2048,
                 lambda_t: int = -1,
                 use_svd: bool = True,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)

        # define the classifier
        self.keypoint_head = MODELS.build(keypoint_classifier)
        self.init_weights_self(keypoint_classifier['tokenizer']['ckpt'])

        # define the liftnet model
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_num = channel_num
        self.lambda_t = lambda_t

        feat_dim = 2 * self.channel_num

        self.liftnet = gMLP(d_model=feat_dim, d_ffn=feat_dim * 2, num_layers=3)
        self.liftnet_classifier = GMLPModel_Large(
            input_size=33, output_size=classifier_num).cuda()
        self.flatten_lay1 = nn.Flatten()
        input_size, output_size = 20 * 8, 60 * 8
        self.fc_lay1 = nn.Linear(input_size, output_size).cuda()

        self.rigid_samples = _gen_rigid_features()

        # define the full connection layer
        self.shape_ncomp = 1
        output_num = 60 + self.shape_ncomp  # 60 （deltaxyz）+ scale
        if use_svd:
            self.output_num = output_num + 21
        else:
            self.output_num = output_num
        self.last_layer = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.SyncBatchNorm(feat_dim), nn.ReLU(),
            nn.Conv2d(feat_dim, self.output_num, kernel_size=1))

        self.lift_loss = MODELS.build(lift_loss)

        # define the fllow parameters
        self.use_svd = use_svd
        self.scale_parameter = 1000
        self.undistort = undistort
        self.use_kp2d_gt = use_kp2d_gt
        self.noRt = noRt

        # define nimble info
        self.kp_index = [
            0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22,
            23, 24
        ]
        self.nimble_layer = sim_NIMBLELayer(
            device='cuda',
            shape_ncomp=20,
            pose_ncomp=60,
            use_pose_pca=False,
            reg_shape_type=0)

    def init_weights_self(self, tokenizer):
        """Weight initialization for model."""
        self.keypoint_head.init_weights()
        self.keypoint_head.tokenizer.init_weights_self(tokenizer)

    def forward(self, feats: Tuple[Tensor], feta_xy: Tuple[Tensor],
                feta_cam: Tuple[Tensor]) -> Tensor:
        output = self.liftnet(feats)
        output = self.last_layer(output).view((feats.shape[0], -1, 1, 1))

        fea_reshape_xy = self.flatten_lay1(feta_xy)
        fea_reshape_xy = self.fc_lay1(fea_reshape_xy)
        fea_reshape_xy = fea_reshape_xy.view(feta_xy.shape[0], 60, -1)
        fea_cam = feta_cam.unsqueeze(1).repeat(1, 60, 1)
        fea_total = torch.cat((fea_reshape_xy, fea_cam), dim=2)
        cls_logits = self.liftnet_classifier(fea_total)

        return cls_logits, output

    def preprocess(self, feats, batch_data_samples):
        xy_coord = feats[..., :2]
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
                if 'nimble_pose' in data_sample.meta.keys():
                    nimble_pose.append(data_sample.meta['nimble_pose'])
                    nimble_trans.append(data_sample.meta['nimble_translation'])
                    nimble_shape.append(data_sample.meta['nimble_shape'])
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
                left_to_right_rt = np.linalg.inv(right_cam_xf)
                lr_rot_matrix.append(
                    lr_t[:3, :3])  # lr_rot_matrix 代表的是 右相机向左相机的变换
                lr_p.append(lr_t[:3, 3])

            warp_mat = data_sample.metainfo[
                'warp_mat']  # warp mat 代表的是从原图到crop图像的映射
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
        left_to_right_rt = torch.tensor(
            np.array(left_to_right_rt)).cuda().float()
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
        nimble_pose = torch.tensor(np.array(nimble_pose)).cuda().float()
        nimble_trans = torch.tensor(np.array(nimble_trans)).cuda().float()
        nimble_shape = torch.tensor(np.array(nimble_shape)).cuda().float()

        def get_nimblehand_input(nimble_pose, zeros_nimble_shape):
            B = nimble_pose.shape[0]
            init_root_rot = torch.zeros((B, 1, 3),
                                        requires_grad=False).cuda().float()
            gt_rot_vector = torch.cat((init_root_rot, nimble_pose[:, 1:, :]),
                                      dim=1)

            _, bone_joints = self.nimble_layer.forward_simple(
                gt_rot_vector, zeros_nimble_shape)
            uesd_joints = bone_joints[:, self.kp_index, :]
            rebuild_joints = (uesd_joints -
                              uesd_joints[:, 0:1, :]) / self.scale_parameter
            return rebuild_joints[:, 1:, :]

        if nimble_pose.shape[0] > 0:
            zeros_nimble_shape = torch.zeros_like(nimble_shape).cuda().float()
            nimble_rebuild_joint = get_nimblehand_input(
                nimble_pose, zeros_nimble_shape)
            with torch.no_grad():
                p_logits, p_joints, g_logits, e_latent_loss = \
                    self.keypoint_head(None, None, nimble_rebuild_joint)
            error_with_classifier = nimble_rebuild_joint - p_joints

            nimble_info = {
                'nimble_pose': nimble_pose,
                'nimble_trans': nimble_trans,
                'nimble_shape': nimble_shape,
                'classifier_gt': g_logits,
                'nimble_rebuild_joint': nimble_rebuild_joint,
                'error_with_classifier': error_with_classifier
            }
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
        # 将crop（仿射变换）的图像的关键点恢复到在原图中位置的点，
        # 得到的uv_coord_im_pred（对应shape为（256, 21, 3））
        # 乘 all_inv_warp_mat（对应shape为（256, 3, 2））= shape为（256, 21, 2）
        uv_coord_im_pred_global_distort = torch.bmm(uv_coord_im_pred,
                                                    all_inv_warp_mat)
        uv_coord_im_pred_global_distort = \
            uv_coord_im_pred_global_distort.view(B, N, K, 2)

        # 这里将原来的点（不区分左右手的），现在经过了resize以及左手翻转回对应的坐标，
        # 这时候是真正的左右的的在原图的坐标位置了（这里指的是还没有去过畸变的点）
        frame_width = batch_data_samples[0].meta['frame_width']
        uv_coord_im_pred_global_distort_noflip = recover_hand(
            uv_coord_im_pred_global_distort, left_hand, frame_width)
        # 执行了相同的操作，只是这里是对真值来做的
        uv_coord_im_gt_global = recover_hand(uv_coord_im_gt_global, left_hand,
                                             frame_width)

        if self.use_kp2d_gt:
            uv_coord_im_pred_global = uv_coord_im_gt_global

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

            feats = torch.cat((feature1, feature2), dim=1).float()

            feta_xy = torch.cat(
                (leftcam_xy[:, 1:, :], rightcam_xy[:, 1:, :],
                 leftcam_xy[:, :1, :].repeat(
                     1, 20, 1), rightcam_xy[:, :1, :].repeat(1, 20, 1)),
                dim=2)
            feta_cam = torch.cat(
                (Tmatrix_leftcam.repeat(B, 1), lr_p.view(
                    (B, -1)), lr_rot_matrix.view(
                        (B, -1)), left_hand.view(B, -1)),
                dim=1)

        return (feats, feta_xy, feta_cam, leftcam_xy, rightcam_xy,
                left_to_right_rt, leftcam_cam_matrix, rightcam_cam_matrix,
                uv_coord_im_pred_global, uv_coord_im_pred_global_distort,
                hand3d_gt, left_hand, nimble_info)

    def postprocess(self,
                    cls_logits,
                    output,
                    left_hand,
                    nimble_info,
                    hand3d_gt,
                    only_pre=False):

        B = output.shape[0]
        cuda_device = output.device

        # decoder the output
        delta_len = 60
        delta_xyz = output[:, :delta_len, 0, 0].reshape(B, 20, 3)
        svd_begin = delta_len + self.shape_ncomp
        shape_v = output[:, delta_len:svd_begin, 0, 0]
        pre_pt_features = output[:, svd_begin:, 0, 0]
        matrix_svd = decode_svd(
            pre_pt_features,
            self.rigid_samples,
        )

        # decoder the classifier result
        cls_logits = cls_logits[:, :, 0, 0]
        cls_logits_softmax = cls_logits.softmax(1)

        # get predict result
        pre_root_xyz = matrix_svd[:, 0:3, 3]
        pre_root_matrix = matrix_svd[:, 0:3, 0:3]
        pre_shape_vector = shape_v
        _, pre_codebook_xyz_part, _, _ = self.keypoint_head(
            None, None, None, cls_logits_softmax=cls_logits_softmax)
        pre_codebook_xyz = torch.cat((torch.zeros_like(
            pre_codebook_xyz_part[:, :1, :]), pre_codebook_xyz_part),
                                     dim=1)
        pre_codebook_delta = torch.cat(
            (torch.zeros_like(delta_xyz[:, :1, :]), delta_xyz), dim=1)

        if not only_pre:
            with torch.no_grad():
                gt_root_xyz = nimble_info['nimble_trans']
                gt_root_matrix = batch_rodrigues(
                    nimble_info['nimble_pose'][:, 0, :]).reshape(-1, 3, 3)
                gt_codebook_xyz_part = nimble_info[
                    'nimble_rebuild_joint'] - nimble_info[
                        'error_with_classifier']
                gt_codebook_xyz = torch.cat((torch.zeros_like(
                    gt_codebook_xyz_part[:, :1, :]), gt_codebook_xyz_part),
                                            dim=1)
                gt_codebook_delta = torch.cat(
                    (torch.zeros_like(
                        nimble_info['error_with_classifier'][:, :1, :]),
                     nimble_info['error_with_classifier']),
                    dim=1)

        def get_nimble_3d(root_xyz, root_matrix, codeboox_xyz, codebook_delta,
                          shape_vector):

            def get_scale_3d(jreg_joints, shape_param):
                scale_factor = 1 + shape_param[:, 0]
                jreg_joints = scale_factor.view(scale_factor.shape[0], 1,
                                                1) * jreg_joints
                return jreg_joints

            local_xyz = codeboox_xyz + codebook_delta
            bone_joints = get_scale_3d(local_xyz, shape_vector)
            rebuild_joints_temp = bone_joints

            mask = left_hand == 1
            add_matrix = torch.eye(3).unsqueeze(0).expand(B, -1,
                                                          -1).to(cuda_device)
            add_matrix[mask, 0, 0] = -add_matrix[mask, 0, 0]
            root_matrix = torch.matmul(add_matrix, root_matrix)
            rebuild_joints_temp = torch.matmul(rebuild_joints_temp,
                                               root_matrix.transpose(1, 2))

            xyz_point = rebuild_joints_temp + root_xyz.unsqueeze(1)
            return xyz_point

        if only_pre:
            pre_root_pre_local_xyz = get_nimble_3d(pre_root_xyz,
                                                   pre_root_matrix,
                                                   pre_codebook_xyz,
                                                   pre_codebook_delta,
                                                   pre_shape_vector)
            return pre_root_pre_local_xyz
        else:
            predict_root = get_nimble_3d(pre_root_xyz, pre_root_matrix,
                                         gt_codebook_xyz, gt_codebook_delta,
                                         pre_shape_vector)
            predict_codebook = get_nimble_3d(gt_root_xyz, gt_root_matrix,
                                             pre_codebook_xyz,
                                             gt_codebook_delta,
                                             pre_shape_vector)
            predict_delta = get_nimble_3d(gt_root_xyz, gt_root_matrix,
                                          gt_codebook_xyz, pre_codebook_delta,
                                          pre_shape_vector)
            predict_all = get_nimble_3d(pre_root_xyz, pre_root_matrix,
                                        pre_codebook_xyz, pre_codebook_delta,
                                        pre_shape_vector)
            return (predict_root, predict_codebook, predict_delta, predict_all,
                    pre_root_xyz, pre_root_matrix, delta_xyz, cls_logits)

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:

        with torch.no_grad():
            (feats, feta_xy, feta_cam, leftcam_xy, rightcam_xy,
             left_to_right_rt, leftcam_cam_matrix, rightcam_cam_matrix,
             uv_coord_im_pred_global, uv_coord_im_pred_global_distort,
             hand3d_gt, left_hand,
             nimble_info) = self.preprocess(feats, batch_data_samples)

            cls_logits, output = self.forward(feats, feta_xy, feta_cam)

        hand3d_pred = self.postprocess(
            cls_logits,
            output,
            left_hand,
            nimble_info,
            hand3d_gt,
            only_pre=True)

        return hand3d_pred, uv_coord_im_pred_global_distort

    def get_class_accuracy(self, output, target, topk):

        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [
            correct[:k].reshape(-1).float().sum(0) * 100. / batch_size
            for k in topk
        ]

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        with torch.no_grad():
            (feats, feta_xy, feta_cam, leftcam_xy, rightcam_xy,
             left_to_right_rt, leftcam_cam_matrix, rightcam_cam_matrix,
             uv_coord_im_pred_global, uv_coord_im_pred_global_distort,
             hand3d_gt, left_hand,
             nimble_info) = self.preprocess(feats, batch_data_samples)

        cls_logits, output = self.forward(feats, feta_xy, feta_cam)

        # 3d 损失
        (pred_3d_way1, pred_3d_way2, pred_3d_way3, hand3d_pred, pre_trans_xyz,
         pre_root_matrix, delta_xyz, pre_cls_logits) = \
            self.postprocess(cls_logits, output, left_hand,
                             nimble_info, hand3d_gt, only_pre=False)

        # 直接监督trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz
        if 'nimble_pose' in nimble_info.keys(
        ) and 'nimble_trans' in nimble_info.keys():
            gt_nimble_trans = nimble_info['nimble_trans']
            # gt_delta_xyz = nimble_info['error_with_classifier']
            gt_classifier = nimble_info['classifier_gt']

        # pinch 损失
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(hand3d_gt[:, 4, :] - hand3d_gt[:, 8, :], dim=-1)

        pred_for_loss = [
            pred_3d_way1, pred_3d_way2, pred_3d_way3, hand3d_pred, dist_pred,
            pre_nimble_trans, pre_cls_logits
        ]

        targ_for_loss = [
            hand3d_gt, hand3d_gt, hand3d_gt, hand3d_gt, dist_gt,
            gt_nimble_trans, gt_classifier
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss)
        (loss_pre_root, loss_pre_codebook, loss_pre_delta, loss_pre_all,
         loss_pinch, loss_nimble_trans, loss_classifier) = losses

        losses_dict = dict(
            loss_pre_root=loss_pre_root,
            loss_pre_codebook=loss_pre_codebook,
            loss_pre_delta=loss_pre_delta,
            loss_pre_all=loss_pre_all,
            loss_pinch=loss_pinch,
            loss_nimble_trans=loss_nimble_trans,
            loss_classifier=loss_classifier)

        topk = (1, 2, 5)
        keypoint_accuracy = \
            self.get_class_accuracy(pre_cls_logits, gt_classifier, topk)
        kpt_accs = {}
        for i in range(len(topk)):
            kpt_accs['top%s-acc' % str(topk[i])] \
                = keypoint_accuracy[i]
        losses_dict.update(kpt_accs)

        return losses_dict
