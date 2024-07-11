# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine.structures import PixelData
from torch import Tensor

from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_coordinates, flip_heatmaps
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import ConfigType, OptConfigType, OptSampleList
from ...utils.siamcc_to_kpt import SimCCToKeypoint3D, SimCCToKeypoint
from ..coord_cls_heads import RTMCCHead
from mmpose.models.utils.gmlp import gMLP

from mmpose.models.heads.nimble.nimble_utils import (
    SkeletonEncoder, _gen_rigid_features, adjust_predicted_angles,
    batch_rodrigues, cal_proportion, convert_vector2matrix, decode_svd,
    euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion,
    rot6D_to_matirx, rot9D_to_matirx, trans_3d_2_2d)
from mmpose.models.heads.nimble.simple_NIMBLELayer import sim_NIMBLELayer


OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RTMCCIPRHeadNimble(RTMCCHead):
    """Top-down head introduced in RTMPose (2023). The head is composed of a
    large-kernel convolutional layer, a fully-connected layer and a Gated
    Attention Unit to generate 1d representation from low-resolution feature
    maps.

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map.
        out_channels (int): Number of channels in the output heatmap.
        input_size (tuple): Size of input image in shape [w, h].
        in_featuremap_size (int | sequence[int]): Size of input feature map.
        simcc_split_ratio (float): Split ratio of pixels.
            Default: 2.0.
        final_layer_kernel_size (int): Kernel size of the convolutional layer.
            Default: 1.
        gau_cfg (Config): Config dict for the Gated Attention Unit.
            Default: dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False).
        input_transform (str): Transformation of input features which should
            be one of the following options:

                - ``'resize_concat'``: Resize multiple feature maps specified
                    by ``input_index`` to the same size as the first one and
                    concat these feature maps
                - ``'select'``: Select feature map(s) specified by
                    ``input_index``. Multiple selected features will be
                    bundled into a tuple

            Defaults to ``'select'``
        input_index (int | sequence[int]): The feature map index used in the
            input transformation. See also ``input_transform``. Defaults to -1
        align_corners (bool): `align_corners` argument of
            :func:`torch.nn.functional.interpolate` used in the input
            transformation. Defaults to ``False``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 input_size: Tuple[int, int],
                 in_featuremap_size: Tuple[int, int],
                 simcc_split_ratio: float = 2.0,
                 final_layer_kernel_size: int = 1,
                 gau_cfg: ConfigType = dict(
                     hidden_dims=256,
                     s=128,
                     expansion_factor=2,
                     dropout_rate=0.,
                     drop_path=0.,
                     act_fn='ReLU',
                     use_rel_bias=False,
                     pos_enc=False),
                 loss: ConfigType = dict(
                     type='KLDiscretLoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 output_sigma: bool = False,
                 deploy: bool = False,
                 with_gau: bool = False,
                 deploy_output='kpt',
                 feat_channel=6,
                 map_type='softmax'):
        super().__init__(
            in_channels,
            out_channels,
            input_size,
            in_featuremap_size,
            simcc_split_ratio,
            final_layer_kernel_size,
            gau_cfg,
            loss,
            decoder,
            init_cfg,
            with_gau=with_gau)
        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)
        # D = int(self.input_size[2] * self.simcc_split_ratio)
        self.ipr_module = SimCCToKeypoint(
            feat_w=W, feat_h=H)
        self.with_gau = with_gau
        self.deploy_output = deploy_output
        self.output_sigma = output_sigma
        self.deploy = deploy
            
        # nimble相关
        self.pose_num = 19*9
        self.scale_parameter=1000
        self.nimble_output_num = self.pose_num + 21
        self.rigid_samples = _gen_rigid_features()
        self.proj_layer = nn.Conv2d(
            256, feat_channel, kernel_size=1, padding=0).to("cuda")
        self.nimble_hidden_num = 512
        self.input_dim = feat_channel * 64
        self.f_standard = 200
        # self.liftnet = gMLP(d_model=self.input_dim, d_ffn=220, num_layers=3).to("cuda")
        self.feature_fuszion_layer = nn.Sequential(
            nn.Conv2d(feat_channel, feat_channel*4, kernel_size=1),
            nn.BatchNorm2d(feat_channel*4, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(feat_channel*4, feat_channel, kernel_size=1))
        self.nimble_last_layer = nn.Sequential(
            nn.Conv2d(self.input_dim, self.nimble_hidden_num, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.nimble_hidden_num, self.nimble_output_num, kernel_size=1))
        self.nimble_layer = sim_NIMBLELayer(
            device='cuda',
            shape_ncomp=1,
            pose_ncomp=30,
            use_pose_pca=False,
            reg_shape_type=1)
        self.kp_index = [
            0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22,
            23, 24
        ]
        self.joint_parents = [
            0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18,
            19
        ]
        self.non_root_indices = []
        for i in range(len(self.joint_parents)):
            if i != self.joint_parents[i]:
                self.non_root_indices.append(i)
        if self.output_sigma:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.sigma_conv = nn.Conv2d(
                self.input_dim, self.out_channels * 3, kernel_size=1)

    def trans_feat(self, image_fea, f_scale):
        B = image_fea.shape[0]
        singlev_scaled_to_orig_xf = torch.eye(4).unsqueeze(0).repeat(B,1,1).to(image_fea.device)
        singlev_scaled_to_orig_xf[..., 2, 2] = f_scale
        scaled_feature_matrix = torch.inverse(singlev_scaled_to_orig_xf)
        
        ftl_image_fea = image_fea.clone()
        point_features_xfed = ftl_image_fea.reshape(B, 3, -1)
        r_in = scaled_feature_matrix[:, 0:3, 0:3].clone()
        t_in = scaled_feature_matrix[:, 0:3, 3].clone()
        point_features_xfed = torch.matmul(
            r_in, point_features_xfed) + t_in.unsqueeze(-1)
        ftl_image_fea = point_features_xfed.reshape(image_fea.shape)
        return ftl_image_fea
        

    def forward(self, feats: Tuple[Tensor], f_scale: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward the network.

        The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        
        # feat_x, feat_y = super().forward(feats)
        # heatmaps = torch.cat([feat_x, feat_y], dim=1)
        raw_feats = feats[-1]
        # pred_x, pred_y = self.ipr_module(feat_x, feat_y)
        # output = torch.cat([pred_x, pred_y], dim=-1)
        
        B = raw_feats.shape[0]
        image_fea = self.proj_layer(raw_feats)
        ftl_image_fea = self.trans_feat(image_fea, f_scale)
        feature_fuszion = self.feature_fuszion_layer(ftl_image_fea)
        # coor_fea = output.clone().reshape(B, -1)
        # if intrix_feats.shape[0] != B:
        #     intrix_fea = intrix_feats.repeat(2, 1)
        # else:
        #     intrix_fea = intrix_feats
            
        # nimble_fea = torch.concat((image_fea, coor_fea),dim=-1)[:,:,None,None]
        # nimble_fea = torch.concat((nimble_fea, intrix_fea),dim=-1)[:,:,None,None]
        # nimble_fea = self.liftnet(nimble_fea)
        nimble_output = self.nimble_last_layer(feature_fuszion.reshape(B,-1,1,1))
        
        
        if self.output_sigma:
            # x = self.gap(raw_feats)
            pred_sigma = self.sigma_conv(image_fea.reshape(B,-1,1,1))
            pred_sigma_reshape = pred_sigma.reshape(
                pred_sigma.size(0), self.out_channels, 3)

            return nimble_output, pred_sigma_reshape

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}):
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """
        left_R = []
        is_left_hands = []
        hand3d_gt = []
        hand2d_gt = []
        intrix_matrix = []
        f_scale = []

        for i, data in enumerate(batch_data_samples):
            keypoint_label = data.gt_instance_labels.keypoint_labels
            camera_model = data.meta['virtual_camera']
            keypoint_2d_lable = data.gt_instances.keypoints[:,:,:2]
            # keypoint_2d_lable = camera_model.undistort(keypoint_2d_lable)
            f_scale.append(camera_model.f[0] / self.f_standard)
            intrix_m = np.array([[camera_model.f[0], 0, camera_model.c[0]],
                                      [0, camera_model.f[1], camera_model.c[1]],
                                      [0,0,1]])
            
            if keypoint_label.shape[-1] == 3:
                if data.meta['category_id'] == 1:
                    is_left_hands.append(1)
                else:
                    is_left_hands.append(0)
            if 'virtual_camera' in data.meta:
                virtual_cam = data.meta['virtual_camera']
                left_R.append(np.linalg.inv(virtual_cam.camera_to_world_xf[:3,:3]))
            hand3d_gt.append(data.gt_instances.keypoints3d[0])
            hand2d_gt.append(keypoint_2d_lable)
            intrix_matrix.append(intrix_m)

        left_R = torch.tensor(np.array(left_R)).cuda().float()
        left_hand = torch.tensor(np.array(is_left_hands)).cuda().float()
        intrix_matrix = torch.tensor(np.array(intrix_matrix)).cuda().float()
        intrix_fea = intrix_matrix.reshape(left_R.shape[0], -1)
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
        hand2d_gt = torch.tensor(np.array(hand2d_gt)).cuda().float()
        f_scale = torch.tensor(np.array(f_scale)).cuda().float()
        
        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']
            _feats, _feats_flip = feats

            _batch_coords, _batch_heatmaps = self.forward(_feats, intrix_fea)

            _batch_coords_flip, _batch_heatmaps_flip = self.forward(
                _feats_flip, intrix_fea)
            _batch_coords_flip = flip_coordinates(
                _batch_coords_flip,
                flip_indices=flip_indices,
                shift_coords=test_cfg.get('shift_coords', True),
                input_size=input_size)
            _batch_heatmaps_flip = flip_heatmaps(
                _batch_heatmaps_flip,
                flip_mode='heatmap',
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))

            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            nimble_output, sigma = self.forward(feats, f_scale)  # (B, K, D)

        
        hand3d_pred = self.decode_nimble_fun(nimble_output, left_R, None, left_hand, None, f_scale, True)
        
        result = []
        for hand3d_sin, batch_data_sample in zip(hand3d_pred, batch_data_samples):
            ori_cam = batch_data_sample.meta['ori_camera']
            result.append(ori_cam.eye_to_window(hand3d_sin.cpu().numpy()))
        uv_reproj = torch.tensor(
            result).cuda()
            
        return hand3d_pred, uv_reproj, sigma
        


    def decode_nimble_fun(self, nimble_output, left_R, nimble_info, left_hand, hand3d_gt, f_scale, only_pre):
        
        B = nimble_output.shape[0]
        cuda_device = nimble_output.device

        pose_len = self.pose_num
        rot_vector_t = nimble_output[:, :pose_len, 0, 0].float()
        pre_pt_features = nimble_output[:, pose_len:, 0, 0]

        matrix_svd = decode_svd(
            pre_pt_features,
            self.rigid_samples,
        )

        pre_root_xyz = matrix_svd[:, 0:3, 3]
        pre_root_matrix = matrix_svd[:, 0:3, 0:3]
        pre_local_matrix = rot9D_to_matirx(rot_vector_t.reshape(
            -1, 9)).reshape(B, 19, -1)
        with torch.no_grad():
            shape_vector = torch.zeros((B, 1)).to(cuda_device)

        mask = left_hand == 1
        add_matrix = torch.eye(3).unsqueeze(0).expand(B, -1,
                                                        -1).to(cuda_device)
        add_matrix[mask, 0, 0] = -add_matrix[mask, 0, 0]

        if not only_pre:
            with torch.no_grad():
                gt_root_xyz = torch.bmm(
                    torch.matmul(left_R, add_matrix), nimble_info['nimble_trans'].unsqueeze(
                        -1))[:, :, 0] / f_scale.unsqueeze(-1)
                # gt_root_matrix = batch_rodrigues(
                #     nimble_info['nimble_pose'][:, 0, :]).reshape(-1, 3, 3)
                
                gt_root_matrix = nimble_info['nibmle_root_matrix']
                gt_root_matrix = torch.matmul(torch.matmul(left_R, add_matrix), torch.matmul(gt_root_matrix, add_matrix))

                init_root_rot = torch.zeros((B, 1, 3),
                                            requires_grad=True,
                                            device=cuda_device)
                gt_rot_vector = torch.cat(
                    (init_root_rot, nimble_info['nimble_pose'][:, 1:, :]),
                    dim=1)
                gt_local_matrix = convert_vector2matrix(
                    gt_rot_vector.view(B, -1)).reshape(B, -1, 9)


        def get_nimble_3d(root_xyz, root_matrix, local_matrix, shape_vector, left_R, f_scale):

            _, bone_joints = self.nimble_layer.forward_simple(
                local_matrix, shape_vector)
            rebuild_joints = bone_joints[:, self.kp_index, :]
            root_rebuild_joints = rebuild_joints[:, 0:1, :]
            rebuild_joints_temp = rebuild_joints - root_rebuild_joints
            
            root_matrix = torch.matmul(torch.inverse(left_R), root_matrix)
            rebuild_joints_temp = torch.matmul(rebuild_joints_temp,
                                               root_matrix.transpose(1, 2))
            rebuild_joints_with_scale = \
                rebuild_joints_temp / self.scale_parameter

            new_root_xyz = torch.bmm(
                root_xyz.unsqueeze(1),
                torch.inverse(left_R).permute(0, 2, 1))
            new_root_xyz = new_root_xyz.mul(f_scale[:,None,None].repeat(1,1,3))
            xyz_point = rebuild_joints_with_scale + new_root_xyz
            return xyz_point

        if only_pre:
            pre_nimble_pre_root_pre_shape__xyz = get_nimble_3d(
                pre_root_xyz, pre_root_matrix, pre_local_matrix,
                shape_vector, left_R, f_scale)

            return pre_nimble_pre_root_pre_shape__xyz
        else:
            pre_root__xyz = get_nimble_3d(pre_root_xyz, pre_root_matrix,
                                          gt_local_matrix, shape_vector, left_R, f_scale)
            pre_nimble__xyz = get_nimble_3d(gt_root_xyz, gt_root_matrix,
                                            pre_local_matrix, shape_vector, left_R, f_scale)
            pre_all__xyz = get_nimble_3d(pre_root_xyz, pre_root_matrix,
                                         pre_local_matrix, shape_vector, left_R, f_scale)
            gt_all__xyz = get_nimble_3d(gt_root_xyz, gt_root_matrix,
                                        gt_local_matrix, shape_vector, left_R, f_scale)
            return (pre_root__xyz, pre_nimble__xyz, pre_all__xyz, gt_all__xyz,
                    pre_root__xyz[:, 0, :], gt_all__xyz[:, 0, :])

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        label_2d_list = []
        label_depth_list = []
        label_depth_id_list = []
        nimble_pose = []
        nimble_trans = []
        nimble_shape = []
        hand3d_gt = []
        hand2d_gt = []
        intrix_matrix = []
        left_R = []
        is_left_hands = []
        f_scale = []
        external_matrix = []
        nimble_info = dict()
        
        for i, data in enumerate(batch_data_samples):
            keypoint_label = data.gt_instance_labels.keypoint_labels
            label_2d_list.append(keypoint_label[..., :2])

            label_depth_list.append(keypoint_label[..., 2:3])
            label_depth_id_list.append(i)
            hand3d_gt.append(data.gt_instances.keypoints3d[0])
            if data.meta['category_id'] == 1:
                is_left_hands.append(1)
            else:
                is_left_hands.append(0)
            if data.meta['camera_name'] == 'right':
                external_matrix.append(data.meta['external'])
            else:
                external_matrix.append(np.eye(4))
                
            keypoint_2d_lable = data.gt_instances.keypoints[:,:,:2]
            camera_model = data.meta['ori_camera']
            vritual_camera = data.meta['virtual_camera']
            keypoint_2d_lable = camera_model.undistort(keypoint_2d_lable)
            intrix_m = np.array([[vritual_camera.f[0], 0, vritual_camera.c[0]],
                                    [0, vritual_camera.f[1], vritual_camera.c[1]],
                                    [0,0,1]])
            f_scale.append(vritual_camera.f[0] / self.f_standard)
            hand2d_gt.append(keypoint_2d_lable)
            intrix_matrix.append(intrix_m)
            
            nimble_pose.append(data.meta['nimble_pose'])
            nimble_trans.append(data.meta['nimble_translation'])
            nimble_shape.append(data.meta['nimble_shape'])
            
            virtual_cam = data.meta['virtual_camera']
            left_R.append(np.linalg.inv(virtual_cam.camera_to_world_xf[:3,:3]))

                
        label_2d = torch.cat(label_2d_list)
        left_R = torch.tensor(np.array(left_R)).cuda().float()
        left_hand = torch.tensor(np.array(is_left_hands)).cuda().float()
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
        hand2d_gt = torch.tensor(np.array(hand2d_gt)).cuda().float()
        f_scale = torch.tensor(np.array(f_scale)).cuda().float()
        intrix_matrix = torch.tensor(np.array(intrix_matrix)).cuda().float()
        external_matrix = torch.tensor(np.array(external_matrix)).cuda().float()
        nimble_pose = torch.tensor(np.array(nimble_pose)).cuda().float()
        nibmle_root_matrix = batch_rodrigues(nimble_pose[:,0,:]).reshape(-1, 3, 3)
        nibmle_root_matrix = torch.matmul(external_matrix[:, :3,:3], nibmle_root_matrix)
        
        nimble_trans = torch.tensor(np.array(nimble_trans)).cuda().float()
        nimble_trans = torch.matmul(external_matrix, torch.concat((nimble_trans, torch.ones(nimble_trans.shape[0], 1).to(nimble_trans.device)),dim=1).unsqueeze(-1))[:,:3,0]
        
        nimble_info = {
            'nibmle_root_matrix': nibmle_root_matrix,
            'nimble_pose': nimble_pose,
            'nimble_trans': nimble_trans,
            'nimble_shape':
            torch.tensor(np.array(nimble_shape)).cuda().float()
        }
        label_depth_id = torch.tensor(
            label_depth_id_list, dtype=torch.int32).cuda()
        
        nimble_output, sigma = self.forward(inputs, f_scale)
        
        
        (pred_3d_way1, pred_3d_way2, hand3d_pred, hand3d_part_gt,
         pre_trans_xyz, gt_trans_xyz) = self.decode_nimble_fun(nimble_output, left_R, nimble_info, left_hand, hand3d_gt, f_scale, False)
        
        # 重投影损失
        hand2d_pred = torch.matmul(intrix_matrix, pred_3d_way1.permute(0,2,1)).permute(0,2,1)
        hand2d_pred = hand2d_pred[...,:2] / (hand2d_pred[...,2:]+1e-8)
        hand2d_gt = torch.matmul(intrix_matrix, hand3d_part_gt.permute(0,2,1)).permute(0,2,1)
        hand2d_gt = hand2d_gt[...,:2] / (hand2d_gt[...,2:]+1e-8)
        

        # 直接监督rot和trans, 只考虑根节点的处理方式
        pre_nimble_trans = pre_trans_xyz
        gt_nimble_trans = gt_trans_xyz


        # pinch 损失
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(
           hand3d_gt[:, 4, :] - hand3d_gt[:, 8, :], dim=-1)
        

        re_all_sigmas = torch.cat((hand3d_pred, sigma), dim=-1)

        pred_for_loss = [
            pred_3d_way1, pred_3d_way2, hand3d_pred, dist_pred, 
            pre_nimble_trans, re_all_sigmas, hand2d_pred
        ]
        targ_for_loss = [
            hand3d_gt, hand3d_gt, hand3d_gt,
            dist_gt, gt_nimble_trans, hand3d_gt, hand2d_gt
        ]

        weight_ini = torch.ones((1, 21, 3))
        weight_ini[0, :9, :] = 2
        weight_ini[0, 4, :], weight_ini[0, 8, :] = 4, 4
        weight_ini = weight_ini.repeat(hand3d_gt.shape[0], 1,
                                       1).to(hand3d_gt.device)
        weight_for_loss = [
            weight_ini,
            weight_ini,
            weight_ini,
            None,
            None,
            None,
            None
        ]

        losses = self.loss_module(pred_for_loss, targ_for_loss, weight_for_loss)
        (loss_pre_root, loss_pre_nimble, loss_pre_all, loss_pinch, 
         loss_nimble_trans, loss_rle_all, loss_reproject) = losses

        # # 子骨骼向量监督
        bone_loss_weight = 0.1
        bone_3d_pre = (hand3d_pred - hand3d_pred[:, self.joint_parents, :]
                        )[:, self.non_root_indices].reshape(-1, 3)
        bone_3d_gt = (hand3d_gt -
                        hand3d_gt[:, self.joint_parents, :]
                        )[:, self.non_root_indices].reshape(-1, 3)

        bone_3d_pre_vector = self.cal_normalize_vector(bone_3d_pre)
        bone_3d_gt_vector = self.cal_normalize_vector(bone_3d_gt)

        squared_diff = (bone_3d_pre_vector - bone_3d_gt_vector)**2
        bone_loss = torch.mean(torch.sum(squared_diff,
                                            dim=1)) * bone_loss_weight

        # 局部子骨骼监督
        major_bone_loss_weight = 0.3
        local_bone_3d_pre = (
            pred_3d_way2 -
            pred_3d_way2[:, self.joint_parents, :])[:,
                                                    self.non_root_indices]
        local_bone_3d_pre = local_bone_3d_pre[:, :8, :].reshape(-1, 3)
        local_bone_3d_gt = (hand3d_part_gt -
                            hand3d_part_gt[:, self.joint_parents, :]
                            )[:, self.non_root_indices]
        local_bone_3d_gt = local_bone_3d_gt[:, :8, :].reshape(-1, 3)

        local_bone_3d_pre_vector = self.cal_normalize_vector(
            local_bone_3d_pre)
        local_bone_3d_gt_vector = self.cal_normalize_vector(
            local_bone_3d_gt)

        local_squared_diff = (local_bone_3d_pre_vector -
                                local_bone_3d_gt_vector)**2
        major_bone_loss = torch.mean(torch.sum(
            local_squared_diff, dim=1)) * major_bone_loss_weight

        losses = dict(
            # loss_kpt2d=loss_kpt2d,
            # label_depth=loss_depth,
            loss_pre_root=loss_pre_root,
            loss_pre_nimble=loss_pre_nimble,
            loss_pre_all=loss_pre_all,
            bone_loss=bone_loss,
            major_bone_loss=major_bone_loss,
            loss_pinch=loss_pinch,
            loss_nimble_trans=loss_nimble_trans,
            loss_rle_all=loss_rle_all,
            loss_reproject=loss_reproject)

        
        # claculate 3d metric
        def cal_mpjpe(pre_kpt, gt_kpt):
            error = np.linalg.norm(pre_kpt - gt_kpt, ord=2, axis=-1).mean() * 1000
            return error
        
        mpjpe_value = cal_mpjpe(hand3d_gt.cpu().numpy(), hand3d_pred.detach().cpu().numpy())
        mpjpe_value = torch.tensor(mpjpe_value).cuda()
        losses.update(mpjpe_value=mpjpe_value)

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to load weights of deconv layers from
        :class:`HeatmapHead` into `simplebaseline_head`.

        The hook will be automatically registered during initialization.
        """

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            # convert fc to conv
            if _k == 'head.sigma_fc.weight':
                state_dict['head.sigma_conv.weight'] = torch.unsqueeze(
                    torch.unsqueeze(v, -1), -1)
            if _k == 'head.sigma_fc.bias':
                state_dict['head.sigma_conv.bias'] = v

    def cal_normalize_vector(self, vector):
        vector_norms = torch.sqrt(
            torch.sum(vector**2, dim=1, keepdim=True) + 1e-8)
        normalized_vector = vector / vector_norms
        return normalized_vector