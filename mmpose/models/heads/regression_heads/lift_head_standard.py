# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.logging import MessageHub
from mmengine.model import BaseModule
from nreal_data_tool.utils import kpt_to_bbox
from nreal_data_tool.utils.affine import from_two_vectors
from torch import Tensor, nn

from mmpose.models.utils.gmlp import gMLP
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions


@MODELS.register_module()
class LiftHeadStandard(BaseModule):
    """liftHead for getting 3d keypoints from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 num_layers: int = 3,
                 d_ffn: int = 220,
                 output_num: int = 42,
                 reproj: bool = False,
                 use_plane_coord=True,
                 baseline=0.13,
                 disparity_input=False,
                 rightcam_3d_disable=False,
                 kpt3d_output=False,
                 kpt3d_output_delta=False,
                 plane_arctan=False,
                 d_model=512,
                 reproj_thre=0,
                 iou_thre=0,
                 pad_2d=False,
                 edge_to_center=False,
                 lambda_t: int = -1,
                 corruption_cam: float = 0.5,
                 all_use_kp2d_gt: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
        self.all_use_kp2d_gt = all_use_kp2d_gt
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_num = 43 if use_plane_coord else 64
        self.disparity_input = disparity_input
        self.lambda_t = lambda_t
        self.rightcam_3d_disable = rightcam_3d_disable
        self.kpt3d_output = kpt3d_output
        self.kpt3d_output_delta = kpt3d_output_delta
        feat_dim = 2 * self.channel_num
        if self.disparity_input:
            feat_dim += 21
        self.liftnet = gMLP(
            d_model=feat_dim, d_ffn=d_ffn, num_layers=num_layers)
        self.corruption_cam = corruption_cam
        if self.rightcam_3d_disable:
            output_num = 21
        if self.kpt3d_output:
            output_num = 63
        self.last_layer = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1), nn.ReLU(),
            nn.Conv2d(feat_dim, output_num, kernel_size=1))
        if self.kpt3d_output_delta:
            self.delta_last_layer = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, kernel_size=1), nn.ReLU(),
                nn.Conv2d(feat_dim, 21, kernel_size=1))
        self.feat_dim = feat_dim
        self.lift_loss = MODELS.build(lift_loss)
        self.reproj = reproj
        self.use_plane_coord = use_plane_coord
        self.baseline = baseline
        self.plane_arctan = plane_arctan
        self.d_model = d_model
        self.reproj_thre = reproj_thre
        self.iou_thre = iou_thre
        self.pad_2d = pad_2d
        self.edge_to_center = edge_to_center
        self.center_rot = None

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        liftnet_output = self.liftnet(feats)
        output = self.last_layer(liftnet_output).view(feats.shape[0], -1, 1, 1)
        if self.kpt3d_output_delta:
            delta_output = self.delta_last_layer(liftnet_output).view(
                feats.shape[0], -1, 1, 1)
            output = torch.cat((output, delta_output), dim=1)
        return output

    @staticmethod
    def recover_hand(uv_coord_im_pred, left_hand, w):
        recover_uv_coord_im_pred = (1 - left_hand.view(
            size=(-1, 1, 1, 1))) * uv_coord_im_pred + left_hand.view(
                size=(-1, 1, 1, 1)) * (
                    torch.tensor([w - 1, 0]).view(size=(1, 1, 1, 2)).cuda() +
                    torch.tensor([-1, 1]).view(size=(1, 1, 1, 2)).cuda() *
                    uv_coord_im_pred)
        return recover_uv_coord_im_pred

    def preprocess(self, feats, batch_data_samples, mode):
        xy_coord = feats[..., :2]
        B = int(len(batch_data_samples) / 2)
        N = 2
        H, W = batch_data_samples[0].input_size
        K = xy_coord.shape[1]
        # kpt2d output to crop wh
        uv_coord_im_pred_crop_right = xy_coord * torch.tensor([W, H]).cuda()
        uv_coord_im_pred_crop_right = uv_coord_im_pred_crop_right.view(
            B, N, K, 2)

        leftcam_cam_matrix = []
        rightcam_cam_matrix = []
        left_R = []
        right_R = []
        baseline_scale = []
        lr_p = []
        lr_rot_matrix = []
        hand3d_gt = []
        is_left_hands = []

        uv_coord_im_gt_global = []

        all_inv_warp_mat = torch.zeros(B * 2, 3, 2).cuda()
        all_inv_warp_mat.requires_grad = False
        for i, data_sample in enumerate(batch_data_samples):
            if i % 2 == 0:
                left_camera = data_sample.meta['ori_camera']
                left_cam_matrix = left_camera.uv_to_window_matrix()
                leftcam_cam_matrix.append(left_cam_matrix)
                left_R.append(data_sample.meta['cam_to_virtual_R'])
                hand3d_gt.append(data_sample.gt_instances.keypoints3d[0])
                if data_sample.meta['category_id'] == 1:
                    is_left_hands.append(1)
                else:
                    is_left_hands.append(0)
            else:
                right_camera = data_sample.meta['ori_camera']
                right_cam_matrix = right_camera.uv_to_window_matrix()
                rightcam_cam_matrix.append(right_cam_matrix)
                right_R.append(data_sample.meta['cam_to_virtual_R'])
                left_cam_xf = left_camera.camera_to_world_xf
                right_cam_xf = right_camera.camera_to_world_xf
                lr_t = np.dot(np.linalg.inv(left_cam_xf),
                              right_cam_xf).astype(np.float32)
                left_to_right_rt = np.linalg.inv(right_cam_xf)
                lr_rot_matrix.append(lr_t[:3, :3])
                lr_p.append(lr_t[:3, 3])
                baseline_scale.append(data_sample.meta['virtual_baseline'] /
                                      self.baseline)
            warp_mat = data_sample.metainfo['warp_mat']
            inv_warp_mat = cv2.invertAffineTransform(warp_mat).astype(
                np.float32)
            inv_warp_mat = torch.from_numpy(inv_warp_mat).cuda()  # (2,3)
            all_inv_warp_mat[i] = inv_warp_mat.transpose(0, 1)  # (3,2)

            uv_coord_im_gt_global.append(data_sample.gt_instances.keypoints)
        leftcam_cam_matrix = torch.tensor(
            np.array(leftcam_cam_matrix)).cuda().float()
        rightcam_cam_matrix = torch.tensor(
            np.array(rightcam_cam_matrix)).cuda().float()

        left_R = torch.tensor(np.array(left_R)).cuda().float()
        right_R = torch.tensor(np.array(right_R)).cuda().float()
        baseline_scale = torch.tensor(np.array(baseline_scale)).cuda().float()
        lr_p = torch.tensor(np.array(lr_p)).cuda().float()
        lr_rot_matrix = torch.tensor(np.array(lr_rot_matrix)).cuda().float()
        left_to_right_rt = torch.tensor(
            np.array(left_to_right_rt)).cuda().float()
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
        left_hand = torch.tensor(np.array(is_left_hands)).cuda().float()
        uv_coord_im_gt_global = torch.tensor(
            np.array(uv_coord_im_gt_global)).cuda().float()
        uv_coord_im_gt_global = uv_coord_im_gt_global[..., :2]
        uv_coord_im_gt_global = uv_coord_im_gt_global.view(B, N, K, 2)

        uv_coord_im_pred_crop_leftright = uv_coord_im_pred_crop_right
        uv_coord_im_pred_crop_leftright = uv_coord_im_pred_crop_leftright.view(
            B * N, K, 2)

        # from crop uv to global uv
        uv_coord_im_pred = torch.cat(
            [uv_coord_im_pred_crop_leftright,
             torch.ones(B * 2, K, 1).cuda()],
            dim=-1)
        uv_coord_im_pred_global_distort = torch.bmm(uv_coord_im_pred,
                                                    all_inv_warp_mat)
        uv_coord_im_pred_global_distort = uv_coord_im_pred_global_distort.view(
            B, N, K, 2)

        frame_width = batch_data_samples[0].meta['frame_width']
        uv_coord_im_pred_global_distort_noflip = self.recover_hand(
            uv_coord_im_pred_global_distort, left_hand,
            frame_width).view(-1, K, 2)
        uv_coord_im_gt_global = self.recover_hand(uv_coord_im_gt_global,
                                                  left_hand,
                                                  frame_width).view(-1, K, 2)

        # Pad 2D keypoints exceeding boundaries
        if self.pad_2d and mode == 'loss':
            for i in range(len(batch_data_samples)):
                pred_kpt = uv_coord_im_pred_global_distort_noflip[i]
                gt_kpt = uv_coord_im_gt_global[i]
                data_sample = batch_data_samples[i]
                mask = self.keypoint_within_bounds(
                    gt_kpt, data_sample.meta['frame_width'],
                    data_sample.meta['frame_height'])
                if mask.any():
                    noise = (4 * torch.rand((mask.sum(), 2)) - 2).cuda()
                    pred_kpt[:, :2][mask] = gt_kpt[:, :2][mask] + noise
        uv_coord_im_pred_global = uv_coord_im_pred_global_distort_noflip.clone(
        )
        if self.all_use_kp2d_gt:
            uv_coord_im_pred_global = uv_coord_im_gt_global.view(-1, K,
                                                                 2).clone()

        for i, data_sample in enumerate(batch_data_samples):
            if data_sample.meta.get('stereo_aug', False):
                uv_coord_im_pred_global[i] = uv_coord_im_gt_global[i].clone()
            camera_model = data_sample.meta['ori_camera']
            kpt2d_u = camera_model.undistort(
                uv_coord_im_pred_global[i].cpu().numpy())
            uv_coord_im_pred_global[i] = torch.from_numpy(kpt2d_u).cuda()
        uv_coord_im_pred_global = uv_coord_im_pred_global.view(B, N, K, 2)

        leftcam_uv = uv_coord_im_pred_global[:, 0]
        leftcam_x = (leftcam_uv[:, :, 0] - leftcam_cam_matrix[:, 0, 2].view(
            (B, 1))) / leftcam_cam_matrix[:, 0, 0].view(B, 1)
        leftcam_y = (leftcam_uv[:, :, 1] - leftcam_cam_matrix[:, 1, 2].view(
            (B, 1))) / leftcam_cam_matrix[:, 1, 1].view(B, 1)
        leftcam_xy = torch.cat(
            (leftcam_x.unsqueeze(-1), leftcam_y.unsqueeze(-1)), dim=2)
        rightcam_uv = uv_coord_im_pred_global[:, 1]
        rightcam_x = (rightcam_uv[:, :, 0] - rightcam_cam_matrix[:, 0, 2].view(
            (B, 1))) / rightcam_cam_matrix[:, 0, 0].view(B, 1)
        rightcam_y = (rightcam_uv[:, :, 1] - rightcam_cam_matrix[:, 1, 2].view(
            (B, 1))) / rightcam_cam_matrix[:, 1, 1].view(B, 1)
        rightcam_xy = torch.cat(
            (rightcam_x.unsqueeze(-1), rightcam_y.unsqueeze(-1)), dim=2)

        norm_leftcam_xyz, norm_rightcam_xyz = self.standardize_stereo(
            leftcam_xy, rightcam_xy, left_R, right_R)

        if self.edge_to_center:
            norm_leftcam_xyz, norm_rightcam_xyz = self.kpt_edge_to_center(
                norm_leftcam_xyz, norm_rightcam_xyz)

        if self.use_plane_coord:
            feature1 = torch.cat((norm_leftcam_xyz[:, :, :2].reshape(
                (B, -1)), left_hand.view(B, -1)),
                                 dim=1)
            feature2 = torch.cat((norm_rightcam_xyz[:, :, :2].reshape(
                (B, -1)), left_hand.view(B, -1)),
                                 dim=1)
        else:
            feature1 = torch.cat((norm_leftcam_xyz.view(
                (B, -1)), left_hand.view(B, -1)),
                                 dim=1)
            feature2 = torch.cat((norm_rightcam_xyz.view(
                (B, -1)), left_hand.view(B, -1)),
                                 dim=1)
        feats = torch.cat((feature1, feature2), dim=1).float()
        if self.disparity_input:
            disparity = norm_leftcam_xyz[:, :, 0] - norm_rightcam_xyz[:, :, 0]
            feats = torch.cat((feats, disparity), dim=1).float()

        feats = feats.reshape(B, self.feat_dim, 1, 1)
        return (feats, norm_leftcam_xyz, norm_rightcam_xyz, lr_rot_matrix,
                lr_p, left_to_right_rt, leftcam_cam_matrix,
                rightcam_cam_matrix, uv_coord_im_pred_global,
                uv_coord_im_gt_global, uv_coord_im_pred_global_distort,
                uv_coord_im_pred_global_distort_noflip, hand3d_gt, left_R,
                right_R, baseline_scale)

    def postprocess(self, output, norm_leftcam_xyz, norm_rightcam_xyz, left_R,
                    right_R, lr_rot_matrix, lr_p, baseline_scale):
        B, K = norm_leftcam_xyz.shape[:2]
        if self.kpt3d_output:
            if self.kpt3d_output_delta:
                delta_output = output[:, 63:]
                output = output[:, :63].view(B, K, 3, 1)
                output = (output * delta_output).view(B, 63, 1, 1)
            baseline_scale = baseline_scale.view(B, 1, 1)
            hand3d_pred = (output.view(B, K, 3) *
                           baseline_scale).reshape(B, K, 3)
            return hand3d_pred, hand3d_pred, hand3d_pred
        baseline_scale = baseline_scale.view(B, 1, 1)
        lr_rot_matrix = lr_rot_matrix.view(B, 1, 3,
                                           3).repeat(1, 21, 1,
                                                     1).view(B * 21, 3, 3)
        lr_p = lr_p.view(B, 1, 3, 1).repeat(1, 21, 1, 1).view(B * 21, 3, 1)
        left_R_inv = torch.inverse(left_R).view(B, 1, 3, 3).repeat(
            1, 21, 1, 1).view(B * 21, 3, 3)
        right_R_inv = torch.inverse(right_R).view(B, 1, 3, 3).repeat(
            1, 21, 1, 1).view(B * 21, 3, 3)
        if self.use_plane_coord:
            leftcam_Z = output[:, :21].view(B, K, 1) * baseline_scale
            leftcam_XYZ = torch.cat(
                (norm_leftcam_xyz[:, :, :2] * leftcam_Z, leftcam_Z),
                dim=2).view(B * K, 3, 1)
            if not self.rightcam_3d_disable:
                rightcam_Z = output[:, 21:21 * 2].reshape(
                    (B, 21, 1)) * baseline_scale
                rightcam_XYZ = torch.cat(
                    (norm_rightcam_xyz[:, :, :2] * rightcam_Z, rightcam_Z),
                    dim=2).view(B * K, 3, 1)
        else:
            leftcam_Z_scale = output[:, :21].view(
                B, K, 1) * baseline_scale / norm_leftcam_xyz[:, :, 2:]
            leftcam_XYZ = (norm_leftcam_xyz *
                           leftcam_Z_scale).view(B * K, 3, 1)
            if not self.rightcam_3d_disable:
                rightcam_Z_scale = output[:, 21:21 * 2].reshape(
                    B, 21, 1) * baseline_scale / norm_rightcam_xyz[:, :, 2:]
                rightcam_XYZ = (norm_rightcam_xyz *
                                rightcam_Z_scale).view(B * K, 3, 1)
        if self.edge_to_center:
            leftcam_XYZ = self.center_to_edge(leftcam_XYZ)
        leftcam_XYZ = torch.bmm(left_R_inv, leftcam_XYZ).view(B, K, 3)
        if not self.rightcam_3d_disable:
            if self.edge_to_center:
                rightcam_XYZ = self.center_to_edge(rightcam_XYZ)
            rightcam_XYZ = torch.bmm(right_R_inv, rightcam_XYZ)
            rightcam_XYZ = (torch.bmm(lr_rot_matrix, rightcam_XYZ) +
                            lr_p).view(B, 21, 3)
            hand3d_pred = (
                self.corruption_cam * leftcam_XYZ +
                (1 - self.corruption_cam) * rightcam_XYZ)
        else:
            hand3d_pred, rightcam_XYZ = leftcam_XYZ, leftcam_XYZ

        return hand3d_pred, leftcam_XYZ, rightcam_XYZ

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            (feats, norm_leftcam_xyz, norm_rightcam_xyz, lr_rot_matrix, lr_p,
             left_to_right_rt, leftcam_cam_matrix, rightcam_cam_matrix,
             uv_coord_im_pred_global, uv_coord_im_gt_global,
             uv_coord_im_pred_global_distort, uv_coord_im_pred_global_distort_noflip,  # noqa
             hand3d_gt, left_R, right_R, baseline_scale) = \
                self.preprocess(feats, batch_data_samples, 'predict')
        output = self.forward(feats)
        hand3d_pred = self.postprocess(output, norm_leftcam_xyz,
                                       norm_rightcam_xyz, left_R, right_R,
                                       lr_rot_matrix, lr_p, baseline_scale)[0]
        if self.reproj:
            camera_model = batch_data_samples[0].meta['ori_camera']
            leftcam_uv_reproj_distort = camera_model.eye_to_window(
                hand3d_pred.cpu().numpy())
            leftcam_uv_reproj_distort = torch.tensor(
                leftcam_uv_reproj_distort).cuda()
            return hand3d_pred, leftcam_uv_reproj_distort[:, None, ...]
        else:
            return hand3d_pred, uv_coord_im_pred_global_distort

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        with torch.no_grad():
            (feats, norm_leftcam_xyz, norm_rightcam_xyz, lr_rot_matrix, lr_p,
             left_to_right_rt, leftcam_cam_matrix, rightcam_cam_matrix,
             uv_coord_im_pred_global, uv_coord_im_gt_global,
             uv_coord_im_pred_global_distort, uv_coord_im_pred_global_distort_noflip,  # noqa
             hand3d_gt, left_R, right_R, baseline_scale) = \
                self.preprocess(feats, batch_data_samples, 'loss')
        output = self.forward(feats)
        hand3d_pred, leftcam_XYZ, rightcam_XYZ = self.postprocess(
            output, norm_leftcam_xyz, norm_rightcam_xyz, left_R, right_R,
            lr_rot_matrix, lr_p, baseline_scale)

        leftcam_uv_gt = uv_coord_im_pred_global[:, 0]
        rightcam_uv_gt = uv_coord_im_pred_global[:, 1]

        leftcam_uv_reproj, rightcam_uv_reproj = \
            self.trans_3d_2_2d(hand3d_pred, leftcam_cam_matrix,
                               rightcam_cam_matrix, left_to_right_rt)

        major_gt = torch.cat((hand3d_gt[:, 4:5, :], hand3d_gt[:, 8:9, :]),
                             dim=1)
        major_pred = torch.cat(
            (hand3d_pred[:, 4:5, :], hand3d_pred[:, 8:9, :]), dim=1)

        # origin distance, no norm
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(hand3d_gt[:, 4, :] - hand3d_gt[:, 8, :], dim=-1)
        pred_for_loss = [
            hand3d_pred, leftcam_XYZ, rightcam_XYZ, leftcam_uv_reproj,
            rightcam_uv_reproj, dist_pred, major_pred
        ]
        targ_for_loss = [
            hand3d_gt, hand3d_gt, hand3d_gt, leftcam_uv_gt, rightcam_uv_gt,
            dist_gt, major_gt
        ]

        # filter abnormal GT
        self.filter_invalid_gt(uv_coord_im_gt_global[::2],
                               uv_coord_im_pred_global_distort_noflip[::2],
                               pred_for_loss, targ_for_loss)

        losses = self.lift_loss(pred_for_loss, targ_for_loss)
        (loss_mse_3d, loss_mse_3d_leftcam, loss_mse_3d_rightcam,
         loss_mse_2d_leftcam, loss_mse_2d_rightcam, loss_pinch,
         loss_major) = losses
        if self.lambda_t > 0:
            mh = MessageHub.get_current_instance()
            cur_epoch = mh.get_info('epoch')
            if cur_epoch <= self.lambda_t:
                loss_mse_2d_leftcam *= 0
                loss_mse_2d_rightcam *= 0
        if self.rightcam_3d_disable or self.kpt3d_output:
            loss_mse_3d_leftcam *= 0
            loss_mse_3d_rightcam *= 0
        losses_dict = dict(
            loss_mse_3d=loss_mse_3d,
            loss_mse_3d_leftcam=loss_mse_3d_leftcam,
            loss_mse_3d_rightcam=loss_mse_3d_rightcam,
            loss_mse_2d_leftcam=loss_mse_2d_leftcam,
            loss_mse_2d_rightcam=loss_mse_2d_rightcam,
            loss_pinch=loss_pinch,
            loss_major=loss_major)
        return losses_dict

    def filter_invalid_gt(self, gt_kpt, pred_kpt, pred_for_loss,
                          targ_for_loss):
        if self.iou_thre and self.reproj_thre:
            kpt2d_error = ((gt_kpt - pred_kpt)).abs().sum(axis=(1, 2))
            gt_bbox = [kpt_to_bbox(np.array(kpt.cpu())) for kpt in gt_kpt]
            pred_bbox = [
                kpt_to_bbox(np.array(kpt.detach().cpu())) for kpt in pred_kpt
            ]
            iou = torch.tensor(
                self.compute_iou(np.array(gt_bbox),
                                 np.array(pred_bbox))).cuda()
            index = (kpt2d_error > self.reproj_thre) & (iou < self.iou_thre)
            pred_for_loss = [loss[~index] for loss in pred_for_loss]
            targ_for_loss = [loss[~index] for loss in targ_for_loss]

    def standardize_stereo(self, leftcam_xy, rightcam_xy, left_R, right_R):
        """transform to standard stereo system."""
        standard_left_xyz = self.align_monocular_to_parallel_stereo(
            leftcam_xy, left_R)
        standard_right_xyz = self.align_monocular_to_parallel_stereo(
            rightcam_xy, right_R)
        if self.use_plane_coord:
            norm_left_xyz = standard_left_xyz / standard_left_xyz[:, :, 2:]
            norm_right_xyz = standard_right_xyz / standard_right_xyz[:, :, 2:]
            if self.plane_arctan:
                norm_left_xyz[:, :, :2] = torch.arctan(norm_left_xyz[:, :, :2])
                norm_right_xyz[:, :, :2] = torch.arctan(
                    norm_right_xyz[:, :, :2])
        else:
            norm_left_xyz = F.normalize(standard_left_xyz, p=2, dim=-1)
            norm_right_xyz = F.normalize(standard_right_xyz, p=2, dim=-1)
        return norm_left_xyz, norm_right_xyz

    @staticmethod
    def align_monocular_to_parallel_stereo(cam_xy, rot):
        """Aligns a monocular camera to a parallel stereo setup using the given
        rotation matrix."""
        B, K = cam_xy.shape[:2]
        cam_xyz = torch.cat((cam_xy, torch.ones(B, K, 1).cuda()),
                            dim=-1).view(B * K, 3, 1)
        rot = rot.view(B, 1, 3, 3).repeat(1, K, 1, 1).view(B * K, 3, 3)
        standard_cam_xyz = torch.matmul(rot, cam_xyz).view(B, K, 3)
        return standard_cam_xyz

    @staticmethod
    def trans_3d_2_2d(hand3d_point, leftcam_cam_matrix, rightcam_cam_matrix,
                      left_to_right_rt):
        B = hand3d_point.shape[0]
        left_to_right_rt = left_to_right_rt.repeat(B, 1, 1)
        leftcam_uv_reproj = torch.matmul(hand3d_point,
                                         leftcam_cam_matrix.permute(
                                             0, 2, 1)).to(torch.float32)
        leftcam_uv_reproj = leftcam_uv_reproj[..., :2] / leftcam_uv_reproj[...,
                                                                           2:]

        column_of_ones = torch.ones((B, 21, 1)).to(hand3d_point.device)
        tensor_with_ones = torch.cat((hand3d_point, column_of_ones), dim=2)
        rightcam_uv_reproj = torch.matmul(tensor_with_ones,
                                          left_to_right_rt.permute(
                                              0, 2, 1)).to(torch.float32)
        rightcam_uv_reproj = rightcam_uv_reproj[..., :3] / rightcam_uv_reproj[
            ..., 3:]
        rightcam_uv_reproj = torch.matmul(rightcam_uv_reproj,
                                          rightcam_cam_matrix.permute(
                                              0, 2, 1)).to(torch.float32)
        rightcam_uv_reproj = rightcam_uv_reproj[..., :2] / rightcam_uv_reproj[
            ..., 2:]
        return leftcam_uv_reproj, rightcam_uv_reproj

    def center_to_edge(self, hand_3d):
        B = self.center_rot.shape[0]
        inv_rot = torch.inverse(self.center_rot)
        inv_rot = inv_rot.view(B, 1, 3, 3).repeat(1, 21, 1,
                                                  1).view(B * 21, 3, 3)
        hand_3d = torch.matmul(inv_rot, hand_3d)
        return hand_3d

    def kpt_edge_to_center(self, norm_leftcam_xyz, norm_rightcam_xyz):
        B, K = norm_leftcam_xyz.shape[:2]
        center_rot = []
        for i in range(norm_leftcam_xyz.shape[0]):
            left_root = np.array(norm_leftcam_xyz[i][9].clone().cpu())
            rot = from_two_vectors(left_root, np.array([0, 0, 1]))
            center_rot.append(rot)
        center_rot = torch.tensor(np.stack(center_rot)).float().cuda()
        rot_leftcam_xyz = torch.matmul(
            center_rot.view(B, 1, 3, 3).repeat(1, K, 1, 1).view(B * K, 3, 3),
            norm_leftcam_xyz.view(B * K, 3, 1)).view(B, K, 3)
        rot_rightcam_xyz = torch.matmul(
            center_rot.view(B, 1, 3, 3).repeat(1, K, 1, 1).view(B * K, 3, 3),
            norm_rightcam_xyz.view(B * K, 3, 1)).view(B, K, 3)

        if self.use_plane_coord:
            norm_leftcam_xyz = rot_leftcam_xyz / rot_leftcam_xyz[:, :, 2:]
            norm_rightcam_xyz = rot_rightcam_xyz / rot_rightcam_xyz[:, :, 2:]
        else:
            norm_leftcam_xyz = F.normalize(rot_leftcam_xyz, p=2, dim=-1)
            norm_rightcam_xyz = F.normalize(rot_rightcam_xyz, p=2, dim=-1)
        self.center_rot = center_rot
        return norm_leftcam_xyz, norm_rightcam_xyz

    @staticmethod
    def keypoint_within_bounds(keypoint, image_width, image_height):
        x, y = keypoint[:, 0], keypoint[:, 1]
        mask = ((0 <= x) & (x < image_width)) & ((0 <= y) & (y < image_height))
        return ~mask

    @staticmethod
    def compute_iou(gt_bboxes, pred_bboxes):
        # 计算交集的坐标范围
        x1 = np.maximum(gt_bboxes[:, 0], pred_bboxes[:, 0])
        y1 = np.maximum(gt_bboxes[:, 1], pred_bboxes[:, 1])
        x2 = np.minimum(gt_bboxes[:, 0] + gt_bboxes[:, 2],
                        pred_bboxes[:, 0] + pred_bboxes[:, 2])
        y2 = np.minimum(gt_bboxes[:, 1] + gt_bboxes[:, 3],
                        pred_bboxes[:, 1] + pred_bboxes[:, 3])

        # 计算交集的面积
        intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # 计算各自框的面积
        area_gt = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        area_pred = pred_bboxes[:, 2] * pred_bboxes[:, 3]

        # 计算并返回 IoU
        ious = intersection_area / (area_gt + area_pred - intersection_area)
        return ious
