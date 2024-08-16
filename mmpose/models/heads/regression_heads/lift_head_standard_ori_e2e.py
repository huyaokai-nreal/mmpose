# Copyright (c) XREAL. All rights reserved.
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.logging import MessageHub
from mmengine.model import BaseModule
from nreal_data_tool.utils import kpt_to_bbox
from torch import Tensor, nn

from mmpose.models.utils.gmlp import gMLP
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList, Predictions
from .lift_head_standard_ori import LiftHeadStandardOri


@MODELS.register_module()
class LiftHeadStandardOriE2e(LiftHeadStandardOri):
    """liftHead for getting 3d keypoints from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 num_layers: int = 3,
                 d_ffn: int = 220,
                 output_num: int = 42,
                 reproj: bool = False,
                 baseline=0.13,
                 d_model=512,
                 reproj_thre=0,
                 iou_thre=0,
                 scale_baseline=0,
                 pad_2d=False,
                 lambda_t: int = -1,
                 corruption_cam: float = 0.5,
                 all_use_kp2d_gt: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(
            lift_loss,
            num_layers,
            d_ffn,
            output_num,
            reproj,
            baseline,
            all_use_kp2d_gt=all_use_kp2d_gt,
            corruption_cam=corruption_cam,
            init_cfg=init_cfg,
            reproj_thre=reproj_thre,
            iou_thre=iou_thre,
            pad_2d=pad_2d)

    def preprocess(self, feats, batch_data_samples, mode):
        xy_coord = feats[..., :2]
        B = int(len(batch_data_samples) / 2)
        N = 2
        H, W = batch_data_samples[0].input_size
        K = xy_coord.shape[1]
        # kpt2d output to crop wh
        uv_coord_im_pred_crop_right = xy_coord * torch.tensor([W, H]).cuda()
        uv_coord_im_pred_crop = uv_coord_im_pred_crop_right.view(B, N, K, 2)
        left_vir_cam_matrix = []
        right_vir_cam_matrix = []
        left_vircam_xf = []
        right_vircam_xf = []
        left_R = []
        right_R = []
        baseline_scale = []
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
                left_vir_camera = data_sample.meta['virtual_camera']
                left_camera = data_sample.meta['ori_camera']
                left_vir_cam_matrix.append(left_vir_camera.uv_to_window_matrix())
                left_R.append(data_sample.meta['cam_to_virtual_R'])
                hand3d_gt.append(data_sample.gt_instances.keypoints3d[0])
                if 'nimble_pose' in data_sample.meta.keys() and not np.equal(
                        data_sample.meta['nimble_pose'].any(), None):
                    nimble_pose.append(data_sample.meta['nimble_pose'])
                    nimble_trans.append(data_sample.meta['nimble_translation'])
                    nimble_shape.append(data_sample.meta['nimble_shape'])
                if data_sample.meta['category_id'] == 1:
                    is_left_hands.append(1)
                    if data_sample.meta['flipped']:
                        uv_coord_im_pred_crop[
                            i // 2, :, :,
                            0] = W - 1 - uv_coord_im_pred_crop[i // 2, :, :, 0]
                else:
                    is_left_hands.append(0)
            else:
                right_vir_camera = data_sample.meta['virtual_camera']
                right_vir_cam_matrix.append(right_vir_camera.uv_to_window_matrix())
                right_R.append(data_sample.meta['cam_to_virtual_R'])
                left_vircam_xf.append(left_vir_camera.camera_to_world_xf[:3,:3])
                right_vircam_xf.append(right_vir_camera.camera_to_world_xf[:3,:3])
                left_cam_xf = left_camera.camera_to_world_xf
                right_cam_xf = data_sample.meta['ori_xf']
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
        left_vir_cam_matrix = torch.tensor(
            np.array(left_vir_cam_matrix)).cuda().float()
        right_vir_cam_matrix = torch.tensor(
            np.array(right_vir_cam_matrix)).cuda().float()
        left_vircam_xf = torch.tensor(
            np.array(left_vircam_xf)).cuda().float()
        right_vircam_xf = torch.tensor(
            np.array(right_vircam_xf)).cuda().float()
        left_R = torch.tensor(np.array(left_R)).cuda().float()
        right_R = torch.tensor(np.array(right_R)).cuda().float()
        baseline_scale = torch.tensor(np.array(baseline_scale)).cuda().float()
        lr_p = torch.tensor(np.array(lr_p)).cuda().float()
        lr_rot_matrix = torch.tensor(np.array(lr_rot_matrix)).cuda().float()
        left_to_right_rt = torch.tensor(
            np.array(left_to_right_rt)).cuda().float()
        hand3d_gt = torch.tensor(np.array(hand3d_gt)).cuda().float()
        if len(nimble_pose) > 0:
            nimble_pose = torch.tensor(np.array(nimble_pose)).cuda().float()
            nimble_trans = torch.tensor(np.array(nimble_trans)).cuda().float()
            nimble_shape = torch.tensor(np.array(nimble_shape)).cuda().float()
        left_hand = torch.tensor(np.array(is_left_hands)).cuda().float()
        uv_coord_im_gt_global = torch.tensor(
            np.array(uv_coord_im_gt_global)).cuda().float()
        uv_coord_im_gt_global = uv_coord_im_gt_global[..., :2]
        uv_coord_im_gt_global = uv_coord_im_gt_global.view(-1, K, 2)

        uv_coord_im_pred_global = uv_coord_im_pred_crop.view(
            B, N, K, 2)

        try:
            nimble_info = {
                'nimble_pose': nimble_pose,
                'nimble_trans': nimble_trans,
                'nimble_shape': nimble_shape
            }
        except Exception as e:
            nimble_info = {
                'nimble_pose': None,
                'nimble_trans': None,
                'nimble_shape': None,
            }
            print(f'An error occurred: {e}')

        leftcam_uv = uv_coord_im_pred_global[:, 0].clone()
        leftcam_x = (leftcam_uv[:, :, 0] - left_vir_cam_matrix[:, 0, 2].view(
            (B, 1))) / left_vir_cam_matrix[:, 0, 0].view(B, 1)
        leftcam_y = (leftcam_uv[:, :, 1] - left_vir_cam_matrix[:, 1, 2].view(
            (B, 1))) / left_vir_cam_matrix[:, 1, 1].view(B, 1)
        leftcam_xy = torch.cat(
            (leftcam_x.unsqueeze(-1), leftcam_y.unsqueeze(-1)), dim=2)
        rightcam_uv = uv_coord_im_pred_global[:, 1].clone()
        rightcam_x = (rightcam_uv[:, :, 0] - right_vir_cam_matrix[:, 0, 2].view(
            (B, 1))) / right_vir_cam_matrix[:, 0, 0].view(B, 1)
        rightcam_y = (rightcam_uv[:, :, 1] - right_vir_cam_matrix[:, 1, 2].view(
            (B, 1))) / right_vir_cam_matrix[:, 1, 1].view(B, 1)
        rightcam_xy = torch.cat(
            (rightcam_x.unsqueeze(-1), rightcam_y.unsqueeze(-1)), dim=2)

        uv_coord_im_pred_global = uv_coord_im_pred_global.view(-1, K, 2)
        for i, data_sample in enumerate(batch_data_samples):
            virtual_cam = batch_data_samples[i].meta['virtual_camera']
            ori_cam = batch_data_samples[i].meta['ori_camera']
            kpt_norm_eye = virtual_cam.window_to_eye(uv_coord_im_pred_global[i].clone().cpu())
            kpt_norm_world = virtual_cam.eye_to_world(kpt_norm_eye)
            kpt2d_ori = ori_cam.eye_to_window(kpt_norm_world)
            uv_coord_im_pred_global[i] = torch.tensor(kpt2d_ori).cuda().float()

        # 相机坐标转标准双目
        norm_leftcam_xyz, norm_rightcam_xyz = self.standardize_stereo(
            leftcam_xy, rightcam_xy, left_R, right_R, left_vircam_xf, right_vircam_xf)

        feature1 = torch.cat((norm_leftcam_xyz[:, :, :2].reshape(
            (B, -1)), left_hand.view(B, -1)),
                             dim=1)
        feature2 = torch.cat((norm_rightcam_xyz[:, :, :2].reshape(
            (B, -1)), left_hand.view(B, -1)),
                             dim=1)
        feats = torch.cat((feature1, feature2), dim=1).float()

        feats = feats.reshape(B, self.feat_dim, 1, 1)
        return {
            'feats': feats,
            'norm_leftcam_xyz': norm_leftcam_xyz,
            'norm_rightcam_xyz': norm_rightcam_xyz,
            'lr_rot_matrix': lr_rot_matrix,
            'lr_p': lr_p,
            'left_to_right_rt': left_to_right_rt,
            'left_vir_cam_matrix': left_vir_cam_matrix,
            'right_vir_cam_matrix': right_vir_cam_matrix,
            'uv_coord_im_pred_global': uv_coord_im_pred_global,
            'hand3d_gt': hand3d_gt,
            'left_R': left_R,
            'right_R': right_R,
            'leftcam_xy': leftcam_xy,
            'left_hand': left_hand,
            'baseline_scale': baseline_scale,
            'nimble_info': nimble_info,
        }


    def postprocess(self, output, norm_leftcam_xyz, norm_rightcam_xyz, left_R,
                    right_R, lr_rot_matrix, lr_p, baseline_scale):
        B, K = norm_leftcam_xyz.shape[:2]
        baseline_scale = baseline_scale.view(B, 1, 1)
        lr_rot_matrix = lr_rot_matrix.view(B, 1, 3,
                                           3).repeat(1, 21, 1,
                                                     1).view(B * 21, 3, 3)
        lr_p = lr_p.view(B, 1, 3, 1).repeat(1, 21, 1, 1).view(B * 21, 3, 1)
        left_R_inv = torch.inverse(left_R).view(B, 1, 3, 3).repeat(
            1, 21, 1, 1).view(B * 21, 3, 3)
        right_R_inv = torch.inverse(right_R).view(B, 1, 3, 3).repeat(
            1, 21, 1, 1).view(B * 21, 3, 3)
        leftcam_Z = output[:, :21].view(B, K, 1) * baseline_scale
        leftcam_XYZ = torch.cat(
            (norm_leftcam_xyz[:, :, :2] * leftcam_Z, leftcam_Z),
            dim=2).view(B * K, 3, 1)
        rightcam_Z = output[:, 21:21 * 2].reshape(
            (B, 21, 1)) * baseline_scale
        rightcam_XYZ = torch.cat(
            (norm_rightcam_xyz[:, :, :2] * rightcam_Z, rightcam_Z),
            dim=2).view(B * K, 3, 1)
        leftcam_XYZ = torch.bmm(left_R_inv, leftcam_XYZ).view(B, K, 3)
        rightcam_XYZ = torch.bmm(right_R_inv, rightcam_XYZ)
        rightcam_XYZ = (torch.bmm(lr_rot_matrix, rightcam_XYZ) +
                        lr_p).view(B, 21, 3)
        hand3d_pred = (
            self.corruption_cam * leftcam_XYZ +
            (1 - self.corruption_cam) * rightcam_XYZ)
        return hand3d_pred, leftcam_XYZ, rightcam_XYZ

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'predict')
        output = self.forward(data['feats'])
        hand3d_pred, leftcam_XYZ, rightcam_XYZ = self.postprocess(output, data['norm_leftcam_xyz'],
                                data['norm_rightcam_xyz'], data['left_R'], data['right_R'],
                                data['lr_rot_matrix'], data['lr_p'], data['baseline_scale'])
        return hand3d_pred, data['uv_coord_im_pred_global']

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'loss')
        output = self.forward(data['feats'])
        
        hand3d_pred, leftcam_XYZ, rightcam_XYZ = self.postprocess(output, data['norm_leftcam_xyz'],
                                data['norm_rightcam_xyz'], data['left_R'], data['right_R'],
                                data['lr_rot_matrix'], data['lr_p'], data['baseline_scale'])
        hand3d_gt = data['hand3d_gt']
        major_gt = torch.cat((hand3d_gt[:, 1:10, :],
                              hand3d_gt[:, 13, :].unsqueeze(1)),
                             dim=1)
        major_pred = torch.cat(
            (hand3d_pred[:, 1:10, :], hand3d_pred[:, 13, :].unsqueeze(1)),
            dim=1)

        # pinch distance, no norm
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(
            hand3d_gt[:, 4, :] - hand3d_gt[:, 8, :], dim=-1)
        pred_for_loss = [
            hand3d_pred, leftcam_XYZ, rightcam_XYZ,
            dist_pred
        ]
        targ_for_loss = [
            hand3d_gt, hand3d_gt, hand3d_gt, dist_gt
        ]

        losses = self.lift_loss(pred_for_loss, targ_for_loss)
        (loss_mse_3d, loss_mse_3d_leftcam, loss_mse_3d_rightcam, loss_pinch) = losses
        if self.lambda_t > 0:
            mh = MessageHub.get_current_instance()
            cur_epoch = mh.get_info('epoch')
            if cur_epoch <= self.lambda_t:
                loss_mse_2d_leftcam *= 0
                loss_mse_2d_rightcam *= 0
        losses_dict = dict(
            loss_mse_3d=loss_mse_3d,
            loss_mse_3d_leftcam=loss_mse_3d_leftcam,
            loss_mse_3d_rightcam=loss_mse_3d_rightcam,
            loss_pinch=loss_pinch)
        return losses_dict

    def standardize_stereo(self, leftcam_xy, rightcam_xy, left_R, right_R, left_vircam_xf, right_vircam_xf):
        """transform to standard stereo system."""
        standard_left_xyz = self.align_monocular_to_parallel_stereo(
            leftcam_xy, left_R, left_vircam_xf)
        standard_right_xyz = self.align_monocular_to_parallel_stereo(
            rightcam_xy, right_R, right_vircam_xf)
        norm_left_xyz = standard_left_xyz / standard_left_xyz[:, :, 2:]
        norm_right_xyz = standard_right_xyz / standard_right_xyz[:, :, 2:]
        return norm_left_xyz, norm_right_xyz

    @staticmethod
    def align_monocular_to_parallel_stereo(cam_xy, R, vircam_xf):
        """Aligns a monocular camera to a parallel stereo setup using the given
        rotation matrix."""
        B, K = cam_xy.shape[:2]
        cam_xyz = torch.cat((cam_xy, torch.ones(B, K, 1).cuda()),
                            dim=-1).view(B * K, 3, 1)
        vircam_xf = vircam_xf.view(B, 1, 3, 3).repeat(1, K, 1, 1).view(B * K, 3, 3)
        R = R.view(B, 1, 3, 3).repeat(1, K, 1, 1).view(B * K, 3, 3)

        oricam_cam_xyz = torch.matmul(vircam_xf, cam_xyz)
        standard_cam_xyz = torch.matmul(R, oricam_cam_xyz).view(B, K, 3)
        return standard_cam_xyz

