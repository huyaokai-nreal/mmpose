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


class UncertaintyModel(nn.Module):

    def __init__(self, feat_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.fc2 = nn.Linear(feat_dim, out_dim)
        self.dropout = nn.Dropout(0.5)
        self.norm1 = nn.BatchNorm1d(feat_dim)
        self.norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.fc2(x)))
        return x


@MODELS.register_module()
class LiftHeadStandard(BaseModule):
    """liftHead for getting 3d keypoints from pair 2d keypoints."""

    def __init__(self,
                 lift_loss: ConfigType,
                 num_layers: int = 3,
                 d_ffn: int = 220,
                 output_num: int = 42,
                 reproj: bool = False,
                 baseline=0.13,
                 score_dim=0,
                 d_model=512,
                 reproj_thre=0,
                 iou_thre=0,
                 pad_2d=False,
                 lambda_t: int = -1,
                 corruption_cam: float = 0.5,
                 all_use_kp2d_gt: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
        self.all_use_kp2d_gt = all_use_kp2d_gt
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_num = 43
        self.lambda_t = lambda_t
        self.score_dim = score_dim
        feat_dim = 2 * self.channel_num
        self.liftnet = gMLP(
            d_model=feat_dim, d_ffn=d_ffn, num_layers=num_layers)
        self.corruption_cam = corruption_cam
        self.last_layer = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1), nn.ReLU(),
            nn.Conv2d(feat_dim, output_num, kernel_size=1))
        if self.score_dim:
            score_feat_dim = feat_dim + output_num * 2
            self.major_score_layer = nn.Sequential(
                nn.Conv2d(score_feat_dim, score_feat_dim, kernel_size=1),
                nn.BatchNorm2d(score_feat_dim),
                nn.ReLU(),
                nn.Conv2d(score_feat_dim, 10, kernel_size=1),
                nn.BatchNorm2d(10),
                nn.ReLU(),
                # gMLP(d_model=score_feat_dim, d_ffn=d_ffn, num_layers=1),
                # UncertaintyModel(score_feat_dim, 10),
            )
            self.pinch_score_layer = nn.Sequential(
                nn.Conv2d(score_feat_dim, score_feat_dim, kernel_size=1),
                nn.BatchNorm2d(score_feat_dim),
                nn.ReLU(),
                nn.Conv2d(score_feat_dim, 2, kernel_size=1),
                nn.BatchNorm2d(2),
                nn.ReLU(),
                # gMLP(d_model=score_feat_dim, d_ffn=d_ffn, num_layers=1),
                # UncertaintyModel(score_feat_dim, 2))
            )
            self.reproj_layer = nn.Sequential(
                nn.Conv2d(output_num * 2, output_num * 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(output_num * 2, output_num * 2, kernel_size=1))
        self.feat_dim = feat_dim
        self.lift_loss = MODELS.build(lift_loss)
        self.reproj = reproj
        self.baseline = baseline
        self.reproj_thre = reproj_thre
        self.iou_thre = iou_thre
        self.pad_2d = pad_2d
        if self.score_dim:
            for param in self.liftnet.parameters():
                param.requires_grad = False
            for param in self.last_layer.parameters():
                param.requires_grad = False

    def forward(self, feats):
        B, K = feats.shape[0], feats.shape[1] // 2 - 1
        virtual_baseline = (torch.ones(B) * self.baseline).cuda()
        # 标准双目归一化平面2d
        norm_leftcam_xyz = torch.cat(
            (feats[:, :K, 0, 0].reshape(B, K // 2, 2), torch.ones(
                B, K // 2, 1).cuda()),
            dim=-1)
        norm_rightcam_xyz = torch.cat((feats[:, K + 1:-1, 0, 0].reshape(
            B, K // 2, 2), torch.ones(B, K // 2, 1).cuda()),
                                      dim=-1)
        liftnet_output = self.liftnet(feats)
        output = self.last_layer(liftnet_output).view(feats.shape[0], -1, 1, 1)

        # 标准双目3d点输出
        hand3d_standard = self.get_standard_kpt3d(output, norm_leftcam_xyz,
                                                  norm_rightcam_xyz,
                                                  virtual_baseline)
        # score output
        if self.score_dim:
            left_reproj, right_reproj = self.trans_3d_2_2d(
                hand3d_standard, virtual_baseline)
            left_reproj_error = (left_reproj - norm_leftcam_xyz[..., :2]).view(
                hand3d_standard.shape[0], -1, 1, 1)
            right_reproj_error = (right_reproj -
                                  norm_rightcam_xyz[..., :2]).view(
                                      hand3d_standard.shape[0], -1, 1, 1)
            reproj_error = torch.cat((left_reproj_error, right_reproj_error),
                                     dim=1)
            reproj_feats = self.reproj_layer(reproj_error)
            score_feats = torch.cat((reproj_feats, liftnet_output), axis=1)
            major_score = self.major_score_layer(score_feats).view(
                hand3d_standard.shape[0], -1)
            pinch_score = self.pinch_score_layer(score_feats).view(
                hand3d_standard.shape[0], -1)
            score = torch.cat((major_score, pinch_score), dim=-1)
        else:
            score = torch.ones(B, 12)
        return hand3d_standard, score

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
        virtual_baseline = []
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
                virtual_baseline.append(data_sample.meta['virtual_baseline'])
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
        virtual_baseline = torch.tensor(
            np.array(virtual_baseline)).cuda().float()
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
                if self.pad_2d <= mask.sum() < 21:  # 可见点足够多时
                    noise = (4 * torch.rand((21 - mask.sum(), 2)) - 2).cuda()
                    pred_kpt[:, :2][~mask] = gt_kpt[:, :2][~mask] + noise
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
            'leftcam_cam_matrix': leftcam_cam_matrix,
            'rightcam_cam_matrix': rightcam_cam_matrix,
            'uv_coord_im_pred_global': uv_coord_im_pred_global,
            'uv_coord_im_gt_global': uv_coord_im_gt_global,
            'uv_coord_im_pred_global_distort': uv_coord_im_pred_global_distort,
            'uv_coord_im_pred_global_distort_noflip':
            uv_coord_im_pred_global_distort_noflip,
            'hand3d_gt': hand3d_gt,
            'left_R': left_R,
            'right_R': right_R,
            'baseline_scale': baseline_scale,
            'virtual_baseline': virtual_baseline,
            'nimble_info': None
        }

    def postprocess(self, hand3d_standard, left_to_right_rt, left_R,
                    baseline_scale):
        B, K = hand3d_standard.shape[:2]
        baseline_scale = baseline_scale.view(B, 1, 1)
        left_to_right_rt = left_to_right_rt.unsqueeze(0)
        left_R_inv = torch.inverse(left_R).view(B, 1, 3,
                                                3).repeat(1, K, 1,
                                                          1).view(B * K, 3, 3)
        # 实际左目3d
        hand_3d = (hand3d_standard * baseline_scale).view(B * K, 3, 1)
        hand3d_pred = torch.bmm(left_R_inv, hand_3d)
        # 实际右目3d
        left_to_right_rot = left_to_right_rt[:1, :3, :3].repeat(
            hand3d_pred.shape[0], 1, 1)
        left_to_right_t = left_to_right_rt[:1, :3,
                                           -1:].repeat(hand3d_pred.shape[0], 1,
                                                       1)
        rightcam_XYZ = (torch.bmm(left_to_right_rot, hand3d_pred) +
                        left_to_right_t).view(B, K, 3)
        hand3d_pred = hand3d_pred.view(B, K, 3)
        return hand3d_pred, hand3d_pred, rightcam_XYZ

    def get_standard_kpt3d(self, output, norm_leftcam_xyz, norm_rightcam_xyz,
                           virtual_baseline):
        B, K = norm_leftcam_xyz.shape[:2]
        leftcam_Z = output[:, :21].view(B, K, 1)
        leftcam_XYZ = torch.cat(
            (norm_leftcam_xyz[:, :, :2] * leftcam_Z, leftcam_Z), dim=2)
        rightcam_Z = output[:, 21:21 * 2].reshape((B, 21, 1))
        rightcam_XYZ = torch.cat(
            (norm_rightcam_xyz[:, :, :2] * rightcam_Z, rightcam_Z), dim=2)
        virtual_baseline = virtual_baseline.reshape(B, 1, 1).repeat(1, 21, 1)
        # 虚拟左目系下的虚拟右目3d点
        rightcam_XYZ[..., :1] = rightcam_XYZ[..., :1] + virtual_baseline
        # 标准双目3d点
        hand3d_pred = (
            self.corruption_cam * leftcam_XYZ +
            (1 - self.corruption_cam) * rightcam_XYZ)
        return hand3d_pred

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'predict')
        hand3d_standard, score = self.forward(data['feats'])
        hand3d_pred, leftcam_XYZ, rightcam_XYZ = self.postprocess(
            hand3d_standard, data['left_to_right_rt'], data['left_R'],
            data['baseline_scale'])

        if self.reproj:
            camera_model = batch_data_samples[0].meta['ori_camera']
            leftcam_uv_reproj_distort = camera_model.eye_to_window(
                hand3d_pred.cpu().numpy())
            leftcam_uv_reproj_distort = torch.tensor(
                leftcam_uv_reproj_distort).cuda()
            return (hand3d_pred, leftcam_uv_reproj_distort[:, None, ...],
                    score) if self.score_dim else (
                        hand3d_pred, leftcam_uv_reproj_distort[:, None, ...])
        else:
            return (hand3d_pred, data['uv_coord_im_pred_global_distort'],
                    score) if self.score_dim else (
                        hand3d_pred, data['uv_coord_im_pred_global_distort'])

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        with torch.no_grad():
            data = self.preprocess(feats, batch_data_samples, 'loss')
        hand3d_standard, score = self.forward(data['feats'])
        hand3d_pred, leftcam_XYZ, rightcam_XYZ = self.postprocess(
            hand3d_standard, data['left_to_right_rt'], data['left_R'],
            data['baseline_scale'])

        left_reproj, right_reproj = self.trans_3d_2_2d(
            hand3d_standard, data['virtual_baseline'])
        major_gt = torch.cat((data['hand3d_gt'][:, 1:10, :],
                              data['hand3d_gt'][:, 13, :].unsqueeze(1)),
                             dim=1)
        major_pred = torch.cat(
            (hand3d_pred[:, 1:10, :], hand3d_pred[:, 13, :].unsqueeze(1)),
            dim=1)

        # pinch distance, no norm
        dist_pred = torch.norm(
            hand3d_pred[:, 4, :] - hand3d_pred[:, 8, :], dim=-1)
        dist_gt = torch.norm(
            data['hand3d_gt'][:, 4, :] - data['hand3d_gt'][:, 8, :], dim=-1)
        pred_for_loss = [
            hand3d_pred, leftcam_XYZ, rightcam_XYZ, left_reproj, right_reproj,
            dist_pred
        ]
        targ_for_loss = [
            data['hand3d_gt'], data['hand3d_gt'], data['hand3d_gt'],
            data['norm_leftcam_xyz'][..., :2],
            data['norm_rightcam_xyz'][..., :2], dist_gt
        ]

        # filter abnormal GT
        self.filter_invalid_gt(
            data['uv_coord_im_gt_global'][::2],
            data['uv_coord_im_pred_global_distort_noflip'][::2], pred_for_loss,
            targ_for_loss)

        losses = self.lift_loss(pred_for_loss, targ_for_loss)
        (loss_mse_3d, loss_mse_3d_leftcam, loss_mse_3d_rightcam,
         loss_mse_2d_leftcam, loss_mse_2d_rightcam, loss_pinch) = losses
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
            loss_mse_2d_leftcam=loss_mse_2d_leftcam,
            loss_mse_2d_rightcam=loss_mse_2d_rightcam,
            loss_pinch=loss_pinch)

        if self.score_dim:
            losses_dict['loss_major_score'], losses_dict[
                'loss_pinch_score_dist'], losses_dict[
                    'loss_pinch_score_mpjpe'] = self.compute_score_loss(
                        major_pred, major_gt, dist_pred, dist_gt, score)
        return losses_dict

    @staticmethod
    def draw_keypoints(img_path, keypoints, color=(0, 0, 0), radius=1):
        from nreal_data_tool import LmdbClient
        lmdb_client = LmdbClient()
        image = lmdb_client.get(img_path)
        output_image = np.copy(image)
        for point in keypoints:
            x, y = int(point[0]), int(point[1])
            cv2.circle(output_image, (x, y), radius, color, -1)

        return output_image

    @staticmethod
    def scale_keypoints(hand3d_gt, camera_model, scale):
        root = hand3d_gt[:1]
        hand3d_gt_ = (hand3d_gt - root) * scale + root
        scaled_kpt2d = camera_model.eye_to_window(
            camera_model.world_to_eye(hand3d_gt_.cpu().numpy()))
        return scaled_kpt2d, hand3d_gt_

    def compute_score_loss(self, major_pred, major_gt, dist_pred, dist_gt,
                           score):
        major_score = score[:, :10]
        major_loss = torch.norm(major_pred - major_gt, dim=-1)
        major_score_loss = ((major_loss) * torch.exp(-1 * major_score) +
                            2 * major_score).mean()
        # pinch score loss
        pinch_score = score[:, 10:]
        pinch_dist_loss = torch.abs(dist_pred - dist_gt).unsqueeze_(-1)
        pinch_score_dist_loss = (
            (pinch_dist_loss) *
            torch.exp(-1 * pinch_score.mean(-1).unsqueeze_(-1)) +
            2 * pinch_score.mean(-1).unsqueeze_(-1)).mean()

        pinch_pred = torch.cat((major_pred[:, 3:4, :], major_pred[:, 7:8, :]),
                               dim=1)
        pinch_gt = torch.cat((major_gt[:, 3:4, :], major_gt[:, 7:8, :]), dim=1)
        pinch_mpjpe_loss = torch.norm(pinch_pred - pinch_gt, dim=-1)
        pinch_score_mpjpe_loss = (
            (pinch_mpjpe_loss) * torch.exp(-1 * pinch_score) +
            2 * pinch_score).mean()
        return (major_score_loss * 3, pinch_score_dist_loss * 4,
                pinch_score_mpjpe_loss * 4)

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
        norm_left_xyz = standard_left_xyz / standard_left_xyz[:, :, 2:]
        norm_right_xyz = standard_right_xyz / standard_right_xyz[:, :, 2:]
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
    def trans_3d_2_2d(hand3d_standard, virtual_baseline):
        B = hand3d_standard.shape[0]
        left_reproj = hand3d_standard[..., :2] / hand3d_standard[..., 2:]
        virtual_baseline = virtual_baseline.reshape(B, 1, 1).repeat(1, 21,
                                                                    1) * -1
        rightcam_XYZ = hand3d_standard.clone()
        rightcam_XYZ[..., :1] = hand3d_standard[..., :1] + virtual_baseline
        right_reproj = rightcam_XYZ[..., :2] / rightcam_XYZ[..., 2:]
        return left_reproj, right_reproj

    @staticmethod
    def keypoint_within_bounds(keypoint, image_width, image_height):
        x, y = keypoint[:, 0], keypoint[:, 1]
        mask = ((0 <= x) & (x < image_width)) & ((0 <= y) & (y < image_height))
        return mask

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
