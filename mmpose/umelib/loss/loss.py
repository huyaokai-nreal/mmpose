# flake8: noqa
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
# from lib.common.hand import scaled_hand_model
from lib.common.hand_skinning import skin_landmarks


class Loss_Pose(nn.Module):

    def __init__(self, lambda_theta=0.05, lambda_w=0.5):
        super(Loss_Pose, self).__init__()
        self.criterion = nn.L1Loss()
        self.lambda_theta = lambda_theta
        self.lambda_w = lambda_w

        self.lambda_kpt = 21

    def forward(self,
                unknown_output,
                known_output,
                generic_hand_model,
                gt_hand_model,
                target,
                mask=None):

        gt_target = target.gt_skel_targets
        preds_target = target.preds_targets
        gt_keypoints = skin_landmarks(gt_hand_model, gt_target.joint_angles,
                                      gt_target.wrist_xfs)
        unknown_gt_keypoints = gt_keypoints
        # bs = gt_keypoints.shape[0]
        # preds_keypoints = skin_landmarks(
        #     generic_hand_model,
        #     preds_target.joint_angles,
        #     preds_target.wrist_xfs
        # )

        known_keypoints = skin_landmarks(
            gt_hand_model,
            known_output.joint_angles,
            known_output.wrist_xfs,
        )
        unknown_keypoints = skin_landmarks(
            generic_hand_model,
            unknown_output.joint_angles,
            unknown_output.wrist_xfs,
        )
        unknown_xfs = unknown_output.wrist_xfs[:, :, :, -1]
        unknown_gt_xfs = preds_target.wrist_xfs[:, :, :, -1]

        unknown_angles = unknown_output.joint_angles
        unknown_gt_angles = preds_target.joint_angles
        if mask is not None:
            mask_kpt = mask.unsqueeze(2).unsqueeze(2)
            mask_xfs = mask.unsqueeze(2)
            unknown_keypoints = unknown_keypoints * mask_kpt
            unknown_gt_keypoints = unknown_gt_keypoints * mask_kpt

            unknown_xfs = unknown_xfs * mask_xfs
            unknown_gt_xfs = unknown_gt_xfs * mask_xfs

            unknown_angles = unknown_angles * mask_xfs
            unknown_gt_angles = unknown_gt_angles * mask_xfs

        loss_keypoints = self.criterion(known_keypoints, gt_keypoints) + \
            self.criterion(unknown_keypoints, unknown_gt_keypoints)

        loss_xfs = self.criterion(known_output.wrist_xfs[:, :, :, -1],
                                  gt_target.wrist_xfs[:, :, :, -1]) + \
            self.criterion(unknown_xfs, unknown_gt_xfs)
        loss_angles = self.criterion(known_output.joint_angles,
                                     gt_target.joint_angles) + \
            self.criterion(unknown_angles, unknown_gt_angles)
        # loss = self.lambda_theta*loss_xfs+self.lambda_w*loss_angles
        # loss_keypoints *= self.lambda_kpt
        loss = loss_keypoints + self.lambda_theta * loss_angles + \
            self.lambda_w * loss_xfs
        # loss = self.criterion(preds_keypoints,gt_keypoints)
        return loss


class Loss_Pose_xv(nn.Module):

    def __init__(self, lambda_theta=0.05, lambda_w=0.5):
        super(Loss_Pose_xv, self).__init__()
        self.criterion = nn.L1Loss()
        self.lambda_theta = lambda_theta
        self.lambda_w = lambda_w
        self.lambda_kpt = 21

        self.mv = 1
        self.sv = 1

    def forward(self, unknown_output, known_output_mv, known_output_l,
                known_output_r, generic_hand_model, gt_hand_model, target):

        gt_target = target.gt_skel_targets
        preds_target = target.preds_targets
        gt_keypoints = skin_landmarks(gt_hand_model, gt_target.joint_angles,
                                      gt_target.wrist_xfs)
        unknown_gt_keypoints = gt_keypoints
        # bs = gt_keypoints.shape[0]
        # preds_keypoints = skin_landmarks(
        #     generic_hand_model,
        #     preds_target.joint_angles,
        #     preds_target.wrist_xfs
        # )
        # unknown_gt_keypoints = preds_keypoints

        known_keypoints_mv = skin_landmarks(
            gt_hand_model,
            known_output_mv.joint_angles,
            known_output_mv.wrist_xfs,
        )
        known_keypoints_l = skin_landmarks(
            gt_hand_model,
            known_output_l.joint_angles,
            known_output_l.wrist_xfs,
        )
        known_keypoints_r = skin_landmarks(
            gt_hand_model,
            known_output_r.joint_angles,
            known_output_r.wrist_xfs,
        )

        unknown_keypoints = skin_landmarks(
            generic_hand_model,
            unknown_output.joint_angles,
            unknown_output.wrist_xfs,
        )
        unknown_xfs = unknown_output.wrist_xfs[:, :, :, -1]
        unknown_gt_xfs = preds_target.wrist_xfs[:, :, :, -1]

        unknown_angles = unknown_output.joint_angles
        unknown_gt_angles = preds_target.joint_angles

        loss_keypoints = (
            self.mv * self.criterion(known_keypoints_mv, gt_keypoints) +
            self.sv * self.criterion(known_keypoints_l, gt_keypoints) +
            self.sv * self.criterion(known_keypoints_r, gt_keypoints) +
            self.criterion(unknown_keypoints, unknown_gt_keypoints))

        loss_xfs = self.mv*self.criterion(known_output_mv.wrist_xfs[:, :, :, -1],
                                  gt_target.wrist_xfs[:, :, :, -1]) + \
                    self.sv*self.criterion(known_output_l.wrist_xfs[:, :, :, -1],
                                  gt_target.wrist_xfs[:, :, :, -1]) + \
                    self.sv*self.criterion(known_output_r.wrist_xfs[:, :, :, -1],
                                  gt_target.wrist_xfs[:, :, :, -1]) + \
                    self.criterion(unknown_xfs,unknown_gt_xfs)
        loss_angles = self.mv*self.criterion(known_output_mv.joint_angles,
                                     gt_target.joint_angles) + \
                    self.sv*self.criterion(known_output_l.joint_angles,
                                     gt_target.joint_angles) + \
                    self.sv*self.criterion(known_output_r.joint_angles,
                                     gt_target.joint_angles) + \
                    self.criterion(unknown_angles,unknown_gt_angles)

        loss = self.lambda_kpt * loss_keypoints + self.lambda_theta * loss_angles + self.lambda_w * loss_xfs

        return loss


class Loss_Temp_xv(nn.Module):

    def __init__(self, lambda_t=0.05):
        super(Loss_Temp_xv, self).__init__()
        self.lambda_t = lambda_t
        self.mv = 1
        self.sv = 1

    def forward(self, unknown_output, known_output_mv, known_output_l,
                known_output_r):

        def _compute_accelerations(pts):
            acc = pts[:, 0:-2] + pts[:, 2:] - 2 * pts[:, 1:-1]
            return torch.norm(acc, p=1, dim=-1).mean()

        unknown_xfs = unknown_output.wrist_xfs[:, :, :, -1]
        unknown_angles = unknown_output.joint_angles
        loss_xfs = self.mv*_compute_accelerations(known_output_mv.wrist_xfs[:,:,:,-1]) + \
                    self.sv*_compute_accelerations(known_output_l.wrist_xfs[:,:,:,-1]) + \
                    self.sv*_compute_accelerations(known_output_r.wrist_xfs[:,:,:,-1]) + \
                    _compute_accelerations(unknown_xfs)
        loss_angles = self.mv*_compute_accelerations(known_output_mv.joint_angles) + \
                    self.sv*_compute_accelerations(known_output_l.joint_angles) + \
                    self.sv*_compute_accelerations(known_output_r.joint_angles) + \
                    _compute_accelerations(unknown_angles)

        loss = self.lambda_t * (loss_angles + loss_xfs)

        return loss


class Loss_Pose_xv_fine_tune(nn.Module):

    def __init__(self, lambda_theta=0.05, lambda_w=0.5):
        super(Loss_Pose_xv_fine_tune, self).__init__()
        self.criterion = nn.L1Loss()
        self.lambda_theta = lambda_theta
        self.lambda_w = lambda_w
        self.lambda_kpt = 21

        self.mv = 1
        self.sv = 1

    def forward(self, known_output_l, known_output_r, gt_hand_model, target):
        gt_target = target.gt_skel_targets

        gt_keypoints = skin_landmarks(gt_hand_model, gt_target.joint_angles,
                                      gt_target.wrist_xfs)
        known_keypoints_l = skin_landmarks(
            gt_hand_model,
            known_output_l.joint_angles,
            known_output_l.wrist_xfs,
        )
        known_keypoints_r = skin_landmarks(
            gt_hand_model,
            known_output_r.joint_angles,
            known_output_r.wrist_xfs,
        )

        loss_keypoints = self.sv*self.criterion(known_keypoints_l,gt_keypoints) + \
                        self.sv*self.criterion(known_keypoints_r,gt_keypoints)
        loss_xfs = self.sv*self.criterion(known_output_l.wrist_xfs[:,:,:,-1],
                                  gt_target.wrist_xfs[:,:,:,-1]) + \
                    self.sv*self.criterion(known_output_r.wrist_xfs[:,:,:,-1],
                                  gt_target.wrist_xfs[:,:,:,-1])
        loss_angles = self.sv*self.criterion(known_output_l.joint_angles,
                                     gt_target.joint_angles) + \
                    self.sv*self.criterion(known_output_r.joint_angles,
                                     gt_target.joint_angles)

        loss = self.lambda_kpt * loss_keypoints + self.lambda_theta * loss_angles + self.lambda_w * loss_xfs

        return loss
