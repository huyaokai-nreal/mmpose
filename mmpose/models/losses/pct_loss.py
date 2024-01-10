# Copyright (c) OpenMMLab. All rights reserved.
# --------------------------------------------------------
# Pose Compositional Tokens
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import torch
import torch.nn as nn

from mmpose.registry import MODELS


@MODELS.register_module()
class JointS1Loss(nn.Module):

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def smooth_l1_loss(self, pred, gt):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < self.beta
        loss = torch.where(cond, 0.5 * l1_loss**2 / self.beta,
                           l1_loss - 0.5 * self.beta)
        return loss

    def forward(self, pred, gt):

        # joint_dim = gt.shape[2] - 1
        # visible = gt[..., joint_dim:]
        # pred, gt = pred[..., :joint_dim], gt[..., :joint_dim]

        loss = self.smooth_l1_loss(pred, gt)
        loss = loss.mean(dim=2).mean(dim=1).mean(dim=0)

        return loss


@MODELS.register_module()
class Tokenizer_loss(nn.Module):

    def __init__(self, joint_loss_w, e_loss_w, beta=0.05):
        super().__init__()

        self.joint_loss = JointS1Loss(beta)
        self.joint_loss_w = joint_loss_w

        self.e_loss_w = e_loss_w

    def forward(self, output_joints, joints, e_latent_loss):

        losses = []
        joint_loss = self.joint_loss(output_joints, joints)
        joint_loss *= self.joint_loss_w
        losses.append(joint_loss)

        e_latent_loss *= self.e_loss_w
        losses.append(e_latent_loss)

        return losses


@MODELS.register_module()
class Classifer_loss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super().__init__()

        self.token_loss = nn.CrossEntropyLoss()
        self.token_loss_w = loss_weight

    def forward(self, p_logits, g_logits, target_weight=None):
        token_loss = self.token_loss(p_logits, g_logits)
        token_loss *= self.token_loss_w

        return token_loss
