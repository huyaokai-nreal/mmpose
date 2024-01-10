# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModule, constant_init, normal_init

from mmpose.models.heads.nimble.modules import BasicBlock
from mmpose.registry import MODELS
from .pct_tokenizer import PCT_Tokenizer


@MODELS.register_module()
class PCT_Head(BaseModule):

    def __init__(
        self,
        stage_pct,
        in_channels,
        num_joints,
        cls_head=None,
        tokenizer=None,
        loss_keypoint=None,
    ):
        super().__init__()

        self.stage_pct = stage_pct

        self.conv_channels = cls_head['conv_channels']
        self.hidden_dim = cls_head['hidden_dim']

        self.num_blocks = cls_head['num_blocks']
        self.hidden_inter_dim = cls_head['hidden_inter_dim']
        self.token_inter_dim = cls_head['token_inter_dim']
        self.dropout = cls_head['dropout']

        self.token_num = tokenizer['codebook']['token_num']
        self.token_class_num = tokenizer['codebook']['token_class_num']

        self.tokenizer = PCT_Tokenizer(
            stage_pct=stage_pct, tokenizer=tokenizer, num_joints=num_joints)

        # self.loss = build_loss(loss_keypoint)

    # def get_loss(self, p_logits, p_joints, g_logits, joints):
    #     """Calculate loss for training classifier.

    #     Note:
    #         batch_size: N
    #         num_keypoints: K
    #         num_token: M
    #         num_token_class: V

    #     Args:
    #         p_logits (torch.Tensor[NxMxV]): Predicted class logits.
    #         p_joints(torch.Tensor[NxKx3]): Predicted joints
    #             recovered from the predicted class.
    #         g_logits(torch.Tensor[NxM]): Groundtruth class labels
    #             calculated by the well-trained tokenizer encoder
    #             and groundtruth joints.
    #         joints(torch.Tensor[NxKx3]): Groundtruth joints.
    #     """

    #     losses = dict()

    #     losses['token_loss'], losses['kpt_loss'] = self.loss(
    #         p_logits, p_joints, g_logits, joints)

    #     unused_losses = []
    #     for name, loss in losses.items():
    #         if loss is None:
    #             unused_losses.append(name)
    #     for unused_loss in unused_losses:
    #         losses.pop(unused_loss)

    #     return losses

    def forward(self,
                x,
                extra_x,
                joints=None,
                cls_logits=None,
                cls_logits_softmax=None,
                train=True):
        """Forward function."""

        encoding_scores = None
        joints_feat = None

        output_joints, cls_label, e_latent_loss = \
            self.tokenizer(joints,
                           joints_feat,
                           cls_logits_softmax,
                           train=train)

        if train:
            return cls_logits, output_joints, cls_label, e_latent_loss
        else:
            return output_joints, encoding_scores

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_cls_head(self, layer_config):
        feature_convs = []
        feature_conv = self._make_layer(
            BasicBlock,
            layer_config['conv_channels'],
            layer_config['conv_channels'],
            layer_config['conv_num_blocks'],
            dilation=layer_config['dilation'])
        feature_convs.append(feature_conv)

        return nn.ModuleList(feature_convs)

    def _make_layer(self,
                    block,
                    inplanes,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(
            block(inplanes, planes, stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
