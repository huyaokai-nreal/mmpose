# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModule, constant_init, normal_init

# from mmpose.models.heads.nimble.modules import BasicBlock
from mmpose.registry import MODELS
from .pct_tokenizer import PCT_Tokenizer


@MODELS.register_module()
class PCT_Head(BaseModule):

    def __init__(
        self,
        stage_pct,
        in_channels,
        image_size,
        num_joints,
        cls_head,
        tokenizer=None,
    ):
        super().__init__()

        self.stage_pct = stage_pct
        self.tokenizer = PCT_Tokenizer(
            stage_pct=stage_pct, tokenizer=tokenizer, num_joints=num_joints)

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

    def init_weights(self):
        if self.stage_pct == 'classifier':
            self.tokenizer.eval()
            for name, params in self.tokenizer.named_parameters():
                params.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
