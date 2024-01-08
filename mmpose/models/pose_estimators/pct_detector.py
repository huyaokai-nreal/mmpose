# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
# from mmpose.models.builder import POSENETS
# from mmpose.models.detectors.base import BasePose
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.models import builder
from mmpose.models.heads.nimble.simple_NIMBLELayer import sim_NIMBLELayer
from mmpose.registry import MODELS


@MODELS.register_module()
class PCT(BaseModel):
    """Detector of Pose Compositional Tokens. paper ref: Zigang Geng et al.
    "Human Pose as Compositional Tokens".

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
    """

    def __init__(self, keypoint_head=None, pretrained=None):
        super().__init__()
        self.stage_pct = keypoint_head['stage_pct']
        assert self.stage_pct in ['tokenizer', 'classifier']

        keypoint_head['loss_keypoint'] \
            = keypoint_head['tokenizer']['loss_keypoint']

        self.keypoint_head = builder.build_head(keypoint_head)
        self.init_weights_self(pretrained, keypoint_head['tokenizer']['ckpt'])
        self.nimble_layer = sim_NIMBLELayer(
            device='cuda',
            shape_ncomp=20,
            pose_ncomp=60,
            use_pose_pca=False,
            reg_shape_type=0)

    def init_weights_self(self, pretrained, tokenizer):
        """Weight initialization for model."""
        self.keypoint_head.init_weights()
        self.keypoint_head.tokenizer.init_weights_self(tokenizer)

    def forward(self, inputs, data_samples, mode: str = 'tensor', **kwargs):
        left_data_samples = data_samples[0]
        nimble_pose = []
        nimble_trans = []
        nimble_shape = []
        # is_left_hands = []
        for data_sample in left_data_samples:
            nimble_pose.append(data_sample.meta['nimble_pose'])
            nimble_trans.append(data_sample.meta['nimble_translation'])
            nimble_shape.append(data_sample.meta['nimble_shape'])
            # if data_sample.meta['category_id'] == 1:  # 1: left, 2: right
            #     is_left_hands.append(1)
            # else:
            #     is_left_hands.append(0)
        nimble_pose = torch.tensor(np.array(nimble_pose)).cuda().float()
        nimble_trans = torch.tensor(np.array(nimble_trans)).cuda().float()
        nimble_shape = torch.tensor(np.array(nimble_shape)).cuda().float()
        # left_hand = torch.tensor(np.array(is_left_hands)).cuda().float()
        zeros_nimble_shape = torch.zeros_like(nimble_shape).cuda().float()

        B = nimble_pose.shape[0]
        init_root_rot = torch.zeros((B, 1, 3),
                                    requires_grad=False).cuda().float()
        gt_rot_vector = torch.cat((init_root_rot, nimble_pose[:, 1:, :]),
                                  dim=1)

        _, bone_joints = self.nimble_layer.forward_simple(
            gt_rot_vector, zeros_nimble_shape)
        kp_index = [
            0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22,
            23, 24
        ]
        uesd_joints = bone_joints[:, kp_index, :]
        rebuild_joints = (uesd_joints - uesd_joints[:, 0:1, :]) / 1000

        if mode == 'loss' or self.stage_pct == 'tokenizer':
            joints = rebuild_joints[:, 1:, :]
        else:
            # Just a placeholder during inference of PCT
            joints = None

        if mode == 'loss':
            return self.forward_train(joints, **kwargs)
        return self.forward_test(joints, data_samples, **kwargs)

    def forward_train(self, joints, **kwargs):
        """Defines the computation performed at every call when training."""

        output = None
        extra_output = None

        p_logits, p_joints, g_logits, e_latent_loss = \
            self.keypoint_head(output, extra_output, joints)

        # if return loss
        losses = dict()
        keypoint_losses = \
            self.keypoint_head.tokenizer.get_loss(
                p_joints, joints, e_latent_loss)
        losses.update(keypoint_losses)

        return losses

    def get_class_accuracy(self, output, target, topk):

        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [(correct[:k].reshape(-1).float().sum(0) * 100. / batch_size)
                for k in topk]

    def forward_test(self, joints, data_samples, **kwargs):
        """Defines the computation performed at every call when testing."""

        output = None
        extra_output = None

        p_joints, encoding_scores = \
            self.keypoint_head(output, extra_output, joints, train=False)

        B = p_joints.shape[0]
        init_root_trans = torch.zeros((B, 1, 3),
                                      requires_grad=False).cuda().float()
        rebuild_p_joints = torch.cat((init_root_trans, p_joints), dim=1)
        rebuild_joints = torch.cat((init_root_trans, joints), dim=1)

        batch_pred_instances = []
        for b in range(p_joints.shape[0]):
            batch_pred_instances.append(
                InstanceData(
                    keypoints3d=rebuild_p_joints[b:b + 1, ...],
                    keypoints3d_scores=torch.ones((1, 21)),
                    keypoints3d_gt=rebuild_joints[b:b + 1, ...],
                    keypoints=np.ones((1, 21, 2)),
                    keypoint_scores=np.ones((1, 21)),
                ))
        batch_data_samples = data_samples[0]
        assert len(batch_pred_instances) == len(batch_data_samples)
        for pred_instances, data_sample in zip(batch_pred_instances,
                                               batch_data_samples):
            pred_instances.keypoints3d = pred_instances.keypoints3d.cpu(
            ).numpy()
            pred_instances.keypoints3d_scores = np.ones(
                (1, pred_instances.keypoints3d.shape[1]))
            data_sample.pred_instances = pred_instances

            data_sample.gt_instances.keypoints3d = \
                pred_instances.keypoints3d_gt.cpu().numpy()
            data_sample.gt_instances.keypoints3d_scores = np.ones(
                (1, data_sample.gt_instances.keypoints3d.shape[1]))

        return batch_data_samples

    def show_result(self):
        # Not implemented
        return None
