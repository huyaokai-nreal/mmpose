# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCCToKeypoint(nn.Module):

    def __init__(self, feat_w: int = 256, feat_h: int = 256) -> None:
        super().__init__()
        linspace_x = torch.arange(0.0, 1.0 * feat_w, 1.0) / feat_w
        linspace_y = torch.arange(0.0, 1.0 * feat_h, 1.0) / feat_h
        self.linspace_x = nn.Parameter(linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(linspace_y, requires_grad=False)

    @staticmethod
    def _flat_softmax(featmaps):
        """Use Softmax to normalize the featmaps in depthwise."""
        heatmaps = F.softmax(featmaps, dim=2)
        return heatmaps

    def forward(self, x, y):
        x = self._flat_softmax(x)
        y = self._flat_softmax(y)
        pred_x = x.mul(self.linspace_x)
        pred_y = y.mul(self.linspace_y)
        pred_x = pred_x.sum(dim=-1, keepdim=True)
        pred_y = pred_y.sum(dim=-1, keepdim=True)
        return pred_x, pred_y


model = SimCCToKeypoint()
