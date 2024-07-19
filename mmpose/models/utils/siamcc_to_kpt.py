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


class SimCCToKeypoint3D(nn.Module):

    def __init__(self,
                 feat_w: int = 256,
                 feat_h: int = 256,
                 feat_d: int = 256,
                 map_type='softmax') -> None:
        super().__init__()
        self.map_type = map_type
        linspace_x = torch.arange(0.0, 1.0 * feat_w, 1.0) / feat_w
        linspace_y = torch.arange(0.0, 1.0 * feat_h, 1.0) / feat_h
        linspace_z = torch.arange(0.0, 1.0 * feat_d, 1.0) / feat_d
        self.linspace_x = nn.Parameter(linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(linspace_y, requires_grad=False)
        self.linspace_z = nn.Parameter(linspace_z, requires_grad=False)

    @staticmethod
    def _flat_softmax(featmaps):
        """Use Softmax to normalize the featmaps in depthwise."""
        heatmaps = F.softmax(featmaps, dim=-1)
        return heatmaps

    @staticmethod
    def _elu_normalize(featmaps):
        heatmaps = F.elu(featmaps) + 1
        heatmaps = heatmaps / heatmaps.sum(dim=-1, keepdim=True)
        return heatmaps

    def forward(self, x, y, z):
        if self.map_type == 'softmax':
            x = self._flat_softmax(x)
            y = self._flat_softmax(y)
            z = self._flat_softmax(z)
        elif self.map_type == 'elu':
            x = self._elu_normalize(x)
            y = self._elu_normalize(y)
            z = self._elu_normalize(z)
        elif self.map_type == 'raw':
            pass
        pred_x = x.mul(self.linspace_x)
        pred_y = y.mul(self.linspace_y)
        pred_z = z.mul(self.linspace_z)
        pred_x = pred_x.sum(dim=-1, keepdim=True)
        pred_y = pred_y.sum(dim=-1, keepdim=True)
        pred_z = pred_z.sum(dim=-1, keepdim=True)
        return pred_x, pred_y, pred_z


model = SimCCToKeypoint()
