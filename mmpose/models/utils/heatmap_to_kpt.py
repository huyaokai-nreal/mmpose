import torch.nn.functional as F
import torch
import torch.nn as nn


class HeatmapToKeypoint(nn.Module):

    def __init__(self, feat_w: int = 32, feat_h: int = 32) -> None:
        super().__init__()
        linspace_x = torch.arange(0.0, 1.0 * feat_w, 1.0) / feat_w
        linspace_y = torch.arange(0.0, 1.0 * feat_h, 1.0) / feat_h
        self.linspace_x = nn.Parameter(linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(linspace_y, requires_grad=False)

    @staticmethod
    def _flat_softmax(featmaps):
        """Use Softmax to normalize the featmaps in depthwise."""
        _, N, H, W = featmaps.shape
        featmaps = featmaps.reshape(-1, N, H * W)
        heatmaps = F.softmax(featmaps, dim=2)
        return heatmaps.reshape(-1, N, H, W)

    def forward(self, feat):
        heatmaps = self._flat_softmax(feat)
        x_fea = heatmaps.sum(dim=2)
        y_fea = heatmaps.sum(dim=3)
        pred_x = x_fea.mul(self.linspace_x)
        pred_y = y_fea.mul(self.linspace_y)
        pred_x = pred_x.sum(dim=-1, keepdim=True)
        pred_y = pred_y.sum(dim=-1, keepdim=True)
        kpt = torch.cat([pred_x, pred_y], dim=-1)
        return kpt


model = HeatmapToKeypoint()
