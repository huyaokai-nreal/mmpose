# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn


# GMLP CGU 2d_conv
class ChannelGatingUnit(nn.Module):

    def __init__(self, d_ffn):
        super().__init__()
        # self.norm = nn.LayerNorm([d_ffn, 1, 1])
        self.norm = nn.BatchNorm2d(d_ffn)
        self.channel_proj = nn.Conv2d(d_ffn, d_ffn, kernel_size=1)
        nn.init.constant_(self.channel_proj.bias, 1.0)
        self.d_ffn = d_ffn

    def forward(self, x):
        u, v = torch.split(x, [self.d_ffn, self.d_ffn], dim=1)
        v = self.norm(v)
        v = self.channel_proj(v)
        out = u * v
        return out


class gMLPBlock(nn.Module):

    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.norm = nn.BatchNorm2d(d_model)
        # self.norm = nn.BatchNorm2d(d_model)
        self.channel_proj1 = nn.Conv2d(d_model, d_ffn * 2, kernel_size=1)
        self.channel_proj2 = nn.Conv2d(d_ffn, d_model, kernel_size=1)
        self.cgu = ChannelGatingUnit(d_ffn)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        # x = F.gelu(self.channel_proj1(x))
        x = F.relu(self.channel_proj1(x))
        x = self.cgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out

class gMLP(nn.Module):

    def __init__(self, d_model=128, d_ffn=256, num_layers=6):
        super().__init__()
        self.model_gmlp = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn) for _ in range(num_layers)])

    def forward(self, x):
        return self.model_gmlp(x)
