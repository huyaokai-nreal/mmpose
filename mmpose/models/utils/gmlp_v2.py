# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import einsum, nn
from torch.nn import functional as F


class SpatialGatingUnit(nn.Module):

    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.d_ffn = d_ffn
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x, att=None):
        u, v = torch.split(x, self.d_ffn, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        if att is not None:
            v = v + att
        out = u * v
        return out


class TinyAttention(nn.Module):

    def __init__(self, d_in, d_out, d_att) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_att * 3)
        self.out_proj = nn.Linear(d_att, d_out)
        self.scale = d_att**(-0.5)

    def forward(self, x):
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.out_proj(out)


class gMLPBlock(nn.Module):

    def __init__(self, d_model, d_ffn, seq_len, with_att):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.with_att = with_att
        if with_att:
            self.att = TinyAttention(d_model, d_ffn, 64)

    def forward(self, x):
        if self.with_att:
            att = self.att(x)
        residual = x
        x = self.norm(x)
        x = F.relu(self.channel_proj1(x))
        if self.with_att:
            x = self.sgu(x, att)
        else:
            x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class gMLP(nn.Module):

    def __init__(self,
                 d_model=256,
                 d_ffn=512,
                 seq_len=256,
                 num_layers=6,
                 with_att=False):
        super().__init__()
        self.model = nn.Sequential(*[
            gMLPBlock(d_model, d_ffn, seq_len, with_att)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.proj(x)
        return self.model(x)


class gMLPForPose(gMLP):

    def __init__(self,
                 in_channels=5,
                 d_model=256,
                 d_ffn=512,
                 seq_len=256,
                 num_layers=6,
                 d_output=8,
                 with_att=False):
        super().__init__(d_model, d_ffn, seq_len, num_layers, with_att)
        self.proj = nn.Linear(in_channels, d_model)
        self.proj_out = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = self.proj(x)
        return self.proj_out(self.model(x))


class gMLPForLanguageModeling(gMLP):

    def __init__(self,
                 num_tokens=10000,
                 d_model=256,
                 d_ffn=512,
                 seq_len=256,
                 num_layers=6):
        super().__init__(d_model, d_ffn, seq_len, num_layers)
        self.embed = nn.Embedding(num_tokens, d_model)

    def forward(self, x):
        embedding = self.embed(x)
        out = self.model(embedding)
        return out


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, '`image_size` must be divisibe by `patch_size`'
    num_patches = sqrt_num_patches**2
    return num_patches


class gMLPForImageClassification(gMLP):

    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=256,
        d_ffn=512,
        seq_len=256,
        num_layers=6,
    ):
        super().__init__(d_model, d_ffn, seq_len, num_layers)
        self.patcher = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = embedding.mean(dim=1)
        out = self.classifier(embedding)
        return out
