# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks import build_activation_layer
from mmengine.model.weight_init import trunc_normal_

from mmpose.registry import MODELS
from .utils.se_layer import SELayer


def _make_divisible(v, divisor, min_value=None):
    """This function is taken from the original tf repo. It ensures that all
    layers have a channel number that is divisible by 8 It can be seen here: ht
    tps://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet
    .

    /mobilenet.py.

    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def repvit_model_convert(model):
    for child_name, child in model.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(model, child_name, fused)
            repvit_model_convert(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(model, child_name, torch.nn.Identity())
        else:
            repvit_model_convert(child)


class Conv2d_BN(torch.nn.Sequential):

    def __init__(self,
                 a,
                 b,
                 ks=1,
                 stride=1,
                 pad=0,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1,
                 resolution=-10000):
        super().__init__()
        self.add_module(
            'c',
            torch.nn.Conv2d(
                a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):

    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(
                x.size(0), 1, 1, 1, device=x.device).ge_(
                    self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):

    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(
            torch.ones(
                conv1_w.shape[0],
                conv1_w.shape[1],
                1,
                1,
                device=conv1_w.device), [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class RepViTBlock(nn.Module):
    """BasicBloc for  RepViT: Revisiting Mobile CNN From ViT Perspective.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se,
                 use_hs, act_cfg):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(
                    inp,
                    inp,
                    kernel_size,
                    stride, (kernel_size - 1) // 2,
                    groups=inp),
                SELayer(inp, 4) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0))
            self.channel_mixer = Residual(
                nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    build_activation_layer(act_cfg),
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SELayer(inp, 4) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    build_activation_layer(act_cfg),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class BN_Linear(torch.nn.Sequential):

    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, linear = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = linear.weight * w[None, :]
        if linear.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (linear.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=linear.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


@MODELS.register_module()
class RepViT(nn.Module):
    arch_zoo = dict(
        m0=[
            # k, t, c, SE, HS, s
            [3, 2, 32, 1, 0, 1],
            [3, 2, 32, 0, 0, 1],
            [3, 2, 32, 0, 0, 1],
            [3, 2, 64, 0, 0, 2],
            [3, 2, 64, 1, 0, 1],
            [3, 2, 64, 0, 0, 1],
            [3, 2, 128, 0, 1, 2],
            [3, 2, 128, 1, 1, 1],
            [3, 2, 128, 0, 1, 1],
            [3, 2, 128, 1, 1, 1],
            [3, 2, 128, 0, 1, 1],
            [3, 2, 128, 1, 1, 1],
            [3, 2, 384, 0, 1, 2],
            [3, 2, 384, 1, 1, 1],
            [3, 2, 384, 0, 1, 1]
        ],
        m1=[
            # k, t, c, SE, HS, s
            [3, 2, 48, 1, 0, 1],
            [3, 2, 48, 0, 0, 1],
            [3, 2, 48, 0, 0, 1],
            [3, 2, 96, 0, 0, 2],
            [3, 2, 96, 1, 0, 1],
            [3, 2, 96, 0, 0, 1],
            [3, 2, 96, 0, 0, 1],
            [3, 2, 192, 0, 1, 2],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 384, 0, 1, 2],
            [3, 2, 384, 1, 1, 1],
            [3, 2, 384, 0, 1, 1]
        ],
        m2=[
            # k, t, c, SE, HS, s
            [3, 2, 64, 1, 0, 1],
            [3, 2, 64, 0, 0, 1],
            [3, 2, 64, 0, 0, 1],
            [3, 2, 128, 0, 0, 2],
            [3, 2, 128, 1, 0, 1],
            [3, 2, 128, 0, 0, 1],
            [3, 2, 128, 0, 0, 1],
            [3, 2, 256, 0, 1, 2],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 512, 0, 1, 2],
            [3, 2, 512, 1, 1, 1],
            [3, 2, 512, 0, 1, 1]
        ],
        m3=[
            # k, t, c, SE, HS, s
            [3, 2, 64, 1, 0, 1],
            [3, 2, 64, 0, 0, 1],
            [3, 2, 64, 1, 0, 1],
            [3, 2, 64, 0, 0, 1],
            [3, 2, 64, 0, 0, 1],
            [3, 2, 128, 0, 0, 2],
            [3, 2, 128, 1, 0, 1],
            [3, 2, 128, 0, 0, 1],
            [3, 2, 128, 1, 0, 1],
            [3, 2, 128, 0, 0, 1],
            [3, 2, 128, 0, 0, 1],
            [3, 2, 256, 0, 1, 2],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 512, 0, 1, 2],
            [3, 2, 512, 1, 1, 1],
            [3, 2, 512, 0, 1, 1]
        ])

    def __init__(self,
                 arch,
                 image_channel=1,
                 act_cfg=dict(type='GELU'),
                 out_indices=(3, )):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        assert arch in self.arch_zoo, f'arch {arch} is not supported'
        self.cfgs = self.arch_zoo[arch]
        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(
            Conv2d_BN(image_channel, input_channel // 2, 3, 2, 1),
            build_activation_layer(act_cfg),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        self.output_index = []
        stage_out_index = []
        for i, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            if s == 2:
                stage_out_index.append(i)
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(
                RepViTBlock(
                    input_channel,
                    exp_size,
                    output_channel,
                    k,
                    s,
                    use_se,
                    use_hs,
                    act_cfg=act_cfg))
            input_channel = output_channel
        stage_out_index.append(len(self.cfgs))
        for i, idx in enumerate(stage_out_index):
            if i in out_indices:
                self.output_index.append(idx)
        self.features = nn.ModuleList(layers)

    def forward(self, x):
        outputs = []
        for i, f in enumerate(self.features):
            x = f(x)
            if i in self.output_index:
                outputs.append(x)
        return outputs

    def switch_to_deploy(self):
        print('convert repvit to deploy mode')
        repvit_model_convert(self)
