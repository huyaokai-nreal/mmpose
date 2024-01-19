# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import constant_init, normal_init

from mmpose.models.utils.gmlp import gMLP


class FCBlock(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ff(x)


class MLPBlock(nn.Module):

    def __init__(self, dim, inter_dim, dropout_ratio):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, inter_dim), nn.GELU(), nn.Dropout(dropout_ratio),
            nn.Linear(inter_dim, dim), nn.Dropout(dropout_ratio))

    def forward(self, x):
        return self.ff(x)


class MixerLayer(nn.Module):

    def __init__(self, hidden_dim, hidden_inter_dim, token_dim,
                 token_inter_dim, dropout_ratio):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.MLP_token = MLPBlock(token_dim, token_inter_dim, dropout_ratio)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.MLP_channel = MLPBlock(hidden_dim, hidden_inter_dim,
                                    dropout_ratio)

    def forward(self, x):
        y = self.layernorm1(x)
        y = y.transpose(2, 1)
        y = self.MLP_token(y)
        y = y.transpose(2, 1)
        z = self.layernorm2(x + y)
        z = self.MLP_channel(z)
        out = x + y + z
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GMLPModel(nn.Module):

    def __init__(self, input_size, hidden_dim=512, output_size=2048):
        super(GMLPModel, self).__init__()
        self.first_layer = nn.Linear(input_size, hidden_dim // 2)
        self.first_relu = nn.ReLU()
        self.second_layer = nn.Linear(hidden_dim // 2, hidden_dim)
        self.second_relu = nn.ReLU()
        self.liftnet = gMLP(
            d_model=hidden_dim, d_ffn=hidden_dim * 2, num_layers=5)
        self.last_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SyncBatchNorm(hidden_dim), nn.ReLU(),
            nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=1),
            nn.SyncBatchNorm(2 * hidden_dim), nn.ReLU(),
            nn.Conv2d(2 * hidden_dim, output_size, kernel_size=1))

    def forward(self, x):
        x = x.view(x.shape[0] * x.shape[1], -1)
        x = self.first_layer(x)
        x = self.first_relu(x)
        x = self.second_layer(x)
        x = self.second_relu(x).unsqueeze(-1).unsqueeze(-1)

        x = self.liftnet(x)
        x = self.last_layer(x)
        return x


class GMLPModel_Large(nn.Module):

    def __init__(self, input_size, hidden_dim=2048, output_size=2048):
        super(GMLPModel_Large, self).__init__()
        self.first_layer = nn.Linear(input_size, hidden_dim // 2)
        self.first_relu = nn.ReLU()
        self.second_layer = nn.Linear(hidden_dim // 2, hidden_dim)
        self.second_relu = nn.ReLU()
        self.liftnet = gMLP(
            d_model=hidden_dim, d_ffn=hidden_dim * 4, num_layers=5)
        self.last_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SyncBatchNorm(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_size, kernel_size=1),
        )

    def forward(self, x):
        x = x.view(x.shape[0] * x.shape[1], -1)
        x = self.first_layer(x)
        x = self.first_relu(x)
        x = self.second_layer(x)
        x = self.second_relu(x).unsqueeze(-1).unsqueeze(-1)

        x = self.liftnet(x)
        x = self.last_layer(x)
        return x


class IMAGE_CLS_Model(nn.Module):

    def __init__(self, image_size, in_channels, cls_head, tokenizer):
        super(IMAGE_CLS_Model, self).__init__()
        self.image_size = image_size
        self.conv_channels = cls_head['conv_channels']
        self.hidden_dim = cls_head['hidden_dim']
        self.num_blocks = cls_head['num_blocks']
        self.hidden_inter_dim = cls_head['hidden_inter_dim']
        self.token_inter_dim = cls_head['token_inter_dim']
        self.dropout = cls_head['dropout']
        self.token_num = tokenizer['codebook']['token_num']
        self.token_class_num = tokenizer['codebook']['token_class_num']

        self.conv_trans = self._make_transition_for_head(
            in_channels, self.conv_channels)
        self.conv_head = self._make_cls_head(cls_head)

        input_size = (image_size[0] // 32) * (image_size[1] // 32)
        self.mixer_trans = FCBlock(self.conv_channels * input_size,
                                   self.token_num * self.hidden_dim)

        self.mixer_head = nn.ModuleList([
            MixerLayer(self.hidden_dim, self.hidden_inter_dim, self.token_num,
                       self.token_inter_dim, self.dropout)
            for _ in range(self.num_blocks)
        ])
        self.mixer_norm_layer = FCBlock(self.hidden_dim, self.hidden_dim)

        self.cls_pred_layer = nn.Linear(self.hidden_dim, self.token_class_num)

    def forward(self, x):
        batch_size = x[-1].shape[0]
        cls_feat = self.conv_head[0](self.conv_trans(x[-1]))

        cls_feat = cls_feat.flatten(2).transpose(2, 1).flatten(1)
        cls_feat = self.mixer_trans(cls_feat)
        cls_feat = cls_feat.reshape(batch_size, self.token_num, -1)

        for mixer_layer in self.mixer_head:
            cls_feat = mixer_layer(cls_feat)
        cls_feat = self.mixer_norm_layer(cls_feat)

        cls_logits = self.cls_pred_layer(cls_feat)

        # encoding_scores = cls_logits.topk(1, dim=2)[0]
        cls_logits = cls_logits.flatten(0, 1)
        return cls_logits

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

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
