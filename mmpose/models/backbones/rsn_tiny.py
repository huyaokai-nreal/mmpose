import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS
from .utils.repvgg import RepVGGBlock


class conv_bn_relu(nn.Module):

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        padding,
        has_bn=True,
        has_relu=True,
    ):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        def _func_factory(conv, bn, relu, has_bn, has_relu):

            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x

            return func

        func = _func_factory(self.conv, self.bn, self.relu, self.has_bn,
                             self.has_relu)

        x = func(x)

        return x


class RepVGGBottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(RepVGGBottleneck, self).__init__()
        self.conv_bn_relu1 = conv_bn_relu(
            in_planes,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bn=True,
            has_relu=True)
        self.conv_bn_relu2 = RepVGGBlock(
            planes,
            planes * self.expansion,
            kernel_size=3,
            stride=stride,
            padding=1)

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv_bn_relu1 = conv_bn_relu(
            in_planes,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bn=True,
            has_relu=True)
        self.conv_bn_relu2 = conv_bn_relu(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            has_bn=True,
            has_relu=True)
        self.conv_bn_relu3 = conv_bn_relu(
            planes,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bn=True,
            has_relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        out = self.conv_bn_relu3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out


class ResNet_top(nn.Module):

    def __init__(self):
        super(ResNet_top, self).__init__()
        self.conv = conv_bn_relu(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            has_bn=True,
            has_relu=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)

        return x


class ResNet_downsample_module(nn.Module):

    def __init__(self,
                 block,
                 num_blocks,
                 has_skip=False,
                 zero_init_residual=False):
        super(ResNet_downsample_module, self).__init__()
        self.has_skip = has_skip
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 48, num_blocks[0])
        self.layer2 = self._make_layer(block, 96, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            block,
            128,
            num_blocks[2],
            stride=2,
        )
        self.layer4 = self._make_layer(
            block,
            160,
            num_blocks[3],
            stride=2,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = conv_bn_relu(
                self.in_planes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                padding=0,
                has_bn=True,
                has_relu=False)

        layers = list()
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, skip1, skip2):
        x1 = self.layer1(x)
        if self.has_skip:
            x1 = x1 + skip1[0] + skip2[0]
        x2 = self.layer2(x1)
        if self.has_skip:
            x2 = x2 + skip1[1] + skip2[1]
        x3 = self.layer3(x2)
        if self.has_skip:
            x3 = x3 + skip1[2] + skip2[2]
        x4 = self.layer4(x3)
        if self.has_skip:
            x4 = x4 + skip1[3] + skip2[3]

        return x4, x3, x2, x1


class Upsample_unit(nn.Module):

    def __init__(self,
                 ind,
                 in_planes,
                 up_size,
                 chl_num=192,
                 gen_skip=False,
                 gen_cross_conv=False):
        super(Upsample_unit, self).__init__()

        self.u_skip = conv_bn_relu(
            in_planes,
            chl_num,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bn=True,
            has_relu=False)
        self.relu = nn.ReLU(inplace=True)

        self.ind = ind
        if self.ind > 0:
            self.up_size = up_size
            self.up_conv = conv_bn_relu(
                chl_num,
                chl_num,
                kernel_size=1,
                stride=1,
                padding=0,
                has_bn=True,
                has_relu=False)

        self.gen_skip = gen_skip
        if self.gen_skip:
            self.skip1 = conv_bn_relu(
                in_planes,
                in_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                has_bn=True,
                has_relu=True)
            self.skip2 = conv_bn_relu(
                chl_num,
                in_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                has_bn=True,
                has_relu=True)

        self.gen_cross_conv = gen_cross_conv
        if self.ind == 3 and self.gen_cross_conv:
            self.cross_conv = conv_bn_relu(
                chl_num,
                64,
                kernel_size=1,
                stride=1,
                padding=0,
                has_bn=True,
                has_relu=True)

    def forward(self, x, up_x):
        out = self.u_skip(x)

        if self.ind > 0:
            up_x = F.interpolate(
                up_x, size=self.up_size, mode='bilinear', align_corners=True)
            up_x = self.up_conv(up_x)
            out += up_x
        out = self.relu(out)
        skip1 = None
        skip2 = None
        if self.gen_skip:
            skip1 = self.skip1(x)
            skip2 = self.skip2(out)

        cross_conv = None
        if self.ind == 3 and self.gen_cross_conv:
            cross_conv = self.cross_conv(out)

        return out, skip1, skip2, cross_conv


class Upsample_module(nn.Module):

    def __init__(self,
                 chl_num=192,
                 gen_skip=False,
                 gen_cross_conv=False,
                 in_planes=[160, 128, 96, 48],
                 output_shape=(32, 32)):
        super(Upsample_module, self).__init__()
        self.in_planes = in_planes
        h, w = output_shape
        self.up_sizes = [(h // 8, w // 8), (h // 4, w // 4), (h // 2, w // 2),
                         (h, w)]
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv

        self.up1 = Upsample_unit(
            0,
            self.in_planes[0],
            self.up_sizes[0],
            chl_num=chl_num,
            gen_skip=self.gen_skip,
            gen_cross_conv=self.gen_cross_conv)
        self.up2 = Upsample_unit(
            1,
            self.in_planes[1],
            self.up_sizes[1],
            chl_num=chl_num,
            gen_skip=self.gen_skip,
            gen_cross_conv=self.gen_cross_conv)
        self.up3 = Upsample_unit(
            2,
            self.in_planes[2],
            self.up_sizes[2],
            chl_num=chl_num,
            gen_skip=self.gen_skip,
            gen_cross_conv=self.gen_cross_conv)
        self.up4 = Upsample_unit(
            3,
            self.in_planes[3],
            self.up_sizes[3],
            chl_num=chl_num,
            gen_skip=self.gen_skip,
            gen_cross_conv=self.gen_cross_conv)

    def forward(self, x4, x3, x2, x1):
        out1, skip1_1, skip2_1, _ = self.up1(x4, None)
        out2, skip1_2, skip2_2, _ = self.up2(x3, out1)
        out3, skip1_3, skip2_3, _ = self.up3(x2, out2)
        out4, skip1_4, skip2_4, cross_conv = self.up4(x1, out3)

        # 'res' starts from small size
        skip1 = [skip1_4, skip1_3, skip1_2, skip1_1]
        skip2 = [skip2_4, skip2_3, skip2_2, skip2_1]
        out = [out1, out2, out3, out4]

        return out, skip1, skip2, cross_conv


bottleneck_map = dict(default=Bottleneck, repvgg=RepVGGBottleneck)


class Single_stage_module(nn.Module):

    def __init__(self,
                 has_skip=False,
                 gen_skip=False,
                 gen_cross_conv=False,
                 chl_num=256,
                 zero_init_residual=False,
                 num_blocks=[2, 2, 2, 2],
                 bottleneck_type='default'):
        super(Single_stage_module, self).__init__()
        self.has_skip = has_skip
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.chl_num = chl_num
        self.zero_init_residual = zero_init_residual
        self.num_blocks = num_blocks
        self.downsample = ResNet_downsample_module(
            bottleneck_map[bottleneck_type], self.num_blocks, self.has_skip,
            self.zero_init_residual)
        self.upsample = Upsample_module(self.chl_num, self.gen_skip,
                                        self.gen_cross_conv)

    def forward(self, x, skip1, skip2):
        x4, x3, x2, x1 = self.downsample(x, skip1, skip2)
        out, skip1, skip2, cross_conv = self.upsample(x4, x3, x2, x1)

        return out, skip1, skip2, cross_conv


@MODELS.register_module()
class RSNTiny(nn.Module):

    def __init__(self,
                 stage_num,
                 upsample_chl_num,
                 output_last_only=False,
                 bottleneck_type='default',
                 num_blocks=[2, 2, 2, 2]):
        super().__init__()
        self.top = ResNet_top()
        self.stage_num = stage_num
        self.upsample_chl_num = upsample_chl_num
        self.output_last_only = output_last_only
        self.mspn_modules = list()
        for i in range(self.stage_num):
            if i == 0:
                has_skip = False
            else:
                has_skip = True
            if i != self.stage_num - 1:
                gen_skip = True
                gen_cross_conv = True
            else:
                gen_skip = False
                gen_cross_conv = False
            self.mspn_modules.append(
                Single_stage_module(
                    has_skip=has_skip,
                    gen_skip=gen_skip,
                    gen_cross_conv=gen_cross_conv,
                    chl_num=self.upsample_chl_num,
                    bottleneck_type=bottleneck_type,
                    num_blocks=num_blocks))
            setattr(self, 'stage%d' % i, self.mspn_modules[i])

    def forward(self, imgs):
        x = self.top(imgs)
        skip1 = None
        skip2 = None
        outputs = list()
        for i in range(self.stage_num):
            out, skip1, skip2, x = eval('self.stage' + str(i))(x, skip1, skip2)
            outputs.append(out)
        if self.output_last_only:
            return outputs[-1]
        else:
            return outputs
