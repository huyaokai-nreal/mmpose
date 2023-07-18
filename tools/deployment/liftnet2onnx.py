# Copyright (c) OpenMMLab. All rights reserved.

import collections
from typing import Tuple

import onnx
import torch
from onnxsim import simplify
from thop import clever_format, profile
from torch import Tensor, nn

from mmpose.models.heads.regression_heads.lift_head import gMLP


class LiftHeadOnnx(nn.Module):

    def __init__(self, channel_num: int = 55, output_num: int = 42):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_num = channel_num
        self.liftnet = gMLP(
            d_model=2 * self.channel_num,
            d_ffn=4 * self.channel_num,
            num_layers=3)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                2 * self.channel_num, 2 * self.channel_num, kernel_size=1),
            nn.SyncBatchNorm(2 * self.channel_num), nn.ReLU(),
            nn.Conv2d(self.channel_num * 2, output_num, kernel_size=1))

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        output = self.liftnet(feats)
        output = self.last_layer(output)
        return output


def process_ckpt(ckpt):
    param = ckpt['state_dict']
    res = collections.OrderedDict()
    for k, v in param.items():
        if 'kpt3d_lift.' in k:
            new_k = k.replace('kpt3d_lift.', '')
            res[new_k] = v
    return res


def main():

    ckpt_path = '/home/jrchen/git-project/mmpose/work_dirs/pair_hand3d/006_td-stage_two_train_55dim_RLE_head_train_flora_finetune/epoch_100.pth'  # noqa

    net = LiftHeadOnnx()

    ckpt_train = torch.load(ckpt_path, map_location='cpu')

    ckpt = process_ckpt(ckpt_train)
    net.load_state_dict(ckpt, strict=True)

    # from IPython import embed; embed()

    dummy_input = torch.randn(1, 110, 1, 1)
    # img_batch1 = torch.randn(1, 1, 128, 128)
    # img = torch.randn(2, 1, 128, 128)
    # left_hand = torch.randn(1, 1)
    # leftcam_cam_matrix = torch.randn(1, 3, 3)
    # rightcam_cam_matrix = torch.randn(1, 3, 3)
    # lr_rot_matrix = torch.randn(1, 3, 3)
    # lr_p = torch.randn(1, 3, 1)
    # # Calculate MACs
    # # input_data = (img_batch1,)
    input_data = (dummy_input, )
    total_ops, total_params = profile(net.cpu(), input_data)
    flops, params = clever_format([total_ops, total_params], '%.3f')
    print('FLOPs : ', flops, '   Params: ', params)

    input_names = ['input']
    output_names = ['output']
    opset_version = 11
    onnx_path = 'flora_lift.onnx'
    onnx_path_simplify = 'flora_lift_simplify.onnx'
    # training = TrainingMode.EVAL

    torch.onnx.export(
        net,
        input_data,
        onnx_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version)

    run_simplify = True

    if run_simplify:
        # model = onnx.load(onnx_path)
        model_simp, check = simplify(onnx_path)
        onnx.save(model_simp, onnx_path_simplify)
        assert check, 'Simplified ONNX model could not be validated'


if __name__ == '__main__':
    main()
