# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from mmengine.config import DictAction

from mmpose.apis.inference import init_model

try:
    from mmengine.analysis import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmengine to >0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device used for model initialization')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 192],
        help='input image size')
    parser.add_argument(
        '--batch-size', '-b', type=int, default=1, help='input batch size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    input_shape = args.shape
    model = init_model(
        args.config,
        checkpoint=None,
        device=args.device,
        cfg_options=args.cfg_options)
    model = model.cpu()

    if hasattr(model, '_forward'):
        model.forward = model._forward
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    model_info = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(model_info['out_table'])
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {model_info["flops"]/1e9} G\n'
          f'Params: {model_info["params"]/1e6} M \n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
