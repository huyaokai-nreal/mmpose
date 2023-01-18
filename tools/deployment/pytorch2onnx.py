# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from torch import nn
import numpy as np
import torch

from mmpose.apis import init_model
from mmpose.models.backbones.utils import repvgg_model_convert
from mmpose.utils import md5sum
try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def _get_conv_layer(submodule):
    for name, child in submodule.named_children():
        if isinstance(child, nn.Conv2d):
            return name, child
        else:
            return _get_conv_layer(child)


def _fuse_preprocess(module):
    mean = model.cfg.mean
    std = model.cfg.std
    mean = torch.as_tensor([mean])
    std = torch.as_tensor([std])
    for name, child in module.named_children():
        if name == 'backbone':
            name, conv_layer = _get_conv_layer(child)
            print(
                f'fuse preprocess with mean {mean}, std {std} to conv {name}')
            w = conv_layer.weight.data
            b = conv_layer.bias.data
            fuse_w = w / std
            fuse_b = -(w * mean / std).view((b.size(0), -1)).sum(dim=-1) + b
            conv_layer.weight.data = fuse_w
            conv_layer.bias.data = fuse_b
            return module
    print('can not find first conv in backbone to fuse preprocess')
    return module


def pytorch2onnx(model,
                 input_shape,
                 output_names,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 simplify=False,
                 dyn_batch=False,
                 graph_mode='eval'):
    """Convert pytorch model to onnx model.

    Args:
        model (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 11.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
        simplify (bool): whether use onnxsim to simply the onnx model
    """
    model.cpu().eval()
    one_img = torch.randn(input_shape)
    dynamic_axes = None
    if dyn_batch:
        dynamic_axes = {
            'input': {
                0: 'batch_size'
            },  # variable length axes
            'output': {
                0: 'batch_size'
            }
        }
    mode = torch.onnx.TrainingMode.EVAL
    if graph_mode == 'train':
        mode = torch.onnx.TrainingMode.TRAINING

    torch.onnx.export(
        model,
        one_img,
        output_file,
        training=mode,
        export_params=True,
        verbose=show,
        do_constant_folding=True,
        opset_version=opset_version,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes)

    if simplify:
        print('Try to refine onnx model with onnxsim')
        try:
            from onnxsim import simplify
        except ImportError as e:
            raise ImportError(f'Please install onnxsim first. {e}')
        onnx_model = onnx.load(output_file)
        from onnxsim import simplify
        model_sim, check = simplify(onnx_model)
        if check:
            onnx.save(model_sim, output_file)
        else:
            print('Failed to refine onnx model with onnxsim')
    md5 = md5sum(output_file)
    onnx_model = onnx.load(output_file)
    output_file = output_file.replace('.onnx', f'_{md5[:6]}.onnx')
    print(f'Successfully exported ONNX model: {output_file}')
    onnx.save(onnx_model, output_file)
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_results = model(one_img)
        if not isinstance(pytorch_results, (list, tuple)):
            assert isinstance(pytorch_results, torch.Tensor)
            pytorch_results = [pytorch_results]

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        sess = rt.InferenceSession(
            output_file, providers=['CPUExecutionProvider'])
        onnx_results = sess.run(None,
                                {net_feed_input[0]: one_img.detach().numpy()})

        # compare results
        assert len(pytorch_results) == len(onnx_results)
        for pt_result, onnx_result in zip(pytorch_results, onnx_results):
            assert np.allclose(
                pt_result.detach().cpu().numpy(), onnx_result, atol=1.e-4
            ), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMPose models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--fuse-pre',
        action='store_true',
        help='fuse preprocess to the first conv')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 256, 192],
        help='input size')
    parser.add_argument(
        '--output-names',
        type=str,
        nargs='+',
        default=['output'],
        help='output name list')
    parser.add_argument(
        '--simplify',
        '-sim',
        action='store_true',
        help='use onnxsim to simplfy the onnx model')
    parser.add_argument(
        '--graph-mode',
        '-gm',
        default='eval',
        help='train or eval graph to export')
    parser.add_argument(
        '--deploy-head',
        '-dh',
        action='store_true',
        help='enable deploy arg for head')
    parser.add_argument(
        '--dyn-batch', action='store_true', help='enable dynamic batch input')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMPose only supports opset 11 now'
    checkpoint = args.checkpoint if args.checkpoint else None
    cfg_options = {
        'model.data_preprocessor': None,
    }
    if args.deploy_head:
        cfg_options['model.head.deploy'] = True
    model = init_model(
        args.config, checkpoint, device='cpu', cfg_options=cfg_options)
    if args.fuse_pre:
        print('enable fuse preprocess mean std to first conv')
        model = _fuse_preprocess(model)
    model = _convert_batchnorm(model)
    model = repvgg_model_convert(model)

    # onnx.export does not support kwargs
    if hasattr(model, '_forward'):
        model.forward = model._forward
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    # convert model to onnx file
    pytorch2onnx(
        model,
        args.shape,
        output_names=args.output_names,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        simplify=args.simplify,
        dyn_batch=args.dyn_batch,
        graph_mode=args.graph_mode)
