# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import numpy as np
import torch

from mmpose.apis import init_model
from mmpose.models.utils.deploy import fuse_preprocess
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


def pytorch2onnx(model,
                 input_shape,
                 input_names,
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
    assert len(input_names) == len(
        input_shape) // 4, f'inputs names {len(input_names)} does not match'
    f'input shapes {len(input_shape)}'
    input_list = []
    for i in range(len(input_names)):
        input_tensor = torch.randn(input_shape[i * 4:i * 4 + 4])
        input_list.append(input_tensor)
    dynamic_axes = None
    if dyn_batch:
        dynamic_axes = dict()
        for input_name in input_names:
            dynamic_axes[input_name] = {0: 'batch_size'}
        for output_name in output_names:
            dynamic_axes[output_name] = {0: 'batch_size'}
    mode = torch.onnx.TrainingMode.EVAL
    if graph_mode == 'train':
        mode = torch.onnx.TrainingMode.TRAINING
    torch.onnx.export(
        model,
        tuple(input_list),
        output_file,
        training=mode,
        export_params=True,
        verbose=show,
        do_constant_folding=True,
        opset_version=opset_version,
        input_names=input_names,
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
    output_file_md5 = output_file.replace('.onnx', f'_{md5[:6]}.onnx')
    print(f'Successfully exported ONNX model: {output_file_md5}')
    onnx.save(onnx_model, output_file_md5)
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_results = model(*input_list)
        if not isinstance(pytorch_results, (list, tuple)):
            assert isinstance(pytorch_results, torch.Tensor)
            pytorch_results = [pytorch_results]

        # get onnx output
        sess = rt.InferenceSession(
            output_file, providers=['CPUExecutionProvider'])
        onnx_input = dict()
        for input_name, input_tensor in zip(input_names, input_list):
            onnx_input[input_name] = input_tensor.detach().numpy()
        onnx_results = sess.run(None, onnx_input)

        # compare results
        assert len(pytorch_results) == len(onnx_results)
        for pt_result, onnx_result in zip(pytorch_results, onnx_results):
            assert np.allclose(
                pt_result.detach().cpu().numpy(), onnx_result, atol=1.e-3
            ), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')

    cmd = f'rm {output_file}'
    print(cmd)
    os.system(cmd)


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
        '--input-names',
        type=str,
        nargs='+',
        default=['input'],
        help='input name list')
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
        '--deploy-module',
        '-dm',
        default='all',
        help='only deploy a submodule of the model, ie. backbone, kpt3d_lift')
    parser.add_argument(
        '--deploy-head',
        '-dh',
        action='store_true',
        help='enable deploy arg for head')
    parser.add_argument(
        '--python-model',
        '-pm',
        action='store_true',
        help='import model from python file')
    parser.add_argument(
        '--dyn-batch', action='store_true', help='enable dynamic batch input')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.python_model:
        exec(f'from {args.config[:-3].replace("/", ".")} import model')
    else:
        assert args.opset_version == 11, 'MMPose only supports opset 11 now'
        checkpoint = args.checkpoint if args.checkpoint else None
        cfg_options = dict()
        if args.deploy_head:
            cfg_options['model.head.deploy'] = True
        model = init_model(
            args.config, checkpoint, device='cpu', cfg_options=cfg_options)
        if args.fuse_pre:
            print('enable fuse preprocess mean std to first conv')
            model = fuse_preprocess(model)
        model = _convert_batchnorm(model)
    if args.deploy_module != 'all':
        model = getattr(model, args.deploy_module)
    # onnx.export does not support kwargs

    if hasattr(model, '_forward'):
        model.forward = model._forward
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'switch_to_deploy'):
            model.backbone.switch_to_deploy()
    # convert model to onnx file
    pytorch2onnx(
        model,
        args.shape,
        input_names=args.input_names,
        output_names=args.output_names,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        simplify=args.simplify,
        dyn_batch=args.dyn_batch,
        graph_mode=args.graph_mode)
