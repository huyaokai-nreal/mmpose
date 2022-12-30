# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import torch
import mmengine
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmpose.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPose prune (and eval) model')
    parser.add_argument('config', help='prune config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir', help='the directory to save evaluation results')
    parser.add_argument('--out', help='the file to save metric results.')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- Dump predictions --------------------
    # if args.dump is not None:
    #     assert args.dump.endswith(('.pkl', '.pickle')), \
    #         'The dump file must be a pkl file.'
    #     dump_metric = dict(type='DumpResults', out_file_path=args.dump)
    #     if isinstance(cfg.test_evaluator, (list, tuple)):
    #         cfg.test_evaluator = list(cfg.test_evaluator).append(dump_metric)
    #     else:
    #         cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    return cfg

import numpy as np
def get_sparsity(module):
    zero = 0
    total = 0

    for name, param in module.named_parameters():
        if 'conv.weight' in name:  # 仅计算卷积权重的稀疏
            total += (param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3])
            p = param.cpu().detach().numpy()
            zero += np.count_nonzero(p == 0)  # 此时处于剪枝之后，等于0的属于被剪枝的，而且在np.countzero中是true，计算其数量就是零值数量

    return 100 * (zero / total)

def main():
    args = parse_args()

    # register all modules in mmpose into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # if args.out:

    #     class SaveMetricHook(Hook):

    #         def after_test_epoch(self, _, metrics=None):
    #             if metrics is not None:
    #                 mmengine.dump(metrics, args.out)

    #     runner.register_hook(SaveMetricHook(), 'LOWEST')


    # prune
    runner.load_or_resume()
    filter_num = 0
    zero_filter_num = 0
    thre = 0.001

    for name, param in runner.model.named_parameters():
        if 'conv.weight' in name:
            filter_num += param.shape[0]
            for i in range(param.shape[0]):
                idx = torch.lt(abs(param.data[i, :, :, :]), thre)  # prune
                param.data[i, :, :, :][idx] = 0
                ith_filter_L1_score = torch.sum(abs(param.data[i, :, :, :]))  # 统计移除的filter的数目
                # print(name,": ",ith_filter_L1_score)
                if ith_filter_L1_score == 0: 
                    zero_filter_num += 1

    #  compute sparsity
    weight_sparsity = get_sparsity(runner.model)
    filter_sparsity = 100 * zero_filter_num / filter_num
    
    print(f'weight_sparsity: {weight_sparsity:.2f},filter_sparsity: {filter_sparsity:.2f}')

    runner.train()  # 新建文件夹，设置epoch=1,lr=0，保存的epoch_1.pth为剪枝后的checkpoint



if __name__ == '__main__':
    main()
