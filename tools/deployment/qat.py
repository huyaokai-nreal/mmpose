# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

from mmengine import Config, DictAction

from mmpose.engine.quant_runner import QuantRunner
from mmpose.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate data for calibration')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='float model weight')
    parser.add_argument('output_dir', help='quant model save dir')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_all_modules()
    runner = QuantRunner(cfg, args.checkpoint)
    runner.run()
    model_name = os.path.basename(args.config).split('.')[0]
    runner.export_quant_model(args.output_dir, model_name)
