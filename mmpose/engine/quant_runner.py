# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy

import aimet_torch.v2 as aimet
import torch
from aimet_torch import batch_norm_fold, model_preparer
from aimet_torch.v2 import quantsim
from loguru import logger
from mmengine import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from mmengine.runner.runner import Runner
from tqdm import tqdm

from mmpose.apis.inference import init_model
from mmpose.models.utils.deploy import fuse_preprocess


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


class QuantRunner:

    def __init__(self, cfg: Config, checkpoint: str) -> None:
        self.evaluator = Evaluator(cfg['val_evaluator'])
        self.quant_cfg = cfg['quant_cfg']
        self.model = init_model(cfg, checkpoint)
        self.model = fuse_preprocess(self.model)
        self.model.deploy()
        self.model.cuda().eval()
        self.data_preprocessor = BaseDataPreprocessor()
        self.data_preprocessor.cuda()
        self.input_shape = self.quant_cfg['input_shape']
        prepared_model = model_preparer.prepare_model(
            copy.deepcopy(self.model))
        dummy_input = torch.zeros(self.input_shape).cuda()
        batch_norm_fold.fold_all_batch_norms(
            prepared_model,
            input_shapes=dummy_input.shape,
            dummy_input=dummy_input)
        self.sim = quantsim.QuantizationSimModel(
            prepared_model,
            dummy_input=dummy_input,
            default_output_bw=self.quant_cfg['act_bitwidth'],
            default_param_bw=self.quant_cfg['weight_bitwidth'])
        logger.info('start init dataloader')
        self.val_dataloader = Runner.build_dataloader(cfg['val_dataloader'])
        self.train_dataloader = Runner.build_dataloader(
            cfg['train_dataloader'])

    def evaluate(self, type='float'):
        model = self.model
        if type == 'quant':
            model = self.sim.model
        logger.info('start running evaluation')
        for data in tqdm(self.val_dataloader):
            data = self.data_preprocessor(data)
            inputs = data['inputs'].float()
            feat = model(inputs)
            outputs = self.model.predict(feat, data['data_samples'])
            self.evaluator.process(outputs, data['data_samples'])
        metrics = self.evaluator.evaluate(len(self.val_dataloader.dataset))
        return metrics

    def run(self):
        # run original float model evaluation
        metrics = self.evaluate()
        logger.info('float32 model metric')
        logger.info(metrics)
        # PTQ
        logger.info('start running ptq encoding')
        with aimet.nn.compute_encodings(self.sim.model):
            dataloader = iter(self.train_dataloader)
            for _ in tqdm(range(self.quant_cfg['ptq_iters'])):
                x = next(dataloader)
                x = x['inputs']
                x = x.cuda().float()
                self.sim.model(x)
        metrics = self.evaluate('quant')
        logger.info('ptq model metric')
        logger.info(metrics)
        if self.quant_cfg['type'] == 'QAT':
            logger.info('start running qat process')
            optimzer = torch.optim.Adam(self.sim.model.parameters(), lr=1e-5)
            for epoch_idx in range(self.quant_cfg.get('qat_epochs', 1)):
                with tqdm(total=len(self.train_dataloader)) as t:
                    for i, item in enumerate(self.train_dataloader):
                        t.set_description('Epoch %i' % epoch_idx)
                        item = self.data_preprocessor(item)
                        inputs = item['inputs'].float()
                        feat = self.sim.model(inputs)
                        loss = self.model.loss(feat, item['data_samples'])
                        total_loss = 0
                        for k, v in loss.items():
                            if k.startswith('loss_'):
                                total_loss += v
                        if (i % 20 == 0):
                            info_dict = dict()
                            loss.update({'total loss': total_loss.item()})
                            for k, v in loss.items():
                                if isinstance(v, torch.Tensor):
                                    info_dict[k] = f'{v.item():.3f}'
                                else:
                                    info_dict[k] = f'{v:.3f}'
                            t.set_postfix(info_dict)
                        t.update(1)
                        total_loss.backward()
                        optimzer.step()
                        optimzer.zero_grad()
            metrics = self.evaluate('quant')
            logger.info('qat model metric')
            logger.info(metrics)

    def export_quant_model(self, output_dir, model_name):
        logger.info(f'export onnx model and encodings to {output_dir}')
        self.sim.export(
            output_dir,
            model_name,
            dummy_input=torch.zeros(self.quant_cfg['input_shape']),
            onnx_export_args={
                'input_names': self.quant_cfg['input_names'],
                'output_names': self.quant_cfg['output_names']
            })
