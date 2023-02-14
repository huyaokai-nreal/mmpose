from mmpose.registry import HOOKS
from mmengine.hooks.hook import Hook
from typing import List, Optional
from nni.compression.pytorch import pruning
from nni.compression.pytorch.speedup import ModelSpeedup
import torch
import os
import pickle as pkl
from mmengine.runner import Runner
from mmengine.device import get_device

PRUNERS = dict(
    l1=pruning.L1NormPruner,
    l2=pruning.L2NormPruner,
    fpgm=pruning.FPGMPruner,
)


@HOOKS.register_module()
class NNIPruneHook(Hook):

    def __init__(self,
                 pruner_name: str = 'l1',
                 config_list: Optional[List] = None,
                 input_shape: Optional[List] = None) -> None:
        super().__init__()
        assert pruner_name in PRUNERS, f'{pruner_name} pruner is not defined'
        self.pruner_name = pruner_name
        self.config_list = config_list
        if self.config_list is None:
            self.config_list = [{
                'sparsity_per_layer': 0.5,
                'op_types': ['Linear', 'Conv2d']
            }, {
                'exclude':
                True,
                'op_names': [
                    'head.predict_layers.1.conv_layers.1.conv',
                    'head.predict_layers.2.conv_layers.1.conv',
                    'head.predict_layers.3.conv_layers.1.conv',
                    'head.predict_layers.0.conv_layers.1.conv',
                ]
            }]
        self.input_shape = input_shape
        if self.input_shape is None:
            self.input_shape = [1, 1, 128, 128]

    def before_train(self, runner: Runner) -> None:
        if hasattr(runner.model, 'module'):
            model = runner.model.module
        else:
            model = runner.model
        self.pruner = PRUNERS[self.pruner_name](
            model,
            config_list=self.config_list,
            dummy_input=torch.rand(*self.input_shape).to(get_device()),
            mode='dependency_aware')
        model, masks = self.pruner.compress()
        # show the masks sparsity
        for name, mask in masks.items():
            print(
                name, ' sparsity : ',
                '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))
        # need to unwrap the model, if the model is wrapped before speedup
        self.pruner._unwrap_model()
        # speedup the model
        ModelSpeedup(model,
                     torch.rand(*self.input_shape).to(get_device()),
                     masks).speedup_model()
        if hasattr(runner.model, 'module'):
            model = runner.wrap_model(
                runner.cfg.get('model_wrapper_cfg'), model)
        runner.model = model
        runner.optim_wrapper = runner.build_optim_wrapper(
            runner.cfg['optim_wrapper'])
        # Automatically scaling lr by linear scaling rule
        runner.scale_lr(runner.optim_wrapper, runner.auto_scale_lr)
        runner.param_schedulers = runner.build_param_scheduler(  # type: ignore
            runner.cfg['param_scheduler'])  # type: ignore
        with open(os.path.join(runner.work_dir, 'prune_mask.pkl'), 'wb') as f:
            pkl.dump(mask, f)
