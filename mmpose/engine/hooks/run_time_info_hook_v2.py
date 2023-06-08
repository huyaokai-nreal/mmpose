# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmengine.hooks import RuntimeInfoHook
from mmengine.registry import HOOKS

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class RuntimeInfoHookV2(RuntimeInfoHook):
    """A hook that updates runtime information into message hub.

    E.g. ``epoch``, ``iter``, ``max_epochs``, and ``max_iters`` for the training state.
    Components that cannot access the runner can get runtime information through the
    message hub.
    """

    priority = 'VERY_HIGH'

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        """Update current iter and learning rate information before every
        iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        runner.message_hub.update_info('iter', runner.iter)
        lr_dict = runner.optim_wrapper.get_lr()
        assert isinstance(lr_dict, dict), (
            '`runner.optim_wrapper.get_lr()` should return a dict '
            'of learning rate when training with OptimWrapper(single '
            'optimizer) or OptimWrapperDict(multiple optimizer), '
            f'but got {type(lr_dict)} please check your optimizer '
            'constructor return an `OptimWrapper` or `OptimWrapperDict` '
            'instance')
        for name, lr in lr_dict.items():
            lr_unique = []
            for lr_ in lr:
                if lr_ not in lr_unique:
                    lr_unique.append(lr_)

            for i, lr_ in enumerate(lr_unique):
                runner.message_hub.update_scalar(f'train/{name}_{i}', lr_)
