# Copyright (c) OpenMMLab. All rights reserved.
from functools import wraps
from typing import Callable

import torch
from mmengine import is_seq_of
from nreal_data_tool.utils.misc import cross_merge_list


def format_data(func: Callable):

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, dict):
                if 'inputs' in arg:
                    inputs = arg['inputs']
                    if torch.is_tensor(inputs):
                        if inputs.dim() == 5:
                            inputs = inputs.reshape(-1, inputs.shape[2],
                                                    inputs.shape[3],
                                                    inputs.shape[4])
                        arg['inputs'] = inputs
            if is_seq_of(arg, tuple):
                tmp_list = cross_merge_list(arg[0], arg[1])
                args[i] = tmp_list
        return func(*args, **kwargs)

    return wrapped_func
