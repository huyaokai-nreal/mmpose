from mmpose.registry import MODELS
import torch.nn as nn
import torch


@MODELS.register_module()
class FilterL2Loss(nn.Module):

    def __init__(self,
                 loss_weight=0.05,
                 filter_scope_list=['backbone']) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.filter_scope_list = filter_scope_list

    def forward(self, model):
        loss = 0
        for name, module in model.named_children():
            if name in self.filter_scope_list:
                for name, param in module.named_parameters():
                    if 'conv.weight' in name:
                        ith_filter_reg_loss = torch.sqrt(
                            torch.sum(torch.pow(param, 2), dim=[1, 2,
                                                                3]))  # GL
                        loss += torch.sum(ith_filter_reg_loss)
        return loss * self.loss_weight
