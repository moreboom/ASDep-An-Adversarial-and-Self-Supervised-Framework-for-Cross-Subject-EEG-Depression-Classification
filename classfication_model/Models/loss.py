import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
#loss.py

def get_loss_module():
        return NoFussCrossEntropyLoss(reduction='none')


def l2_reg_loss(model):


    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


