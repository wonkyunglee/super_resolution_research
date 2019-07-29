from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)


def l2loss(reduction='sum', **_):
    return torch.nn.MSELoss(reduction=reduction)
    
    
def l1loss(reduction='sum', **_):
    return torch.nn.L1Loss(reduction=reduction)

