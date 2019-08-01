from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)


def l2loss(reduction='sum', **_):
    return torch.nn.MSELoss(reduction=reduction)
    
    
def l1loss(reduction='sum', **_):
    return torch.nn.L1Loss(reduction=reduction)


def l1loss(reduction='sum', **_):
    return torch.nn.L1Loss(reduction=reduction)


def distillation_loss(distill, reduction='sum', **_):
    layers_for_distill = []
    for d in distill:
        teacher_layer, student_layer, weight = d.split(':')
        weight = float(weight)
        layers_for_distill.append((teacher_layer, student_layer, weight))
    
    l1loss_fn = l1loss(reduction=reduction)
    l2loss_fn = l2loss(reduction=reduction)
    def loss_fn(teacher_pred_dict, student_pred_dict, HR):
        total_loss = 0
        student_pred_hr = student_pred_dict['hr']
        total_loss += l1loss_fn(student_pred_hr, HR) 
        
        for teacher_layer, student_layer, weight in layers_for_distill:
            tl = teacher_pred_dict[teacher_layer]
            sl = student_pred_dict[student_layer]
            total_loss += weight * l2loss_fn(tl, sl)
        return total_loss
    return {'train':loss_fn,
            'val':l1loss_fn}


def attend_similarity_loss(reduction='sum', **_):
    l1loss_fn = l1loss(reduction=reduction)
    l2loss_fn = l2loss(reduction=reduction)
    cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()
    def loss_fn(teacher_pred_dict, student_pred_dict, HR):
        total_loss = 0
        student_pred_hr = student_pred_dict['hr']
        total_loss += l1loss_fn(student_pred_hr, HR) 
        
        for layer, value in student_pred_dict.items():
            if 'attention' in layer:
                print('attention loss layer : %s'%layer)
                ones = torch.ones([*value.shape], dtype=torch.long).to(device)
                total_loss += cross_entropy_loss_fn(value, ones)
        return total_loss
    
    return {'train':loss_fn,
            'val':l1loss_fn}
