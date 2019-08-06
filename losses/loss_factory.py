from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import os

device = None

def get_loss(config):
    f = globals().get(config.loss.name)
    global device
    os.environ["CUDA_VISIBLE_DEVICES"]= str(config.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        gt_loss = 0
        distill_loss = 0
        student_pred_hr = student_pred_dict['hr']
        gt_loss += l1loss_fn(student_pred_hr, HR)

        for teacher_layer, student_layer, weight in layers_for_distill:
            tl = teacher_pred_dict[teacher_layer]
            sl = student_pred_dict[student_layer]
            distill_loss += weight * l2loss_fn(tl, sl)

        loss_dict = dict()
        total_loss = gt_loss + distill_loss
        loss_dict['loss'] = total_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['distill_loss'] = distill_loss
        return loss_dict

    return {'train':loss_fn,
            'val':l1loss_fn}


def attend_similarity_loss(reduction='sum', lambda1=1, lambda2=1, lambda3=1, **_):
    l1loss_fn = l1loss(reduction=reduction)
    l2loss_fn = l2loss(reduction=reduction)
    cross_entropy_loss_fn = torch.nn.BCELoss()
    def loss_fn(teacher_pred_dict, student_pred_dict, HR):
        total_loss = 0
        gt_loss = 0
        cross_entropy_loss = 0
        attended_distill_loss = 0
        student_pred_hr = student_pred_dict['hr']
        gt_loss = l1loss_fn(student_pred_hr, HR)

        for layer, value in student_pred_dict.items():
            if 'attention' in layer:
                values = value.view(-1,1)
                ones = torch.ones([*value.shape], dtype=torch.float32).to(device)
                cross_entropy_loss += cross_entropy_loss_fn(value, ones)

                layer_name = layer.split('_attention')[0]
                sl = student_pred_dict[layer_name]
                tl = teacher_pred_dict[layer_name]
                attended_distill_loss += l2loss_fn(tl * value, sl)

        total_loss = lambda1 * gt_loss + lambda2 * attended_distill_loss # + lambda3 * cross_entropy_loss
        loss_dict = dict()
        loss_dict['loss'] = total_loss
        loss_dict['gt_loss'] = lambda1 * gt_loss
        loss_dict['attended_distill_loss'] = lambda2 * attended_distill_loss
        loss_dict['cross_entropy_loss'] = lambda3 * cross_entropy_loss

        return loss_dict

    return {'train':loss_fn,
            'val':l1loss_fn}
