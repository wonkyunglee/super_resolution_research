from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = None

def standardize(tensor, dim=(2,3)):
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    standardized = (tensor - mean) / (std + 1e-8)
    return standardized

def get_loss(config):
    f = globals().get(config.loss.name)
    global device
    os.environ["CUDA_VISIBLE_DEVICES"]= str(config.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return f(**config.loss.params)


def l2loss(reduction='sum', **_):
    return torch.nn.MSELoss(reduction=reduction)


# def l1loss(reduction='sum', **_):
#     return torch.nn.L1Loss(reduction=reduction)


def l1loss(reduction='sum', **_):
    l1loss_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(pred_dict, HR, **_):
        gt_loss = 0
        loss_dict = dict()
        pred_hr = pred_dict['hr']
        gt_loss = l1loss_fn(pred_hr, HR)

        loss_dict['loss'] = gt_loss
        loss_dict['gt_loss'] = gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':l1loss_fn}


def gaussian_kernel_loss(reduction='sum', scale=2, **_):

    l1loss_fn = torch.nn.L1Loss(reduction=reduction)
    max_val = 5
    epsilon = 1e-8
    def loss_fn(pred_dict, LR, HR):
        loss_dict = dict()
        pred_hr = pred_dict['hr']
        LR = nn.functional.interpolate(LR, scale_factor=scale, mode='bicubic')
        diff = torch.abs(HR - LR)
        std = torch.clamp(torch.max(diff) / (diff + epsilon), min=0.1, max=10)
        temperature = 0.1
        loss = 1 - torch.mean(torch.exp(-((HR - pred_hr) / std / temperature).pow(2)))

        loss_dict['loss'] = loss
        loss_dict['gt_loss'] = loss

        return loss_dict

    return {'train': loss_fn,
            'val': l1loss_fn}


def focal_l1_loss(reduction='sum', scale=2, **_):

    l1loss_fn = torch.nn.L1Loss(reduction=reduction)
    max_val = 5
    epsilon = 1e-8
    def loss_fn(pred_dict, LR, HR):
        loss_dict = dict()
        pred_hr = pred_dict['hr']
        LR = nn.functional.interpolate(LR, scale_factor=scale, mode='bicubic')
        diff = torch.abs(HR - LR)
        weight = torch.clamp(torch.min(diff) / (diff + epsilon), min=0.1, max=1)
        loss = torch.mean(torch.abs(HR - pred_hr) * weight)

        loss_dict['loss'] = loss
        loss_dict['gt_loss'] = loss

        return loss_dict

    return {'train': loss_fn,
            'val': l1loss_fn}


def distillation_loss(distill, reduction='sum', standardization=False,
                      lambda1=1, lambda2=1, gt_loss_type='l1',
                      distill_loss_type='l1', **_):
    layers_for_distill = []
    for d in distill:
        teacher_layer, student_layer, weight = d.split(':')
        weight = float(weight)
        layers_for_distill.append((teacher_layer, student_layer, weight))

    l1loss_fn = torch.nn.L1Loss(reduction=reduction)
    l2loss_fn = torch.nn.MSELoss(reduction=reduction)

    if gt_loss_type == 'l1':
        gt_loss_fn = l1loss_fn
    elif gt_loss_type == 'l2':
        gt_loss_fn = l2loss_fn

    if distill_loss_type == 'l1':
        distill_loss_fn = l1loss_fn
    elif distill_loss_type == 'l2':
        distill_loss_fn = l2loss_fn

    def loss_fn(teacher_pred_dict, student_pred_dict, HR):
        gt_loss = 0
        distill_loss = 0
        loss_dict = dict()
        student_pred_hr = student_pred_dict['hr']
        gt_loss = gt_loss_fn(student_pred_hr, HR)

        for teacher_layer, student_layer, weight in layers_for_distill:
            tl = teacher_pred_dict[teacher_layer]
            sl = student_pred_dict[student_layer]
            if standardization:
                tl = standardize(tl, dim=(2,3))
                sl = standardize(sl, dim=(2,3))
            distill_loss += weight * distill_loss_fn(tl, sl)

        loss_dict['loss'] = lambda1 * gt_loss + lambda2 * distill_loss
        loss_dict['gt_loss'] = lambda1 * gt_loss
        loss_dict['distill_loss'] = lambda2 * distill_loss
        return loss_dict

    return {'train':loss_fn,
            'val':l1loss_fn}


def attend_similarity_loss(attend, reduction='sum', standardization=False,
                           lambda1=1, lambda2=1, lambda3=1, reverse=False, **_):

    layers_for_attend = []
    for d in attend:
        teacher_layer, student_layer, weight = d.split(':')
        weight = float(weight)
        layers_for_attend.append((teacher_layer, student_layer, weight))

    l1loss_fn = l1loss(reduction=reduction)
    l2loss_fn = l2loss(reduction=reduction)
    cross_entropy_loss_fn = torch.nn.BCELoss()

    def distill_loss_fn(tl, sl, attention_map, reverse=False):
        if reverse:
            distill_loss = torch.mean(torch.abs(tl - sl) * (1-attention_map).pow(5)) # focal loss
        else:
            distill_loss = torch.mean(torch.abs(tl - sl) * attention_map)
        return distill_loss


    def get_attention_map(x, y):
        attention = F.cosine_similarity(x, y, dim=1).unsqueeze(1)
        attention = (attention + 1.0) / 2.01
        return attention


    def loss_fn(teacher_pred_dict, student_pred_dict, HR):
        total_loss = 0
        gt_loss = 0
        cross_entropy_loss = 0
        attended_distill_loss = 0
        student_pred_hr = student_pred_dict['hr']
        gt_loss = l1loss_fn(student_pred_hr, HR)

        for teacher_layer, student_layer, weight in layers_for_attend:
            tl = teacher_pred_dict[teacher_layer]
            sl = student_pred_dict[student_layer]
            if standardization:
                tl = standardize(tl, dim=(2,3))
                sl = standardize(sl, dim=(2,3))
            attention_map = get_attention_map(tl, sl).detach()
            attended_distill_loss += weight * distill_loss_fn(tl, sl, attention_map, reverse)

            ones = torch.ones([*attention_map.shape], dtype=torch.float32).to(device)
            cross_entropy_loss += cross_entropy_loss_fn(attention_map, ones)

        total_loss = lambda1 * gt_loss + lambda2 * attended_distill_loss + lambda3 * cross_entropy_loss
        loss_dict = dict()
        loss_dict['loss'] = total_loss
        loss_dict['gt_loss'] = lambda1 * gt_loss
        loss_dict['attended_distill_loss'] = lambda2 * attended_distill_loss
        loss_dict['cross_entropy_loss'] = lambda3 * cross_entropy_loss

        return loss_dict

    return {'train':loss_fn,
            'val':l1loss_fn}
