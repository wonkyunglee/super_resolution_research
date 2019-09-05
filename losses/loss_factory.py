from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np

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


def l2loss(reduction='mean', **_):
    return torch.nn.MSELoss(reduction=reduction)


# def l1loss(reduction='sum', **_):
#     return torch.nn.L1Loss(reduction=reduction)


def l1loss(reduction='mean', **_):
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


def gaussian_kernel_loss(reduction='mean', scale=2, **_):

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


def focal_l1_loss(reduction='mean', scale=2, max_val=5, min_val=0.5, **_):

    l1loss_fn = torch.nn.L1Loss(reduction=reduction)
    def loss_fn(pred_dict, LR, HR):
        loss_dict = dict()
        pred_hr = pred_dict['hr']
        LR = nn.functional.interpolate(LR, scale_factor=scale, mode='bicubic')
        diff = torch.abs(HR - LR)
        # weight = torch.clamp(torch.min(diff) / (diff + epsilon), min=0.1, max=1)
        weight = torch.clamp(diff / torch.mean(diff), min=min_val, max=max_val)
        loss = torch.mean(torch.abs(HR - pred_hr) * weight)

        loss_dict['loss'] = loss
        loss_dict['gt_loss'] = loss

        return loss_dict

    return {'train': loss_fn,
            'val': l1loss_fn}


def distillation_loss(distill, reduction='mean', standardization=False,
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


def attend_similarity_loss(attend, reduction='mean', standardization=False,
                           lambda1=1, lambda2=1, lambda3=1, reverse=False, **_):

    layers_for_attend = []
    for d in attend:
        teacher_layer, student_layer, weight = d.split(':')
        weight = float(weight)
        layers_for_attend.append((teacher_layer, student_layer, weight))

    cross_entropy_loss_fn = torch.nn.BCELoss()
    l1loss_fn = torch.nn.L1Loss(reduction=reduction)
    l2loss_fn = torch.nn.MSELoss(reduction=reduction)

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


def gaussian_mle_loss(reduction='mean', **_):
    l1loss_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(pred_dict, HR, **_):
        gt_loss = 0
        loss_dict = dict()
        pred_hr_mu = pred_dict['hr_mu']
        pred_hr_sigma = pred_dict['hr_sigma']
        mle_loss = (1/2 * torch.log(pred_hr_sigma) +  1/(2 * pred_hr_sigma) * (HR - pred_hr_mu).pow(2)).mean()

        loss_dict['loss'] = mle_loss
        loss_dict['mle_loss'] = mle_loss
        return loss_dict


    return {'train':loss_fn,
            'val':l1loss_fn}


def contrastive_loss(reduction='mean', lambda1=1, lambda2=1, margin=0.1, **_):

    l1loss_fn = torch.nn.L1Loss(reduction=reduction)
    l2loss_fn = torch.nn.MSELoss(reduction=reduction)
    loss_dict = dict()
    def loss_fn(pred_dict_lr, pred_dict_hr, label):
        mapping_lr = pred_dict_lr['mapping']
        mapping_hr = pred_dict_hr['mapping']
        diff = l2loss_fn(mapping_lr, mapping_hr)
        loss = lambda1 * label * diff + lambda2 * (1 - label) * max(0, margin - diff)
        print(label,  label * diff.item(),  (1 - label) * max(0, margin - diff.item()), diff.item())
        loss_dict['loss'] = loss
        loss_dict['diff'] = diff

        return loss_dict

    return {'train': loss_fn,
            'val': loss_fn}


def diff_metric_loss(reduction='mean', w=15, h=15, stride=5, offset=1,
                     lambda1=1.0, lambda2=0.1, **_):
    l1loss_fn = torch.nn.L1Loss(reduction=reduction)

    def get_diff_map(tensor):
        batch_size = tensor.shape[0]
        feature_dim = tensor.shape[1]
        h, w = tensor.shape[-2:]
        diff_map = torch.zeros((batch_size, h*w, h*w)).to(device)
        tensor = tensor.reshape(batch_size, feature_dim, h*w)

        index_x_upper = torch.arange(w*h).long()
        index_y_upper = torch.arange(w*h).long()
        index_x_lower = torch.arange(w*h).long()
        index_y_lower = torch.arange(w*h).long()

        for i in range(1, w*h):
            index_x_upper += 1
            index_x_upper = index_x_upper[:-1]
            index_y_upper = index_y_upper[:-1]

            index_y_lower += 1
            index_x_lower = index_x_lower[:-1]
            index_y_lower = index_y_lower[:-1]


            diff_map[:, index_y_upper, index_x_upper] = torch.abs(tensor[:,:, i:] - tensor[:,:,:-i]).sum(1)
            diff_map[:, index_y_lower, index_x_lower] = torch.abs(tensor[:,:, i:] - tensor[:,:,:-i]).sum(1)
        return diff_map


    def loss_fn(pred_dict, HR,**_):
        gt_loss = 0
        loss_dict = dict()
        pred_hr = pred_dict['hr']
        gt_loss = l1loss_fn(pred_hr, HR)

        upscaled_lr = pred_dict['upscaled_lr']
        diff = torch.abs(HR - upscaled_lr)
        diff_map = get_diff_map(diff[:,:,:stride*h:stride, :stride*w:stride])

        rx = np.random.randint(stride-1)
        ry = np.random.randint(stride-1)
        features = pred_dict['mapping']
        feature_diff_map = get_diff_map(features[:,:,ry:ry+stride*h:stride,rx:rx+stride*w:stride])
        metric_loss = l1loss_fn(feature_diff_map, diff_map * offset)


        loss_dict['loss'] = lambda1 * gt_loss + lambda2 * metric_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['metric_loss'] = metric_loss
        return loss_dict


    return {'train':loss_fn,
            'val':l1loss_fn}


def diff_metric_margin_loss(reduction='mean', w=15, h=15, stride=5, offset=1,
                     lambda1=1.0, lambda2=0.1, **_):
    l1loss_fn = torch.nn.L1Loss(reduction=reduction)

    def get_diff_map(tensor):
        batch_size = tensor.shape[0]
        feature_dim = tensor.shape[1]
        h, w = tensor.shape[-2:]
        diff_map = torch.zeros((batch_size, h*w, h*w)).to(device)
        tensor = tensor.reshape(batch_size, feature_dim, h*w)

        index_x_upper = torch.arange(w*h).long()
        index_y_upper = torch.arange(w*h).long()
        index_x_lower = torch.arange(w*h).long()
        index_y_lower = torch.arange(w*h).long()

        for i in range(1, w*h):
            index_x_upper += 1
            index_x_upper = index_x_upper[:-1]
            index_y_upper = index_y_upper[:-1]

            index_y_lower += 1
            index_x_lower = index_x_lower[:-1]
            index_y_lower = index_y_lower[:-1]


            diff_map[:, index_y_upper, index_x_upper] = torch.abs(tensor[:,:, i:] - tensor[:,:,:-i]).sum(1)
            diff_map[:, index_y_lower, index_x_lower] = torch.abs(tensor[:,:, i:] - tensor[:,:,:-i]).sum(1)
        return diff_map


    def loss_fn(pred_dict, HR,**_):
        gt_loss = 0
        loss_dict = dict()
        pred_hr = pred_dict['hr']
        gt_loss = l1loss_fn(pred_hr, HR)

        upscaled_lr = pred_dict['upscaled_lr']
        diff = torch.abs(HR - upscaled_lr)
        diff_map = get_diff_map(diff[:,:,:stride*h:stride, :stride*w:stride])

        rx = np.random.randint(stride-1)
        ry = np.random.randint(stride-1)
        features = pred_dict['mapping']
        feature_diff_map = get_diff_map(features[:,:,ry:ry+stride*h:stride,rx:rx+stride*w:stride])
        metric_loss = max(0, torch.abs(diff_map * offset - feature_diff_map).mean())

        loss_dict['loss'] = lambda1 * gt_loss + lambda2 * metric_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['metric_loss'] = metric_loss
        return loss_dict


    return {'train':loss_fn,
            'val':l1loss_fn}


