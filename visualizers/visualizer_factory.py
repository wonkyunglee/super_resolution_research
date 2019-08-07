from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

sys.path.append('../')
from utils.utils import quantize

def float2uint8(image, normalize=False, rgb_range=1):
    if type(image) == torch.Tensor:
        image = image.detach().cpu().numpy()
    if len(image.shape) == 4:
        image = image[0]
    if normalize:
        min_val = np.min(image)
        image += min_val
        max_val = np.max(image)
        image /= max_val
    image *= 255/rgb_range
    image = image.astype('uint8')
    if image.shape[-1] != 1:
        image = np.transpose(image, (1,2,0))
    if image.shape[-1] == 1:
        image = np.squeeze(image)
    return image


def get_figure_basic(LR, HR, pred):
    LR = LR[0]
    HR = HR[0]
    pred_residual_hr = pred['residual_hr'][0]
    pred_hr = pred['hr'][0]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,4))
    cmap = 'gray'
    ax1.imshow(float2uint8(pred_residual_hr), cmap=cmap)
    ax1.set_title('pred_residual_hr, mean_val : %.4f'%pred_residual_hr.mean())
    ax2.imshow(float2uint8(pred_hr), cmap=cmap)
    ax2.set_title('pred_hr, mean_val : %.4f'%pred_hr.mean())
    ax3.imshow(float2uint8(HR), cmap=cmap)
    ax3.set_title('ground_truth, mean_val : %.4f'%HR.mean())
    ax4.imshow(float2uint8(HR - pred_hr), cmap=cmap)
    ax4.set_title('GT - pred_hr, mean_val : %.4f'%(HR-pred_hr).mean())

    return fig

def step0_visualizer(rgb_range):
    return get_figure_basic


def step1_visualizer(rgb_range):
    return get_figure_basic


def step2_visualizer(rgb_range):
    def get_figure(LR, HR, pred_student, pred_teacher):
        LR = LR[0]
        HR = HR[0]
        pred_teacher_residual_hr = pred_teacher['residual_hr'][0]
        pred_student_residual_hr = pred_student['residual_hr'][0]
        pred_student_hr = pred_student['hr'][0]
        pred_teacher_hr = pred_teacher['hr'][0]
        residual_diff = pred_teacher_residual_hr - pred_student_residual_hr

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,4))
        cmap = 'gray'
        ax1.imshow(float2uint8(quantize(pred_student_residual_hr, rgb_range), rgb_range), cmap=cmap)
        ax1.set_title('pred_s_residual_hr, mean_val : %.4f'% pred_student_residual_hr.mean())
        ax2.imshow(float2uint8(quantize(pred_student_hr, rgb_range), rgb_range), cmap=cmap)
        ax2.set_title('pred_s_hr, mean_val : %.4f'%pred_student_hr.mean())
        ax3.imshow(float2uint8(quantize(residual_diff, rgb_range), rgb_range), cmap=cmap)
        ax3.set_title('residual_diff, mean_val : %.4f'%residual_diff.mean())
        ax4.imshow(float2uint8(quantize(HR - pred_student_hr, rgb_range), rgb_range), cmap=cmap)
        ax4.set_title('GT - pred_hr, mean_val : %.4f'%(HR-pred_student_hr).mean())

        return fig

    return get_figure


def step2_attention_visualizer(rgb_range):
    def get_figure(LR, HR, pred_student, pred_teacher):
        LR = LR[0]
        HR = HR[0]
        pred_teacher_residual_hr = pred_teacher['residual_hr'][0]
        pred_student_residual_hr = pred_student['residual_hr'][0]
        pred_student_hr = pred_student['hr'][0]
        pred_teacher_hr = pred_teacher['hr'][0]
        residual_diff = pred_teacher_residual_hr - pred_student_residual_hr
        attention = pred_student['mapping_attention'][0]

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15,3))
        cmap = 'gray'
        ax1.imshow(float2uint8(quantize(pred_student_residual_hr, rgb_range), rgb_range), cmap=cmap)
        ax1.set_title('pred_s_residual_hr, mean_val : %.4f'% pred_student_residual_hr.mean())
        ax2.imshow(float2uint8(quantize(pred_student_hr, rgb_range), rgb_range), cmap=cmap)
        ax2.set_title('pred_s_hr, mean_val : %.4f'%pred_student_hr.mean())
        ax3.imshow(float2uint8(quantize(residual_diff, rgb_range), rgb_range), cmap=cmap)
        ax3.set_title('residual_diff, mean_val : %.4f'%residual_diff.mean())
        ax4.imshow(float2uint8(quantize(HR - pred_student_hr, rgb_range), rgb_range), cmap=cmap)
        ax4.set_title('GT - pred_hr, mean_val : %.4f'%(HR-pred_student_hr).mean())
        ax5.imshow(float2uint8(quantize(attention, rgb_range), rgb_range), cmap=cmap)
        ax5.set_title('attention, mean_val : %.4f'%(attention).mean())
        return fig

    return get_figure




def get_visualizer(config):
    func = globals().get(config.visualizer.name + '_visualizer')
    return func(config.data.rgb_range)

