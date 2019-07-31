from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def prepare_train_directories(config, model_type):
    out_dir = config.train[model_type + '_dir']
    os.makedirs(os.path.join(out_dir, 'checkpoint'), exist_ok=True)
    

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def float2uint8(image):
    if type(image) == torch.Tensor:
        image = image.detach().cpu().numpy()
    if len(image.shape) == 4:
        image = image[0]
    min_val = np.min(image)
    image += min_val
    max_val = np.max(image)
    image /= max_val
    image *= 255
    image = image.astype('uint8')
    if image.shape[-1] != 1:
        image = np.transpose(image, (1,2,0))
    if image.shape[-1] == 1:
        image = np.squeeze(image)
    return image 


def get_SR_image_figure(pred_lr, pred_hr, GT):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    ax1.imshow(float2uint8(pred_lr))
    ax1.set_title('pred_lr, mean_val : %.4f'%pred_lr.mean())
    ax2.imshow(float2uint8(pred_hr))        
    ax2.set_title('pred_hr, mean_val : %.4f'%pred_hr.mean())
    ax3.imshow(float2uint8(GT))
    ax3.set_title('ground_truth, mean_val : %.4f'%GT.mean())
    return fig