from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np

def prepare_train_directories(config, model_type):
    out_dir = config.train[model_type + '_dir']
    os.makedirs(os.path.join(out_dir, 'checkpoint'), exist_ok=True)
    
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
