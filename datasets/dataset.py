import os
import pickle
import numpy as np
import torch
import torch.utils.data as utils
from .base_dataset import BaseDataset, BaseBenchmarkDataset

class DIV2K(BaseDataset):
    def __init__(self, scale=2, n_colors=1, rgb_range=255,
                 batch_size=128,
                 test_every=None, patch_size=192, augment=True, data_range='1-800',
                 base_dir=None,
                 name='DIV2K', train=True
                ):
        data_range = data_range.split('-')
        begin, end = list(map(lambda x: int(x)-1, data_range))
        super(DIV2K, self).__init__(scale, n_colors, rgb_range, batch_size,
                 test_every, patch_size, augment, base_dir, name, train,
                  begin, end)


class Benchmark(BaseBenchmarkDataset):
    def __init__(self, scale=2, n_colors=1, rgb_range=255, batch_size=1,
                 base_dir=None, name=''):
        super(Benchmark, self).__init__(
            scale, n_colors, rgb_range, batch_size, base_dir, name
        )


class LargeDiffDIV2K(utils.TensorDataset):

    def __init__(self, is_train=True, base_dir='data/LargeDiffDIV2K/',
                 rgb_range=1, data_type='large', **_):
        super(LargeDiffDIV2K, self).__init__()

        train_x_path = os.path.join(base_dir, 'train_x.pkl')
        valid_x_path = os.path.join(base_dir, 'valid_x.pkl')
        train_y_path = os.path.join(base_dir, 'train_y.pkl')
        valid_y_path = os.path.join(base_dir, 'valid_y.pkl')

        train_x_small_path = os.path.join(base_dir, 'train_x_small.pkl')
        valid_x_small_path = os.path.join(base_dir, 'valid_x_small.pkl')
        train_y_small_path = os.path.join(base_dir, 'train_y_small.pkl')
        valid_y_small_path = os.path.join(base_dir, 'valid_y_small.pkl')

        if is_train:
            with open(train_x_path, 'rb') as f:
                tensor_x = pickle.load(f)
            with open(train_y_path, 'rb') as f:
                tensor_y = pickle.load(f)
            with open(train_x_small_path, 'rb') as f:
                tensor_x_small = pickle.load(f)
            with open(train_y_small_path, 'rb') as f:
                tensor_y_small = pickle.load(f)
        else:
            with open(valid_x_path, 'rb') as f:
                tensor_x = pickle.load(f)
            with open(valid_y_path, 'rb') as f:
                tensor_y = pickle.load(f)
            with open(valid_x_small_path, 'rb') as f:
                tensor_x_small = pickle.load(f)
            with open(valid_y_small_path, 'rb') as f:
                tensor_y_small = pickle.load(f)

        if data_type == 'large':
            x = tensor_x
            y = tensor_y
        elif data_type == 'small':
            x = tensor_x_small
            y = tensor_y_small
        elif data_type == 'mix':
            x = np.concatenate((tensor_x, tensor_x_small), axis=0)
            y = np.concatenate((tensor_y, tensor_y_small), axis=0)

        x = torch.stack([torch.Tensor(i) for i in x])
        y = torch.Tensor(y)
        x *= rgb_range / 255
        y *= rgb_range / 255

        self.tensors = (x, y)







