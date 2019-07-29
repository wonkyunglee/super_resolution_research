import os
import glob
import random
import pickle

from .data_utils import *

import numpy as np
import imageio
import torch
import torch.utils.data as data

class ParentDataset(data.Dataset):
    
    def __init__(self, scale=2, n_colors=1, rgb_range=255, batch_size=128, base_dir=None, name=''):
        self.scale = scale
        self.n_colors = n_colors
        self.rgb_range = rgb_range
        self.batch_size = batch_size
        self.augment = augment
        self.base_dir = base_dir
        self.name = name
        self.begin = None
        self.end = None 
        
    def _initialize(self):
        raise NotImplementedError
        
        
    def __len__(self):
        raise NotImplementedError
    
    
    def __get_index__(self, idx):
        raise NotImplementedError
    
    
    def _get_hr_and_lr_img_dir(self, base_dir):
        raise NotImplementedError

        
    def get_patch(self, lr, hr):
        raise NotImplementedError
        
    
    def _get_imagepath_list(self, hr_dir, lr_dir, scale, ext='png'):
        hr_paths = sorted(
            glob.glob(os.path.join(hr_dir, '*' + ext))
        )
        lr_paths = []
        for path in hr_paths:
            filename, _ = os.path.splitext(os.path.basename(path))
            
            lr_paths.append(os.path.join(
                lr_dir, 'X{}/{}x{}.{}'.format(
                    scale, filename, scale, ext
                )
            ))
        if self.begin is not None and self.end is not None:
            return hr_paths[self.begin:self.end], lr_paths[self.begin:self.end]
        else:
            return hr_paths, lr_paths
    
    
    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = set_channel(*pair, n_channels=self.n_colors)
        pair_t = np2Tensor(*pair, rgb_range=self.rgb_range)

        return pair_t[0], pair_t[1], filename

        
    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.hr_data_paths[idx]
        f_lr = self.lr_data_paths[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        with open(f_hr, 'rb') as _f:
            hr = pickle.load(_f)
        with open(f_lr, 'rb') as _f:
            lr = pickle.load(_f)

        return lr, hr, filename


class BaseDataset(ParentDataset):
    
    def __init__(self, scale=2, n_colors=1, rgb_range=255, batch_size=128, 
                 test_every=None, patch_size=192, augment=True, 
                 base_dir=None, name='', train=True, begin=None, end=None):
        super().__init__(scale, n_colors, rgb_range, batch_size, 
                        base_dir, name)
        self.train = train
        self.test_every = test_every
        self.augment = augment
        self.patch_size = patch_size
        self.begin = begin
        self.end = end
        self._initialize()
        if train:
            self._init_repeat()
        
        
    def _initialize(self):
        base_dir = self.base_dir
        scale = self.scale
        hr_dir, lr_dir = self._get_hr_and_lr_img_dir(base_dir)
        hr_bin_dir, lr_bin_dir = self._get_hr_and_lr_binary_dir(base_dir)
        
        hr_paths, lr_paths = self._get_imagepath_list(hr_dir, lr_dir, scale, 'png')
        
        self.hr_data_paths = self._init_data_paths(hr_paths, hr_dir, hr_bin_dir, reset=False, verbose=False)
        self.lr_data_paths = self._init_data_paths(lr_paths, lr_dir, lr_bin_dir, reset=False, verbose=False)
        
        
    def _init_repeat(self):
        n_patches = self.batch_size * self.test_every
        n_images = len(self.hr_data_paths)
        assert n_images != 0
        self.repeat = max(n_patches // n_images, 1)
        
        
    def _init_data_paths(self, img_paths, img_dir, bin_dir, reset, verbose):
        bin_paths = self._convert2binarypath(img_dir, bin_dir, img_paths, 'pt')
        data_paths = []
        for img_path, bin_path in zip(img_paths, bin_paths):
            data_paths.append(bin_path)
            self._check_and_save(img_path, bin_path, reset=reset,
                                 verbose=verbose)
        return data_paths
    
    
    def _convert2binarypath(self, img_dir, bin_dir, paths, ext='pt'):
        paths = [os.path.splitext(
            path.replace(img_dir, bin_dir))[0] + '.' + ext for path in paths]
        return paths
    
    
    def _get_hr_and_lr_img_dir(self, base_dir):
        lr_dir = os.path.join(base_dir, 
                              self.name + '_' + 'train' + '_' + 'LR' + '_' + 'bicubic')
        hr_dir = os.path.join(base_dir, 
                             self.name + '_' + 'train' + '_' + 'HR')
        return hr_dir, lr_dir
    
    
    def _get_hr_and_lr_binary_dir(self, base_dir):
        lr_dir = os.path.join(base_dir, 'bin', 
                              self.name + '_' + 'train' + '_' + 'LR' + '_' + 'bicubic')
        hr_dir = os.path.join(base_dir, 'bin', 
                             self.name + '_' + 'train' + '_' + 'HR')
        return hr_dir, lr_dir   

    
    def _check_and_save(self, img_path, bin_path, reset=False, verbose=False):
        bin_dir = '/'.join(bin_path.split('/')[:-1])
        os.makedirs(bin_dir, exist_ok=True)

        if not os.path.isfile(bin_path) or reset:
            if verbose:
                print('Making a binary: {}'.format(bin_path))
            with open(bin_path, 'wb') as f:
                pickle.dump(imageio.imread(img_path), f)

                
    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.hr_data_paths[idx]
        f_lr = self.lr_data_paths[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        with open(f_hr, 'rb') as _f:
            hr = pickle.load(_f)
        with open(f_lr, 'rb') as _f:
            lr = pickle.load(_f)

        return lr, hr, filename
        
    
    def __len__(self):
        if self.train:
            return len(self.hr_data_paths) * self.repeat
        else:
            return len(self.hr_data_paths)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.hr_data_paths)
        else:
            return idx
    
    def get_patch(self, lr, hr):
        scale = self.scale
        if self.train:
            lr, hr = get_patch(
                lr, hr,
                patch_size=self.patch_size,
                scale=scale,
            )
            if self.augment: 
                lr, hr = augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr
    
    
    
    
class BaseBenchmarkDataset(ParentDataset):
    
    def __init__(self, scale=2, n_colors=1, rgb_range=255, batch_size=128, 
                 base_dir=None, name=''):
        super().__init__(scale, n_colors, rgb_range, batch_size, 
                         base_dir, name)
        print(base_dir)
        self._initialize()
        
    
    def _initialize(self):
        base_dir = self.base_dir
        scale = self.scale
        hr_dir, lr_dir = self._get_hr_and_lr_img_dir(base_dir)
        hr_paths, lr_paths = self._get_imagepath_list(hr_dir, lr_dir, scale, 'png')
        
        self.hr_data_paths = hr_paths
        self.lr_data_paths = lr_paths
        
    
    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.hr_data_paths[idx]
        f_lr = self.lr_data_paths[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = imageio.imread(f_hr)
        lr = imageio.imread(f_lr)

        return lr, hr, filename

    
    def __len__(self):
        return len(self.hr_data_paths)

    
    def _get_index(self, idx):
        return idx
    
    
    def get_patch(self, lr, hr):
        scale = self.scale
        ih, iw = lr.shape[:2]
        hr = hr[0:ih * scale, 0:iw * scale]
        return lr, hr
    
    
    def _get_hr_and_lr_img_dir(self, base_dir):
        lr_dir = os.path.join(base_dir, 'LR' + '_' + 'bicubic')
        hr_dir = os.path.join(base_dir, 'HR')
        return hr_dir, lr_dir
        
        
        

# class BaseDataset(data.Dataset):
#     def __init__(self, config
#                  name='', train=True, benchmark=False):
        
#         self.name = name
#         self.train = train
#         self.benchmark = benchmark
#         self.scale = scale
            
#         self.augment = config.augment
#         self.patch_size = config.patch_size
#         self.ext = config.ext
#         self.n_colors = config.n_colors
#         self.rgb_range = config.rgb_range
#         self.batch_size = config.batch_size
#         self.test_every = config.test_every
#         self.data_train = config.data_train
#         self.dir_data = config.dir_data
        
#         self._set_filesystem(dir_data)
#         if self.ext.find('img') < 0:
#             path_bin = os.path.join(self.apath, 'bin')
#             os.makedirs(path_bin, exist_ok=True)

#         list_hr, list_lr = self._scan()
#         if self.ext.find('img') >= 0 or benchmark:
#             self.images_hr, self.images_lr = list_hr, list_lr
#         elif self.ext.find('sep') >= 0:
#             os.makedirs(
#                 self.dir_hr.replace(self.apath, path_bin),
#                 exist_ok=True
#             )
#             os.makedirs(
#                 os.path.join(
#                     self.dir_lr.replace(self.apath, path_bin),
#                     'X{}'.format(self.scale)
#                 ),
#                 exist_ok=True
#             )
            
#             self.images_hr, self.images_lr = [], []
#             for h in list_hr:
#                 b = h.replace(self.apath, path_bin)
#                 b = b.replace(self.ext[0], '.pt')
#                 self.images_hr.append(b)
#                 self._check_and_load(self.ext, h, b, verbose=True) 
#             for i, ll in enumerate(list_lr):
#                 b = l.replace(self.apath, path_bin)
#                 b = b.replace(self.ext[1], '.pt')
#                 self.images_lr.append(b)
#                 self._check_and_load(self.ext, l, b, verbose=True) 
#         if train:
#             n_patches = self.batch_size * self.test_every
#             n_images = len(self.data_train) * len(self.images_hr)
#             if n_images == 0:
#                 self.repeat = 0
#             else:
#                 self.repeat = max(n_patches // n_images, 1)

#     # Below functions as used to prepare images
#     def _scan(self):
#         names_hr = sorted(
#             glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
#         )
#         names_lr = []
#         for f in names_hr:
#             filename, _ = os.path.splitext(os.path.basename(f))
            
#             names_lr.append(os.path.join(
#                 self.dir_lr, 'X{}/{}x{}{}'.format(
#                     self.scale, filename, self.scale, self.ext[1]
#                 )
#             ))

#         return names_hr, names_lr

#     def _set_filesystem(self, dir_data):
#         self.apath = os.path.join(dir_data, self.name)
#         self.dir_hr = os.path.join(self.apath, 'HR')
#         self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
#         self.ext = ('.png', '.png')

#     def _check_and_load(self, ext, img, f, verbose=True):
#         if not os.path.isfile(f) or ext.find('reset') >= 0:
#             if verbose:
#                 print('Making a binary: {}'.format(f))
#             with open(f, 'wb') as _f:
#                 pickle.dump(imageio.imread(img), _f)

#     def __getitem__(self, idx):
#         lr, hr, filename = self._load_file(idx)
#         pair = self.get_patch(lr, hr)
#         pair = data_utils.set_channel(*pair, n_channels=self.n_colors)
#         pair_t = data_utils.np2Tensor(*pair, rgb_range=self.rgb_range)

#         return pair_t[0], pair_t[1], filename

#     def __len__(self):
#         if self.train:
#             return len(self.images_hr) * self.repeat
#         else:
#             return len(self.images_hr)

#     def _get_index(self, idx):
#         if self.train:
#             return idx % len(self.images_hr)
#         else:
#             return idx

#     def _load_file(self, idx):
#         idx = self._get_index(idx)
#         f_hr = self.images_hr[idx]
#         f_lr = self.images_lr[idx]

#         filename, _ = os.path.splitext(os.path.basename(f_hr))
#         if self.ext == 'img' or self.benchmark:
#             hr = imageio.imread(f_hr)
#             lr = imageio.imread(f_lr)
#         elif self.ext.find('sep') >= 0:
#             with open(f_hr, 'rb') as _f:
#                 hr = pickle.load(_f)
#             with open(f_lr, 'rb') as _f:
#                 lr = pickle.load(_f)

#         return lr, hr, filename

#     def get_patch(self, lr, hr):
#         scale = self.scale
#         if self.train:
#             lr, hr = data_utils.get_patch(
#                 lr, hr,
#                 patch_size=self.patch_size,
#                 scale=scale,
#             )
#             if self.augment: 
#                 lr, hr = data_utils.augment(lr, hr)
#         else:
#             ih, iw = lr.shape[:2]
#             hr = hr[0:ih * scale, 0:iw * scale]

#         return lr, hr
