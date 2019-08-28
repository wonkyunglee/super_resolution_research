from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import torch.utils.data as utils

from .data_utils import MSDataLoader
from .dataset import DIV2K, Benchmark, LargeDiffDIV2K



def get_train_dataset(config, transform=None):
    scale = config.data.scale
    n_colors = config.data.n_colors
    rgb_range = config.data.rgb_range
    batch_size = config.train.batch_size
    datasets = []
    for train_dict in config.data.train:
        name = train_dict.name
        params = train_dict.params
        f = globals().get(name)
        datasets.append(f(**params,
                          scale=scale, n_colors=n_colors, rgb_range=rgb_range,
                          batch_size=batch_size,
                          name=name, train=True,
                         ))
    dataset = ConcatDataset(datasets)

    return dataset


def get_train_dataloader(config, transform=None):
    num_workers = config.data.num_workers
    dataset = get_train_dataset(config, transform)
    batch_size = config.train.batch_size
    dataloader = MSDataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=num_workers,
                              pin_memory=False)
    return dataloader


def get_valid_dataset(config, transform=None):
    scale = config.data.scale
    n_colors = config.data.n_colors
    rgb_range = config.data.rgb_range
    batch_size = 1
    datasets = []
    for valid_dict in config.data.valid:
        name = valid_dict.name
        params = valid_dict.params
        f = globals().get(name)
        datasets.append(f(**params,
                          scale=scale, n_colors=n_colors, rgb_range=rgb_range,
                          batch_size=batch_size,
                          name=name, train=False,
                         ))
    dataset = ConcatDataset(datasets)

    return dataset


def get_valid_dataloader(config, transform=None):
    num_workers = config.data.num_workers
    dataset = get_valid_dataset(config, transform)
    batch_size = 1
    dataloader = MSDataLoader(dataset,
                              shuffle=False,
                              batch_size=batch_size,
                              drop_last=False,
                              num_workers=num_workers,
                              pin_memory=False)
    return dataloader


def get_test_dataset(config, transform=None):
    scale = config.data.scale
    n_colors = config.data.n_colors
    rgb_range = config.data.rgb_range
    batch_size = 1
    datasets = []
    for test_dict in config.data.test:
        name = test_dict.name
        params = test_dict.params
        datasets.append(Benchmark(**params,
                          scale=scale, n_colors=n_colors,
                          rgb_range=rgb_range, batch_size=batch_size,
                          name=name))
    dataset = ConcatDataset(datasets)
    return dataset


def get_test_dataloader(config, transform=None):
    num_workers = config.data.num_workers
    dataset = get_test_dataset(config, transform)
    dataloader = MSDataLoader(dataset,
                              shuffle=False,
                              batch_size=1,
                              drop_last=False,
                              pin_memory=False,
                              num_workers=num_workers)
    return dataloader


def get_train_dataloader_largediff(config, transform=None):
    base_dir = config.data.base_dir
    params = config.data.train.params
    train_dataset = LargeDiffDIV2K(is_train=True, base_dir=base_dir,
                                  rgb_range=config.data.rgb_range, 
                                  **params)
    train_dataloader = utils.DataLoader(train_dataset, 
                                        batch_size=config.train.batch_size,
                                        shuffle=True ) 
    return train_dataloader


def get_valid_dataloader_largediff(config, transform=None):
    base_dir = config.data.base_dir
    params = config.data.valid.params
    valid_dataset = LargeDiffDIV2K(is_train=False, base_dir=base_dir,
                                  rgb_range=config.data.rgb_range,
                                  **params)
    valid_dataloader = utils.DataLoader(valid_dataset, 
                                        batch_size=config.eval.batch_size,
                                        shuffle=False) 
    return valid_dataloader