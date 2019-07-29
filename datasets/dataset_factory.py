from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from .data_utils import MSDataLoader
from .dataset import DIV2K, Benchmark



def get_train_dataset(config, transform=None):
    name = config.data.train.name
    scale = config.data.scale
    n_colors = config.data.n_colors
    rgb_range = config.data.rgb_range
    batch_size = config.train.batch_size
    if type(name) == list:
        for n in name:
            f = globals().get(n)
            datasets.append(f(**config.data.train.params,
                              scale=scale, n_colors=n_colors, rgb_range=rgb_range,
                              batch_size=batch_size,
                              name=n, train=True,
                             ))
        dataset = ConcatDataset(datasets)
    else:
        f = globals().get(name)
        dataset = f(**config.data.train.params,
                      scale=scale, n_colors=n_colors, rgb_range=rgb_range,
                      batch_size=batch_size,
                      name=name, train=True,
                     )

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
    name = config.data.valid.name
    scale = config.data.scale
    n_colors = config.data.n_colors
    rgb_range = config.data.rgb_range
    batch_size = 1
    if type(name) == list:
        for n in name:
            f = globals().get(n)
            datasets.append(f(**config.data.valid.params,
                              scale=scale, n_colors=n_colors, rgb_range=rgb_range,
                              batch_size=batch_size,
                              name=n, train=False,
                             ))
        dataset = ConcatDataset(datasets)
    else:
        f = globals().get(name)
        dataset = f(**config.data.valid.params,
                      scale=scale, n_colors=n_colors, rgb_range=rgb_range,
                      batch_size=batch_size,
                      name=name, train=False,
                     )

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
    name = config.data.test.name
    scale = config.data.scale
    n_colors = config.data.n_colors
    rgb_range = config.data.rgb_range
    batch_size = 1
    if type(name) == list:
        datasets = []
        for n in name:
            datasets.append(Benchmark(**config.data.test.params,
                              scale=scale, n_colors=n_colors,
                              rgb_range=rgb_range, batch_size=batch_size,
                              name=n))
        dataset = ConcatDataset(datasets)
    else:
        dataset = Benchmark(**config.data.test.params,
                  scale=scale, n_colors=n_colors,
                  rgb_range=rgb_range, batch_size=batch_size,
                  name=name)

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







# class Data:
#     def __init__(self, args):
#         self.loader_train = None
#         if not args.test_only:
#             datasets = []
#             for d in args.data_train:
#                 f = globals().get(d)
#                 datasets.append(f(args, name=d))

#             self.loader_train = MSDataLoader(
#                 args,
#                 MyConcatDataset(datasets),
#                 batch_size=args.batch_size,
#                 shuffle=True,
#                 pin_memory=not args.cpu
#             )

#         self.loader_test = []
#         for d in args.data_test:
#             if d in ['Set5', 'Set14', 'B100', 'Urban100']:
#                 testset = BenchMark(args, train=False, name=d)
#             else:
#                 module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
#                 m = import_module('data.' + module_name.lower())
#                 testset = getattr(m, module_name)(args, train=False, name=d)

#             self.loader_test.append(MSDataLoader(
#                 args,
#                 testset,
#                 batch_size=1,
#                 shuffle=False,
#                 pin_memory=not args.cpu
#             ))


