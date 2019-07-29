import os
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



