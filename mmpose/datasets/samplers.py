# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
from typing import Iterator, List, Optional, Sized, Union
import random

import torch
from mmengine.dataset.sampler import DefaultSampler
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmpose.datasets import CombinedDataset
from mmpose.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class DistributedRangeSampler(DefaultSampler):

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        super().__init__(dataset, shuffle, seed, round_up)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        start_id = self.rank * self.num_samples
        end_id = min(start_id + self.num_samples, len(indices))
        indices = indices[start_id:end_id]

        return iter(indices)


@DATA_SAMPLERS.register_module()
class MultiSourceSampler(Sampler):
    """Multi-Source Sampler. According to the sampling ratio, sample data from
    different datasets to form batches.

    Args:
        dataset (Sized): The dataset
        batch_size (int): Size of mini-batch
        source_ratio (list[int | float]): The sampling ratio of different
            source datasets in a mini-batch
        shuffle (bool): Whether shuffle the dataset or not. Defaults to
            ``True``
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
        seed (int, optional): Random seed. If ``None``, set a random seed.
            Defaults to ``None``
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 source_ratio: List[Union[int, float]],
                 data_ratio: List[Union[int, float]],
                 shuffle: bool = True,
                 round_up: bool = True,
                 seed: Optional[int] = None) -> None:

        assert isinstance(dataset, CombinedDataset),\
            f'The dataset must be CombinedDataset, but get {dataset}'
        assert isinstance(batch_size, int) and batch_size > 0, \
            'batch_size must be a positive integer value, ' \
            f'but got batch_size={batch_size}'
        assert isinstance(source_ratio, list), \
            f'source_ratio must be a list, but got source_ratio={source_ratio}'
        assert len(source_ratio) == len(dataset._lens), \
            'The length of source_ratio must be equal to ' \
            f'the number of datasets, but got source_ratio={source_ratio}'

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.cumulative_sizes = [0] + list(itertools.accumulate(dataset._lens))
        self.batch_size = batch_size
        self.source_ratio = source_ratio
        self.data_ratio = data_ratio
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / world_size))
        self.num_samples = int(math.ceil(sum([len(ds) * self.data_ratio[idx] for idx, ds in enumerate(self.dataset.datasets)]) * 1.0 / world_size))
        self.num_per_source = [
            int(batch_size * sr / sum(source_ratio)) for sr in source_ratio
        ]
        self.num_per_source[0] = batch_size - sum(self.num_per_source[1:])

        assert sum(self.num_per_source) == batch_size, \
            'The sum of num_per_source must be equal to ' \
            f'batch_size, but get {self.num_per_source}'

        self.seed = sync_random_seed() if seed is None else seed
        self.shuffle = shuffle
        self.round_up = round_up
        self.epoch = 0

    def _infinite_indices(self, sample_size: int, data_ratio: float) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        random.seed(self.seed + self.epoch)
        
        while True:
            if self.shuffle:
                if self.epoch // 10:
                    yield from random.sample(torch.randperm(sample_size, generator=g).tolist()[:int(sample_size * data_ratio)], int(sample_size * data_ratio))
                else:
                    yield from random.sample(torch.randperm(sample_size, generator=g).tolist()[int(sample_size * data_ratio):], sample_size - int(sample_size * data_ratio))
            else:
                yield from random.sample(torch.arange(sample_size).tolist(), int(sample_size * data_ratio))

    def _indices_of_rank(self, sample_size: int, data_ratio: float) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(
            self._infinite_indices(sample_size, data_ratio), self.rank, None,
            self.world_size)

    def __iter__(self) -> Iterator[int]:
        
        self.source2inds = {
            source: self._indices_of_rank(len(ds), self.data_ratio[source])
            for source, ds in enumerate(self.dataset.datasets)
        }
        
        batch_buffer = []
        num_iters = self.num_samples // self.batch_size
        if self.round_up and self.num_samples > num_iters * self.batch_size:
            num_iters += 1
        for i in range(num_iters):
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.source2inds[source]:
                    idx += self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
        
        return iter(batch_buffer)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Compatible in `epoch-based runner."""
        self.epoch = epoch
