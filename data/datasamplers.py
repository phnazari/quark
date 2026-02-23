"""Stateful data samplers for resumable training.

References:
- https://discuss.pytorch.org/t/resume-iterating-dataloader-from-checkpoint-batch-idx/60683/3
- https://github.com/facebookresearch/vissl/blob/main/vissl/data/data_helper.py#L93
"""

from typing import Optional, Sized

import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


class StatefulSequentialSampler(Sampler):
    """Samples elements sequentially with optional start offset."""

    def __init__(self, data_source: Sized, batch_size=None, start_idx: int = 0):
        self.data_source = data_source
        self.start_idx = start_idx * batch_size

    def __iter__(self):
        """Return an iterator over the dataset."""
        return iter(range(self.start_idx, len(self.data_source)))

    def __len__(self):
        """Return the number of elements in the dataset."""
        return len(self.data_source) - self.start_idx


class StatefulRandomSampler(Sampler):
    """Samples elements with shuffling, with optional start offset."""

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        start_idx: int = 0,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.start_idx = start_idx * batch_size
        self.shuffle = shuffle
        if shuffle:
            if seed is None:
                raise ValueError("Seed must be set if shuffle is True in a stateful sampler.")
            self.g = torch.Generator()
            self.g.manual_seed(seed)

    def __iter__(self):
        """Return an iterator over the dataset."""
        n = len(self.data_source)
        indices = list(range(n))
        if self.shuffle:
            indices = torch.randperm(n, generator=self.g).tolist()
        return iter(indices[self.start_idx :])

    def __len__(self):
        """Return the number of elements in the dataset."""
        return len(self.data_source) - self.start_idx


class StatefulDistributedSampler(DistributedSampler):
    """DistributedSampler that supports resuming from a specific iteration."""

    def __init__(self, dataset, batch_size=None, seed: int = 0, start_iter: int = 0):
        super().__init__(dataset, shuffle=False, seed=seed)

        self.start_iter = start_iter
        self.batch_size = batch_size
        self.total_size = len(dataset) - (len(dataset) % self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas
        print(f"rank: {self.rank}: sampler created, start_iter: {self.start_iter}")

    def __iter__(self):
        """Return an iterator over the dataset."""
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        shuffling = torch.randperm(self.num_samples, generator=g).tolist()
        indices = np.array(
            list(range(self.rank * self.num_samples, (self.rank + 1) * self.num_samples))
        )[shuffling].tolist()

        assert len(indices) == self.num_samples
        assert self.batch_size > 0, "batch_size not set for the sampler"

        start_index = self.start_iter * self.batch_size
        indices = indices[start_index:]
        return iter(indices)
