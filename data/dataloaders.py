"""DataLoader construction for HF datasets."""

import torch
from datasets import Dataset, load_from_disk
from torch import distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from data.datasamplers import (
    StatefulDistributedSampler,
    StatefulRandomSampler,
    StatefulSequentialSampler,
)


def get_dataloaders(cfg):
    """Load trainset and validset, return DataLoaders."""
    train_set = load_from_disk(cfg.data.trainset_path)
    if not isinstance(train_set, Dataset):
        raise ValueError("dataset should be a datasets.Dataset")

    train_sampler = _get_sampler(train_set, cfg)

    # Custom collate for variable-length docs_lengths (intra-document masking)
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
            "docs_lengths": [x["docs_lengths"].tolist() for x in batch],
        }

    has_docs_lengths = "docs_lengths" in train_set.column_names

    trainloader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=cfg.training.micro_batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
        persistent_workers=cfg.training.num_workers > 0,
        collate_fn=collate_fn if has_docs_lengths else None,
    )

    if not cfg.data.eval or not cfg.data.validset_path:
        return trainloader, None

    valid_set = load_from_disk(cfg.data.validset_path)
    if not isinstance(valid_set, Dataset):
        raise ValueError("dataset should be a datasets.Dataset")

    if cfg.data.valid_tokens:
        valid_rows = cfg.data.valid_tokens // (cfg.data.seq_len + 1)
        valid_set = valid_set.take(min(len(valid_set), valid_rows))

    if dist.is_initialized():
        valid_sampler = DistributedSampler(valid_set, drop_last=True)
    else:
        valid_sampler = SequentialSampler(valid_set)

    has_docs_lengths_valid = "docs_lengths" in valid_set.column_names

    validloader = DataLoader(
        valid_set,
        batch_size=cfg.training.micro_batch_size,
        drop_last=True,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
        persistent_workers=False,
        collate_fn=collate_fn if has_docs_lengths_valid else None,
    )

    return trainloader, validloader


def _get_sampler(train_set, cfg):
    """Initialize a sampler for the training DataLoader."""
    ddp = dist.is_initialized()

    if cfg.training.sampler == "random":
        if ddp:
            sampler = DistributedSampler(
                train_set, shuffle=True, seed=cfg.training.sampler_seed, drop_last=True
            )
        else:
            sampler = RandomSampler(
                train_set,
                generator=(
                    torch.Generator().manual_seed(cfg.training.sampler_seed)
                    if cfg.training.sampler_seed
                    else None
                ),
            )

    elif cfg.training.sampler == "sequential":
        if ddp:
            sampler = DistributedSampler(train_set, shuffle=False, drop_last=True)
        else:
            sampler = SequentialSampler(train_set)

    elif cfg.training.sampler == "stateful_random":
        micro_step_start = (
            cfg.checkpoint.resume_step * cfg.training.grad_accumulation_steps
            if cfg.checkpoint.resume
            else 0
        )
        if ddp:
            sampler = StatefulDistributedSampler(
                train_set,
                batch_size=cfg.training.micro_batch_size,
                seed=cfg.training.sampler_seed,
                start_iter=micro_step_start,
            )
        else:
            sampler = StatefulRandomSampler(
                train_set,
                batch_size=cfg.training.micro_batch_size,
                shuffle=True,
                seed=cfg.training.sampler_seed,
                start_idx=micro_step_start,
            )

    elif cfg.training.sampler == "stateful_sequential":
        micro_step_start = (
            cfg.checkpoint.resume_step * cfg.training.grad_accumulation_steps
            if cfg.checkpoint.resume
            else 0
        )
        if ddp:
            raise NotImplementedError("StatefulDistributedSampler currently needs a seed.")
        sampler = StatefulSequentialSampler(
            train_set, batch_size=cfg.training.micro_batch_size, start_idx=micro_step_start
        )

    else:
        raise NotImplementedError(f"Sampler {cfg.training.sampler} is not implemented.")

    return sampler
