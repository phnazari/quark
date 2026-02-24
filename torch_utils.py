"""PyTorch setup utilities for DDP and device configuration."""

import os
import random

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group


def pytorch_setup(cfg):
    """Initialize PyTorch, DDP, and return (local_rank, world_size, device, master_process)."""
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master_process = rank == 0
        seed_offset = rank
    else:
        master_process = True
        seed_offset = 0
        local_rank = None
        world_size = 1
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    random.seed(cfg.seed + seed_offset)
    np.random.seed(cfg.seed + seed_offset)
    torch.manual_seed(cfg.seed + seed_offset)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return local_rank, world_size, device, master_process


def destroy_ddp():
    """Clean up DDP process group."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        destroy_process_group()
