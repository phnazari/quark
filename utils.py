"""Project-level utilities."""

import math
import os
import shutil

import torch
from omegaconf import DictConfig


def print_master(msg):
    """Print only in the master process."""
    rank = os.environ.get("RANK", -1)
    ddp = int(rank) != -1
    master_process = (not ddp) or (int(rank) == 0)
    if master_process:
        print(msg)


def get_variant_name(cfg: DictConfig) -> str:
    """Derive variant name from model config."""
    model_type = cfg.model.model_type
    names = {"transformer": "Transformer", "delta_net": "DeltaNet"}
    return names.get(model_type, model_type)


def get_param_groups(model, weight_decay):
    """Separate parameters into decay and no-decay groups."""
    decay = set()
    no_decay = set()

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
                decay.add(fpn)
            else:
                no_decay.add(fpn)

    param_dict = dict(model.named_parameters())

    decay &= param_dict.keys()
    no_decay &= param_dict.keys()

    inter_params = decay & no_decay
    assert len(inter_params) == 0, f"parameters in both decay/no_decay: {inter_params}"
    assert len(param_dict.keys() - (decay | no_decay)) == 0, (
        f"parameters not in either: {param_dict.keys() - (decay | no_decay)}"
    )

    return [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]


def maybe_make_dir(cfg):
    """Create experiment directory if checkpointing is enabled."""
    if not cfg.checkpoint.save_intermediate_checkpoints and not cfg.checkpoint.save_last_checkpoint:
        return
    if cfg.checkpoint.resume and cfg.checkpoint.resume_exp_name is None:
        return

    exp_dir = os.path.join(cfg.out_dir, cfg.checkpoint.exp_name)

    if os.path.exists(exp_dir):
        if not cfg.checkpoint.over_write:
            raise ValueError(f"Found existing exp_dir at {exp_dir}.")
        print(f"Removing experiment dir: {exp_dir}")
        shutil.rmtree(exp_dir)

    print(f"Creating experiment directory: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)


def log(
    cfg,
    metrics,
    micro_step,
    train_loss,
    train_loss_array,
    valid_loss,
    optimizer,
    world_size,
    grad_norm=None,
    throughput=None,
    step_time=None,
):
    """Update metrics, print to console, and log to W&B."""
    if isinstance(train_loss_array, list):
        train_loss_avg = torch.stack(train_loss_array).mean().item()
    elif isinstance(train_loss_array, torch.Tensor):
        train_loss_avg = train_loss_array.item()

    new_metrics = {
        "micro_step": micro_step,
        "step": micro_step // cfg.training.grad_accumulation_steps,
        "tokens": micro_step * cfg.training.micro_batch_size * cfg.data.seq_len * world_size,
        "lr": optimizer.param_groups[0].get("lr", float("NaN")),
        "train/loss": train_loss.item(),
        "train/loss_avg": train_loss_avg,
        "train/ppl": math.exp(train_loss),
        "train/ppl_avg": math.exp(train_loss_avg),
    }

    if valid_loss is not None:
        new_metrics["valid/loss"] = valid_loss
        new_metrics["valid/ppl"] = math.exp(valid_loss)

    if grad_norm is not None:
        new_metrics["train/grad_norm"] = (
            grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
        )

    if throughput is not None:
        new_metrics["train/throughput"] = throughput
    if step_time is not None:
        new_metrics["train/step_time"] = step_time

    for k, v in new_metrics.items():
        metrics.setdefault(k, []).append(v)

    msg = " | ".join(
        f"{key}: {value:.3e}" if isinstance(value, float) else f"{key}: {value}"
        for key, value in new_metrics.items()
    )
    print(msg)

    if cfg.logging.wandb_log:
        import wandb

        wandb.log(dict(new_metrics))
