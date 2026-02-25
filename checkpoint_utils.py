"""Checkpoint saving and loading utilities."""

import json
import os
import re

import torch


def _latest_checkpoint(ckpt_dir: str, prefix: str = "checkpoint_") -> str | None:
    """Retrieve the latest checkpoint path in a directory."""
    if not os.path.isdir(ckpt_dir):
        return None

    checkpoints = [f for f in os.listdir(ckpt_dir) if re.match(rf"^{prefix}\d+\.pth$", f)]
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: int(re.search(r"\d+", x).group()))

    return os.path.join(ckpt_dir, checkpoints[-1])


def save_checkpoint(step, model, engine, cfg, metrics=None):
    """Save a training checkpoint."""
    state = {
        "step": step,
        "state_dict": model.state_dict(),
        "optimizer": engine.optimizer.state_dict(),
        "scheduler": engine.scheduler.state_dict() if engine.scheduler else {},
        "scaler": engine.scaler.state_dict(),
    }

    exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    save_path = os.path.join(exp_dir, f"ckpt_step_{step}.pth")
    print(f"Saving checkpoint to {save_path}")
    torch.save(state, save_path)

    if metrics is not None:
        metrics_path = os.path.join(exp_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(dict(metrics), f)


def maybe_load_checkpoint(cfg):
    """Load a checkpoint if resuming, else return None."""
    if not cfg.resume:
        return None

    resume_exp_name = cfg.resume_exp_name if cfg.resume_exp_name is not None else cfg.exp_name
    ckpt_dir = os.path.join(cfg.out_dir, resume_exp_name)

    if cfg.resume_step is not None:
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{cfg.resume_step}.pth")
    else:
        ckpt_path = _latest_checkpoint(ckpt_dir, prefix="ckpt_step_")

    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    return ckpt


def match_state_dict_keys(state_dict: dict, state_dict_orig: dict) -> dict:
    """Modify keys of state_dict to match state_dict_orig (handles DDP/compile prefixes)."""
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    orig_key = next(iter(state_dict_orig.keys()))

    if orig_key.startswith("_orig_mod.module."):
        state_dict = {"_orig_mod.module." + k: v for k, v in state_dict.items()}
    elif orig_key.startswith("_orig_mod."):
        state_dict = {"_orig_mod." + k: v for k, v in state_dict.items()}
    elif orig_key.startswith("module._orig_mod."):
        state_dict = {"module._orig_mod." + k: v for k, v in state_dict.items()}
    elif orig_key.startswith("module."):
        state_dict = {"module." + k: v for k, v in state_dict.items()}

    return state_dict
