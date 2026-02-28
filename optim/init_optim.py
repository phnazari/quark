"""Initialize optimizer and scheduler."""

import torch

from .lr_schedule import WSD, LinearCooldown, WarmupConstant, WarmupCosine


def initialize_optimizer(param_groups, cfg):
    """Initialize an optimizer from config."""
    if cfg.training.optim == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg.training.lr,
            betas=[cfg.training.beta1, cfg.training.beta2],
            weight_decay=cfg.training.weight_decay,
            fused=cfg.training.fused_optim,
            eps=cfg.training.eps,
        )

    elif cfg.training.optim == "nadamw":
        kwargs = dict(
            lr=cfg.training.lr,
            betas=[cfg.training.beta1, cfg.training.beta2],
            weight_decay=cfg.training.weight_decay,
            decoupled_weight_decay=True,
            eps=cfg.training.eps,
        )
        # fused only supported on CUDA
        if cfg.training.fused_optim and torch.cuda.is_available():
            kwargs["fused"] = True
        optimizer = torch.optim.NAdam(param_groups, **kwargs)

    elif cfg.training.optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=cfg.training.lr,
            momentum=cfg.training.beta1,
            dampening=getattr(cfg.training, "dampening", 0.0),
            weight_decay=cfg.training.weight_decay,
        )

    elif cfg.training.optim == "signSGD":
        from .sign_sgd import signSGD

        optimizer = signSGD(
            param_groups,
            lr=cfg.training.lr,
            momentum=cfg.training.beta1,
            dampening=getattr(cfg.training, "dampening", 0.0),
            weight_decay=cfg.training.weight_decay,
        )

    else:
        raise NotImplementedError(f"Not implemented optim: {cfg.training.optim}.")

    return optimizer


def initialize_scheduler(optimizer, cfg):
    """Initialize a learning rate scheduler from config."""
    if cfg.training.scheduler is None:
        return None

    # Warmup steps: int or fraction of steps_budget
    warmup_steps = None
    if cfg.training.warmup_steps is not None:
        warmup_steps = (
            cfg.training.warmup_steps
            if isinstance(cfg.training.warmup_steps, int)
            else int(cfg.training.warmup_steps * cfg.training.steps_budget)
        )

    # Cooldown steps: int or fraction of steps_budget
    cooldown_steps = None
    if cfg.training.cooldown_steps is not None:
        cooldown_steps = (
            cfg.training.cooldown_steps
            if isinstance(cfg.training.cooldown_steps, int)
            else int(cfg.training.cooldown_steps * cfg.training.steps_budget)
        )

    # Final LR: direct or as fraction of peak lr
    lr_end = None
    if cfg.training.lr_end is not None or cfg.training.lr_end_pct is not None:
        lr_end = (
            cfg.training.lr_end
            if cfg.training.lr_end is not None
            else (cfg.training.lr_end_pct * cfg.training.lr)
        )

    if cfg.training.scheduler == "warmup_cosine":
        scheduler = WarmupCosine(
            optimizer,
            lr_start=cfg.training.lr_start,
            lr_max=cfg.training.lr,
            lr_end=lr_end,
            warmup_steps=warmup_steps,
            T=cfg.training.steps_budget,
        )

    elif cfg.training.scheduler == "wsd":
        cooldown_start_step = cfg.training.steps_budget - cooldown_steps
        scheduler = WSD(
            optimizer,
            lr_start=cfg.training.lr_start,
            lr_max=cfg.training.lr,
            lr_end=lr_end,
            warmup_steps=warmup_steps,
            cooldown_start_step=cooldown_start_step,
            cooldown_steps=cooldown_steps,
        )

    elif cfg.training.scheduler == "warmup_constant":
        scheduler = WarmupConstant(
            optimizer,
            lr_start=cfg.training.lr_start,
            lr_max=cfg.training.lr,
            warmup_steps=warmup_steps,
        )

    elif cfg.training.scheduler == "linear_cooldown":
        cooldown_start_step = cfg.checkpoint.resume_step
        scheduler = LinearCooldown(
            optimizer,
            lr_max=cfg.training.lr,
            lr_end=lr_end,
            cooldown_start_step=cooldown_start_step,
            cooldown_steps=cooldown_steps,
        )

    else:
        raise NotImplementedError(f"Not implemented scheduler: {cfg.training.scheduler}.")

    return scheduler
