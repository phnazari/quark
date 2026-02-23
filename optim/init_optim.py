"""Initialize optimizer and scheduler."""

import torch

from .lr_schedule import WSD, LinearCooldown, WarmupConstant, WarmupCosine


def initialize_optimizer(param_groups, cfg):
    """Initialize an optimizer from config."""
    if cfg.optim == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg.lr,
            betas=[cfg.beta1, cfg.beta2],
            weight_decay=cfg.weight_decay,
            fused=cfg.fused_optim,
            eps=cfg.eps,
        )

    elif cfg.optim == "nadamw":
        kwargs = dict(
            lr=cfg.lr,
            betas=[cfg.beta1, cfg.beta2],
            weight_decay=cfg.weight_decay,
            decoupled_weight_decay=True,
            eps=cfg.eps,
        )
        # fused only supported on CUDA
        if cfg.fused_optim and torch.cuda.is_available():
            kwargs["fused"] = True
        optimizer = torch.optim.NAdam(param_groups, **kwargs)

    elif cfg.optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=cfg.lr,
            momentum=cfg.beta1,
            dampening=getattr(cfg, "dampening", 0.0),
            weight_decay=cfg.weight_decay,
        )

    elif cfg.optim == "signSGD":
        from .sign_sgd import signSGD

        optimizer = signSGD(
            param_groups,
            lr=cfg.lr,
            momentum=cfg.beta1,
            dampening=getattr(cfg, "dampening", 0.0),
            weight_decay=cfg.weight_decay,
        )

    else:
        raise NotImplementedError(f"Not implemented optim: {cfg.optim}.")

    return optimizer


def initialize_scheduler(optimizer, cfg):
    """Initialize a learning rate scheduler from config."""
    if cfg.scheduler is None:
        return None

    # Warmup steps: int or fraction of steps_budget
    warmup_steps = None
    if getattr(cfg, "warmup_steps", None) is not None:
        warmup_steps = (
            cfg.warmup_steps
            if isinstance(cfg.warmup_steps, int)
            else int(cfg.warmup_steps * cfg.steps_budget)
        )

    # Cooldown steps: int or fraction of steps_budget
    cooldown_steps = None
    if getattr(cfg, "cooldown_steps", None) is not None:
        cooldown_steps = (
            cfg.cooldown_steps
            if isinstance(cfg.cooldown_steps, int)
            else int(cfg.cooldown_steps * cfg.steps_budget)
        )

    # Final LR: direct or as fraction of peak lr
    lr_end = None
    if getattr(cfg, "lr_end", None) is not None or getattr(cfg, "lr_end_pct", None) is not None:
        lr_end = cfg.lr_end if (cfg.lr_end is not None) else (cfg.lr_end_pct * cfg.lr)

    if cfg.scheduler == "warmup_cosine":
        scheduler = WarmupCosine(
            optimizer,
            lr_start=cfg.lr_start,
            lr_max=cfg.lr,
            lr_end=lr_end,
            warmup_steps=warmup_steps,
            T=cfg.steps_budget,
        )

    elif cfg.scheduler == "wsd":
        cooldown_start_step = cfg.steps_budget - cooldown_steps
        scheduler = WSD(
            optimizer,
            lr_start=cfg.lr_start,
            lr_max=cfg.lr,
            lr_end=lr_end,
            warmup_steps=warmup_steps,
            cooldown_start_step=cooldown_start_step,
            cooldown_steps=cooldown_steps,
        )

    elif cfg.scheduler == "warmup_constant":
        scheduler = WarmupConstant(
            optimizer,
            lr_start=cfg.lr_start,
            lr_max=cfg.lr,
            warmup_steps=warmup_steps,
        )

    elif cfg.scheduler == "linear_cooldown":
        cooldown_start_step = cfg.resume_step
        scheduler = LinearCooldown(
            optimizer,
            lr_max=cfg.lr,
            lr_end=lr_end,
            cooldown_start_step=cooldown_start_step,
            cooldown_steps=cooldown_steps,
        )

    else:
        raise NotImplementedError(f"Not implemented scheduler: {cfg.scheduler}.")

    return scheduler
