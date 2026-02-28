"""Tests for optimizer and scheduler initialization."""

import torch
from omegaconf import OmegaConf
from optim import initialize_optimizer, initialize_scheduler


def _dummy_param_groups(lr=1e-3):
    """Create minimal param groups for testing."""
    param = torch.nn.Parameter(torch.zeros(4))
    return [{"params": [param], "weight_decay": 0.1}]


# -- Optimizer ----------------------------------------------------------------


def test_initialize_adamw(cfg):
    """Should create an AdamW optimizer from structured config."""
    groups = _dummy_param_groups()
    optimizer = initialize_optimizer(groups, cfg)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == cfg.training.lr


def test_initialize_nadamw(cfg):
    """Should create an NAdam optimizer when optim=nadamw."""
    test_cfg = OmegaConf.merge(cfg, OmegaConf.create({"training": {"optim": "nadamw"}}))
    groups = _dummy_param_groups()
    optimizer = initialize_optimizer(groups, test_cfg)
    assert isinstance(optimizer, torch.optim.NAdam)


def test_initialize_sgd(cfg):
    """Should create an SGD optimizer when optim=sgd."""
    test_cfg = OmegaConf.merge(cfg, OmegaConf.create({"training": {"optim": "sgd"}}))
    groups = _dummy_param_groups()
    optimizer = initialize_optimizer(groups, test_cfg)
    assert isinstance(optimizer, torch.optim.SGD)


# -- Scheduler ----------------------------------------------------------------


def test_initialize_warmup_cosine(cfg):
    """Should create a WarmupCosine scheduler."""
    groups = _dummy_param_groups()
    optimizer = torch.optim.SGD(groups, lr=1e-3)
    scheduler = initialize_scheduler(optimizer, cfg)
    assert scheduler is not None
    assert hasattr(scheduler, "step")
    assert hasattr(scheduler, "warmup_steps")


def test_initialize_wsd(cfg):
    """Should create a WSD scheduler."""
    test_cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "training": {
                    "scheduler": "wsd",
                    "cooldown_steps": 100,
                    "lr_end": 1e-5,
                },
            }
        ),
    )
    groups = _dummy_param_groups()
    optimizer = torch.optim.SGD(groups, lr=1e-3)
    scheduler = initialize_scheduler(optimizer, test_cfg)
    assert scheduler is not None


def test_initialize_warmup_constant(cfg):
    """Should create a WarmupConstant scheduler."""
    test_cfg = OmegaConf.merge(
        cfg, OmegaConf.create({"training": {"scheduler": "warmup_constant"}})
    )
    groups = _dummy_param_groups()
    optimizer = torch.optim.SGD(groups, lr=1e-3)
    scheduler = initialize_scheduler(optimizer, test_cfg)
    assert scheduler is not None


def test_scheduler_none(cfg):
    """Should return None when scheduler is null."""
    test_cfg = OmegaConf.merge(cfg, OmegaConf.create({"training": {"scheduler": None}}))
    groups = _dummy_param_groups()
    optimizer = torch.optim.SGD(groups, lr=1e-3)
    scheduler = initialize_scheduler(optimizer, test_cfg)
    assert scheduler is None


def test_scheduler_step_changes_lr(cfg):
    """Scheduler.step() should modify the optimizer's learning rate."""
    groups = _dummy_param_groups()
    optimizer = torch.optim.SGD(groups, lr=1e-3)
    scheduler = initialize_scheduler(optimizer, cfg)

    initial_lr = optimizer.param_groups[0]["lr"]
    # Step through warmup
    for _ in range(cfg.training.warmup_steps + 10):
        scheduler.step()
    post_warmup_lr = optimizer.param_groups[0]["lr"]

    # LR should have changed from initial (lr_start=0.0)
    assert initial_lr != post_warmup_lr
