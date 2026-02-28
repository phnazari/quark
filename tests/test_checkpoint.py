"""Tests for checkpoint save/load utilities."""

import os
import tempfile

import torch
from checkpoint_utils import maybe_load_checkpoint, save_checkpoint
from omegaconf import OmegaConf


def _make_checkpoint_cfg(tmpdir, exp_name="test_exp", resume=False, resume_step=None):
    """Create a minimal config for checkpoint tests."""
    return OmegaConf.create(
        {
            "out_dir": tmpdir,
            "checkpoint": {
                "exp_name": exp_name,
                "resume": resume,
                "resume_step": resume_step,
                "resume_exp_name": None,
                "save_intermediate_checkpoints": True,
                "save_last_checkpoint": True,
                "over_write": True,
            },
        }
    )


def _dummy_engine():
    """Create a minimal mock engine with optimizer, scheduler, scaler."""

    class MockEngine:
        pass

    param = torch.nn.Parameter(torch.zeros(4))
    engine = MockEngine()
    engine.optimizer = torch.optim.SGD([param], lr=0.01)
    engine.scheduler = None
    engine.scaler = torch.amp.GradScaler(enabled=False)
    return engine


def test_save_and_load_checkpoint():
    """Should save and reload a checkpoint correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = torch.nn.Linear(4, 4)
        engine = _dummy_engine()
        cfg = _make_checkpoint_cfg(tmpdir)

        save_checkpoint(step=100, model=model, engine=engine, cfg=cfg, metrics={"loss": [1.0]})

        # Verify file was created
        ckpt_path = os.path.join(tmpdir, "test_exp", "ckpt_step_100.pth")
        assert os.path.exists(ckpt_path)

        # Load it back
        load_cfg = _make_checkpoint_cfg(tmpdir, resume=True, resume_step=100)
        ckpt = maybe_load_checkpoint(load_cfg)

        assert ckpt is not None
        assert ckpt["step"] == 100
        assert "state_dict" in ckpt
        assert "optimizer" in ckpt


def test_maybe_load_checkpoint_returns_none_when_not_resuming():
    """Should return None when resume=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_checkpoint_cfg(tmpdir, resume=False)
        assert maybe_load_checkpoint(cfg) is None


def test_save_checkpoint_creates_metrics_file():
    """Should write a metrics.json alongside the checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = torch.nn.Linear(4, 4)
        engine = _dummy_engine()
        cfg = _make_checkpoint_cfg(tmpdir)

        metrics = {"train/loss": [2.5, 2.3, 2.1]}
        save_checkpoint(step=50, model=model, engine=engine, cfg=cfg, metrics=metrics)

        metrics_path = os.path.join(tmpdir, "test_exp", "metrics.json")
        assert os.path.exists(metrics_path)


def test_load_latest_checkpoint():
    """Should load the latest checkpoint when resume_step is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = torch.nn.Linear(4, 4)
        engine = _dummy_engine()
        cfg = _make_checkpoint_cfg(tmpdir)

        # Save two checkpoints
        save_checkpoint(step=100, model=model, engine=engine, cfg=cfg)
        save_checkpoint(step=200, model=model, engine=engine, cfg=cfg)

        # Load latest (resume_step=None)
        load_cfg = _make_checkpoint_cfg(tmpdir, resume=True, resume_step=None)
        ckpt = maybe_load_checkpoint(load_cfg)

        assert ckpt is not None
        assert ckpt["step"] == 200
