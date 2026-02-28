"""Tests for TorchEngine initialization with structured config."""

import torch
from engine import TorchEngine
from models.transformer import Transformer, TransformerConfig
from omegaconf import OmegaConf


def _small_model():
    """Create a tiny transformer for testing."""
    config = TransformerConfig(
        vocab_size=256, hidden_size=32, num_layers=2, num_heads=2, head_dim=16, block_size=64
    )
    return Transformer(config)


def test_engine_init(cfg):
    """TorchEngine should initialize from structured config without errors."""
    # Override to avoid needing GPU / real data paths
    test_cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "system": {"compile_model": False, "dtype": "float32"},
                "checkpoint": {"resume": False},
                "data": {"seq_len": 64},
            }
        ),
    )
    model = _small_model()
    engine = TorchEngine(model, test_cfg, device="cpu", local_rank=None, ckpt=None)

    assert engine.seq_len == 64
    assert engine.accumulation_steps == test_cfg.training.grad_accumulation_steps
    assert engine.grad_clip == test_cfg.training.grad_clip
    assert engine.dtype == "float32"
    assert engine.optimizer is not None
    assert engine.scheduler is not None


def test_engine_step(cfg):
    """TorchEngine.step should return loss and grad_norm."""
    test_cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "system": {"compile_model": False, "dtype": "float32"},
                "checkpoint": {"resume": False},
                "data": {"seq_len": 16},
                "training": {"grad_accumulation_steps": 1},
            }
        ),
    )
    model = _small_model()
    engine = TorchEngine(model, test_cfg, device="cpu", local_rank=None, ckpt=None)

    # Create a fake batch
    batch = {"input_ids": torch.randint(0, 256, (2, 17))}  # seq_len + 1
    loss, grad_norm = engine.step(batch)

    assert loss is not None
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert grad_norm is not None


def test_engine_eval(cfg):
    """TorchEngine.eval should return average loss."""
    test_cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "system": {"compile_model": False, "dtype": "float32"},
                "checkpoint": {"resume": False},
                "data": {"seq_len": 16},
            }
        ),
    )
    model = _small_model()
    engine = TorchEngine(model, test_cfg, device="cpu", local_rank=None, ckpt=None)

    # Create a tiny DataLoader-like iterable
    batches = [{"input_ids": torch.randint(0, 256, (2, 17))} for _ in range(3)]
    avg_loss = engine.eval(batches)

    assert isinstance(avg_loss, float)
    assert avg_loss > 0
