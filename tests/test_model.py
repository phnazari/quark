"""Tests for model building and instantiation."""

import pytest
import torch
from models.transformer import Transformer, TransformerConfig

try:
    from models.delta_net import DeltaNet, DeltaNetWrapperConfig

    HAS_FLA = True
except ImportError:
    HAS_FLA = False


# -- build_model (transformer only, no triton needed) -------------------------


def test_build_model_transformer(cfg):
    """build_model should create a Transformer with correct dimensions from config."""
    vocab_size = cfg.data.vocab_size
    config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        head_dim=cfg.model.head_dim,
        block_size=cfg.data.seq_len,
        dropout=cfg.model.dropout,
        bias=cfg.model.bias,
    )
    model = Transformer(config)

    assert isinstance(model, Transformer)
    assert config.hidden_size == 384
    assert config.num_layers == 21
    assert config.num_heads == 8
    assert config.vocab_size == 50304


@pytest.mark.skipif(not HAS_FLA, reason="flash-linear-attention / triton not available")
def test_build_model_delta_net(cfg_delta_net):
    """build_model should create a DeltaNet with correct dimensions."""
    cfg = cfg_delta_net
    config = DeltaNetWrapperConfig(
        vocab_size=cfg.data.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
    )
    model = DeltaNet(config)

    assert isinstance(model, DeltaNet)
    assert config.hidden_size == 1024
    assert config.num_layers == 23


# -- Transformer forward ------------------------------------------------------


def test_transformer_forward():
    """Transformer forward should return (logits, loss) with correct shapes."""
    config = TransformerConfig(
        vocab_size=256, hidden_size=32, num_layers=2, num_heads=2, head_dim=16, block_size=64
    )
    model = Transformer(config)

    idx = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))

    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, 256)
    assert loss is not None
    assert loss.ndim == 0  # scalar


def test_transformer_forward_no_targets():
    """Transformer forward without targets should return loss=None."""
    config = TransformerConfig(
        vocab_size=256, hidden_size=32, num_layers=2, num_heads=2, head_dim=16, block_size=64
    )
    model = Transformer(config)

    idx = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx)
    assert logits.shape == (2, 16, 256)
    assert loss is None


def test_transformer_weight_tying():
    """Embedding and lm_head weights should be tied."""
    config = TransformerConfig(
        vocab_size=256, hidden_size=32, num_layers=2, num_heads=2, head_dim=16, block_size=64
    )
    model = Transformer(config)
    assert model.wte.weight is model.lm_head.weight
