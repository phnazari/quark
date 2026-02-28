"""Tests for utility functions."""

import os
import tempfile

import torch
from omegaconf import OmegaConf
from utils import get_param_groups, get_variant_name, log, maybe_make_dir

# -- get_variant_name --------------------------------------------------------


def test_variant_name_transformer(cfg):
    """Transformer config should return 'Transformer'."""
    assert get_variant_name(cfg) == "Transformer"


def test_variant_name_delta_net(cfg_delta_net):
    """DeltaNet config should return 'DeltaNet'."""
    assert get_variant_name(cfg_delta_net) == "DeltaNet"


# -- get_param_groups --------------------------------------------------------


def test_param_groups_separates_decay():
    """Linear weights should decay, biases and norms should not."""
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8, bias=True),
        torch.nn.LayerNorm(8),
        torch.nn.Linear(8, 4, bias=False),
    )
    groups = get_param_groups(model, weight_decay=0.1)
    assert len(groups) == 2

    decay_group = groups[0]
    no_decay_group = groups[1]

    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    # Linear weights should be in decay group
    decay_numel = sum(p.numel() for p in decay_group["params"])
    no_decay_numel = sum(p.numel() for p in no_decay_group["params"])

    # 4*8 + 8*4 = 64 in decay (two Linear weights)
    assert decay_numel == 64
    # 8 (bias) + 8 + 8 (LayerNorm weight + bias) = 24 in no_decay
    assert no_decay_numel == 24


def test_param_groups_all_params_accounted():
    """Every parameter should be in exactly one group."""
    model = torch.nn.Sequential(
        torch.nn.Embedding(100, 16),
        torch.nn.Linear(16, 32),
        torch.nn.Linear(32, 16, bias=False),
    )
    groups = get_param_groups(model, weight_decay=0.05)
    group_params = set()
    for g in groups:
        for p in g["params"]:
            group_params.add(id(p))

    all_params = {id(p) for p in model.parameters()}
    assert group_params == all_params


# -- maybe_make_dir ----------------------------------------------------------


def test_maybe_make_dir_creates_directory():
    """Should create the experiment directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = OmegaConf.create(
            {
                "out_dir": tmpdir,
                "checkpoint": {
                    "save_intermediate_checkpoints": True,
                    "save_last_checkpoint": True,
                    "resume": False,
                    "resume_exp_name": None,
                    "over_write": True,
                    "exp_name": "test_exp",
                },
            }
        )
        maybe_make_dir(cfg)
        assert os.path.isdir(os.path.join(tmpdir, "test_exp"))


def test_maybe_make_dir_skips_when_no_checkpoints():
    """Should not create a directory when checkpointing is disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = OmegaConf.create(
            {
                "out_dir": tmpdir,
                "checkpoint": {
                    "save_intermediate_checkpoints": False,
                    "save_last_checkpoint": False,
                    "resume": False,
                    "resume_exp_name": None,
                    "over_write": True,
                    "exp_name": "should_not_exist",
                },
            }
        )
        maybe_make_dir(cfg)
        assert not os.path.exists(os.path.join(tmpdir, "should_not_exist"))


def test_maybe_make_dir_overwrites_existing():
    """Should remove and recreate when over_write is True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = os.path.join(tmpdir, "overwrite_exp")
        os.makedirs(exp_dir)
        marker = os.path.join(exp_dir, "old_file.txt")
        with open(marker, "w") as f:
            f.write("old")

        cfg = OmegaConf.create(
            {
                "out_dir": tmpdir,
                "checkpoint": {
                    "save_intermediate_checkpoints": True,
                    "save_last_checkpoint": True,
                    "resume": False,
                    "resume_exp_name": None,
                    "over_write": True,
                    "exp_name": "overwrite_exp",
                },
            }
        )
        maybe_make_dir(cfg)
        assert os.path.isdir(exp_dir)
        assert not os.path.exists(marker)


# -- log ---------------------------------------------------------------------


def test_log_populates_metrics(cfg):
    """Log should append metrics to the metrics dict."""
    metrics = {}
    # We need a mock optimizer with param_groups
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([dummy_param], lr=0.001)

    # Disable wandb for this test
    test_cfg = OmegaConf.merge(cfg, OmegaConf.create({"logging": {"wandb_log": False}}))

    train_loss = torch.tensor(2.5)

    log(
        test_cfg,
        metrics,
        micro_step=100,
        train_loss=train_loss,
        train_loss_array=[train_loss],
        valid_loss=None,
        optimizer=optimizer,
        world_size=1,
        grad_norm=torch.tensor(0.5),
    )

    assert "step" in metrics
    assert "train/loss" in metrics
    assert "train/grad_norm" in metrics
    assert len(metrics["step"]) == 1
    assert metrics["step"][0] == 100 // cfg.training.grad_accumulation_steps


def test_log_includes_throughput(cfg):
    """Log should include throughput and step_time when provided."""
    metrics = {}
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([dummy_param], lr=0.001)
    test_cfg = OmegaConf.merge(cfg, OmegaConf.create({"logging": {"wandb_log": False}}))
    train_loss = torch.tensor(2.5)

    log(
        test_cfg,
        metrics,
        micro_step=100,
        train_loss=train_loss,
        train_loss_array=[train_loss],
        valid_loss=None,
        optimizer=optimizer,
        world_size=1,
        throughput=50000.0,
        step_time=0.5,
    )

    assert "train/throughput" in metrics
    assert "train/step_time" in metrics
    assert metrics["train/throughput"][0] == 50000.0


def test_log_includes_valid_loss(cfg):
    """Log should include valid/loss when provided."""
    metrics = {}
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([dummy_param], lr=0.001)
    test_cfg = OmegaConf.merge(cfg, OmegaConf.create({"logging": {"wandb_log": False}}))
    train_loss = torch.tensor(2.5)

    log(
        test_cfg,
        metrics,
        micro_step=100,
        train_loss=train_loss,
        train_loss_array=[train_loss],
        valid_loss=1.8,
        optimizer=optimizer,
        world_size=1,
    )

    assert "valid/loss" in metrics
    assert "valid/ppl" in metrics
