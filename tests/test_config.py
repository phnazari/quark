"""Tests for Hydra config composition."""

from hydra import compose, initialize
from omegaconf import OmegaConf

# -- Structure tests ---------------------------------------------------------


def test_default_config_has_all_groups(cfg):
    """Default config should contain all top-level groups."""
    for group in ("model", "data", "training", "system", "logging", "checkpoint"):
        assert group in cfg, f"Missing config group: {group}"


def test_no_flat_keys_at_root(cfg):
    """No training/system/logging/checkpoint keys should leak to the root."""
    root_keys = set(OmegaConf.to_container(cfg, resolve=True).keys())
    expected_root = {"model", "data", "training", "system", "logging", "checkpoint", "out_dir"}
    assert root_keys == expected_root, f"Unexpected root keys: {root_keys - expected_root}"


# -- Training config ---------------------------------------------------------


def test_training_keys(cfg):
    """Training group should contain all expected keys."""
    expected = {
        "steps_budget",
        "micro_batch_size",
        "eval_every_steps",
        "log_every_steps",
        "grad_accumulation_steps",
        "num_workers",
        "sampler",
        "sampler_seed",
        "optim",
        "fused_optim",
        "lr",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
        "eps",
        "scheduler",
        "warmup_steps",
        "cooldown_steps",
        "lr_start",
        "lr_end",
        "lr_end_pct",
        "early_stopping_patience",
    }
    actual = set(OmegaConf.to_container(cfg.training, resolve=True).keys())
    assert expected == actual, f"Missing: {expected - actual}, Extra: {actual - expected}"


def test_training_default_values(cfg):
    """Spot-check default training values."""
    assert cfg.training.optim == "adamw"
    assert cfg.training.scheduler == "warmup_cosine"
    assert cfg.training.lr == 7e-4
    assert cfg.training.grad_accumulation_steps == 2
    assert cfg.training.micro_batch_size == 16
    assert cfg.training.sampler == "stateful_random"


# -- System config -----------------------------------------------------------


def test_system_keys(cfg):
    """System group should contain expected keys."""
    expected = {"dtype", "compile_model", "seed", "ddp_backend"}
    actual = set(OmegaConf.to_container(cfg.system, resolve=True).keys())
    assert expected == actual


def test_system_default_values(cfg):
    """Spot-check default system values."""
    assert cfg.system.dtype == "bfloat16"
    assert cfg.system.compile_model is False
    assert cfg.system.seed == 42


# -- Logging config ----------------------------------------------------------


def test_logging_keys(cfg):
    """Logging group should contain expected keys."""
    expected = {"wandb_log", "wandb_project", "wandb_log_layer_stats"}
    actual = set(OmegaConf.to_container(cfg.logging, resolve=True).keys())
    assert expected == actual


def test_logging_default_values(cfg):
    """Spot-check default logging values."""
    assert cfg.logging.wandb_project == "quark"
    assert cfg.logging.wandb_log is True


# -- Checkpoint config -------------------------------------------------------


def test_checkpoint_keys(cfg):
    """Checkpoint group should contain expected keys."""
    expected = {
        "save_last_checkpoint",
        "save_intermediate_checkpoints",
        "save_every_steps",
        "resume",
        "resume_step",
        "resume_exp_name",
        "over_write",
        "exp_name",
    }
    actual = set(OmegaConf.to_container(cfg.checkpoint, resolve=True).keys())
    assert expected == actual


def test_checkpoint_default_values(cfg):
    """Spot-check default checkpoint values."""
    assert cfg.checkpoint.resume is False
    assert cfg.checkpoint.exp_name == "default"
    assert cfg.checkpoint.save_last_checkpoint is True


# -- Data config -------------------------------------------------------------


def test_data_keys(cfg):
    """Data group should contain expected keys."""
    expected = {
        "dataset",
        "vocab_size",
        "trainset_path",
        "validset_path",
        "seq_len",
        "eval",
        "valid_tokens",
    }
    actual = set(OmegaConf.to_container(cfg.data, resolve=True).keys())
    assert expected == actual


def test_data_has_vocab_size(cfg):
    """Data config should define vocab_size."""
    assert cfg.data.vocab_size == 50304


def test_data_does_not_have_training_fields(cfg):
    """Data config should not contain fields that moved to training."""
    data_keys = set(OmegaConf.to_container(cfg.data, resolve=True).keys())
    for field in ("micro_batch_size", "num_workers", "sampler", "sampler_seed"):
        assert field not in data_keys, f"'{field}' should be in training, not data"


# -- Model configs -----------------------------------------------------------


def test_default_model_is_transformer(cfg):
    """Default model should be transformer."""
    assert cfg.model.model_type == "transformer"
    assert cfg.model.hidden_size == 384
    assert cfg.model.num_layers == 21


def test_delta_net_model_override(cfg_delta_net):
    """model=delta_net should switch to delta_net config."""
    assert cfg_delta_net.model.model_type == "delta_net"
    assert cfg_delta_net.model.hidden_size == 1024
    assert cfg_delta_net.model.num_layers == 23


def test_model_override_preserves_other_groups(cfg_delta_net):
    """Switching model should not affect other config groups."""
    assert cfg_delta_net.training.optim == "adamw"
    assert cfg_delta_net.system.dtype == "bfloat16"
    assert cfg_delta_net.data.vocab_size == 50304


# -- CLI overrides -----------------------------------------------------------


def test_cli_override_training():
    """CLI overrides should propagate to the correct group."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["training.lr=1e-4"])
        assert cfg.training.lr == 1e-4


def test_cli_override_system():
    """CLI override for system group should work."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["system.seed=123"])
        assert cfg.system.seed == 123


def test_cli_override_checkpoint():
    """CLI override for checkpoint group should work."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=["checkpoint.exp_name=my_experiment", "checkpoint.resume=true"],
        )
        assert cfg.checkpoint.exp_name == "my_experiment"
        assert cfg.checkpoint.resume is True


def test_cli_override_multiple():
    """Multiple CLI overrides should all take effect."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=delta_net",
                "training.lr=1e-3",
                "training.steps_budget=5000",
                "system.compile_model=true",
            ],
        )
        assert cfg.model.model_type == "delta_net"
        assert cfg.training.lr == 1e-3
        assert cfg.training.steps_budget == 5000
        assert cfg.system.compile_model is True
