"""Shared fixtures for tests."""

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig


@pytest.fixture()
def cfg() -> DictConfig:
    """Return default resolved config."""
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="config")


@pytest.fixture()
def cfg_delta_net() -> DictConfig:
    """Return config with model=delta_net."""
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="config", overrides=["model=delta_net"])
