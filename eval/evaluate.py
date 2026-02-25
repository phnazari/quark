"""Evaluation script for Quark models."""

import eval.harness  # noqa: F401
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
