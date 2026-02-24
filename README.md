# Quark

A minimal playground for language modeling research. The goal is to provide a clean, hackable base for training and experimenting with GPT-style models — without the overhead of a large framework. Ships with a standard pre-norm transformer (causal attention + SwiGLU MLP) and a training pipeline built on Hydra, W&B, and DDP.

## Setup

Clone the repo with submodules (required for flash-linear-attention):

```bash
git clone --recurse-submodules https://github.com/philippnazari/quark
```

If you already cloned without `--recurse-submodules`, fetch the submodules with:

```bash
git submodule update --init --recursive
```

Then install dependencies:

```bash
uv sync
uv run pre-commit install
```

For development (adds ruff and pre-commit):

```bash
uv sync --extra dev
```

## Data

Download and tokenize FineWeb-Edu 10B into chunked Arrow files. Only needs to be run once — the result is reused across training runs.

```bash
.venv/bin/python -m data.datasets.prepare \
  --dataset_path HuggingFaceFW/fineweb-edu \
  --dataset_name sample-10BT \
  --tokenizer gpt2 \
  --seq_length 256 \
  --out_path data/fineweb10B \
  --n_tokens_valid 10000000
```

## Training

Config is managed by Hydra (`configs/`). All keys can be overridden from the CLI.

```bash
# Single GPU
.venv/bin/python train.py

# Multi-GPU (DDP)
torchrun --standalone --nproc_per_node=4 train.py

# Override config values
.venv/bin/python train.py training.lr=1e-4 training.steps_budget=10000

# Print resolved config without running
.venv/bin/python train.py --cfg job
```

Training logs to W&B and optionally saves checkpoints to `out_dir/exp_name` (configured in `configs/config.yaml`).

Training pipeline adapted from [PlainLM](https://github.com/Niccolo-Ajroldi/plainLM).
