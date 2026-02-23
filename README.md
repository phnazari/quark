# Quark

Vanilla language modeling playground. Standard pre-norm GPT (causal attention + SwiGLU MLP) with a clean training pipeline built on Hydra, W&B, and DDP.

## Setup

```bash
uv sync
uv run pre-commit install
```

## Data

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

```bash
# Single GPU
.venv/bin/python train.py

# Multi-GPU
torchrun --standalone --nproc_per_node=4 train.py

# Override config
.venv/bin/python train.py training.lr=1e-4 training.steps_budget=10000
```

Config lives in `configs/`. Training pipeline adapted from [PlainLM](https://github.com/Niccolo-Ajroldi/plainLM).
