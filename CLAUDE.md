# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Quark** — a vanilla language modeling playground built around a standard GPT-style transformer. Clean, minimal infrastructure for experimenting with language model training.

## Commands

```bash
# Setup
uv sync

# Prepare data (FineWeb-Edu 10B, required before first training run)
.venv/bin/python -m data.datasets.prepare \
  --dataset_path HuggingFaceFW/fineweb-edu \
  --dataset_name sample-10BT \
  --tokenizer gpt2 \
  --seq_length 256 \
  --out_path data/fineweb10B \
  --n_tokens_valid 10000000

# Train
.venv/bin/python train.py

# Override params from CLI
.venv/bin/python train.py training.steps_budget=10000

# Select optimizer and scheduler
.venv/bin/python train.py training.optim=nadamw
.venv/bin/python train.py training.scheduler=wsd training.cooldown_steps=100

# Multi-GPU training
torchrun --standalone --nproc_per_node=4 train.py

# Show resolved config
.venv/bin/python train.py --cfg job
```

## Python Environment

- Virtual environment at `.venv`; use `uv` for package management (not pip)
- Run scripts with `.venv/bin/python` or activate with `source .venv/bin/activate`

## Linting

- Pre-commit hook runs `ruff check --fix` and `ruff format` automatically on commit
- Do not run ruff check manually; the hook handles it
- Ruff config: Google-style docstrings, line length 100, Python 3.12+

## Architecture

### Model (`models/`)

- **`transformer.py`** — `Transformer` (`TransformerConfig`): standard pre-norm GPT with causal self-attention and SwiGLU MLP blocks.
- **`__init__.py`** — model registry; import and re-export models here when adding new ones.
- **`utils.py`** — shared model utilities: `RMSNorm`, `init_gpt_weights`.

To add a new model: create `models/my_model.py`, export it from `models/__init__.py`, add a branch in `build_model()` in `train.py`, and add a `configs/model/my_model.yaml`.

### Training Infrastructure

- **`train.py`** — slim orchestrator using Hydra for config.
- **`engine/engine.py`** — `TorchEngine`: encapsulates model, optimizer, scheduler, grad scaler, AMP. Handles forward/backward/step with gradient accumulation and DDP sync.
- **`optim/`** — optimizer initialization (`adamw`, `nadamw`, `sgd`, `signSGD`) and LR schedules (`warmup_cosine`, `wsd`, `warmup_constant`, `linear_cooldown`).
- **`data/`** — HF datasets DataLoaders with stateful samplers for resumable training. Data is stored in Arrow format via the `datasets` library.
- **`torch_utils.py`** — DDP setup, device detection, seed initialization.
- **`checkpoint_utils.py`** — checkpoint save/load with full state (model, optimizer, scheduler, scaler).
- **`utils.py`** — config flattening (Hydra nested → flat namespace), logging, param groups.

YAML configs live in `configs/` with groups for `model/` and `data/`. Supports DDP, AMP (bfloat16/float16), gradient accumulation, multiple LR schedules, checkpoint resume, and W&B logging.

## Code Style

- **Minimality**: Prefer minimal, functional implementations; don't be verbose
- **Early Returns**: Use early returns to avoid nesting
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **DRY**: Don't repeat yourself
- **Function Ordering**: Define composing functions before their components
- **Keep changes focused**: Only modify code related to the task at hand
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Iterative development**: Start minimal and verify before adding complexity
- **Clean logic**: Keep core logic clean; push implementation details to the edges
- **Simplicity**: Prioritize readability over clever solutions
- **File Organization**: Balance organization with simplicity for project scale
