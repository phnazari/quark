#!/bin/bash
# Launcher script called by train.sub.
# Any extra Hydra overrides are forwarded as positional arguments.

cd ~/workspace/quark

uv run python -m torch.distributed.run --standalone --nproc_per_node=4 \
  train.py \
  model=transformer \
  "$@"
