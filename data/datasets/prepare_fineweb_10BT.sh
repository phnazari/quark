#!/bin/bash

# This script will download and preprocess FineWebEdu-100BT.
# Expect some token loss by batched concat_chunk.

mkdir -p /fast/pnazari/tmp
cd ~/workspace/quark

source /home/pnazari/workspace/quark/.venv/bin/activate

PYTHONPATH=. python data/datasets/prepare.py \
  --out_path="/fast/pnazari/data/lm/fwedu/fwedu_sample_10B_tokenizer_GPTNeoX" \
  --cache_path="/fast/pnazari/tmp" \
  --download --tokenize --chunk \
  --save_tokenized --save_tokenizer \
  --dataset_path="HuggingFaceFW/fineweb-edu" \
  --dataset_split="train" \
  --dataset_name="sample-10BT" \
  --tokenizer="EleutherAI/gpt-neox-20b" \
  --seq_length=2048 \
  --split_train_valid \
  --n_tokens_valid=10000000 \
  --num_proc 32