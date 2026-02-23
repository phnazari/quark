r"""Prepare datasets for training.

Downloads, tokenizes, chunks, and splits a HuggingFace dataset into
train/valid Arrow datasets ready for training.

These three ops can be done independently:
    python -m data.datasets.prepare --download
    python -m data.datasets.prepare --tokenize
    python -m data.datasets.prepare --chunk

Or all at once:
    python -m data.datasets.prepare --download --tokenize --chunk

Example: preprocess FineWebEdu 10BT sample:
    python -m data.datasets.prepare \
        --out_path data/fineweb10B \
        --download --tokenize --chunk \
        --save_tokenized --save_tokenizer \
        --dataset_path HuggingFaceFW/fineweb-edu \
        --dataset_split train \
        --dataset_name sample-10BT \
        --tokenizer gpt2 \
        --seq_length 2048 \
        --split_train_valid \
        --n_tokens_valid 10000000

Output structure:
    {out_path}/
    ├── raw_dataset/
    ├── tokenized_{tokenizer}/
    │   ├── tokenizer/
    │   ├── tokenized_dataset/
    │   └── ctx_{seq_length}/
    │       ├── train/
    │       └── valid/
"""

import argparse
import os
from functools import partial
from timeit import default_timer as timer

from dotenv import load_dotenv

load_dotenv()

# This will override FileLock globally to use SoftFileLock.
# Uncomment on filesystems do not support FileLock.
import filelock

filelock.FileLock = filelock.SoftFileLock
os.environ["SOFT_FILELOCK"] = "1"


from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer

from data.datasets.data_prep_utils import concat_chunk


def tokenize_batched(examples, tokenizer):
    """Tokenize a batch of examples, adding BOS/EOS tokens."""
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token

    def add_special(seq):
        return (bos_token + seq + eos_token) if seq else seq

    return tokenizer(
        [add_special(seq) for seq in examples["text"]],
        add_special_tokens=False,
        return_special_tokens_mask=False,
        return_attention_mask=False,
    )


def main():  # noqa: C901
    """Run data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare datasets for training")

    # Paths
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--cache_path", type=str, default=None)

    # Stage flags
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--chunk", action="store_true")

    # Dataset
    parser.add_argument("--dataset_path", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_columns", nargs="*", default=None)

    # Download
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--nrows", type=int, default=None)

    # Tokenization
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument(
        "--num_proc", type=int, default=8, help="Number of processes for data processing"
    )

    # Splitting
    parser.add_argument("--split_train_valid", action="store_true")
    parser.add_argument("--n_tokens_valid", type=int, default=None)

    # Save options
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--save_tokenized", action="store_true")
    parser.add_argument("--save_tokenizer", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    tokenizer_name = args.tokenizer.replace("/", "_") if args.tokenizer else None
    map_setup = dict(batched=True, batch_size=1024, num_proc=args.num_proc)

    raw_ds = None
    tokenized_ds = None

    # -----------------------------------------------------------------
    # Download
    if args.download:
        t0 = timer()

        raw_ds = load_dataset(
            args.dataset_path,
            split=args.dataset_split,
            name=args.dataset_name,
            streaming=args.streaming,
            cache_dir=args.cache_path,
            **({"columns": args.dataset_columns} if args.dataset_columns else {}),
        )

        if args.nrows is not None:
            raw_ds = raw_ds.take(args.nrows)

        if args.streaming:
            print("Converting IterableDataset to Dataset.")

            def custom_generator(iterable_ds):
                yield from iterable_ds

            raw_ds = Dataset.from_generator(
                partial(custom_generator, raw_ds),
                features=raw_ds.features,
            )

        if args.save_raw:
            raw_path = os.path.join(args.out_path, "raw_dataset")
            if os.path.exists(raw_path):
                raise FileExistsError("Raw dataset already exists.")
            print("Saving Raw Dataset")
            raw_ds.save_to_disk(raw_path)

        print(f"Downloading time: {(timer() - t0) // 60:.0f} min")

    # -----------------------------------------------------------------
    # Tokenize
    if args.tokenize:
        t0 = timer()

        if raw_ds is None:
            raw_ds = load_from_disk(os.path.join(args.out_path, "raw_dataset"))

        raw_ds = raw_ds.shuffle(seed=1996)

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"Length of tokenizer = {len(tokenizer)}")
        tokenizer.model_max_length = int(1e30)

        print("Tokenizing.")
        tokenized_ds = raw_ds.map(
            partial(tokenize_batched, tokenizer=tokenizer),
            remove_columns=["text"],
            **map_setup,
        )

        if args.seq_length is not None:
            tokenizer.model_max_length = args.seq_length

        if args.save_tokenized:
            tok_path = os.path.join(
                args.out_path, f"tokenized_{tokenizer_name}", "tokenized_dataset"
            )
            if os.path.exists(tok_path):
                raise FileExistsError("Tokenized dataset already exists.")
            print("Saving Tokenized Dataset")
            tokenized_ds.save_to_disk(tok_path)

        if args.save_tokenizer:
            tok_save_path = os.path.join(args.out_path, f"tokenized_{tokenizer_name}", "tokenizer")
            if os.path.exists(tok_save_path):
                raise FileExistsError("Tokenizer already exists.")
            print("Saving Tokenizer")
            tokenizer.save_pretrained(tok_save_path)

        print(f"Tokenization time: {(timer() - t0) // 60:.0f} min")

    # -----------------------------------------------------------------
    # Chunk
    if args.chunk:
        t0 = timer()

        if tokenized_ds is None:
            tokenized_ds = load_from_disk(
                os.path.join(args.out_path, f"tokenized_{tokenizer_name}", "tokenized_dataset")
            )

        tokenized_ds = tokenized_ds.remove_columns(
            [c for c in tokenized_ds.column_names if c != "input_ids"]
        )

        max_seq_length = args.seq_length + 1
        print(f"Concatenating and chunking into sequences of length {max_seq_length}.")
        chunked_ds = tokenized_ds.map(
            partial(concat_chunk, max_seq_length=max_seq_length), **map_setup
        )
        print(f"Number of tokens in chunked_ds: {len(chunked_ds) * max_seq_length:_}")
        chunked_ds.set_format("torch")

        print(f"Chunkization time: {(timer() - t0) // 60:.0f} min")

        # -------------------------------------------------------------
        # Split or save as-is
        if not args.split_train_valid:
            save_path = os.path.join(
                args.out_path,
                f"tokenized_{tokenizer_name}",
                f"ctx_{args.seq_length}",
                args.dataset_split,
            )
            chunked_ds.save_to_disk(save_path)
        else:
            n_chunks_valid = args.n_tokens_valid // max_seq_length
            valid_ds = chunked_ds.select(range(n_chunks_valid))
            train_ds = chunked_ds.select(range(n_chunks_valid, len(chunked_ds)))

            print(f"Number of tokens in train_ds: {len(train_ds) * max_seq_length:_}")
            print(f"Number of tokens in valid_ds: {len(valid_ds) * max_seq_length:_}")

            train_ds = train_ds.shuffle(seed=96)
            valid_ds = valid_ds.shuffle(seed=96)

            train_path = os.path.join(
                args.out_path,
                f"tokenized_{tokenizer_name}",
                f"ctx_{args.seq_length}",
                "train",
            )
            valid_path = os.path.join(
                args.out_path,
                f"tokenized_{tokenizer_name}",
                f"ctx_{args.seq_length}",
                "valid",
            )

            if os.path.exists(train_path):
                raise FileExistsError("Trainset already exists.")
            if os.path.exists(valid_path):
                raise FileExistsError("Validset already exists.")

            print(f"Saving trainset to {train_path}")
            train_ds.save_to_disk(train_path)
            print(f"Saving validset to {valid_path}")
            valid_ds.save_to_disk(valid_path)

    print("Successful completion.")


if __name__ == "__main__":
    main()
