"""Data preparation utilities for chunking tokenized datasets."""

from itertools import chain
from typing import Any, Dict, List

import torch


def intra_doc_causal_mask(doc_boundaries: list, max_seq_length: int, device="cpu") -> torch.Tensor:
    """Create a block diagonal causal mask for intra-document segments."""
    if sum(doc_boundaries) != max_seq_length:
        raise ValueError("Sum of doc_boundaries does not match max_seq_length.")

    sub_masks_bool = []
    for segment_length in doc_boundaries:
        segment_causal_mask_bool = torch.tril(
            torch.ones((segment_length, segment_length), dtype=torch.bool, device=device)
        )
        sub_masks_bool.append(segment_causal_mask_bool)

    return torch.block_diag(*sub_masks_bool)


def _get_docs_boundaries(
    doc_lengths: List[int], n_chunks: int, max_seq_length: int
) -> List[List[int]]:
    """Get the boundaries of documents in concatenated chunks.

    A list of documents has been concatenated and chunked into `n_chunks`
    of `max_seq_length`. Each original document had a different length,
    defined in `doc_lengths`.

    Returns a list of lists, where each inner list contains the lengths of
    document segments present in the corresponding chunk.
    """
    doc_boundaries = [[] for _ in range(n_chunks)]

    doc_idx = 0
    current_doc_remainder = 0

    for chunk_idx in range(n_chunks):
        current_chunk_filled_length = 0

        while current_chunk_filled_length < max_seq_length:
            if current_doc_remainder == 0:
                if doc_idx < len(doc_lengths):
                    current_doc_remainder = doc_lengths[doc_idx]
                    doc_idx += 1
                else:
                    break

            space_in_chunk = max_seq_length - current_chunk_filled_length
            amount_to_add = min(current_doc_remainder, space_in_chunk)

            doc_boundaries[chunk_idx].append(amount_to_add)
            current_chunk_filled_length += amount_to_add
            current_doc_remainder -= amount_to_add

    return doc_boundaries


def concat_chunk(examples: Dict[str, List[Any]], max_seq_length: int) -> Dict[str, List[Any]]:
    """Concatenate all texts and split them into chunks of max_seq_length."""
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length

    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }

    # Track document boundaries for intra-document masking
    original_docs_lengths = [len(example) for example in examples["input_ids"]]
    n_chunks = len(result["input_ids"])
    result["docs_lengths"] = _get_docs_boundaries(original_docs_lengths, n_chunks, max_seq_length)

    return result
