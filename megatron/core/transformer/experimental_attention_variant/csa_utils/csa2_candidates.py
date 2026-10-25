# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compact CSA2 candidate blocks for packed and unpacked sequences."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# Mirrors csa2.py's _SELECT_ROW_CHUNK_ELEMENTS: rows per stable-sort chunk, as
# an element budget, so the CUDA sort workspace stays bounded.
_SELECT_ROW_CHUNK_ELEMENTS = 1 << 24

# Mirrors csa2.py's _MASK_ROW_CHUNK_ELEMENTS: rows per row-chunk when a bool mask
# is expanded onto a [rows, keys] slab, as a bool-element budget (1 byte per
# element, so a 16 MiB live slab).
_MASK_ROW_CHUNK_ELEMENTS = 16 * 1024**2


@dataclass(frozen=True)
class CSA2CandidateBlocks:
    """Sorted local block IDs; invalid slots are -1 after the valid IDs."""

    indices: Tensor
    block_size: int

    def to_mask(self, num_keys: int) -> Tensor:
        """Expand only while scoring a consumer; the shared state retains block IDs."""
        shape = (*self.indices.shape[:-1], num_keys)
        if num_keys == 0 or self.indices.shape[-1] == 0:
            return torch.zeros(shape, dtype=torch.bool, device=self.indices.device)
        num_blocks = (num_keys + self.block_size - 1) // self.block_size
        flags = torch.zeros(
            (*self.indices.shape[:-1], num_blocks + 1), dtype=torch.bool, device=self.indices.device
        )
        # Invalid slots write to a discarded column instead of overwriting block zero.
        flags.scatter_(-1, self.indices.masked_fill(self.indices < 0, num_blocks).long(), True)
        return flags[..., :num_blocks].repeat_interleave(self.block_size, -1)[..., :num_keys]

    def to_mask_thd(
        self, query_sequence_ids: Tensor, key_sequence_ids: Tensor, key_local_positions: Tensor
    ) -> Tensor:
        """Expand sequence-local block IDs at physical packed key addresses."""
        num_keys = key_local_positions.numel()
        if num_keys == 0 or self.indices.shape[-1] == 0:
            return torch.zeros(
                (*self.indices.shape[:-1], num_keys), dtype=torch.bool, device=self.indices.device
            )
        num_blocks = (num_keys + self.block_size - 1) // self.block_size
        flags = torch.zeros(
            (*self.indices.shape[:-1], num_blocks + 1), dtype=torch.bool, device=self.indices.device
        )
        # Avoid a [query, candidate, key] broadcast for long packed sequences.
        invalid = (self.indices < 0) | (self.indices >= num_blocks)
        flags.scatter_(-1, self.indices.masked_fill(invalid, num_blocks).long(), True)
        key_blocks = key_local_positions.div(self.block_size, rounding_mode="floor").long()
        # Row-chunked gather + compare. `indices` is [batch, rows, candidates] and
        # the packed THD select runs batch 1, so chunking the batch dimension left
        # the whole [rows, num_keys] bool matrix live -- the gather slab, the
        # sequence-membership comparison and their `&`, ~2.7 GiB each at 9.5K packed
        # rows x compressed keys. Chunk the flattened row dimension instead, so one
        # bool row-slab is live per step; gather + compare is elementwise, so the
        # packed [batch, rows, keys] mask is unchanged.
        leading = self.indices.shape[:-1]
        n_seq = self.indices.shape[-2] if len(leading) >= 2 else 1
        n_rows = self.indices[..., 0].numel()
        batch = n_rows // n_seq if n_seq else 0
        out = torch.empty((*leading, num_keys), dtype=torch.bool, device=self.indices.device)
        flat_flags = flags.reshape(n_rows, flags.shape[-1])
        flat_out = out.reshape(n_rows, num_keys)
        # Rows are batch-major and the sequence-membership test ignores the batch,
        # so a batch-major repeat of the per-row query sequence ids is exact.
        row_query = query_sequence_ids.repeat(batch) if batch > 1 else query_sequence_ids
        blocks = key_blocks.unsqueeze(0).expand(n_rows, num_keys)
        row_chunk = max(1, _MASK_ROW_CHUNK_ELEMENTS // max(num_keys, 1))
        for start in range(0, n_rows, row_chunk):
            end = min(start + row_chunk, n_rows)
            slab = flat_flags[start:end].gather(-1, blocks[start:end])
            slab &= row_query[start:end, None] == key_sequence_ids[None, :]
            flat_out[start:end] = slab
        return out


@torch.no_grad()
def candidate_blocks_from_block_scores(
    block_scores: Tensor, visible: Tensor | int, topk_blocks: int, block_size: int
) -> CSA2CandidateBlocks:
    """Rank precomputed causal block maxima (the tail of candidate_blocks_from_scores).

    ``block_scores`` is already each row's per-block amax over its sequence-local
    key blocks; ``visible`` broadcasts against block rows with a trailing
    singleton dimension. Earlier block IDs win equal scores and the newest
    visible block is forced, exactly as in candidate_blocks_from_scores.
    """
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("CSA2 candidate block count and size must be positive")
    num_blocks = block_scores.shape[-1]
    newest = (visible - 1) // block_size
    block_scores = block_scores.masked_fill(
        torch.arange(num_blocks, device=block_scores.device) == newest, torch.inf
    )
    keep_blocks = min(topk_blocks, num_blocks)
    # Zero blocks (no complete compression group) or zero rows keep the
    # pre-chunking contract, which reshape/cat below cannot express.
    if block_scores.numel() == 0:
        return CSA2CandidateBlocks(
            torch.empty(
                *block_scores.shape[:-1], keep_blocks, dtype=torch.int32, device=block_scores.device
            ),
            block_size,
        )
    # A stable per-row sort is row-independent, so chunk the rows: one
    # full-width argsort needs a workspace that scales with rows*num_blocks
    # (12 GiB at 512K context's 16384x65536) and OOMs. Same discipline as
    # CSA2Indexer.select_indices.
    flat = block_scores.reshape(-1, num_blocks)
    row_chunk = max(1, _SELECT_ROW_CHUNK_ELEMENTS // max(num_blocks, 1))
    parts = []
    for start in range(0, flat.shape[0], row_chunk):
        block = flat[start : start + row_chunk]
        # Clone the slice: `[..., :keep_blocks]` of the argsort result is a *view*
        # that keeps the whole full-width int64 sort output alive. Chunking bounds
        # the sort workspace but not what is retained afterwards, so at 512K the
        # collected views still hold rows*num_blocks*8 bytes per chunk sum
        # (measured on CPU for the pinned shapes: a 256 B slice retains 4096 B).
        # The copy keeps the same stable order and lets the full-width result go.
        parts.append(block.argsort(dim=-1, descending=True, stable=True)[..., :keep_blocks].clone())
    ids = torch.cat(parts, dim=0).reshape(*block_scores.shape[:-1], keep_blocks)
    valid = block_scores.gather(-1, ids) > -torch.inf
    sentinel = torch.iinfo(torch.int32).max
    ids = ids.masked_fill(~valid, sentinel).sort(-1).values
    return CSA2CandidateBlocks(ids.masked_fill(ids == sentinel, -1).int(), block_size)


@torch.no_grad()
def candidate_blocks_from_scores(
    scores: Tensor, visible: Tensor | int, topk_blocks: int, block_size: int
) -> CSA2CandidateBlocks:
    """Rank causal block maxima, forcing the newest visible block and stable ties.

    ``visible`` broadcasts against score rows with a trailing singleton dimension.
    Earlier block IDs win equal scores, independent of masked future capacity.
    """
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("CSA2 candidate block count and size must be positive")
    padded = F.pad(scores, (0, -scores.shape[-1] % block_size), value=-torch.inf)
    num_blocks = padded.shape[-1] // block_size
    block_scores = padded.unflatten(-1, (num_blocks, block_size)).amax(-1)
    return candidate_blocks_from_block_scores(block_scores, visible, topk_blocks, block_size)
