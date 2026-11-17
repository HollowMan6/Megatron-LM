# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed row coordinates and reconstruction for outer-sequence chunking."""

import torch
from torch.distributed import ProcessGroup


def get_outer_chunk_cp_coordinates(
    *, cp_rank: int, cp_size: int, chunk_index: int, chunk_count: int
) -> tuple[int, int]:
    """Apply the chunk offset to the layout while keeping the CP group physical."""
    if not 0 <= chunk_index < chunk_count:
        raise ValueError(
            f"CP chunk index outside the chunk count: cp_chunk_index={chunk_index}, "
            f"cp_chunk_count={chunk_count}"
        )
    return chunk_index * cp_size + cp_rank, chunk_count * cp_size


def restore_outer_chunk_thd_rows(
    output: torch.Tensor,
    *,
    cp_group: ProcessGroup,
    chunk_index: int,
    chunk_count: int,
    sequence_length: int,
    padded_length: int,
    cp_layout: str,
) -> torch.Tensor:
    """Restore one packed sequence's rows, zeroing positions from other chunks.

    Keep the local shard differentiable. Other ranks have their own replay graphs,
    so only exchange their tensor data on the physical CP group.
    """
    physical_size = cp_group.size()
    first_rank, virtual_size = get_outer_chunk_cp_coordinates(
        cp_rank=0, cp_size=physical_size, chunk_index=chunk_index, chunk_count=chunk_count
    )
    local_rows = output[0].contiguous()
    shards = [torch.empty_like(local_rows) for _ in range(physical_size)]
    torch.distributed.all_gather(shards, local_rows.detach(), group=cp_group)
    shards[cp_group.rank()] = local_rows

    restored = local_rows.new_zeros((sequence_length, *local_rows.shape[1:]))

    def place(rows: torch.Tensor, start: int) -> None:
        # Even an all-padding pass needs a graph edge back to its local shard.
        stop = min(sequence_length, start + rows.shape[0])
        restored[start:stop] = rows[: max(0, stop - start)]

    if cp_layout == "contiguous":
        place(torch.cat(shards, dim=0), first_rank * local_rows.shape[0])
    elif cp_layout == "zigzag":
        shard_length = padded_length // virtual_size
        half = shard_length // 2
        place(torch.cat([shard[:half] for shard in shards], dim=0), first_rank * half)
        back = shard_length * virtual_size - (first_rank + physical_size) * half
        place(torch.cat([shard[half:shard_length] for shard in reversed(shards)], dim=0), back)
    else:
        raise ValueError(f"Unsupported context parallel layout: {cp_layout}")
    return restored
