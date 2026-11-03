# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 local compression and canonical shared-K addressing on the DSv4 CP path."""

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from .thd_utils import CSA2THDCompressionLayout, CSA2THDLayout


def get_cp_compressor_capacity(local_rows: int, ratio: int) -> int:
    """Reserve aligned local compressed slots without reading device lengths."""
    group_alignment = 32 // math.gcd(32, ratio)
    capacity = max(1, (local_rows + ratio) // ratio)
    return (capacity + group_alignment - 1) // group_alignment * group_alignment


class _LeftBoundaryExchange(torch.autograd.Function):
    """Exchange one neighbor's fixed halo and return its gradients to the owner."""

    @staticmethod
    def forward(
        ctx,
        hidden: Tensor,
        halo_rows: int,
        cp_group: dist.ProcessGroup,
        chunk_tape=None,
        chunk_slot: int = -1,
    ) -> Tensor:
        """Exchange adjacent tails and snapshot the last rank's tail for the next pass."""
        if hidden.shape[0] < halo_rows:
            raise ValueError("CSA2 CP local rows must cover the attention halo")
        rank, size = cp_group.rank(), cp_group.size()
        ctx.cp_group = cp_group
        ctx.halo_rows = halo_rows
        ctx.input_shape = hidden.shape
        ctx.chunk_tape = chunk_tape
        ctx.chunk_slot = chunk_slot
        ctx.chunk_index = chunk_tape.index if chunk_tape is not None else 0
        ctx.consumed_previous_tail = False
        tail = hidden[-halo_rows:].contiguous()
        if size > 1:
            gathered = tail.new_empty((size,) + tuple(tail.shape))
            dist.all_gather_into_tensor(
                gathered.view(size * tail.shape[0], *tail.shape[1:]), tail, group=cp_group
            )
            boundary = tail.new_zeros(tail.shape)
            if rank > 0:
                # Copy out so the halo does not keep every rank's tail alive.
                boundary = gathered[rank - 1].clone()
        else:
            gathered = tail.unsqueeze(0)
            boundary = torch.zeros_like(tail)
        if chunk_tape is not None:
            chunk_tape.ship_tail(chunk_slot, gathered[-1])
            prior = chunk_tape.tail_from_previous(chunk_slot)
            ctx.consumed_previous_tail = prior is not None
            if rank == 0 and prior is not None:
                boundary = prior.to(boundary.dtype) + boundary * 0
        return boundary

    @staticmethod
    def backward(ctx, grad_boundary: Tensor):
        """Return halo and prior-pass gradients with one group collective."""
        group = ctx.cp_group
        rank, size = group.rank(), group.size()
        grad_input = grad_boundary.new_zeros(ctx.input_shape)
        chunk_tape = getattr(ctx, "chunk_tape", None)

        # Whether a previous tail exists changes values, never participation.
        halo = grad_boundary.contiguous()
        if rank == 0 and not ctx.consumed_previous_tail:
            halo = torch.zeros_like(halo)
        if size == 1:
            gathered = halo.unsqueeze(0)
        else:
            gathered = halo.new_empty((size,) + tuple(halo.shape))
            dist.all_gather_into_tensor(
                gathered.view(size * halo.shape[0], *halo.shape[1:]), halo, group=group
            )
        if rank + 1 < size:
            grad_input[-ctx.halo_rows :] = gathered[rank + 1]
        if chunk_tape is not None and rank == size - 1:
            if ctx.chunk_index > 0:
                chunk_tape.ship_tail_grad(ctx.chunk_index, ctx.chunk_slot, gathered[0])
            later = chunk_tape.tail_grad_from_next(ctx.chunk_index, ctx.chunk_slot)
            if later is not None:
                grad_input[-ctx.halo_rows :] += later.to(grad_input.dtype)
        return grad_input, None, None, None, None


def exchange_csa2_boundary_hidden(
    hidden: Tensor,
    halo_rows: int,
    cp_group: dist.ProcessGroup,
    chunk_tape=None,
    chunk_slot: int = -1,
) -> Tensor:
    """Fetch adjacent-rank halos with one CP all-gather per forward and backward.

    ``chunk_tape`` carries the last-rank tail across outer chunks. All CP ranks
    enter each collective in the same order; pass index and prior-tail availability
    change values after communication, never collective participation. With no tape,
    no cross-pass state is stored. K=1 keeps the ordinary halo values and gradients.
    """
    return _LeftBoundaryExchange.apply(hidden, halo_rows, cp_group, chunk_tape, chunk_slot)


@dataclass(frozen=True)
class CSA2CPCompressionLayout:
    """Local DSv4 compact groups and their canonical sequence-major gather map.

    DSv4 compaction can include an extra group entirely in the left halo. Each
    global key is read only from the rank owning its last source token, so that
    redundant group receives no gradient. Physical and logical padding keep
    their original addresses; only floating-point K tensors are communicated.
    """

    local: CSA2THDCompressionLayout
    global_layout: CSA2THDCompressionLayout
    gather_indices: Tensor

    def to_sequence_major(self, gathered: Tensor) -> Tensor:
        """Select each group's unique owner, retaining the all-gather autograd edge."""
        if gathered.shape[0] != self.local.cp_size * self.local.capacity:
            raise ValueError("CSA2 gathered K must match the rank-major compression capacity")
        # build_csa2_cp_compression_layout addresses invalid groups as row 0 and checks that
        # every index lies inside the gathered rows, whose count is pinned just above.
        output = gathered.index_select(0, self.gather_indices)
        invalid = ~self.global_layout.valid_groups.reshape(-1, *([1] * (gathered.ndim - 1)))
        return output.masked_fill(invalid, 0)


def build_csa2_cp_compression_layout(
    tokens: CSA2THDLayout, ratio: int, halo_rows: int
) -> CSA2CPCompressionLayout:
    """Mirror DSv4's compact group order using device-only prefix arithmetic.

    Local source indices address ``cat([left_halo, local_hidden])``. Group
    positions remain relative to the original sequence, including a group
    straddling a CP cut. Capacity depends only on host-known tensor geometry.
    """
    if tokens.cp_size <= 1 or tokens.total_tokens <= 0:
        raise ValueError("CSA2 CP compression requires nonempty contiguous CP partitions")
    if ratio not in (1, 2) or halo_rows < ratio:
        raise ValueError("CSA2 CP compression requires r1/r2 and the DSv4 compression halo")
    global_layout = tokens.for_compression(ratio)
    capacity = get_cp_compressor_capacity(tokens.total_tokens, ratio)
    cu = tokens.cu_seqlens_padded
    starts, ends = cu[:-1].long(), cu[1:].long()
    begin, end = tokens.global_start, tokens.global_start + tokens.total_tokens
    # Match CompressorInputCompact, including the preceding complete group
    # when a sequence crosses an aligned CP boundary.
    first = (begin - ratio - starts).clamp_min(0).add(ratio - 1) // ratio
    stop = (ends.clamp_max(end) - starts).clamp_min(0) // ratio
    counts = (stop - first).clamp_min(0).masked_fill(ends.clamp_max(end) <= begin, 0)
    zero = cu.new_zeros(1)
    local_cu = torch.cat((zero, counts.cumsum(0).to(cu.dtype)))
    logical_counts = torch.minimum(
        (tokens.cu_seqlens.diff().long() // ratio - first).clamp_min(0), counts
    )
    logical_cu = torch.cat((zero, logical_counts.cumsum(0).to(cu.dtype)))
    rows = torch.arange(capacity, device=cu.device, dtype=torch.int64)
    if starts.numel() == 0:
        sequence_ids = torch.full_like(rows, -1)
        valid = torch.zeros_like(rows, dtype=torch.bool)
        positions = torch.zeros_like(rows)
        sources = torch.zeros((capacity, ratio), dtype=torch.int64, device=cu.device)
    else:
        sequence_ids = torch.bucketize(rows, local_cu[1:], right=True).clamp_max(starts.numel() - 1)
        group_ids = rows - local_cu[sequence_ids] + first[sequence_ids]
        assigned = rows < local_cu[-1]
        valid = assigned & ((group_ids + 1) * ratio <= tokens.cu_seqlens.diff()[sequence_ids])
        positions = (group_ids * ratio).masked_fill(~valid, 0)
        sources = (
            starts[sequence_ids, None]
            + group_ids[:, None] * ratio
            + torch.arange(ratio, device=cu.device)
            - begin
            + halo_rows
        ).masked_fill(~valid[:, None], 0)
        sequence_ids = sequence_ids.masked_fill(~assigned, -1)
    local = CSA2THDCompressionLayout(
        ratio=ratio,
        total_tokens=tokens.total_tokens + halo_rows,
        capacity=capacity,
        max_seqlen=global_layout.max_seqlen,
        cu_seqlens=logical_cu,
        cu_seqlens_padded=local_cu,
        sequence_ids=sequence_ids,
        position_ids=positions,
        source_indices=sources,
        valid_groups=valid,
        cp_rank=tokens.cp_rank,
        cp_size=tokens.cp_size,
    )
    if global_layout.capacity == 0:
        gather_indices = rows[:0]
    else:
        # The source identifies a virtual owner (chunk_index * cp_size + cp_rank).
        # Use that coordinate for the owner's first compressed group, but address the
        # gathered tensor through its corresponding physical CP slot.
        virtual_owner = global_layout.source_indices[:, -1] // tokens.total_tokens
        owner = virtual_owner % tokens.cp_size
        rank_start = virtual_owner * tokens.total_tokens
        first_seq = torch.bucketize(rank_start, cu[1:], right=True).clamp_max(starts.numel() - 1)
        first_group = (rank_start - ratio - starts[first_seq]).clamp_min(0).add(ratio - 1) // ratio
        first_global = global_layout.cu_seqlens_padded[first_seq] + first_group
        gather_indices = (
            owner * capacity + torch.arange(global_layout.capacity, device=cu.device) - first_global
        ).masked_fill(
            ~global_layout.valid_groups | (virtual_owner // tokens.cp_size != tokens.chunk_index), 0
        )
        # Invalid and other-pass groups read row 0 and are masked or replaced afterwards, so
        # an index outside the gathered rows means a VALID group's mapping is wrong. Check it
        # once per layout, without a host sync on CUDA.
        gathered_rows = tokens.cp_size * capacity
        bad = (gather_indices < 0) | (gather_indices >= gathered_rows)
        message = (
            f"a valid CSA2 group maps outside the gathered rows [0, {gathered_rows}) "
            f"(capacity={capacity}, cp_size={tokens.cp_size})"
        )
        if bad.is_cuda:
            torch._assert_async(~bad.any(), message)
        elif bool(bad.any()):
            raise ValueError(message)
    return CSA2CPCompressionLayout(local, global_layout, gather_indices)
