"""Outer-sequence chunking: the tape and the autograd driver.

A K-way temporal split of one sequence runs as K passes over time-ordered slices. Each
pass is a full model pass, so any state a later pass needs from an earlier one has to be
carried explicitly. Only three things cross a pass boundary:

* the compressed ``global_kv`` each KV source slot published (value + gradient),
* the 127-row attention-input halo tail at the cut (value + gradient),
* metadata with no gradient (Engram prefix ids, positions, group ownership).

Everything else - ``global_indices``, candidate sets, ``pre_mix``, routing - is indexed by
the *current* pass's query rows and is recomputed per pass, which is why splitting inside
a layer failed while splitting the timeline works. "Recomputed" means every pass rebuilds
those tensors from its own rows; it does not mean a pass may skip its own contribution to
the cumulative KV - see :func:`extend_chunk_kv`.

Values retain only each pass's new rows, plus corrections for groups completed inside
the prior capacity. Cumulative tensors are reconstructed on demand and are not cached.

Gradients accumulate through ordinary autograd: reconstructed history becomes a leaf with
``requires_grad`` during the gradient passes, and whatever the later passes accumulate into
``leaf.grad`` is injected into the earlier pass's own graph before its backward runs. There
is no hand-written projection backward and no giant custom Function wrapping the model.
"""

from __future__ import annotations

import os
import sys

import torch


class ChunkTape:
    """Cross-pass state for one outer-chunked sequence.

    For K equal publications of R rows, values retain K*R rows instead of
    R*K*(K+1)/2. Prefix corrections additionally retain their values and int64
    row indices. Leaves and adjoint accumulators keep their cumulative shapes.
    """

    def __init__(self, chunk_count: int) -> None:
        if chunk_count < 1:
            raise ValueError("chunk_count must be >= 1")
        self.chunk_count = chunk_count
        self.index = 0
        self.mode = "prefill"  # "prefill" | "grad"
        self.values: dict = {}
        self.value_parents: dict = {}
        self.value_patches: dict = {}
        self.leaves: dict = {}
        self.accum: dict = {}
        self.tails: dict = {}
        self.tail_grads: dict = {}
        self._tape_mem_enabled = os.environ.get("DSV41_TAPE_MEM") == "1"
        self._tape_mem_reported = False

    def accounting(self) -> tuple[dict[str, int], list[str]]:
        """Return logical tensor bytes and notes for shared views.

        Each category is counted with ``numel() * element_size()``. A shared leaf
        is counted in both categories because this is a logical tape ledger, not
        an allocator-unique byte measurement; ``aliases`` identifies that sharing.
        """
        sizes: dict[str, int] = {
            "values.indexer_k": 0,
            "values.global_kv": 0,
            "patches": 0,
            "leaves": 0,
            "accum": 0,
        }
        aliases: list[str] = []
        storage_owners: list[tuple[torch.device, int, int, int, str]] = []

        def add(category: str, key, tensor: torch.Tensor) -> None:
            sizes[category] = sizes.get(category, 0) + tensor.numel() * tensor.element_size()
            if not tensor.numel():
                return
            label = f"{category}[{key!r}]"
            storage = tensor.untyped_storage().data_ptr()
            start = tensor.storage_offset() * tensor.element_size()
            end = start + tensor.numel() * tensor.element_size()
            owner = next(
                (
                    prior_label
                    for (
                        prior_device,
                        prior_storage,
                        prior_start,
                        prior_end,
                        prior_label,
                    ) in storage_owners
                    if prior_device == tensor.device
                    and prior_storage == storage
                    and start < prior_end
                    and prior_start < end
                ),
                None,
            )
            if owner is not None:
                aliases.append(f"{label}->{owner}")
            elif tensor._base is not None:
                aliases.append(f"{label}->external_base")
            storage_owners.append((tensor.device, storage, start, end, label))

        for key, tensor in self.values.items():
            add(f"values.{key[0]}", key, tensor)
        for key, (indices, values) in self.value_patches.items():
            add("patches", (key, "indices"), indices)
            add("patches", (key, "values"), values)
        for category in ("leaves", "accum"):
            for key, tensor in getattr(self, category).items():
                add(category, key, tensor)
        sizes["total"] = sum(sizes.values())
        return sizes, aliases

    def _report_tape_memory(self) -> None:
        """Print the final logical-step ledger on CP rank zero."""
        from megatron.core.parallel_state import get_context_parallel_rank

        if get_context_parallel_rank() != 0:
            return
        sizes, aliases = self.accounting()
        required = ("values.indexer_k", "values.global_kv")
        value_names = sorted(name for name in sizes if name.startswith("values."))
        ordered = (*required, *(name for name in value_names if name not in required))
        ordered += ("patches", "leaves", "accum", "total")
        fields = " ".join(f"{name}={sizes[name] / 1e6:.6f}MB" for name in ordered)
        view_note = ";".join(aliases) if aliases else "none"
        sys.stdout.write(f"[tape-mem] {fields} basis=logical_bytes aliases={view_note}\n")
        sys.stdout.flush()
        self._tape_mem_reported = True

    # -- pass lifecycle -----------------------------------------------------
    def begin(self, index: int, mode: str) -> None:
        """Select the pass and whether it is filling or consuming the tape."""
        if mode not in ("prefill", "grad"):
            raise ValueError(f"unknown tape mode {mode!r}")
        if not 0 <= index < self.chunk_count:
            raise ValueError("chunk index outside the chunk count")
        self.index = index
        self.mode = mode

    # -- values -------------------------------------------------------------
    def store(self, key, value: torch.Tensor) -> torch.Tensor:
        """Snapshot a publication without retaining its producer's graph."""
        self.values[key] = value.detach().clone()
        return value

    def value(self, key) -> torch.Tensor:
        """Reconstruct a cumulative publication without caching the result."""
        keys = self.value_keys(key)
        if len(keys) == 1:
            return self.values[key]
        out = torch.cat([self.values[item] for item in keys], dim=0)
        for item in keys:
            patch = self.value_patches.get(item)
            if patch is not None:
                indices, values = patch
                out.index_copy_(0, indices, values)
        return out

    def value_keys(self, key) -> list:
        """Return the chronological delta chain ending at a publication."""
        keys = []
        while key is not None:
            keys.append(key)
            key = self.value_parents.get(key)
        return keys[::-1]

    def store_delta(self, key, parent, current: torch.Tensor, prefix_mask) -> None:
        """Snapshot new rows, including groups completed inside the prior capacity.

        A replay must leave its prefill baseline intact so a later comparison can
        detect mismatches. Cloning the tail prevents a view from retaining the
        current gather's entire storage.
        """
        if self.mode == "grad" and key in self.values:
            return
        prior_rows = (
            sum(self.values[item].shape[0] for item in self.value_keys(parent))
            if parent is not None
            else 0
        )
        self.store(key, current[prior_rows:])
        self.value_parents[key] = parent
        self.value_patches.pop(key, None)
        if parent is not None and prefix_mask is not None:
            indices = (~prefix_mask[:prior_rows]).nonzero(as_tuple=True)[0]
            if indices.numel():
                self.value_patches[key] = (indices, current.detach().index_select(0, indices))

    # -- differentiable leaves ---------------------------------------------
    def make_leaf(self, key, value: torch.Tensor, *, copy: bool = True) -> torch.Tensor:
        """A leaf that replay can hand back to a *later* pass unchanged.

        Returning the same object every time is what makes the gradient land on the
        tensor the consumer used; a fresh leaf per replay would silently drop it.
        """
        leaf = self.leaves.get(key)
        if leaf is None:
            leaf_value = value.detach().clone() if copy else value.detach()
            leaf = leaf_value.requires_grad_(True)
            self.leaves[key] = leaf
        return leaf

    def collect_adjoint(self, key) -> torch.Tensor | None:
        """Move ``leaf.grad`` into the accumulator after a pass's backward."""
        leaf = self.leaves.get(key)
        if leaf is None or leaf.grad is None:
            return None
        got = leaf.grad.detach()
        prev = self.accum.get(key)
        self.accum[key] = got.clone() if prev is None else prev + got
        leaf.grad = None
        return self.accum[key]

    def collect_all_adjoints(self) -> dict:
        """Collect every leaf that received a gradient.

        Collecting by key is a footgun: a later pass consumes the *prefix* leaf, so its
        adjoint lands on key g-1, and asking for key g silently collects nothing and drops
        the contribution. The tape already owns its leaves, so it can just look.
        """
        found = {}
        for key in list(self.leaves):
            got = self.collect_adjoint(key)
            if got is not None:
                found[key] = got
        if (
            self._tape_mem_enabled
            and not self._tape_mem_reported
            and self.mode == "grad"
            and self.index == 0
        ):
            self._report_tape_memory()
        return found

    def has_adjoint(self, key) -> bool:
        """Return whether a producer pass has a carried gradient."""
        return key in self.accum

    def adjoint(self, key) -> torch.Tensor:
        """Return the gradient accumulated for a producer pass."""
        return self.accum[key]

    # -- halo tails ---------------------------------------------------------
    def ship_tail(self, slot, tail: torch.Tensor) -> None:
        """Snapshot a tail already replicated by the caller's CP exchange.

        Tape presence must not control collective participation. Communication
        belongs to the exchange; both attention and Engram pass replicated tails.
        """
        self.tails[(self.index, slot)] = tail.detach().clone()

    def tail_from_previous(self, slot) -> torch.Tensor | None:
        """Return the saved boundary tail from the previous pass."""
        return self.tails.get((self.index - 1, slot))

    def ship_tail_grad(self, index: int, slot, grad: torch.Tensor) -> None:
        """Record an adjoint under the consumer pass captured by its forward."""
        self.tail_grads[(index, slot)] = grad.detach().clone()

    def tail_grad_from_next(self, index: int, slot) -> torch.Tensor | None:
        """Read the next pass's adjoint without consulting the mutable cursor."""
        return self.tail_grads.get((index + 1, slot))


class InjectChunkAdjoint(torch.autograd.Function):
    """Add a carried adjoint to the gradient flowing into a pass's cumulative output.

    The consumer sees sequence-major values after the CP gather. Backward through
    that gather reduce-scatters the total (own loss plus later-pass contributions)
    to the owning projection exactly once; no manual all-reduce is needed.
    """

    @staticmethod
    def forward(ctx, tensor, tape, key):
        """Pass cumulative values through while binding their producer key."""
        ctx.tape = tape
        ctx.key = key
        return tensor

    @staticmethod
    def backward(ctx, grad):
        """Add later-pass gradient to this pass's cumulative output."""
        extra = ctx.tape.adjoint(ctx.key)
        return grad + extra.to(grad.dtype), None, None


def extend_chunk_kv(
    tape: ChunkTape,
    layer_idx: int,
    name: str,
    current: torch.Tensor,
    prefix_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return cumulative compressed KV in global sequence coordinates.

    The current gather uses cumulative coordinates but contains only this pass's
    projections. Both prefill and replay restore completed historical groups from
    the tape; replay uses a leaf so its adjoint reaches the producer's replay.
    """
    if tape.chunk_count <= 1:
        return current
    prior_key = (name, layer_idx, tape.index - 1)
    this_key = (name, layer_idx, tape.index)
    prior_keys = tape.value_keys(prior_key) if prior_key in tape.values else []
    prior_rows = sum(tape.values[key].shape[0] for key in prior_keys)
    if current.shape[0] < prior_rows:
        raise ValueError(
            f"cumulative {name} has {current.shape[0]} rows before pass {tape.index}, "
            f"but the prior pass published {prior_rows}"
        )
    if not prior_keys:
        out = current
    elif (
        tape.mode != "grad"
        and prefix_mask is None
        and not any(key in tape.value_patches for key in prior_keys)
    ):
        out = torch.cat([*(tape.values[key] for key in prior_keys), current[prior_rows:]], dim=0)
    else:
        prefix = tape.value(prior_key)
        if tape.mode == "grad":
            prefix = tape.make_leaf(prior_key, prefix, copy=False)
        # Capacity can include incomplete groups at the pass cut. Only completed
        # historical groups belong to the tape; newly completed groups use this pass.
        if prefix_mask is not None:
            mask = prefix_mask[:prior_rows].reshape(-1, *([1] * (current.ndim - 1)))
            prefix = torch.where(mask, prefix, current[:prior_rows])
        out = torch.cat((prefix, current[prior_rows:]), dim=0)
    tape.store_delta(this_key, prior_key if prior_keys else None, current, prefix_mask)
    # The adjoint belongs to this pass's cumulative output, not to its prefix.
    if tape.mode == "grad" and tape.has_adjoint(this_key):
        out = InjectChunkAdjoint.apply(out, tape, this_key)
    return out


def assert_replay_matches(tape: ChunkTape, key, value: torch.Tensor, atol: float = 0.0):
    """Compare replay to reconstructed prefill, including historical corrections.

    Comparing the cumulative result catches mismatches in reconstructed history,
    including corrections retained for previously uncommitted rows. Equality is
    bit-for-bit by default.
    """
    stored = tape.value(key)
    if stored.shape != value.shape:
        raise AssertionError(
            f"replay shape mismatch for {key!r}: {tuple(value.shape)} != {tuple(stored.shape)}"
        )
    if atol == 0.0:
        if not torch.equal(stored, value.detach()):
            raise AssertionError(f"replay is not bit-identical for {key!r}")
    elif not torch.allclose(stored, value.detach(), atol=atol, rtol=0.0):
        raise AssertionError(f"replay differs beyond {atol} for {key!r}")
