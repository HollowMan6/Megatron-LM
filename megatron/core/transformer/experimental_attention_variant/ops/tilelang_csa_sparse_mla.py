# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TileLang SparseMLA adapter for the CSA2 sparse-attention path.

CSA2's fused backend (``dsa_kernel_backend='cudnn'``) needs SM90+ cuDNN
Frontend DSA kernels plus FlashMLA, so accelerators that do not ship those
(e.g. PPU-class devices behind an ``sm_89`` compatibility layer) fell through
to :func:`...csa.unfused_compressed_sparse_attn`, which materializes a
``[batch, heads, seq, topk]`` FP32 score block plus the gathered KV block.
The TileLang SparseMLA kernels vendored in this directory are source-portable
and already serve the ordinary DSA absorbed path, so this module exposes them
behind the same call contract as the unfused reference.

Three shape/semantic bridges live here, each of which has to be exact for the
fused path to be interchangeable with the reference:

* **576 padding.**  The copied kernels are specialized for the DeepSeek V3.2
  absorbed shape (``dim_plus_tail_dim == 576``, ``v_channels == 512``): the 512
  value channels plus a 64-wide rotary tail.  CSA2 keeps q/kv at the 512 value
  channels only (rotary is applied to the latent before this call), so both are
  zero-padded to 576 here.  A zero tail adds nothing to a score and nothing to
  the value reduction, so the result is mathematically unchanged.
* **Sink.**  The kernel runs one softmax over the top-k keys and returns a
  base-2 LSE of the key mass only.  CSA2's reference adds the per-head sink as
  one extra always-valid key with no value, so the sink changes the denominator
  but never the numerator.  :func:`_sink_lse` builds the sink-inclusive LSE and
  the forward rescales the kernel output by ``exp2(lse_ns - lse_s)``.
* **Sink backward.**  Feeding the kernel backward the *sink-inclusive* LSE
  together with ``delta = sum(O * dO)`` of the *sink-scaled* output makes the
  kernel's ``p = exp2(q.k * scale - lse)`` the true final attention probability,
  so its dq/dkv are the gradients of the sink-aware softmax rather than of the
  key-only one.  The sink itself gets the closed-form ``-sum p_sink * delta``.

The kernel interfaces pad ``topk`` to their own tile multiple with ``-1``
sentinels, so callers do not need a top-k that is a multiple of any block size.
"""

from typing import Optional

import warnings

import torch

try:
    from .tilelang_dsa import _is_supported_sparse_mla_head_count
    from .tilelang_sparse_mla_bwd import HAVE_TILELANG as _HAVE_TILELANG_BWD
    from .tilelang_sparse_mla_bwd import sparse_mla_bwd
    from .tilelang_sparse_mla_fwd import HAVE_TILELANG as _HAVE_TILELANG_FWD
    from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface

    _HAVE_TILELANG_SPARSE_MLA = _HAVE_TILELANG_FWD and _HAVE_TILELANG_BWD
except (ImportError, OSError):
    # Without the kernels this backend is simply unavailable; the caller keeps
    # the PyTorch reference path instead of failing the model build.
    sparse_mla_bwd = None
    sparse_mla_fwd_interface = None
    _is_supported_sparse_mla_head_count = None
    _HAVE_TILELANG_SPARSE_MLA = False

#: ``log2(e)``, converting natural-log logits into the kernel's base-2 domain.
_LOG2_E = 1.4426950408889634

#: Value channels carried by the absorbed q/kv and by the attention output.
_VALUE_CHANNELS = 512

#: ``dim + tail_dim`` the copied kernels are specialized for (512 + rotary tail).
_ABSORBED_DIM = 576

_TAIL_DIM = _ABSORBED_DIM - _VALUE_CHANNELS


def _pad_last_dim(x: torch.Tensor, target: int) -> torch.Tensor:
    """Zero-pad the last axis to ``target`` and return a contiguous tensor.

    One allocation serves both the copy and the zero tail, which keeps the
    padding cost to a single pass over ``x``; ``F.pad`` would add a second
    tensor (and a copy) for the same result on the permuted views passed here.
    """
    padded = x.new_zeros((*x.shape[:-1], target))
    padded[..., : x.size(-1)] = x
    return padded


def _sink_lse(lse_ns: torch.Tensor, sink_l2: torch.Tensor) -> torch.Tensor:
    """Combine the kernel's base-2 key LSE with the per-head sink logit.

    Args:
        lse_ns: ``[b, sq, heads]`` FP32 ``log2`` of the key-only log mass.
        sink_l2: ``[heads]`` FP32 sink logit in ``log2`` units.

    Returns:
        ``[b, sq, heads]`` FP32 ``log2(2 ** lse_ns + 2 ** sink_l2)``.

    A ``-inf`` sink is valid input: the shift is then ``lse_ns`` and its mass
    term underflows to zero, which is exactly the no-sink softmax.
    """
    sink_l2 = sink_l2.view(1, 1, -1)
    shift = torch.maximum(lse_ns, sink_l2)
    return shift + torch.log2(torch.exp2(lse_ns - shift) + torch.exp2(sink_l2 - shift))


class _TileLangCSASparseMLA(torch.autograd.Function):
    """Autograd bridge between the CSA2 call contract and the SparseMLA kernels."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        kv_full: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_indices: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        seq_len, batch, heads = query.shape[:3]
        q_bshd = _pad_last_dim(query.permute(1, 0, 2, 3), _ABSORBED_DIM)
        # [n_kv, b, d] -> [b, n_kv, 1, d]: the kernels carry one KV head.
        kv_bsd = _pad_last_dim(kv_full.permute(1, 0, 2).unsqueeze(2), _ABSORBED_DIM)
        # [b, sq, topk] -> [b, sq, kv_group=1, topk]; the kernels index int32.
        indices_bsgk = topk_indices.to(torch.int32).unsqueeze(2).contiguous()

        out_ns, lse_ns = sparse_mla_fwd_interface(
            q_bshd, kv_bsd, indices_bsgk, sm_scale=softmax_scale
        )
        sink_l2 = attn_sink.detach().to(torch.float32) * _LOG2_E
        lse_s = _sink_lse(lse_ns, sink_l2)
        # Fold the sink into the denominator only. The sink carries no value, so
        # the numerator is untouched and the output is a pure rescale.
        out = out_ns.to(torch.float32) * torch.exp2(lse_ns - lse_s).unsqueeze(-1)
        out = out.to(out_ns.dtype)

        # Do not save ``out``: backward recomputes the key-only output in order to
        # form delta, which costs one more forward kernel but keeps a
        # [sq, b, heads, 512] activation out of the saved-tensor set (and out of
        # the offload path).
        ctx.save_for_backward(q_bshd, kv_bsd, indices_bsgk, lse_s, sink_l2)
        ctx.softmax_scale = softmax_scale
        ctx.sink_dtype = attn_sink.dtype
        out = out.permute(1, 0, 2, 3).contiguous()
        return out.reshape(seq_len, batch, heads * _VALUE_CHANNELS)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q_bshd, kv_bsd, indices_bsgk, lse_s, sink_l2 = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        batch, seq_len, heads = q_bshd.shape[:3]
        # The forward returned [sq, b, heads * 512]; the kernels want [b, sq, heads,
        # 512] dO, narrowed to their bf16 operand dtype.
        do = (
            grad_output.to(q_bshd.dtype)
            .reshape(seq_len, batch, heads, _VALUE_CHANNELS)
            .permute(1, 0, 2, 3)
            .contiguous()
        )

        with torch.no_grad():
            out_ns, lse_ns = sparse_mla_fwd_interface(
                q_bshd, kv_bsd, indices_bsgk, sm_scale=softmax_scale
            )
            # delta of the sink-scaled output; out = exp2(lse_ns - lse_s) * out_ns,
            # so its dot product with dO is the same scalar factor times dO . out_ns.
            delta = torch.sum(do.to(torch.float32) * out_ns.to(torch.float32), dim=-1)
            delta = delta * torch.exp2(lse_ns - lse_s)

            dq, dkv = sparse_mla_bwd(
                q_bshd,
                kv_bsd,
                None,
                do,
                indices_bsgk,
                lse_s.contiguous(),
                sm_scale=softmax_scale,
                delta=delta.contiguous(),
            )
            # The sink is one extra key with no value: d sink = -sum p_sink * delta.
            p_sink = torch.exp2(sink_l2.view(1, 1, -1) - lse_s)
            dsink = -(p_sink * delta).sum(dim=(0, 1))

        dq = dq[..., :_VALUE_CHANNELS].permute(1, 0, 2, 3).contiguous()
        # [b, n_kv, kv_group=1, 576] -> the caller's [n_kv, b, 512] layout.
        dkv = dkv[:, :, 0, :_VALUE_CHANNELS].permute(1, 0, 2).contiguous()
        return dq, dkv, dsink.to(ctx.sink_dtype), None, None


#: Decline reasons seen so far, so a silent fallback becomes a visible counter:
#: a run that reports using this backend can be checked against what it actually served.
_DECLINE_COUNTS: "dict[str, int]" = {}
_DECLINE_WARNED: "set[str]" = set()


def _decline(reason: str):
    """Record why the fused path declined (once per reason) and return None."""
    _DECLINE_COUNTS[reason] = _DECLINE_COUNTS.get(reason, 0) + 1
    if reason not in _DECLINE_WARNED:
        _DECLINE_WARNED.add(reason)
        warnings.warn(
            f"CSA2 TileLang sparse attention declined ({reason}); falling back to the "
            "PyTorch reference for this call. This is a silent per-shape switch, so the "
            f"run may be using the reference path while configured for 'tilelang'. "
            f"Counts: {dict(_DECLINE_COUNTS)}",
            stacklevel=3,
        )
    return None


def decline_counts() -> "dict[str, int]":
    """Per-reason decline counts, for periodic logging by the caller."""
    return dict(_DECLINE_COUNTS)


def tilelang_csa_sparse_attn(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> Optional[torch.Tensor]:
    """Differentiable sparse CSA attention through the TileLang SparseMLA kernels.

    Same contract as :func:`...csa.unfused_compressed_sparse_attn`:

    Args:
        query:         ``[sq, b, heads, 512]`` BF16 multi-head query.
        kv_full:       ``[n_kv, b, 512]`` BF16 single-head KV (window + compressed).
        attn_sink:     ``[heads]`` per-head sink logit (natural log scale).
        topk_indices:  ``[b, sq, topk]`` indices into ``kv_full``, ``-1`` = invalid.
        softmax_scale: Natural-log softmax scale (the kernels apply ``log2(e)``).

    Returns:
        ``[sq, b, heads * 512]`` attention output, or ``None`` when the shape or
        dtype falls outside the kernel specialization so the caller can run the
        PyTorch reference instead.

        ``topk_indices`` must hold only ``-1`` (invalid, skipped by the kernels --
        measured: rows whose top-k is entirely invalid produce the same zeros the
        reference produces) or a valid ``kv_full`` row. The kernels do not
        bound-check positive indices, so an out-of-range index is rejected here
        instead of reading past the KV buffer; it means the caller's index space
        and ``kv_full`` disagree (e.g. a mis-computed CP offset), never an
        unsupported shape.
    """
    if not _HAVE_TILELANG_SPARSE_MLA:
        return _decline("tilelang kernels unavailable")
    if query.ndim != 4 or kv_full.ndim != 3 or topk_indices.ndim != 3:
        return _decline("rank mismatch")
    if query.dtype != torch.bfloat16 or kv_full.dtype != torch.bfloat16:
        # The copied kernels are BF16-only; FP32 CSA2 keeps the reference path.
        return _decline("q/kv not bf16")

    seq_len, batch, heads, head_dim = query.shape
    if head_dim != _VALUE_CHANNELS or kv_full.size(-1) != _VALUE_CHANNELS:
        return _decline("value width != 512")
    if kv_full.size(1) != batch:
        return _decline("kv batch axis mismatch")
    if topk_indices.size(0) != batch or topk_indices.size(1) != seq_len:
        return _decline("index shape mismatch")
    if seq_len == 0 or kv_full.size(0) == 0 or topk_indices.size(-1) == 0:
        return _decline("empty query/kv/topk")
    if attn_sink.ndim != 1 or attn_sink.numel() != heads:
        return _decline("sink shape mismatch")
    # The kernels pad the head tile to max(next_power_of_two(heads), 16) and index
    # the unpadded head axis by that count, so only exact power-of-two tiles >= 16
    # stay in bounds. Reuse the DSA kernel's own contract rather than restating it.
    if not _is_supported_sparse_mla_head_count(heads, kv_group=1):
        return _decline(f"unsupported head count {heads}")

    # The kernels gather without bounds checks, so an out-of-range positive index
    # would read past the KV buffer. That is an index-space bug (the caller's
    # offsets and ``kv_full`` disagree), not an unsupported shape: fall back would
    # hide it, so raise.
    index_max = int(topk_indices.max()) if topk_indices.numel() else -1
    index_min = int(topk_indices.min()) if topk_indices.numel() else -1
    if index_max >= kv_full.size(0):
        raise IndexError(
            "topk_indices contain "
            f"{index_max} but kv_full has {kv_full.size(0)} rows: the index space and "
            "kv_full disagree (check the context-parallel offsets), not a shape the "
            "reference path could serve either."
        )
    if index_min < -1:
        raise IndexError(f"topk_indices contain {index_min}; only -1 marks an invalid slot")

    return _TileLangCSASparseMLA.apply(query, kv_full, attn_sink, topk_indices, softmax_scale)
