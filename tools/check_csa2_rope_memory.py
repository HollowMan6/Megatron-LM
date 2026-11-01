#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU exactness and table-size checks for CSA2 RoPE."""

import argparse
import ast
import math
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
_LAYOUTS = None


class RotaryBase(nn.Module):
    """Minimal Megatron rotary base used to AST-load the production Yarn class."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.cp_group = kwargs.get("cp_group")
        self.rotary_interleaved = kwargs.get("rotary_interleaved", False)
        self.seq_len_interpolation_factor = kwargs.get("seq_len_interpolation_factor")


def load_layouts():
    """Load production CPU THD layout builders without importing the full stack."""
    global _LAYOUTS
    if _LAYOUTS is not None:
        return _LAYOUTS
    sys.path.insert(0, str(ROOT / "tools"))
    from check_outer_chunk_numerics import CPGroup, load_modules

    _, thd, cp = load_modules()
    from megatron.core.packed_seq_params import PackedSeqParams

    _LAYOUTS = (CPGroup, thd, cp, PackedSeqParams)
    return _LAYOUTS


def load_functions():
    """AST-load the real Yarn, RoPE application, and CSA2 RoPE functions."""
    yarn = ROOT / "megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py"
    rope = ROOT / "megatron/core/models/common/embeddings/rope_utils.py"
    csa = ROOT / "megatron/core/transformer/experimental_attention_variant/csa2.py"
    yt, rt, ct = (ast.parse(path.read_text()) for path in (yarn, rope, csa))
    _, thd, _, _ = load_layouts()
    ns = dict(
        torch=torch,
        Tensor=Tensor,
        Optional=Optional,
        nn=nn,
        math=math,
        lru_cache=lru_cache,
        internal_api=lambda fn: fn,
        TransformerConfig=object,
        RotaryEmbedding=RotaryBase,
        get_pos_emb_on_this_cp_rank=lambda emb, _dim, _group: emb,
        parallel_state=SimpleNamespace(get_context_parallel_group=lambda **_kw: None),
        CSA2THDLayout=thd.CSA2THDLayout,
        CSA2THDCompressionLayout=thd.CSA2THDCompressionLayout,
        MLATransformerConfig=object,
        fused_mla_rope_out_of_place=None,
    )
    nodes = [n for n in yt.body if isinstance(n, ast.FunctionDef) and n.name.startswith("_yarn_")]
    nodes += [
        next(n for n in yt.body if isinstance(n, ast.ClassDef) and n.name == "YarnRotaryEmbedding")
    ]
    nodes += [
        n
        for n in rt.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_rotate_half", "_apply_rotary_pos_emb_bshd", "apply_rotary_pos_emb"}
    ]
    apply = next(
        n for n in ct.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_thd_rope"
    )
    for node in nodes + [apply]:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(yarn), "exec"), ns)
    return ns, apply, csa


def fixed_baseline(template, path):
    """Freeze the pre-fix max(total_tokens, base length) policy in an AST copy."""
    node = ast.parse(ast.unparse(template)).body[0]
    i = next(
        i
        for i, s in enumerate(node.body)
        if isinstance(s, ast.Assign) and "layout.max_seqlen" in ast.unparse(s)
    )
    body = node.body[: i + 1] + ast.parse("length = max(length, int(layout.total_tokens))").body
    body += [s for s in node.body[i + 1 :] if "layout.total_tokens" not in ast.unparse(s)]
    node.body = body
    ast.fix_missing_locations(node)
    ns, _, _ = load_functions()
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns["_apply_thd_rope"]


class Recorder:
    """Record requested table lengths while delegating to real Yarn."""

    def __init__(self, rope) -> None:
        self.rope = rope
        self.lengths: list[int] = []

    def __call__(self, length, packed_seq=False):
        self.lengths.append(int(length))
        return self.rope(length, packed_seq=packed_seq)


def make_layout(k, cp_size, rank, ratio, scenario, kind):
    """Build a ragged or chunked layout with the production THD builders."""
    CPGroup, thd, cp, PackedSeqParams = load_layouts()
    local = 14 if cp_size == 1 else 7
    capacity = local * cp_size * k
    if scenario == "ragged":
        physical = torch.arange(0, capacity + 1, 7, dtype=torch.int32)
        lengths = [5 + i % 3 for i in range(physical.numel() - 1)]
        logical = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        maximum = 7
    else:
        physical = torch.tensor([0, capacity], dtype=torch.int32)
        logical = torch.tensor([0, capacity - 1], dtype=torch.int32)
        maximum = capacity
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=maximum,
        max_seqlen_kv=maximum,
        cp_partition_mode="contiguous",
        cp_chunk_index=k - 1,
        cp_chunk_count=k,
    )
    tokens = thd.build_csa2_thd_layout(params, local, cp_group=CPGroup(rank, cp_size))
    if kind == "tokens":
        return tokens
    return (
        cp.build_csa2_cp_compression_layout(tokens, ratio, 2).local
        if cp_size > 1
        else tokens.for_compression(ratio)
    )


def one_case(ns, old_apply, k, cp_size, rank, ratio, scenario, kind, dtype):
    """Compare independent baseline/current graphs and their table rows exactly."""
    old_device = torch.cuda.current_device
    try:
        torch.cuda.current_device = lambda: "cpu"
        layout = make_layout(k, cp_size, rank, ratio, scenario, kind)
        config = SimpleNamespace(
            qk_pos_emb_head_dim=16, apply_rope_fusion=False, rotary_interleaved=False
        )
        generator = torch.Generator().manual_seed(1201 + k * 100 + cp_size * 10 + rank + ratio)
        x0 = torch.randn((layout.position_ids.numel(), 1, 20), generator=generator, dtype=dtype)
        w0 = torch.randn((20, 20), generator=generator, dtype=dtype)
        outputs, lengths = [], []
        for apply in (old_apply, ns["_apply_thd_rope"]):
            # Separate instances prevent a smaller request from reusing a prior
            # larger cache and making the equivalence check vacuous.
            rope = ns["YarnRotaryEmbedding"](
                16,
                use_cpu_initialization=False,
                scaling_factor=2.0,
                original_max_position_embeddings=8,
            )
            x, weight = x0.clone().requires_grad_(), w0.clone().requires_grad_()
            recorder = Recorder(rope)
            projected = torch.nn.functional.linear(x, weight)
            result = apply(projected, recorder, config, layout, load_layouts()[0](rank, cp_size))
            probe = torch.arange(result.numel(), dtype=dtype).reshape_as(result) / 37
            loss = (result * probe).sum()
            grads = torch.autograd.grad(loss, (x, weight))
            outputs.append((result.detach(), loss.detach(), *grads))
            lengths.append(recorder.lengths[-1])
        required = layout.max_seqlen * (layout.ratio if hasattr(layout, "ratio") else 1)
        baseline_cos_sin = ns["YarnRotaryEmbedding"](
            16, use_cpu_initialization=False, scaling_factor=2.0, original_max_position_embeddings=8
        ).get_cached_cos_sin(lengths[0], dtype=dtype, packed_seq=True, mscale=1.0)
        current_cos_sin = ns["YarnRotaryEmbedding"](
            16, use_cpu_initialization=False, scaling_factor=2.0, original_max_position_embeddings=8
        ).get_cached_cos_sin(required, dtype=dtype, packed_seq=True, mscale=1.0)
        assert int(layout.position_ids.max()) < required
        assert all(
            torch.equal(
                a.contiguous().reshape(-1).view(torch.uint8),
                b.contiguous().reshape(-1).view(torch.uint8),
            )
            for a, b in zip(outputs[0], outputs[1])
        )
        ids = layout.position_ids
        assert all(
            torch.equal(
                old.index_select(0, ids).contiguous().reshape(-1).view(torch.uint8),
                new.index_select(0, ids).contiguous().reshape(-1).view(torch.uint8),
            )
            for old, new in zip(baseline_cos_sin, current_cos_sin)
        )
        return lengths, required, int(layout.position_ids.max()) + 1
    finally:
        torch.cuda.current_device = old_device


def run(check_memory=True):
    """Run exactness checks and optionally assert the host-known RoPE bound."""
    ns, template, path = load_functions()
    old_apply = fixed_baseline(template, path)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for k in (1, 2):
            for ratio in (1, 2):
                for cp_size in (1, 2):
                    kinds = ("tokens", "compressed") if cp_size == 2 or k == 1 else ("tokens",)
                    for rank in range(cp_size):
                        for scenario in ("ragged", "chunk"):
                            for kind in kinds:
                                for dtype in (torch.float32, torch.bfloat16):
                                    lengths, required, referenced = one_case(
                                        ns,
                                        old_apply,
                                        k,
                                        cp_size,
                                        rank,
                                        ratio,
                                        scenario,
                                        kind,
                                        dtype,
                                    )
                                    assert lengths[1] >= referenced
                                    sys.stdout.write(
                                        f"PASS exact K={k} r={ratio} CP={cp_size} rank={rank} "
                                        f"{scenario}/{kind} {dtype}; "
                                        f"old={lengths[0]} actual={lengths[1]} M={required}\n"
                                    )
                                    if check_memory:
                                        assert lengths[1] == required, (
                                            k,
                                            ratio,
                                            cp_size,
                                            rank,
                                            scenario,
                                            kind,
                                            dtype,
                                            lengths[1],
                                            required,
                                        )
    finally:
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerics-only", action="store_true")
    run(check_memory=not parser.parse_args().numerics_only)
