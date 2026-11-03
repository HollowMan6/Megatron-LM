#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU exactness checks against frozen CSA2 candidate oracle outputs."""

import argparse
import ast
import hashlib
import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "megatron/core/transformer/experimental_attention_variant/csa_utils/csa2_candidates.py"
)
BASELINE = (
    ROOT
    / "tests/unit_tests/transformer/experimental_attention_variant/data/csa2_candidate_baseline.json"
)
FUSED_SOURCE = ROOT / "megatron/core/transformer/experimental_attention_variant/csa2.py"


def require_original_source(path, hash_key):
    """Prevent an implementation change from silently rewriting its frozen oracle."""
    if BASELINE.exists():
        recorded_hash = json.loads(BASELINE.read_text()).get(hash_key)
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if recorded_hash is not None and actual_hash != recorded_hash:
            raise RuntimeError("Capture requires the original pre-streaming source revision")


def load_streaming():
    """Execute the real fused CPU operators without accelerator-stack imports."""
    namespace = runpy.run_path(str(SOURCE))
    namespace.update(
        torch=torch,
        F=F,
        Tuple=Tuple,
        CSA2THDLayout=SimpleNamespace,
        CSA2THDCompressionLayout=SimpleNamespace,
        _SELECT_MERGE_STAGE_KEYS=64,
        _SELECT_ROW_CHUNK_ELEMENTS=256,
        _CAUSAL_LIMIT_CACHE={},
        _CAUSAL_LIMIT_CACHE_MAX=64,
        _SLOT_COUNT_CACHE={},
        _SLOT_COUNT_CACHE_MAX=64,
    )
    tree = ast.parse(FUSED_SOURCE.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_causal_key_limit",
            "_sequence_slot_count",
            "_streaming_indexer_topk",
            "_chunked_csa2_indexer_scores",
            "_chunked_candidate_block_scores",
            "_dense_candidate_block_scores",
        }
    ]
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(FUSED_SOURCE), "exec"), namespace
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CompressedSparseAttention2"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_shared_global_attention"
    )
    namespace["CSA2State"] = SimpleNamespace
    namespace["_FUSED_CHECK"] = True
    namespace["_FUSED_DISABLE"] = False
    namespace["_check_reported"] = set()
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(FUSED_SOURCE), "exec"), namespace)
    return namespace


def fused_inputs(n_slots, chunk_count):
    """Build exact integer-score inputs with ragged blocks and sequence slots."""
    capacities = [131] if n_slots == 1 else [65, 47, 19]
    row_sequences = torch.arange(n_slots).repeat_interleave(3)
    rows = row_sequences.numel()
    row_positions = torch.cat(
        [torch.tensor([0, 2 * capacity - 1, 2 * capacity - 3]) for capacity in capacities]
    )
    valid_rows = torch.ones(rows, dtype=torch.bool)
    valid_rows[-1] = False
    prefixes = torch.tensor([0, *torch.tensor(capacities).cumsum(0).tolist()])
    keys = sum(capacities)
    key_sequences = torch.cat(
        [torch.full((capacity,), sequence) for sequence, capacity in enumerate(capacities)]
    )
    key_positions = torch.cat([2 * torch.arange(capacity) for capacity in capacities])
    valid_groups = torch.ones(keys, dtype=torch.bool)
    valid_groups[-2:] = False
    thd = SimpleNamespace(
        sequence_ids=row_sequences,
        position_ids=row_positions,
        valid_tokens=valid_rows,
        chunk_count=chunk_count,
    )
    comp = SimpleNamespace(
        cu_seqlens_padded=prefixes,
        total_tokens=keys,
        max_seqlen=max(capacities),
        valid_groups=valid_groups,
        sequence_ids=key_sequences,
        position_ids=key_positions,
    )
    q = ((torch.arange(rows * 8).reshape(rows, 1, 2, 4) % 5) - 2).float()
    indexer_k = ((torch.arange(keys * 4).reshape(keys, 1, 4) % 7) - 3).float()
    weights = ((torch.arange(rows * 2).reshape(rows, 1, 2) % 3) - 1).float()
    ids = torch.tensor([0, 2, 5, 9]).expand(rows, 4).clone().int()
    ids[-1] = -1
    return q, indexer_k, weights, thd, comp, ids


def capture_fused_baseline():
    """Freeze old fused outputs as an additional K=1 and integration oracle."""
    snapshot = json.loads(BASELINE.read_text())
    assert hashlib.sha256(FUSED_SOURCE.read_bytes()).hexdigest() == snapshot["fused_source_sha256"]
    require_original_source(SOURCE, "source_sha256")
    require_original_source(FUSED_SOURCE, "fused_source_sha256")
    namespace = load_streaming()
    cases = []
    for n_slots in (1, 3):
        for chunk_count in (1, 2):
            for use_candidates in (False, True):
                q, keys, weights, thd, comp, ids = fused_inputs(n_slots, chunk_count)
                candidates = namespace["CSA2CandidateBlocks"](ids, 3) if use_candidates else None
                indices, scores = namespace["_streaming_indexer_topk"](
                    q, keys, weights, 8, 2, thd, comp, candidates, True, 3
                )
                visible = ((thd.position_ids + 1) // 2).unsqueeze(-1)
                block_ids = namespace["candidate_blocks_from_block_scores"](
                    scores, visible, 4, 3
                ).indices
                cases.append(
                    {
                        "n_slots": n_slots,
                        "chunk_count": chunk_count,
                        "use_candidates": use_candidates,
                        "indices": encode_tensor(indices),
                        "block_scores": encode_tensor(scores),
                        "block_indices": encode_tensor(block_ids),
                        "block_values": encode_tensor(oracle_values(scores, visible, 3, block_ids)),
                    }
                )
    snapshot = json.loads(BASELINE.read_text())
    snapshot["fused_source_sha256"] = hashlib.sha256(FUSED_SOURCE.read_bytes()).hexdigest()
    snapshot["fused_cases"] = cases
    BASELINE.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    print(f"Captured {len(cases)} immutable old fused CPU cases")


def encode_tensor(tensor):
    """Represent infinities as strings so the fixture is standard JSON."""
    if tensor.ndim:
        return [encode_tensor(row) for row in tensor]
    value = tensor.item()
    if isinstance(value, float) and not torch.isfinite(tensor):
        return "-inf" if value < 0 else "inf"
    return value


def decode_tensor(values, dtype=torch.float32):
    """Restore the fixture without decimal or dtype conversions."""
    if isinstance(values, list):
        return torch.stack([decode_tensor(value, dtype) for value in values])
    return torch.tensor(float(values) if isinstance(values, str) else values, dtype=dtype)


def oracle_values(scores, visible, block_size, indices):
    """Gather values at the oracle's exact ids, including its forced newest block."""
    newest = (visible - 1) // block_size
    forced = scores.masked_fill(torch.arange(scores.shape[-1]) == newest, torch.inf)
    return forced.gather(-1, indices.clamp_min(0).long()).masked_fill(indices < 0, -torch.inf)


def capture_baseline():
    """Explicitly snapshot the untouched production selector before implementation."""
    snapshot = json.loads(BASELINE.read_text())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == snapshot["source_sha256"]
    require_original_source(SOURCE, "source_sha256")
    select = runpy.run_path(str(SOURCE))["candidate_blocks_from_block_scores"]
    cases = []
    patterns = {
        "all_equal": torch.full((4, 11), 2.0),
        "all_negative_infinity": torch.full((4, 11), -torch.inf),
        "partial_ties": torch.tensor(
            [
                [
                    1.0001220703125,
                    1.000244140625,
                    1.000244140625,
                    1,
                    -torch.inf,
                    1.000244140625,
                    0,
                    1.00018310546875,
                    1.00018310546875,
                    1.000244140625,
                    0,
                ]
            ]
        ).repeat(4, 1),
        "multiple_sequence_slots": torch.tensor([[7, 7, 1, 3, 3, -torch.inf, 7, 0, 0]]).repeat(
            4, 1
        ),
        "fewer_blocks_than_chunk": torch.tensor([[1, 1, 1]]).repeat(4, 1),
        "partial_final_chunk": torch.tensor([[5, 5, 0, 0, 5, 5, 1, 1, 5, 5]]).repeat(4, 1),
    }
    for name, scores in patterns.items():
        scores = scores.float()
        num_blocks = scores.shape[-1]
        visible = torch.tensor([[0], [2], [num_blocks], [2 * num_blocks]])
        topk = min(6 if name == "partial_ties" else 3, num_blocks)
        indices = select(scores, visible, topk, 2).indices
        cases.append(
            {
                "name": name,
                "chunk_size": 4,
                "n_slots": 3 if name == "multiple_sequence_slots" else 1,
                "row_sequences": [0, 2, 1, 2] if name == "multiple_sequence_slots" else [0] * 4,
                "block_size": 2,
                "topk_blocks": topk,
                "scores": encode_tensor(scores),
                "visible": encode_tensor(visible),
                "indices": encode_tensor(indices),
                "values": encode_tensor(oracle_values(scores, visible, 2, indices)),
            }
        )
    snapshot = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "oracle": "candidate_blocks_from_block_scores (stable descending argsort)",
        "values": "Scores gathered at oracle ids after newest-block forcing; -1 slots are -inf.",
        "cases": cases,
    }
    BASELINE.parent.mkdir(exist_ok=True)
    BASELINE.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    print(f"Captured {len(cases)} immutable CPU oracle cases at {snapshot['source_revision']}")


def check_oracle():
    """Require both current full-table and streaming selection to match frozen ids/values."""
    namespace = runpy.run_path(str(SOURCE))
    select = namespace["candidate_blocks_from_block_scores"]
    topk_class = namespace["CSA2BlockTopK"]
    for case in json.loads(BASELINE.read_text())["cases"]:
        scores = decode_tensor(case["scores"])
        visible = decode_tensor(case["visible"], torch.int64)
        expected_ids = decode_tensor(case["indices"], torch.int32)
        expected_values = decode_tensor(case["values"])
        k, block_size = case["topk_blocks"], case["block_size"]
        full = select(scores, visible, k, block_size)
        assert torch.equal(full.indices, expected_ids), (case["name"], "full indices")
        assert torch.equal(
            oracle_values(scores, visible, block_size, full.indices), expected_values
        ), (case["name"], "full values")
        rows, num_blocks = scores.shape
        # Packed slots contain deliberately different foreign scores. Extract only
        # the owning sequence's slab, as the fused path must do before ranking.
        row_sequences = torch.tensor(case["row_sequences"])
        slots = scores.new_full((rows, case["n_slots"], num_blocks), 999.0)
        slots[torch.arange(rows), row_sequences] = scores
        for chunk_size in (1, case["chunk_size"], num_blocks + 1):
            state = topk_class(rows, num_blocks, k, block_size, visible, scores.device)
            for start in range(0, num_blocks, chunk_size):
                slab = slots[torch.arange(rows), row_sequences, start : start + chunk_size]
                state.update(slab, start)
            actual = select(state, visible, k, block_size)
            assert torch.equal(actual.indices, expected_ids), (case["name"], chunk_size, "indices")
            # State values are compared, rather than re-gathering source values:
            # this catches errors in score/index pairing during rolling merges.
            same_id = actual.indices.long().unsqueeze(-1) == state.indices.unsqueeze(-2)
            paired = torch.where(same_id, state.values.unsqueeze(-2), -torch.inf).amax(-1)
            paired = paired.masked_fill(actual.indices < 0, -torch.inf)
            assert torch.equal(paired, expected_values), (case["name"], chunk_size, "values")
        print(f"PASS frozen oracle: {case['name']} indices/values (three chunk widths)")


def check_fused():
    """Pin old fused outputs and both explicit fallback branches on CPU."""
    namespace = load_streaming()
    namespace["_CANDIDATE_BLOCK_CHUNK_SIZE"] = 4
    for case in json.loads(BASELINE.read_text())["fused_cases"]:
        q, keys, weights, thd, comp, ids = fused_inputs(case["n_slots"], case["chunk_count"])
        candidates = namespace["CSA2CandidateBlocks"](ids, 3) if case["use_candidates"] else None
        indices, scores = namespace["_streaming_indexer_topk"](
            q, keys, weights, 8, 2, thd, comp, candidates, True, 3, candidate_topk_blocks=4
        )
        assert torch.equal(indices, decode_tensor(case["indices"], torch.int32)), (
            case["n_slots"],
            case["chunk_count"],
            case["use_candidates"],
            "token indices",
        )
        visible = ((thd.position_ids + 1) // 2).unsqueeze(-1)
        if isinstance(scores, torch.Tensor):
            assert torch.equal(scores, decode_tensor(case["block_scores"])), "fallback values"
            values = oracle_values(
                scores, visible, 3, decode_tensor(case["block_indices"], torch.int32)
            )
        else:
            expected_ids = decode_tensor(case["block_indices"], torch.int32)
            same_id = expected_ids.long().unsqueeze(-1) == scores.indices.unsqueeze(-2)
            values = torch.where(same_id, scores.values.unsqueeze(-2), -torch.inf).amax(-1)
            values = values.masked_fill(expected_ids < 0, -torch.inf)
        assert isinstance(scores, torch.Tensor) == (case["chunk_count"] == 1), "K=1 fallback"
        actual = namespace["candidate_blocks_from_block_scores"](scores, visible, 4, 3)
        assert torch.equal(actual.indices, decode_tensor(case["block_indices"], torch.int32))
        assert torch.equal(values, decode_tensor(case["block_values"]))
        print(
            f"PASS frozen fused: K={case['chunk_count']} slots={case['n_slots']} "
            f"member={case['use_candidates']} token/block indices and fp32 values"
        )
    # A chunked pass with a block table narrower than C explicitly takes the old path.
    namespace["_CANDIDATE_BLOCK_CHUNK_SIZE"] = 4096
    q, keys, weights, thd, comp, _ = fused_inputs(1, 2)
    _, scores = namespace["_streaming_indexer_topk"](
        q, keys, weights, 8, 2, thd, comp, None, True, 3, candidate_topk_blocks=4
    )
    assert isinstance(scores, torch.Tensor), "n_blocks <= C fallback"
    print("PASS explicit n_blocks <= C fallback")


def check_allocations():
    """Reject the exact full-width bool/fp32 table allocations on the streamed path."""
    namespace = load_streaming()
    namespace["_CANDIDATE_BLOCK_CHUNK_SIZE"] = 4
    full, zeros = torch.full, torch.zeros
    for n_slots in (1, 3):
        q, keys, weights, thd, comp, ids = fused_inputs(n_slots, 2)
        rows = q.shape[0]
        num_blocks = (max(comp.total_tokens, comp.max_seqlen) + 2) // 3
        forbidden = {(rows, num_blocks), (rows, n_slots * num_blocks)}
        candidates = namespace["CSA2CandidateBlocks"](ids, 3)
        recorded = []

        def record(function):
            def allocate(*args, **kwargs):
                output = function(*args, **kwargs)
                if output.dtype in (torch.bool, torch.float32) and output.ndim == 2:
                    recorded.append((tuple(output.shape), output.dtype))
                    assert tuple(output.shape) not in forbidden, (
                        "full candidate table allocation",
                        tuple(output.shape),
                        output.dtype,
                    )
                return output

            return allocate

        with patch.object(torch, "full", record(full)), patch.object(torch, "zeros", record(zeros)):
            for want_scores in (False, True):
                namespace["_streaming_indexer_topk"](
                    q,
                    keys,
                    weights,
                    8,
                    2,
                    thd,
                    comp,
                    candidates,
                    want_scores,
                    3,
                    candidate_topk_blocks=4,
                )
        assert recorded, "allocation recorder must observe actual production buffers"
        print(f"PASS no full bool/fp32 candidate-table allocations: slots={n_slots}")

    # Empty key spans still need candidate forcing but must not resurrect a
    # full table just because the fused scorer returned early.
    namespace = load_streaming()
    namespace["_CANDIDATE_BLOCK_CHUNK_SIZE"] = 4
    q, keys, weights, thd, comp, _ = fused_inputs(1, 2)
    empty_keys = keys[:0]
    comp.sequence_ids = comp.sequence_ids[:0]
    comp.position_ids = comp.position_ids[:0]
    comp.valid_groups = comp.valid_groups[:0]
    namespace["_streaming_indexer_topk"](
        q, empty_keys, weights, 8, 2, thd, comp, None, True, 3, candidate_topk_blocks=4
    )
    print("PASS empty-key large-table path uses rolling candidate state")


def check_consumer_and_crosscheck():
    """Execute the real candidate publisher and FUSED_CHECK consumer on CPU."""
    namespace = load_streaming()
    namespace["_CANDIDATE_BLOCK_CHUNK_SIZE"] = 4
    for n_slots in (1, 3):
        q, keys, weights, thd, comp, ids = fused_inputs(n_slots, 2)
        candidates = namespace["CSA2CandidateBlocks"](ids.unsqueeze(0), 3)
        args = (q, keys, weights, 8, 2, thd, comp, candidates, False, 3)
        actual, _ = namespace["_streaming_indexer_topk"](*args)
        thd.chunk_count = 1
        expected, _ = namespace["_streaming_indexer_topk"](*args)
        assert torch.equal(actual, expected), "consumer-only production shape"
        thd.chunk_count = 2
        dense = namespace["_chunked_csa2_indexer_scores"](q, keys, weights)
        causal = (
            thd.valid_tokens[:, None]
            & comp.valid_groups[None, :]
            & (thd.sequence_ids[:, None] == comp.sequence_ids[None, :])
            & (comp.position_ids[None, :] + 2 <= thd.position_ids[:, None] + 1)
        )
        dense = dense.masked_fill(~causal, -torch.inf)
        fused = namespace["_streaming_indexer_topk"](
            q, keys, weights, 8, 2, thd, comp, None, True, 3, candidate_topk_blocks=4
        )

        def select(scores):
            indices = scores.argsort(dim=-1, descending=True, stable=True)[..., :8].sort(-1).values
            return indices.masked_fill(~scores.gather(-1, indices).isfinite(), -1).int()

        indexer = SimpleNamespace(
            select_indices_fused=lambda *a, **kw: fused,
            forward_before_topk=lambda *a, **kw: dense,
            select_indices=select,
        )
        config = SimpleNamespace(csa2_candidate_topk_blocks=4, csa2_candidate_block_size=3)
        model = SimpleNamespace(
            is_kv_source=False,
            kv_source_layer=0,
            is_index_source=True,
            uses_candidates=False,
            is_candidate_source=True,
            layer_idx=0,
            compress_ratio=2,
            config=config,
            indexer=indexer,
            rotary_pos_emb=None,
        )
        state = SimpleNamespace(
            thd_layout=thd,
            compressed_layout=comp,
            kv_source_layer=0,
            global_kv=keys,
            indexer_k=keys,
        )
        namespace["_shared_global_attention"](model, q, q, state)
        expected = namespace["candidate_blocks_from_block_scores"](
            namespace["_chunked_candidate_block_scores"](dense[0], thd.sequence_ids, comp, 3),
            ((thd.position_ids + 1) // 2)[:, None],
            4,
            3,
        )
        assert torch.equal(state.candidates.indices, expected.indices.unsqueeze(0))
        print(f"PASS real FUSED_CHECK and candidate consumer shape: slots={n_slots}")


def check_tie_mutation():
    """A topk substitution must break the frozen equal-score oracle."""
    text = SOURCE.read_text()
    original = "values.argsort(dim=-1, descending=True, stable=True)[..., :keep]"
    assert text.count(original) == 1
    mutated = text.replace(original, "values.topk(keep, dim=-1, sorted=True).indices")
    namespace = {"__name__": "tie_mutation"}
    exec(compile(mutated, str(SOURCE), "exec"), namespace)
    scores = torch.ones(1, 4)
    visible = torch.zeros(1, 1, dtype=torch.int64)
    oracle = runpy.run_path(str(SOURCE))["candidate_blocks_from_block_scores"](
        scores, visible, 1, 1
    )
    state = namespace["CSA2BlockTopK"](1, 4, 1, 1, visible, scores.device)
    state.update(scores, 0)
    actual = namespace["candidate_blocks_from_block_scores"](state, visible, 1, 1)
    assert not torch.equal(actual.indices, oracle.indices), "topk mutation escaped the tie guard"
    print("PASS tie mutation rejected: topk changes frozen equal-score indices")


def run(oracle_only=False, fused_only=False):
    """Run reproducible CPU checks with a bounded thread count."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if not fused_only:
            check_oracle()
            check_tie_mutation()
        if not oracle_only:
            check_fused()
            check_allocations()
            check_consumer_and_crosscheck()
    finally:
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-baseline", action="store_true")
    parser.add_argument("--capture-fused-baseline", action="store_true")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--fused-only", action="store_true")
    args = parser.parse_args()
    if args.capture_baseline:
        capture_baseline()
    elif args.capture_fused_baseline:
        capture_fused_baseline()
    else:
        run(args.oracle_only, args.fused_only)
