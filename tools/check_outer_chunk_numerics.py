#!/usr/bin/env python3
"""CPU checks for outer-chunk layouts, replay, halos and indexer row partitions.

Run with a CPU PyTorch interpreter; no accelerator packages or model weights needed.
The lightweight loaders bypass Megatron's eager accelerator imports, not any of
the production operators under test. CPU equality does not certify device GEMMs.
"""

import argparse
import ast
import importlib.util
import sys
import tempfile
import types
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]


def load_modules():
    for name, relative in (
        ("megatron", "megatron"),
        ("megatron.core", "megatron/core"),
        (
            "outer_chunk_checks",
            "megatron/core/transformer/experimental_attention_variant/csa_utils",
        ),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / relative)]
        sys.modules[name] = package
    modules = []
    for name, relative in (
        ("outer_chunk_tape", "megatron/core/pipeline_parallel/chunk_tape.py"),
        (
            "outer_chunk_checks.thd_utils",
            "megatron/core/transformer/experimental_attention_variant/csa_utils/thd_utils.py",
        ),
        (
            "outer_chunk_checks.csa2_context_parallel",
            "megatron/core/transformer/experimental_attention_variant/csa_utils/csa2_context_parallel.py",
        ),
    ):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


class CPGroup:
    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def check_layouts(tape_module, thd, cp):
    from megatron.core.packed_seq_params import PackedSeqParams

    for chunks in (1, 2, 3, 4):
        for ratio in (1, 2):
            if chunks == 4:
                total = 96
                cases = (([0, 96], [0, 93]), ([0, 3, 96], [0, 3, 94]))
            else:
                total = 84
                cases = (([0, 84], [0, 81]), ([0, 3, 84], [0, 3, 82]))
            for physical, logical in cases:
                size = 2
                local_rows = total // (size * chunks)
                x = torch.arange(total, dtype=torch.float64).requires_grad_()
                params = PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=torch.tensor(logical, dtype=torch.int32),
                    cu_seqlens_kv=torch.tensor(logical, dtype=torch.int32),
                    cu_seqlens_q_padded=torch.tensor(physical, dtype=torch.int32),
                    cu_seqlens_kv_padded=torch.tensor(physical, dtype=torch.int32),
                    max_seqlen_q=total,
                    max_seqlen_kv=total,
                    cp_partition_mode="contiguous",
                    cp_chunk_count=chunks,
                )
                tape = tape_module.ChunkTape(chunks)
                old_cumulative_bytes = 0

                def forward(index, mode):
                    tape.begin(index, mode)
                    params.cp_chunk_index = index
                    layouts, parts = [], []
                    for rank in range(size):
                        tokens = thd.build_csa2_thd_layout(
                            params, local_rows, cp_group=CPGroup(rank, size)
                        )
                        layout = cp.build_csa2_cp_compression_layout(tokens, ratio, 2)
                        start = tokens.global_start
                        halo = x.new_zeros(2) if start == 0 else x[start - 2 : start]
                        source = torch.cat((halo, x[start : start + local_rows]))
                        parts.append(
                            source[layout.local.source_indices]
                            .sum(-1)
                            .masked_fill(~layout.local.valid_groups, 0)
                        )
                        layouts.append(layout)
                    assert all(
                        layout.global_layout.capacity == layouts[0].global_layout.capacity
                        for layout in layouts
                    )
                    gathered = torch.cat(parts)
                    mapped = layouts[0].to_sequence_major(gathered)
                    global_layout = layouts[0].global_layout
                    historical = global_layout.source_indices[:, -1] < index * size * local_rows
                    actual = tape_module.extend_chunk_kv(tape, 0, "kv", mapped, historical)
                    expected = (
                        x[global_layout.source_indices]
                        .sum(-1)
                        .masked_fill(~global_layout.valid_groups, 0)
                    )
                    assert torch.equal(actual, expected), (chunks, ratio, physical, index, mode)
                    if chunks == 1:
                        assert actual is mapped
                    return actual, expected

                for index in range(chunks):
                    with torch.no_grad():
                        stored, _ = forward(index, "prefill")
                        old_cumulative_bytes += stored.numel() * stored.element_size()
                tape_bytes = sum(value.untyped_storage().nbytes() for value in tape.values.values())
                correction_bytes = sum(
                    tensor.untyped_storage().nbytes()
                    for indices, values in tape.value_patches.values()
                    for tensor in (indices, values)
                )
                correction_rows = sum(values.shape[0] for _, values in tape.value_patches.values())
                resident_bytes = tape_bytes + correction_bytes
                if chunks > 1:
                    assert resident_bytes < old_cumulative_bytes, (
                        chunks,
                        ratio,
                        physical,
                        logical,
                        correction_rows,
                        resident_bytes,
                        old_cumulative_bytes,
                    )
                actual, expected = forward(chunks - 1, "grad")
                actual.sum().backward()
                tape.collect_all_adjoints()
                for index in reversed(range(chunks - 1)):
                    replay, _ = forward(index, "grad")
                    (replay.sum() * 0).backward()
                    tape.collect_all_adjoints()
                actual_grad = x.grad.clone()
                x.grad = None
                expected.sum().backward()
                assert torch.equal(actual_grad, x.grad), (chunks, ratio, physical, "gradient")
                print(
                    f"PASS: tape bytes K={chunks} r={ratio} corrections={correction_rows} "
                    f"new={resident_bytes} old={old_cumulative_bytes}",
                    flush=True,
                )
    print(
        "PASS: exact cumulative groups/replay gradients and lower tape bytes "
        "(K=1/2/3/4, r1/r2, padding, odd sequence cut)",
        flush=True,
    )


def check_indexer_row_partitions():
    """Check the production score contraction at several CPU row partitions."""
    path = ROOT / "megatron/core/transformer/experimental_attention_variant/csa2.py"
    tree = ast.parse(path.read_text())
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_chunked_csa2_indexer_scores"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    score = namespace["_chunked_csa2_indexer_scores"]
    generator = torch.Generator().manual_seed(20261029)
    for dtype in (torch.float32, torch.bfloat16):
        q = torch.randn(128, 1, 32, 128, generator=generator, dtype=dtype)
        keys = torch.randn(80, 1, 128, generator=generator, dtype=dtype)
        weights = torch.randn(128, 1, 32, generator=generator)
        with torch.no_grad():
            full = score(q, keys, weights)
            for chunks in (1, 2, 4):
                for index in range(chunks):
                    begin, end = index * 128 // chunks, (index + 1) * 128 // chunks
                    local = score(q[begin:end], keys, weights[begin:end])
                    assert torch.equal(full[:, begin:end], local), (dtype, chunks, index)
    print("PASS: production indexer score row partitions (CPU FP32/BF16, K=1/2/4)", flush=True)


def halo_worker(rank, rendezvous):
    """Exercise real collectives, cross-pass adjoints and an absent prior tail."""
    torch.set_num_threads(1)
    tape_module, _, cp = load_modules()
    size = 4
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=size, timeout=timedelta(seconds=10)
    )
    trace = []
    gather = dist.all_gather_into_tensor

    def traced_gather(output, tensor, *, group):
        trace.append((group is dist.group.WORLD, tuple(output.shape), tuple(tensor.shape)))
        return gather(output, tensor, group=group)

    dist.all_gather_into_tensor = traced_gather
    try:
        group = dist.group.WORLD
        tape = tape_module.ChunkTape(2)
        first = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1) + rank * 4
        tape.begin(0, "prefill")
        with torch.no_grad():
            cp.exchange_csa2_boundary_hidden(first, 2, group, tape, 7)
        tape.begin(1, "grad")
        second = (first + size * 4).requires_grad_()
        halo = cp.exchange_csa2_boundary_hidden(second, 2, group, tape, 7)
        expected = first[-2:] - 4 + size * 4
        if rank == 0:
            expected = torch.tensor([14, 15], dtype=torch.float64).reshape(2, 1, 1)
        assert torch.equal(halo, expected), (rank, halo, expected)
        # Fault injection: live cursor divergence must not alter collective order
        # or the pass key used to return an adjoint. The real driver is synchronous.
        tape.begin(rank % 2, "grad")
        (halo * (rank + 1)).sum().backward()
        expected_grad = torch.zeros_like(second)
        if rank + 1 < size:
            expected_grad[-2:] = rank + 2
        assert torch.equal(second.grad, expected_grad), (rank, second.grad)
        tape.begin(0, "grad")
        assert tape.tail_from_previous(7) is None
        replay = first.clone().requires_grad_()
        old_halo = cp.exchange_csa2_boundary_hidden(replay, 2, group, tape, 7)
        expected = torch.zeros_like(old_halo) if rank == 0 else first[-2:] - 4
        assert torch.equal(old_halo, expected), (rank, old_halo)
        # Zero local loss must still run every rank's backward collective.
        (old_halo.sum() * 0).backward()
        expected_grad = torch.zeros_like(replay)
        if rank == size - 1:
            expected_grad[-2:] = 1
        assert torch.equal(replay.grad, expected_grad), (rank, replay.grad)
        # K=1 supplies no tape: only ordinary adjacent-rank halo derivatives exist.
        plain = first.clone().requires_grad_()
        plain_halo = cp.exchange_csa2_boundary_hidden(plain, 2, group)
        assert torch.equal(plain_halo, expected), (rank, plain_halo)
        (plain_halo * (rank + 1)).sum().backward()
        expected_grad = torch.zeros_like(plain)
        if rank + 1 < size:
            expected_grad[-2:] = rank + 2
        assert torch.equal(plain.grad, expected_grad), (rank, plain.grad)
        # Pass 1 without a published slot: there is no cross-pass derivative.
        tape.begin(1, "grad")
        missing = first.clone().requires_grad_()
        missing_halo = cp.exchange_csa2_boundary_hidden(missing, 2, group, tape, 99)
        assert torch.equal(missing_halo, expected), (rank, missing_halo)
        (missing_halo * (rank + 1)).sum().backward()
        assert torch.equal(missing.grad, expected_grad), (rank, missing.grad)
        if rank == size - 1:
            assert torch.equal(tape.tail_grads[(1, 99)], torch.zeros_like(missing_halo))
        # Rank-local tape presence must not change the collective trace either.
        mixed = first.clone().requires_grad_()
        mixed_tape = tape if rank % 2 else None
        mixed_halo = cp.exchange_csa2_boundary_hidden(mixed, 2, group, mixed_tape, 101)
        assert torch.equal(mixed_halo, expected), (rank, mixed_halo)
        (mixed_halo * (rank + 1)).sum().backward()
        assert torch.equal(mixed.grad, expected_grad), (rank, mixed.grad)
        traces = [None] * size
        dist.all_gather_object(traces, trace, group=group)
        assert all(item == [(True, (8, 1, 1), (2, 1, 1))] * 11 for item in traces), traces
    finally:
        dist.all_gather_into_tensor = gather
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-halo", action="store_true", help="Run layout/tape checks only")
    parser.add_argument(
        "--tape-tests",
        action="store_true",
        help="Run tape pytest without the accelerator-only parent conftest",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    modules = load_modules()
    check_layouts(*modules)
    check_indexer_row_partitions()
    if not args.skip_halo:
        with tempfile.TemporaryDirectory(prefix="outer-chunk-numerics-") as directory:
            mp.spawn(halo_worker, args=(f"file://{directory}/rendezvous",), nprocs=4, join=True)
        print("PASS: Gloo CP4 halos, replay adjoints and K=1", flush=True)
    if args.tape_tests:
        import pytest

        package = types.ModuleType("megatron.core.pipeline_parallel")
        package.__path__ = [str(ROOT / "megatron/core/pipeline_parallel")]
        sys.modules[package.__name__] = package
        sys.modules[package.__name__ + ".chunk_tape"] = modules[0]
        test_dir = ROOT / "tests/unit_tests/pipeline_parallel"
        raise SystemExit(
            pytest.main(
                [
                    "-q",
                    f"--confcutdir={test_dir}",
                    str(test_dir / "test_outer_chunk_tape_equivalence.py"),
                ]
            )
        )


if __name__ == "__main__":
    main()
