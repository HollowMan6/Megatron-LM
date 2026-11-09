# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 dense-oracle and shared-gradient tests, adapted from NVIDIA/Megatron-LM#7224.
Original implementation and oracles by Hongxiao Bai. The oracle follows DeepSeek's
published inference implementation and does not call the CSA2 attention operators.

Run this file directly with Python for CPU-only FP4 checks (no Megatron imports);
add --benchmark for isolated old/new timing and peak RSS measurements.
"""

import math
import warnings
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from types import SimpleNamespace


def _fp4_grid_search_reference(x, block_size):
    """Frozen pre-vectorization implementation, including its FP32 distance arithmetic."""
    width = x.shape[-1]
    pad = (-width) % block_size
    values = F.pad(x.float(), (0, pad)).reshape(*x.shape[:-1], -1, block_size)
    amax = values.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * 2.0**-126)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    normalized = (values / scale).clamp(-6.0, 6.0).abs()
    grid = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float32)
    distances = (normalized.unsqueeze(-1) - grid).abs()
    minimum = distances.amin(dim=-1, keepdim=True)
    tied = distances == minimum
    even_indices = torch.arange(8)
    even = even_indices.remainder(2) == 0
    even_tie = torch.where(tied & even, even_indices, 8).amin(dim=-1).clamp_max(7)
    nearest = distances.argmin(dim=-1)
    indices = torch.where(tied.sum(dim=-1).squeeze(-1) > 1, even_tie, nearest)
    rounded = grid[indices] * values.sign() * scale
    return rounded.reshape(*x.shape[:-1], -1)[..., :width].to(x.dtype)


def _fp4_cpu_cases(block_size, dtype):
    generator = torch.Generator().manual_seed(429)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    midpoints = (grid[:-1] + grid[1:]) / 2
    neighbors = torch.cat(
        (
            midpoints,
            torch.nextafter(midpoints, torch.zeros_like(midpoints)),
            torch.nextafter(midpoints, torch.full_like(midpoints, float("inf"))),
        )
    )

    def anchored(probes):
        # A separate row per probe pins scale=1 even for block_size=16.
        rows = torch.zeros(probes.numel(), block_size)
        rows[:, 0], rows[:, -1] = probes, 6.0
        return rows

    boundary = torch.tensor([6.0 * 2.0**e for e in (-126, -125, -10, -1, 0, 1, 10)])
    boundary = torch.cat(
        (
            boundary,
            torch.nextafter(boundary, torch.zeros_like(boundary)),
            torch.nextafter(boundary, torch.full_like(boundary, float("inf"))),
        )
    )
    cases = {
        "normal": torch.randn(5, 3, block_size * 3, generator=generator),
        "wide_exponents": torch.randint(
            0, 0x7F000000, (4096, block_size), generator=generator, dtype=torch.int32
        ).view(torch.float32).clamp_max(torch.finfo(dtype).max),
        "uniform": torch.rand(5, block_size * 2, generator=generator) * 24 - 12,
        "grid_and_negative_grid": anchored(torch.cat((grid, -grid))),
        "midpoints_and_neighbors": anchored(torch.cat((neighbors, -neighbors))),
        "zero_blocks": torch.zeros(3, block_size * 2),
        "signed_zero": anchored(torch.tensor([0.0, -0.0, -0.01, 0.01])),
        "amax_power_boundaries": boundary[:, None].expand(-1, block_size).clone(),
        "two_block_scales": torch.cat((anchored(midpoints), anchored(midpoints) * 16), -1),
        "padded_width": torch.randn(3, block_size + 3, generator=generator),
        "noncontiguous": torch.randn(3, block_size * 2, 2, generator=generator)[..., 0],
    }
    # fp16 cannot represent the extreme scale probes; keep all inputs finite.
    if dtype == torch.float16:
        cases["amax_power_boundaries"] = cases["amax_power_boundaries"].clamp(-65504, 65504)
    return {name: values.to(dtype) for name, values in cases.items()}


def _check_fp4_cpu_case(values, block_size):
    before = values.clone()
    expected = _fp4_grid_search_reference(values, block_size)
    actual = _fp4_fake_quant(values, block_size)
    assert torch.equal(actual, expected)
    # torch.equal treats +0 and -0 as equal; compare bytes as well.
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )
    assert actual.dtype == values.dtype and actual.shape == values.shape
    assert torch.equal(values, before)
    if values.shape[-1] == block_size * 2:
        separate = torch.cat(
            [_fp4_fake_quant(block, block_size) for block in values.split(block_size, -1)], -1
        )
        assert torch.equal(actual, separate)


def _check_fp4_cpu_contract(dtype, block_size):
    values = torch.linspace(-7, 7, block_size + 3, dtype=dtype).requires_grad_()
    before = values.detach().clone()
    gradient = torch.linspace(-3, 2, values.numel(), dtype=dtype)
    _fp4_fake_quant(values, block_size).backward(gradient)
    assert torch.equal(values.grad, gradient)
    assert torch.equal(values.detach(), before)
    disabled = _maybe_fp4_fake_quant(values, False, block_size)
    assert disabled is values and disabled.data_ptr() == values.data_ptr()
    assert torch.equal(
        _maybe_fp4_fake_quant(values, True, block_size), _fp4_fake_quant(values, block_size)
    )


def _check_fp4_cpu_workspace(quantize=None):
    """Reject any intermediate larger than the padded FP32 input."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    values = torch.randn(2, 64, dtype=torch.bfloat16)

    class InputSizedWorkspace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for tensor in tree_leaves(result):
                if isinstance(tensor, torch.Tensor):
                    assert tensor.numel() <= values.numel(), (func, tensor.shape)
            return result

    with InputSizedWorkspace():
        (quantize or _fp4_fake_quant)(values, 32)


def _load_fp4_cpu_definitions():
    """Execute the actual quantizer AST without importing GPU-only Megatron modules."""
    import ast

    source = Path(__file__).resolve().parents[4] / "megatron/core/transformer"
    source = source / "experimental_attention_variant/csa2.py"
    tree = ast.parse(source.read_text())
    names = {"_FP4FakeQuantSTE", "_fp4_fake_quant", "_maybe_fp4_fake_quant"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), globals())


def _fp4_cpu_main():
    import argparse
    import json
    import resource
    import statistics
    import subprocess
    import sys
    import time

    parser = argparse.ArgumentParser(description="CPU-only FP4 equivalence and RSS benchmark")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--backend", choices=("old", "new"), help=argparse.SUPPRESS)
    parser.add_argument("--shape", type=int, nargs=3, default=(2048, 64, 128))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    _load_fp4_cpu_definitions()
    if args.backend:
        quantize = _fp4_grid_search_reference if args.backend == "old" else _fp4_fake_quant
        torch.manual_seed(429)
        values = torch.randn(*args.shape, dtype=torch.bfloat16)
        baseline_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        quantize(values[:1], 32)  # Warm the operators without retaining a full-size output.
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = quantize(values, 32)
            times.append(time.perf_counter() - start)
            del result
        peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "shape": args.shape,
                    "dtype": "bfloat16",
                    "threads": args.threads,
                    "repeats": args.repeats,
                    "seconds": times,
                    "median_seconds": statistics.median(times),
                    "baseline_mib": baseline_kib / 1024,
                    "peak_rss_mib": peak_kib / 1024,
                    "peak_increase_mib": (peak_kib - baseline_kib) / 1024,
                }
            )
        )
        return
    passed = failed = 0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for block_size in (16, 32):
            cases = _fp4_cpu_cases(block_size, dtype)
            for name, values in cases.items():
                label = f"{dtype} block={block_size} {name}"
                try:
                    _check_fp4_cpu_case(values, block_size)
                except AssertionError:
                    print(f"FAIL {label}")
                    failed += 1
                else:
                    print(f"PASS {label}")
                    passed += 1
            try:
                _check_fp4_cpu_contract(dtype, block_size)
            except AssertionError:
                print(f"FAIL {dtype} block={block_size} STE/disabled identity")
                failed += 1
            else:
                print(f"PASS {dtype} block={block_size} STE/disabled identity")
                passed += 1
    try:
        _check_fp4_cpu_workspace()
    except AssertionError:
        print("FAIL input-sized workspace")
        failed += 1
    else:
        print("PASS input-sized workspace")
        passed += 1
    print(f"TOTAL {passed} PASS / {failed} FAIL")
    if failed:
        raise SystemExit(1)
    if args.benchmark:
        sys.stdout.flush()
        for backend in ("old", "new"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--backend",
                backend,
                "--shape",
                *map(str, args.shape),
                "--threads",
                str(args.threads),
                "--repeats",
                str(args.repeats),
            ]
            subprocess.run(command, check=True)
        elements = math.prod(args.shape)
        print(
            f"Old distances alone: {elements * 8 * 4 / 1024**2:.1f} MiB; "
            f"new FP32 input-shaped temporary: {elements * 4 / 1024**2:.1f} MiB"
        )


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    _fp4_cpu_main()
    raise SystemExit(0)


import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fp8_utils import get_fp8_context, get_fp8_disabled_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CompressedSparseAttention2,
    CSA2Compressor,
    CSA2Indexer,
    CSA2IndexerSubmodules,
    CSA2State,
    _fp4_fake_quant,
    _maybe_fp4_fake_quant,
    select_candidate_blocks,
)
from megatron.core.transformer.experimental_attention_variant.csa2_config import (
    validate_csa2_config,
)
from megatron.core.transformer.experimental_attention_variant.csa2_module_spec import (
    csa2_attention_spec,
    get_csa2_module_spec_for_backend,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    csa2_candidates as csa2_candidates_module,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
    CSA2CandidateBlocks,
    candidate_blocks_from_block_scores,
    candidate_blocks_from_scores,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_context_parallel import (
    build_csa2_cp_compression_layout,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    build_csa2_thd_layout,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import _make_config


def test_fp4_fake_quant_cpu_workspace():
    """A broadcast grid search would exceed this per-intermediate allocation bound."""
    _check_fp4_cpu_workspace()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_fp4_fake_quant_cpu_grid_search_equivalence(dtype, block_size):
    """Pin the vectorized quantizer against the original grid-search arithmetic."""
    for values in _fp4_cpu_cases(block_size, dtype).values():
        _check_fp4_cpu_case(values, block_size)
    _check_fp4_cpu_contract(dtype, block_size)


@pytest.mark.parametrize("block_size", [16, 32])
def test_fp4_fake_quant_matches_e2m1_grid_and_scale(block_size):
    """FP4 blocks use power-of-two scales and round ties to even grid indices."""
    values = torch.zeros(1, block_size * 2, dtype=torch.float32)
    values[0, :8] = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 5.0, 6.0])
    values[0, block_size] = 24.0
    quantized = _fp4_fake_quant(values, block_size)
    expected_grid = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 6.0])
    torch.testing.assert_close(quantized[0, :8], expected_grid)
    assert torch.equal(quantized[0, block_size], torch.tensor(24.0))
    assert torch.equal(quantized[0, 8], torch.tensor(0.0))


def test_fp4_fake_quant_uses_independent_block_scales_and_power_boundaries():
    """Each block chooses its own ceil-log2 scale, including amax power crossings."""
    values = torch.tensor([[6.0, 10.0, 12.01, 11.0]])
    quantized = _fp4_fake_quant(values, block_size=2)
    torch.testing.assert_close(quantized, torch.tensor([[6.0, 8.0, 12.0, 12.0]]))


def test_fp4_fake_quant_has_straight_through_gradient():
    """Quantization changes forward values while preserving the input gradient."""
    values = torch.tensor([[0.3, 1.2, 7.0]], requires_grad=True)
    _fp4_fake_quant(values, block_size=2).sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


def test_fp4_fake_quant_disabled_path_is_bitwise_baseline():
    """Default-off config flags preserve the baseline tensor bit for bit."""
    config = _make_config()
    assert config.csa2_fp4_compressed_kv is False
    assert config.csa2_fp4_indexer_qk is False
    baseline = torch.randn(3, 32, dtype=torch.bfloat16)
    after_change = _maybe_fp4_fake_quant(
        baseline, config.csa2_fp4_compressed_kv, block_size=16
    )
    assert torch.equal(after_change, baseline)
    assert after_change.data_ptr() == baseline.data_ptr()


@pytest.mark.parametrize(
    ("compressed_kv", "indexer_qk"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_fp4_fake_quant_switches_are_independent(compressed_kv, indexer_qk):
    """Both fake-quant sites can be enabled independently or together."""
    config = _make_config(
        csa2_fp4_compressed_kv=compressed_kv, csa2_fp4_indexer_qk=indexer_qk
    )
    assert config.csa2_fp4_compressed_kv is compressed_kv
    assert config.csa2_fp4_indexer_qk is indexer_qk


def test_global_fp8_is_accepted_by_csa2_validation():
    """A global FP8 run is valid: CSA2 keeps its own attention GEMMs out of the autocast."""
    # Construction runs validate_csa2_config() through MLATransformerConfig.__post_init__.
    config = _make_config(fp8="hybrid")
    assert config.fp8 == "hybrid"
    validate_csa2_config(config)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("fp4", "e2m1"), ("quant_recipe", object())],
    ids=["fp4", "quant_recipe"],
)
def test_fp4_and_weight_qat_are_still_rejected(field_name, value):
    """FP4 and a QAT recipe have no CSA2 exemption path, so their guard stays."""
    config = _make_config()
    setattr(config, field_name, value)
    with pytest.raises(NotImplementedError, match="QAT"):
        validate_csa2_config(config)


@pytest.mark.skipif(
    not HAVE_TE or not torch.cuda.is_available(),
    reason="TE FP8 global state needs Transformer Engine on a CUDA device",
)
def test_fp8_disabled_context_turns_fp8_off_around_csa2_gemms():
    """Inside a global FP8 run the wrapper used by CSA2 reports TE FP8 as disabled."""
    import transformer_engine.pytorch.fp8  # noqa: F401  (TE registers its fp8 submodule)
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    config = _make_config(fp8="hybrid")
    # With quantization off the wrapper must be a genuine no-op, not a disabled context.
    assert isinstance(get_fp8_disabled_context(_make_config()), nullcontext)
    assert not isinstance(get_fp8_disabled_context(config), nullcontext)
    with get_fp8_context(config):
        assert FP8GlobalStateManager.is_fp8_enabled()
        with get_fp8_disabled_context(config):
            assert not FP8GlobalStateManager.is_fp8_enabled()
        assert FP8GlobalStateManager.is_fp8_enabled()
    assert not FP8GlobalStateManager.is_fp8_enabled()


def test_forward_runs_under_the_disabled_quantization_context(pg_collection, monkeypatch):
    """CompressedSparseAttention2.forward enters the disabled context around its GEMMs."""
    contexts = []
    real_context = get_fp8_disabled_context

    def record(config):
        context = real_context(config)
        contexts.append((config, context))
        return context

    monkeypatch.setattr(
        "megatron.core.transformer.experimental_attention_variant.csa2.get_fp8_disabled_context",
        record,
    )
    layer = _layer(pg_collection, 2)
    x = torch.randn(5, 2, layer.config.hidden_size, device="cuda")
    layer(x, None)
    assert len(contexts) == 1
    assert contexts[0][0] is layer.config
    assert isinstance(contexts[0][1], nullcontext)


class _IEEEFP32Linear(nn.Module):
    """TP=1 test backend separating attention math from TE's TF32 GEMM policy.

    BF16 oracle and fused-attention cases still exercise the real TE projections.
    """

    def __init__(
        self,
        input_size,
        output_size,
        config,
        init_method,
        bias=False,
        skip_bias_add=False,
        **kwargs,
    ):
        super().__init__()
        assert config.tensor_model_parallel_size == 1
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=config.params_dtype))
        self.bias = (
            nn.Parameter(torch.zeros(output_size, dtype=config.params_dtype)) if bias else None
        )
        self.skip_bias_add = skip_bias_add
        if config.perform_initialization:
            init_method(self.weight)

    def forward(self, hidden_states):
        """Use IEEE PyTorch GEMM and the standard optional-bias return contract."""
        bias = None if self.skip_bias_add else self.bias
        return F.linear(hidden_states, self.weight, bias), self.bias if self.skip_bias_add else None


class _IEEEFP32Backend(TESpecProvider):
    """Keep real normalization/RoPE modules while making FP32 projections IEEE."""

    def linear(self):
        """Build an unpartitioned IEEE projection."""
        return _IEEEFP32Linear

    def column_parallel_linear(self):
        """TP=1 column projection has the same IEEE contract."""
        return _IEEEFP32Linear

    def row_parallel_linear(self):
        """TP=1 row projection has the same IEEE contract."""
        return _IEEEFP32Linear


def _attention_spec(dtype):
    """Keep strict FP32 math tests independent of cached vendor GEMM precision."""
    return (
        get_csa2_module_spec_for_backend(_IEEEFP32Backend())
        if dtype == torch.float32
        else csa2_attention_spec
    )


@pytest.fixture(autouse=True)
def ieee_fp32_oracle():
    # CUDA libraries may cache NVIDIA_TF32_OVERRIDE before a per-test fixture runs.
    # FP32 projection oracles therefore use the explicit IEEE backend above;
    # BF16 continues to cover TE, including real fused forward/backward kernels.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_fused_attention_matches_native_for_multiple_batches(pg_collection, ratio):
    """Exercise fused index packing and KV gradients with distinct batch elements."""
    pytest.importorskip("flash_mla")
    cudnn = pytest.importorskip("cudnn")
    if not hasattr(cudnn, "DSA") or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires cuDNN DSA on SM90 or newer")
    torch.manual_seed(1234)
    shape = dict(
        v_head_dim=512,
        qk_pos_emb_head_dim=64,
        num_attention_heads=8,
        dsa_indexer_head_dim=128,
        dsa_indexer_n_heads=4,
    )
    native = _layer(pg_collection, ratio, torch.bfloat16, **shape)
    fused = _layer(pg_collection, ratio, torch.bfloat16, dsa_kernel_backend="cudnn", **shape)
    # Preserve output variance when moving the oracle from 16- to 512-wide heads.
    # A fixed std=0.2 projection over 2048 channels amplifies BF16 kernel rounding.
    torch.nn.init.normal_(
        native.linear_o_group_proj, std=native.linear_o_group_proj.shape[-1] ** -0.5
    )
    fused.load_state_dict(native.state_dict())
    core_outputs = {}
    native.core_attention.register_forward_hook(
        lambda _module, _inputs, output: core_outputs.update(native=output.detach())
    )
    fused.core_attention.register_forward_hook(
        lambda _module, _inputs, output: core_outputs.update(fused=output.detach())
    )
    x = torch.randn(9, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    expected = native(x, None)[0]
    try:
        actual = fused(other, None)[0]
    except RuntimeError as error:
        if torch.cuda.get_device_capability()[0] == 9 and "no kernel image is available" in str(
            error
        ):
            pytest.skip("Installed FlashMLA was built without SM90 kernels")
        raise
    # Match the existing CSA backend's BF16 kernel tolerance before projection.
    torch.testing.assert_close(core_outputs["fused"], core_outputs["native"], atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=3e-2)
    gradient = torch.randn_like(expected)
    (expected * gradient).sum().backward()
    (actual * gradient).sum().backward()
    torch.testing.assert_close(other.grad, x.grad, atol=5e-2, rtol=8e-2)
    if ratio:
        torch.testing.assert_close(
            fused.core_attention.compressor.linear_wkv.weight.grad,
            native.core_attention.compressor.linear_wkv.weight.grad,
            atol=5e-2,
            rtol=8e-2,
        )


@pytest.fixture
def pg_collection():
    # Native indexer-loss tests below do not need CUDA or Transformer Engine.
    if not torch.cuda.is_available():
        pytest.skip("Megatron RoPE requires CUDA")
    if not HAVE_TE:
        pytest.skip("Transformer Engine is not installed")
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


def _layer(pg_collection, ratio, dtype=torch.float32, **overrides):
    config_values = dict(
        num_layers=1,
        csa_compress_ratios=[ratio],
        csa2_kv_source_layers=[0] if ratio else [],
        csa2_index_source_layers=[0] if ratio else [],
        csa2_candidate_source_layer=None,
        csa2_candidate_topk_blocks=0,
        csa2_candidate_block_size=0,
        dsa_indexer_topk=2,
    )
    config_values.update(overrides)
    config = _make_config(params_dtype=dtype, **config_values)
    layer = build_module(
        _attention_spec(dtype), config=config, layer_number=1, pg_collection=pg_collection
    ).cuda()
    # Nontrivial query magnitudes, gates, and sink expose normalization/softmax mistakes.
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if "norm.weight" in name:
                param.uniform_(0.7, 1.3)
            else:
                param.normal_(std=0.2)
    return layer


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_thd_core_matches_independent_sequences_with_padding(pg_collection, ratio):
    """Packed CSA2 must restart windows and compression at each logical segment."""
    layer = _layer(pg_collection, ratio)
    core = layer.core_attention
    config = layer.config
    lengths = (3, 4)
    logical = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
    physical = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=7,
        max_seqlen_kv=7,
    )
    shape = (12, 1)
    q = torch.randn(*shape, config.num_attention_heads, config.v_head_dim, device="cuda")
    k = torch.randn(*shape, 1, config.v_head_dim, device="cuda")
    x = torch.randn(*shape, config.hidden_size, device="cuda")
    qr = torch.randn(*shape, config.q_lora_rank, device="cuda")
    valid = torch.tensor([0, 1, 2, 5, 6, 7, 8], device="cuda")
    padding = torch.tensor([3, 4, 9, 10, 11], device="cuda")
    for tensor in (q, k, x, qr):
        tensor[padding] = float("nan")
    inputs = [tensor.requires_grad_() for tensor in (q, k, x, qr)]
    output = core(
        inputs[0], inputs[1], inputs[1], None, packed_seq_params=packed, x=inputs[2], qr=inputs[3]
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[padding], torch.zeros_like(output[padding]))
    gradient = torch.randn_like(output)
    gradient[padding] = 0
    (output * gradient).sum().backward()
    packed_grads = [
        tensor.grad[valid].clone() if tensor.grad is not None else None for tensor in inputs
    ]
    weight_grad = core.compressor.linear_wkv.weight.grad.clone() if ratio else None
    core.zero_grad(set_to_none=True)

    references = [tensor.detach().clone().requires_grad_() for tensor in (q, k, x, qr)]
    expected = []
    for start, length in zip((0, 5), lengths):
        result = core(
            references[0][start : start + length],
            references[1][start : start + length],
            references[1][start : start + length],
            None,
            x=references[2][start : start + length],
            qr=references[3][start : start + length],
        )
        expected.append(result)
        (result * gradient[start : start + length]).sum().backward()
    torch.testing.assert_close(output[valid], torch.cat(expected, dim=0), atol=2e-5, rtol=2e-4)
    for actual, reference in zip(packed_grads, references):
        if actual is None:
            assert reference.grad is None
        else:
            torch.testing.assert_close(actual, reference.grad[valid], atol=2e-5, rtol=2e-4)
    if ratio:
        torch.testing.assert_close(
            weight_grad, core.compressor.linear_wkv.weight.grad, atol=2e-5, rtol=2e-4
        )


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_thd_attention_wrapper_matches_independent_sequences(pg_collection, ratio):
    """QKV and output RoPE must restart at every packed sequence boundary."""
    layer = _layer(pg_collection, ratio)
    logical = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
    physical = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=7,
        max_seqlen_kv=7,
    )
    x = torch.randn(12, 1, layer.config.hidden_size, device="cuda")
    valid = torch.tensor([0, 1, 2, 5, 6, 7, 8], device="cuda")
    x[[3, 4, 9, 10, 11]] = float("nan")
    x.requires_grad_()
    actual, _ = layer(x, None, packed_seq_params=packed)
    assert torch.isfinite(actual).all()
    actual[valid].square().sum().backward()
    packed_grad = x.grad[valid].clone()
    layer.zero_grad(set_to_none=True)
    reference_x = x.detach().clone().requires_grad_()
    expected = []
    for start, length in ((0, 3), (5, 4)):
        output, _ = layer(reference_x[start : start + length], None)
        expected.append(output)
        output.square().sum().backward()
    torch.testing.assert_close(actual[valid], torch.cat(expected), atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(packed_grad, reference_x.grad[valid], atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("backend", ["none", "cudnn"])
def test_thd_cp_matches_unsplit_forward_and_gradients(ratio, backend):
    """A group crossing the CP cut has one owner and receives both ranks' gradients."""
    if Utils.world_size not in (2, 4):
        pytest.skip("run with torchrun --nproc-per-node=2 or 4")
    if backend == "cudnn":
        pytest.importorskip("flash_mla")
        cudnn = pytest.importorskip("cudnn")
        if not hasattr(cudnn, "DSA") or torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("requires cuDNN DSA on SM90 or newer")
    world = Utils.world_size
    Utils.initialize_model_parallel(context_parallel_size=world)
    model_parallel_cuda_manual_seed(1234)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    singleton_groups = [torch.distributed.new_group([rank]) for rank in range(world)]
    reference_groups = copy(groups)
    reference_groups.cp = singleton_groups[torch.distributed.get_rank()]
    try:
        torch.manual_seed(1234)
        overrides = {"context_parallel_size": world}
        dtype = torch.float32
        if backend == "cudnn":
            dtype = torch.bfloat16
            overrides.update(
                dsa_kernel_backend="cudnn",
                v_head_dim=512,
                qk_pos_emb_head_dim=64,
                num_attention_heads=8,
                dsa_indexer_head_dim=128,
                dsa_indexer_n_heads=4,
            )
        distributed_layer = _layer(groups, ratio, dtype, **overrides)
        overrides["context_parallel_size"] = 1
        reference_layer = _layer(reference_groups, ratio, dtype, **overrides)
        reference_layer.load_state_dict(distributed_layer.state_dict())
        total_rows = 4 * world
        global_x = torch.randn(
            total_rows, 1, distributed_layer.config.hidden_size, device="cuda", dtype=dtype
        )
        rank = torch.distributed.get_rank()
        local_x = global_x[rank * 4 : (rank + 1) * 4].detach().clone().requires_grad_()
        reference_x = global_x.detach().clone().requires_grad_()
        logical = torch.tensor([0, 3, total_rows], device="cuda", dtype=torch.int32)
        packed_cp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=logical,
            cu_seqlens_kv=logical,
            max_seqlen_q=total_rows - 3,
            max_seqlen_kv=total_rows - 3,
            cp_partition_mode="contiguous",
        )
        packed_ref = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=logical,
            cu_seqlens_kv=logical,
            max_seqlen_q=total_rows - 3,
            max_seqlen_kv=total_rows - 3,
        )
        actual, _ = distributed_layer(local_x, None, packed_seq_params=packed_cp)
        expected, _ = reference_layer(reference_x, None, packed_seq_params=packed_ref)
        expected_local = expected[rank * 4 : (rank + 1) * 4]
        error = (actual - expected_local).abs().amax()
        torch.distributed.all_reduce(error, op=torch.distributed.ReduceOp.MAX, group=groups.cp)
        tolerance = 3e-2 if backend == "cudnn" else 1e-4
        if error > tolerance:
            pytest.fail(f"CP forward maximum error {error.item():.6g}")
        actual.square().sum().backward()
        expected.square().sum().backward()
        grad_error = (local_x.grad - reference_x.grad[rank * 4 : (rank + 1) * 4]).abs().amax()
        grad_scale = reference_x.grad[rank * 4 : (rank + 1) * 4].abs().amax()
        torch.distributed.all_reduce(grad_error, op=torch.distributed.ReduceOp.MAX, group=groups.cp)
        torch.distributed.all_reduce(grad_scale, op=torch.distributed.ReduceOp.MAX, group=groups.cp)
        grad_tolerance = 8e-2 if backend == "cudnn" else 1e-4
        if grad_error > grad_tolerance + (2e-2 if backend == "cudnn" else 0) * grad_scale:
            pytest.fail(
                f"CP input gradient maximum error {grad_error.item():.6g}, "
                f"reference maximum {grad_scale.item():.6g}"
            )
        if ratio:
            distributed_grad = distributed_layer.core_attention.compressor.linear_wkv.weight.grad
            torch.distributed.all_reduce(distributed_grad, group=groups.cp)
            weight_error = (
                (
                    distributed_grad
                    - reference_layer.core_attention.compressor.linear_wkv.weight.grad
                )
                .abs()
                .amax()
            )
            weight_scale = (
                reference_layer.core_attention.compressor.linear_wkv.weight.grad.abs().amax()
            )
            torch.distributed.all_reduce(
                weight_error, op=torch.distributed.ReduceOp.MAX, group=groups.cp
            )
            torch.distributed.all_reduce(
                weight_scale, op=torch.distributed.ReduceOp.MAX, group=groups.cp
            )
            if weight_error > grad_tolerance + (2e-2 if backend == "cudnn" else 0) * weight_scale:
                pytest.fail(
                    f"CP compressor gradient maximum error {weight_error.item():.6g}, "
                    f"reference maximum {weight_scale.item():.6g}"
                )
    finally:
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
        Utils.inited = False


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("ratio", [1, 2])
def test_thd_cp_indexer_loss_matches_unsplit(ratio, per_token, sparse):
    """Under CP the indexer KL loss stays exact: the tracker receives the CP group
    so its flush averages the per-shard token means into the unsplit value, and
    the indexer K-projection gradients average (mean mode) or sum (per-token
    mode) across shards to the unsplit gradient."""
    if Utils.world_size not in (2, 4):
        pytest.skip("run with torchrun --nproc-per-node=2 or 4")
    world = Utils.world_size
    Utils.initialize_model_parallel(context_parallel_size=world)
    model_parallel_cuda_manual_seed(1234)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    singleton_groups = [torch.distributed.new_group([rank]) for rank in range(world)]
    reference_groups = copy(groups)
    reference_groups.cp = singleton_groups[torch.distributed.get_rank()]
    try:
        torch.manual_seed(1234)
        overrides = {
            "context_parallel_size": world,
            "dsa_indexer_loss_coeff": 1.0,
            "calculate_per_token_loss": per_token,
            "dsa_indexer_use_sparse_loss": sparse,
        }
        dtype = torch.float32
        distributed_layer = _layer(groups, ratio, dtype, **overrides)
        overrides["context_parallel_size"] = 1
        reference_layer = _layer(reference_groups, ratio, dtype, **overrides)
        reference_layer.load_state_dict(distributed_layer.state_dict())
        total_rows = 4 * world
        global_x = torch.randn(
            total_rows, 1, distributed_layer.config.hidden_size, device="cuda", dtype=dtype
        )
        rank = torch.distributed.get_rank()
        local_x = global_x[rank * 4 : (rank + 1) * 4].detach().clone().requires_grad_()
        reference_x = global_x.detach().clone().requires_grad_()
        logical = torch.tensor([0, 3, total_rows], device="cuda", dtype=torch.int32)
        packed_cp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=logical,
            cu_seqlens_kv=logical,
            max_seqlen_q=total_rows - 3,
            max_seqlen_kv=total_rows - 3,
            cp_partition_mode="contiguous",
        )
        packed_ref = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=logical,
            cu_seqlens_kv=logical,
            max_seqlen_q=total_rows - 3,
            max_seqlen_kv=total_rows - 3,
        )
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        actual, _ = distributed_layer(local_x, None, packed_seq_params=packed_cp)
        tracker = DSAIndexerLossLoggingHelper.tracker
        assert "values" in tracker, "CP forward must log the indexer loss"
        # This rank's shard mean is stored together with the CP group for the flush;
        # reduce_group stays empty because the saved value is a token mean in both
        # reduction modes (unlike the DSA line, which SUMs in per-token mode).
        assert tracker["avg_group"] is not None
        assert torch.distributed.get_process_group_ranks(
            tracker["avg_group"]
        ) == torch.distributed.get_process_group_ranks(groups.cp)
        assert tracker["reduce_group"] is None
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
            distributed_layer.core_attention.pg_collection
        )
        distributed_logged = tracker["values"].clone()
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        expected, _ = reference_layer(reference_x, None, packed_seq_params=packed_ref)
        assert DSAIndexerLossLoggingHelper.tracker["avg_group"] is None
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
            reference_layer.core_attention.pg_collection
        )
        reference_logged = DSAIndexerLossLoggingHelper.tracker["values"].clone()
        # The flushed CP value is the unsplit token mean, and the loss is real.
        assert reference_logged.abs().sum() > 0
        torch.testing.assert_close(distributed_logged, reference_logged, atol=1e-4, rtol=1e-4)
        # Forward parity: the loss attaches without changing the attention output.
        expected_local = expected[rank * 4 : (rank + 1) * 4]
        error = (actual - expected_local).abs().amax()
        torch.distributed.all_reduce(error, op=torch.distributed.ReduceOp.MAX, group=groups.cp)
        if error > 1e-4:
            pytest.fail(f"CP forward maximum error {error.item():.6g}")
        actual.square().sum().backward()
        expected.square().sum().backward()
        # Indexer K-projection gradients: the loss normalizer differs by mode, so
        # the shard reduction matching the unsplit gradient is AVG in mean mode
        # (each shard saves a local mean) and SUM in per-token mode (each shard
        # saves a raw local sum).
        distributed_grad = distributed_layer.core_attention.indexer.linear_wk.weight.grad
        if per_token:
            torch.distributed.all_reduce(distributed_grad, group=groups.cp)
        else:
            torch.distributed.all_reduce(
                distributed_grad, op=torch.distributed.ReduceOp.AVG, group=groups.cp
            )
        reference_grad = reference_layer.core_attention.indexer.linear_wk.weight.grad
        weight_error = (distributed_grad - reference_grad).abs().amax()
        grad_scale = reference_grad.abs().amax()
        torch.distributed.all_reduce(
            weight_error, op=torch.distributed.ReduceOp.MAX, group=groups.cp
        )
        if weight_error > 1e-4 + 1e-2 * grad_scale:
            pytest.fail(
                f"CP indexer gradient maximum error {weight_error.item():.6g}, "
                f"reference maximum {grad_scale.item():.6g}"
            )
    finally:
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
        Utils.inited = False


def _reference_rope(x, config, ratio, position_stride=1, inverse=False):
    dim = config.qk_pos_emb_head_dim
    base = config.csa_compress_rotary_base if ratio else config.rotary_base
    freq = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    if ratio:

        def corrected_dim(rotations):
            return (
                dim
                * math.log(config.original_max_position_embeddings / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(config.beta_fast)), 0)
        high = min(math.ceil(corrected_dim(config.beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, device=x.device, dtype=torch.float32) - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        freq = freq / config.rotary_scaling_factor * ramp + freq * (1 - ramp)
    positions = torch.arange(x.shape[0], device=x.device, dtype=torch.float32) * position_stride
    angles = torch.outer(positions, freq)
    # Preserve V4's RoPE precision: cast trig values to the activation dtype before
    # the two products and their sum, instead of promoting the activation to FP32.
    shape = (x.shape[0], *([1] * (x.ndim - 2)), dim // 2)
    cos = angles.cos().to(x.dtype).reshape(shape)
    sin = angles.sin().to(x.dtype).reshape(shape)
    if inverse:
        sin = -sin
    even, odd = x[..., -dim::2], x[..., -dim + 1 :: 2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    return torch.cat((x[..., :-dim], rotated), dim=-1)


def _reference_norm(x, weight, eps):
    xf = x.float()
    return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt() * weight).to(x.dtype)


def _reference(x, weights, config, ratio):
    """Dense oracle including the grouped output projection and indexer score graph."""

    def linear(tensor, name):
        return F.linear(tensor, weights[name + ".weight"])

    def norm(tensor, name):
        return _reference_norm(
            tensor, weights[name + ".weight"], config.attention_latent_norm_epsilon
        )

    seq_len, batch = x.shape[:2]
    qr = norm(linear(x, "linear_q_down_proj"), "q_layernorm")
    q = linear(qr, "linear_q_up_proj").reshape(
        seq_len, batch, config.num_attention_heads, config.v_head_dim
    )
    q = _reference_rope(q, config, ratio)
    local_kv = _reference_rope(norm(linear(x, "linear_kv_proj"), "kv_layernorm"), config, ratio)
    qi = torch.arange(seq_len, device=x.device)[:, None]
    ki = torch.arange(seq_len, device=x.device)[None, :]
    visible = ((ki <= qi) & (ki > qi - config.csa_window_size)).expand(batch, -1, -1)
    kv = local_kv
    index_scores = None
    if ratio:
        prefix = "core_attention.compressor."
        if ratio == 1:
            latent = linear(x, prefix + "linear_wkv")
        else:
            # Stack whole groups explicitly, without the implementation's reshape/pooling path.
            groups = []
            for start in range(0, seq_len - ratio + 1, ratio):
                tokens = x[start : start + ratio].float()
                projected = linear(tokens, prefix + "linear_wkv")
                logits = linear(tokens, prefix + "linear_wgate")
                groups.append((projected * logits.softmax(dim=0)).sum(dim=0))
            if groups:
                latent = torch.stack(groups).to(x.dtype)
            else:
                # Preserve zero derivatives to both projections when no group is complete.
                latent = (
                    linear(x[:0].float(), prefix + "linear_wkv")
                    + linear(x[:0].float(), prefix + "linear_wgate")
                ).to(x.dtype)
        latent = norm(latent, prefix + "norm")
        prefix = "core_attention.indexer."
        index_q = linear(qr, prefix + "linear_wq_b").reshape(
            seq_len, batch, config.dsa_indexer_n_heads, config.dsa_indexer_head_dim
        )
        index_q = _reference_rope(index_q, config, ratio)
        index_k = norm(linear(latent, prefix + "linear_wk"), prefix + "k_norm")
        index_k = _reference_rope(index_k, config, ratio, position_stride=ratio)
        index_weights = linear(x, prefix + "linear_weights_proj").float()
        index_scores = torch.einsum("sbhd,tbd->bsht", index_q.float(), index_k.float()).relu()
        index_scores = (
            index_scores
            * index_weights.permute(1, 0, 2).unsqueeze(-1)
            * (config.dsa_indexer_head_dim**-0.5 * config.dsa_indexer_n_heads**-0.5)
        ).sum(dim=2)
        global_visible = torch.arange(latent.shape[0], device=x.device) < (qi + 1) // ratio
        index_scores = index_scores.masked_fill(~global_visible, -torch.inf)
        selected = index_scores.argsort(dim=-1, descending=True, stable=True)[
            ..., : min(config.dsa_indexer_topk, latent.shape[0])
        ]
        selected_mask = torch.zeros_like(index_scores, dtype=torch.bool).scatter(-1, selected, True)
        visible = torch.cat((visible, selected_mask & global_visible), dim=-1)
        kv = torch.cat((kv, _reference_rope(latent, config, ratio, position_stride=ratio)))
    # K and V are the same latent. Combine both gradient contributions in FP32 before
    # casting back, just as for the FP32 attention computation in the sparse path.
    kv_float = kv.float()
    scores = torch.einsum("sbhd,tbd->bsht", q.float(), kv_float) * config.v_head_dim**-0.5
    scores = scores.masked_fill(~visible.unsqueeze(2), -torch.inf)
    sink = weights["core_attention.attn_sink"].reshape(1, 1, -1, 1).expand(batch, seq_len, -1, 1)
    probs = torch.cat((scores, sink), dim=-1).softmax(dim=-1)[..., :-1]
    output = torch.einsum("bsht,tbd->sbhd", probs, kv_float).to(x.dtype)
    output = _reference_rope(output, config, ratio, inverse=True)
    output = output.reshape(seq_len, batch, config.output_projection_groups, -1)
    group_weight = weights["linear_o_group_proj"].reshape(
        config.output_projection_groups, config.output_projection_lora_rank, -1
    )
    # One independent projection per group; use matmul instead of the wrapper's einsum.
    output = torch.stack(
        [
            F.linear(output[:, :, group], group_weight[group])
            for group in range(config.output_projection_groups)
        ],
        dim=2,
    ).flatten(-2)
    return linear(output, "linear_proj"), index_scores


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("seq_len", [1, 3, 5, 9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_and_backward_match_dense_reference(pg_collection, ratio, seq_len, dtype):
    torch.manual_seed(1234)
    layer = _layer(pg_collection, ratio, dtype)
    x = torch.randn(
        seq_len, 2, layer.config.hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = {
        name: p.detach().clone().requires_grad_() for name, p in layer.named_parameters()
    }
    actual, bias = layer(x, None)
    expected, _ = _reference(ref_x, ref_weights, layer.config, ratio)
    assert bias is None
    tol = dict(atol=2e-6, rtol=2e-5) if dtype == torch.float32 else dict(atol=1.5e-2, rtol=3e-2)
    torch.testing.assert_close(actual, expected, **tol)
    grad = torch.randn_like(actual)
    (actual * grad).sum().backward()
    (expected * grad).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, **tol)
    for name, param in layer.named_parameters():
        if "indexer." in name:
            # Discrete selection intentionally does not imply an indexer training objective.
            assert param.grad is None and ref_weights[name].grad is None
        else:
            assert param.grad is not None, name
            torch.testing.assert_close(
                param.grad, ref_weights[name].grad, msg=lambda msg: f"{name}: {msg}", **tol
            )
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_indexer_scores_and_auxiliary_gradients(pg_collection, ratio, dtype):
    torch.manual_seed(42)
    layer = _layer(pg_collection, ratio, dtype)
    x = torch.randn(9, 2, layer.config.hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = {
        name: p.detach().clone().requires_grad_() for name, p in layer.named_parameters()
    }
    qr = layer.q_layernorm(layer.linear_q_down_proj(x)[0])
    latent = layer.core_attention.compressor(x)
    scores = layer.core_attention.indexer.forward_before_topk(x, qr, latent, layer.rotary_pos_emb)
    _, ref_scores = _reference(ref_x, ref_weights, layer.config, ratio)
    tol = dict(atol=3e-6, rtol=3e-5) if dtype == torch.float32 else dict(atol=2e-2, rtol=3e-2)
    torch.testing.assert_close(scores, ref_scores, **tol)
    finite = scores.isfinite()
    scores[finite].square().mean().backward()
    ref_scores[finite].square().mean().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, **tol)
    for name, param in layer.named_parameters():
        if "indexer." in name:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            assert param.grad.abs().sum() > 0, name
            torch.testing.assert_close(
                param.grad, ref_weights[name].grad, msg=lambda msg: f"{name}: {msg}", **tol
            )


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_causality_and_multiple_live_microbatches(pg_collection, ratio):
    torch.manual_seed(7)
    layer = _layer(pg_collection, ratio)
    x = torch.randn(9, 2, layer.config.hidden_size, device="cuda", requires_grad=True)
    changed = x.detach().clone()
    changed[5:] = torch.randn_like(changed[5:]) * 10
    changed.requires_grad_()
    first, _ = layer(x, None)
    second, _ = layer(changed, None)
    # Query 4 cannot see token 5, which would complete its ratio-2 global group.
    torch.testing.assert_close(first[:5], second[:5], atol=1e-6, rtol=1e-5)
    grad_x, grad_changed = torch.autograd.grad(first[:5].sum() + second[:5].sum(), (x, changed))
    assert torch.count_nonzero(grad_x[5:]) == 0
    assert torch.count_nonzero(grad_changed[5:]) == 0
    assert torch.isfinite(grad_x).all() and torch.isfinite(grad_changed).all()


def test_bf16_conversion_keeps_compressor_and_sink_in_fp32(pg_collection):
    layer = _layer(pg_collection, 2)
    convert_module_to_dtype_except_fp32_marked(layer, torch.bfloat16)
    core = layer.core_attention
    assert core.attn_sink.dtype == torch.float32
    assert core.compressor.linear_wkv.weight.dtype == torch.float32
    assert core.compressor.linear_wgate.weight.dtype == torch.float32
    assert core.compressor.norm.weight.dtype == torch.bfloat16
    assert layer.linear_q_up_proj.weight.dtype == torch.bfloat16
    ratio_one = _layer(pg_collection, 1, torch.bfloat16).core_attention.compressor
    assert ratio_one.linear_wkv.weight.dtype == torch.bfloat16
    assert ratio_one.linear_wgate is None


@pytest.mark.parametrize("layer_number", [3, 5, 6])
def test_cross_layer_modes_require_forward_state(pg_collection, layer_number):
    config = _make_config()
    layer = build_module(
        csa2_attention_spec, config=config, layer_number=layer_number, pg_collection=pg_collection
    ).cuda()
    x = torch.randn(5, 2, config.hidden_size, device="cuda")
    with pytest.raises(ValueError, match="explicit csa2_state"):
        layer(x, None)


def test_candidate_source_can_run_standalone(pg_collection):
    config = _make_config()
    layer = build_module(
        csa2_attention_spec, config=config, layer_number=4, pg_collection=pg_collection
    ).cuda()
    x = torch.randn(5, 2, config.hidden_size, device="cuda", requires_grad=True)
    output, _ = layer(x, None)
    output.float().square().mean().backward()
    assert output.shape == x.shape
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()


def test_causal_mask_and_unsupported_packing(pg_collection):
    layer = _layer(pg_collection, 2)
    x = torch.randn(5, 2, layer.config.hidden_size, device="cuda")
    mask = torch.ones(1, 1, 5, 5, device="cuda", dtype=torch.bool).triu(1)
    torch.testing.assert_close(layer(x, mask)[0], layer(x, None)[0])
    mask[..., 3, 0] = True
    with pytest.raises(NotImplementedError, match="ordinary causal"):
        layer(x, mask)
    with pytest.raises(NotImplementedError, match="requires V4.1 THD"):
        layer(x, None, packed_seq_params=SimpleNamespace(qkv_format="bshd"))


def _layers(pg_collection, dtype=torch.float32, candidates=True):
    overrides = (
        {}
        if candidates
        else dict(
            csa2_candidate_source_layer=None,
            csa2_candidate_topk_blocks=0,
            csa2_candidate_block_size=0,
        )
    )
    config = _make_config(params_dtype=dtype, dsa_indexer_topk=2, **overrides)
    spec = _attention_spec(dtype)
    layers = torch.nn.ModuleList(
        build_module(spec, config=config, layer_number=i + 1, pg_collection=pg_collection)
        for i in range(config.num_layers)
    ).cuda()
    with torch.no_grad():
        for name, parameter in layers.named_parameters():
            if "norm.weight" in name:
                parameter.uniform_(0.7, 1.3)
            else:
                parameter.normal_(std=0.2)
    return layers


def _reference_candidates(scores, ratio, blocks_to_keep, block_size):
    """Rank explicit slices, including the newest partly visible block."""
    width = scores.shape[-1]
    if width == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    blocks = [
        scores[..., start : start + block_size].amax(-1) for start in range(0, width, block_size)
    ]
    block_scores = torch.stack(blocks, -1)
    visible = torch.arange(1, scores.shape[1] + 1, device=scores.device) // ratio
    newest = (visible - 1) // block_size
    for block in range(len(blocks)):
        block_scores[..., block] = torch.where(newest == block, torch.inf, block_scores[..., block])
    order = block_scores.argsort(dim=-1, descending=True, stable=True)
    indices = order[..., : min(blocks_to_keep, len(blocks))]
    selected = torch.zeros_like(block_scores, dtype=torch.bool).scatter(
        -1, indices, block_scores.gather(-1, indices) > -torch.inf
    )
    return torch.stack([selected[..., position // block_size] for position in range(width)], -1)


def _reference_layer(x, weights, config, layer_idx, state):
    def linear(tensor, name):
        return F.linear(tensor, weights[name + ".weight"])

    def norm(tensor, name):
        return _reference_norm(
            tensor, weights[name + ".weight"], config.attention_latent_norm_epsilon
        )

    seq_len, batch = x.shape[:2]
    ratio = config.csa_compress_ratios[layer_idx]
    qr = norm(linear(x, "linear_q_down_proj"), "q_layernorm")
    q = linear(qr, "linear_q_up_proj").reshape(
        seq_len, batch, config.num_attention_heads, config.v_head_dim
    )
    q = _reference_rope(q, config, ratio)
    kv = _reference_rope(norm(linear(x, "linear_kv_proj"), "kv_layernorm"), config, ratio)
    query_positions = torch.arange(seq_len, device=x.device)[:, None]
    local_positions = torch.arange(seq_len, device=x.device)[None, :]
    visible = (local_positions <= query_positions) & (
        local_positions > query_positions - config.csa_window_size
    )
    visible = visible.expand(batch, -1, -1)
    if ratio:
        if layer_idx in config.csa2_kv_source_layers:
            prefix = "core_attention.compressor."
            if ratio == 1:
                latent = linear(x, prefix + "linear_wkv")
            else:
                groups = []
                for start in range(0, seq_len - ratio + 1, ratio):
                    tokens = x[start : start + ratio].float()
                    projected = linear(tokens, prefix + "linear_wkv")
                    gate = linear(tokens, prefix + "linear_wgate").softmax(0)
                    groups.append((projected * gate).sum(0))
                latent = (
                    torch.stack(groups)
                    if groups
                    else (
                        linear(x[:0].float(), prefix + "linear_wkv")
                        + linear(x[:0].float(), prefix + "linear_wgate")
                    )
                ).to(x.dtype)
            latent = norm(latent, prefix + "norm")
            state["global_kv"] = _reference_rope(latent, config, ratio, position_stride=ratio)
            prefix = "core_attention.indexer."
            index_k = norm(linear(latent, prefix + "linear_wk"), prefix + "k_norm")
            state["indexer_k"] = _reference_rope(index_k, config, ratio, position_stride=ratio)
        global_len = state["global_kv"].shape[0]
        causal = torch.arange(global_len, device=x.device) < (query_positions + 1) // ratio
        if layer_idx in config.csa2_index_source_layers:
            prefix = "core_attention.indexer."
            q_index = linear(qr, prefix + "linear_wq_b").reshape(
                seq_len, batch, config.dsa_indexer_n_heads, config.dsa_indexer_head_dim
            )
            q_index = _reference_rope(q_index, config, ratio)
            scores = torch.einsum(
                "sbhd,tbd->bsht", q_index.float(), state["indexer_k"].float()
            ).relu()
            weights_index = linear(x, prefix + "linear_weights_proj").float()
            weights_index = weights_index * (
                config.dsa_indexer_head_dim**-0.5 * config.dsa_indexer_n_heads**-0.5
            )
            scores = (scores * weights_index.permute(1, 0, 2).unsqueeze(-1)).sum(2)
            scores = scores.masked_fill(~causal, -torch.inf)
            candidate_owner = config.csa2_candidate_source_layer
            if candidate_owner == layer_idx:
                state["candidates"] = _reference_candidates(
                    scores,
                    ratio,
                    config.csa2_candidate_topk_blocks,
                    config.csa2_candidate_block_size,
                )
            elif candidate_owner is not None and candidate_owner < layer_idx:
                scores = scores.masked_fill(~state["candidates"], -torch.inf)
            state["scores"] = scores
            indices = scores.argsort(dim=-1, descending=True, stable=True)[
                ..., : min(config.dsa_indexer_topk, global_len)
            ]
            state["selected"] = torch.zeros_like(scores, dtype=torch.bool).scatter(
                -1, indices, scores.gather(-1, indices).isfinite()
            )
        visible = torch.cat((visible, state["selected"] & causal), -1)
        kv = torch.cat((kv, state["global_kv"]), 0)
    kv_float = kv.float()
    scores = torch.einsum("sbhd,tbd->bsht", q.float(), kv_float) * config.v_head_dim**-0.5
    scores = scores.masked_fill(~visible.unsqueeze(2), -torch.inf)
    sink = weights["core_attention.attn_sink"].reshape(1, 1, -1, 1).expand(batch, seq_len, -1, 1)
    probabilities = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
    output = torch.einsum("bsht,tbd->sbhd", probabilities, kv_float).to(x.dtype)
    output = _reference_rope(output, config, ratio, inverse=True).reshape(
        seq_len, batch, config.output_projection_groups, -1
    )
    group_weights = weights["linear_o_group_proj"].reshape(
        config.output_projection_groups, config.output_projection_lora_rank, -1
    )
    output = torch.stack(
        [
            F.linear(output[:, :, group], group_weights[group])
            for group in range(config.output_projection_groups)
        ],
        2,
    ).flatten(-2)
    return linear(output, "linear_proj")


def _stack(layers, x, state):
    # A residual attention stack isolates sharing from the still-separate mHC work.
    outputs = []
    for layer in layers:
        output, bias = layer(x, None, csa2_state=state)
        assert bias is None
        outputs.append(output)
        x = x + output * 0.2
    return x, outputs


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [1, 9])
@pytest.mark.parametrize("candidates", [False, True])
def test_shared_stack_matches_dense_reference(pg_collection, dtype, seq_len, candidates):
    torch.manual_seed(1234)
    layers = _layers(pg_collection, dtype, candidates)
    x = torch.randn(
        seq_len, 2, layers[0].config.hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = [
        {name: p.detach().clone().requires_grad_() for name, p in layer.named_parameters()}
        for layer in layers
    ]
    actual, outputs = _stack(layers, x, CSA2State())
    expected, ref_state = ref_x, {}
    tol = dict(atol=5e-6, rtol=5e-5) if dtype == torch.float32 else dict(atol=3e-2, rtol=6e-2)
    for layer_idx, (layer, weights) in enumerate(zip(layers, ref_weights)):
        ref_output = _reference_layer(expected, weights, layer.config, layer_idx, ref_state)
        torch.testing.assert_close(outputs[layer_idx], ref_output, **tol)
        expected = expected + ref_output * 0.2
    torch.testing.assert_close(actual, expected, **tol)
    gradient = torch.randn_like(actual)
    (actual * gradient).sum().backward()
    (expected * gradient).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, **tol)
    for layer_idx, (layer, weights) in enumerate(zip(layers, ref_weights)):
        for name, parameter in layer.named_parameters():
            if "indexer." in name:
                assert parameter.grad is None and weights[name].grad is None
            else:
                assert parameter.grad is not None, (layer_idx, name)
                torch.testing.assert_close(
                    parameter.grad,
                    weights[name].grad,
                    msg=lambda msg: f"{layer_idx}.{name}: {msg}",
                    **tol,
                )


def test_only_full_owns_kv_and_reindex_owns_only_query(pg_collection):
    layers = _layers(pg_collection)
    for layer_idx, layer in enumerate(layers):
        core = layer.core_attention
        assert (core.compressor is not None) == (layer_idx in [1, 3])
        assert (core.indexer is not None) == (layer_idx in [1, 3, 4])
        if core.indexer is not None:
            names = dict(core.indexer.named_parameters())
            assert ("linear_wk.weight" in names) == (layer_idx in [1, 3])
            assert ("k_norm.weight" in names) == (layer_idx in [1, 3])
            assert "linear_wq_b.weight" in names and "linear_weights_proj.weight" in names
    x = torch.randn(9, 2, layers[0].config.hidden_size, device="cuda", requires_grad=True)
    state = CSA2State()
    layers[1](x, None, csa2_state=state)
    owner_kv, owner_index_k, owner_indices = state.global_kv, state.indexer_k, state.global_indices
    layers[2](x, None, csa2_state=state)
    assert state.global_kv is owner_kv and state.indexer_k is owner_index_k
    assert state.global_indices is owner_indices
    layers[3](x, None, csa2_state=state)
    decoder_kv, decoder_index_k = state.global_kv, state.indexer_k
    assert decoder_kv is not owner_kv and decoder_index_k is not owner_index_k
    assert decoder_kv.grad_fn is not None and decoder_index_k.grad_fn is not None
    layers[4](x, None, csa2_state=state)
    assert state.global_kv is decoder_kv and state.indexer_k is decoder_index_k
    assert state.kv_source_layer == 3 and state.index_source_layer == 4


def test_reuse_consumer_gradient_reaches_full_owner(pg_collection):
    torch.manual_seed(47)
    layers = _layers(pg_collection, candidates=False)
    source = torch.randn(9, 2, layers[0].config.hidden_size, device="cuda", requires_grad=True)
    consumer = torch.randn_like(source, requires_grad=True)
    state = CSA2State()
    # Deliberately discard Full's output; the only path to source is the shared global KV.
    layers[1](source, None, csa2_state=state)
    reused, _ = layers[2](consumer, None, csa2_state=state)
    reused.square().sum().backward()
    assert source.grad is not None and source.grad.abs().sum() > 0
    assert consumer.grad is not None and consumer.grad.abs().sum() > 0
    owner = layers[1].core_attention.compressor
    for parameter in owner.parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert layers[1].linear_q_up_proj.weight.grad is None
    # Position 8 belongs to an incomplete ratio-2 group and has no source path here.
    assert torch.count_nonzero(source.grad[-1]) == 0


@torch.no_grad()
def test_shared_state_rejects_stale_order_missing_owners_and_shape_changes(pg_collection):
    layers = _layers(pg_collection)
    x = torch.randn(9, 2, layers[0].config.hidden_size, device="cuda")
    state = CSA2State()
    layers[1](x, None, csa2_state=state)
    with pytest.raises(ValueError, match="increasing order.*fresh CSA2State"):
        layers[1](x, None, csa2_state=state)
    # Reindex cannot run before its Full source, even when another Full populated state.
    with pytest.raises(ValueError, match="KV and indexer K from Full layer 3"):
        layers[4](x, None, csa2_state=CSA2State())
    with pytest.raises(ValueError, match="KV and indexer K from Full layer 3"):
        layers[4](x, None, csa2_state=state)
    layers[3](x, None, csa2_state=state)
    # The decoder Reuse layer needs the latest Reindex selection, not Full's selection.
    with pytest.raises(ValueError, match="requires indices from layer 4"):
        layers[5](x, None, csa2_state=state)
    with pytest.raises(ValueError, match="sequence length, batch size, device and dtype"):
        layers[4](x[:-1], None, csa2_state=state)


def test_two_live_forwards_preserve_causality_and_independent_graphs(pg_collection):
    torch.manual_seed(17)
    layers = _layers(pg_collection)
    x = torch.randn(9, 2, layers[0].config.hidden_size, device="cuda", requires_grad=True)
    changed = x.detach().clone()
    changed[5:] = torch.randn_like(changed[5:]) * 10
    changed.requires_grad_()
    state, changed_state = CSA2State(), CSA2State()
    first, _ = _stack(layers, x, state)
    second, _ = _stack(layers, changed, changed_state)
    assert state.global_kv is not changed_state.global_kv
    assert state.global_indices is not changed_state.global_indices
    torch.testing.assert_close(first[:5], second[:5], atol=5e-6, rtol=5e-5)
    (first[:5].sum() + second[:5].sum()).backward()
    assert torch.count_nonzero(x.grad[5:]) == 0
    assert torch.count_nonzero(changed.grad[5:]) == 0
    # TE linears release backward context after one traversal. Compare with fresh, isolated
    # forwards so every graph is differentiated once, including the two live graphs above.
    for original in (x, changed):
        isolated = original.detach().clone().requires_grad_()
        output, _ = _stack(layers, isolated, CSA2State())
        expected_gradient = torch.autograd.grad(output[:5].sum(), isolated)[0]
        torch.testing.assert_close(original.grad, expected_gradient)


def test_reindex_scores_preserve_owner_key_graph_and_candidate_mask(pg_collection):
    torch.manual_seed(24)
    layers = _layers(pg_collection)
    source = torch.randn(9, 2, layers[0].config.hidden_size, device="cuda", requires_grad=True)
    consumer = torch.randn_like(source, requires_grad=True)
    ref_source = source.detach().clone().requires_grad_()
    ref_consumer = consumer.detach().clone().requires_grad_()
    owner, reindex = layers[3], layers[4]
    ref_owner_weights = {
        name: p.detach().clone().requires_grad_() for name, p in owner.named_parameters()
    }
    ref_reindex_weights = {
        name: p.detach().clone().requires_grad_() for name, p in reindex.named_parameters()
    }
    state, ref_state = CSA2State(), {}
    owner(source, None, csa2_state=state)
    _reference_layer(ref_source, ref_owner_weights, owner.config, 3, ref_state)
    _reference_layer(ref_consumer, ref_reindex_weights, reindex.config, 4, ref_state)
    qr = reindex.q_layernorm(reindex.linear_q_down_proj(consumer)[0])
    indexer = reindex.core_attention.indexer
    scores = indexer.forward_before_topk(
        consumer,
        qr,
        None,
        reindex.rotary_pos_emb,
        indexer_k=state.indexer_k,
        candidates=state.candidates,
    )
    torch.testing.assert_close(scores, ref_state["scores"], atol=3e-6, rtol=3e-5)
    candidate_mask = state.candidates.to_mask(scores.shape[-1])
    assert (~candidate_mask[:, -1]).any(), "The case must exclude old global positions."
    assert torch.isneginf(scores.masked_select(~candidate_mask)).all()
    selected = indexer.select_indices(scores)
    valid = selected >= 0
    assert candidate_mask.gather(-1, selected.clamp_min(0).long())[valid].all()
    finite = scores.isfinite()
    scores[finite].square().mean().backward()
    ref_state["scores"][finite].square().mean().backward()
    torch.testing.assert_close(source.grad, ref_source.grad, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(consumer.grad, ref_consumer.grad, atol=3e-6, rtol=3e-5)
    for layer, ref_weights in [(owner, ref_owner_weights), (reindex, ref_reindex_weights)]:
        for name, parameter in layer.named_parameters():
            expected_gradient = ref_weights[name].grad
            if expected_gradient is None:
                assert parameter.grad is None, name
            else:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                torch.testing.assert_close(parameter.grad, expected_gradient, atol=3e-6, rtol=3e-5)
    assert owner.core_attention.indexer.linear_wk.weight.grad.abs().sum() > 0
    assert reindex.core_attention.indexer.linear_wq_b.weight.grad.abs().sum() > 0


def test_select_indices_stable_selection_survives_chunking(monkeypatch):
    """The chunked selection keeps the stable-argsort contract: lowest index wins
    ties regardless of the packed width, -1 lands exactly on slots whose own score
    is masked (interior intervals, rows with fewer visible positions than topk),
    rows keep their alignment across chunks, and the per-chunk sort calls actually
    happen at the budgeted size."""
    import megatron.core.transformer.experimental_attention_variant.csa2 as csa2_module

    topk = 4
    selector = SimpleNamespace(topk=topk)
    rows, keys, batch = 7, 24, 2
    scores = torch.randn(batch, rows, keys, device="cuda", dtype=torch.float32)
    scores[:, 0, -5:] = -torch.inf  # partial visibility: fewer winners than topk
    scores[:, 1, :] = -torch.inf  # fully masked row
    scores[:, 2, 1] = -torch.inf  # hole between valid positions
    scores[0, 3, :5] = -torch.inf  # interior interval: only {5, 6} visible
    scores[0, 3, 5:7] = 0.5
    scores[0, 3, 7:] = -torch.inf
    selected = CSA2Indexer.select_indices(selector, scores)
    assert selected.dtype == torch.int32
    assert selected.shape[-1] == topk
    valid = selected >= 0
    assert torch.equal(valid.sum(-1), scores.isfinite().sum(-1).clamp_max(topk))
    assert torch.equal(selected[0, 3][valid[0, 3]], torch.tensor([5, 6], device="cuda"))
    # A zero-key matrix (no complete compression group) keeps the empty-selection
    # contract instead of crashing the chunked path.
    empty = CSA2Indexer.select_indices(selector, torch.zeros(1, 3, 0, device="cuda"))
    assert empty.shape == (1, 3, 0) and empty.dtype == torch.int32
    # The stable tie contract: all-tied visible positions keep the lowest indices.
    tied = torch.zeros(1, 3, 31, device="cuda", dtype=torch.float32)
    tied[..., 0, :] = -torch.inf
    tied[..., 1, 7:] = -torch.inf  # 7 visible positions, all tied at zero
    tied_selected = CSA2Indexer.select_indices(selector, tied)
    torch.testing.assert_close(
        tied_selected[0, 1], torch.arange(topk, device="cuda", dtype=torch.int32)
    )
    # topk larger than the key count clamps to the key count.
    narrow = torch.randn(1, 4, 2, device="cuda", dtype=torch.float32)
    narrow[..., 0, 1] = -torch.inf
    narrow_selected = CSA2Indexer.select_indices(selector, narrow)
    assert narrow_selected.shape[-1] == 2
    # The chunked sort must both chunk (more than one call, no call above the
    # budgeted rows) and stay bit-identical to the single-shot stable argsort.
    monkeypatch.setattr(csa2_module, "_SELECT_ROW_CHUNK_ELEMENTS", 3 * keys)
    sort_rows = []
    real_argsort = torch.Tensor.argsort

    def spy(self, *args, **kwargs):
        sort_rows.append(self.shape[0])
        return real_argsort(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "argsort", spy)
    chunked = CSA2Indexer.select_indices(selector, scores)
    assert len(sort_rows) > 1 and max(sort_rows) <= 3
    single = real_argsort(scores, dim=-1, descending=True, stable=True)[..., :topk]
    single = single.sort(dim=-1).values
    single = single.masked_fill(~scores.gather(-1, single).isfinite(), -1).int()
    torch.testing.assert_close(chunked, single)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Candidate tensors use CUDA")
def test_thd_candidate_mask_matches_sequence_local_block_lookup(monkeypatch):
    blocks = CSA2CandidateBlocks(
        torch.tensor([[[0, 1, -1], [0, -1, -1], [1, 9, -1], [1, -1, -1]]]), block_size=2
    )
    query_sequences = torch.tensor([0, 1, 1, 0])
    key_sequences = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    key_positions = torch.tensor([0, 1, 2, 0, 1, 2, 3])
    expected = (blocks.indices[..., None] == (key_positions // 2)[None, None, None, :]).any(-2) & (
        query_sequences[None, :, None] == key_sequences[None, None, :]
    )
    torch.testing.assert_close(
        blocks.to_mask_thd(query_sequences, key_sequences, key_positions), expected
    )
    # A two-row budget forces several row chunks: the helper chunks the row
    # dimension (not the batch dimension, which is 1 here) and folds the
    # sequence-membership test into each slab, so the mask must be unchanged.
    monkeypatch.setattr(
        csa2_candidates_module, "_MASK_ROW_CHUNK_ELEMENTS", 2 * key_positions.numel()
    )
    chunked = blocks.to_mask_thd(query_sequences, key_sequences, key_positions)
    assert chunked.shape == expected.shape
    torch.testing.assert_close(chunked, expected)


def test_candidate_blocks_exclude_old_blocks_and_force_newest():
    # Latest position has the lowest score; pinning must displace the second-best old block.
    scores = torch.tensor([[[9.0, 7.0, 8.0, 6.0, 5.0, 4.0, -100.0]]], device="cuda")
    selected = select_candidate_blocks(scores, torch.tensor([[7]], device="cuda"), 2, 2)
    expected = torch.tensor([[[True, True, False, False, False, False, True]]], device="cuda")
    torch.testing.assert_close(selected, expected)
    # No completed ratio-2 group means no candidate, despite having storage for later keys.
    empty = select_candidate_blocks(
        torch.full_like(scores, -torch.inf), torch.tensor([[0]], device="cuda"), 2, 2
    )
    assert not empty.any()
    partial = scores.masked_fill(torch.arange(7, device="cuda") >= 3, -torch.inf)
    selected = select_candidate_blocks(partial, torch.tensor([[3]], device="cuda"), 2, 2)
    expected = torch.tensor([[[True, True, True, True, False, False, False]]], device="cuda")
    torch.testing.assert_close(selected, expected)


def test_chunked_candidate_block_scores_match_full_gather(monkeypatch):
    """The chunked sequence-local reduction is bit-identical to the full gather.

    Covers ragged block tails, sequence boundaries inside shards, multi-chunk
    splits via a tiny element budget, and the all-masked (-inf) path that the
    candidate-source layer takes when a row's sequence has no visible keys.
    """
    import dataclasses

    import megatron.core.transformer.experimental_attention_variant.csa2 as csa2_module

    @dataclasses.dataclass
    class _Compressed:
        cu_seqlens_padded: torch.Tensor
        max_seqlen: int

    torch.manual_seed(93)
    rows, keys, nseq, block_size = 50, 300, 3, 8
    # Padded per-sequence capacities are not block multiples, so the -inf
    # block tail and the OOB mask vary per sequence.
    capacities = torch.tensor([104, 136, 56], device="cuda", dtype=torch.int64)
    cu = torch.zeros(nseq + 1, device="cuda", dtype=torch.int64)
    cu[1:] = capacities.cumsum(0)
    max_seqlen = int(capacities.max())
    sequence_ids = torch.randint(0, nseq, (rows,), device="cuda")
    scores = torch.randn(rows, keys, device="cuda", dtype=torch.float32)
    # One row's sequence entirely OOB so its blocks come out all -inf.
    scores[0] = -torch.inf

    local_keys = torch.arange(max_seqlen, device="cuda")
    physical = cu[sequence_ids, None] + local_keys
    full_local = scores.gather(1, physical.clamp_max(keys - 1).long())
    full_local = full_local.masked_fill(physical >= cu[sequence_ids + 1, None], -torch.inf)
    padded = F.pad(full_local, (0, -max_seqlen % block_size), value=-torch.inf)
    num_blocks = padded.shape[-1] // block_size
    expected = padded.unflatten(-1, (num_blocks, block_size)).amax(-1)

    # Force several chunks (and chunk-boundary rows) with a tiny budget.
    monkeypatch.setattr(csa2_module, "_SELECT_ROW_CHUNK_ELEMENTS", 3 * max_seqlen)
    chunked = csa2_module._chunked_candidate_block_scores(
        scores, sequence_ids, _Compressed(cu, max_seqlen), block_size
    )
    assert chunked.shape == expected.shape
    torch.testing.assert_close(chunked, expected, rtol=0, atol=0)

    # The reduced form feeds the same ranking as the full-matrix path.
    visible = torch.randint(1, max_seqlen + 1, (1, rows, 1), device="cuda")
    from_blocks = candidate_blocks_from_block_scores(chunked.unsqueeze(0), visible, 2, block_size)
    from_full = candidate_blocks_from_scores(full_local.unsqueeze(0), visible, 2, block_size)
    torch.testing.assert_close(from_blocks.indices, from_full.indices)


def test_streaming_topk_matches_dense_thd(monkeypatch):
    """The fused THD select is bit-identical to the dense path.

    Pins _streaming_indexer_topk against the dense composition it replaces
    (chunked einsum -> the forward_before_topk causal chain -> the select
    contract; and the candidate-source fold against
    _chunked_candidate_block_scores, the already-pinned oracle) across
    ragged multi-sequence packs, exact-zero ReLU tie storms, the candidate
    block mask, -1 sentinel rows, and multi-chunk splits via a tiny budget.
    """
    import dataclasses

    import megatron.core.transformer.experimental_attention_variant.csa2 as csa2_module

    # A dedicated generator instead of torch.manual_seed(): seeding here would reset
    # the process-wide CUDA stream that earlier tests in this file already advanced,
    # so the draws below would depend on test order and would perturb whatever runs
    # next. A fresh CUDA generator seeded 101 reproduces the same draws and leaves
    # global RNG state untouched.
    gen = torch.Generator(device="cuda")
    gen.manual_seed(101)
    ratio, topk, block_size = 2, 8, 8

    @dataclasses.dataclass
    class _Thd:
        valid_tokens: torch.Tensor
        sequence_ids: torch.Tensor
        position_ids: torch.Tensor

    @dataclasses.dataclass
    class _Compressed:
        cu_seqlens_padded: torch.Tensor
        max_seqlen: int
        valid_groups: torch.Tensor
        sequence_ids: torch.Tensor
        position_ids: torch.Tensor

    # Three ragged sequences; per-sequence compressed capacity aligned to the block size.
    row_counts = [96, 40, 52]
    capacities = [48, 24, 32]  # compressed groups per sequence (ceil(rows/ratio) rounded)
    s = sum(row_counts)
    rows_seq = torch.cat(
        [torch.full((n,), i, dtype=torch.int64, device="cuda") for i, n in enumerate(row_counts)]
    )
    rows_pos = torch.cat([torch.arange(n, dtype=torch.int64, device="cuda") for n in row_counts])
    rows_valid = torch.ones(s, dtype=torch.bool, device="cuda")
    rows_valid[-5:] = False  # padding tail: all-masked -> -1 sentinel rows
    cu = torch.zeros(4, dtype=torch.int64, device="cuda")
    cu[1:] = torch.tensor(capacities, device="cuda").cumsum(0)
    keys = int(cu[-1])
    key_seq = torch.cat(
        [torch.full((c,), i, dtype=torch.int64, device="cuda") for i, c in enumerate(capacities)]
    )
    key_pos = torch.cat(
        [torch.arange(c, dtype=torch.int64, device="cuda") * ratio for c in capacities]
    )
    # The last few groups of sequence 2 are padding (invalid).
    valid_groups = torch.ones(keys, dtype=torch.bool, device="cuda")
    valid_groups[-3:] = False
    thd = _Thd(rows_valid, rows_seq, rows_pos)
    comp = _Compressed(cu, max(capacities), valid_groups, key_seq, key_pos)

    h, d = 4, 32
    q = torch.randn(s, 1, h, d, device="cuda", generator=gen)
    indexer_k = torch.randn(keys, 1, d, device="cuda", generator=gen)
    indexer_k[: keys // 2].neg_()  # ReLU emits exact zeros: the tie-storm regime
    weights = torch.randn(s, 1, h, device="cuda", generator=gen)

    # Dense reference: the forward_before_topk no-grad THD math, verbatim.
    scores = csa2_module._chunked_csa2_indexer_scores(q, indexer_k, weights)
    causal = (
        thd.valid_tokens[:, None]
        & comp.valid_groups[None, :]
        & (thd.sequence_ids[:, None] == comp.sequence_ids[None, :])
        & (comp.position_ids[None, :] + ratio <= thd.position_ids[:, None] + 1)
    )
    scores = scores.masked_fill(~causal, -torch.inf)
    k = min(topk, scores.shape[-1])
    winners = scores.reshape(-1, scores.shape[-1]).argsort(dim=-1, descending=True, stable=True)
    ref_indices = winners[..., :k].sort(dim=-1).values.reshape(1, s, k)
    ref_indices = ref_indices.masked_fill(~scores.gather(-1, ref_indices).isfinite(), -1).int()
    ref_blocks = csa2_module._chunked_candidate_block_scores(
        scores.squeeze(0), thd.sequence_ids.clamp_min(0), comp, block_size
    )

    # Candidate mask case: random per-row candidate blocks.
    ids = torch.randint(0, 4, (1, s, 3), dtype=torch.int32, device="cuda", generator=gen)
    ids[0, -5:] = -1  # padding rows have no candidates
    cand = CSA2CandidateBlocks(ids, block_size)
    cand_mask = cand.to_mask_thd(thd.sequence_ids, comp.sequence_ids, comp.position_ids // ratio)
    scores_c = scores.masked_fill(~cand_mask, -torch.inf)
    winners_c = scores_c.reshape(-1, scores_c.shape[-1]).argsort(
        dim=-1, descending=True, stable=True
    )
    ref_c = winners_c[..., :k].sort(dim=-1).values.reshape(1, s, k)
    ref_c = ref_c.masked_fill(~scores_c.gather(-1, ref_c).isfinite(), -1).int()

    # Fused: multi-chunk via a tiny budget. The merge stage is driven by
    # _SELECT_MERGE_STAGE_KEYS, not by the budget, so shrink it too: this
    # fixture holds 104 keys (causal limit 101), where 64 is the only
    # multiple of the 64-key einsum tiling that leaves more than one merge.
    monkeypatch.setattr(csa2_module, "_SELECT_ROW_CHUNK_ELEMENTS", 256)
    monkeypatch.setattr(csa2_module, "_SELECT_MERGE_STAGE_KEYS", 64)
    got_indices, got_blocks = csa2_module._streaming_indexer_topk(
        q, indexer_k, weights, topk, ratio, thd, comp, None, True, block_size
    )
    got_c, _ = csa2_module._streaming_indexer_topk(
        q, indexer_k, weights, topk, ratio, thd, comp, cand, False, block_size
    )

    torch.testing.assert_close(got_indices, ref_indices, rtol=0, atol=0)
    torch.testing.assert_close(got_blocks, ref_blocks, rtol=0, atol=0)
    torch.testing.assert_close(got_c, ref_c, rtol=0, atol=0)
    assert (got_indices[0, -5:] == -1).all()  # sentinel rows survive the streaming merge

    # ---- Key-skip dimensions (same bit-identity contract) ----
    # (1) Causal skip: a single-sequence layout with trailing padding groups
    #     must produce identical outputs while skipping the dead tail keys.
    solo_rows = 96
    solo_capacity = 48
    solo_thd = _Thd(
        torch.ones(solo_rows, dtype=torch.bool, device="cuda"),
        torch.zeros(solo_rows, dtype=torch.int64, device="cuda"),
        torch.arange(solo_rows, dtype=torch.int64, device="cuda"),
    )
    solo_cu = torch.tensor([0, solo_capacity], dtype=torch.int64, device="cuda")
    solo_key_seq = torch.zeros(solo_capacity, dtype=torch.int64, device="cuda")
    solo_key_pos = torch.arange(solo_capacity, dtype=torch.int64, device="cuda") * ratio
    # trailing invalid groups: the last 5 groups are padding
    solo_valid_groups = torch.ones(solo_capacity, dtype=torch.bool, device="cuda")
    solo_valid_groups[-5:] = False
    solo_comp = _Compressed(solo_cu, solo_capacity, solo_valid_groups, solo_key_seq, solo_key_pos)
    solo_scores = csa2_module._chunked_csa2_indexer_scores(
        q[:solo_rows], indexer_k[:solo_capacity], weights[:solo_rows]
    )
    solo_causal = (
        solo_thd.valid_tokens[:, None]
        & solo_comp.valid_groups[None, :]
        & (solo_thd.sequence_ids[:, None] == solo_comp.sequence_ids[None, :])
        & (solo_comp.position_ids[None, :] + ratio <= solo_thd.position_ids[:, None] + 1)
    )
    solo_scores = solo_scores.masked_fill(~solo_causal, -torch.inf)
    solo_winners = (
        solo_scores.reshape(-1, solo_scores.shape[-1])
        .argsort(dim=-1, descending=True, stable=True)[..., :k]
        .sort(dim=-1)
        .values.reshape(1, solo_rows, k)
    )
    solo_ref = solo_winners.masked_fill(~solo_scores.gather(-1, solo_winners).isfinite(), -1).int()
    solo_got, _ = csa2_module._streaming_indexer_topk(
        q[:solo_rows],
        indexer_k[:solo_capacity],
        weights[:solo_rows],
        topk,
        ratio,
        solo_thd,
        solo_comp,
        None,
        False,
        block_size,
    )
    torch.testing.assert_close(solo_got, solo_ref, rtol=0, atol=0)

    # (2) Candidate skip: keys outside every row's candidate tiles are dead;
    #     the skip form must match the already-verified candidate select.
    # Give rows disjoint candidate blocks so live tiles are sparse and
    # interleaved (some 64-key tiles fully dead, some live).
    sparse_ids = torch.randint(0, 4, (1, s, 2), dtype=torch.int32, device="cuda", generator=gen)
    sparse_ids[0, -5:] = -1
    sparse_cand = CSA2CandidateBlocks(sparse_ids, block_size)
    dense_skip_got, _ = csa2_module._streaming_indexer_topk(
        q, indexer_k, weights, topk, ratio, thd, comp, sparse_cand, False, block_size
    )
    # dense-path reference for the sparse candidate set
    sparse_mask = sparse_cand.to_mask_thd(
        thd.sequence_ids, comp.sequence_ids, comp.position_ids // ratio
    )
    sparse_scores = scores.masked_fill(~sparse_mask, -torch.inf)
    sparse_winners = (
        sparse_scores.reshape(-1, sparse_scores.shape[-1])
        .argsort(dim=-1, descending=True, stable=True)[..., :k]
        .sort(dim=-1)
        .values.reshape(1, s, k)
    )
    sparse_ref = sparse_winners.masked_fill(
        ~sparse_scores.gather(-1, sparse_winners).isfinite(), -1
    ).int()
    torch.testing.assert_close(dense_skip_got, sparse_ref, rtol=0, atol=0)

    # (3) Dead trailing tile: the live-tile run-length walk must iterate the
    #     LIVE count, not the tile count. Only sequence 0 (keys 0..47) keeps a
    #     candidate block here, so the second 64-key tile (keys 64..103) has no
    #     candidate key for any row -- exactly the shape the causal skip
    #     produces in production, and the one that indexes tile_ids past its
    #     end if the walk is bounded by the tile count.
    dead_tail_ids = torch.full((1, s, 2), -1, dtype=torch.int32, device="cuda")
    dead_tail_ids[0, : row_counts[0], 0] = 0  # only block 0 -> keys 0..7, seq 0
    dead_tail_cand = CSA2CandidateBlocks(dead_tail_ids, block_size)
    dead_tail_got, _ = csa2_module._streaming_indexer_topk(
        q, indexer_k, weights, topk, ratio, thd, comp, dead_tail_cand, False, block_size
    )
    dead_tail_mask = dead_tail_cand.to_mask_thd(
        thd.sequence_ids, comp.sequence_ids, comp.position_ids // ratio
    )
    dead_tail_scores = scores.masked_fill(~dead_tail_mask, -torch.inf)
    dead_tail_winners = (
        dead_tail_scores.reshape(-1, dead_tail_scores.shape[-1])
        .argsort(dim=-1, descending=True, stable=True)[..., :k]
        .sort(dim=-1)
        .values.reshape(1, s, k)
    )
    dead_tail_ref = dead_tail_winners.masked_fill(
        ~dead_tail_scores.gather(-1, dead_tail_winners).isfinite(), -1
    ).int()
    torch.testing.assert_close(dead_tail_got, dead_tail_ref, rtol=0, atol=0)

    # (4) Every tile dead: an all -1 candidate set leaves no live 64-key tile,
    #     so the live-tile walk yields no bounds and no stage ever scores a
    #     key. The select must still materialize the empty selection (all -1,
    #     shape [b, s, k]) instead of raising on an unmaterialized run.
    none_ids = torch.full((1, s, 2), -1, dtype=torch.int32, device="cuda")
    none_cand = CSA2CandidateBlocks(none_ids, block_size)
    none_got, _ = csa2_module._streaming_indexer_topk(
        q, indexer_k, weights, topk, ratio, thd, comp, none_cand, False, block_size
    )
    assert none_got.shape == (1, s, k) and none_got.dtype == torch.int32
    assert (none_got == -1).all()


def _dense_thd_select(q, indexer_k, weights, thd, comp, ratio, topk, candidates=None):
    """Dense oracle: chunked scores -> the forward_before_topk causal chain -> select_indices.

    Returns ``(indices, causal_scores)``; the scores are the pre-candidate slab the
    candidate-source block statistic is reduced from.
    """
    import megatron.core.transformer.experimental_attention_variant.csa2 as csa2_module

    scores = csa2_module._chunked_csa2_indexer_scores(q, indexer_k, weights)
    causal = (
        thd.valid_tokens[:, None]
        & comp.valid_groups[None, :]
        & (thd.sequence_ids[:, None] == comp.sequence_ids[None, :])
        & (comp.position_ids[None, :] + ratio <= thd.position_ids[:, None] + 1)
    )
    scores = scores.masked_fill(~causal, -torch.inf)
    selected = scores
    if candidates is not None:
        selected = scores.masked_fill(
            ~candidates.to_mask_thd(
                thd.sequence_ids, comp.sequence_ids, comp.position_ids // ratio
            ),
            -torch.inf,
        )
    return CSA2Indexer.select_indices(SimpleNamespace(topk=topk), selected), scores


@pytest.mark.parametrize(
    ("cu_seqlens", "max_seqlen", "total_tokens", "cp_size"),
    [
        # Static-shape cumulative lengths: zero-length trailing entries and a padding tail.
        ([0, 100, 150, 150, 150, 150, 150, 150, 150], 100, 512, 1),
        # CP rank 0 of 2 holds only sequence 0; the trailing 1-token sequence owns no group.
        ([0, 128, 129], 128, 128, 2),
    ],
    ids=["zero-length-tail", "cp-groupless-trailing-sequence"],
)
def test_streaming_topk_keeps_unassigned_tail_keys_in_the_block_table(
    cu_seqlens, max_seqlen, total_tokens, cp_size
):
    """Capacity-tail keys stay inside the K=1 sequence-local block table.

    The table has one slot per sequence that owns rows or groups. The keys past the
    last group belong to no sequence; addressing them by their absolute index ran past
    the table whenever the trailing sequences owned no compressed group.
    """
    import megatron.core.transformer.experimental_attention_variant.csa2 as csa2_module

    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    ratio, topk, block_size = 2, 8, 64
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu.clone(),
        cu_seqlens_q_padded=cu.clone(),
        cu_seqlens_kv_padded=cu.clone(),
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        cp_partition_mode="contiguous",
    )
    # The layout only reads the group's size and rank; rank 0's rows never reach the
    # trailing sequence.
    cp_group = SimpleNamespace(size=lambda: cp_size, rank=lambda: 0) if cp_size > 1 else None
    thd = build_csa2_thd_layout(params, total_tokens, cp_group=cp_group)
    comp = thd.for_compression(ratio)
    n_slots = max(int(thd.sequence_ids.max()), int(comp.sequence_ids.max())) + 1
    n_blocks = -(-comp.max_seqlen // block_size)
    tail = (comp.sequence_ids < 0).nonzero().flatten()
    assert int(tail.max()) // block_size >= n_slots * n_blocks, "tail must reach past the table"

    s, keys, h, d = total_tokens, comp.capacity, 2, 16
    q = torch.randn(s, 1, h, d, device="cuda", generator=gen)
    indexer_k = torch.randn(keys, 1, d, device="cuda", generator=gen)
    weights = torch.randn(s, 1, h, device="cuda", generator=gen)

    scores = csa2_module._chunked_csa2_indexer_scores(q, indexer_k, weights)
    causal = (
        thd.valid_tokens[:, None]
        & comp.valid_groups[None, :]
        & (thd.sequence_ids[:, None] == comp.sequence_ids[None, :])
        & (comp.position_ids[None, :] + ratio <= thd.position_ids[:, None] + 1)
    )
    scores = scores.masked_fill(~causal, -torch.inf)

    def dense_select(masked):
        k = min(topk, masked.shape[-1])
        order = masked.reshape(-1, masked.shape[-1]).argsort(dim=-1, descending=True, stable=True)
        picked = order[..., :k].sort(dim=-1).values.reshape(1, s, k)
        return picked.masked_fill(~masked.gather(-1, picked).isfinite(), -1).int()

    ids = torch.randint(0, n_blocks, (1, s, 2), dtype=torch.int32, device="cuda", generator=gen)
    cand = CSA2CandidateBlocks(ids, block_size)
    cand_mask = cand.to_mask_thd(thd.sequence_ids, comp.sequence_ids, comp.position_ids // ratio)

    got, got_blocks = csa2_module._streaming_indexer_topk(
        q, indexer_k, weights, topk, ratio, thd, comp, None, True, block_size
    )
    got_c, _ = csa2_module._streaming_indexer_topk(
        q, indexer_k, weights, topk, ratio, thd, comp, cand, False, block_size
    )

    torch.testing.assert_close(got, dense_select(scores), rtol=0, atol=0)
    ref_blocks = csa2_module._dense_candidate_block_scores(scores, thd, comp, block_size)
    torch.testing.assert_close(got_blocks, ref_blocks, rtol=0, atol=0)
    ref_c = dense_select(scores.masked_fill(~cand_mask, -torch.inf))
    torch.testing.assert_close(got_c, ref_c, rtol=0, atol=0)


@pytest.mark.parametrize("chunk_count", [1, 2], ids=["k1", "k2-streamed"])
@pytest.mark.parametrize("want_block_scores", [False, True], ids=["consumer", "source"])
def test_streaming_topk_host_syncs_do_not_grow_with_the_pack(
    monkeypatch, chunk_count, want_block_scores
):
    """The fused select reads the device a fixed number of times per call.

    Neither the live tiles nor the merge stages and their block buckets may each
    synchronise the host, and the per-layout values are read once per layout pair: a
    pack with four times the keys syncs exactly as often, at most three times per call.
    """
    import megatron.core.transformer.experimental_attention_variant.csa2 as csa2_module

    # Small stages and buckets, so even the small pack spans several of each.
    monkeypatch.setattr(csa2_module, "_SELECT_MERGE_STAGE_KEYS", 64)
    monkeypatch.setattr(csa2_module, "_CANDIDATE_BLOCK_CHUNK_SIZE", 4)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(11)
    block_size = 3

    def inputs(scale):
        capacities = [65 * scale, 47 * scale, 19 * scale]
        rows = 3 * len(capacities)
        keys = sum(capacities)
        valid_rows = torch.ones(rows, dtype=torch.bool)
        valid_rows[-1] = False
        valid_groups = torch.ones(keys, dtype=torch.bool)
        valid_groups[-2:] = False
        thd = SimpleNamespace(
            sequence_ids=torch.arange(len(capacities)).repeat_interleave(3).cuda(),
            position_ids=torch.tensor(
                [p for c in capacities for p in (0, 2 * c - 1, 2 * c - 3)]
            ).cuda(),
            valid_tokens=valid_rows.cuda(),
            chunk_count=chunk_count,
        )
        comp = SimpleNamespace(
            cu_seqlens_padded=torch.tensor(
                [0, *torch.tensor(capacities).cumsum(0).tolist()]
            ).cuda(),
            total_tokens=keys,
            max_seqlen=max(capacities),
            valid_groups=valid_groups.cuda(),
            sequence_ids=torch.cat([torch.full((c,), i) for i, c in enumerate(capacities)]).cuda(),
            position_ids=torch.cat([2 * torch.arange(c) for c in capacities]).cuda(),
        )
        q = torch.randn(rows, 1, 2, 4, device="cuda", generator=gen)
        indexer_k = torch.randn(keys, 1, 4, device="cuda", generator=gen)
        weights = torch.randn(rows, 1, 2, device="cuda", generator=gen)
        ids = torch.tensor([0, 2, 5, 9], dtype=torch.int32).expand(rows, 4).clone()
        ids[-1] = -1
        candidates = None if want_block_scores else CSA2CandidateBlocks(ids.cuda(), block_size)
        return (
            q,
            indexer_k,
            weights,
            8,
            2,
            thd,
            comp,
            candidates,
            want_block_scores,
            block_size,
            4,
        )

    # Count only the syncs Megatron code issues: PyTorch can add a one-time internal one
    # the first time the debug mode reports.
    megatron_root = Path(csa2_module.__file__).resolve().parents[3]

    def count_syncs(args):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                csa2_module._streaming_indexer_topk(*args)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        return sum(
            "synchroniz" in str(w.message)
            and Path(w.filename).resolve().is_relative_to(megatron_root)
            for w in caught
        )

    counts = []
    for scale in (1, 4):
        args = inputs(scale)
        csa2_module._streaming_indexer_topk(*args)  # fill the per-layout caches
        counts.append(count_syncs(args))
    assert counts[0] == counts[1], f"host syncs grow with the pack: {counts}"
    assert counts[1] <= 3, f"too many host syncs per call: {counts}"


def test_streaming_topk_pads_the_selection_to_k():
    """A pack with fewer live keys than k keeps the dense select's fixed width.

    Both key skips can leave the merge with fewer columns than k: the causal skip
    when the pack's valid groups stop before k keys, and the candidate skip when
    fewer than k keys sit in the live 64-key tiles. select_indices always returns
    min(topk, keys) columns with -1 past the live entries, and the selection is
    concatenated with the local indices, so the streamed select must right-pad
    instead of returning a narrower tensor.
    """
    import megatron.core.transformer.experimental_attention_variant.csa2 as csa2_module

    gen = torch.Generator(device="cuda")
    gen.manual_seed(701)
    ratio, block_size = 2, 8

    # (1) Causal skip: the pack's valid groups stop after three keys, so only
    #     three live keys can ever be staged against k = 8.
    topk, keys, rows = 8, 16, 16
    thd = SimpleNamespace(
        valid_tokens=torch.ones(rows, dtype=torch.bool, device="cuda"),
        sequence_ids=torch.zeros(rows, dtype=torch.int64, device="cuda"),
        position_ids=torch.arange(rows, dtype=torch.int64, device="cuda"),
    )
    valid_groups = torch.zeros(keys, dtype=torch.bool, device="cuda")
    valid_groups[:3] = True
    comp = SimpleNamespace(
        cu_seqlens_padded=torch.tensor([0, keys], dtype=torch.int64, device="cuda"),
        max_seqlen=keys,
        valid_groups=valid_groups,
        sequence_ids=torch.zeros(keys, dtype=torch.int64, device="cuda"),
        position_ids=torch.arange(keys, dtype=torch.int64, device="cuda") * ratio,
    )
    q = torch.randn(rows, 1, 4, 32, device="cuda", generator=gen)
    indexer_k = torch.randn(keys, 1, 32, device="cuda", generator=gen)
    weights = torch.randn(rows, 1, 4, device="cuda", generator=gen)
    got, _ = csa2_module._streaming_indexer_topk(
        q, indexer_k, weights, topk, ratio, thd, comp, None, False, block_size
    )
    ref, _ = _dense_thd_select(q, indexer_k, weights, thd, comp, ratio, topk)
    assert got.shape == (1, rows, topk) == ref.shape
    assert got.dtype == torch.int32
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    assert (got[0, :, 3:] == -1).all()  # nothing past the three live keys is selectable

    # (2) Candidate skip: k exceeds one live 64-key tile. Only block 0 (the first
    #     eight keys) is a candidate block, so the staged tile holds 64 keys while
    #     k = 72: the merge is narrower than k even though 128 keys exist.
    topk, keys, rows = 72, 128, 40
    thd = SimpleNamespace(
        valid_tokens=torch.ones(rows, dtype=torch.bool, device="cuda"),
        sequence_ids=torch.zeros(rows, dtype=torch.int64, device="cuda"),
        position_ids=torch.arange(rows, dtype=torch.int64, device="cuda"),
    )
    comp = SimpleNamespace(
        cu_seqlens_padded=torch.tensor([0, keys], dtype=torch.int64, device="cuda"),
        max_seqlen=keys,
        valid_groups=torch.ones(keys, dtype=torch.bool, device="cuda"),
        sequence_ids=torch.zeros(keys, dtype=torch.int64, device="cuda"),
        position_ids=torch.arange(keys, dtype=torch.int64, device="cuda") * ratio,
    )
    q = torch.randn(rows, 1, 4, 32, device="cuda", generator=gen)
    indexer_k = torch.randn(keys, 1, 32, device="cuda", generator=gen)
    weights = torch.randn(rows, 1, 4, device="cuda", generator=gen)
    cand = CSA2CandidateBlocks(
        torch.zeros((1, rows, 1), dtype=torch.int32, device="cuda"), block_size
    )
    got, _ = csa2_module._streaming_indexer_topk(
        q, indexer_k, weights, topk, ratio, thd, comp, cand, False, block_size
    )
    ref, _ = _dense_thd_select(q, indexer_k, weights, thd, comp, ratio, topk, cand)
    assert got.shape == (1, rows, topk) == ref.shape
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    assert (got[0, :, block_size:] == -1).all()  # only block 0's keys are candidates


def test_streaming_topk_sizes_the_slot_space_by_rows_not_keys():
    """A trailing sequence with rows but no compressed key must not index past the slot space.

    The candidate slot space and the candidate-source block table are addressed by the ROW
    sequence id, but were sized from the compressed keys' maximum sequence id. A packed batch
    whose trailing sequence is shorter than the compression ratio owns no compressed group,
    so it had no slot at all and the fused select raised an out-of-bounds index. Those rows
    match no key's sequence id, so nothing is causally visible to them: the dense path selects
    an all -1 row, and the candidate-source block table's slot for them is all -inf.
    """
    import megatron.core.transformer.experimental_attention_variant.csa2 as csa2_module

    gen = torch.Generator(device="cuda")
    gen.manual_seed(909)
    ratio, topk, block_size = 2, 8, 8
    row_counts = [96, 1]  # the trailer is shorter than `ratio`: no compressed group
    capacities = [48, 0]
    s = sum(row_counts)
    rows_seq = torch.cat(
        [torch.full((n,), i, dtype=torch.int64, device="cuda") for i, n in enumerate(row_counts)]
    )
    rows_pos = torch.cat([torch.arange(n, dtype=torch.int64, device="cuda") for n in row_counts])
    thd = SimpleNamespace(
        valid_tokens=torch.ones(s, dtype=torch.bool, device="cuda"),
        sequence_ids=rows_seq,
        position_ids=rows_pos,
    )
    cu = torch.zeros(len(capacities) + 1, dtype=torch.int64, device="cuda")
    cu[1:] = torch.tensor(capacities, dtype=torch.int64, device="cuda").cumsum(0)
    keys = int(cu[-1])
    n_local_blocks = -(-max(capacities) // block_size)
    comp = SimpleNamespace(
        cu_seqlens_padded=cu,
        max_seqlen=max(capacities),
        valid_groups=torch.ones(keys, dtype=torch.bool, device="cuda"),
        sequence_ids=torch.zeros(keys, dtype=torch.int64, device="cuda"),
        position_ids=torch.arange(keys, dtype=torch.int64, device="cuda") * ratio,
    )
    q = torch.randn(s, 1, 4, 32, device="cuda", generator=gen)
    indexer_k = torch.randn(keys, 1, 32, device="cuda", generator=gen)
    weights = torch.randn(s, 1, 4, device="cuda", generator=gen)
    # Every row carries candidate blocks, the trailer included: its sequence id is
    # the index that used to run past the end of the key-sized slot space.
    ids = torch.randint(
        0, n_local_blocks, (1, s, 2), dtype=torch.int32, device="cuda", generator=gen
    )
    cand = CSA2CandidateBlocks(ids, block_size)

    ref, scores = _dense_thd_select(q, indexer_k, weights, thd, comp, ratio, topk, cand)
    got, _ = csa2_module._streaming_indexer_topk(
        q, indexer_k, weights, topk, ratio, thd, comp, cand, False, block_size
    )
    assert got.shape == (1, s, topk) == ref.shape
    assert got.dtype == torch.int32
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    assert (got[0, -row_counts[1] :] == -1).all()  # no key can be visible to the trailer

    # The candidate-source block table is sized the same way: the trailer must read
    # its own all -inf slot instead of indexing past the end.
    _, block_scores = csa2_module._streaming_indexer_topk(
        q, indexer_k, weights, topk, ratio, thd, comp, None, True, block_size
    )
    ref_blocks = csa2_module._chunked_candidate_block_scores(
        scores.squeeze(0), thd.sequence_ids.clamp_min(0), comp, block_size
    )
    assert block_scores.shape == (s, n_local_blocks)
    torch.testing.assert_close(block_scores, ref_blocks, rtol=0, atol=0)
    assert (block_scores[-row_counts[1] :] == -torch.inf).all()


def test_streaming_block_scores_fold_causal_maxima_with_a_candidate_set():
    """want_block_scores folds the causal maxima even when candidates are supplied.

    The candidate-source layer derives its block set from these maxima, so on the
    dense path they come from forward_before_topk's causally masked scores, with
    _chunked_candidate_block_scores as the pinned oracle. The module never reaches
    this combination on its own (the candidate source does not consume candidates),
    so this test drives it directly: one candidate set exercises both halves of the
    split, namely folding before the membership mask (a candidate-excluded block
    whose keys are still causally scorable keeps its causal maximum) and staging the
    tiles a candidate-only skip would call dead. The selection must still honor the
    candidate mask.
    """
    import dataclasses

    import megatron.core.transformer.experimental_attention_variant.csa2 as csa2_module

    gen = torch.Generator(device="cuda")
    gen.manual_seed(202)
    ratio, topk, block_size = 2, 8, 8

    @dataclasses.dataclass
    class _Thd:
        valid_tokens: torch.Tensor
        sequence_ids: torch.Tensor
        position_ids: torch.Tensor

    @dataclasses.dataclass
    class _Compressed:
        cu_seqlens_padded: torch.Tensor
        max_seqlen: int
        valid_groups: torch.Tensor
        sequence_ids: torch.Tensor
        position_ids: torch.Tensor

    row_counts = [96, 40]
    capacities = [48, 24]  # compressed groups per sequence
    s = sum(row_counts)
    rows_seq = torch.cat(
        [torch.full((n,), i, dtype=torch.int64, device="cuda") for i, n in enumerate(row_counts)]
    )
    rows_pos = torch.cat([torch.arange(n, dtype=torch.int64, device="cuda") for n in row_counts])
    cu = torch.zeros(len(capacities) + 1, dtype=torch.int64, device="cuda")
    cu[1:] = torch.tensor(capacities, device="cuda").cumsum(0)
    keys = int(cu[-1])
    key_seq = torch.cat(
        [torch.full((c,), i, dtype=torch.int64, device="cuda") for i, c in enumerate(capacities)]
    )
    key_pos = torch.cat(
        [torch.arange(c, dtype=torch.int64, device="cuda") * ratio for c in capacities]
    )
    thd = _Thd(torch.ones(s, dtype=torch.bool, device="cuda"), rows_seq, rows_pos)
    comp = _Compressed(
        cu, max(capacities), torch.ones(keys, dtype=torch.bool, device="cuda"), key_seq, key_pos
    )

    h, d = 4, 32
    q = torch.randn(s, 1, h, d, device="cuda", generator=gen)
    indexer_k = torch.randn(keys, 1, d, device="cuda", generator=gen)
    weights = torch.randn(s, 1, h, device="cuda", generator=gen)

    # Dense reference: the forward_before_topk no-grad THD math, verbatim.
    scores = csa2_module._chunked_csa2_indexer_scores(q, indexer_k, weights)
    causal = (
        thd.valid_tokens[:, None]
        & comp.valid_groups[None, :]
        & (thd.sequence_ids[:, None] == comp.sequence_ids[None, :])
        & (comp.position_ids[None, :] + ratio <= thd.position_ids[:, None] + 1)
    )
    scores = scores.masked_fill(~causal, -torch.inf)
    k = min(topk, keys)
    ref_blocks = csa2_module._chunked_candidate_block_scores(
        scores.squeeze(0), thd.sequence_ids.clamp_min(0), comp, block_size
    )

    # Sequence 0 rows drop block 1: its keys stay causally scorable, so the causal
    # maximum differs from the candidate-masked one. Sequence 1 rows drop block 2,
    # the only block with keys (64..71) in the second 64-key tile, so a candidate
    # skip alone would call that tile dead while the causal fold still needs it.
    ids = torch.full((1, s, 3), -1, dtype=torch.int32, device="cuda")
    ids[0, : row_counts[0], :3] = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    ids[0, row_counts[0] :, :2] = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    cand = CSA2CandidateBlocks(ids, block_size)
    cand_mask = cand.to_mask_thd(thd.sequence_ids, comp.sequence_ids, comp.position_ids // ratio)
    scores_c = scores.masked_fill(~cand_mask, -torch.inf)
    winners_c = scores_c.reshape(-1, keys).argsort(dim=-1, descending=True, stable=True)
    ref_c = winners_c[..., :k].sort(dim=-1).values.reshape(1, s, k)
    ref_c = ref_c.masked_fill(~scores_c.gather(-1, ref_c).isfinite(), -1).int()
    # The fixture is only discriminating if the candidate mask moves some maxima.
    candidate_masked_blocks = csa2_module._chunked_candidate_block_scores(
        scores_c.squeeze(0), thd.sequence_ids.clamp_min(0), comp, block_size
    )
    assert not torch.equal(candidate_masked_blocks, ref_blocks)

    got_indices, got_blocks = csa2_module._streaming_indexer_topk(
        q, indexer_k, weights, topk, ratio, thd, comp, cand, True, block_size
    )
    torch.testing.assert_close(got_blocks, ref_blocks, rtol=0, atol=0)
    torch.testing.assert_close(got_indices, ref_c, rtol=0, atol=0)


def test_fused_check_also_compares_the_candidate_source_block_scores(pg_collection, monkeypatch):
    """CSA2_FUSED_CHECK gates the block-score fold, not just the selection.

    Every later Reindex layer ranks its candidate set from the candidate-source
    layer's block maxima, so a fold bug can leave this layer's selection
    bit-equal and still corrupt the next layer's candidates. The gate must
    compare fused block scores against the dense reduction bit-exactly; a fold
    that only perturbs the maxima must raise the block-score RuntimeError while
    the index comparison still passes.
    """
    import megatron.core.transformer.experimental_attention_variant.csa2 as csa2_module

    layer = _layer(
        pg_collection,
        2,
        csa2_candidate_source_layer=0,
        csa2_candidate_topk_blocks=2,
        csa2_candidate_block_size=8,
    )
    # eval() drops the indexer-loss path, which is what enables the fused select.
    layer.eval()
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32),
        max_seqlen_q=7,
        max_seqlen_kv=7,
    )
    x = torch.randn(12, 1, layer.config.hidden_size, device="cuda")
    x[[3, 4, 9, 10, 11]] = float("nan")

    monkeypatch.setattr(csa2_module, "_FUSED_CHECK", True)
    monkeypatch.setattr(csa2_module, "_FUSED_DISABLE", False)  # the fused path must run
    csa2_module._check_reported.clear()
    compared = []
    dense_block_scores = csa2_module._dense_candidate_block_scores

    def recording_reduction(*args, **kwargs):
        compared.append(1)
        return dense_block_scores(*args, **kwargs)

    monkeypatch.setattr(csa2_module, "_dense_candidate_block_scores", recording_reduction)
    layer(x, None, packed_seq_params=packed)  # the real fold passes the gate
    assert compared, "the cross-check compared indices but not block scores"

    # Perturb only the block-score fold: the returned selection is untouched, so
    # the index comparison passes and only the block-score comparison can fire.
    streaming = csa2_module._streaming_indexer_topk
    folds = []

    def perturbed_fold(*args, **kwargs):
        indices, block_scores = streaming(*args, **kwargs)
        folds.append(block_scores.clone())
        return indices, torch.where(torch.isfinite(block_scores), block_scores + 0.25, block_scores)

    monkeypatch.setattr(csa2_module, "_streaming_indexer_topk", perturbed_fold)
    with pytest.raises(RuntimeError) as failure:
        layer(x, None, packed_seq_params=packed)
    message = str(failure.value)
    fused_blocks = folds[-1]
    differing = int(torch.isfinite(fused_blocks).any(dim=-1).sum())
    assert differing > 0  # the fixture has rows whose block maxima are finite
    assert "block-score mismatch" in message
    assert f"{differing} of {fused_blocks.shape[0]} rows differ" in message
    magnitude = float(message.rsplit("max|fused-dense|=", 1)[1])
    assert 0.24 < magnitude < 0.26


@pytest.mark.parametrize("width", [0, 1, 3, 17, 1025])
def test_compact_candidates_match_dense_mask_and_bound_retained_storage(width):
    torch.manual_seed(92)
    scores = torch.randint(-2, 3, (2, 9, width), device="cuda").float()
    visible = torch.arange(1, 10, device="cuda").unsqueeze(-1) // 2
    scores.masked_fill_(torch.arange(width, device="cuda") >= visible, -torch.inf)
    compact = candidate_blocks_from_scores(scores, visible, 2, 4)
    expected = _reference_candidates(scores, 2, 2, 4)
    torch.testing.assert_close(compact.to_mask(width), expected)
    assert compact.indices.dtype == torch.int32
    assert compact.indices.shape[-1] <= 2
    if width == 1025:
        retained_bytes = compact.indices.numel() * compact.indices.element_size()
        assert retained_bytes < expected.numel() * expected.element_size() / 50


def test_tied_selection_is_independent_of_masked_future_capacity():
    visible = torch.tensor([[0], [3], [7]], device="cuda")
    expected_indices = torch.tensor([[-1, -1], [0, 1], [0, 1]], device="cuda", dtype=torch.int32)
    expected_blocks = torch.tensor([[-1, -1], [0, 1], [0, 3]], device="cuda", dtype=torch.int32)
    for width in (7, 8, 31, 128):
        scores = torch.zeros(3, width, device="cuda")
        scores.masked_fill_(torch.arange(width, device="cuda") >= visible, -torch.inf)
        indices = CSA2Indexer.select_indices(SimpleNamespace(topk=2), scores)
        compact = candidate_blocks_from_scores(scores, visible, 2, 2)
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(compact.indices, expected_blocks)
        # Repeated invalid slots must not overwrite the valid first block in other rows.
        assert not compact.to_mask(width)[0].any()
        assert compact.to_mask(width)[1, :3].all()


def test_tied_shared_indexers_preserve_prefix_when_sequence_grows(pg_collection):
    torch.manual_seed(93)
    layers = _layers(pg_collection)
    with torch.no_grad():
        for layer in layers:
            indexer = layer.core_attention.indexer
            if indexer is not None:
                indexer.linear_weights_proj.weight.zero_()
        x = torch.randn(17, 2, layers[0].config.hidden_size, device="cuda")
        prefix, _ = _stack(layers, x[:9], CSA2State())
        extended, _ = _stack(layers, x, CSA2State())
    torch.testing.assert_close(prefix, extended[:9], atol=3e-6, rtol=3e-5)


def _groups():
    # A real singleton ProcessGroup supplies size/rank without starting collectives.
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = torch.distributed.ProcessGroup(0, 1)
    return groups


def _math_core(sparse=False, per_token=False, coefficient=0.3):
    core = CompressedSparseAttention2.__new__(CompressedSparseAttention2)
    nn.Module.__init__(core)
    core.config = SimpleNamespace(
        dsa_indexer_loss_coeff=coefficient,
        dsa_indexer_use_sparse_loss=sparse,
        calculate_per_token_loss=per_token,
    )
    core.pg_collection = _groups()
    core.softmax_scale = 0.5
    core.attn_sink = nn.Parameter(torch.tensor([-1.0, 2.0, 4.0]))
    return core


def _inputs(dtype=torch.float32, candidates=False, empty=False):
    torch.manual_seed(102)
    length, batch, heads, dim = (1 if empty else 7), 2, 3, 4
    width = length // 2
    query = torch.randn(length, batch, heads, dim, dtype=dtype, requires_grad=True)
    local_kv = torch.randn(length, batch, dim, dtype=dtype, requires_grad=True)
    global_kv = torch.randn(width, batch, dim, dtype=dtype, requires_grad=True)
    raw_scores = torch.randn(batch, length, width, requires_grad=True)
    visible = torch.arange(width)[None, :] < torch.arange(1, length + 1)[:, None] // 2
    visible = visible.expand(batch, -1, -1).clone()
    if candidates:
        visible[0, 5:, 1] = False
        visible[1, 3:, 0] = False
    scores = raw_scores.masked_fill(~visible, -torch.inf)
    topk = scores.detach().topk(min(2, width), dim=-1).indices
    topk = topk.masked_fill(~scores.detach().gather(-1, topk).isfinite(), -1)
    window = torch.full((batch, length, 2), -1, dtype=torch.int64)
    for row in range(length):
        positions = torch.arange(max(row - 1, 0), row + 1)
        window[:, row, : positions.numel()] = positions
    return (query, local_kv, global_kv, window, topk, scores), raw_scores, visible


def _teacher_oracle(core, query, local_kv, global_kv, window, topk, scores):
    """Independent per-query KL, including window/sink mass before summing heads."""
    losses = []
    for batch in range(scores.shape[0]):
        for row in range(scores.shape[1]):
            positions = scores[batch, row].detach().isfinite().nonzero().flatten()
            if core.config.dsa_indexer_use_sparse_loss:
                positions = positions[torch.isin(positions, topk[batch, row])]
            if positions.numel() == 0:
                losses.append(scores[batch, row, :0].sum() * 0)
                continue
            q = query[row, batch].detach().float()
            selected = global_kv[positions, batch].detach().float()
            global_logits = (q @ selected.T) * core.softmax_scale
            local_positions = window[batch, row]
            local_positions = local_positions[local_positions >= 0]
            local = local_kv[local_positions, batch].detach().float()
            window_logits = (q @ local.T) * core.softmax_scale
            all_logits = torch.cat(
                (global_logits, window_logits, core.attn_sink.detach()[:, None]), dim=-1
            )
            teacher = all_logits.softmax(-1)[:, : positions.numel()].sum(0)
            teacher = teacher / teacher.sum()
            log_prediction = scores[batch, row, positions].float().log_softmax(-1)
            losses.append((teacher * (teacher.clamp_min(1e-10).log() - log_prediction)).sum())
    loss = torch.stack(losses).sum() * core.config.dsa_indexer_loss_coeff
    if not core.config.calculate_per_token_loss:
        loss = loss / (scores.shape[0] * scores.shape[1])
    return loss


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("candidates", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_loss_and_score_gradients_match_joint_teacher(sparse, per_token, candidates, dtype):
    core = _math_core(sparse, per_token)
    args, raw_scores, visible = _inputs(dtype, candidates)
    scores_before = args[-1].detach().clone()
    reference_raw = raw_scores.detach().clone().requires_grad_()
    reference_scores = reference_raw.masked_fill(~visible, -torch.inf)
    expected = _teacher_oracle(core, *args[:-1], reference_scores)
    actual = core._compute_indexer_loss(*args)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(args[-1], scores_before)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(raw_scores.grad, reference_raw.grad, atol=2e-7, rtol=2e-6)
    assert torch.isfinite(raw_scores.grad).all()
    assert raw_scores.grad.abs().sum() > 0
    assert torch.count_nonzero(raw_scores.grad.masked_select(~visible)) == 0
    assert all(tensor.grad is None for tensor in args[:3])
    assert core.attn_sink.grad is None


@pytest.mark.parametrize("sparse", [False, True])
def test_teacher_depends_on_window_and_sink_mass(sparse):
    core = _math_core(sparse)
    args, _, _ = _inputs()
    baseline = core._compute_indexer_loss(*args)
    changed_window = (args[0], args[1] * 5, *args[2:])
    changed = core._compute_indexer_loss(*changed_window)
    assert not torch.isclose(baseline, changed, atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        core.attn_sink.copy_(torch.tensor([9.0, -7.0, 0.5]))
    changed = core._compute_indexer_loss(*args)
    assert not torch.isclose(baseline, changed, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("empty_width", [False, True])
def test_no_visible_global_positions_have_finite_zero_loss(sparse, empty_width):
    core = _math_core(sparse)
    args, raw_scores, _ = _inputs(empty=empty_width)
    if not empty_width:
        args = (
            *args[:-2],
            torch.full_like(args[-2], -1),
            raw_scores.masked_fill(torch.ones_like(raw_scores, dtype=torch.bool), -torch.inf),
        )
    loss = core._compute_indexer_loss(*args)
    assert loss.item() == 0
    loss.backward()
    assert raw_scores.grad is not None
    assert torch.isfinite(raw_scores.grad).all()
    assert torch.count_nonzero(raw_scores.grad) == 0


def test_coefficient_and_token_sum_scaling():
    args, _, _ = _inputs()
    base = _math_core()._compute_indexer_loss(*args)
    scaled = _math_core(coefficient=0.9)._compute_indexer_loss(*args)
    token_sum = _math_core(per_token=True)._compute_indexer_loss(*args)
    torch.testing.assert_close(scaled, base * 3)
    torch.testing.assert_close(token_sum, base * 14)


def test_autoscaler_preserves_output_and_scales_only_auxiliary_backward():
    args, raw_scores, _ = _inputs()
    loss = _math_core()._compute_indexer_loss(*args)
    expected_gradient = torch.autograd.grad(loss, raw_scores, retain_graph=True)[0]
    output = torch.randn(3, requires_grad=True)
    previous = DSAIndexerLossAutoScaler.main_loss_backward_scale
    try:
        DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(0.25))
        attached = DSAIndexerLossAutoScaler.apply(output, loss)
        torch.testing.assert_close(attached, output)
        (attached.sum() * 3).backward()
        torch.testing.assert_close(output.grad, torch.full_like(output, 3))
        torch.testing.assert_close(raw_scores.grad, expected_gradient * 0.25)
    finally:
        DSAIndexerLossAutoScaler.main_loss_backward_scale = previous


class _NativeLinear(nn.Module):
    def __init__(self, input_size, output_size, config, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=config.params_dtype))
        nn.init.normal_(self.weight, std=0.2)

    def forward(self, x):
        return F.linear(x, self.weight), None


class _NativeNorm(nn.RMSNorm):
    def __init__(self, hidden_size, eps, config):
        super().__init__(hidden_size, eps=eps, dtype=config.params_dtype)


class _CPURotaryTable(nn.Module):
    """Ordinary CPU frequency table; the production RoPE application remains unchanged."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("frequencies", 10000 ** (-torch.arange(0, dim, 2).float() / dim))

    def forward(self, length, packed_seq=False):
        angles = torch.outer(torch.arange(length).float(), self.frequencies)
        return torch.cat((angles, angles), dim=-1)[:, None, None, :]


def _sharing_cores(per_token=False, reuse=False):
    config = _make_config(
        num_layers=4 if reuse else 3,
        csa_compress_ratios=[1] * (4 if reuse else 3),
        csa2_kv_source_layers=[0],
        csa2_index_source_layers=[0, 1, 2],
        csa2_candidate_source_layer=0,
        dsa_indexer_topk=2,
        dsa_indexer_loss_coeff=0.3,
        calculate_per_token_loss=per_token,
    )
    modules = CompressedSparseAttentionSubmodules(
        compressor=ModuleSpec(
            module=CSA2Compressor,
            submodules=CompressorSubmodules(_NativeLinear, _NativeLinear, _NativeNorm),
        ),
        indexer=ModuleSpec(
            module=CSA2Indexer,
            submodules=CSA2IndexerSubmodules(
                _NativeLinear, _NativeLinear, _NativeNorm, _NativeLinear
            ),
        ),
    )
    return [
        CompressedSparseAttention2(
            config,
            modules,
            layer_number=layer + 1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=_groups(),
            rotary_pos_emb=_CPURotaryTable(config.qk_pos_emb_head_dim),
        )
        for layer in range(config.num_layers)
    ]


def test_shared_keys_accumulate_consumer_auxiliary_gradients_without_backbone_gradients():
    torch.manual_seed(91)
    owner, first, second = _sharing_cores()
    state = CSA2State()
    hidden = [torch.randn(7, 2, 32, requires_grad=True) for _ in range(3)]
    query_latents = [torch.randn(7, 2, 16, requires_grad=True) for _ in range(3)]
    records = [
        core._shared_global_attention(x, qr, state, use_indexer_loss=True)
        for core, x, qr in zip((owner, first, second), hidden, query_latents)
    ]
    assert records[0][0] is records[1][0] is records[2][0]
    assert state.indexer_k.requires_grad
    assert state.candidates is not None and not state.candidates.indices.requires_grad
    losses = []
    for core, (global_kv, indices, scores) in zip((owner, first, second), records):
        query = torch.randn(7, 2, 4, 16, requires_grad=True)
        local = torch.randn(7, 2, 16, requires_grad=True)
        window = torch.arange(7).view(1, 7, 1).expand(2, -1, -1)
        losses.append(core._compute_indexer_loss(query, local, global_kv, window, indices, scores))
    weight = owner.indexer.linear_wk.weight
    individual = [torch.autograd.grad(loss, weight, retain_graph=True)[0] for loss in losses]
    assert all(gradient.abs().sum() > 0 for gradient in individual)
    sum(losses).backward()
    torch.testing.assert_close(weight.grad, sum(individual), atol=2e-6, rtol=2e-5)
    assert all(x.grad is None for x in hidden + query_latents)
    assert all(parameter.grad is None for parameter in owner.compressor.parameters())
    for core in (owner, first, second):
        for parameter in core.indexer.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    # Detaching the indexer branch must preserve the independent LM/global-KV graph.
    records[0][0].square().sum().backward()
    assert hidden[0].grad is not None and hidden[0].grad.abs().sum() > 0
    assert owner.compressor.linear_wkv.weight.grad.abs().sum() > 0


def _forward_inputs(count):
    return [
        (
            torch.randn(7, 2, 4, 16, requires_grad=True),
            torch.randn(7, 2, 1, 16, requires_grad=True),
            torch.randn(7, 2, 32, requires_grad=True),
            torch.randn(7, 2, 16, requires_grad=True),
        )
        for _ in range(count)
    ]


def _forward_cores(cores, inputs):
    state = CSA2State()
    return [
        core(q, kv, kv, None, x=x, qr=qr, csa2_state=state)
        for core, (q, kv, x, qr) in zip(cores, inputs)
    ]


def _record_losses(monkeypatch):
    records = []

    def record(**kwargs):
        assert not kwargs["loss"].requires_grad
        records.append(kwargs)

    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "save_loss_to_tracker", record)
    monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)
    return records


@pytest.mark.parametrize("per_token", [False, True])
def test_forward_auxiliary_loss_preserves_main_gradients_and_excludes_reuse(monkeypatch, per_token):
    torch.manual_seed(307)
    records = _record_losses(monkeypatch)
    cores = _sharing_cores(per_token, reuse=True)
    baseline = _sharing_cores(per_token, reuse=True)
    for core, reference in zip(cores, baseline):
        reference.load_state_dict(core.state_dict())
    baseline[0].config.dsa_indexer_loss_coeff = 0
    inputs = _forward_inputs(len(cores))
    reference_inputs = [
        tuple(tensor.detach().clone().requires_grad_() for tensor in row) for row in inputs
    ]
    outputs = _forward_cores(cores, inputs)
    expected = _forward_cores(baseline, reference_inputs)
    assert [record["layer_number"] for record in records] == [1, 2, 3]
    assert all(record["num_layers"] == 4 for record in records)
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    sum(output.square().mean() for output in outputs).backward()
    sum(output.square().mean() for output in expected).backward()
    for core, reference in zip(cores, baseline):
        parameters = dict(reference.named_parameters())
        for name, parameter in core.named_parameters():
            if name.startswith("indexer."):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert parameters[name].grad is None
            else:
                torch.testing.assert_close(parameter.grad, parameters[name].grad, atol=0, rtol=0)
    for row, reference_row in zip(inputs, reference_inputs):
        for tensor, reference in zip(row, reference_row):
            torch.testing.assert_close(tensor.grad, reference.grad, atol=0, rtol=0)


def test_token_sum_mode_keeps_same_logged_metric_and_scales_gradients(monkeypatch):
    torch.manual_seed(81)
    records = _record_losses(monkeypatch)
    average_cores = _sharing_cores()
    sum_cores = _sharing_cores(per_token=True)
    for core, summed in zip(average_cores, sum_cores):
        summed.load_state_dict(core.state_dict())
    inputs = _forward_inputs(3)
    average_outputs = _forward_cores(average_cores, inputs)
    sum_outputs = _forward_cores(sum_cores, inputs)
    assert len(records) == 6
    for average, summed in zip(records[:3], records[3:]):
        torch.testing.assert_close(average["loss"], summed["loss"])
    sum(output.sum() for output in average_outputs).backward()
    sum(output.sum() for output in sum_outputs).backward()
    for average, summed in zip(average_cores, sum_cores):
        for parameter, summed_parameter in zip(
            average.indexer.parameters(), summed.indexer.parameters()
        ):
            torch.testing.assert_close(
                summed_parameter.grad, parameter.grad * 14, atol=3e-6, rtol=3e-5
            )


@pytest.mark.parametrize("evaluation", [False, True])
def test_evaluation_and_no_grad_do_not_attach_auxiliary_loss(monkeypatch, evaluation):
    records = _record_losses(monkeypatch)
    cores = _sharing_cores(reuse=True)
    inputs = _forward_inputs(len(cores))
    if evaluation:
        for core in cores:
            core.eval()
        outputs = _forward_cores(cores, inputs)
        sum(output.square().mean() for output in outputs).backward()
        assert all(
            parameter.grad is None
            for core in cores
            if core.indexer is not None
            for parameter in core.indexer.parameters()
        )
    else:
        with torch.no_grad():
            outputs = _forward_cores(cores, inputs)
        assert all(not output.requires_grad for output in outputs)
    assert not records


@pytest.mark.parametrize("ratio", [1, 2])
def test_thd_cp_compression_keeps_global_groups_and_gradients(ratio):
    """A group straddling the CP cut has one owner and the right source gradient."""

    class CPGroup:
        def __init__(self, rank):
            self._rank = rank

        def rank(self):
            return self._rank

        def size(self):
            return 2

    logical = torch.tensor([0, 4, 13], dtype=torch.int32)
    physical = torch.tensor([0, 4, 14], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=10,
        max_seqlen_kv=10,
        cp_partition_mode="contiguous",
    )
    x = torch.arange(14, dtype=torch.float32, requires_grad=True)
    local_values = []
    layouts = []
    for rank in range(2):
        tokens = build_csa2_thd_layout(params, 7, cp_group=CPGroup(rank))
        cp_layout = build_csa2_cp_compression_layout(tokens, ratio, halo_rows=2)
        start = rank * 7
        halo = x.new_zeros(2) if rank == 0 else x[start - 2 : start]
        source = torch.cat((halo, x[start : start + 7]))
        local_values.append(source[cp_layout.local.source_indices].mean(-1))
        layouts.append(cp_layout)
    global_layout = layouts[0].global_layout
    actual = layouts[0].to_sequence_major(torch.cat(local_values))
    expected = x[global_layout.source_indices].mean(-1).masked_fill(~global_layout.valid_groups, 0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward(retain_graph=True)
    actual_grad = x.grad.clone()
    x.grad = None
    expected.sum().backward()
    torch.testing.assert_close(actual_grad, x.grad, atol=0, rtol=0)


def test_thd_layout_cache_rebuilds_when_reused_metadata_changes():
    """A PackedSeqParams reused with new boundaries must not return the old layout."""

    def packed(boundaries, max_seqlen):
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=boundaries,
            cu_seqlens_kv=boundaries,
            cu_seqlens_q_padded=boundaries,
            cu_seqlens_kv_padded=boundaries,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

    boundaries = torch.tensor([0, 4, 8], dtype=torch.int32)
    params = packed(boundaries, 5)
    first = build_csa2_thd_layout(params, 8)
    assert build_csa2_thd_layout(params, 8) is first

    # Refill the same buffer in place for the next micro-batch, as a static-input
    # loader does; no attribute of the metadata object changes.
    boundaries.copy_(torch.tensor([0, 3, 8], dtype=torch.int32))
    second = build_csa2_thd_layout(params, 8)
    expected = build_csa2_thd_layout(packed(torch.tensor([0, 3, 8], dtype=torch.int32), 5), 8)
    assert second is not first
    for name in ("sequence_ids", "position_ids", "valid_tokens"):
        torch.testing.assert_close(getattr(second, name), getattr(expected, name), atol=0, rtol=0)


@pytest.mark.parametrize("chunk_index", [0, 1])
def test_thd_cp_chunked_compression_matches_global_groups_and_gradients(chunk_index):
    """Current CP owners plus tape history preserve global values and adjoints."""
    from megatron.core.pipeline_parallel.chunk_tape import ChunkTape, extend_chunk_kv

    class CPGroup:
        def __init__(self, rank):
            self._rank = rank

        def rank(self):
            return self._rank

        def size(self):
            return 2

    logical = torch.tensor([0, 32], dtype=torch.int32)
    physical = torch.tensor([0, 32], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=32,
        max_seqlen_kv=32,
        cp_partition_mode="contiguous",
        cp_chunk_index=chunk_index,
        cp_chunk_count=2,
    )
    x = torch.arange(32, dtype=torch.float64, requires_grad=True)
    tape = ChunkTape(2)

    def forward(index, mode):
        tape.begin(index, mode)
        params.cp_chunk_index = index
        layouts, gathered_parts = [], []
        for rank in range(2):
            tokens = build_csa2_thd_layout(params, 8, cp_group=CPGroup(rank))
            layout = build_csa2_cp_compression_layout(tokens, ratio=2, halo_rows=2)
            start = tokens.global_start
            halo = x.new_zeros(2) if start == 0 else x[start - 2 : start]
            local = torch.cat((halo, x[start : start + 8]))
            gathered_parts.append(local[layout.local.source_indices].mean(-1))
            layouts.append(layout)
        global_layout = layouts[0].global_layout
        assert all(layout.global_layout.capacity == global_layout.capacity for layout in layouts)
        current = layouts[0].to_sequence_major(torch.cat(gathered_parts))
        prefix_mask = global_layout.source_indices[:, -1] < index * 16
        actual = extend_chunk_kv(tape, 0, "global_kv", current, prefix_mask)
        expected = (
            x[global_layout.source_indices].mean(-1).masked_fill(~global_layout.valid_groups, 0)
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        return actual, expected

    for index in range(chunk_index):
        with torch.no_grad():
            forward(index, "prefill")
    actual, expected = forward(chunk_index, "grad")
    actual.sum().backward()
    tape.collect_all_adjoints()
    for index in reversed(range(chunk_index)):
        replay, _ = forward(index, "grad")
        (replay.sum() * 0).backward()
        tape.collect_all_adjoints()
    actual_grad = x.grad.clone()
    x.grad = None
    expected.sum().backward()
    torch.testing.assert_close(actual_grad, x.grad, atol=0, rtol=0)
