# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the production output boundary without importing accelerator backends."""

import ast
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[4] / "megatron/core/transformer"


@pytest.fixture
def attention_code():
    path = SOURCE / "experimental_attention_variant/deepseek_v4_hybrid_attention.py"
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body if getattr(node, "name", "") == "DSv4HybridAttention")
    nodes = [node for node in tree.body if getattr(node, "name", "") == "_fp8_round_trip"]
    nodes += [node for node in owner.body if getattr(node, "name", "") == "_forward_impl"]
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    csa2_path = SOURCE / "experimental_attention_variant/csa2.py"
    csa2 = ast.parse(csa2_path.read_text())
    nodes = [
        node
        for node in csa2.body
        if getattr(node, "name", "") in ("_FP8FakeQuantSTE", "_fp8_fake_quant")
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(csa2_path), "exec"), namespace)
    return namespace


def test_output_act_defaults_preserve_the_existing_quantization_switches():
    tree = ast.parse((SOURCE / "transformer_config.py").read_text())
    defaults = {
        node.target.id: ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in ("dsv4_fp8_output_act", "kv_fp8_emulation", "csa2_fp8_output_act")
    }
    assert defaults == {
        "dsv4_fp8_output_act": False,
        "kv_fp8_emulation": False,
        "csa2_fp8_output_act": True,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_output_round_trip_preserves_groups_and_dtype(attention_code, dtype):
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(2, 3, 96, generator=generator).to(dtype).transpose(0, 1)
    x[..., :32] *= 2.0**-20
    x[..., 32:64] *= 8
    x[0] = 0
    x[1, 0, 32] = 448
    x[1, 1, 32] = 450
    expected = x.detach().clone().contiguous()
    for row in expected.reshape(-1, 96):
        for block in row.split(32):
            fraction, exponent = math.frexp(max(float(block.abs().max()), 1e-4) / 448)
            scale = 2.0 ** (exponent - (fraction == 0.5))
            block.copy_((block.float() / scale).to(torch.float8_e4m3fn).float() * scale)
    actual = attention_code["_fp8_round_trip"](x)
    assert actual.dtype == dtype and actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, x)


class _NoOffload(nullcontext):
    def __init__(self, enabled, tensor, name):
        super().__init__(tensor)

    def group_offload(self, tensor, **kwargs):
        return tensor


@pytest.mark.parametrize("fused_rope", [False, True])
@pytest.mark.parametrize("output_fp8", [False, True])
@pytest.mark.parametrize("kv_fp8", [False, True])
@pytest.mark.parametrize("csa2_fp8", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_output_act_quantizes_after_inverse_rope_and_before_wo_a(
    attention_code, monkeypatch, fused_rope, output_fp8, kv_fp8, csa2_fp8, dtype
):
    generator = torch.Generator().manual_seed(9)
    core_output = torch.randn(3, 1, 2, 32, generator=generator).to(dtype)
    core_output[0] *= 2.0**-20
    core_output[1, :, 0] = 0
    core_output.requires_grad_()
    events = []
    quantize = attention_code["_fp8_round_trip"]
    quantizer_inputs = []

    def quantize_output(value):
        events.append("output-quantize")
        quantizer_inputs.append(value.detach().clone())
        return quantize(value)

    def inverse_rope(value):
        first, second = value.chunk(2, dim=-1)
        return torch.cat((0.6 * first + 0.8 * second, -0.8 * first + 0.6 * second), -1)

    def rotary(value, *args, **kwargs):
        assert kwargs["inverse"] is True
        events.append("inverse-rope")
        return inverse_rope(value)

    def fused_rotary(value, cos, sin, nope_dim, *args, **kwargs):
        assert kwargs["inverse"] is True
        events.append("inverse-rope")
        return torch.cat((value[..., :nope_dim], inverse_rope(value[..., nope_dim:])), -1)

    def core(query, key, value, *args, **kwargs):
        # The native hybrid passes value-space output, without an absorbed up projection.
        assert "up_v_weight" not in kwargs
        if csa2_fp8:
            events.append("csa2-quantize")
            return attention_code["_fp8_fake_quant"](core_output).flatten(2)
        return core_output.flatten(2)

    def wo_b(value):
        events.append("wo-b")
        return value, None

    einsum = torch.einsum

    def wo_a(*args):
        events.append("wo-a")
        return einsum(*args)

    config = SimpleNamespace(
        dsv4_version="v4.1",
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        apply_rope_fusion=fused_rope,
        kv_fp8_emulation=kv_fp8,
        dsv4_fp8_output_act=output_fp8,
        csa2_fp8_output_act=csa2_fp8,
        output_projection_lora_rank=32,
    )
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    fake = SimpleNamespace(
        config=config,
        pg_collection=SimpleNamespace(cp=group),
        get_query_key_value_tensors=lambda x, *args, **kwargs: (x, x, x, x, x),
        offload_core_attention=False,
        offload_attn_proj=False,
        training=True,
        core_attention=core,
        recompute_up_proj=False,
        num_attention_heads_per_partition=2,
        output_projection_local_groups=2,
        _dsv4_uses_yarn_rope=False,
        rotary_pos_emb=lambda *args, **kwargs: None,
        linear_o_group_proj=torch.eye(32, dtype=dtype).repeat(2, 1),
        linear_proj=wo_b,
    )
    if fused_rope:
        fake.rotary_pos_emb = SimpleNamespace(
            get_cached_cos_sin=lambda *args, **kwargs: (None, None)
        )
    attention_code.update(
        apply_module=lambda module: module,
        get_pg_size=lambda group: group.size(),
        off_interface=_NoOffload,
        apply_rotary_pos_emb=rotary,
        fused_mla_rope_inplace=object(),
        fused_mla_rope_out_of_place=fused_rotary,
        _fp8_round_trip=quantize_output,
    )
    monkeypatch.setattr(torch, "einsum", wo_a)
    actual, bias = attention_code["_forward_impl"](fake, core_output.flatten(2), None)
    assert bias is None
    expected = attention_code["_fp8_fake_quant"](core_output) if csa2_fp8 else core_output
    expected = torch.cat((expected[..., :16], inverse_rope(expected[..., 16:])), -1)
    output_site = output_fp8 or kv_fp8
    if output_site:
        assert torch.equal(quantizer_inputs[0], expected)
        rounded = quantize(expected)
        expected = rounded.detach() + (expected - expected.detach()) if output_fp8 else rounded
    expected = expected.flatten(2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert events == (
        (["csa2-quantize"] if csa2_fp8 else [])
        + ["inverse-rope"]
        + (["output-quantize"] if output_site else [])
        + ["wo-a", "wo-b"]
    )
    upstream = torch.linspace(-1, 1, actual.numel()).reshape_as(actual)
    gradient = torch.autograd.grad(actual, core_output, upstream, retain_graph=True)[0]
    reference = torch.autograd.grad(expected, core_output, upstream)[0]
    torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
    if output_fp8:
        unrounded = torch.cat((core_output[..., :16], inverse_rope(core_output[..., 16:])), -1)
        identity_gradient = torch.autograd.grad(unrounded.flatten(2), core_output, upstream)[0]
        torch.testing.assert_close(gradient, identity_gradient, rtol=0, atol=0)
