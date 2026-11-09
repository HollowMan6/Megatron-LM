# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The CSA2 FP8 output round trip must run without an absorbed V up projection.

The switch quantizes the attention output before the output projection, which is
the official anchor. Its absorbed form needs ``up_v_weight``; the DSv4.1 hybrid
attention path has no such weight because its core attention already returns
value-space output. These tests pin that branch: no raise, the same blockwise
quantizer, and a single log line stating the branch.
"""

import logging

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import csa2

BLOCK = 32
HEADS = 4
V_HEAD_DIM = 64
LATENT = 16


def _quantize(output, up_v_weight=None):
    return csa2._csa2_fp8_act_output(output, up_v_weight)


@pytest.fixture(autouse=True)
def _reset_report_flag(monkeypatch):
    monkeypatch.setattr(csa2, "_CSA2_FP8_ACT_DEGRADED_REPORTED", False)


def test_no_up_v_weight_does_not_raise_and_quantizes():
    torch.manual_seed(0)
    output = torch.randn(2, 3, HEADS, V_HEAD_DIM, dtype=torch.bfloat16)
    quantized = _quantize(output, None)
    assert quantized.shape == (2, 3, HEADS * V_HEAD_DIM)
    assert quantized.dtype == torch.bfloat16
    assert not torch.equal(quantized, output.reshape(2, 3, -1))


def test_the_round_trip_is_a_rounding():
    torch.manual_seed(1)
    output = torch.randn(2, 2, HEADS, V_HEAD_DIM, dtype=torch.bfloat16)
    once = _quantize(output, None)
    twice = _quantize(once.reshape(2, 2, HEADS, -1), None)
    assert torch.equal(once, twice)


def test_blocks_stay_independent_along_the_value_dimension():
    """Blocking must follow the last dimension, so one block cannot move another."""
    torch.manual_seed(2)
    output = torch.randn(1, 2, HEADS, V_HEAD_DIM, dtype=torch.bfloat16)
    base = _quantize(output, None)
    scaled = output.reshape(1, 2, HEADS, -1).clone()
    scaled[..., BLOCK:] *= 8.0
    moved = _quantize(scaled, None)
    assert torch.equal(moved[..., :BLOCK], base[..., :BLOCK])
    assert not torch.equal(moved[..., BLOCK:], base[..., BLOCK:])


def test_absorbed_path_keeps_the_preexisting_arithmetic():
    torch.manual_seed(3)
    latent = torch.randn(2, 3, HEADS, LATENT, dtype=torch.bfloat16)
    up_v_weight = torch.randn(HEADS, V_HEAD_DIM, LATENT, dtype=torch.bfloat16)
    absorbed = _quantize(latent, up_v_weight)
    reference = csa2._fp8_fake_quant(
        torch.einsum("sbhc,hdc->sbhd", latent, up_v_weight).contiguous().to(torch.bfloat16),
        BLOCK,
    ).reshape(2, 3, HEADS * V_HEAD_DIM)
    assert torch.equal(absorbed, reference)


def test_the_degraded_branch_is_reported_once_per_process(caplog):
    torch.manual_seed(4)
    output = torch.randn(1, 1, HEADS, V_HEAD_DIM, dtype=torch.bfloat16)
    with caplog.at_level(logging.INFO, logger=csa2.__name__):
        _quantize(output, None)
        _quantize(output, None)
    stated = [record for record in caplog.records if "no up_v_weight supplied" in record.getMessage()]
    assert len(stated) == 1, [record.getMessage() for record in caplog.records]
    assert "block 32" in stated[0].getMessage()


def test_the_absorbed_branch_stays_silent(caplog):
    torch.manual_seed(5)
    latent = torch.randn(1, 1, HEADS, LATENT, dtype=torch.bfloat16)
    up_v_weight = torch.randn(HEADS, V_HEAD_DIM, LATENT, dtype=torch.bfloat16)
    with caplog.at_level(logging.INFO, logger=csa2.__name__):
        _quantize(latent, up_v_weight)
    assert not [record for record in caplog.records if "no up_v_weight" in record.getMessage()]


def test_a_value_dimension_that_is_not_a_multiple_of_the_block_still_works():
    """v_head_dim need not be 32-aligned; the quantizer pads internally."""
    torch.manual_seed(6)
    output = torch.randn(1, 2, HEADS, 48, dtype=torch.bfloat16)
    quantized = _quantize(output, None)
    assert quantized.shape == (1, 2, HEADS * 48)
    assert torch.isfinite(quantized).all()
