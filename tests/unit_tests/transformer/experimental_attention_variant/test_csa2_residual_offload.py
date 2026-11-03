# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Residual offload for the CSA2 full-recompute checkpoint."""

import contextlib

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    csa2_hybrid_adapter as adapter,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offload moves CUDA tensors")
def test_residual_offload_matches_device_resident_gradients(monkeypatch):
    torch.manual_seed(0)
    weight = torch.randn(256, 256, device="cuda", requires_grad=True)
    residual = torch.randn(1024, 256, device="cuda")
    offloaded = []

    class RecordingPinnedResidual(adapter._PinnedResidual):
        __slots__ = ()

        def __init__(self, tensor):
            offloaded.append(tensor.shape)
            super().__init__(tensor)

    monkeypatch.setattr(adapter, "_PinnedResidual", RecordingPinnedResidual)

    def region(x, w):
        return torch.tanh(x @ w).square() + x

    def gradients(offload):
        x = residual.clone().requires_grad_()
        hooks = adapter._residual_offload_hooks(x) if offload else contextlib.nullcontext()
        with hooks:
            out = checkpoint(region, x, weight, use_reentrant=False)
        # Overwrite the freed device block's neighbours before backward reads the copy.
        torch.randn(4096, 4096, device="cuda")
        out.square().sum().backward()
        grads = x.grad.clone(), weight.grad.clone()
        weight.grad = None
        return grads

    expected = gradients(offload=False)
    actual = gradients(offload=True)
    assert offloaded == [residual.shape]
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
