# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the opt-in release of blockwise FP8 columnwise weight copies.

``fp8_param_free_columnwise_after_backward`` makes a columnwise copy exist only while a
dgrad needs it: the module's backward hook drops it, and the step-start all-gather
post-processing drops the stale copy instead of transposing every weight back, so TE
recreates it on demand from the rowwise copy. These tests pin the properties the change
relies on:

1. the free happens *after* the module's backward, i.e. after the dgrad already consumed
   the copy, and it leaves no columnwise copy behind;
2. freeing does not perturb the math: outputs and gradients match a run without hooks;
3. TE rebuilds the copy bitwise-identically from the rowwise copy when it is needed
   again, so a second step keeps the same gradients;
4. the all-gather post-processing drops the copy only when the flag is on, and a two-step
   run through that path keeps bitwise-identical outputs and gradients.
"""

import pytest
import torch

from megatron.core import fp8_utils
from megatron.core.fp8_utils import (
    _FP8_COLUMN_WISE_FREE_HOOK_FLAG,
    free_fp8_columnwise_after_backward,
    install_fp8_columnwise_free_hooks,
    is_fp8_columnwise_free_installed,
    post_all_gather_processing,
    quantize_param_shard,
)
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Float8BlockScaling, Format
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
    from transformer_engine_torch import DType as TEDType

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

requires_te_cuda = pytest.mark.skipif(
    not (HAVE_TE and torch.cuda.is_available()),
    reason="requires Transformer Engine and a CUDA device",
)


@pytest.fixture(autouse=True)
def _restore_transient_mode():
    """The transient contract lives in module-level switches; never leak it across tests."""
    saved = (
        fp8_utils._FP8_COLUMN_WISE_EXPECTED,
        fp8_utils._FP8_COLUMN_WISE_TRANSIENT,
        fp8_utils._FP8_COLUMN_WISE_INSTALLED_HOOKS,
    )
    yield
    (
        fp8_utils._FP8_COLUMN_WISE_EXPECTED,
        fp8_utils._FP8_COLUMN_WISE_TRANSIENT,
        fp8_utils._FP8_COLUMN_WISE_INSTALLED_HOOKS,
    ) = saved


BLOCK_LEN = 128  # blockwise (2D) FP8 scaling quantizes in 128x128 tiles.
SIZE = 2 * BLOCK_LEN


def _base_config(**overrides):
    """Minimal valid TransformerConfig for the validation tests."""
    kwargs = dict(num_layers=1, hidden_size=128, num_attention_heads=8, num_query_groups=8)
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _make_fp8_param(shape, seed):
    """Build a blockwise-quantized FP8 weight parameter, as ``fp8_param=True`` does."""
    torch.manual_seed(seed)
    quantizer = Float8BlockQuantizer(
        fp8_dtype=TEDType.kFloat8E4M3, rowwise=True, columnwise=True, block_scaling_dim=2
    )
    weight = quantizer(torch.randn(*shape, device="cuda", dtype=torch.bfloat16))
    return torch.nn.Parameter(weight)


def _make_te_linear(seed):
    """A TE linear whose weight was created under ``fp8_model_init`` (fp8_param path)."""
    torch.manual_seed(seed)
    with te.fp8_model_init(enabled=True, recipe=Float8BlockScaling(fp8_format=Format.HYBRID)):
        return te.Linear(SIZE, SIZE, bias=False, params_dtype=torch.bfloat16)


class _QuantizedWeightModule(torch.nn.Module):
    """Synthetic owner of a real blockwise FP8 weight.

    The forward consumes the rowwise copy through ``dequantize`` and requests the
    columnwise copy the way TE's ``Linear.backward`` does for the dgrad operands, so the
    parameter is used the same way the production path uses it.
    """

    def __init__(self, seed):
        super().__init__()
        self.weight = _make_fp8_param((SIZE, SIZE), seed)

    def forward(self, x):
        self.weight.update_usage(columnwise_usage=True)
        return x @ self.weight.dequantize().to(x.dtype).t()


def _hooked_modules(model):
    return [m for m in model.modules() if getattr(m, _FP8_COLUMN_WISE_FREE_HOOK_FLAG, False)]


# --------------------------------------------------------------------------------------
# 1. The helper itself
# --------------------------------------------------------------------------------------


def test_install_is_a_noop_without_fp8_parameters():
    """No quantized weight -> nothing to hook (the flag-off / TE-absent path)."""
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    assert install_fp8_columnwise_free_hooks(model) == 0
    assert not _hooked_modules(model)
    # Plain bf16 parameters expose neither ``update_usage`` nor a columnwise copy.
    assert all(free_fp8_columnwise_after_backward(m) == 0 for m in model.modules())


def test_free_skips_parameters_without_a_rowwise_copy():
    """A columnwise-only weight keeps its last copy: TE asserts one copy survives."""

    class _ColumnwiseOnly:
        _columnwise_data = object()
        _rowwise_data = None

        def __init__(self):
            self.calls = []

        def update_usage(self, **kwargs):
            self.calls.append(kwargs)

    class _Owner(torch.nn.Module):
        def __init__(self, param):
            super().__init__()
            self.param = param

    param = _ColumnwiseOnly()
    owner = _Owner(param)
    assert free_fp8_columnwise_after_backward(owner) == 0
    assert param.calls == []


@requires_te_cuda
def test_free_returns_zero_when_the_copy_is_already_gone():
    module = _QuantizedWeightModule(seed=0).cuda()
    assert module.weight._columnwise_data is not None  # quantized with both copies
    assert free_fp8_columnwise_after_backward(module) == 1
    assert module.weight._columnwise_data is None
    assert module.weight._rowwise_data is not None
    assert free_fp8_columnwise_after_backward(module) == 0  # already gone


# --------------------------------------------------------------------------------------
# 2. Synthetic module: the hook fires after backward, gradients are unchanged
# --------------------------------------------------------------------------------------


@requires_te_cuda
def test_hook_frees_after_backward_and_gradients_match():
    hooked = _QuantizedWeightModule(seed=0).cuda()
    plain = _QuantizedWeightModule(seed=0).cuda()
    assert install_fp8_columnwise_free_hooks(hooked) == 1
    assert len(_hooked_modules(hooked)) == 1

    x = torch.randn(8, SIZE, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_plain = x.detach().clone().requires_grad_(True)

    y = hooked(x)
    y.sum().backward()
    y_plain = plain(x_plain)
    y_plain.sum().backward()

    # Identical math with and without the hook.
    assert torch.equal(y, y_plain)
    assert torch.equal(x.grad, x_plain.grad)
    # Freed once the module's backward is done ...
    assert hooked.weight._columnwise_data is None
    # ... while the unhooked module keeps paying for the copy (the memory we reclaim).
    assert plain.weight._columnwise_data is not None


@requires_te_cuda
def test_recreated_columnwise_copy_is_bitwise_identical():
    hooked = _QuantizedWeightModule(seed=0).cuda()
    plain = _QuantizedWeightModule(seed=0).cuda()
    install_fp8_columnwise_free_hooks(hooked)
    x = torch.randn(8, SIZE, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    hooked(x).sum().backward()
    plain(x.detach().clone().requires_grad_(True)).sum().backward()
    assert hooked.weight._columnwise_data is None

    # The next dgrad rebuilds the copy from the rowwise data, bit for bit.
    hooked.weight.update_usage(columnwise_usage=True)
    assert hooked.weight._columnwise_data is not None
    assert torch.equal(hooked.weight._columnwise_data, plain.weight._columnwise_data)
    assert torch.equal(hooked.weight._columnwise_scale_inv, plain.weight._columnwise_scale_inv)


# --------------------------------------------------------------------------------------
# 3. Real TE linear: the hook fires on TE's own backward, across two steps
# --------------------------------------------------------------------------------------


@requires_te_cuda
def test_te_linear_gradients_match_across_two_steps():
    recipe = Float8BlockScaling(fp8_format=Format.HYBRID)
    hooked = _make_te_linear(seed=0)
    plain = _make_te_linear(seed=0)
    # Same initialization, so the two runs are comparable bit for bit.
    assert torch.equal(hooked.weight._rowwise_data, plain.weight._rowwise_data)

    installed = install_fp8_columnwise_free_hooks(hooked)
    assert installed == 1

    torch.manual_seed(7)
    inputs = [
        torch.randn(16, SIZE, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2)
    ]

    for step in range(2):
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            # Step 2 runs with the columnwise copy freed at the end of step 1's backward,
            # so its dgrad has to recreate the copy on demand.
            x = inputs[step].detach().clone().requires_grad_(True)
            y = hooked(x)
            (y.float() ** 2).sum().backward()
            x_plain = inputs[step].detach().clone().requires_grad_(True)
            y_plain = plain(x_plain)
            (y_plain.float() ** 2).sum().backward()

        assert torch.equal(y, y_plain), f"forward diverged on step {step}"
        assert torch.equal(x.grad, x_plain.grad), f"dgrad diverged on step {step}"
        assert torch.equal(
            hooked.weight.grad.dequantize(), plain.weight.grad.dequantize()
        ), f"wgrad diverged on step {step}"
        assert hooked.weight._columnwise_data is None, f"copy survived step {step}"
        assert plain.weight._columnwise_data is not None
        # The wgrad path keeps the rowwise copy that TE rebuilds the columnwise one from.
        assert hooked.weight._rowwise_data is not None


# --------------------------------------------------------------------------------------
# 4. All-gather post-processing: drop instead of transpose when the flag is on
# --------------------------------------------------------------------------------------


@requires_te_cuda
def test_post_all_gather_processing_drops_the_copy_only_when_transient():
    module = _QuantizedWeightModule(seed=0).cuda()
    weight = module.weight
    assert weight._columnwise_data is not None  # quantized with both copies

    # Flag off: the historical behavior, the all-gather hook transposes the copy back.
    post_all_gather_processing([weight])
    assert weight._columnwise_data is not None

    # Flag on: TransformerBlock installs the hooks, which is what switches the contract.
    install_fp8_columnwise_free_hooks(module)
    post_all_gather_processing([weight])
    assert weight._columnwise_data is None
    assert weight._rowwise_data is not None


def test_post_all_gather_processing_is_untouched_by_default():
    """Without the flag nothing changes, not even the attribute lookup path."""
    assert fp8_utils._FP8_COLUMN_WISE_TRANSIENT is False
    post_all_gather_processing([])  # empty batch is a no-op


@requires_te_cuda
def test_transient_all_gather_keeps_two_steps_bitwise_identical():
    """A two-step run through the drop-at-all-gather path matches the flag-off run."""
    recipe = Float8BlockScaling(fp8_format=Format.HYBRID)
    transient = _make_te_linear(seed=5)
    plain = _make_te_linear(seed=5)
    assert torch.equal(transient.weight._rowwise_data, plain.weight._rowwise_data)

    torch.manual_seed(11)
    inputs = [torch.randn(16, SIZE, device="cuda", dtype=torch.bfloat16) for _ in range(2)]

    def run_two_steps(linear):
        """Two steps with the optimizer's step-start all-gather hook between them."""
        recorded = []
        for step in range(2):
            # What _post_param_sync does after every param all-gather, i.e. before the
            # forward and with the rowwise data just refreshed.
            post_all_gather_processing([linear.weight])
            copy_after_all_gather = linear.weight._columnwise_data is not None
            x = inputs[step].detach().clone().requires_grad_(True)
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                y = linear(x)
                (y.float() ** 2).sum().backward()
            recorded.append(
                {
                    "copy_after_all_gather": copy_after_all_gather,
                    "copy_after_backward": linear.weight._columnwise_data is not None,
                    "y": y.detach().clone(),
                    "dx": x.grad.detach().clone(),
                    "dw": linear.weight.grad.dequantize().clone(),
                }
            )
            linear.weight.grad = None
        return recorded

    # Flag off first: installing is what turns the transient contract on.
    plain_steps = run_two_steps(plain)
    install_fp8_columnwise_free_hooks(transient)
    transient_steps = run_two_steps(transient)

    for step, (plain_step, transient_step) in enumerate(zip(plain_steps, transient_steps)):
        assert plain_step["copy_after_all_gather"] is True
        assert plain_step["copy_after_backward"] is True
        assert transient_step["copy_after_all_gather"] is False, f"step {step}"
        assert transient_step["copy_after_backward"] is False, f"step {step}"
        assert torch.equal(transient_step["y"], plain_step["y"]), f"forward, step {step}"
        assert torch.equal(transient_step["dx"], plain_step["dx"]), f"dgrad, step {step}"
        assert torch.equal(transient_step["dw"], plain_step["dw"]), f"wgrad, step {step}"


# --------------------------------------------------------------------------------------
# 5. Config flag: default off, and meaningless without fp8_param
# --------------------------------------------------------------------------------------


def test_flag_on_without_an_install_raises():
    """A flag that is on while nothing installed the hooks must fail loudly, not no-op.

    This is the guard that keeps a block class without an install call (which would make
    the flag save nothing at all) from looking like a clean negative measurement.
    """
    _base_config(fp8="hybrid", fp8_param=True, fp8_param_free_columnwise_after_backward=True)
    assert fp8_utils._FP8_COLUMN_WISE_EXPECTED is True
    assert is_fp8_columnwise_free_installed() is False

    with pytest.raises(RuntimeError, match="install_fp8_columnwise_free_hooks"):
        post_all_gather_processing([])
    with pytest.raises(RuntimeError, match="install_fp8_columnwise_free_hooks"):
        quantize_param_shard([], [], [], None)


@requires_te_cuda
def test_install_satisfies_the_guard():
    module = _QuantizedWeightModule(seed=0).cuda()
    assert install_fp8_columnwise_free_hooks(module) == 1
    assert is_fp8_columnwise_free_installed() is True
    # The per-step path no longer complains once a block class installed the hooks.
    post_all_gather_processing([module.weight])
    quantize_param_shard([], [], [], None)


def test_flag_defaults_to_false():
    assert _base_config().fp8_param_free_columnwise_after_backward is False


def test_flag_requires_fp8_param():
    with pytest.raises(ValueError, match="fp8_param_free_columnwise_after_backward"):
        _base_config(fp8_param_free_columnwise_after_backward=True)
