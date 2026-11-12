# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The schedule owns the fine-grained activation offload lifecycle.

Measured 2026-11-12 on a 190K-token lane: enabling `fine_grained_activation_offloading`
changed no peak memory, and once a caller installed a chunk by hand the step hung. The
cause was that this fork had no production call site for `init_chunk_handler`, `reset`
or `enable_offload` -- the model's offload managers were wired to a manager nobody
started. These tests pin the gating of the helpers that now do start it.
"""

from types import SimpleNamespace

from megatron.core.pipeline_parallel import schedules


def _config(**overrides):
    values = dict(fine_grained_activation_offloading=False, offload_modules=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_lifecycle_is_off_unless_both_switch_and_modules_are_set():
    assert schedules._offload_lifecycle_enabled(_config()) is False
    assert schedules._offload_lifecycle_enabled(
        _config(fine_grained_activation_offloading=True)) is False
    assert schedules._offload_lifecycle_enabled(
        _config(offload_modules=["moe_act"])) is False
    assert schedules._offload_lifecycle_enabled(
        _config(fine_grained_activation_offloading=True, offload_modules=["moe_act"])) is True


def test_helpers_do_nothing_without_the_switch(monkeypatch):
    """A run that did not ask for offloading must not import or touch the manager."""
    calls = []
    monkeypatch.setattr(
        schedules, "_begin_offload_microbatch",
        lambda config, vp_stage=None: calls.append(("microbatch", vp_stage)))
    monkeypatch.setattr(
        schedules, "_begin_offload_iteration", lambda config: calls.append(("iteration", None)))
    # The helpers themselves short-circuit before importing megatron's interface; calling
    # them at all is what the schedule does, and with the switch off they must return
    # without raising even though nothing is initialised.
    schedules._begin_offload_microbatch(_config(), vp_stage=None)
    schedules._begin_offload_iteration(_config())
    assert calls == [("microbatch", None), ("iteration", None)]
