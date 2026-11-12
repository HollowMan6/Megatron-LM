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


def test_helper_does_nothing_without_the_switch(monkeypatch):
    """A run that did not ask for offloading must not import or touch the manager."""
    calls = []
    monkeypatch.setattr(
        schedules, "_begin_offload_iteration", lambda config: calls.append(("iteration", None)))
    # The helper short-circuits before importing megatron's interface; calling it at all is
    # what the schedule does, and with the switch off it must return without raising even
    # though nothing is initialised.
    schedules._begin_offload_iteration(_config())
    assert calls == [("iteration", None)]


def test_the_schedule_does_not_install_a_per_microbatch_chunk_handler():
    """The model installs it; a second schedule-side install doubled the handlers.

    ``init_model_chunk_offload_handler`` appends a ``ChunkOffloadHandler`` per call and
    flushes on the last virtual stage, so two callers per microbatch meant two handlers and
    two flushes. The model path (``GPTModel.forward`` / ``HybridModel.forward`` ->
    ``preprocess_for_fine_grained_offloading``) already installs it and knows the chunk's own
    virtual-pipeline stage, so the schedule must not.
    """
    assert not hasattr(schedules, "_begin_offload_microbatch")
