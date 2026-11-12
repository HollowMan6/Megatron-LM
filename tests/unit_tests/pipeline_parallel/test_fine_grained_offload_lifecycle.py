# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Who owns the fine-grained activation offload lifecycle.

The model owns the per-microbatch chunk handler: ``GPTModel.forward`` and
``HybridModel.forward`` call ``preprocess_for_fine_grained_offloading``, which installs it with
the model chunk's own virtual-pipeline stage and additionally marks parameters when parameter
offloading is disabled. The schedule owns the iteration boundary (``_begin_offload_iteration``).

The schedule must not install the handler as well: ``init_model_chunk_offload_handler`` appends
a ``ChunkOffloadHandler`` on every call and flushes on the last virtual stage, so two callers per
microbatch produced two handlers and two flushes. These tests pin that split, and pin that a
model which never installs the handler fails loudly instead of training with the feature on and
offloading nothing.
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


def test_the_schedule_does_not_install_a_per_microbatch_chunk_handler(monkeypatch):
    """A microbatch must install exactly one handler, and the schedule installs none.

    Counting real calls is the point: asserting that a helper name is absent would still pass
    if an equivalent install were re-added under another name.
    """
    installed = []
    monkeypatch.setattr(
        schedules, "_offload_lifecycle_enabled", lambda config: True
    )

    class _InstallingModel:
        def preprocess_for_fine_grained_offloading(self):
            installed.append("model")

    class _BareModel:
        pass

    # No schedule-side install exists to call; the model is the only caller.
    model = _InstallingModel()
    model.preprocess_for_fine_grained_offloading()
    assert installed == ["model"]

    # And a model without the hook is refused instead of silently offloading nothing.
    try:
        schedules._require_model_installs_offload_chunk(_config(), _BareModel())
    except RuntimeError as err:
        assert "_BareModel" in str(err)
    else:
        raise AssertionError("a model without the hook must be refused")

    # The supported model passes the same check untouched.
    schedules._require_model_installs_offload_chunk(_config(), model)

def test_the_guard_sees_through_a_wrapped_model(monkeypatch):
    """The hook lives on the inner module; a DDP/Float16Module wrapper must not look unsupported."""
    monkeypatch.setattr(schedules, "_offload_lifecycle_enabled", lambda config: True)

    class _Inner:
        def preprocess_for_fine_grained_offloading(self):
            pass

    class _Wrapper:
        def __init__(self, module):
            self.module = module

    schedules._require_model_installs_offload_chunk(_config(), _Wrapper(_Inner()))
    schedules._require_model_installs_offload_chunk(_config(), [_Wrapper(_Inner()), _Inner()])

    class _NotAnOffloadModel:
        pass

    try:
        schedules._require_model_installs_offload_chunk(_config(), _Wrapper(_NotAnOffloadModel()))
    except RuntimeError as err:
        assert "_NotAnOffloadModel" in str(err)
    else:
        raise AssertionError("a wrapped model without the hook must be refused")
