# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Static guards on where the fine-grained activation-offload chunk handler is installed.

The per-microbatch chunk handler belongs to the model: ``GPTModel.forward`` and
``HybridModel.forward`` call ``preprocess_for_fine_grained_offloading``, which installs it with
the model chunk's own virtual-pipeline stage. ``init_model_chunk_offload_handler`` appends a
``ChunkOffloadHandler`` on every call and flushes on the last virtual stage, so a second install
from the schedule doubles the handlers and the flushes.

This module deliberately imports nothing from ``megatron``: it reads the schedule's source as
text. That keeps the guard runnable anywhere (no torch, no GPU) and lets it ask the question the
runtime tests cannot ask cheaply -- *can any helper reachable from ``forward_step`` call the
installer?* -- which a test that patches one helper by name cannot answer, because re-adding the
install under a different name would still pass it.
"""

import ast
import pathlib

SCHEDULES = (
    pathlib.Path(__file__).resolve().parents[3] / "megatron/core/pipeline_parallel/schedules.py"
)


def _module_functions():
    """Map every module-level function name to the names of the functions it calls."""
    tree = ast.parse(SCHEDULES.read_text())
    calls = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        callees = set()
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                func = inner.func
                if isinstance(func, ast.Name):
                    callees.add(func.id)
                elif isinstance(func, ast.Attribute):
                    callees.add(func.attr)
        calls[node.name] = callees
    return tree, calls


def _reachable(calls, root):
    """Every module-level function name reachable from ``root`` through the call graph."""
    seen = set()
    stack = [root]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(calls.get(name, ()))
    return seen


def test_forward_step_cannot_reach_the_chunk_handler_installer():
    """Nothing ``forward_step`` calls may install a per-microbatch chunk handler.

    A re-added install fails this even when it hides behind a new helper name, because the check
    follows the call graph instead of looking for one particular symbol.
    """
    _, calls = _module_functions()
    reachable = _reachable(calls, "forward_step")

    offenders = sorted(name for name in reachable if "init_chunk_handler" in calls.get(name, ()))
    assert offenders == [], (
        "forward_step reaches a chunk-handler install through: "
        f"{offenders}. The model installs it (GPTModel/HybridModel.forward -> "
        "preprocess_for_fine_grained_offloading); a second install appends a second "
        "ChunkOffloadHandler per microbatch and flushes twice on the last virtual stage."
    )

    assert "init_chunk_handler" not in calls.get(
        "forward_step", set()
    ), "forward_step must not call the installer directly"


def test_forward_step_still_refuses_models_that_do_not_install_a_handler():
    """The guard that replaced the install has to stay wired into ``forward_step``."""
    _, calls = _module_functions()
    reachable = _reachable(calls, "forward_step")
    assert "_require_model_installs_offload_chunk" in reachable, (
        "forward_step no longer calls _require_model_installs_offload_chunk; a model without "
        "preprocess_for_fine_grained_offloading would then accept the feature and offload nothing"
    )


def test_forward_step_carries_the_microbatch_trace_decorator():
    """The microbatch trace belongs on ``forward_step``.

    It has drifted before: an insertion between the decorator and ``def forward_step(`` hands the
    decorator to whatever function was inserted.
    """
    tree, _ = _module_functions()
    traced = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            segment = ast.get_source_segment(SCHEDULES.read_text(), decorator) or ""
            if "_otel_trace_fn" in segment and "microbatch" in segment:
                traced.setdefault(node.name, []).append(segment)

    assert (
        "forward_step" in traced
    ), f"forward_step lost its microbatch trace; traced: {sorted(traced)}"
    forward_traces = [s for s in traced["forward_step"] if "microbatch.forward" in s]
    assert len(forward_traces) == 1, f"expected exactly one forward trace, found {forward_traces}"

    others = [
        name
        for name in traced
        if name != "forward_step" and "microbatch.forward" in " ".join(traced[name])
    ]
    assert others == [], f"the forward trace also decorates: {others}"
