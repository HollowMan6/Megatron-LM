"""Opt-in CUDA memory accounting for DeepSeek-V4.1 training investigations.

``DSV41_MEM_PROFILE=1`` enables the rank-zero timeline and the step-end component
walk. The component walk visits model parameters and optimizer state, so its cost is
proportional to the number of tensors and is intentionally absent by default.
``DSV41_MEM_SNAPSHOT`` adds the more expensive allocator snapshot write at step end.
``DSV41_MEM_HISTORY`` enables the allocator's stack-trace event recorder for one
training step.  It is intended for diagnosis only: the recorder adds per-event
metadata and can make allocation-heavy steps slower.
"""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import torch

_GB = 1024**3
_ENABLED = os.environ.get("DSV41_MEM_PROFILE") == "1"
_PHASE_INDEX = 0
# A diagnostic step can contain hundreds of thousands of allocator events; one
# million entries leaves headroom without the unbounded growth of ``None``.
_HISTORY_MAX_ENTRIES = 1_000_000
_HISTORY_ACTIVE = False
_HISTORY_START_BLOCKS: list[dict[str, Any]] | None = None
_HISTORY_START_ERROR: str | None = None


def enabled() -> bool:
    """Return whether the low-overhead memory probe is enabled."""
    return _ENABLED


def parts_enabled() -> bool:
    """Return whether the expensive parameter/state walk is enabled."""
    return os.environ.get("DSV41_MEM_PARTS", "1") == "1"


def _rank_zero(cp_group: Any | None = None) -> bool:
    """Return whether this process is global rank zero and CP rank zero."""
    if cp_group is not None:
        try:
            cp_rank = cp_group.rank()
        except AttributeError:
            cp_rank = torch.distributed.get_rank(cp_group)
        if cp_rank != 0:
            return False
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _history_path() -> Path | None:
    path = os.environ.get("DSV41_MEM_HISTORY")
    return Path(path) if path else None


def _capture_history_start_blocks() -> list[dict[str, Any]]:
    """Keep the live allocations that predate this step's event history."""
    snapshot = torch.cuda.memory._snapshot()
    blocks = []
    default_device = torch.cuda.current_device()
    for segment in snapshot.get("segments", []):
        if not isinstance(segment, dict):
            continue
        device = segment.get("device", default_device)
        for block in segment.get("blocks", []):
            if not isinstance(block, dict) or block.get("state") not in (
                "active_allocated",
                "active_awaiting_free",
            ):
                continue
            blocks.append(
                {
                    "device": device,
                    "address": block.get("address"),
                    "size": block.get("requested_size", block.get("size", 0)),
                    "frames": block.get("frames", []),
                }
            )
    return blocks


def _finish_history(path: Path) -> None:
    """Write and disable the active recorder while preserving the original error."""
    global _HISTORY_ACTIVE, _HISTORY_START_BLOCKS, _HISTORY_START_ERROR
    if not _HISTORY_ACTIVE:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(path))
        if _HISTORY_START_BLOCKS is not None:
            with path.open("rb") as input_file:
                snapshot = pickle.load(input_file)
            snapshot["history_max_entries"] = _HISTORY_MAX_ENTRIES
            snapshot["history_start_blocks"] = _HISTORY_START_BLOCKS
            snapshot["history_start_error"] = _HISTORY_START_ERROR
            with path.open("wb") as output:
                pickle.dump(snapshot, output)
        print(f"[mem] history={path}", flush=True)
    except Exception as exc:  # Diagnostics must not change training behavior.
        print(f"[mem] history_error={type(exc).__name__}: {exc}", flush=True)
    finally:
        _HISTORY_ACTIVE = False
        _HISTORY_START_BLOCKS = None
        _HISTORY_START_ERROR = None
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception as exc:  # Diagnostics must not change training behavior.
            print(f"[mem] history_disable_error={type(exc).__name__}: {exc}", flush=True)


def step_start(cp_group: Any | None = None) -> None:
    """Start allocator history recording for the next training step."""
    path = _history_path()
    rank_zero = _rank_zero(cp_group) if cp_group is not None else _rank_zero()
    if path is None or not rank_zero or not torch.cuda.is_available():
        return
    global _HISTORY_ACTIVE, _HISTORY_START_BLOCKS, _HISTORY_START_ERROR
    try:
        torch.cuda.memory._record_memory_history(max_entries=_HISTORY_MAX_ENTRIES)
        _HISTORY_ACTIVE = True
        try:
            _HISTORY_START_BLOCKS = _capture_history_start_blocks()
            _HISTORY_START_ERROR = None
        except Exception as exc:  # Diagnostics must not change training behavior.
            _HISTORY_START_BLOCKS = None
            _HISTORY_START_ERROR = f"{type(exc).__name__}: {exc}"
            print(f"[mem] history_start_error={type(exc).__name__}: {exc}", flush=True)
        print(f"[mem] history_start={path} max_entries={_HISTORY_MAX_ENTRIES}", flush=True)
    except Exception as exc:  # Diagnostics must not change training behavior.
        print(f"[mem] history_error={type(exc).__name__}: {exc}", flush=True)


def step_abort(cp_group: Any | None = None) -> None:
    """Dump history when a training step raises before its normal end hook."""
    path = _history_path()
    rank_zero = _rank_zero(cp_group) if cp_group is not None else _rank_zero()
    if path is None or not rank_zero or not torch.cuda.is_available():
        return
    _finish_history(path)


def phase(name: str) -> None:
    """Print one rank-zero memory timeline sample without synchronizing CUDA."""
    global _PHASE_INDEX
    if not _ENABLED or not _rank_zero() or not torch.cuda.is_available():
        return
    _PHASE_INDEX += 1
    alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(
        f"[mem] phase={name} alloc={alloc / _GB:.3f}GB peak={peak / _GB:.3f}GB "
        f"reserved={reserved / _GB:.3f}GB",
        flush=True,
    )


def _tensor_bytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        return 0
    return value.numel() * value.element_size()


def _iter_models(model: Any) -> Iterable[Any]:
    if isinstance(model, list | tuple):
        for item in model:
            yield from _iter_models(item)
    elif model is not None:
        yield model


def _iter_parameters(model: Any) -> Iterable[Any]:
    for module in _iter_models(model):
        yield from module.parameters()


def _iter_optimizers(optimizer: Any) -> Iterable[Any]:
    """Expand chains before reading properties forwarded to the inner optimizer."""
    pending = [optimizer]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        children = getattr(current, "chained_optimizers", None)
        if isinstance(children, list | tuple):
            pending.extend(children)
        else:
            yield current


def _nested_tensor_bytes(value: Any, seen: set[tuple[Any, ...]]) -> int:
    if isinstance(value, torch.Tensor):
        if not value.is_cuda:
            return 0
        key = (value.device, value.data_ptr(), value.numel(), value.element_size())
        if key in seen:
            return 0
        seen.add(key)
        return _tensor_bytes(value)
    if isinstance(value, Mapping):
        return sum(_nested_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, list | tuple):
        return sum(_nested_tensor_bytes(item, seen) for item in value)
    return 0


def _sum_main_grads(model: Any, optimizer: Any) -> tuple[int | None, str | None]:
    """Use DDP gradients, falling back to optimizer-owned main/shard gradients."""
    total = 0
    seen: set[tuple[Any, ...]] = set()
    missing = False
    try:
        for parameter in _iter_parameters(model):
            if not parameter.requires_grad:
                continue
            if not hasattr(parameter, "main_grad"):
                missing = True
                continue
            total += _nested_tensor_bytes(parameter.main_grad, seen)
        if not missing:
            return total, None

        # DistOpt installs .grad (or precision-aware .decoupled_grad) on the
        # inner optimizer's main parameters; frozen model weights are absent.
        total = 0
        seen.clear()
        found = False
        for current in _iter_optimizers(optimizer):
            if getattr(current, "is_stub_optimizer", False):
                continue
            groups = getattr(current, "param_groups", None)
            if groups is None:
                return (None, "main gradients unavailable: optimizer param_groups unavailable")
            for group in groups:
                for parameter in group["params"]:
                    value = getattr(parameter, "decoupled_grad", None)
                    if value is None:
                        value = getattr(parameter, "grad", None)
                    found = found or value is not None
                    total += _nested_tensor_bytes(value, seen)
                    del value
        if found:
            return total, None
    except Exception as exc:  # Diagnostics must not change training behavior.
        return None, f"main gradients unavailable: {type(exc).__name__}: {exc}"
    return (None, "main gradients unavailable: no main_grad or optimizer gradient tensors")


def _sum_optimizer_states(optimizer: Any) -> tuple[dict[str, int | str] | None, str | None]:
    totals: dict[str, int | str] = {
        "optim_exp_avg": 0,
        "optim_exp_avg_sq": 0,
        "optim_master": 0,
        "optim_other": 0,
    }
    observed = {name: False for name in totals}
    seen: set[tuple[Any, ...]] = set()
    found = False
    try:
        for current in _iter_optimizers(optimizer):
            if getattr(current, "is_stub_optimizer", False):
                continue
            state = getattr(current, "state", None)
            if state is None:
                inner = getattr(current, "optimizer", None)
                state = getattr(inner, "state", None)
            if not isinstance(state, Mapping):
                return (None, "optimizer state unavailable: optimizer exposes no state mapping")
            found = True
            for values in state.values():
                if not isinstance(values, Mapping):
                    totals["optim_other"] += _nested_tensor_bytes(values, seen)
                    continue
                for key, value in values.items():
                    if key == "exp_avg":
                        bucket = "optim_exp_avg"
                    elif key == "exp_avg_sq":
                        bucket = "optim_exp_avg_sq"
                    elif key in ("master_param", "fp32_param", "main_param"):
                        bucket = "optim_master"
                    else:
                        bucket = "optim_other"
                    totals[bucket] += _nested_tensor_bytes(value, seen)
                    observed[bucket] = True
                    del value
            # Ordinary mixed precision stores masters outside state; precision-
            # aware FusedAdam keeps them under the master_param state key.
            for name in ("shard_fp32_from_float16_groups", "fp32_from_float16_groups"):
                master = getattr(current, name, None)
                if master is not None:
                    totals["optim_master"] += _nested_tensor_bytes(master, seen)
                    observed["optim_master"] = True
                    del master
    except Exception as exc:  # Diagnostics must not change training behavior.
        return None, f"optimizer state unavailable: {type(exc).__name__}: {exc}"
    if not found:
        return None, "optimizer exposes no state mapping"
    for bucket in ("optim_exp_avg", "optim_exp_avg_sq", "optim_master"):
        if not observed[bucket]:
            totals[bucket] = "n/a"
    return totals, None


def _account_components(model: Any, optimizer: Any) -> tuple[dict[str, int | str], str | None]:
    parts: dict[str, int | str] = {"params": 0, "grads": 0}
    try:
        for param in _iter_parameters(model):
            parts["params"] += _tensor_bytes(param)
            parts["grads"] += _tensor_bytes(param.grad)
    except Exception as exc:
        return {
            key: "n/a"
            for key in (
                "params",
                "grads",
                "main_grads",
                "optim_exp_avg",
                "optim_exp_avg_sq",
                "optim_master",
                "optim_other",
            )
        }, f"model traversal failed: {exc}"
    main_grads, main_reason = _sum_main_grads(model, optimizer)
    parts["main_grads"] = main_grads if main_grads is not None else "n/a"
    totals, state_reason = _sum_optimizer_states(optimizer)
    parts.update(
        totals
        if totals is not None
        else {
            key: "n/a"
            for key in ("optim_exp_avg", "optim_exp_avg_sq", "optim_master", "optim_other")
        }
    )
    reason = "; ".join(value for value in (main_reason, state_reason) if value)
    return parts, reason or None


def step_end(
    model: Any, optimizer: Any, step: int | None = None, cp_group: Any | None = None
) -> None:
    """Print component accounting and optionally save CUDA allocator snapshots."""
    history_path = _history_path()
    if not _ENABLED and history_path is None:
        return
    rank_zero = _rank_zero(cp_group) if cp_group is not None else _rank_zero()
    if not rank_zero or not torch.cuda.is_available():
        return
    if history_path is not None:
        _finish_history(history_path)

    if not _ENABLED:
        return

    phase("step_end")
    if parts_enabled():
        parts, reason = _account_components(model, optimizer)
        allocated = torch.cuda.memory_allocated()
        if reason is None and all(isinstance(value, int) for value in parts.values()):
            accounted = sum(parts.values())
            parts["other"] = allocated - accounted
        else:
            parts["other"] = "n/a"
        fields = (
            "params",
            "grads",
            "main_grads",
            "optim_exp_avg",
            "optim_exp_avg_sq",
            "optim_master",
            "optim_other",
            "other",
        )
        rendered = " ".join(
            (
                f"{field}={parts[field] / _GB:.3f}GB"
                if isinstance(parts[field], int)
                else f"{field}=n/a"
            )
            for field in fields
        )
        suffix = f" reason={reason}" if reason else ""
        print(f"[mem-parts] {rendered}{suffix}", flush=True)
    else:
        print("[mem-parts] skipped=DSV41_MEM_PARTS=0", flush=True)

    snapshot_path = os.environ.get("DSV41_MEM_SNAPSHOT")
    if snapshot_path:
        try:
            # The allocator snapshot is made of JSON-compatible records. JSON keeps
            # the artifact readable without importing CUDA in the analyzer.
            path = Path(snapshot_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            snapshot = torch.cuda.memory_snapshot()
            with path.open("w", encoding="utf-8") as output:
                json.dump(snapshot, output)
            del snapshot
            print(f"[mem] snapshot={path}", flush=True)
        except Exception as exc:  # Snapshot diagnostics must not change training behavior.
            print(f"[mem] snapshot_error={type(exc).__name__}: {exc}", flush=True)
