#!/usr/bin/env python3
"""Analyze a CUDA allocator snapshot or a recorded memory-history snapshot.

``torch.cuda.memory_snapshot()`` produces a JSON-friendly live-state snapshot,
whereas ``torch.cuda.memory._dump_snapshot()`` writes a pickle containing
``device_traces``. The latter is the useful format for finding a transient
peak: this tool replays allocation events and reports the active allocations at
the largest point in that timeline.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any


def _frame_location(frame: Any) -> str:
    if isinstance(frame, dict):
        filename = frame.get("filename") or frame.get("file") or "<unknown>"
        line = frame.get("line") or frame.get("lineno") or "?"
        return f"{filename}:{line}"
    if isinstance(frame, str):
        return frame.strip() or "<unknown>"
    return str(frame) if frame is not None else "<unknown>"


def _frames(entry: dict[str, Any]) -> list[Any]:
    value = entry.get("frames") or entry.get("trace") or []
    if isinstance(value, str):
        return value.splitlines() if value else []
    return value if isinstance(value, list) else []


def _trace_key(entry: dict[str, Any]) -> tuple[str, ...]:
    """Return a stable full-stack key while retaining a useful display location."""
    return tuple(_frame_location(frame) for frame in _frames(entry)) or ("<unknown>",)


def _display_trace(trace: tuple[str, ...]) -> str:
    return " <- ".join(trace) if trace else "<unknown>"


def _event_size(event: dict[str, Any]) -> int:
    value = event.get("size", event.get("requested_size", 0))
    return value if isinstance(value, int) and value >= 0 else 0


def _live_blocks(snapshot: Any):
    """Yield live allocator blocks from the legacy segment snapshot format."""
    if isinstance(snapshot, dict):
        direct = snapshot.get("blocks")
        if isinstance(direct, list):
            yield from (block for block in direct if isinstance(block, dict))
            return
        segments = snapshot.get("segments")
        if isinstance(segments, list):
            for segment in segments:
                yield from _live_blocks(segment)
        return
    if isinstance(snapshot, list):
        for segment in snapshot:
            yield from _live_blocks(segment)


def aggregate(snapshot: Any) -> dict[str, int]:
    """Aggregate active blocks in a legacy live-state snapshot by location."""
    totals: dict[str, int] = defaultdict(int)
    for block in _live_blocks(snapshot):
        if block.get("state") not in (None, "active_allocated"):
            continue
        size = block.get("requested_size", block.get("size", 0))
        if isinstance(size, int):
            totals[_display_trace(_trace_key(block))] += size
    return dict(totals)


def _trace_events(snapshot: Any) -> list[list[dict[str, Any]]]:
    traces = snapshot.get("device_traces") if isinstance(snapshot, dict) else None
    if not isinstance(traces, list):
        return []
    return [
        [event for event in trace if isinstance(event, dict)]
        for trace in traces
        if isinstance(trace, list)
    ]


def _initial_active(snapshot: Any, device: int):
    """Recover allocations that were live when recording began.

    History starts in the middle of a process, so allocations that survive the
    step may have no alloc event in ``device_traces``.  The final segment state
    is the only available source for those blocks.  Do not seed an address that
    appears in an alloc event: it may have been freed and reused during the
    step, and seeding it would double-count the later allocation.
    """
    if not isinstance(snapshot, dict):
        return {}
    active: dict[int, tuple[int, tuple[str, ...]]] = {}
    start_blocks = snapshot.get("history_start_blocks")
    blocks: list[Any] = []
    if isinstance(start_blocks, list):
        blocks = start_blocks
    elif "history_start_blocks" in snapshot:
        return {}
    elif isinstance(snapshot.get("segments"), list):
        for segment in snapshot["segments"]:
            if not isinstance(segment, dict) or segment.get("device", device) != device:
                continue
            for block in segment.get("blocks", []):
                if isinstance(block, dict):
                    blocks.append(block)
    for block in blocks:
        if not isinstance(block, dict) or block.get("device", device) != device:
            continue
        if start_blocks is None and block.get("state") not in (
            "active_allocated",
            "active_awaiting_free",
        ):
            continue
        address = block.get("address")
        if not isinstance(address, int):
            continue
        active[address] = (_event_size(block), _trace_key(block))
    return active


def _history_peak(snapshot: Any) -> dict[str, Any] | None:
    """Replay alloc/free events and return the highest active alloc point."""
    peak: dict[str, Any] | None = None
    for device, events in enumerate(_trace_events(snapshot)):
        active = _initial_active(snapshot, device)
        active_bytes = sum(size for size, _ in active.values())
        if active_bytes and (peak is None or active_bytes > peak["active_bytes"]):
            peak = {
                "active_bytes": active_bytes,
                "device": device,
                "event_index": -1,
                "active": dict(active),
            }
        for index, event in enumerate(events):
            action = event.get("action")
            address = event.get("addr")
            if action == "alloc" and isinstance(address, int):
                size = _event_size(event)
                # Replacing an address is defensive; normal allocator traces
                # free an address before reusing it.
                previous = active.pop(address, None)
                if previous is not None:
                    active_bytes -= previous[0]
                active[address] = (size, _trace_key(event))
                active_bytes += size
                if peak is None or active_bytes > peak["active_bytes"]:
                    peak = {
                        "active_bytes": active_bytes,
                        "device": device,
                        "event_index": index,
                        "active": dict(active),
                    }
            elif action == "free_completed" and isinstance(address, int):
                previous = active.pop(address, None)
                if previous is not None:
                    active_bytes -= previous[0]
    return peak


def _aggregate_peak(peak: dict[str, Any]) -> dict[tuple[str, ...], int]:
    totals: dict[tuple[str, ...], int] = defaultdict(int)
    for size, trace in peak["active"].values():
        totals[trace] += size
    return dict(totals)


def _largest_request(snapshot: Any) -> dict[str, Any] | None:
    largest: dict[str, Any] | None = None
    for events in _trace_events(snapshot):
        for event in events:
            if event.get("action") not in ("alloc", "oom"):
                continue
            size = _event_size(event)
            if largest is None or size > largest["size"]:
                largest = {
                    "size": size,
                    "action": event.get("action"),
                    "trace": _trace_key(event),
                    "device_free": event.get("device_free"),
                }
    return largest


def _load_snapshot(path: Path) -> Any:
    with path.open("rb") as input_file:
        prefix = input_file.read(1)
        input_file.seek(0)
        if prefix in (b"{", b"["):
            return json.load(input_file)
        return pickle.load(input_file)


def _print_size(size: int) -> str:
    return f"{size} bytes ({size / (1024**3):.3f} GiB)"


def _history_warnings(snapshot: Any) -> list[str]:
    if not isinstance(snapshot, dict):
        return []
    warnings = []
    if "history_start_blocks" in snapshot and not isinstance(
        snapshot.get("history_start_blocks"), list
    ):
        warnings.append(
            "Recording-start allocations are unavailable; the transient peak may be undercounted."
        )
    max_entries = snapshot.get("history_max_entries")
    if isinstance(max_entries, int) and max_entries > 0:
        if any(len(events) >= max_entries for events in _trace_events(snapshot)):
            warnings.append(
                f"At least one device trace reached max_entries={max_entries}; "
                "older events may have been overwritten and the peak may be inaccurate."
            )
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    snapshot = _load_snapshot(args.snapshot)

    for warning in _history_warnings(snapshot):
        print(f"Warning: {warning}")

    peak = _history_peak(snapshot)
    largest = _largest_request(snapshot)
    if peak is not None:
        print(
            f"Peak active allocation: {_print_size(peak['active_bytes'])} "
            f"(device={peak['device']} event={peak['event_index']})"
        )
        print("Top active allocation stack traces at peak:")
        totals = _aggregate_peak(peak)
        for trace, size in sorted(
            totals.items(), key=lambda item: item[1], reverse=True
        )[: args.top]:
            print(f"{_print_size(size)} | {_display_trace(trace)}")
        print()
    else:
        totals = aggregate(snapshot)
        print("Live snapshot (no allocation history):")
        for frame, size in sorted(
            totals.items(), key=lambda item: item[1], reverse=True
        )[: args.top]:
            print(f"{_print_size(size)} | {frame}")

    if largest is not None:
        print(
            f"Largest allocation request: {_print_size(largest['size'])} "
            f"(action={largest['action']}) | {_display_trace(largest['trace'])}"
        )
        if largest["device_free"] is not None:
            print(f"Device free at OOM: {_print_size(largest['device_free'])}")


if __name__ == "__main__":
    main()
