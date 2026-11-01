"""CPU tests for the CUDA memory-history analyzer."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path


def _load_analyzer():
    path = Path(__file__).resolve().parents[3] / "tools/analyze_memory_snapshot.py"
    spec = importlib.util.spec_from_file_location("analyze_memory_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _frame(filename: str, line: int):
    return [{"filename": filename, "line": line, "name": "allocate"}]


def test_history_peak_replays_allocations_and_frees():
    analyzer = _load_analyzer()
    snapshot = {
        "segments": [
            {
                "device": 0,
                "blocks": [
                    {
                        "address": 10,
                        "requested_size": 7,
                        "state": "active_allocated",
                        "frames": _frame("baseline.py", 3),
                    }
                ],
            }
        ],
        "device_traces": [
            [
                {"action": "alloc", "addr": 20, "size": 11, "frames": _frame("cat.py", 42)},
                {"action": "free_requested", "addr": 20, "size": 11},
                {"action": "alloc", "addr": 30, "size": 13, "frames": _frame("cat.py", 42)},
                {"action": "free_completed", "addr": 20, "size": 11},
            ]
        ],
    }

    peak = analyzer._history_peak(snapshot)

    assert peak is not None
    assert peak["active_bytes"] == 31
    assert peak["event_index"] == 2
    assert analyzer._aggregate_peak(peak) == {("baseline.py:3",): 7, ("cat.py:42",): 24}
    assert analyzer._display_trace(("cat.py:42", "torch.py:8")) == "cat.py:42 <- torch.py:8"


def test_largest_request_includes_failed_oom_request():
    analyzer = _load_analyzer()
    snapshot = {
        "device_traces": [
            [
                {"action": "alloc", "addr": 1, "size": 100, "frames": _frame("small.py", 1)},
                {
                    "action": "oom",
                    "size": 6_583_008_256,
                    "device_free": 1_000,
                    "frames": _frame("chunk_tape.py", 99),
                },
            ]
        ]
    }

    largest = analyzer._largest_request(snapshot)

    assert largest == {
        "size": 6_583_008_256,
        "action": "oom",
        "trace": ("chunk_tape.py:99",),
        "device_free": 1_000,
    }


def test_full_event_buffer_warns_that_early_events_may_be_missing():
    analyzer = _load_analyzer()
    snapshot = {"history_max_entries": 2, "device_traces": [[{}, {}]]}

    assert any(
        "older events may have been overwritten" in warning
        for warning in analyzer._history_warnings(snapshot)
    )


def test_history_peak_uses_recording_start_blocks_even_when_they_are_freed():
    analyzer = _load_analyzer()
    snapshot = {
        "history_start_blocks": [
            {"device": 0, "address": 1, "size": 50, "frames": _frame("before_step.py", 7)}
        ],
        "segments": [{"device": 0, "blocks": []}],
        "device_traces": [
            [
                {"action": "free_completed", "addr": 1, "size": 50},
                {"action": "alloc", "addr": 1, "size": 10, "frames": _frame("step.py", 8)},
            ]
        ],
    }

    peak = analyzer._history_peak(snapshot)

    assert peak is not None
    assert peak["active_bytes"] == 50
    assert peak["event_index"] == -1


def test_load_snapshot_accepts_history_pickle_and_legacy_json(tmp_path):
    analyzer = _load_analyzer()
    history = {"device_traces": [[]], "segments": []}
    pickle_path = tmp_path / "history.pickle"
    with pickle_path.open("wb") as output:
        pickle.dump(history, output)
    assert analyzer._load_snapshot(pickle_path) == history

    json_path = tmp_path / "live.json"
    json_path.write_text('{"segments": []}', encoding="utf-8")
    assert analyzer._load_snapshot(json_path) == {"segments": []}
