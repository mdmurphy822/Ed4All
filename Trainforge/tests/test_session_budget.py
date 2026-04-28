"""Wave 110 / Phase D — _BudgetTracker tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from Trainforge.generators._session_budget import (
    SynthesisBudgetExceeded,
    _BudgetTracker,
)


def test_records_dispatched_and_cached_calls(tmp_path: Path) -> None:
    tel = tmp_path / "telemetry.jsonl"
    bt = _BudgetTracker(telemetry_path=tel)
    bt.record(kind="instruction", chunk_id="c1", cached=False, elapsed_seconds=0.5)
    bt.record(kind="instruction", chunk_id="c2", cached=True, elapsed_seconds=0.0)
    assert bt.dispatched == 1
    assert bt.cache_hits == 1
    assert bt.total_calls == 2
    rows = [json.loads(l) for l in tel.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    assert rows[0]["cached"] is False
    assert rows[1]["cached"] is True


def test_max_dispatches_cap_raises(tmp_path: Path) -> None:
    bt = _BudgetTracker(max_dispatches=2)
    bt.record(kind="instruction", chunk_id="c1", cached=False, elapsed_seconds=0.1)
    bt.record(kind="preference", chunk_id="c1", cached=False, elapsed_seconds=0.1)
    with pytest.raises(SynthesisBudgetExceeded) as ei:
        bt.record(kind="instruction", chunk_id="c2", cached=False, elapsed_seconds=0.1)
    assert "max_dispatches=2" in str(ei.value)
    assert ei.value.dispatched == 2
    assert ei.value.cache_hits == 0


def test_cache_hits_dont_count_against_cap(tmp_path: Path) -> None:
    bt = _BudgetTracker(max_dispatches=1)
    for _ in range(100):
        bt.record(kind="instruction", chunk_id="c", cached=True, elapsed_seconds=0.0)
    bt.record(kind="instruction", chunk_id="c1", cached=False, elapsed_seconds=0.1)
    with pytest.raises(SynthesisBudgetExceeded):
        bt.record(kind="instruction", chunk_id="c2", cached=False, elapsed_seconds=0.1)


def test_telemetry_path_optional() -> None:
    """No telemetry_path = no on-disk file; counters still work."""
    bt = _BudgetTracker(telemetry_path=None)
    bt.record(kind="instruction", chunk_id="c1", cached=False, elapsed_seconds=0.1)
    assert bt.dispatched == 1


def test_summary_dict_for_reporting() -> None:
    bt = _BudgetTracker(max_dispatches=10)
    for i in range(3):
        bt.record(kind="instruction", chunk_id=f"c{i}", cached=False, elapsed_seconds=0.2)
    summary = bt.summary()
    assert summary["dispatched"] == 3
    assert summary["cache_hits"] == 0
    assert summary["max_dispatches"] == 10
    assert summary["remaining"] == 7
    assert summary["elapsed_seconds_total"] >= 0.6 - 1e-9
