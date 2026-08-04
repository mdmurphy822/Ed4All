"""Run-scoped training-synthesis progress snapshot tests."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from Trainforge.synthesis.synthesis_progress import (
    SynthesisProgressWriter,
    resolve_progress_path,
)


def _isolate(monkeypatch, tmp_path: Path, run_id: str = "WF-TEST") -> Path:
    runs = tmp_path / "runs"
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.setenv("ED4ALL_RUN_ID", run_id)
    return runs / run_id / "telemetry" / "training_synthesis.json"


def test_atomic_snapshot_has_authoritative_fields(monkeypatch, tmp_path):
    path = _isolate(monkeypatch, tmp_path)
    writer = SynthesisProgressWriter(
        run_id=None,
        total_units=100,
        max_concurrent=32,
        provider="local",
        model="teacher-model",
    )
    writer.update(
        completed_units=20,
        terminal_units=20,
        accepted_count=30,
        rejected_count=5,
        transient_count=2,
        sft_count=18,
        dpo_count=12,
        provider_results=40,
        active_workers=0,
        queued_units=32,
        in_flight=32,
        rejection_reasons={"claim_support:unsupported": 5},
        checkpointed=True,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["state"] == "running"
    assert payload["completed_units"] == 20
    assert payload["terminal_units"] == 20
    assert payload["accepted_count"] == 30
    assert payload["rejected_count"] == 5
    assert payload["sft_count"] == 18
    assert payload["dpo_count"] == 12
    assert payload["max_concurrent"] == 32
    assert payload["backpressure"] is True
    assert payload["checkpoint_timestamp"]
    assert payload["eta_seconds"] is not None
    assert "fresh_start_id" not in payload
    assert "marker_digest" not in payload
    assert not list(path.parent.glob("*.tmp"))


def test_resume_preserves_prior_terminal_counts_and_started_at(
    monkeypatch, tmp_path
):
    path = _isolate(monkeypatch, tmp_path)
    first = SynthesisProgressWriter(
        run_id=None,
        total_units=100,
        max_concurrent=32,
        provider="local",
        model="teacher-model",
    )
    first.update(
        completed_units=16,
        terminal_units=16,
        accepted_count=12,
        rejected_count=4,
        sft_count=7,
        dpo_count=5,
        provider_results=18,
        cached_replays=0,
        checkpointed=True,
    )
    prior = json.loads(path.read_text(encoding="utf-8"))

    resumed = SynthesisProgressWriter(
        run_id=None,
        total_units=100,
        max_concurrent=48,
        provider="local",
        model="teacher-model",
    )
    current = resumed.payload
    assert current["started_at"] == prior["started_at"]
    assert current["completed_units"] == 16
    assert current["terminal_units"] == 16
    assert current["accepted_count"] == 12
    assert current["rejected_count"] == 4
    assert current["provider_results"] == 18
    assert current["max_concurrent"] == 48
    resumed.update(
        completed_units=1,
        terminal_units=2,
        accepted_count=1,
        rejected_count=1,
        provider_results=20,
        cached_replays=7,
    )
    assert resumed.payload["completed_units"] == 16
    assert resumed.payload["terminal_units"] == 16
    assert resumed.payload["provider_results"] == 20
    assert resumed.payload["cached_replays"] == 7

    third = SynthesisProgressWriter(
        run_id=None,
        total_units=100,
        max_concurrent=48,
        provider="local",
        model="teacher-model",
    )
    third.update(
        completed_units=3,
        provider_results=24,
        cached_replays=11,
    )
    assert third.payload["completed_units"] == 16
    assert third.payload["provider_results"] == 24
    assert third.payload["cached_replays"] == 18


def test_malformed_or_wrong_run_snapshot_is_not_reused(monkeypatch, tmp_path):
    path = _isolate(monkeypatch, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    writer = SynthesisProgressWriter(
        run_id=None,
        total_units=5,
        max_concurrent=1,
        provider="mock",
        model="deterministic",
    )
    assert writer.payload["completed_units"] == 0

    payload = writer.payload
    payload["run_id"] = "ANOTHER-RUN"
    path.write_text(json.dumps(payload), encoding="utf-8")
    writer2 = SynthesisProgressWriter(
        run_id=None,
        total_units=5,
        max_concurrent=1,
        provider="mock",
        model="deterministic",
    )
    assert writer2.payload["completed_units"] == 0


def test_fresh_identity_same_id_resume_preserves_prior_counts(
    monkeypatch, tmp_path
):
    path = _isolate(monkeypatch, tmp_path)
    identity = {
        "fresh_start_id": "fresh-attempt-0123456789",
        "marker_digest": "a" * 64,
    }
    first = SynthesisProgressWriter(
        run_id=None,
        total_units=12,
        max_concurrent=4,
        provider="local",
        model="teacher-model",
        **identity,
    )
    first.update(
        completed_units=5,
        terminal_units=5,
        accepted_count=3,
        rejected_count=2,
        checkpointed=True,
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["fresh_start_id"] == identity["fresh_start_id"]
    assert persisted["marker_digest"] == identity["marker_digest"]

    resumed = SynthesisProgressWriter(
        run_id=None,
        total_units=12,
        max_concurrent=8,
        provider="local",
        model="teacher-model",
        **identity,
    )
    assert resumed.payload["completed_units"] == 5
    assert resumed.payload["terminal_units"] == 5
    assert resumed.payload["accepted_count"] == 3
    assert resumed.payload["rejected_count"] == 2
    assert resumed.payload["fresh_start_id"] == identity["fresh_start_id"]
    assert resumed.payload["marker_digest"] == identity["marker_digest"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.pop("fresh_start_id"),
            "missing its fresh-start identity",
        ),
        (
            lambda payload: payload.__setitem__(
                "fresh_start_id", "different-fresh-attempt"
            ),
            "does not match the current synthesis attempt",
        ),
        (
            lambda payload: payload.__setitem__("marker_digest", "b" * 64),
            "does not match the current synthesis attempt",
        ),
    ],
)
def test_fresh_identity_rejects_missing_or_mismatched_prior_snapshot(
    monkeypatch, tmp_path, mutation, message
):
    path = _isolate(monkeypatch, tmp_path)
    identity = {
        "fresh_start_id": "fresh-attempt-0123456789",
        "marker_digest": "a" * 64,
    }
    writer = SynthesisProgressWriter(
        run_id=None,
        total_units=12,
        max_concurrent=4,
        provider="local",
        model="teacher-model",
        **identity,
    )
    writer.update(completed_units=5, terminal_units=5)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        SynthesisProgressWriter(
            run_id=None,
            total_units=12,
            max_concurrent=4,
            provider="local",
            model="teacher-model",
            **identity,
        )


def test_concurrent_updates_always_leave_valid_json(monkeypatch, tmp_path):
    path = _isolate(monkeypatch, tmp_path)
    writer = SynthesisProgressWriter(
        run_id=None,
        total_units=48,
        max_concurrent=48,
        provider="local",
        model="teacher-model",
    )

    threads = [
        threading.Thread(
            target=writer.update,
            kwargs={
                "completed_units": index + 1,
                "active_workers": 16,
                "queued_units": 16,
                "in_flight": 32,
            },
        )
        for index in range(16)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert 1 <= payload["completed_units"] <= 16
    assert payload["active_workers"] == 16
    assert payload["queued_units"] == 16
    assert payload["in_flight"] == 32
    assert not list(path.parent.glob("*.tmp"))


def test_paused_and_complete_snapshots_have_no_eta(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    writer = SynthesisProgressWriter(
        run_id=None,
        total_units=10,
        max_concurrent=32,
        provider="local",
        model="teacher-model",
    )
    writer.update(
        state="paused",
        completed_units=3,
        stop_requested=True,
        gate_readiness="pending",
        checkpointed=True,
    )
    assert writer.payload["state"] == "paused"
    assert writer.payload["eta_seconds"] is None
    assert writer.payload["stop_requested"] is True

    writer.update(
        state="complete",
        completed_units=10,
        stop_requested=False,
        gate_readiness="ready",
    )
    assert writer.payload["state"] == "complete"
    assert writer.payload["eta_seconds"] is None
    assert writer.payload["gate_readiness"] == "ready"


def test_no_run_id_means_no_filesystem_side_effect(monkeypatch, tmp_path):
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    writer = SynthesisProgressWriter(
        run_id=None,
        total_units=1,
        max_concurrent=1,
        provider="mock",
        model="deterministic",
    )
    writer.update(completed_units=1, state="complete")
    assert resolve_progress_path() is None
    assert not (tmp_path / "runs").exists()
