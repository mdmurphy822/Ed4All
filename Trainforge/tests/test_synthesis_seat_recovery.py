from __future__ import annotations

import json
from pathlib import Path

import pytest

from Trainforge.synthesis_journal import (
    GenerationJournal,
    load_generation_journal,
)
from Trainforge.synthesize_training import (
    _call_with_seat_recovery,
    _run_generation_unit,
)
from Trainforge.seat_recovery import SynthesisSeatRecoveryCoordinator


_TIMEOUT = (
    "local request failed after 3 transport attempts: timed out"
)


class _Recovered:
    recovery_id = "incident-1"

    def recover(self, exc: BaseException, **kwargs) -> bool:
        return True


class _RecoveryFailed:
    recovery_id = "incident-failed"
    run_id = "run-generic"

    def recover(self, exc: BaseException, **kwargs) -> bool:
        return False


def test_sequential_transport_hang_recovers_and_retries_exact_call() -> None:
    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(_TIMEOUT)
        return {"ok": True}

    assert _call_with_seat_recovery(
        provider,
        recovery_coordinator=_Recovered(),
    ) == {"ok": True}
    assert calls == 2


def test_sequential_second_transport_failure_pauses_after_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.generation.stop_control import GracefulStopRequested

    monkeypatch.setattr(
        "lib.generation.stop_control.request_stop",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(GracefulStopRequested):
        _call_with_seat_recovery(
            lambda: (_ for _ in ()).throw(RuntimeError(_TIMEOUT)),
            recovery_coordinator=_Recovered(),
        )


def test_sequential_healthy_call_is_byte_path_single_call() -> None:
    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        return "healthy"

    assert _call_with_seat_recovery(
        provider,
        recovery_coordinator=_RecoveryFailed(),
    ) == "healthy"
    assert calls == 1


def test_sequential_non_engine_failure_is_not_recovered() -> None:
    with pytest.raises(ValueError, match="invalid content"):
        _call_with_seat_recovery(
            lambda: (_ for _ in ()).throw(ValueError("invalid content")),
            recovery_coordinator=_Recovered(),
        )


def test_engine_recovery_retries_same_semantic_attempt(tmp_path: Path) -> None:
    path = tmp_path / "generation.jsonl"
    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        if calls == 1:
            from Trainforge.generators._synthesis_provider import (
                SynthesisProviderError,
            )
            raise SynthesisProviderError(_TIMEOUT, code="max_retries_exceeded")
        from Trainforge.generators.preference_factory import (
            PreferenceSynthesisResult,
        )
        return PreferenceSynthesisResult(
            pair={"prompt": "p", "chosen": "c", "rejected": "r"},
            quality={"passed": True},
            rationale="Recovered exact unit after engine recycle",
            source="rule_synthesized",
        )

    outcome = _run_generation_unit(
        chunk_id="chunk-generic",
        kind="preference",
        variant_index=0,
        fingerprint="fp",
        generation_cache={},
        journal=GenerationJournal(path),
        call=provider,
        recovery_coordinator=_Recovered(),
    )

    assert calls == 2
    assert outcome.error is None
    latest = load_generation_journal(path)[("chunk-generic", "preference", 0)]
    assert latest["attempt"] == 1
    assert latest["disposition"] == "success"
    assert latest["recovered_engine_incident"] == "incident-1"


def test_concurrent_second_transport_failure_pauses_without_journal_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.generation.stop_control import GracefulStopRequested

    path = tmp_path / "generation.jsonl"
    monkeypatch.setattr(
        "lib.generation.stop_control.request_stop",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(GracefulStopRequested):
        _run_generation_unit(
            chunk_id="chunk-generic",
            kind="instruction",
            variant_index=0,
            fingerprint="fp",
            generation_cache={},
            journal=GenerationJournal(path),
            call=lambda: (_ for _ in ()).throw(RuntimeError(_TIMEOUT)),
            recovery_coordinator=_Recovered(),
        )
    assert load_generation_journal(path) == {}


def test_legacy_fatal_rows_are_never_reclassified_by_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.vllm_container_lifecycle import SeatRecoveryResult

    path = tmp_path / "generation.jsonl"
    journal = GenerationJournal(path)
    for chunk in ("a", "b"):
        journal.append({
            "chunk_id": chunk,
            "kind": "preference",
            "variant_index": 0,
            "fingerprint": f"fp-{chunk}",
            "attempt": 3,
            "disposition": "fatal",
            "was_transient": True,
            "error_type": "SynthesisProviderError",
            "message": _TIMEOUT,
        })
    before = path.read_bytes()
    coordinator = SynthesisSeatRecoveryCoordinator(
        base_url="http://localhost:8000/v1",
        run_dir=None,
        marker_path=tmp_path / "recovery.jsonl",
    )
    monkeypatch.setattr(
        "lib.vllm_container_lifecycle.recover_seat_for_base_url",
        lambda *args, **kwargs: SeatRecoveryResult(
            True, "recovered", "local-seat", True, ("Hang detected",), 9.0
        ),
    )
    assert coordinator.recover(RuntimeError(_TIMEOUT)) is True
    assert path.read_bytes() == before


def test_failed_engine_recovery_pauses_without_spending_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Trainforge.generators._synthesis_provider import SynthesisProviderError
    from lib.generation.stop_control import GracefulStopRequested

    path = tmp_path / "generation.jsonl"
    stop_calls = []
    monkeypatch.setattr(
        "lib.generation.stop_control.request_stop",
        lambda run_id, **kwargs: stop_calls.append((run_id, kwargs)),
    )

    def provider():
        raise SynthesisProviderError(_TIMEOUT, code="max_retries_exceeded")

    with pytest.raises(GracefulStopRequested):
        _run_generation_unit(
            chunk_id="chunk-generic",
            kind="instruction",
            variant_index=0,
            fingerprint="fp",
            generation_cache={},
            journal=GenerationJournal(path),
            call=provider,
            recovery_coordinator=_RecoveryFailed(),
        )

    assert stop_calls == [(
        "run-generic",
        {
            "scope": "run",
            "reason": "seat_unhealthy",
            "source": "synthesis_recovery",
        },
    )]
    assert load_generation_journal(path) == {}


def test_sequential_failed_recovery_pauses_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.generation.stop_control import GracefulStopRequested

    stop_calls = []
    monkeypatch.setattr(
        "lib.generation.stop_control.request_stop",
        lambda run_id, **kwargs: stop_calls.append((run_id, kwargs)),
    )
    with pytest.raises(GracefulStopRequested):
        _call_with_seat_recovery(
            lambda: (_ for _ in ()).throw(RuntimeError(_TIMEOUT)),
            recovery_coordinator=_RecoveryFailed(),
        )
    assert stop_calls and stop_calls[0][0] == "run-generic"


def test_recovery_budget_does_not_blindly_accept_second_incident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.vllm_container_lifecycle import SeatRecoveryResult

    coordinator = SynthesisSeatRecoveryCoordinator(
        base_url="http://localhost:8000/v1",
        run_dir=None,
        marker_path=tmp_path / "recovery.jsonl",
    )
    monkeypatch.setattr(
        "lib.vllm_container_lifecycle.recover_seat_for_base_url",
        lambda *args, **kwargs: SeatRecoveryResult(
            True, "recovered", "local-seat", True, ("Hang detected",), 9.0
        ),
    )
    assert coordinator.recover(RuntimeError(_TIMEOUT)) is True
    monkeypatch.setattr(
        "lib.vllm_container_lifecycle.coherence_probe",
        lambda *args, **kwargs: False,
    )
    assert coordinator.recover(RuntimeError(_TIMEOUT)) is False


def test_marker_failure_releases_waiters_as_failed(tmp_path: Path) -> None:
    coordinator = SynthesisSeatRecoveryCoordinator(
        base_url="http://localhost:8000/v1",
        run_dir=None,
        marker_path=tmp_path / "recovery.jsonl",
    )

    def marker_failure(*args, **kwargs):
        raise OSError("disk unavailable")

    coordinator._append_marker = marker_failure  # type: ignore[method-assign]
    assert coordinator.recover(RuntimeError(_TIMEOUT)) is False
    assert coordinator.recover(RuntimeError(_TIMEOUT)) is False


def test_final_recovery_marker_failure_cannot_publish_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.vllm_container_lifecycle import SeatRecoveryResult

    coordinator = SynthesisSeatRecoveryCoordinator(
        base_url="http://localhost:8000/v1",
        run_dir=None,
        marker_path=tmp_path / "recovery.jsonl",
    )
    monkeypatch.setattr(
        "lib.vllm_container_lifecycle.recover_seat_for_base_url",
        lambda *args, **kwargs: SeatRecoveryResult(
            True, "recovered", "local-seat", True, ("Hang detected",), 9.0
        ),
    )
    real_append = coordinator._append_marker

    def fail_success_marker(state: str, **extra):
        if state == "recovered":
            raise OSError("durable evidence failed")
        return real_append(state, **extra)

    coordinator._append_marker = fail_success_marker  # type: ignore[method-assign]
    assert coordinator.recover(RuntimeError(_TIMEOUT)) is False


def test_crash_resume_keeps_prior_marker_and_records_correlated_incident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.vllm_container_lifecycle import SeatRecoveryResult

    marker = tmp_path / "recovery.jsonl"
    marker.write_text(
        json.dumps({
            "schema_version": 1,
            "recovery_id": "crashed-process",
            "state": "started",
            "run_id": "run-generic",
            "task": {"chunk_id": "old-chunk"},
        }) + "\n",
        encoding="utf-8",
    )
    coordinator = SynthesisSeatRecoveryCoordinator(
        base_url="http://localhost:8000/v1",
        run_dir=tmp_path / "run-generic",
        marker_path=marker,
    )
    monkeypatch.setattr(
        "lib.vllm_container_lifecycle.recover_seat_for_base_url",
        lambda *args, **kwargs: SeatRecoveryResult(
            True, "recovered", "local-seat", True, ("Hang detected",), 9.0
        ),
    )
    context = {
        "chunk_id": "current-chunk",
        "kind": "instruction",
        "variant_index": 0,
        "fingerprint": "fp-current",
    }
    assert coordinator.recover(
        RuntimeError(_TIMEOUT),
        incident_context=context,
    ) is True
    rows = [
        json.loads(line)
        for line in marker.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["recovery_id"] == "crashed-process"
    assert rows[-1]["state"] == "recovered"
    assert rows[-1]["run_id"] == "run-generic"
    assert rows[-1]["seat_name"] == "local-seat"
    assert rows[-1]["task"] == context


def test_crash_started_incident_cannot_recycle_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "recovery.jsonl"
    coordinator = SynthesisSeatRecoveryCoordinator(
        base_url="http://localhost:8000/v1",
        run_dir=tmp_path / "run-generic",
        marker_path=marker,
    )
    context = {
        "chunk_id": "current-chunk",
        "kind": "instruction",
        "variant_index": 0,
        "fingerprint": "fp-current",
    }
    incident_key = coordinator._incident_key(context)
    marker.write_text(
        json.dumps({
            "schema_version": 1,
            "recovery_id": incident_key,
            "incident_key": incident_key,
            "state": "started",
            "run_id": "run-generic",
            "seat_name": "local-seat",
            "task": context,
        }) + "\n",
        encoding="utf-8",
    )
    lifecycle_calls = []
    monkeypatch.setattr(
        "lib.vllm_container_lifecycle.recover_seat_for_base_url",
        lambda *args, **kwargs: lifecycle_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "lib.vllm_container_lifecycle.coherence_probe",
        lambda base_url, **kwargs: False,
    )
    before = marker.read_bytes()
    assert coordinator.recover(
        RuntimeError(_TIMEOUT),
        incident_context=context,
    ) is False
    assert lifecycle_calls == []
    assert marker.read_bytes() == before


def test_healthy_crash_probe_appends_durable_recovery_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "recovery.jsonl"
    coordinator = SynthesisSeatRecoveryCoordinator(
        base_url="http://localhost:8000/v1",
        run_dir=tmp_path / "run-generic",
        marker_path=marker,
    )
    context = {
        "workflow_phase": "training_synthesis",
        "task_id": "chunk-generic:instruction:0",
        "chunk_id": "chunk-generic",
        "kind": "instruction",
        "variant_index": 0,
        "fingerprint": "fp-current",
    }
    incident_key = coordinator._incident_key(context)
    marker.write_text(
        json.dumps({
            "schema_version": 1,
            "recovery_id": incident_key,
            "incident_key": incident_key,
            "state": "started",
            "run_id": "run-generic",
            "seat_name": "local-seat",
            "task": context,
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "lib.vllm_container_lifecycle.coherence_probe",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "lib.vllm_container_lifecycle._seat_for_base_url",
        lambda *args, **kwargs: "local-seat",
    )
    assert coordinator.recover(
        RuntimeError(_TIMEOUT),
        incident_context=context,
    ) is True
    rows = [
        json.loads(line)
        for line in marker.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[-1]["state"] == "recovered_after_crash_probe"
    assert rows[-1]["workflow_phase"] == "training_synthesis"
    assert rows[-1]["task_id"] == "chunk-generic:instruction:0"


def test_canonical_v1_url_is_normalized_for_late_follower_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = SynthesisSeatRecoveryCoordinator(
        base_url="http://localhost:8000/v1",
        run_dir=None,
        marker_path=tmp_path / "recovery.jsonl",
    )
    coordinator._state = "recovered"
    seen = []
    monkeypatch.setattr(
        "lib.vllm_container_lifecycle.coherence_probe",
        lambda base_url, **kwargs: seen.append(base_url) or True,
    )
    assert coordinator.recover(RuntimeError(_TIMEOUT)) is True
    assert seen == ["http://localhost:8000"]


@pytest.mark.parametrize(
    "marker_text",
    [
        '{"schema_version":1,"state":"started"',
        json.dumps({
            "schema_version": 1,
            "incident_key": "wrong-key-is-otherwise-valid",
            "state": "unknown",
        }),
    ],
)
def test_corrupt_recovery_marker_fails_closed_without_recycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_text: str,
) -> None:
    marker = tmp_path / "recovery.jsonl"
    marker.write_text(marker_text, encoding="utf-8")
    coordinator = SynthesisSeatRecoveryCoordinator(
        base_url="http://localhost:8000/v1",
        run_dir=tmp_path / "run-generic",
        marker_path=marker,
    )
    lifecycle_calls = []
    monkeypatch.setattr(
        "lib.vllm_container_lifecycle.recover_seat_for_base_url",
        lambda *args, **kwargs: lifecycle_calls.append((args, kwargs)),
    )
    assert coordinator.recover(
        RuntimeError(_TIMEOUT),
        incident_context={"chunk_id": "current", "kind": "instruction"},
    ) is False
    assert lifecycle_calls == []
