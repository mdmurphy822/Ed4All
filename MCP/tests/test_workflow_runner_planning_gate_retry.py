"""course_planning gate-failure retry regression tests.

Owner directive 2026-07-17: "a failed entailment should not kill the whole
run; it should gracefully retry (max 10 retries, if 10 fails then log and
loudly warn)."

Pins the ``ED4ALL_PLANNING_GATE_RETRIES`` contract in
``MCP/core/workflow_runner.py`` (``resolve_planning_gate_retries`` +
``WorkflowRunner._retry_course_planning_gates``):

- retries=0 (unset / blank / garbage / negative) → byte-identical legacy
  behavior: ONE dispatch, gate failure stops the workflow (FAILED).
- gate fails N-1 times then passes → phase completes, N synthesis
  dispatches observed, the CLUSTER sidecar
  (``.stage2_clusters_checkpoint.jsonl``) is evicted between attempts while
  the WINDOW sidecar (``.stage2_windows_checkpoint.jsonl``) is kept, and the
  per-attempt ``ED4ALL_PLANNING_REROLL_SALT`` PLUS the failure-directed
  ``ED4ALL_PLANNING_REROLL_FEEDBACK`` remediation digest are set for each
  retry (and cleaned up after).
- all attempts fail → the phase completes WITH the
  ``PLANNING_GATE_RETRIES_EXHAUSTED`` warning (fail-open) and the workflow
  CONTINUES to downstream phases.
- the retry is scoped to ``course_planning`` only.

Everything is mocked (scripted ``executor.execute_phase``) — no LLM, no GPU.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from unittest.mock import AsyncMock, MagicMock

from MCP.core import workflow_runner as wr_mod
from MCP.core.config import WorkflowConfig, WorkflowPhase
from MCP.core.executor import ExecutionResult
from MCP.core.workflow_runner import (
    WorkflowRunner,
    _planning_failure_digest,
    resolve_planning_gate_retries,
)

_WORKFLOW_ID = "WF-TEST-PLANNING-RETRY"
_SALT_ENV = "ED4ALL_PLANNING_REROLL_SALT"
_FEEDBACK_ENV = "ED4ALL_PLANNING_REROLL_FEEDBACK"
_RETRIES_ENV = "ED4ALL_PLANNING_GATE_RETRIES"


def _result(
    task_id: str, status: str, payload: Optional[Dict[str, Any]] = None
) -> ExecutionResult:
    return ExecutionResult(task_id=task_id, status=status, result=payload)


def _failing_gates() -> List[Dict[str, Any]]:
    """One failed critical gate, dict-shaped like the executor returns."""
    return [{
        "gate_id": "objective_entailment",
        "validator_name": (
            "lib.validators.objective_entailment.ObjectiveEntailmentValidator"
        ),
        "validator_version": "1.0",
        "passed": False,
        "severity": "critical",
        "issues": [{
            "severity": "critical",
            "code": "OBJECTIVE_NOT_ENTAILED",
            "message": "TO-01 statement not entailed by its cited chunks",
        }],
    }]


class _PlanningExecutorStub:
    """Scripted ``execute_phase``: course_planning fails its gates
    ``fail_attempts`` times, then passes; every other phase passes.

    Records per-course_planning-call observations (sidecar presence + salt
    env at ENTRY) and simulates the synthesis pass flushing BOTH stage-2
    sidecars + the objectives doc into the export's
    ``01_learning_objectives/`` dir on every attempt.
    """

    def __init__(self, obj_dir: Path, fail_attempts: int) -> None:
        self.obj_dir = obj_dir
        self.fail_attempts = fail_attempts
        self.calls: List[str] = []
        self.cluster_sidecar_at_entry: List[bool] = []
        self.window_sidecar_at_entry: List[bool] = []
        self.salt_at_entry: List[Optional[str]] = []
        self.feedback_at_entry: List[Optional[str]] = []

    def __call__(self, *_a: Any, **kw: Any):
        # Sync callable — AsyncMock(side_effect=...) awaits the mock and
        # returns this value directly (an async __call__ on an instance is
        # NOT detected as a coroutine function by AsyncMock).
        phase_name = kw.get("phase_name") or ""
        self.calls.append(phase_name)
        if phase_name != "course_planning":
            return (
                {"T-d": _result("T-d", "COMPLETE", {"ok": True})},
                True,
                [],
            )
        cluster = self.obj_dir / ".stage2_clusters_checkpoint.jsonl"
        window = self.obj_dir / ".stage2_windows_checkpoint.jsonl"
        self.cluster_sidecar_at_entry.append(cluster.exists())
        self.window_sidecar_at_entry.append(window.exists())
        self.salt_at_entry.append(os.environ.get(_SALT_ENV))
        self.feedback_at_entry.append(os.environ.get(_FEEDBACK_ENV))
        # Simulate the synthesis pass writing its artifacts + sidecars.
        self.obj_dir.mkdir(parents=True, exist_ok=True)
        cluster.write_text('{"unit": "cluster01"}\n', encoding="utf-8")
        window.write_text('{"unit": "ch1#w0"}\n', encoding="utf-8")
        obj_path = self.obj_dir / "synthesized_objectives.json"
        obj_path.write_text("{}", encoding="utf-8")
        payload = {
            "project_id": "PROJ-TEST",
            "synthesized_objectives_path": str(obj_path),
            "objective_ids": ["TO-01"],
            "terminal_count": 1,
            "chapter_count": 3,
            "grounding_signals": {},
        }
        results = {"T-1": _result("T-1", "COMPLETE", payload)}
        planning_calls = sum(1 for c in self.calls if c == "course_planning")
        if planning_calls <= self.fail_attempts:
            return results, False, _failing_gates()
        return results, True, []


def _drive(
    tmp_path,
    monkeypatch,
    *,
    stub: _PlanningExecutorStub,
    phases: List[WorkflowPhase],
) -> Dict[str, Any]:
    state_root = tmp_path / "state"
    monkeypatch.setattr(wr_mod, "STATE_PATH", state_root)
    monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")
    monkeypatch.setenv(
        "ED4ALL_TRAINING_CAPTURES_DIR", str(tmp_path / "captures")
    )

    wf_dir = state_root / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{_WORKFLOW_ID}.json").write_text(
        json.dumps({
            "workflow_id": _WORKFLOW_ID,
            "type": "test_wf",
            "params": {"course_name": "FXALPHA_101"},
            "phase_outputs": {},
            "tasks": [],
        })
    )

    executor = MagicMock()
    executor.execute_phase = AsyncMock(side_effect=stub)

    config = MagicMock()
    config.get_workflow.return_value = WorkflowConfig(
        description="test", phases=phases
    )

    runner = WorkflowRunner(executor=executor, config=config)
    return asyncio.run(runner.run_workflow(_WORKFLOW_ID))


def _load_persisted(tmp_path) -> Dict[str, Any]:
    state_path = tmp_path / "state" / "workflows" / f"{_WORKFLOW_ID}.json"
    return json.loads(state_path.read_text()).get("phase_outputs", {})


def _phases() -> List[WorkflowPhase]:
    return [
        WorkflowPhase(name="course_planning", agents=["course-outliner"]),
        WorkflowPhase(name="downstream_phase", agents=["content-generator"]),
    ]


# ---------------------------------------------------------------------------
# resolve_planning_gate_retries — parse-with-fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 0),          # unset
        ("", 0),            # blank
        ("  ", 0),          # whitespace
        ("garbage", 0),     # non-int
        ("-3", 0),          # negative clamps to off
        ("0", 0),
        ("10", 10),
        (" 7 ", 7),
    ],
)
def test_resolve_planning_gate_retries_parse_with_fallback(
    monkeypatch, raw, expected
) -> None:
    if raw is None:
        monkeypatch.delenv(_RETRIES_ENV, raising=False)
    else:
        monkeypatch.setenv(_RETRIES_ENV, raw)
    assert resolve_planning_gate_retries() == expected


# ---------------------------------------------------------------------------
# (a) retries=0 → byte-identical legacy block behavior
# ---------------------------------------------------------------------------


def test_retries_off_keeps_legacy_gate_block(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(_RETRIES_ENV, raising=False)
    stub = _PlanningExecutorStub(
        tmp_path / "export" / "01_learning_objectives", fail_attempts=99
    )
    out = _drive(tmp_path, monkeypatch, stub=stub, phases=_phases())

    assert out["status"] == "FAILED"
    assert out["failed_phase"] == "course_planning"
    # Exactly ONE dispatch — no retry ran; downstream never executed.
    assert stub.calls == ["course_planning"]
    persisted = _load_persisted(tmp_path)
    assert persisted["course_planning"]["_gates_passed"] is False
    assert "_planning_gate_retries_exhausted" not in (
        persisted["course_planning"]
    )
    # The salt + feedback envs were never set (legacy path).
    assert stub.salt_at_entry == [None]
    assert stub.feedback_at_entry == [None]


# ---------------------------------------------------------------------------
# (b) gate fails N-1 times then passes → recovery
# ---------------------------------------------------------------------------


def test_retry_recovers_after_transient_gate_failures(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(_RETRIES_ENV, "10")
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")
    obj_dir = tmp_path / "export" / "01_learning_objectives"
    # Fails twice (initial + retry 1), passes on retry 2 → 3 dispatches.
    stub = _PlanningExecutorStub(obj_dir, fail_attempts=2)
    out = _drive(tmp_path, monkeypatch, stub=stub, phases=_phases())

    assert out["status"] == "COMPLETE"
    planning_calls = [c for c in stub.calls if c == "course_planning"]
    assert len(planning_calls) == 3, "initial + 2 retries expected"
    # The workflow continued into the downstream phase.
    assert "downstream_phase" in stub.calls

    # The CLUSTER sidecar was evicted before every retry attempt (absent at
    # entry of every dispatch), while the WINDOW/CO sidecar was KEPT (it
    # survives from the prior attempt).
    assert stub.cluster_sidecar_at_entry == [False, False, False]
    assert stub.window_sidecar_at_entry == [False, True, True]

    # Per-attempt re-roll salt: none on the first attempt, attempt-N on the
    # retries; cleaned up after the retry loop.
    assert stub.salt_at_entry == [None, "attempt-1", "attempt-2"]
    assert _SALT_ENV not in os.environ

    # Failure-DIRECTED feedback: none on the first attempt; each retry
    # carries the serialized digest of the failing critical gates' issues
    # (objective id + issue code + directive/message); cleaned up after.
    assert stub.feedback_at_entry[0] is None
    for digest in stub.feedback_at_entry[1:]:
        assert digest, "each retry must carry the remediation digest"
        assert "objective_entailment" in digest
        assert "OBJECTIVE_NOT_ENTAILED" in digest
        assert "TO-01" in digest
        assert len(digest) <= 1200
    assert _FEEDBACK_ENV not in os.environ

    persisted = _load_persisted(tmp_path)
    assert persisted["course_planning"]["_gates_passed"] is True
    assert persisted["course_planning"]["_completed"] is True
    assert "_planning_gate_retries_exhausted" not in (
        persisted["course_planning"]
    )

    # DecisionCapture fired per retry decision (canonical
    # decision_type=validation_result; dynamic rationale interpolating the
    # attempt + failing gate ids + issue codes).
    captures_root = tmp_path / "captures"
    decision_lines: List[Dict[str, Any]] = []
    for path in captures_root.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                decision_lines.append(json.loads(line))
            except ValueError:
                continue
    retry_events = [
        d for d in decision_lines
        if d.get("decision_type") == "validation_result"
        and "gate retry" in str(d.get("decision", ""))
    ]
    assert retry_events, "expected a validation_result retry decision event"
    rationale = retry_events[0].get("rationale", "")
    assert len(rationale) >= 20
    assert "attempt 1" in rationale
    assert "objective_entailment" in rationale
    assert "OBJECTIVE_NOT_ENTAILED" in rationale
    # Capture quality contract: the retry decision interpolates the
    # failure-directed remediation digest it fed back to the synthesizer.
    assert "remediation digest" in rationale
    assert "TO-01" in rationale
    assert all(
        isinstance(alternative.get("option"), str)
        and isinstance(alternative.get("reason_rejected"), str)
        for alternative in retry_events[0]["alternatives_considered"]
    )


# ---------------------------------------------------------------------------
# (c) all attempts fail → fail-open with PLANNING_GATE_RETRIES_EXHAUSTED
# ---------------------------------------------------------------------------


def test_retries_exhausted_fail_open_with_loud_warning(
    tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setenv(_RETRIES_ENV, "2")
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")
    obj_dir = tmp_path / "export" / "01_learning_objectives"
    stub = _PlanningExecutorStub(obj_dir, fail_attempts=99)
    with caplog.at_level("ERROR", logger="MCP.core.workflow_runner"):
        out = _drive(tmp_path, monkeypatch, stub=stub, phases=_phases())

    # Fail-open: phase complete-with-warning, workflow CONTINUES.
    assert out["status"] == "COMPLETE"
    planning_calls = [c for c in stub.calls if c == "course_planning"]
    assert len(planning_calls) == 3, "initial + 2 retries expected"
    assert "downstream_phase" in stub.calls

    persisted = _load_persisted(tmp_path)
    planning = persisted["course_planning"]
    assert planning["_gates_passed"] is True
    assert planning["_completed"] is True
    assert planning["_planning_gate_retries_exhausted"] is True

    # The GateIssue-style warning surfaced in the phase's gate chain.
    exhausted = [
        g for g in planning.get("_gate_results", [])
        if isinstance(g, dict) and any(
            i.get("code") == "PLANNING_GATE_RETRIES_EXHAUSTED"
            for i in g.get("issues", [])
            if isinstance(i, dict)
        )
    ]
    assert exhausted, "expected a PLANNING_GATE_RETRIES_EXHAUSTED gate entry"
    assert exhausted[0]["passed"] is True
    assert exhausted[0]["severity"] == "warning"

    # The LOUD logger.error line fired.
    assert any(
        "PLANNING_GATE_RETRIES_EXHAUSTED" in r.message
        and r.levelname == "ERROR"
        for r in caplog.records
    ), "expected the loud PLANNING_GATE_RETRIES_EXHAUSTED error line"

    # Salt + feedback envs cleaned up.
    assert _SALT_ENV not in os.environ
    assert _FEEDBACK_ENV not in os.environ


# ---------------------------------------------------------------------------
# Scope: course_planning only
# ---------------------------------------------------------------------------


def test_non_planning_phase_is_never_retried(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(_RETRIES_ENV, "5")

    calls: List[str] = []

    async def side_effect(*_a: Any, **kw: Any):
        calls.append(kw.get("phase_name") or "")
        results = {"T-1": _result("T-1", "COMPLETE", {"ok": True})}
        return results, False, _failing_gates()

    state_root = tmp_path / "state"
    monkeypatch.setattr(wr_mod, "STATE_PATH", state_root)
    monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")
    wf_dir = state_root / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{_WORKFLOW_ID}.json").write_text(
        json.dumps({
            "workflow_id": _WORKFLOW_ID,
            "type": "test_wf",
            "params": {},
            "phase_outputs": {},
            "tasks": [],
        })
    )
    executor = MagicMock()
    executor.execute_phase = AsyncMock(side_effect=side_effect)
    config = MagicMock()
    config.get_workflow.return_value = WorkflowConfig(
        description="test",
        phases=[WorkflowPhase(name="other_gated_phase", agents=["a"])],
    )
    runner = WorkflowRunner(executor=executor, config=config)
    out = asyncio.run(runner.run_workflow(_WORKFLOW_ID))

    assert out["status"] == "FAILED"
    assert out["failed_phase"] == "other_gated_phase"
    assert calls == ["other_gated_phase"], (
        "gate retry must be scoped to course_planning only"
    )


# ---------------------------------------------------------------------------
# _planning_failure_digest — failure-directed remediation digest builder
# ---------------------------------------------------------------------------


def _gate(
    gate_id: str,
    issues: List[Dict[str, Any]],
    *,
    passed: bool = False,
    severity: str = "critical",
) -> Dict[str, Any]:
    return {
        "gate_id": gate_id,
        "passed": passed,
        "severity": severity,
        "issues": issues,
    }


def test_digest_mapped_code_yields_directive_and_objective_id() -> None:
    digest = _planning_failure_digest([
        _gate("objective_entailment", [{
            "severity": "critical",
            "code": "OBJECTIVE_UNENTAILED",
            "message": (
                "LO 'TO-08' statement is NOT entailed by its cited chunk(s)"
            ),
            "location": "TO-08",
        }]),
    ])
    assert digest.startswith("- [objective_entailment] TO-08 "
                             "OBJECTIVE_UNENTAILED:")
    # The code→directive map fired (not the raw-message fallback).
    assert "directly supported by its cited chunks" in digest


def test_digest_unmapped_code_falls_back_to_truncated_message() -> None:
    long_msg = "CO-12 " + ("x" * 500)
    digest = _planning_failure_digest([
        _gate("some_gate", [{
            "severity": "critical",
            "code": "SOME_UNKNOWN_CODE",
            "message": long_msg,
            "location": None,
        }]),
    ])
    assert "CO-12 SOME_UNKNOWN_CODE:" in digest
    # Raw message fallback is truncated per line.
    assert len(digest) < 300


def test_digest_ignores_passing_and_warning_gates() -> None:
    digest = _planning_failure_digest([
        _gate("passed_gate", [{
            "severity": "critical", "code": "X", "message": "m",
        }], passed=True),
        _gate("warning_gate", [{
            "severity": "critical", "code": "Y", "message": "m",
        }], severity="warning"),
        _gate("crit_gate", [{
            "severity": "warning", "code": "Z", "message": "m",
        }]),
    ])
    assert digest == ""


def test_digest_dedupes_and_caps_at_1200_chars() -> None:
    issues = [
        {
            "severity": "critical",
            "code": "OBJECTIVE_UNENTAILED",
            "message": f"LO 'CO-{i:02d}' statement is NOT entailed",
            "location": f"CO-{i:02d}",
        }
        for i in range(1, 60)
    ]
    # Duplicate of the first issue → deduped on (gate, objective, code).
    issues.append(dict(issues[0]))
    digest = _planning_failure_digest([_gate("objective_entailment", issues)])
    assert len(digest) <= 1200
    assert digest.count("CO-01 OBJECTIVE_UNENTAILED") == 1
    # Whole lines only — the cap never leaves a dangling partial directive.
    for line in digest.splitlines():
        assert line.startswith("- [objective_entailment] ")
