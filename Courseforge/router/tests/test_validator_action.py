"""Tests for ``GateResult.action`` consumption in the Courseforge router (Phase 3 Subtask 49).

Exercises the validator-action signal landed by Subtasks 46-48:

- Legacy validators that don't set ``action`` map to ``"pass"`` on
  success and ``"block"`` on failure via
  :meth:`GateResult.derive_default_action` for the canonical audit
  event, while the router's self-consistency loop preserves the
  pre-Phase-3 retry semantics for back-compat.
- Phase-3-aware validators that set ``action="regenerate"`` /
  ``"escalate"`` / ``"block"`` directly engage the corresponding
  router-level dispatch:
  - ``regenerate`` continues the loop (retry).
  - ``escalate`` breaks the loop, stamps
    ``escalation_marker="validator_consensus_fail"``, returns for the
    rewrite tier's enriched-prompt branch.
  - ``block`` breaks the loop, stamps
    ``escalation_marker="structural_unfixable"``, returns for
    downstream consumers to skip the rewrite tier entirely.
- The action-priority order ``block > escalate > regenerate > pass``
  is honoured when multiple validators emit different actions for the
  same candidate.
- Each non-pass validator result emits one ``block_validation_action``
  decision-capture event with the gate_id, action, score, and top-3
  issues interpolated into the rationale.

Stub validators return canned :class:`GateResult` instances so the
loop's branching is fully observable without any LLM dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.router.router import (  # noqa: E402
    BlockProviderSpec,
    CourseforgeRouter,
)
from MCP.hardening.validation_gates import GateIssue, GateResult  # noqa: E402
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _block(
    *,
    block_type: str = "concept",
    block_id: str = "page1#concept_intro_0",
    content: Any = "hello",
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page1",
        sequence=0,
        content=content,
    )


class _SequenceOutlineProvider:
    """Stub OutlineProvider returning canned Blocks in sequence."""

    def __init__(self, outputs: List[Block]) -> None:
        if not outputs:
            raise ValueError("_SequenceOutlineProvider needs at least one canned output")
        self._outputs = list(outputs)
        self.calls: List[Dict[str, Any]] = []

    def generate_outline(
        self, block: Block, *, source_chunks: Any, objectives: Any
    ) -> Block:
        idx = min(len(self.calls), len(self._outputs) - 1)
        self.calls.append(
            {"block": block, "source_chunks": source_chunks, "objectives": objectives}
        )
        return self._outputs[idx]


class _CannedValidator:
    """Stub validator returning canned :class:`GateResult` instances.

    The i-th call returns ``self._results[i]``; once exhausted the
    last result is re-yielded so a test can assert the loop ran a
    bounded number of times without an IndexError.
    """

    def __init__(
        self,
        *,
        validator_name: str,
        results: List[GateResult],
    ) -> None:
        if not results:
            raise ValueError("_CannedValidator needs at least one canned result")
        self.validator_name = validator_name
        self._results = list(results)
        self.calls: List[Dict[str, Any]] = []

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        idx = min(len(self.calls), len(self._results) - 1)
        self.calls.append({"inputs": inputs})
        return self._results[idx]


def _gate_result(
    *,
    passed: bool,
    validator_name: str = "outline_curie_anchoring",
    gate_id: Optional[str] = None,
    action: Optional[str] = None,
    score: Optional[float] = None,
    issues: Optional[List[GateIssue]] = None,
) -> GateResult:
    return GateResult(
        gate_id=gate_id or validator_name,
        validator_name=validator_name,
        validator_version="1.0.0",
        passed=passed,
        score=score,
        issues=list(issues or []),
        action=action,
    )


class _FakeCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_legacy_validator_passed_true_treated_as_pass_action(monkeypatch):
    """Legacy validator with ``passed=True`` and ``action=None`` is
    treated as a pass: the first candidate wins, no retry, no
    block_validation_action event fires."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk])
    validator = _CannedValidator(
        validator_name="legacy_validator",
        results=[_gate_result(passed=True)],
    )
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=provider, capture=capture, n_candidates=3
    )
    out = r.route_with_self_consistency(blk, validators=[validator])
    # First candidate wins → only 1 dispatch.
    assert len(provider.calls) == 1
    # Winning Touch appended.
    assert any(
        t.purpose == "self_consistency_winner" for t in out.touched_by
    )
    # No structural marker stamped.
    assert out.escalation_marker is None
    # No block_validation_action events emitted (steady-state pass).
    bva_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_validation_action"
    ]
    assert bva_events == []


def test_legacy_validator_passed_false_treated_as_block_action(monkeypatch):
    """Legacy validator with ``passed=False`` and ``action=None``
    emits ``block_validation_action`` events whose ``action`` field is
    ``"block"`` (per ``derive_default_action`` contract). The router's
    loop-control still treats it as ``regenerate`` for back-compat —
    the loop continues until budget exhaustion."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk, blk, blk])
    validator = _CannedValidator(
        validator_name="legacy_validator",
        results=[_gate_result(passed=False)],
    )
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=provider, capture=capture, n_candidates=3
    )
    out = r.route_with_self_consistency(blk, validators=[validator])
    # Loop ran all 3 candidates (back-compat: legacy fail → retry).
    assert len(provider.calls) == 3
    # Budget exhaustion path fires (validation_attempts==3 == budget).
    assert out.escalation_marker == "outline_budget_exhausted"
    # Each non-pass result emits one block_validation_action event with
    # action="block" (the derive_default_action canonical mapping).
    bva_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_validation_action"
    ]
    assert len(bva_events) == 3
    for evt in bva_events:
        assert evt["ml_features"]["action"] == "block"


def test_validator_emitting_regenerate_triggers_self_consistency_retry(
    monkeypatch,
):
    """A validator explicitly emitting ``action="regenerate"`` keeps
    the loop running until either a candidate passes or the budget
    exhausts. Here the second candidate passes."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk, blk])
    validator = _CannedValidator(
        validator_name="curie_anchoring",
        results=[
            _gate_result(passed=False, action="regenerate"),
            _gate_result(passed=True, action="pass"),
        ],
    )
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=provider, capture=capture, n_candidates=3
    )
    out = r.route_with_self_consistency(blk, validators=[validator])
    # Two dispatches: first failed → retry → second passed.
    assert len(provider.calls) == 2
    # Winning Touch on the second candidate.
    assert any(
        t.purpose == "self_consistency_winner" for t in out.touched_by
    )
    assert out.escalation_marker is None
    # One block_validation_action event for the regenerate fail.
    bva_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_validation_action"
    ]
    assert len(bva_events) == 1
    assert bva_events[0]["ml_features"]["action"] == "regenerate"


def test_validator_emitting_escalate_skips_remaining_outline_retries(
    monkeypatch,
):
    """A validator returning ``action="escalate"`` breaks the loop
    immediately: subsequent outline candidates are NOT dispatched, the
    block carries ``escalation_marker="validator_consensus_fail"``,
    and a ``block_escalation`` event fires."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk, blk, blk])
    validator = _CannedValidator(
        validator_name="content_type",
        results=[_gate_result(passed=False, action="escalate")],
    )
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=provider, capture=capture, n_candidates=3
    )
    out = r.route_with_self_consistency(blk, validators=[validator])
    # Only one candidate dispatched — the escalate broke the loop.
    assert len(provider.calls) == 1
    # Validator-consensus marker stamped.
    assert out.escalation_marker == "validator_consensus_fail"
    # No winning Touch.
    assert not any(
        t.purpose == "self_consistency_winner" for t in out.touched_by
    )
    # block_escalation event fired with the consensus-fail marker.
    esc_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_escalation"
    ]
    assert len(esc_events) == 1
    assert esc_events[0]["ml_features"]["marker"] == "validator_consensus_fail"


def test_validator_emitting_block_marks_block_failed(monkeypatch):
    """A validator returning ``action="block"`` breaks the loop
    immediately and stamps ``escalation_marker="structural_unfixable"``
    so downstream consumers (route_all, packaging) skip the rewrite
    tier entirely. The structural-unfixable path bypasses the regen
    budget — it's a "give up entirely" signal."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk, blk, blk])
    validator = _CannedValidator(
        validator_name="page_objectives",
        results=[_gate_result(passed=False, action="block")],
    )
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=provider, capture=capture, n_candidates=3
    )
    out = r.route_with_self_consistency(blk, validators=[validator])
    # Only one candidate dispatched — block broke the loop.
    assert len(provider.calls) == 1
    # Structural-unfixable marker stamped.
    assert out.escalation_marker == "structural_unfixable"
    assert not any(
        t.purpose == "self_consistency_winner" for t in out.touched_by
    )
    # block_escalation event fired with the structural-unfixable
    # marker; attempts=0 signals the bypass-budget path.
    esc_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_escalation"
    ]
    assert len(esc_events) == 1
    assert esc_events[0]["ml_features"]["marker"] == "structural_unfixable"
    assert esc_events[0]["ml_features"]["attempts"] == 0


def test_block_validation_action_event_includes_gate_id_and_action(
    monkeypatch,
):
    """The ``block_validation_action`` event carries gate_id, action,
    score, and top-3 issues in both the rationale string and the
    ml_features payload."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk])
    issues = [
        GateIssue(
            severity="critical",
            code="curie_missing",
            message="No CURIEs detected in outline content",
        ),
        GateIssue(
            severity="warning",
            code="curie_unresolved",
            message="CURIE 'foo:bar' did not resolve",
        ),
    ]
    validator = _CannedValidator(
        validator_name="curie_anchoring",
        results=[
            _gate_result(
                passed=False,
                action="regenerate",
                gate_id="outline_curie_anchoring",
                score=0.42,
                issues=issues,
            )
        ],
    )
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=provider, capture=capture, n_candidates=1
    )
    r.route_with_self_consistency(blk, validators=[validator])
    bva_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_validation_action"
    ]
    assert len(bva_events) == 1
    evt = bva_events[0]
    # Decision string carries gate_id + action.
    assert "outline_curie_anchoring" in evt["decision"]
    assert "regenerate" in evt["decision"]
    # Rationale ≥20 chars and interpolates the gate_id, action, score.
    assert len(evt["rationale"]) >= 20
    assert "outline_curie_anchoring" in evt["rationale"]
    assert "regenerate" in evt["rationale"]
    assert "0.420" in evt["rationale"]  # score formatted to 3 decimals
    # ml_features carries the structured fields.
    ml = evt["ml_features"]
    assert ml["gate_id"] == "outline_curie_anchoring"
    assert ml["action"] == "regenerate"
    assert ml["score"] == pytest.approx(0.42)
    assert ml["issues_count"] == 2
    # Top-3 issues recorded with code + severity + message.
    issues_top3 = ml["issues_top3"]
    assert len(issues_top3) == 2
    assert issues_top3[0]["code"] == "curie_missing"
    assert issues_top3[0]["severity"] == "critical"
    assert issues_top3[1]["code"] == "curie_unresolved"


def test_action_priority_block_over_escalate_over_regenerate_over_pass(
    monkeypatch,
):
    """When multiple validators return DIFFERENT non-pass actions for
    the same candidate, the router dispatches on the highest-priority
    action. Priority order: ``block > escalate > regenerate > pass``.

    Setup: one candidate passes through three validators that emit
    ``regenerate``, ``escalate``, ``block`` respectively. The block
    action wins → loop breaks immediately, structural-unfixable
    marker stamped."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk, blk, blk])
    v_regen = _CannedValidator(
        validator_name="curie_anchoring",
        results=[_gate_result(passed=False, action="regenerate", validator_name="curie_anchoring")],
    )
    v_escalate = _CannedValidator(
        validator_name="content_type",
        results=[_gate_result(passed=False, action="escalate", validator_name="content_type")],
    )
    v_block = _CannedValidator(
        validator_name="page_objectives",
        results=[_gate_result(passed=False, action="block", validator_name="page_objectives")],
    )
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=provider, capture=capture, n_candidates=3
    )
    out = r.route_with_self_consistency(
        blk,
        validators=[v_regen, v_escalate, v_block],
        fast_fail=False,
    )
    # Only one outline dispatch — block broke the loop after the first
    # candidate's gate-result aggregation.
    assert len(provider.calls) == 1
    # Block action wins → structural-unfixable, NOT validator-
    # consensus-fail.
    assert out.escalation_marker == "structural_unfixable"


def test_multiple_validators_emit_separate_events(monkeypatch):
    """Each non-pass validator on a single candidate emits its own
    ``block_validation_action`` event with its own gate_id."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk])
    v_a = _CannedValidator(
        validator_name="curie_anchoring",
        results=[
            _gate_result(
                passed=False,
                action="regenerate",
                validator_name="curie_anchoring",
                gate_id="outline_curie_anchoring",
            )
        ],
    )
    v_b = _CannedValidator(
        validator_name="content_type",
        results=[
            _gate_result(
                passed=False,
                action="regenerate",
                validator_name="content_type",
                gate_id="outline_content_type",
            )
        ],
    )
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=provider, capture=capture, n_candidates=1
    )
    # fast_fail=False so BOTH validators run on the candidate even
    # after the first reports a non-pass action.
    r.route_with_self_consistency(
        blk, validators=[v_a, v_b], fast_fail=False
    )
    bva_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_validation_action"
    ]
    assert len(bva_events) == 2
    gate_ids = {evt["ml_features"]["gate_id"] for evt in bva_events}
    assert gate_ids == {"outline_curie_anchoring", "outline_content_type"}
