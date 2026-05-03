"""Tests for the Phase 3.5 remediation-suffix injection (Subtask 22).

Covers Subtasks 18 (outline-tier suffix injection in
``route_with_self_consistency``), 19
(``route_rewrite_with_remediation`` rewrite-tier symmetric loop), 20
(``_DEFAULT_REWRITE_REGEN_BUDGET`` + ``_resolve_rewrite_regen_budget``),
and 21 (``regen_budget_rewrite`` policy field).

The shared seam under test is the ``remediation_suffix`` kwarg that
flows from the router's per-loop suffix-builder
(``_append_remediation_for_gates`` from Wave A) into the provider's
``generate_outline`` / ``generate_rewrite`` ``_render_user_prompt``
seam. Test stubs record ``remediation_suffix`` in their calls log so
each assertion checks both that the suffix was supplied AND that it
carries the expected per-failure remediation directive.

Loosely mirrors the fixture style in ``test_self_consistency.py`` and
``test_validator_action.py`` for consistency.
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
    CourseforgeRouter,
    _DEFAULT_OUTLINE_REGEN_BUDGET,
    _DEFAULT_REWRITE_REGEN_BUDGET,
)
from Courseforge.router.policy import BlockRoutingPolicy  # noqa: E402
from MCP.hardening.validation_gates import GateIssue, GateResult  # noqa: E402
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
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


def _gate_result(
    *,
    gate_id: str = "outline_curie_anchoring",
    passed: bool = False,
    action: Optional[str] = "regenerate",
    issues: Optional[List[GateIssue]] = None,
) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        validator_name=gate_id,
        validator_version="1.0.0",
        passed=passed,
        action=action,
        issues=issues or [
            GateIssue(
                severity="critical",
                code=f"{gate_id.upper()}_FAIL",
                message=f"missing some {gate_id} signal in block",
            )
        ],
    )


class _RecordingOutlineProvider:
    """Records every ``generate_outline`` call including the
    ``remediation_suffix`` kwarg so assertions can verify the suffix
    flow."""

    def __init__(self, outputs: List[Block]) -> None:
        self._outputs = list(outputs)
        self.calls: List[Dict[str, Any]] = []

    def generate_outline(
        self,
        block: Block,
        *,
        source_chunks: Any,
        objectives: Any,
        remediation_suffix: Optional[str] = None,
        **kwargs: Any,
    ) -> Block:
        idx = min(len(self.calls), len(self._outputs) - 1)
        self.calls.append({
            "block_id": block.block_id,
            "remediation_suffix": remediation_suffix,
        })
        return self._outputs[idx]


class _RecordingRewriteProvider:
    """Sibling of ``_RecordingOutlineProvider`` for the rewrite tier."""

    def __init__(self, outputs: List[Block]) -> None:
        self._outputs = list(outputs)
        self.calls: List[Dict[str, Any]] = []

    def generate_rewrite(
        self,
        block: Block,
        *,
        source_chunks: Any,
        objectives: Any,
        remediation_suffix: Optional[str] = None,
        **kwargs: Any,
    ) -> Block:
        idx = min(len(self.calls), len(self._outputs) - 1)
        self.calls.append({
            "block_id": block.block_id,
            "remediation_suffix": remediation_suffix,
        })
        return self._outputs[idx]


class _SequenceValidator:
    """Returns canned ``GateResult`` instances in sequence."""

    name = "sequence_validator"
    version = "1.0.0"

    def __init__(self, results: List[GateResult]) -> None:
        if not results:
            raise ValueError("_SequenceValidator needs at least one result")
        self._results = list(results)
        self.calls: int = 0

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


# ---------------------------------------------------------------------------
# Subtask 18 — outline-tier remediation injection
# ---------------------------------------------------------------------------


def test_first_iteration_has_no_remediation_suffix(monkeypatch):
    """Subtask 18: first outline candidate dispatch carries
    ``remediation_suffix=None`` because no prior failures exist."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    blk = _block()
    outline = _RecordingOutlineProvider([blk, blk])
    validator = _SequenceValidator([
        _gate_result(passed=True, action="pass"),
    ])
    r = CourseforgeRouter(outline_provider=outline, n_candidates=2)
    r.route_with_self_consistency(blk, validators=[validator])
    assert outline.calls[0]["remediation_suffix"] is None


def test_second_iteration_carries_remediation_suffix_after_failure(monkeypatch):
    """Subtask 18: when iteration 0 fails, iteration 1 dispatches with a
    non-None ``remediation_suffix`` carrying the canonical
    ``_append_remediation_for_gates`` header + per-failure block."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    blk = _block()
    outline = _RecordingOutlineProvider([blk, blk])
    validator = _SequenceValidator([
        _gate_result(passed=False, action="regenerate"),
        _gate_result(passed=True, action="pass"),
    ])
    r = CourseforgeRouter(outline_provider=outline, n_candidates=2)
    r.route_with_self_consistency(blk, validators=[validator])
    assert len(outline.calls) == 2
    assert outline.calls[0]["remediation_suffix"] is None
    second_suffix = outline.calls[1]["remediation_suffix"]
    assert second_suffix is not None
    assert "Your previous attempt failed validation" in second_suffix
    assert "outline_curie_anchoring" in second_suffix


def test_remediation_suffix_carries_failure_directive(monkeypatch):
    """Subtask 18: the suffix contains the canonical "Correct by:" line
    drawn from the per-gate-id directive table in
    ``Courseforge.router.remediation``."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    blk = _block()
    outline = _RecordingOutlineProvider([blk, blk])
    validator = _SequenceValidator([
        _gate_result(
            gate_id="outline_curie_anchoring",
            passed=False,
            action="regenerate",
        ),
        _gate_result(passed=True, action="pass"),
    ])
    r = CourseforgeRouter(outline_provider=outline, n_candidates=2)
    r.route_with_self_consistency(blk, validators=[validator])
    second_suffix = outline.calls[1]["remediation_suffix"]
    assert second_suffix is not None
    # Directive copy from _REMEDIATION_DIRECTIVES_BY_GATE_ID for
    # outline_curie_anchoring.
    assert "Correct by: Preserve every CURIE verbatim" in second_suffix


def test_winning_candidate_does_not_re_dispatch_with_suffix(monkeypatch):
    """Subtask 18: when iteration 0 PASSES, the loop short-circuits and
    no second dispatch (and so no suffix) is built."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    outline = _RecordingOutlineProvider([blk, blk])
    validator = _SequenceValidator([
        _gate_result(passed=True, action="pass"),
    ])
    r = CourseforgeRouter(outline_provider=outline, n_candidates=2)
    r.route_with_self_consistency(blk, validators=[validator])
    assert len(outline.calls) == 1
    assert outline.calls[0]["remediation_suffix"] is None


# ---------------------------------------------------------------------------
# Subtask 19 — rewrite-tier remediation loop
# ---------------------------------------------------------------------------


def test_route_rewrite_with_remediation_winner_short_circuits(monkeypatch):
    """Subtask 19: when iteration 0 passes the rewrite-tier validator
    chain, the loop short-circuits and a ``self_consistency_winner``
    Touch is appended."""
    monkeypatch.delenv("COURSEFORGE_REWRITE_REGEN_BUDGET", raising=False)
    blk = _block()
    rewrite = _RecordingRewriteProvider([blk])
    validator = _SequenceValidator([
        _gate_result(passed=True, action="pass"),
    ])
    r = CourseforgeRouter(rewrite_provider=rewrite)
    out = r.route_rewrite_with_remediation(
        blk, n_candidates=3, validators=[validator]
    )
    assert len(rewrite.calls) == 1
    assert out.escalation_marker is None
    assert any(
        t.purpose == "self_consistency_winner" and t.tier == "rewrite"
        for t in out.touched_by
    )


def test_route_rewrite_with_remediation_threads_suffix_on_retry(monkeypatch):
    """Subtask 19: a failed rewrite candidate threads the canonical
    remediation suffix into the next dispatch."""
    monkeypatch.delenv("COURSEFORGE_REWRITE_REGEN_BUDGET", raising=False)
    blk = _block()
    rewrite = _RecordingRewriteProvider([blk, blk])
    validator = _SequenceValidator([
        _gate_result(
            gate_id="rewrite_curie_anchoring",
            passed=False,
            action="regenerate",
        ),
        _gate_result(passed=True, action="pass"),
    ])
    r = CourseforgeRouter(rewrite_provider=rewrite)
    r.route_rewrite_with_remediation(
        blk, n_candidates=2, regen_budget=5, validators=[validator]
    )
    assert len(rewrite.calls) == 2
    assert rewrite.calls[0]["remediation_suffix"] is None
    second = rewrite.calls[1]["remediation_suffix"]
    assert second is not None
    assert "rewrite_curie_anchoring" in second
    # Plan §3.5 reworded directive: rewrite-tier remediation now names
    # the three pedagogical-context shapes the gate accepts and
    # explicitly forbids attribute-value / fake-triple stuffing.
    assert "pedagogical voice" in second
    assert "<code>" in second


def test_route_rewrite_with_remediation_budget_exhaustion_stamps_consensus_fail(
    monkeypatch,
):
    """Subtask 19 + 20: when every rewrite-tier candidate fails and the
    budget is exhausted, the surviving candidate is stamped with
    ``escalation_marker="validator_consensus_fail"`` (NOT
    ``outline_budget_exhausted`` — that marker is reserved for the
    outline-tier escalation path)."""
    monkeypatch.delenv("COURSEFORGE_REWRITE_REGEN_BUDGET", raising=False)
    blk = _block()
    rewrite = _RecordingRewriteProvider([blk, blk, blk])
    validator = _SequenceValidator([
        _gate_result(passed=False, action="regenerate"),
    ])
    r = CourseforgeRouter(rewrite_provider=rewrite)
    out = r.route_rewrite_with_remediation(
        blk, n_candidates=3, regen_budget=3, validators=[validator]
    )
    assert out.escalation_marker == "validator_consensus_fail"
    assert out.validation_attempts == 3
    assert len(rewrite.calls) == 3


def test_route_rewrite_with_remediation_action_block_short_circuits(
    monkeypatch,
):
    """Subtask 19: a validator returning ``action="block"`` exits the
    rewrite-tier loop immediately with ``validator_consensus_fail``."""
    monkeypatch.delenv("COURSEFORGE_REWRITE_REGEN_BUDGET", raising=False)
    blk = _block()
    rewrite = _RecordingRewriteProvider([blk, blk, blk])
    validator = _SequenceValidator([
        _gate_result(passed=False, action="block"),
    ])
    r = CourseforgeRouter(rewrite_provider=rewrite)
    out = r.route_rewrite_with_remediation(
        blk, n_candidates=3, regen_budget=10, validators=[validator]
    )
    # Loop broke after iteration 0 — only one rewrite dispatch.
    assert len(rewrite.calls) == 1
    assert out.escalation_marker == "validator_consensus_fail"


# ---------------------------------------------------------------------------
# Subtask 20 + 21 — budget resolver precedence
# ---------------------------------------------------------------------------


def test_default_outline_regen_budget_is_ten():
    """Subtask 20: the outline-tier hardcoded default was bumped from 3
    to 10 to give the remediation-suffix loop room to converge."""
    assert _DEFAULT_OUTLINE_REGEN_BUDGET == 10


def test_default_rewrite_regen_budget_is_ten():
    """Subtask 20: the rewrite-tier hardcoded default mirrors the
    outline-tier default."""
    assert _DEFAULT_REWRITE_REGEN_BUDGET == 10


def test_resolve_rewrite_regen_budget_per_call_kwarg_wins(monkeypatch):
    """Subtask 20: per-call ``regen_budget`` kwarg beats env var + policy
    + hardcoded default."""
    monkeypatch.setenv("COURSEFORGE_REWRITE_REGEN_BUDGET", "20")
    blk = _block()
    r = CourseforgeRouter()
    assert r._resolve_rewrite_regen_budget(blk, override=7) == 7


def test_resolve_rewrite_regen_budget_policy_beats_env(monkeypatch):
    """Subtask 21: per-block-type policy entry beats the env var."""
    monkeypatch.setenv("COURSEFORGE_REWRITE_REGEN_BUDGET", "15")
    blk = _block(block_type="concept")
    policy = BlockRoutingPolicy(
        regen_budget_rewrite_by_block_type={"concept": 4},
    )
    r = CourseforgeRouter(policy=policy)
    assert r._resolve_rewrite_regen_budget(blk, override=None) == 4


def test_resolve_rewrite_regen_budget_env_beats_default(monkeypatch):
    """Subtask 20: env var beats the hardcoded default when no policy
    entry or per-call kwarg overrides it."""
    monkeypatch.setenv("COURSEFORGE_REWRITE_REGEN_BUDGET", "25")
    blk = _block()
    r = CourseforgeRouter()
    assert r._resolve_rewrite_regen_budget(blk, override=None) == 25


def test_resolve_rewrite_regen_budget_falls_through_to_default(monkeypatch):
    """Subtask 20: when nothing pins the budget, the hardcoded default
    (10) fires."""
    monkeypatch.delenv("COURSEFORGE_REWRITE_REGEN_BUDGET", raising=False)
    blk = _block()
    r = CourseforgeRouter()
    assert r._resolve_rewrite_regen_budget(blk, override=None) == 10
