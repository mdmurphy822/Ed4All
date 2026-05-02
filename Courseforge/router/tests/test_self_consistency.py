"""Tests for ``CourseforgeRouter.route_with_self_consistency`` (Phase 3 Subtask 40).

Exercises the self-consistency dispatch loop documented in Phase 3 §3.6:

- N-candidate resolution chain (kwarg → policy → env → constructor → default).
- First-passing-candidate short-circuit.
- All-fail behaviour (validation_attempts incremented; escalation_marker
  left untouched — Subtask 41 handles that contract).
- Decision-capture metadata in the structured ml_features payload of
  the block_outline_call audit event.
- Per-validator failure distribution aggregation.

The fixture uses a stub OutlineProvider that returns canned outputs in
sequence and a stub validator that emits canned GateResults so the
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
from MCP.hardening.validation_gates import GateResult  # noqa: E402
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
    """Stub OutlineProvider returning canned Blocks in sequence.

    Each ``generate_outline`` call returns the next entry from
    ``self._outputs``; if the list is exhausted the last entry is
    re-yielded so a test that asks for more candidates than canned
    outputs still works without an IndexError.
    """

    def __init__(self, outputs: List[Block]) -> None:
        if not outputs:
            raise ValueError("_SequenceOutlineProvider needs at least one canned output")
        self._outputs = list(outputs)
        self.calls: List[Dict[str, Any]] = []

    def generate_outline(
        self,
        block: Block,
        *,
        source_chunks: Any,
        objectives: Any,
        **kwargs: Any,
    ) -> Block:
        # Phase 3.5 Subtask 18: ``**kwargs`` swallows the new
        # ``remediation_suffix`` kwarg the router threads in on regen
        # iterations; the stub records its presence in the calls log
        # so tests can assert the suffix flowed through.
        idx = min(len(self.calls), len(self._outputs) - 1)
        self.calls.append({
            "block": block,
            "source_chunks": source_chunks,
            "objectives": objectives,
            **{k: v for k, v in kwargs.items() if k == "remediation_suffix"},
        })
        return self._outputs[idx]


class _SequenceValidator:
    """Stub validator returning canned GateResult instances in sequence.

    ``self._results`` is a list of GateResults; the i-th call returns
    ``self._results[i]``. When the list is exhausted the last result is
    re-yielded.
    """

    def __init__(
        self,
        *,
        validator_name: str,
        results: List[GateResult],
    ) -> None:
        if not results:
            raise ValueError("_SequenceValidator needs at least one canned result")
        self.validator_name = validator_name
        self._results = list(results)
        self.calls: List[Dict[str, Any]] = []

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        idx = min(len(self.calls), len(self._results) - 1)
        self.calls.append({"inputs": inputs})
        return self._results[idx]


def _make_gate_result(
    *,
    passed: bool,
    validator_name: str = "outline_curie_anchoring",
    action: Optional[str] = None,
) -> GateResult:
    return GateResult(
        gate_id=validator_name,
        validator_name=validator_name,
        validator_version="1.0.0",
        passed=passed,
        action=action,
    )


class _FakeCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


class _StubPolicy:
    """Minimal BlockRoutingPolicy stand-in.

    Exposes the ``n_candidates_by_block_type`` fast-lookup map that
    Worker G's policy carries, so the router's
    :meth:`_resolve_n_candidates` can branch into the policy layer
    without depending on the full ``BlockRoutingPolicy.resolve`` shape.
    """

    def __init__(
        self,
        *,
        n_candidates_by_block_type: Optional[Dict[str, int]] = None,
        spec: Optional[BlockProviderSpec] = None,
    ) -> None:
        self.n_candidates_by_block_type = n_candidates_by_block_type or {}
        self._spec = spec

    def resolve(self, block_id: str, block_type: str, tier: str) -> Any:
        return self._spec


# ---------------------------------------------------------------------------
# Self-consistency loop tests
# ---------------------------------------------------------------------------


def test_first_candidate_passes_returns_immediately(monkeypatch):
    """When the first candidate passes every validator the loop exits
    after one dispatch and the winning Touch carries the
    self_consistency_winner purpose."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    outputs = [blk, blk, blk]
    provider = _SequenceOutlineProvider(outputs)
    validator = _SequenceValidator(
        validator_name="outline_curie_anchoring",
        results=[_make_gate_result(passed=True)],
    )
    r = CourseforgeRouter(outline_provider=provider, n_candidates=3)
    out = r.route_with_self_consistency(blk, validators=[validator])
    # Only one outline dispatch happened.
    assert len(provider.calls) == 1
    # Winning Touch appended.
    assert any(t.purpose == "self_consistency_winner" for t in out.touched_by)
    # validation_attempts NOT bumped on a pass.
    assert out.validation_attempts == 0


def test_third_candidate_passes_after_two_fail(monkeypatch):
    """After two failing candidates, the third passes; the loop returns
    the third candidate and the audit event records winning_candidate_index=2."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk, blk, blk])
    validator = _SequenceValidator(
        validator_name="outline_curie_anchoring",
        results=[
            _make_gate_result(passed=False),
            _make_gate_result(passed=False),
            _make_gate_result(passed=True),
        ],
    )
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=provider, capture=capture, n_candidates=3
    )
    out = r.route_with_self_consistency(blk, validators=[validator])
    assert len(provider.calls) == 3
    assert any(t.purpose == "self_consistency_winner" for t in out.touched_by)
    # ml_features carries winning_candidate_index=2.
    sc_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_outline_call"
        and "self_consistency:" in e.get("decision", "")
    ]
    assert len(sc_events) == 1
    ml = sc_events[0]["ml_features"]
    assert ml["winning_candidate_index"] == 2
    assert ml["failed_candidate_count"] == 2


def test_all_candidates_fail_returns_last_with_validation_attempts_n(monkeypatch):
    """When every candidate fails AND the regen budget is exhausted at
    or before N, the router returns the LAST candidate with
    validation_attempts incremented per fail and the canonical
    ``outline_budget_exhausted`` marker stamped (Subtask 41).

    With N=3 and the default regen_budget=3, the third failure brings
    validation_attempts to 3 == budget → escalation marker fires and
    the loop breaks early after the third dispatch. The earlier
    Subtask-37 contract (marker left None for budget>N runs) is
    exercised by ``test_validation_attempts_increments_below_budget``
    in ``test_regen_budget.py``.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk, blk, blk])
    validator = _SequenceValidator(
        validator_name="outline_curie_anchoring",
        results=[_make_gate_result(passed=False)],
    )
    r = CourseforgeRouter(outline_provider=provider, n_candidates=3)
    out = r.route_with_self_consistency(blk, validators=[validator])
    assert len(provider.calls) == 3
    # validation_attempts bumped per failure to budget=N=3.
    assert out.validation_attempts == 3
    # Subtask 41: canonical marker stamped on budget exhaustion.
    assert out.escalation_marker == "outline_budget_exhausted"
    # No winning Touch on a full-loop failure.
    assert not any(t.purpose == "self_consistency_winner" for t in out.touched_by)


def test_n_candidates_resolves_from_env_var(monkeypatch):
    """``COURSEFORGE_OUTLINE_N_CANDIDATES`` env var beats the hardcoded
    default when neither per-call kwarg nor policy entry is set.

    Subtask 41 follow-on: also pin the regen budget to a value >= N
    so the budget-exhaustion check doesn't fire early and the loop
    actually runs through all N dispatches (otherwise the env-var
    resolution itself wouldn't be observable).
    """
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "5")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", "10")
    blk = _block()
    provider = _SequenceOutlineProvider([blk] * 5)
    validator = _SequenceValidator(
        validator_name="outline_curie_anchoring",
        results=[_make_gate_result(passed=False)],
    )
    r = CourseforgeRouter(outline_provider=provider)
    out = r.route_with_self_consistency(blk, validators=[validator])
    # Env var wins → 5 dispatches.
    assert len(provider.calls) == 5
    assert out.validation_attempts == 5


def test_n_candidates_resolves_from_policy_block_override(monkeypatch):
    """When the policy supplies ``n_candidates_by_block_type``, that
    value beats env var + constructor default.

    Subtask 41 follow-on: pin a per-call ``regen_budget`` >= N so the
    budget-exhaustion path doesn't short-circuit before the policy
    N-candidates resolution is observable.
    """
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "10")
    blk = _block(block_type="concept")
    provider = _SequenceOutlineProvider([blk] * 5)
    validator = _SequenceValidator(
        validator_name="outline_curie_anchoring",
        results=[_make_gate_result(passed=False)],
    )
    policy = _StubPolicy(n_candidates_by_block_type={"concept": 4})
    r = CourseforgeRouter(
        policy=policy, outline_provider=provider, n_candidates=2
    )
    out = r.route_with_self_consistency(
        blk, validators=[validator], regen_budget=10
    )
    # Policy wins over env var (10) and constructor default (2).
    assert len(provider.calls) == 4
    assert out.validation_attempts == 4


def test_decision_event_includes_winning_candidate_index(monkeypatch):
    """The block_outline_call self-consistency event's ml_features
    payload carries n_candidates_requested + winning_candidate_index +
    failed_candidate_count + validator_failure_distribution."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk, blk])
    validator = _SequenceValidator(
        validator_name="outline_curie_anchoring",
        results=[
            _make_gate_result(passed=False),
            _make_gate_result(passed=True),
        ],
    )
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=provider, capture=capture, n_candidates=2
    )
    r.route_with_self_consistency(blk, validators=[validator])
    sc_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_outline_call"
        and "self_consistency:" in e.get("decision", "")
    ]
    assert len(sc_events) == 1
    ml = sc_events[0].get("ml_features")
    assert ml is not None
    assert ml["n_candidates_requested"] == 2
    assert ml["winning_candidate_index"] == 1
    assert ml["failed_candidate_count"] == 1
    assert isinstance(ml["validator_failure_distribution"], dict)


def test_validator_failure_distribution_is_aggregated(monkeypatch):
    """When multiple candidates fail multiple validators, the
    distribution dict accumulates per-validator failure counts."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk, blk, blk])

    validator_a = _SequenceValidator(
        validator_name="outline_curie_anchoring",
        results=[_make_gate_result(passed=False, validator_name="outline_curie_anchoring")],
    )
    validator_b = _SequenceValidator(
        validator_name="outline_content_type",
        results=[_make_gate_result(passed=False, validator_name="outline_content_type")],
    )
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=provider, capture=capture, n_candidates=3
    )
    # fast_fail=False so BOTH validators are exercised per candidate;
    # the distribution dict ends up with both names ticked once per
    # candidate (3 candidates × 2 validators = 6 total fails).
    r.route_with_self_consistency(
        blk, validators=[validator_a, validator_b], fast_fail=False
    )
    sc_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_outline_call"
        and "self_consistency:" in e.get("decision", "")
    ]
    assert len(sc_events) == 1
    ml = sc_events[0]["ml_features"]
    dist = ml["validator_failure_distribution"]
    assert dist.get("outline_curie_anchoring") == 3
    assert dist.get("outline_content_type") == 3
    # All-fail outcome → winning_candidate_index is None.
    assert ml["winning_candidate_index"] is None
    assert ml["failed_candidate_count"] == 3


def test_empty_validators_list_treats_first_candidate_as_winner(monkeypatch):
    """Pre-Phase-4 default: when no validators are wired the loop
    treats the first dispatched candidate as the winner. Confirms the
    Wave-N self-consistency surface degrades cleanly to the
    single-candidate path the existing route() method takes."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block()
    provider = _SequenceOutlineProvider([blk, blk, blk])
    r = CourseforgeRouter(outline_provider=provider, n_candidates=3)
    out = r.route_with_self_consistency(blk, validators=None)
    # Only one dispatch — first-pass winner.
    assert len(provider.calls) == 1
    assert any(t.purpose == "self_consistency_winner" for t in out.touched_by)
