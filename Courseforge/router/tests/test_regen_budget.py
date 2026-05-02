"""Tests for the regen-budget + escalation contract (Phase 3 Subtask 44).

Exercises the per-block regeneration budget tracking and escalation
paths Worker I landed in Subtasks 41-43:

- Subtask 41 (``route_with_self_consistency``): per-fail
  ``validation_attempts`` increment, ``COURSEFORGE_OUTLINE_REGEN_BUDGET``
  env var resolution, policy ``regen_budget_by_block_type`` override,
  and the canonical ``escalation_marker="outline_budget_exhausted"``
  stamp + early-loop break when the cumulative attempt count meets the
  resolved budget.
- Subtask 42 (``route``): the ``escalate_immediately=True`` short-
  circuit on the outline tier — skips the LLM dispatch entirely, stamps
  the canonical ``outline_budget_exhausted`` marker (NOTE: the plan's
  wording said ``outline_skipped_by_policy`` but
  ``Block._ESCALATION_MARKERS`` only admits the budget-exhausted /
  structural-unfixable / validator-consensus-fail values; provenance is
  preserved via the ``Touch.purpose="escalate_immediately"`` audit
  record on the same return block).
- Subtask 42-43 (``_emit_block_escalation``): the canonical
  ``block_escalation`` decision-event seam — fired from BOTH the
  budget-exhausted and policy-skip paths so a postmortem reader sees a
  single event class regardless of how the block reached the rewrite
  tier.

Test fixtures use the same stub-OutlineProvider / stub-validator /
fake-capture pattern as ``test_self_consistency.py`` so the loop
branching is fully observable without any LLM dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import dataclasses
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
    validation_attempts: int = 0,
    escalation_marker: Optional[str] = None,
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page1",
        sequence=0,
        content=content,
        validation_attempts=validation_attempts,
        escalation_marker=escalation_marker,
    )


class _AlwaysFailingOutlineProvider:
    """Stub OutlineProvider that returns the input block unchanged.

    Used as the outline-tier stand-in for every test in this module: the
    paired stub validator always returns ``passed=False`` so the
    self-consistency loop accumulates ``validation_attempts`` until the
    regen budget triggers the escalation path. The provider records each
    call so a test can assert how many dispatches actually fired before
    the budget check short-circuited the loop.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def generate_outline(
        self, block: Block, *, source_chunks: Any, objectives: Any
    ) -> Block:
        self.calls.append(
            {
                "block": block,
                "source_chunks": source_chunks,
                "objectives": objectives,
            }
        )
        # Return the block unchanged so the per-fail
        # ``dataclasses.replace(..., validation_attempts=...)`` rebind
        # in ``route_with_self_consistency`` is the ONLY thing bumping
        # the attempt count; otherwise the provider could mask a missing
        # increment in the loop.
        return block


class _AlwaysFailingValidator:
    """Stub validator that always returns a failing GateResult."""

    def __init__(
        self,
        *,
        validator_name: str = "outline_curie_anchoring",
        action: Optional[str] = None,
    ) -> None:
        self._validator_name = validator_name
        self._action = action
        self.calls: List[Dict[str, Any]] = []

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        self.calls.append({"inputs": inputs})
        return GateResult(
            gate_id=self._validator_name,
            validator_name=self._validator_name,
            validator_version="1.0.0",
            passed=False,
            action=self._action,
        )


class _RecordingRewriteProvider:
    """Stub RewriteProvider that records the input Block and returns it.

    Used by ``test_escalated_block_routes_to_rewrite_with_richer_prompt``
    to confirm the router hands the rewrite tier a Block that carries
    the canonical ``outline_budget_exhausted`` escalation marker — the
    rewrite provider's ``generate_rewrite`` branches on that field to
    pick the richer ``_render_escalated_user_prompt`` template. Without
    a true LLM call we cannot inspect the rendered prompt directly, but
    we CAN assert the input shape that drives the prompt branch.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def generate_rewrite(
        self, block: Block, *, source_chunks: Any, objectives: Any
    ) -> Block:
        self.calls.append(
            {
                "block": block,
                "source_chunks": source_chunks,
                "objectives": objectives,
                "escalation_marker": block.escalation_marker,
                "validation_attempts": block.validation_attempts,
            }
        )
        return block


class _FakeCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


class _StubPolicy:
    """Minimal BlockRoutingPolicy stand-in.

    Carries the two fast-lookup maps the router consults during
    n-candidate / regen-budget resolution. ``resolve`` returns ``None``
    so the spec resolution path falls through to the hardcoded defaults.
    """

    def __init__(
        self,
        *,
        n_candidates_by_block_type: Optional[Dict[str, int]] = None,
        regen_budget_by_block_type: Optional[Dict[str, int]] = None,
    ) -> None:
        self.n_candidates_by_block_type = n_candidates_by_block_type or {}
        self.regen_budget_by_block_type = regen_budget_by_block_type or {}

    def resolve(self, block_id: str, block_type: str, tier: str) -> Any:
        return None


# ---------------------------------------------------------------------------
# Subtask 41: validation_attempts + budget-exhaustion tests
# ---------------------------------------------------------------------------


def test_validation_attempts_increments_on_failure(monkeypatch):
    """Each failed validator pass increments the cumulative
    ``validation_attempts`` count on the candidate Block.

    Pin ``regen_budget`` higher than ``n_candidates`` so the loop runs
    through every candidate and the increment is observable on the
    final return block (the budget-exhaustion early-break is exercised
    in ``test_budget_exhaustion_sets_outline_budget_exhausted_marker``).
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    blk = _block()
    provider = _AlwaysFailingOutlineProvider()
    validator = _AlwaysFailingValidator()
    r = CourseforgeRouter(
        outline_provider=provider, n_candidates=2, regen_budget=10
    )
    out = r.route_with_self_consistency(blk, validators=[validator])
    # Loop ran through both candidates (budget=10 > N=2 so no early break).
    assert len(provider.calls) == 2
    assert out.validation_attempts == 2
    # No escalation marker because budget was not exhausted.
    assert out.escalation_marker is None


def test_budget_exhaustion_sets_outline_budget_exhausted_marker(monkeypatch):
    """When cumulative ``validation_attempts`` meets the resolved budget,
    the router stamps the canonical
    ``escalation_marker="outline_budget_exhausted"`` on the return block
    and breaks out of the candidate loop early.

    NOTE: The Phase 3 plan section §3.7 lists ``outline_skipped_by_policy``
    as a candidate marker name, but ``Block._ESCALATION_MARKERS``
    (``Courseforge/scripts/blocks.py:105-111``) only contains
    ``{outline_budget_exhausted, structural_unfixable,
    validator_consensus_fail}``. Worker 3F's deviation note: the router
    uses ``outline_budget_exhausted`` for both the regen-budget and the
    policy-skip paths so the marker stays Block-validation-clean;
    provenance is preserved via the ``Touch.purpose="escalate_immediately"``
    audit field on the policy-skip path. This test asserts the actual
    marker the implementation lands.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    blk = _block()
    provider = _AlwaysFailingOutlineProvider()
    validator = _AlwaysFailingValidator()
    # n_candidates=5 but regen_budget=2 → early break after 2 fails.
    r = CourseforgeRouter(
        outline_provider=provider, n_candidates=5, regen_budget=2
    )
    out = r.route_with_self_consistency(blk, validators=[validator])
    # Loop broke early after exactly 2 failed candidates.
    assert len(provider.calls) == 2
    assert out.validation_attempts == 2
    assert out.escalation_marker == "outline_budget_exhausted"


def test_budget_exhaustion_emits_block_escalation_event(monkeypatch):
    """The Subtask 41 path emits ONE ``block_escalation`` decision-
    capture event when the regen budget is exhausted mid-loop.

    The ml_features payload carries the dynamic signals required by the
    Subtask 43 contract: ``block_id`` / ``block_type`` / ``marker`` /
    ``attempts`` / ``n_candidates``.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    blk = _block(block_type="concept", block_id="page1#concept_intro_0")
    provider = _AlwaysFailingOutlineProvider()
    validator = _AlwaysFailingValidator()
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=provider,
        capture=capture,
        n_candidates=5,
        regen_budget=3,
    )
    r.route_with_self_consistency(blk, validators=[validator])
    escalation_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_escalation"
    ]
    assert len(escalation_events) == 1
    ev = escalation_events[0]
    ml = ev.get("ml_features", {})
    assert ml.get("block_id") == "page1#concept_intro_0"
    assert ml.get("block_type") == "concept"
    assert ml.get("marker") == "outline_budget_exhausted"
    # Budget=3 → 3 failed dispatches before the early break fires.
    assert ml.get("attempts") == 3
    assert ml.get("n_candidates") == 3
    # Rationale must be at least 20 chars (Subtask 43 contract).
    assert isinstance(ev.get("rationale"), str)
    assert len(ev["rationale"]) >= 20


def test_per_block_type_regen_budget_overrides_env(monkeypatch):
    """Per-block-type policy ``regen_budget_by_block_type`` beats the
    ``COURSEFORGE_OUTLINE_REGEN_BUDGET`` env var.

    Resolution chain (highest first): per-call kwarg → policy →
    env var → constructor → default. With env=10 and policy={concept:2}
    the policy value wins → loop breaks after 2 fails.
    """
    monkeypatch.setenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", "10")
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _block(block_type="concept")
    provider = _AlwaysFailingOutlineProvider()
    validator = _AlwaysFailingValidator()
    policy = _StubPolicy(regen_budget_by_block_type={"concept": 2})
    r = CourseforgeRouter(
        policy=policy, outline_provider=provider, n_candidates=5
    )
    out = r.route_with_self_consistency(blk, validators=[validator])
    # Policy budget=2 wins over env=10.
    assert len(provider.calls) == 2
    assert out.validation_attempts == 2
    assert out.escalation_marker == "outline_budget_exhausted"


# ---------------------------------------------------------------------------
# Subtask 42: escalate_immediately short-circuit tests
# ---------------------------------------------------------------------------


def test_escalate_immediately_skips_outline_entirely(monkeypatch):
    """When the resolved spec carries ``escalate_immediately=True``,
    ``route_with_self_consistency`` delegates to ``route(tier="outline")``
    which short-circuits BEFORE any LLM dispatch — the candidate loop
    never runs.

    Verified by asserting the outline-provider ``calls`` list stays
    empty across the dispatch.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    blk = _block()
    provider = _AlwaysFailingOutlineProvider()
    validator = _AlwaysFailingValidator()
    r = CourseforgeRouter(
        outline_provider=provider, n_candidates=5, regen_budget=5
    )
    # Trigger the escalate_immediately short-circuit via per-call override.
    out = r.route_with_self_consistency(
        blk,
        validators=[validator],
        escalate_immediately=True,
    )
    # No outline dispatch ever fired.
    assert provider.calls == []
    # No validator calls either — the loop never entered.
    assert validator.calls == []
    # The block carries the canonical escalation marker.
    assert out.escalation_marker == "outline_budget_exhausted"


def test_escalate_immediately_sets_outline_skipped_by_policy(monkeypatch):
    """Policy-skip provenance is preserved on the short-circuit return
    block via a ``Touch(purpose="escalate_immediately")`` audit record.

    NOTE on the marker name: the plan's contract section §3.7 calls this
    case ``outline_skipped_by_policy``, but ``Block._ESCALATION_MARKERS``
    only admits the canonical 3-value set, so the implementation lands
    ``escalation_marker="outline_budget_exhausted"`` and signals the
    policy-skip path exclusively via ``Touch.purpose="escalate_immediately"``
    (the audit field is the canonical discriminator a postmortem reader
    consults — the marker is the broader rewrite-tier-routing flag both
    paths share). This test pins both signals so a future refactor can't
    silently drop either side without breaking the contract.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    blk = _block()
    provider = _AlwaysFailingOutlineProvider()
    capture = _FakeCapture()
    r = CourseforgeRouter(outline_provider=provider, capture=capture)
    out = r.route(
        blk,
        tier="outline",
        escalate_immediately=True,
    )
    # Marker is the canonical Block-validated value.
    assert out.escalation_marker == "outline_budget_exhausted"
    # Provenance is preserved via the Touch.purpose audit field — this
    # is the discriminator that distinguishes the policy-skip path from
    # the regen-budget-exhaustion path.
    skip_touches = [
        t for t in out.touched_by if t.purpose == "escalate_immediately"
    ]
    assert len(skip_touches) == 1
    skip_touch = skip_touches[0]
    assert skip_touch.tier == "outline"
    # Capture path: the policy-skip path emits a block_escalation event
    # with attempts=0 + n_candidates=0 (signalling no outline dispatch).
    escalation_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_escalation"
    ]
    assert len(escalation_events) == 1
    ml = escalation_events[0].get("ml_features", {})
    assert ml.get("attempts") == 0
    assert ml.get("n_candidates") == 0
    assert ml.get("marker") == "outline_budget_exhausted"


# ---------------------------------------------------------------------------
# Cross-tier handoff: escalated block → rewrite tier
# ---------------------------------------------------------------------------


def test_escalated_block_routes_to_rewrite_with_richer_prompt(monkeypatch):
    """A block carrying ``escalation_marker="outline_budget_exhausted"``
    that's dispatched to the rewrite tier must reach
    ``RewriteProvider.generate_rewrite`` with the marker still attached
    — the rewrite provider branches on that field to select
    ``_render_escalated_user_prompt`` over the standard
    ``_render_user_prompt`` template.

    We don't drive a real LLM call; we assert the input-shape contract
    that drives the prompt branch (the prompt-rendering branch itself
    is exercised by the rewrite-provider unit tests in
    ``Courseforge/tests/test_rewrite_provider*.py``).
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    escalated = _block(
        validation_attempts=3,
        escalation_marker="outline_budget_exhausted",
    )
    rewrite_provider = _RecordingRewriteProvider()
    r = CourseforgeRouter(rewrite_provider=rewrite_provider)
    r.route(escalated, tier="rewrite")
    assert len(rewrite_provider.calls) == 1
    call = rewrite_provider.calls[0]
    # The marker survived the router-tier handoff so the rewrite
    # provider's `if block.escalation_marker is not None` branch picks
    # the escalated prompt template.
    assert call["escalation_marker"] == "outline_budget_exhausted"
    assert call["validation_attempts"] == 3


# ---------------------------------------------------------------------------
# Frozen-dataclass invariant: validation_attempts persists through replace
# ---------------------------------------------------------------------------


def test_validation_attempts_persists_through_block_replace():
    """``Block`` is frozen, so the per-fail rebind in
    ``route_with_self_consistency`` happens via
    ``dataclasses.replace(block, validation_attempts=...)``.

    This test pins the invariant at the dataclass level: a Block
    constructed with ``validation_attempts=N`` round-trips through
    ``dataclasses.replace`` (with no other field changed) preserving
    ``N``, AND a subsequent stamp of ``escalation_marker`` does NOT
    reset the attempt counter. Both halves matter because the router's
    Subtask 41 escalation block is built as
    ``dataclasses.replace(last_candidate, escalation_marker=...)``
    where ``last_candidate`` already carries the cumulative
    ``validation_attempts`` count from the per-failure rebind.
    """
    blk = _block(validation_attempts=4)
    # Trivial replace preserves the count.
    same = dataclasses.replace(blk, validation_attempts=blk.validation_attempts)
    assert same.validation_attempts == 4
    # Stamping escalation_marker on top does not reset validation_attempts.
    escalated = dataclasses.replace(
        same, escalation_marker="outline_budget_exhausted"
    )
    assert escalated.validation_attempts == 4
    assert escalated.escalation_marker == "outline_budget_exhausted"
    # And the original is unchanged (frozen dataclass invariant).
    assert blk.validation_attempts == 4
    assert blk.escalation_marker is None
