"""Tests for the W5 best-of-N NLI-verifier selector seam (§3, §5 cases 2-3).

The self-consistency loops already sample N candidates and run the validator
chain; W5 flips the winner selector to ``argmax(entailment_rate)`` among PASSING
candidates when ``COURSEFORGE_BEST_OF_N_SELECT_BY=entailment_argmax`` (the C5
mode). These tests assert:

* case 2 — best-of-N picks the highest-grounding PASSING candidate +
  the ``block_best_of_n_selection`` capture fires with the spread in its
  rationale.
* case 3 — a contradicted candidate is rejected even at the highest entailment;
  with no clean candidate the selector keeps the highest-entailment passing one +
  stamps ``best_of_n_no_clean_candidate`` (never fabricates a winner).
* additive guard — without the env flag the legacy first-passing selector runs
  unchanged.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SCRIPTS = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from Courseforge.router.router import CourseforgeRouter  # noqa: E402
from MCP.hardening.validation_gates import GateResult  # noqa: E402
from blocks import Block  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class _FakeScore:
    def __init__(self, ent: float, con: float = 0.0):
        self.entailment = ent
        self.neutral = max(0.0, 1.0 - ent - con)
        self.contradiction = con


class _ScriptedNLI:
    """Token-driven fake NLI: the candidate prose carries a planted token that
    determines its grounding (``HIGH`` / ``MID`` / ``LOW``) or a contradiction
    (``FABRICATED``). Returns one score per (premise, hypothesis) pair."""

    _revision = "fake-nli"
    device = "cpu"

    def score_batch(self, *, pairs):
        out = []
        for _premise, hyp in pairs:
            h = str(hyp)
            if "FABRICATED" in h:
                out.append(_FakeScore(0.05, con=0.95))
            elif "HIGH" in h:
                out.append(_FakeScore(0.95))
            elif "MID" in h:
                out.append(_FakeScore(0.75))
            else:  # LOW
                out.append(_FakeScore(0.20))
        return out


class _SequenceOutlineProvider:
    def __init__(self, outputs: List[Block]) -> None:
        self._outputs = list(outputs)
        self.calls: List[Dict[str, Any]] = []

    def generate_outline(self, block, *, source_chunks, objectives, **kwargs):
        idx = min(len(self.calls), len(self._outputs) - 1)
        self.calls.append({"block": block})
        return self._outputs[idx]


class _AlwaysPassValidator:
    """Every candidate PASSES the validator chain (so selection is driven purely
    by the NLI verifier, not the gate)."""

    validator_name = "always_pass"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        return GateResult(
            gate_id="always_pass",
            validator_name="always_pass",
            validator_version="1.0.0",
            passed=True,
            action=None,
        )


class _FakeCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


# Per-sentence tokens: each scored sentence carries ≥4 content tokens (the
# claim-support floor) and a planted token (HIGH/LOW/FABRICATED) driving its NLI
# verdict. ``score_groundedness`` reads entailed/scored, so a candidate with k of
# n sentences HIGH lands at rate k/n.
_HIGH = "This HIGH supported sentence matches the cited reference material closely."
_LOW = "This LOW unsupported sentence drifts well beyond the cited reference text."
_FAB = "This FABRICATED sentence directly conflicts with the cited reference material."


def _candidate(idx: int, *, prose: str) -> Block:
    return Block(
        block_id=f"page1#concept_intro_{idx}",
        block_type="concept",
        page_id="page1",
        sequence=0,
        content=prose,
        objective_ids=("CO-01",),
        source_ids=("c1",),
    )


_SOURCE_CHUNKS = [{"id": "c1", "text": "Reference text describing the concept in detail."}]
_OBJECTIVES = [{"id": "CO-01", "statement": "Understand the concept."}]


def _argmax_router(monkeypatch, outputs, *, capture=None):
    monkeypatch.setenv("COURSEFORGE_BEST_OF_N_SELECT_BY", "entailment_argmax")
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    provider = _SequenceOutlineProvider(outputs)
    return (
        CourseforgeRouter(
            outline_provider=provider,
            capture=capture,
            n_candidates=len(outputs),
            nli=_ScriptedNLI(),
        ),
        provider,
    )


# --------------------------------------------------------------------------- #
# Case 2 — best-of-N picks the highest-grounding sample
# --------------------------------------------------------------------------- #

def test_best_of_n_picks_highest_grounding(monkeypatch):
    # 3 passing candidates with graded grounding: cand0=0.5 (1 HIGH/1 LOW),
    # cand1=1.0 (2 HIGH), cand2≈0.667 (2 HIGH/1 LOW). cand1 is the argmax.
    outputs = [
        _candidate(0, prose=f"{_HIGH} {_LOW}"),
        _candidate(1, prose=f"{_HIGH} {_HIGH}"),
        _candidate(2, prose=f"{_HIGH} {_HIGH} {_LOW}"),
    ]
    capture = _FakeCapture()
    router, provider = _argmax_router(monkeypatch, outputs, capture=capture)
    out = router.route_with_self_consistency(
        outputs[0],
        validators=[_AlwaysPassValidator()],
        source_chunks=_SOURCE_CHUNKS,
        objectives=_OBJECTIVES,
    )
    # All N sampled (no early return under argmax).
    assert len(provider.calls) == 3
    # The fully-grounded candidate (index 1) wins.
    assert out.block_id == "page1#concept_intro_1"
    assert any(t.purpose == "self_consistency_winner" for t in out.touched_by)
    # The best_of_n capture fired with the spread in its rationale.
    sel = [e for e in capture.events if e.get("decision_type") == "block_best_of_n_selection"]
    assert len(sel) == 1
    rationale = sel[0]["rationale"]
    assert "spread" in rationale and "0.50" in rationale and "1.00" in rationale
    assert len(rationale) >= 20
    ml = sel[0]["ml_features"]
    assert ml["winning_index"] == 1
    rates = ml["per_candidate_entailment_rates"]
    assert rates[0] == pytest.approx(0.5) and rates[1] == pytest.approx(1.0)
    assert ml["select_by"] == "entailment_argmax"


# --------------------------------------------------------------------------- #
# Case 3 — contradiction is fatal to selection
# --------------------------------------------------------------------------- #

def test_contradiction_rejected_falls_to_next(monkeypatch):
    # Candidate 1 is contradicted (FABRICATED) → rejected even though a
    # contradicted block would otherwise tie; candidate 0 (fully grounded, clean)
    # wins. Candidate 2 is only half grounded.
    outputs = [
        _candidate(0, prose=f"{_HIGH} {_HIGH}"),
        _candidate(1, prose=f"{_HIGH} {_FAB}"),
        _candidate(2, prose=f"{_HIGH} {_LOW}"),
    ]
    router, provider = _argmax_router(monkeypatch, outputs)
    out = router.route_with_self_consistency(
        outputs[0],
        validators=[_AlwaysPassValidator()],
        source_chunks=_SOURCE_CHUNKS,
        objectives=_OBJECTIVES,
    )
    # The contradicted candidate (index 1) is rejected → grounded clean (0) wins.
    assert out.block_id == "page1#concept_intro_0"
    assert out.escalation_marker != "best_of_n_no_clean_candidate"


def test_all_contradicted_sets_no_clean_marker(monkeypatch):
    # Every passing candidate is contradicted → no clean winner; selector keeps
    # the highest-entailment one + stamps the no-clean marker (never fabricates).
    outputs = [
        _candidate(0, prose=f"{_FAB} {_HIGH}"),
        _candidate(1, prose=f"{_FAB} {_LOW}"),
    ]
    router, provider = _argmax_router(monkeypatch, outputs)
    out = router.route_with_self_consistency(
        outputs[0],
        validators=[_AlwaysPassValidator()],
        source_chunks=_SOURCE_CHUNKS,
        objectives=_OBJECTIVES,
    )
    assert out.escalation_marker == "best_of_n_no_clean_candidate"
    # Still a winner Touch (the block ships — it passed validation).
    assert any(t.purpose == "self_consistency_winner" for t in out.touched_by)


# --------------------------------------------------------------------------- #
# Additive guard — legacy selector unchanged without the flag
# --------------------------------------------------------------------------- #

def test_legacy_first_passing_without_flag(monkeypatch):
    monkeypatch.delenv("COURSEFORGE_BEST_OF_N_SELECT_BY", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    outputs = [
        _candidate(0, prose=f"{_HIGH} {_LOW}"),
        _candidate(1, prose=f"{_HIGH} {_HIGH}"),
    ]
    provider = _SequenceOutlineProvider(outputs)
    router = CourseforgeRouter(
        outline_provider=provider, n_candidates=2, nli=_ScriptedNLI()
    )
    out = router.route_with_self_consistency(
        outputs[0],
        validators=[_AlwaysPassValidator()],
        source_chunks=_SOURCE_CHUNKS,
        objectives=_OBJECTIVES,
    )
    # Legacy: first PASSING candidate (index 0) wins, only ONE dispatch.
    assert len(provider.calls) == 1
    assert out.block_id == "page1#concept_intro_0"


def test_rewrite_tier_best_of_n_argmax(monkeypatch):
    # The rewrite-tier loop carries the identical seam.
    monkeypatch.setenv("COURSEFORGE_BEST_OF_N_SELECT_BY", "entailment_argmax")
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    outputs = [
        _candidate(0, prose=f"{_HIGH} {_LOW}"),
        _candidate(1, prose=f"{_HIGH} {_HIGH}"),
    ]

    class _RewriteProvider:
        def __init__(self, outs):
            self._outs = list(outs)
            self.calls = []

        def generate_rewrite(self, block, *, source_chunks, objectives, **kwargs):
            idx = min(len(self.calls), len(self._outs) - 1)
            self.calls.append({})
            return self._outs[idx]

    provider = _RewriteProvider(outputs)
    router = CourseforgeRouter(
        rewrite_provider=provider, n_candidates=2, nli=_ScriptedNLI()
    )
    out = router.route_rewrite_with_remediation(
        outputs[0],
        validators=[_AlwaysPassValidator()],
        source_chunks=_SOURCE_CHUNKS,
        objectives=_OBJECTIVES,
    )
    assert out.block_id == "page1#concept_intro_1"
