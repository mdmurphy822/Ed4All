"""Tests for the lazy best-of-N / first-passing-candidate early exit on the
REWRITE tier (``COURSEFORGE_REWRITE_EARLY_EXIT``).

Under the ``entailment_argmax`` selector the rewrite loop normally samples all
N candidates and picks the post-loop argmax. With ``COURSEFORGE_REWRITE_EARLY_EXIT``
truthy it STOPS at the first validator-PASSING candidate whose NLI verdict also
clears the entailment floors (``covers_objective`` + not contradicted +
``entailment_rate >= 0.60`` — the ``block_prose_entailment`` production floor),
never sampling the remaining candidates. On a floor MISS it samples the next;
if none clear the floors the post-loop argmax runs byte-for-byte as legacy.

Cases:
* (a) flag off → byte-identical (all N sampled, post-loop argmax).
* (b) flag on + first candidate passes floors → exactly ONE sample drawn +
  ``block_best_of_n_selection`` capture fires.
* (c) first fails floor, second passes → exactly TWO samples drawn.
* (d) none pass floors → argmax fallback identical to legacy (all N sampled,
  same winner as flag-off).
* (e) contradiction disqualifies regardless of entailment_rate (a contradicted
  high-grounding candidate does NOT early-exit; the next clean one does).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

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
# Fakes (mirror test_best_of_n_selection.py)
# --------------------------------------------------------------------------- #

class _FakeScore:
    def __init__(self, ent: float, con: float = 0.0):
        self.entailment = ent
        self.neutral = max(0.0, 1.0 - ent - con)
        self.contradiction = con


class _ScriptedNLI:
    """Token-driven fake NLI: a planted token in the hypothesis sets grounding
    (``HIGH`` / ``MID`` / ``LOW``) or a contradiction (``FABRICATED``)."""

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


class _SequenceRewriteProvider:
    def __init__(self, outputs: List[Block]) -> None:
        self._outputs = list(outputs)
        self.calls: List[Dict[str, Any]] = []

    def generate_rewrite(self, block, *, source_chunks, objectives, **kwargs):
        idx = min(len(self.calls), len(self._outputs) - 1)
        self.calls.append({"block": block})
        return self._outputs[idx]


class _AlwaysPassValidator:
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


def _rewrite_router(monkeypatch, outputs, *, early_exit, capture=None):
    monkeypatch.setenv("COURSEFORGE_BEST_OF_N_SELECT_BY", "entailment_argmax")
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    if early_exit:
        monkeypatch.setenv("COURSEFORGE_REWRITE_EARLY_EXIT", "1")
    else:
        monkeypatch.delenv("COURSEFORGE_REWRITE_EARLY_EXIT", raising=False)
    provider = _SequenceRewriteProvider(outputs)
    router = CourseforgeRouter(
        rewrite_provider=provider,
        capture=capture,
        n_candidates=len(outputs),
        nli=_ScriptedNLI(),
    )
    return router, provider


def _run(router, outputs):
    return router.route_rewrite_with_remediation(
        outputs[0],
        validators=[_AlwaysPassValidator()],
        source_chunks=_SOURCE_CHUNKS,
        objectives=_OBJECTIVES,
    )


# --------------------------------------------------------------------------- #
# (a) flag off → byte-identical (all N sampled, post-loop argmax)
# --------------------------------------------------------------------------- #

def test_flag_off_samples_all_and_argmax(monkeypatch):
    # cand0 rate 0.5 (HIGH+LOW), cand1 rate 1.0 (2 HIGH) — argmax is cand1.
    outputs = [
        _candidate(0, prose=f"{_HIGH} {_LOW}"),
        _candidate(1, prose=f"{_HIGH} {_HIGH}"),
    ]
    router, provider = _rewrite_router(monkeypatch, outputs, early_exit=False)
    out = _run(router, outputs)
    # All N sampled; the fully-grounded candidate wins (argmax, not first-pass).
    assert len(provider.calls) == 2
    assert out.block_id == "page1#concept_intro_1"


# --------------------------------------------------------------------------- #
# (b) flag on + first passes floors → exactly ONE sample + capture
# --------------------------------------------------------------------------- #

def test_early_exit_first_candidate_passes(monkeypatch):
    # cand0 rate 1.0 (2 HIGH) clears the 0.60 floor → stop after ONE sample.
    outputs = [
        _candidate(0, prose=f"{_HIGH} {_HIGH}"),
        _candidate(1, prose=f"{_HIGH} {_HIGH}"),
    ]
    capture = _FakeCapture()
    router, provider = _rewrite_router(
        monkeypatch, outputs, early_exit=True, capture=capture
    )
    out = _run(router, outputs)
    # Only the first candidate was dispatched — candidates 2..N never sampled.
    assert len(provider.calls) == 1
    assert out.block_id == "page1#concept_intro_0"
    assert any(t.purpose == "self_consistency_winner" for t in out.touched_by)
    # The early-exit choice is auditable via the same best_of_n capture.
    sel = [
        e for e in capture.events
        if e.get("decision_type") == "block_best_of_n_selection"
    ]
    assert len(sel) == 1
    ml = sel[0]["ml_features"]
    assert ml["winning_index"] == 0
    assert ml["per_candidate_entailment_rates"][0] == pytest.approx(1.0)
    assert len(sel[0]["rationale"]) >= 20


# --------------------------------------------------------------------------- #
# (c) first fails floor, second passes → exactly TWO samples
# --------------------------------------------------------------------------- #

def test_early_exit_second_candidate_passes(monkeypatch):
    # cand0 rate 0.5 (HIGH+LOW) < 0.60 floor → keep sampling; cand1 rate 1.0
    # clears the floor → stop after TWO samples (never reaches cand2).
    outputs = [
        _candidate(0, prose=f"{_HIGH} {_LOW}"),
        _candidate(1, prose=f"{_HIGH} {_HIGH}"),
        _candidate(2, prose=f"{_HIGH} {_HIGH}"),
    ]
    router, provider = _rewrite_router(monkeypatch, outputs, early_exit=True)
    out = _run(router, outputs)
    assert len(provider.calls) == 2
    assert out.block_id == "page1#concept_intro_1"


# --------------------------------------------------------------------------- #
# (d) none pass floors → argmax fallback identical to legacy
# --------------------------------------------------------------------------- #

def test_early_exit_none_pass_falls_back_to_argmax(monkeypatch):
    # cand0 rate 0.5, cand1 rate ~0.333 — neither clears 0.60. Early-exit never
    # fires; the post-loop argmax keeps the highest (cand0), sampling all N —
    # byte-identical to the flag-off run.
    outputs = [
        _candidate(0, prose=f"{_HIGH} {_LOW}"),
        _candidate(1, prose=f"{_HIGH} {_LOW} {_LOW}"),
    ]
    router_on, provider_on = _rewrite_router(monkeypatch, outputs, early_exit=True)
    out_on = _run(router_on, outputs)
    assert len(provider_on.calls) == 2  # all N sampled — no early exit fired
    assert out_on.block_id == "page1#concept_intro_0"

    # Flag-off control produces the identical winner + sample count.
    router_off, provider_off = _rewrite_router(monkeypatch, outputs, early_exit=False)
    out_off = _run(router_off, outputs)
    assert len(provider_off.calls) == 2
    assert out_off.block_id == out_on.block_id


# --------------------------------------------------------------------------- #
# (e) contradiction disqualifies regardless of entailment_rate
# --------------------------------------------------------------------------- #

def test_early_exit_contradiction_disqualifies(monkeypatch):
    # cand0 is contradicted (FABRICATED) yet HIGH-grounded (rate 2/3 ~0.667 >=
    # 0.60) — the contradiction must veto the early exit, so we sample cand1
    # (clean, 2 HIGH) and stop there. Proves the floor is not entailment-only.
    outputs = [
        _candidate(0, prose=f"{_FAB} {_HIGH} {_HIGH}"),
        _candidate(1, prose=f"{_HIGH} {_HIGH}"),
    ]
    router, provider = _rewrite_router(monkeypatch, outputs, early_exit=True)
    out = _run(router, outputs)
    assert len(provider.calls) == 2  # cand0's contradiction blocked early exit
    assert out.block_id == "page1#concept_intro_1"
