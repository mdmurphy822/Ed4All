"""Bloom-ladder initiative (WI-12) — caller-side window/rung wiring.

WI-11 (``Trainforge/generators/synthesis_window_contract.py``) built the
per-CARD Bloom-rung recovery plumbing and ``build_evidence_window``'s
optional ``target_rung`` ceiling, but explicitly left it dead code on the
caller side: neither call site threaded a ``target_rung`` and
``micro_preference_eligibility`` Arm A minted every candidate's canonical id
from the single function-scoped (objective/chunk-wide) Bloom level, even
when the authored misconception card carried its own, different rung.

This module covers the caller-side half:

* ``build_evidence_window`` call sites in ``staged_synthesis_provider.py``
  and ``staged_synthesis_micro.py`` now pass ``target_rung=`` the focus
  objective's own ``bloom_level`` -- proven here by observing the ceiling
  actually exclude an over-rung evidence block, not just by inspecting the
  call signature.
* ``micro_preference_eligibility`` Arm A recomputes (and rejects on
  disagreement with) the canonical misconception id using the CARD's own
  ``bloom_level`` when ``TRAINFORGE_BLOOM_WINDOWS`` is on, instead of
  applying the chunk-wide value to every card.
* A ``dpo_pair_from_misconception`` decision fires once per Arm-A admitted
  candidate (flag on only), carrying the minted id, the rung used, and the
  source-backing outcome in its rationale.

Every behavior is gated on ``TRAINFORGE_BLOOM_WINDOWS`` (default OFF); the
flag-off tests below pin the pre-ladder byte-identical contract.
"""

from __future__ import annotations

import pytest

from lib.ontology.misconception_id import canonical_mc_id
from Trainforge.generators._synthesis_common import SynthesisProviderError
from Trainforge.generators.synthesis_window_contract import BLOOM_WINDOWS_ENV
import Trainforge.generators.staged_synthesis_provider as provider_module
from Trainforge.generators.staged_synthesis_micro import micro_preference_eligibility


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_OBJECTIVE_ID = "CO-1"
_MISCONCEPTION = "atomicity preserves partial updates"
_CORRECTION = "atomicity prevents partial updates"
_MECHANISM = "all grouped operations succeed or fail together"

_FLAT_TEXT = (
    "A mistaken view says atomicity preserves partial updates. The "
    "correction is that atomicity prevents partial updates by making all "
    "grouped operations succeed or fail together."
)


def _flat_chunk(*, chunk_bloom: str, item_bloom: str | None, incoming_id: str) -> dict:
    """A misconception-only, no-HTML chunk (flat_source evidence path)."""
    item: dict = {
        "id": incoming_id,
        "misconception": _MISCONCEPTION,
        "correction": _CORRECTION,
        "mechanism_evidence": _MECHANISM,
    }
    if item_bloom is not None:
        item["bloom_level"] = item_bloom
    return {
        "id": "chunk-atomicity",
        "text": _FLAT_TEXT,
        "bloom_level": chunk_bloom,
        "misconceptions": [item],
    }


def _focus(bloom_level: str) -> dict:
    return {
        "id": "co-1",
        "statement": "Analyze how atomicity prevents partial updates.",
        "bloom_level": bloom_level,
    }


class _RecordingCapture:
    """Minimal ``DecisionCapture``-shaped fake: records every log_decision call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log_decision(self, **kwargs) -> None:
        self.calls.append(kwargs)


# ---------------------------------------------------------------------------
# (b) Arm A id mint + candidate bloom_level use the CARD's own rung
# ---------------------------------------------------------------------------


def test_flag_off_mints_id_from_chunk_wide_rung_byte_identical(monkeypatch):
    """Pre-ladder (legacy) contract: the card's own rung is ignored."""
    monkeypatch.delenv(BLOOM_WINDOWS_ENV, raising=False)
    card_id = canonical_mc_id(_MISCONCEPTION, _CORRECTION, "analyze")
    chunk = _flat_chunk(
        chunk_bloom="understand", item_bloom="analyze", incoming_id=card_id,
    )
    result = micro_preference_eligibility(chunk, focus=_focus("understand"))

    # The card supplied its OWN canonical id (minted at its own "analyze"
    # rung). Off, the recompute still uses the chunk-wide "understand"
    # value, so the recomputed id disagrees with the supplied one and the
    # candidate is rejected as a mismatch -- the exact legacy behavior.
    assert result["eligible"] is False
    assert result["reason"] == "preference_misconception_id_mismatch"
    assert result["telemetry"]["rejection_counts"]["id_mismatch"] == 1


def test_flag_on_mints_id_from_the_cards_own_rung(monkeypatch):
    """The fix: a card-minted id at the card's own rung is now recognized."""
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    card_id = canonical_mc_id(_MISCONCEPTION, _CORRECTION, "analyze")
    chunk = _flat_chunk(
        chunk_bloom="understand", item_bloom="analyze", incoming_id=card_id,
    )
    result = micro_preference_eligibility(chunk, focus=_focus("understand"))

    assert result["eligible"] is True, result["reason"]
    assert result["telemetry"]["rejection_counts"]["id_mismatch"] == 0
    assert result["candidates"][0]["id"] == card_id
    assert result["candidates"][0]["bloom_level"] == "analyze"


def test_flag_on_falls_back_to_chunk_wide_rung_when_card_carries_none(monkeypatch):
    """A card without its own rung keeps the chunk-wide fallback, flag on."""
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    fallback_id = canonical_mc_id(_MISCONCEPTION, _CORRECTION, "understand")
    chunk = _flat_chunk(
        chunk_bloom="understand", item_bloom=None, incoming_id=fallback_id,
    )
    result = micro_preference_eligibility(chunk, focus=_focus("understand"))

    assert result["eligible"] is True, result["reason"]
    assert result["candidates"][0]["id"] == fallback_id
    assert result["candidates"][0]["bloom_level"] == "understand"


# ---------------------------------------------------------------------------
# (c) dpo_pair_from_misconception decision capture
# ---------------------------------------------------------------------------


def test_admission_emits_dpo_pair_from_misconception_decision(monkeypatch):
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    card_id = canonical_mc_id(_MISCONCEPTION, _CORRECTION, "analyze")
    chunk = _flat_chunk(
        chunk_bloom="understand", item_bloom="analyze", incoming_id=card_id,
    )
    capture = _RecordingCapture()
    result = micro_preference_eligibility(
        chunk, focus=_focus("understand"), capture=capture,
    )

    assert result["eligible"] is True, result["reason"]
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "dpo_pair_from_misconception"
    assert len(call["rationale"]) >= 20
    assert card_id in call["rationale"]
    assert "analyze" in call["rationale"]
    assert "source_backs" in call["rationale"]


def test_no_capture_supplied_is_a_silent_no_op(monkeypatch):
    """``capture`` defaults to ``None``; every pre-existing caller is unaffected."""
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    card_id = canonical_mc_id(_MISCONCEPTION, _CORRECTION, "analyze")
    chunk = _flat_chunk(
        chunk_bloom="understand", item_bloom="analyze", incoming_id=card_id,
    )
    # No exception, no capture kwarg -- must not raise.
    result = micro_preference_eligibility(chunk, focus=_focus("understand"))
    assert result["eligible"] is True


def test_rejected_candidate_never_emits_a_decision(monkeypatch):
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    chunk = _flat_chunk(
        chunk_bloom="understand", item_bloom="analyze", incoming_id="mc_deadbeef",
    )
    capture = _RecordingCapture()
    result = micro_preference_eligibility(
        chunk, focus=_focus("understand"), capture=capture,
    )
    assert result["eligible"] is False
    assert capture.calls == []


def test_flag_off_admission_emits_no_decision_byte_identical(monkeypatch):
    """Flag off, an ADMITTED candidate still emits nothing -- the capture
    stream (like every other behavior in this item) is gated on
    ``TRAINFORGE_BLOOM_WINDOWS``, so a flag-off run's decision JSONL is
    byte-identical to the pre-ladder contract."""
    monkeypatch.delenv(BLOOM_WINDOWS_ENV, raising=False)
    fallback_id = canonical_mc_id(_MISCONCEPTION, _CORRECTION, "understand")
    chunk = _flat_chunk(
        chunk_bloom="understand", item_bloom=None, incoming_id=fallback_id,
    )
    capture = _RecordingCapture()
    result = micro_preference_eligibility(
        chunk, focus=_focus("understand"), capture=capture,
    )
    assert result["eligible"] is True, result["reason"]
    assert capture.calls == []


# ---------------------------------------------------------------------------
# (a) target_rung threading is not dead code -- micro side
# ---------------------------------------------------------------------------


def _rung_block(block_id: str, text: str, bloom_level: str | None) -> str:
    attr = f' data-cf-bloom-level="{bloom_level}"' if bloom_level else ""
    return (
        f'<div data-cf-block-id="{block_id}" '
        f'data-cf-objective-id="{_OBJECTIVE_ID}"{attr}><p>{text}</p></div>'
    )


_CEILING_CLAIM = "the derived proof shows every intermediate step commutes"
_CEILING_CORRECTION = "the derived proof requires several non commuting steps"


def _two_rung_html_chunk() -> dict:
    allowed_text = "Every factor of twelve divides twelve evenly with no remainder."
    ceiling_text = f"{_CEILING_CLAIM}. {_CEILING_CORRECTION}."
    html = (
        _rung_block("blk-allowed", allowed_text, "understand")
        + _rung_block("blk-above-ceiling", ceiling_text, "create")
    )
    return {
        "id": "chunk-ladder",
        "html": html,
        "bloom_level": "understand",
        "misconceptions": [{
            "misconception": _CEILING_CLAIM,
            "correction": _CEILING_CORRECTION,
            "mechanism_evidence": _CEILING_CORRECTION,
        }],
    }


def test_flag_off_target_rung_never_filters_the_window(monkeypatch):
    monkeypatch.delenv(BLOOM_WINDOWS_ENV, raising=False)
    result = micro_preference_eligibility(
        _two_rung_html_chunk(), focus=_focus("understand"),
    )
    # The "create"-rung block is not excluded, so its claim/correction are
    # backed by the (unfiltered) evidence and the candidate is admitted.
    assert result["eligible"] is True, result["reason"]


def test_flag_on_target_rung_excludes_the_over_ceiling_block(monkeypatch):
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    result = micro_preference_eligibility(
        _two_rung_html_chunk(), focus=_focus("understand"),
    )
    # focus bloom_level="understand" is threaded as target_rung into
    # build_evidence_window; the "create"-rung block (above the ceiling) is
    # dropped from the evidence window, so the claim/correction that live
    # ONLY in that block are no longer source-backed.
    assert result["eligible"] is False
    assert result["telemetry"]["rejection_counts"]["source_unbacked"] == 1


# ---------------------------------------------------------------------------
# (a) target_rung threading is not dead code -- staged_synthesis_provider side
# ---------------------------------------------------------------------------


class _NoCallBaseProvider:
    _capture = None
    _model = "test"
    _provider_name = "local"
    _plan_nli_scorer = None


def test_plan_threads_the_objectives_own_rung_into_evidence_window(monkeypatch):
    calls: list[tuple] = []

    def fake_build_evidence_window(chunk, focus, target_rung=None):
        calls.append((chunk, focus, target_rung))
        raise ValueError("evidence unavailable (test double)")

    monkeypatch.setattr(
        provider_module, "build_evidence_window", fake_build_evidence_window,
    )
    provider = provider_module.StagedSynthesisProvider(_NoCallBaseProvider())
    chunk = {
        "id": "chunk-1",
        "text": "irrelevant",
        "learning_outcome_refs": ["co-1"],
        "bloom_level": "apply",
        "synthesis_focus_objective": {
            "id": "co-1",
            "statement": "Apply the divisibility rule to a given number.",
            "bloom_level": "apply",
        },
    }
    with pytest.raises(SynthesisProviderError) as caught:
        provider._plan(chunk, need_misconception=False)
    assert caught.value.code == "staged_evidence_window_unavailable"
    assert len(calls) == 1
    _chunk_arg, _focus_arg, target_rung = calls[0]
    assert target_rung == "apply"
