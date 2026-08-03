"""Feature 2 / PRONG C — bloom-complement invariants (hermetic fake provider).

Verifies: (1) default-OFF is a byte-stable no-op (no dispatch, canonical
untouched), (2) provider-None / share-already-met short-circuits, (3) a grounded
analyze complement is ADDED and the higher-order share rises (share math),
(4) anti-fabrication — a candidate citing a FOREIGN chunk id is dropped whole,
(5) a below-target-level (create / understand) candidate is dropped, (6) the
dedup helper drops a near-duplicate + the echo helper drops a same-skill
restatement, (7) the LLM call site fires DecisionCapture, (8) the
parse-with-fallback resolvers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.objectives.bloom_complement import (  # noqa: E402
    _echoes_existing,
    _near_duplicate,
    complement_bloom_profile,
    higher_order_share,
    resolve_bloom_complement,
    resolve_max_additions,
    resolve_min_share,
)
from lib.objectives.tests._fakes import FakeEmbed  # noqa: E402


class _Capture:
    def __init__(self) -> None:
        self.events = []

    def log_decision(self, **kwargs) -> None:  # noqa: ANN003
        self.events.append(kwargs)


class _FakeProvider:
    """Scripted provider stub exposing the ``_dispatch_call`` seam.

    ``responses`` is a queue of payload dicts; each dispatch pops one and
    returns ``(json_text, 0)``. An exhausted queue returns ``"{}"``.
    """

    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self._provider = "local"
        self._model = "qwen2.5:7b-instruct-q4_K_M"
        self.calls = 0

    def _dispatch_call(self, prompt, *, extra_payload=None):  # noqa: ANN001
        self.calls += 1
        if self._responses:
            payload = self._responses.pop(0)
        else:
            payload = {}
        return json.dumps(payload), 0


def _co(statement, chunk_ids, bloom_level="apply"):
    return {
        "statement": statement,
        "bloom_level": bloom_level,
        "source_chunk_ids": list(chunk_ids),
        "chapter_id": "ch1",
    }


def _chunks(*ids):
    return {
        cid: {
            "id": cid,
            "chapter_id": "ch1",
            "chunk_type": "explanation",
            "text": (
                "The distributive property lets you expand a product over a "
                "sum by multiplying each addend, a foundational algebra idea. "
                * 4
            ),
        }
        for cid in ids
    }


def _resp(statement, chunk_ids, bloom_level="analyze"):
    return {
        "complement_objectives": [
            {
                "statement": statement,
                "bloom_level": bloom_level,
                "bloom_verb": bloom_level,
                "source_chunk_ids": list(chunk_ids),
            }
        ]
    }


# ---------------------------------------------------------------------------
# Off / short-circuit
# ---------------------------------------------------------------------------


def test_flag_off_noop_bytestable():
    """enabled=None + env unset → no dispatch, canonical untouched."""
    canonical = [_co("Simplify fractions", ["c1"])]
    before = [dict(c) for c in canonical]
    prov = _FakeProvider([_resp("Analyze the fraction model", ["c1"])])
    result = complement_bloom_profile(
        canonical=canonical,
        chunks_by_id=_chunks("c1"),
        provider=prov,
        course_name="ALG",
        enabled=None,  # env unset → OFF
    )
    assert result.available is False
    assert prov.calls == 0
    assert canonical == before


def test_provider_none_noop():
    canonical = [_co("Simplify fractions", ["c1"])]
    result = complement_bloom_profile(
        canonical=canonical,
        chunks_by_id=_chunks("c1"),
        provider=None,
        course_name="ALG",
        enabled=True,
    )
    assert result.available is False
    assert len(canonical) == 1


def test_share_already_met_no_call():
    """A course already above the higher-order floor makes no LLM call."""
    canonical = [
        _co("Simplify fractions", ["c1"], bloom_level="apply"),
        _co("Analyze the fraction model", ["c1"], bloom_level="analyze"),
    ]
    prov = _FakeProvider([_resp("Evaluate two methods", ["c1"])])
    result = complement_bloom_profile(
        canonical=canonical,
        chunks_by_id=_chunks("c1"),
        provider=prov,
        course_name="ALG",
        enabled=True,
        min_share=0.15,
    )
    assert result.available is True
    assert prov.calls == 0
    assert result.cos_added == 0


# ---------------------------------------------------------------------------
# Happy path — grounded analyze complement added
# ---------------------------------------------------------------------------


def test_adds_grounded_analyze_complement_and_share_rises():
    canonical = [
        _co("Simplify fractions using common factors", ["c1"], bloom_level="apply"),
        _co("Add and subtract fractions", ["c2"], bloom_level="apply"),
    ]
    assert higher_order_share(canonical) == 0.0
    cap = _Capture()
    prov = _FakeProvider([
        _resp(
            "Analyze how changing the numerator affects a fraction's magnitude",
            ["c1"],
        ),
    ])
    result = complement_bloom_profile(
        canonical=canonical,
        chunks_by_id=_chunks("c1", "c2"),
        provider=prov,
        course_name="ALG",
        capture=cap,
        embed=FakeEmbed(),
        enabled=True,
        min_share=0.15,
    )
    assert result.available is True
    assert result.cos_added == 1
    assert result.share_after > result.share_before
    added = [c for c in canonical if c.get("bloom_complement")]
    assert len(added) == 1
    assert added[0]["bloom_level"] == "analyze"
    assert added[0]["source_chunk_ids"] == ["c1"]
    # Capture fired on the LLM call path (regression).
    assert any(
        e["decision_type"] == "objective_bloom_complement" for e in cap.events
    )
    event = next(
        e for e in cap.events
        if e["decision_type"] == "objective_bloom_complement"
    )
    alternatives = event["alternatives_considered"]
    assert all(a["option"] and len(a["reason_rejected"]) >= 20 for a in alternatives)
    assert all(isinstance(a.get("score"), (int, float)) for a in alternatives)


# ---------------------------------------------------------------------------
# Anti-fabrication + target-level filter
# ---------------------------------------------------------------------------


def test_foreign_chunk_id_candidate_dropped():
    """A candidate citing an id NOT in the offered set is dropped whole."""
    canonical = [_co("Simplify fractions", ["c1"], bloom_level="apply")]
    prov = _FakeProvider([
        _resp("Analyze the geometry proof", ["c_foreign"]),  # c_foreign not offered
    ])
    result = complement_bloom_profile(
        canonical=canonical,
        chunks_by_id=_chunks("c1"),
        provider=prov,
        course_name="ALG",
        embed=FakeEmbed(),
        enabled=True,
        max_additions=2,
    )
    assert result.cos_added == 0
    assert not any(c.get("bloom_complement") for c in canonical)


def test_below_target_level_candidate_dropped():
    """An understand-level candidate is dropped (must target analyze/evaluate)."""
    canonical = [_co("Simplify fractions", ["c1"], bloom_level="apply")]
    prov = _FakeProvider([
        _resp("Describe the fraction model", ["c1"], bloom_level="analyze"),
    ])
    # Statement main verb is 'describe' (understand) despite the analyze label —
    # honesty check drops it.
    result = complement_bloom_profile(
        canonical=canonical,
        chunks_by_id=_chunks("c1"),
        provider=prov,
        course_name="ALG",
        embed=FakeEmbed(),
        enabled=True,
        max_additions=2,
    )
    assert result.cos_added == 0


def test_create_level_candidate_dropped():
    """A create-verb candidate is dropped (create is unassessable here)."""
    canonical = [_co("Simplify fractions", ["c1"], bloom_level="apply")]
    prov = _FakeProvider([
        _resp("Design a brand-new number system", ["c1"], bloom_level="create"),
    ])
    result = complement_bloom_profile(
        canonical=canonical,
        chunks_by_id=_chunks("c1"),
        provider=prov,
        course_name="ALG",
        embed=FakeEmbed(),
        enabled=True,
        max_additions=2,
    )
    assert result.cos_added == 0


# ---------------------------------------------------------------------------
# Dedup + echo helpers (unit)
# ---------------------------------------------------------------------------


def test_near_duplicate_helper():
    embed = FakeEmbed()
    existing = ["Analyze the linear equation structure carefully"]
    vecs = embed.encode_batch(existing)
    # Same statement → cosine 1.0 ≥ dedup threshold → near-dup.
    assert _near_duplicate(
        "Analyze the linear equation structure carefully", vecs, embed
    ) is True
    # Disjoint statement → not a near-dup.
    assert _near_duplicate(
        "Evaluate a completely unrelated biology hypothesis", vecs, embed
    ) is False
    # Embed absent → no dedup (echo is the fallback guard).
    assert _near_duplicate("anything", vecs, None) is False


def test_echoes_existing_helper():
    from lib.objectives.objective_dedup import _skill_signature

    existing = [{"statement": "Simplify algebraic expressions using properties"}]
    sigs = [_skill_signature(e) for e in existing if _skill_signature(e)]
    # Same skill (simplify expressions) → echo.
    assert _echoes_existing(
        {"statement": "Simplify the algebraic expressions with properties"}, sigs
    ) is True
    # Distinct skill → not an echo.
    assert _echoes_existing(
        {"statement": "Analyze data-set variance patterns"}, sigs
    ) is False


def test_integrated_dedup_rejection():
    """A candidate near-duplicating an existing analyze CO is rejected."""
    canonical = [
        _co("Simplify fractions", ["c1"], bloom_level="apply"),
        _co(
            "Analyze how the fraction denominator scales the value",
            ["c1"],
            bloom_level="analyze",
        ),
    ]
    # share = 1/2 = 0.5; force topping-up with a high floor so the pass runs.
    prov = _FakeProvider([
        _resp(
            "Analyze how the fraction denominator scales the value",  # dup
            ["c1"],
        ),
    ])
    result = complement_bloom_profile(
        canonical=canonical,
        chunks_by_id=_chunks("c1"),
        provider=prov,
        course_name="ALG",
        embed=FakeEmbed(),
        enabled=True,
        min_share=0.99,
        max_additions=1,
    )
    assert result.cos_added == 0
    assert result.candidates_rejected >= 1


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


def test_resolvers_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT", raising=False)
    monkeypatch.delenv("ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MIN_SHARE", raising=False)
    monkeypatch.delenv("ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MAX", raising=False)
    assert resolve_bloom_complement() is False
    assert resolve_min_share() == 0.15
    assert resolve_max_additions() == 8

    monkeypatch.setenv("ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT", "on")
    assert resolve_bloom_complement() is True

    monkeypatch.setenv("ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MIN_SHARE", "0.4")
    assert resolve_min_share() == 0.4
    monkeypatch.setenv("ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MIN_SHARE", "banana")
    assert resolve_min_share() == 0.15
    monkeypatch.setenv("ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MIN_SHARE", "9")  # >1
    assert resolve_min_share() == 0.15

    monkeypatch.setenv("ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MAX", "3")
    assert resolve_max_additions() == 3
    monkeypatch.setenv("ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MAX", "-2")
    assert resolve_max_additions() == 8
    monkeypatch.setenv("ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MAX", "0")  # measure-only
    assert resolve_max_additions() == 0
