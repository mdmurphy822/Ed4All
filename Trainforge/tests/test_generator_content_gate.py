"""Regression net: the generator must never manufacture content-free pairs.

Three defects are pinned here, all instances of the project's
no-design-intent-fallbacks rule:

* FIX 1 — a chunk with no summary / key_terms / concept_tags must NOT produce
  a completion consisting only of the fixed Bloom tail.
* FIX 2 — the ``learning outcome {lo_ref}`` topic fallback must not fire; a
  database key is never a human-readable topic.
* FIX 3 — the deterministic pre-generation content gate admits a real prose
  chunk and excludes degenerate auto-generated MCQ residue.

Fixtures are written inline from the SHAPES observed in the audited corpora;
no course slug, corpus path, or book title is referenced.
"""

from __future__ import annotations

import pytest

from Trainforge.generators.pairs.instruction import (
    _build_completion,
    _derive_topic,
    synthesize_instruction_pair,
)
from Trainforge.generators.pairs import preference as preference_factory
from Trainforge.synthesis_eligibility import (
    CONTENT_GATE_ENV,
    MIN_PROSE_WORDS_ENV,
    MIN_STEM_CONTENT_WORDS_ENV,
    content_gate_eligibility,
    describe_content_sources,
    is_degenerate_stem,
    leading_stem,
    resolve_content_gate_enabled,
    resolve_min_prose_words,
    resolve_min_stem_content_words,
)


# --------------------------------------------------------------------------- #
# Inline fixtures — shapes only, never a real course's content.
# --------------------------------------------------------------------------- #

_REAL_PROSE = (
    "The distributive property states that multiplying a sum by a number "
    "gives the same result as multiplying each addend by that number and "
    "then adding the products. Expanding an expression in this way is the "
    "first move when simplifying, because it removes the grouping symbols "
    "and leaves a sum of like terms that can then be collected together."
)

# Verbatim SHAPES of the degenerate auto-generated MCQ stems found in the
# audited corpora: a template whose slot was filled with a sentence fragment
# rather than a concept, leaving a whitespace gap before the terminator.
_DEGENERATE_STEMS = [
    (
        "Compare and contrast Shown below and How . Explain how they differ "
        "and where they connect, using specific examples from the material."
    ),
    "Solve process . Show your work step by step and justify each step.",
    "Which definition best matches the term Because of this, it ?",
    "Identify the statement that best describes That .",
]


def _bare_chunk(**overrides):
    """A chunk with LO refs but no content source at all."""
    chunk = {
        "id": "chunk-0001",
        "text": "",
        "chunk_type": "explanation",
        "bloom_level": "analyze",
        "learning_outcome_refs": ["co-117"],
        "concept_tags": [],
        "key_terms": [],
    }
    chunk.update(overrides)
    return chunk


def _prose_chunk(**overrides):
    chunk = _bare_chunk(text=_REAL_PROSE, concept_tags=["distributive-property"])
    chunk.update(overrides)
    return chunk


# --------------------------------------------------------------------------- #
# FIX 1 — no Bloom-tail-only completion
# --------------------------------------------------------------------------- #


# The six fixed Bloom tails, verbatim, minus the {topic} slot. If a completion
# is nothing but one of these it is boilerplate, not training data.
_BLOOM_TAIL_MARKERS = (
    "Learners should be able to recall and restate this about",
    "Learners should be able to explain this about",
    "Learners should be able to use this about",
    "Learners should be able to break this down and explain the parts of",
    "Learners should be able to judge the quality of claims about",
    "Learners should be able to generate a fresh example or application of",
    "A proficient learner can",
)


def test_build_completion_returns_none_when_every_content_source_is_empty():
    import random

    for bloom in (
        "remember", "understand", "apply", "analyze", "evaluate", "create",
    ):
        out = _build_completion(
            _bare_chunk(bloom_level=bloom),
            topic="place value",
            bloom=bloom,
            content_type="explanation",
            rng=random.Random(0),
        )
        assert out is None, (
            f"bloom={bloom} produced a completion with no content source: {out!r}"
        )


def test_no_content_source_never_yields_a_bloom_tail_only_pair():
    # The audited failure mode: a chunk whose every content source is empty
    # used to emit the fixed Bloom tail with the topic slot filled.
    result = synthesize_instruction_pair(
        _bare_chunk(), seed=7, provider="mock",
    )
    assert result.pair is None
    assert result.quality.get("ineligible") is True
    assert len(result.rationale) >= 20


def test_emitted_completion_is_never_only_the_bloom_tail():
    result = synthesize_instruction_pair(
        _prose_chunk(summary="Multiplying a sum distributes over each addend."),
        seed=3,
        provider="mock",
    )
    assert result.pair is not None
    completion = result.pair["completion"]
    stripped = completion
    for marker in _BLOOM_TAIL_MARKERS:
        idx = stripped.find(marker)
        if idx != -1:
            stripped = stripped[:idx]
    assert stripped.strip(), (
        f"completion carried nothing but the Bloom tail: {completion!r}"
    )


def test_bloom_tail_only_unit_is_ineligible_not_rejected():
    result = synthesize_instruction_pair(_bare_chunk(), seed=1, provider="mock")
    assert result.pair is None
    # ``ineligible`` routes the caller to the ineligible disposition, keeping
    # the unit out of the quality-rejection denominator.
    assert result.quality["ineligible"] is True
    assert result.quality["reason"] in {
        "no_derivable_topic", "no_groundable_completion_content",
    }


# --------------------------------------------------------------------------- #
# FIX 2 — the LO-id topic fallback cannot fire
# --------------------------------------------------------------------------- #


def test_derive_topic_never_interpolates_a_learning_outcome_id():
    chunk = _bare_chunk(learning_outcome_refs=["co-117", "to-05"])
    topic = _derive_topic(chunk)
    assert topic == ""
    assert "co-117" not in topic
    assert "learning outcome" not in topic


def test_preference_derive_topic_never_interpolates_a_learning_outcome_id():
    chunk = _bare_chunk(learning_outcome_refs=["co-154"])
    assert preference_factory._derive_topic(chunk) == ""


@pytest.mark.parametrize("lo_ref", ["co-117", "to-05", "co-154"])
def test_no_emitted_pair_surface_contains_a_learning_outcome_id(lo_ref):
    chunk = _prose_chunk(
        learning_outcome_refs=[lo_ref],
        summary="Multiplying a sum distributes across each addend.",
    )
    result = synthesize_instruction_pair(chunk, seed=11, provider="mock")
    assert result.pair is not None
    blob = (result.pair["prompt"] + " " + result.pair["completion"]).lower()
    assert lo_ref not in blob
    assert "learning outcome" not in blob


def test_topicless_chunk_is_ineligible_rather_than_lo_id_filled():
    result = synthesize_instruction_pair(
        _bare_chunk(text=_REAL_PROSE), seed=5, provider="mock",
    )
    assert result.pair is None
    assert result.quality["reason"] == "no_derivable_topic"
    assert result.quality["ineligible"] is True
    # Rationale must carry real signals for the DecisionCapture the caller
    # emits on the skip.
    assert "chunk-0001" in result.rationale
    assert len(result.rationale) >= 20


def test_topicless_preference_chunk_is_ineligible():
    result = preference_factory.synthesize_preference_pair(
        _bare_chunk(text=_REAL_PROSE), seed=5, provider="mock",
    )
    assert result.pair is None
    assert result.quality["reason"] == "no_derivable_topic"
    assert result.quality["ineligible"] is True


# --------------------------------------------------------------------------- #
# FIX 3 — pre-generation content gate
# --------------------------------------------------------------------------- #


def test_content_gate_admits_a_real_prose_chunk():
    verdict = content_gate_eligibility(_prose_chunk())
    assert verdict.eligible, verdict


def test_content_gate_admits_prose_with_no_tags_or_terms():
    # The prose arm alone is enough: >= the word floor is groundable content.
    verdict = content_gate_eligibility(_bare_chunk(text=_REAL_PROSE))
    assert verdict.eligible, verdict


def test_content_gate_admits_a_short_chunk_that_carries_tags():
    verdict = content_gate_eligibility(
        _bare_chunk(
            text="Absolute value is a distance, so it is never negative here.",
            concept_tags=["absolute-value"],
        )
    )
    assert verdict.eligible, verdict


@pytest.mark.parametrize("stem", _DEGENERATE_STEMS)
def test_content_gate_excludes_degenerate_mcq_stems(stem):
    # concept_tags are present so the exclusion is attributable to the stem
    # check alone, not to the no-groundable-content arm.
    verdict = content_gate_eligibility(
        _bare_chunk(
            text=stem,
            chunk_type="assessment_item",
            concept_tags=["place-value"],
        )
    )
    assert not verdict.eligible
    assert verdict.reason == "degenerate_source_stem"
    assert verdict.detail and len(verdict.detail) >= 20


def test_content_gate_excludes_a_chunk_with_no_groundable_content():
    verdict = content_gate_eligibility(
        _bare_chunk(text="Step 2. See below.")
    )
    assert not verdict.eligible
    assert verdict.reason == "chunk_carries_no_groundable_content"
    # Detail names every empty source so the capture rationale can interpolate
    # which ones were missing.
    for token in ("concept_tags=", "key_terms=", "prose_words="):
        assert token in verdict.detail


def test_content_gate_does_not_judge_long_prose_on_its_opening_sentence():
    # Flattened OCR/math text routinely emits a spaced terminal period
    # mid-prose. A long instructional chunk must never be excluded for that.
    chunk = _bare_chunk(
        text="79 . " + _REAL_PROSE,
        concept_tags=["distributive-property"],
    )
    assert content_gate_eligibility(chunk).eligible


def test_content_gate_does_not_fire_on_math_notation_stems():
    chunk = _bare_chunk(
        text="Simplify: 2 x 2 + 3 x + 7 + x 2 + 4 x + 5 . " + _REAL_PROSE,
        concept_tags=["like-terms"],
    )
    assert content_gate_eligibility(chunk).eligible


# --------------------------------------------------------------------------- #
# Gate helpers + env plumbing
# --------------------------------------------------------------------------- #


def test_describe_content_sources_counts_each_axis():
    sources = describe_content_sources(
        {
            "text": "one two three",
            "concept_tags": ["a", "", "b"],
            "key_terms": [{"term": "x"}, {"term": ""}, {"definition": "y"}],
            "summary": " abc ",
        }
    )
    assert sources == {
        "concept_tags": 2,
        "key_terms": 1,
        "summary_chars": 3,
        "prose_words": 3,
    }


def test_leading_stem_returns_the_first_sentence():
    assert leading_stem("Solve process . Show your work.") == "Solve process ."
    assert leading_stem("") == ""


def test_is_degenerate_stem_signals():
    assert is_degenerate_stem("")
    assert is_degenerate_stem("Solve process .")
    assert is_degenerate_stem("Which definition best matches the term it ?")
    # A numeral or single-letter variable before a spaced period is math
    # notation, so the SLOT-GAP signal must not fire on it (the content-word
    # floor is evaluated separately, and the gate applies it to the whole
    # item rather than to this stem).
    assert not is_degenerate_stem(
        "Simplify 2 x 2 + 3 x + 5 .", min_content_words=0,
    )
    assert not is_degenerate_stem("79 .", min_content_words=0)
    assert not is_degenerate_stem(
        "The distributive property expands a grouped sum."
    )
    # min_content_words=0 disables the content-word floor.
    assert is_degenerate_stem("y < 3")
    assert not is_degenerate_stem("y < 3", min_content_words=0)


def test_content_gate_defaults_on_and_env_can_disable(monkeypatch):
    monkeypatch.delenv(CONTENT_GATE_ENV, raising=False)
    assert resolve_content_gate_enabled() is True

    degenerate = _bare_chunk(text="Step 2. See below.")
    assert not content_gate_eligibility(degenerate).eligible

    monkeypatch.setenv(CONTENT_GATE_ENV, "0")
    assert resolve_content_gate_enabled() is False
    assert content_gate_eligibility(degenerate).eligible

    # Garbage parses back to ON (parse-with-fallback).
    monkeypatch.setenv(CONTENT_GATE_ENV, "banana")
    assert resolve_content_gate_enabled() is True


@pytest.mark.parametrize(
    "raw,expected",
    [(None, 40), ("12", 12), ("0", 40), ("-3", 40), ("banana", 40), ("", 40)],
)
def test_min_prose_words_parse_with_fallback(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(MIN_PROSE_WORDS_ENV, raising=False)
    else:
        monkeypatch.setenv(MIN_PROSE_WORDS_ENV, raw)
    assert resolve_min_prose_words() == expected


@pytest.mark.parametrize(
    "raw,expected",
    [(None, 3), ("5", 5), ("0", 3), ("nope", 3)],
)
def test_min_stem_content_words_parse_with_fallback(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(MIN_STEM_CONTENT_WORDS_ENV, raising=False)
    else:
        monkeypatch.setenv(MIN_STEM_CONTENT_WORDS_ENV, raw)
    assert resolve_min_stem_content_words() == expected


def test_min_prose_words_env_moves_the_admission_boundary(monkeypatch):
    short = _bare_chunk(
        text=(
            "Absolute value measures distance from zero along the number "
            "line, so the result is never a negative quantity at all."
        )
    )
    monkeypatch.setenv(MIN_PROSE_WORDS_ENV, "10")
    assert content_gate_eligibility(short).eligible
    monkeypatch.setenv(MIN_PROSE_WORDS_ENV, "500")
    assert not content_gate_eligibility(short).eligible


# --------------------------------------------------------------------------- #
# DecisionCapture on the skipped (ineligible) unit
# --------------------------------------------------------------------------- #


class _RecordingCapture:
    """Minimal DecisionCapture stand-in that records log_decision kwargs."""

    def __init__(self):
        self.events = []

    def log_decision(self, **kwargs):
        self.events.append(kwargs)
        return f"evt-{len(self.events)}"


def test_ineligible_skip_emits_a_capture_with_interpolated_signals():
    from Trainforge.synthesize_training import (
        SynthesisStats,
        _record_ineligible_disposition,
    )

    verdict = content_gate_eligibility(_bare_chunk(text="Step 2. See below."))
    assert not verdict.eligible

    stats = SynthesisStats()
    capture = _RecordingCapture()
    _record_ineligible_disposition(
        stats=stats,
        checkpoint_fh=None,
        capture=capture,
        chunk_id="chunk-0001",
        kind="instruction",
        variant_index=0,
        provider="local",
        seed=42,
        reason=verdict.reason,
        detail=verdict.detail,
        contract_fingerprint="deadbeef",
    )

    assert stats.instruction_pairs_ineligible == 1
    assert stats.ineligible_reasons == {
        "instruction:chunk_carries_no_groundable_content": 1,
    }
    assert len(capture.events) == 1
    event = capture.events[0]
    assert event["decision_type"] == "instruction_pair_synthesis"
    rationale = event["rationale"]
    assert len(rationale) >= 20
    # Real per-unit signals, not a static boilerplate rationale.
    assert "chunk-0001" in rationale
    assert "chunk_carries_no_groundable_content" in rationale
    assert "concept_tags=0" in rationale
    assert "key_terms=0" in rationale
    assert "prose_words=" in rationale
    assert "chunk-0001" in event["context"]
