"""IB2.3 — Block 8-dimension quality-rubric DATA MODEL (empty scaffold).

Pins the contracts of the rubric SHAPE this wave stands up (no scoring —
IB6 fills it):

  1. ``QUALITY_DIMENSIONS`` is the eight ordered dimension names;
     ``CORE_QUALITY_DIMENSIONS`` is the load-bearing four ⊂ the eight.
  2. ``QualityScore`` validates its dimension + anchored 0–3 score.
  3. ``Block.quality_rubric`` defaults to ``()`` and is EXCLUDED from
     ``compute_content_hash()`` (the byte-stability proof, mirroring
     ``observed_bloom_level``).
  4. ``Block.to_jsonld_entry`` emits ``qualityRubric`` only when non-empty
     (so emit stays byte-identical this wave) and ``quality_score_for``
     reads the tuple.
"""

from __future__ import annotations

import dataclasses

import pytest

from Courseforge.scripts.blocks import (
    CORE_QUALITY_DIMENSIONS,
    QUALITY_DIMENSIONS,
    Block,
    QualityScore,
)


def _make_block(**overrides) -> Block:
    """Minimal Block at a minimal-block-jsonld block_type (prereq_set)."""
    base = dict(
        block_id="week_01_overview#prereq_set_intro_0",
        block_type="prereq_set",
        page_id="week_01_overview",
        sequence=0,
        content="Prereq set content.",
    )
    base.update(overrides)
    return Block(**base)


# ---------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------- #


def test_quality_dimensions_eight_and_core_four():
    assert len(QUALITY_DIMENSIONS) == 8
    assert QUALITY_DIMENSIONS == (
        "alignment",
        "cognitive_load",
        "multimedia",
        "retrieval",
        "feedback",
        "engagement",
        "accessibility",
        "coherence",
    )
    assert len(CORE_QUALITY_DIMENSIONS) == 4
    assert CORE_QUALITY_DIMENSIONS < set(QUALITY_DIMENSIONS)
    assert CORE_QUALITY_DIMENSIONS == {
        "alignment",
        "cognitive_load",
        "accessibility",
        "coherence",
    }


# ---------------------------------------------------------------------- #
# QualityScore validation
# ---------------------------------------------------------------------- #


def test_quality_score_validates_dimension_and_score():
    with pytest.raises(ValueError):
        QualityScore("bogus", 1)
    with pytest.raises(ValueError):
        QualityScore("alignment", 5)


@pytest.mark.parametrize("score", [None, 0, 1, 2, 3])
def test_quality_score_accepts_valid_scores(score):
    qs = QualityScore("alignment", score)
    assert qs.score == score


def test_quality_score_defaults():
    qs = QualityScore("coherence")
    assert qs.score is None
    assert qs.applicable is True
    assert qs.rationale is None


def test_quality_score_to_jsonld_omits_none_rationale():
    assert "rationale" not in QualityScore("alignment", 2).to_jsonld()
    assert QualityScore("alignment", 2, True, "why").to_jsonld()["rationale"] == "why"


# ---------------------------------------------------------------------- #
# Block.quality_rubric — default + hash exclusion + projection
# ---------------------------------------------------------------------- #


def test_quality_rubric_defaults_empty():
    assert _make_block().quality_rubric == ()


def test_quality_rubric_excluded_from_hash():
    """Two Blocks identical but for the rubric hash equally (byte-stability)."""
    base = _make_block()
    extended = dataclasses.replace(
        base,
        quality_rubric=(
            QualityScore("alignment", 3, True, "strong"),
            QualityScore("coherence", None, False),
        ),
    )
    assert base.compute_content_hash() == extended.compute_content_hash()


def test_quality_rubric_omitted_from_jsonld_when_empty():
    assert "qualityRubric" not in _make_block().to_jsonld_entry()


def test_quality_rubric_emitted_when_populated():
    block = _make_block(
        quality_rubric=(QualityScore("alignment", 2, True, "ok"),)
    )
    rubric = block.to_jsonld_entry()["qualityRubric"]
    assert rubric == [
        {"dimension": "alignment", "score": 2, "applicable": True, "rationale": "ok"}
    ]


def test_quality_score_for_reads_tuple():
    block = _make_block(
        quality_rubric=(QualityScore("alignment", 2), QualityScore("coherence", 1))
    )
    assert block.quality_score_for("alignment").score == 2
    assert block.quality_score_for("coherence").score == 1
    assert block.quality_score_for("multimedia") is None
