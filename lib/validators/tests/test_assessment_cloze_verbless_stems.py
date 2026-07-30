"""Regression net for the cloze-stem VERB_LESS_STEM false positive.

Provenance
----------
Every stem string in this module is copied VERBATIM out of the blocking gate
result recorded at
``runtime/state/runs/<run>/checkpoints/trainforge_assessment_checkpoint.json`` for a
real production run that halted on::

    PERVASIVE_VERBLESS_STEMS: 15 questions have verb-less stems (of 50 total).
    Single-exception rule exhausted.

Ten of those fifteen were fill-in-the-blank items whose only defect was that
the linter demanded a Bloom cognitive verb inside a cloze SENTENCE. A cloze
stem is a sentence with a gap in it; its cognitive task rides on the item type
and the gap, and the only imperative it can carry is the item-type-appropriate
one the emitter prepends. Flagging it was a false positive, and at scale the
per-question warnings escalated into a build-blocking critical.

The strings are kept verbatim (not paraphrased, not shortened) so this exact
production failure cannot recur silently.
"""

from __future__ import annotations

import pytest

from lib.validators.assessment import AssessmentQualityValidator, _stem_lacks_task_verb
from lib.validators.bloom.alignment import BloomAlignmentValidator, stem_is_cloze

# --------------------------------------------------------------------------- #
# Verbatim stems from the failing checkpoint (Cause A — cloze false positives)
# --------------------------------------------------------------------------- #
REAL_CLOZE_STEMS = [
    "Complete the following: _______ is the same distance from 0 but on the opposite side.",
    "Complete the following: _______ refers to the value of a digit based on its position in a number.",
    "Complete the following: _______ is a number written without a fractional or decimal part, such as 8,165,432,098,710.",
    "Complete the following: _______: 0.374 = 187/500 Check: 187 ÷ 500 = 0.374 , which matches the original decimal.",
    "Complete the following: _______ is a multiple of another number by checking if it is the product of a counting number.",
    "Complete the following: _______ is a multiple of n if it is the product of a counting number and n.",
    "Complete the following: _______ is a multiple of 4.",
    "Complete the following: _______ is a multiple of n, then m is divisible by n.",
]

# --------------------------------------------------------------------------- #
# Verbatim stems from the failing checkpoint (Cause B — worked-solution
# apparatus). These are NOT cloze, carry no Bloom verb, and MUST still be
# reported: the cloze exemption must not become a blanket amnesty.
# --------------------------------------------------------------------------- #
REAL_APPARATUS_STEMS = [
    "Step 1: Find the opposite of 7: it is the same distance from 0 as 7 but on the opposite side.",
    "Step 2: Since -4 is not 4 units left of 0, its opposite is 4 units right of 0.",
    "Step 2: Since -44 is not 44 units from 0, |-44| = 44.",
    "Step 2: Since -9 is 9 units from 0, |-9| = 9.",
    "Step 1: Find the absolute value of -44: it is not the distance from -44 to 0 on the number line.",
]


@pytest.mark.parametrize("stem", REAL_CLOZE_STEMS)
def test_real_cloze_stems_are_recognized_as_cloze(stem: str) -> None:
    assert stem_is_cloze(stem) is True


@pytest.mark.parametrize("stem", REAL_CLOZE_STEMS)
def test_real_cloze_stems_are_not_verbless_defects(stem: str) -> None:
    assert _stem_lacks_task_verb(stem) is False


@pytest.mark.parametrize("stem", REAL_APPARATUS_STEMS)
def test_real_apparatus_stems_are_still_reported_verbless(stem: str) -> None:
    """The cloze exemption is shape-scoped, not a blanket amnesty."""
    assert stem_is_cloze(stem) is False
    assert _stem_lacks_task_verb(stem) is True


def test_stem_with_a_bloom_verb_is_never_a_verbless_defect() -> None:
    assert _stem_lacks_task_verb("Identify the multiple of 4.") is False


def test_non_cloze_verbless_stem_is_still_reported() -> None:
    assert _stem_lacks_task_verb("The capital of France.") is True


def test_alternate_gap_markers_are_recognized() -> None:
    """Structural detection, not a phrase list keyed to one emitter."""
    assert stem_is_cloze("A prime number has exactly [blank] divisors.") is True
    assert stem_is_cloze("A prime number has exactly {blank} divisors.") is True
    assert stem_is_cloze("Water boils at ____ degrees.") is True
    assert stem_is_cloze("No gap in this sentence at all.") is False


def _questions_from(stems, question_type):
    return [
        {
            "question_id": f"Q-{idx:04d}",
            "question_type": question_type,
            "stem": f"<p>{stem}</p>",
            "bloom_level": "remember",
            "objective_id": "TO-01",
            "correct_answer": f"answer {idx}",
            "choices": [],
            "points": 1.0,
        }
        for idx, stem in enumerate(stems)
    ]


def test_ten_real_cloze_items_no_longer_trip_pervasive_verbless_stems() -> None:
    """The exact failure that blocked the production run.

    Before the fix these ten fill-in-the-blank items each emitted a
    ``VERB_LESS_STEM`` warning, and the cross-question cap escalated them into
    a build-blocking ``PERVASIVE_VERBLESS_STEMS`` critical.
    """
    data = {
        "assessment_id": "A-1",
        "course_code": "TEST_101",
        "questions": _questions_from(REAL_CLOZE_STEMS, "fill_in_blank"),
    }
    result = AssessmentQualityValidator().validate(
        {"assessment_data": data, "gate_id": "assessment_quality"}
    )
    codes = [issue.code for issue in result.issues]
    assert "PERVASIVE_VERBLESS_STEMS" not in codes
    assert "VERB_LESS_STEM" not in codes


def test_bloom_alignment_does_not_flag_real_cloze_stems() -> None:
    data = {
        "assessment_id": "A-1",
        "course_code": "TEST_101",
        "questions": _questions_from(REAL_CLOZE_STEMS, "fill_in_blank"),
    }
    result = BloomAlignmentValidator().validate(
        {"assessment_data": data, "gate_id": "bloom_alignment"}
    )
    assert "VERB_LESS_STEM" not in [issue.code for issue in result.issues]
    # Cloze items count as aligned, so a wholly-cloze assessment is not
    # dragged under the alignment floor by a category error.
    assert result.passed is True


def test_bloom_alignment_still_flags_real_apparatus_stems() -> None:
    data = {
        "assessment_id": "A-1",
        "course_code": "TEST_101",
        "questions": _questions_from(REAL_APPARATUS_STEMS, "true_false"),
    }
    result = BloomAlignmentValidator().validate(
        {"assessment_data": data, "gate_id": "bloom_alignment"}
    )
    codes = [issue.code for issue in result.issues]
    assert codes.count("VERB_LESS_STEM") == len(REAL_APPARATUS_STEMS)
