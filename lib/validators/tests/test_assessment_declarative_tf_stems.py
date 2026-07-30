"""Regression net for the declarative true/false VERB_LESS_STEM false positive.

Provenance
----------
After the cloze fix + the first apparatus-guard widening landed (07fa61eb), the
production run regenerated and the ``assessment_quality`` gate at
``trainforge_assessment`` still failed at score 0.75 on five accumulated
warnings. Four were ``VERB_LESS_STEM``, verbatim from
``runtime/state/runs/<run>/checkpoints/trainforge_assessment_checkpoint.json``::

    Q-cbc366e9: 'Show solution The opposite of 7 is -7 because it is the same distance from 0 but...'
    Q-5a4deb7d: 'The opposite of -10 is 10 because it is the same distance from 0 but on the oppo...'
    Q-c17ddc7e: 'Key Idea: Each digit in a whole number has a place value based on its position, ...'
    Q-a0d6406d: 'This is determined by its position as the third digit from the right.'

They split three ways, and this module pins all three outcomes:

* ``Q-5a4deb7d`` is a well-formed declarative PROPOSITION on a true/false
  item. Like a cloze stem, its cognitive demand rides on the item type, so it
  can never carry a Bloom verb — flagging it was a category error. EXEMPT.
* ``Q-cbc366e9`` / ``Q-c17ddc7e`` lead with pedagogical apparatus labels
  (``Show solution``, ``Key Idea:``). Those are mining defects fixed at the
  harvest layer; the linter must keep reporting them. STILL WARNS.
* ``Q-a0d6406d`` is a dangling anaphoric fragment with no antecedent —
  unanswerable standalone. Also a mining defect. STILL WARNS.

The exemption must not degenerate into "any true/false stem is fine", so the
"still warns" half of this module is as load-bearing as the exempt half.
"""

from __future__ import annotations

import pytest

from lib.validators.assessment import AssessmentQualityValidator, _stem_lacks_task_verb
from lib.validators.bloom.alignment import (
    BloomAlignmentValidator,
    stem_is_declarative_proposition,
)

# --------------------------------------------------------------------------- #
# Verbatim from the failing checkpoint.
# --------------------------------------------------------------------------- #
REAL_DECLARATIVE_TF_STEM = (
    "The opposite of -10 is 10 because it is the same distance from 0 but on "
    "the opposite side."
)

REAL_APPARATUS_TF_STEMS = [
    "Show solution The opposite of 7 is -7 because it is the same distance "
    "from 0 but on the opposite side.",
    "Key Idea: Each digit in a whole number has a place value based on its "
    "position, such as trillions, billions, millions, thousands, hundreds, "
    "tens, or ones.",
]

REAL_ANAPHORIC_TF_STEM = (
    "This is determined by its position as the third digit from the right."
)


def _q(stem: str, question_type: str = "true_false", idx: int = 0):
    return {
        "question_id": f"Q-{idx:04d}",
        "question_type": question_type,
        "stem": f"<p>{stem}</p>",
        "bloom_level": "remember",
        "objective_id": "TO-01",
        "correct_answer": "True" if idx % 2 == 0 else "False",
        "choices": [
            {"text": "True", "is_correct": idx % 2 == 0},
            {"text": "False", "is_correct": idx % 2 != 0},
        ],
        "points": 1.0,
    }


# --------------------------------------------------------------------------- #
# EXEMPT — the category error being corrected.
# --------------------------------------------------------------------------- #

def test_real_declarative_tf_stem_is_a_proposition() -> None:
    assert stem_is_declarative_proposition(REAL_DECLARATIVE_TF_STEM) is True


def test_real_declarative_tf_stem_is_not_a_verbless_defect() -> None:
    stem = REAL_DECLARATIVE_TF_STEM
    assert _stem_lacks_task_verb(stem, _q(stem)) is False


def test_exemption_requires_the_true_false_item_type() -> None:
    """The identical stem on a multiple-choice item still warns.

    An MCQ stem SHOULD carry a task verb; only the true/false instrument
    carries its demand in the type.
    """
    stem = REAL_DECLARATIVE_TF_STEM
    assert _stem_lacks_task_verb(stem, _q(stem, "multiple_choice")) is True
    assert _stem_lacks_task_verb(stem, _q(stem, "short_answer")) is True


def test_exemption_is_inert_without_the_question_dict() -> None:
    """A bare-stem caller gets only the cloze exemption (back-compat)."""
    assert _stem_lacks_task_verb(REAL_DECLARATIVE_TF_STEM) is True


# --------------------------------------------------------------------------- #
# STILL WARNS — the exemption is not a blanket amnesty.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stem", REAL_APPARATUS_TF_STEMS)
def test_real_apparatus_tf_stems_still_warn(stem: str) -> None:
    assert stem_is_declarative_proposition(stem) is False
    assert _stem_lacks_task_verb(stem, _q(stem)) is True


def test_real_anaphoric_tf_stem_still_warns() -> None:
    """A dangling anaphor has no antecedent — a defect, not a shape."""
    assert stem_is_declarative_proposition(REAL_ANAPHORIC_TF_STEM) is False
    assert _stem_lacks_task_verb(REAL_ANAPHORIC_TF_STEM, _q(REAL_ANAPHORIC_TF_STEM)) is True


@pytest.mark.parametrize(
    "stem",
    [
        # Imperative exercise directive mis-shaped as a T/F stem: has a finite
        # verb ("is") but opens on a bare verb, not a subject.
        "Find three consecutive integers whose sum is -36.",
        "Write the inequality shown by the graph.",
        # Fragments: no finite verb / no terminal punctuation.
        "Step 2",
        "The capital of France.",
        "The opposite of -10 is 10 but on the opposite side",
        # Interrogative.
        "Is the opposite of -10 equal to 10?",
        # Worked-solution apparatus.
        "Step 2: Since -9 is 9 units from 0, |-9| = 9.",
    ],
)
def test_malformed_tf_stems_are_not_exempted(stem: str) -> None:
    assert stem_is_declarative_proposition(stem) is False
    assert _stem_lacks_task_verb(stem, _q(stem)) is True


# --------------------------------------------------------------------------- #
# Gate-level: the real four-warning population.
# --------------------------------------------------------------------------- #

def test_gate_reports_exactly_the_three_real_defects() -> None:
    """One exemption, three retained findings — measured on the real strings."""
    stems = [
        REAL_DECLARATIVE_TF_STEM,          # exempt
        *REAL_APPARATUS_TF_STEMS,          # 2 x still warns
        REAL_ANAPHORIC_TF_STEM,            # still warns
    ]
    data = {
        "assessment_id": "A-1",
        "course_code": "TEST_101",
        "questions": [_q(s, idx=i) for i, s in enumerate(stems)],
    }
    result = AssessmentQualityValidator().validate(
        {"assessment_data": data, "gate_id": "assessment_quality"}
    )
    codes = [i.code for i in result.issues]
    assert codes.count("VERB_LESS_STEM") == 3
    # All four are true_false, so the cross-question cap's T/F bucket never
    # escalates on its own — the critical must not fire.
    assert "PERVASIVE_VERBLESS_STEMS" not in codes


def test_bloom_alignment_exempts_the_declarative_tf_stem() -> None:
    data = {
        "assessment_id": "A-1",
        "course_code": "TEST_101",
        "questions": [_q(REAL_DECLARATIVE_TF_STEM)],
    }
    result = BloomAlignmentValidator().validate(
        {"assessment_data": data, "gate_id": "bloom_alignment"}
    )
    assert "VERB_LESS_STEM" not in [i.code for i in result.issues]
    assert result.passed is True


def test_bloom_alignment_still_flags_the_anaphoric_tf_stem() -> None:
    data = {
        "assessment_id": "A-1",
        "course_code": "TEST_101",
        "questions": [_q(REAL_ANAPHORIC_TF_STEM)],
    }
    result = BloomAlignmentValidator().validate(
        {"assessment_data": data, "gate_id": "bloom_alignment"}
    )
    assert "VERB_LESS_STEM" in [i.code for i in result.issues]
