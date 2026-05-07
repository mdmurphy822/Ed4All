"""Wave 6 W6.A — per-question-type quality threshold matrix.

Pins the per-question-type threshold table on ``AssessmentQualityValidator``,
the operator-overlay resolution chain, the QTI ``type:`` field
normalization (W4.B Lesson 1 defense), and the warning-severity
day-1 emit on per-type diversity sub-checks. Mirrors the W1.7.C
``_PER_BLOCK_TYPE_ENTAILMENT_FLOOR`` and W4.C MEDIUM
``_PER_PAIR_KIND_ENTAILMENT_FLOOR`` test patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.assessment import (  # noqa: E402
    AssessmentQualityValidator,
    _DEFAULT_QUESTION_TYPE_THRESHOLDS,
    _PER_QUESTION_TYPE_THRESHOLDS,
    _normalize_question_type,
    _resolve_per_type_thresholds,
    _thresholds_for_type,
)


# --------------------------------------------------------------------- #
# Test fixture helpers
# --------------------------------------------------------------------- #


def _essay(qid: str, stem: str, answer: str = "An essay-length answer with multiple paragraphs covering the topic.") -> dict:
    """Build an essay question dict (no choices)."""
    return {
        "question_id": qid,
        "question_type": "essay",
        "stem": stem,
        "bloom_level": "evaluate",
        "objective_id": "LO-01",
        "correct_answer": answer,
    }


def _mc(qid: str, stem: str, correct: str, distractors=None) -> dict:
    """Build an MC question dict with 3 distractors."""
    if distractors is None:
        distractors = [f"Distractor {qid}-{i}" for i in range(3)]
    choices = [{"text": correct, "is_correct": True}]
    for d in distractors:
        choices.append({"text": d, "is_correct": False})
    return {
        "question_id": qid,
        "question_type": "multiple_choice",
        "stem": stem,
        "bloom_level": "understand",
        "objective_id": "LO-01",
        "choices": choices,
    }


# --------------------------------------------------------------------- #
# Test 1 — module-level table loaded with 5 entries
# --------------------------------------------------------------------- #


def test_per_type_thresholds_loaded() -> None:
    """The per-question-type threshold table must carry exactly 5 entries
    (multiple_choice / true_false / short_answer / essay / fill_in_blank).
    Each entry must carry the 4 required threshold axes
    (stem_diversity / correct_answer_diversity /
    distractor_template_max_ratio / min_stem_chars).

    Mirror of the W1.7.C
    ``test_per_block_type_entailment_floor_table_has_6_entries`` shape.
    """
    assert set(_PER_QUESTION_TYPE_THRESHOLDS.keys()) == {
        "multiple_choice",
        "true_false",
        "short_answer",
        "essay",
        "fill_in_blank",
    }
    required_axes = {
        "stem_diversity",
        "correct_answer_diversity",
        "distractor_template_max_ratio",
        "min_stem_chars",
    }
    for q_type, axes in _PER_QUESTION_TYPE_THRESHOLDS.items():
        missing = required_axes - axes.keys()
        assert not missing, (
            f"{q_type!r} missing axes: {missing!r}; got {sorted(axes.keys())}"
        )

    # Sanity-check a couple of pinned values from the plan §3 W6.A spec.
    assert _PER_QUESTION_TYPE_THRESHOLDS["multiple_choice"]["stem_diversity"] == 0.75
    assert _PER_QUESTION_TYPE_THRESHOLDS["essay"]["stem_diversity"] == 0.55
    assert _PER_QUESTION_TYPE_THRESHOLDS["true_false"]["distractor_template_max_ratio"] == 1.0
    assert _PER_QUESTION_TYPE_THRESHOLDS["essay"]["min_stem_chars"] == 15


# --------------------------------------------------------------------- #
# Test 2 — operator overlay via inputs["per_question_type_thresholds"]
# --------------------------------------------------------------------- #


def test_resolve_per_type_thresholds_overlay_via_inputs() -> None:
    """Operator override via ``inputs["per_question_type_thresholds"]``
    merges on top of the canonical table; per-axis omitted by the
    operator keeps the day-1 default.

    Resolution chain mirrors W1.7.C
    ``block_objective_delivery._resolve_threshold_table`` (operator
    overlay merges on top of module-level defaults).
    """
    overlay = {
        "multiple_choice": {"stem_diversity": 0.95},  # operator tightens
        "essay": {"min_stem_chars": 25},               # operator tightens
        "matching": {"stem_diversity": 0.6},           # NEW type added
    }
    resolved = _resolve_per_type_thresholds(
        {"per_question_type_thresholds": overlay}
    )

    # Operator override applies to MC.stem_diversity.
    assert resolved["multiple_choice"]["stem_diversity"] == 0.95
    # MC.correct_answer_diversity NOT in overlay → keeps day-1 default 0.65.
    assert resolved["multiple_choice"]["correct_answer_diversity"] == 0.65

    # Essay min_stem_chars overridden; other essay axes preserved.
    assert resolved["essay"]["min_stem_chars"] == 25
    assert resolved["essay"]["stem_diversity"] == 0.55

    # New type (matching) added with the default-axes scaffold and the
    # operator-supplied stem_diversity override.
    assert "matching" in resolved
    assert resolved["matching"]["stem_diversity"] == 0.6
    assert resolved["matching"]["correct_answer_diversity"] == _DEFAULT_QUESTION_TYPE_THRESHOLDS[
        "correct_answer_diversity"
    ]

    # Unaffected types preserve their full day-1 entry.
    assert resolved["true_false"] == _PER_QUESTION_TYPE_THRESHOLDS["true_false"]


# --------------------------------------------------------------------- #
# Test 3 — essay diversity warns at per-type floor (not legacy critical)
# --------------------------------------------------------------------- #


def test_essay_low_distinct_stem_ratio_warns_not_critical() -> None:
    """2 essay questions both starting "Discuss" → identical stems.

    Pre-W6.A: corpus-wide LOW_STEM_DIVERSITY at severity=critical
    (0.7 floor; ratio 0.5 < 0.7).
    Post-W6.A: ALSO emits LOW_STEM_DIVERSITY_ESSAY at severity=warning
    (essay-bucket floor 0.55; ratio 0.5 < 0.55).

    The legacy critical stays for back-compat (the legacy regression
    suite at ``lib/tests/test_assessment_validator_real_failures.py``
    pins the corpus-wide critical-severity emit). The W6.A landing
    adds the warning-severity per-type emit ON TOP — not a replacement.
    """
    # Both stems literally identical → distinct ratio 1/2 = 0.5.
    stem = "Discuss the role of mitochondria in cellular respiration in detail."
    questions = [
        _essay("q-001", stem, answer="Answer A: long essay response."),
        _essay("q-002", stem, answer="Answer B: long essay response."),
    ]
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
    })
    codes = {i.code: i for i in result.issues}

    # NEW per-type warning fires at warning severity.
    assert "LOW_STEM_DIVERSITY_ESSAY" in codes, (
        f"per-type code missing; emitted: {sorted(codes.keys())}"
    )
    assert codes["LOW_STEM_DIVERSITY_ESSAY"].severity == "warning", (
        f"per-type severity must be warning day-1, got "
        f"{codes['LOW_STEM_DIVERSITY_ESSAY'].severity!r}"
    )


# --------------------------------------------------------------------- #
# Test 4 — MC diversity floor tighter than corpus-wide default
# --------------------------------------------------------------------- #


def test_mc_diversity_floor_tighter_than_default() -> None:
    """5 MC questions with distinct stem ratio 0.6 (3 unique / 5 total).

    Pre-W6.A: corpus-wide threshold 0.7 → 0.6 < 0.7 → fires
    LOW_STEM_DIVERSITY at critical severity.
    Post-W6.A: MC bucket floor 0.75 → ALSO fires
    LOW_STEM_DIVERSITY_MULTIPLE_CHOICE at warning severity.

    Confirms the per-type MC floor (0.75) is strictly tighter than the
    corpus-wide default (0.7) — MC has the tightest stem-diversity
    requirement because it carries the highest stem volume per
    assessment.
    """
    # 3 unique stems, 5 total → ratio 0.6.
    stems = [
        "<p>Explain how mitochondria produce ATP in cellular respiration</p>",
        "<p>Describe the photosynthesis Calvin cycle stages clearly</p>",
        "<p>Compare and contrast prokaryotic vs eukaryotic ribosomes</p>",
    ]
    questions = [
        _mc("q-001", stems[0], "ATP via oxidative phosphorylation"),
        _mc("q-002", stems[1], "Carbon fixation in stroma"),
        _mc("q-003", stems[2], "70S vs 80S subunit composition"),
        _mc("q-004", stems[0], "Different correct answer"),
        _mc("q-005", stems[1], "Yet another distinct answer"),
    ]
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
    })
    codes = {i.code: i for i in result.issues}

    # NEW per-type MC warning fires at warning severity.
    assert "LOW_STEM_DIVERSITY_MULTIPLE_CHOICE" in codes, (
        f"MC bucket warning missing; emitted: {sorted(codes.keys())}"
    )
    assert codes["LOW_STEM_DIVERSITY_MULTIPLE_CHOICE"].severity == "warning"

    # Per-type floor tighter than default — pin the relationship in the
    # threshold table itself (asymmetry is the load-bearing property).
    assert (
        _PER_QUESTION_TYPE_THRESHOLDS["multiple_choice"]["stem_diversity"]
        > _DEFAULT_QUESTION_TYPE_THRESHOLDS["stem_diversity"]
    )


# --------------------------------------------------------------------- #
# Test 5 — QTI field-name normalization (Lesson 1 defense)
# --------------------------------------------------------------------- #


def test_qti_field_normalization() -> None:
    """A question dict carrying ``"type":"multiple_choice"`` (QTI shape,
    not generator shape) must dispatch to the MC bucket via
    ``_normalize_question_type``.

    W4.B Lesson 1 defense: pair-schema field-name divergence (lo_refs
    vs learning_outcome_refs) was caught in flight on Wave 4. Same
    class of risk applies here — ``Trainforge/parsers/qti_parser.py:24``
    uses ``type:`` while ``Trainforge/generators/assessment_generator.py:101``
    uses ``question_type:``. Day-1 only generator-emit reaches this
    validator; this helper is defense-in-depth against a future call
    site that routes QTI-parsed dicts directly.
    """
    # Helper-level: bare resolution.
    qti_q = {"type": "multiple_choice", "stem": "Some stem"}
    assert _normalize_question_type(qti_q) == "multiple_choice"

    # ``question_type`` wins over ``type`` when both are present
    # (generator shape is preferred since canonical).
    both_q = {"question_type": "essay", "type": "multiple_choice"}
    assert _normalize_question_type(both_q) == "essay"

    # Validator-level: QTI-shape questions dispatch to the MC bucket.
    # 5 QTI-shape MC questions, 2 unique stems → MC bucket warning.
    qti_questions = [
        {
            "question_id": f"q-{i:03d}",
            "type": "multiple_choice",  # QTI shape only
            "stem": (
                "<p>Explain how mitochondria produce ATP</p>"
                if i % 2 == 0
                else "<p>Describe photosynthesis</p>"
            ),
            "bloom_level": "understand",
            "objective_id": "LO-01",
            "choices": [
                {"text": f"correct-{i}", "is_correct": True},
                {"text": f"distractor-A-{i}", "is_correct": False},
                {"text": f"distractor-B-{i}", "is_correct": False},
                {"text": f"distractor-C-{i}", "is_correct": False},
            ],
        }
        for i in range(5)
    ]
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": qti_questions},
    })
    codes = {i.code for i in result.issues}
    # 2/5 = 0.4 distinct ratio < MC bucket floor 0.75 → warning fires.
    assert "LOW_STEM_DIVERSITY_MULTIPLE_CHOICE" in codes, (
        f"QTI-shape questions did not dispatch to MC bucket; "
        f"emitted: {sorted(codes)}"
    )


# --------------------------------------------------------------------- #
# Test 6 — unknown question_type falls through to default
# --------------------------------------------------------------------- #


def test_unknown_question_type_dispatches_to_default() -> None:
    """A ``question_type:"matching"`` (the QTI-only 6th value) routes
    through ``_DEFAULT_QUESTION_TYPE_THRESHOLDS`` rather than KeyError'ing.

    Defensive mitigation against future QTI-side type additions before
    the threshold table is updated. Mirrors W4.C MEDIUM
    ``_DEFAULT_ENTAILMENT_FLOOR`` fallback shape.
    """
    # Helper-level: ``matching`` resolves through default scaffold.
    fallback = _thresholds_for_type("matching", _PER_QUESTION_TYPE_THRESHOLDS)
    assert fallback == _DEFAULT_QUESTION_TYPE_THRESHOLDS

    # Validator-level: 4 ``matching`` questions, all stems identical
    # (1/4 = 0.25 distinct ratio). Default floor 0.7 fires → emit
    # LOW_STEM_DIVERSITY_MATCHING (the upper-cased question_type).
    matching_questions = [
        {
            "question_id": f"q-{i:03d}",
            "question_type": "matching",
            "stem": "<p>Match the term to its definition from the list</p>",
            "bloom_level": "understand",
            "objective_id": "LO-01",
            "correct_answer": f"answer-{i}",
        }
        for i in range(4)
    ]
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": matching_questions},
        "min_score": 0.0,  # only checking warning emit, not gate-pass
    })
    codes = {i.code: i for i in result.issues}
    assert "LOW_STEM_DIVERSITY_MATCHING" in codes, (
        f"unknown-type bucket warning missing; emitted: {sorted(codes.keys())}"
    )
    # Default-floor fallback emits at warning severity (per-type
    # contract — calibration-deferred regardless of type).
    assert codes["LOW_STEM_DIVERSITY_MATCHING"].severity == "warning"
