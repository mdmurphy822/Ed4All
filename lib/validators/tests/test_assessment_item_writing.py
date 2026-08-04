"""Tests for the deterministic Haladyna item-writing linter.

Covers each rule pass+fail on the structured-item surface, the QTI-XML parse
path, the Bloom-honesty ceiling arm, and the warning-day-1 severity contract.
Offline-safe: no network, no models (the generator ceiling import is
pure-Python).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from lib.validators.assessment_item_writing import AssessmentItemWritingValidator


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _codes(result) -> set:
    return {i.code for i in result.issues}


def _run(items=None, **inputs):
    v = AssessmentItemWritingValidator()
    payload = dict(inputs)
    if items is not None:
        payload["assessment_items"] = items
    return v.validate(payload)


def _mc(qid="q1", stem="What is the capital of France?", choices=None,
        subtype="mc_single", bloom=None, qtype="multiple_choice"):
    if choices is None:
        choices = [
            {"id": "A", "text": "Paris", "is_correct": True},
            {"id": "B", "text": "Lyon", "is_correct": False},
            {"id": "C", "text": "Nice", "is_correct": False},
        ]
    q = {
        "question_id": qid,
        "question_type": qtype,
        "stem": stem,
        "choices": choices,
        "item_subtype": subtype,
    }
    if bloom is not None:
        q["bloom_level"] = bloom
    return q


def _qti_doc(questions):
    """Wrap emitter-built items into a namespace-naked questestinterop doc."""
    from Courseforge.scripts.cartridge.qti_emitter import question_to_qti_item

    root = ET.Element("questestinterop")
    assessment = ET.SubElement(root, "assessment")
    assessment.set("ident", "A1")
    section = ET.SubElement(assessment, "section")
    section.set("ident", "S1")
    for q in questions:
        section.append(question_to_qti_item(q))
    return ET.tostring(root, encoding="unicode")


# --------------------------------------------------------------------------- #
# Severity + no-input contract
# --------------------------------------------------------------------------- #
def test_warning_day_one_always_passes():
    result = _run(items=[_mc()])
    assert result.passed is True
    assert result.action is None


def test_no_input_is_info_noop():
    result = _run()
    assert result.passed is True
    assert "ITEM_WRITING_NO_INPUT" in _codes(result)


def test_clean_item_has_no_writing_issues():
    result = _run(items=[_mc()])
    writing = {c for c in _codes(result) if c.startswith("ITEM_") and c not in (
        "ITEM_WRITING_NO_BLOOM_META",)}
    assert writing == set()


# --------------------------------------------------------------------------- #
# Rule: single-key (+ MR exemption)
# --------------------------------------------------------------------------- #
def test_non_mr_multiple_keys_flagged():
    choices = [
        {"id": "A", "text": "Paris", "is_correct": True},
        {"id": "B", "text": "Lyon", "is_correct": True},
        {"id": "C", "text": "Nice", "is_correct": False},
    ]
    result = _run(items=[_mc(choices=choices, subtype="mc_single")])
    assert "ITEM_NON_MR_MULTIPLE_KEYS" in _codes(result)


def test_multiple_response_two_keys_exempt():
    choices = [
        {"id": "A", "text": "Distribute the sign", "is_correct": True},
        {"id": "B", "text": "Combine like terms", "is_correct": True},
        {"id": "C", "text": "Ignore the parentheses", "is_correct": False},
    ]
    result = _run(items=[_mc(
        choices=choices, subtype="mc_multiple_response",
        qtype="multiple_response",
    )])
    assert "ITEM_NON_MR_MULTIPLE_KEYS" not in _codes(result)


# --------------------------------------------------------------------------- #
# Rule: generic non-problem stem
# --------------------------------------------------------------------------- #
def test_generic_stem_flagged():
    result = _run(items=[_mc(stem="Which statement is correct?")])
    assert "ITEM_STEM_GENERIC_NONPROBLEM" in _codes(result)


# --------------------------------------------------------------------------- #
# Rule: stem too short vs options
# --------------------------------------------------------------------------- #
def test_stem_too_short_vs_options_flagged():
    choices = [
        {"id": "A", "text": "A prime number is a natural number greater than "
                            "one with no positive divisors other than one and "
                            "itself.", "is_correct": True},
        {"id": "B", "text": "A composite number has more than two positive "
                            "divisors including one and itself.", "is_correct": False},
        {"id": "C", "text": "An even number is any integer divisible by two "
                            "without any remainder at all.", "is_correct": False},
    ]
    result = _run(items=[_mc(stem="Define:", choices=choices)])
    assert "ITEM_STEM_TOO_SHORT_VS_OPTIONS" in _codes(result)


def test_short_stem_not_flagged_for_true_false():
    choices = [
        {"id": "T", "text": "True", "is_correct": True},
        {"id": "F", "text": "False", "is_correct": False},
    ]
    result = _run(items=[_mc(
        stem="Pi is rational.", choices=choices, subtype="tf",
        qtype="true_false",
    )])
    assert "ITEM_STEM_TOO_SHORT_VS_OPTIONS" not in _codes(result)


# --------------------------------------------------------------------------- #
# Rule: unemphasized negative
# --------------------------------------------------------------------------- #
def test_unemphasized_negative_flagged():
    result = _run(items=[_mc(stem="Which of these is not a prime number?")])
    assert "ITEM_STEM_UNEMPHASIZED_NEGATIVE" in _codes(result)


def test_uppercased_negative_not_flagged():
    result = _run(items=[_mc(stem="Which of these is NOT a prime number?")])
    assert "ITEM_STEM_UNEMPHASIZED_NEGATIVE" not in _codes(result)


def test_emphasis_markup_negative_not_flagged():
    result = _run(items=[_mc(
        stem="Which of these is <strong>not</strong> a prime number?")])
    assert "ITEM_STEM_UNEMPHASIZED_NEGATIVE" not in _codes(result)


# --------------------------------------------------------------------------- #
# Rule: None/All of the above + absolute terms
# --------------------------------------------------------------------------- #
def test_none_of_the_above_flagged():
    choices = [
        {"id": "A", "text": "Paris", "is_correct": True},
        {"id": "B", "text": "Lyon", "is_correct": False},
        {"id": "C", "text": "None of the above", "is_correct": False},
    ]
    result = _run(items=[_mc(choices=choices)])
    assert "ITEM_OPTION_NONE_ALL_OF_ABOVE" in _codes(result)


def test_absolute_term_option_flagged():
    choices = [
        {"id": "A", "text": "A remainder is always zero", "is_correct": False},
        {"id": "B", "text": "A remainder is the left-over amount", "is_correct": True},
        {"id": "C", "text": "A remainder is the quotient", "is_correct": False},
    ]
    result = _run(items=[_mc(choices=choices)])
    assert "ITEM_OPTION_ABSOLUTE_TERM" in _codes(result)


# --------------------------------------------------------------------------- #
# Rule: longest option is key
# --------------------------------------------------------------------------- #
def test_longest_option_is_key_flagged():
    choices = [
        {"id": "A", "text": "A prime number is a natural number greater than "
                            "one whose only positive divisors are one and "
                            "itself, exactly.", "is_correct": True},
        {"id": "B", "text": "A square.", "is_correct": False},
        {"id": "C", "text": "An even.", "is_correct": False},
    ]
    result = _run(items=[_mc(stem="Which best defines a prime number here?",
                             choices=choices)])
    assert "ITEM_LONGEST_OPTION_IS_KEY" in _codes(result)


# --------------------------------------------------------------------------- #
# Rule: duplicate / overlapping options
# --------------------------------------------------------------------------- #
def test_duplicate_options_flagged():
    choices = [
        {"id": "A", "text": "Paris", "is_correct": True},
        {"id": "B", "text": "Paris", "is_correct": False},
        {"id": "C", "text": "Nice", "is_correct": False},
    ]
    result = _run(items=[_mc(choices=choices)])
    assert "ITEM_OPTIONS_DUPLICATE" in _codes(result)


def test_overlapping_options_flagged():
    choices = [
        {"id": "A", "text": "the number greater than one with two divisors",
         "is_correct": True},
        {"id": "B", "text": "the number greater than one with two divisors only",
         "is_correct": False},
        {"id": "C", "text": "a triangle", "is_correct": False},
    ]
    result = _run(items=[_mc(stem="Which describes a prime?", choices=choices)])
    assert "ITEM_OPTIONS_OVERLAP" in _codes(result)


# --------------------------------------------------------------------------- #
# Rule: a/an article agreement
# --------------------------------------------------------------------------- #
def test_article_disagreement_flagged():
    choices = [
        {"id": "A", "text": "apple", "is_correct": True},
        {"id": "B", "text": "banana", "is_correct": False},
        {"id": "C", "text": "cherry", "is_correct": False},
    ]
    result = _run(items=[_mc(stem="The fruit shown is a", choices=choices)])
    assert "ITEM_OPTION_ARTICLE_DISAGREEMENT" in _codes(result)


# --------------------------------------------------------------------------- #
# Rule: Bloom-honesty ceiling (structured items only)
# --------------------------------------------------------------------------- #
def test_bloom_ceiling_exceeded_flagged():
    result = _run(items=[_mc(subtype="mc_single", bloom="apply")])
    assert "ITEM_BLOOM_CEILING_EXCEEDED" in _codes(result)


def test_bloom_within_ceiling_not_flagged():
    result = _run(items=[_mc(subtype="error_analysis", bloom="analyze")])
    assert "ITEM_BLOOM_CEILING_EXCEEDED" not in _codes(result)


def test_bloom_at_ceiling_not_flagged():
    result = _run(items=[_mc(subtype="mc_single", bloom="understand")])
    assert "ITEM_BLOOM_CEILING_EXCEEDED" not in _codes(result)


# --------------------------------------------------------------------------- #
# QTI-XML surface
# --------------------------------------------------------------------------- #
def test_qti_surface_parses_and_flags():
    q = {
        "question_id": "q1",
        "question_type": "multiple_choice",
        "stem": "Which statement is correct?",
        "choices": [
            {"id": "A", "text": "Paris", "is_correct": True},
            {"id": "B", "text": "Lyon", "is_correct": False},
            {"id": "C", "text": "Nice", "is_correct": False},
        ],
        "objective_id": "TO-01",
    }
    xml = _qti_doc([q])
    result = _run(qti_strings=[{"id": "quiz", "xml": xml}])
    assert result.passed is True
    # Generic stem fires on the parsed QTI surface.
    assert "ITEM_STEM_GENERIC_NONPROBLEM" in _codes(result)
    # Bloom-honesty degrades to the seam note on the metadata-free QTI surface.
    assert "ITEM_WRITING_NO_BLOOM_META" in _codes(result)


def test_qti_surface_mr_exempt_from_single_key():
    q = {
        "question_id": "mr1",
        "question_type": "multiple_response",
        "stem": "Select every step used to simplify the expression correctly.",
        "choices": [
            {"id": "A", "text": "Distribute the negative sign", "is_correct": True},
            {"id": "B", "text": "Combine the like terms", "is_correct": True},
            {"id": "C", "text": "Drop the exponent", "is_correct": False},
        ],
        "objective_id": "TO-01",
    }
    xml = _qti_doc([q])
    result = _run(qti_strings=[{"id": "quiz", "xml": xml}])
    assert "ITEM_NON_MR_MULTIPLE_KEYS" not in _codes(result)


# --------------------------------------------------------------------------- #
# Written-response rules (short_answer / extended_response)
# --------------------------------------------------------------------------- #
def _written(qid="w1", stem=None, subtype="short_answer", rubric="default",
             source_chunks=("c1",)):
    if stem is None:
        stem = ("In 2-4 sentences, explain why a coefficient multiplies "
                "the variable it precedes.")
    if rubric == "default":
        rubric = {
            "criteria": [
                {"criterion": "Correctly explains the role of a coefficient.",
                 "cites": ["c1"],
                 "levels": [{"score": 2, "descriptor": "Full."},
                            {"score": 0, "descriptor": "None."}]},
            ],
            "deductions": [],
        }
    q = {
        "question_id": qid,
        "question_type": "essay",
        "stem": stem,
        "item_subtype": subtype,
        "source_chunks": list(source_chunks),
    }
    if rubric is not None:
        q["rubric"] = rubric
    return q


def test_written_good_item_passes_clean():
    result = _run(items=[_written()])
    codes = _codes(result)
    assert not any(c.startswith("ITEM_WRITTEN_") for c in codes), codes
    assert result.passed is True


def test_written_bare_discuss_stem_flagged():
    result = _run(items=[_written(stem="Discuss photosynthesis.")])
    assert "ITEM_WRITTEN_STEM_NOT_SPECIFIC" in _codes(result)
    assert result.passed is True  # warning-severity day-1


def test_written_short_stem_flagged():
    result = _run(items=[_written(stem="Explain it.")])
    assert "ITEM_WRITTEN_STEM_NOT_SPECIFIC" in _codes(result)


def test_written_missing_rubric_flagged():
    result = _run(items=[_written(rubric=None)])
    assert "ITEM_WRITTEN_RUBRIC_MISSING" in _codes(result)


def test_written_empty_criteria_flagged():
    result = _run(items=[_written(rubric={"criteria": [], "deductions": []})])
    assert "ITEM_WRITTEN_RUBRIC_MISSING" in _codes(result)


def test_written_ungrounded_criterion_flagged():
    bad = {
        "criteria": [
            {"criterion": "Ungrounded criterion.", "cites": [],
             "levels": [{"score": 2, "descriptor": "x"}]},
        ],
        "deductions": [],
    }
    result = _run(items=[_written(rubric=bad)])
    assert "ITEM_WRITTEN_RUBRIC_CRITERION_UNGROUNDED" in _codes(result)


def test_written_criterion_cite_not_in_source_flagged():
    bad = {
        "criteria": [
            {"criterion": "Cites a foreign chunk.", "cites": ["c_other"],
             "levels": [{"score": 2, "descriptor": "x"}]},
        ],
        "deductions": [],
    }
    result = _run(items=[_written(rubric=bad, source_chunks=("c1",))])
    assert "ITEM_WRITTEN_RUBRIC_CRITERION_UNGROUNDED" in _codes(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
