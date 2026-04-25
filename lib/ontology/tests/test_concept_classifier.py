"""Wave 75: tests for the concept-graph node classifier.

Pins the rule precedence and the curated stoplists. All assertions are
deterministic — the classifier is side-effect-free.
"""

from __future__ import annotations

import pytest

from lib.ontology.concept_classifier import (
    ASSESSMENT_OPTION,
    DOMAIN_CONCEPT,
    INSTRUCTIONAL_ARTIFACT,
    LEARNING_OBJECTIVE,
    LOW_SIGNAL,
    MISCONCEPTION,
    PEDAGOGICAL_MARKER,
    classify_concept,
)


# ---------------------------------------------------------------------------
# Required cases from the Wave 75 task spec.
# ---------------------------------------------------------------------------


def test_rdf_graph_is_domain_concept():
    assert classify_concept("rdf-graph") == DOMAIN_CONCEPT


def test_key_takeaway_is_pedagogical_marker():
    assert classify_concept("key-takeaway") == PEDAGOGICAL_MARKER


def test_answer_b_is_assessment_option():
    assert classify_concept("answer-b") == ASSESSMENT_OPTION


def test_submission_format_is_instructional_artifact():
    assert classify_concept("submission-format") == INSTRUCTIONAL_ARTIFACT


def test_to_04_is_learning_objective():
    assert classify_concept("to-04") == LEARNING_OBJECTIVE


def test_not_is_low_signal():
    assert classify_concept("not") == LOW_SIGNAL


def test_sh_path_is_domain_concept():
    # Colon-prefixed CURIE for SHACL must NOT be flagged as anything
    # except a real domain concept.
    assert classify_concept("sh:path") == DOMAIN_CONCEPT


def test_owl_2_rl_is_domain_concept():
    assert classify_concept("owl-2-rl") == DOMAIN_CONCEPT


def test_do_not_is_low_signal():
    assert classify_concept("do-not") == LOW_SIGNAL


def test_empty_input_is_low_signal_graceful():
    assert classify_concept("") == LOW_SIGNAL
    assert classify_concept(None) == LOW_SIGNAL  # type: ignore[arg-type]
    assert classify_concept("   ") == LOW_SIGNAL


# ---------------------------------------------------------------------------
# Precedence + range coverage.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_id,expected",
    [
        ("to-01", LEARNING_OBJECTIVE),
        ("co-12", LEARNING_OBJECTIVE),
        ("TO-99", LEARNING_OBJECTIVE),
        ("answer-a", ASSESSMENT_OPTION),
        ("answer-d", ASSESSMENT_OPTION),
        ("option-c", ASSESSMENT_OPTION),
        ("OPTION-A", ASSESSMENT_OPTION),
        ("rubric", PEDAGOGICAL_MARKER),
        ("learning-objective", PEDAGOGICAL_MARKER),
        ("self-check", PEDAGOGICAL_MARKER),
        ("practice", PEDAGOGICAL_MARKER),
        ("callout", PEDAGOGICAL_MARKER),
        ("exercise", PEDAGOGICAL_MARKER),
        ("the", LOW_SIGNAL),
        ("an", LOW_SIGNAL),
        ("with", LOW_SIGNAL),
        ("by", LOW_SIGNAL),
        ("of", LOW_SIGNAL),
        ("deadline", INSTRUCTIONAL_ARTIFACT),
        ("week-overview", INSTRUCTIONAL_ARTIFACT),
        ("module-header", INSTRUCTIONAL_ARTIFACT),
        ("what-you-will-produce", INSTRUCTIONAL_ARTIFACT),
        ("turtle", DOMAIN_CONCEPT),
        ("blank-node", DOMAIN_CONCEPT),
        ("sparql-select", DOMAIN_CONCEPT),
    ],
)
def test_classification_table(node_id, expected):
    assert classify_concept(node_id) == expected


def test_misconception_hint_is_consulted():
    # A node that would otherwise classify as DomainConcept should pivot
    # to Misconception when the explicit hint is supplied.
    assert (
        classify_concept(
            "students-believe-rdf-is-xml",
            hints={"is_misconception": True},
        )
        == MISCONCEPTION
    )


def test_misconception_hint_does_not_override_lo_or_option():
    # Precedence: LO / AssessmentOption / Pedagogical / LowSignal /
    # InstructionalArtifact all win against the misconception hint.
    assert classify_concept("to-05", hints={"is_misconception": True}) == LEARNING_OBJECTIVE
    assert classify_concept("answer-a", hints={"is_misconception": True}) == ASSESSMENT_OPTION
    assert classify_concept("rubric", hints={"is_misconception": True}) == PEDAGOGICAL_MARKER
    assert classify_concept("not", hints={"is_misconception": True}) == LOW_SIGNAL
    assert (
        classify_concept("submission-format", hints={"is_misconception": True})
        == INSTRUCTIONAL_ARTIFACT
    )


def test_classifier_is_side_effect_free():
    # Calling repeatedly must always produce the same answer with no
    # state leakage between calls.
    for _ in range(5):
        assert classify_concept("rdf-graph") == DOMAIN_CONCEPT
        assert classify_concept("answer-b") == ASSESSMENT_OPTION
        assert classify_concept("not") == LOW_SIGNAL


def test_label_argument_is_advisory_not_load_bearing():
    # The label is accepted but classification is keyed off node_id.
    # A pedagogical-marker slug stays a PedagogicalMarker even if the
    # label is set to something that looks like a domain concept.
    assert (
        classify_concept("rubric", label="RDF Graph Construction")
        == PEDAGOGICAL_MARKER
    )


def test_lo_pattern_does_not_match_co_authored_or_to_string():
    # ``co-`` / ``to-`` prefixes are only LO IDs when followed by 2+
    # digits and nothing else. Free-form concepts that happen to start
    # with those prefixes must NOT be misclassified as LOs.
    assert classify_concept("co-author") == DOMAIN_CONCEPT
    assert classify_concept("to-string") == DOMAIN_CONCEPT
    # Single-digit suffix is below the canonical LO pattern (NN, 2+).
    assert classify_concept("to-1") == DOMAIN_CONCEPT


def test_answer_pattern_only_matches_a_through_d():
    assert classify_concept("answer-e") == DOMAIN_CONCEPT
    assert classify_concept("answer-key") == DOMAIN_CONCEPT
