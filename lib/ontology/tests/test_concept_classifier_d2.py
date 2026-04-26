"""Wave 82 Phase D2 tests for the procedural-noise stoplist additions.

Audit reproducer (rdf-shacl-551 Section F): top concepts included
``Plan`` (frequency 10), ``Verify`` (14), and ``Step 1``/``Step 2``
(6 each) — pedagogical scaffolding misclassified as domain vocabulary.

This test file pins the new stoplist entries so future edits can't
silently regress them. Pairs with the existing
``test_concept_classifier.py`` Wave 75/76 coverage.
"""

from __future__ import annotations

from lib.ontology.concept_classifier import (
    INSTRUCTIONAL_ARTIFACT,
    PEDAGOGICAL_MARKER,
    classify_concept,
)


class TestProceduralVerbStoplist:
    def test_plan_classified_as_pedagogical_marker(self):
        assert classify_concept("plan") == PEDAGOGICAL_MARKER

    def test_verify_classified_as_pedagogical_marker(self):
        assert classify_concept("verify") == PEDAGOGICAL_MARKER

    def test_plan_uppercase_still_matches(self):
        # Classifier lowercases input via _normalize, so uppercase still
        # routes to PedagogicalMarker.
        assert classify_concept("Plan") == PEDAGOGICAL_MARKER
        assert classify_concept("PLAN") == PEDAGOGICAL_MARKER


class TestStepNumberLogistics:
    def test_step_1_classified_as_instructional_artifact(self):
        assert classify_concept("step-1") == INSTRUCTIONAL_ARTIFACT

    def test_step_2_classified_as_instructional_artifact(self):
        assert classify_concept("step-2") == INSTRUCTIONAL_ARTIFACT

    def test_step_n_with_compound_suffix(self):
        # "step-3-attach-datatype-hints" is a procedural fragment slug —
        # the logistics regex matches the prefix and routes it accordingly.
        assert classify_concept("step-3-attach-datatype-hints") == INSTRUCTIONAL_ARTIFACT

    def test_high_step_number_still_matches(self):
        assert classify_concept("step-15") == INSTRUCTIONAL_ARTIFACT


class TestRegressionGuards:
    def test_step_alone_not_in_logistics(self):
        # "step" without a number doesn't match the logistics prefix
        # regex (the regex requires \d+). It's not a domain concept
        # either but isn't on the stoplist — let it fall through as
        # DomainConcept rather than over-matching.
        result = classify_concept("step")
        # Length 4 → not LowSignal (needs <3); not in PEDAGOGICAL_MARKERS;
        # so it falls through to DomainConcept. This is intentional —
        # bare "step" is rare in real corpora.
        from lib.ontology.concept_classifier import DOMAIN_CONCEPT
        assert result == DOMAIN_CONCEPT

    def test_planet_not_misclassified(self):
        # The classifier matches whole slugs, not substrings — "planet"
        # must NOT match the "plan" stoplist entry.
        from lib.ontology.concept_classifier import DOMAIN_CONCEPT
        assert classify_concept("planet") == DOMAIN_CONCEPT

    def test_verifiable_not_misclassified(self):
        from lib.ontology.concept_classifier import DOMAIN_CONCEPT
        assert classify_concept("verifiable") == DOMAIN_CONCEPT
