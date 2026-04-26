"""Wave 82 tests for the concept-ID prefix helper."""

from __future__ import annotations

from lib.ontology.concept_id import (
    CONCEPT_PREFIX,
    add_concept_prefix,
    strip_concept_prefix,
)


class TestStripConceptPrefix:
    def test_strips_prefix_when_present(self):
        assert strip_concept_prefix("concept:rdf-graph") == "rdf-graph"

    def test_passthrough_when_no_prefix(self):
        assert strip_concept_prefix("rdf-graph") == "rdf-graph"

    def test_empty_input_returns_empty(self):
        assert strip_concept_prefix("") == ""

    def test_does_not_strip_other_prefixes(self):
        # Pedagogy graph also emits module:foo, bloom:bar — these must
        # pass through unchanged.
        assert strip_concept_prefix("module:week-3") == "module:week-3"
        assert strip_concept_prefix("bloom:apply") == "bloom:apply"

    def test_idempotent(self):
        # Stripping twice = stripping once.
        once = strip_concept_prefix("concept:foo")
        twice = strip_concept_prefix(once)
        assert once == twice == "foo"


class TestAddConceptPrefix:
    def test_adds_prefix_to_bare_slug(self):
        assert add_concept_prefix("rdf-graph") == "concept:rdf-graph"

    def test_idempotent_when_prefix_present(self):
        assert add_concept_prefix("concept:rdf-graph") == "concept:rdf-graph"

    def test_empty_input_returns_empty(self):
        assert add_concept_prefix("") == ""


class TestRoundTrip:
    def test_strip_then_add_recovers_prefix_form(self):
        assert add_concept_prefix(strip_concept_prefix("concept:foo")) == "concept:foo"

    def test_add_then_strip_recovers_bare_form(self):
        assert strip_concept_prefix(add_concept_prefix("foo")) == "foo"


def test_constant_value():
    # Pin the exact prefix string — emit sites at
    # Trainforge/pedagogy_graph_builder.py:731,767,864,895,953,954 use
    # f"concept:{slug}" verbatim.
    assert CONCEPT_PREFIX == "concept:"
