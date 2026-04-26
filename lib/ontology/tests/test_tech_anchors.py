"""Wave 82 tests for the W3C tech-anchor detector.

Pins the detection contract: the bounded vocabulary fires on standard
acronyms and full names but does NOT over-match on adjacent compound
forms (``OWL 2`` is its own concept and must not trigger plain ``owl``).
"""

from __future__ import annotations

import pytest

from lib.ontology.tech_anchors import anchor_slugs, detect_anchors


# ---------------------------------------------------------------------------
# Standalone acronym detection — the audit-named failing cases.
# ---------------------------------------------------------------------------


def test_rdf_acronym_detected():
    assert "rdf" in detect_anchors("RDF is a graph data model.")


def test_rdfs_acronym_detected():
    assert "rdfs" in detect_anchors("Use RDFS for vocabulary definitions.")


def test_rdf_schema_full_name_detected():
    assert "rdfs" in detect_anchors("RDF Schema lets you declare classes.")


def test_owl_standalone_detected():
    assert "owl" in detect_anchors("OWL is a description-logic vocabulary.")


def test_web_ontology_language_full_name_detected():
    assert "owl" in detect_anchors("The Web Ontology Language extends RDFS.")


def test_shacl_detected():
    assert "shacl" in detect_anchors("Validate the graph with SHACL.")


def test_shapes_constraint_language_full_name_detected():
    assert "shacl" in detect_anchors(
        "The Shapes Constraint Language defines node shapes."
    )


def test_sparql_detected():
    assert "sparql" in detect_anchors("Run a SPARQL SELECT query.")


def test_turtle_detected():
    assert "turtle" in detect_anchors("Serialize as Turtle for readability.")


def test_ttl_acronym_detected():
    # Uppercase TTL is the file-extension acronym; lowercase ``ttl`` is
    # noise in body text and must NOT fire.
    assert "turtle" in detect_anchors("Saved the graph as TTL.")


def test_lowercase_ttl_does_not_fire():
    # Pattern is case-sensitive on TTL specifically — see tech_anchors.py
    # rationale comment. Lowercase ``ttl`` could just be the literal
    # English string in unrelated prose; we don't want a false positive.
    assert "turtle" not in detect_anchors("the cattle were grazing in ttl fields")


def test_json_ld_detected():
    assert "json-ld" in detect_anchors("Embed metadata as JSON-LD.")


def test_n_triples_detected():
    assert "n-triples" in detect_anchors("Export to N-Triples.")


def test_ntriples_no_hyphen_detected():
    assert "n-triples" in detect_anchors("Export to NTriples.")


# ---------------------------------------------------------------------------
# Predicate-level detection — the owl:sameAs failing case.
# ---------------------------------------------------------------------------


def test_owl_sameas_detected():
    assert "same-as" in detect_anchors("Use owl:sameAs to assert identity.")


def test_sameas_camelcase_detected():
    # Bare ``sameAs`` (no namespace prefix) is also valid in prose.
    assert "same-as" in detect_anchors("The sameAs predicate is dangerous.")


# ---------------------------------------------------------------------------
# Negative cases — guard against over-matching.
# ---------------------------------------------------------------------------


def test_owl_2_not_flagged_as_plain_owl():
    # ``OWL 2`` and ``OWL-2`` are their own concepts (``owl-2``,
    # ``owl-2-dl`` etc. are pre-existing nodes). Plain ``owl`` must NOT
    # fire when the version qualifier follows.
    assert "owl" not in detect_anchors("OWL 2 DL is decidable.")
    assert "owl" not in detect_anchors("Use OWL-2 for classification.")


def test_rdf_substring_in_other_word_not_matched():
    # Word-boundary regex must reject substring-only matches.
    assert "rdf" not in detect_anchors("XMLRDFReader is deprecated.")


def test_empty_text_returns_empty_set():
    assert detect_anchors("") == set()
    assert detect_anchors(None) == set()  # type: ignore[arg-type]


def test_no_match_returns_empty_set():
    assert detect_anchors("This text has no semantic-web vocabulary.") == set()


# ---------------------------------------------------------------------------
# Multi-hit text — realistic chunk content.
# ---------------------------------------------------------------------------


def test_multi_anchor_text():
    text = (
        "An RDF graph serialized as Turtle, validated with SHACL, "
        "and queried via SPARQL is the standard semantic-web stack."
    )
    hits = detect_anchors(text)
    assert {"rdf", "turtle", "shacl", "sparql"} <= hits


# ---------------------------------------------------------------------------
# Slug stability — anchor_slugs() returns the canonical alphabetized list.
# ---------------------------------------------------------------------------


def test_anchor_slugs_returns_sorted_tuple():
    slugs = anchor_slugs()
    assert slugs == tuple(sorted(slugs))
    assert len(slugs) == 9  # 8 W3C standards + same-as predicate
    assert "rdf" in slugs
    assert "same-as" in slugs
