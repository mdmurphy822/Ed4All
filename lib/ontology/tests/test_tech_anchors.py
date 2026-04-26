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


# ---------------------------------------------------------------------------
# Wave 84 — additional anchors from the rdf-shacl-551-2 weak-chunk audit.
# ---------------------------------------------------------------------------


class TestWave84Serializations:
    """Audit's worked-example pages were tagged ``turtle/n-triples/rdf``
    only — but they cover TriG, N-Quads, and RDF/XML too. Detect them."""

    def test_trig_detected_case_sensitive(self):
        assert "trig" in detect_anchors(
            "TriG extends Turtle with named-graph blocks."
        )

    def test_trig_does_not_match_lowercase_math_function(self):
        # Case-sensitive: the math word "trig" (trigonometry) must NOT fire.
        assert "trig" not in detect_anchors("Solve the trig identity.")

    def test_n_quads_detected(self):
        assert "n-quads" in detect_anchors("Serialize the dataset as N-Quads.")

    def test_rdf_xml_detected_with_slash(self):
        assert "rdf-xml" in detect_anchors("Parse the RDF/XML file.")

    def test_rdfxml_compact_form_detected(self):
        assert "rdf-xml" in detect_anchors("RDFXML is the legacy serialization.")


class TestWave84Foundationals:
    """RDF foundational vocabulary: IRI, literal, datatype, blank node."""

    def test_iri_detected(self):
        assert "iri" in detect_anchors("Use an IRI to identify the resource.")

    def test_iri_plural_detected(self):
        assert "iri" in detect_anchors("Mint IRIs from a stable namespace.")

    def test_blank_node_detected(self):
        assert "blank-node" in detect_anchors("Use a blank node for anonymity.")

    def test_blank_node_turtle_syntax_detected(self):
        # Turtle blank-node syntax: _:b1, _:foo
        assert "blank-node" in detect_anchors(":alice :knows _:b1 .")

    def test_literal_with_rdf_context_detected(self):
        assert "literal" in detect_anchors("RDF literals carry datatype tags.")

    def test_plain_word_literal_does_not_fire(self):
        # The English word "literal" should NOT trigger without RDF/datatype context.
        assert "literal" not in detect_anchors("The literal interpretation.")

    def test_datatype_detected(self):
        assert "datatype" in detect_anchors("xsd:string is a datatype.")

    def test_rdf_dataset_detected(self):
        assert "rdf-dataset" in detect_anchors(
            "An RDF dataset contains a default graph."
        )

    def test_named_graphs_detected_as_dataset(self):
        assert "rdf-dataset" in detect_anchors(
            "Use named graphs to scope assertions."
        )


class TestWave84ShaclShapes:
    def test_node_shape_detected(self):
        assert "node-shape" in detect_anchors("Author a NodeShape per class.")

    def test_node_shape_qname_detected(self):
        assert "node-shape" in detect_anchors("a sh:NodeShape ;")

    def test_property_shape_detected(self):
        assert "property-shape" in detect_anchors(
            "PropertyShape constraints target paths."
        )


class TestWave84RdfsPredicates:
    def test_subclassof_detected(self):
        assert "subclassof" in detect_anchors(
            "Apply rdfs:subClassOf entailment to the graph."
        )

    def test_subclassof_english_form(self):
        assert "subclassof" in detect_anchors(
            "Person is a subclass of Agent."
        )

    def test_subpropertyof_detected(self):
        assert "subpropertyof" in detect_anchors(
            "Declare rdfs:subPropertyOf to chain predicates."
        )

    def test_rdf_type_detected(self):
        assert "rdf-type" in detect_anchors(":alice rdf:type :Person .")


class TestWave84TurtlePrefix:
    def test_turtle_at_prefix_detected(self):
        assert "turtle-prefix" in detect_anchors("@prefix : <http://ex.org/> .")

    def test_sparql_prefix_detected(self):
        assert "turtle-prefix" in detect_anchors("PREFIX foaf: <http://...>")


def test_anchor_slugs_returns_sorted_tuple():
    slugs = anchor_slugs()
    assert slugs == tuple(sorted(slugs))
    # Wave 82 set: 8 W3C standards + same-as.
    # Wave 84 additions: trig, n-quads, rdf-xml, iri, literal, datatype,
    # blank-node, rdf-dataset, node-shape, property-shape, subclassof,
    # subpropertyof, rdf-type, turtle-prefix. Total = 23.
    assert len(slugs) >= 9, f"Lost a Wave 82 anchor: only {len(slugs)} present"
    assert "rdf" in slugs
    assert "same-as" in slugs
