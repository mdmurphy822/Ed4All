"""SKOS broader/narrower emission on the Concept layer.

The W3C-canonical pattern for taxonomic hierarchy between two
``skos:Concept`` instances is ``skos:broader`` / ``skos:narrower``, NOT
``rdfs:subClassOf`` (which is for class-level subsumption). The
vocabulary file declares ``cf:Concept rdfs:subClassOf skos:Concept``,
so concept-graph nodes ARE skos:Concepts.

Before this change, ``Trainforge/rag/inference_rules/is_a_from_key_terms.py``
emitted the ``is-a`` slug (mapped to ``rdfs:subClassOf`` in
``lib/ontology/edge_predicates.SLUG_TO_IRI``) for every concept-pair
match. After this change, when both endpoints are concept-graph
nodes (i.e. cf:Concept instances — the canonical case), the rule
emits the ``broader-than`` slug (mapped to ``skos:broader``). The
``is-a`` slug remains reserved for class-level subsumption (an
endpoint that is an ``rdfs:Class``).

This test:

1. Synthesizes a tiny concept graph with two cf:Concept nodes.
2. Feeds a chunk whose key_term definition triggers the is-a
   inference pattern.
3. Asserts the rule emits ``broader-than`` (mapping to
   ``skos:broader``) — the W3C-canonical SKOS Concept-layer hierarchy
   predicate.
4. Asserts the slug-to-IRI registry resolves ``broader-than`` to
   ``http://www.w3.org/2004/02/skos/core#broader`` and
   ``narrower-than`` to ``...#narrower``.
5. Asserts the JSON schema enum admits both slugs (so emit-side and
   schema-side stay in sync).
6. Asserts the legacy ``is-a`` slug still maps to ``rdfs:subClassOf``
   (slot remains for class-level subsumption).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.ontology.edge_predicates import IRI_TO_SLUG, SLUG_TO_IRI, lookup_iri
from Trainforge.rag.inference_rules import infer_is_a

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "knowledge" / "concept_graph_semantic.schema.json"
)


# ---------------------------------------------------------------------------
# 1. Registry: broader-than / narrower-than resolve to skos:broader / narrower
# ---------------------------------------------------------------------------


def test_broader_than_resolves_to_skos_broader() -> None:
    iri = SLUG_TO_IRI.get("broader-than")
    assert iri == "http://www.w3.org/2004/02/skos/core#broader", (
        "broader-than must resolve to skos:broader (W3C-canonical SKOS "
        "Concept-layer hierarchy predicate)"
    )
    # lookup_iri shim returns the same answer.
    assert lookup_iri("broader-than") == iri


def test_narrower_than_resolves_to_skos_narrower() -> None:
    iri = SLUG_TO_IRI.get("narrower-than")
    assert iri == "http://www.w3.org/2004/02/skos/core#narrower", (
        "narrower-than must resolve to skos:narrower"
    )
    assert lookup_iri("narrower-than") == iri


def test_skos_slugs_round_trip_bijective() -> None:
    """Adding the two SKOS slugs must keep SLUG_TO_IRI bijective —
    no other slug shares the new IRIs."""
    assert IRI_TO_SLUG[SLUG_TO_IRI["broader-than"]] == "broader-than"
    assert IRI_TO_SLUG[SLUG_TO_IRI["narrower-than"]] == "narrower-than"


def test_is_a_slug_still_maps_to_rdfs_subclass_of() -> None:
    """``is-a`` (rdfs:subClassOf) remains in the registry for
    class-level subsumption — the slot was NOT replaced, only joined
    by the SKOS Concept-layer slugs."""
    assert SLUG_TO_IRI["is-a"] == "http://www.w3.org/2000/01/rdf-schema#subClassOf"


# ---------------------------------------------------------------------------
# 2. Schema enum admits both new slugs
# ---------------------------------------------------------------------------


def test_concept_graph_schema_enum_includes_skos_slugs() -> None:
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    enum = schema["properties"]["edges"]["items"]["properties"]["type"]["enum"]
    assert "broader-than" in enum, (
        "concept_graph_semantic.schema.json must accept ``broader-than`` "
        "as a valid edge type"
    )
    assert "narrower-than" in enum
    # Legacy is-a slot remains.
    assert "is-a" in enum


# ---------------------------------------------------------------------------
# 3. Inference rule emits broader-than when both endpoints are cf:Concepts
# ---------------------------------------------------------------------------


def _minimal_concept_graph(*node_ids: str) -> dict:
    return {
        "kind": "concept",
        "nodes": [
            {"id": nid, "label": nid, "frequency": 2} for nid in node_ids
        ],
        "edges": [],
    }


def test_is_a_rule_emits_broader_than_between_two_cf_concepts() -> None:
    """Both endpoints of the inferred edge are concept-graph nodes
    (which are cf:Concept instances per the vocabulary). The W3C-
    canonical SKOS Concept-layer hierarchy predicate is skos:broader,
    so the rule must emit the ``broader-than`` slug."""
    graph = _minimal_concept_graph("aria-role", "accessibility-attribute")
    chunks = [
        {
            "id": "c-aria",
            "concept_tags": ["aria-role", "accessibility-attribute"],
            "learning_outcome_refs": [],
            "key_terms": [
                {
                    "term": "aria-role",
                    "definition": (
                        "An ARIA role is a type of accessibility-attribute "
                        "that describes a widget."
                    ),
                }
            ],
        }
    ]

    edges = infer_is_a(chunks, None, graph)

    assert len(edges) == 1, edges
    edge = edges[0]
    assert edge["source"] == "aria-role"
    assert edge["target"] == "accessibility-attribute"
    assert edge["type"] == "broader-than", (
        f"Expected broader-than (skos:broader) for two cf:Concept "
        f"endpoints; got {edge['type']!r}. The W3C-canonical SKOS "
        f"Concept-layer hierarchy predicate is skos:broader, not "
        f"rdfs:subClassOf — the latter is reserved for class-level "
        f"subsumption."
    )
    # The provenance rule label is unchanged: rule identity is
    # determined by the rule module, not by the emit slug.
    assert edge["provenance"]["rule"] == "is_a_from_key_terms"


def test_is_a_rule_emits_nothing_when_parent_not_in_graph() -> None:
    """Sanity: the rule still gates on node-existence, regardless of
    whether the emit slug is is-a or broader-than."""
    graph = _minimal_concept_graph("aria-role")  # parent absent
    chunks = [
        {
            "id": "c-aria",
            "concept_tags": ["aria-role"],
            "learning_outcome_refs": [],
            "key_terms": [
                {
                    "term": "aria-role",
                    "definition": (
                        "An ARIA role is a type of accessibility-attribute."
                    ),
                }
            ],
        }
    ]
    assert infer_is_a(chunks, None, graph) == []


def test_emitted_broader_than_edge_validates_against_schema() -> None:
    """End-to-end: an emit-and-validate roundtrip ensures the schema
    enum, the rule emit, and the JSON shape all agree."""
    jsonschema = pytest.importorskip("jsonschema")

    graph = _minimal_concept_graph("widget", "gadget")
    chunks = [
        {
            "id": "c1",
            "concept_tags": ["widget", "gadget"],
            "learning_outcome_refs": [],
            "key_terms": [
                {
                    "term": "widget",
                    "definition": "A widget is a type of gadget for testing.",
                }
            ],
        }
    ]
    edges = infer_is_a(chunks, None, graph)
    assert len(edges) == 1
    artifact = {
        "kind": "concept_semantic",
        "generated_at": "2026-04-26T00:00:00+00:00",
        "nodes": [{"id": "widget"}, {"id": "gadget"}],
        "edges": [
            {
                **edges[0],
                # confidence already populated by rule; ensure explicit
                # for the schema validator's clarity.
            }
        ],
    }
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(instance=artifact, schema=schema)
