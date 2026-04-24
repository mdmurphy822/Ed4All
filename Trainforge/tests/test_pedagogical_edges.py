"""Tests for Worker U's Wave 5.2 pedagogical-edge rule modules.

Covers the five new edge types added via REC-LNK-04:

- ``derived-from-objective`` — chunk→LO from ``chunk.learning_outcome_refs``.
- ``defined-by`` — concept→chunk from Worker S's ``occurrences[0]``.
- ``exemplifies`` — example-chunk→concept from ``chunk_type``/``content_type_label``.
- ``misconception-of`` — misconception→concept from explicit ``concept_id``.
- ``assesses`` — question→LO from explicit ``objective_id``.

Plus the Worker S handoff carry-forward in ``_build_nodes``, the schema
enum extension, precedence non-interference with taxonomic edges,
deterministic output, and an integration test that fires all five at once.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from Trainforge.rag.inference_rules import (
    infer_assesses,
    infer_defined_by,
    infer_derived_from_objective,
    infer_exemplifies,
    infer_misconception_of,
)
from Trainforge.rag.typed_edge_inference import build_semantic_graph

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "knowledge"
    / "concept_graph_semantic.schema.json"
)

FIXED_NOW = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _empty_graph():
    return {"kind": "concept", "nodes": [], "edges": []}


def _graph(node_ids, *, occurrences_by_id=None):
    occurrences_by_id = occurrences_by_id or {}
    nodes = []
    for nid in node_ids:
        n = {"id": nid, "label": nid, "frequency": 2}
        if nid in occurrences_by_id:
            n["occurrences"] = list(occurrences_by_id[nid])
        nodes.append(n)
    return {"kind": "concept", "nodes": nodes, "edges": []}


# ---------------------------------------------------------------------------
# derived-from-objective
# ---------------------------------------------------------------------------

def test_derived_from_lo_ref_happy_path():
    chunks = [
        {
            "id": "chunk_alpha",
            "learning_outcome_refs": ["to-01", "co-05"],
            "concept_tags": [],
        }
    ]
    edges = infer_derived_from_objective(chunks, None, _empty_graph())
    assert len(edges) == 2
    targets = sorted(e["target"] for e in edges)
    assert targets == ["co-05", "to-01"]
    for e in edges:
        assert e["type"] == "derived-from-objective"
        assert e["source"] == "chunk_alpha"
        assert e["confidence"] == 1.0
        ev = e["provenance"]["evidence"]
        assert ev["chunk_id"] == "chunk_alpha"
        assert ev["objective_id"] == e["target"]


def test_derived_from_lo_ref_empty_when_no_refs():
    chunks = [{"id": "c1", "learning_outcome_refs": [], "concept_tags": []}]
    assert infer_derived_from_objective(chunks, None, _empty_graph()) == []


def test_derived_from_lo_ref_dedups_same_chunk_lo_pair():
    chunks = [
        {
            "id": "c1",
            "learning_outcome_refs": ["to-01", "to-01"],
        }
    ]
    edges = infer_derived_from_objective(chunks, None, _empty_graph())
    assert len(edges) == 1


# ---------------------------------------------------------------------------
# defined-by
# ---------------------------------------------------------------------------

def test_defined_by_from_first_mention_happy_path():
    # occurrences pre-unsorted — rule re-sorts ASC before picking [0].
    graph = _graph(["widget"], occurrences_by_id={"widget": ["chunk_b", "chunk_a", "chunk_c"]})
    edges = infer_defined_by([], None, graph)
    assert len(edges) == 1
    e = edges[0]
    assert e["source"] == "widget"
    assert e["target"] == "chunk_a"  # ASCII-ASC
    assert e["type"] == "defined-by"
    assert e["confidence"] == 0.7
    ev = e["provenance"]["evidence"]
    assert ev["chunk_id"] == "chunk_a"
    assert ev["concept_slug"] == "widget"
    assert ev["first_mention_position"] == 0


def test_defined_by_empty_when_no_occurrences():
    graph = _graph(["widget"])  # no occurrences
    assert infer_defined_by([], None, graph) == []


def test_defined_by_strips_scoped_id_prefix_in_evidence():
    graph = _graph(
        ["CRS_101:widget"],
        occurrences_by_id={"CRS_101:widget": ["chunk_a"]},
    )
    edges = infer_defined_by([], None, graph)
    assert edges[0]["provenance"]["evidence"]["concept_slug"] == "widget"
    assert edges[0]["source"] == "CRS_101:widget"


# ---------------------------------------------------------------------------
# exemplifies
# ---------------------------------------------------------------------------

def test_exemplifies_chunk_type_example():
    graph = _graph(["widget", "stage"])
    chunks = [
        {
            "id": "chunk_ex",
            "chunk_type": "example",
            "concept_tags": ["widget", "stage"],
        }
    ]
    edges = infer_exemplifies(chunks, None, graph)
    assert len(edges) == 2
    for e in edges:
        assert e["type"] == "exemplifies"
        assert e["source"] == "chunk_ex"
        assert e["confidence"] == 0.8
        assert e["provenance"]["evidence"]["content_type"] == "chunk_type"


def test_exemplifies_content_type_label_example():
    graph = _graph(["widget"])
    chunks = [
        {
            "id": "chunk_ex",
            "chunk_type": "document_text",
            "content_type_label": "example",
            "concept_tags": ["widget"],
        }
    ]
    edges = infer_exemplifies(chunks, None, graph)
    assert len(edges) == 1
    assert edges[0]["provenance"]["evidence"]["content_type"] == "content_type_label"


def test_exemplifies_skips_non_example_chunks():
    graph = _graph(["widget"])
    chunks = [
        {
            "id": "chunk_ex",
            "chunk_type": "document_text",
            "content_type_label": "explanation",
            "concept_tags": ["widget"],
        }
    ]
    assert infer_exemplifies(chunks, None, graph) == []


def test_exemplifies_filters_unknown_concept_tags():
    graph = _graph(["widget"])  # "stage" is NOT a graph node
    chunks = [
        {
            "id": "chunk_ex",
            "chunk_type": "example",
            "concept_tags": ["widget", "stage"],
        }
    ]
    edges = infer_exemplifies(chunks, None, graph)
    assert len(edges) == 1
    assert edges[0]["target"] == "widget"


# ---------------------------------------------------------------------------
# misconception-of
# ---------------------------------------------------------------------------

def test_misconception_of_empty_when_no_misconceptions_kwarg():
    assert infer_misconception_of([], None, _empty_graph()) == []


def test_misconception_of_empty_when_no_concept_id():
    mcs = [{"id": "mc_" + "a" * 16, "misconception": "x", "correction": "y"}]
    assert infer_misconception_of([], None, _empty_graph(), misconceptions=mcs) == []


def test_misconception_of_happy_path():
    mcs = [
        {
            "id": "mc_" + "a" * 16,
            "misconception": "x",
            "correction": "y",
            "concept_id": "widget",
        }
    ]
    edges = infer_misconception_of([], None, _empty_graph(), misconceptions=mcs)
    assert len(edges) == 1
    e = edges[0]
    assert e["source"] == "mc_" + "a" * 16
    assert e["target"] == "widget"
    assert e["type"] == "misconception-of"
    assert e["confidence"] == 1.0
    assert e["provenance"]["evidence"]["misconception_id"] == e["source"]
    assert e["provenance"]["evidence"]["concept_id"] == "widget"


# ---------------------------------------------------------------------------
# assesses
# ---------------------------------------------------------------------------

def test_assesses_empty_when_no_questions_kwarg():
    assert infer_assesses([], None, _empty_graph()) == []


def test_assesses_happy_path():
    qs = [
        {"id": "q1", "objective_id": "to-01"},
        {"id": "q2", "objective_id": "co-05", "source_chunk_id": "chunk_a"},
    ]
    edges = infer_assesses([], None, _empty_graph(), questions=qs)
    assert len(edges) == 2
    for e in edges:
        assert e["type"] == "assesses"
        assert e["confidence"] == 1.0
    q2_edge = next(e for e in edges if e["source"] == "q2")
    assert q2_edge["provenance"]["evidence"]["source_chunk_id"] == "chunk_a"
    q1_edge = next(e for e in edges if e["source"] == "q1")
    assert "source_chunk_id" not in q1_edge["provenance"]["evidence"]


def test_assesses_skips_questions_without_objective_id():
    qs = [{"id": "q1"}]
    assert infer_assesses([], None, _empty_graph(), questions=qs) == []


# ---------------------------------------------------------------------------
# Schema + precedence + deterministic output
# ---------------------------------------------------------------------------

def test_edge_enum_includes_new_types():
    jsonschema = pytest.importorskip("jsonschema")
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    enum_values = set(
        schema["properties"]["edges"]["items"]["properties"]["type"]["enum"]
    )
    assert enum_values == {
        "prerequisite",
        "is-a",
        "related-to",
        "assesses",
        "exemplifies",
        "misconception-of",
        "derived-from-objective",
        "defined-by",
        "targets-concept",
    }

    # Sanity: validate a synthetic artifact containing all 5 new edge types.
    artifact = {
        "kind": "concept_semantic",
        "generated_at": FIXED_NOW.isoformat(),
        "nodes": [
            {"id": "widget"},
            {"id": "chunk_a"},
        ],
        "edges": [
            {
                "source": "chunk_a",
                "target": "to-01",
                "type": "derived-from-objective",
                "confidence": 1.0,
                "provenance": {"rule": "derived_from_lo_ref", "rule_version": 1},
            },
            {
                "source": "widget",
                "target": "chunk_a",
                "type": "defined-by",
                "confidence": 0.7,
                "provenance": {"rule": "defined_by_from_first_mention", "rule_version": 1},
            },
            {
                "source": "chunk_a",
                "target": "widget",
                "type": "exemplifies",
                "confidence": 0.8,
                "provenance": {"rule": "exemplifies_from_example_chunks", "rule_version": 1},
            },
            {
                "source": "mc_" + "a" * 16,
                "target": "widget",
                "type": "misconception-of",
                "confidence": 1.0,
                "provenance": {"rule": "misconception_of_from_misconception_ref", "rule_version": 1},
            },
            {
                "source": "q1",
                "target": "to-01",
                "type": "assesses",
                "confidence": 1.0,
                "provenance": {"rule": "assesses_from_question_lo", "rule_version": 1},
            },
        ],
    }
    jsonschema.validate(instance=artifact, schema=schema)


def test_precedence_new_types_do_not_drop_is_a():
    """is-a (tier 3) still wins over a tier-2 pedagogical edge on the same pair."""
    graph = {
        "kind": "concept",
        "nodes": [
            {"id": "widget", "frequency": 10},
            {"id": "gadget", "frequency": 10},
        ],
        "edges": [],
    }
    chunks = [
        {
            "id": "c1",
            "concept_tags": ["widget", "gadget"],
            "learning_outcome_refs": [],
            "key_terms": [
                {
                    "term": "widget",
                    "definition": "A widget is a type of gadget used in testing.",
                }
            ],
        }
    ]
    artifact = build_semantic_graph(chunks, None, graph, now=FIXED_NOW)
    pair_edges = [
        e for e in artifact["edges"]
        if {e["source"], e["target"]} == {"widget", "gadget"}
    ]
    # is-a fires on (widget, gadget); no new pedagogical edge targets that
    # same concept↔concept pair, so is-a stands alone.
    assert len(pair_edges) == 1
    assert pair_edges[0]["type"] == "is-a"


def test_precedence_exemplifies_beats_related_to_on_same_pair():
    """Tier-2 ``exemplifies`` should drop a tier-1 ``related-to`` on the same pair."""
    # Set up so (chunk_ex, widget) would be emitted by both rules. Since
    # related-to is concept↔concept undirected, we need a pair that actually
    # collides: craft a graph where chunk_ex appears as a node for the
    # test (synthetic scenario — normally chunks aren't in the concept
    # graph, but the precedence logic should still honour tiers).
    graph = {
        "kind": "concept",
        "nodes": [
            {"id": "chunk_ex", "frequency": 5},
            {"id": "widget", "frequency": 5},
        ],
        "edges": [
            # High weight so related-to fires at default threshold=3.
            {"source": "chunk_ex", "target": "widget", "weight": 9, "relation_type": "co-occurs"},
        ],
    }
    chunks = [
        {
            "id": "chunk_ex",
            "chunk_type": "example",
            "concept_tags": ["widget"],
            "learning_outcome_refs": [],
        }
    ]
    artifact = build_semantic_graph(chunks, None, graph, now=FIXED_NOW)
    # Now (chunk_ex, widget) has both an exemplifies edge (tier 2, directed)
    # and a related-to (tier 1, undirected). The directed-pair drop rule in
    # _apply_precedence should drop related-to because the directed pair
    # for (chunk_ex, widget) is already claimed.
    edges_on_pair = [
        e for e in artifact["edges"]
        if {e["source"], e["target"]} == {"chunk_ex", "widget"}
    ]
    types_on_pair = {e["type"] for e in edges_on_pair}
    assert "exemplifies" in types_on_pair
    assert "related-to" not in types_on_pair


def test_deterministic_output():
    chunks = [
        {
            "id": "c1",
            "chunk_type": "example",
            "concept_tags": ["widget"],
            "learning_outcome_refs": ["to-01"],
        }
    ]
    graph = _graph(["widget"], occurrences_by_id={"widget": ["c1"]})
    a = build_semantic_graph(chunks, None, graph, now=FIXED_NOW)
    b = build_semantic_graph(chunks, None, graph, now=FIXED_NOW)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_occurrences_carry_forward_in_build_nodes():
    """Worker S handoff: occurrences[] flows from concept_graph into the
    semantic graph node output via _build_nodes."""
    graph = _graph(["widget"], occurrences_by_id={"widget": ["chunk_a", "chunk_b"]})
    artifact = build_semantic_graph([], None, graph, now=FIXED_NOW)
    widget_node = next(n for n in artifact["nodes"] if n["id"] == "widget")
    assert widget_node["occurrences"] == ["chunk_a", "chunk_b"]


def test_occurrences_absent_when_source_lacks_it():
    graph = _graph(["widget"])  # no occurrences
    artifact = build_semantic_graph([], None, graph, now=FIXED_NOW)
    widget_node = next(n for n in artifact["nodes"] if n["id"] == "widget")
    assert "occurrences" not in widget_node


# ---------------------------------------------------------------------------
# Integration: all 5 new edge types fire together
# ---------------------------------------------------------------------------

def test_integration_all_five_new_edge_types():
    graph = _graph(
        ["widget", "gadget"],
        occurrences_by_id={"widget": ["chunk_ex"], "gadget": ["chunk_a"]},
    )
    # Add a related-to edge candidate too; not the focus of this test.
    chunks = [
        {
            "id": "chunk_ex",
            "chunk_type": "example",
            "concept_tags": ["widget"],
            "learning_outcome_refs": ["to-01"],
        },
        {
            "id": "chunk_a",
            "chunk_type": "document_text",
            "concept_tags": ["gadget"],
            "learning_outcome_refs": ["co-05"],
        },
    ]
    misconceptions = [
        {
            "id": "mc_" + "a" * 16,
            "misconception": "w is g",
            "correction": "w is not g",
            "concept_id": "widget",
        }
    ]
    questions = [
        {"id": "q1", "objective_id": "to-01", "source_chunk_id": "chunk_ex"},
    ]
    artifact = build_semantic_graph(
        chunks,
        None,
        graph,
        now=FIXED_NOW,
        misconceptions=misconceptions,
        questions=questions,
    )

    types_emitted = {e["type"] for e in artifact["edges"]}
    for required in (
        "derived-from-objective",
        "defined-by",
        "exemplifies",
        "misconception-of",
        "assesses",
    ):
        assert required in types_emitted, (required, types_emitted)

    # And the existing 3-tier system still validates on this artifact.
    jsonschema = pytest.importorskip("jsonschema")
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.validate(instance=artifact, schema=schema)
