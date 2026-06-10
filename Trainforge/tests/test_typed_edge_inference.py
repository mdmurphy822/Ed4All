"""Tests for Worker F's typed-edge concept-graph inference.

Covers the eight must-have checks from the Worker F spec:

1. `is_a` rule emits an edge when the definition phrase and the parent term
   both resolve to concept-graph nodes.
2. `is_a` rule emits nothing when the parent term is not in the graph.
3. `prerequisite` rule emits `B --prerequisite--> A` when A first appears at
   an earlier LO position than B.
4. `prerequisite` rule emits nothing when both concepts first appear at the
   same LO position.
5. `related-to` rule respects the co-occurrence threshold (>=3 default).
6. Precedence: `is-a` wins over `related-to` on the same (source, target)
   pair.
7. Deterministic fallback: two back-to-back invocations produce
   byte-identical artifacts when `generated_at` is held fixed.
8. Emitted artifact validates against the schema.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from Trainforge.rag.inference_rules import (
    infer_is_a,
    infer_prerequisite,
    infer_related,
)
from Trainforge.rag.typed_edge_inference import build_semantic_graph

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "mini_course_typed_graph"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "knowledge"
    / "concept_graph_semantic.schema.json"
)

FIXED_NOW = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _load_fixture():
    with open(FIXTURE_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    with open(FIXTURE_DIR / "course.json", encoding="utf-8") as f:
        course = json.load(f)
    with open(FIXTURE_DIR / "concept_graph.json", encoding="utf-8") as f:
        concept_graph = json.load(f)
    with open(FIXTURE_DIR / "expected_semantic_graph.json", encoding="utf-8") as f:
        expected = json.load(f)
    return chunks, course, concept_graph, expected


def _minimal_graph(node_ids):
    return {
        "kind": "concept",
        "nodes": [{"id": n, "label": n, "frequency": 2} for n in node_ids],
        "edges": [],
    }


# ---------------------------------------------------------------------------
# 1. is-a fires when both terms are nodes
# ---------------------------------------------------------------------------

def test_is_a_fires_when_both_terms_are_nodes():
    graph = _minimal_graph(["aria-role", "accessibility-attribute"])
    chunks = [
        {
            "id": "c-aria",
            "concept_tags": ["aria-role", "accessibility-attribute"],
            "learning_outcome_refs": [],
            "key_terms": [
                {
                    "term": "aria-role",
                    "definition": "An ARIA role is a type of accessibility-attribute that describes a widget.",
                }
            ],
        }
    ]
    edges = infer_is_a(chunks, None, graph)
    assert len(edges) == 1, edges
    e = edges[0]
    assert e["source"] == "aria-role"
    assert e["target"] == "accessibility-attribute"
    # Both endpoints are concept-graph nodes (cf:Concept instances), so
    # the rule emits ``broader-than`` (skos:broader) — the W3C-canonical
    # predicate for SKOS-concept hierarchy. ``is-a`` (rdfs:subClassOf)
    # remains reserved for class-level subsumption (see
    # lib/ontology/edge_predicates.py).
    assert e["type"] == "broader-than"
    assert e["provenance"]["rule"] == "is_a_from_key_terms"


# ---------------------------------------------------------------------------
# 2. is-a suppresses edges when the parent isn't in the graph
# ---------------------------------------------------------------------------

def test_is_a_no_edge_when_parent_missing():
    graph = _minimal_graph(["aria-role"])  # no parent node present
    chunks = [
        {
            "id": "c-aria",
            "concept_tags": ["aria-role"],
            "learning_outcome_refs": [],
            "key_terms": [
                {
                    "term": "aria-role",
                    "definition": "An ARIA role is a type of accessibility-attribute.",
                }
            ],
        }
    ]
    edges = infer_is_a(chunks, None, graph)
    assert edges == []


# ---------------------------------------------------------------------------
# 3. prerequisite fires when earliest-LO positions differ
# ---------------------------------------------------------------------------

def test_prerequisite_fires_on_lo_order_skew():
    course = {
        "learning_outcomes": [
            {"id": "co-01", "statement": "A"},
            {"id": "co-05", "statement": "B"},
        ]
    }
    graph = _minimal_graph(["a", "b"])
    chunks = [
        {
            "id": "ca",
            "concept_tags": ["a", "b"],  # share a chunk so co-occurrence is true
            "learning_outcome_refs": ["co-01"],
        },
        {
            "id": "cb",
            "concept_tags": ["b"],
            "learning_outcome_refs": ["co-05"],
        },
    ]
    # "a" first at position 0; "b" first at position 0 too (both in ca).
    # Adjust: remove "b" from ca so b's first position is co-05.
    chunks[0]["concept_tags"] = ["a"]
    # But they still need to co-occur. Add a third chunk where both appear
    # at a non-constraining LO — position has to be derived from first
    # occurrence.
    chunks.append({
        "id": "cc",
        "concept_tags": ["a", "b"],
        "learning_outcome_refs": ["co-05"],
    })
    edges = infer_prerequisite(chunks, course, graph)
    # a first at co-01 (pos 0); b first at co-05 (pos 1) → b depends on a.
    assert any(e["source"] == "b" and e["target"] == "a" and e["type"] == "prerequisite" for e in edges), edges


# ---------------------------------------------------------------------------
# 4. prerequisite rule emits nothing when both concepts share their first LO
# ---------------------------------------------------------------------------

def test_prerequisite_no_edge_when_same_lo_position():
    course = {
        "learning_outcomes": [
            {"id": "co-01", "statement": "A"},
            {"id": "co-02", "statement": "B"},
        ]
    }
    graph = _minimal_graph(["x", "y"])
    chunks = [
        {
            "id": "c1",
            "concept_tags": ["x", "y"],
            "learning_outcome_refs": ["co-01"],
        }
    ]
    edges = infer_prerequisite(chunks, course, graph)
    assert edges == []


# ---------------------------------------------------------------------------
# 4b. TRAINFORGE_PREREQ_LO_ADJACENT_ONLY — quadratic-closure mitigation
# ---------------------------------------------------------------------------

def _linear_chain_inputs():
    """3 concepts on a linear LO chain, all co-occurring in one chunk.

    a@co-01 (pos0), b@co-02 (pos1), c@co-03 (pos2). All three share a chunk,
    so legacy emits the full closure (b<-a, c<-a, c<-b). The reduced path
    keeps only the adjacent links (b<-a, c<-b) — c<-a is the skip edge.
    """
    course = {
        "learning_outcomes": [
            {"id": "co-01", "statement": "A"},
            {"id": "co-02", "statement": "B"},
            {"id": "co-03", "statement": "C"},
        ]
    }
    graph = _minimal_graph(["a", "b", "c"])
    chunks = [
        # First-occurrence LO positions: a@co-01 (pos0), b@co-02 (pos1),
        # c@co-03 (pos2). Each concept's EARLIEST LO must be distinct, so
        # the introducing chunks come first and are single-tag.
        {"id": "c-a", "concept_tags": ["a"], "learning_outcome_refs": ["co-01"]},
        {"id": "c-b", "concept_tags": ["b"], "learning_outcome_refs": ["co-02"]},
        {"id": "c-c", "concept_tags": ["c"], "learning_outcome_refs": ["co-03"]},
        # All three co-occur in a later chunk (gives the co-occurrence +
        # intermediate signal without disturbing the earliest-LO positions).
        {"id": "c-all", "concept_tags": ["a", "b", "c"], "learning_outcome_refs": ["co-03"]},
    ]
    return chunks, course, graph


def test_prerequisite_legacy_emits_transitive_closure():
    """Flag OFF (default): linear chain emits the full O(n^2) closure."""
    chunks, course, graph = _linear_chain_inputs()
    edges = infer_prerequisite(chunks, course, graph)
    pairs = {(e["source"], e["target"]) for e in edges}
    # later-depends-on-earlier: b<-a, c<-a (skip), c<-b
    assert pairs == {("b", "a"), ("c", "a"), ("c", "b")}, pairs


def test_prerequisite_adjacent_only_drops_skip_edge(monkeypatch):
    """Flag ON: transitive reduction drops the c<-a skip edge."""
    monkeypatch.setenv("TRAINFORGE_PREREQ_LO_ADJACENT_ONLY", "true")
    chunks, course, graph = _linear_chain_inputs()
    edges = infer_prerequisite(chunks, course, graph)
    pairs = {(e["source"], e["target"]) for e in edges}
    # Only the adjacent chain links survive; the closure skip edge is gone.
    assert pairs == {("b", "a"), ("c", "b")}, pairs
    assert ("c", "a") not in pairs
    # Reachability preserved: c -> b -> a is intact.


def test_prerequisite_adjacent_only_no_intermediate_keeps_edge(monkeypatch):
    """Flag ON but no co-occurring concept between the pair → edge kept.

    Reduction only suppresses an edge when a genuine intermediate exists; a
    bare two-concept span must still emit its edge.
    """
    monkeypatch.setenv("TRAINFORGE_PREREQ_LO_ADJACENT_ONLY", "true")
    course = {
        "learning_outcomes": [
            {"id": "co-01", "statement": "A"},
            {"id": "co-03", "statement": "C"},
        ]
    }
    graph = _minimal_graph(["a", "c"])
    chunks = [
        {"id": "c-a", "concept_tags": ["a"], "learning_outcome_refs": ["co-01"]},
        {"id": "c-c", "concept_tags": ["c"], "learning_outcome_refs": ["co-03"]},
        {"id": "c-all", "concept_tags": ["a", "c"], "learning_outcome_refs": ["co-03"]},
    ]
    edges = infer_prerequisite(chunks, course, graph)
    pairs = {(e["source"], e["target"]) for e in edges}
    assert pairs == {("c", "a")}, pairs


def test_prerequisite_adjacent_only_demotes_text_order_conflict(monkeypatch):
    """Flag ON: LO order disagreeing with text order → confidence demoted."""
    monkeypatch.setenv("TRAINFORGE_PREREQ_LO_ADJACENT_ONLY", "true")
    course = {
        "learning_outcomes": [
            {"id": "co-01", "statement": "A"},
            {"id": "co-05", "statement": "B"},
        ]
    }
    graph = _minimal_graph(["a", "b"])
    # Text order (chunk list order): "b" is introduced FIRST (index 0),
    # "a" later (index 1) — but LO order says "a" (co-01) precedes
    # "b" (co-05). The orderings disagree → demotion + annotation.
    chunks = [
        {"id": "c-b", "concept_tags": ["b"], "learning_outcome_refs": ["co-05"]},
        {"id": "c-a", "concept_tags": ["a"], "learning_outcome_refs": ["co-01"]},
        {"id": "c-both", "concept_tags": ["a", "b"], "learning_outcome_refs": ["co-05"]},
    ]
    edges = infer_prerequisite(chunks, course, graph)
    assert len(edges) == 1, edges
    edge = edges[0]
    # b depends on a (a is the earlier-LO prerequisite).
    assert edge["source"] == "b" and edge["target"] == "a"
    assert edge["confidence"] == 0.3
    ev = edge["provenance"]["evidence"]
    assert ev.get("lo_order_vs_text_order_conflict") is True


def test_prerequisite_adjacent_only_no_demotion_when_orders_agree(monkeypatch):
    """Flag ON: LO order and text order agree → base confidence, no flag."""
    monkeypatch.setenv("TRAINFORGE_PREREQ_LO_ADJACENT_ONLY", "true")
    course = {
        "learning_outcomes": [
            {"id": "co-01", "statement": "A"},
            {"id": "co-05", "statement": "B"},
        ]
    }
    graph = _minimal_graph(["a", "b"])
    # "a" introduced first in text (index 0) AND earlier by LO (co-01) —
    # orderings agree, so no demotion.
    chunks = [
        {"id": "c-a", "concept_tags": ["a"], "learning_outcome_refs": ["co-01"]},
        {"id": "c-b", "concept_tags": ["b"], "learning_outcome_refs": ["co-05"]},
        {"id": "c-both", "concept_tags": ["a", "b"], "learning_outcome_refs": ["co-05"]},
    ]
    edges = infer_prerequisite(chunks, course, graph)
    assert len(edges) == 1, edges
    edge = edges[0]
    assert edge["source"] == "b" and edge["target"] == "a"
    assert edge["confidence"] == 0.6
    assert "lo_order_vs_text_order_conflict" not in edge["provenance"]["evidence"]


def test_prerequisite_flag_off_byte_identical_to_legacy(monkeypatch):
    """Flag OFF path is byte-identical to the pre-mitigation output.

    Pins the backward-compat contract: with the flag unset, the reduced
    code path must produce the exact closure + 0.6 confidence + no
    conflict annotation that legacy corpora regenerate.
    """
    monkeypatch.delenv("TRAINFORGE_PREREQ_LO_ADJACENT_ONLY", raising=False)
    chunks, course, graph = _linear_chain_inputs()
    edges = infer_prerequisite(chunks, course, graph)
    # Full closure, all base confidence, no conflict annotation anywhere.
    assert {(e["source"], e["target"]) for e in edges} == {
        ("b", "a"), ("c", "a"), ("c", "b"),
    }
    for e in edges:
        assert e["confidence"] == 0.6
        assert "lo_order_vs_text_order_conflict" not in e["provenance"]["evidence"]
        assert "target_first_text_index" not in e["provenance"]["evidence"]


# ---------------------------------------------------------------------------
# 5. related-to threshold
# ---------------------------------------------------------------------------

def test_related_threshold_default_three():
    graph = {
        "kind": "concept",
        "nodes": [
            {"id": "a", "frequency": 4},
            {"id": "b", "frequency": 4},
            {"id": "c", "frequency": 2},
        ],
        "edges": [
            {"source": "a", "target": "b", "weight": 3, "relation_type": "co-occurs"},
            {"source": "a", "target": "c", "weight": 2, "relation_type": "co-occurs"},
        ],
    }
    edges = infer_related([], None, graph)
    tuples = {(e["source"], e["target"]) for e in edges}
    # a↔b passes, a↔c does not.
    assert ("a", "b") in tuples
    assert ("a", "c") not in tuples and ("c", "a") not in tuples


# ---------------------------------------------------------------------------
# 6. precedence: is-a beats related-to on the same pair
# ---------------------------------------------------------------------------

def test_precedence_is_a_beats_related_to():
    graph = {
        "kind": "concept",
        "nodes": [
            {"id": "aria-role", "frequency": 10},
            {"id": "accessibility-attribute", "frequency": 10},
        ],
        # High co-occurrence so related-to would fire.
        "edges": [
            {"source": "aria-role", "target": "accessibility-attribute", "weight": 9, "relation_type": "co-occurs"},
        ],
    }
    chunks = [
        {
            "id": "c1",
            "concept_tags": ["aria-role", "accessibility-attribute"],
            "learning_outcome_refs": [],
            "key_terms": [
                {
                    "term": "aria-role",
                    "definition": "An ARIA role is a type of accessibility-attribute describing intent.",
                }
            ],
        }
    ]
    graph_out = build_semantic_graph(chunks, None, graph, now=FIXED_NOW)
    # The (aria-role, accessibility-attribute) pair should appear as
    # ``broader-than`` (the canonical SKOS-concept hierarchy slug; both
    # endpoints are cf:Concept instances), not as related-to. Both the
    # broader-than and is-a slugs share the same precedence tier, so
    # the precedence semantics versus related-to are unchanged.
    pair_edges = [
        e for e in graph_out["edges"]
        if set([e["source"], e["target"]]) == {"aria-role", "accessibility-attribute"}
    ]
    assert len(pair_edges) == 1, pair_edges
    assert pair_edges[0]["type"] == "broader-than"


# ---------------------------------------------------------------------------
# 7. Deterministic fallback — two runs produce byte-identical output
# ---------------------------------------------------------------------------

def test_deterministic_fallback_produces_identical_artifacts():
    chunks, course, concept_graph, _ = _load_fixture()
    g1 = build_semantic_graph(chunks, course, concept_graph, now=FIXED_NOW)
    g2 = build_semantic_graph(chunks, course, concept_graph, now=FIXED_NOW)
    assert json.dumps(g1, sort_keys=True) == json.dumps(g2, sort_keys=True)


# ---------------------------------------------------------------------------
# 8. Schema validation
# ---------------------------------------------------------------------------

def test_emitted_artifact_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    chunks, course, concept_graph, _ = _load_fixture()
    artifact = build_semantic_graph(chunks, course, concept_graph, now=FIXED_NOW)
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.validate(instance=artifact, schema=schema)


# ---------------------------------------------------------------------------
# Extra: golden expected tuples on the fixture exercise all rules together.
# ---------------------------------------------------------------------------

def test_fixture_golden_edge_tuples():
    chunks, course, concept_graph, expected = _load_fixture()
    artifact = build_semantic_graph(chunks, course, concept_graph, now=FIXED_NOW)
    actual_tuples = [[e["type"], e["source"], e["target"]] for e in artifact["edges"]]
    expected_tuples = [list(t) for t in expected["expected_edge_tuples"]]
    assert actual_tuples == expected_tuples, {
        "expected": expected_tuples,
        "actual": actual_tuples,
    }
