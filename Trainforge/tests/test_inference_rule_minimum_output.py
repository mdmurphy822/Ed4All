"""Wave 82 regression tests for typed-edge inference rule outputs.

Background: the deep audit on rdf-shacl-551-2 found 5 of 7 inference rules
silently emitting zero edges in a shipped `concept_graph_semantic.json`
while the same rule code, run against the same persisted inputs, produced
the expected edge counts. The failure mode was orchestration-level
(stale-input drift between chunk regen and graph regen passes), not a
bug in the rule code itself — the rule unit tests were missing, so the
silent-zero failure mode had no automated detection.

These tests pin **minimum-viable input → non-zero output** contracts for
each rule. They run in milliseconds (inline fixtures), and any change
that makes a rule emit zero edges from these representative inputs will
fail loudly. Pair with the broader per-rule output gate
(`lib/validators/semantic_graph_rule_output.py`, Phase A3) which catches
regressions on real corpora.
"""

from __future__ import annotations

from typing import Any, Dict, List

from Trainforge.rag.inference_rules.defined_by_from_first_mention import (
    infer as infer_defined_by,
)
from Trainforge.rag.inference_rules.derived_from_lo_ref import (
    infer as infer_derived_from_lo_ref,
)


# ---------------------------------------------------------------------------
# defined_by_from_first_mention
# ---------------------------------------------------------------------------


def _concept_graph_with_occurrences(
    nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Wrap a list of concept-graph nodes in the canonical artifact shape."""
    return {
        "kind": "concept",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "nodes": nodes,
        "edges": [],
    }


class TestDefinedByFromFirstMention:
    """Pin: any node with non-empty `occurrences` produces exactly one edge."""

    def test_single_node_single_occurrence_produces_one_edge(self):
        cg = _concept_graph_with_occurrences([
            {
                "id": "rdf-graph",
                "label": "RDF Graph",
                "frequency": 1,
                "occurrences": ["chunk_001"],
            },
        ])
        edges = infer_defined_by(chunks=[], course=None, concept_graph=cg)
        assert len(edges) == 1
        e = edges[0]
        assert e["source"] == "rdf-graph"
        assert e["target"] == "chunk_001"
        assert e["type"] == "defined-by"
        assert e["confidence"] == 0.7
        assert e["provenance"]["rule"] == "defined_by_from_first_mention"
        assert e["provenance"]["evidence"]["first_mention_position"] == 0

    def test_multiple_occurrences_picks_first_by_id_sort(self):
        cg = _concept_graph_with_occurrences([
            {
                "id": "shacl-shape",
                "label": "SHACL Shape",
                "frequency": 3,
                "occurrences": ["chunk_055", "chunk_003", "chunk_021"],
            },
        ])
        edges = infer_defined_by(chunks=[], course=None, concept_graph=cg)
        assert len(edges) == 1
        # ASC sort → chunk_003 wins regardless of input order.
        assert edges[0]["target"] == "chunk_003"

    def test_node_without_occurrences_emits_no_edge(self):
        cg = _concept_graph_with_occurrences([
            {"id": "orphan-concept", "label": "Orphan", "frequency": 1},
            {"id": "empty-occ", "label": "Empty", "frequency": 1, "occurrences": []},
        ])
        edges = infer_defined_by(chunks=[], course=None, concept_graph=cg)
        assert edges == []

    def test_many_nodes_one_edge_per_node_with_occurrences(self):
        # Regression for the audit's 0/424 failure: with 10 nodes all
        # carrying occurrences, we MUST get 10 edges. Anything less is a
        # silent-zero regression of the kind that shipped on rdf-shacl-551-2.
        nodes = [
            {
                "id": f"concept-{i:03d}",
                "label": f"Concept {i}",
                "frequency": 1,
                "occurrences": [f"chunk_{i:03d}"],
            }
            for i in range(10)
        ]
        edges = infer_defined_by(
            chunks=[], course=None, concept_graph=_concept_graph_with_occurrences(nodes)
        )
        assert len(edges) == 10
        assert {e["source"] for e in edges} == {n["id"] for n in nodes}

    def test_output_deterministic_sort(self):
        cg = _concept_graph_with_occurrences([
            {"id": "z-concept", "label": "Z", "frequency": 1, "occurrences": ["chunk_a"]},
            {"id": "a-concept", "label": "A", "frequency": 1, "occurrences": ["chunk_z"]},
            {"id": "m-concept", "label": "M", "frequency": 1, "occurrences": ["chunk_m"]},
        ])
        edges = infer_defined_by(chunks=[], course=None, concept_graph=cg)
        assert [e["source"] for e in edges] == ["a-concept", "m-concept", "z-concept"]


# ---------------------------------------------------------------------------
# derived_from_lo_ref
# ---------------------------------------------------------------------------


class TestDerivedFromLoRef:
    """Pin: any chunk with non-empty `learning_outcome_refs` produces edges."""

    def test_single_chunk_single_ref_produces_one_edge(self):
        chunks = [
            {"id": "chunk_001", "learning_outcome_refs": ["TO-01"]},
        ]
        edges = infer_derived_from_lo_ref(chunks=chunks, course=None, concept_graph={})
        assert len(edges) == 1
        e = edges[0]
        assert e["source"] == "chunk_001"
        assert e["target"] == "TO-01"
        assert e["type"] == "derived-from-objective"
        assert e["confidence"] == 1.0
        assert e["provenance"]["rule"] == "derived_from_lo_ref"
        assert e["provenance"]["evidence"]["objective_id"] == "TO-01"

    def test_multiple_refs_per_chunk_produce_multiple_edges(self):
        # Regression for the audit's 0/969 failure: a chunk with N LO refs
        # must produce N edges. The rdf-shacl-551 corpus averages 20.9 refs
        # per chunk, so the "0 produced" outcome is exactly what this test
        # detects.
        chunks = [
            {
                "id": "chunk_001",
                "learning_outcome_refs": ["TO-01", "TO-02", "CO-03", "CO-04"],
            },
        ]
        edges = infer_derived_from_lo_ref(chunks=chunks, course=None, concept_graph={})
        assert len(edges) == 4
        assert {e["target"] for e in edges} == {"TO-01", "TO-02", "CO-03", "CO-04"}

    def test_chunk_without_refs_emits_nothing(self):
        chunks = [
            {"id": "chunk_001"},
            {"id": "chunk_002", "learning_outcome_refs": []},
            {"id": "chunk_003", "learning_outcome_refs": None},
        ]
        edges = infer_derived_from_lo_ref(chunks=chunks, course=None, concept_graph={})
        assert edges == []

    def test_duplicate_refs_collapsed_per_chunk(self):
        chunks = [
            {
                "id": "chunk_001",
                "learning_outcome_refs": ["TO-01", "TO-01", "TO-02"],
            },
        ]
        edges = infer_derived_from_lo_ref(chunks=chunks, course=None, concept_graph={})
        assert len(edges) == 2

    def test_chunks_without_id_skipped(self):
        chunks = [
            {"learning_outcome_refs": ["TO-01"]},  # no id
            {"id": "chunk_001", "learning_outcome_refs": ["TO-02"]},
        ]
        edges = infer_derived_from_lo_ref(chunks=chunks, course=None, concept_graph={})
        assert len(edges) == 1
        assert edges[0]["source"] == "chunk_001"

    def test_corpus_scale_minimum_output(self):
        # Pin: 100 chunks × 5 refs each = 500 edges. The rdf-shacl-551
        # ratio is 295 × 20.9 ≈ 6,166 — anything that emits zero in this
        # shape is the audit's exact failure mode.
        chunks = [
            {
                "id": f"chunk_{i:03d}",
                "learning_outcome_refs": [f"TO-{j:02d}" for j in range(1, 6)],
            }
            for i in range(100)
        ]
        edges = infer_derived_from_lo_ref(chunks=chunks, course=None, concept_graph={})
        assert len(edges) == 500

    def test_output_deterministic_sort(self):
        chunks = [
            {"id": "chunk_z", "learning_outcome_refs": ["TO-99"]},
            {"id": "chunk_a", "learning_outcome_refs": ["TO-01"]},
            {"id": "chunk_m", "learning_outcome_refs": ["TO-50"]},
        ]
        edges = infer_derived_from_lo_ref(chunks=chunks, course=None, concept_graph={})
        assert [e["source"] for e in edges] == ["chunk_a", "chunk_m", "chunk_z"]
