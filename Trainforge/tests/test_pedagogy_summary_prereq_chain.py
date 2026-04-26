"""Wave 82 tests for pedagogy_model.prerequisite_chain population.

Audit reproducer: rdf-shacl-551-2 shipped pedagogy_model.json with
``prerequisite_chain: []`` and ``prerequisite_violations: []`` despite
the pedagogy graph carrying 404 ``prerequisite_of`` edges. The
chunk-based legacy path consults ``chunk.prereq_concepts``, which the
Courseforge IMSCC parser doesn't populate for this corpus, so the
chain stays empty. Wave 82 threads the pedagogy_graph through and
reads its prerequisite_of edges directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from Trainforge.process_course import CourseProcessor


def _bare_processor() -> CourseProcessor:
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "TEST_101"
    proc.objectives = {}
    return proc


def _make_chunks() -> List[Dict[str, Any]]:
    """Minimum chunks needed for module_sequence to populate."""
    return [
        {
            "id": "test_101_chunk_00001",
            "source": {"module_id": "week_01", "module_title": "Intro"},
            "concept_tags": ["rdf", "graph"],
            "learning_outcome_refs": ["TO-01"],
            "bloom_level": "understand",
        },
        {
            "id": "test_101_chunk_00002",
            "source": {"module_id": "week_02", "module_title": "Schema"},
            "concept_tags": ["rdfs", "rdf"],
            "learning_outcome_refs": ["TO-02"],
            "bloom_level": "apply",
        },
    ]


def _make_pedagogy_graph_with_prereq_edges() -> Dict[str, Any]:
    """Pedagogy graph with two prerequisite_of edges (concept: prefix shape)."""
    return {
        "kind": "pedagogy",
        "schema_version": "v2",
        "course_id": "TEST_101",
        "nodes": [],
        "edges": [
            {
                "source": "concept:rdf",
                "target": "concept:rdfs",
                "relation_type": "prerequisite_of",
                "confidence": 3,
            },
            {
                "source": "concept:rdfs",
                "target": "concept:owl",
                "relation_type": "prerequisite_of",
                "confidence": 2,
            },
            {
                # Non-prereq edge: must be ignored.
                "source": "concept:rdf",
                "target": "concept:graph",
                "relation_type": "co-occurs",
                "confidence": 5,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Graph-driven path (Wave 82 — preferred when pedagogy_graph is supplied)
# ---------------------------------------------------------------------------


class TestGraphDrivenPath:
    def test_audit_reproducer_chain_populates_from_graph(self):
        # Pre-Wave-82: empty chain because chunks don't carry prereq_concepts.
        # Post-Wave-82: chain has 2 entries from the graph's 2 prereq edges.
        proc = _bare_processor()
        chunks = _make_chunks()
        graph = _make_pedagogy_graph_with_prereq_edges()
        summary = proc._build_pedagogy_summary(
            chunks=chunks, pedagogy_graph=graph
        )
        chain = summary["prerequisite_chain"]
        assert len(chain) == 2
        # Concept slugs come through with the concept: prefix stripped.
        rdf_entry = next(c for c in chain if c["concept"] == "rdf")
        assert rdf_entry["required_for"] == "rdfs"
        assert rdf_entry["confidence"] == 3
        rdfs_entry = next(c for c in chain if c["concept"] == "rdfs")
        assert rdfs_entry["required_for"] == "owl"
        assert rdfs_entry["confidence"] == 2

    def test_co_occurs_edges_excluded(self):
        proc = _bare_processor()
        graph = _make_pedagogy_graph_with_prereq_edges()
        summary = proc._build_pedagogy_summary(
            chunks=_make_chunks(), pedagogy_graph=graph
        )
        # 3 edges in graph; only 2 are prerequisite_of → 2 entries.
        assert len(summary["prerequisite_chain"]) == 2

    def test_chain_deterministic_sort(self):
        proc = _bare_processor()
        graph = {
            "edges": [
                {
                    "source": f"concept:z-{i}",
                    "target": f"concept:a-{i}",
                    "relation_type": "prerequisite_of",
                    "confidence": 1,
                }
                for i in range(3)
            ] + [
                {
                    "source": "concept:alpha",
                    "target": "concept:beta",
                    "relation_type": "prerequisite_of",
                    "confidence": 1,
                },
            ],
        }
        summary = proc._build_pedagogy_summary(
            chunks=_make_chunks(), pedagogy_graph=graph
        )
        chain = summary["prerequisite_chain"]
        # Sorted by (concept, required_for). "alpha" < "z-*".
        assert chain[0]["concept"] == "alpha"
        # Remaining are z-0, z-1, z-2 in sorted order.
        z_concepts = [c["concept"] for c in chain[1:]]
        assert z_concepts == sorted(z_concepts)

    def test_empty_graph_gives_empty_chain(self):
        proc = _bare_processor()
        empty_graph = {"edges": []}
        summary = proc._build_pedagogy_summary(
            chunks=_make_chunks(), pedagogy_graph=empty_graph
        )
        assert summary["prerequisite_chain"] == []
        assert summary["prerequisite_violations"] == []

    def test_malformed_edges_skipped(self):
        proc = _bare_processor()
        graph = {
            "edges": [
                None,  # not a dict
                {"relation_type": "prerequisite_of"},  # missing source/target
                {"source": "", "target": "concept:x", "relation_type": "prerequisite_of"},
                {
                    "source": "concept:a",
                    "target": "concept:b",
                    "relation_type": "prerequisite_of",
                    "confidence": 1,
                },
            ],
        }
        summary = proc._build_pedagogy_summary(
            chunks=_make_chunks(), pedagogy_graph=graph
        )
        # Only the well-formed edge produces an entry.
        assert len(summary["prerequisite_chain"]) == 1
        assert summary["prerequisite_chain"][0]["concept"] == "a"


# ---------------------------------------------------------------------------
# Legacy chunk-driven path (no pedagogy_graph supplied)
# ---------------------------------------------------------------------------


class TestLegacyChunkPath:
    def test_legacy_path_still_works_when_no_graph(self):
        # Chunk in week_01 defines `triple`; chunk in week_02 declares
        # `triple` as a prereq → valid chain entry.
        proc = _bare_processor()
        chunks = [
            {
                "id": "test_101_chunk_00001",
                "source": {"module_id": "week_01", "module_title": "Intro"},
                "concept_tags": ["triple"],
                "bloom_level": "understand",
            },
            {
                "id": "test_101_chunk_00002",
                "source": {"module_id": "week_02", "module_title": "Schema"},
                "concept_tags": ["graph"],
                "prereq_concepts": ["triple"],
                "bloom_level": "apply",
            },
        ]
        summary = proc._build_pedagogy_summary(chunks=chunks)
        # Legacy shape: defined_in + first_used_in keys.
        chain = summary["prerequisite_chain"]
        assert len(chain) == 1
        assert chain[0]["concept"] == "triple"
        assert "defined_in" in chain[0]
        assert "first_used_in" in chain[0]

    def test_legacy_path_detects_violations(self):
        # Use site (week_01) precedes definition (week_02) → violation.
        proc = _bare_processor()
        chunks = [
            {
                "id": "test_101_chunk_00001",
                "source": {"module_id": "week_01", "module_title": "Intro"},
                "concept_tags": ["graph"],
                "prereq_concepts": ["triple"],
                "bloom_level": "understand",
            },
            {
                "id": "test_101_chunk_00002",
                "source": {"module_id": "week_02", "module_title": "Triples"},
                "concept_tags": ["triple"],
                "bloom_level": "apply",
            },
        ]
        summary = proc._build_pedagogy_summary(chunks=chunks)
        assert len(summary["prerequisite_violations"]) == 1
        assert summary["prerequisite_violations"][0]["concept"] == "triple"

    def test_no_chunks_no_chain(self):
        proc = _bare_processor()
        summary = proc._build_pedagogy_summary(chunks=None)
        # No chunks → no chain key at all (legacy contract).
        assert "prerequisite_chain" not in summary
