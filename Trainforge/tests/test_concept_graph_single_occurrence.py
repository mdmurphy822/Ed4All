"""Wave 82 tests for the single-occurrence backfill flag.

The rdf-shacl-551 audit found that ``concept_graph.json`` carried 424
nodes while the pedagogy graph named 660 Concept-class nodes — 236
concepts (``alldisjointclasses``, ``aggregate-projection``,
``annotation-properties``, etc.) appeared in exactly 1 chunk each and
were filtered out by the legacy "2+ chunks to admit" gate.

With ``TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE=true`` the
threshold drops to 1, so single-occurrence legitimate domain terms
survive into the concept graph. Wave 76's classifier filter still gates
upstream — pedagogical/assessment scaffolding never enters
``concept_tags`` — so flipping this on doesn't reintroduce the original
noise that motivated the 2+ rule.

Default remains off for backward compatibility; flipping the default
is a separate decision after retrieval-quality evaluation on real
corpora.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from Trainforge.process_course import CourseProcessor


def _bare_processor() -> CourseProcessor:
    """Construct a CourseProcessor that bypasses __init__ (test helper)."""
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "TEST_101"
    proc.domain_concept_seeds = []
    return proc


def _make_chunks(tag_distribution: Dict[str, int]) -> List[Dict[str, Any]]:
    """Build chunks where each tag appears in N chunks per the distribution.

    Each chunk gets a unique ID. Tags are spread across chunks so that
    ``tag: 1`` produces a single-chunk occurrence.
    """
    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0
    for tag, n in tag_distribution.items():
        for _ in range(n):
            chunks.append({
                "id": f"test_101_chunk_{chunk_idx:05d}",
                "concept_tags": [tag],
            })
            chunk_idx += 1
    return chunks


class TestSingleOccurrenceFlag:
    def test_default_excludes_single_occurrence(self, monkeypatch):
        # Audit reproducer: legitimate W3C term appears in only 1 chunk.
        # Default behavior drops it.
        monkeypatch.delenv(
            "TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE", raising=False
        )
        proc = _bare_processor()
        chunks = _make_chunks({
            "alldisjointclasses": 1,   # single occurrence — dropped
            "rdf-graph": 5,            # multi-occurrence — admitted
        })
        graph = proc._build_tag_graph(chunks, graph_kind="concept")
        node_ids = {n["id"] for n in graph["nodes"]}
        assert "alldisjointclasses" not in node_ids
        assert "rdf-graph" in node_ids

    def test_flag_on_includes_single_occurrence(self, monkeypatch):
        monkeypatch.setenv(
            "TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE", "true"
        )
        proc = _bare_processor()
        chunks = _make_chunks({
            "alldisjointclasses": 1,
            "haskey": 1,
            "rdf-graph": 5,
        })
        graph = proc._build_tag_graph(chunks, graph_kind="concept")
        node_ids = {n["id"] for n in graph["nodes"]}
        assert "alldisjointclasses" in node_ids
        assert "haskey" in node_ids
        assert "rdf-graph" in node_ids

    def test_flag_false_value_keeps_default(self, monkeypatch):
        # Only "true" enables — "false", "0", "no" stay default.
        for value in ["false", "0", "no", ""]:
            monkeypatch.setenv(
                "TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE", value
            )
            proc = _bare_processor()
            chunks = _make_chunks({"single": 1, "multi": 3})
            graph = proc._build_tag_graph(chunks, graph_kind="concept")
            node_ids = {n["id"] for n in graph["nodes"]}
            assert "single" not in node_ids, f"value={value!r} should not enable"
            assert "multi" in node_ids

    def test_flag_on_preserves_classifier_filter(self, monkeypatch):
        # Single-occurrence pedagogical-marker tags are STILL filtered
        # at extraction by Wave 76's classifier — _build_tag_graph operates
        # on already-filtered concept_tags, so noise doesn't sneak back in
        # via this flag. This test asserts that the gate is purely
        # frequency-based — the upstream filter is the noise gate.
        monkeypatch.setenv(
            "TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE", "true"
        )
        proc = _bare_processor()
        # Single-occurrence DomainConcept slug — admitted.
        chunks = _make_chunks({"shacl-shape": 1})
        graph = proc._build_tag_graph(chunks, graph_kind="concept")
        node_ids = {n["id"] for n in graph["nodes"]}
        assert "shacl-shape" in node_ids
        # Pedagogical-marker tags should never be in concept_tags in
        # the first place (Wave 76 filter), so their absence here is
        # the upstream contract — not this flag's responsibility.

    def test_flag_off_audit_state_reproduces(self, monkeypatch):
        # Off → 236-missing-concepts state matches the audit's finding.
        monkeypatch.delenv(
            "TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE", raising=False
        )
        proc = _bare_processor()
        # Mix of singles and multis. Default drops 5 single-occurrence tags.
        single_tags = {f"single-{i}": 1 for i in range(5)}
        multi_tags = {f"multi-{i}": 3 for i in range(2)}
        chunks = _make_chunks({**single_tags, **multi_tags})
        graph = proc._build_tag_graph(chunks, graph_kind="concept")
        node_ids = {n["id"] for n in graph["nodes"]}
        # Only the 2 multi-occurrence tags survive.
        assert len(node_ids) == 2
        assert all(nid.startswith("multi-") for nid in node_ids)

    def test_flag_on_admits_all_classifier_passing_tags(self, monkeypatch):
        # On → all 7 tags survive (5 singles + 2 multis).
        monkeypatch.setenv(
            "TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE", "true"
        )
        proc = _bare_processor()
        single_tags = {f"single-{i}": 1 for i in range(5)}
        multi_tags = {f"multi-{i}": 3 for i in range(2)}
        chunks = _make_chunks({**single_tags, **multi_tags})
        graph = proc._build_tag_graph(chunks, graph_kind="concept")
        node_ids = {n["id"] for n in graph["nodes"]}
        assert len(node_ids) == 7
