"""Fix-2 — instance-free co-occurrence concept-graph builder.

Locks the contract of ``lib.ontology.cooccurrence_graph.build_cooccurrence_graph``
— the helper extracted from ``CourseProcessor._build_tag_graph`` so the
``textbook_to_course::concept_extraction`` workflow phase can build the
``kind: "concept"`` co-occurrence graph that ``build_semantic_graph``
consumes, WITHOUT a ``CourseProcessor`` instance.

Contract assertions:

1. **Callable with no instance.** The helper builds a graph from a bare
   chunk list + course_id string — no ``CourseProcessor`` needed.
2. **Tagged chunks → ``kind: "concept"`` nodes with ``class``.** Each
   node carries the canonical ``{id, label, frequency, class}`` shape;
   tags appearing in ≥2 chunks become nodes (default ``min_freq``).
3. **Empty-tag chunks → empty node set.** Regression-pins the Risk-2
   failure mode: chunks with ``concept_tags: []`` contribute zero nodes.
4. **Delegation parity.** ``CourseProcessor._build_tag_graph`` delegates
   to the helper — the two produce identical graphs from the same input.
5. **Co-occurrence edges.** Tags that share a chunk emit a
   ``relation_type: "co-occurs"`` edge.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.cooccurrence_graph import build_cooccurrence_graph  # noqa: E402


def _tagged_chunks() -> List[Dict[str, Any]]:
    """Chunks with hand-set ``concept_tags`` so two tags appear in 2+
    chunks (clearing the default ``min_freq=2`` floor).
    """
    return [
        {
            "id": "chunk_001",
            "text": "alpha and beta intro",
            "chunk_type": "content",
            "concept_tags": ["alpha-concept", "beta-concept"],
        },
        {
            "id": "chunk_002",
            "text": "more alpha and beta",
            "chunk_type": "content",
            "concept_tags": ["alpha-concept", "beta-concept"],
        },
        {
            "id": "chunk_003",
            "text": "alpha and gamma",
            "chunk_type": "content",
            "concept_tags": ["alpha-concept", "gamma-concept"],
        },
    ]


def test_callable_without_processor_instance() -> None:
    """The helper builds a graph from a bare chunk list — no
    ``CourseProcessor`` instance, no instance state.
    """
    graph = build_cooccurrence_graph(_tagged_chunks(), "FIX2_TEST")
    assert isinstance(graph, dict)
    assert graph.get("kind") == "concept"
    assert "nodes" in graph and "edges" in graph


def test_tagged_chunks_yield_concept_nodes_with_class() -> None:
    """Tags appearing in ≥2 chunks become nodes carrying the canonical
    ``{id, label, frequency, class}`` shape.
    """
    graph = build_cooccurrence_graph(_tagged_chunks(), "FIX2_TEST")
    nodes = graph["nodes"]
    # alpha-concept (3), beta-concept (2) clear min_freq=2;
    # gamma-concept (1) is filtered out.
    labels = {n["id"] for n in nodes}
    assert "alpha-concept" in labels
    assert "beta-concept" in labels
    assert "gamma-concept" not in labels, (
        "Single-occurrence tag should be filtered by default min_freq=2."
    )
    for node in nodes:
        assert isinstance(node.get("id"), str) and node["id"]
        assert isinstance(node.get("label"), str)
        assert isinstance(node.get("frequency"), int)
        assert isinstance(node.get("class"), str) and node["class"], (
            "Every node must be stamped with a non-empty class."
        )


def test_empty_tag_chunks_yield_empty_node_set() -> None:
    """Regression-pin the Risk-2 failure mode: chunks with empty
    ``concept_tags`` contribute zero nodes (a naive build_pedagogy →
    build_semantic swap would have produced a zero-node graph).
    """
    empty_chunks = [
        {"id": "c1", "concept_tags": [], "chunk_type": "content"},
        {"id": "c2", "concept_tags": [], "chunk_type": "content"},
    ]
    graph = build_cooccurrence_graph(empty_chunks, "FIX2_TEST")
    assert graph["nodes"] == []
    assert graph["edges"] == []


def test_cooccurrence_edges_emitted() -> None:
    """Tags sharing a chunk emit a ``co-occurs`` edge between the two
    surviving nodes.
    """
    graph = build_cooccurrence_graph(_tagged_chunks(), "FIX2_TEST")
    edges = graph["edges"]
    assert edges, "Expected at least one co-occurrence edge."
    for edge in edges:
        assert edge.get("relation_type") == "co-occurs"
        assert "source" in edge and "target" in edge
        assert isinstance(edge.get("weight"), int)
    # alpha-concept <-> beta-concept co-occur in chunk_001 + chunk_002.
    pairs = {frozenset((e["source"], e["target"])) for e in edges}
    assert frozenset(("alpha-concept", "beta-concept")) in pairs


def test_delegation_parity_with_course_processor() -> None:
    """``CourseProcessor._build_tag_graph`` delegates to the helper —
    the two produce identical graphs (modulo the ``generated_at``
    timestamp) from the same chunk input.
    """
    from Trainforge.process_course import CourseProcessor

    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "FIX2_TEST"

    chunks = _tagged_chunks()
    via_method = proc._build_tag_graph(chunks)
    via_helper = build_cooccurrence_graph(chunks, "FIX2_TEST")

    # generated_at is wall-clock; compare everything else.
    assert via_method["kind"] == via_helper["kind"]
    assert via_method["nodes"] == via_helper["nodes"]
    assert via_method["edges"] == via_helper["edges"]


def test_graph_kind_override() -> None:
    """The ``graph_kind`` kwarg controls the emitted ``kind`` value."""
    graph = build_cooccurrence_graph(
        _tagged_chunks(), "FIX2_TEST", graph_kind="pedagogy"
    )
    assert graph["kind"] == "pedagogy"
