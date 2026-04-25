"""Wave 75: classifier integration tests for concept-graph builders.

Two contracts:

1. The production ``CourseProcessor._build_tag_graph`` stamps a
   ``class`` field on every emitted concept node, drawn from
   :func:`lib.ontology.concept_classifier.classify_concept`.
2. The Wave 75 retroactive script
   (``scripts/wave75_classify_concept_graph.py``) regenerates ``class``
   on a stub concept graph that lacks it, without dropping or merging
   nodes and without disturbing the edges.

The tests follow the lightweight helper pattern used by
``test_concept_occurrences.py``: construct the processor via ``__new__``
to skip IMSCC ingestion. We do not exercise the full pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.concept_classifier import (  # noqa: E402
    ASSESSMENT_OPTION,
    DOMAIN_CONCEPT,
    INSTRUCTIONAL_ARTIFACT,
    LEARNING_OBJECTIVE,
    LOW_SIGNAL,
    PEDAGOGICAL_MARKER,
)
from Trainforge.rag import typed_edge_inference  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (mirrors test_concept_occurrences.py)
# ---------------------------------------------------------------------------

def _build_concept_graph(chunks, course_id=""):
    from Trainforge.process_course import CourseProcessor

    processor = CourseProcessor.__new__(CourseProcessor)
    processor.course_code = course_id
    return processor._build_tag_graph(chunks)


def _mk_chunk(chunk_id, tags):
    return {"id": chunk_id, "concept_tags": list(tags)}


# ---------------------------------------------------------------------------
# Contract 1 — the builder stamps `class` on every node.
# ---------------------------------------------------------------------------

def test_every_emitted_node_carries_class(monkeypatch):
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    # Mix domain concepts + the contamination flagged by ChatGPT's
    # review so we verify the classifier is actually consulted.
    chunks = [
        _mk_chunk("c_001", ["rdf-graph", "key-takeaway", "answer-b", "to-04"]),
        _mk_chunk("c_002", ["rdf-graph", "key-takeaway", "answer-b", "submission-format"]),
        _mk_chunk("c_003", ["sh-path", "owl-2-rl", "not", "do-not"]),
        _mk_chunk("c_004", ["sh-path", "owl-2-rl", "not", "do-not", "to-04"]),
        _mk_chunk("c_005", ["sparql-select", "rubric"]),
        _mk_chunk("c_006", ["sparql-select", "rubric", "submission-format"]),
    ]
    graph = _build_concept_graph(chunks)

    nodes = graph["nodes"]
    assert nodes, "graph emitted no nodes"

    # Every node must carry a class.
    for node in nodes:
        assert "class" in node, f"node missing class: {node}"
        assert node["class"], f"node has empty class: {node}"

    by_id = {n["id"]: n for n in nodes}

    # Spot-check the high-confidence cases.
    assert by_id["rdf-graph"]["class"] == DOMAIN_CONCEPT
    assert by_id["key-takeaway"]["class"] == PEDAGOGICAL_MARKER
    assert by_id["answer-b"]["class"] == ASSESSMENT_OPTION
    assert by_id["submission-format"]["class"] == INSTRUCTIONAL_ARTIFACT
    assert by_id["to-04"]["class"] == LEARNING_OBJECTIVE
    assert by_id["not"]["class"] == LOW_SIGNAL
    assert by_id["do-not"]["class"] == LOW_SIGNAL
    assert by_id["owl-2-rl"]["class"] == DOMAIN_CONCEPT
    assert by_id["rubric"]["class"] == PEDAGOGICAL_MARKER
    assert by_id["sparql-select"]["class"] == DOMAIN_CONCEPT


def test_classification_does_not_drop_or_merge_nodes(monkeypatch):
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    chunks = [
        _mk_chunk("c_001", ["rdf-graph", "key-takeaway"]),
        _mk_chunk("c_002", ["rdf-graph", "key-takeaway"]),
        _mk_chunk("c_003", ["answer-a", "answer-b"]),
        _mk_chunk("c_004", ["answer-a", "answer-b"]),
    ]
    graph = _build_concept_graph(chunks)

    # All four tags appear in 2+ chunks → all four must be present as
    # nodes regardless of how they classify.
    ids = {n["id"] for n in graph["nodes"]}
    assert ids == {"rdf-graph", "key-takeaway", "answer-a", "answer-b"}

    # Edges are still present (classification is metadata-only).
    assert graph["edges"], "edges should be preserved by classification"


# ---------------------------------------------------------------------------
# Contract 2 — retroactive regen on a stub graph.
# ---------------------------------------------------------------------------

def _stub_graph_payload():
    """A concept-graph payload that pre-dates the classifier wiring."""
    return {
        "kind": "concept",
        "nodes": [
            {"id": "rdf-graph", "label": "Rdf Graph", "frequency": 12},
            {"id": "key-takeaway", "label": "Key Takeaway", "frequency": 32},
            {"id": "answer-b", "label": "Answer B", "frequency": 9},
            {"id": "to-04", "label": "To 04", "frequency": 4},
            {"id": "not", "label": "Not", "frequency": 5},
            {"id": "submission-format", "label": "Submission Format", "frequency": 6},
            {"id": "owl-2-rl", "label": "Owl 2 Rl", "frequency": 7},
        ],
        "edges": [
            {"source": "rdf-graph", "target": "owl-2-rl", "weight": 4, "relation_type": "co-occurs"},
            {"source": "key-takeaway", "target": "rdf-graph", "weight": 2, "relation_type": "co-occurs"},
        ],
        "generated_at": "2026-01-01T00:00:00",
    }


def test_retroactive_regen_adds_class_to_every_node(tmp_path):
    # Stage a course directory under tmp_path with both graph files.
    course_dir = tmp_path / "course"
    graph_dir = course_dir / "graph"
    graph_dir.mkdir(parents=True)

    primary = graph_dir / "concept_graph.json"
    semantic = graph_dir / "concept_graph_semantic.json"
    payload = _stub_graph_payload()
    primary.write_text(json.dumps(payload), encoding="utf-8")
    semantic.write_text(json.dumps(payload), encoding="utf-8")

    # Import lazily so the script's REPO_ROOT side-effect doesn't fire
    # at module-collection time.
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        import wave75_classify_concept_graph as regen  # noqa: WPS433
    finally:
        sys.path.pop(0)

    report = regen.regen_course(course_dir)

    # Both files should have been processed.
    assert len(report["graphs"]) == 2

    # Every node in both regenerated files must now carry a class field.
    for graph_path in (primary, semantic):
        with graph_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for node in data["nodes"]:
            assert "class" in node, f"missing class after regen: {node}"
            assert node["class"], f"empty class after regen: {node}"

        # Backups must exist so the operation is reversible.
        bak = graph_path.with_suffix(graph_path.suffix + ".bak")
        assert bak.exists(), f"missing .bak for {graph_path}"

        # Class assignment uses the documented rule precedence.
        by_id = {n["id"]: n for n in data["nodes"]}
        assert by_id["rdf-graph"]["class"] == DOMAIN_CONCEPT
        assert by_id["key-takeaway"]["class"] == PEDAGOGICAL_MARKER
        assert by_id["answer-b"]["class"] == ASSESSMENT_OPTION
        assert by_id["to-04"]["class"] == LEARNING_OBJECTIVE
        assert by_id["not"]["class"] == LOW_SIGNAL
        assert by_id["submission-format"]["class"] == INSTRUCTIONAL_ARTIFACT
        assert by_id["owl-2-rl"]["class"] == DOMAIN_CONCEPT

        # Edges and node count survive.
        assert len(data["nodes"]) == len(payload["nodes"])
        assert len(data["edges"]) == len(payload["edges"])


def test_retroactive_regen_dry_run_does_not_write(tmp_path):
    course_dir = tmp_path / "course"
    graph_dir = course_dir / "graph"
    graph_dir.mkdir(parents=True)

    primary = graph_dir / "concept_graph.json"
    payload = _stub_graph_payload()
    primary.write_text(json.dumps(payload), encoding="utf-8")
    pre_mtime = primary.stat().st_mtime

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        import wave75_classify_concept_graph as regen  # noqa: WPS433
    finally:
        sys.path.pop(0)

    regen.regen_course(course_dir, dry_run=True)

    # File untouched + no .bak created.
    assert primary.stat().st_mtime == pre_mtime
    assert not primary.with_suffix(primary.suffix + ".bak").exists()


def test_typed_edge_semantic_graph_carries_class(monkeypatch):
    """The semantic graph builder must propagate ``class`` from concept_graph nodes."""
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    concept_graph = {
        "kind": "concept",
        "nodes": [
            {
                "id": "rdf-graph",
                "label": "Rdf Graph",
                "frequency": 5,
                "class": DOMAIN_CONCEPT,
            },
            {
                "id": "key-takeaway",
                "label": "Key Takeaway",
                "frequency": 3,
                "class": PEDAGOGICAL_MARKER,
            },
        ],
        "edges": [],
    }
    semantic = typed_edge_inference.build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph=concept_graph,
    )
    nodes = {n["id"]: n for n in semantic["nodes"]}
    assert nodes["rdf-graph"]["class"] == DOMAIN_CONCEPT
    assert nodes["key-takeaway"]["class"] == PEDAGOGICAL_MARKER


def test_typed_edge_semantic_graph_backfills_missing_class(monkeypatch):
    """When a legacy concept_graph lacks ``class``, the semantic builder backfills."""
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    concept_graph = {
        "kind": "concept",
        "nodes": [
            {"id": "rdf-graph", "label": "Rdf Graph", "frequency": 5},
            {"id": "answer-c", "label": "Answer C", "frequency": 4},
            {"id": "rubric", "label": "Rubric", "frequency": 3},
        ],
        "edges": [],
    }
    semantic = typed_edge_inference.build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph=concept_graph,
    )
    nodes = {n["id"]: n for n in semantic["nodes"]}
    assert nodes["rdf-graph"]["class"] == DOMAIN_CONCEPT
    assert nodes["answer-c"]["class"] == ASSESSMENT_OPTION
    assert nodes["rubric"]["class"] == PEDAGOGICAL_MARKER
