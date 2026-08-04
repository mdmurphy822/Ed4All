"""Production concept classification and typed-edge propagation contracts.

The tests verify that concept-graph nodes receive deterministic classes without
changing graph topology and that semantic graph construction preserves or
supplies those classes. The processor is constructed without ingestion so the
tests exercise only graph-building behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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


def _build_concept_graph(chunks, course_id=""):
    from Trainforge.pipeline.process_course import CourseProcessor

    processor = CourseProcessor.__new__(CourseProcessor)
    processor.course_code = course_id
    return processor._build_tag_graph(chunks)


def _mk_chunk(chunk_id, tags):
    return {"id": chunk_id, "concept_tags": list(tags)}


def test_every_emitted_node_carries_class(monkeypatch):
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    # Exercise every classification family emitted by the graph builder.
    chunks = [
        _mk_chunk("c_001", ["photosynthesis", "key-takeaway", "answer-b", "to-01"]),
        _mk_chunk("c_002", ["photosynthesis", "key-takeaway", "answer-b", "submission-format"]),
        _mk_chunk("c_003", ["energy-transfer", "cellular-respiration", "not", "do-not"]),
        _mk_chunk("c_004", ["energy-transfer", "cellular-respiration", "not", "do-not", "to-01"]),
        _mk_chunk("c_005", ["ecosystem", "rubric"]),
        _mk_chunk("c_006", ["ecosystem", "rubric", "submission-format"]),
    ]
    graph = _build_concept_graph(chunks)

    nodes = graph["nodes"]
    assert nodes, "graph emitted no nodes"

    for node in nodes:
        assert "class" in node, f"node missing class: {node}"
        assert node["class"], f"node has empty class: {node}"

    by_id = {n["id"]: n for n in nodes}

    assert by_id["photosynthesis"]["class"] == DOMAIN_CONCEPT
    assert by_id["key-takeaway"]["class"] == PEDAGOGICAL_MARKER
    assert by_id["answer-b"]["class"] == ASSESSMENT_OPTION
    assert by_id["submission-format"]["class"] == INSTRUCTIONAL_ARTIFACT
    assert by_id["to-01"]["class"] == LEARNING_OBJECTIVE
    assert by_id["not"]["class"] == LOW_SIGNAL
    assert by_id["do-not"]["class"] == LOW_SIGNAL
    assert by_id["cellular-respiration"]["class"] == DOMAIN_CONCEPT
    assert by_id["rubric"]["class"] == PEDAGOGICAL_MARKER
    assert by_id["ecosystem"]["class"] == DOMAIN_CONCEPT


def test_classification_does_not_drop_or_merge_nodes(monkeypatch):
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    chunks = [
        _mk_chunk("c_001", ["photosynthesis", "key-takeaway"]),
        _mk_chunk("c_002", ["photosynthesis", "key-takeaway"]),
        _mk_chunk("c_003", ["answer-a", "answer-b"]),
        _mk_chunk("c_004", ["answer-a", "answer-b"]),
    ]
    graph = _build_concept_graph(chunks)

    ids = {n["id"] for n in graph["nodes"]}
    assert ids == {"photosynthesis", "key-takeaway", "answer-a", "answer-b"}

    assert graph["edges"], "edges should be preserved by classification"


def test_typed_edge_semantic_graph_carries_class(monkeypatch):
    """The semantic graph builder propagates classes from concept nodes."""
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    concept_graph = {
        "kind": "concept",
        "nodes": [
            {
                "id": "photosynthesis",
                "label": "Photosynthesis",
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
    assert nodes["photosynthesis"]["class"] == DOMAIN_CONCEPT
    assert nodes["key-takeaway"]["class"] == PEDAGOGICAL_MARKER


def test_typed_edge_semantic_graph_backfills_missing_class(monkeypatch):
    """The semantic graph builder supplies classes missing from concept nodes."""
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    concept_graph = {
        "kind": "concept",
        "nodes": [
            {"id": "photosynthesis", "label": "Photosynthesis", "frequency": 5},
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
    assert nodes["photosynthesis"]["class"] == DOMAIN_CONCEPT
    assert nodes["answer-c"]["class"] == ASSESSMENT_OPTION
    assert nodes["rubric"]["class"] == PEDAGOGICAL_MARKER
