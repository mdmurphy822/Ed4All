"""Assesses + misconception-of edges fire when the orchestrator threads
questions/misconceptions through ``build_semantic_graph``.

The rule modules (``infer_assesses``, ``infer_misconception_of``) were
already correct — they emit ``[]`` gracefully when their kwargs are None.
The bug was in ``process_course.CourseProcessor._generate_semantic_concept_graph``:
neither kwarg was ever populated, so neither rule fired in production.

These tests lock in:

1. The two helpers (``_build_misconceptions_for_graph`` +
   ``_build_questions_for_graph``) extract well-shaped entities from real
   chunks carrying ``misconceptions[]`` and ``learning_outcome_refs[]``.
2. Calling ``build_semantic_graph`` through the processor's wrapper
   (``_generate_semantic_concept_graph``) produces both ``assesses`` and
   ``misconception-of`` edges on a fixture chunk set that includes both
   signal types.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.process_course import CourseProcessor  # noqa: E402
from Trainforge.rag.typed_edge_inference import build_semantic_graph  # noqa: E402


FIXED_NOW = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _bare_processor() -> CourseProcessor:
    """Instantiate a CourseProcessor without running __init__.

    The helpers under test (``_build_misconceptions_for_graph`` +
    ``_build_questions_for_graph``) only read ``self.course_code``; we set
    that manually. This sidesteps the IMSCC-extraction path which requires
    a real zip file and output directory."""
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "TST_101"
    return proc


def _fixture_chunks() -> list:
    """Chunks with both misconception and assessment signals.

    One content chunk carries ``misconceptions[]`` + ``concept_tags[]`` so
    the misconception-of rule has a concept target.

    One assessment_item chunk carries ``learning_outcome_refs[]`` so the
    assesses rule has an LO target.
    """
    return [
        {
            "id": "chunk_content_01",
            "chunk_type": "explanation",
            "concept_tags": ["contrast-ratio", "accessibility"],
            "learning_outcome_refs": ["to-01"],
            "text": "Contrast ratio is the luminance difference between colours.",
            "misconceptions": [
                {
                    "misconception": "Any bold text meets contrast requirements.",
                    "correction": "Contrast is measured by luminance ratio, not weight.",
                }
            ],
        },
        {
            "id": "chunk_quiz_01",
            "chunk_type": "assessment_item",
            "concept_tags": ["contrast-ratio"],
            "learning_outcome_refs": ["to-01", "co-02"],
            "text": "Which pair meets the minimum contrast ratio?",
        },
    ]


def _fixture_concept_graph() -> dict:
    return {
        "kind": "concept",
        "nodes": [
            {"id": "contrast-ratio", "label": "contrast-ratio", "frequency": 2},
            {"id": "accessibility", "label": "accessibility", "frequency": 2},
        ],
        "edges": [],
    }


def test_build_misconceptions_extracts_entities_from_chunks():
    proc = _bare_processor()
    entities = proc._build_misconceptions_for_graph(_fixture_chunks())
    assert len(entities) == 1
    entity = entities[0]
    # Content-hash id shape.
    assert entity["id"].startswith("mc_")
    assert len(entity["id"]) == len("mc_") + 16
    # Concept target resolved from the chunk's first concept tag.
    assert "concept_id" in entity
    # Flat slug by default (SCOPE_CONCEPT_IDS flag off).
    assert entity["concept_id"] == "contrast-ratio"
    assert entity["misconception"].startswith("Any bold text")


def test_build_questions_extracts_one_per_objective_ref():
    proc = _bare_processor()
    questions = proc._build_questions_for_graph(_fixture_chunks())
    # One assessment_item chunk with 2 LO refs => 2 question entities.
    assert len(questions) == 2
    targets = {q["objective_id"] for q in questions}
    assert targets == {"to-01", "co-02"}
    for q in questions:
        assert q["id"].startswith("q_chunk_quiz_01_")
        assert q["source_chunk_id"] == "chunk_quiz_01"


def test_semantic_graph_emits_both_edge_types():
    """End-to-end through ``build_semantic_graph`` with the orchestrator's
    derived kwargs: both ``assesses`` and ``misconception-of`` edges must
    appear in the output artifact."""
    proc = _bare_processor()
    chunks = _fixture_chunks()
    graph = _fixture_concept_graph()

    misconceptions = proc._build_misconceptions_for_graph(chunks)
    questions = proc._build_questions_for_graph(chunks)
    assert misconceptions, "Precondition: misconception helper produced nothing."
    assert questions, "Precondition: question helper produced nothing."

    artifact = build_semantic_graph(
        chunks=chunks,
        course=None,
        concept_graph=graph,
        misconceptions=misconceptions,
        questions=questions,
        now=FIXED_NOW,
    )

    edge_types = {e["type"] for e in artifact["edges"]}
    assert "assesses" in edge_types, (
        f"Expected 'assesses' edges; got types={sorted(edge_types)!r}"
    )
    assert "misconception-of" in edge_types, (
        f"Expected 'misconception-of' edges; got types={sorted(edge_types)!r}"
    )

    # Spot-check each edge's shape.
    assesses = [e for e in artifact["edges"] if e["type"] == "assesses"]
    assert {e["target"] for e in assesses} == {"to-01", "co-02"}
    for e in assesses:
        assert e["source"].startswith("q_chunk_quiz_01_")
        assert e["provenance"]["rule"] == "assesses_from_question_lo"

    mis_edges = [e for e in artifact["edges"] if e["type"] == "misconception-of"]
    assert len(mis_edges) == 1
    assert mis_edges[0]["source"].startswith("mc_")
    assert mis_edges[0]["target"] == "contrast-ratio"
    assert mis_edges[0]["provenance"]["rule"] == "misconception_of_from_misconception_ref"


def test_chunks_without_signal_produce_no_new_edges():
    """Negative-control: a corpus lacking misconceptions / assessment items
    still produces no edges of these two types — the helpers must not
    synthesize signal."""
    proc = _bare_processor()
    chunks = [
        {
            "id": "plain_01",
            "chunk_type": "explanation",
            "concept_tags": ["alpha"],
            "learning_outcome_refs": ["to-01"],
            "text": "Some neutral prose.",
        }
    ]
    graph = {
        "kind": "concept",
        "nodes": [{"id": "alpha", "label": "alpha", "frequency": 2}],
        "edges": [],
    }
    misconceptions = proc._build_misconceptions_for_graph(chunks)
    questions = proc._build_questions_for_graph(chunks)
    assert misconceptions == []
    assert questions == []
    artifact = build_semantic_graph(
        chunks=chunks,
        course=None,
        concept_graph=graph,
        misconceptions=misconceptions or None,
        questions=questions or None,
        now=FIXED_NOW,
    )
    edge_types = {e["type"] for e in artifact["edges"]}
    assert "assesses" not in edge_types
    assert "misconception-of" not in edge_types
