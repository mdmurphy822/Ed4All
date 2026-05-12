"""Tests for the three misconception-anchored typed-edge rules.

GPT Feedback (12 May 2026) item 4 — making Misconception a first-class
learning target via three new typed edges that materialize implicit
pointers already in the corpus:

- ``corrected-by-chunk`` from chunk-level ``misconceptions[]`` lists
  (the chunk that emits the misconception IS the corrector-of-record).
- ``detected-by-question`` from questions carrying a
  ``misconception_id`` pointer (the rejected branch of a preference
  pair, or a distractor encoding a known misconception).
- ``interferes-with-outcome`` from misconception entities carrying an
  ``lo_id`` pointer.

Test surface mirrors ``test_pedagogical_edges.py`` exactly: per-rule
happy-path + empty-signal + dedupe + integration + strict-evidence
schema round-trip.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from lib.ontology.misconception_id import canonical_mc_id
from Trainforge.rag.inference_rules import (
    infer_corrected_by_chunk,
    infer_detected_by_question,
    infer_interferes_with_outcome,
)
from Trainforge.rag.typed_edge_inference import build_semantic_graph

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "knowledge"
    / "concept_graph_semantic.schema.json"
)

FIXED_NOW = datetime(2026, 5, 12, 0, 0, 0, tzinfo=timezone.utc)


def _empty_graph():
    return {"kind": "concept", "nodes": [], "edges": []}


# ---------------------------------------------------------------------------
# corrected-by-chunk
# ---------------------------------------------------------------------------


def test_corrected_by_chunk_happy_path():
    """A single chunk with one misconception emits one typed edge whose
    source is the canonical content-hash misconception ID and whose
    target is the chunk ID."""
    statement = "All birds can fly."
    correction = "Some birds, like penguins and ostriches, cannot fly."
    chunks = [
        {
            "id": "chunk_001",
            "misconceptions": [
                {"misconception": statement, "correction": correction}
            ],
        }
    ]
    edges = infer_corrected_by_chunk(chunks, None, _empty_graph())
    assert len(edges) == 1
    e = edges[0]
    expected_id = canonical_mc_id(statement, correction)
    assert e["source"] == expected_id
    assert e["target"] == "chunk_001"
    assert e["type"] == "corrected-by-chunk"
    assert e["confidence"] == 1.0
    assert e["provenance"]["rule"] == "corrected_by_from_chunk_misconception"
    assert e["provenance"]["rule_version"] == 1
    assert e["provenance"]["evidence"] == {
        "misconception_id": expected_id,
        "chunk_id": "chunk_001",
    }


def test_corrected_by_chunk_empty_when_no_misconceptions():
    chunks = [{"id": "chunk_001"}, {"id": "chunk_002", "misconceptions": []}]
    assert infer_corrected_by_chunk(chunks, None, _empty_graph()) == []


def test_corrected_by_chunk_skips_string_entries():
    """Plain-string misconception entries from non-Courseforge IMSCC
    passthrough lack a correction text — skip rather than fabricate."""
    chunks = [
        {"id": "chunk_001", "misconceptions": ["a bare string with no correction"]}
    ]
    assert infer_corrected_by_chunk(chunks, None, _empty_graph()) == []


def test_corrected_by_chunk_supports_wave74_statement_alias():
    """Chunk schema allows ``statement`` as an alias for ``misconception``
    (Wave 74+ subagents). The rule honours the alias."""
    chunks = [
        {
            "id": "chunk_001",
            "misconceptions": [
                {"statement": "All birds fly.", "correction": "Some cannot."}
            ],
        }
    ]
    edges = infer_corrected_by_chunk(chunks, None, _empty_graph())
    assert len(edges) == 1
    assert edges[0]["target"] == "chunk_001"
    assert edges[0]["source"] == canonical_mc_id(
        "All birds fly.", "Some cannot."
    )


def test_corrected_by_chunk_bloom_level_participates_in_id():
    """Bloom level participates in the misconception ID seed (Wave 69),
    so the same statement+correction at different Bloom levels emits
    distinct edges sharing the same chunk target."""
    chunks = [
        {
            "id": "chunk_001",
            "misconceptions": [
                {
                    "misconception": "x",
                    "correction": "y",
                    "bloom_level": "apply",
                },
                {
                    "misconception": "x",
                    "correction": "y",
                    "bloom_level": "analyze",
                },
            ],
        }
    ]
    edges = infer_corrected_by_chunk(chunks, None, _empty_graph())
    assert len(edges) == 2
    sources = {e["source"] for e in edges}
    assert canonical_mc_id("x", "y", "apply") in sources
    assert canonical_mc_id("x", "y", "analyze") in sources


def test_corrected_by_chunk_dedups_same_mc_chunk_pair():
    """Two identical dicts on the same chunk collapse to one edge."""
    chunks = [
        {
            "id": "chunk_001",
            "misconceptions": [
                {"misconception": "x", "correction": "y"},
                {"misconception": "x", "correction": "y"},
            ],
        }
    ]
    edges = infer_corrected_by_chunk(chunks, None, _empty_graph())
    assert len(edges) == 1


def test_corrected_by_chunk_fans_out_across_chunks():
    """The same misconception emitted in two chunks produces two edges
    sharing source (the content-hash ID) but with distinct chunk targets."""
    chunks = [
        {
            "id": "chunk_001",
            "misconceptions": [{"misconception": "x", "correction": "y"}],
        },
        {
            "id": "chunk_002",
            "misconceptions": [{"misconception": "x", "correction": "y"}],
        },
    ]
    edges = infer_corrected_by_chunk(chunks, None, _empty_graph())
    assert len(edges) == 2
    targets = sorted(e["target"] for e in edges)
    assert targets == ["chunk_001", "chunk_002"]
    expected_id = canonical_mc_id("x", "y")
    assert all(e["source"] == expected_id for e in edges)


# ---------------------------------------------------------------------------
# detected-by-question
# ---------------------------------------------------------------------------


def test_detected_by_question_empty_when_no_questions_kwarg():
    assert infer_detected_by_question([], None, _empty_graph()) == []


def test_detected_by_question_empty_when_no_misconception_id():
    qs = [{"id": "q1", "objective_id": "to-01"}]
    assert infer_detected_by_question([], None, _empty_graph(), questions=qs) == []


def test_detected_by_question_happy_path():
    mc_id = "mc_" + "a" * 16
    qs = [
        {
            "id": "MCQ-a1b2c3d4",
            "misconception_id": mc_id,
            "objective_id": "to-01",
        }
    ]
    edges = infer_detected_by_question([], None, _empty_graph(), questions=qs)
    assert len(edges) == 1
    e = edges[0]
    assert e["source"] == mc_id
    assert e["target"] == "MCQ-a1b2c3d4"
    assert e["type"] == "detected-by-question"
    assert e["confidence"] == 1.0
    assert e["provenance"]["rule"] == "detected_by_from_distractor_misconception_id"
    assert e["provenance"]["evidence"] == {
        "misconception_id": mc_id,
        "question_id": "MCQ-a1b2c3d4",
    }


def test_detected_by_question_dedups_same_mc_question_pair():
    mc_id = "mc_" + "a" * 16
    qs = [
        {"id": "q1", "misconception_id": mc_id},
        {"id": "q1", "misconception_id": mc_id},
    ]
    edges = infer_detected_by_question([], None, _empty_graph(), questions=qs)
    assert len(edges) == 1


def test_detected_by_question_distinct_questions_distinct_edges():
    mc_id = "mc_" + "a" * 16
    qs = [
        {"id": "MCQ-1", "misconception_id": mc_id},
        {"id": "MRQ-2", "misconception_id": mc_id},
    ]
    edges = infer_detected_by_question([], None, _empty_graph(), questions=qs)
    assert len(edges) == 2
    targets = sorted(e["target"] for e in edges)
    assert targets == ["MCQ-1", "MRQ-2"]


# ---------------------------------------------------------------------------
# interferes-with-outcome
# ---------------------------------------------------------------------------


def test_interferes_with_outcome_empty_when_no_misconceptions_kwarg():
    assert infer_interferes_with_outcome([], None, _empty_graph()) == []


def test_interferes_with_outcome_empty_when_no_lo_id():
    mcs = [{"id": "mc_" + "a" * 16, "misconception": "x", "correction": "y"}]
    assert (
        infer_interferes_with_outcome([], None, _empty_graph(), misconceptions=mcs)
        == []
    )


def test_interferes_with_outcome_happy_path():
    mc_id = "mc_" + "a" * 16
    mcs = [
        {
            "id": mc_id,
            "misconception": "x",
            "correction": "y",
            "lo_id": "to-01",
        }
    ]
    edges = infer_interferes_with_outcome(
        [], None, _empty_graph(), misconceptions=mcs
    )
    assert len(edges) == 1
    e = edges[0]
    assert e["source"] == mc_id
    assert e["target"] == "to-01"
    assert e["type"] == "interferes-with-outcome"
    assert e["confidence"] == 1.0
    assert e["provenance"]["rule"] == "interferes_with_outcome_from_misconception_lo"
    assert e["provenance"]["evidence"] == {
        "misconception_id": mc_id,
        "lo_id": "to-01",
    }


def test_interferes_with_outcome_dedups_same_mc_lo_pair():
    mc_id = "mc_" + "a" * 16
    mcs = [
        {"id": mc_id, "lo_id": "to-01"},
        {"id": mc_id, "lo_id": "to-01"},
    ]
    edges = infer_interferes_with_outcome(
        [], None, _empty_graph(), misconceptions=mcs
    )
    assert len(edges) == 1


# ---------------------------------------------------------------------------
# Integration: all three new edges emit together via build_semantic_graph
# ---------------------------------------------------------------------------


def test_integration_all_three_misconception_edges_via_build_semantic_graph():
    """Wire all three signals into ``build_semantic_graph`` end-to-end."""
    statement = "Wrong belief"
    correction = "Right answer"
    mc_id = canonical_mc_id(statement, correction)
    chunks = [
        {
            "id": "chunk_001",
            "concept_tags": [],
            "learning_outcome_refs": [],
            "misconceptions": [
                {"misconception": statement, "correction": correction}
            ],
        }
    ]
    misconceptions = [
        {
            "id": mc_id,
            "misconception": statement,
            "correction": correction,
            "lo_id": "to-01",
        }
    ]
    questions = [
        {
            "id": "MCQ-deadbeef",
            "misconception_id": mc_id,
        }
    ]
    artifact = build_semantic_graph(
        chunks,
        None,
        _empty_graph(),
        now=FIXED_NOW,
        misconceptions=misconceptions,
        questions=questions,
    )
    types_emitted = {e["type"] for e in artifact["edges"]}
    assert "corrected-by-chunk" in types_emitted
    assert "detected-by-question" in types_emitted
    assert "interferes-with-outcome" in types_emitted

    # Per-edge structural check.
    corrected = next(e for e in artifact["edges"] if e["type"] == "corrected-by-chunk")
    assert corrected["source"] == mc_id and corrected["target"] == "chunk_001"
    detected = next(
        e for e in artifact["edges"] if e["type"] == "detected-by-question"
    )
    assert detected["source"] == mc_id and detected["target"] == "MCQ-deadbeef"
    interferes = next(
        e for e in artifact["edges"] if e["type"] == "interferes-with-outcome"
    )
    assert interferes["source"] == mc_id and interferes["target"] == "to-01"


def test_all_three_edges_empty_when_no_signal():
    """No chunks, no questions, no misconceptions → zero edges of all three
    types (the rules emit nothing rather than erroring)."""
    artifact = build_semantic_graph(
        [],
        None,
        _empty_graph(),
        now=FIXED_NOW,
        misconceptions=None,
        questions=None,
    )
    types_emitted = {e["type"] for e in artifact["edges"]}
    for t in ("corrected-by-chunk", "detected-by-question", "interferes-with-outcome"):
        assert t not in types_emitted


# ---------------------------------------------------------------------------
# Schema validation — strict-evidence round trip on all three edge types
# ---------------------------------------------------------------------------


def _new_edge_artifact():
    """Build a synthetic artifact carrying one edge per new type plus
    valid evidence shapes. Used by the lenient + strict round-trip tests."""
    mc_id = "mc_" + "f" * 16
    return {
        "kind": "concept_semantic",
        "generated_at": FIXED_NOW.isoformat(),
        "nodes": [{"id": "chunk_001"}],
        "edges": [
            {
                "source": mc_id,
                "target": "chunk_001",
                "type": "corrected-by-chunk",
                "confidence": 1.0,
                "provenance": {
                    "rule": "corrected_by_from_chunk_misconception",
                    "rule_version": 1,
                    "evidence": {
                        "misconception_id": mc_id,
                        "chunk_id": "chunk_001",
                    },
                },
            },
            {
                "source": mc_id,
                "target": "MCQ-feedface",
                "type": "detected-by-question",
                "confidence": 1.0,
                "provenance": {
                    "rule": "detected_by_from_distractor_misconception_id",
                    "rule_version": 1,
                    "evidence": {
                        "misconception_id": mc_id,
                        "question_id": "MCQ-feedface",
                    },
                },
            },
            {
                "source": mc_id,
                "target": "to-01",
                "type": "interferes-with-outcome",
                "confidence": 1.0,
                "provenance": {
                    "rule": "interferes_with_outcome_from_misconception_lo",
                    "rule_version": 1,
                    "evidence": {
                        "misconception_id": mc_id,
                        "lo_id": "to-01",
                    },
                },
            },
        ],
    }


def test_new_edge_types_validate_against_lenient_schema():
    jsonschema = pytest.importorskip("jsonschema")
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.validate(instance=_new_edge_artifact(), schema=schema)


def test_new_edge_types_validate_under_strict_evidence_mode():
    """All three new edges round-trip through strict-evidence validation
    (TRAINFORGE_STRICT_EVIDENCE=true strips the FallbackProvenance arm
    so the per-rule arms must match)."""
    jsonschema = pytest.importorskip("jsonschema")
    from lib.validators.evidence import get_schema

    schema = get_schema(strict=True)
    jsonschema.validate(instance=_new_edge_artifact(), schema=schema)


def test_strict_evidence_rejects_malformed_corrected_by_evidence():
    """Strict mode rejects an edge with the new rule name but a
    malformed evidence shape (no fallback arm to fall through to)."""
    jsonschema = pytest.importorskip("jsonschema")
    from lib.validators.evidence import get_schema

    mc_id = "mc_" + "f" * 16
    bad_artifact = {
        "kind": "concept_semantic",
        "generated_at": FIXED_NOW.isoformat(),
        "nodes": [{"id": "chunk_001"}],
        "edges": [
            {
                "source": mc_id,
                "target": "chunk_001",
                "type": "corrected-by-chunk",
                "confidence": 1.0,
                "provenance": {
                    "rule": "corrected_by_from_chunk_misconception",
                    "rule_version": 1,
                    # MALFORMED: missing required chunk_id.
                    "evidence": {"misconception_id": mc_id},
                },
            }
        ],
    }
    schema = get_schema(strict=True)
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=bad_artifact, schema=schema)


# ---------------------------------------------------------------------------
# Edge-kind classification — all three new rules are asserted
# ---------------------------------------------------------------------------


def test_new_rules_classified_as_asserted_in_edge_kind_registry():
    """All three new materializer rules are classified as ``asserted``
    in the canonical edge_kind registry (lib/ontology/edge_kind.py)."""
    from lib.ontology.edge_kind import EDGE_KIND_ASSERTED, edge_kind_for_rule

    for rule_name in (
        "corrected_by_from_chunk_misconception",
        "detected_by_from_distractor_misconception_id",
        "interferes_with_outcome_from_misconception_lo",
    ):
        assert edge_kind_for_rule(rule_name) == EDGE_KIND_ASSERTED, rule_name


def test_new_rules_stamp_edge_kind_on_emitted_edges():
    """Edges produced by ``build_semantic_graph`` for the three new
    rules carry ``edge_kind: "asserted"`` per Worker 1's stamping."""
    mc_id = "mc_" + "f" * 16
    chunks = [
        {
            "id": "chunk_001",
            "concept_tags": [],
            "learning_outcome_refs": [],
            "misconceptions": [{"misconception": "x", "correction": "y"}],
        }
    ]
    misconceptions = [{"id": mc_id, "lo_id": "to-01"}]
    questions = [{"id": "MCQ-1", "misconception_id": mc_id}]
    artifact = build_semantic_graph(
        chunks,
        None,
        _empty_graph(),
        now=FIXED_NOW,
        misconceptions=misconceptions,
        questions=questions,
    )
    for edge in artifact["edges"]:
        if edge["type"] in (
            "corrected-by-chunk",
            "detected-by-question",
            "interferes-with-outcome",
        ):
            assert edge.get("edge_kind") == "asserted", edge
