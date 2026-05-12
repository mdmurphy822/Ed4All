"""Regression tests for GPT Feedback (May 12) item 3 lineage fields.

Validates the three new top-level fields on the typed concept graph:

- ``rulepack_version`` (aggregate version of the rulepack)
- ``graph_build_hash`` (deterministic SHA over canonicalised inputs)
- ``course_package_version`` (Courseforge package version; null today)

Plus the threading on the canonical Source dict.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from jsonschema import Draft7Validator

from Trainforge.rag.inference_rules import RULEPACK_VERSION
from Trainforge.rag.typed_edge_inference import build_semantic_graph


_FIXED_NOW = datetime(2026, 5, 12, 12, 0, 0, tzinfo=timezone.utc)
_EMPTY_CONCEPT_GRAPH = {
    "kind": "cooccurrence",
    "generated_at": "2026-05-12T00:00:00+00:00",
    "nodes": [],
    "edges": [],
}


def _load_graph_schema():
    schema_path = (
        "/home/mdmur/Projects/Ed4All/schemas/knowledge/"
        "concept_graph_semantic.schema.json"
    )
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_rulepack_version_shape_matches_schema():
    """``RULEPACK_VERSION`` matches ``^v[0-9a-f]+$``."""
    import re

    assert re.match(r"^v[0-9a-f]+$", RULEPACK_VERSION), RULEPACK_VERSION


def test_graph_carries_lineage_fields():
    """``build_semantic_graph`` stamps the three new top-level fields."""
    graph = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph=_EMPTY_CONCEPT_GRAPH,
        now=_FIXED_NOW,
    )
    assert graph["rulepack_version"] == RULEPACK_VERSION
    assert isinstance(graph["graph_build_hash"], str)
    import re

    assert re.match(r"^[a-f0-9]{64}$", graph["graph_build_hash"])
    # course_package_version is intentionally null today (Courseforge does
    # not emit a stable signature). Schema admits it; producer threads it.
    assert graph["course_package_version"] is None


def test_graph_build_hash_is_deterministic_independent_of_now():
    """Two runs with the same inputs but different ``now`` hash the same."""
    g1 = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph=_EMPTY_CONCEPT_GRAPH,
        now=_FIXED_NOW,
    )
    g2 = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph=_EMPTY_CONCEPT_GRAPH,
        now=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    # Timestamps differ.
    assert g1["generated_at"] != g2["generated_at"]
    # But the build hash does NOT — that's the whole point of the field.
    assert g1["graph_build_hash"] == g2["graph_build_hash"]


def test_graph_build_hash_changes_with_chunk_ids():
    """Adding a chunk changes the build hash."""
    g_no_chunks = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph=_EMPTY_CONCEPT_GRAPH,
        now=_FIXED_NOW,
    )
    g_with_chunks = build_semantic_graph(
        chunks=[
            {
                "id": "test_chunk_00001",
                "text": "hello",
                "concept_tags": [],
            }
        ],
        course=None,
        concept_graph=_EMPTY_CONCEPT_GRAPH,
        now=_FIXED_NOW,
    )
    assert g_no_chunks["graph_build_hash"] != g_with_chunks["graph_build_hash"]


def test_graph_build_hash_changes_with_rulepack_version_change(monkeypatch):
    """Bumping the rulepack version flips the build hash."""
    g1 = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph=_EMPTY_CONCEPT_GRAPH,
        now=_FIXED_NOW,
    )
    # Patch in a different rulepack version and rebuild.
    monkeypatch.setattr(
        "Trainforge.rag.typed_edge_inference.RULEPACK_VERSION",
        "vDEADBEEF",
    )
    g2 = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph=_EMPTY_CONCEPT_GRAPH,
        now=_FIXED_NOW,
    )
    assert g1["graph_build_hash"] != g2["graph_build_hash"]


def test_course_package_version_pickup_from_course_dict():
    """When ``course.package_version`` is set, it propagates."""
    graph = build_semantic_graph(
        chunks=[],
        course={"course_id": "TEST_101", "package_version": "imscc-2026-05-12-abc123"},
        concept_graph=_EMPTY_CONCEPT_GRAPH,
        now=_FIXED_NOW,
    )
    assert graph["course_package_version"] == "imscc-2026-05-12-abc123"


def test_course_package_version_explicit_kwarg_wins():
    """Explicit kwarg overrides the course-dict pickup."""
    graph = build_semantic_graph(
        chunks=[],
        course={"course_id": "TEST_101", "package_version": "from-course-json"},
        concept_graph=_EMPTY_CONCEPT_GRAPH,
        now=_FIXED_NOW,
        course_package_version="from-explicit-kwarg",
    )
    assert graph["course_package_version"] == "from-explicit-kwarg"


def test_legacy_graph_without_new_fields_still_validates():
    """A pre-May-2026 graph (no new fields) validates against the schema."""
    schema = _load_graph_schema()
    legacy_graph = {
        "kind": "concept_semantic",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "rule_versions": {"is_a_from_key_terms": 1},
        "nodes": [],
        "edges": [],
    }
    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(legacy_graph))
    assert not errors, [str(e) for e in errors]


def test_new_graph_with_lineage_fields_validates():
    """A graph carrying the three new fields validates."""
    schema = _load_graph_schema()
    graph = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph=_EMPTY_CONCEPT_GRAPH,
        now=_FIXED_NOW,
    )
    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(graph))
    assert not errors, [str(e) for e in errors]
