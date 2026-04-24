"""Wave 66 — ``targets-concept`` typed edges materialize Wave 57 targetedConcepts[].

The inference rule ``targets_concept_from_lo.infer`` reads an
``objectives_metadata`` kwarg shaped like the Courseforge JSON-LD
``learningObjectives[]`` emit and produces one edge per ``{concept,
bloomLevel}`` entry in each LO's ``targetedConcepts[]`` array. This
closes the loop from Wave 57's emit to a typed KG edge.

Covers:

* Basic emit: every well-formed entry produces a ``targets-concept``
  edge with the expected shape (source=lo_id_lowercased, target=concept
  slug, type + confidence + provenance with bloom_level).
* LO IDs are normalized to lowercase (matches Trainforge's
  process_course case-insensitive reference-resolution convention).
* Deduplication: the same ``(lo_id, concept_id)`` pair only emits one
  edge even if repeated in the input.
* Defensive parsing: missing ``objectives_metadata``, empty list,
  missing ``id``, missing ``targetedConcepts``, malformed entries all
  produce an empty output without raising.
* Non-canonical Bloom levels (outside the 6-value enum) are dropped
  with a logged warning — prevents schema-drift from shipping into
  downstream validators.
* Schema round trip: the rule's output validates against
  ``schemas/knowledge/concept_graph_semantic.schema.json`` including
  the new ``TargetsConceptProvenance`` arm.
* Integration with ``build_semantic_graph``: an end-to-end call with
  ``objectives_metadata`` produces a graph whose edges include the
  new type, precedence table permits it, and schema validates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Trainforge.rag.inference_rules.targets_concept_from_lo import (  # noqa: E402
    EDGE_TYPE,
    RULE_NAME,
    RULE_VERSION,
    infer,
)
from Trainforge.rag.typed_edge_inference import build_semantic_graph  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


def _lo(lo_id: str, targets: list) -> dict:
    """Compact helper for building a LearningObjective fixture."""
    return {
        "id": lo_id,
        "statement": f"{lo_id} statement",
        "bloomLevel": "apply",
        "targetedConcepts": targets,
    }


def _target(concept: str, bloom: str) -> dict:
    return {"concept": concept, "bloomLevel": bloom}


# ---------------------------------------------------------------------- #
# 1. Basic emit shape
# ---------------------------------------------------------------------- #


def test_basic_emit_produces_expected_edge_shape():
    los = [
        _lo("TO-01", [_target("framework", "apply"), _target("sample-data", "apply")])
    ]
    edges = infer([], None, {}, objectives_metadata=los)
    assert len(edges) == 2
    for e in edges:
        assert e["type"] == EDGE_TYPE
        assert e["source"] == "to-01"
        assert e["confidence"] == 1.0
        assert e["provenance"]["rule"] == RULE_NAME
        assert e["provenance"]["rule_version"] == RULE_VERSION
        evidence = e["provenance"]["evidence"]
        assert evidence["lo_id"] == "to-01"
        assert evidence["bloom_level"] == "apply"
        assert evidence["concept_id"] == e["target"]
    # Deterministic sort by (source, target) → alphabetical concept order
    assert [e["target"] for e in edges] == ["framework", "sample-data"]


def test_lo_id_lowercased_to_match_trainforge_convention():
    """Uppercase TO-NN / CO-NN inputs must normalize to lowercase on edge."""
    los = [_lo("CO-05", [_target("x", "analyze")])]
    edges = infer([], None, {}, objectives_metadata=los)
    assert edges[0]["source"] == "co-05"
    assert edges[0]["provenance"]["evidence"]["lo_id"] == "co-05"


def test_bloom_level_carried_through_on_evidence():
    """Each entry's bloomLevel survives to edge provenance verbatim."""
    los = [
        _lo("TO-01", [_target("a", "remember"), _target("b", "create")]),
    ]
    edges = infer([], None, {}, objectives_metadata=los)
    by_concept = {e["target"]: e for e in edges}
    assert by_concept["a"]["provenance"]["evidence"]["bloom_level"] == "remember"
    assert by_concept["b"]["provenance"]["evidence"]["bloom_level"] == "create"


# ---------------------------------------------------------------------- #
# 2. Deduplication
# ---------------------------------------------------------------------- #


def test_same_lo_concept_pair_deduplicated():
    """Repeating the same (lo_id, concept) within targetedConcepts collapses."""
    los = [
        _lo(
            "TO-01",
            [
                _target("x", "apply"),
                _target("x", "apply"),  # duplicate
                _target("x", "analyze"),  # duplicate key — different bloom dropped
            ],
        )
    ]
    edges = infer([], None, {}, objectives_metadata=los)
    assert len(edges) == 1
    assert edges[0]["target"] == "x"
    # First-wins: the initial "apply" bloom stays
    assert edges[0]["provenance"]["evidence"]["bloom_level"] == "apply"


def test_same_concept_across_multiple_los_produces_distinct_edges():
    """Different LOs targeting the same concept produce one edge per LO."""
    los = [
        _lo("TO-01", [_target("shared-concept", "apply")]),
        _lo("CO-01", [_target("shared-concept", "analyze")]),
    ]
    edges = infer([], None, {}, objectives_metadata=los)
    assert len(edges) == 2
    sources = sorted(e["source"] for e in edges)
    assert sources == ["co-01", "to-01"]


# ---------------------------------------------------------------------- #
# 3. Defensive parsing
# ---------------------------------------------------------------------- #


def test_missing_kwarg_returns_empty():
    assert infer([], None, {}) == []


def test_empty_metadata_list_returns_empty():
    assert infer([], None, {}, objectives_metadata=[]) == []


def test_non_list_metadata_returns_empty():
    assert infer([], None, {}, objectives_metadata="not a list") == []


def test_lo_without_targeted_concepts_produces_no_edges():
    los = [
        {
            "id": "TO-01",
            "statement": "s",
            # no targetedConcepts field
        }
    ]
    assert infer([], None, {}, objectives_metadata=los) == []


def test_lo_with_empty_targeted_concepts_produces_no_edges():
    los = [_lo("TO-01", [])]
    assert infer([], None, {}, objectives_metadata=los) == []


def test_lo_without_id_skipped():
    los = [
        {
            "statement": "s",
            "targetedConcepts": [_target("x", "apply")],
        }
    ]
    assert infer([], None, {}, objectives_metadata=los) == []


def test_entry_with_missing_concept_skipped(caplog):
    import logging

    los = [_lo("TO-01", [{"bloomLevel": "apply"}])]  # no concept
    with caplog.at_level(logging.WARNING):
        edges = infer([], None, {}, objectives_metadata=los)
    assert edges == []
    assert any("missing/empty concept" in rec.getMessage() for rec in caplog.records)


def test_entry_with_non_canonical_bloom_level_dropped(caplog):
    import logging

    los = [_lo("TO-01", [_target("x", "bogus-level")])]
    with caplog.at_level(logging.WARNING):
        edges = infer([], None, {}, objectives_metadata=los)
    assert edges == []
    assert any(
        "non-canonical bloomLevel" in rec.getMessage() for rec in caplog.records
    )


def test_non_dict_entries_silently_skipped():
    los = [_lo("TO-01", ["not a dict", _target("ok", "apply")])]
    edges = infer([], None, {}, objectives_metadata=los)
    assert len(edges) == 1
    assert edges[0]["target"] == "ok"


# ---------------------------------------------------------------------- #
# 4. Schema round trip
# ---------------------------------------------------------------------- #


def test_edge_output_validates_against_semantic_graph_schema():
    """Rule output must conform to concept_graph_semantic.schema.json (incl.
    the new TargetsConceptProvenance arm)."""
    from jsonschema import Draft7Validator

    schema_path = (
        _PROJECT_ROOT / "schemas" / "knowledge" / "concept_graph_semantic.schema.json"
    )
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    los = [_lo("TO-01", [_target("x", "apply"), _target("y", "analyze")])]
    edges = infer([], None, {}, objectives_metadata=los)
    graph = {
        "kind": "concept_semantic",
        "generated_at": "2026-04-24T00:00:00Z",
        "nodes": [{"id": "x"}, {"id": "y"}],
        "edges": edges,
    }
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(graph), key=lambda e: list(e.absolute_path))
    assert not errors, (
        f"Schema violations: {[e.message for e in errors]}\n"
        f"Failing payload: {json.dumps(graph, indent=2)}"
    )


# ---------------------------------------------------------------------- #
# 5. Integration with build_semantic_graph
# ---------------------------------------------------------------------- #


def test_build_semantic_graph_surfaces_targets_concept_edges():
    """End-to-end: passing objectives_metadata through build_semantic_graph
    produces targets-concept edges in the final artifact."""
    los = [_lo("TO-01", [_target("framework", "apply")])]
    graph = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph={"nodes": [], "edges": []},
        objectives_metadata=los,
        now=None,
    )
    target_edges = [e for e in graph.get("edges", []) if e["type"] == EDGE_TYPE]
    assert len(target_edges) == 1
    edge = target_edges[0]
    assert edge["source"] == "to-01"
    assert edge["target"] == "framework"
    # Run-provenance stamping from the orchestrator (REC-PRV-01)
    assert "created_at" in edge
    # Rule versions map records the new rule.
    assert RULE_NAME in graph.get("rule_versions", {})
    assert graph["rule_versions"][RULE_NAME] == RULE_VERSION


def test_build_semantic_graph_without_metadata_produces_no_targets_concept_edges():
    """Backward compat: calling build_semantic_graph without the new kwarg
    produces zero targets-concept edges — no behavior change for legacy corpora."""
    graph = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph={"nodes": [], "edges": []},
        now=None,
    )
    target_edges = [e for e in graph.get("edges", []) if e["type"] == EDGE_TYPE]
    assert target_edges == []
