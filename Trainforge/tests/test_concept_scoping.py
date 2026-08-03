"""Verify the course-scoped concept ID behavior flag and artifact contract.

The tests cover five behavior contracts:

1. Flag OFF (default): concept node IDs are flat slugs; nodes have no
   ``course_id`` key.
2. Flag ON: concept node IDs are composite ``{course_id}:{slug}``; nodes
   carry a ``course_id`` field.
3. Schema accepts both flag-off and flag-on formats (optional field).
4. With flag on, two courses that share a concept slug do NOT silently
   merge into a single node.
5. ``course_id`` field is present when flag on, absent when flag off.

Tests patch ``typed_edge_inference.SCOPE_CONCEPT_IDS`` because the helper and
its consumers read that module-level setting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Trainforge.rag import typed_edge_inference
from Trainforge.rag.typed_edge_inference import _make_concept_id

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "knowledge"
    / "concept_graph_semantic.schema.json"
)


# ---------------------------------------------------------------------------
# This focused graph builder exercises concept ID construction without running
# the IMSCC ingestion path.
# ---------------------------------------------------------------------------


def _build_concept_graph(chunks, course_id):
    """Mirror of ``CourseProcessor._build_tag_graph`` scoped to these tests."""
    from collections import defaultdict

    tag_frequency = defaultdict(int)
    co_occurrence = defaultdict(int)
    for chunk in chunks:
        tags = chunk.get("concept_tags", [])
        for tag in tags:
            tag_frequency[tag] += 1
        for i, a in enumerate(tags):
            for b in tags[i + 1 :]:
                key = tuple(sorted([a, b]))
                co_occurrence[key] += 1

    nodes = []
    for tag, freq in sorted(tag_frequency.items(), key=lambda x: -x[1]):
        if freq < 2:
            # Retain low-frequency concepts so scoping is tested independently
            # of the production frequency threshold.
            pass
        node_id = _make_concept_id(tag, course_id)
        node = {
            "id": node_id,
            "label": tag.replace("-", " ").title(),
            "frequency": freq,
        }
        if typed_edge_inference.SCOPE_CONCEPT_IDS and course_id:
            node["course_id"] = course_id
        nodes.append(node)

    return {
        "kind": "concept",
        "generated_at": "2026-04-19T00:00:00+00:00",
        "nodes": nodes,
        "edges": [],
    }


def _chunks_for(course_id, tags_per_chunk):
    """Build test chunks carrying ``source.course_id`` and concept_tags."""
    return [
        {
            "id": f"{course_id}_chunk_{i:05d}",
            "source": {"course_id": course_id},
            "concept_tags": tags,
            "key_terms": [],
        }
        for i, tags in enumerate(tags_per_chunk)
    ]


# ---------------------------------------------------------------------------
# Test 1 — default (flag off) produces flat slugs and no course_id field
# ---------------------------------------------------------------------------


def test_flag_off_flat_slug_ids(monkeypatch):
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    chunks = _chunks_for("fixture_course", [["accessibility", "wcag"], ["accessibility"]])
    graph = _build_concept_graph(chunks, course_id="fixture_course")

    ids = [n["id"] for n in graph["nodes"]]
    assert "accessibility" in ids, ids
    # No composite form should appear.
    assert not any(":" in nid for nid in ids), ids
    # And no node should carry course_id in flag-off mode.
    for node in graph["nodes"]:
        assert "course_id" not in node, node


# ---------------------------------------------------------------------------
# Test 2 — flag on produces composite {course_id}:{slug} IDs
# ---------------------------------------------------------------------------


def test_flag_on_composite_ids(monkeypatch):
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", True)

    chunks = _chunks_for("fixture_course", [["accessibility", "wcag"], ["accessibility"]])
    graph = _build_concept_graph(chunks, course_id="fixture_course")

    ids = [n["id"] for n in graph["nodes"]]
    assert "fixture_course:accessibility" in ids, ids
    # Flat slug should NOT appear when the scope is enabled.
    assert "accessibility" not in ids, ids
    # Every node should carry the course_id field.
    for node in graph["nodes"]:
        assert node.get("course_id") == "fixture_course", node


# ---------------------------------------------------------------------------
# Test 3 — schema accepts both formats
# ---------------------------------------------------------------------------


def test_schema_accepts_both_formats():
    jsonschema = pytest.importorskip("jsonschema")
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    flat_artifact = {
        "kind": "concept_semantic",
        "generated_at": "2026-04-19T00:00:00+00:00",
        "rule_versions": {},
        "nodes": [{"id": "accessibility", "label": "Accessibility", "frequency": 3}],
        "edges": [],
    }
    jsonschema.validate(instance=flat_artifact, schema=schema)

    scoped_artifact = {
        "kind": "concept_semantic",
        "generated_at": "2026-04-19T00:00:00+00:00",
        "rule_versions": {},
        "nodes": [
            {
                "id": "fixture_course:accessibility",
                "label": "Accessibility",
                "frequency": 3,
                "course_id": "fixture_course",
            }
        ],
        "edges": [],
    }
    jsonschema.validate(instance=scoped_artifact, schema=schema)


# ---------------------------------------------------------------------------
# Test 4 — with flag on, two courses sharing a concept slug stay distinct
# ---------------------------------------------------------------------------


def test_cross_course_no_silent_merge_when_scoped(monkeypatch):
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", True)

    chunks_a = _chunks_for("course_a", [["accessibility", "markup"], ["accessibility"]])
    chunks_b = _chunks_for("course_b", [["accessibility", "markup"], ["accessibility"]])
    graph_a = _build_concept_graph(chunks_a, course_id="course_a")
    graph_b = _build_concept_graph(chunks_b, course_id="course_b")

    # Simulate a multi-course aggregation: union of node IDs.
    combined_ids = {n["id"] for n in graph_a["nodes"]}
    combined_ids.update(n["id"] for n in graph_b["nodes"])

    assert "course_a:accessibility" in combined_ids, combined_ids
    assert "course_b:accessibility" in combined_ids, combined_ids
    # Each course contributes a distinct scoped node.
    accessibility_variants = [nid for nid in combined_ids if nid.endswith(":accessibility")]
    assert len(accessibility_variants) == 2, accessibility_variants


# ---------------------------------------------------------------------------
# Test 5 — course_id field populated only when flag on
# ---------------------------------------------------------------------------


def test_course_id_field_populated_when_flag_on(monkeypatch):
    chunks = _chunks_for("fixture_course", [["accessibility", "wcag"], ["accessibility"]])

    # Flag OFF: course_id absent.
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)
    graph_off = _build_concept_graph(chunks, course_id="fixture_course")
    assert all("course_id" not in n for n in graph_off["nodes"]), graph_off["nodes"]

    # Flag ON: course_id present and equal to the scope.
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", True)
    graph_on = _build_concept_graph(chunks, course_id="fixture_course")
    assert all(n.get("course_id") == "fixture_course" for n in graph_on["nodes"]), graph_on["nodes"]


# ---------------------------------------------------------------------------
# Bonus — helper sanity (both branches)
# ---------------------------------------------------------------------------


def test_make_concept_id_flag_off(monkeypatch):
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)
    # Resolve the helper after patching the module-level behavior setting.
    from Trainforge.rag.typed_edge_inference import _make_concept_id as h

    assert h("accessibility", "fixture_course") == "accessibility"
    assert h("accessibility", None) == "accessibility"


def test_make_concept_id_flag_on(monkeypatch):
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", True)
    from Trainforge.rag.typed_edge_inference import _make_concept_id as h

    assert h("accessibility", "fixture_course") == "fixture_course:accessibility"
    # When course_id is missing/empty we fall back to flat slug even when
    # flag is on — prevents emitting a stray leading ``:``.
    assert h("accessibility", "") == "accessibility"
    assert h("accessibility", None) == "accessibility"
