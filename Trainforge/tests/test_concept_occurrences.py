"""Regression tests for REC-LNK-01 (Wave 5.1, Worker S).

Covers the ``occurrences[]`` back-reference on concept-graph nodes — the
list of chunk IDs that mention each concept. Populated from the
chunk->concept inverted index inside ``CourseProcessor._build_tag_graph``.

Five behaviour contracts (per the master plan):

1. Nodes carry a populated ``occurrences[]`` after graph build.
2. ``occurrences[]`` is deterministically sorted.
3. ``occurrences[]`` matches a manually computed inverted index.
4. Legacy nodes WITHOUT ``occurrences[]`` still validate against the
   semantic-graph schema (optional field).
5. Under ``TRAINFORGE_CONTENT_HASH_IDS=true`` (Worker N's Wave 4 flag),
   ``occurrences[]`` is stable across re-builds of the same source text.

The implementation lives in ``Trainforge/process_course.py::_build_tag_graph``.
These tests mirror the lightweight helper pattern from
``test_concept_scoping.py`` so they don't spin up the full IMSCC
ingestion pipeline — the production function's logic is copy-free replicated
only up to the fields we assert on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.rag import typed_edge_inference  # noqa: E402

SCHEMA_PATH = (
    PROJECT_ROOT
    / "schemas"
    / "knowledge"
    / "concept_graph_semantic.schema.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_concept_graph(chunks, course_id=""):
    """Invoke the production ``_build_tag_graph`` via the CourseProcessor.

    We intentionally call through the real implementation (instead of
    re-implementing inline) so these tests exercise the shipped code
    path. The processor is constructed via ``__new__`` to skip the
    IMSCC-ingestion ``__init__`` — only ``course_code`` is attached so
    the helper's course-id fallback branch behaves.
    """
    from Trainforge.process_course import CourseProcessor

    processor = CourseProcessor.__new__(CourseProcessor)
    processor.course_code = course_id
    return processor._build_tag_graph(chunks)


def _mk_chunk(chunk_id, tags):
    """Minimal chunk shape: ``_build_tag_graph`` reads ``id`` + ``concept_tags``."""
    return {"id": chunk_id, "concept_tags": list(tags)}


# ---------------------------------------------------------------------------
# Test 1 — nodes carry a populated occurrences[] list
# ---------------------------------------------------------------------------

def test_node_carries_occurrences_list(monkeypatch):
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    # Three chunks, tags engineered to produce two qualifying nodes (a, b)
    # at min_freq=2; c appears once and is filtered out.
    chunks = [
        _mk_chunk("c_00001", ["a", "b"]),
        _mk_chunk("c_00002", ["a", "c"]),
        _mk_chunk("c_00003", ["b"]),
    ]
    graph = _build_concept_graph(chunks)

    by_id = {n["id"]: n for n in graph["nodes"]}
    # Only nodes with freq>=2 survive — a and b qualify, c filtered.
    assert set(by_id.keys()) == {"a", "b"}, by_id.keys()

    assert by_id["a"].get("occurrences") == ["c_00001", "c_00002"], by_id["a"]
    assert by_id["b"].get("occurrences") == ["c_00001", "c_00003"], by_id["b"]


# ---------------------------------------------------------------------------
# Test 2 — occurrences[] is deterministically sorted
# ---------------------------------------------------------------------------

def test_occurrences_are_sorted(monkeypatch):
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    # Build in reverse order so any lurking "insertion-order" assumption
    # would produce a non-sorted result.
    chunks = [
        _mk_chunk("c_00009", ["x"]),
        _mk_chunk("c_00005", ["x"]),
        _mk_chunk("c_00003", ["x"]),
        _mk_chunk("c_00001", ["x"]),
    ]
    graph = _build_concept_graph(chunks)

    nodes = [n for n in graph["nodes"] if n["id"] == "x"]
    assert len(nodes) == 1
    occurrences = nodes[0].get("occurrences")
    assert occurrences == sorted(occurrences), occurrences
    assert occurrences == ["c_00001", "c_00003", "c_00005", "c_00009"]

    # Duplicate-tag-on-same-chunk sanity: chunk ID must appear only once.
    dup_chunks = [
        _mk_chunk("c_00001", ["y", "y", "y"]),
        _mk_chunk("c_00002", ["y"]),
    ]
    dup_graph = _build_concept_graph(dup_chunks)
    y_nodes = [n for n in dup_graph["nodes"] if n["id"] == "y"]
    assert len(y_nodes) == 1
    # chunk_00001 listed "y" three times — appears in occurrences ONCE.
    assert y_nodes[0].get("occurrences") == ["c_00001", "c_00002"], y_nodes[0]


# ---------------------------------------------------------------------------
# Test 3 — occurrences[] matches a manually computed inverted index
# ---------------------------------------------------------------------------

def test_occurrences_match_inverted_index(monkeypatch):
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    chunks = [
        _mk_chunk("c_00001", ["alpha", "beta", "gamma"]),
        _mk_chunk("c_00002", ["alpha", "beta"]),
        _mk_chunk("c_00003", ["beta", "gamma"]),
        _mk_chunk("c_00004", ["alpha", "gamma"]),
        _mk_chunk("c_00005", ["alpha"]),
    ]
    graph = _build_concept_graph(chunks)

    # Manually compute the inverted index.
    manual = {}
    for chunk in chunks:
        for tag in chunk["concept_tags"]:
            manual.setdefault(tag, set()).add(chunk["id"])

    # For every emitted node, occurrences[] must equal sorted(manual[node_id]).
    for node in graph["nodes"]:
        node_id = node["id"]
        expected = sorted(manual[node_id])
        assert node.get("occurrences") == expected, (
            f"node {node_id} occurrences mismatch: got "
            f"{node.get('occurrences')}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Test 4 — legacy nodes without occurrences[] validate against the schema
# ---------------------------------------------------------------------------

def test_legacy_nodes_without_occurrences_validate():
    jsonschema = pytest.importorskip("jsonschema")
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    legacy_artifact = {
        "kind": "concept_semantic",
        "generated_at": "2026-04-20T00:00:00+00:00",
        "rule_versions": {},
        "nodes": [
            # No occurrences field — legacy Wave 4 shape.
            {"id": "accessibility", "label": "Accessibility", "frequency": 3},
            {"id": "wcag", "label": "WCAG", "frequency": 2},
        ],
        "edges": [],
    }
    # Must validate — schema addition is optional.
    jsonschema.validate(instance=legacy_artifact, schema=schema)

    # And the new-shape artifact (with occurrences) validates too.
    new_artifact = {
        "kind": "concept_semantic",
        "generated_at": "2026-04-20T00:00:00+00:00",
        "rule_versions": {},
        "nodes": [
            {
                "id": "accessibility",
                "label": "Accessibility",
                "frequency": 3,
                "occurrences": ["c_00001", "c_00002", "c_00003"],
            }
        ],
        "edges": [],
    }
    jsonschema.validate(instance=new_artifact, schema=schema)


# ---------------------------------------------------------------------------
# Test 5 — occurrences[] survives re-chunk under content-hash IDs
# ---------------------------------------------------------------------------

def test_occurrences_survive_rechunk_under_content_hash(monkeypatch):
    """Under TRAINFORGE_CONTENT_HASH_IDS=true, re-processing the same
    semantic content produces identical chunk IDs (Worker N's Wave 4
    contract). Therefore, occurrences[] must also be identical across
    two independent graph builds from the same source.

    We simulate the "re-chunk" by producing chunks with content-hash
    IDs via the shipped ``_generate_chunk_id`` helper and feeding them
    into two independent graph-build calls.
    """
    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)
    monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "true")

    from Trainforge.process_course import _generate_chunk_id

    # Fixture data: three "chunks" of text; hash-ID helper produces
    # content-addressed IDs that are stable across runs.
    payload = [
        ("course_content/page_01.html", "Accessibility is a core principle.", ["accessibility", "principles"]),
        ("course_content/page_02.html", "WCAG standards codify accessibility.", ["accessibility", "wcag"]),
        ("course_content/page_03.html", "Principles underlie every WCAG rule.", ["principles", "wcag"]),
    ]

    def _build_chunks_for_run():
        """Build a fresh chunk list — IDs derived from the same content
        hash each run, so across runs the list is byte-identical.
        """
        chunks = []
        for idx, (source, text, tags) in enumerate(payload):
            chunk_id = _generate_chunk_id(
                prefix="testcourse_chunk_",
                start_id=idx,
                text=text,
                source_locator=source,
            )
            chunks.append({
                "id": chunk_id,
                "concept_tags": tags,
            })
        return chunks

    run_a = _build_concept_graph(_build_chunks_for_run())
    run_b = _build_concept_graph(_build_chunks_for_run())

    # Every node in run_a must appear in run_b with identical occurrences[].
    by_a = {n["id"]: n.get("occurrences") for n in run_a["nodes"]}
    by_b = {n["id"]: n.get("occurrences") for n in run_b["nodes"]}
    assert by_a.keys() == by_b.keys(), (by_a.keys(), by_b.keys())
    for node_id, occurrences_a in by_a.items():
        assert occurrences_a == by_b[node_id], (
            f"occurrences drift for {node_id}: "
            f"run_a={occurrences_a} run_b={by_b[node_id]}"
        )
    # Sanity: at least one node must have non-empty occurrences — otherwise
    # the test proves nothing.
    assert any(v for v in by_a.values()), by_a
