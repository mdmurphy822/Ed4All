"""Regression tests for typed endpoint-node materialization.

The pedagogical inference rules (``assesses_from_question_lo``,
``derived_from_lo_ref``, ...) emit edges whose endpoints are chunk IDs,
synthetic question IDs (``q_<chunk>_<lo>``), and learning-objective IDs
(``to-NN`` / ``co-NN``) — none of which are co-occurrence concept nodes.
Pre-fix, ``build_semantic_graph`` copied nodes ONLY from the upstream
concept_graph, so a course whose co-occurrence graph was degenerate (empty
``concept_tags`` → no DomainConcept nodes) shipped a
``concept_graph_semantic.json`` with typed edges and ZERO nodes.

The LibV2 ``packet_integrity`` ``edge_endpoint_typing`` rule then could not
classify the ``assesses`` endpoints and failed the typed-endpoint contract
(``EDGE_ENDPOINT_TYPE_MISMATCH``) — the archival blocker this fix closes.

These tests pin the fix:

* ``assesses`` edges get a ``Chunk``-class source node and an
  ``Outcome``/``ComponentObjective``-class target node.
* Every resolved-edge endpoint becomes a materialized node.
* The materialized graph passes the typing contract via the actual
  PacketIntegrityValidator rule (no ``EDGE_ENDPOINT_TYPE_MISMATCH``).
* The rich concept-graph case (every endpoint already a concept node) is
  unaffected — no spurious extra nodes.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lib.validators.libv2._packet_integrity_severity import EDGE_TYPING_CONTRACT
from Trainforge.rag.typed_edge_inference import (
    _classify_endpoint_id,
    build_semantic_graph,
)

FIXED_NOW = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _degenerate_assessment_inputs():
    """A course with assessment-item chunks + LO refs but NO concept nodes.

    Mirrors the real-course shape that triggered the archival blocker:
    an empty co-occurrence concept_graph, so only the chunk-anchored
    pedagogical rules fire.
    """
    chunks = [
        {
            "id": "course_chunk_00001",
            "chunk_type": "explanation",
            "text": "An explanation chunk.",
            "learning_outcome_refs": ["to-01"],
        },
        {
            "id": "course_chunk_00002",
            "chunk_type": "explanation",
            "text": "Another explanation chunk.",
            "learning_outcome_refs": ["co-01"],
        },
        {
            "id": "course_chunk_00007",
            "chunk_type": "assessment_item",
            "text": "Which option is correct?",
            "learning_outcome_refs": ["to-01"],
        },
        {
            "id": "course_chunk_00008",
            "chunk_type": "assessment_item",
            "text": "Pick the best answer.",
            "learning_outcome_refs": ["co-01"],
        },
    ]
    # Reproduces process_course._build_questions_for_graph.
    questions = []
    for c in chunks:
        if c.get("chunk_type") != "assessment_item":
            continue
        cid = c["id"]
        for ref in c.get("learning_outcome_refs") or []:
            questions.append(
                {"id": f"q_{cid}_{ref}", "objective_id": ref, "source_chunk_id": cid}
            )
    concept_graph = {"nodes": [], "edges": []}
    course = {"course_id": "course", "learning_outcomes": []}
    return chunks, course, concept_graph, questions


def test_classify_endpoint_id_namespaces():
    """ID-namespace classifier maps each endpoint to a canonical class."""
    # Chunk IDs.
    assert _classify_endpoint_id("course_chunk_00007") == "Chunk"
    # Synthetic question IDs carry a chunk_ token → Chunk (assesses source).
    assert _classify_endpoint_id("q_course_chunk_00007_to-01") == "Chunk"
    # Terminal vs component objective.
    assert _classify_endpoint_id("to-01") == "Outcome"
    assert _classify_endpoint_id("co-03") == "ComponentObjective"
    # Misconception IDs.
    assert _classify_endpoint_id("mc_0123456789abcdef") == "Misconception"
    # A plain concept slug is NOT classified here (it's already a node from
    # the co-occurrence graph; we must not guess a class for it).
    assert _classify_endpoint_id("accessibility") is None
    assert _classify_endpoint_id("") is None
    assert _classify_endpoint_id(None) is None  # type: ignore[arg-type]


def test_assesses_endpoints_get_typed_nodes():
    """assesses edges materialize Chunk source + Outcome/CO target nodes."""
    chunks, course, concept_graph, questions = _degenerate_assessment_inputs()
    graph = build_semantic_graph(
        chunks=chunks,
        course=course,
        concept_graph=concept_graph,
        questions=questions,
        now=FIXED_NOW,
    )

    node_class = {n["id"]: n.get("class") for n in graph["nodes"]}
    assesses = [e for e in graph["edges"] if e.get("type") == "assesses"]
    assert assesses, "expected at least one assesses edge"

    allowed_src, allowed_tgt = EDGE_TYPING_CONTRACT["assesses"]
    for edge in assesses:
        src, tgt = edge["source"], edge["target"]
        assert src in node_class, f"assesses source {src!r} not materialized"
        assert tgt in node_class, f"assesses target {tgt!r} not materialized"
        assert node_class[src] in allowed_src, (
            f"assesses source class {node_class[src]!r} not in {allowed_src}"
        )
        assert node_class[tgt] in allowed_tgt, (
            f"assesses target class {node_class[tgt]!r} not in {allowed_tgt}"
        )


def test_every_edge_endpoint_is_a_node():
    """No dangling endpoints: every resolved-edge endpoint has a node."""
    chunks, course, concept_graph, questions = _degenerate_assessment_inputs()
    graph = build_semantic_graph(
        chunks=chunks,
        course=course,
        concept_graph=concept_graph,
        questions=questions,
        now=FIXED_NOW,
    )
    node_ids = {n["id"] for n in graph["nodes"]}
    for edge in graph["edges"]:
        assert edge["source"] in node_ids, f"dangling source {edge['source']!r}"
        assert edge["target"] in node_ids, f"dangling target {edge['target']!r}"


def test_materialized_graph_passes_edge_typing_rule():
    """The materialized graph clears the packet_integrity typing rule."""
    from lib.validators.libv2._packet_integrity_result import ValidationResult
    from lib.validators.libv2.packet_integrity import PacketIntegrityValidator

    chunks, course, concept_graph, questions = _degenerate_assessment_inputs()
    graph = build_semantic_graph(
        chunks=chunks,
        course=course,
        concept_graph=concept_graph,
        questions=questions,
        now=FIXED_NOW,
    )

    result = ValidationResult(archive_root="<mem>")
    PacketIntegrityValidator._rule_edge_endpoint_typing(
        "edge_endpoint_typing",
        "critical",
        result,
        {},        # concept_graph
        {},        # pedagogy_graph
        graph,     # concept_graph_semantic
    )
    mismatches = [
        i for i in result.issues if i.issue_code == "EDGE_ENDPOINT_TYPE_MISMATCH"
    ]
    assert mismatches == [], f"unexpected typing mismatches: {mismatches}"


def test_rich_concept_graph_unaffected():
    """When every endpoint is already a concept node, no extra nodes appear."""
    concept_graph = {
        "nodes": [
            {"id": "alpha", "label": "Alpha", "frequency": 5, "class": "DomainConcept"},
            {"id": "beta", "label": "Beta", "frequency": 4, "class": "DomainConcept"},
        ],
        "edges": [],
    }
    # No assessment chunks / questions → only concept-anchored rules can fire,
    # and their endpoints are concept nodes that already exist.
    chunks = [
        {
            "id": "c1",
            "chunk_type": "explanation",
            "text": "Alpha relates to Beta.",
            "concept_tags": ["alpha", "beta"],
            "key_terms": [],
        }
    ]
    course = {"course_id": "course", "learning_outcomes": []}
    graph = build_semantic_graph(
        chunks=chunks,
        course=course,
        concept_graph=concept_graph,
        now=FIXED_NOW,
    )
    node_ids = {n["id"] for n in graph["nodes"]}
    # The two concept nodes carry through; no chunk/objective endpoint nodes
    # are synthesized because no pedagogical edge referenced one.
    assert node_ids == {"alpha", "beta"}
