"""Edge-consensus aggregator tests (GPT feedback 12-may, item 2).

Coverage matrix (mirrors the wave plan at
``plans/gptfeedback-may12-item2-edge-consensus-2026-05.md``):

1. Confirming rule wires consensus → both edges land ``confirmed``.
2. Reverse-direction prerequisite → ``contradicted`` on both edges
   AND each pair lands in ``report['contradictions']``.
3. Orthogonal rule → ``pending`` with empty ``consensus_signals``.
4. Graceful degrade — missing graph path → empty-summary build.
5. Decision capture wiring — one ``edge_consensus_resolution``
   event per ``build()``; rationale ≥ 20 chars.
6. ``apply_to_graph`` is in-place + idempotent.
7. NLI flag (``TRAINFORGE_EDGE_NLI``) is plumbed through report
   metadata without flipping any edge_status today.
8. Drift defence — an edge with the wrong shape for
   ``consensus_signals`` does not crash the build.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from lib.aggregators.edge_consensus import (
    EdgeConsensusAggregator,
    MATRIX_VERSION,
    SCHEMA_VERSION,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

class _FakeCapture:
    """Captures every log_decision call as a dict for assertion."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


def _write_graph(tmp_path: Path, edges: List[Dict[str, Any]]) -> Path:
    """Persist a minimal semantic graph and return the path."""
    graph = {
        "kind": "concept_semantic",
        "generated_at": "2026-05-12T00:00:00Z",
        "nodes": [],
        "edges": edges,
        "rule_versions": {},
    }
    path = tmp_path / "concept_graph_semantic.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    return path


def _edge(
    *,
    source: str,
    target: str,
    edge_type: str,
    rule: str,
    rule_version: int = 1,
    confidence: float = 0.6,
) -> Dict[str, Any]:
    """Build a minimal edge with one provenance arm."""
    return {
        "source": source,
        "target": target,
        "type": edge_type,
        "confidence": confidence,
        "provenance": {
            "rule": rule,
            "rule_version": rule_version,
            "evidence": {},
        },
    }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_confirming_rule_wires_consensus(tmp_path: Path) -> None:
    """Test #1 — same-pair is_a + defined_by → both confirmed."""
    edges = [
        _edge(
            source="concept_a", target="concept_b",
            edge_type="is-a",
            rule="is_a_from_key_terms",
            confidence=0.7,
        ),
        _edge(
            source="concept_a", target="concept_b",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
            confidence=0.5,
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    capture = _FakeCapture()
    agg = EdgeConsensusAggregator(
        graph_path,
        course_slug="test-course",
        run_id="run_test_1",
        decision_capture=capture,
    )
    report = agg.build()

    # Apply the consensus stamps so the test can read them off the
    # graph dict the same way a downstream consumer would.
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    statuses = {(e["source"], e["target"], e["type"]): e["edge_status"]
                for e in graph_data["edges"]}
    assert statuses[("concept_a", "concept_b", "is-a")] == "confirmed"
    assert statuses[("concept_a", "concept_b", "defined-by")] == "confirmed"

    # Each edge carries one consensus_signals entry pointing at the
    # other rule.
    is_a_edge = next(
        e for e in graph_data["edges"] if e["type"] == "is-a"
    )
    assert any(
        sig["other_rule"] == "defined_by_from_first_mention"
        and sig["signal"] == "agree"
        for sig in is_a_edge["consensus_signals"]
    )

    # Report summary reflects the two confirmed edges.
    summary = report["summary"]
    assert summary["total_edges"] == 2
    assert summary["confirmed_count"] == 2
    assert summary["contradicted_count"] == 0
    assert summary["pending_count"] == 0
    assert summary["consensus_rate"] == 1.0


def test_reverse_direction_prerequisite_contradicted(tmp_path: Path) -> None:
    """Test #2 — A→B AND B→A prerequisite → contradicted both sides."""
    edges = [
        _edge(
            source="lo_a", target="lo_b",
            edge_type="prerequisite",
            rule="prerequisite_from_lo_order",
        ),
        _edge(
            source="lo_b", target="lo_a",
            edge_type="prerequisite",
            rule="prerequisite_from_lo_order",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    report = agg.build()

    summary = report["summary"]
    assert summary["contradicted_count"] == 2
    assert summary["confirmed_count"] == 0
    assert summary["contradiction_rate"] == 1.0

    # Both directions land in report['contradictions'] with detail
    # 'circular_prerequisite'.
    details = [c["detail"] for c in report["contradictions"]]
    assert details.count("circular_prerequisite") == 2

    # apply_to_graph stamps both edges with contradicted.
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    statuses = [e["edge_status"] for e in graph_data["edges"]]
    assert statuses == ["contradicted", "contradicted"]


def test_orthogonal_rule_is_pending(tmp_path: Path) -> None:
    """Test #3 — lone related_from_cooccurrence → pending."""
    edges = [
        _edge(
            source="c_x", target="c_y",
            edge_type="related-to",
            rule="related_from_cooccurrence",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    report = agg.build()

    summary = report["summary"]
    assert summary["pending_count"] == 1
    assert summary["confirmed_count"] == 0
    assert summary["contradicted_count"] == 0

    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    edge = graph_data["edges"][0]
    assert edge["edge_status"] == "pending"
    assert edge["consensus_signals"] == []


def test_graceful_degrade_on_missing_graph(tmp_path: Path) -> None:
    """Test #4 — non-existent path resolves to empty summary."""
    missing = tmp_path / "does_not_exist.json"
    agg = EdgeConsensusAggregator(missing, course_slug="t", run_id="r")
    report = agg.build()
    assert report["summary"]["total_edges"] == 0
    assert report["summary"]["confirmed_count"] == 0
    assert report["summary"]["contradicted_count"] == 0
    assert report["matrix_version"] == MATRIX_VERSION
    assert report["schema_version"] == SCHEMA_VERSION


def test_decision_capture_wired(tmp_path: Path) -> None:
    """Test #5 — one edge_consensus_resolution event per build()."""
    edges = [
        _edge(
            source="a", target="b",
            edge_type="related-to",
            rule="related_from_cooccurrence",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    capture = _FakeCapture()
    agg = EdgeConsensusAggregator(
        graph_path,
        course_slug="course-001",
        run_id="run-001",
        decision_capture=capture,
    )
    agg.build()
    assert len(capture.events) == 1
    event = capture.events[0]
    assert event["decision_type"] == "edge_consensus_resolution"
    assert len(event["rationale"]) >= 20
    # Rationale must interpolate the dynamic per-call signals required
    # by the wave plan's decision-capture contract.
    assert "course-001" in event["rationale"]
    assert "run-001" in event["rationale"]
    assert MATRIX_VERSION in event["rationale"]
    assert "total_edges=1" in event["rationale"]


def test_apply_to_graph_in_place_idempotent(tmp_path: Path) -> None:
    """Test #6 — apply_to_graph mutates in place; twice == once."""
    edges = [
        _edge(
            source="c_a", target="c_b",
            edge_type="is-a",
            rule="is_a_from_key_terms",
        ),
        _edge(
            source="c_a", target="c_b",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")

    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    first_pass = json.dumps(graph_data, sort_keys=True)

    agg.apply_to_graph(graph_data)
    second_pass = json.dumps(graph_data, sort_keys=True)

    assert first_pass == second_pass


def test_nli_flag_plumbed_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test #7 — TRAINFORGE_EDGE_NLI flag surfaces in the report and
    does NOT flip edge_status today (NLI stub returns None)."""
    edges = [
        _edge(
            source="c_x", target="c_y",
            edge_type="related-to",
            rule="related_from_cooccurrence",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)

    monkeypatch.setenv("TRAINFORGE_EDGE_NLI", "true")
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    report = agg.build()
    assert report["nli_extension_enabled"] is True
    # Today's stub returns None — status stays pending.
    assert report["summary"]["retracted_count"] == 0
    assert report["summary"]["pending_count"] == 1

    monkeypatch.setenv("TRAINFORGE_EDGE_NLI", "0")
    agg2 = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    report2 = agg2.build()
    assert report2["nli_extension_enabled"] is False


def test_drift_defence_malformed_edges(tmp_path: Path) -> None:
    """Test #8 — edges with wrong shape don't crash the build."""
    edges = [
        # Missing source
        {"target": "b", "type": "is-a", "provenance": {"rule": "x", "rule_version": 1}},
        # Missing type
        {"source": "a", "target": "b", "provenance": {"rule": "x", "rule_version": 1}},
        # Provenance missing rule
        {"source": "a", "target": "b", "type": "is-a", "provenance": {}},
        # Whole edge is not a dict
        "not a dict",
        # Valid edge to confirm we still process the survivors
        _edge(
            source="c_a", target="c_b",
            edge_type="related-to",
            rule="related_from_cooccurrence",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    report = agg.build()
    # The third edge has provenance but no rule → treated as pending.
    # The valid edge → pending (lone related-to).
    # The first / second / fourth → dropped before scoring.
    assert report["summary"]["total_edges"] >= 1


def test_write_emits_report(tmp_path: Path) -> None:
    """write() persists JSON next to the graph by default."""
    edges = [
        _edge(
            source="a", target="b",
            edge_type="is-a",
            rule="is_a_from_key_terms",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    out = agg.write()
    assert out is not None
    assert out.name == "edge_consensus_report.json"
    assert out.parent == graph_path.parent
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["schema_version"] == SCHEMA_VERSION
    assert body["matrix_version"] == MATRIX_VERSION


# ----------------------------------------------------------------------
# v2 cross-stratum triangulation tests
# ----------------------------------------------------------------------


def _prov_edge(
    *,
    source: str,
    target: str,
    edge_type: str,
    rule: str,
    evidence: Optional[Dict[str, Any]] = None,
    confidence: float = 0.6,
) -> Dict[str, Any]:
    """Edge builder with a populated evidence sub-dict on provenance."""
    return {
        "source": source,
        "target": target,
        "type": edge_type,
        "confidence": confidence,
        "provenance": {
            "rule": rule,
            "rule_version": 1,
            "evidence": evidence or {},
        },
    }


def _status_map(graph_data: Dict[str, Any]) -> Dict[Any, str]:
    return {
        (e["source"], e["target"], e["type"]): e["edge_status"]
        for e in graph_data["edges"]
    }


def test_defined_by_exemplifies_same_pair_both_confirmed(tmp_path: Path) -> None:
    """v2 #i — symmetry fix: a surviving defined_by x exemplifies same-pair
    co-fire (reversed directions, like the real corpus) confirms BOTH edges.
    """
    edges = [
        # defined-by: concept -> chunk
        _prov_edge(
            source="concept_z", target="chunk_00001",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
            evidence={"chunk_id": "chunk_00001", "concept_slug": "concept_z"},
        ),
        # exemplifies: chunk -> concept (reversed direction, same node pair)
        _prov_edge(
            source="chunk_00001", target="concept_z",
            edge_type="exemplifies",
            rule="exemplifies_from_example_chunks",
            evidence={"chunk_id": "chunk_00001", "concept_slug": "concept_z",
                      "content_type": "chunk_type"},
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    statuses = _status_map(graph_data)
    assert statuses[("concept_z", "chunk_00001", "defined-by")] == "confirmed"
    assert statuses[("chunk_00001", "concept_z", "exemplifies")] == "confirmed"


def test_prereq_corroborated_by_text_order(tmp_path: Path) -> None:
    """v2 #ii — prerequisite B->A where A's first-mention precedes B's →
    confirmed + agree detail text_order_corroborates_lo_order.
    """
    edges = [
        # prereq: lo_b depends on lo_a (edge source=B target=A, A is prereq)
        _prov_edge(
            source="concept_b", target="concept_a",
            edge_type="prerequisite",
            rule="prerequisite_from_lo_order",
        ),
        # defined-by anchoring first-mention ordinals: A in chunk 1, B in chunk 5
        _prov_edge(
            source="concept_a", target="chunk_00001",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
        _prov_edge(
            source="concept_b", target="chunk_00005",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    report = agg.build()
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    statuses = _status_map(graph_data)
    assert statuses[("concept_b", "concept_a", "prerequisite")] == "confirmed"
    prereq_edge = next(e for e in graph_data["edges"] if e["type"] == "prerequisite")
    assert any(
        s["signal"] == "agree"
        and s["detail"] == "text_order_corroborates_lo_order"
        for s in prereq_edge["consensus_signals"]
    )
    assert report["summary"]["contradicted_count"] == 0


def test_prereq_conflicting_text_order(tmp_path: Path) -> None:
    """v2 #iii — prerequisite B->A where A's first-mention is AFTER B's →
    contradicted, in report['contradictions'] with the distinct soft-conflict
    detail; contradiction_rate > 0.
    """
    edges = [
        # prereq: B depends on A; A is the prerequisite (target).
        _prov_edge(
            source="concept_b", target="concept_a",
            edge_type="prerequisite",
            rule="prerequisite_from_lo_order",
        ),
        # text order conflicts: A appears LATER (chunk 9) than B (chunk 2).
        _prov_edge(
            source="concept_a", target="chunk_00009",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
        _prov_edge(
            source="concept_b", target="chunk_00002",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    report = agg.build()
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    statuses = _status_map(graph_data)
    assert statuses[("concept_b", "concept_a", "prerequisite")] == "contradicted"
    details = [c["detail"] for c in report["contradictions"]]
    assert "lo_order_vs_text_order_conflict" in details
    # Distinct from a hard cycle — operators must be able to tell them apart.
    assert "circular_prerequisite" not in details
    assert report["summary"]["contradiction_rate"] > 0


def test_prereq_unparseable_chunk_id_pending(tmp_path: Path) -> None:
    """v2 #iv — prerequisite whose anchoring chunk IDs have no parseable
    ordinal → no T1 signal → pending (graceful degrade).
    """
    edges = [
        _prov_edge(
            source="concept_b", target="concept_a",
            edge_type="prerequisite",
            rule="prerequisite_from_lo_order",
        ),
        _prov_edge(
            source="concept_a", target="intro_section_alpha",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
        _prov_edge(
            source="concept_b", target="appendix_section_beta",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    statuses = _status_map(graph_data)
    assert statuses[("concept_b", "concept_a", "prerequisite")] == "pending"


def test_related_co_located_is_supported(tmp_path: Path) -> None:
    """v2 #v — related-to whose concepts share a first-mention chunk →
    supported, NOT in consensus_rate, counted in supported_rate.
    """
    edges = [
        _prov_edge(
            source="concept_a", target="concept_b",
            edge_type="related-to",
            rule="related_from_cooccurrence",
        ),
        # both concepts defined by the SAME chunk → co-location.
        _prov_edge(
            source="concept_a", target="chunk_00003",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
        _prov_edge(
            source="concept_b", target="chunk_00003",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    report = agg.build()
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    statuses = _status_map(graph_data)
    assert statuses[("concept_a", "concept_b", "related-to")] == "supported"
    summary = report["summary"]
    assert summary["supported_count"] == 1
    assert summary["supported_rate"] > 0
    # supported must NOT inflate consensus_rate. The two defined-by edges
    # share no confirming rule here (no is_a / exemplifies), so they're
    # pending; only the related-to is supported.
    assert summary["confirmed_count"] == 0
    assert summary["consensus_rate"] == 0.0
    rel_edge = next(e for e in graph_data["edges"] if e["type"] == "related-to")
    assert any(
        s["signal"] == "support" and s["detail"] == "co_located_first_mention"
        for s in rel_edge["consensus_signals"]
    )


def test_intra_chunk_link_co_located_stays_pending(tmp_path: Path) -> None:
    """v2 #vi — intra_chunk_link edge with co-located anchors stays pending
    (tautology exclusion — a co-location signal over a co-location edge is
    circular).
    """
    edges = [
        _prov_edge(
            source="concept_a", target="concept_b",
            edge_type="related-to",
            rule="intra_chunk_link",
        ),
        _prov_edge(
            source="concept_a", target="chunk_00003",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
        _prov_edge(
            source="concept_b", target="chunk_00003",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    statuses = _status_map(graph_data)
    assert statuses[("concept_a", "concept_b", "related-to")] == "pending"


def test_assesses_derived_triangle_both_confirmed(tmp_path: Path) -> None:
    """v2 #vii — assesses (question->LO) x derived (chunk->LO) closes a
    (chunk, LO) triangle → both confirmed, incl. case-insensitive LO match.
    """
    edges = [
        # assesses: question -> LO 'CO-01', evidence chunk 'chunk_00007'
        _prov_edge(
            source="q-001", target="CO-01",
            edge_type="assesses",
            rule="assesses_from_question_lo",
            evidence={"question_id": "q-001", "objective_id": "CO-01",
                      "source_chunk_id": "chunk_00007"},
            confidence=1.0,
        ),
        # derived: chunk_00007 -> lo 'co-01' (lowercase — case-insensitive)
        _prov_edge(
            source="chunk_00007", target="co-01",
            edge_type="derived-from-objective",
            rule="derived_from_lo_ref",
            evidence={"chunk_id": "chunk_00007", "objective_id": "co-01"},
            confidence=1.0,
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    statuses = _status_map(graph_data)
    assert statuses[("q-001", "CO-01", "assesses")] == "confirmed"
    assert statuses[("chunk_00007", "co-01", "derived-from-objective")] == "confirmed"
    for e in graph_data["edges"]:
        assert any(
            s["detail"] == "chunk_lo_triangle" for s in e["consensus_signals"]
        )


def test_targets_derived_defined_triangle(tmp_path: Path) -> None:
    """v2 #viii — targets-concept (LO->C) x derived (chunk->LO) x defined-by
    (C->chunk) closes an LO-chunk-concept triangle → confirmed on both the
    targets edge and the symmetric derived edge.
    """
    edges = [
        # targets-concept: LO 'co-02' -> concept 'concept_q'
        _prov_edge(
            source="co-02", target="concept_q",
            edge_type="targets-concept",
            rule="targets_concept_from_lo",
            evidence={"lo_id": "co-02", "concept_id": "concept_q",
                      "bloom_level": "understand"},
            confidence=1.0,
        ),
        # derived: chunk_00010 -> lo 'CO-02' (case-insensitive)
        _prov_edge(
            source="chunk_00010", target="CO-02",
            edge_type="derived-from-objective",
            rule="derived_from_lo_ref",
            evidence={"chunk_id": "chunk_00010", "objective_id": "CO-02"},
            confidence=1.0,
        ),
        # defined-by: concept_q -> chunk_00010 (concept's chunk == LO's chunk)
        _prov_edge(
            source="concept_q", target="chunk_00010",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
            evidence={"chunk_id": "chunk_00010", "concept_slug": "concept_q"},
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    statuses = _status_map(graph_data)
    assert statuses[("co-02", "concept_q", "targets-concept")] == "confirmed"
    assert statuses[("chunk_00010", "CO-02", "derived-from-objective")] == "confirmed"
    targets_edge = next(e for e in graph_data["edges"] if e["type"] == "targets-concept")
    assert any(
        s["detail"] == "lo_chunk_concept_triangle"
        for s in targets_edge["consensus_signals"]
    )


def test_apply_to_graph_idempotent_all_signal_kinds(tmp_path: Path) -> None:
    """v2 #ix — apply_to_graph twice is byte-identical on a fixture
    exercising every new signal kind (T1 agree, T1 disagree, T2, T3, T4).
    """
    edges = [
        # T1 disagree (soft conflict)
        _prov_edge(source="c_b", target="c_a", edge_type="prerequisite",
                   rule="prerequisite_from_lo_order"),
        _prov_edge(source="c_a", target="chunk_00009", edge_type="defined-by",
                   rule="defined_by_from_first_mention"),
        _prov_edge(source="c_b", target="chunk_00002", edge_type="defined-by",
                   rule="defined_by_from_first_mention"),
        # T2/T3
        _prov_edge(source="q-9", target="CO-09", edge_type="assesses",
                   rule="assesses_from_question_lo",
                   evidence={"question_id": "q-9", "objective_id": "CO-09",
                             "source_chunk_id": "chunk_00050"}, confidence=1.0),
        _prov_edge(source="chunk_00050", target="co-09",
                   edge_type="derived-from-objective", rule="derived_from_lo_ref",
                   evidence={"chunk_id": "chunk_00050", "objective_id": "co-09"},
                   confidence=1.0),
        _prov_edge(source="co-09", target="concept_w", edge_type="targets-concept",
                   rule="targets_concept_from_lo",
                   evidence={"lo_id": "co-09", "concept_id": "concept_w",
                             "bloom_level": "apply"}, confidence=1.0),
        _prov_edge(source="concept_w", target="chunk_00050", edge_type="defined-by",
                   rule="defined_by_from_first_mention",
                   evidence={"chunk_id": "chunk_00050", "concept_slug": "concept_w"}),
        # T4 support
        _prov_edge(source="concept_w", target="concept_x", edge_type="related-to",
                   rule="related_from_cooccurrence"),
        _prov_edge(source="concept_x", target="chunk_00050", edge_type="defined-by",
                   rule="defined_by_from_first_mention",
                   evidence={"chunk_id": "chunk_00050", "concept_slug": "concept_x"}),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    first = json.dumps(graph_data, sort_keys=True)
    agg.apply_to_graph(graph_data)
    second = json.dumps(graph_data, sort_keys=True)
    assert first == second
    # Sanity: the fixture really exercised a spread of verdicts.
    statuses = set(_status_map(graph_data).values())
    assert {"contradicted", "confirmed", "supported"} <= statuses


# ----------------------------------------------------------------------
# Contradicted-edge policy (TRAINFORGE_CONTRADICTED_EDGE_POLICY)
# ----------------------------------------------------------------------


def _reverse_prereq_pair() -> List[Dict[str, Any]]:
    """A→B AND B→A prerequisite cycle → both edges land 'contradicted'."""
    return [
        _edge(source="lo_a", target="lo_b", edge_type="prerequisite",
              rule="prerequisite_from_lo_order", confidence=0.6),
        _edge(source="lo_b", target="lo_a", edge_type="prerequisite",
              rule="prerequisite_from_lo_order", confidence=0.6),
    ]


def test_policy_unset_byte_identical_stamp_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Policy unset → apply_to_graph output is byte-identical to the
    pre-flag stamp-only behaviour (no confidence touch, status stays
    'contradicted', no sentinel field)."""
    monkeypatch.delenv("TRAINFORGE_CONTRADICTED_EDGE_POLICY", raising=False)
    graph_path = _write_graph(tmp_path, _reverse_prereq_pair())
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")

    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)

    for e in graph_data["edges"]:
        assert e["edge_status"] == "contradicted"
        # Confidence untouched, no decay sentinel stamped.
        assert e["confidence"] == 0.6
        assert "consensus_confidence_decayed" not in e


def test_policy_decay_multiplies_confidence_status_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """decay → contradicted edge's confidence × 0.5, status stays
    'contradicted'; idempotent (twice == once)."""
    monkeypatch.setenv("TRAINFORGE_CONTRADICTED_EDGE_POLICY", "decay")
    graph_path = _write_graph(tmp_path, _reverse_prereq_pair())
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")

    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    for e in graph_data["edges"]:
        assert e["edge_status"] == "contradicted"
        assert e["confidence"] == 0.3  # 0.6 * 0.5
        assert e["consensus_confidence_decayed"] is True

    # Idempotent: a second pass must NOT halve again (0.3 stays 0.3).
    first = json.dumps(graph_data, sort_keys=True)
    agg.apply_to_graph(graph_data)
    second = json.dumps(graph_data, sort_keys=True)
    assert first == second
    for e in graph_data["edges"]:
        assert e["confidence"] == 0.3


def test_policy_retract_restatuses_edge_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retract → contradicted edge re-statused 'retracted', edge retained
    in graph['edges'] (NOT deleted), confidence untouched."""
    monkeypatch.setenv("TRAINFORGE_CONTRADICTED_EDGE_POLICY", "retract")
    graph_path = _write_graph(tmp_path, _reverse_prereq_pair())
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")

    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    edge_count_before = len(graph_data["edges"])
    agg.apply_to_graph(graph_data)

    # Edges physically retained — provenance/replay preserved.
    assert len(graph_data["edges"]) == edge_count_before
    for e in graph_data["edges"]:
        assert e["edge_status"] == "retracted"
        assert e["confidence"] == 0.6  # confidence untouched by retract
        assert "consensus_confidence_decayed" not in e

    # Idempotent.
    first = json.dumps(graph_data, sort_keys=True)
    agg.apply_to_graph(graph_data)
    second = json.dumps(graph_data, sort_keys=True)
    assert first == second


def test_policy_only_touches_contradicted_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """decay/retract leave non-contradicted (confirmed/pending) edges
    completely alone."""
    monkeypatch.setenv("TRAINFORGE_CONTRADICTED_EDGE_POLICY", "retract")
    edges = [
        # confirmed same-pair pair
        _edge(source="c_a", target="c_b", edge_type="is-a",
              rule="is_a_from_key_terms", confidence=0.7),
        _edge(source="c_a", target="c_b", edge_type="defined-by",
              rule="defined_by_from_first_mention", confidence=0.5),
        # lone pending related-to
        _edge(source="c_x", target="c_y", edge_type="related-to",
              rule="related_from_cooccurrence", confidence=0.4),
    ]
    graph_path = _write_graph(tmp_path, edges)
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph_data)
    statuses = _status_map(graph_data)
    assert statuses[("c_a", "c_b", "is-a")] == "confirmed"
    assert statuses[("c_a", "c_b", "defined-by")] == "confirmed"
    assert statuses[("c_x", "c_y", "related-to")] == "pending"
    # None retracted (none were contradicted).
    assert all(e["edge_status"] != "retracted" for e in graph_data["edges"])


def test_policy_invalid_value_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid flag value raises ContradictedEdgePolicyError BEFORE any
    edge is mutated (no half-written graph)."""
    from lib.aggregators.edge_consensus import ContradictedEdgePolicyError

    monkeypatch.setenv("TRAINFORGE_CONTRADICTED_EDGE_POLICY", "delete")
    graph_path = _write_graph(tmp_path, _reverse_prereq_pair())
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))

    with pytest.raises(ContradictedEdgePolicyError):
        agg.apply_to_graph(graph_data)

    # No edge was stamped before the raise (fail-closed up front).
    for e in graph_data["edges"]:
        assert "edge_status" not in e
        assert "consensus_signals" not in e


def test_policy_does_not_affect_build_contradiction_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build()'s contradiction_rate counts the contradicted VERDICT
    regardless of the policy — coherence with KGQualityValidator's
    (1 - contradiction_rate) consistency attenuation. A retracted/decayed
    edge still counts as evidence of contradiction."""
    monkeypatch.setenv("TRAINFORGE_CONTRADICTED_EDGE_POLICY", "retract")
    graph_path = _write_graph(tmp_path, _reverse_prereq_pair())
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    report = agg.build()
    summary = report["summary"]
    # build() does not apply the policy; the contradicted verdict survives.
    assert summary["contradicted_count"] == 2
    assert summary["contradiction_rate"] == 1.0
    assert summary["retracted_count"] == 0


def test_matrix_keys_cover_registered_rules() -> None:
    """Drift defence — _RULE_PAIR_MATRIX keys ⊇ registered Trainforge rule
    names ∪ {'intra_chunk_link'}. Skipped when Trainforge import unavailable.
    """
    pytest.importorskip("Trainforge.rag.inference_rules")
    from lib.aggregators.edge_consensus import _RULE_PAIR_MATRIX
    import Trainforge.rag.inference_rules as ir

    registered = set()
    for name in dir(ir):
        mod = getattr(ir, name)
        rule_name = getattr(mod, "RULE_NAME", None)
        if isinstance(rule_name, str) and rule_name:
            registered.add(rule_name)
    registered.add("intra_chunk_link")
    missing = registered - set(_RULE_PAIR_MATRIX.keys())
    assert not missing, f"matrix missing rule names: {sorted(missing)}"


# ----------------------------------------------------------------------
# NLI extension (TRAINFORGE_EDGE_NLI) — chunk-anchored contradiction arm.
# Tests monkeypatch NliClassifier.get_or_load so the real ~750MB DeBERTa
# model NEVER loads.
# ----------------------------------------------------------------------

class _FakeNliScore:
    def __init__(self, entailment: float, neutral: float, contradiction: float) -> None:
        self.entailment = entailment
        self.neutral = neutral
        self.contradiction = contradiction


class _FakeNliClassifier:
    """Scripted NLI classifier. ``contradiction_for`` maps a hypothesis
    substring → contradiction probability; default 0.0 (no signal)."""

    def __init__(self, contradiction_for: Optional[Dict[str, float]] = None) -> None:
        self.contradiction_for = contradiction_for or {}
        self.calls: List[Dict[str, str]] = []

    def score_pair(self, *, premise: str, hypothesis: str) -> _FakeNliScore:
        self.calls.append({"premise": premise, "hypothesis": hypothesis})
        contradiction = 0.0
        for needle, value in self.contradiction_for.items():
            if needle in hypothesis:
                contradiction = value
                break
        return _FakeNliScore(
            entailment=max(0.0, 1.0 - contradiction),
            neutral=0.0,
            contradiction=contradiction,
        )


def _patch_nli(monkeypatch: pytest.MonkeyPatch, fake: _FakeNliClassifier) -> None:
    import lib.classifiers.nli_classifier as nli_mod

    monkeypatch.setattr(
        nli_mod.NliClassifier, "get_or_load", classmethod(lambda cls: fake)
    )


def _defined_by_edges() -> List[Dict[str, Any]]:
    # defined-by: concept -> chunk (chunk endpoint = target).
    return [
        _prov_edge(
            source="photosynthesis", target="chunk_00001",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
    ]


def test_nli_no_lookup_no_op_even_with_flag_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag on but NO chunk_text_lookup supplied → extension no-ops
    exactly as the deferred stub did (default-off no-op preserved)."""
    monkeypatch.setenv("TRAINFORGE_EDGE_NLI", "true")
    fake = _FakeNliClassifier({"defines": 0.99})
    _patch_nli(monkeypatch, fake)
    graph_path = _write_graph(tmp_path, _defined_by_edges())
    agg = EdgeConsensusAggregator(graph_path, course_slug="t", run_id="r")
    report = agg.build()
    assert report["summary"]["retracted_count"] == 0
    assert fake.calls == []  # classifier never invoked without a lookup


def test_nli_contradiction_retracts_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag on + lookup + strong contradiction (>=0.5) → edge retracted."""
    monkeypatch.setenv("TRAINFORGE_EDGE_NLI", "true")
    fake = _FakeNliClassifier({"defines photosynthesis": 0.87})
    _patch_nli(monkeypatch, fake)
    edges = _defined_by_edges()
    graph_path = _write_graph(tmp_path, edges)
    lookup = {"chunk_00001": "This passage is about mitochondria and ATP only."}
    agg = EdgeConsensusAggregator(
        graph_path, course_slug="t", run_id="r", chunk_text_lookup=lookup,
    )
    report = agg.build()
    assert report["summary"]["retracted_count"] == 1
    # Hypothesis rendered from the template + humanized concept.
    assert fake.calls
    assert "This text defines photosynthesis." == fake.calls[0]["hypothesis"]

    # apply_to_graph stamps edge_status: retracted + the nli signal.
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    agg.apply_to_graph(graph)
    edge = graph["edges"][0]
    assert edge["edge_status"] == "retracted"
    nli_sigs = [
        s for s in edge["consensus_signals"]
        if s.get("other_rule") == "nli_text_entailment"
    ]
    assert nli_sigs and nli_sigs[0]["signal"] == "disagree"
    assert nli_sigs[0]["confidence"] == pytest.approx(0.87)


def test_nli_below_threshold_no_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contradiction below 0.5 → no signal (entailment/neutral add nothing)."""
    monkeypatch.setenv("TRAINFORGE_EDGE_NLI", "true")
    fake = _FakeNliClassifier({"defines": 0.42})
    _patch_nli(monkeypatch, fake)
    graph_path = _write_graph(tmp_path, _defined_by_edges())
    lookup = {"chunk_00001": "Photosynthesis converts light to chemical energy."}
    agg = EdgeConsensusAggregator(
        graph_path, course_slug="t", run_id="r", chunk_text_lookup=lookup,
    )
    report = agg.build()
    assert report["summary"]["retracted_count"] == 0
    assert report["summary"]["pending_count"] == 1
    assert fake.calls  # classifier WAS invoked, just under threshold


def test_nli_missing_chunk_text_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lookup supplied but the cited chunk id is absent → no forward pass."""
    monkeypatch.setenv("TRAINFORGE_EDGE_NLI", "true")
    fake = _FakeNliClassifier({"defines": 0.99})
    _patch_nli(monkeypatch, fake)
    graph_path = _write_graph(tmp_path, _defined_by_edges())
    lookup = {"chunk_99999": "unrelated"}
    agg = EdgeConsensusAggregator(
        graph_path, course_slug="t", run_id="r", chunk_text_lookup=lookup,
    )
    report = agg.build()
    assert report["summary"]["retracted_count"] == 0
    assert fake.calls == []


def test_nli_per_run_cap_bounds_forward_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TRAINFORGE_EDGE_NLI_MAX_EDGES caps the number of scored edges."""
    monkeypatch.setenv("TRAINFORGE_EDGE_NLI", "true")
    monkeypatch.setenv("TRAINFORGE_EDGE_NLI_MAX_EDGES", "1")
    fake = _FakeNliClassifier({"defines": 0.99})
    _patch_nli(monkeypatch, fake)
    edges = [
        _prov_edge(
            source="alpha", target="chunk_00001",
            edge_type="defined-by", rule="defined_by_from_first_mention",
        ),
        _prov_edge(
            source="beta", target="chunk_00002",
            edge_type="defined-by", rule="defined_by_from_first_mention",
        ),
    ]
    graph_path = _write_graph(tmp_path, edges)
    lookup = {
        "chunk_00001": "Text one about alpha.",
        "chunk_00002": "Text two about beta.",
    }
    agg = EdgeConsensusAggregator(
        graph_path, course_slug="t", run_id="r", chunk_text_lookup=lookup,
    )
    report = agg.build()
    # Only one edge scored under the cap; only that one can retract.
    assert len(fake.calls) == 1
    assert report["summary"]["retracted_count"] == 1


def test_nli_flag_off_ignores_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag OFF + lookup supplied → extension never runs (byte-identical)."""
    monkeypatch.delenv("TRAINFORGE_EDGE_NLI", raising=False)
    fake = _FakeNliClassifier({"defines": 0.99})
    _patch_nli(monkeypatch, fake)
    graph_path = _write_graph(tmp_path, _defined_by_edges())
    lookup = {"chunk_00001": "anything"}
    agg = EdgeConsensusAggregator(
        graph_path, course_slug="t", run_id="r", chunk_text_lookup=lookup,
    )
    report = agg.build()
    assert report["summary"]["retracted_count"] == 0
    assert fake.calls == []


def test_load_chunk_text_lookup_reads_course_tree(tmp_path: Path) -> None:
    """load_chunk_text_lookup builds {id: text} from a course chunkset."""
    from lib.aggregators.edge_consensus import load_chunk_text_lookup

    course_dir = tmp_path / "course"
    dart = course_dir / "dart_chunks"
    dart.mkdir(parents=True)
    (dart / "chunks.jsonl").write_text(
        json.dumps({"id": "chunk_00001", "text": "hello"}) + "\n"
        + json.dumps({"chunk_id": "chunk_00002", "text": "world"}) + "\n"
        + "not json\n"
        + json.dumps({"id": "chunk_00003"}) + "\n",  # no text → skipped
        encoding="utf-8",
    )
    lookup = load_chunk_text_lookup(course_dir)
    assert lookup == {"chunk_00001": "hello", "chunk_00002": "world"}
    # No chunkset → None (extension no-ops).
    assert load_chunk_text_lookup(tmp_path / "empty") is None
    assert load_chunk_text_lookup(None) is None
