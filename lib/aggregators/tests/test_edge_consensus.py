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
