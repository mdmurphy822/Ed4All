"""Post-loop edge-consensus aggregator regression tests (NVIDIA-KG item 3).

Mirror of the authoring-time stamping in
``MCP/tools/pipeline_tools.py::_run_concept_extraction`` for semantic
graphs that exist on disk at workflow end but were authored via another
route (process_course / IMSCC path writing
``<libv2_course>/graph/concept_graph_semantic.json``, or pre-fix corpora
under ``concept_graph/``). The post-loop helper
``WorkflowRunner._maybe_write_edge_consensus_reports`` must:

1. Stamp every edge of an un-stamped graph with a non-None
   ``edge_status`` + ``consensus_signals[]`` and write the sibling
   ``edge_consensus_report.json`` — for BOTH on-disk layouts
   (``graph/`` and ``concept_graph/``; real LibV2 courses carry either
   and some carry both).
2. Leave an already-stamped graph (e.g. stamped at authoring time by
   ``_run_concept_extraction``) byte-untouched when its sibling report
   already exists — the idempotency contract with the authoring-time
   wiring (``apply_to_graph`` itself is idempotent per
   ``lib/aggregators/tests/test_edge_consensus.py`` test #6, but
   re-serializing would churn formatting/mtime and re-writing the
   report would churn ``generated_at``).
3. Never flip ``final_status`` on aggregator failure (best-effort
   contract shared by all post-loop aggregators).

These tests drive ``run_workflow`` end-to-end against a tmp on-disk
workflow-state file with a mocked ``executor.execute_phase``, following
the harness in ``MCP/tests/test_workflow_runner_zombie_phase_guard.py``.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from unittest.mock import AsyncMock, MagicMock

from MCP.core import workflow_runner as wr_mod
from MCP.core.config import WorkflowConfig, WorkflowPhase
from MCP.core.executor import ExecutionResult
from MCP.core.workflow_runner import WorkflowRunner


WORKFLOW_ID = "WF-TEST-EC01"


# --------------------------------------------------------------------------
# Harness helpers (style of test_workflow_runner_zombie_phase_guard.py)
# --------------------------------------------------------------------------


def _write_workflow_state(state_root: Path, workflow_id: str) -> None:
    """Materialise the minimal on-disk workflow-state JSON run_workflow reads."""
    wf_dir = state_root / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{workflow_id}.json").write_text(
        json.dumps(
            {
                "workflow_id": workflow_id,
                "type": "test_wf",
                "params": {"course_name": "TEST_COURSE"},
                "phase_outputs": {},
                "tasks": [],
            }
        )
    )


def _result(
    task_id: str, status: str, payload: Dict[str, Any] | None = None
) -> ExecutionResult:
    return ExecutionResult(task_id=task_id, status=status, result=payload)


def _run_with_archival(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    course_dir: Path,
) -> Dict[str, Any]:
    """Drive ``run_workflow`` whose single ``libv2_archival`` phase
    completes carrying ``course_dir`` — the canonical resolution input
    of every post-loop aggregator (matches the sibling aggregators'
    ``phase_outputs.libv2_archival.course_dir`` priority-1 source).
    """
    state_root = tmp_path / "state"
    monkeypatch.setattr(wr_mod, "STATE_PATH", state_root)
    _write_workflow_state(state_root, WORKFLOW_ID)

    async def side_effect(*_a, **_kw):  # noqa: ANN001
        results = {
            "T-1": _result(
                "T-1",
                "COMPLETE",
                {
                    "course_slug": "test-course",
                    "course_dir": str(course_dir),
                },
            )
        }
        return results, True, []

    executor = MagicMock()
    executor.execute_phase = AsyncMock(side_effect=side_effect)

    config = MagicMock()
    config.get_workflow.return_value = WorkflowConfig(
        description="test",
        phases=[WorkflowPhase(name="libv2_archival", agents=["libv2-archivist"])],
    )

    runner = WorkflowRunner(executor=executor, config=config)
    return asyncio.run(runner.run_workflow(WORKFLOW_ID))


# --------------------------------------------------------------------------
# Graph fixture helpers (edge shapes mirror
# lib/aggregators/tests/test_edge_consensus.py)
# --------------------------------------------------------------------------


def _edge(
    *,
    source: str,
    target: str,
    edge_type: str,
    rule: str,
    confidence: float = 0.6,
) -> Dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "type": edge_type,
        "confidence": confidence,
        "provenance": {"rule": rule, "rule_version": 1, "evidence": {}},
    }


def _confirming_pair() -> List[Dict[str, Any]]:
    """Two same-pair edges whose rules confirm each other in the matrix."""
    return [
        _edge(
            source="concept_a",
            target="concept_b",
            edge_type="is-a",
            rule="is_a_from_key_terms",
        ),
        _edge(
            source="concept_a",
            target="concept_b",
            edge_type="defined-by",
            rule="defined_by_from_first_mention",
        ),
    ]


def _write_graph(
    course_dir: Path, layout: str, edges: List[Dict[str, Any]]
) -> Path:
    """Persist a minimal semantic graph under ``<course_dir>/<layout>/``."""
    graph_dir = course_dir / layout
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "concept_graph_semantic.json"
    graph_path.write_text(
        json.dumps(
            {
                "kind": "concept_semantic",
                "generated_at": "2026-05-12T00:00:00Z",
                "nodes": [],
                "edges": edges,
                "rule_versions": {},
            }
        ),
        encoding="utf-8",
    )
    return graph_path


def _load_edges(graph_path: Path) -> List[Dict[str, Any]]:
    return json.loads(graph_path.read_text(encoding="utf-8"))["edges"]


# --------------------------------------------------------------------------
# (1) un-stamped graph under graph/ -> stamped + sibling report written
# --------------------------------------------------------------------------


def test_unstamped_graph_layout_stamped_and_report_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    course_dir = tmp_path / "libv2" / "courses" / "test-course"
    graph_path = _write_graph(course_dir, "graph", _confirming_pair())

    out = _run_with_archival(tmp_path, monkeypatch, course_dir=course_dir)

    assert out["status"] == "COMPLETE"

    # Every edge stamped with a non-None edge_status + consensus_signals.
    edges = _load_edges(graph_path)
    assert edges, "graph lost its edges"
    for edge in edges:
        assert edge.get("edge_status") is not None, edge
        assert isinstance(edge.get("consensus_signals"), list), edge
    # The confirming pair resolves to confirmed/confirmed (matrix:
    # is_a_from_key_terms <-> defined_by_from_first_mention agree).
    assert {e["edge_status"] for e in edges} == {"confirmed"}

    # Sibling report written next to the graph.
    report_path = graph_path.parent / "edge_consensus_report.json"
    assert report_path.is_file(), "edge_consensus_report.json not written"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["total_edges"] == 2
    assert report["summary"]["confirmed_count"] == 2
    assert report["run_id"] == WORKFLOW_ID

    # The run_workflow return payload surfaces the report path.
    assert out["edge_consensus_report_paths"] == [str(report_path)]


# --------------------------------------------------------------------------
# (2) BOTH layouts present -> each stamped, each gets a sibling report
# --------------------------------------------------------------------------


def test_both_layouts_each_stamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    course_dir = tmp_path / "libv2" / "courses" / "test-course"
    graph_a = _write_graph(course_dir, "graph", _confirming_pair())
    graph_b = _write_graph(course_dir, "concept_graph", _confirming_pair())

    out = _run_with_archival(tmp_path, monkeypatch, course_dir=course_dir)

    assert out["status"] == "COMPLETE"
    for graph_path in (graph_a, graph_b):
        for edge in _load_edges(graph_path):
            assert edge.get("edge_status") is not None, (graph_path, edge)
        assert (graph_path.parent / "edge_consensus_report.json").is_file()

    assert sorted(out["edge_consensus_report_paths"]) == sorted(
        [
            str(graph_a.parent / "edge_consensus_report.json"),
            str(graph_b.parent / "edge_consensus_report.json"),
        ]
    )


# --------------------------------------------------------------------------
# (3) already-stamped graph + existing report -> byte-untouched no-op
# --------------------------------------------------------------------------


def test_already_stamped_graph_with_report_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    course_dir = tmp_path / "libv2" / "courses" / "test-course"
    # Simulate the authoring-time wiring's output: every edge already
    # carries edge_status + consensus_signals.
    stamped_edges = []
    for edge in _confirming_pair():
        edge["edge_status"] = "confirmed"
        edge["consensus_signals"] = [
            {"other_rule": "sentinel_rule", "signal": "agree"}
        ]
        stamped_edges.append(edge)
    graph_path = _write_graph(course_dir, "concept_graph", stamped_edges)
    graph_bytes_before = graph_path.read_bytes()

    # Pre-existing sibling report with sentinel content — must survive.
    report_path = graph_path.parent / "edge_consensus_report.json"
    report_path.write_text(
        json.dumps({"sentinel": True}), encoding="utf-8"
    )
    report_bytes_before = report_path.read_bytes()

    out = _run_with_archival(tmp_path, monkeypatch, course_dir=course_dir)

    assert out["status"] == "COMPLETE"
    assert graph_path.read_bytes() == graph_bytes_before, (
        "already-stamped graph was rewritten"
    )
    assert report_path.read_bytes() == report_bytes_before, (
        "pre-existing edge_consensus_report.json was clobbered"
    )
    # The skipped-but-present report still surfaces in the payload.
    assert out["edge_consensus_report_paths"] == [str(report_path)]


# --------------------------------------------------------------------------
# (4) stamped graph but MISSING report -> report written, graph untouched
# --------------------------------------------------------------------------


def test_stamped_graph_missing_report_writes_report_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    course_dir = tmp_path / "libv2" / "courses" / "test-course"
    stamped_edges = []
    for edge in _confirming_pair():
        edge["edge_status"] = "confirmed"
        edge["consensus_signals"] = []
        stamped_edges.append(edge)
    graph_path = _write_graph(course_dir, "graph", stamped_edges)
    graph_bytes_before = graph_path.read_bytes()

    out = _run_with_archival(tmp_path, monkeypatch, course_dir=course_dir)

    assert out["status"] == "COMPLETE"
    assert graph_path.read_bytes() == graph_bytes_before, (
        "stamped graph was rewritten while only the report was missing"
    )
    report_path = graph_path.parent / "edge_consensus_report.json"
    assert report_path.is_file()
    assert out["edge_consensus_report_paths"] == [str(report_path)]


# --------------------------------------------------------------------------
# (5) aggregator failure -> warning logged, final_status NOT flipped
# --------------------------------------------------------------------------


def test_aggregator_failure_does_not_flip_final_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    course_dir = tmp_path / "libv2" / "courses" / "test-course"
    graph_path = _write_graph(course_dir, "graph", _confirming_pair())

    # The helper imports EdgeConsensusAggregator at call time from
    # lib.aggregators.edge_consensus, so patching the module attribute
    # makes instantiation raise inside the aggregator path.
    import lib.aggregators.edge_consensus as ec_mod

    def _boom(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("synthetic edge-consensus failure")

    monkeypatch.setattr(ec_mod, "EdgeConsensusAggregator", _boom)

    with caplog.at_level("WARNING", logger="MCP.core.workflow_runner"):
        out = _run_with_archival(
            tmp_path, monkeypatch, course_dir=course_dir
        )

    # Best-effort contract: workflow still completes.
    assert out["status"] == "COMPLETE"
    assert out["edge_consensus_report_paths"] is None
    # The failure surfaced as a warning, not silence.
    assert any(
        "edge_consensus" in rec.getMessage()
        and "non-fatal" in rec.getMessage()
        for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]
    # Graph left as-is (no partial stamp, no report).
    for edge in _load_edges(graph_path):
        assert "edge_status" not in edge
    assert not (graph_path.parent / "edge_consensus_report.json").exists()


# --------------------------------------------------------------------------
# (6) direct-helper: concept_extraction.concept_graph_path fallback
# --------------------------------------------------------------------------


def test_helper_resolves_concept_extraction_graph_path(
    tmp_path: Path,
) -> None:
    """Partial run (no libv2_archival output): the helper falls back to
    ``phase_outputs.concept_extraction.concept_graph_path``."""
    course_dir = tmp_path / "libv2" / "courses" / "test-course"
    graph_path = _write_graph(course_dir, "concept_graph", _confirming_pair())

    runner = WorkflowRunner(executor=MagicMock(), config=MagicMock())
    written = runner._maybe_write_edge_consensus_reports(
        workflow_id=WORKFLOW_ID,
        workflow_params={"course_name": "TEST_COURSE"},
        phase_outputs={
            "concept_extraction": {
                "concept_graph_path": str(graph_path),
                "course_slug": "test-course",
            }
        },
    )

    assert written == [graph_path.parent / "edge_consensus_report.json"]
    for edge in _load_edges(graph_path):
        assert edge.get("edge_status") is not None, edge
    report = json.loads(written[0].read_text(encoding="utf-8"))
    assert report["course_slug"] == "test-course"


# --------------------------------------------------------------------------
# (7) direct-helper: nothing resolvable -> clean empty no-op
# --------------------------------------------------------------------------


def test_helper_noop_when_nothing_resolvable(tmp_path: Path) -> None:
    runner = WorkflowRunner(executor=MagicMock(), config=MagicMock())
    written = runner._maybe_write_edge_consensus_reports(
        workflow_id=WORKFLOW_ID,
        workflow_params={},
        phase_outputs={},
    )
    assert written == []
