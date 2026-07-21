"""Regression — `_run_concept_extraction` stamps edge-consensus on the graph.

Locks the GPT-fb-12-may item-2 contract at the AUTHORING point: the
``concept_extraction`` workflow phase
(``MCP/tools/pipeline_tools.py::_run_concept_extraction``) must run
``lib.aggregators.edge_consensus.EdgeConsensusAggregator.apply_to_graph``
over the freshly-built semantic graph BEFORE serialization, and write the
sibling ``edge_consensus_report.json``.

Why this exists: pre-fix, the EdgeConsensusAggregator only ran post-loop in
``MCP/core/workflow_runner.py`` (and only for the ``graph/`` artifact on the
process_course path). It was NEVER wired into the ``concept_extraction`` phase
that authors ``LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json``
for the textbook_to_course pipeline. The symptom on a real corpus:
~980 edges, every one with ``edge_status=None``, and no
sibling ``edge_consensus_report.json`` — the documented anti-silent-degradation
contract in root CLAUDE.md § Aggregators was silently unmet.

Contract assertions:

1. **Every edge carries a non-None ``edge_status``.** The string is one of
   the four documented consensus verdicts.
2. **Edges carry ``consensus_signals[]``.** The cross-rule matrix output is
   stamped (may be empty per edge, but the key is present).
3. **Sibling ``edge_consensus_report.json`` exists** next to the graph.
4. **The envelope surfaces ``edge_consensus_report_path``.**
5. **The report's edge_count matches the graph's edge count** — the report
   and the stamped graph agree.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


_CONCEPTS = [
    "vector-space", "linear-map", "eigenvalue", "eigenvector",
    "matrix-rank", "determinant", "basis-set", "orthogonality",
    "inner-product", "subspace", "kernel-space", "image-space",
]

_VALID_STATUSES = {"confirmed", "supported", "contradicted", "pending", "retracted"}


def _tagged_chunkset() -> List[Dict[str, Any]]:
    """A SemantiK chunkset whose chunks carry populated ``concept_tags`` so the
    co-occurrence + typed-edge build produces a non-trivial edge set.
    """
    chunks: List[Dict[str, Any]] = []
    for i in range(24):
        window = [_CONCEPTS[(i + j) % len(_CONCEPTS)] for j in range(3)]
        chunks.append({
            "id": f"linalg_chunk_{i:05d}",
            "text": (
                f"This passage explains {window[0]} in relation to "
                f"{window[1]} and {window[2]} for a linear-algebra course."
            ),
            "chunk_type": "explanation",
            "concept_tags": window,
            "learning_outcome_refs": [],
            "bloom_level": "understand",
            "difficulty": "intermediate",
            "source": {
                "module_id": f"week_{(i // 4) + 1:02d}",
                "item_path": f"week_{(i // 4) + 1:02d}/page_{i:03d}.html",
            },
        })
    return chunks


def _write_chunkset(path: Path, chunks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")


def _run(tmp_path: Path, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunks_path = tmp_path / "semantik_chunks" / "chunks.jsonl"
    _write_chunkset(chunks_path, chunks)
    custom_libv2 = tmp_path / "libv2"

    registry = _build_tool_registry()
    tool = registry["run_concept_extraction"]
    result = asyncio.run(
        tool(
            project_id="",
            course_name="LINALG_CONSENSUS",
            staging_dir="",
            dart_chunks_path=str(chunks_path),
            libv2_root=str(custom_libv2),
            run_id="WF-TEST-CONSENSUS",
        )
    )
    payload = json.loads(result)
    graph_path = Path(payload["concept_graph_path"])
    payload["_graph"] = json.loads(graph_path.read_text(encoding="utf-8"))
    payload["_graph_path"] = graph_path
    return payload


def test_every_edge_stamped_with_status(tmp_path: Path) -> None:
    payload = _run(tmp_path, _tagged_chunkset())
    edges = payload["_graph"].get("edges") or []
    assert edges, "Fixture should produce a non-empty edge set."
    for edge in edges:
        status = edge.get("edge_status")
        assert status in _VALID_STATUSES, (
            "Every concept-extraction edge must carry a non-None "
            f"edge_status; got {status!r} on edge "
            f"{edge.get('source')}->{edge.get('target')}."
        )


def test_edges_carry_consensus_signals(tmp_path: Path) -> None:
    payload = _run(tmp_path, _tagged_chunkset())
    edges = payload["_graph"].get("edges") or []
    for edge in edges:
        assert "consensus_signals" in edge, (
            "apply_to_graph must stamp consensus_signals[] on every edge; "
            f"missing on {edge.get('source')}->{edge.get('target')}."
        )
        assert isinstance(edge["consensus_signals"], list)


def test_sibling_report_written_and_surfaced(tmp_path: Path) -> None:
    payload = _run(tmp_path, _tagged_chunkset())
    report_path = payload["_graph_path"].parent / "edge_consensus_report.json"
    assert report_path.is_file(), (
        "edge_consensus_report.json must be written next to "
        "concept_graph_semantic.json."
    )
    assert payload.get("edge_consensus_report_path") == str(report_path), (
        "The phase envelope must surface edge_consensus_report_path."
    )


def test_report_edge_count_matches_graph(tmp_path: Path) -> None:
    payload = _run(tmp_path, _tagged_chunkset())
    report_path = payload["_graph_path"].parent / "edge_consensus_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    graph_edges = len(payload["_graph"].get("edges") or [])
    # The report's corpus-wide rollup must account for every graph edge
    # (confirmed + supported + contradicted + pending + retracted == total
    # edges). The 'supported' tier (v2 cross-stratum redesign) is a 5th
    # disjoint status that the co-occurrence fixture exercises (co-located
    # related-to edges), so it must be in the accounting sum.
    summary = report.get("summary") or report
    counted = (
        int(summary.get("confirmed_count", 0))
        + int(summary.get("supported_count", 0))
        + int(summary.get("contradicted_count", 0))
        + int(summary.get("pending_count", 0))
        + int(summary.get("retracted_count", 0))
    )
    assert counted == graph_edges, (
        f"Consensus report accounted for {counted} edges but the graph "
        f"has {graph_edges}."
    )
