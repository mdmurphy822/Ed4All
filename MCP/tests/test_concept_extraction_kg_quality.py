"""Regression — `_run_concept_extraction` stamps kg_quality on the graph.

Locks the NVIDIA-KG item-2 contract at the AUTHORING point: the
``concept_extraction`` workflow phase
(``MCP/tools/pipeline_tools.py::_run_concept_extraction``) must compute
the four KG-quality dimensions over the freshly-built (and
edge-consensus-stamped) semantic graph BEFORE serialization, stamp the
``kg_quality`` field on the graph, and write a sibling
``quality/kg_quality_report.json`` under the LibV2 course dir.

Why this exists: pre-fix, the KG-quality reporter only ran as a
blocking gate at ``libv2_archival`` (``lib/validators/kg_quality.py``)
and never stamped the graph's ``kg_quality`` field at authoring time.
The symptom on a real corpus: ~200 nodes / ~980 edges,
``kg_quality: null`` on the graph, and no ``quality/kg_quality_report.json``
sidecar.

Contract assertions:

1. **The graph's ``kg_quality`` is populated** with the four metric keys
   and sensible ``[0, 1]`` values.
2. **A ``quality/kg_quality_report.json`` is written** under the LibV2
   course dir, with the four dimensions.
3. **The envelope surfaces ``kg_quality_report_path``.**
4. **The consistency axis composes with the edge-consensus attenuation**
   — its score equals ``1 - contradiction_rate`` derived from the
   stamped edge_status.
5. **A KG-quality computation error degrades gracefully** — ``kg_quality``
   stays null, no sidecar, but the phase still succeeds.
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

import MCP.tools.pipeline_tools as pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


_CONCEPTS = [
    "vector-space", "linear-map", "eigenvalue", "eigenvector",
    "matrix-rank", "determinant", "basis-set", "orthogonality",
    "inner-product", "subspace", "kernel-space", "image-space",
]

_DIMENSIONS = ("completeness", "consistency", "accuracy", "coverage")


def _tagged_chunkset() -> List[Dict[str, Any]]:
    """A DART chunkset whose chunks carry populated ``concept_tags`` so the
    co-occurrence + typed-edge build produces a non-trivial node + edge set.
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
    chunks_path = tmp_path / "dart_chunks" / "chunks.jsonl"
    _write_chunkset(chunks_path, chunks)
    custom_libv2 = tmp_path / "libv2"

    registry = _build_tool_registry()
    tool = registry["run_concept_extraction"]
    result = asyncio.run(
        tool(
            project_id="",
            course_name="LINALG_KGQ",
            staging_dir="",
            dart_chunks_path=str(chunks_path),
            libv2_root=str(custom_libv2),
            run_id="WF-TEST-KGQ",
        )
    )
    payload = json.loads(result)
    graph_path = Path(payload["concept_graph_path"])
    payload["_graph"] = json.loads(graph_path.read_text(encoding="utf-8"))
    payload["_graph_path"] = graph_path
    payload["_course_dir"] = graph_path.parent.parent
    return payload


def test_graph_kg_quality_populated(tmp_path: Path) -> None:
    payload = _run(tmp_path, _tagged_chunkset())
    kgq = payload["_graph"].get("kg_quality")
    assert isinstance(kgq, dict), (
        "The concept-extraction graph must carry a populated kg_quality "
        f"dict; got {kgq!r}."
    )
    for dim in _DIMENSIONS:
        assert dim in kgq, f"kg_quality missing dimension {dim!r}."
        val = kgq[dim]
        assert isinstance(val, (int, float)), (
            f"kg_quality[{dim!r}] must be numeric; got {val!r}."
        )
        assert 0.0 <= float(val) <= 1.0, (
            f"kg_quality[{dim!r}]={val!r} must be in [0, 1]."
        )


def test_report_written_and_surfaced(tmp_path: Path) -> None:
    payload = _run(tmp_path, _tagged_chunkset())
    report_path = (
        payload["_course_dir"] / "quality" / "kg_quality_report.json"
    )
    assert report_path.is_file(), (
        "quality/kg_quality_report.json must be written under the LibV2 "
        "course dir."
    )
    assert payload.get("kg_quality_report_path") == str(report_path), (
        "The phase envelope must surface kg_quality_report_path."
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dims = report.get("dimensions") or {}
    assert set(dims.keys()) == set(_DIMENSIONS), (
        f"Report must carry all four dimensions; got {sorted(dims)}."
    )
    for dim in _DIMENSIONS:
        assert "score" in dims[dim]


def test_consistency_composes_with_consensus_attenuation(
    tmp_path: Path,
) -> None:
    payload = _run(tmp_path, _tagged_chunkset())
    graph = payload["_graph"]
    edges = graph.get("edges") or []
    stamped = [
        e for e in edges
        if isinstance(e.get("edge_status"), str) and e.get("edge_status")
    ]
    assert stamped, "Edge-consensus stamping should run before kg-quality."
    contradicted = sum(
        1 for e in stamped if e.get("edge_status") == "contradicted"
    )
    expected_contradiction_rate = contradicted / len(stamped)
    expected_consistency = round(
        max(0.0, 1.0 - expected_contradiction_rate), 4
    )
    actual = float(graph["kg_quality"]["consistency"])
    assert abs(actual - expected_consistency) < 1e-6, (
        "consistency must equal 1 - contradiction_rate derived from the "
        f"stamped edge_status; expected {expected_consistency}, got {actual}."
    )


def test_kg_quality_error_degrades_gracefully(
    tmp_path: Path, monkeypatch,
) -> None:
    """A KGQualityReporter failure must leave kg_quality null and NOT
    fail the phase (no kg_quality_report.json, no envelope key)."""
    import Trainforge.rag.kg_quality_report as kgq_mod

    class _BoomReporter:
        def __init__(self, *a, **k):
            raise RuntimeError("synthetic kg-quality failure")

    monkeypatch.setattr(kgq_mod, "KGQualityReporter", _BoomReporter)

    payload = _run(tmp_path, _tagged_chunkset())
    assert payload.get("success") is True, (
        "The phase must still succeed when kg-quality computation raises."
    )
    assert payload["_graph"].get("kg_quality") is None, (
        "kg_quality must stay null on a computation error."
    )
    assert "kg_quality_report_path" not in payload, (
        "No kg_quality_report_path should be surfaced on failure."
    )
    report_path = (
        payload["_course_dir"] / "quality" / "kg_quality_report.json"
    )
    assert not report_path.exists(), (
        "No quality/kg_quality_report.json should be written on failure."
    )
