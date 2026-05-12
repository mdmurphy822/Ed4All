"""archive_to_libv2 mirrors lineage fields onto course_manifest.json.

GPT Feedback (May 12) item 3: the LibV2 manifest carries four new optional
top-level fields:

- ``source_documents_sha256_index`` (deduped list of upstream-source SHAs)
- ``graph_build_hash`` (mirrored from concept_graph_semantic.json)
- ``rulepack_version`` (mirrored)
- ``course_package_version`` (mirrored; null today)

These are best-effort: missing graph / pre-May-2026 manifests still
validate. New archives carry them when the graph is present on disk.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import register_pipeline_tools  # noqa: E402


class _CapturingMCP:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def tool(self):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return _decorator


@pytest.fixture
def archive_tool(monkeypatch, tmp_path):
    """Return @mcp.tool archive_to_libv2 rooted at tmp_path."""
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )

    mcp = _CapturingMCP()
    register_pipeline_tools(mcp)
    return mcp.tools["archive_to_libv2"]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_archive_emits_source_documents_sha256_index(archive_tool, tmp_path):
    """When a PDF is archived, its checksum lands in the lineage index."""
    pdf = tmp_path / "input.pdf"
    pdf_bytes = b"%PDF-1.4 some bytes" * 20
    pdf.write_bytes(pdf_bytes)
    expected_sha = _sha256_bytes(pdf_bytes)

    result_str = asyncio.run(archive_tool(
        course_name="LINEAGE_TEST_1",
        domain="test",
        pdf_paths=str(pdf),
    ))
    result = json.loads(result_str)
    assert "manifest_path" in result, f"archive_to_libv2 errored: {result}"
    manifest = json.loads(Path(result["manifest_path"]).read_text())

    assert "source_documents_sha256_index" in manifest
    assert expected_sha in manifest["source_documents_sha256_index"]


def test_archive_mirrors_graph_lineage_fields_when_graph_present(
    archive_tool, tmp_path
):
    """When concept_graph_semantic.json exists at the LibV2 course path,
    the manifest mirrors graph_build_hash + rulepack_version +
    course_package_version onto the top level."""
    # Pre-create the course dir with a hand-crafted concept_graph_semantic.json
    # so the archival call's mirror logic has something to read.
    slug = "lineage-test-2"
    course_dir = tmp_path / "LibV2" / "courses" / slug
    (course_dir / "graph").mkdir(parents=True)
    graph = {
        "kind": "concept_semantic",
        "generated_at": "2026-05-12T00:00:00+00:00",
        "rule_versions": {"is_a_from_key_terms": 2},
        "rulepack_version": "v1a2b3c4d",
        "graph_build_hash": "a" * 64,
        "course_package_version": None,
        "nodes": [],
        "edges": [],
    }
    (course_dir / "graph" / "concept_graph_semantic.json").write_text(
        json.dumps(graph),
        encoding="utf-8",
    )

    result_str = asyncio.run(archive_tool(
        course_name="LINEAGE_TEST_2",
        domain="test",
    ))
    result = json.loads(result_str)
    assert "manifest_path" in result, f"archive_to_libv2 errored: {result}"
    manifest = json.loads(Path(result["manifest_path"]).read_text())

    assert manifest.get("graph_build_hash") == "a" * 64
    assert manifest.get("rulepack_version") == "v1a2b3c4d"
    # course_package_version present, value is null.
    assert "course_package_version" in manifest
    assert manifest["course_package_version"] is None


def test_archive_omits_lineage_mirror_when_graph_missing(archive_tool, tmp_path):
    """Without a concept_graph_semantic.json on disk, the manifest
    omits the graph-mirror fields (best-effort)."""
    result_str = asyncio.run(archive_tool(
        course_name="LINEAGE_TEST_3",
        domain="test",
    ))
    result = json.loads(result_str)
    assert "manifest_path" in result
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    # No graph on disk → no mirror.
    assert "graph_build_hash" not in manifest
    assert "rulepack_version" not in manifest
    assert "course_package_version" not in manifest
