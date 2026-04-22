"""Wave 10 — archive_to_libv2 features.source_provenance manifest flag.

Contract: when ``MCP/tools/pipeline_tools.py::archive_to_libv2`` emits the
LibV2 manifest, it scans the archived corpus's ``chunks.jsonl`` for chunks
carrying ``source.source_references[]``. If at least one chunk has the
field populated, ``manifest.features.source_provenance = true``. Otherwise
(missing file, malformed lines, no refs anywhere) it's ``false``.

This advisory flag lets LibV2 retrieval callers fast-skip source-grounded
queries on pre-Wave-9 / pre-Wave-10 corpora.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Callable, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import (  # noqa: E402
    _detect_source_provenance,
    register_pipeline_tools,
)


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
    """Return archive_to_libv2 coroutine, rooted at tmp_path for LibV2."""
    # Redirect the libv2 write target by monkey-patching PROJECT_ROOT.
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    # Also redirect the Courseforge staging dir so _ensure_directories
    # doesn't splat into the repo.
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )

    mcp = _CapturingMCP()
    register_pipeline_tools(mcp)
    return mcp.tools["archive_to_libv2"]


# --------------------------------------------------------------------- #
# Unit tests on _detect_source_provenance
# --------------------------------------------------------------------- #


def test_detect_returns_false_on_missing_chunks_file(tmp_path):
    course_dir = tmp_path / "courses" / "slug"
    (course_dir / "corpus").mkdir(parents=True)
    # No chunks.jsonl
    assert _detect_source_provenance(course_dir) is False


def test_detect_returns_false_on_legacy_chunks(tmp_path):
    """Chunks without source_references → false."""
    course_dir = tmp_path / "courses" / "slug"
    corpus = course_dir / "corpus"
    corpus.mkdir(parents=True)
    chunks_file = corpus / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        # Legacy chunk — no source.source_references.
        f.write(json.dumps({
            "id": "c_00001",
            "source": {"course_id": "A", "module_id": "m", "lesson_id": "l"},
        }) + "\n")
    assert _detect_source_provenance(course_dir) is False


def test_detect_returns_true_when_chunk_carries_refs(tmp_path):
    course_dir = tmp_path / "courses" / "slug"
    corpus = course_dir / "corpus"
    corpus.mkdir(parents=True)
    chunks_file = corpus / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": "c_00001",
            "source": {
                "course_id": "A",
                "module_id": "m",
                "lesson_id": "l",
                "source_references": [
                    {"sourceId": "dart:a#s0_p0", "role": "primary"},
                ],
            },
        }) + "\n")
    assert _detect_source_provenance(course_dir) is True


def test_detect_returns_false_on_empty_source_references_array(tmp_path):
    """Empty source_references array does NOT count as populated."""
    course_dir = tmp_path / "courses" / "slug"
    corpus = course_dir / "corpus"
    corpus.mkdir(parents=True)
    chunks_file = corpus / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": "c_00001",
            "source": {
                "course_id": "A",
                "module_id": "m",
                "lesson_id": "l",
                "source_references": [],
            },
        }) + "\n")
    assert _detect_source_provenance(course_dir) is False


def test_detect_skips_malformed_lines_safely(tmp_path):
    """Malformed JSONL lines don't crash the detector."""
    course_dir = tmp_path / "courses" / "slug"
    corpus = course_dir / "corpus"
    corpus.mkdir(parents=True)
    chunks_file = corpus / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write("not valid json\n")
        f.write("{invalid\n")
        f.write("\n")
        f.write(json.dumps({
            "id": "c_00001",
            "source": {
                "course_id": "A",
                "module_id": "m",
                "lesson_id": "l",
                "source_references": [
                    {"sourceId": "dart:a#s0_p0", "role": "primary"},
                ],
            },
        }) + "\n")
    # Should still find the valid chunk carrying refs.
    assert _detect_source_provenance(course_dir) is True


def test_detect_returns_false_on_non_dict_jsonl(tmp_path):
    """A chunk line that's a JSON array (not dict) is ignored gracefully."""
    course_dir = tmp_path / "courses" / "slug"
    corpus = course_dir / "corpus"
    corpus.mkdir(parents=True)
    chunks_file = corpus / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write(json.dumps([1, 2, 3]) + "\n")
        f.write(json.dumps({"id": "c1", "source": "not-a-dict"}) + "\n")
    assert _detect_source_provenance(course_dir) is False


def test_detect_mixed_legacy_and_wave10_chunks_returns_true(tmp_path):
    """Even a single Wave-10 chunk in a mostly-legacy file → true."""
    course_dir = tmp_path / "courses" / "slug"
    corpus = course_dir / "corpus"
    corpus.mkdir(parents=True)
    chunks_file = corpus / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        # 5 legacy chunks
        for i in range(5):
            f.write(json.dumps({
                "id": f"c_{i}",
                "source": {
                    "course_id": "A", "module_id": "m", "lesson_id": "l",
                },
            }) + "\n")
        # 1 Wave-10 chunk
        f.write(json.dumps({
            "id": "c_99",
            "source": {
                "course_id": "A",
                "module_id": "m",
                "lesson_id": "l",
                "source_references": [
                    {"sourceId": "dart:a#s0_p0", "role": "contributing"},
                ],
            },
        }) + "\n")
    assert _detect_source_provenance(course_dir) is True


# --------------------------------------------------------------------- #
# End-to-end archive_to_libv2: manifest carries the flag
# --------------------------------------------------------------------- #


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_archive_manifest_omits_flag_when_no_corpus(archive_tool, tmp_path):
    """No chunks file (no assessment_path) → features.source_provenance=false."""
    result_str = asyncio.run(archive_tool(
        course_name="TEST_101",
        domain="test-domain",
    ))
    result = json.loads(result_str)
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    assert "features" in manifest
    assert manifest["features"]["source_provenance"] is False


def test_archive_manifest_flag_true_when_chunks_carry_refs(
    archive_tool, tmp_path
):
    """Copy a chunks.jsonl with Wave-10 refs → features flag is true."""
    # Build a fake corpus dir with chunks.jsonl.
    fake_corpus = tmp_path / "src_corpus"
    fake_corpus.mkdir()
    chunks_path = fake_corpus / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": "c_00001",
            "source": {
                "course_id": "TEST_101",
                "module_id": "m",
                "lesson_id": "l",
                "source_references": [
                    {"sourceId": "dart:a#s0_p0", "role": "primary"},
                ],
            },
        }) + "\n")

    result_str = asyncio.run(archive_tool(
        course_name="TEST_101",
        domain="test-domain",
        assessment_path=str(chunks_path),
    ))
    result = json.loads(result_str)
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["features"]["source_provenance"] is True


def test_archive_manifest_flag_false_on_legacy_chunks_corpus(
    archive_tool, tmp_path
):
    """Copy a chunks.jsonl with legacy chunks only → flag false."""
    fake_corpus = tmp_path / "src_corpus"
    fake_corpus.mkdir()
    chunks_path = fake_corpus / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": "c_00001",
            "source": {
                "course_id": "TEST_101",
                "module_id": "m",
                "lesson_id": "l",
            },
        }) + "\n")

    result_str = asyncio.run(archive_tool(
        course_name="TEST_101",
        domain="test-domain",
        assessment_path=str(chunks_path),
    ))
    result = json.loads(result_str)
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["features"]["source_provenance"] is False


def test_archive_features_field_is_dict(archive_tool, tmp_path):
    """Ensure the manifest structure is a dict under features (not scalar)."""
    result_str = asyncio.run(archive_tool(
        course_name="TEST_102",
        domain="test-domain",
    ))
    result = json.loads(result_str)
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert isinstance(manifest["features"], dict)
    assert "source_provenance" in manifest["features"]
