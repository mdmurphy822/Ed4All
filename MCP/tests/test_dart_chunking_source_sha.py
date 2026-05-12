"""``_run_dart_chunking`` stamps ``source.source_document_sha256``.

GPT Feedback (May 12) item 3: every chunk emitted by the canonical DART
chunker carries the aggregate ``source_dart_html_sha256`` from the
sidecar manifest in its ``source.source_document_sha256`` field. Lets
downstream consumers join chunk → upstream-source bytes without a
sidecar lookup.
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
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


@pytest.fixture
def dart_chunking_tool(monkeypatch, tmp_path):
    """Return the _run_dart_chunking registry entry rooted at tmp_path.

    Sets both ``ED4ALL_LIBV2_ROOT`` (consumed by ``_resolve_libv2_root``)
    and monkeypatches the module-level paths so the chunker emits its
    artifacts under the test's tmp_path rather than the real LibV2 tree.
    """
    libv2_root = tmp_path / "LibV2"
    libv2_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )

    registry = _build_tool_registry()
    return registry["run_dart_chunking"]


def _write_simple_html(path: Path, title: str = "Intro") -> None:
    """Write a minimally-parseable HTML doc the HTMLContentParser accepts."""
    path.write_text(
        "<!doctype html><html><head><title>"
        + title
        + "</title></head><body>"
        + "<h1>" + title + "</h1>"
        + "<section><h2>Section A</h2>"
        + "<p>The mitochondrion is the powerhouse of the cell. "
        + "Cells use mitochondria to produce ATP through oxidative "
        + "phosphorylation, which is the primary mechanism by which "
        + "cells generate energy for metabolic activity.</p></section>"
        + "</body></html>",
        encoding="utf-8",
    )


def test_dart_chunks_carry_source_document_sha256(dart_chunking_tool, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_simple_html(staging / "lesson_01.html", title="Lesson 1")
    _write_simple_html(staging / "lesson_02.html", title="Lesson 2")

    result_str = asyncio.run(dart_chunking_tool(
        course_name="DART_SHA_TEST",
        staging_dir=str(staging),
    ))
    result = json.loads(result_str)
    assert result.get("success"), f"chunking errored: {result}"

    chunks_path = Path(result["dart_chunks_path"])
    assert chunks_path.exists(), f"no chunks at {chunks_path}"

    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = manifest["source_dart_html_sha256"]
    assert len(expected_sha) == 64

    # Every emitted chunk should carry source.source_document_sha256 ==
    # source_dart_html_sha256 (the aggregate of the staged HTML inputs).
    with chunks_path.open("r", encoding="utf-8") as fh:
        chunks = [json.loads(line) for line in fh if line.strip()]
    assert chunks, "expected at least one chunk emitted"
    for chunk in chunks:
        sha = chunk.get("source", {}).get("source_document_sha256")
        assert sha == expected_sha, (
            f"chunk {chunk.get('id')!r} has source_document_sha256={sha!r}; "
            f"expected manifest's source_dart_html_sha256={expected_sha!r}"
        )
