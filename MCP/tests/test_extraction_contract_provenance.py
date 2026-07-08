"""``extraction_contract`` provenance marker on freshly-emitted chunksets.

The chunk-TEXT extraction contract (``HTMLTextExtractor`` sr-scaffolding
suppression + structural delimiters) is versioned as
``Trainforge.chunker.EXTRACTION_TEXT_CONTRACT_VERSION`` and stamped as the
integer ``extraction_contract`` field on every chunkset sidecar manifest (and
``course_manifest.json``). This is orthogonal to ``chunker_version`` (which
stays ``"v4"`` across extraction-text changes) so a provenance consumer — the
tier-2 citation-anchoring floor gate — can tell a chunk-only ``--stop-after
chunking`` slice was produced under the CURRENT extraction-text contract even
though it never gets a course manifest.

Covers:
1. a freshly emitted dart chunkset manifest carries
   ``extraction_contract == EXTRACTION_TEXT_CONTRACT_VERSION``;
2. that manifest still validates against ``chunkset_manifest.schema.json``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402
from Trainforge.chunker import EXTRACTION_TEXT_CONTRACT_VERSION  # noqa: E402

_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "library" / "chunkset_manifest.schema.json"
)


@pytest.fixture
def dart_chunking_tool(monkeypatch, tmp_path):
    """Return the _run_dart_chunking registry entry rooted at tmp_path."""
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


def _emit_manifest(dart_chunking_tool, tmp_path) -> dict:
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_simple_html(staging / "lesson_01.html", title="Lesson 1")
    result = json.loads(
        asyncio.run(
            dart_chunking_tool(
                course_name="EXTRACTION_CONTRACT_TEST",
                staging_dir=str(staging),
            )
        )
    )
    assert result.get("success"), f"chunking errored: {result}"
    manifest_path = Path(result["manifest_path"])
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_fresh_dart_manifest_carries_extraction_contract(
    dart_chunking_tool, tmp_path
):
    manifest = _emit_manifest(dart_chunking_tool, tmp_path)
    assert manifest.get("extraction_contract") == EXTRACTION_TEXT_CONTRACT_VERSION, (
        "a freshly emitted dart chunkset manifest must stamp the CURRENT "
        f"extraction_contract={EXTRACTION_TEXT_CONTRACT_VERSION!r}; got "
        f"{manifest.get('extraction_contract')!r}"
    )
    # chunker_version stays coarse ("v4") — the whole reason the finer int
    # marker exists.
    assert manifest.get("chunker_version") == "v4"


def test_fresh_dart_manifest_validates_against_schema(
    dart_chunking_tool, tmp_path
):
    jsonschema = pytest.importorskip("jsonschema")
    manifest = _emit_manifest(dart_chunking_tool, tmp_path)
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    # additionalProperties:false on the chunkset schema means the new
    # extraction_contract field MUST be declared or this raises.
    jsonschema.validate(instance=manifest, schema=schema)
    assert "extraction_contract" in manifest
