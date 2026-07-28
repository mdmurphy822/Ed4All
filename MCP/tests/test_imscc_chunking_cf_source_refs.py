"""Courseforge source IDs survive the IMSCC chunking boundary."""

from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path

from MCP.tools import pipeline_tools
from MCP.tools.pipeline_tools import (
    _build_tool_registry,
    _canonical_source_id_references,
)


def test_canonical_source_id_references_preserves_and_deduplicates():
    refs = _canonical_source_id_references([
        " semantik:chapter-one#s12 ",
        "semantik:chapter-one#s12",
        "not-a-canonical-source-id",
        "",
        "semantik:chapter-one#s13",
    ])

    assert refs == [
        {
            "sourceId": "semantik:chapter-one#s12",
            "role": "primary",
            "extractor": "synthesized",
        },
        {
            "sourceId": "semantik:chapter-one#s13",
            "role": "primary",
            "extractor": "synthesized",
        },
    ]


def test_imscc_chunks_preserve_data_cf_source_ids(tmp_path, monkeypatch):
    libv2_root = tmp_path / "LibV2"
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )

    source_ids = [
        "semantik:chapter-one#s12",
        "semantik:chapter-one#s13",
    ]
    padding = " ".join(
        ["Grounded instructional content remains attributable."] * 90
    )
    html = (
        "<!doctype html><html><head><title>Grounded Page</title></head>"
        '<body><main><section data-cf-source-ids="'
        + ",".join(source_ids)
        + '"><h1>Grounded Page</h1><p>'
        + padding
        + "</p></section></main></body></html>"
    )
    imscc_path = tmp_path / "grounded.imscc"
    with zipfile.ZipFile(
        imscc_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(
            "imsmanifest.xml",
            '<?xml version="1.0"?><manifest></manifest>',
        )
        archive.writestr("html/grounded.html", html)

    tool = _build_tool_registry()["run_imscc_chunking"]
    result = json.loads(asyncio.run(tool(
        course_name="GROUNDING_DEMO",
        imscc_path=str(imscc_path),
    )))
    assert result["success"] is True

    chunks_path = Path(result["imscc_chunks_path"])
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert chunks
    expected = set(source_ids)
    for chunk in chunks:
        emitted = {
            ref["sourceId"]
            for ref in chunk["source"].get("source_references", [])
        }
        assert emitted == expected
