"""Wave 19 figures-directory propagation tests.

The staging + archival tools previously only copied the HTML file,
dropping the sibling ``{stem}_figures/`` directory Wave 17 persists.
Courseforge's ``<img src>`` references to that directory then dangled.
These tests lock in the Wave 19 restoration.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Callable

import pytest


def _make_tool_capturing_mcp():
    """Build a minimal MCP shim that records every ``@mcp.tool()`` call."""

    class _ToolBox:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):  # mimics @mcp.tool() decorator
            def _wrap(fn: Callable):
                self.tools[fn.__name__] = fn
                return fn
            return _wrap

    return _ToolBox()


def _bootstrap_tools():
    """Register pipeline tools against a capture-only mcp shim."""
    from MCP.tools import pipeline_tools

    mcp = _make_tool_capturing_mcp()
    pipeline_tools.register_pipeline_tools(mcp)
    return mcp.tools


def _write_dart_bundle(base: Path, stem: str) -> tuple[Path, Path]:
    """Create a minimal DART output bundle (html + figures dir)."""
    html_path = base / f"{stem}.html"
    html_path.write_text(
        f"<html><body><p>{stem}</p></body></html>", encoding="utf-8",
    )
    figures_dir = base / f"{stem}_figures"
    figures_dir.mkdir()
    (figures_dir / "0001-ab12cd34.png").write_bytes(b"fake-png-bytes")
    (figures_dir / "0002-ef56ab78.png").write_bytes(b"another-fake-bytes")
    return html_path, figures_dir


# ---------------------------------------------------------------------------
# stage_dart_outputs copies the figures dir
# ---------------------------------------------------------------------------


def test_stage_dart_outputs_copies_figures_dir(tmp_path, monkeypatch):
    """Wave 19: ``stage_dart_outputs`` must copy ``{stem}_figures/``
    into the staging dir alongside the HTML so Courseforge's
    ``<img src>`` paths resolve to real files."""
    # Redirect COURSEFORGE_INPUTS to a tempdir under the test's control.
    src_dir = tmp_path / "dart_src"
    src_dir.mkdir()
    cf_inputs = tmp_path / "cf_inputs"
    cf_inputs.mkdir()

    from MCP.tools import pipeline_tools
    monkeypatch.setattr(pipeline_tools, "COURSEFORGE_INPUTS", cf_inputs)

    html_path, figures_dir = _write_dart_bundle(src_dir, "bates")

    tools = _bootstrap_tools()
    stage = tools["stage_dart_outputs"]
    result = asyncio.run(
        stage(
            run_id="run-1",
            dart_html_paths=str(html_path),
            course_name="Bates",
        )
    )
    result_doc = json.loads(result)
    assert result_doc["success"] is True

    staged_html = cf_inputs / "run-1" / "bates.html"
    staged_figs = cf_inputs / "run-1" / "bates_figures"
    assert staged_html.exists()
    assert staged_figs.is_dir()
    # Both images survive the copytree.
    assert (staged_figs / "0001-ab12cd34.png").exists()
    assert (staged_figs / "0002-ef56ab78.png").exists()


def test_stage_dart_outputs_missing_figures_dir_is_silent(tmp_path, monkeypatch):
    """Backward compat: bundles without a ``{stem}_figures/`` dir
    still stage successfully."""
    src_dir = tmp_path / "dart_src"
    src_dir.mkdir()
    cf_inputs = tmp_path / "cf_inputs"
    cf_inputs.mkdir()

    from MCP.tools import pipeline_tools
    monkeypatch.setattr(pipeline_tools, "COURSEFORGE_INPUTS", cf_inputs)

    html_path = src_dir / "plain.html"
    html_path.write_text("<html><body><p>x</p></body></html>", encoding="utf-8")

    tools = _bootstrap_tools()
    stage = tools["stage_dart_outputs"]
    result_doc = json.loads(
        asyncio.run(
            stage(
                run_id="run-2",
                dart_html_paths=str(html_path),
                course_name="Plain",
            )
        )
    )
    assert result_doc["success"] is True
    staged_html = cf_inputs / "run-2" / "plain.html"
    assert staged_html.exists()
    # No figures dir was ever written — copytree silently skipped.
    assert not any(p.is_dir() for p in (cf_inputs / "run-2").iterdir())


# ---------------------------------------------------------------------------
# archive_to_libv2 copies the figures dir
# ---------------------------------------------------------------------------


def test_archive_to_libv2_copies_figures_dir(tmp_path, monkeypatch):
    """Wave 19: ``archive_to_libv2`` must copy ``{stem}_figures/`` into
    ``{course}/source/html/{stem}_figures/`` when present."""
    src_dir = tmp_path / "dart_src"
    src_dir.mkdir()
    libv2_root = tmp_path / "LibV2"
    libv2_root.mkdir()

    from MCP.tools import pipeline_tools

    # Redirect PROJECT_ROOT so LibV2 lands inside the tmp dir.
    original_root = pipeline_tools.PROJECT_ROOT
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    try:
        html_path, figures_dir = _write_dart_bundle(src_dir, "textbook")

        tools = _bootstrap_tools()
        archive = tools["archive_to_libv2"]
        result_doc = json.loads(
            asyncio.run(
                archive(
                    course_name="TEST_101",
                    domain="biology",
                    html_paths=str(html_path),
                )
            )
        )
        assert result_doc.get("success") is True
        slug = result_doc["course_slug"]
        dest = (
            tmp_path / "LibV2" / "courses" / slug / "source" / "html"
            / "textbook_figures"
        )
        assert dest.is_dir()
        assert (dest / "0001-ab12cd34.png").exists()
    finally:
        monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", original_root)


def test_archive_to_libv2_missing_figures_dir_is_silent(tmp_path, monkeypatch):
    """HTML-only archival (no figures dir) still succeeds."""
    src_dir = tmp_path / "dart_src"
    src_dir.mkdir()
    from MCP.tools import pipeline_tools

    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)

    html_path = src_dir / "plain.html"
    html_path.write_text("<html><body>x</body></html>", encoding="utf-8")

    tools = _bootstrap_tools()
    archive = tools["archive_to_libv2"]
    result_doc = json.loads(
        asyncio.run(
            archive(
                course_name="PLAIN_101",
                domain="generic",
                html_paths=str(html_path),
            )
        )
    )
    assert result_doc.get("success") is True
    slug = result_doc["course_slug"]
    figures_path = (
        tmp_path / "LibV2" / "courses" / slug / "source" / "html"
        / "plain_figures"
    )
    assert not figures_path.exists()


# ---------------------------------------------------------------------------
# Registry wrapper also copies the figures dir (pipeline-dispatch parity)
# ---------------------------------------------------------------------------


def test_registry_stage_dart_outputs_copies_figures_dir(tmp_path, monkeypatch):
    """The pipeline-dispatch registry variant of ``stage_dart_outputs``
    must behave identically to the MCP-tool variant (Wave 8 audit
    already enforced parity; Wave 19 extends it to the figures dir)."""
    from MCP.tools import pipeline_tools

    src_dir = tmp_path / "dart_src"
    src_dir.mkdir()
    cf_inputs = tmp_path / "cf_inputs"
    cf_inputs.mkdir()
    monkeypatch.setattr(pipeline_tools, "COURSEFORGE_INPUTS", cf_inputs)

    html_path, figures_dir = _write_dart_bundle(src_dir, "rich")

    registry = pipeline_tools._build_tool_registry()
    stage = registry["stage_dart_outputs"]
    result_doc = json.loads(
        asyncio.run(
            stage(
                run_id="run-3",
                dart_html_paths=str(html_path),
                course_name="RichDoc",
            )
        )
    )
    assert result_doc["success"] is True
    staged_figs = cf_inputs / "run-3" / "rich_figures"
    assert staged_figs.is_dir()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
