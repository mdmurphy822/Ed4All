"""Wave4-Anew4 / F6 — _archive_to_libv2 registers course in catalog.

Contract: after ``_archive_to_libv2`` (registry variant) and
``archive_to_libv2`` (@mcp.tool() variant) finish writing
``manifest.json`` they must update
``LibV2/catalog/master_catalog.json`` (and ``course_index.json``) so
that ``libv2 catalog list`` / ``libv2 info <slug>`` see the course
without a manual ``index rebuild`` step.

Tests:
  - fresh LibV2 root with no pre-existing catalog → catalog is created
    with one entry.
  - re-archival of the same slug → exactly one entry remains (idempotent).
  - second slug → both entries present.
  - course_index.json kept in sync alongside master_catalog.json.
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
    _build_tool_registry,
    register_pipeline_tools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CapturingMCP:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def tool(self):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return _decorator


def _catalog(tmp_path: Path) -> dict:
    path = tmp_path / "LibV2" / "catalog" / "master_catalog.json"
    return json.loads(path.read_text())


def _index(tmp_path: Path) -> dict:
    path = tmp_path / "LibV2" / "catalog" / "course_index.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Tests — registry (workflow-phase) variant
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRegistryVariantCatalogEntry:
    """_archive_to_libv2 (registry / workflow-phase variant) registers courses."""

    def test_catalog_created_on_first_archive(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
        registry = _build_tool_registry()
        tool = registry["archive_to_libv2"]

        result = json.loads(asyncio.run(
            tool(course_name="PHYS_101", domain="physics", division="STEM")
        ))
        assert result.get("success"), result

        cat = _catalog(tmp_path)
        slugs = [c["slug"] for c in cat["courses"]]
        assert "phys-101" in slugs, f"Expected phys-101 in {slugs}"
        assert cat["total_courses"] == 1

    def test_catalog_entry_carries_classification(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
        registry = _build_tool_registry()
        tool = registry["archive_to_libv2"]

        asyncio.run(
            tool(course_name="CHEM_201", domain="chemistry", division="STEM",
                 subdomains="organic,inorganic")
        )
        cat = _catalog(tmp_path)
        entry = next(c for c in cat["courses"] if c["slug"] == "chem-201")
        assert entry["division"] == "STEM"
        assert entry["primary_domain"] == "chemistry"

    def test_idempotent_rearchival_single_entry(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
        registry = _build_tool_registry()
        tool = registry["archive_to_libv2"]

        asyncio.run(tool(course_name="PHYS_101", domain="physics", division="STEM"))
        asyncio.run(tool(course_name="PHYS_101", domain="physics", division="STEM"))

        cat = _catalog(tmp_path)
        matching = [c for c in cat["courses"] if c["slug"] == "phys-101"]
        assert len(matching) == 1, f"Expected 1 entry, got {len(matching)}"
        assert cat["total_courses"] == 1

    def test_two_slugs_both_present(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
        registry = _build_tool_registry()
        tool = registry["archive_to_libv2"]

        asyncio.run(tool(course_name="PHYS_101", domain="physics"))
        asyncio.run(tool(course_name="CHEM_201", domain="chemistry"))

        cat = _catalog(tmp_path)
        slugs = {c["slug"] for c in cat["courses"]}
        assert "phys-101" in slugs
        assert "chem-201" in slugs
        assert cat["total_courses"] == 2

    def test_course_index_updated(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
        registry = _build_tool_registry()
        tool = registry["archive_to_libv2"]

        asyncio.run(tool(course_name="BIO_301", domain="biology"))

        idx = _index(tmp_path)
        assert "bio-301" in idx
        assert idx["bio-301"]["path"] == "courses/bio-301"


# ---------------------------------------------------------------------------
# Tests — @mcp.tool() variant
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMcpToolVariantCatalogEntry:
    """archive_to_libv2 (@mcp.tool() variant) also registers courses."""

    def test_catalog_created_on_first_archive(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
        )
        mcp = _CapturingMCP()
        register_pipeline_tools(mcp)
        tool = mcp.tools["archive_to_libv2"]

        result = json.loads(asyncio.run(
            tool(course_name="ART_101", domain="art-history", division="ARTS")
        ))
        assert result.get("success"), result

        cat = _catalog(tmp_path)
        slugs = [c["slug"] for c in cat["courses"]]
        assert "art-101" in slugs

    def test_idempotent_rearchival_single_entry(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
        )
        mcp = _CapturingMCP()
        register_pipeline_tools(mcp)
        tool = mcp.tools["archive_to_libv2"]

        asyncio.run(tool(course_name="ART_101", domain="art-history", division="ARTS"))
        asyncio.run(tool(course_name="ART_101", domain="art-history", division="ARTS"))

        cat = _catalog(tmp_path)
        matching = [c for c in cat["courses"] if c["slug"] == "art-101"]
        assert len(matching) == 1, f"Expected 1 entry, got {len(matching)}"
