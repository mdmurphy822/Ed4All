"""Wave 28e — runtime DeprecationWarnings on @mcp.tool() legacy surfaces.

Three tools carry Wave-7 / Wave-24 deprecation status but historically
emitted no runtime warning on invocation:

1. ``create_textbook_pipeline_tool`` (Wave 7 deprecated)
2. ``run_textbook_pipeline_tool`` (Wave 7 deprecated)
3. ``create_course_project`` (Wave 28e documented deprecation — the
   tool remains functional for external MCP clients but new
   integrations should route through ``extract_textbook_structure``
   + ``plan_course_structure`` per Wave 24).

Wave 28e adds ``warnings.warn(DeprecationWarning, ...)`` at the top
of each so the MCP surface matches the CLI ``ed4all
textbook-to-course`` stderr deprecation notice. Warnings are
non-blocking — the call site must still succeed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import warnings
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import courseforge_tools, pipeline_tools  # noqa: E402


class _MCPStub:
    def __init__(self):
        self.registered = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.registered[fn.__name__] = fn
            return fn
        return decorator


def _collect_pipeline_tools():
    mcp = _MCPStub()
    pipeline_tools.register_pipeline_tools(mcp)
    return mcp.registered


def _collect_courseforge_tools():
    mcp = _MCPStub()
    courseforge_tools.register_courseforge_tools(mcp)
    return mcp.registered


class TestCreateTextbookPipelineToolDeprecation:
    def test_emits_deprecation_warning(self, monkeypatch):
        """create_textbook_pipeline_tool fires DeprecationWarning."""
        tools = _collect_pipeline_tools()
        fn = tools["create_textbook_pipeline_tool"]

        # Stub out the heavy downstream call so we only measure the
        # warning emission, not the full pipeline wiring.
        async def _stub(*args, **kwargs):
            return json.dumps({"success": True, "stubbed": True})

        monkeypatch.setattr(pipeline_tools, "create_textbook_pipeline", _stub)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = asyncio.run(fn(
                pdf_paths="/tmp/fake.pdf",
                course_name="TEST_101",
            ))

        payload = json.loads(result)
        assert payload.get("success") is True  # non-blocking warning
        dep_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert dep_warnings, (
            "create_textbook_pipeline_tool should emit DeprecationWarning"
        )
        msg = str(dep_warnings[0].message)
        assert "create_textbook_pipeline_tool" in msg
        assert "Wave 7" in msg
        assert (
            "ed4all run textbook-to-course" in msg
            or "create_workflow" in msg
        )


class TestRunTextbookPipelineToolDeprecation:
    def test_emits_deprecation_warning(self, monkeypatch):
        """run_textbook_pipeline_tool fires DeprecationWarning."""
        tools = _collect_pipeline_tools()
        fn = tools["run_textbook_pipeline_tool"]

        async def _stub(workflow_id):
            return json.dumps({"success": True, "workflow_id": workflow_id})

        monkeypatch.setattr(pipeline_tools, "run_textbook_pipeline", _stub)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = asyncio.run(fn(workflow_id="WF-TEST"))

        payload = json.loads(result)
        assert payload.get("success") is True
        dep_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert dep_warnings, (
            "run_textbook_pipeline_tool should emit DeprecationWarning"
        )
        msg = str(dep_warnings[0].message)
        assert "run_textbook_pipeline_tool" in msg
        assert "Wave 7" in msg


class TestCreateCourseProjectDeprecation:
    def test_emits_deprecation_warning(self, monkeypatch, tmp_path):
        """create_course_project fires DeprecationWarning pointing at Wave 24 replacements."""
        exports_root = tmp_path / "Courseforge" / "exports"
        exports_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(courseforge_tools, "EXPORTS_PATH", exports_root)

        tools = _collect_courseforge_tools()
        fn = tools["create_course_project"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = asyncio.run(fn(
                course_name="TEST_101",
                objectives_path="/tmp/fake_objectives.json",
            ))

        payload = json.loads(result)
        # Non-blocking — call site still succeeds.
        assert payload.get("success") is True, payload

        dep_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert dep_warnings, (
            "create_course_project should emit DeprecationWarning"
        )
        msg = str(dep_warnings[0].message)
        assert "create_course_project" in msg
        # Warning cites the Wave 24 replacements.
        assert "extract_textbook_structure" in msg
        assert "plan_course_structure" in msg

    def test_schema_description_marks_deprecated(self):
        """TOOL_SCHEMAS entry description is prefixed with the deprecation notice."""
        from MCP.core.tool_schemas import TOOL_SCHEMAS

        desc = TOOL_SCHEMAS["create_course_project"]["description"]
        assert desc.startswith("[DEPRECATED"), desc
        assert "extract_textbook_structure" in desc
        assert "plan_course_structure" in desc
