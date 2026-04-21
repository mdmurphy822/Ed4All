"""Runtime DeprecationWarnings on @mcp.tool() legacy surfaces.

Wave 28e originally covered three tools:

1. ``create_textbook_pipeline_tool`` (Wave 7 deprecated) — REMOVED in Wave 28f.
2. ``run_textbook_pipeline_tool`` (Wave 7 deprecated) — REMOVED in Wave 28f.
3. ``create_course_project`` (Wave 28e documented deprecation — the
   tool remains functional for external MCP clients but new
   integrations should route through ``extract_textbook_structure``
   + ``plan_course_structure`` per Wave 24).

Wave 28f: the two pipeline wrapper tools were deleted outright once the
grace window elapsed. Only the ``create_course_project`` deprecation
warning is pinned here now. Warnings are non-blocking — the call site
must still succeed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import courseforge_tools  # noqa: E402


class _MCPStub:
    def __init__(self):
        self.registered = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.registered[fn.__name__] = fn
            return fn
        return decorator


def _collect_courseforge_tools():
    mcp = _MCPStub()
    courseforge_tools.register_courseforge_tools(mcp)
    return mcp.registered


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
