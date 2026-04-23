"""Wave 28e — ``@mcp.tool() package_imscc`` parity with the mature packager.

Wave 27 folded the registry-side ``_package_imscc`` wrapper at
``MCP/tools/pipeline_tools.py`` onto the mature
``Courseforge.scripts.package_multifile_imscc.package_imscc`` module.
The ``@mcp.tool()`` surface in ``MCP/tools/courseforge_tools.py`` was
not included in that fold until Wave 28e — pre-Wave-28e it flipped
``project_config.status = "packaged"`` and attempted a LibV2 copy
without ever building the zip.

These tests exercise the ``@mcp.tool() package_imscc`` entry point
directly (not the registry variant) and verify:

1. A real IMSCC zip is produced at the expected path.
2. ``course_metadata.json`` is bundled at the zip root.
3. The input parameter contract is preserved (positional + kwargs).
4. LO-contract failure surfaces as a structured error envelope.

The fixtures are hermetic — no external corpora or network access.
"""

from __future__ import annotations

import asyncio
import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import courseforge_tools  # noqa: E402

COURSE_CODE = "WAVE28E_101"


class _MCPStub:
    """Minimal MCP stub that captures registered @mcp.tool() functions."""

    def __init__(self):
        self.registered = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.registered[fn.__name__] = fn
            return fn
        return decorator


def _page_html(title: str, lo_ids: list[str]) -> str:
    """Minimal page HTML with JSON-LD block for LO-contract validation."""
    lo_entries = ",".join(
        f'{{"id": "{lo_id}", "statement": "Describe {lo_id}."}}'
        for lo_id in lo_ids
    )
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        f"  <title>{title}</title>\n"
        "  <script type=\"application/ld+json\">\n"
        f"  {{\"@type\": \"CourseModule\", \"learningObjectives\": [{lo_entries}]}}\n"
        "  </script>\n"
        "</head>\n"
        "<body><main id=\"main-content\"><h1>" + title + "</h1></main></body>\n"
        "</html>\n"
    )


def _make_project(
    exports_root: Path,
    *,
    project_id: str,
    duration_weeks: int = 2,
    emit_course_metadata: bool = True,
    lo_ids_per_week: dict[int, list[str]] | None = None,
) -> tuple[Path, Path]:
    """Build a Courseforge exports workspace with 2 weeks of pages."""
    project_path = exports_root / project_id
    content_dir = project_path / "03_content_development"
    content_dir.mkdir(parents=True, exist_ok=True)
    (project_path / "05_final_package").mkdir(parents=True, exist_ok=True)

    config = {
        "project_id": project_id,
        "course_name": COURSE_CODE,
        "course_title": f"{COURSE_CODE} Sample Course",
        "duration_weeks": duration_weeks,
    }
    (project_path / "project_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    lo_ids_per_week = lo_ids_per_week or {
        1: ["TO-01", "CO-01"],
        2: ["TO-01", "CO-02"],
    }

    for week_num in range(1, duration_weeks + 1):
        week_dir = content_dir / f"week_{week_num:02d}"
        week_dir.mkdir(parents=True, exist_ok=True)
        for role in ("overview", "content_01", "summary"):
            page = week_dir / f"week_{week_num:02d}_{role}.html"
            page.write_text(
                _page_html(
                    f"Week {week_num} {role}",
                    lo_ids_per_week.get(week_num, ["TO-01"]),
                ),
                encoding="utf-8",
            )

    course_objectives = {
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Terminal objective 1."},
        ],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {"id": "CO-01", "statement": "Chapter objective 1."},
                ],
            },
            {
                "chapter": "Week 2",
                "objectives": [
                    {"id": "CO-02", "statement": "Chapter objective 2."},
                ],
            },
        ],
    }
    (content_dir / "course.json").write_text(
        json.dumps(course_objectives, indent=2), encoding="utf-8"
    )

    if emit_course_metadata:
        (content_dir / "course_metadata.json").write_text(
            json.dumps({
                "course_code": COURSE_CODE,
                "course_title": f"{COURSE_CODE} Sample Course",
                "classification": {"taxonomy": "sample"},
            }),
            encoding="utf-8",
        )

    return project_path, content_dir


@pytest.fixture
def package_imscc_tool(monkeypatch, tmp_path):
    """Register courseforge tools against a tmp exports root and return the tool fn."""
    exports_root = tmp_path / "Courseforge" / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(courseforge_tools, "EXPORTS_PATH", exports_root)

    mcp = _MCPStub()
    courseforge_tools.register_courseforge_tools(mcp)
    return mcp.registered["package_imscc"], exports_root


class TestPackageImsccMCPToolParity:
    def test_successful_package_produces_real_zip(self, package_imscc_tool):
        """Real IMSCC zip lands on disk with manifest + HTML pages."""
        tool, exports_root = package_imscc_tool
        project_id = "PROJ-28E-ZIP"
        _make_project(exports_root, project_id=project_id)

        result = asyncio.run(tool(project_id=project_id))
        payload = json.loads(result)
        assert payload.get("success") is True, payload

        package_path = Path(payload["package_path"])
        assert package_path.exists(), f"zip not created: {package_path}"
        assert package_path.stat().st_size > 0

        with zipfile.ZipFile(package_path, "r") as zf:
            names = zf.namelist()
            assert "imsmanifest.xml" in names
            assert any(n.endswith(".html") for n in names)

    def test_course_metadata_json_bundled(self, package_imscc_tool):
        """``course_metadata.json`` lands at the zip root (Wave 3 REC-TAX-01)."""
        tool, exports_root = package_imscc_tool
        project_id = "PROJ-28E-META"
        _make_project(
            exports_root,
            project_id=project_id,
            emit_course_metadata=True,
        )

        result = asyncio.run(tool(project_id=project_id))
        payload = json.loads(result)
        assert payload["success"] is True

        with zipfile.ZipFile(payload["package_path"], "r") as zf:
            names = zf.namelist()
            assert "course_metadata.json" in names
            meta = json.loads(zf.read("course_metadata.json"))
            assert meta.get("course_code") == COURSE_CODE

    def test_parameter_contract_positional_and_kwargs(self, package_imscc_tool):
        """Legacy contract preserved: positional + kwargs both work."""
        tool, exports_root = package_imscc_tool
        project_id = "PROJ-28E-CONTRACT"
        _make_project(exports_root, project_id=project_id)

        # Positional form (pre-fold signature was
        # ``package_imscc(project_id, validate=True)``).
        result = asyncio.run(tool(project_id, True))
        payload = json.loads(result)
        assert payload.get("success") is True

        # Kwargs form — same as above.
        project_id_2 = "PROJ-28E-CONTRACT-KW"
        _make_project(exports_root, project_id=project_id_2)
        result_kw = asyncio.run(
            tool(project_id=project_id_2, validate=False)
        )
        payload_kw = json.loads(result_kw)
        assert payload_kw.get("success") is True

        # Response envelope keys match the Wave 27 registry variant.
        required_keys = {
            "success",
            "project_id",
            "package_path",
            "libv2_package_path",
            "html_modules",
            "package_size_bytes",
        }
        assert required_keys.issubset(payload.keys()), payload
        assert required_keys.issubset(payload_kw.keys()), payload_kw

    def test_lo_contract_failure_structured_error(self, package_imscc_tool):
        """Page with out-of-week LO => structured error, not silent pass."""
        tool, exports_root = package_imscc_tool
        project_id = "PROJ-28E-LOFAIL"
        _make_project(
            exports_root,
            project_id=project_id,
            lo_ids_per_week={
                1: ["TO-01", "CO-02"],  # CO-02 is week 2's only — violates.
                2: ["TO-01", "CO-02"],
            },
        )

        result = asyncio.run(tool(project_id=project_id))
        payload = json.loads(result)
        assert payload.get("success") is False, payload
        assert "error" in payload
        assert payload.get("exit_code") == 2
        assert payload.get("project_id") == project_id

    def test_no_html_content_returns_error(self, package_imscc_tool):
        """Missing content dir / no HTML pages => structured error envelope."""
        tool, exports_root = package_imscc_tool
        project_id = "PROJ-28E-NOHTML"
        project_path = exports_root / project_id
        (project_path / "03_content_development").mkdir(parents=True)
        (project_path / "project_config.json").write_text(
            json.dumps({"course_name": COURSE_CODE, "duration_weeks": 2})
        )

        result = asyncio.run(tool(project_id=project_id))
        payload = json.loads(result)
        assert "error" in payload
        assert "HTML" in payload["error"]
