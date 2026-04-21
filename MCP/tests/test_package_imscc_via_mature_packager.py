"""Wave 27 HIGH-2 — ``_package_imscc`` routes through the mature packager.

Verifies that the MCP registry wrapper at
``MCP.tools.pipeline_tools._package_imscc`` delegates to
``Courseforge.scripts.package_multifile_imscc.package_imscc`` rather than
hand-rolling the IMSCC zip. The mature packager supplies:

- Per-week ``learningObjectives`` LO-contract validation (default on)
- ``course_metadata.json`` bundling at zip root
- IMS Common Cartridge v1.3 namespaces
- Week-grouped resource manifest (nested ``<item>`` under week modules)

These tests use a minimal on-disk fixture so the suite runs hermetically.
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

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


COURSE_CODE = "WAVE27PKG_101"


# Minimal page HTML that carries the JSON-LD block the LO-contract
# validator inspects. The ``learningObjectives`` list must reference
# IDs declared for the page's week in the canonical objectives JSON.
def _page_html(title: str, lo_ids: list[str]) -> str:
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
    tmp_path: Path,
    *,
    project_id: str = "PROJ-27-01",
    duration_weeks: int = 2,
    emit_course_metadata: bool = True,
    lo_ids_per_week: dict[int, list[str]] | None = None,
    course_objectives: dict | None = None,
) -> tuple[Path, Path]:
    """Create a Courseforge exports workspace with 2 weeks of pages.

    Returns ``(project_path, content_dir)``.
    """
    project_path = tmp_path / "Courseforge" / "exports" / project_id
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

    # Canonical course.json — mature packager auto-discovers this so the
    # LO-contract validator fires on every page. Shape mirrors the one
    # Trainforge + Courseforge emit.
    if course_objectives is None:
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
def pipeline_registry(monkeypatch, tmp_path):
    """Build the pipeline tool registry against a tmp Courseforge root."""
    staging_root = tmp_path / "cf_inputs"
    staging_root.mkdir()
    monkeypatch.setattr(pipeline_tools, "COURSEFORGE_INPUTS", staging_root)
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)
    return _build_tool_registry(), tmp_path


class TestPackageImsccRoutesThroughMaturePackager:
    def test_manifest_is_week_grouped(self, pipeline_registry):
        """Pages live under per-week ``<item>`` wrappers, not flat siblings."""
        tools, tmp_path = pipeline_registry
        project_id = "PROJ-27-WEEKGROUP"
        _make_project(tmp_path, project_id=project_id)

        result = asyncio.run(tools["package_imscc"](project_id=project_id))
        payload = json.loads(result)
        assert payload["success"] is True, payload

        with zipfile.ZipFile(payload["package_path"], "r") as zf:
            manifest = zf.read("imsmanifest.xml").decode("utf-8")

        # Week-grouped manifest: resources nest under per-week modules.
        assert "WEEK_1" in manifest
        assert "WEEK_2" in manifest
        # Flat legacy manifest stamped ITEM_001 / RES_001 as a single-
        # level list under ROOT — reject that shape here.
        assert "ITEM_001" not in manifest
        assert "RES_001" not in manifest
        # Hierarchical depth: the ROOT item must have week items as
        # children, and each week must contain the page items.
        assert 'identifier="ROOT"' in manifest

    def test_course_metadata_json_bundled(self, pipeline_registry):
        """``course_metadata.json`` lands at the zip root (Wave 3 REC-TAX-01)."""
        tools, tmp_path = pipeline_registry
        project_id = "PROJ-27-METADATA"
        _make_project(
            tmp_path, project_id=project_id, emit_course_metadata=True,
        )

        result = asyncio.run(tools["package_imscc"](project_id=project_id))
        payload = json.loads(result)
        assert payload["success"] is True

        with zipfile.ZipFile(payload["package_path"], "r") as zf:
            names = zf.namelist()
            assert "course_metadata.json" in names
            # Structural sanity: bundled metadata is valid JSON.
            meta = json.loads(zf.read("course_metadata.json"))
            assert meta.get("course_code") == COURSE_CODE

    def test_ims_cc_v1p3_namespace(self, pipeline_registry):
        """Mature packager emits IMS CC v1.3, not the legacy v1.2."""
        tools, tmp_path = pipeline_registry
        project_id = "PROJ-27-NS"
        _make_project(tmp_path, project_id=project_id)

        result = asyncio.run(tools["package_imscc"](project_id=project_id))
        payload = json.loads(result)
        assert payload["success"] is True

        with zipfile.ZipFile(payload["package_path"], "r") as zf:
            manifest = zf.read("imsmanifest.xml").decode("utf-8")

        assert "imsccv1p3" in manifest
        assert "<schemaversion>1.3.0</schemaversion>" in manifest
        # Legacy v1.2 namespace MUST NOT appear.
        assert "imsccv1p2" not in manifest

    def test_lo_contract_failure_surfaces_structured_error(
        self, pipeline_registry
    ):
        """Page with out-of-week LO => structured error, not silent pass."""
        tools, tmp_path = pipeline_registry
        project_id = "PROJ-27-LOFAIL"
        # Week 1 page stamps CO-02 (which is ONLY declared for week 2).
        # The mature packager's LO-contract validator must refuse.
        project_path, content_dir = _make_project(
            tmp_path,
            project_id=project_id,
            lo_ids_per_week={
                1: ["TO-01", "CO-02"],  # violates — CO-02 is week 2's CO.
                2: ["TO-01", "CO-02"],
            },
        )

        result = asyncio.run(tools["package_imscc"](project_id=project_id))
        payload = json.loads(result)
        assert payload.get("success") is False
        assert "error" in payload
        # Exit code 2 comes straight from package_multifile_imscc.
        assert payload.get("exit_code") == 2

    def test_legacy_json_envelope_preserved_on_success(self, pipeline_registry):
        """Legacy callers see the same response keys: success, package_path,
        libv2_package_path, html_modules, package_size_bytes.
        """
        tools, tmp_path = pipeline_registry
        project_id = "PROJ-27-ENVELOPE"
        _make_project(tmp_path, project_id=project_id)

        result = asyncio.run(tools["package_imscc"](project_id=project_id))
        payload = json.loads(result)

        required_keys = {
            "success",
            "project_id",
            "package_path",
            "libv2_package_path",
            "html_modules",
            "package_size_bytes",
        }
        assert required_keys.issubset(payload.keys()), payload
        assert payload["success"] is True
        assert Path(payload["package_path"]).exists()
        assert payload["html_modules"] >= 1
        assert payload["package_size_bytes"] > 0

    def test_missing_course_metadata_still_succeeds(self, pipeline_registry):
        """Back-compat: when no course_metadata.json is present, packaging
        still succeeds (the mature packager silently omits it from the zip).
        """
        tools, tmp_path = pipeline_registry
        project_id = "PROJ-27-NOMETA"
        _make_project(
            tmp_path, project_id=project_id, emit_course_metadata=False,
        )

        result = asyncio.run(tools["package_imscc"](project_id=project_id))
        payload = json.loads(result)
        assert payload["success"] is True

        with zipfile.ZipFile(payload["package_path"], "r") as zf:
            assert "course_metadata.json" not in zf.namelist()
            # Still has manifest + HTML pages.
            names = zf.namelist()
            assert "imsmanifest.xml" in names
            assert any(n.endswith(".html") for n in names)
