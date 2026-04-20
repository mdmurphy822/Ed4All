"""Worker α — ``_generate_course_content`` unit tests.

Verifies the textbook-to-course content-generation tool produces the
5-page weekly module structure with full ``data-cf-*`` + JSON-LD
metadata per the Wave Pipeline contract
(``plans/pipeline-execution-fixes/contracts.md`` §1).

Uses a minimal staged DART HTML fixture so the test runs in the default
suite (no network, no subprocess, no real pipeline). The tool is
registered inside ``_build_tool_registry``; we reach it by calling that
function directly.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


COURSE_CODE = "TESTPIPE_101"

# Minimal DART-shaped HTML fixture: two <section> blocks with text
# matching the reference fixture's photosynthesis topic.
_DART_HTML = """<!DOCTYPE html>
<html lang="en">
<head><title>Photosynthesis Basics</title></head>
<body>
<main id="main-content" role="main">
<section id="intro" aria-labelledby="intro-heading">
  <h2 id="intro-heading">Introduction to Photosynthesis</h2>
  <p>Photosynthesis is the biological process by which plants, algae, and
  some bacteria convert light energy into chemical energy stored as
  glucose. This fundamental process sustains nearly all life on Earth by
  producing the oxygen we breathe and forming the base of most food webs.</p>
  <p>Photosynthesis occurs primarily in chloroplasts, specialized
  organelles found in the cells of plant leaves. Chloroplasts contain
  chlorophyll, a green pigment that absorbs light energy most effectively
  in the red and blue portions of the visible spectrum. The Calvin cycle
  is the second major stage.</p>
</section>
<section id="stages" aria-labelledby="stages-heading">
  <h2 id="stages-heading">The Two Stages of Photosynthesis</h2>
  <p>Photosynthesis proceeds in two interconnected stages: the
  light-dependent reactions and the Calvin cycle, also known as the
  light-independent reactions.</p>
  <p>The light-dependent reactions occur in the thylakoid membranes of
  the chloroplast. The Calvin cycle takes place in the stroma, the
  fluid-filled space surrounding the thylakoids. Both stages work
  together to fix atmospheric carbon dioxide into organic glucose.</p>
</section>
</main>
</body>
</html>
"""


@pytest.fixture
def pipeline_registry(monkeypatch, tmp_path):
    """Build the pipeline tool registry against a tmp Courseforge inputs dir.

    Redirects both the Courseforge staging root and the ``_PROJECT_ROOT``
    used by ``_generate_course_content`` so the tool writes to tmp paths
    (no pollution of the real exports/).
    """
    staging_root = tmp_path / "cf_inputs"
    staging_root.mkdir()
    monkeypatch.setattr(pipeline_tools, "COURSEFORGE_INPUTS", staging_root)
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)

    registry = _build_tool_registry()
    return registry, tmp_path, staging_root


def _make_project(tmp_path: Path, project_id: str, duration_weeks: int = 2):
    """Create a Courseforge exports/{project_id} workspace with config."""
    project_path = tmp_path / "Courseforge" / "exports" / project_id
    (project_path / "03_content_development").mkdir(parents=True, exist_ok=True)
    (project_path / "01_learning_objectives").mkdir(parents=True, exist_ok=True)
    config = {
        "project_id": project_id,
        "course_name": COURSE_CODE,
        "duration_weeks": duration_weeks,
        "objectives_path": None,
    }
    (project_path / "project_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return project_path


def _stage_dart(staging_root: Path, run_id: str):
    staging_dir = staging_root / run_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "photosynthesis.html").write_text(_DART_HTML, encoding="utf-8")
    return staging_dir


# ---------------------------------------------------------------------- #
# Shape tests
# ---------------------------------------------------------------------- #


class TestContentGenerationShape:
    def test_emits_five_pages_per_week(self, pipeline_registry):
        tools, tmp_path, staging_root = pipeline_registry
        project_id = "PROJ-TESTPIPE-01"
        project_path = _make_project(tmp_path, project_id, duration_weeks=2)
        staging_dir = _stage_dart(staging_root, "WF-01")

        result = asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_dir),
        ))
        payload = json.loads(result)
        assert payload.get("success") is True, payload
        assert payload["weeks_prepared"] == 2

        week_01 = project_path / "03_content_development" / "week_01"
        assert week_01.exists()
        html_files = sorted(week_01.glob("*.html"))
        # Contract: 5 pages — overview/content/application/self_check/summary.
        assert len(html_files) >= 5, [f.name for f in html_files]

        # Module types represented.
        names = [f.name for f in html_files]
        assert any("overview" in n for n in names)
        assert any("content_" in n for n in names)
        assert any("application" in n for n in names)
        assert any("self_check" in n for n in names)
        assert any("summary" in n for n in names)

    def test_each_page_has_data_cf_role_and_jsonld(self, pipeline_registry):
        tools, tmp_path, staging_root = pipeline_registry
        project_id = "PROJ-TESTPIPE-02"
        project_path = _make_project(tmp_path, project_id, duration_weeks=2)
        staging_dir = _stage_dart(staging_root, "WF-02")

        asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_dir),
        ))

        week_01 = project_path / "03_content_development" / "week_01"
        for html_file in week_01.glob("*.html"):
            body = html_file.read_text(encoding="utf-8")
            assert 'data-cf-role="template-chrome"' in body, html_file.name
            assert 'application/ld+json' in body, html_file.name
            assert 'data-cf-objective-id=' in body, html_file.name
            # Not the old DIGPED 101 hardcoded template.
            assert "DIGPED 101" not in body, html_file.name

    def test_jsonld_validates_against_schema(self, pipeline_registry):
        """Every JSON-LD block must validate against courseforge_jsonld_v1."""
        pytest.importorskip("jsonschema")
        pytest.importorskip("referencing")
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource

        tools, tmp_path, staging_root = pipeline_registry
        project_id = "PROJ-TESTPIPE-03"
        project_path = _make_project(tmp_path, project_id, duration_weeks=2)
        staging_dir = _stage_dart(staging_root, "WF-03")

        asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_dir),
        ))

        schemas = PROJECT_ROOT / "schemas"
        main_schema_path = schemas / "knowledge" / "courseforge_jsonld_v1.schema.json"
        source_ref_path = schemas / "knowledge" / "source_reference.schema.json"
        main_schema = json.loads(main_schema_path.read_text())
        source_ref_schema = json.loads(source_ref_path.read_text())

        resources = [
            (main_schema["$id"], Resource.from_contents(main_schema)),
            (source_ref_schema["$id"], Resource.from_contents(source_ref_schema)),
        ]
        for name in [
            "bloom_verbs.json", "module_type.json", "content_type.json",
            "cognitive_domain.json", "question_type.json",
        ]:
            tax = json.loads((schemas / "taxonomies" / name).read_text())
            resources.append((tax["$id"], Resource.from_contents(tax)))
        registry = Registry().with_resources(resources)
        validator = Draft202012Validator(main_schema, registry=registry)

        jsonld_re = re.compile(
            r'<script type="application/ld\+json">(.*?)</script>',
            re.DOTALL,
        )

        week_01 = project_path / "03_content_development" / "week_01"
        pages_checked = 0
        for html_file in week_01.glob("*.html"):
            body = html_file.read_text(encoding="utf-8")
            match = jsonld_re.search(body)
            assert match, f"{html_file.name} missing JSON-LD block"
            meta = json.loads(match.group(1))
            errors = sorted(
                validator.iter_errors(meta), key=lambda e: list(e.path)
            )
            assert not errors, (
                f"{html_file.name} JSON-LD invalid: "
                + "; ".join(e.message for e in errors[:3])
            )
            pages_checked += 1
        assert pages_checked >= 5

    def test_self_check_module_type_is_assessment(self, pipeline_registry):
        """Schema gap: self_check pages emit moduleType: 'assessment'."""
        tools, tmp_path, staging_root = pipeline_registry
        project_id = "PROJ-TESTPIPE-04"
        project_path = _make_project(tmp_path, project_id, duration_weeks=1)
        staging_dir = _stage_dart(staging_root, "WF-04")

        asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_dir),
        ))

        sc_path = (
            project_path / "03_content_development" / "week_01"
            / "week_01_self_check.html"
        )
        assert sc_path.exists()
        body = sc_path.read_text(encoding="utf-8")
        match = re.search(
            r'<script type="application/ld\+json">(.*?)</script>',
            body, re.DOTALL,
        )
        assert match
        meta = json.loads(match.group(1))
        assert meta["moduleType"] == "assessment"

    def test_works_without_dart_staging(self, pipeline_registry):
        """Missing staging dir must not crash; fall back to synthesis."""
        tools, tmp_path, _ = pipeline_registry
        project_id = "PROJ-TESTPIPE-05"
        project_path = _make_project(tmp_path, project_id, duration_weeks=1)

        result = asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(tmp_path / "nonexistent"),
        ))
        payload = json.loads(result)
        assert payload.get("success") is True
        week_01 = project_path / "03_content_development" / "week_01"
        html_files = list(week_01.glob("*.html"))
        assert len(html_files) >= 5

    def test_honors_source_module_map_when_populated(
        self, pipeline_registry,
    ):
        """When source_module_map.json is non-empty, sourceReferences[] appears."""
        tools, tmp_path, staging_root = pipeline_registry
        project_id = "PROJ-TESTPIPE-06"
        project_path = _make_project(tmp_path, project_id, duration_weeks=1)
        staging_dir = _stage_dart(staging_root, "WF-06")

        # Write a populated source_module_map.json.
        map_path = project_path / "source_module_map.json"
        map_path.write_text(json.dumps({
            "week_01": {
                "week_01_overview": {
                    "primary": ["dart:photosynthesis#s1_p0"],
                    "contributing": [],
                    "confidence": 0.9,
                }
            }
        }), encoding="utf-8")

        asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_dir),
            source_module_map_path=str(map_path),
        ))

        overview = (
            project_path / "03_content_development" / "week_01"
            / "week_01_overview.html"
        )
        assert overview.exists()
        body = overview.read_text(encoding="utf-8")
        assert "sourceReferences" in body
        assert "dart:photosynthesis" in body
