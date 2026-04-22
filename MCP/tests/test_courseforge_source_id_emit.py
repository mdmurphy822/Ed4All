"""Wave 27 CRITICAL-2 — Courseforge emits ``data-cf-source-ids`` per element.

Pre-Wave-27, zero chunks in any real LibV2 course carried
``source.source_references[]`` — the Wave 9 source-router's
``source_module_map`` path was populated, but the Courseforge
content-generator never stamped per-element DART block IDs onto the
page HTML. Downstream Trainforge chunk harvesting (see
``Trainforge/parsers/html_content_parser.py``) looks for
``data-cf-source-ids`` on the nearest wrapper but found nothing.

Wave 27 closes the loop:

1. ``MCP.tools._content_gen_helpers.parse_dart_html_files`` captures
   ``data-dart-block-id`` from every DART section wrapper.
2. ``_topic_to_section`` emits ``source_references[]`` on the
   generated section dict when a DART block ID is present.
3. ``Courseforge.scripts.generate_course._render_content_sections``
   stamps ``data-cf-source-ids`` on the rendered ``<h2>`` and
   ``_build_sections_metadata`` propagates the same refs into the
   page's JSON-LD ``sections[].sourceReferences``.

Tests below verify every step of that chain against a minimal staged
DART HTML fixture so the suite runs hermetically.
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
from MCP.tools import _content_gen_helpers as _cgh  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


COURSE_CODE = "WAVE27SRC_101"


# Minimal DART HTML fixture with ``data-dart-block-id`` on every section.
# Multiple sections so the dedup path can't accidentally drop all block
# IDs. All section content is real educational prose so the no-placeholder
# policy emits real pages.
_DART_HTML_WITH_BLOCK_IDS = """<!DOCTYPE html>
<html lang="en">
<head><title>Photosynthesis Basics</title></head>
<body>
<main id="main-content" role="main" class="dart-document">
<section class="dart-section" data-dart-block-id="s1_c0"
  data-dart-source="pdfplumber" data-dart-pages="12">
  <h2>Introduction to Photosynthesis</h2>
  <p>Photosynthesis is the biological process by which plants, algae, and
  some bacteria convert light energy into chemical energy stored as
  glucose. This fundamental process sustains nearly all life on Earth by
  producing the oxygen we breathe and forming the base of most food webs.</p>
  <p>Photosynthesis occurs primarily in chloroplasts, specialized
  organelles found in the cells of plant leaves. Chloroplasts contain
  chlorophyll, a green pigment that absorbs light energy most effectively
  in the red and blue portions of the visible spectrum.</p>
</section>
<section class="dart-section" data-dart-block-id="s2_c0"
  data-dart-source="pdftotext" data-dart-pages="14">
  <h2>The Two Stages of Photosynthesis</h2>
  <p>Photosynthesis proceeds in two interconnected stages: the
  light-dependent reactions and the Calvin cycle, also known as the
  light-independent reactions.</p>
  <p>The light-dependent reactions occur in the thylakoid membranes of
  the chloroplast. The Calvin cycle takes place in the stroma, the
  fluid-filled space surrounding the thylakoids. Both stages work
  together to fix atmospheric carbon dioxide into organic glucose.</p>
</section>
<section class="dart-section" data-dart-block-id="s3_c0"
  data-dart-source="pdftotext" data-dart-pages="16">
  <h2>Cellular Respiration and Energy</h2>
  <p>Cellular respiration is the complementary process to
  photosynthesis. While photosynthesis stores chemical energy, cellular
  respiration releases that energy to fuel the work of the cell.</p>
  <p>Both processes are coupled globally: the oxygen produced by
  photosynthesis is consumed during respiration, and the carbon dioxide
  produced during respiration is consumed during photosynthesis.</p>
</section>
</main>
</body>
</html>
"""


# Legacy DART HTML — same content, but no ``data-dart-block-id`` stamped
# (pre-Wave-12 output). Used to verify the back-compat path: no
# ``data-cf-source-ids`` emitted, but the page still renders cleanly.
_DART_HTML_LEGACY = """<!DOCTYPE html>
<html lang="en">
<head><title>Photosynthesis Basics</title></head>
<body>
<main id="main-content" role="main">
<section>
  <h2>Introduction to Photosynthesis</h2>
  <p>Photosynthesis is the biological process by which plants, algae, and
  some bacteria convert light energy into chemical energy stored as
  glucose. This fundamental process sustains nearly all life on Earth by
  producing the oxygen we breathe and forming the base of most food webs.</p>
  <p>Photosynthesis occurs primarily in chloroplasts, specialized
  organelles found in the cells of plant leaves.</p>
</section>
</main>
</body>
</html>
"""


@pytest.fixture
def pipeline_registry(monkeypatch, tmp_path):
    staging_root = tmp_path / "cf_inputs"
    staging_root.mkdir()
    monkeypatch.setattr(pipeline_tools, "COURSEFORGE_INPUTS", staging_root)
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)
    return _build_tool_registry(), tmp_path, staging_root


def _make_project(tmp_path: Path, project_id: str, duration_weeks: int = 1):
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


def _stage_dart(staging_root: Path, run_id: str, html: str, filename: str):
    staging_dir = staging_root / run_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / filename).write_text(html, encoding="utf-8")
    return staging_dir


class TestParseCapturesDartBlockIds:
    """Unit test: ``parse_dart_html_files`` harvests ``data-dart-block-id``."""

    def test_dart_block_ids_captured_on_topics(self, tmp_path):
        path = tmp_path / "photosynthesis.html"
        path.write_text(_DART_HTML_WITH_BLOCK_IDS, encoding="utf-8")

        topics = _cgh.parse_dart_html_files([path])
        # Should have 3 non-low-signal topics; each carries its block_id.
        assert len(topics) >= 1
        all_block_ids: list[str] = []
        for t in topics:
            block_ids = t.get("dart_block_ids") or []
            all_block_ids.extend(block_ids)
        assert "s1_c0" in all_block_ids or "s2_c0" in all_block_ids or (
            "s3_c0" in all_block_ids
        ), f"No DART block IDs captured: {all_block_ids}"

    def test_legacy_dart_no_block_ids(self, tmp_path):
        path = tmp_path / "legacy.html"
        path.write_text(_DART_HTML_LEGACY, encoding="utf-8")

        topics = _cgh.parse_dart_html_files([path])
        for t in topics:
            assert t.get("dart_block_ids") == [], (
                f"Legacy DART topic should have empty block_ids, "
                f"got {t.get('dart_block_ids')}"
            )


class TestTopicSourceReferences:
    """Unit test: ``_topic_source_references`` produces schema-clean refs."""

    def test_single_block_id_primary(self):
        topic = {
            "dart_block_ids": ["s3_c0"],
            "source_file": "photosynthesis",
        }
        refs = _cgh._topic_source_references(topic)
        assert refs == [
            {"sourceId": "dart:photosynthesis#s3_c0", "role": "primary"},
        ]

    def test_multiple_block_ids_first_primary_rest_contributing(self):
        topic = {
            "dart_block_ids": ["s3_c0", "s4_c1", "s5_c2"],
            "source_file": "photosynthesis",
        }
        refs = _cgh._topic_source_references(topic)
        assert [r["role"] for r in refs] == [
            "primary", "contributing", "contributing",
        ]

    def test_empty_block_ids_empty_refs(self):
        assert _cgh._topic_source_references(
            {"dart_block_ids": [], "source_file": "x"}
        ) == []
        assert _cgh._topic_source_references(
            {"source_file": "x"}
        ) == []

    def test_source_id_matches_schema_pattern(self):
        """``dart:{slug}#{block_id}`` must match the canonical pattern
        from ``schemas/knowledge/source_reference.schema.json``.
        """
        pattern = re.compile(r"^dart:[a-z0-9_-]+#[a-z0-9_-]+$")
        topic = {
            "dart_block_ids": ["s3_c0"],
            "source_file": "photosynthesis",
        }
        refs = _cgh._topic_source_references(topic)
        assert refs
        assert pattern.match(refs[0]["sourceId"]), refs[0]


class TestCourseforgeEmitsDataCfSourceIds:
    """Integration: end-to-end content generation stamps source-ids."""

    def test_data_cf_source_ids_present_on_content_page(
        self, pipeline_registry
    ):
        tools, tmp_path, staging_root = pipeline_registry
        project_id = "PROJ-27-EMIT-01"
        project_path = _make_project(tmp_path, project_id)
        _stage_dart(
            staging_root, "WF-27-01",
            _DART_HTML_WITH_BLOCK_IDS, "photosynthesis.html",
        )

        asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_root / "WF-27-01"),
        ))

        week_01 = project_path / "03_content_development" / "week_01"
        # At least one content page should carry a data-cf-source-ids
        # attribute whose value starts with ``dart:photosynthesis#``.
        found = False
        for html_file in week_01.glob("*content*.html"):
            body = html_file.read_text(encoding="utf-8")
            if 'data-cf-source-ids="dart:photosynthesis#' in body:
                found = True
                break
        assert found, (
            "No Courseforge page carries data-cf-source-ids with the "
            "expected dart:photosynthesis# prefix"
        )

    def test_legacy_dart_no_source_ids_emitted(self, pipeline_registry):
        """Back-compat: DART HTML without ``data-dart-block-id`` => no
        ``data-cf-source-ids`` on section wrappers (still renders page).
        """
        tools, tmp_path, staging_root = pipeline_registry
        project_id = "PROJ-27-EMIT-02"
        project_path = _make_project(tmp_path, project_id)
        _stage_dart(
            staging_root, "WF-27-02",
            _DART_HTML_LEGACY, "legacy.html",
        )

        asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_root / "WF-27-02"),
        ))

        week_01 = project_path / "03_content_development" / "week_01"
        for html_file in week_01.glob("*content*.html"):
            body = html_file.read_text(encoding="utf-8")
            # The Wave 9 page-level path may still inject a
            # data-cf-source-ids from source_module_map.json, but this
            # test's project has no such map — assert there is no
            # ``dart:legacy#`` attribute value derived from a real block.
            assert 'data-cf-source-ids="dart:legacy#' not in body, (
                "Legacy DART HTML (no data-dart-block-id) must not yield "
                "dart:legacy# attributes on Courseforge pages"
            )

    def test_jsonld_sections_carry_source_references(self, pipeline_registry):
        """JSON-LD ``sections[].sourceReferences`` populated end-to-end."""
        tools, tmp_path, staging_root = pipeline_registry
        project_id = "PROJ-27-EMIT-03"
        project_path = _make_project(tmp_path, project_id)
        _stage_dart(
            staging_root, "WF-27-03",
            _DART_HTML_WITH_BLOCK_IDS, "photosynthesis.html",
        )

        asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_root / "WF-27-03"),
        ))

        week_01 = project_path / "03_content_development" / "week_01"
        jsonld_re = re.compile(
            r'<script type="application/ld\+json">(.*?)</script>',
            re.DOTALL,
        )

        any_page_has_refs = False
        for html_file in week_01.glob("*content*.html"):
            body = html_file.read_text(encoding="utf-8")
            m = jsonld_re.search(body)
            if not m:
                continue
            meta = json.loads(m.group(1))
            sections = meta.get("sections") or []
            for sec in sections:
                refs = sec.get("sourceReferences") or []
                if refs:
                    any_page_has_refs = True
                    # Pattern-validate every ref.
                    for ref in refs:
                        assert "sourceId" in ref
                        assert ref["sourceId"].startswith("dart:photosynthesis#"), ref
        assert any_page_has_refs, (
            "No section-level sourceReferences populated in JSON-LD"
        )

    def test_course_slug_derived_from_source_stem(self, pipeline_registry):
        """Course slug tracks the staged DART file stem.

        Wave 35: the emitted slug now preserves underscores (lowercase
        + space-to-hyphen only) so it matches the slug the
        ``ContentGroundingValidator`` + Wave 9 source-router build when
        they read the staged HTML's ``path.stem``. Pre-Wave-35 used
        :func:`canonical_slug`, which collapsed ``XYZ_201`` to
        ``xyz201`` and diverged from the validator's ``xyz_201``.
        """
        tools, tmp_path, staging_root = pipeline_registry
        project_id = "PROJ-27-EMIT-04"
        project_path = _make_project(tmp_path, project_id)
        _stage_dart(
            staging_root, "WF-27-04",
            _DART_HTML_WITH_BLOCK_IDS.replace(
                "<title>Photosynthesis Basics</title>",
                "<title>XYZ_201 Textbook</title>",
            ),
            "XYZ_201.html",
        )

        asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_root / "WF-27-04"),
        ))

        week_01 = project_path / "03_content_development" / "week_01"
        # The validator-compatible slug is "xyz_201" (underscore kept).
        found = False
        for html_file in week_01.glob("*content*.html"):
            body = html_file.read_text(encoding="utf-8")
            if "dart:xyz_201#" in body:
                found = True
                break
        assert found, "Expected source slug 'xyz_201' derived from XYZ_201 filename"

    def test_source_id_pattern_validates_schema(self, pipeline_registry):
        """Emitted sourceIds match the source_reference schema pattern."""
        tools, tmp_path, staging_root = pipeline_registry
        project_id = "PROJ-27-EMIT-05"
        project_path = _make_project(tmp_path, project_id)
        _stage_dart(
            staging_root, "WF-27-05",
            _DART_HTML_WITH_BLOCK_IDS, "photosynthesis.html",
        )

        asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_root / "WF-27-05"),
        ))

        week_01 = project_path / "03_content_development" / "week_01"
        pattern = re.compile(r"^dart:[a-z0-9_-]+#[a-z0-9_-]+$")
        attr_re = re.compile(r'data-cf-source-ids="([^"]+)"')
        checked = 0
        for html_file in week_01.glob("*.html"):
            body = html_file.read_text(encoding="utf-8")
            for match in attr_re.finditer(body):
                for sid in match.group(1).split(","):
                    sid = sid.strip()
                    if not sid:
                        continue
                    assert pattern.match(sid), (
                        f"sourceId {sid!r} in {html_file.name} does not "
                        f"match canonical pattern"
                    )
                    checked += 1
        assert checked >= 1, (
            "Expected at least one data-cf-source-ids attribute to verify"
        )


class TestTrainforgeHarvestsSourceReferences:
    """Downstream integration: Trainforge's ``html_content_parser`` picks
    up the newly-emitted ``data-cf-source-ids`` on Courseforge chunks.
    """

    def test_module_carries_source_references(
        self, pipeline_registry,
    ):
        tools, tmp_path, staging_root = pipeline_registry
        project_id = "PROJ-27-EMIT-06"
        project_path = _make_project(tmp_path, project_id)
        _stage_dart(
            staging_root, "WF-27-06",
            _DART_HTML_WITH_BLOCK_IDS, "photosynthesis.html",
        )

        asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_root / "WF-27-06"),
        ))

        # Locate a content page, feed it through Trainforge's parser.
        # HTMLContentParser.parse returns a single ``ParsedHTMLModule``
        # whose ``source_references`` field is populated from JSON-LD
        # + ``data-cf-source-ids`` per Wave 10's priority chain.
        from Trainforge.parsers.html_content_parser import HTMLContentParser

        week_01 = project_path / "03_content_development" / "week_01"
        content_pages = sorted(week_01.glob("*content*.html"))
        assert content_pages, "No content pages emitted"

        parser = HTMLContentParser()
        found_ref = False
        for page in content_pages:
            html = page.read_text(encoding="utf-8")
            module = parser.parse(html)
            refs = getattr(module, "source_references", None) or []
            if refs:
                found_ref = True
                # Pattern-validate: every ref has the canonical sourceId
                # shape and traces back to our staged DART fixture.
                for ref in refs:
                    assert isinstance(ref, dict)
                    sid = ref.get("sourceId", "")
                    assert sid.startswith("dart:photosynthesis#"), ref
                break

        assert found_ref, (
            "Trainforge parse harvested no source_references — "
            "Wave 27 carry-through broke"
        )


class TestPageSourceRefValidatorNonVacuous:
    """``PageSourceRefValidator`` must pass non-vacuously when the page
    carries real source-refs (Wave 27 closes the "empty => passes"
    vacuous path for real runs).
    """

    def test_validator_passes_with_real_refs(self, pipeline_registry):
        tools, tmp_path, staging_root = pipeline_registry
        project_id = "PROJ-27-EMIT-07"
        project_path = _make_project(tmp_path, project_id)
        _stage_dart(
            staging_root, "WF-27-07",
            _DART_HTML_WITH_BLOCK_IDS, "photosynthesis.html",
        )

        asyncio.run(tools["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_root / "WF-27-07"),
        ))

        week_01 = project_path / "03_content_development" / "week_01"
        page_paths = [str(p) for p in week_01.glob("*.html")]

        from lib.validators.source_refs import PageSourceRefValidator

        validator = PageSourceRefValidator()
        # Seed the valid-id set with every dart:photosynthesis# id we
        # can possibly emit (s1_c0, s2_c0, s3_c0); validator passes when
        # every emitted sid resolves against this set.
        valid_ids = {
            f"dart:photosynthesis#s{i}_c0" for i in range(1, 10)
        }
        result = validator.validate({
            "gate_id": "source_refs",
            "page_paths": page_paths,
            "valid_source_ids": valid_ids,
        })
        assert result.passed is True, [
            (i.code, i.message) for i in result.issues
        ]
