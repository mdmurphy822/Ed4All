"""Wave 30 Gap 4 — course.json is always written.

Pre-Wave-30, ``Trainforge/process_course.py::_build_course_json`` was
only called when ``self.objectives`` was truthy, which required the
caller to thread ``objectives_path`` through. Pipeline runs that
auto-synthesize LOs at
``{project_path}/01_learning_objectives/synthesized_objectives.json``
never did — ``course.json`` never landed, and LibV2 retrieval /
validator joins had nothing to look at.

Wave 30 Gap 4 does two things:

1. When ``objectives_path`` is ``None``, the constructor probes the
   canonical auto-synthesized location so ``self.objectives`` gets
   populated whenever the Wave-24 sidecar exists.
2. ``_write_metadata`` now always calls ``_build_course_json`` and
   writes ``course.json`` even when no objectives resolved — the
   result is a schema-valid shell with ``learning_outcomes: []``
   and a ``note`` field explaining the absence.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.process_course import CourseProcessor  # noqa: E402

COURSE_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "knowledge" / "course.schema.json"
)


def _minimal_imscc(tmp_path: Path) -> Path:
    """Write a throwaway IMSCC package that parses but is otherwise empty.
    CourseProcessor needs a real zip to get past Stage 1; we don't care
    about its content for course.json materialisation testing."""
    imscc = tmp_path / "minimal.imscc"
    with zipfile.ZipFile(imscc, "w") as zf:
        # Minimal manifest + one empty HTML resource.
        manifest = """<?xml version="1.0" encoding="UTF-8"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1"
          xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p1/LOM/resource">
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.1.0</schemaversion>
    <lomimscc:lom>
      <lomimscc:general>
        <lomimscc:title>
          <lomimscc:string language="en">Wave 30 Gap 4 Fixture</lomimscc:string>
        </lomimscc:title>
      </lomimscc:general>
    </lomimscc:lom>
  </metadata>
  <resources>
    <resource identifier="r1" type="webcontent" href="content.html">
      <file href="content.html"/>
    </resource>
  </resources>
</manifest>
"""
        zf.writestr("imsmanifest.xml", manifest)
        zf.writestr(
            "content.html",
            "<!DOCTYPE html><html><body><p>Stub body for Wave 30 Gap 4.</p></body></html>",
        )
    return imscc


def _minimal_objectives_json() -> dict:
    """Realistic shape for synthesized_objectives.json (Wave 24)."""
    return {
        "course_name": "WAVE30_GAP4_TEST",
        "generated_from": "synthetic",
        "mint_method": "test_fixture",
        "duration_weeks": 4,
        "learning_outcomes": [
            {
                "id": "to-01",
                "statement": "Recall the course purpose.",
                "bloomLevel": "remember",
                "hierarchy_level": "terminal",
            }
        ],
        "terminal_objectives": [
            {
                "id": "to-01",
                "statement": "Recall the course purpose.",
                "bloomLevel": "remember",
            }
        ],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {
                        "id": "co-01",
                        "statement": "List course artifacts.",
                        "bloomLevel": "remember",
                    }
                ],
            }
        ],
    }


def _load_schema() -> dict:
    with COURSE_SCHEMA_PATH.open() as fh:
        return json.load(fh)


@pytest.mark.unit
def test_course_json_written_when_objectives_path_supplied(tmp_path):
    """Pre-Wave-30 contract regression: when ``objectives_path`` is
    supplied, course.json must land with the supplied LOs."""
    imscc = _minimal_imscc(tmp_path)
    output_dir = tmp_path / "trainforge_out"
    output_dir.mkdir()

    objectives_path = tmp_path / "objectives.json"
    objectives_path.write_text(
        json.dumps(_minimal_objectives_json()), encoding="utf-8"
    )

    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(output_dir),
        course_code="WAVE30_KWARG_TEST",
        objectives_path=str(objectives_path),
    )
    processor.process()

    course_json_path = output_dir / "course.json"
    assert course_json_path.exists()
    data = json.loads(course_json_path.read_text())
    assert data["course_code"] == "WAVE30_KWARG_TEST"
    # Both TO + CO emit from the objectives file.
    assert len(data["learning_outcomes"]) == 2
    assert "note" not in data


@pytest.mark.unit
def test_course_json_written_from_auto_synthesized_sidecar(tmp_path):
    """When objectives_path is None BUT the canonical
    ``{project_path}/01_learning_objectives/synthesized_objectives.json``
    sidecar exists, the constructor must auto-detect + load it so
    course.json lands with real LOs (not the empty shell).
    """
    imscc = _minimal_imscc(tmp_path)
    # Simulate the textbook_to_course layout: project_path parent of
    # trainforge/ with 01_learning_objectives/synthesized_objectives.json.
    project_path = tmp_path / "project"
    project_path.mkdir()
    output_dir = project_path / "trainforge"
    output_dir.mkdir()

    lo_dir = project_path / "01_learning_objectives"
    lo_dir.mkdir()
    (lo_dir / "synthesized_objectives.json").write_text(
        json.dumps(_minimal_objectives_json()), encoding="utf-8"
    )

    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(output_dir),
        course_code="WAVE30_AUTO_TEST",
        # Note: NO objectives_path kwarg passed.
    )
    # Detection must populate self.objectives before process() runs.
    assert processor.objectives is not None, (
        "Wave 30 Gap 4: auto-detection of synthesized_objectives.json regressed"
    )
    assert processor._objectives_source == "auto_synthesized"

    processor.process()
    course_json_path = output_dir / "course.json"
    assert course_json_path.exists()
    data = json.loads(course_json_path.read_text())
    assert len(data["learning_outcomes"]) == 2


@pytest.mark.unit
def test_course_json_empty_shell_when_no_objectives_available(tmp_path):
    """Neither ``objectives_path`` kwarg NOR an auto-synthesized sidecar
    available → course.json still materialises as an empty-LOs shell
    with a ``note`` field. LibV2 archival always finds a file, and the
    result still validates against the course schema."""
    imscc = _minimal_imscc(tmp_path)
    output_dir = tmp_path / "trainforge_out_empty"
    output_dir.mkdir()

    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(output_dir),
        course_code="WAVE30_EMPTY_TEST",
    )
    # No objectives discovered.
    assert processor.objectives is None
    processor.process()

    course_json_path = output_dir / "course.json"
    assert course_json_path.exists(), (
        "Wave 30 Gap 4 regression: course.json must be written even with no objectives"
    )
    data = json.loads(course_json_path.read_text())
    assert data["course_code"] == "WAVE30_EMPTY_TEST"
    assert data["learning_outcomes"] == []
    assert "note" in data
    assert len(data["note"]) > 20  # human-readable explanation


@pytest.mark.unit
def test_course_json_shell_validates_against_schema(tmp_path):
    """The empty-shell course.json must schema-validate against
    ``schemas/knowledge/course.schema.json`` — the schema accepts empty
    arrays + optional ``note`` via additionalProperties:true."""
    try:
        import jsonschema
    except ImportError:
        pytest.skip("jsonschema not available in this environment")

    imscc = _minimal_imscc(tmp_path)
    output_dir = tmp_path / "trainforge_out_schema"
    output_dir.mkdir()

    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(output_dir),
        course_code="WAVE30_SCHEMA_TEST",
    )
    processor.process()

    course_json_path = output_dir / "course.json"
    data = json.loads(course_json_path.read_text())
    schema = _load_schema()
    # Should not raise.
    jsonschema.validate(data, schema)


@pytest.mark.unit
def test_build_course_json_returns_shell_when_objectives_none(tmp_path):
    """Unit-level: ``_build_course_json`` called directly with
    ``self.objectives=None`` must return a dict with ``learning_outcomes:
    []`` and ``note`` populated. Exercised without running the full
    pipeline so we can pin the API contract."""
    imscc = _minimal_imscc(tmp_path)
    output_dir = tmp_path / "unit_test_output"
    output_dir.mkdir()

    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(output_dir),
        course_code="UNIT_TEST",
    )
    # Force no-objectives state.
    processor.objectives = None

    manifest = {"title": "Unit Test Course"}
    result = processor._build_course_json(manifest)
    assert result["course_code"] == "UNIT_TEST"
    assert result["title"] == "Unit Test Course"
    assert result["learning_outcomes"] == []
    assert "note" in result
    assert "No learning objectives" in result["note"]


@pytest.mark.unit
def test_kwarg_objectives_path_overrides_auto_detection(tmp_path):
    """When the caller supplies ``objectives_path`` explicitly AND the
    auto-synthesized sidecar exists, the kwarg wins. Preserves the
    Wave-24 caller contract."""
    imscc = _minimal_imscc(tmp_path)
    project_path = tmp_path / "project_override"
    project_path.mkdir()
    output_dir = project_path / "trainforge"
    output_dir.mkdir()

    # Plant the auto-synthesized file.
    lo_dir = project_path / "01_learning_objectives"
    lo_dir.mkdir()
    (lo_dir / "synthesized_objectives.json").write_text(
        json.dumps(_minimal_objectives_json()), encoding="utf-8"
    )

    # Supply a DIFFERENT kwarg file with a single distinct LO.
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({
        "terminal_objectives": [
            {
                "id": "to-99",
                "statement": "Explicit-only outcome.",
                "bloomLevel": "understand",
            }
        ],
        "chapter_objectives": [],
    }), encoding="utf-8")

    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(output_dir),
        course_code="WAVE30_OVERRIDE_TEST",
        objectives_path=str(explicit),
    )
    assert processor._objectives_source == "kwarg"
    # Ensure the explicit objectives wiped out any auto-detection.
    ids = [
        to["id"] for to in processor.objectives.get("terminal_objectives", [])
    ]
    assert "to-99" in ids
    assert "to-01" not in ids
