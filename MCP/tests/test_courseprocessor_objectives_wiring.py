"""Tests for Wave 24 CourseProcessor objectives wiring.

Before Wave 24, pipeline_tools.py invoked CourseProcessor without
objectives_path, so self.objectives stayed None, _build_valid_outcome_ids
returned an empty set, and _build_course_json was never called → no
course.json, every chunk ref flagged as broken.

These tests cover the fix:
  1. CourseProcessor accepts objectives_path kwarg (already did) and
     loads objectives from it.
  2. _build_valid_outcome_ids returns the expected lowercase IDs.
  3. _build_course_json produces a schema-compliant shape.
  4. Empty/missing objectives_path falls back without crashing.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_imscc(tmp_path: Path) -> Path:
    """Create a minimal IMSCC-ish zip file with just a manifest-like entry."""
    path = tmp_path / "minimal.imscc"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "imsmanifest.xml",
            '<?xml version="1.0"?><manifest identifier="x"/>',
        )
    return path


def _make_objectives(tmp_path: Path) -> Path:
    path = tmp_path / "objectives.json"
    path.write_text(json.dumps({
        "terminal_objectives": [
            {"id": "TO-01", "statement": "First terminal outcome.",
             "bloom_level": "understand"},
            {"id": "TO-02", "statement": "Second terminal outcome.",
             "bloom_level": "apply"},
        ],
        "chapter_objectives": [{
            "chapter": "Week 1",
            "objectives": [
                {"id": "CO-01", "statement": "First chapter objective.",
                 "bloom_level": "remember"},
                {"id": "CO-02", "statement": "Second chapter objective.",
                 "bloom_level": "apply"},
            ],
        }],
    }), encoding="utf-8")
    return path


def test_courseprocessor_accepts_objectives_path(tmp_path):
    """CourseProcessor with objectives_path loads them and exposes outcomes."""
    from Trainforge.process_course import CourseProcessor

    imscc = _make_imscc(tmp_path)
    objectives = _make_objectives(tmp_path)
    output = tmp_path / "out"

    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(output),
        course_code="TEST_101",
        objectives_path=str(objectives),
    )
    assert processor.objectives is not None
    assert len(processor.objectives.get("terminal_objectives", [])) == 2


def test_valid_outcome_ids_populated_from_objectives(tmp_path):
    """_build_valid_outcome_ids returns lowercased TO/CO IDs."""
    from Trainforge.process_course import CourseProcessor

    imscc = _make_imscc(tmp_path)
    objectives = _make_objectives(tmp_path)
    output = tmp_path / "out"

    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(output),
        course_code="TEST_101",
        objectives_path=str(objectives),
    )
    valid_ids = processor._build_valid_outcome_ids()
    # All four LOs should surface (lowercased).
    assert "to-01" in valid_ids
    assert "to-02" in valid_ids
    assert "co-01" in valid_ids
    assert "co-02" in valid_ids


def test_build_course_json_shape(tmp_path):
    """_build_course_json produces the canonical schema shape."""
    from Trainforge.process_course import CourseProcessor

    imscc = _make_imscc(tmp_path)
    objectives = _make_objectives(tmp_path)
    output = tmp_path / "out"

    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(output),
        course_code="TEST_101",
        objectives_path=str(objectives),
    )
    manifest = {"title": "Test Course"}
    course_data = processor._build_course_json(manifest)
    assert course_data["course_code"] == "TEST_101"
    assert course_data["title"] == "Test Course"
    outcomes = course_data["learning_outcomes"]
    assert len(outcomes) == 4
    # Schema-required fields are all present.
    for lo in outcomes:
        assert "id" in lo
        assert "statement" in lo
        assert "hierarchy_level" in lo
        assert lo["hierarchy_level"] in ("terminal", "chapter")


def test_empty_objectives_path_falls_back(tmp_path):
    """No objectives_path → self.objectives is None, valid_ids empty, no crash."""
    from Trainforge.process_course import CourseProcessor

    imscc = _make_imscc(tmp_path)
    output = tmp_path / "out"

    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(output),
        course_code="TEST_LEGACY",
        objectives_path=None,
    )
    assert processor.objectives is None
    valid_ids = processor._build_valid_outcome_ids()
    assert valid_ids == set()
