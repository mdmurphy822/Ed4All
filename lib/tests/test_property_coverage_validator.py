"""Wave 109 / Phase C: PropertyCoverageValidator must fail closed when
synthesis output misses any declared property's minimum-pairs floor."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from lib.ontology.property_manifest import load_property_manifest
from lib.validators.property_coverage import PropertyCoverageValidator


def _write_pairs(course_dir: Path, rows: List[dict]) -> Path:
    p = course_dir / "training_specs" / "instruction_pairs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _rows_for_all_manifest_properties(slug: str, count_offset: int = 0) -> List[dict]:
    """Build instruction pairs covering every CURIE in the course's
    manifest at `min_pairs + count_offset`. Wave 130a expanded the
    rdf_shacl manifest 6 -> 40 properties so hardcoded fixture lists
    don't scale; iterate the manifest directly."""
    manifest = load_property_manifest(slug)
    rows: List[dict] = []
    for prop in manifest.properties:
        n = max(0, prop.min_pairs + count_offset)
        for i in range(n):
            rows.append({
                "prompt": f"Prompt {i} about {prop.curie}",
                "completion": f"Use {prop.curie} to describe the schema.",
            })
    return rows


def test_passing_when_every_property_meets_floor(tmp_path: Path) -> None:
    course = tmp_path / "courses" / "rdf-shacl-551-2"
    # +2 over the floor so every property comfortably passes.
    rows = _rows_for_all_manifest_properties("rdf-shacl-551-2", count_offset=2)
    _write_pairs(course, rows)
    result = PropertyCoverageValidator().validate({
        "course_dir": str(course),
        "course_slug": "rdf-shacl-551-2",
    })
    assert result.passed is True
    assert not [i for i in result.issues if i.severity == "critical"]


def test_fails_critical_when_property_missing(tmp_path: Path) -> None:
    course = tmp_path / "courses" / "rdf-shacl-551-2"
    # Cover every manifest property except `owl:sameAs` to assert the
    # validator names the single missing one.
    manifest = load_property_manifest("rdf-shacl-551-2")
    rows: List[dict] = []
    for prop in manifest.properties:
        if prop.id == "owl_sameas":
            continue
        for i in range(prop.min_pairs + 2):
            rows.append({
                "prompt": f"Prompt {i} about {prop.curie}",
                "completion": f"Use {prop.curie} to describe the schema.",
            })
    _write_pairs(course, rows)
    result = PropertyCoverageValidator().validate({
        "course_dir": str(course),
        "course_slug": "rdf-shacl-551-2",
    })
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "PROPERTY_COVERAGE_BELOW_FLOOR" in codes
    msgs = " ".join(i.message for i in result.issues if i.severity == "critical")
    assert "owl_sameas" in msgs


def test_fails_critical_when_property_under_floor(tmp_path: Path) -> None:
    course = tmp_path / "courses" / "rdf-shacl-551-2"
    # Each property gets only 1 pair, well below any min_pairs floor.
    manifest = load_property_manifest("rdf-shacl-551-2")
    rows = [
        {
            "prompt": f"Prompt {i} about {prop.curie}",
            "completion": f"Use {prop.curie}.",
        }
        for prop in manifest.properties
        for i in range(1)
    ]
    _write_pairs(course, rows)
    result = PropertyCoverageValidator().validate({
        "course_dir": str(course),
        "course_slug": "rdf-shacl-551-2",
    })
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "PROPERTY_COVERAGE_BELOW_FLOOR" in codes


def test_missing_instruction_pairs_fails_critical(tmp_path: Path) -> None:
    course = tmp_path / "courses" / "rdf-shacl-551-2"
    course.mkdir(parents=True)
    result = PropertyCoverageValidator().validate({
        "course_dir": str(course),
        "course_slug": "rdf-shacl-551-2",
    })
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "INSTRUCTION_PAIRS_NOT_FOUND" in codes


def test_no_manifest_for_course_passes(tmp_path: Path) -> None:
    """Courses without a property manifest are out-of-scope for this
    gate. The validator no-ops so legacy workflows don't break."""
    course = tmp_path / "courses" / "unknown-course-001"
    _write_pairs(course, [{"prompt": "p", "completion": "c"}])
    result = PropertyCoverageValidator().validate({
        "course_dir": str(course),
        "course_slug": "unknown-course-001",
    })
    assert result.passed is True
