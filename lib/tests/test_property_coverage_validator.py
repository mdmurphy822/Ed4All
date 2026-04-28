"""Wave 109 / Phase C: PropertyCoverageValidator must fail closed when
synthesis output misses any declared property's minimum-pairs floor."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from lib.validators.property_coverage import PropertyCoverageValidator


def _write_pairs(course_dir: Path, rows: List[dict]) -> Path:
    p = course_dir / "training_specs" / "instruction_pairs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_passing_when_every_property_meets_floor(tmp_path: Path) -> None:
    course = tmp_path / "courses" / "rdf-shacl-551-2"
    rows = []
    for prop in ("sh:datatype", "sh:class", "sh:NodeShape",
                 "sh:PropertyShape", "rdfs:subClassOf", "owl:sameAs"):
        for i in range(8):
            rows.append({
                "prompt": f"Prompt {i} about {prop}",
                "completion": f"Use {prop} to describe the schema.",
            })
    _write_pairs(course, rows)
    result = PropertyCoverageValidator().validate({
        "course_dir": str(course),
        "course_slug": "rdf-shacl-551-2",
    })
    assert result.passed is True
    assert not [i for i in result.issues if i.severity == "critical"]


def test_fails_critical_when_property_missing(tmp_path: Path) -> None:
    course = tmp_path / "courses" / "rdf-shacl-551-2"
    rows = []
    for prop in ("sh:datatype", "sh:class", "sh:NodeShape",
                 "sh:PropertyShape", "rdfs:subClassOf"):
        for i in range(8):
            rows.append({
                "prompt": f"Prompt {i} about {prop}",
                "completion": f"Use {prop} to describe the schema.",
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
    rows = []
    for prop in ("sh:datatype", "sh:class", "sh:NodeShape",
                 "sh:PropertyShape", "rdfs:subClassOf", "owl:sameAs"):
        for i in range(3):
            rows.append({
                "prompt": f"Prompt {i} about {prop}",
                "completion": f"Use {prop}.",
            })
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
