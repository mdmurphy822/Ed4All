"""Wave 110 / Phase D — SynthesisQuotaValidator tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.validators.synthesis_quota import SynthesisQuotaValidator


def _write_chunks(course_dir: Path, n: int) -> Path:
    p = course_dir / "corpus" / "chunks.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "id": f"chunk_{i:04d}",
                "learning_outcome_refs": ["TO-01"],
                "text": "x",
            }) + "\n")
    return p


def test_passes_when_estimate_under_ceiling(tmp_path: Path) -> None:
    course = tmp_path / "courses" / "small"
    _write_chunks(course, 100)
    result = SynthesisQuotaValidator().validate({
        "course_dir": str(course),
        "instruction_variants_per_chunk": 1,
        "thresholds": {"max_estimated_dispatches": 1500},
    })
    assert result.passed is True


def test_warns_when_estimate_exceeds_ceiling(tmp_path: Path) -> None:
    """1000 chunks × (3 variants + 1 pref) = 4000 > 1500 ceiling."""
    course = tmp_path / "courses" / "huge"
    _write_chunks(course, 1000)
    result = SynthesisQuotaValidator().validate({
        "course_dir": str(course),
        "instruction_variants_per_chunk": 3,
        "thresholds": {"max_estimated_dispatches": 1500},
    })
    warnings = [i for i in result.issues if i.severity == "warning"]
    assert any(i.code == "SYNTHESIS_QUOTA_OVER_CEILING" for i in warnings)
    msg = warnings[0].message
    assert "4000" in msg


def test_critical_severity_when_explicitly_set(tmp_path: Path) -> None:
    course = tmp_path / "courses" / "huge2"
    _write_chunks(course, 1000)
    result = SynthesisQuotaValidator().validate({
        "course_dir": str(course),
        "instruction_variants_per_chunk": 3,
        "thresholds": {"max_estimated_dispatches": 1500},
        "severity": "critical",
    })
    assert result.passed is False
    crits = [i for i in result.issues if i.severity == "critical"]
    assert any(i.code == "SYNTHESIS_QUOTA_OVER_CEILING" for i in crits)


def test_missing_chunks_passes_gracefully(tmp_path: Path) -> None:
    """No chunks file = nothing to estimate; validator no-ops."""
    course = tmp_path / "courses" / "empty"
    course.mkdir(parents=True)
    result = SynthesisQuotaValidator().validate({
        "course_dir": str(course),
    })
    assert result.passed is True


def test_default_ceiling_is_1500(tmp_path: Path) -> None:
    course = tmp_path / "courses" / "right-at-default"
    # 800 chunks × (1 variant + 1 pref) = 1600 — over default 1500.
    _write_chunks(course, 800)
    result = SynthesisQuotaValidator().validate({
        "course_dir": str(course),
        "instruction_variants_per_chunk": 1,
    })
    warnings = [i for i in result.issues if i.severity == "warning"]
    assert any(i.code == "SYNTHESIS_QUOTA_OVER_CEILING" for i in warnings)
