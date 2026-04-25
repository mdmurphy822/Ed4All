"""Wave 76 — outcome-minimum rule regression tests.

Pre-Wave-76 ``validate_learning_outcomes`` counted only
``course.json::learning_outcomes[]`` and rejected courses with fewer
than 10 entries. Two failure modes:

  1. course.json was emitted before Wave 75's component-objective merge
     (held only the 7 terminal LOs).
  2. course.json held the full 36 but a later filter dropped entries
     where ``type == "component"``.

The fix counts the union of ``course.json::learning_outcomes[]`` AND
``objectives.json::terminal_outcomes`` + ``objectives.json::
component_objectives``, deduplicated by lowercase ID.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from LibV2.tools.libv2.validator import (
    _count_total_learning_outcomes,
    validate_learning_outcomes,
)


def _write_course_archive(
    tmp_path: Path,
    *,
    course_outcomes: list[dict] | None,
    objectives: dict | None = None,
) -> Path:
    course_dir = tmp_path / "course"
    course_dir.mkdir()
    if course_outcomes is not None:
        (course_dir / "course.json").write_text(
            json.dumps({
                "course_code": "TEST_101",
                "title": "Test Course",
                "learning_outcomes": course_outcomes,
            })
        )
    if objectives is not None:
        (course_dir / "objectives.json").write_text(json.dumps(objectives))
    return course_dir


def _terminal(idx: int) -> dict:
    return {"id": f"to-{idx:02d}", "statement": f"Terminal {idx}", "hierarchy_level": "terminal"}


def _component(idx: int) -> dict:
    return {"id": f"co-{idx:02d}", "statement": f"Component {idx}", "hierarchy_level": "chapter"}


# ---------------------------------------------------------------------------
# _count_total_learning_outcomes
# ---------------------------------------------------------------------------


def test_count_terminals_plus_components_via_course_json(tmp_path: Path) -> None:
    """The Wave-75-aligned shape: course.json carries all 36 in one flat list."""
    archive = _write_course_archive(
        tmp_path,
        course_outcomes=[_terminal(i) for i in range(1, 8)] + [_component(i) for i in range(1, 30)],
    )
    assert _count_total_learning_outcomes(archive) == 36


def test_count_via_objectives_when_course_json_partial(tmp_path: Path) -> None:
    """course.json missing components — objectives.json fills in.

    Mirrors the rdf-shacl-550 mid-migration state: course.json had only
    the 7 terminals, but objectives.json carried the full 7 + 29 split.
    """
    archive = _write_course_archive(
        tmp_path,
        course_outcomes=[_terminal(i) for i in range(1, 8)],  # 7 terminals only
        objectives={
            "schema_version": "wave75",
            "terminal_outcomes": [_terminal(i) for i in range(1, 8)],
            "component_objectives": [_component(i) for i in range(1, 30)],
        },
    )
    assert _count_total_learning_outcomes(archive) == 36


def test_count_dedups_by_id(tmp_path: Path) -> None:
    """A LO present in both files should be counted once."""
    archive = _write_course_archive(
        tmp_path,
        course_outcomes=[_terminal(1), _terminal(2)],
        objectives={
            "terminal_outcomes": [_terminal(1), _terminal(2)],  # duplicates
            "component_objectives": [_component(1)],
        },
    )
    assert _count_total_learning_outcomes(archive) == 3


def test_count_legacy_objectives_keys(tmp_path: Path) -> None:
    """Pre-Wave-75 archives use ``terminal_objectives`` / ``chapter_objectives``."""
    archive = _write_course_archive(
        tmp_path,
        course_outcomes=None,
        objectives={
            "terminal_objectives": [_terminal(i) for i in range(1, 4)],
            "chapter_objectives": [
                {"chapter": "Week 1", "objectives": [_component(1), _component(2)]},
                {"chapter": "Week 2", "objectives": [_component(3)]},
            ],
        },
    )
    # course.json absent → returns 6 because objectives.json is readable.
    assert _count_total_learning_outcomes(archive) == 6


def test_count_returns_none_when_neither_file_exists(tmp_path: Path) -> None:
    course_dir = tmp_path / "empty"
    course_dir.mkdir()
    assert _count_total_learning_outcomes(course_dir) is None


# ---------------------------------------------------------------------------
# validate_learning_outcomes — minimum-10 gate
# ---------------------------------------------------------------------------


def test_minimum_passes_with_36_total_outcomes(tmp_path: Path) -> None:
    """7 terminals + 29 components = 36 total. Passes minimum-10."""
    archive = _write_course_archive(
        tmp_path,
        course_outcomes=[_terminal(i) for i in range(1, 8)] + [_component(i) for i in range(1, 30)],
    )
    result = validate_learning_outcomes(archive)
    assert result.valid, f"errors: {result.errors}"
    # Should not produce a "minimum is 10" error
    assert not any("minimum is 10" in e for e in result.errors)


def test_minimum_fails_with_9_total_outcomes(tmp_path: Path) -> None:
    """5 terminals + 4 components = 9 total. Below minimum-10."""
    archive = _write_course_archive(
        tmp_path,
        course_outcomes=[_terminal(i) for i in range(1, 6)] + [_component(i) for i in range(1, 5)],
    )
    result = validate_learning_outcomes(archive)
    assert not result.valid
    assert any("minimum is 10" in e for e in result.errors)


def test_minimum_legacy_archive_with_only_7_terminals_still_fails(tmp_path: Path) -> None:
    """Documents the legacy expected behavior: an archive that genuinely
    has 7 terminals and 0 components should still fail the minimum-10
    rule. The Wave 76 fix only avoids rejecting GOOD archives — it does
    not relax the threshold itself.
    """
    archive = _write_course_archive(
        tmp_path,
        course_outcomes=[_terminal(i) for i in range(1, 8)],
    )
    result = validate_learning_outcomes(archive)
    assert not result.valid
    assert any("minimum is 10" in e for e in result.errors)


def test_minimum_uses_objectives_when_course_json_partial(tmp_path: Path) -> None:
    """The exact rdf-shacl-550 case: course.json had only 7, but
    objectives.json had the full 36. The validator must consult both
    and pass.
    """
    archive = _write_course_archive(
        tmp_path,
        course_outcomes=[_terminal(i) for i in range(1, 8)],
        objectives={
            "schema_version": "wave75",
            "terminal_outcomes": [_terminal(i) for i in range(1, 8)],
            "component_objectives": [_component(i) for i in range(1, 30)],
        },
    )
    result = validate_learning_outcomes(archive)
    assert result.valid, f"errors: {result.errors}"


def test_minimum_passes_above_30_no_warning_under_60(tmp_path: Path) -> None:
    """Wave 76 — 36 LOs is now expected; the legacy ``recommended max 25``
    warning bumped to 60. 36 should not warn.
    """
    archive = _write_course_archive(
        tmp_path,
        course_outcomes=[_terminal(i) for i in range(1, 8)] + [_component(i) for i in range(1, 30)],
    )
    result = validate_learning_outcomes(archive)
    assert result.valid
    assert not any("recommended max" in w for w in result.warnings)


def test_minimum_warns_only_above_60(tmp_path: Path) -> None:
    archive = _write_course_archive(
        tmp_path,
        course_outcomes=[_terminal(i) for i in range(1, 65)],  # 64 outcomes
    )
    result = validate_learning_outcomes(archive)
    assert any("recommended max" in w for w in result.warnings)
