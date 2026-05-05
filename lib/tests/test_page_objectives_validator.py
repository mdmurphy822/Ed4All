"""Tests for ``lib.validators.page_objectives.PageObjectivesValidator``.

The audit (page_objectives audit fix) inverts the legacy
``passed=True`` warning-only branch on missing ``objectives_path`` to
fail-closed at critical severity. The gate is wired ``critical`` on
both ``course_generation`` and ``textbook_to_course`` packaging phases
(see ``config/workflows.yaml::validation_gates``); silent-skipping the
LO-specificity check on a critical-severity gate was the silent-
degradation failure mode.
"""

from __future__ import annotations

import json
from pathlib import Path

from lib.validators.page_objectives import PageObjectivesValidator


def test_missing_objectives_path_fails_closed(tmp_path):
    """No ``objectives_path`` AND no ``content_dir/course.json`` →
    fail closed with the named code at critical severity.
    """
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    # Intentionally do NOT emit course.json — this is the silent-degrade
    # failure mode the audit fix targets.
    result = PageObjectivesValidator().validate({
        "content_dir": str(content_dir),
    })
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "PAGE_OBJECTIVES_PATH_MISSING" in codes
    crit = [
        i for i in result.issues
        if i.code == "PAGE_OBJECTIVES_PATH_MISSING"
    ]
    assert crit and crit[0].severity == "critical"
    msg = crit[0].message
    # Operator hint: name the upstream phase + the path that was looked for.
    assert "course.json" in msg
    assert "course_planning" in msg or "packaging" in msg


def test_auto_discover_course_json_still_works(tmp_path):
    """Backward-compat: when ``content_dir/course.json`` exists, the
    auto-discover path engages and the missing-path branch is not hit.
    The downstream ``load_canonical_objectives`` may still reject the
    file — that's outside this test's scope.
    """
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    course_json = content_dir / "course.json"
    course_json.write_text(json.dumps({
        "courseCode": "TEST_101",
        "courseTitle": "Test",
        "terminalObjectives": [],
        "chapterObjectives": [],
    }))
    result = PageObjectivesValidator().validate({
        "content_dir": str(content_dir),
    })
    # We don't assert passed/score (that depends on the downstream
    # helper); we only assert the missing-path branch wasn't taken.
    codes = {i.code for i in result.issues}
    assert "PAGE_OBJECTIVES_PATH_MISSING" not in codes


def test_explicit_objectives_path_engages(tmp_path):
    """Explicit ``objectives_path`` overrides auto-discover and
    bypasses the missing-path branch."""
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    objectives = tmp_path / "synthesized_objectives.json"
    objectives.write_text(json.dumps({
        "courseCode": "TEST_101",
        "courseTitle": "Test",
        "terminalObjectives": [],
        "chapterObjectives": [],
    }))
    result = PageObjectivesValidator().validate({
        "content_dir": str(content_dir),
        "objectives_path": str(objectives),
    })
    codes = {i.code for i in result.issues}
    assert "PAGE_OBJECTIVES_PATH_MISSING" not in codes


def test_explicit_objectives_path_not_found_fails(tmp_path):
    """Explicit path that doesn't exist → existing
    ``OBJECTIVES_FILE_NOT_FOUND`` (unchanged behavior)."""
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    result = PageObjectivesValidator().validate({
        "content_dir": str(content_dir),
        "objectives_path": str(tmp_path / "no_such_file.json"),
    })
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "OBJECTIVES_FILE_NOT_FOUND" in codes


def test_missing_content_dir_fails(tmp_path):
    """Missing ``content_dir`` → existing
    ``MISSING_CONTENT_DIR`` (unchanged)."""
    result = PageObjectivesValidator().validate({})
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "MISSING_CONTENT_DIR" in codes
