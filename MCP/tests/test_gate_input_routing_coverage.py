"""Wave 29 gate input router coverage tests (Defect 2).

The 5-persona OLSR_SIM_01 run surfaced four gates with runtime issues:

* ``libv2_manifest`` → skipped: missing inputs: manifest_path
* ``assessment_objective_alignment`` → skipped: missing inputs: chunks_path
* ``dart_markers`` → skipped: missing inputs: html_path
* ``assessment_quality`` → CRASH on ``json.loads`` of empty/absent file

Wave 29 closes the coverage gaps:

* ``libv2_manifest`` derives ``manifest_path`` from ``course_dir``.
* ``assessment_objective_alignment`` falls back to the LibV2-archived
  ``course_dir/corpus/chunks.jsonl``.
* ``dart_markers`` picks up batch outputs from ``output_paths`` and
  surfaces ``html_paths[]`` alongside a representative ``html_path``.
* ``assessment_quality`` checks existence + non-empty before handing
  the path off, so a missing / truncated file yields a structured
  skip rather than a JSON-decode crash.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from MCP.hardening.gate_input_routing import (
    _build_assessment_objective_alignment,
    _build_assessment_quality,
    _build_dart_markers,
    _build_libv2_manifest,
    default_router,
)


# --------------------------------------------------------------------- #
# libv2_manifest — derive from course_dir
# --------------------------------------------------------------------- #


def test_libv2_manifest_explicit_path(tmp_path: Path):
    """Explicit manifest_path in phase outputs short-circuits the
    course_dir derivation."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    outputs = {"libv2_archival": {"manifest_path": str(manifest)}}
    inputs, missing = _build_libv2_manifest(outputs, {})
    assert missing == []
    assert inputs["manifest_path"] == str(manifest)


def test_libv2_manifest_derives_from_course_dir(tmp_path: Path):
    """When ``manifest_path`` isn't surfaced but ``course_dir`` is, the
    builder derives ``course_dir/manifest.json`` if it exists."""
    course_dir = tmp_path / "MY_COURSE"
    course_dir.mkdir()
    (course_dir / "manifest.json").write_text('{"course_id": "X"}', encoding="utf-8")

    outputs = {"libv2_archival": {"course_dir": str(course_dir)}}
    inputs, missing = _build_libv2_manifest(outputs, {})
    assert missing == []
    assert inputs["manifest_path"] == str(course_dir / "manifest.json")
    assert inputs["course_dir"] == str(course_dir)


def test_libv2_manifest_skipped_when_no_signals():
    """No manifest_path, no course_dir → structured skip."""
    inputs, missing = _build_libv2_manifest({}, {})
    assert missing == ["manifest_path"]


# --------------------------------------------------------------------- #
# assessment_objective_alignment — chunks fallback
# --------------------------------------------------------------------- #


def test_assessment_objective_alignment_explicit_chunks(tmp_path: Path):
    """Explicit chunks_path short-circuits the LibV2 fallback."""
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("{}\n", encoding="utf-8")
    assessments = tmp_path / "assessments.json"
    assessments.write_text("{}", encoding="utf-8")

    outputs = {
        "trainforge_assessment": {
            "output_path": str(assessments),
            "chunks_path": str(chunks),
        },
    }
    inputs, missing = _build_assessment_objective_alignment(outputs, {})
    assert missing == []
    assert inputs["assessments_path"] == str(assessments)
    assert inputs["chunks_path"] == str(chunks)


def test_assessment_objective_alignment_falls_back_to_libv2_archive(tmp_path: Path):
    """When chunks_path isn't surfaced and the assessments path has no
    sibling corpus/chunks.jsonl, the builder pulls from the archived
    ``course_dir/corpus/chunks.jsonl``."""
    # Assessments in a lonely dir with no corpus sibling.
    tf_dir = tmp_path / "trainforge_run"
    tf_dir.mkdir()
    assessments = tf_dir / "assessments.json"
    assessments.write_text("{}", encoding="utf-8")

    # LibV2-archived course dir with the chunks.
    archive_dir = tmp_path / "LibV2" / "courses" / "archived_course"
    (archive_dir / "corpus").mkdir(parents=True)
    chunks = archive_dir / "corpus" / "chunks.jsonl"
    chunks.write_text('{"chunk_id": "c1"}\n', encoding="utf-8")

    outputs = {
        "trainforge_assessment": {"output_path": str(assessments)},
        "libv2_archival": {"course_dir": str(archive_dir)},
    }
    inputs, missing = _build_assessment_objective_alignment(outputs, {})
    assert missing == []
    assert inputs["chunks_path"] == str(chunks)


def test_assessment_objective_alignment_skipped_without_chunks(tmp_path: Path):
    """Assessments found but chunks nowhere → structured skip."""
    tf_dir = tmp_path / "tf"
    tf_dir.mkdir()
    assessments = tf_dir / "assessments.json"
    assessments.write_text("{}", encoding="utf-8")

    outputs = {"trainforge_assessment": {"output_path": str(assessments)}}
    inputs, missing = _build_assessment_objective_alignment(outputs, {})
    assert missing == ["chunks_path"]
    assert inputs["assessments_path"] == str(assessments)


# --------------------------------------------------------------------- #
# dart_markers — batch-aware html_paths
# --------------------------------------------------------------------- #


def test_dart_markers_single_html(tmp_path: Path):
    """Single-file DART output still returns a representative html_path."""
    html = tmp_path / "out.html"
    html.write_text("<html></html>", encoding="utf-8")

    outputs = {"dart_conversion": {"output_path": str(html)}}
    inputs, missing = _build_dart_markers(outputs, {})
    assert missing == []
    assert inputs["html_path"] == str(html)
    assert inputs["html_paths"] == [str(html)]


def test_dart_markers_batch_html(tmp_path: Path):
    """Comma-joined ``output_paths`` surfaces as a list under
    ``html_paths``."""
    a = tmp_path / "chapter1.html"
    b = tmp_path / "chapter2.html"
    c = tmp_path / "chapter3.html"
    for p in (a, b, c):
        p.write_text("<html></html>", encoding="utf-8")

    outputs = {
        "dart_conversion": {"output_paths": f"{a},{b},{c}"},
    }
    inputs, missing = _build_dart_markers(outputs, {})
    assert missing == []
    assert inputs["html_path"] == str(a)
    assert inputs["html_paths"] == [str(a), str(b), str(c)]


def test_dart_markers_skipped_on_no_html():
    """No DART outputs anywhere → structured skip, not a crash."""
    inputs, missing = _build_dart_markers({}, {})
    assert missing == ["html_path"]


# --------------------------------------------------------------------- #
# assessment_quality — missing / empty file handling
# --------------------------------------------------------------------- #


def test_assessment_quality_valid_nonempty(tmp_path: Path):
    """Well-formed non-empty path short-circuits to the happy path."""
    p = tmp_path / "assessments.json"
    p.write_text('{"questions": []}', encoding="utf-8")

    outputs = {"trainforge_assessment": {"output_path": str(p)}}
    inputs, missing = _build_assessment_quality(outputs, {})
    assert missing == []
    assert inputs["assessment_path"] == str(p)


def test_assessment_quality_missing_path_returns_skip():
    """No candidate path at all → structured skip."""
    inputs, missing = _build_assessment_quality({}, {})
    assert missing == ["ASSESSMENTS_FILE_MISSING"]


def test_assessment_quality_nonexistent_file_returns_skip(tmp_path: Path):
    """Path surfaced but file doesn't exist → skip (no crash)."""
    fake = tmp_path / "never_created.json"
    outputs = {"trainforge_assessment": {"output_path": str(fake)}}
    inputs, missing = _build_assessment_quality(outputs, {})
    assert missing == ["ASSESSMENTS_FILE_MISSING"]


def test_assessment_quality_empty_file_returns_skip(tmp_path: Path):
    """Path exists but the file is empty → skip rather than
    ``json.JSONDecodeError``.

    This is the exact crash Defect 6 surfaced: pre-Wave-29 the
    validator was handed an empty path and crashed on
    ``json.loads``. Under Wave 29 the builder intercepts and emits
    a structured skip reason.
    """
    empty = tmp_path / "empty_assessments.json"
    empty.write_text("", encoding="utf-8")

    outputs = {"trainforge_assessment": {"output_path": str(empty)}}
    inputs, missing = _build_assessment_quality(outputs, {})
    assert missing == ["ASSESSMENTS_FILE_MISSING"]


# --------------------------------------------------------------------- #
# Registry integrity — all four gates reachable through default_router
# --------------------------------------------------------------------- #


def test_default_router_covers_all_defect2_gates():
    """The default router must include builders for every gate Defect 2
    called out."""
    r = default_router()
    must_have = {
        "lib.validators.libv2_manifest.LibV2ManifestValidator",
        "lib.validators.assessment_objective_alignment.AssessmentObjectiveAlignmentValidator",
        "lib.validators.dart_markers.DartMarkersValidator",
        "lib.validators.assessment.AssessmentQualityValidator",
    }
    missing = must_have - set(r.builders.keys())
    assert not missing, f"Missing builders: {missing}"
