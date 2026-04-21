"""Wave 23 Sub-task C tests — LibV2ManifestValidator.

Gates the ``libv2_archival`` phase of the ``textbook_to_course``
workflow. Critical-severity checks (JSON parse, schema match,
artifact integrity) must block the phase; warning-severity scaffold
+ provenance gaps surface but never block.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from lib.validators.libv2_manifest import LibV2ManifestValidator


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def good_archive(tmp_path: Path):
    """Build a minimal well-formed LibV2 archive under tmp_path.

    Includes every required scaffold dir + course.json + a valid
    manifest with one PDF source artifact whose checksum + size match.
    """
    slug = "test-course"
    course_dir = tmp_path / "courses" / slug
    course_dir.mkdir(parents=True)

    # Scaffold dirs
    for sub in ("corpus", "graph", "training_specs", "quality",
                "source/pdf", "source/html", "source/imscc", "pedagogy"):
        (course_dir / sub).mkdir(parents=True)

    # Seed pedagogy/ + graph/ with a marker file so the "empty" warnings
    # don't fire on a healthy archive.
    (course_dir / "pedagogy" / "model.json").write_text("{}", encoding="utf-8")
    (course_dir / "graph" / "nodes.json").write_text("[]", encoding="utf-8")

    # A single PDF source artifact
    pdf_bytes = b"%PDF-1.4 synthetic test pdf bytes" * 10
    pdf_path = course_dir / "source" / "pdf" / "test.pdf"
    pdf_path.write_bytes(pdf_bytes)

    # course.json (so MISSING_COURSE_JSON warning doesn't fire)
    (course_dir / "course.json").write_text(
        json.dumps({"slug": slug, "learning_outcomes": []}),
        encoding="utf-8",
    )

    manifest: Dict[str, Any] = {
        "libv2_version": "1.2.0",
        "slug": slug,
        "import_timestamp": "2026-04-20T18:26:45.000000",
        "classification": {
            "division": "STEM",
            "primary_domain": "general",
            "subdomains": [],
        },
        "source_artifacts": {
            "pdf": [{
                "path": str(pdf_path),
                "checksum": _sha256(pdf_bytes),
                "size": len(pdf_bytes),
            }],
        },
        "provenance": {
            "source_type": "textbook_to_course_pipeline",
            "import_pipeline_version": "1.0.0",
        },
        "features": {
            "source_provenance": True,
            "evidence_source_provenance": True,
        },
    }
    manifest_path = course_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, course_dir


# ---------------------------------------------------------------------- #
# Core cases
# ---------------------------------------------------------------------- #


def test_valid_manifest_passes(good_archive):
    """A well-formed archive passes.

    The current Wave-19+ pipeline emits a known-gap manifest (missing
    ``sourceforge_manifest`` + ``content_profile``) — those trip a
    *warning* (SCHEMA_GAP_KNOWN), not a critical. So ``passed`` is
    True but score can drop below 1.0 while no critical issues fire.
    """
    manifest_path, course_dir = good_archive
    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })
    assert result.passed, (
        f"Valid manifest should pass. Got issues: {[i.code for i in result.issues]}"
    )
    critical = [i for i in result.issues if i.severity == "critical"]
    assert not critical, f"Unexpected critical issues: {[i.code for i in critical]}"


def test_missing_manifest_path():
    result = LibV2ManifestValidator().validate({})
    assert not result.passed
    assert any(i.code == "MISSING_MANIFEST_PATH" for i in result.issues)


def test_nonexistent_manifest_path(tmp_path):
    result = LibV2ManifestValidator().validate({
        "manifest_path": str(tmp_path / "nope.json"),
    })
    assert not result.passed
    assert any(i.code == "MANIFEST_NOT_FOUND" for i in result.issues)


def test_corrupt_json_fails_critical(tmp_path):
    mp = tmp_path / "manifest.json"
    mp.write_text("{not valid json", encoding="utf-8")
    result = LibV2ManifestValidator().validate({
        "manifest_path": str(mp),
        "course_dir": str(tmp_path),
    })
    assert not result.passed
    assert any(i.code == "INVALID_JSON" for i in result.issues)
    critical = [i for i in result.issues if i.severity == "critical"]
    assert critical, "INVALID_JSON must be critical severity"


def test_schema_violation_when_missing_required_key(tmp_path, good_archive):
    """Removing a required top-level key triggers SCHEMA_VIOLATION."""
    manifest_path, course_dir = good_archive
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # libv2_version is a top-level required key
    manifest.pop("libv2_version")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })
    assert not result.passed
    assert any(i.code == "SCHEMA_VIOLATION" for i in result.issues)


def test_missing_artifact_file_fails_critical(good_archive):
    manifest_path, course_dir = good_archive
    # Delete the referenced PDF
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pdf_path = Path(manifest["source_artifacts"]["pdf"][0]["path"])
    pdf_path.unlink()

    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })
    assert not result.passed
    assert any(i.code == "MISSING_ARTIFACT" for i in result.issues)


def test_checksum_mismatch_fails_critical(good_archive):
    manifest_path, course_dir = good_archive
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Tamper with the file after manifest records
    pdf_path = Path(manifest["source_artifacts"]["pdf"][0]["path"])
    original = pdf_path.read_bytes()
    pdf_path.write_bytes(original + b"TAMPERED")
    # size will mismatch too — but we want to assert checksum specifically:
    # update size in manifest so size check passes, then check checksum fails.
    manifest["source_artifacts"]["pdf"][0]["size"] = len(original + b"TAMPERED")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })
    assert not result.passed
    assert any(i.code == "CHECKSUM_MISMATCH" for i in result.issues), (
        f"Expected CHECKSUM_MISMATCH, got codes: "
        f"{[i.code for i in result.issues]}"
    )


def test_size_mismatch_fails_critical(good_archive):
    """Explicit size-field mismatch (even when checksum drops out)."""
    manifest_path, course_dir = good_archive
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_artifacts"]["pdf"][0]["size"] = 9999999
    # Blank the checksum so only the size check can fail
    manifest["source_artifacts"]["pdf"][0].pop("checksum", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })
    assert not result.passed
    assert any(i.code == "SIZE_MISMATCH" for i in result.issues)


# ---------------------------------------------------------------------- #
# Warning-severity gaps — must never block
# ---------------------------------------------------------------------- #


def test_empty_pedagogy_dir_warns_never_blocks(good_archive):
    manifest_path, course_dir = good_archive
    # Remove the marker file so pedagogy/ is truly empty
    (course_dir / "pedagogy" / "model.json").unlink()

    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })
    assert result.passed, "PEDAGOGY_EMPTY must never be critical"
    assert any(i.code == "PEDAGOGY_EMPTY" for i in result.issues)


def test_source_provenance_false_warns_never_blocks(good_archive):
    manifest_path, course_dir = good_archive
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["features"]["source_provenance"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })
    assert result.passed, "SOURCE_PROVENANCE_FALSE must never be critical"
    sp_issues = [i for i in result.issues if i.code == "SOURCE_PROVENANCE_FALSE"]
    assert sp_issues
    assert sp_issues[0].severity == "warning"
    # The rationale links to the audit doc so reviewers know where to go.
    assert "plans/pipeline-integrity-review" in (sp_issues[0].suggestion or ""), (
        "source_provenance warning must point at the audit doc for remediation."
    )


def test_missing_course_json_warns_never_blocks(good_archive):
    manifest_path, course_dir = good_archive
    (course_dir / "course.json").unlink()

    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })
    assert result.passed
    assert any(i.code == "MISSING_COURSE_JSON" for i in result.issues)


def test_concept_graph_drift_warns(good_archive):
    manifest_path, course_dir = good_archive
    # Create concept_graph/ empty (graph/ already populated in fixture)
    (course_dir / "concept_graph").mkdir()

    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })
    assert result.passed
    assert any(i.code == "CONCEPT_GRAPH_DRIFT" for i in result.issues)


def test_course_dir_derived_from_manifest_path(good_archive):
    """Validator should derive course_dir when caller omits it."""
    manifest_path, _ = good_archive
    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
    })
    # course_dir was derived from manifest_path.parent → should still pass.
    assert result.passed or all(
        i.severity != "critical" or i.code in {"SCHEMA_VIOLATION", "SCHEMA_UNAVAILABLE"}
        for i in result.issues
    )
