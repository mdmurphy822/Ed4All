"""Wave 70 — SHACL gate on LibV2 manifest import.

Covers:

* Well-formed manifest → conforms, no warnings, no raise.
* Malformed manifest + strict env → raises ValueError with report.
* Malformed manifest + lenient env → logs a WARNING, import proceeds.
* Missing pyld/pyshacl/rdflib → logs an INFO, import skipped cleanly.

Mirrors the ``pytest.importorskip`` pattern from
``schemas/tests/test_courseforge_shacl_shapes.py`` so this file is
hermetic when the RDF toolchain isn't installed.
"""

from __future__ import annotations

import logging
from unittest import mock

import pytest

# Skip the whole module when the RDF toolchain isn't importable — the
# importer path degrades gracefully in that case, but the SHACL gate
# assertions below require real pyld/pyshacl.
pytest.importorskip(
    "pyld",
    reason="pyld required for SHACL import tests; install with `pip install pyld`.",
)
pytest.importorskip(
    "pyshacl",
    reason="pyshacl required for SHACL import tests; install with `pip install pyshacl`.",
)
pytest.importorskip("rdflib", reason="rdflib comes with pyshacl.")


from LibV2.tools.libv2.importer import (  # noqa: E402
    _shacl_validate_manifest,
)


# -------------------------------------------------------------------- #
# Well-formed payload conforms cleanly
# -------------------------------------------------------------------- #


def _well_formed_course_module() -> dict:
    """Minimal CourseModule payload that passes every required SHACL shape."""
    return {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
    }


def test_well_formed_manifest_passes_lenient():
    conforms, report = _shacl_validate_manifest(
        _well_formed_course_module(), strict=False,
    )
    assert conforms is True
    # Report is SHACL's textual payload — must not contain "Violation".
    assert "Violation" not in report


def test_well_formed_manifest_passes_strict():
    # strict=True on a conforming payload must NOT raise.
    conforms, report = _shacl_validate_manifest(
        _well_formed_course_module(), strict=True,
    )
    assert conforms is True


# -------------------------------------------------------------------- #
# Malformed payloads
# -------------------------------------------------------------------- #


def _malformed_missing_course_code() -> dict:
    """CourseModule without the required schema:courseCode — fails
    cfshapes:CourseModuleShape's minCount=1 constraint."""
    return {
        "@type": "CourseModule",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
    }


def _malformed_bad_bloom_level() -> dict:
    """LearningObjective with a bloomLevel IRI outside the closed set.
    Wave 67's sh:in check catches this — the Wave 63 pattern check would
    have accepted it."""
    return {
        "@type": "LearningObjective",
        "statement": "Apply concepts to solve problems",
        # Out-of-vocab IRI — cfshapes:LearningObjectiveShape's sh:in must fail.
        "bloomLevel": "https://ed4all.dev/vocab/bloom#aplly",
    }


def test_malformed_manifest_strict_raises():
    with pytest.raises(ValueError) as excinfo:
        _shacl_validate_manifest(_malformed_missing_course_code(), strict=True)
    assert "SHACL" in str(excinfo.value)
    assert "failed" in str(excinfo.value).lower() or "strict" in str(excinfo.value).lower()


def test_malformed_manifest_lenient_warns(caplog):
    caplog.set_level(logging.WARNING, logger="LibV2.tools.libv2.importer")
    conforms, report = _shacl_validate_manifest(
        _malformed_missing_course_code(), strict=False,
    )
    assert conforms is False
    assert "Violation" in report or "violation" in report.lower()
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_records, (
        "Expected a WARNING-level log for non-conforming manifest in lenient "
        "mode, got nothing."
    )
    joined = " ".join(r.getMessage() for r in warning_records)
    assert "SHACL" in joined


def test_malformed_bloom_level_strict_raises():
    """Wave 67 sh:in check must fire on typo'd bloom IRIs."""
    with pytest.raises(ValueError):
        _shacl_validate_manifest(_malformed_bad_bloom_level(), strict=True)


# -------------------------------------------------------------------- #
# Missing RDF deps → skip, not crash
# -------------------------------------------------------------------- #


def test_missing_deps_logs_info_and_skips(caplog):
    """When pyld/pyshacl/rdflib aren't importable, the validator raises
    ShaclDepsMissing and ``_shacl_validate_manifest`` degrades to a
    conforms=True skip with an INFO log. The import flow must not
    hard-fail."""
    from LibV2.tools.libv2 import _shacl_validator

    caplog.set_level(logging.INFO, logger="LibV2.tools.libv2.importer")

    # Patch _ensure_deps to raise ShaclDepsMissing so we simulate the
    # bare-install path without actually uninstalling the packages.
    with mock.patch.object(
        _shacl_validator,
        "_ensure_deps",
        side_effect=_shacl_validator.ShaclDepsMissing("simulated"),
    ):
        conforms, report = _shacl_validate_manifest(
            _well_formed_course_module(), strict=True,
        )

    # strict=True must still not raise when deps are missing — the
    # validator skipped.
    assert conforms is True
    assert "skipped" in report.lower() or "missing" in report.lower()
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info_records, (
        "Expected an INFO log indicating SHACL validation was skipped due "
        "to missing deps."
    )
    joined = " ".join(r.getMessage() for r in info_records)
    assert "SHACL" in joined and ("skipped" in joined.lower() or "missing" in joined.lower())
