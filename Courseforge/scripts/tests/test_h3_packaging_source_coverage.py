"""W3.H sub-task H3 — IMSCC packaging ``source_coverage`` emit.

Pins the canonical W3.H ``source_coverage`` block on the new
``packaging_report.json`` sidecar emitted by
``Courseforge/scripts/packaging/package_multifile_imscc.py::package_imscc``.
Four tests:

* canonical shape (5 fields present, coverage_pct calc, dropped_count
  == sum invariant) on a happy-path build;
* ``outline_filter`` drops surface when ``outline_only=True``;
* ``INTERNAL_DROP_REASON_MISSING`` fires on a forced histogram
  imbalance via the helper;
* back-compat — sidecar emit is opt-in (no path -> no sidecar);
  legacy callers see no behaviour change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import package_multifile_imscc as pmi  # noqa: E402

from lib.governance.source_coverage import (  # noqa: E402
    INTERNAL_DROP_REASON_MISSING,
    build_source_coverage,
)


SCHEMA_PATH = (
    _REPO_ROOT / "schemas" / "library" / "packaging_report.schema.json"
)


def _validate_packaging_report(doc: dict) -> None:
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator

    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(doc)


def _make_minimal_course(content_dir: Path) -> None:
    """Create a 1-week minimum course shell so package_imscc can package it.

    No JSON-LD, no LO contract — passes the LO validator's "no
    learningObjectives" silent-skip path. Files are intentionally
    bare so we don't depend on the rest of Courseforge's content
    pipeline.
    """
    week = content_dir / "week_01"
    week.mkdir(parents=True)
    (week / "week_01_overview.html").write_text(
        "<html><body><h1>Week 1: Intro</h1></body></html>",
        encoding="utf-8",
    )
    (week / "week_01_content.html").write_text(
        "<html><body><h2>Lesson 1</h2></body></html>",
        encoding="utf-8",
    )
    (week / "week_01_summary.html").write_text(
        "<html><body><h2>Recap</h2></body></html>",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_h3_source_coverage_canonical_shape(tmp_path: Path) -> None:
    """Happy path: 3 pages authored, 3 packaged, no drops."""
    content_dir = tmp_path / "content"
    output_imscc = tmp_path / "out.imscc"
    sidecar = tmp_path / "packaging_report.json"
    _make_minimal_course(content_dir)

    pmi.package_imscc(
        content_dir,
        output_imscc,
        course_code="H3_TEST",
        course_title="H3 Test Course",
        skip_validation=True,
        coverage_sidecar_path=sidecar,
    )
    assert sidecar.exists()
    doc = json.loads(sidecar.read_text())
    _validate_packaging_report(doc)
    block = doc["source_coverage"]
    assert set(block.keys()) == {
        "consumed_count",
        "emitted_count",
        "dropped_count",
        "drop_reasons",
        "coverage_pct",
    }
    assert block["consumed_count"] == 3
    assert block["emitted_count"] == 3
    assert block["dropped_count"] == 0
    assert block["coverage_pct"] == pytest.approx(1.0, rel=1e-3)
    assert doc["outline_only"] is False


@pytest.mark.unit
def test_h3_outline_only_filter_drops(tmp_path: Path) -> None:
    """``outline_only=True`` excludes content pages → ``outline_filter`` bucket fires."""
    content_dir = tmp_path / "content"
    output_imscc = tmp_path / "out.imscc"
    sidecar = tmp_path / "packaging_report.json"
    _make_minimal_course(content_dir)

    pmi.package_imscc(
        content_dir,
        output_imscc,
        course_code="H3_OUTLINE",
        course_title="H3 Outline Test",
        skip_validation=True,
        outline_only=True,
        coverage_sidecar_path=sidecar,
    )
    doc = json.loads(sidecar.read_text())
    _validate_packaging_report(doc)
    block = doc["source_coverage"]
    # 3 authored - 1 content page filtered = 2 packaged.
    assert block["consumed_count"] == 3
    assert block["emitted_count"] == 2
    assert block["dropped_count"] == 1
    assert block["drop_reasons"].get("outline_filter") == 1
    assert doc["outline_only"] is True


@pytest.mark.unit
def test_h3_internal_drop_reason_missing_fires() -> None:
    """Force the silent-gaming check via the helper."""
    import logging

    rec: list[logging.LogRecord] = []
    h = logging.Handler()
    h.emit = lambda r: rec.append(r)
    lg = logging.getLogger("lib.governance.source_coverage")
    lg.addHandler(h)
    lg.setLevel(logging.WARNING)
    try:
        block = build_source_coverage(
            consumed_count=10,
            emitted_count=2,
            drop_reasons={"missing_lo": 3},
            dropped_count=8,
            label="package_imscc_test",
        )
    finally:
        lg.removeHandler(h)
    assert block["drop_reasons"][INTERNAL_DROP_REASON_MISSING] == 5
    assert block["dropped_count"] == sum(block["drop_reasons"].values())
    assert any(
        "package_imscc_test" in (r.getMessage() or "")
        for r in rec
    )


@pytest.mark.unit
def test_h3_back_compat_no_sidecar_path(tmp_path: Path) -> None:
    """When ``coverage_sidecar_path=None`` no sidecar is emitted (legacy callers)."""
    content_dir = tmp_path / "content"
    output_imscc = tmp_path / "out.imscc"
    _make_minimal_course(content_dir)

    pmi.package_imscc(
        content_dir,
        output_imscc,
        course_code="H3_LEGACY",
        course_title="H3 Legacy Test",
        skip_validation=True,
    )
    # Final dir should NOT have a packaging_report.json.
    assert not (tmp_path / "packaging_report.json").exists()
    # And the IMSCC must still build successfully.
    assert output_imscc.exists()
