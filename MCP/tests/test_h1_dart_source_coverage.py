"""W3.H sub-task H1 — DART chunkset ``source_coverage`` emit.

Pins the canonical W3.H ``source_coverage`` block on the DART
``manifest.json`` sidecar emitted by ``_run_dart_chunking``. Three
tests:

* canonical shape (5 fields present, coverage_pct calc, dropped_count
  == sum invariant);
* ``INTERNAL_DROP_REASON_MISSING`` fires when drop reasons don't
  balance;
* back-compat — a legacy chunkset_manifest without ``source_coverage``
  still validates against the schema (the field is optional).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.governance.source_coverage import (  # noqa: E402
    INTERNAL_DROP_REASON_MISSING,
    build_source_coverage,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "library" / "chunkset_manifest.schema.json"


def _validate_against_schema(doc: dict) -> None:
    """Minimal Draft-07 validation via jsonschema; soft-skip when missing."""
    pytest.importorskip("jsonschema")
    from jsonschema import Draft7Validator

    schema = json.loads(SCHEMA_PATH.read_text())
    Draft7Validator.check_schema(schema)
    Draft7Validator(schema).validate(doc)


@pytest.mark.unit
def test_h1_source_coverage_canonical_shape() -> None:
    """The canonical 5 fields exist and ``coverage_pct`` is correct."""
    block = build_source_coverage(
        consumed_count=10,
        emitted_count=8,
        drop_reasons={"boilerplate": 1, "image_only": 1},
        dropped_count=2,
    )
    assert set(block.keys()) == {
        "consumed_count",
        "emitted_count",
        "dropped_count",
        "drop_reasons",
        "coverage_pct",
    }
    assert block["consumed_count"] == 10
    assert block["emitted_count"] == 8
    assert block["dropped_count"] == 2
    assert block["coverage_pct"] == pytest.approx(0.8, rel=1e-3)
    # Invariant: dropped_count == sum(drop_reasons.values()).
    assert block["dropped_count"] == sum(block["drop_reasons"].values())


@pytest.mark.unit
def test_h1_internal_drop_reason_missing_fires(caplog: pytest.LogCaptureFixture) -> None:
    """A drop without a reason augments the histogram with the canonical bucket."""
    import logging

    caplog.set_level(logging.WARNING, logger="lib.governance.source_coverage")
    block = build_source_coverage(
        consumed_count=10,
        emitted_count=4,
        drop_reasons={"boilerplate": 1},  # only 1 attributable drop, but 6 dropped
        dropped_count=6,
        label="dart_chunking_test",
    )
    # The missing-reason bucket fires for the unattributed delta (6 - 1 = 5).
    assert INTERNAL_DROP_REASON_MISSING in block["drop_reasons"]
    assert block["drop_reasons"][INTERNAL_DROP_REASON_MISSING] == 5
    assert block["dropped_count"] == sum(block["drop_reasons"].values())
    # The warning includes the operator-actionable context.
    warning_lines = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("dart_chunking_test" in m for m in warning_lines)
    assert any(INTERNAL_DROP_REASON_MISSING in m for m in warning_lines)


@pytest.mark.unit
def test_h1_zero_consumed_yields_zero_coverage() -> None:
    """``coverage_pct`` is 0.0 (not NaN / divide-by-zero) when consumed == 0."""
    block = build_source_coverage(consumed_count=0, emitted_count=0)
    assert block["coverage_pct"] == 0.0
    assert block["consumed_count"] == 0
    assert block["dropped_count"] == 0


@pytest.mark.unit
def test_h1_legacy_manifest_back_compat() -> None:
    """A pre-W3.H manifest (no ``source_coverage`` field) still validates."""
    legacy = {
        "chunks_sha256": "0" * 64,
        "chunker_version": "v4",
        "chunkset_kind": "dart",
        "source_dart_html_sha256": "1" * 64,
    }
    _validate_against_schema(legacy)


@pytest.mark.unit
def test_h1_manifest_with_source_coverage_validates() -> None:
    """A new W3.H-shaped manifest validates against the extended schema."""
    block = build_source_coverage(
        consumed_count=5,
        emitted_count=4,
        drop_reasons={"boilerplate": 1},
        dropped_count=1,
    )
    manifest = {
        "chunks_sha256": "a" * 64,
        "chunker_version": "v4",
        "chunkset_kind": "dart",
        "source_dart_html_sha256": "b" * 64,
        "chunks_count": 4,
        "generated_at": "2026-05-06T12:00:00Z",
        "source_coverage": block,
    }
    _validate_against_schema(manifest)


@pytest.mark.unit
def test_h1_run_dart_chunking_emits_source_coverage(tmp_path: Path) -> None:
    """End-to-end: the helper writes a manifest carrying ``source_coverage``.

    Drives the helper against an empty staging dir so we exercise the
    fail-soft empty-input shell without needing a real DART HTML
    fixture. The coverage block must still emit (consumed_count=0,
    coverage_pct=0.0).
    """
    pytest.importorskip("Trainforge.parsers.html_content_parser")
    from MCP.tools.pipeline_tools import _build_tool_registry

    registry = _build_tool_registry()
    runner = registry.get("run_dart_chunking")
    assert runner is not None, "run_dart_chunking should be registered"

    staging = tmp_path / "staging"
    staging.mkdir()
    libv2_root = tmp_path / "libv2"
    libv2_root.mkdir()

    import asyncio

    envelope = asyncio.run(runner(
        course_name="TEST_H1_COURSE",
        staging_dir=str(staging),
        libv2_root=str(libv2_root),
    ))
    payload = json.loads(envelope)
    assert payload.get("success") is True
    manifest_path = Path(payload["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "source_coverage" in manifest
    block = manifest["source_coverage"]
    assert set(block.keys()) == {
        "consumed_count",
        "emitted_count",
        "dropped_count",
        "drop_reasons",
        "coverage_pct",
    }
    # Empty-input shell — zero blocks consumed, zero emitted.
    assert block["consumed_count"] == 0
    assert block["emitted_count"] == 0
    assert block["coverage_pct"] == 0.0
