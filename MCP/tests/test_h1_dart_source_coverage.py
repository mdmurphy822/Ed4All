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
def test_h1_fanout_coverage_pct_never_exceeds_one() -> None:
    """Regression: a FAN-OUT stage (one block → many chunks) must NOT
    emit ``coverage_pct > 1``.

    On a coarse-block full-book corpus one page-level DART block splits
    into several chunks, so ``emitted_count`` (chunks) exceeds
    ``consumed_count`` (blocks). The naive ``emitted / consumed`` form
    overshoots the schema ``maximum: 1`` (observed in the field:
    768 chunks / 661 blocks = 1.161876, which BLOCKED the
    ``chunkset_manifest`` gate). ``coverage_pct`` is the covered-share
    ``(consumed - dropped) / consumed``, bounded ``[0, 1]``: with zero
    drops every block was covered, so coverage is exactly ``1.0``.
    """
    block = build_source_coverage(
        consumed_count=661,
        emitted_count=768,  # fan-out: more chunks than blocks
        drop_reasons=None,
        dropped_count=0,
        label="dart_chunking",
    )
    assert block["coverage_pct"] == pytest.approx(1.0, rel=1e-9)
    assert 0.0 <= block["coverage_pct"] <= 1.0, block
    # The emitted shape must still validate against the schema
    # (coverage_pct maximum is 1).
    manifest = {
        "chunks_sha256": "c" * 64,
        "chunker_version": "v4",
        "chunkset_kind": "dart",
        "source_dart_html_sha256": "d" * 64,
        "chunks_count": 768,
        "generated_at": "2026-07-01T00:00:00Z",
        "source_coverage": block,
    }
    _validate_against_schema(manifest)


@pytest.mark.unit
def test_h1_fanout_with_boilerplate_drops_bounds_coverage() -> None:
    """A fan-out stage that also drops some blocks reports covered-share
    < 1 (dropped blocks were not covered), still bounded ``[0, 1]``."""
    # 661 blocks in, 768 chunks out, but 61 blocks dropped as boilerplate.
    # dropped_count passed as 0 (emitted > consumed → naive delta clamps to
    # 0), but the boilerplate histogram over-attributes, so the helper
    # reconciles dropped up to 61 → covered = 600 → 600/661 ≈ 0.907716.
    block = build_source_coverage(
        consumed_count=661,
        emitted_count=768,
        drop_reasons={"boilerplate": 61},
        dropped_count=0,
        label="dart_chunking",
    )
    assert block["dropped_count"] == 61
    assert block["coverage_pct"] == pytest.approx(600 / 661, rel=1e-6)
    assert 0.0 <= block["coverage_pct"] <= 1.0, block


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


# Minimal DART HTML fixture: two content sections with non-trivial
# paragraphs so the HTML parser yields real ContentSections and the
# chunker emits at least one chunk. Wave1-I2 made ``_run_dart_chunking``
# fail-closed on an empty staging dir (RuntimeError — it refuses to write
# an empty-bytes shell over a possibly-real prior artifact), so the
# end-to-end source_coverage emit MUST run against non-empty input. The
# companion test-inversion commit (c8ea9a2) flipped the other empty-input
# tests but missed this one; seeding a real HTML file is the fix that
# keeps the W3.H schema-shape coverage alive against live chunker output.
_DART_HTML_FIXTURE = """<!DOCTYPE html>
<html><head><title>Intro to RDF</title></head><body>
<h1>Introduction to RDF</h1>
<section><h2>What is a Triple</h2>
<p>An RDF triple is composed of a subject, a predicate, and an object.
The subject denotes the resource being described, while the predicate
denotes a trait or aspect of the resource and expresses a relationship
between the subject and the object. This paragraph is long enough to
clear any non-trivial word minimum the chunker enforces.</p>
</section>
<section><h2>Serialization Formats</h2>
<p>RDF data can be serialized in several formats including Turtle,
N-Triples, JSON-LD, and RDF/XML. Each format has tradeoffs in human
readability and machine parsing efficiency, and choosing among them
depends on the downstream tooling and intended audience.</p>
</section>
</body></html>"""


@pytest.mark.unit
def test_h1_run_dart_chunking_emits_source_coverage(tmp_path: Path) -> None:
    """End-to-end: the helper writes a manifest carrying ``source_coverage``.

    Seeds ONE minimal DART HTML file into the staging dir so the helper
    runs against non-empty input (Wave1-I2 fail-closes on an empty
    staging dir). The W3.H ``source_coverage`` block must emit with the
    canonical 5 fields, the dropped-count invariant holding against live
    chunker output, and ``coverage_pct == emitted/consumed``.
    """
    pytest.importorskip("Trainforge.parsers.html_content_parser")
    from MCP.tools.pipeline_tools import _build_tool_registry

    registry = _build_tool_registry()
    runner = registry.get("run_dart_chunking")
    assert runner is not None, "run_dart_chunking should be registered"

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "intro_rdf.html").write_text(
        _DART_HTML_FIXTURE, encoding="utf-8",
    )
    libv2_root = tmp_path / "libv2"
    libv2_root.mkdir()

    import asyncio

    envelope = asyncio.run(runner(
        course_name="TEST_H1_COURSE",
        staging_dir=str(staging),
        libv2_root=str(libv2_root),
    ))
    payload = json.loads(envelope)
    assert payload.get("success") is True, payload
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
    # Non-empty input — at least one block consumed and at least one
    # chunk emitted from the seeded HTML.
    assert block["consumed_count"] >= 1, block
    assert block["emitted_count"] >= 1, block
    # Canonical invariant: dropped_count == sum(drop_reasons.values()).
    assert block["dropped_count"] == sum(block["drop_reasons"].values()), block
    # coverage_pct is emitted/consumed (rounded), within (0, 1].
    expected_pct = round(
        block["emitted_count"] / block["consumed_count"], 6
    )
    assert block["coverage_pct"] == pytest.approx(expected_pct, rel=1e-3), block
    assert 0.0 < block["coverage_pct"] <= 1.0, block
