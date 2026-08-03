"""W8.1 regression — the calibration-gated severity flip has a REAL input.

Before W8.1 the flip read ``trainforge_assessment_quality_report.json::
summary.answerability_alignment_rate``, a field NO producer ever wrote
(the schema-locked ``summary`` has no such key), so the flip was dead —
it could only ever defer. W8.1 (a) makes
``AssessmentRetrievalGroundingValidator`` emit the alignment rate and
(b) teaches ``resolve_severity_flip`` to fall back to the schema-present
``answerable_rate`` sibling (same retrieval-grounding score). These tests
pin the fallback semantics + prove the flip stays calibration-gated.

All fixtures synthesized under a tmp LibV2 root — no real course slug.
"""
from __future__ import annotations

import json
from pathlib import Path

from lib.governance.calibration_gate import resolve_severity_flip


def _write_report(root: Path, summary: dict, slug: str = "calibration-fixture") -> None:
    d = root / "courses" / slug / "quality"
    d.mkdir(parents=True, exist_ok=True)
    (d / "trainforge_assessment_quality_report.json").write_text(
        json.dumps({"schema_version": "1.0", "summary": summary}),
        encoding="utf-8",
    )


def test_fallback_reads_answerable_rate_when_canonical_absent(tmp_path):
    # Only the schema-present sibling is populated (the real-world case).
    _write_report(tmp_path, {"answerable_rate": 0.92, "source_support_rate": 0.92})
    apply_flip, payload = resolve_severity_flip(
        calibration_signal="answerability_alignment_rate",
        threshold=0.85,
        libv2_root=tmp_path,
        fallback_signals=("answerable_rate",),
    )
    assert apply_flip is True
    assert payload["decision_type"] == "severity_flip_applied"
    # Records the field actually read + the canonical name requested.
    assert payload["ml_features"]["calibration_signal"] == "answerable_rate"
    assert payload["ml_features"]["requested_signal"] == (
        "answerability_alignment_rate"
    )
    assert payload["ml_features"]["observed_rate"] == 0.92


def test_canonical_signal_still_wins_over_fallback(tmp_path):
    # When the canonical field IS present it takes precedence (test_6 path).
    _write_report(
        tmp_path,
        {"answerability_alignment_rate": 0.90, "answerable_rate": 0.10},
    )
    apply_flip, payload = resolve_severity_flip(
        calibration_signal="answerability_alignment_rate",
        threshold=0.85,
        libv2_root=tmp_path,
        fallback_signals=("answerable_rate",),
    )
    assert apply_flip is True
    assert payload["ml_features"]["calibration_signal"] == (
        "answerability_alignment_rate"
    )
    assert payload["ml_features"]["observed_rate"] == 0.90


def test_flip_stays_gated_below_floor_via_fallback(tmp_path):
    # Fallback value below the floor → flip still holds (calibration-gated).
    _write_report(tmp_path, {"answerable_rate": 0.40})
    apply_flip, payload = resolve_severity_flip(
        calibration_signal="answerability_alignment_rate",
        threshold=0.85,
        libv2_root=tmp_path,
        fallback_signals=("answerable_rate",),
    )
    assert apply_flip is False
    assert payload["ml_features"]["reason"] == "alignment_rate_below_floor"


def test_default_no_fallback_is_byte_identical(tmp_path):
    # Without opting into a fallback, a report carrying ONLY answerable_rate
    # defers exactly as before W8.1 (single-signal read).
    _write_report(tmp_path, {"answerable_rate": 0.99})
    apply_flip, payload = resolve_severity_flip(
        calibration_signal="answerability_alignment_rate",
        threshold=0.85,
        libv2_root=tmp_path,
    )
    assert apply_flip is False
    assert payload["ml_features"]["reason"] == "calibration_signal_missing"


def test_missing_report_defers(tmp_path):
    apply_flip, payload = resolve_severity_flip(
        calibration_signal="answerability_alignment_rate",
        threshold=0.85,
        libv2_root=tmp_path,
        fallback_signals=("answerable_rate",),
    )
    assert apply_flip is False
    assert payload["ml_features"]["reason"] == "calibration_report_missing"


def test_bool_value_is_not_a_valid_rate(tmp_path):
    # A JSON ``true`` must NOT be read as 1.0 (bool is an int subclass).
    _write_report(tmp_path, {"answerable_rate": True})
    apply_flip, payload = resolve_severity_flip(
        calibration_signal="answerability_alignment_rate",
        threshold=0.85,
        libv2_root=tmp_path,
        fallback_signals=("answerable_rate",),
    )
    assert apply_flip is False
    assert payload["ml_features"]["reason"] == "calibration_signal_missing"
