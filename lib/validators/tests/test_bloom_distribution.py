"""Tests for the course-level Bloom-distribution-vs-target-curve gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.validators.bloom_distribution import (
    BloomDistributionValidator,
    DEFAULT_MIN_LOS,
    DEFAULT_TOLERANCE,
    resolve_bloom_distribution,
    resolve_bloom_target,
    resolve_min_los,
    resolve_tolerance,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _FakeCapture:
    def __init__(self) -> None:
        self.calls = []

    def log_decision(self, **kwargs):
        self.calls.append(kwargs)


def _objectives_doc(levels):
    """Build a synthesized-objectives doc with one TO per declared level."""
    tos = []
    for i, lvl in enumerate(levels, start=1):
        entry = {"id": f"TO-{i:02d}", "statement": f"Objective {i}"}
        if lvl is not None:
            entry["bloom_level"] = lvl
        tos.append(entry)
    return {"terminal_objectives": tos, "chapter_objectives": []}


def _write_doc(tmp_path: Path, doc) -> str:
    p = tmp_path / "synthesized_objectives.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


def _codes(result):
    return {i.code for i in result.issues}


# --------------------------------------------------------------------------- #
# Flag off → byte-stable no-op
# --------------------------------------------------------------------------- #


def test_flag_off_byte_identical(monkeypatch, tmp_path):
    monkeypatch.delenv("ED4ALL_BLOOM_DISTRIBUTION", raising=False)
    cap = _FakeCapture()
    path = _write_doc(tmp_path, _objectives_doc(["remember"] * 8))
    result = BloomDistributionValidator(decision_capture=cap).validate(
        {"synthesized_objectives_path": path}
    )
    assert result.passed is True
    assert _codes(result) == {"BLOOM_DISTRIBUTION_DISABLED"}
    assert result.metadata["bloom_distribution"] == {"enabled": False}
    assert cap.calls == []  # no decision event when off


# --------------------------------------------------------------------------- #
# Recall-only → NO_HIGHER_ORDER + RECALL_HEAVY
# --------------------------------------------------------------------------- #


def test_recall_only_fires_no_higher_order_and_recall_heavy(monkeypatch, tmp_path):
    monkeypatch.setenv("ED4ALL_BLOOM_DISTRIBUTION", "1")
    path = _write_doc(
        tmp_path,
        _objectives_doc(["remember", "understand"] * 4),  # 8 LOs, all recall
    )
    result = BloomDistributionValidator().validate(
        {"synthesized_objectives_path": path}
    )
    assert result.passed is True  # warning-day-1
    codes = _codes(result)
    assert "BLOOM_DISTRIBUTION_NO_HIGHER_ORDER" in codes
    assert "BLOOM_DISTRIBUTION_RECALL_HEAVY" in codes


# --------------------------------------------------------------------------- #
# Balanced set within tolerance → no off-target / recall-heavy / top-heavy
# --------------------------------------------------------------------------- #


def test_balanced_set_within_tolerance(monkeypatch, tmp_path):
    monkeypatch.setenv("ED4ALL_BLOOM_DISTRIBUTION", "true")
    # Mirror the default curve proportions over 20 LOs: 3/5/5/4/2/1.
    levels = (
        ["remember"] * 3
        + ["understand"] * 5
        + ["apply"] * 5
        + ["analyze"] * 4
        + ["evaluate"] * 2
        + ["create"] * 1
    )
    path = _write_doc(tmp_path, _objectives_doc(levels))
    result = BloomDistributionValidator().validate(
        {"synthesized_objectives_path": path}
    )
    codes = _codes(result)
    assert "BLOOM_DISTRIBUTION_OFF_TARGET" not in codes
    assert "BLOOM_DISTRIBUTION_RECALL_HEAVY" not in codes
    assert "BLOOM_DISTRIBUTION_TOP_HEAVY" not in codes
    assert "BLOOM_DISTRIBUTION_NO_HIGHER_ORDER" not in codes
    assert result.passed is True


# --------------------------------------------------------------------------- #
# Top-heavy → TOP_HEAVY + OFF_TARGET
# --------------------------------------------------------------------------- #


def test_top_heavy_fires(monkeypatch, tmp_path):
    monkeypatch.setenv("ED4ALL_BLOOM_DISTRIBUTION", "on")
    path = _write_doc(
        tmp_path,
        _objectives_doc(["evaluate", "create"] * 4 + ["apply", "analyze"]),
    )
    result = BloomDistributionValidator().validate(
        {"synthesized_objectives_path": path}
    )
    codes = _codes(result)
    assert "BLOOM_DISTRIBUTION_TOP_HEAVY" in codes
    assert "BLOOM_DISTRIBUTION_OFF_TARGET" in codes


# --------------------------------------------------------------------------- #
# Anti-fabrication: null/absent bloom never imputed
# --------------------------------------------------------------------------- #


def test_null_bloom_not_imputed(monkeypatch, tmp_path):
    monkeypatch.setenv("ED4ALL_BLOOM_DISTRIBUTION", "1")
    # 6 declared apply + 4 with no bloom field at all.
    path = _write_doc(
        tmp_path,
        _objectives_doc(["apply"] * 6 + [None, None, None, None]),
    )
    result = BloomDistributionValidator().validate(
        {"synthesized_objectives_path": path}
    )
    meta = result.metadata["bloom_distribution"]
    assert meta["n_objectives"] == 6  # the 4 None LOs excluded
    assert meta["n_skipped"] == 4
    # The skipped objectives never contribute a level.
    assert sum(meta["observed_counts"].values()) == 6


def test_bloom_verb_resolves_level(monkeypatch, tmp_path):
    """A declared bloom_verb (real data shape) resolves to its canonical level
    — this is a lookup of an EXISTING field, not free-text imputation."""
    monkeypatch.setenv("ED4ALL_BLOOM_DISTRIBUTION", "1")
    doc = {
        "terminal_objectives": [
            {"id": "TO-01", "bloom_verb": "Implement", "statement": "x"},
            {"id": "TO-02", "bloom_verb": "Analyze", "statement": "y"},
        ],
        "chapter_objectives": [],
    }
    path = _write_doc(tmp_path, doc)
    result = BloomDistributionValidator().validate(
        {"synthesized_objectives_path": path}
    )
    meta = result.metadata["bloom_distribution"]
    assert meta["n_objectives"] == 2
    assert meta["observed_counts"]["apply"] == 1  # Implement -> apply
    assert meta["observed_counts"]["analyze"] == 1


# --------------------------------------------------------------------------- #
# Graceful skip on empty
# --------------------------------------------------------------------------- #


def test_graceful_skip_no_objectives(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOOM_DISTRIBUTION", "1")
    result = BloomDistributionValidator().validate({})
    assert result.passed is True
    assert "BLOOM_DISTRIBUTION_NO_OBJECTIVES" in _codes(result)


# --------------------------------------------------------------------------- #
# Grouped chapter_objectives shape is iterated
# --------------------------------------------------------------------------- #


def test_grouped_chapter_objectives(monkeypatch, tmp_path):
    monkeypatch.setenv("ED4ALL_BLOOM_DISTRIBUTION", "1")
    doc = {
        "terminal_objectives": [{"id": "TO-01", "bloom_level": "apply"}],
        "chapter_objectives": [
            {
                "chapter": "Ch1",
                "objectives": [
                    {"id": "CO-01", "bloom_level": "understand"},
                    {"id": "CO-02", "bloom_level": "analyze"},
                ],
            }
        ],
    }
    path = _write_doc(tmp_path, doc)
    result = BloomDistributionValidator().validate(
        {"synthesized_objectives_path": path}
    )
    meta = result.metadata["bloom_distribution"]
    assert meta["n_objectives"] == 3


# --------------------------------------------------------------------------- #
# Operator overrides
# --------------------------------------------------------------------------- #


def test_inline_target_override_changes_verdict(monkeypatch, tmp_path):
    monkeypatch.setenv("ED4ALL_BLOOM_DISTRIBUTION", "1")
    # All-apply skills course: off-target vs the default curve.
    path = _write_doc(tmp_path, _objectives_doc(["apply"] * 10))
    base = BloomDistributionValidator().validate(
        {"synthesized_objectives_path": path}
    )
    assert "BLOOM_DISTRIBUTION_OFF_TARGET" in _codes(base)
    # Operator pins an apply-dominant curve → off-target clears.
    monkeypatch.setenv(
        "ED4ALL_BLOOM_DISTRIBUTION_TARGET",
        json.dumps({"apply": 1.0}),
    )
    tuned = BloomDistributionValidator().validate(
        {"synthesized_objectives_path": path}
    )
    assert "BLOOM_DISTRIBUTION_OFF_TARGET" not in _codes(tuned)


def test_garbage_target_falls_back():
    # Garbage inline JSON / non-subset key → canonical curve.
    assert resolve_bloom_target("not json {{{")["target_shares"]["apply"] == 0.25
    assert resolve_bloom_target('{"bogus_level": 0.5}')["target_shares"]["apply"] == 0.25
    # Out-of-range share → fallback.
    assert resolve_bloom_target('{"apply": 5}')["target_shares"]["apply"] == 0.25


def test_garbage_tolerance_falls_back():
    assert resolve_tolerance("abc") == DEFAULT_TOLERANCE
    assert resolve_tolerance("-1") == DEFAULT_TOLERANCE
    assert resolve_tolerance("0") == DEFAULT_TOLERANCE
    assert resolve_tolerance("0.05") == 0.05


def test_garbage_min_los_falls_back():
    assert resolve_min_los("abc") == DEFAULT_MIN_LOS
    assert resolve_min_los("-3") == DEFAULT_MIN_LOS
    assert resolve_min_los("3") == 3


def test_small_n_suppresses_share_verdicts(monkeypatch, tmp_path):
    monkeypatch.setenv("ED4ALL_BLOOM_DISTRIBUTION", "1")
    # 3 LOs (< default floor 6), all evaluate → would be top-heavy/off-target,
    # but share-based verdicts are suppressed; only SMALL_N surfaces.
    path = _write_doc(tmp_path, _objectives_doc(["evaluate", "evaluate", "create"]))
    result = BloomDistributionValidator().validate(
        {"synthesized_objectives_path": path}
    )
    codes = _codes(result)
    assert "BLOOM_DISTRIBUTION_SMALL_N" in codes
    assert "BLOOM_DISTRIBUTION_TOP_HEAVY" not in codes
    assert "BLOOM_DISTRIBUTION_OFF_TARGET" not in codes


def test_small_n_recall_only_still_fires_no_higher_order(monkeypatch, tmp_path):
    monkeypatch.setenv("ED4ALL_BLOOM_DISTRIBUTION", "1")
    path = _write_doc(tmp_path, _objectives_doc(["remember", "understand"]))
    result = BloomDistributionValidator().validate(
        {"synthesized_objectives_path": path}
    )
    codes = _codes(result)
    assert "BLOOM_DISTRIBUTION_NO_HIGHER_ORDER" in codes  # defect at any size


# --------------------------------------------------------------------------- #
# Decision capture fires exactly once with dynamic rationale
# --------------------------------------------------------------------------- #


def test_decision_capture_fires_once(monkeypatch, tmp_path):
    monkeypatch.setenv("ED4ALL_BLOOM_DISTRIBUTION", "1")
    cap = _FakeCapture()
    path = _write_doc(tmp_path, _objectives_doc(["apply"] * 8))
    BloomDistributionValidator(decision_capture=cap).validate(
        {"synthesized_objectives_path": path}
    )
    assert len(cap.calls) == 1
    call = cap.calls[0]
    assert call["decision_type"] == "validation_result"
    assert len(call["rationale"]) >= 20
    # Dynamic rationale interpolates the observed/target signal, not a static
    # string.
    assert "L1" in call["rationale"]
    assert "n_objectives" in call["ml_features"]


# --------------------------------------------------------------------------- #
# resolve_bloom_distribution parse-with-fallback
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "val,expected",
    [("1", True), ("true", True), ("YES", True), ("on", True),
     ("0", False), ("false", False), ("garbage", False), ("", False)],
)
def test_resolve_flag_parse_with_fallback(val, expected):
    assert resolve_bloom_distribution(val) is expected
