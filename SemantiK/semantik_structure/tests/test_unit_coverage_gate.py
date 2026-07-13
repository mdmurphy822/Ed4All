"""SEMANTIK_UNIT_COVERAGE_GATE — deterministic content-loss fidelity gate (task #43).

Synthetic extraction units + emitted HTML (NO course/corpus text). Asserts:
(1) output drops a chunk of extraction text → the gate fires and NAMES the
missing span; (2) full coverage → passes; (3) flag off → no-op resolver.
"""

from __future__ import annotations

import pytest

from semantik_structure import unit_coverage
from semantik_structure.unit_coverage import (
    compute_unit_coverage,
    resolve_unit_coverage_gate_mode,
    resolve_unit_coverage_min,
    resolve_unit_coverage_page_floor,
    run_unit_coverage_gate,
    strip_html_text,
)


# ---------------------------------------------------------------------------
# Resolvers.
# ---------------------------------------------------------------------------


def test_gate_default_off(monkeypatch):
    monkeypatch.delenv(unit_coverage.SEMANTIK_UNIT_COVERAGE_GATE_ENV, raising=False)
    assert resolve_unit_coverage_gate_mode() is False


@pytest.mark.parametrize("val,expected", [("1", True), ("on", True), ("0", False), ("x", False)])
def test_gate_resolver(monkeypatch, val, expected):
    monkeypatch.setenv(unit_coverage.SEMANTIK_UNIT_COVERAGE_GATE_ENV, val)
    assert resolve_unit_coverage_gate_mode() is expected


def test_thresholds_defaults(monkeypatch):
    monkeypatch.delenv(unit_coverage.SEMANTIK_UNIT_COVERAGE_MIN_ENV, raising=False)
    monkeypatch.delenv(unit_coverage.SEMANTIK_UNIT_COVERAGE_PAGE_FLOOR_ENV, raising=False)
    assert resolve_unit_coverage_min() == 0.90
    assert resolve_unit_coverage_page_floor() == 0.70


@pytest.mark.parametrize("val", ["nan", "inf", "-0.1", "1.5", "garbage", ""])
def test_threshold_parse_with_fallback(monkeypatch, val):
    monkeypatch.setenv(unit_coverage.SEMANTIK_UNIT_COVERAGE_MIN_ENV, val)
    assert resolve_unit_coverage_min() == 0.90


def test_threshold_valid_override(monkeypatch):
    monkeypatch.setenv(unit_coverage.SEMANTIK_UNIT_COVERAGE_MIN_ENV, "0.8")
    assert resolve_unit_coverage_min() == 0.8


# ---------------------------------------------------------------------------
# strip_html_text.
# ---------------------------------------------------------------------------


def test_strip_html_text():
    html = "<h2>Title</h2><p>hello&nbsp;world &amp; more</p>"
    txt = strip_html_text(html)
    assert "Title" in txt and "hello" in txt and "world" in txt and "more" in txt
    assert "<" not in txt


# ---------------------------------------------------------------------------
# Coverage computation — the content-loss class.
# ---------------------------------------------------------------------------


def test_full_coverage_passes():
    units = [
        (1, "the quick brown fox"),
        (1, "jumps over the lazy dog"),
    ]
    html = "<p>the quick brown fox jumps over the lazy dog</p>"
    rep = compute_unit_coverage(units, html)
    assert rep["document_passed"] is True
    assert rep["warned_pages"] == []
    assert rep["failed_pages"] == []
    assert all(p["coverage"] == 1.0 for p in rep["pages"])


def test_dropped_chunk_fires_and_names_span():
    # A whole "worked example" span present in extraction but absent from output.
    units = [
        (1, "Introduction paragraph one two three four five"),
        (1, "Worked Example compute the discriminant of the quadratic seven eight"),
    ]
    # Output keeps only the intro; the worked example is DROPPED.
    html = "<p>Introduction paragraph one two three four five</p>"
    rep = compute_unit_coverage(units, html)
    assert rep["document_passed"] is False  # below hard floor → fail-closed
    assert rep["failed_pages"] == [1]
    page = rep["pages"][0]
    assert page["below_min"] is True
    assert page["below_floor"] is True
    # The longest missing span is named (the worked example line).
    assert page["missing_spans"], "expected a named missing span"
    joined = " ".join(page["missing_spans"]).lower()
    assert "worked example" in joined
    assert "discriminant" in joined


def test_warn_band_between_floor_and_min():
    # ~85% coverage: below the 0.90 min, above the 0.70 floor → WARN not FAIL.
    kept = " ".join(f"w{i}" for i in range(17))
    dropped = "aa bb cc"  # 3 of 20 tokens missing → 0.85 coverage
    units = [(1, kept + " " + dropped)]
    html = f"<p>{kept}</p>"
    rep = compute_unit_coverage(units, html, min_coverage=0.90, page_floor=0.70)
    page = rep["pages"][0]
    assert page["below_min"] is True
    assert page["below_floor"] is False
    assert rep["document_passed"] is True  # warn, not fail
    assert rep["warned_pages"] == [1]


def test_span_prefix_truncated_to_80_chars():
    long_missing = " ".join(f"tok{i}" for i in range(60))  # far over 80 chars
    units = [(1, long_missing)]
    rep = compute_unit_coverage(units, "<p></p>")
    assert rep["pages"][0]["missing_spans"]
    assert all(len(s) <= 80 for s in rep["pages"][0]["missing_spans"])


def test_image_only_page_full_coverage():
    # A page whose extraction unit has no alphanumeric tokens → nothing to lose.
    units = [(2, "!!! ---")]
    rep = compute_unit_coverage(units, "<p></p>")
    p2 = [p for p in rep["pages"] if p["page"] == 2][0]
    assert p2["coverage"] == 1.0
    assert p2["below_floor"] is False


# ---------------------------------------------------------------------------
# run_unit_coverage_gate — loud logging + theta-bypass note.
# ---------------------------------------------------------------------------


def test_run_gate_logs_theta_bypass_and_fail():
    logs: list[str] = []
    units = [(1, "alpha beta gamma delta epsilon zeta eta theta")]
    rep = run_unit_coverage_gate(units, "<p>alpha</p>", log=logs.append)
    assert any("theta bypassed" in m for m in logs)
    assert any("FAIL-CLOSED" in m for m in logs)
    assert rep["document_passed"] is False
