"""Tests for lib.retrieval.grounded_eval_diff — the regression-diff CLI.

CI-safe: builds tiny inline report dicts (the published grounded_answer_eval
JSON shape), never touches an LLM / model weights / a real course slug. Covers
improvement, within-tolerance, regression (exit-1), diagnostic-bucket exemption,
config-diff detection, and 1.6-vs-1.7 (additive unknown field) tolerance.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.retrieval.grounded_eval_diff import (
    diff_reports,
    main,
    render_text,
)


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _report(
    *,
    schema_version: str = "1.6",
    answer_rate: float = 0.95,
    citation_resolution_rate: float = 1.0,
    citation_precision: float = 0.35,
    citation_precision_primary: float = 0.40,
    groundedness_rate_mean: float = 0.80,
    unsupported_claim_rate: float = 0.09,
    refusal_recall: float = 0.90,
    refusal_precision: float = 0.85,
    false_refusals_on_gold: int = 1,
    blocked_invalid: int = 0,
    blocked_gate: int = 0,
    composer_exhausted: int = 0,
    macro: float = 0.80,
    micro: float = 0.78,
    phrasing: dict | None = None,
    by_question_type: dict | None = None,
    config: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Build one grounded_answer_eval report dict in the published shape."""
    if phrasing is None:
        phrasing = {
            "canonical": {"n": 20, "answered_count": 19, "answered_rate": 0.95,
                          "blocked_count": 0, "refused_count": 1},
            "colloquial": {"n": 6, "answered_count": 5, "answered_rate": 0.83,
                           "blocked_count": 0, "refused_count": 1},
        }
    if by_question_type is None:
        by_question_type = {
            "definition": {"macro_groundedness": 0.85, "micro_groundedness": 0.84},
            "application": {"macro_groundedness": 0.72, "micro_groundedness": 0.70},
        }
    report: dict = {
        "schema_version": schema_version,
        "course_slug": "fixture-course-1",
        "engine": "lexical",
        "model_id": "fixture-model",
        "prompt_version": "p1",
        "refusal_policy_version": "r1",
        "generated_at": "2026-07-19T00:00:00Z",
        "headline": {
            "answer_rate": answer_rate,
            "citation_resolution_rate": citation_resolution_rate,
            "citation_precision": citation_precision,
            "citation_precision_primary": citation_precision_primary,
            "groundedness_rate_mean": groundedness_rate_mean,
            "unsupported_claim_rate": unsupported_claim_rate,
            "refusal": {
                "n_probes": 12,
                "refusal_recall": refusal_recall,
                "refusal_precision": refusal_precision,
                "false_refusals_on_gold": false_refusals_on_gold,
                "unsupported_answer_rate_on_answered_probes": 0.10,
            },
            "key_point_coverage": {"coverage_rate": 0.75},
            "groundedness_breakdown": {
                "macro_groundedness": macro,
                "micro_groundedness": micro,
                "by_question_type": by_question_type,
                "by_stratum": {},
                "_diagnostic": "diagnostic breakdown, NOT a pinned milestone.",
            },
            "phrasing_breakdown": phrasing,
        },
        "blocked": {
            "invalid_citation": blocked_invalid,
            "citation_gate": blocked_gate,
        },
        "composer_exhausted": composer_exhausted,
    }
    if config is not None:
        report["config"] = config
    if extra is not None:
        report.update(extra)
    return report


# --------------------------------------------------------------------------- #
# Improvement / within-tolerance / regression
# --------------------------------------------------------------------------- #

def test_improvement_exits_zero():
    old = _report(answer_rate=0.90, groundedness_rate_mean=0.70)
    new = _report(answer_rate=0.96, groundedness_rate_mean=0.82)
    result = diff_reports(old, new, tolerance_pp=5.0)
    assert not result.has_regression
    assert result.exit_code == 0
    ans = next(r for r in result.metrics if r.label == "answer_rate")
    assert ans.status == "improved"
    assert ans.delta_pp == pytest.approx(6.0)


def test_within_tolerance_exits_zero():
    # answer_rate drops 3pp (< 5pp tolerance) → within tolerance, no exit-1.
    old = _report(answer_rate=0.95)
    new = _report(answer_rate=0.92)
    result = diff_reports(old, new, tolerance_pp=5.0)
    assert not result.has_regression
    assert result.exit_code == 0
    ans = next(r for r in result.metrics if r.label == "answer_rate")
    assert ans.status == "within_tolerance"


def test_regression_beyond_tolerance_exits_one():
    # answer_rate drops 10pp → pinned regression → exit 1.
    old = _report(answer_rate=0.95)
    new = _report(answer_rate=0.85)
    result = diff_reports(old, new, tolerance_pp=5.0)
    assert result.has_regression
    assert result.exit_code == 1
    ans = next(r for r in result.metrics if r.label == "answer_rate")
    assert ans.status == "regressed"
    assert ans.counts_toward_exit is True


def test_lower_better_regression_on_unsupported_rate():
    # unsupported_claim_rate RISES 8pp → regression on a lower-better metric.
    old = _report(unsupported_claim_rate=0.05)
    new = _report(unsupported_claim_rate=0.13)
    result = diff_reports(old, new, tolerance_pp=5.0)
    assert result.has_regression
    row = next(r for r in result.metrics if r.label == "unsupported_claim_rate")
    assert row.status == "regressed"
    assert row.counts_toward_exit is True


# --------------------------------------------------------------------------- #
# Diagnostic-bucket exemption
# --------------------------------------------------------------------------- #

def test_diagnostic_metric_regression_never_exits_one():
    # groundedness_macro (diagnostic) collapses 30pp, all pinned metrics steady.
    old = _report(macro=0.80)
    new = _report(macro=0.50)
    result = diff_reports(old, new, tolerance_pp=5.0)
    row = next(r for r in result.metrics if r.label == "groundedness_macro")
    assert row.status == "regressed"
    assert row.counts_toward_exit is False
    assert not result.has_regression
    assert result.exit_code == 0


def test_diagnostic_bucket_regression_never_exits_one():
    # A per-phrasing bucket collapses; a per-category bucket collapses too.
    old = _report(
        phrasing={"colloquial": {"answered_rate": 0.90}},
        by_question_type={"application": {"macro_groundedness": 0.90}},
    )
    new = _report(
        phrasing={"colloquial": {"answered_rate": 0.40}},
        by_question_type={"application": {"macro_groundedness": 0.40}},
    )
    result = diff_reports(old, new, tolerance_pp=5.0)
    ph = result.buckets["phrasing"][0]
    qt = result.buckets["question_type"][0]
    assert ph.status == "regressed" and ph.counts_toward_exit is False
    assert qt.status == "regressed" and qt.counts_toward_exit is False
    assert not result.has_regression
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# Config-diff detection
# --------------------------------------------------------------------------- #

def test_config_diff_detected():
    old = _report(config={"ED4ALL_ANSWER_NLI_ADD": "shadow",
                          "ED4ALL_ANSWER_NUM_CTX": "4096"})
    new = _report(config={"ED4ALL_ANSWER_NLI_ADD": "on",
                          "ED4ALL_ANSWER_NUM_CTX": "4096"})
    result = diff_reports(old, new, tolerance_pp=5.0)
    cfg = result.config_diff
    assert cfg["present"] is True
    assert cfg["changed"] is True
    assert "ED4ALL_ANSWER_NLI_ADD" in cfg["changes"]
    assert cfg["changes"]["ED4ALL_ANSWER_NLI_ADD"] == {"old": "shadow", "new": "on"}
    # Unchanged flag not listed.
    assert "ED4ALL_ANSWER_NUM_CTX" not in cfg["changes"]
    # Human render calls out the attribution.
    text = render_text(result)
    assert "config changed between runs" in text
    assert "ED4ALL_ANSWER_NLI_ADD" in text


def test_config_diff_absent_when_no_stamp():
    old = _report()
    new = _report()
    result = diff_reports(old, new, tolerance_pp=5.0)
    assert result.config_diff["present"] is False
    assert result.config_diff["changed"] is False


def test_config_unchanged_when_same_stamp():
    cfg = {"ED4ALL_ANSWER_NLI_ADD": "shadow"}
    result = diff_reports(_report(config=dict(cfg)), _report(config=dict(cfg)))
    assert result.config_diff["present"] is True
    assert result.config_diff["changed"] is False


# --------------------------------------------------------------------------- #
# 1.6-vs-1.7 additive-field tolerance
# --------------------------------------------------------------------------- #

def test_diff_tolerates_1_6_vs_1_7_additive_fields():
    old = _report(schema_version="1.6")
    # Simulate a 1.7 report: bumped schema + a brand-new headline field + a new
    # top-level block the differ has never seen. It must not crash and must
    # still diff the shared metrics.
    new = _report(schema_version="1.7")
    new["headline"]["brand_new_1_7_metric"] = 0.42
    new["some_new_top_level_block"] = {"whatever": [1, 2, 3]}
    result = diff_reports(old, new, tolerance_pp=5.0)
    assert result.old_meta["schema_version"] == "1.6"
    assert result.new_meta["schema_version"] == "1.7"
    # Shared metric still compared; no regression on identical values.
    assert not result.has_regression
    ans = next(r for r in result.metrics if r.label == "answer_rate")
    assert ans.status == "within_tolerance"


def test_metric_missing_on_one_side_is_na_not_regression():
    old = _report()
    new = _report()
    # Drop a pinned metric from the new report entirely.
    del new["headline"]["citation_precision"]
    result = diff_reports(old, new, tolerance_pp=5.0)
    row = next(r for r in result.metrics if r.label == "citation_precision")
    assert row.status == "n/a"
    assert row.counts_toward_exit is False
    assert not result.has_regression


# --------------------------------------------------------------------------- #
# CLI entry point (exit codes + --json)
# --------------------------------------------------------------------------- #

def _write(tmp_path: Path, name: str, report: dict) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(report), encoding="utf-8")
    return str(p)


def test_cli_regression_exit_one(tmp_path, capsys):
    old = _write(tmp_path, "old.json", _report(answer_rate=0.95))
    new = _write(tmp_path, "new.json", _report(answer_rate=0.80))
    rc = main([old, new, "--tolerance-pp", "5.0"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "REGRESSION" in out


def test_cli_clean_exit_zero_json(tmp_path, capsys):
    old = _write(tmp_path, "old.json", _report())
    new = _write(tmp_path, "new.json", _report())
    rc = main([old, new, "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code"] == 0
    assert payload["regression"] is False
    assert payload["tolerance_pp"] == 5.0
    assert any(m["label"] == "answer_rate" for m in payload["metrics"])


def test_cli_bad_path_exits_two(tmp_path, capsys):
    old = _write(tmp_path, "old.json", _report())
    rc = main([old, str(tmp_path / "nope.json")])
    # argparse would reject a missing positional only under click; the module
    # CLI reads paths directly, so a missing file surfaces as the read-error 2.
    assert rc == 2
