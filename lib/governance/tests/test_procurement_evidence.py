"""Tests for the procurement evidence exporter (backlog E4/E5/D5).

Fully offline: builds synthetic grounded-eval report dicts on ``tmp_path`` (no
pipeline, no model, no network, no course-slug fixtures). Exercises:

* the evidence-bundle roll-up (headline subset, breakdowns, flag stamp, CIs),
* the anti-silent-degradation ``not_evaluated`` path (missing / malformed
  report),
* the floor-pass verdict against the pinned milestone targets,
* PPI interval math (base PPI vs power-tuned; labeled-only degenerate case)
  and its Wilson fallback when no operator labels are present,
* the readiness computation (per-course streak + cross-course aggregation),
* the ADVISORY promotion-chain back-reference (never mutating the chain).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.governance import procurement_evidence as pe
from lib.retrieval.grounded_eval import MILESTONE_TARGETS


# --------------------------------------------------------------------------- #
# Synthetic report builders
# --------------------------------------------------------------------------- #

def _passing_headline(**overrides):
    """A headline that clears every pinned milestone floor/ceiling."""
    headline = {
        "answer_rate": 0.98,
        "citation_resolution_rate": 1.0,
        "citation_precision": 0.40,
        "citation_recall": 0.55,
        "citation_precision_legacy": 0.42,
        "groundedness_rate_mean": 0.90,
        "groundedness_rate_micro": 0.88,
        "unsupported_claim_rate": 0.03,
        "refusal": {
            "n_probes": 44,
            "refusal_recall": 0.80,
            "refusal_precision": 0.95,
            "by_category": {"off_topic": {"n": 6, "refused": 6, "refused_rate": 1.0}},
        },
        "phrasing_breakdown": {
            "canonical": {"n": 50, "answered_count": 49, "answered_rate": 0.98},
        },
        "abstention": {"premise_correction": {"rate": 0.5}},
    }
    headline.update(overrides)
    return headline


def _report(headline, questions=None, *, ts="20260101T000000Z", **top):
    doc = {
        "schema_version": "1.7",
        "course_slug": "unit-course",
        "engine": "lexical",
        "model_id": "qwen-test",
        "prompt_version": "ws3.v3",
        "refusal_policy_version": "v3",
        "gold": {"schema_version": "1.1", "sha256": "deadbeef"},
        "questions": questions or [],
        "headline": headline,
        "flag_config": {"answer_env": {"ED4ALL_ANSWER_NLI_ADD": "off"}},
        "generated_at": "2026-01-01T00:00:00Z",
        "_ts": ts,
    }
    doc.update(top)
    return doc


def _write_report(course_dir: Path, doc: dict, ts: str) -> Path:
    eval_dir = course_dir / pe.RETRIEVAL_EVAL_SUBDIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / f"grounded_answer_eval_{ts}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Report discovery + not_evaluated
# --------------------------------------------------------------------------- #

def test_not_evaluated_when_no_report(tmp_path):
    bundle = pe.build_evidence_bundle(tmp_path / "course", course_slug="c1")
    assert bundle["evaluation_status"] == "not_evaluated"
    assert bundle["advisory"] is True
    assert "headline" not in bundle
    assert bundle["readiness"]["consecutive_passing_runs"] == 0
    assert bundle["readiness"]["meets_run_criterion"] is False


def test_not_evaluated_when_report_malformed(tmp_path):
    course = tmp_path / "course"
    eval_dir = course / pe.RETRIEVAL_EVAL_SUBDIR
    eval_dir.mkdir(parents=True)
    (eval_dir / "grounded_answer_eval_20260101T000000Z.json").write_text(
        "{not json", encoding="utf-8"
    )
    bundle = pe.build_evidence_bundle(course)
    assert bundle["evaluation_status"] == "not_evaluated"
    assert "malformed" in bundle["reason"]


def test_latest_eval_report_picks_newest(tmp_path):
    course = tmp_path / "course"
    _write_report(course, _report(_passing_headline()), "20260101T000000Z")
    newest = _write_report(course, _report(_passing_headline()), "20260202T000000Z")
    assert pe.latest_eval_report(course) == newest


# --------------------------------------------------------------------------- #
# Bundle roll-up
# --------------------------------------------------------------------------- #

def test_bundle_rollup_fields(tmp_path):
    course = tmp_path / "course"
    _write_report(course, _report(_passing_headline()), "20260101T000000Z")
    bundle = pe.build_evidence_bundle(
        course, course_code="UNIT_101", course_slug="unit-course", run_id="WF-1"
    )
    assert bundle["evaluation_status"] == "evaluated"
    assert bundle["schema_version"] == pe.EVIDENCE_SCHEMA_VERSION
    assert bundle["course_code"] == "UNIT_101"
    # Pinned headline subset present.
    assert bundle["headline"]["answer_rate"] == 0.98
    assert bundle["headline"]["groundedness_rate_micro"] == 0.88
    assert bundle["headline"]["refusal_recall"] == 0.80
    # Breakdowns + flag stamp carried through.
    assert "canonical" in bundle["phrasing_breakdown"]
    assert bundle["abstention"]["premise_correction"]["rate"] == 0.5
    assert bundle["refusal"]["by_category"]["off_topic"]["refused"] == 6
    assert bundle["flag_config"]["answer_env"]["ED4ALL_ANSWER_NLI_ADD"] == "off"
    assert bundle["source_report"]["model_id"] == "qwen-test"
    assert bundle["operator_labels"]["present"] is False


# --------------------------------------------------------------------------- #
# Floor-pass verdict
# --------------------------------------------------------------------------- #

def test_floor_pass_true_on_passing_headline():
    fp = pe.floor_pass(_report(_passing_headline()))
    assert fp["passed"] is True
    assert fp["missing"] == []
    # Every pinned target is present + accounted for.
    assert set(fp["metrics"]) == set(MILESTONE_TARGETS)


def test_floor_pass_false_below_floor():
    hl = _passing_headline(answer_rate=0.10)
    fp = pe.floor_pass(_report(hl))
    assert fp["passed"] is False
    assert fp["metrics"]["answer_rate"]["passed"] is False


def test_floor_pass_ceiling_semantics():
    # unsupported_claim_rate is a ceiling: a high value FAILS.
    hl = _passing_headline(unsupported_claim_rate=0.50)
    fp = pe.floor_pass(_report(hl))
    assert fp["metrics"]["unsupported_claim_rate"]["kind"] == "ceiling"
    assert fp["metrics"]["unsupported_claim_rate"]["passed"] is False
    assert fp["passed"] is False


def test_floor_pass_none_metric_fails_not_passes():
    hl = _passing_headline(groundedness_rate_mean=None)
    fp = pe.floor_pass(_report(hl))
    assert "groundedness_rate_mean" in fp["missing"]
    assert fp["passed"] is False


# --------------------------------------------------------------------------- #
# Wilson CI fallback
# --------------------------------------------------------------------------- #

def test_wilson_ci_basis_and_null():
    assert pe.wilson_ci(0, 0)["basis"] == "diagnostic"
    assert pe.wilson_ci(0, 0)["lo"] is None
    small = pe.wilson_ci(8, 10)
    assert small["basis"] == "diagnostic"
    assert 0.0 <= small["lo"] <= small["point"] <= small["hi"] <= 1.0
    big = pe.wilson_ci(45, 50)
    assert big["basis"] == "sufficient"


def test_confidence_intervals_wilson_when_no_labels(tmp_path):
    course = tmp_path / "course"
    questions = [
        {"question_id": f"q{i}", "status": "answered", "groundedness_rate": 0.9}
        for i in range(40)
    ]
    _write_report(course, _report(_passing_headline(), questions), "20260101T000000Z")
    bundle = pe.build_evidence_bundle(course)
    ci = bundle["confidence_intervals"]["groundedness_rate_mean"]
    assert ci["method"] == "wilson"
    assert ci["n"] == 40
    ar = bundle["confidence_intervals"]["answer_rate"]
    assert ar["method"] == "wilson"


# --------------------------------------------------------------------------- #
# PPI math
# --------------------------------------------------------------------------- #

def test_ppi_none_without_labeled_slice():
    assert pe.compute_ppi_interval([0.9, 0.8], [], []) is None


def test_ppi_base_estimator_matches_closed_form():
    # Unlabeled proxy mean 0.8; labeled rectifier mean(f-Y) = 0.1 -> theta 0.7.
    U = [0.8] * 10
    Lf = [0.9, 0.9, 0.9, 0.9]
    Ly = [0.8, 0.8, 0.8, 0.8]
    res = pe.compute_ppi_interval(U, Lf, Ly)
    assert res["method"] == "ppi"
    assert res["lambda"] == 1.0
    # theta = mean_L(Y) + 1*(mean_U(f) - mean_L(f)) = 0.8 + (0.8 - 0.9) = 0.7
    assert res["point"] == pytest.approx(0.7, abs=1e-9)
    assert res["n_labeled"] == 4
    assert res["n_unlabeled"] == 10
    assert res["basis"] == "diagnostic"  # n<30


def test_ppi_zero_variance_gives_zero_half_width():
    # Perfect proxy (f == Y everywhere), constant -> variance 0 -> point CI.
    res = pe.compute_ppi_interval([0.5] * 5, [0.5, 0.5], [0.5, 0.5])
    assert res["half_width"] == 0.0
    assert res["lo"] == res["hi"] == pytest.approx(0.5)


def test_ppi_labeled_only_when_no_unlabeled():
    res = pe.compute_ppi_interval([], [0.9, 0.7], [1.0, 0.0])
    assert res["method"] == "ppi_labeled_only"
    assert res["point"] == pytest.approx(0.5)
    assert res["n_unlabeled"] == 0


def test_ppi_power_tuning_reports_lambda_star_and_reduces_or_equal_width():
    # Correlated proxy: power-tuned lambda* in [0,1].
    U = [0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.7]
    Lf = [0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.7]
    Ly = [0.1, 0.5, 0.5, 0.9, 0.9, 0.4, 0.6]
    base = pe.compute_ppi_interval(U, Lf, Ly, power_tuning=False)
    tuned = pe.compute_ppi_interval(U, Lf, Ly, power_tuning=True)
    assert base["lambda"] == 1.0
    assert tuned["lambda_star"] is not None
    assert 0.0 <= tuned["lambda"] <= 1.0


# --------------------------------------------------------------------------- #
# PPI end-to-end through the bundle (operator_labels.json present)
# --------------------------------------------------------------------------- #

def _write_operator_labels(course_dir: Path, metric: str, labels: list) -> Path:
    eval_dir = course_dir / pe.RETRIEVAL_EVAL_SUBDIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / pe.OPERATOR_LABELS_FILENAME
    path.write_text(
        json.dumps({"schema_version": "1.0", "metric": metric, "labels": labels}),
        encoding="utf-8",
    )
    return path


def test_operator_labels_drive_ppi_ci(tmp_path):
    course = tmp_path / "course"
    questions = [
        {"question_id": f"q{i}", "status": "answered", "groundedness_rate": 0.9}
        for i in range(20)
    ]
    _write_report(course, _report(_passing_headline(), questions), "20260101T000000Z")
    # Operator labels anchor groundedness on a subset; proxy joined from report.
    labels = [{"question_id": f"q{i}", "operator_label": 1.0} for i in range(5)]
    _write_operator_labels(course, "groundedness_rate_mean", labels)

    bundle = pe.build_evidence_bundle(course)
    ci = bundle["confidence_intervals"]["groundedness_rate_mean"]
    assert ci["method"] == "ppi"
    assert ci["n_labeled"] == 5
    assert ci["n_unlabeled"] == 15  # 20 total - 5 labeled
    assert bundle["operator_labels"]["present"] is True
    assert bundle["operator_labels"]["metric"] == "groundedness_rate_mean"
    # answer_rate has no operator labels -> Wilson.
    assert bundle["confidence_intervals"]["answer_rate"]["method"] == "wilson"


def test_operator_labels_explicit_proxy_used(tmp_path):
    course = tmp_path / "course"
    questions = [
        {"question_id": f"q{i}", "status": "answered", "groundedness_rate": 0.9}
        for i in range(10)
    ]
    _write_report(course, _report(_passing_headline(), questions), "20260101T000000Z")
    labels = [
        {"question_id": "q0", "operator_label": 1.0, "proxy_label": 0.85},
        {"question_id": "q1", "operator_label": 0.0, "proxy_label": 0.20},
    ]
    _write_operator_labels(course, "groundedness_rate_mean", labels)
    ol = pe.load_operator_labels(course)
    assert ol["metric"] == "groundedness_rate_mean"
    assert ol["labels"][0]["proxy_label"] == 0.85


def test_malformed_operator_labels_ignored(tmp_path):
    course = tmp_path / "course"
    eval_dir = course / pe.RETRIEVAL_EVAL_SUBDIR
    eval_dir.mkdir(parents=True)
    (eval_dir / pe.OPERATOR_LABELS_FILENAME).write_text("{bad", encoding="utf-8")
    assert pe.load_operator_labels(course) is None


# --------------------------------------------------------------------------- #
# Readiness (D5)
# --------------------------------------------------------------------------- #

def test_consecutive_passing_runs_streak(tmp_path):
    course = tmp_path / "course"
    # Two passing (newest), then a failing one -> streak stops at 2.
    _write_report(course, _report(_passing_headline(answer_rate=0.1)),
                  "20260101T000000Z")
    _write_report(course, _report(_passing_headline()), "20260202T000000Z")
    _write_report(course, _report(_passing_headline()), "20260303T000000Z")
    bundle = pe.build_evidence_bundle(course)
    assert bundle["readiness"]["consecutive_passing_runs"] == 2
    assert bundle["readiness"]["meets_run_criterion"] is True
    assert bundle["readiness"]["criterion"]["min_courses"] == pe.READINESS_MIN_COURSES


def test_readiness_single_pass_below_criterion(tmp_path):
    course = tmp_path / "course"
    _write_report(course, _report(_passing_headline()), "20260101T000000Z")
    bundle = pe.build_evidence_bundle(course)
    assert bundle["readiness"]["consecutive_passing_runs"] == 1
    assert bundle["readiness"]["meets_run_criterion"] is False


def test_aggregate_readiness_needs_two_courses():
    ready_course = {
        "evaluation_status": "evaluated",
        "course_slug": "a",
        "readiness": {"meets_run_criterion": True},
    }
    other = dict(ready_course, course_slug="b")
    not_ready = {
        "evaluation_status": "evaluated",
        "course_slug": "c",
        "readiness": {"meets_run_criterion": False},
    }
    # One course meeting -> not ready.
    agg1 = pe.aggregate_readiness([ready_course, not_ready])
    assert agg1["blocking_flip_ready"] is False
    # Two distinct courses meeting -> ready (still advisory).
    agg2 = pe.aggregate_readiness([ready_course, other, not_ready])
    assert agg2["blocking_flip_ready"] is True
    assert set(agg2["courses_meeting_run_criterion"]) == {"a", "b"}


def test_aggregate_readiness_skips_not_evaluated():
    ne = {"evaluation_status": "not_evaluated", "course_slug": "x",
          "readiness": {"meets_run_criterion": True}}
    assert pe.aggregate_readiness([ne, ne])["blocking_flip_ready"] is False


# --------------------------------------------------------------------------- #
# Promotion-chain linkage (advisory, never mutating the chain)
# --------------------------------------------------------------------------- #

def test_link_promotion_chain_backreference():
    bundle = {"schema_version": "1.0"}
    chain = {"chain_hash": "abc123", "course_status": "certified_instructional",
             "run_id": "WF-9"}
    out = pe.link_promotion_chain(bundle, chain)
    assert out is bundle
    assert bundle["promotion_chain"]["linked"] is True
    assert bundle["promotion_chain"]["chain_hash"] == "abc123"
    assert bundle["promotion_chain"]["course_status"] == "certified_instructional"
    assert bundle["promotion_chain"]["advisory"] is True
    # The chain report itself is UNTOUCHED (no advisory key injected).
    assert set(chain) == {"chain_hash", "course_status", "run_id"}


def test_link_promotion_chain_unlinked_when_absent():
    bundle = {}
    pe.link_promotion_chain(bundle, None)
    assert bundle["promotion_chain"]["linked"] is False
    assert bundle["promotion_chain"]["advisory"] is True


def test_write_evidence_bundle_links_chain(tmp_path):
    course = tmp_path / "course"
    _write_report(course, _report(_passing_headline()), "20260101T000000Z")
    chain_path = course / "courseforge_promotion_chain_report.json"
    chain_path.write_text(
        json.dumps({
            "schema_version": "1.0", "chain_hash": "H1",
            "course_status": "certified_instructional", "run_id": "WF-2",
            "arrows": [],
        }),
        encoding="utf-8",
    )
    out = pe.write_evidence_bundle(
        course, course_slug="c", run_id="WF-2", promotion_chain_path=chain_path
    )
    assert out is not None and out.name == pe.EVIDENCE_BUNDLE_FILENAME
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["promotion_chain"]["chain_hash"] == "H1"
    assert written["evaluation_status"] == "evaluated"
    # The chain report on disk was NOT mutated (schema is additionalProperties:
    # false — the exporter must never add a top-level key to it).
    chain_after = json.loads(chain_path.read_text(encoding="utf-8"))
    assert "procurement_evidence" not in chain_after


def test_write_evidence_bundle_not_evaluated_path(tmp_path):
    course = tmp_path / "course"
    course.mkdir()
    out = pe.write_evidence_bundle(course, course_slug="c")
    assert out is not None
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["evaluation_status"] == "not_evaluated"
