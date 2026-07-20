"""Tests for the E3 risk-coverage / selective-QA view (lib.retrieval.grounded_eval_risk).

Pure post-processing math — no network, no model, no torch. Every metric is
checked against a hand-computed value on tiny fixtures, plus the degenerate
(empty / single-class) paths that must return None (never a fabricated number).
The report-extraction path is checked against a stored-report-shaped dict.
"""
from __future__ import annotations

import pytest

from lib.retrieval.grounded_eval_risk import (
    RISK_VIEW_SCHEMA_VERSION,
    abstention_auroc,
    aurc,
    expected_calibration_error,
    pairs_from_report,
    risk_coverage_curve,
    selective_qa_from_report,
    selective_qa_view,
)


# ===========================================================================
# risk_coverage_curve
# ===========================================================================

def test_risk_coverage_curve_descending_thresholds():
    # 4 items: confidences 0.9(ok) 0.8(ok) 0.4(bad) 0.2(ok).
    pairs = [(0.9, True), (0.8, True), (0.4, False), (0.2, True)]
    curve = risk_coverage_curve(pairs)
    # thresholds are the distinct confidences, descending.
    assert [row["threshold"] for row in curve] == [0.9, 0.8, 0.4, 0.2]
    # highest-confidence-only slice: 1 answered, 1 correct → risk 0.
    assert curve[0]["n_answered"] == 1
    assert curve[0]["coverage"] == pytest.approx(0.25)
    assert curve[0]["accuracy"] == pytest.approx(1.0)
    assert curve[0]["risk"] == pytest.approx(0.0)
    # full coverage (threshold 0.2): 4 answered, 3 correct → accuracy 0.75.
    assert curve[-1]["coverage"] == pytest.approx(1.0)
    assert curve[-1]["accuracy"] == pytest.approx(0.75)
    assert curve[-1]["risk"] == pytest.approx(0.25)


def test_risk_coverage_curve_empty():
    assert risk_coverage_curve([]) == []


# ===========================================================================
# AURC
# ===========================================================================

def test_aurc_perfect_ordering_all_correct_is_zero():
    # All correct → risk 0 at every prefix → AURC 0.
    pairs = [(0.9, True), (0.5, True), (0.1, True)]
    assert aurc(pairs) == pytest.approx(0.0)


def test_aurc_all_incorrect_is_one():
    pairs = [(0.9, False), (0.5, False), (0.1, False)]
    assert aurc(pairs) == pytest.approx(1.0)


def test_aurc_hand_computed():
    # Sorted desc: (0.9,ok)(0.6,bad)(0.3,ok). Prefix risks: 0/1, 1/2, 1/3.
    # AURC = (0 + 0.5 + 1/3) / 3.
    pairs = [(0.3, True), (0.9, True), (0.6, False)]
    assert aurc(pairs) == pytest.approx((0.0 + 0.5 + 1 / 3) / 3)


def test_aurc_empty_is_none():
    assert aurc([]) is None


# ===========================================================================
# abstention AUROC
# ===========================================================================

def test_auroc_perfect_separation():
    # Every correct outranks every incorrect → AUROC 1.0.
    pairs = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    assert abstention_auroc(pairs) == pytest.approx(1.0)


def test_auroc_inverted_separation():
    # Confidence anti-correlated with correctness → AUROC 0.0.
    pairs = [(0.1, True), (0.2, True), (0.8, False), (0.9, False)]
    assert abstention_auroc(pairs) == pytest.approx(0.0)


def test_auroc_ties_credited_half():
    # One correct + one incorrect at the SAME confidence → tie → AUROC 0.5.
    pairs = [(0.5, True), (0.5, False)]
    assert abstention_auroc(pairs) == pytest.approx(0.5)


def test_auroc_single_class_is_none():
    assert abstention_auroc([(0.9, True), (0.5, True)]) is None
    assert abstention_auroc([(0.9, False)]) is None
    assert abstention_auroc([]) is None


# ===========================================================================
# ECE
# ===========================================================================

def test_ece_perfectly_calibrated_is_zero():
    # Confidence 1.0 + correct, confidence 0.0 + incorrect → per-bin gap 0.
    pairs = [(1.0, True), (1.0, True), (0.0, False), (0.0, False)]
    out = expected_calibration_error(pairs, bins=10)
    assert out["ece"] == pytest.approx(0.0)


def test_ece_miscalibrated():
    # 2 items at conf 0.9 but both wrong → bin accuracy 0, gap 0.9, weight 1.0.
    pairs = [(0.9, False), (0.9, False)]
    out = expected_calibration_error(pairs, bins=10)
    assert out["ece"] == pytest.approx(0.9)
    assert len(out["bin_table"]) == 1
    assert out["bin_table"][0]["accuracy"] == pytest.approx(0.0)
    assert out["bin_table"][0]["mean_confidence"] == pytest.approx(0.9)


def test_ece_empty_is_none():
    out = expected_calibration_error([], bins=10)
    assert out["ece"] is None
    assert out["bin_table"] == []


def test_ece_bins_fallback_on_garbage():
    # bins <= 0 / non-int → default bins.
    out = expected_calibration_error([(0.5, True)], bins=0)
    assert out["bins"] == 10
    out2 = expected_calibration_error([(0.5, True)], bins=-3)
    assert out2["bins"] == 10


# ===========================================================================
# selective_qa_view top-level
# ===========================================================================

def test_selective_qa_view_rollup():
    records = [
        {"confidence": 0.9, "correct": True},
        {"confidence": 0.8, "correct": True},
        {"confidence": 0.3, "correct": False},
    ]
    view = selective_qa_view(records)
    assert view["schema_version"] == RISK_VIEW_SCHEMA_VERSION
    assert view["basis"] == "records"
    assert view["n_scored"] == 3
    assert view["n_correct"] == 2
    assert view["n_incorrect"] == 1
    assert view["base_accuracy"] == pytest.approx(2 / 3)
    # correct all rank above incorrect → AUROC 1.0.
    assert view["abstention_auroc"] == pytest.approx(1.0)
    assert view["aurc"] is not None
    assert view["ece"] is not None
    assert view["coverage_accuracy_curve"]


def test_selective_qa_view_empty_is_none_basis():
    view = selective_qa_view([])
    assert view["basis"] == "none"
    assert view["n_scored"] == 0
    assert view["aurc"] is None
    assert view["abstention_auroc"] is None
    assert view["ece"] is None
    assert view["base_accuracy"] is None
    assert view["coverage_accuracy_curve"] == []


def test_selective_qa_view_drops_null_confidence():
    # A record with a null / non-numeric confidence carries no basis → dropped.
    records = [
        {"confidence": None, "correct": True},
        {"confidence": "x", "correct": True},
        {"confidence": True, "correct": True},  # bool guarded out
        {"confidence": 0.5, "correct": False},
    ]
    view = selective_qa_view(records)
    assert view["n_scored"] == 1


# ===========================================================================
# report extraction
# ===========================================================================

def _report_with_rows(rows):
    return {"schema_version": "1.8", "questions": rows}


def test_pairs_from_report_uses_groundedness_and_relevance():
    rows = [
        # answered, grounded 0.9, cited a relevant primary → correct.
        {"question_id": "q1", "status": "answered", "groundedness_rate": 0.9,
         "citation_relevant_primary": 1},
        # answered, grounded 0.4, cited NO relevant primary → incorrect.
        {"question_id": "q2", "status": "answered", "groundedness_rate": 0.4,
         "citation_relevant_primary": 0},
        # null groundedness → dropped (no confidence basis).
        {"question_id": "q3", "status": "answered", "groundedness_rate": None,
         "citation_relevant_primary": 1},
    ]
    pairs = pairs_from_report(_report_with_rows(rows))
    assert len(pairs) == 2
    by_id = {p["question_id"]: p for p in pairs}
    assert by_id["q1"]["confidence"] == pytest.approx(0.9)
    assert by_id["q1"]["correct"] is True
    assert by_id["q2"]["correct"] is False


def test_selective_qa_from_report_end_to_end():
    rows = [
        {"question_id": "q1", "status": "answered", "groundedness_rate": 1.0,
         "citation_relevant_primary": 1},
        {"question_id": "q2", "status": "answered", "groundedness_rate": 0.2,
         "citation_relevant_primary": 0},
    ]
    view = selective_qa_from_report(_report_with_rows(rows))
    assert view["n_scored"] == 2
    assert view["n_correct"] == 1
    # perfect separation (1.0 correct, 0.2 incorrect) → AUROC 1.0.
    assert view["abstention_auroc"] == pytest.approx(1.0)


def test_pairs_from_report_no_questions_key():
    assert pairs_from_report({"schema_version": "1.8"}) == []
    assert pairs_from_report({"questions": "not-a-list"}) == []


# ===========================================================================
# Integration: risk_coverage section wired into run_grounded_eval
# ===========================================================================

import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

from lib.retrieval.grounded_eval import run_grounded_eval  # noqa: E402

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "retrieval" / "mini_course"
)
_GOLD_MAP = {
    "What does a vector store index?": "mini_alpha_chunk_001",
    "How is retrieval quality commonly measured?": "mini_alpha_chunk_003",
    "Where does the course cover chunking strategies?": "mini_beta_chunk_005",
}
_GROUNDED_OK = {
    "available": True, "groundedness_rate": 1.0, "scored_count": 2,
    "unsupported_count": 0, "contradicted_count": 0, "claims": [],
}


class _FakeCitation:
    def __init__(self, chunk_id):
        self.chunk_id = chunk_id
        self.anchor_status = "resolved_exact"
        self.page_label = "P"
        self.text_quote = "q"

    def to_dict(self):
        return {"chunk_id": self.chunk_id, "anchor_status": self.anchor_status,
                "page_label": self.page_label, "text_quote": self.text_quote}


class _FakeAnswer:
    def __init__(self, status, citations, groundedness=None):
        self.status = status
        self.answer_text = "A." if citations else None
        self.citations = citations
        self.groundedness = groundedness
        self.latency_ms = 1.0
        self.model_id = "fake"
        self.prompt_version = "v"
        self.confidence = {"policy_version": "p"}


def _fn(repo_root, course_slug, query, *, with_groundedness=False, **kwargs):
    cid = _GOLD_MAP.get(query)
    if cid is None:
        return _FakeAnswer("refused_low_confidence", [])
    return _FakeAnswer(
        "answered", [_FakeCitation(cid)],
        groundedness=(_GROUNDED_OK if with_groundedness else None),
    )


@pytest.fixture
def libv2_course(tmp_path, monkeypatch):
    slug = "mini-retrieval-101"
    libv2_root = tmp_path / "LibV2"
    shutil.copytree(_FIXTURE, libv2_root / "courses" / slug)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return tmp_path, slug


def test_report_has_risk_coverage_section(libv2_course):
    repo_root, slug = libv2_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fn,
        with_groundedness=True, write=False,
    )
    rc = report["risk_coverage"]
    # 3 answered questions, each grounded 1.0 and cited its relevant primary →
    # all correct at confidence 1.0.
    assert rc["basis"] == "records"
    assert rc["n_scored"] == 3
    assert rc["n_correct"] == 3
    assert rc["base_accuracy"] == pytest.approx(1.0)
    # AURC of all-correct is 0; AUROC undefined (one class only) → None.
    assert rc["aurc"] == pytest.approx(0.0)
    assert rc["abstention_auroc"] is None


def test_report_risk_coverage_none_basis_when_groundedness_off(libv2_course):
    repo_root, slug = libv2_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fn,
        with_groundedness=False, write=False,
    )
    rc = report["risk_coverage"]
    # No groundedness scored → no confidence basis → basis "none".
    assert rc["basis"] == "none"
    assert rc["n_scored"] == 0
    assert rc["aurc"] is None
