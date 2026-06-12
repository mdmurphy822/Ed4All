"""Tests for the P4 refusal-calibration extensions (refusal.py).

Covers: distribution-overlap statistics (separable vs overlapping, the honest
stays-unpinned evidence — risk R8), the precision>=0.90 AND recall>=0.95 pin
rule against the overlap evidence, and the per-engine v1.1 probe dry_runs[]
negative extraction (with legacy single-dry_run fall-back).

CI-safe: pure math on synthetic distributions + synthetic probe docs; no live
retriever, no LLM, no real slugs.
"""
from __future__ import annotations

from lib.retrieval.refusal import (
    calibrate_from_distributions,
    negatives_from_probe_dry_runs,
)
from lib.retrieval.refusal import _overlap_stats


# ===========================================================================
# Overlap statistics
# ===========================================================================

def test_overlap_separable():
    # positives all above negatives → separable, min_gap>0, overlap 0.
    stats = _overlap_stats(positives=[0.6, 0.7, 0.9], negatives=[0.1, 0.2, 0.3])
    assert stats["separable"] is True
    assert stats["min_gap"] > 0
    assert stats["overlap_fraction"] == 0.0


def test_overlap_overlapping():
    # min(pos)=0.3 < max(neg)=0.5 → overlap band [0.3,0.5]; several samples in it.
    stats = _overlap_stats(positives=[0.3, 0.4, 0.8], negatives=[0.2, 0.5])
    assert stats["separable"] is False
    assert stats["min_gap"] <= 0
    assert stats["overlap_fraction"] > 0.0


def test_overlap_needs_both_arms():
    stats = _overlap_stats(positives=[0.5], negatives=[])
    assert stats["separable"] is None
    assert stats["overlap_fraction"] is None


# ===========================================================================
# calibrate_from_distributions records overlap + honest unpinnable verdict
# ===========================================================================

def test_separable_distribution_pins_and_reports_overlap():
    # Clean separation → a threshold meets precision>=0.90 AND recall>=0.95.
    result = calibrate_from_distributions(
        course_slug="syn", engine="hybrid-rrf",
        positives=[0.6, 0.7, 0.8, 0.9, 0.65],
        negatives=[0.1, 0.15, 0.2, 0.05, 0.12],
        score_floor=0.0, min_passages_above_floor=1,
    )
    assert result.recommended is not None
    assert result.overlap["separable"] is True
    assert result.overlap["overlap_fraction"] == 0.0
    d = result.to_dict()
    assert "overlap" in d
    assert d["recommended"]["refusal_precision"] >= 0.90
    assert d["recommended"]["answer_recall"] >= 0.95


def test_overlapping_distribution_stays_unpinned_with_evidence():
    # Heavy overlap → no threshold meets the rule; recommended=None BUT the
    # overlap stats carry the credible stay-unpinned evidence (risk R8).
    result = calibrate_from_distributions(
        course_slug="syn", engine="hybrid-rrf",
        positives=[0.30, 0.35, 0.40, 0.45, 0.50],
        negatives=[0.30, 0.35, 0.40, 0.45, 0.50],
        score_floor=0.0, min_passages_above_floor=1,
    )
    assert result.recommended is None
    assert result.overlap["separable"] is False
    assert result.overlap["overlap_fraction"] > 0.0
    assert result.overlap["min_gap"] <= 0


# ===========================================================================
# Per-engine v1.1 probe dry_runs[] extraction
# ===========================================================================

def _probe(pid, dry_runs=None, legacy=None):
    p = {"probe_id": pid, "question_text": "q", "category": "off_topic",
         "why_unanswerable": "x" * 25}
    if legacy is not None:
        p["dry_run"] = legacy
    if dry_runs is not None:
        p["dry_runs"] = dry_runs
    return p


def _dr(engine, top_score, n_results=1):
    return {"verified": True, "engine": engine, "limit": 8,
            "n_results": n_results, "top_score": top_score,
            "top_passage_answers": False}


def test_negatives_prefers_per_engine_dry_runs():
    doc = {
        "probes": [
            _probe("rp-x-0001", dry_runs=[
                _dr("lexical", 4.5), _dr("semantic", 0.3), _dr("hybrid-rrf", 0.02),
            ]),
            _probe("rp-x-0002", dry_runs=[
                _dr("lexical", 3.9), _dr("semantic", 0.25), _dr("hybrid-rrf", 0.01),
            ]),
        ]
    }
    sem_scores, sem_above = negatives_from_probe_dry_runs(doc, "semantic")
    assert sem_scores == [0.3, 0.25]
    assert sem_above == [1, 1]
    hyb_scores, _ = negatives_from_probe_dry_runs(doc, "hybrid-rrf")
    assert hyb_scores == [0.02, 0.01]


def test_negatives_falls_back_to_legacy_dry_run():
    # No dry_runs[] block — only the legacy single dry_run (lexical).
    doc = {"engine": "lexical", "probes": [
        _probe("rp-x-0001", legacy=_dr("lexical", 4.2)),
    ]}
    lex_scores, _ = negatives_from_probe_dry_runs(doc, "lexical")
    assert lex_scores == [4.2]
    # The legacy lexical dry_run does NOT speak for the semantic engine.
    sem_scores, _ = negatives_from_probe_dry_runs(doc, "semantic")
    assert sem_scores == []


def test_negatives_n_above_zero_when_no_results():
    doc = {"probes": [
        _probe("rp-x-0001", dry_runs=[_dr("lexical", 0.0, n_results=0)]),
    ]}
    scores, n_above = negatives_from_probe_dry_runs(doc, "lexical")
    assert scores == [0.0]
    assert n_above == [0]  # zero-result dry-run never clears a count floor of 1


def test_negatives_empty_doc():
    assert negatives_from_probe_dry_runs(None, "lexical") == ([], [])
    assert negatives_from_probe_dry_runs({}, "lexical") == ([], [])


def test_per_engine_sweep_pins_hybrid_when_separable():
    """End-to-end: hybrid-rrf negatives from dry_runs[] + separable positives →
    the sweep pins (the scaled per-engine evidence the plan calls for)."""
    doc = {"probes": [
        _probe(f"rp-x-{i:04d}", dry_runs=[_dr("hybrid-rrf", 0.01)])
        for i in range(5)
    ]}
    neg_scores, neg_above = negatives_from_probe_dry_runs(doc, "hybrid-rrf")
    result = calibrate_from_distributions(
        course_slug="syn", engine="hybrid-rrf",
        positives=[0.030, 0.028, 0.025, 0.032, 0.027],  # rank-1-in-both fused
        negatives=neg_scores,
        score_floor=0.0, min_passages_above_floor=1,
        neg_n_above=neg_above,
    )
    assert result.recommended is not None
    assert result.overlap["separable"] is True
