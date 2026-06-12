"""Tests for attribution-threshold calibration (attribution_calibrate.py).

CI-safe: pure calibration math on synthetic score sets + a fake retriever for
the driver (no live retriever, no LLM, no real slugs). Covers the sweep
precision/recall/F1 math, both recommendation rules, the claim-proxy selection,
the live-capture rationale parser, and the ADD_MIN_SHINGLE-knob assessment.
"""
from __future__ import annotations

import json

from lib.retrieval.attribution_calibrate import (
    ScoreSample,
    calibrate,
    calibrate_from_samples,
    claim_proxy_for_question,
    ingest_prune_captures,
)
from lib.retrieval.attribution_calibrate import _parse_prune_rationale


# ===========================================================================
# Claim proxy selection
# ===========================================================================

def test_claim_proxy_prefers_key_points():
    q = {
        "question_text": "qt",
        "answer_note": "an",
        "expected_key_points": ["point one", "point two"],
    }
    proxy, source = claim_proxy_for_question(q)
    assert source == "key_points"
    assert "point one" in proxy and "point two" in proxy


def test_claim_proxy_falls_back_to_answer_note_then_question():
    assert claim_proxy_for_question(
        {"question_text": "qt", "answer_note": "the note"}
    ) == ("the note", "answer_note")
    assert claim_proxy_for_question({"question_text": "the q"}) == (
        "the q", "question_text"
    )
    # empty key points list → not used
    assert claim_proxy_for_question(
        {"question_text": "the q", "expected_key_points": []}
    )[1] == "question_text"


# ===========================================================================
# Sweep + recommendation math on synthetic separable / overlapping sets
# ===========================================================================

def _samples(rel_scores, non_rel_scores):
    return (
        [ScoreSample(score=s, relevant=True) for s in rel_scores]
        + [ScoreSample(score=s, relevant=False) for s in non_rel_scores]
    )


def test_separable_recommendation_precision_floored():
    # Relevant scores high (>=0.6), non-relevant low (<=0.2): perfectly
    # separable. A threshold ~0.6 gives precision 1.0, recall 1.0.
    samples = _samples([0.6, 0.7, 0.9], [0.0, 0.1, 0.2])
    result = calibrate_from_samples(
        course_slug="syn", engine="lexical", samples=samples, n_gold_questions=3,
    )
    rec = result.recommended
    assert rec is not None
    assert rec["rule"].startswith("precision_floored")
    assert rec["precision"] == 1.0
    assert rec["recall"] == 1.0
    # f1_max is also reported and is 1.0 here.
    assert rec["f1_max"]["f1"] == 1.0


def test_overlapping_falls_back_to_f1_max_when_floor_unmet():
    # Heavy overlap so NO threshold reaches precision 0.80: 3 relevant and 3
    # non-relevant all in [0.3,0.5]. precision_floored is None → f1_max rule.
    samples = _samples([0.3, 0.4, 0.5], [0.3, 0.4, 0.5])
    result = calibrate_from_samples(
        course_slug="syn", engine="lexical", samples=samples, n_gold_questions=3,
        precision_floor=0.80,
    )
    rec = result.recommended
    assert rec is not None
    assert rec["rule"].startswith("f1_max")
    assert rec["precision_floored"] is None


def test_sweep_precision_recall_monotonicity():
    samples = _samples([0.5, 0.8], [0.1, 0.6])
    result = calibrate_from_samples(
        course_slug="syn", engine="lexical", samples=samples, n_gold_questions=2,
    )
    sweep = result.sweep
    # recall is non-increasing as threshold rises (higher bar → fewer tp).
    recalls = [row["recall"] for row in sweep]
    assert recalls == sorted(recalls, reverse=True)
    # threshold 0.0 captures everything: recall 1.0.
    assert sweep[0]["threshold"] == 0.0
    assert sweep[0]["recall"] == 1.0


def test_distributions_recorded():
    samples = _samples([0.6, 0.8], [0.1])
    result = calibrate_from_samples(
        course_slug="syn", engine="lexical", samples=samples, n_gold_questions=2,
    )
    d = result.to_dict()["distributions"]
    assert d["gold_relevant"]["n"] == 2
    assert d["gold_non_relevant"]["n"] == 1
    assert d["gold_relevant"]["median"] == 0.7


def test_empty_samples_recommendation_none():
    result = calibrate_from_samples(
        course_slug="syn", engine="lexical", samples=[], n_gold_questions=0,
    )
    assert result.recommended is None


# ===========================================================================
# ADD_MIN_SHINGLE knob assessment
# ===========================================================================

def test_add_knob_warranted_when_cited_median_far_below():
    # cited shingle median ~0.1, far below the 0.50 ADD bar → warranted.
    result = calibrate_from_samples(
        course_slug="syn", engine="lexical", samples=_samples([0.6], [0.1]),
        n_gold_questions=1, capture_cited_scores=[0.05, 0.1, 0.15],
    )
    w = result.add_min_shingle_knob_warranted
    assert w["warranted"] is True
    assert "BELOW" in w["reason"]
    assert w["evidence_stream"] == "capture_cited"


def test_add_knob_not_warranted_near_bar():
    result = calibrate_from_samples(
        course_slug="syn", engine="lexical", samples=_samples([0.6], [0.1]),
        n_gold_questions=1, capture_cited_scores=[0.48, 0.5, 0.52],
    )
    w = result.add_min_shingle_knob_warranted
    assert w["warranted"] is False


def test_add_knob_uses_gold_relevant_when_no_captures():
    result = calibrate_from_samples(
        course_slug="syn", engine="lexical",
        samples=_samples([0.05, 0.1], []), n_gold_questions=1,
    )
    w = result.add_min_shingle_knob_warranted
    assert w["evidence_stream"] == "gold_relevant"


# ===========================================================================
# Live citation-prune capture ingestion (second evidence stream)
# ===========================================================================

def test_parse_prune_rationale():
    rationale = (
        "citation prune+add pruned for query abc (course=x, engine=lexical, "
        "mode=on); n_claims=3 min_overlap=0.25 shingle_size=4 add_min_shingle=0.5; "
        "per_citation=[c1@0.8/0.9*2(cited), c2@0.1/0.2*0(uncited), "
        "c3@0.6/0.7*1(cited)] kept=[c1, c3] pruned=[c2] added=[] inline_vetoed=[]"
    )
    parsed = _parse_prune_rationale(rationale)
    assert len(parsed) == 3
    cited = [p for p in parsed if p[4] == "cited"]
    uncited = [p for p in parsed if p[4] == "uncited"]
    assert sorted(p[1] for p in cited) == [0.6, 0.8]
    assert uncited[0][1] == 0.1


def test_parse_prune_rationale_tolerates_garbage():
    assert _parse_prune_rationale("") == []
    assert _parse_prune_rationale("no per_citation block here") == []


def test_ingest_prune_captures(tmp_path):
    cap_dir = tmp_path / "phase_libv2-answer"
    cap_dir.mkdir()
    events = [
        {
            "decision_type": "grounded_answer_citation_prune",
            "rationale": "per_citation=[c1@0.8/0.9*2(cited), c2@0.1/0.0*0(uncited)]",
        },
        {
            "decision_type": "grounded_answer_citation_prune",
            "rationale": "per_citation=[c3@0.55/0.6*1(cited)]",
        },
        # A non-prune event must be ignored.
        {"decision_type": "something_else", "rationale": "per_citation=[x@9/9*9(cited)]"},
    ]
    (cap_dir / "decisions_001.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    cited, uncited = ingest_prune_captures(cap_dir)
    assert sorted(cited) == [0.55, 0.8]
    assert uncited == [0.1]


def test_ingest_prune_captures_missing_dir(tmp_path):
    cited, uncited = ingest_prune_captures(tmp_path / "nope")
    assert cited == [] and uncited == []


# ===========================================================================
# Driver with a fake retriever (no live retrieval, no LLM, no real slug)
# ===========================================================================

class _FakePassage:
    def __init__(self, chunk_id, text):
        self.chunk_id = chunk_id
        self.text = text


def test_calibrate_driver_with_fake_retriever(tmp_path, monkeypatch):
    slug = "syn-course"
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / slug / "retrieval_eval"
    course_dir.mkdir(parents=True)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))

    gold = {
        "schema_version": "1.1",
        "course_slug": slug,
        "questions": [
            {
                "question_id": "gq-syn-course-0001",
                "question_text": "what is a widget",
                "expected_key_points": [
                    "A widget is a small reusable interface component",
                ],
                "relevant_passages": [{"chunk_id": "rel1"}],
            }
        ],
    }
    (course_dir / "gold_set.json").write_text(json.dumps(gold), encoding="utf-8")

    def _fake_retrieve(libv2_root, course_slug, query, *, engine, limit):
        # rel1 contains the key-point text → high shingle; noise1 does not.
        return [
            _FakePassage(
                "rel1",
                "A widget is a small reusable interface component used widely.",
            ),
            _FakePassage("noise1", "Completely different sentence about geology."),
        ]

    result = calibrate(
        slug, engine="lexical", repo_root=tmp_path, retrieve_fn=_fake_retrieve,
        capture_dir=tmp_path / "no-captures", write=True,
    )
    assert result.n_gold_questions == 1
    assert result.claim_source_counts == {"key_points": 1}
    d = result.to_dict()["distributions"]
    # rel1 is the gold positive (high score); noise1 the negative (~0).
    assert d["gold_relevant"]["n"] == 1
    assert d["gold_non_relevant"]["n"] == 1
    assert d["gold_relevant"]["max"] > d["gold_non_relevant"]["max"]
    # Artifact written under retrieval_eval/.
    written = list(course_dir.glob("attribution_calibration_*.json"))
    assert len(written) == 1
