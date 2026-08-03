"""Phase IA WS3 (D4) — tests for lib/retrieval/refusal.py + the seed probe sets.

Covers (no LLM, no models, no network — wave A is deterministic arithmetic):

* Verdict truth table + boundary cases (top-score floor AND count-above-floor).
* Permissive v0-uncalibrated default behaviour.
* Calibration math on CONSTRUCTED distributions: separable arms propose a
  threshold; overlapping arms honestly fall back to recommended=None.
* Deterministic calibration output (same inputs → byte-identical dict).
* The calibration-report shape.
* The neutral mini-course refusal-probe fixture validates against the schema,
  every probe is dry-run-verified, and the categories match the authoring
  procedure. Ignored operator courses are never test inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from lib.retrieval import refusal
from lib.retrieval.refusal import (
    DEFAULT_POLICIES,
    PINNED_POLICIES,
    RRF_K,
    POLICY_VERSION_PINNED,
    POLICY_VERSION_UNCALIBRATED,
    REASON_LOW_CONFIDENCE,
    RefusalPolicy,
    calibrate_from_distributions,
    default_policy_for,
    evaluate_confidence,
    resolve_policy,
    should_refuse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "retrieval" / "refusal_probes.schema.json"
# --------------------------------------------------------------------------- #
# A tiny score-bearing stand-in (duck-typed like RetrievedPassage / RetrievalResult)
# --------------------------------------------------------------------------- #


@dataclass
class _P:
    score: float


def _passages(*scores):
    return [_P(s) for s in scores]


# --------------------------------------------------------------------------- #
# Verdict truth table + boundaries
# --------------------------------------------------------------------------- #


def _policy(min_top=1.0, floor=0.5, min_above=1):
    return RefusalPolicy(
        engine="lexical",
        min_top_score=min_top,
        score_floor=floor,
        min_passages_above_floor=min_above,
        policy_version="test.v0",
    )


def test_confident_when_top_score_and_count_met():
    v = evaluate_confidence(_passages(2.0, 0.6, 0.4), _policy())
    assert v.confident is True
    assert v.signals["top_score"] == 2.0
    assert v.signals["n_above_floor"] == 2.0  # 2.0 and 0.6 clear floor=0.5


def test_refuse_when_top_score_below_floor():
    v = evaluate_confidence(_passages(0.9, 0.6), _policy(min_top=1.0))
    assert v.confident is False  # top 0.9 < 1.0


def test_refuse_when_count_below_min_above():
    # top clears min_top, but only one passage clears floor and we require 2
    v = evaluate_confidence(_passages(2.0, 0.3), _policy(min_above=2))
    assert v.confident is False


def test_boundary_top_score_exactly_at_floor_is_confident():
    v = evaluate_confidence(_passages(1.0, 0.5), _policy(min_top=1.0, floor=0.5))
    assert v.confident is True  # >= is inclusive on both signals


def test_empty_passages_never_confident():
    v = evaluate_confidence([], _policy())
    assert v.confident is False
    assert v.signals["top_score"] == 0.0
    assert v.signals["n_above_floor"] == 0.0
    assert v.signals["n_passages"] == 0.0


def test_should_refuse_inverts_confidence_and_sets_reason_code():
    refuse_v = should_refuse(_passages(0.1), _policy(min_top=1.0))
    assert refuse_v.refuse is True
    assert refuse_v.reason_code == REASON_LOW_CONFIDENCE
    assert refuse_v.engine == "lexical"

    ok_v = should_refuse(_passages(5.0, 1.0), _policy(min_top=1.0))
    assert ok_v.refuse is False
    assert ok_v.reason_code is None


def test_should_refuse_accepts_dict_passages():
    refuse_v = should_refuse([{"score": 0.0}], _policy(min_top=1.0))
    assert refuse_v.refuse is True


def test_verdict_to_dict_round_trips_signals_and_policy_version():
    v = evaluate_confidence(_passages(2.0), _policy())
    d = v.to_dict()
    assert d["policy_version"] == "test.v0"
    assert d["engine"] == "lexical"
    assert set(d["signals"]) >= {"top_score", "n_above_floor", "mean_top3"}


# --------------------------------------------------------------------------- #
# Permissive v0-uncalibrated defaults
# --------------------------------------------------------------------------- #


def test_default_policies_are_marked_uncalibrated():
    for engine in ("semantic", "lexical", "hybrid-rrf"):
        assert DEFAULT_POLICIES[engine].policy_version == POLICY_VERSION_UNCALIBRATED


def test_default_policy_for_unknown_engine_falls_back_to_lexical():
    assert default_policy_for("nonsense") is DEFAULT_POLICIES["lexical"]


def test_lexical_default_is_permissive_passes_a_typical_hit():
    # A single retained BM25 passage scoring well above the floor is confident.
    v = evaluate_confidence(_passages(3.8), DEFAULT_POLICIES["lexical"])
    assert v.confident is True


def test_semantic_default_refuses_low_cosine():
    v = evaluate_confidence(_passages(0.20, 0.18), DEFAULT_POLICIES["semantic"])
    assert v.confident is False  # 0.20 < min_top_score 0.30


def test_hybrid_rrf_default_does_not_refuse_a_perfect_dual_list_retrieval():
    # A real RRF fusion: the top doc is rank-1 in BOTH arms (top_score
    # 2/(RRF_K+1)), and the next passages are single-arm rank-r hits scoring
    # 1/(RRF_K+r). The old lexical-borrowed floors (min_top_score=1.0) made this
    # mathematically unreachable and refused 100% of hybrid-rrf queries.
    top = 2.0 / (RRF_K + 1)
    rest = [1.0 / (RRF_K + r) for r in (1, 2, 3, 4)]
    verdict = should_refuse(_passages(top, *rest), DEFAULT_POLICIES["hybrid-rrf"])
    assert verdict.refuse is False
    assert verdict.signals["top_score"] >= DEFAULT_POLICIES["hybrid-rrf"].min_top_score


def test_hybrid_rrf_default_min_top_score_is_satisfiable_by_max_achievable_score():
    # Structural guard: the floor must be reachable by the engine's MAXIMUM
    # achievable fused score (a doc ranked 1 in both arms = 2/(RRF_K+1)).
    # An unsatisfiable floor refuses every query unconditionally.
    max_achievable = 2.0 / (RRF_K + 1)
    assert DEFAULT_POLICIES["hybrid-rrf"].min_top_score <= max_achievable


def test_refusal_rrf_k_matches_semantic_retriever_constant():
    # Cross-check the module-local RRF_K mirrors the retriever's DEFAULT_RRF_K.
    # Skip cleanly if importing the retriever needs optional deps it can't load.
    try:
        from LibV2.tools.libv2.semantic_retriever import DEFAULT_RRF_K
    except Exception as exc:  # pragma: no cover - optional-dep guard
        pytest.skip(f"semantic_retriever not importable here: {exc}")
    assert RRF_K == DEFAULT_RRF_K


# --------------------------------------------------------------------------- #
# Calibration math on constructed distributions
# --------------------------------------------------------------------------- #


def test_calibration_separable_proposes_threshold():
    # answerable scores high, unanswerable scores low — cleanly separable.
    positives = [5.0, 6.0, 7.0, 8.0, 5.5]
    negatives = [0.5, 0.7, 0.3, 0.6, 0.4]
    result = calibrate_from_distributions(
        course_slug="sep",
        engine="semantic",
        positives=positives,
        negatives=negatives,
        score_floor=0.0,  # all pass the count floor
        min_passages_above_floor=1,
    )
    assert result.recommended is not None
    rec = result.recommended
    # A threshold between the arms refuses all negatives, answers all positives.
    assert rec["refusal_precision"] >= 0.90
    assert rec["answer_recall"] >= 0.95
    assert 0.7 <= rec["min_top_score"] <= 5.0


def test_calibration_overlapping_falls_back_to_none():
    # arms fully interleaved — no threshold separates them.
    positives = [1.0, 2.0, 3.0, 4.0, 5.0]
    negatives = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = calibrate_from_distributions(
        course_slug="overlap",
        engine="lexical",
        positives=positives,
        negatives=negatives,
        score_floor=0.0,
        min_passages_above_floor=1,
    )
    assert result.recommended is None
    assert result.fallback_policy_version == POLICY_VERSION_UNCALIBRATED


def test_calibration_report_shape():
    result = calibrate_from_distributions(
        course_slug="shape",
        engine="semantic",
        positives=[5.0, 6.0],
        negatives=[0.5, 0.6],
        score_floor=0.0,
        min_passages_above_floor=1,
        embedding_model_id="test-embed-1",
        generated_at="2026-06-09T00:00:00Z",
    )
    d = result.to_dict()
    assert d["schema_version"] == "1.0"
    assert d["course_slug"] == "shape"
    assert d["engine"] == "semantic"
    assert d["embedding_model_id"] == "test-embed-1"
    assert d["positives"]["n"] == 2
    assert d["negatives"]["n"] == 2
    assert "top_score" in d["positives"]
    assert isinstance(d["sweep"], list) and d["sweep"]
    for row in d["sweep"]:
        assert {
            "threshold",
            "answer_recall",
            "refusal_recall",
            "refusal_precision",
            "false_refusals_on_positives",
        } <= set(row)
    assert d["fallback_policy_version"] == POLICY_VERSION_UNCALIBRATED
    assert d["generated_at"] == "2026-06-09T00:00:00Z"


def test_calibration_is_deterministic():
    kwargs = dict(
        course_slug="det",
        engine="lexical",
        positives=[3.0, 4.0, 5.0],
        negatives=[0.4, 0.6, 0.5],
        score_floor=0.0,
        min_passages_above_floor=1,
        generated_at="2026-06-09T00:00:00Z",
    )
    a = calibrate_from_distributions(**kwargs).to_dict()
    b = calibrate_from_distributions(**kwargs).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_calibration_count_floor_forces_refusal_when_zero_results():
    # A negative that returned ZERO passages above the floor is always refused,
    # regardless of threshold — the count signal alone catches it.
    result = calibrate_from_distributions(
        course_slug="countfloor",
        engine="semantic",
        positives=[5.0, 6.0],
        negatives=[0.0, 0.0],
        score_floor=0.5,
        min_passages_above_floor=1,
        pos_n_above=[2, 2],
        neg_n_above=[0, 0],  # nothing cleared the floor
    )
    # at threshold 0.0 the negatives still refuse (n_above 0 < 1).
    row0 = result.sweep[0]
    assert row0["threshold"] == 0.0
    assert row0["refusal_recall"] == 1.0


# --------------------------------------------------------------------------- #
# Seed probe sets — schema validation + verification invariants
# --------------------------------------------------------------------------- #

MINI_PROBES = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "retrieval"
    / "mini_course"
    / "retrieval_eval"
    / "refusal_probes.json"
)


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_mini_course_fixture_probe_set_validates_and_is_verified():
    jsonschema = pytest.importorskip("jsonschema")
    assert MINI_PROBES.exists(), f"mini-course probes missing: {MINI_PROBES}"
    doc = json.loads(MINI_PROBES.read_text(encoding="utf-8"))
    jsonschema.validate(doc, _load_schema())
    assert len(doc["probes"]) == 3
    cats = {p["category"] for p in doc["probes"]}
    assert cats == {"off_topic", "adjacent_domain", "out_of_scope_detail"}
    for p in doc["probes"]:
        assert p["dry_run"]["verified"] is True
        assert p["dry_run"]["top_passage_answers"] is False


def test_fixture_probes_drive_an_end_to_end_overlap_calibration():
    """Overlapping neutral-fixture distributions select honest fallback."""
    doc = json.loads(MINI_PROBES.read_text(encoding="utf-8"))
    neg_scores = [p["dry_run"]["top_score"] for p in doc["probes"]]
    # An identical answerable arm is the worst-case overlap.
    pos_scores = list(neg_scores)  # worst case: identical → must fall back
    result = calibrate_from_distributions(
        course_slug=doc["course_slug"],
        engine="lexical",
        positives=pos_scores,
        negatives=neg_scores,
        score_floor=0.5,
        min_passages_above_floor=1,
    )
    assert result.recommended is None


# --------------------------------------------------------------------------- #
# Pinned (measured) policy resolution — § D4.2 measure-then-pin
# --------------------------------------------------------------------------- #

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
_BGE_LARGE = "BAAI/bge-large-en-v1.5"


def test_pinned_policy_selected_for_bge_large_semantic_pair():
    """The current production embedding default (bge-large) resolves to its own
    measured pin, distinct from the MiniLM pin (cosine scales differ per model)."""
    policy = resolve_policy("semantic", _BGE_LARGE)
    assert policy is PINNED_POLICIES[("semantic", _BGE_LARGE)]
    assert policy.policy_version == POLICY_VERSION_PINNED
    assert policy.embedding_model_id == _BGE_LARGE
    # Floor + count signal inherited from the measured semantic default.
    assert policy.score_floor == DEFAULT_POLICIES["semantic"].score_floor
    assert (
        policy.min_passages_above_floor
        == DEFAULT_POLICIES["semantic"].min_passages_above_floor
    )
    # The two semantic pins are genuinely different objects with different
    # thresholds — a model swap is NOT a silent reuse of the other's cosine.
    assert policy is not PINNED_POLICIES[("semantic", _MINILM)]
    assert policy.min_top_score != PINNED_POLICIES[("semantic", _MINILM)].min_top_score


def test_hybrid_rrf_is_embedder_keyed_and_pinned_for_bge_large():
    """hybrid-rrf is keyed by embedding model (its fused score depends on the
    semantic arm). As of the 2026-06-12 single-course union-corpus calibration it
    is PINNED for bge-large (the precision-/recall-clean recommendation on the
    scaled-up frozen gold set), but UNKNOWN embedders + the no-embedder case
    still fall through to the v0-uncalibrated default (never a stale reuse)."""
    # A hybrid-rrf pin now exists — and only for bge-large.
    assert ("hybrid-rrf", _BGE_LARGE) in PINNED_POLICIES
    p_known = resolve_policy("hybrid-rrf", _BGE_LARGE)
    assert p_known is PINNED_POLICIES[("hybrid-rrf", _BGE_LARGE)]
    assert p_known.policy_version == POLICY_VERSION_PINNED
    assert p_known.embedding_model_id == _BGE_LARGE
    # Floor + count signal inherit the RRF-scale v0 default (only min_top_score
    # is re-pinned). The pin is a fused-RRF threshold, far below the cosine scale.
    assert p_known.score_floor == DEFAULT_POLICIES["hybrid-rrf"].score_floor
    assert (
        p_known.min_passages_above_floor
        == DEFAULT_POLICIES["hybrid-rrf"].min_passages_above_floor
    )
    assert p_known.min_top_score < DEFAULT_POLICIES["semantic"].min_top_score
    # An unknown embedder is unpinned (never a stale reuse of bge-large's pin).
    p_unknown = resolve_policy("hybrid-rrf", "some-other/embedder-v9")
    assert p_unknown is DEFAULT_POLICIES["hybrid-rrf"]
    assert p_unknown.policy_version == POLICY_VERSION_UNCALIBRATED
    # And a hybrid-rrf request with no embedder also falls back (never a pin).
    assert resolve_policy("hybrid-rrf", None) is DEFAULT_POLICIES["hybrid-rrf"]


def test_pinned_policy_selected_for_matching_semantic_pair():
    """A (semantic, <pinned model>) pair returns the measured pin, not the
    permissive v0 default."""
    policy = resolve_policy("semantic", _MINILM)
    assert policy is PINNED_POLICIES[("semantic", _MINILM)]
    assert policy.policy_version == POLICY_VERSION_PINNED
    assert policy.embedding_model_id == _MINILM
    # The pin is measurably stricter than the v0-uncalibrated cosine guess.
    assert policy.min_top_score > DEFAULT_POLICIES["semantic"].min_top_score
    # Floor + count signal are inherited from the measured default.
    assert policy.score_floor == DEFAULT_POLICIES["semantic"].score_floor
    assert (
        policy.min_passages_above_floor
        == DEFAULT_POLICIES["semantic"].min_passages_above_floor
    )


def test_pinned_policy_selected_for_lexical():
    """The model-agnostic lexical pin is keyed (lexical, None)."""
    policy = resolve_policy("lexical")
    assert policy is PINNED_POLICIES[("lexical", None)]
    assert policy.policy_version == POLICY_VERSION_PINNED
    assert policy.embedding_model_id is None
    # An incidental embedding_model_id is ignored for lexical (no embedder).
    assert resolve_policy("lexical", "anything") is policy


def test_unknown_semantic_model_falls_back_to_uncalibrated():
    """A semantic engine with an UNKNOWN embedder must NOT silently reuse the
    pinned cosine — it falls back to the v0-uncalibrated default."""
    policy = resolve_policy("semantic", "some-other/embedder-v9")
    assert policy is DEFAULT_POLICIES["semantic"]
    assert policy.policy_version == POLICY_VERSION_UNCALIBRATED
    # A semantic engine with NO model id is also unknown (never reuse a pin).
    none_policy = resolve_policy("semantic", None)
    assert none_policy is DEFAULT_POLICIES["semantic"]
    assert none_policy.policy_version == POLICY_VERSION_UNCALIBRATED


def test_unknown_engine_falls_back_to_lexical_default():
    """An engine with no pin and no default rides the permissive lexical default."""
    policy = resolve_policy("hybrid-rrf")  # no embedder → no pin match
    assert policy is DEFAULT_POLICIES["hybrid-rrf"]
    assert policy.policy_version == POLICY_VERSION_UNCALIBRATED
    assert resolve_policy("nonsense") is DEFAULT_POLICIES["lexical"]
