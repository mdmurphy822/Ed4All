"""Phase IA WS3 (D4) — tests for lib/retrieval/refusal.py + the seed probe sets.

Covers (no LLM, no models, no network — wave A is deterministic arithmetic):

* Verdict truth table + boundary cases (top-score floor AND count-above-floor).
* Permissive v0-uncalibrated default behaviour.
* Calibration math on CONSTRUCTED distributions: separable arms propose a
  threshold; overlapping arms honestly fall back to recommended=None.
* Deterministic calibration output (same inputs → byte-identical dict).
* The calibration-report shape.
* The seed probe sets (any discovered real courses + the mini_course fixture)
  validate against the schema, every probe is dry-run-verified, and the
  categories / counts match the authoring procedure. Real courses are
  discovered dynamically from LibV2/courses/*; tier-2 tests skip when absent.
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

try:
    from lib.paths import libv2_path

    _LIBV2_COURSES = libv2_path() / "courses"
except Exception:  # pragma: no cover - defensive
    _LIBV2_COURSES = PROJECT_ROOT / "LibV2" / "courses"


def _discover_probe_courses():
    """Dynamically discover real courses carrying a seed refusal_probes.json.

    Course data dirs are gitignored user data; tests must NOT name course
    slugs. Glob whatever is present and skip cleanly when none exist.
    """
    if not _LIBV2_COURSES.is_dir():
        return []
    found = []
    for probe in sorted(_LIBV2_COURSES.glob("*/retrieval_eval/refusal_probes.json")):
        found.append(probe.parent.parent.name)
    return found


def _discover_calibration_courses():
    """Discover real courses carrying a committed refusal_calibration.json."""
    if not _LIBV2_COURSES.is_dir():
        return []
    found = []
    for cal in sorted(
        _LIBV2_COURSES.glob("*/retrieval_eval/refusal_calibration.json")
    ):
        found.append(cal.parent.parent.name)
    return found


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


def _real_probe_path(slug: str) -> Path:
    return _LIBV2_COURSES / slug / "retrieval_eval" / "refusal_probes.json"


_PROBE_COURSES = _discover_probe_courses()


@pytest.mark.skipif(
    not _PROBE_COURSES,
    reason="no gitignored seed refusal_probes.json under LibV2/courses/*",
)
@pytest.mark.parametrize("slug", _PROBE_COURSES)
def test_real_course_probe_set_validates_and_is_verified(slug):
    jsonschema = pytest.importorskip("jsonschema")
    path = _real_probe_path(slug)
    if not path.exists():
        pytest.skip(f"gitignored seed probes absent for {slug}: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(doc, _load_schema())

    assert doc["course_slug"] == slug
    probes = doc["probes"]
    # Authoring procedure: 9 probes, 3 per category.
    assert len(probes) == 9
    by_cat = {}
    for p in probes:
        by_cat.setdefault(p["category"], 0)
        by_cat[p["category"]] += 1
        # Every probe is dry-run-verified, never assumed.
        assert p["dry_run"]["verified"] is True
        assert p["dry_run"]["top_passage_answers"] is False
    assert by_cat == {
        "off_topic": 3,
        "adjacent_domain": 3,
        "out_of_scope_detail": 3,
    }
    # probe_ids unique.
    ids = [p["probe_id"] for p in probes]
    assert len(ids) == len(set(ids))


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


def test_seed_probes_drive_an_end_to_end_overlap_calibration():
    """Real-corpus distributions overlap (R4): a calibration over the recorded
    probe top-scores + a plausible answerable arm honestly returns None when the
    arms interleave — proving the honest-fallback path on real-shaped data.

    Discovers the first available seed probe course dynamically (no hardcoded
    slug); skips when no real course data is present on this checkout."""
    if not _PROBE_COURSES:
        pytest.skip("gitignored seed probes absent")
    slug = _PROBE_COURSES[0]
    path = _real_probe_path(slug)
    doc = json.loads(path.read_text(encoding="utf-8"))
    neg_scores = [p["dry_run"]["top_score"] for p in doc["probes"]]
    # Answerable arm that overlaps the negatives' range (real lexical behaviour).
    pos_scores = list(neg_scores)  # worst case: identical → must fall back
    result = calibrate_from_distributions(
        course_slug=slug,
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


def test_hybrid_rrf_is_embedder_keyed_and_unpinned():
    """hybrid-rrf is keyed by embedding model (its fused score depends on the
    semantic arm) but ships UNPINNED — the calibration distributions overlapped
    on every local archive, so it falls through to the v0-uncalibrated default
    for BOTH a known and an unknown embedder until a clean threshold exists."""
    # No hybrid-rrf pin exists yet for any embedder.
    assert not any(eng == "hybrid-rrf" for (eng, _model) in PINNED_POLICIES)
    # The current production embedder resolves to the default (not a stale pin).
    p_known = resolve_policy("hybrid-rrf", _BGE_LARGE)
    assert p_known is DEFAULT_POLICIES["hybrid-rrf"]
    assert p_known.policy_version == POLICY_VERSION_UNCALIBRATED
    # An unknown embedder is equally unpinned.
    p_unknown = resolve_policy("hybrid-rrf", "some-other/embedder-v9")
    assert p_unknown is DEFAULT_POLICIES["hybrid-rrf"]
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
    policy = resolve_policy("hybrid-rrf")  # no pin yet
    assert policy is DEFAULT_POLICIES["hybrid-rrf"]
    assert policy.policy_version == POLICY_VERSION_UNCALIBRATED
    assert resolve_policy("nonsense") is DEFAULT_POLICIES["lexical"]


# --- Tier 2: the pins reproduce the committed calibration verdicts --------- #


def _calibration_path(slug: str) -> Path:
    return _LIBV2_COURSES / slug / "retrieval_eval" / "refusal_calibration.json"


def _discover_pinned_calibrations():
    """Find committed refusal_calibration.json artifacts whose (engine,
    embedding_model_id) provenance matches a PINNED_POLICIES key.

    Keyed off artifact fields (no hardcoded slug). Returns
    ``[(slug, key), ...]`` for each discovered artifact that maps to a pin AND
    carries a clean (non-null) recommendation. Several courses may map to the
    SAME key (a single pin calibrated across a corpus) — each is returned so the
    pin is checked safe on every one.
    """
    out = []
    for slug in _discover_calibration_courses():
        try:
            doc = json.loads(
                _calibration_path(slug).read_text(encoding="utf-8")
            )
        except Exception:  # pragma: no cover - skip malformed
            continue
        key = (doc.get("engine"), doc.get("embedding_model_id"))
        if key in PINNED_POLICIES and doc.get("recommended") is not None:
            out.append((slug, key))
    return out


_PINNED_CALIBRATIONS = _discover_pinned_calibrations()


def _effective_sweep_row(doc: dict, threshold: float) -> dict:
    """The sweep row the policy actually applies at ``threshold`` on this course.

    The verdict is ``top_score >= threshold`` (monotone in the threshold), so a
    threshold falling between two observed scores behaves like the largest
    observed score that is ``<= threshold``. A cross-course pin (the corpus-wide
    minimum of the per-course recommendations) is an exact sweep threshold only
    on the course it was drawn from; on the others it lands between observed
    scores, so we read the effective row rather than requiring an exact match.
    """
    candidates = [
        r for r in doc["sweep"] if r["threshold"] <= threshold + 1e-9
    ]
    assert candidates, "sweep always includes threshold 0.0"
    return max(candidates, key=lambda r: r["threshold"])


@pytest.mark.skipif(
    not _PINNED_CALIBRATIONS,
    reason="no committed refusal_calibration.json mapping to a pinned policy",
)
@pytest.mark.parametrize(
    "slug,key",
    _PINNED_CALIBRATIONS,
    ids=[f"{s}:{k[0]}" for s, k in _PINNED_CALIBRATIONS],
)
def test_pinned_threshold_is_corpus_safe_on_committed_calibration(slug, key):
    """Tier-2: every committed refusal_calibration.json mapping to a pin must
    confirm the pinned min_top_score is SAFE on that course — the effective
    sweep row at the pin still clears the pin rule (refusal_precision >= 0.90
    AND answer_recall >= 0.95) with zero false refusals on answerable queries.

    A pin shipped in the binary is corpus-wide, not per-course: a single pin can
    map to several committed artifacts. The cross-course derivation (the MINIMUM
    of the per-course recommendations) guarantees answer_recall is preserved on
    every calibrated course — this test enforces that guarantee against the
    on-disk artifacts rather than trusting the derivation comment.

    Course data dirs are gitignored user data; the (slug, key) pairs are
    discovered dynamically by the artifact's own (engine, embedding_model_id)
    provenance, never by a hardcoded slug."""
    path = _calibration_path(slug)
    if not path.exists():
        pytest.skip(f"gitignored calibration absent for {slug}: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["recommended"] is not None, (
        f"{slug} calibration has no clean recommendation"
    )

    pin = PINNED_POLICIES[key]
    # Engine + embedder match the artifact provenance.
    assert doc["engine"] == pin.engine
    assert doc["embedding_model_id"] == pin.embedding_model_id

    # The pin must never EXCEED a course's own clean recommendation — exceeding
    # it would refuse answerable queries the course's own sweep deemed answerable
    # (the corpus-wide pin is the MIN across courses, so this always holds).
    assert pin.min_top_score <= doc["recommended"]["min_top_score"] + 1e-6

    # The effective sweep row at the pin clears the rule on this course.
    row = _effective_sweep_row(doc, pin.min_top_score)
    assert row["refusal_precision"] >= 0.90
    assert row["answer_recall"] >= 0.95
    assert row["false_refusals_on_positives"] == 0


@pytest.mark.skipif(
    not _PINNED_CALIBRATIONS,
    reason="no committed refusal_calibration.json mapping to a pinned policy",
)
def test_pin_value_anchored_to_a_committed_recommendation():
    """Tier-2: each pinned key's numeric min_top_score must equal at least one
    committed artifact's clean recommendation (no drift / invented number).

    The corpus-wide pin is the MINIMUM of the per-course recommendations, so it
    coincides exactly with one course's committed recommendation; the others it
    only has to be SAFE on (the per-course test above). This anchors every
    shipped pin value to an on-disk artifact without requiring a 1:1 mapping."""
    recs_by_key: dict = {}
    for slug, key in _PINNED_CALIBRATIONS:
        doc = json.loads(_calibration_path(slug).read_text(encoding="utf-8"))
        recs_by_key.setdefault(key, []).append(
            doc["recommended"]["min_top_score"]
        )
    for key, recs in recs_by_key.items():
        pin = PINNED_POLICIES[key]
        assert any(
            pin.min_top_score == pytest.approx(r, abs=1e-6) for r in recs
        ), (
            f"pin {key} min_top_score={pin.min_top_score} matches no committed "
            f"recommendation in {recs}"
        )
        # Corpus-wide pin is the minimum across courses.
        assert pin.min_top_score == pytest.approx(min(recs), abs=1e-6)
