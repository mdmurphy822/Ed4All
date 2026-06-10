"""Phase IA WS3 (D4) — calibrated refusal policy for the grounded-answer path.

The refusal layer is the *pre-LLM* honesty gate: when retrieval confidence is
too low to ground an answer, the grounded-answer pipeline (wave B) refuses
BEFORE ever calling the local model — cheap, deterministic, and verifiable
("zero client calls on refusal"). This module owns:

* :class:`RefusalPolicy` — the engine- and (for semantic engines)
  embedding-model-pinned thresholds. Cosine scales are NOT comparable across
  embedding models, so a semantic policy carries the model id it was pinned
  against; the pipeline warns (never silently trusts) a mismatched live index.
* :func:`evaluate_confidence` / :func:`should_refuse` — the verdict over the
  retrieved passages: a top-score floor AND a count-above-floor signal.
* :data:`DEFAULT_POLICIES` — **v0-uncalibrated, permissive** defaults. Lexical
  BM25 / blended scores are corpus-dependent (stopword overlap inflates
  off-topic scores), so we ship permissive and calibrate later. An honest weak
  signal beats a fabricated strong one.
* The ``python -m lib.retrieval.refusal --calibrate`` entry point: runs the
  per-course gold set (answerable positives) and refusal probes (verified
  unanswerable negatives) through the live retriever, measures the two score
  distributions, sweeps thresholds, and emits
  ``retrieval_eval/refusal_calibration.json``. When the distributions overlap
  so badly that no threshold meets the precision/recall rule, it HONESTLY
  reports ``recommended: null`` and leaves the policy ``v0-uncalibrated``
  (the plan's pre-declared honest fallback).

No LLM, no network, no decision capture on the policy path — the verdict is
pure arithmetic over retrieval scores. The calibration ``__main__`` does touch
the (local, file-backed) retriever and reads the WS1 gold set, but performs no
generation and no cloud call.

WS4 + the wave-B pipeline consume :class:`RefusalVerdict`; they emit the
``reason_code`` as data and own the learner-facing display copy.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "RefusalPolicy",
    "ConfidenceVerdict",
    "RefusalVerdict",
    "DEFAULT_POLICIES",
    "PINNED_POLICIES",
    "POLICY_VERSION_UNCALIBRATED",
    "POLICY_VERSION_PINNED",
    "REASON_LOW_CONFIDENCE",
    "REASON_NOT_IN_COURSE_MODEL",
    "evaluate_confidence",
    "should_refuse",
    "default_policy_for",
    "resolve_policy",
    "calibrate",
    "CalibrationResult",
]

# --------------------------------------------------------------------------- #
# Reason codes (WS4 owns the display copy; we emit codes only — § D4.2 / § 3.3)
# --------------------------------------------------------------------------- #

#: Pre-LLM refusal: retrieval confidence below the policy floor.
REASON_LOW_CONFIDENCE = "low_confidence"
#: Model-side refusal: the composed envelope set ``not_in_course: true``.
#: (Emitted by the wave-B pipeline, not this module — exported so the enum of
#: reason codes lives in one place.)
REASON_NOT_IN_COURSE_MODEL = "not_in_course_model"

POLICY_VERSION_UNCALIBRATED = "ws3.v0-uncalibrated"
#: First measured pin (D4.2 measure-then-pin). Bumped on every re-pin so a
#: consumer can tell a measured threshold from the permissive v0 default.
POLICY_VERSION_PINNED = "ws3.v1-pinned-2026-06-10"


# --------------------------------------------------------------------------- #
# Policy + verdict dataclasses (frozen — § D4)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RefusalPolicy:
    """Thresholds the refusal verdict is computed against.

    ``confident`` (the inverse of refusal) is::

        top_score >= min_top_score
        AND  count(passages with score >= score_floor) >= min_passages_above_floor

    Semantic policies are pinned PER EMBEDDING MODEL: a cosine threshold tuned
    for one embedder is meaningless for another, so ``embedding_model_id``
    records the model the pin was measured against. ``evaluate_confidence`` does
    not enforce the match (the pipeline warns), but the calibration artifact and
    the policy carry it so a re-pin after a model swap is one constant edit.
    """

    engine: str
    min_top_score: float
    score_floor: float
    min_passages_above_floor: int
    policy_version: str
    embedding_model_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "min_top_score": self.min_top_score,
            "score_floor": self.score_floor,
            "min_passages_above_floor": self.min_passages_above_floor,
            "policy_version": self.policy_version,
            "embedding_model_id": self.embedding_model_id,
        }


@dataclass(frozen=True)
class ConfidenceVerdict:
    """The arithmetic outcome of applying a policy to retrieved passages."""

    confident: bool
    signals: Dict[str, float]
    policy: RefusalPolicy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confident": self.confident,
            "signals": dict(self.signals),
            "policy_version": self.policy.policy_version,
            "engine": self.policy.engine,
            "embedding_model_id": self.policy.embedding_model_id,
        }


@dataclass(frozen=True)
class RefusalVerdict:
    """The refusal-shaped projection of a confidence verdict.

    ``refuse`` is the inverse of ``ConfidenceVerdict.confident``. When refusing,
    ``reason_code`` is :data:`REASON_LOW_CONFIDENCE` (the only pre-LLM refusal
    reason); the model-side ``not_in_course`` refusal carries
    :data:`REASON_NOT_IN_COURSE_MODEL` and is set by the pipeline, not here.
    WS4 renders the learner-facing copy from the code — WS3 never hardcodes it.
    """

    refuse: bool
    reason_code: Optional[str]
    signals: Dict[str, float]
    policy_version: str
    engine: str
    embedding_model_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "refuse": self.refuse,
            "reason_code": self.reason_code,
            "signals": dict(self.signals),
            "policy_version": self.policy_version,
            "engine": self.engine,
            "embedding_model_id": self.embedding_model_id,
        }


# --------------------------------------------------------------------------- #
# Default (v0-uncalibrated, permissive) policies — § D4
# --------------------------------------------------------------------------- #

# Permissive on purpose: lexical blended scores are corpus-dependent (a single
# common token can score > 1.0; stopword overlap pushes genuinely off-topic
# probes to several points), so a strict lexical floor would fabricate
# confidence it cannot defend. The semantic floor is a conservative cosine guess
# pending the per-model calibration sweep. Both carry POLICY_VERSION_UNCALIBRATED
# so any consumer can see at a glance these are not measured pins.
DEFAULT_POLICIES: Dict[str, RefusalPolicy] = {
    "semantic": RefusalPolicy(
        engine="semantic",
        min_top_score=0.30,
        score_floor=0.25,
        min_passages_above_floor=1,
        policy_version=POLICY_VERSION_UNCALIBRATED,
        embedding_model_id=None,
    ),
    "lexical": RefusalPolicy(
        engine="lexical",
        # LazyBM25 DEFAULT_MIN_RELEVANCE=0.5 already prunes the floor; the
        # blended-score scale is corpus-dependent, so we keep a low top-score
        # floor (any retained passage clears it) — permissive pre-calibration.
        min_top_score=1.0,
        score_floor=0.5,
        min_passages_above_floor=1,
        policy_version=POLICY_VERSION_UNCALIBRATED,
        embedding_model_id=None,
    ),
}
# hybrid-rrf rides the lexical floor until WS2 lands fused-score calibration —
# RRF scores share neither the cosine nor the BM25 scale, so the permissive
# lexical-shaped policy is the honest pre-calibration default.
DEFAULT_POLICIES["hybrid-rrf"] = RefusalPolicy(
    engine="hybrid-rrf",
    min_top_score=DEFAULT_POLICIES["lexical"].min_top_score,
    score_floor=DEFAULT_POLICIES["lexical"].score_floor,
    min_passages_above_floor=1,
    policy_version=POLICY_VERSION_UNCALIBRATED,
    embedding_model_id=None,
)


def default_policy_for(engine: str) -> RefusalPolicy:
    """Return the v0-uncalibrated default policy for ``engine``.

    Unknown engines fall back to the lexical (most permissive) default rather
    than raising — refusal under-firing is the safe failure direction.
    """
    return DEFAULT_POLICIES.get(engine, DEFAULT_POLICIES["lexical"])


# --------------------------------------------------------------------------- #
# Pinned (measured) policies — § D4.2 measure-then-pin
# --------------------------------------------------------------------------- #

# Each pin is the clean ``recommended.min_top_score`` from a course's
# refusal_calibration.json (refusal_precision >= 0.90 AND answer_recall >= 0.95
# — the LARGEST qualifying threshold). Keyed by (engine, embedding_model_id):
#
#   * SEMANTIC pins bind to a SPECIFIC embedding model. Cosine scales are not
#     comparable across embedders, so a pin tuned for one model is meaningless
#     for another — an unknown (semantic, <model>) pair MUST fall back to the
#     v0-uncalibrated default, never silently reuse a stale cosine threshold.
#   * LEXICAL / hybrid-rrf pins are model-agnostic (no embedder), keyed
#     (engine, None). BM25 blended scores carry no model id.
#
# score_floor + min_passages_above_floor are inherited from the matching
# DEFAULT_POLICIES entry the calibration sweep was measured against (the sweep
# only re-pins min_top_score; the count signal + floor are unchanged).
#
# Measured pins (D4.2):
#   (semantic, sentence-transformers/all-MiniLM-L6-v2)  sample-rag-101 2026-06-09
#       min_top_score=0.369377  refusal_precision=1.0  answer_recall=1.0
#   (lexical, None)  sample-course-a 2026-06-10
#       min_top_score=4.455352  refusal_precision=1.0  answer_recall=1.0
PINNED_POLICIES: Dict[tuple, RefusalPolicy] = {
    ("semantic", "sentence-transformers/all-MiniLM-L6-v2"): RefusalPolicy(
        engine="semantic",
        min_top_score=0.369377,
        score_floor=DEFAULT_POLICIES["semantic"].score_floor,
        min_passages_above_floor=DEFAULT_POLICIES[
            "semantic"
        ].min_passages_above_floor,
        policy_version=POLICY_VERSION_PINNED,
        embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
    ),
    ("lexical", None): RefusalPolicy(
        engine="lexical",
        min_top_score=4.455352,
        score_floor=DEFAULT_POLICIES["lexical"].score_floor,
        min_passages_above_floor=DEFAULT_POLICIES[
            "lexical"
        ].min_passages_above_floor,
        policy_version=POLICY_VERSION_PINNED,
        embedding_model_id=None,
    ),
}


def resolve_policy(
    engine: str, embedding_model_id: Optional[str] = None
) -> RefusalPolicy:
    """Select the pinned (measured) policy for ``(engine, embedding_model_id)``,
    else the v0-uncalibrated default.

    Lookup rules:

    * SEMANTIC engines are keyed by ``(engine, embedding_model_id)`` — the
      cosine threshold is only valid for the embedder it was measured against.
      An unknown model id (or ``None`` for a semantic engine) returns the
      v0-uncalibrated default; we NEVER reuse another model's pinned cosine.
    * LEXICAL / hybrid-rrf pins are keyed ``(engine, None)`` (model-agnostic);
      any incidental ``embedding_model_id`` argument is ignored for these.

    Falling back to the (permissive) default is the safe direction: refusal
    under-firing is preferable to fabricated confidence on a stale threshold.
    """
    # Lexical / hybrid-rrf carry no embedder — always look up under None.
    key_model = None if engine != "semantic" else embedding_model_id
    pinned = PINNED_POLICIES.get((engine, key_model))
    if pinned is not None:
        return pinned
    return default_policy_for(engine)


# --------------------------------------------------------------------------- #
# Verdict logic — § D4
# --------------------------------------------------------------------------- #


def _score_of(passage: Any) -> float:
    """Read ``.score`` (RetrievedPassage / RetrievalResult) or ``["score"]``."""
    score = getattr(passage, "score", None)
    if score is None and isinstance(passage, dict):
        score = passage.get("score")
    try:
        return float(score) if score is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _compute_signals(
    passages: Sequence[Any], policy: RefusalPolicy
) -> Dict[str, float]:
    scores = [_score_of(p) for p in passages]
    top_score = max(scores) if scores else 0.0
    n_above_floor = sum(1 for s in scores if s >= policy.score_floor)
    top3 = sorted(scores, reverse=True)[:3]
    mean_top3 = (sum(top3) / len(top3)) if top3 else 0.0
    return {
        "top_score": float(top_score),
        "n_above_floor": float(n_above_floor),
        "mean_top3": float(mean_top3),
        "n_passages": float(len(scores)),
    }


def evaluate_confidence(
    passages: Sequence[Any], policy: RefusalPolicy
) -> ConfidenceVerdict:
    """Apply ``policy`` to ``passages`` → :class:`ConfidenceVerdict`.

    ``confident = (top_score >= min_top_score) AND
    (n_above_floor >= min_passages_above_floor)``. Empty passages are never
    confident. ``passages`` may be :class:`RetrievedPassage`,
    ``RetrievalResult``, or score-bearing dicts (duck-typed on ``.score`` /
    ``["score"]``).
    """
    signals = _compute_signals(passages, policy)
    confident = bool(passages) and (
        signals["top_score"] >= policy.min_top_score
        and signals["n_above_floor"] >= policy.min_passages_above_floor
    )
    return ConfidenceVerdict(confident=confident, signals=signals, policy=policy)


def should_refuse(
    retrieval_results: Sequence[Any], policy: RefusalPolicy
) -> RefusalVerdict:
    """The refusal-shaped entry point the wave-B pipeline + WS4 consume.

    Returns :class:`RefusalVerdict`: ``refuse=True`` with
    ``reason_code=REASON_LOW_CONFIDENCE`` when confidence fails, else
    ``refuse=False`` with ``reason_code=None``. Display copy is WS4's.
    """
    verdict = evaluate_confidence(retrieval_results, policy)
    refuse = not verdict.confident
    return RefusalVerdict(
        refuse=refuse,
        reason_code=REASON_LOW_CONFIDENCE if refuse else None,
        signals=verdict.signals,
        policy_version=policy.policy_version,
        engine=policy.engine,
        embedding_model_id=policy.embedding_model_id,
    )


# --------------------------------------------------------------------------- #
# Calibration (measure-then-pin) — § D4.2
# --------------------------------------------------------------------------- #

CALIBRATION_SCHEMA_VERSION = "1.0"

#: The pin rule: choose the LARGEST min_top_score whose refusal_precision and
#: answer_recall both clear these floors. Pre-declared so the artifact is
#: reproducible and the "no clean threshold" fallback is principled.
_REFUSAL_PRECISION_TARGET = 0.90
_ANSWER_RECALL_TARGET = 0.95


@dataclass(frozen=True)
class _ScoreDist:
    """Top-score distribution of one arm (positives or negatives)."""

    n: int
    scores: List[float]

    def percentiles(self) -> Dict[str, Optional[float]]:
        if not self.scores:
            return {"p05": None, "p25": None, "p50": None, "p75": None, "p95": None}
        ordered = sorted(self.scores)

        def pct(q: float) -> float:
            if len(ordered) == 1:
                return ordered[0]
            # nearest-rank, deterministic
            idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
            return ordered[idx]

        return {
            "p05": pct(0.05),
            "p25": pct(0.25),
            "p50": pct(0.50),
            "p75": pct(0.75),
            "p95": pct(0.95),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "top_score": self.percentiles()}


@dataclass(frozen=True)
class CalibrationResult:
    """In-memory shape of ``refusal_calibration.json`` (also the report dict)."""

    schema_version: str
    course_slug: str
    engine: str
    embedding_model_id: Optional[str]
    positives: Dict[str, Any]
    negatives: Dict[str, Any]
    sweep: List[Dict[str, float]]
    recommended: Optional[Dict[str, Any]]
    fallback_policy_version: str
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "course_slug": self.course_slug,
            "engine": self.engine,
            "embedding_model_id": self.embedding_model_id,
            "positives": self.positives,
            "negatives": self.negatives,
            "sweep": self.sweep,
            "recommended": self.recommended,
            "fallback_policy_version": self.fallback_policy_version,
            "generated_at": self.generated_at,
        }


def _sweep_thresholds(
    positives: Sequence[float],
    negatives: Sequence[float],
    floor: float,
    min_passages_above_floor: int,
    pos_n_above: Sequence[int],
    neg_n_above: Sequence[int],
) -> List[Dict[str, float]]:
    """Sweep candidate ``min_top_score`` values over the observed score range.

    For each candidate threshold ``t`` an item is judged *confident* (answered)
    when ``top_score >= t AND n_above_floor >= min_passages_above_floor``.

    * ``answer_recall``   = answered positives / positives
      (a positive is answerable, so answering it is correct).
    * ``refusal_recall``  = refused negatives / negatives
      (a negative is unanswerable, so refusing it is correct).
    * ``refusal_precision`` = refused negatives / total refused
      (of everything we refused, how much was correctly refused).

    Candidate thresholds are every observed score (positives + negatives), so
    the sweep is deterministic and corpus-anchored — no arbitrary grid.
    """
    candidates = sorted(set([0.0] + list(positives) + list(negatives)))
    n_pos = len(positives)
    n_neg = len(negatives)
    sweep: List[Dict[str, float]] = []
    for t in candidates:
        ans_pos = sum(
            1
            for s, na in zip(positives, pos_n_above)
            if s >= t and na >= min_passages_above_floor
        )
        ans_neg = sum(
            1
            for s, na in zip(negatives, neg_n_above)
            if s >= t and na >= min_passages_above_floor
        )
        refused_pos = n_pos - ans_pos
        refused_neg = n_neg - ans_neg
        total_refused = refused_pos + refused_neg
        answer_recall = (ans_pos / n_pos) if n_pos else 0.0
        refusal_recall = (refused_neg / n_neg) if n_neg else 0.0
        refusal_precision = (refused_neg / total_refused) if total_refused else 1.0
        sweep.append(
            {
                "threshold": round(float(t), 6),
                "answer_recall": round(answer_recall, 6),
                "refusal_recall": round(refusal_recall, 6),
                "refusal_precision": round(refusal_precision, 6),
                "false_refusals_on_positives": int(refused_pos),
            }
        )
    return sweep


def _recommend(sweep: Sequence[Dict[str, float]]) -> Optional[Dict[str, Any]]:
    """Pick the LARGEST threshold meeting both targets, or None (honest fallback).

    Larger thresholds refuse more aggressively — we want the most-aggressive
    refusal that still keeps ``answer_recall >= target`` and
    ``refusal_precision >= target``. None when no candidate qualifies (the
    distributions overlap): the caller keeps v0-uncalibrated.
    """
    qualifying = [
        row
        for row in sweep
        if row["refusal_precision"] >= _REFUSAL_PRECISION_TARGET
        and row["answer_recall"] >= _ANSWER_RECALL_TARGET
        and row["refusal_recall"] > 0.0  # a threshold that refuses nothing is useless
    ]
    if not qualifying:
        return None
    best = max(qualifying, key=lambda r: r["threshold"])
    return {
        "min_top_score": best["threshold"],
        "answer_recall": best["answer_recall"],
        "refusal_recall": best["refusal_recall"],
        "refusal_precision": best["refusal_precision"],
        "rule": (
            f"max threshold with refusal_precision >= {_REFUSAL_PRECISION_TARGET} "
            f"and answer_recall >= {_ANSWER_RECALL_TARGET}"
        ),
    }


def calibrate_from_distributions(
    *,
    course_slug: str,
    engine: str,
    positives: Sequence[float],
    negatives: Sequence[float],
    score_floor: float,
    min_passages_above_floor: int,
    pos_n_above: Optional[Sequence[int]] = None,
    neg_n_above: Optional[Sequence[int]] = None,
    embedding_model_id: Optional[str] = None,
    generated_at: Optional[str] = None,
) -> CalibrationResult:
    """Pure calibration math over pre-measured top-score distributions.

    Separated from the retriever I/O so it is unit-testable on constructed
    distributions (separable → a threshold is proposed; overlapping → honest
    ``recommended=None`` fallback). ``*_n_above`` default to "every item clears
    the count signal" when omitted (the common case where the count floor is 1
    and every answerable/probe retrieval returns >= 1 passage).
    """
    positives = [float(s) for s in positives]
    negatives = [float(s) for s in negatives]
    if pos_n_above is None:
        pos_n_above = [min_passages_above_floor] * len(positives)
    if neg_n_above is None:
        # negatives that returned zero passages have n_above=0; but when the
        # caller does not supply counts we conservatively assume they cleared
        # the floor (worst case for refusal — makes the threshold work harder).
        neg_n_above = [min_passages_above_floor] * len(negatives)

    sweep = _sweep_thresholds(
        positives,
        negatives,
        score_floor,
        min_passages_above_floor,
        list(pos_n_above),
        list(neg_n_above),
    )
    recommended = _recommend(sweep)
    return CalibrationResult(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        course_slug=course_slug,
        engine=engine,
        embedding_model_id=embedding_model_id,
        positives=_ScoreDist(len(positives), positives).to_dict(),
        negatives=_ScoreDist(len(negatives), negatives).to_dict(),
        sweep=sweep,
        recommended=recommended,
        fallback_policy_version=POLICY_VERSION_UNCALIBRATED,
        generated_at=generated_at or _utcnow_iso(),
    )


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Live-retriever calibration driver + __main__ — § D4.2
# --------------------------------------------------------------------------- #

RETRIEVAL_EVAL_SUBDIR = "retrieval_eval"
GOLD_SET_FILENAME = "gold_set.json"
REFUSAL_PROBES_FILENAME = "refusal_probes.json"
CALIBRATION_FILENAME = "refusal_calibration.json"


def _libv2_root(repo_root: Optional[Path] = None) -> Path:
    """Resolve the LibV2 root the retriever expects as its ``repo_root``."""
    import os

    env = os.environ.get("ED4ALL_LIBV2_ROOT")
    if env:
        return Path(env)
    if repo_root is not None:
        cand = Path(repo_root) / "LibV2"
        if cand.exists():
            return cand
        return Path(repo_root)
    here = Path(__file__).resolve()
    # lib/retrieval/refusal.py → repo root is parents[2]
    return here.parents[2] / "LibV2"


def _retrieve_top(
    libv2_root: Path,
    course_slug: str,
    query: str,
    *,
    engine: str,
    limit: int,
    score_floor: float,
):
    """Read-only retrieval dry-run → (top_score, n_above_floor, n_results, top_id)."""
    from LibV2.tools.libv2.retriever import retrieve_chunks  # lazy, file-backed

    results = retrieve_chunks(
        libv2_root, query, course_slug=course_slug, engine=engine, limit=limit
    )
    scores = [float(getattr(r, "score", 0.0)) for r in results]
    top_score = max(scores) if scores else 0.0
    n_above = sum(1 for s in scores if s >= score_floor)
    top_id = getattr(results[0], "chunk_id", "") if results else ""
    return top_score, n_above, len(results), top_id


def calibrate(
    course_slug: str,
    *,
    engine: str = "lexical",
    repo_root: Optional[Path] = None,
    limit: int = 8,
    policy: Optional[RefusalPolicy] = None,
    write: bool = True,
    output_path: Optional[Path] = None,
) -> CalibrationResult:
    """Run the gold set + refusal probes through the live retriever and calibrate.

    Positives = gold-set questions (answerable). Negatives = refusal probes
    (verified unanswerable). Measures each arm's rank-1 score + count-above-floor,
    sweeps thresholds, and writes ``retrieval_eval/refusal_calibration.json``.
    Honest fallback: ``recommended=None`` when the arms overlap.
    """
    policy = policy or default_policy_for(engine)
    libv2_root = _libv2_root(repo_root)
    course_dir = libv2_root / "courses" / course_slug
    embedding_model_id = (
        _embedding_model_id_from_manifest(course_dir) if engine != "lexical" else None
    )

    gold = _load_json(course_dir / RETRIEVAL_EVAL_SUBDIR / GOLD_SET_FILENAME)
    probes = _load_json(course_dir / RETRIEVAL_EVAL_SUBDIR / REFUSAL_PROBES_FILENAME)

    pos_scores: List[float] = []
    pos_n_above: List[int] = []
    for q in (gold or {}).get("questions", []):
        ts, na, _, _ = _retrieve_top(
            libv2_root,
            course_slug,
            q["question_text"],
            engine=engine,
            limit=limit,
            score_floor=policy.score_floor,
        )
        pos_scores.append(ts)
        pos_n_above.append(na)

    neg_scores: List[float] = []
    neg_n_above: List[int] = []
    for p in (probes or {}).get("probes", []):
        ts, na, _, _ = _retrieve_top(
            libv2_root,
            course_slug,
            p["question_text"],
            engine=engine,
            limit=limit,
            score_floor=policy.score_floor,
        )
        neg_scores.append(ts)
        neg_n_above.append(na)

    result = calibrate_from_distributions(
        course_slug=course_slug,
        engine=engine,
        positives=pos_scores,
        negatives=neg_scores,
        score_floor=policy.score_floor,
        min_passages_above_floor=policy.min_passages_above_floor,
        pos_n_above=pos_n_above,
        neg_n_above=neg_n_above,
        embedding_model_id=embedding_model_id,
    )

    if write:
        out = output_path or (course_dir / RETRIEVAL_EVAL_SUBDIR / CALIBRATION_FILENAME)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _embedding_model_id_from_manifest(course_dir: Path) -> Optional[str]:
    """Best-effort read of the live vector-index model id (semantic engines)."""
    for rel in ("vector_index/manifest.json", "vector_index.manifest.json"):
        doc = _load_json(course_dir / rel)
        if isinstance(doc, dict):
            model = doc.get("embedding_model_id") or doc.get("model_id")
            if model:
                return str(model)
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lib.retrieval.refusal",
        description=(
            "Calibrate the grounded-answer refusal threshold for one course by "
            "measuring answerable (gold-set) vs unanswerable (refusal-probe) "
            "retrieval-score distributions. Honest fallback: keeps "
            "v0-uncalibrated when the distributions overlap."
        ),
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Run the calibration sweep."
    )
    parser.add_argument("--course", required=False, help="Course slug.")
    parser.add_argument(
        "--engine",
        default="lexical",
        choices=["lexical", "semantic", "hybrid-rrf"],
        help="Retrieval engine (default: lexical).",
    )
    parser.add_argument(
        "--limit", type=int, default=8, help="top-k retrieval limit (default: 8)."
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo or LibV2 root override (else ED4ALL_LIBV2_ROOT / autodetect).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the calibration JSON to stdout without writing the artifact.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if not args.calibrate:
        print("nothing to do; pass --calibrate --course <slug>", flush=True)
        return 2
    if not args.course:
        print("--calibrate requires --course <slug>", flush=True)
        return 2
    result = calibrate(
        args.course,
        engine=args.engine,
        repo_root=Path(args.repo_root) if args.repo_root else None,
        limit=args.limit,
        write=not args.no_write,
    )
    rec = result.recommended
    if rec is None:
        print(
            f"[{args.course}/{args.engine}] distributions overlap — keeping "
            f"{POLICY_VERSION_UNCALIBRATED} (no threshold met the rule).",
            flush=True,
        )
    else:
        print(
            f"[{args.course}/{args.engine}] recommended min_top_score="
            f"{rec['min_top_score']} (refusal_precision={rec['refusal_precision']}, "
            f"answer_recall={rec['answer_recall']}).",
            flush=True,
        )
    if args.no_write:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())
