"""Attribution-threshold calibration (retrieval-answer-eval-set §4 / P4).

Calibrates the citation-attribution support ``min_overlap`` knob
(:data:`lib.retrieval.citation_attribution.DEFAULT_MIN_OVERLAP`) against a
course's gold set. Replays each gold question through the LIVE retriever
(retrieval only — NO LLM, NO answer generation), scores every retrieved passage
with the SAME attribution machinery the answer path uses
(:func:`citation_attribution.attribute_citations`), and joins each passage's
best support score against gold relevance (a passage is a gold POSITIVE iff its
chunk_id is in the question's ``relevant_passages`` set).

**Claim proxy (documented choice).** Attribution support scoring needs an
*answer text* to split into claims. There is no answer here (we deliberately do
not call the LLM — calibration must be cheap, deterministic, and replayable).
The gold question's ``expected_key_points`` (v1.1) are the operator-reviewed
statements the answer *should* make, so we use them as the claim proxy: the
synthetic "answer" is the key points joined into sentences. A question with no
key points falls back to its ``answer_note`` then its ``question_text`` so every
gold question still contributes evidence (recorded per-question as
``claim_source``). This biases the measured shingle scores toward the
operator's wording rather than a model's phrasing — calibration on this stream
is therefore a CEILING on achievable support overlap; the second evidence stream
(live captures) grounds it against real model phrasings.

**Second evidence stream: live citation-prune captures.** The answer path emits
``grounded_answer_citation_prune`` decision captures whose rationale carries
``per_citation=[chunk@shingle/coverage*N(cited|uncited)]`` for the REAL model
answers served in production. :func:`ingest_prune_captures` parses those into a
``cited`` / ``uncited`` shingle-score distribution — the real-phrasing
complement to the gold-key-point ceiling.

**Recommendation.** Over candidate thresholds (every observed positive/negative
score) we compute precision / recall / F1 treating "best support score >=
threshold" as the relevance predictor. We report BOTH:

  * ``f1_max`` — the threshold maximizing F1 (balanced).
  * ``precision_floored`` — the LARGEST-recall threshold whose precision clears
    a floor (default 0.80, mirroring the attribution module's precision-first
    posture: a spurious support attribution is a fabricated-looking source,
    costlier than a missed weak supporter).

The headline ``recommended`` is ``precision_floored`` when one qualifies
(precision-first, consistent with ``citation_attribution``'s design comments),
else ``f1_max`` (so an always-low-precision corpus still gets a balanced
recommendation rather than nothing). The artifact records the rule used.

Also flags whether :data:`citation_attribution.ADD_MIN_SHINGLE` warrants its own
env knob: if the gold-positive shingle distribution's median sits below the
current constant (additions would essentially never fire) or far above it
(additions would over-fire), the report sets ``add_min_shingle_knob_warranted``
with the evidence.

Writes ``retrieval_eval/attribution_calibration_<ts>.json``. Decision capture is
NOT needed: there is no LLM call on this path (retrieval + pure arithmetic),
mirroring the refusal-calibration precedent ("no decision capture on the policy
path" — :mod:`lib.retrieval.refusal`). Verified against the project rule: the
root CLAUDE.md contract scopes mandatory ``DecisionCapture`` to *LLM call
sites*; this module makes none.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.retrieval.citation_attribution import (
    ADD_MIN_SHINGLE,
    ATTRIBUTION_SHINGLE_SIZE,
    DEFAULT_MIN_OVERLAP,
    _normalize,
    _shingle_support,
)

__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "DEFAULT_PRECISION_FLOOR",
    "ScoreSample",
    "ThresholdRow",
    "AttributionCalibrationResult",
    "calibrate_from_samples",
    "ingest_prune_captures",
    "claim_proxy_for_question",
    "calibrate",
]

CALIBRATION_SCHEMA_VERSION = "1.0"
RETRIEVAL_EVAL_SUBDIR = "retrieval_eval"
GOLD_SET_FILENAME = "gold_set.json"
CALIBRATION_FILENAME_PREFIX = "attribution_calibration_"

#: Precision floor for the ``precision_floored`` recommendation. Mirrors the
#: attribution module's precision-first posture (a spurious support attribution
#: is costlier than a missed weak supporter).
DEFAULT_PRECISION_FLOOR = 0.80


# --------------------------------------------------------------------------- #
# Claim proxy
# --------------------------------------------------------------------------- #


def claim_proxy_for_question(question: Dict[str, Any]) -> Tuple[str, str]:
    """Return ``(proxy_text, source)`` for a gold question's claim proxy.

    Priority: ``expected_key_points`` (joined) -> ``answer_note`` ->
    ``question_text``. ``source`` records which arm supplied it
    (``"key_points"`` / ``"answer_note"`` / ``"question_text"``) so the bias of
    the measured distribution is auditable.
    """
    kps = question.get("expected_key_points")
    if isinstance(kps, list):
        points = [str(k).strip() for k in kps if str(k).strip()]
        if points:
            # Join into sentence-shaped text so split_claims (used downstream by
            # attribute_citations) sees discrete claims.
            joined = ". ".join(p.rstrip(".") for p in points) + "."
            return joined, "key_points"
    note = question.get("answer_note")
    if isinstance(note, str) and note.strip():
        return note.strip(), "answer_note"
    return str(question.get("question_text", "")).strip(), "question_text"


# --------------------------------------------------------------------------- #
# Sample / threshold shapes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScoreSample:
    """One (passage, gold-relevance) support-score observation.

    ``score`` is the passage's best shingle support of the question's claim
    proxy; ``relevant`` is gold membership (chunk_id in relevant_passages).
    ``stream`` is ``"gold"`` (replayed gold) or ``"capture"`` (live prune
    capture) so the two evidence streams stay distinguishable.
    """

    score: float
    relevant: bool
    stream: str = "gold"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "relevant": self.relevant,
            "stream": self.stream,
        }


@dataclass(frozen=True)
class ThresholdRow:
    """Precision/recall/F1 at one candidate ``min_overlap`` threshold."""

    threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": round(self.threshold, 6),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
        }


def _dist(scores: Sequence[float]) -> Dict[str, Any]:
    """Summary stats of a score list (deterministic, stdlib)."""
    vals = sorted(float(s) for s in scores)
    if not vals:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": len(vals),
        "min": round(vals[0], 6),
        "median": round(statistics.median(vals), 6),
        "mean": round(statistics.fmean(vals), 6),
        "max": round(vals[-1], 6),
    }


@dataclass(frozen=True)
class AttributionCalibrationResult:
    """In-memory shape of ``attribution_calibration_<ts>.json``."""

    schema_version: str
    course_slug: str
    engine: str
    n_gold_questions: int
    relevant_distribution: Dict[str, Any]
    non_relevant_distribution: Dict[str, Any]
    capture_cited_distribution: Dict[str, Any]
    capture_uncited_distribution: Dict[str, Any]
    sweep: List[Dict[str, Any]]
    recommended: Optional[Dict[str, Any]]
    current_min_overlap: float
    add_min_shingle: float
    add_min_shingle_knob_warranted: Dict[str, Any]
    claim_source_counts: Dict[str, int]
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "course_slug": self.course_slug,
            "engine": self.engine,
            "shingle_size": ATTRIBUTION_SHINGLE_SIZE,
            "n_gold_questions": self.n_gold_questions,
            "distributions": {
                "gold_relevant": self.relevant_distribution,
                "gold_non_relevant": self.non_relevant_distribution,
                "capture_cited": self.capture_cited_distribution,
                "capture_uncited": self.capture_uncited_distribution,
            },
            "sweep": self.sweep,
            "recommended": self.recommended,
            "current_min_overlap": self.current_min_overlap,
            "add_min_shingle": self.add_min_shingle,
            "add_min_shingle_knob_warranted": self.add_min_shingle_knob_warranted,
            "claim_source_counts": self.claim_source_counts,
            "generated_at": self.generated_at,
        }


# --------------------------------------------------------------------------- #
# Pure calibration math
# --------------------------------------------------------------------------- #


def _sweep(samples: Sequence[ScoreSample]) -> List[ThresholdRow]:
    """Precision/recall/F1 over every observed score as a candidate threshold.

    A sample is PREDICTED relevant at threshold ``t`` iff ``score >= t``. tp/fp
    are over the predicted-relevant set; fn are gold-relevant samples scoring
    below ``t``. Candidate thresholds are the observed scores (+0.0), so the
    sweep is corpus-anchored, not an arbitrary grid.
    """
    if not samples:
        return []
    n_relevant = sum(1 for s in samples if s.relevant)
    candidates = sorted({0.0} | {round(s.score, 6) for s in samples})
    rows: List[ThresholdRow] = []
    for t in candidates:
        tp = sum(1 for s in samples if s.score >= t and s.relevant)
        fp = sum(1 for s in samples if s.score >= t and not s.relevant)
        fn = n_relevant - tp
        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (tp / n_relevant) if n_relevant else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        rows.append(
            ThresholdRow(
                threshold=t, tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1,
            )
        )
    return rows


def _recommend(
    sweep: Sequence[ThresholdRow], *, precision_floor: float
) -> Optional[Dict[str, Any]]:
    """Pick the headline recommendation + report both candidate rules.

    ``f1_max``: max-F1 threshold (tie-broken by the larger threshold — a more
    selective support bar is preferred at equal F1, consistent with precision-
    first). ``precision_floored``: the threshold with the HIGHEST recall whose
    precision >= ``precision_floor`` (tie-broken by larger threshold). Headline
    ``recommended`` = precision_floored when it qualifies, else f1_max.
    """
    if not sweep:
        return None

    f1_max = max(sweep, key=lambda r: (r.f1, r.threshold))
    qualifying = [r for r in sweep if r.precision >= precision_floor]
    precision_floored = (
        max(qualifying, key=lambda r: (r.recall, r.threshold))
        if qualifying
        else None
    )

    if precision_floored is not None:
        headline = precision_floored
        rule = (
            f"precision_floored: largest-recall threshold with precision >= "
            f"{precision_floor}"
        )
    else:
        headline = f1_max
        rule = "f1_max (no threshold cleared the precision floor)"

    out: Dict[str, Any] = {
        "min_overlap": round(headline.threshold, 6),
        "precision": round(headline.precision, 6),
        "recall": round(headline.recall, 6),
        "f1": round(headline.f1, 6),
        "rule": rule,
        "precision_floor": precision_floor,
        "f1_max": f1_max.to_dict(),
        "precision_floored": (
            precision_floored.to_dict() if precision_floored is not None else None
        ),
    }
    return out


def _add_min_shingle_assessment(
    relevant_scores: Sequence[float], capture_cited_scores: Sequence[float]
) -> Dict[str, Any]:
    """Decide whether ADD_MIN_SHINGLE warrants its own env knob.

    The ADD bar gates ADDITIONS (uncited passages promoted to citations). If the
    median gold-relevant (or, preferably, real-cited) shingle sits well BELOW
    the current constant, additions would essentially never fire — too strict.
    If it sits well ABOVE, additions would over-fire — too loose. Either case
    argues for a tunable env knob. The threshold ``+/- 0.15`` band around the
    constant is the "leave it constant" zone.
    """
    # Prefer the real-cited distribution (model phrasing) when present; fall
    # back to the gold-relevant ceiling.
    evidence = list(capture_cited_scores) or list(relevant_scores)
    if not evidence:
        return {
            "warranted": False,
            "reason": "no evidence (no captures, no gold-relevant samples)",
            "median": None,
            "current": ADD_MIN_SHINGLE,
        }
    median = statistics.median(evidence)
    gap = median - ADD_MIN_SHINGLE
    band = 0.15
    if gap < -band:
        warranted, reason = True, (
            f"median cited shingle {median:.3f} sits {abs(gap):.3f} BELOW the "
            f"{ADD_MIN_SHINGLE} ADD bar — additions would rarely fire; a tunable "
            f"ED4ALL_ANSWER_ADD_MIN_SHINGLE knob is warranted"
        )
    elif gap > band:
        warranted, reason = True, (
            f"median cited shingle {median:.3f} sits {gap:.3f} ABOVE the "
            f"{ADD_MIN_SHINGLE} ADD bar — additions would over-fire; a tunable "
            f"ED4ALL_ANSWER_ADD_MIN_SHINGLE knob is warranted"
        )
    else:
        warranted, reason = False, (
            f"median cited shingle {median:.3f} is within +/-{band} of the "
            f"{ADD_MIN_SHINGLE} ADD bar — constant is adequate"
        )
    return {
        "warranted": warranted,
        "reason": reason,
        "median": round(median, 6),
        "current": ADD_MIN_SHINGLE,
        "evidence_stream": "capture_cited" if capture_cited_scores else "gold_relevant",
    }


def calibrate_from_samples(
    *,
    course_slug: str,
    engine: str,
    samples: Sequence[ScoreSample],
    n_gold_questions: int,
    capture_cited_scores: Optional[Sequence[float]] = None,
    capture_uncited_scores: Optional[Sequence[float]] = None,
    claim_source_counts: Optional[Dict[str, int]] = None,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
    generated_at: Optional[str] = None,
) -> AttributionCalibrationResult:
    """Pure calibration over pre-measured (score, relevant) samples.

    Separated from retriever I/O so it is unit-testable on synthetic score sets.
    ``samples`` are the gold-replay observations; the capture score lists feed
    the second-stream distributions + the ADD-bar assessment.
    """
    capture_cited_scores = list(capture_cited_scores or [])
    capture_uncited_scores = list(capture_uncited_scores or [])

    relevant_scores = [s.score for s in samples if s.relevant]
    non_relevant_scores = [s.score for s in samples if not s.relevant]

    sweep = _sweep(samples)
    recommended = _recommend(sweep, precision_floor=precision_floor)
    add_assessment = _add_min_shingle_assessment(
        relevant_scores, capture_cited_scores
    )

    return AttributionCalibrationResult(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        course_slug=course_slug,
        engine=engine,
        n_gold_questions=n_gold_questions,
        relevant_distribution=_dist(relevant_scores),
        non_relevant_distribution=_dist(non_relevant_scores),
        capture_cited_distribution=_dist(capture_cited_scores),
        capture_uncited_distribution=_dist(capture_uncited_scores),
        sweep=[r.to_dict() for r in sweep],
        recommended=recommended,
        current_min_overlap=float(DEFAULT_MIN_OVERLAP),
        add_min_shingle=float(ADD_MIN_SHINGLE),
        add_min_shingle_knob_warranted=add_assessment,
        claim_source_counts=dict(claim_source_counts or {}),
        generated_at=generated_at or _utcnow_iso(),
    )


# --------------------------------------------------------------------------- #
# Live citation-prune capture ingestion (second evidence stream)
# --------------------------------------------------------------------------- #

# Parses the rationale token shape emitted by
# grounded_answer._emit_citation_prune: per_citation=[chunk@shingle/coverage*N(cited|uncited), ...]
_PER_CITATION_BLOCK_RE = re.compile(r"per_citation=\[(?P<body>.*?)\]")
_PER_CITATION_ENTRY_RE = re.compile(
    r"(?P<cid>[^@,\[\]]+)@(?P<shingle>[0-9.]+)/(?P<coverage>[0-9.]+)"
    r"\*(?P<n>\d+)\((?P<flag>cited|uncited)\)"
)


def _parse_prune_rationale(rationale: str) -> List[Tuple[str, float, float, int, str]]:
    """Parse ``per_citation=[...]`` entries from one prune-capture rationale.

    Returns ``(chunk_id, shingle, coverage, supported_claim_count, flag)`` tuples
    (flag in ``{"cited","uncited"}``). Tolerant of a missing/garbled block (an
    empty list) — never raises on a malformed capture line.
    """
    if not rationale:
        return []
    block = _PER_CITATION_BLOCK_RE.search(rationale)
    if not block:
        return []
    out: List[Tuple[str, float, float, int, str]] = []
    for m in _PER_CITATION_ENTRY_RE.finditer(block.group("body")):
        try:
            out.append(
                (
                    m.group("cid").strip(),
                    float(m.group("shingle")),
                    float(m.group("coverage")),
                    int(m.group("n")),
                    m.group("flag"),
                )
            )
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
    return out


def ingest_prune_captures(
    capture_dir: Path,
) -> Tuple[List[float], List[float]]:
    """Read live ``grounded_answer_citation_prune`` captures into score lists.

    Walks ``decisions_*.jsonl`` under ``capture_dir``, parses each
    ``grounded_answer_citation_prune`` event's ``per_citation`` block, and
    returns ``(cited_shingle_scores, uncited_shingle_scores)`` — the real-model
    second evidence stream. Best-effort: a missing dir / unparsable line is
    skipped (empty lists when nothing parses).
    """
    cited: List[float] = []
    uncited: List[float] = []
    if not capture_dir.is_dir():
        return cited, uncited
    for path in sorted(capture_dir.glob("decisions_*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("decision_type") != "grounded_answer_citation_prune":
                continue
            rationale = str(event.get("rationale", ""))
            for _cid, shingle, _cov, _n, flag in _parse_prune_rationale(rationale):
                (cited if flag == "cited" else uncited).append(shingle)
    return cited, uncited


# --------------------------------------------------------------------------- #
# Live-retriever calibration driver
# --------------------------------------------------------------------------- #


def _relevant_ids(question: Dict[str, Any]) -> set:
    ids: set = set()
    for p in question.get("relevant_passages") or []:
        if isinstance(p, dict) and p.get("chunk_id"):
            ids.add(str(p["chunk_id"]))
    return ids


def _passage_best_shingle(claim_proxy: str, passage_text: str) -> float:
    """Best shingle support of the claim proxy in one passage.

    Uses the SAME normalization + shingle arm as
    ``citation_attribution.attribute_citations`` (one passage, the proxy as the
    whole-answer claim text), so the measured score is exactly the support score
    the answer path would compute.
    """
    return _shingle_support(_normalize(claim_proxy), _normalize(passage_text))


def calibrate(
    course_slug: str,
    *,
    engine: str = "lexical",
    repo_root: Optional[Path] = None,
    limit: int = 8,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
    capture_dir: Optional[Path] = None,
    write: bool = True,
    output_path: Optional[Path] = None,
    retrieve_fn: Optional[Any] = None,
) -> AttributionCalibrationResult:
    """Replay a course's gold set through the retriever and calibrate min_overlap.

    For each gold question: build its claim proxy, retrieve top-``limit``
    passages, and emit one :class:`ScoreSample` per retrieved passage (score =
    best shingle support of the proxy; relevant = chunk_id in gold). Also ingest
    live prune captures as the second stream. ``retrieve_fn`` is injectable for
    tests (signature ``(libv2_root, course_slug, query, engine, limit) ->
    list[passage]``); the default uses the live LibV2 retriever (lazy import).
    """
    from lib.retrieval.refusal import _libv2_root

    libv2_root = _libv2_root(repo_root)
    course_dir = libv2_root / "courses" / course_slug
    gold = _load_json(course_dir / RETRIEVAL_EVAL_SUBDIR / GOLD_SET_FILENAME) or {}
    questions = [q for q in gold.get("questions", []) if isinstance(q, dict)]

    retriever = retrieve_fn if retrieve_fn is not None else _default_retrieve

    samples: List[ScoreSample] = []
    claim_source_counts: Dict[str, int] = {}
    for q in questions:
        proxy, source = claim_proxy_for_question(q)
        claim_source_counts[source] = claim_source_counts.get(source, 0) + 1
        rel_ids = _relevant_ids(q)
        passages = retriever(libv2_root, course_slug, q.get("question_text", ""),
                             engine=engine, limit=limit)
        for p in passages:
            cid = str(getattr(p, "chunk_id", "") or
                      (p.get("chunk_id") if isinstance(p, dict) else "") or "")
            ptext = str(getattr(p, "text", "") or
                        (p.get("text") if isinstance(p, dict) else "") or "")
            score = _passage_best_shingle(proxy, ptext)
            samples.append(ScoreSample(score=score, relevant=cid in rel_ids))

    cap_dir = capture_dir if capture_dir is not None else _default_capture_dir(
        repo_root, course_slug
    )
    cited_scores, uncited_scores = ingest_prune_captures(cap_dir)

    result = calibrate_from_samples(
        course_slug=course_slug,
        engine=engine,
        samples=samples,
        n_gold_questions=len(questions),
        capture_cited_scores=cited_scores,
        capture_uncited_scores=uncited_scores,
        claim_source_counts=claim_source_counts,
        precision_floor=precision_floor,
    )

    if write:
        out = output_path or (
            course_dir / RETRIEVAL_EVAL_SUBDIR
            / f"{CALIBRATION_FILENAME_PREFIX}{_ts()}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _default_retrieve(
    libv2_root: Path, course_slug: str, query: str, *, engine: str, limit: int
) -> List[Any]:
    from LibV2.tools.libv2.retriever import retrieve_chunks

    import inspect

    params = inspect.signature(retrieve_chunks).parameters
    kwargs: Dict[str, Any] = {"course_slug": course_slug, "limit": limit}
    if "engine" in params:
        kwargs["engine"] = engine
    elif engine != "lexical":
        raise RuntimeError(
            f"engine={engine!r} requested but the installed retrieve_chunks has "
            f"no 'engine' parameter; refusing to silently downgrade to lexical."
        )
    return list(retrieve_chunks(libv2_root, query, **kwargs))


def _default_capture_dir(repo_root: Optional[Path], course_slug: str) -> Path:
    """Resolve the live citation-prune capture dir for a course.

    ``training-captures/libv2/<slug>/phase_libv2-answer/`` (plan §4). Honors
    ``ED4ALL_TRAINING_CAPTURES_DIR`` via ``lib.paths`` when importable.
    """
    try:
        from lib.paths import get_training_captures_dir

        base = get_training_captures_dir()
    except Exception:  # noqa: BLE001 — fall back to repo-relative
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        base = root / "training-captures"
    return Path(base) / "libv2" / course_slug / "phase_libv2-answer"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- #
# CLI (__main__)
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lib.retrieval.attribution_calibrate",
        description=(
            "Calibrate the citation-attribution support min_overlap knob "
            "against a course gold set (retrieval only — no LLM). Emits "
            "retrieval_eval/attribution_calibration_<ts>.json."
        ),
    )
    parser.add_argument("--course", required=True, help="course slug")
    parser.add_argument(
        "--engine", default="lexical",
        choices=["lexical", "semantic", "hybrid-rrf"],
        help="retrieval engine for the replay (default: lexical)",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--precision-floor", type=float, default=DEFAULT_PRECISION_FLOOR,
        help="precision floor for the precision_floored recommendation",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="print the calibration JSON without writing the artifact",
    )
    parser.add_argument("--repo-root", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    args = _build_arg_parser().parse_args(argv)
    result = calibrate(
        args.course,
        engine=args.engine,
        repo_root=Path(args.repo_root) if args.repo_root else None,
        limit=args.limit,
        precision_floor=args.precision_floor,
        write=not args.no_write,
    )
    rec = result.recommended
    if rec is None:
        print(f"[{args.course}/{args.engine}] no gold samples — nothing to calibrate.")
    else:
        print(
            f"[{args.course}/{args.engine}] recommended min_overlap="
            f"{rec['min_overlap']} (precision={rec['precision']}, "
            f"recall={rec['recall']}, f1={rec['f1']}; rule={rec['rule']})."
        )
        warranted = result.add_min_shingle_knob_warranted
        print(f"  ADD_MIN_SHINGLE knob warranted: {warranted['warranted']} "
              f"({warranted['reason']})")
    if args.no_write:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
