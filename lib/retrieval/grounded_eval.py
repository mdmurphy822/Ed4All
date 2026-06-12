"""Grounded-answer eval harness (D8) — runs the grounded-answer pipeline over a
course's gold set + refusal probes and rolls up groundedness + citation-precision
metrics into a ``retrieval_eval/grounded_answer_eval_<ts>.json`` report plus a
deterministic, seeded human-rubric review sample.

This is master-plan headline metrics #2 (groundedness) and #3 (citation
precision). The nightly command mirrors the WS2 benchmark operationally::

    python -m lib.retrieval.grounded_eval --course <slug> --engine lexical

writing a timestamped sibling of WS2's ``benchmark_*.json``. CI never runs the
real-model eval — the report-shape tests run this harness on the mini-course
fixture with a ``FakeAnswerClient`` + a deterministic fake NLI.

Pipeline dependency: this harness drives
``lib.retrieval.grounded_answer.answer_course_question`` (E6 wave B). That
module is import-guarded — if it has not landed,
:func:`run_grounded_eval` raises :class:`PipelineUnavailable` (a
``NotImplementedError`` subclass) naming the missing dependency. No fake
pipeline results are ever fabricated.

Filenames written here are disjoint from WS1 (``gold_set.json``), WS2
(``benchmark_*.json``), and E6 (``refusal_probes.json`` /
``refusal_calibration.json``): this harness owns
``grounded_answer_eval_*.json`` and ``groundedness_review_sample.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.retrieval.answer_scoring import (
    score_key_point_coverage,
    score_part_coverage,
    score_population_breakdown,
)
from lib.retrieval.gold_coverage import _population_of_chunk
from lib.retrieval.gold_set import (
    _load_chunks_by_id,
    has_critical_issues,
    load_gold_set,
)

#: Bumped 1.0 -> 1.1 for the P4 additive per-question scoring fields
#: (key_point_coverage, part_coverage, population breakdown). Every v1.0
#: report field is preserved; the new fields are additive and present only
#: when the v1.1 gold question carries the matching authored data.
EVAL_SCHEMA_VERSION = "1.1"
RETRIEVAL_EVAL_SUBDIR = "retrieval_eval"
REVIEW_SAMPLE_FILENAME = "groundedness_review_sample.json"

# --------------------------------------------------------------------------- #
# Milestone targets — measure-then-pin (floors CURRENT evidence meets)
# --------------------------------------------------------------------------- #

#: Date the targets below were pinned. RE-PINNED 2026-06-12 against the first
#: scaled-up frozen-gold eval run (single-course union-corpus calibration basis).
MILESTONE_TARGETS_PINNED_AT = "2026-06-12"

#: COMPARABILITY BOUNDARY (plan §4): these targets are RE-PINNED against the
#: 2026-06-12 frozen-gold run and are NOT comparable to the prior 2026-06-10
#: pins. The basis changed in two ways the plan flags as a hard boundary:
#:   (1) Gold scaled up from a 10-question/9-probe seed to a 77-question frozen
#:       gold v1.1 + 34 refusal probes (single course, union corpus). With one
#:       relevant passage pinned per drafted gold question, citation_precision is
#:       EXPECTED to move via denominator semantics (a multi-citation answer
#:       mathematically deflates the per-question precision), NOT via a pipeline
#:       change — so the pre/post citation_precision floors are not comparable.
#:   (2) refusal metrics move to a NEW hard-probe basis (34 probes incl. scaled
#:       near-miss / adjacent-domain negatives) — NOT comparable to the old
#:       9-probe basis.
#:
#: SINGLE-COURSE CAVEAT (risk R4): with one calibrated course the MIN-pin
#: convention degenerates to that course's measurement; every floor below carries
#: a single-corpus caveat until a second different-family course joins. Re-pin
#: (tighten) only after a new measured run on a different course clears a
#: stricter bound — never tighten from fewer courses than the prior pin.
#:
#: Measured run (single-course union-corpus calibration basis, 2026-06-12;
#: model qwen2.5:7b-instruct-q4_K_M, prompt ws3.v2, bge-large + hybrid-rrf,
#: 77q frozen gold v1.1 / 34 probes, local backend):
#:   answer_rate 0.961  citation_resolution_rate 1.0  citation_precision 0.3495
#:   groundedness_rate_mean 0.572  unsupported_claim_rate 0.2326
#:   refusal_recall 0.50  refusal_precision 0.8947  key_point_coverage 0.5439
#:
#: citation_resolution_rate stays pinned at 1.0: every emitted citation resolved
#: on this run too, so the floor IS the measurement — any dip is a real
#: anchoring break, not noise.
MILESTONE_TARGETS: Dict[str, float] = {
    # FLOOR. measured 0.961 (single course, 77q gold v1.1, 2026-06-12);
    # pinned to 0.95 (NOT comparable to the old 0.50/10q basis).
    "answer_rate": 0.95,
    # FLOOR == measurement. Every emitted citation resolved on the 2026-06-12
    # run (and on all prior runs). Pinned at 1.0 exactly; any dip is a real
    # anchoring break.
    "citation_resolution_rate": 1.0,
    # FLOOR. measured 0.3495 (single course, 77q gold v1.1, 2026-06-12),
    # pinned to 0.30. DENOMINATOR ARTIFACT: each drafted gold question pins
    # exactly ONE relevant passage, so a multi-citation answer mathematically
    # deflates per-question citation_precision. This floor is therefore NOT
    # comparable to the old 0.45/10q-basis pin — the metric moved via gold
    # denominator semantics (plan §4 comparability boundary), not via a
    # pipeline regression. A richer multi-passage gold would raise it.
    "citation_precision": 0.30,
    # FLOOR. measured 0.572 (single course 2026-06-12), pinned conservatively
    # to 0.15 — the prior cross-course floor, kept because this single run does
    # not establish a credible cross-corpus bound (R4); tightening waits on a
    # second different-family course.
    "groundedness_rate_mean": 0.15,
    # CEILING. measured 0.2326 (single course 2026-06-12); pinned unchanged at
    # 0.25 (lower is better). Comfortably under the ceiling on this run.
    "unsupported_claim_rate": 0.25,
    # FLOOR. measured 0.50 on the NEW 34-probe hard-probe basis (single course
    # 2026-06-12), pinned to 0.45. NOT comparable to the old 9-probe basis. The
    # near-miss / adjacent-domain gap (the 0.5 of probes the retrieval threshold
    # cannot catch) is MODEL-policy-owned, not a retrieval-threshold matter —
    # see lib/retrieval/refusal.py PINNED_POLICIES hybrid-rrf overlap block.
    "refusal_precision_floor_note": 0.0,  # placeholder removed below
    # FLOOR. measured 0.50 (new hard-probe basis 2026-06-12), pinned 0.45.
    "refusal_recall": 0.45,
    # FLOOR. measured 0.8947 (single course 2026-06-12), pinned to 0.85.
    "refusal_precision": 0.85,
}
# Remove the inline placeholder key used only to anchor the refusal_recall note.
MILESTONE_TARGETS.pop("refusal_precision_floor_note", None)

#: DIAGNOSTIC (unpinned). key_point_coverage is a FIRST-measurement diagnostic
#: (2026-06-12 baseline 0.54 on the single-course 77q frozen gold v1.1) — it is
#: deliberately NOT in MILESTONE_TARGETS until a second measured run establishes
#: it is stable enough to floor (plan risk R7 posture for new additive metrics).
#: Recorded here as a commented baseline so a future re-pin has the starting
#: point; the staleness test does NOT gate on it.
KEY_POINT_COVERAGE_DIAGNOSTIC_BASELINE = 0.54

#: Names of milestone targets that are CEILINGS (measured value must be <= the
#: target) rather than floors. Lower is better for these (e.g. unsupported
#: claims). Everything else in :data:`MILESTONE_TARGETS` is a floor (>=).
MILESTONE_CEILINGS: frozenset = frozenset({"unsupported_claim_rate"})

#: Statuses whose ``answer_text`` is a real answer (vs refusal / block).
_ANSWERED_STATUSES = frozenset({"answered", "answered_with_warnings"})
_REFUSED_STATUSES = frozenset(
    {"refused_low_confidence", "refused_not_in_course"}
)
_BLOCKED_INVALID = "blocked_invalid_citation"
_BLOCKED_GATE = "blocked_citation_gate"

#: Per-question status recorded when the composer EXHAUSTS its retry budget
#: (parse exhaustion, post-remediation invalid citations) or fails closed on a
#: silently-truncated prompt. These are per-question composer failures, NOT a
#: systemic backend outage (which stays fatal and aborts the whole eval). A
#: composer_exhausted question counts as not-answered for answer_rate and is
#: NOT a refusal (it never produced a refusal verdict).
_COMPOSER_EXHAUSTED = "composer_exhausted"


class PipelineUnavailable(NotImplementedError):
    """The grounded-answer pipeline (E6 wave B) has not landed yet.

    Raised (never silently degraded) when
    ``lib.retrieval.grounded_answer.answer_course_question`` is not importable,
    so an operator sees the exact missing dependency rather than a fabricated
    eval result.
    """


def _composer_exhaustion_errors() -> Tuple[type, ...]:
    """Lazily resolve the composer error types treated as per-question failures.

    ``AnswerComposeError`` (parse exhaustion), ``InvalidCitationError``
    (post-remediation unknown citations), and ``PromptTruncatedError`` (silent
    head-truncation fail-closed) are PER-QUESTION composer failures: the eval
    records them and moves on. ``AnswerBackendUnavailable`` is deliberately
    EXCLUDED — a missing/refusing backend is systemic and must abort the run
    (catching it per-question would silently hollow out every metric).

    Returns an empty tuple when the composer module can't be imported (then the
    caller catches nothing extra and any error propagates as before).
    """
    try:
        from lib.retrieval.answer_composer import (
            AnswerComposeError,
            InvalidCitationError,
        )
        from lib.retrieval.answer_backend import PromptTruncatedError
    except Exception:  # pragma: no cover - composer absent → no extra catch
        return tuple()
    # InvalidCitationError subclasses AnswerComposeError; list both for clarity.
    return (AnswerComposeError, InvalidCitationError, PromptTruncatedError)


def _import_answer_pipeline() -> Any:
    """Import-guard ``answer_course_question`` (E6 wave B dependency).

    Returns the callable, or raises :class:`PipelineUnavailable` naming the
    missing module / symbol. The import is lazy so the harness module is
    importable (and its non-pipeline helpers testable) before E6 lands.
    """
    try:
        from lib.retrieval.grounded_answer import answer_course_question
    except Exception as exc:  # noqa: BLE001 — surface as a typed, named error
        raise PipelineUnavailable(
            "lib.retrieval.grounded_answer.answer_course_question is "
            "required by the grounded-answer eval harness but is not yet "
            f"importable ({type(exc).__name__}: {exc}). This is the E6 "
            "wave-B pipeline dependency; the eval harness fabricates no "
            "results in its absence."
        ) from exc
    return answer_course_question


# --------------------------------------------------------------------------- #
# Gold-set / probe helpers
# --------------------------------------------------------------------------- #

def _course_dir(repo_root: Path, course_slug: str) -> Path:
    libv2_root = os.environ.get("ED4ALL_LIBV2_ROOT")
    base = Path(libv2_root) if libv2_root else (Path(repo_root) / "LibV2")
    return base / "courses" / course_slug


def _gold_questions(gold: Dict[str, Any]) -> List[Dict[str, Any]]:
    questions = gold.get("questions")
    return list(questions) if isinstance(questions, list) else []


def _relevant_chunk_ids(
    question: Dict[str, Any],
) -> Tuple[set, set]:
    """Return ``(all_relevant_ids, primary_ids)`` for a gold question.

    Tolerant of both the ``relevant_passages`` (gold-set v1.0) and a flat
    ``relevant_chunk_ids`` shape; relevance defaults to ``supporting``.
    """
    all_ids: set = set()
    primary: set = set()
    passages = question.get("relevant_passages")
    if isinstance(passages, list):
        for entry in passages:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("chunk_id")
            if not cid:
                continue
            all_ids.add(str(cid))
            if str(entry.get("relevance", "supporting")) == "primary":
                primary.add(str(cid))
    flat = question.get("relevant_chunk_ids")
    if isinstance(flat, list):
        for cid in flat:
            if cid:
                all_ids.add(str(cid))
    return all_ids, primary


def _resolve_keypoint_nli(
    with_groundedness: bool, nli: Optional[Any]
) -> Tuple[Optional[Any], Optional[float]]:
    """Resolve the optional NLI classifier + entailment floor for the key-point
    completeness arm.

    Only consulted when ``with_groundedness`` is True (the key-point NLI arm is
    a no-op on the fast lane). Reuses the SAME classifier resolution +
    entailment floor that groundedness uses, so the completeness verdict and the
    groundedness verdict never disagree on what "the answer supports this claim"
    means. Returns ``(None, None)`` when the NLI model is unavailable (graceful
    degrade — the deterministic shingle/coverage arms still score key points).
    """
    if not with_groundedness:
        return None, None
    try:
        from lib.retrieval.groundedness import _resolve_nli
        from lib.validators.pair._claim_support_thresholds import (
            _DEFAULT_ENTAILMENT_FLOOR,
        )
    except Exception:  # noqa: BLE001 — groundedness deps absent → deterministic only
        return None, None
    resolved = _resolve_nli(nli)
    if resolved is None:
        return None, None
    return resolved, float(_DEFAULT_ENTAILMENT_FLOOR)


def _load_pinned_chunks(
    course_dir: Path, gold: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Load the gold set's pinned chunkset indexed by id (population breakdown).

    Best-effort: returns ``{}`` when the chunkset path is absent/unreadable so a
    missing chunkset degrades the population slice to empty rather than aborting
    the whole eval (the headline metrics never depended on the chunkset bytes).
    """
    chunks_rel = (gold.get("chunkset") or {}).get("chunks_path") or ""
    if not chunks_rel:
        return {}
    chunks_path = course_dir / chunks_rel
    if not chunks_path.is_file():
        return {}
    try:
        return _load_chunks_by_id(chunks_path)
    except Exception:  # noqa: BLE001 — chunkset unreadable → empty population slice
        return {}


def _question_expected_population(question: Dict[str, Any]) -> str:
    val = question.get("expected_citation_population")
    return val if val in ("source", "course", "both", "any") else "any"


def _question_key_points(question: Dict[str, Any]) -> List[str]:
    kps = question.get("expected_key_points")
    return [str(k) for k in kps if str(k).strip()] if isinstance(kps, list) else []


def _question_parts(question: Dict[str, Any]) -> List[Dict[str, Any]]:
    parts = question.get("parts")
    return [p for p in parts if isinstance(p, dict)] if isinstance(parts, list) else []


def _load_probes(probes_path: Path) -> List[Dict[str, Any]]:
    if not probes_path.exists() or not probes_path.is_file():
        return []
    try:
        doc = json.loads(probes_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    probes = doc.get("probes")
    return list(probes) if isinstance(probes, list) else []


# --------------------------------------------------------------------------- #
# Metric arithmetic
# --------------------------------------------------------------------------- #

def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (deterministic; no numpy dependency)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = pct / 100.0 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


def _citations_list(answer: Any) -> List[Dict[str, Any]]:
    """Read a pipeline answer's citations as a list of dicts (duck-typed)."""
    cites = getattr(answer, "citations", None)
    if cites is None and isinstance(answer, dict):
        cites = answer.get("citations")
    out: List[Dict[str, Any]] = []
    for c in cites or []:
        if isinstance(c, dict):
            out.append(c)
        elif hasattr(c, "to_dict"):
            out.append(c.to_dict())
        else:
            out.append({"chunk_id": getattr(c, "chunk_id", None)})
    return out


def _answer_field(answer: Any, name: str, default: Any = None) -> Any:
    val = getattr(answer, name, None)
    if val is None and isinstance(answer, dict):
        val = answer.get(name, default)
    return default if val is None else val


# --------------------------------------------------------------------------- #
# Review sample (deterministic, seeded)
# --------------------------------------------------------------------------- #

def _sample_size(n_questions: int) -> int:
    """n = max(10, 20% of questions), capped at the question count."""
    target = max(10, int(round(0.20 * n_questions)))
    return min(target, n_questions)


def _seeded_order(question_ids: Sequence[str], course_slug: str) -> List[str]:
    """Deterministic ordering of question ids, seeded by course slug.

    Sorts by ``sha256(course_slug + "::" + question_id)`` so the sample is
    reproducible across runs and machines (no PRNG state, no insertion-order
    dependence) yet shuffled relative to the gold-set order.
    """
    def _key(qid: str) -> str:
        return hashlib.sha256(
            f"{course_slug}::{qid}".encode("utf-8")
        ).hexdigest()

    return sorted(question_ids, key=_key)


def _build_review_sample(
    course_slug: str,
    per_question: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the deterministic human-rubric review sample (D6)."""
    by_id = {row["question_id"]: row for row in per_question}
    ordered = _seeded_order(list(by_id.keys()), course_slug)
    n = _sample_size(len(ordered))
    chosen = ordered[:n]
    samples = []
    for qid in chosen:
        row = by_id[qid]
        samples.append(
            {
                "question_id": qid,
                "question_text": row.get("question_text", ""),
                "answer_text": row.get("answer_text"),
                "status": row.get("status"),
                "citations": row.get("review_citations", []),
                "per_claim_nli_verdicts": row.get("per_claim_nli_verdicts", []),
                "reviewer_fields": {
                    "claims_all_supported": None,
                    "citation_correct": None,
                    "refusal_correct": None,
                    "notes": "",
                },
            }
        )
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "course_slug": course_slug,
        "sample_seed": "sha256(course_slug::question_id)",
        "n_sampled": len(samples),
        "n_questions": len(ordered),
        "samples": samples,
    }


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #

def run_grounded_eval(
    repo_root: Path,
    course_slug: str,
    *,
    engine: str = "semantic",
    client: Optional[Any] = None,
    nli: Optional[Any] = None,
    gold_set_path: Optional[Path] = None,
    refusal_probes_path: Optional[Path] = None,
    refusal_policy: Optional[Any] = None,
    with_groundedness: bool = True,
    limit: int = 8,
    output_path: Optional[Path] = None,
    review_sample_path: Optional[Path] = None,
    capture: Optional[Any] = None,
    answer_fn: Optional[Any] = None,
    write: bool = True,
) -> Dict[str, Any]:
    """Run the grounded-answer pipeline over a course's gold set + probes.

    For each answerable gold question: drive ``answer_course_question`` (E6
    wave B), score per-question groundedness (when ``with_groundedness`` and the
    pipeline returned NLI verdicts) and citation precision against the gold
    relevant set. For each refusal probe: drive the pipeline and record whether
    it refused / answered (refusal recall + precision).

    Returns the D8 report dict; when ``write`` is True also writes
    ``grounded_answer_eval_<ts>.json`` + ``groundedness_review_sample.json``
    into ``retrieval_eval/``.

    Raises :class:`PipelineUnavailable` when the E6 pipeline is absent (unless
    a test injects ``answer_fn``). Raises ``RuntimeError`` on any critical
    gold-set issue (WS1 fail-closed contract).
    """
    repo_root = Path(repo_root)
    course_dir = _course_dir(repo_root, course_slug)

    # 1) Load + verify the gold set. ANY critical issue is a hard error.
    gold, issues = load_gold_set(course_dir, verify=True)
    if has_critical_issues(issues):
        codes = sorted({i.code for i in issues if i.severity == "critical"})
        raise RuntimeError(
            f"gold set for {course_slug!r} has critical issues {codes}; "
            f"refusing to run the grounded eval on an unverified gold set."
        )
    questions = _gold_questions(gold)

    # 2) Resolve the pipeline callable (import-guarded; injectable for tests).
    pipeline = answer_fn if answer_fn is not None else _import_answer_pipeline()

    # 3) Resolve probe set.
    probes_path = (
        Path(refusal_probes_path)
        if refusal_probes_path is not None
        else course_dir / RETRIEVAL_EVAL_SUBDIR / "refusal_probes.json"
    )
    probes = _load_probes(probes_path)

    chunkset = gold.get("chunkset", {}) if isinstance(gold, dict) else {}
    gold_pin = {
        "path": str(gold_set_path) if gold_set_path else None,
        "chunks_sha256": chunkset.get("chunks_sha256"),
        "chunkset_kind": chunkset.get("kind"),
        # Measure-then-pin feed (plan §4): record the gold set's identity in the
        # report so a re-pin of MILESTONE_TARGETS records which gold set + how
        # many questions produced the measured floors (pre/post scale-up is a
        # comparability boundary — citation_precision is EXPECTED to move with
        # the question count via denominator semantics, not pipeline change).
        "schema_version": gold.get("schema_version"),
        "question_count": len(questions),
        "authored_at": gold.get("authored_at"),
    }

    # Pinned chunkset (for the per-population citation breakdown) + the NLI
    # entailment floor (shared with groundedness so the key-point completeness
    # arm agrees with the groundedness verdict on "the answer supports this").
    chunks_by_id = _load_pinned_chunks(course_dir, gold)
    nli_for_keypoints, entailment_floor = _resolve_keypoint_nli(
        with_groundedness, nli
    )
    # The gold set's chunkset.kind is the eval's source of truth for which
    # chunkset retrieval + the citation gate must resolve against; thread it
    # into the pipeline so a course with multiple chunksets (e.g. dart_chunks/
    # alongside an imscc-pinned index) is evaluated against the pinned kind
    # rather than the pipeline's directory-presence guess.
    gold_chunkset_kind = chunkset.get("kind")

    per_question_report: List[Dict[str, Any]] = []
    per_question_detail: List[Dict[str, Any]] = []
    latencies: List[float] = []
    model_id: Optional[str] = None
    prompt_version: Optional[str] = None
    refusal_policy_version: Optional[str] = None

    n_emitted_citations = 0
    n_resolved_citations = 0
    n_relevant_citations = 0
    n_relevant_primary_citations = 0
    answered_count = 0
    blocked_invalid = 0
    blocked_gate = 0
    composer_exhausted_count = 0
    false_refusals_on_gold = 0
    groundedness_rates: List[float] = []
    unsupported_rates: List[float] = []

    # P4 additive aggregates (key-point completeness, part coverage, per-
    # population citations). All roll up only over ANSWERED questions that
    # carry the matching authored v1.1 data; absent on v1.0 gold sets.
    # Per-population citation precision (plan §4): tally emitted + relevant
    # citations by the population (source/course) of the cited chunk, joined the
    # same way the coverage report classifies a chunk. 'both' counts a citation
    # that lands in a question whose gold passages span both populations.
    pop_emitted = {"source": 0, "course": 0}
    pop_relevant = {"source": 0, "course": 0}
    kp_total = 0
    kp_covered = 0
    kp_questions = 0
    part_covered_total = 0
    part_answered_total = 0
    part_uncovered_total = 0
    part_flagged_total = 0
    part_questions = 0
    pop_cited_source = 0
    pop_cited_course = 0
    pop_expected_checked = 0
    pop_expected_satisfied = 0

    catchable = _composer_exhaustion_errors()

    # --- answerable gold questions ---------------------------------------
    for q in questions:
        qid = str(q.get("question_id", ""))
        qtext = str(q.get("question_text", ""))
        all_rel, primary_rel = _relevant_chunk_ids(q)

        try:
            answer = pipeline(
                repo_root,
                course_slug,
                qtext,
                engine=engine,
                limit=limit,
                client=client,
                refusal_policy=refusal_policy,
                with_groundedness=with_groundedness,
                capture=capture,
                chunkset_kind=gold_chunkset_kind,
            )
        except catchable as exc:
            # Per-question composer failure (parse/citation exhaustion, or a
            # fail-closed truncation). Record it, count it as NOT-answered for
            # answer_rate, do NOT count it as a refusal, and keep evaluating
            # the rest of the gold set. The eval artifact still writes.
            composer_exhausted_count += 1
            latencies.append(0.0)
            per_question_report.append(
                {
                    "question_id": qid,
                    "status": _COMPOSER_EXHAUSTED,
                    "n_citations": 0,
                    "citations_resolved": 0,
                    "citation_relevant_primary": 0,
                    "groundedness_rate": None,
                    "latency_ms": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            per_question_detail.append(
                {
                    "question_id": qid,
                    "question_text": qtext,
                    "status": _COMPOSER_EXHAUSTED,
                    "answer_text": None,
                    "review_citations": [],
                    "per_claim_nli_verdicts": [],
                }
            )
            continue

        status = str(_answer_field(answer, "status", "unknown"))
        latency = float(_answer_field(answer, "latency_ms", 0.0))
        latencies.append(latency)
        if model_id is None:
            model_id = _answer_field(answer, "model_id", None)
        if prompt_version is None:
            prompt_version = _answer_field(answer, "prompt_version", None)
        if refusal_policy_version is None:
            conf = _answer_field(answer, "confidence", {}) or {}
            if isinstance(conf, dict):
                refusal_policy_version = conf.get("policy_version")

        cites = _citations_list(answer)
        n_cites = len(cites)
        resolved_here = 0
        relevant_here = 0
        relevant_primary_here = 0
        for c in cites:
            cid = str(c.get("chunk_id", ""))
            anchor_status = str(c.get("anchor_status", ""))
            # resolution: an emitted citation that resolved (the gate would
            # have blocked unresolved ones, so on-gate this is 1.0 by
            # construction; counted pre-gate so the metric measures the model).
            if anchor_status.startswith("resolved"):
                resolved_here += 1
            is_relevant = cid in all_rel
            if is_relevant:
                relevant_here += 1
            if cid in primary_rel:
                relevant_primary_here += 1
            # Per-population tally: classify the cited chunk by its population
            # (skip ids absent from the pinned chunkset — unclassifiable).
            chunk = chunks_by_id.get(cid)
            if chunk is not None:
                pop = _population_of_chunk(chunk)
                if pop in pop_emitted:
                    pop_emitted[pop] += 1
                    if is_relevant:
                        pop_relevant[pop] += 1

        n_emitted_citations += n_cites
        n_resolved_citations += resolved_here
        n_relevant_citations += relevant_here
        n_relevant_primary_citations += relevant_primary_here

        if status in _ANSWERED_STATUSES:
            answered_count += 1
        elif status == _BLOCKED_INVALID:
            blocked_invalid += 1
        elif status == _BLOCKED_GATE:
            blocked_gate += 1
        elif status in _REFUSED_STATUSES:
            # A refusal on an answerable gold question is a false refusal.
            false_refusals_on_gold += 1

        grounded = _answer_field(answer, "groundedness", None)
        g_rate: Optional[float] = None
        u_rate: Optional[float] = None
        per_claim: List[Dict[str, Any]] = []
        if isinstance(grounded, dict) and grounded.get("available"):
            scored = int(grounded.get("scored_count", 0) or 0)
            g_rate = float(grounded.get("groundedness_rate", 0.0) or 0.0)
            if scored:
                u_rate = float(grounded.get("unsupported_count", 0) or 0) / scored
            else:
                u_rate = 0.0
            groundedness_rates.append(g_rate)
            unsupported_rates.append(u_rate)
            per_claim = list(grounded.get("claims", []) or [])

        # --- P4 additive per-question scoring (only on answered questions) ---
        is_answered = status in _ANSWERED_STATUSES
        answer_text_val = _answer_field(answer, "answer_text", None)
        cited_ids = [str(c.get("chunk_id", "")) for c in cites if c.get("chunk_id")]

        kp_cov = None
        part_cov = None
        pop_break = None
        if is_answered:
            kp_cov = score_key_point_coverage(
                answer_text_val,
                _question_key_points(q),
                nli=nli_for_keypoints,
                entailment_floor=entailment_floor,
            )
            if kp_cov is not None:
                kp_total += kp_cov.total
                kp_covered += kp_cov.covered
                kp_questions += 1

            if str(q.get("question_type", "")) == "multi_part":
                part_cov = score_part_coverage(
                    answer_text_val, _question_parts(q)
                )
                if part_cov is not None:
                    part_covered_total += part_cov.n_covered_parts
                    part_answered_total += part_cov.n_answered
                    part_uncovered_total += part_cov.n_uncovered_parts
                    part_flagged_total += part_cov.n_correctly_flagged
                    part_questions += 1

            expected_pop = _question_expected_population(q)
            pop_break = score_population_breakdown(
                cited_ids,
                chunks_by_id,
                expected_population=expected_pop,
                answered=True,
            )
            pop_cited_source += pop_break.cited_source
            pop_cited_course += pop_break.cited_course
            if pop_break.expected_satisfied is not None:
                pop_expected_checked += 1
                if pop_break.expected_satisfied:
                    pop_expected_satisfied += 1

        row: Dict[str, Any] = {
            "question_id": qid,
            "status": status,
            "n_citations": n_cites,
            "citations_resolved": resolved_here,
            "citation_relevant_primary": relevant_primary_here,
            "groundedness_rate": g_rate,
            "latency_ms": latency,
        }
        if kp_cov is not None:
            row["key_point_coverage"] = kp_cov.to_dict()
        if part_cov is not None:
            row["part_coverage"] = part_cov.to_dict()
        if pop_break is not None:
            row["citation_population"] = pop_break.to_dict()
        per_question_report.append(row)
        per_question_detail.append(
            {
                "question_id": qid,
                "question_text": qtext,
                "status": status,
                "answer_text": _answer_field(answer, "answer_text", None),
                "review_citations": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "page_label": c.get("page_label"),
                        "text": c.get("text_quote"),
                    }
                    for c in cites
                ],
                "per_claim_nli_verdicts": per_claim,
            }
        )

    # --- refusal probes ---------------------------------------------------
    n_probes = len(probes)
    probe_refused = 0
    probe_answered = 0
    probe_composer_exhausted = 0
    for probe in probes:
        ptext = str(probe.get("question_text", ""))
        try:
            answer = pipeline(
                repo_root,
                course_slug,
                ptext,
                engine=engine,
                limit=limit,
                client=client,
                refusal_policy=refusal_policy,
                with_groundedness=False,
                capture=capture,
                chunkset_kind=gold_chunkset_kind,
            )
        except catchable:
            # A composer exhaustion on a probe is neither a clean refusal nor
            # an answer — it never reached a verdict, so it must NOT inflate
            # refusal_recall. Counted separately, surfaced in the report.
            probe_composer_exhausted += 1
            continue
        status = str(_answer_field(answer, "status", "unknown"))
        if status in _REFUSED_STATUSES:
            probe_refused += 1
        elif status in _ANSWERED_STATUSES:
            probe_answered += 1
        # blocked_* on a probe is neither a clean refusal nor an answer; it is
        # surfaced via the blocked counters only (not double-counted here).

    # refusal_recall = probes correctly refused / probes
    refusal_recall = (probe_refused / n_probes) if n_probes else 0.0
    # refusal_precision = correct refusals / all refusals (probes + false
    # gold refusals are the wrong refusals).
    total_refusals = probe_refused + false_refusals_on_gold
    refusal_precision = (
        (probe_refused / total_refusals) if total_refusals else 1.0
    )

    n_answerable = len(questions)
    headline = {
        "answer_rate": (answered_count / n_answerable) if n_answerable else 0.0,
        "citation_resolution_rate": (
            (n_resolved_citations / n_emitted_citations)
            if n_emitted_citations
            else 0.0
        ),
        "citation_precision": (
            (n_relevant_citations / n_emitted_citations)
            if n_emitted_citations
            else 0.0
        ),
        "citation_precision_primary": (
            (n_relevant_primary_citations / n_emitted_citations)
            if n_emitted_citations
            else 0.0
        ),
        "groundedness_rate_mean": (
            (sum(groundedness_rates) / len(groundedness_rates))
            if groundedness_rates
            else None
        ),
        "unsupported_claim_rate": (
            (sum(unsupported_rates) / len(unsupported_rates))
            if unsupported_rates
            else None
        ),
        "refusal": {
            "n_probes": n_probes,
            "refusal_recall": refusal_recall,
            "refusal_precision": refusal_precision,
            "false_refusals_on_gold": false_refusals_on_gold,
            "composer_exhausted": probe_composer_exhausted,
        },
        # P4 additive: per-population citation precision (source/course) +
        # expected-population satisfaction. precision_* is None when that
        # population emitted no citations (no denominator). On a v1.0 / non-
        # union gold set these stay zero-denominator (None) — additive, never
        # a headline milestone.
        "citation_precision_by_population": {
            "source": (
                (pop_relevant["source"] / pop_emitted["source"])
                if pop_emitted["source"]
                else None
            ),
            "course": (
                (pop_relevant["course"] / pop_emitted["course"])
                if pop_emitted["course"]
                else None
            ),
            "both": (
                ((pop_relevant["source"] + pop_relevant["course"])
                 / (pop_emitted["source"] + pop_emitted["course"]))
                if (pop_emitted["source"] + pop_emitted["course"])
                else None
            ),
            "emitted_source": pop_emitted["source"],
            "emitted_course": pop_emitted["course"],
        },
        "expected_population_satisfaction": {
            "checked": pop_expected_checked,
            "satisfied": pop_expected_satisfied,
            "rate": (
                (pop_expected_satisfied / pop_expected_checked)
                if pop_expected_checked
                else None
            ),
        },
        # P4 additive: claim-level completeness (key-point coverage) — only over
        # answered questions that authored expected_key_points (v1.1).
        "key_point_coverage": {
            "questions_scored": kp_questions,
            "total_key_points": kp_total,
            "covered_key_points": kp_covered,
            "coverage_rate": (kp_covered / kp_total) if kp_total else None,
        },
        # P4 additive (diagnostic, NOT pinned — risk R7): multi_part coverage.
        "part_coverage": {
            "questions_scored": part_questions,
            "covered_parts": part_covered_total,
            "answered_parts": part_answered_total,
            "uncovered_parts": part_uncovered_total,
            "correctly_flagged_parts": part_flagged_total,
            "answered_rate": (
                (part_answered_total / part_covered_total)
                if part_covered_total
                else None
            ),
            "correctly_flagged_rate": (
                (part_flagged_total / part_uncovered_total)
                if part_uncovered_total
                else None
            ),
            "_diagnostic": (
                "uncovered-part flagging is heuristic (risk R7); diagnostic, "
                "not a pinned milestone"
            ),
        },
        "latency_ms": {
            "p50": _percentile(latencies, 50.0),
            "p95": _percentile(latencies, 95.0),
        },
    }

    report = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "course_slug": course_slug,
        "engine": engine,
        "model_id": model_id,
        "prompt_version": prompt_version,
        "refusal_policy_version": refusal_policy_version,
        "gold": gold_pin,
        "questions": per_question_report,
        "headline": headline,
        "blocked": {
            "invalid_citation": blocked_invalid,
            "citation_gate": blocked_gate,
        },
        # Per-question composer failures (parse/citation exhaustion or a
        # fail-closed truncation) — top-level so it is NOT mistaken for a
        # headline milestone metric. A composer_exhausted gold question is
        # NOT-answered (in the answer_rate denominator, out of answered_count)
        # and is NOT a refusal; the probe count is carried under
        # headline.refusal.composer_exhausted.
        "composer_exhausted": composer_exhausted_count,
        "generated_at": _utcnow_iso(),
    }

    review_sample = _build_review_sample(course_slug, per_question_detail)

    if write:
        eval_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
        eval_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = (
            Path(output_path)
            if output_path is not None
            else eval_dir / f"grounded_answer_eval_{ts}.json"
        )
        out.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        review_out = (
            Path(review_sample_path)
            if review_sample_path is not None
            else eval_dir / REVIEW_SAMPLE_FILENAME
        )
        review_out.write_text(
            json.dumps(review_sample, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        report["_written"] = {
            "report_path": str(out),
            "review_sample_path": str(review_out),
        }

    report["_review_sample"] = review_sample
    return report


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# CLI (__main__)
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lib.retrieval.grounded_eval",
        description=(
            "Run the grounded-answer eval over a course's gold set + refusal "
            "probes (groundedness + citation precision). Writes a timestamped "
            "retrieval_eval/grounded_answer_eval_<ts>.json + review sample."
        ),
    )
    parser.add_argument("--course", required=True, help="course slug")
    parser.add_argument(
        "--engine",
        default="semantic",
        help="retrieval engine (lexical | semantic | hybrid-rrf)",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--no-groundedness",
        action="store_true",
        help="skip the per-answer NLI groundedness pass",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="repo root (default: auto-detect from this file)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    args = _build_arg_parser().parse_args(argv)
    repo_root = (
        Path(args.repo_root)
        if args.repo_root
        else Path(__file__).resolve().parents[2]
    )
    try:
        report = run_grounded_eval(
            repo_root,
            args.course,
            engine=args.engine,
            limit=args.limit,
            with_groundedness=not args.no_groundedness,
            write=True,
        )
    except PipelineUnavailable as exc:
        print(f"grounded-answer pipeline unavailable: {exc}", file=sys.stderr)
        return 3
    except RuntimeError as exc:
        print(f"grounded eval refused: {exc}", file=sys.stderr)
        return 2
    written = report.get("_written", {})
    print(
        json.dumps(
            {
                "course_slug": report["course_slug"],
                "engine": report["engine"],
                "headline": report["headline"],
                "blocked": report["blocked"],
                "written": written,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "run_grounded_eval",
    "PipelineUnavailable",
    "EVAL_SCHEMA_VERSION",
    "MILESTONE_TARGETS",
    "MILESTONE_CEILINGS",
    "MILESTONE_TARGETS_PINNED_AT",
    "KEY_POINT_COVERAGE_DIAGNOSTIC_BASELINE",
]
