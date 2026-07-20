"""Grounded-answer eval harness — runs the grounded-answer pipeline over a
course's gold set + refusal probes and rolls up groundedness + citation-precision
metrics into a ``retrieval_eval/grounded_answer_eval_<ts>.json`` report plus a
deterministic, seeded human-rubric review sample::

    python -m lib.retrieval.grounded_eval --course <slug> --engine lexical

CI never runs the real-model eval — the report-shape tests run this harness on
the mini-course fixture with a ``FakeAnswerClient`` + a deterministic fake NLI.

Pipeline dependency: this harness drives
``lib.retrieval.grounded_answer.answer_course_question``. That module is
import-guarded — when it is not importable, :func:`run_grounded_eval` raises
:class:`PipelineUnavailable` (a ``NotImplementedError`` subclass) naming the
missing dependency. No fake pipeline results are ever fabricated.

Filenames written here are disjoint from the other eval artifacts
(``gold_set.json``, ``benchmark_*.json``, ``refusal_probes.json``,
``refusal_calibration.json``): this harness owns
``grounded_answer_eval_*.json`` and ``groundedness_review_sample.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
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
    question_expected_behavior,
    question_phrasing,
    question_prior_turns,
)
from lib.retrieval.grounded_eval_library import run_library_eval
from lib.retrieval.grounded_eval_risk import selective_qa_view

#: Report schema for ``grounded_answer_eval_*.json``. The schema has only ever
#: grown ADDITIVELY: every key from an earlier version is still emitted, so a
#: consumer pinned to an older version keeps working. Beyond the core
#: per-question rows and headline metrics, the report carries:
#:   * ``headline.refusal`` — LEGACY refusal recall/precision over the probe set,
#:     plus groundedness scored on ANSWERED probes (probes the pipeline failed to
#:     refuse) as an unsupported-vs-corpus signal; per-probe rows live under the
#:     top-level ``probe_results``.
#:   * ``headline.abstention`` — the current two-axis scorer, classifying each
#:     item's observed outcome {answered, premise_corrected, refused, blocked,
#:     composer_exhausted} against its expected axis {answerable, false_premise,
#:     unanswerable}, with an outcome matrix, ``premise_correction_rate``, and
#:     ``corrected_refusal`` recall/precision. ``premise_corrected`` on a
#:     false-premise item that ANSWERED is credited by a local-LLM judge only
#:     when the answer actually flags the false premise.
#:   * ``headline.citation_precision`` — supported-over-CITED PER-QUESTION macro
#:     average (mean over questions emitting >=1 citation). The older POOLED
#:     number is retained as ``headline.citation_precision_legacy`` for
#:     cross-run comparability. ``headline.citation_precision_preadd`` is the
#:     same metric over the citation set BEFORE the NLI-ADD pass appended
#:     anything (equal to ``citation_precision`` whenever NLI-ADD is off/shadow).
#:     ``headline.citation_recall`` is cited-over-required, macro-averaged over
#:     answered questions pinning >=1 relevant passage — single-passage questions
#:     contribute 1.0-or-0.0, which is stated honestly in the field docs.
#:   * ``headline.groundedness_rate_micro`` — the claim-weighted micro average
#:     (entailed/scored), beside the macro mean.
#:   * ``headline.groundedness_breakdown`` — macro + micro groundedness sliced by
#:     difficulty stratum and by question_type, over answered questions with an
#:     available groundedness report.
#:   * ``headline.computational_numeric_check`` — roll-up of the per-answer
#:     numeric-grounding diagnostic; all-zero when
#:     ``ED4ALL_GROUNDEDNESS_COMPUTATIONAL`` is off.
#:   * ``headline.phrasing_breakdown`` — per-learner-phrasing answerability slice
#:     (canonical / colloquial / malformed) so an operator can see whether bare
#:     and malformed-but-reasonable learner questions pass, not only well-formed
#:     textbook phrasings.
#:   * ``headline.multiturn`` — reachability diagnostic counting gold questions
#:     that carried ``prior_turns`` and therefore actually drove the
#:     ``ED4ALL_ANSWER_MULTITURN`` retrieval-rewrite seam. All-zero on a
#:     single-turn gold set.
#:   * 95% Wilson score intervals (``*_ci``) on every rate in
#:     ``phrasing_breakdown``, the abstention block, and
#:     ``headline.refusal.by_category``; any bucket with n<30 is stamped
#:     ``"basis": "diagnostic"``.
#:   * top-level ``flag_config`` — the resolved ``ED4ALL_ANSWER_*`` + eval-judge
#:     env config, so a stored report is attributable to the config that made it.
#:   * top-level ``risk_coverage`` — selective-QA view (AURC, abstention AUROC,
#:     ECE, coverage-vs-accuracy curve) computed as PURE post-processing over the
#:     groundedness scores already in the report (no new model calls);
#:     ``basis == "none"`` when nothing was scored.
#:   * top-level ``library_wide`` — present ONLY when a caller passes a
#:     ``library_questions`` slice; drives ``answer_library_question`` over a
#:     cross-course question set and is skipped when the library holds a single
#:     course.
#: The breakdown/diagnostic blocks above are diagnostics, NOT pinned milestones.
EVAL_SCHEMA_VERSION = "1.8"
RETRIEVAL_EVAL_SUBDIR = "retrieval_eval"

#: Progress-writer arm name for the grounded eval (matches eval_arms.ARM_GROUNDED
#: without importing it — eval_arms imports from THIS module, so importing back
#: would be circular). Kept in sync by the test suite.
_PROGRESS_ARM_GROUNDED = "grounded"


def _progress_call(progress: Optional[Any], method: str, *args: Any, **kwargs: Any) -> None:
    """Invoke ``progress.<method>(...)`` best-effort (``None`` ⇒ no-op).

    Mirrors :func:`lib.retrieval.eval_arms._progress_call`: a ``None`` writer is
    the default (no file written, legacy behavior); any error is swallowed so
    progress instrumentation can NEVER raise into the grounded eval.
    """
    if progress is None:
        return
    try:
        getattr(progress, method)(*args, **kwargs)
    except Exception:  # noqa: BLE001 — progress must never abort the eval
        pass
REVIEW_SAMPLE_FILENAME = "groundedness_review_sample.json"

#: Review-sample artifact schema version. Tracked independently of
#: ``EVAL_SCHEMA_VERSION`` (the report-level schema): bumped 3 → 4 with the
#: groundedness scorer-v2 surface (per-claim ``windowed`` / ``best_chunk_cited``
#: + computational/filtered diagnostics now appear in the embedded verdicts).
REVIEW_SAMPLE_SCHEMA_VERSION = "4"

# --------------------------------------------------------------------------- #
# Milestone targets — measure-then-pin (floors CURRENT evidence meets)
# --------------------------------------------------------------------------- #

#: Date the targets below were pinned. Consumed by the staleness test, which
#: nags when the pins age relative to the newest committed eval report.
MILESTONE_TARGETS_PINNED_AT = "2026-06-12"

#: COMPARABILITY BOUNDARY. A target may only be compared against runs sharing
#: its basis. Two things break comparability and have both happened here:
#:   (1) GOLD-SET SHAPE. Each drafted gold question pins exactly ONE relevant
#:       passage, so citation_precision moves on gold DENOMINATOR semantics — a
#:       multi-citation answer mathematically deflates per-question precision —
#:       not on any pipeline change. A richer multi-passage gold raises it.
#:       Refusal metrics likewise move with the probe basis (probe count and how
#:       many are near-miss vs pure off-topic).
#:   (2) SCORER VERSION. Groundedness scorer v2 (artifact filter + computational
#:       exemption + windowed rescue + gate-eligible evidence pool) admits a
#:       wider evidence pool than v1 and raises the rate by construction, so v1
#:       and v2 groundedness/unsupported numbers are NOT comparable.
#:
#: SINGLE-COURSE CAVEAT: with one calibrated course the MIN-pin convention
#: degenerates to that course's measurement, so every floor below carries a
#: single-corpus caveat until a second different-family course joins. NEVER
#: tighten a pin from fewer courses than the prior pin established — re-pin only
#: after a measured run on a different course clears the stricter bound.
MILESTONE_TARGETS: Dict[str, float] = {
    # FLOOR. Non-answers that remain are a mix of false refusal,
    # composer_exhausted, and citation-gate fail-closed blocks — all real
    # per-question outcomes, not harness noise. Single-corpus caveat applies.
    "answer_rate": 0.95,
    # FLOOR == measurement: every emitted citation has resolved on every run.
    # Pinned at 1.0 exactly, so any dip is a real anchoring break, not noise.
    "citation_resolution_rate": 1.0,
    # FLOOR. Kept well under the measured value because the metric is dominated
    # by gold DENOMINATOR semantics (one relevant passage pinned per question),
    # so it moves whenever the gold set's shape changes without any pipeline
    # change. A richer multi-passage gold would raise it.
    "citation_precision": 0.30,
    # FLOOR. Deliberately far below the scorer-v2 measurement: a single course
    # does not establish a credible cross-corpus bound, so this stays at the
    # older cross-course floor until a second different-family course is
    # measured. (v1 and v2 groundedness numbers are not comparable.)
    "groundedness_rate_mean": 0.15,
    # CEILING. Retuned on the scorer-v2 basis, which removes the v1 scorer's
    # false negatives on glossary-style multi-topic chunks, novel-but-correct
    # computations, and claim-splitter artifacts — so v1 and v2 values are NOT
    # comparable and the never-tighten-from-one-course rule does not bind a
    # basis-invalidated pin. The residual is dominated by question-induced
    # evaluative synthesis (gold questions asking "which is better?"), which is
    # honestly unsupported rather than fabricated. Single-corpus caveat applies.
    "unsupported_claim_rate": 0.05,
    # FLOOR (refusal_recall). Not comparable across probe bases (probe count +
    # near-miss vs pure off-topic mix). The remaining recall gap is near-miss /
    # adjacent-domain probes inside the score-overlap band — MODEL-policy-owned,
    # not a retrieval-threshold matter — see lib/retrieval/refusal.py
    # PINNED_POLICIES hybrid-rrf overlap block.
    "refusal_precision_floor_note": 0.0,  # placeholder anchoring the note above; popped below
    # FLOOR.
    "refusal_recall": 0.55,
    # FLOOR.
    "refusal_precision": 0.90,
}
# Remove the inline placeholder key used only to anchor the refusal_recall note.
MILESTONE_TARGETS.pop("refusal_precision_floor_note", None)

#: DIAGNOSTIC (unpinned) key_point_coverage baseline. Deliberately NOT in
#: MILESTONE_TARGETS until the metric is stable across a second
#: different-family course; the staleness test does NOT gate on it.
KEY_POINT_COVERAGE_DIAGNOSTIC_BASELINE = 0.6420

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
    """The grounded-answer pipeline is not importable.

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
    """Import-guard the ``answer_course_question`` dependency.

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


def _part_chunk_bodies(
    parts: List[Dict[str, Any]], chunks_by_id: Dict[str, Dict[str, Any]]
) -> Dict[str, List[str]]:
    """Map each part_id → the bodies of its ``relevant_passage_refs`` chunks.

    This is the answer-content signal for ``score_part_coverage``: a covered
    part's ``part_text`` is only the sub-question prompt, so "did the answer
    address this part" is decided against the chunk bodies that answer it
    (reusing the claim-support machinery). Missing/unknown chunk ids are
    skipped — a part with no resolvable refs falls back to part_text."""
    out: Dict[str, List[str]] = {}
    for part in parts:
        pid = str(part.get("part_id", ""))
        refs = part.get("relevant_passage_refs")
        if not isinstance(refs, list):
            continue
        bodies: List[str] = []
        for ref in refs:
            chunk = chunks_by_id.get(str(ref))
            body = str((chunk or {}).get("text", "")).strip()
            if body:
                bodies.append(body)
        if bodies:
            out[pid] = bodies
    return out


def _load_probes(probes_path: Path) -> List[Dict[str, Any]]:
    if not probes_path.exists() or not probes_path.is_file():
        return []
    try:
        doc = json.loads(probes_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    probes = doc.get("probes")
    return list(probes) if isinstance(probes, list) else []


#: The valid refusal-probe expected_outcome enum + its default (refusal-probes
#: schema v1.1, additive optional field). Absent / unrecognized → ``refuse`` (the
#: outcome every legacy probe expects). ``correct_premise`` marks an ill_posed
#: probe whose false premise a correct pipeline must correct rather than answer
#: or refuse blankly.
_PROBE_EXPECTED_OUTCOME_VALUES = ("refuse", "correct_premise")
_DEFAULT_PROBE_EXPECTED_OUTCOME = "refuse"


def probe_expected_outcome(probe: Dict[str, Any]) -> str:
    """Return a refusal probe's expected_outcome, defaulting to ``refuse``.

    A probe with no (or an unrecognized) ``expected_outcome`` field reads as
    ``refuse`` — the outcome every legacy probe expects (back-compat: existing
    probe sets carry no such field). ``correct_premise`` is the ill_posed
    variant the eval can score separately."""
    val = probe.get("expected_outcome") if isinstance(probe, dict) else None
    return val if val in _PROBE_EXPECTED_OUTCOME_VALUES else _DEFAULT_PROBE_EXPECTED_OUTCOME


# --------------------------------------------------------------------------- #
# Two-axis abstention scorer + premise-correction judge (schema 1.7)
# --------------------------------------------------------------------------- #

#: Env overrides for the premise-correction judge backend. Both default to the
#: answer backend's OWN resolution (``ED4ALL_ANSWER_PROVIDER`` / ``_MODEL`` →
#: registry), so a run that pins neither routes the judge through the same
#: loopback-only local seat the answer path uses. Loopback enforcement lives in
#: ``answer_backend.resolve_answer_backend`` (a non-loopback resolved base_url
#: raises); the judge inherits it for free.
ENV_EVAL_JUDGE_PROVIDER = "ED4ALL_EVAL_JUDGE_PROVIDER"
ENV_EVAL_JUDGE_MODEL = "ED4ALL_EVAL_JUDGE_MODEL"

#: Expected-axis universe: the correct-handling class an item belongs to.
_AXIS_ANSWERABLE = "answerable"
_AXIS_FALSE_PREMISE = "false_premise"
_AXIS_UNANSWERABLE = "unanswerable"
_EXPECTED_AXES = (_AXIS_ANSWERABLE, _AXIS_FALSE_PREMISE, _AXIS_UNANSWERABLE)

#: Observed-outcome universe.
_OUTCOME_ANSWERED = "answered"
_OUTCOME_PREMISE_CORRECTED = "premise_corrected"
_OUTCOME_REFUSED = "refused"
_OUTCOME_BLOCKED = "blocked"
_OUTCOME_COMPOSER_EXHAUSTED = "composer_exhausted"
_OBSERVED_OUTCOMES = (
    _OUTCOME_ANSWERED,
    _OUTCOME_PREMISE_CORRECTED,
    _OUTCOME_REFUSED,
    _OUTCOME_BLOCKED,
    _OUTCOME_COMPOSER_EXHAUSTED,
)


@dataclass(frozen=True)
class _JudgeVerdict:
    """Result of the false-premise-correction judge on one answered item."""

    corrected: bool
    confidence: Optional[float]
    raw: str


def _gold_expected_axis(q: Dict[str, Any]) -> str:
    """Classify a gold question onto the expected-abstention axis.

    ``expected_behavior == "correct_premise"`` (v1.2) → false_premise; an
    explicitly unanswerable / refusal-flagged question → unanswerable; else
    answerable (gold sets are answerable by construction, so the default is
    answerable and a legacy behavior-less gold set slices entirely there)."""
    if question_expected_behavior(q) == "correct_premise":
        return _AXIS_FALSE_PREMISE
    if q.get("unanswerable") is True or q.get("is_refusal") is True:
        return _AXIS_UNANSWERABLE
    return _AXIS_ANSWERABLE


_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluation judge. You are given a learner QUESTION that "
    "is believed to contain a FALSE PREMISE (an assumption the course material "
    "does not support, or contradicts) and a tutor's ANSWER. Decide ONE thing: "
    "did the answer FLAG or CORRECT the false premise instead of going along "
    "with it? Think step by step using the rubric, then emit a single JSON "
    "object.\n"
    "Rubric:\n"
    "1. Identify the false premise embedded in the question.\n"
    "2. Check whether the answer explicitly disputes, corrects, or flags that "
    "premise (e.g. 'that is not accurate', 'the course actually says ...').\n"
    "3. An answer that silently accepts the premise and answers as if it were "
    "true does NOT correct it.\n"
    'Return JSON only: {"reasoning": "<one sentence>", '
    '"flags_false_premise": true|false, "confidence": <0.0-1.0>}.'
)


def _build_judge_messages(
    question_text: str, answer_text: str, cited_texts: List[str]
) -> List[Dict[str, str]]:
    context = "\n".join(f"- {t}" for t in cited_texts if t) or "(no citations)"
    user = (
        f"QUESTION (believed to contain a false premise):\n{question_text}\n\n"
        f"ANSWER:\n{answer_text}\n\n"
        f"CITED COURSE PASSAGES:\n{context}\n\n"
        "Did the answer flag or correct the false premise? Respond with the "
        "JSON object only."
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _parse_judge_response(raw: Any) -> Optional[Tuple[bool, Optional[float]]]:
    """Lenient parse of the judge JSON envelope.

    Returns ``(corrected, confidence)`` or ``None`` when nothing parseable was
    returned (the caller then degrades the outcome to 'answered')."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    flag = obj.get("flags_false_premise")
    if isinstance(flag, str):
        flag = flag.strip().lower() in ("true", "yes", "1")
    corrected = bool(flag)
    conf: Optional[float]
    try:
        conf = float(obj.get("confidence"))
    except (TypeError, ValueError):
        conf = None
    if conf is not None:
        conf = max(0.0, min(1.0, conf))
    return corrected, conf


def _emit_judge_decision(
    capture: Any,
    candidate: Dict[str, Any],
    verdict: Optional[_JudgeVerdict],
    error: Optional[str],
) -> None:
    """Emit the per-call judge decision-capture event (dynamic rationale).

    This is the NEW LLM call site's mandatory capture (root CLAUDE.md § LLM
    call-site instrumentation). Best-effort: a capture failure never aborts the
    eval."""
    item_id = candidate.get("item_id", "")
    qprefix = (candidate.get("question_text") or "")[:80]
    if verdict is None:
        outcome = "degraded_to_answered"
        conf_s = "n/a"
        corrected_s = "unknown"
    else:
        outcome = (
            "premise_corrected" if verdict.corrected else "answered_uncorrected"
        )
        conf_s = (
            f"{verdict.confidence:.2f}"
            if verdict.confidence is not None
            else "n/a"
        )
        corrected_s = str(verdict.corrected)
    rationale = (
        f"False-premise correction judge on {candidate.get('kind', 'item')} "
        f"{item_id!r} (q='{qprefix}'): flags_false_premise={corrected_s}, "
        f"confidence={conf_s}, outcome={outcome}"
        + (f"; judge_error={error}" if error else "")
    )
    try:
        capture.log_decision(
            decision_type="fresh_eval_invocation",
            decision=outcome,
            rationale=rationale,
            context=str(candidate.get("kind") or ""),
        )
    except Exception:  # noqa: BLE001 — decision capture must never abort the eval
        pass


def _judge_premise_correction(
    judge_client: Any, candidate: Dict[str, Any], capture: Optional[Any]
) -> Optional[_JudgeVerdict]:
    """Run the GEval-style false-premise-correction judge on ONE candidate.

    Returns a :class:`_JudgeVerdict`, or ``None`` on ANY failure/timeout/parse
    failure — the caller degrades that candidate's outcome to 'answered' with a
    warning (the judge NEVER crashes the eval). Emits one decision-capture event
    per call when a ``capture`` handle is threaded."""
    messages = _build_judge_messages(
        candidate["question_text"],
        candidate.get("answer_text") or "",
        candidate.get("cited_texts") or [],
    )
    raw: Any = ""
    parsed: Optional[Tuple[bool, Optional[float]]] = None
    error: Optional[str] = None
    try:
        raw = judge_client.chat_completion(
            messages, max_tokens=256, temperature=0.0
        )
        parsed = _parse_judge_response(raw)
    except Exception as exc:  # noqa: BLE001 — judge failure degrades gracefully
        error = f"{type(exc).__name__}: {exc}"
    verdict: Optional[_JudgeVerdict]
    if parsed is None:
        verdict = None
    else:
        corrected, conf = parsed
        verdict = _JudgeVerdict(
            corrected=corrected, confidence=conf, raw=str(raw)
        )
    if capture is not None:
        _emit_judge_decision(capture, candidate, verdict, error)
    return verdict


def _resolve_judge_client(capture: Optional[Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """Build the loopback-only judge client via the answer-backend registry.

    ``ED4ALL_EVAL_JUDGE_PROVIDER`` / ``_MODEL`` win when set, else the answer
    backend's own resolution (``ED4ALL_ANSWER_*`` → registry). Returns
    ``(client, resolved)`` or ``(None, None)`` on any resolution/build failure
    (graceful degrade — every candidate then falls back to 'answered')."""
    try:
        from lib.retrieval.answer_backend import (
            build_answer_client,
            resolve_answer_backend,
        )

        provider = os.environ.get(ENV_EVAL_JUDGE_PROVIDER) or None
        model = os.environ.get(ENV_EVAL_JUDGE_MODEL) or None
        resolved = resolve_answer_backend(provider_name=provider, model_id=model)
        client = build_answer_client(resolved, capture=capture, json_mode=True)
        return client, resolved
    except Exception:  # noqa: BLE001 — judge unavailable → degrade, never crash
        return None, None


# --------------------------------------------------------------------------- #
# Metric arithmetic
# --------------------------------------------------------------------------- #

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Dict[str, Any]:
    """95% Wilson score interval for a binomial proportion.

    ``n < 30`` → ``"basis": "diagnostic"`` (too small to trust the point
    estimate), else ``"sufficient"``. An empty basis (``n <= 0``) returns
    ``lo``/``hi`` ``None`` — never a fabricated 0.0 interval.
    """
    if n <= 0:
        return {"lo": None, "hi": None, "n": 0, "basis": "diagnostic"}
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)) / denom
    return {
        "lo": max(0.0, center - margin),
        "hi": min(1.0, center + margin),
        "n": n,
        "basis": "diagnostic" if n < 30 else "sufficient",
    }


def _abstention_block(
    records: List[Dict[str, Any]], judge_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Two-axis abstention scorer (schema 1.7).

    ``records`` = one row per gold question AND per probe:
    ``{"expected_axis", "outcome"}``. Builds the expected-axis × observed-outcome
    matrix, the premise-correction rate over false-premise items, and the
    corrected-refusal recall/precision that credits BOTH a refusal and a
    premise-correction as an acceptable decline. Every rate carries a Wilson CI.
    Additive diagnostic — NOT a pinned milestone.
    """
    matrix = {
        axis: {o: 0 for o in _OBSERVED_OUTCOMES} for axis in _EXPECTED_AXES
    }
    axis_totals = {axis: 0 for axis in _EXPECTED_AXES}
    for r in records:
        axis = r.get("expected_axis")
        outcome = r.get("outcome")
        if axis not in matrix:
            continue
        axis_totals[axis] += 1
        if outcome in matrix[axis]:
            matrix[axis][outcome] += 1

    fp = matrix[_AXIS_FALSE_PREMISE]
    un = matrix[_AXIS_UNANSWERABLE]
    ans = matrix[_AXIS_ANSWERABLE]
    fp_total = axis_totals[_AXIS_FALSE_PREMISE]
    corrected = fp[_OUTCOME_PREMISE_CORRECTED]
    premise_rate = (corrected / fp_total) if fp_total else None

    # Corrected-refusal: the "should-not-answer-blindly" universe is every
    # false-premise + unanswerable item. A false-premise item is handled well by
    # EITHER a refusal OR a premise-correction; an unanswerable item only by a
    # refusal.
    refusable = fp_total + axis_totals[_AXIS_UNANSWERABLE]
    recall_num = (
        un[_OUTCOME_REFUSED]
        + fp[_OUTCOME_REFUSED]
        + fp[_OUTCOME_PREMISE_CORRECTED]
    )
    recall = (recall_num / refusable) if refusable else None
    # All declines = every refusal (any axis) + every false-premise correction.
    all_refusals = (
        ans[_OUTCOME_REFUSED] + fp[_OUTCOME_REFUSED] + un[_OUTCOME_REFUSED]
    )
    all_declines = all_refusals + corrected
    # Correct declines = refusals on should-not-answer items + corrections
    # (a refusal on an answerable gold item is a WRONG decline / false refusal).
    correct_declines = (
        fp[_OUTCOME_REFUSED] + un[_OUTCOME_REFUSED] + corrected
    )
    precision = (correct_declines / all_declines) if all_declines else None

    return {
        "outcome_matrix": matrix,
        "axis_totals": axis_totals,
        "premise_correction": {
            "false_premise_items": fp_total,
            "corrected": corrected,
            "refused": fp[_OUTCOME_REFUSED],
            "answered_uncorrected": fp[_OUTCOME_ANSWERED],
            "rate": premise_rate,
            "rate_ci": _wilson_ci(corrected, fp_total),
        },
        "corrected_refusal": {
            "refusable_items": refusable,
            "recall": recall,
            "recall_ci": _wilson_ci(recall_num, refusable),
            "precision": precision,
            "precision_ci": _wilson_ci(correct_declines, all_declines),
            "declines": all_declines,
            "correct_declines": correct_declines,
        },
        "judge": judge_meta,
        "_diagnostic": (
            "Two-axis abstention scorer (schema 1.7): expected axis "
            "{answerable, false_premise, unanswerable} x observed outcome "
            "{answered, premise_corrected, refused, blocked, composer_exhausted}."
            " On false-premise items premise_corrected=SUCCESS, refused="
            "acceptable, answered-without-correction=FAILURE. premise_corrected "
            "is detected by a local-LLM judge (FreshQA rule: credited only when "
            "the answer flags the false premise). See headline.refusal for the "
            "legacy single-axis refusal metrics."
        ),
    }


def _probe_category_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-category probe refused-rate + Wilson CI over ``{category, refused}``
    rows. Additive diagnostic; each bucket n<30 stamps ``basis=diagnostic``."""
    by_cat: Dict[str, Dict[str, int]] = {}
    for r in records:
        cat = str(r.get("category") or "")
        b = by_cat.setdefault(cat, {"n": 0, "refused": 0})
        b["n"] += 1
        if r.get("refused"):
            b["refused"] += 1
    out: Dict[str, Any] = {}
    for cat in sorted(by_cat):
        b = by_cat[cat]
        out[cat] = {
            "n": b["n"],
            "refused": b["refused"],
            "refused_rate": (b["refused"] / b["n"]) if b["n"] else None,
            "refused_rate_ci": _wilson_ci(b["refused"], b["n"]),
        }
    return out


#: Documented answer-path env names recorded in the flag-config stamp. Kept in
#: sync with the ``ED4ALL_ANSWER_*`` rows in root CLAUDE.md's cross-cutting flag
#: table so a stored eval report is attributable to its exact answer-path config.
_DOCUMENTED_ANSWER_ENV = (
    "ED4ALL_ANSWER_PROVIDER",
    "ED4ALL_ANSWER_MODEL",
    "ED4ALL_ANSWER_TIMEOUT_SECONDS",
    "ED4ALL_ANSWER_NUM_CTX",
    "ED4ALL_ANSWER_CITATION_PRUNE",
    "ED4ALL_ANSWER_ASSESSMENT_GUARD",
    "ED4ALL_ANSWER_ASSESSMENT_GUARD_THRESHOLD",
    "ED4ALL_ANSWER_PRUNE_MIN_OVERLAP",
    "ED4ALL_ANSWER_ADD_MIN_SHINGLE",
    "ED4ALL_ANSWER_NLI_ADD",
    "ED4ALL_ANSWER_COMPLETENESS_RECHECK",
    "ED4ALL_ANSWER_LIBRARY_WIDE",
    "ED4ALL_ANSWER_INTENT_ROUTE",
    "ED4ALL_ANSWER_MULTITURN",
    "ED4ALL_ANSWER_DECOMPOSE",
    "ED4ALL_ANSWER_HYDE",
    "ED4ALL_ANSWER_GRAPH_EXPAND",
    "ED4ALL_ANSWER_GRAPH_EXPAND_MAX",
    "ED4ALL_ANSWER_COMPLETENESS_RERETRIEVE",
    "ED4ALL_ANSWER_HEDGE_TIER",
    "ED4ALL_ANSWER_HEDGE_MARGIN",
    "ED4ALL_ANSWER_RERANK_PROVIDER",
    "ED4ALL_ANSWER_HYDE_MODEL",
    "ED4ALL_ANSWER_NLI_ADD_MODEL",
)


def _flag_config_stamp() -> Dict[str, Any]:
    """Attribution stamp: the resolved ``ED4ALL_ANSWER_*`` + eval-judge env
    config for this run.

    Records the documented answer-path env names (value ``None`` when unset)
    PLUS any other ``ED4ALL_ANSWER_*`` key present in the environment, and the
    two eval-judge overrides — so a stored eval report is attributable to the
    exact answer-path flag config it ran under (a re-pin or A/B comparison can
    read which flags were on)."""
    answer_env: Dict[str, Optional[str]] = {
        name: os.environ.get(name) for name in _DOCUMENTED_ANSWER_ENV
    }
    for key, val in os.environ.items():
        if key.startswith("ED4ALL_ANSWER_") and key not in answer_env:
            answer_env[key] = val
    return {
        "answer_env": answer_env,
        "eval_judge": {
            ENV_EVAL_JUDGE_PROVIDER: os.environ.get(ENV_EVAL_JUDGE_PROVIDER),
            ENV_EVAL_JUDGE_MODEL: os.environ.get(ENV_EVAL_JUDGE_MODEL),
        },
    }

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


def _breakdown_agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one slice of per-question groundedness records.

    ``rows`` = ``[{"g_rate": float, "scored": int, "entailed": int}, ...]``.
    Returns the MACRO (unweighted per-question mean of ``groundedness_rate``,
    matching ``headline.groundedness_rate_mean`` semantics) AND the MICRO
    (claim-weighted: total entailed / total scored) groundedness for the slice.
    Both are ``None`` when the slice has no basis (no questions / no scored
    claims) — never a fabricated 0.0.
    """
    n = len(rows)
    scored = sum(r["scored"] for r in rows)
    entailed = sum(r["entailed"] for r in rows)
    return {
        "questions": n,
        "macro_groundedness": (
            (sum(r["g_rate"] for r in rows) / n) if n else None
        ),
        "micro_groundedness": (entailed / scored) if scored else None,
        "scored_claims": scored,
        "entailed_claims": entailed,
    }


def _groundedness_breakdown(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Macro + per-stratum + per-question-type groundedness breakdown.

    ``records`` carries one row per ANSWERED gold question whose groundedness
    report was available: ``{"difficulty", "question_type", "g_rate", "scored",
    "entailed"}``. The overall block reports macro (per-question mean) AND micro
    (claim-weighted) groundedness; ``by_stratum`` slices by the question's
    difficulty band (missing → ``"unknown"``) and ``by_question_type`` by its
    declared type. Additive diagnostic — NOT a pinned milestone.
    """
    overall = _breakdown_agg(records)
    by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_stratum.setdefault(r["difficulty"], []).append(r)
        by_type.setdefault(r["question_type"], []).append(r)
    return {
        "questions_scored": overall["questions"],
        "macro_groundedness": overall["macro_groundedness"],
        "micro_groundedness": overall["micro_groundedness"],
        "scored_claims": overall["scored_claims"],
        "entailed_claims": overall["entailed_claims"],
        "by_stratum": {
            k: _breakdown_agg(v) for k, v in sorted(by_stratum.items())
        },
        "by_question_type": {
            k: _breakdown_agg(v) for k, v in sorted(by_type.items())
        },
        "_diagnostic": (
            "macro = unweighted per-question mean (matches "
            "groundedness_rate_mean); micro = claim-weighted (entailed/scored). "
            "Diagnostic breakdown, NOT a pinned milestone."
        ),
    }


def _phrasing_breakdown(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-learner-phrasing answerability breakdown (GUI-phrasing axis).

    ``records`` carries one row per gold question: ``{"phrasing", "status"}``
    (phrasing is the v1.1 ``phrasing`` field resolved via
    ``gold_set.question_phrasing`` — a phrasing-less question reads as
    ``canonical``). For each observed phrasing value the block reports:
      * ``n`` — questions with that phrasing,
      * ``answered_count`` / ``answered_rate`` — status in the answered set,
      * ``blocked_count`` — citation-gate / invalid-citation blocks,
      * ``refused_count`` — false refusals on an answerable gold question.
    ``answered_rate`` is ``None`` for an empty bucket (never a fabricated 0.0).
    Additive diagnostic — NOT a pinned milestone; the goal is a glance-level read
    of whether the bare/colloquial and malformed-but-reasonable learner
    questions pass, not only the well-formed textbook phrasings.
    """
    by_ph: Dict[str, List[str]] = {}
    for r in records:
        by_ph.setdefault(str(r.get("phrasing") or _DEFAULT_PHRASING_LABEL), []).append(
            str(r.get("status") or "")
        )
    out: Dict[str, Any] = {}
    for ph in sorted(by_ph):
        statuses = by_ph[ph]
        n = len(statuses)
        answered = sum(1 for s in statuses if s in _ANSWERED_STATUSES)
        refused = sum(1 for s in statuses if s in _REFUSED_STATUSES)
        blocked = sum(
            1 for s in statuses if s in (_BLOCKED_INVALID, _BLOCKED_GATE)
        )
        out[ph] = {
            "n": n,
            "answered_count": answered,
            "answered_rate": (answered / n) if n else None,
            # Schema 1.7: 95% Wilson CI on the answered_rate; n<30 buckets carry
            # basis="diagnostic" (a bare/malformed phrasing slice is usually
            # small — the CI keeps an operator from over-reading a point rate).
            "answered_rate_ci": _wilson_ci(answered, n),
            "blocked_count": blocked,
            "refused_count": refused,
        }
    return out


#: Label a phrasing-less gold question falls into (mirrors
#: ``gold_set._DEFAULT_PHRASING`` without importing the private name).
_DEFAULT_PHRASING_LABEL = "canonical"


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
        "schema_version": REVIEW_SAMPLE_SCHEMA_VERSION,
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
    judge_client: Optional[Any] = None,
    write: bool = True,
    progress: Optional[Any] = None,
    library_questions: Optional[Sequence[Dict[str, Any]]] = None,
    library_answer_fn: Optional[Any] = None,
    library_course_slugs: Optional[Sequence[str]] = None,
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
        # Measure-then-pin feed: record the gold set's identity in the
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
    # PRE-ADD citation precision (mirrors the prune arm's *_preprune pattern):
    # the emitted/relevant counts computed over the citation set BEFORE the
    # NLI-ADD pass appended any citation. When NLI-ADD is off/shadow (today) no
    # citation was added, so the pre-add set == the emitted set and
    # citation_precision_preadd == citation_precision exactly. When NLI-ADD is
    # ON, the pre-add headline stays comparable to today's precision while the
    # post-add precision can be read separately (an ADD against a one-passage
    # gold can only grow the denominator — see plans/nli-add-flip-2026-06.md §2).
    n_emitted_citations_preadd = 0
    n_relevant_citations_preadd = 0
    answered_count = 0
    blocked_invalid = 0
    blocked_gate = 0
    composer_exhausted_count = 0
    false_refusals_on_gold = 0
    groundedness_rates: List[float] = []
    unsupported_rates: List[float] = []
    # Scorer-v2 additive diagnostics (NOT pinned milestones): how many claims
    # the v2 scorer exempted as computational, dropped as structural artifacts,
    # and entailed via an UNCITED corpus chunk (best_chunk_cited is False).
    computational_claim_count = 0
    filtered_claim_count = 0
    entailed_uncited_count = 0
    # Numeric-grounding roll-up (all-zero on the default off path where no answer
    # carries a ``computational_numeric_check`` block): computational sentences
    # whose numeric literals were verified against the cited chunks, and how
    # many carried an ungrounded (fabricated-looking) numeric literal.
    comp_numeric_claims_checked = 0
    comp_numeric_ungrounded_claims = 0
    comp_numeric_ungrounded_total = 0
    # Per-question groundedness records (difficulty stratum + question_type
    # + macro/micro basis). One row per answered question whose groundedness
    # report was available.
    groundedness_breakdown_records: List[Dict[str, Any]] = []
    # Per-learner-phrasing answerability records (schema 1.6). One row per gold
    # question: its resolved phrasing (canonical/colloquial/malformed) + final
    # status. Recorded for EVERY question incl. composer-exhausted ones so the
    # per-phrasing n counts the whole gold set.
    phrasing_records: List[Dict[str, Any]] = []
    # Schema 1.8 multiturn REACHABILITY (E2): how many gold questions carried
    # prior_turns (v1.2) and therefore actually drove the ED4ALL_ANSWER_MULTITURN
    # retrieval-rewrite seam (answer_course_question(prior_turns=...)). All-zero on
    # a legacy / single-turn gold set. Also tallies how many of those still
    # answered (the multiturn path did not break answerability).
    multiturn_gold_count = 0
    multiturn_answered_count = 0
    multiturn_prior_turns_total = 0
    # Schema 1.8 risk-coverage records (E3): one {confidence, correct} row per
    # ANSWERED gold question whose groundedness report was available. confidence =
    # per-answer groundedness rate; correct = cited a gold-relevant chunk. Fed to
    # the pure-post-processing selective-QA view (no new model calls).
    risk_records: List[Dict[str, Any]] = []
    # Schema 1.7 two-axis abstention records: one row per gold question AND per
    # probe carrying its expected axis {answerable, false_premise, unanswerable}
    # + observed outcome. false-premise items that ANSWERED become judge
    # candidates (the outcome may be upgraded to premise_corrected post-loop).
    abstention_records: List[Dict[str, Any]] = []
    judge_candidates: List[Dict[str, Any]] = []
    # Per-category probe refused-rate records ({category, refused}) for the
    # Wilson-CI'd per-category probe stats.
    probe_category_records: List[Dict[str, Any]] = []
    # Schema 1.7 citation metrics (macro, per-question): supported-over-CITED
    # precision (mean over questions with >=1 emitted citation) + cited-over-
    # required recall (mean over answered questions pinning >=1 relevant passage;
    # single-passage questions contribute 1.0-or-0.0).
    citation_precision_macro_terms: List[float] = []
    citation_recall_terms: List[float] = []
    multi_passage_question_count = 0
    # NLI-ADD shadow diagnostics (NOT a pinned milestone): how many answers the
    # NLI-ADD arm WOULD have credited an uncited supporter on, the total would-add
    # count, and how many zero-citation answers it would have recovered. Rolls up
    # only over answers carrying an ``nli_citation_add`` block (the arm ran in
    # shadow/on mode); 0 on the default (off) path where the block is absent.
    nli_answers_with_would_adds = 0
    nli_total_would_adds = 0
    nli_zero_citation_answers_recovered = 0

    # P4 additive aggregates (key-point completeness, part coverage, per-
    # population citations). All roll up only over ANSWERED questions that
    # carry the matching authored v1.1 data; absent on v1.0 gold sets.
    # Per-population citation precision: tally emitted + relevant
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

    # Best-effort incremental progress: the grounded arm visits every gold
    # question PLUS every refusal probe. Refine the writer's planned total to
    # that sum up front (no-op when no writer was threaded).
    _progress_call(
        progress, "start_arm", _PROGRESS_ARM_GROUNDED, len(questions) + len(probes)
    )

    # --- answerable gold questions ---------------------------------------
    for q in questions:
        qid = str(q.get("question_id", ""))
        qtext = str(q.get("question_text", ""))
        phrasing = question_phrasing(q)
        expected_axis = _gold_expected_axis(q)
        all_rel, primary_rel = _relevant_chunk_ids(q)
        # E2 reachability: convert the gold question's v1.2 prior_turns
        # ({role, text} entries) into the flat text sequence
        # answer_course_question(prior_turns=...) consumes, so a multi-turn gold
        # item actually drives the ED4ALL_ANSWER_MULTITURN retrieval-rewrite seam
        # (byte-identical when off — the pipeline gates on multiturn_enabled()).
        prior_turn_texts = [t["text"] for t in question_prior_turns(q)]
        if prior_turn_texts:
            multiturn_gold_count += 1
            multiturn_prior_turns_total += len(prior_turn_texts)

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
                prior_turns=(prior_turn_texts or None),
            )
        except catchable as exc:
            # Per-question composer failure (parse/citation exhaustion, or a
            # fail-closed truncation). Record it, count it as NOT-answered for
            # answer_rate, do NOT count it as a refusal, and keep evaluating
            # the rest of the gold set. The eval artifact still writes.
            composer_exhausted_count += 1
            latencies.append(0.0)
            phrasing_records.append(
                {"phrasing": phrasing, "status": _COMPOSER_EXHAUSTED}
            )
            abstention_records.append(
                {
                    "item_id": qid,
                    "kind": "gold",
                    "expected_axis": expected_axis,
                    "outcome": _OUTCOME_COMPOSER_EXHAUSTED,
                }
            )
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
            _progress_call(
                progress,
                "record_item",
                _PROGRESS_ARM_GROUNDED,
                qid,
                _COMPOSER_EXHAUSTED,
            )
            continue

        status = str(_answer_field(answer, "status", "unknown"))
        phrasing_records.append({"phrasing": phrasing, "status": status})
        if prior_turn_texts and status in _ANSWERED_STATUSES:
            multiturn_answered_count += 1
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
        # Ids the NLI-ADD pass ACTUALLY appended to this answer (empty in
        # off/shadow — the only modes shipping today; the real add set under ON).
        # Used to recompute citation_precision over the PRE-ADD set so the
        # pre-add headline stays comparable to today's number after the flip.
        _nli_add_block = _answer_field(answer, "nli_citation_add", None)
        nli_added_ids = set()
        if isinstance(_nli_add_block, dict):
            nli_added_ids = {
                str(x) for x in (_nli_add_block.get("added_ids") or [])
            }
        resolved_here = 0
        relevant_here = 0
        relevant_primary_here = 0
        emitted_preadd_here = 0
        relevant_preadd_here = 0
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
            # PRE-ADD set: every emitted citation EXCEPT the ones the NLI-ADD
            # pass appended on this answer. Under off/shadow nli_added_ids is
            # empty, so the pre-add tally equals the full emitted tally.
            if cid not in nli_added_ids:
                emitted_preadd_here += 1
                if is_relevant:
                    relevant_preadd_here += 1
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
        n_emitted_citations_preadd += emitted_preadd_here
        n_relevant_citations_preadd += relevant_preadd_here

        if status in _ANSWERED_STATUSES:
            answered_count += 1
        elif status == _BLOCKED_INVALID:
            blocked_invalid += 1
        elif status == _BLOCKED_GATE:
            blocked_gate += 1
        elif status in _REFUSED_STATUSES:
            # A refusal on an answerable gold question is a false refusal.
            false_refusals_on_gold += 1

        # --- Schema 1.7 two-axis abstention record + macro citation metrics ---
        if status in _ANSWERED_STATUSES:
            base_outcome = _OUTCOME_ANSWERED
        elif status in _REFUSED_STATUSES:
            base_outcome = _OUTCOME_REFUSED
        elif status in (_BLOCKED_INVALID, _BLOCKED_GATE):
            base_outcome = _OUTCOME_BLOCKED
        else:
            base_outcome = _OUTCOME_ANSWERED
        abst_record = {
            "item_id": qid,
            "kind": "gold",
            "expected_axis": expected_axis,
            "outcome": base_outcome,
        }
        abstention_records.append(abst_record)
        # A false-premise gold item that ANSWERED is a judge candidate: only the
        # judge can tell an answer that CORRECTED the premise from one that went
        # along with it. Cheap — a handful per run.
        if expected_axis == _AXIS_FALSE_PREMISE and status in _ANSWERED_STATUSES:
            judge_candidates.append(
                {
                    "record": abst_record,
                    "item_id": qid,
                    "kind": "gold",
                    "question_text": qtext,
                    "answer_text": _answer_field(answer, "answer_text", None),
                    "cited_texts": [
                        c.get("text_quote") for c in cites if c.get("text_quote")
                    ],
                }
            )
        # Macro citation precision (supported-over-CITED per question) + recall
        # (cited-over-required per question). relevant_here already counted the
        # cited chunks in the gold-relevant set.
        if n_cites > 0:
            citation_precision_macro_terms.append(relevant_here / n_cites)
        if all_rel:
            if len(all_rel) > 1:
                multi_passage_question_count += 1
            if status in _ANSWERED_STATUSES:
                cited_set = {
                    str(c.get("chunk_id", "")) for c in cites if c.get("chunk_id")
                }
                citation_recall_terms.append(
                    len(cited_set & all_rel) / len(all_rel)
                )

        grounded = _answer_field(answer, "groundedness", None)
        g_rate: Optional[float] = None
        u_rate: Optional[float] = None
        per_claim: List[Dict[str, Any]] = []
        if isinstance(grounded, dict) and grounded.get("available"):
            scored = int(grounded.get("scored_count", 0) or 0)
            g_rate = float(grounded.get("groundedness_rate", 0.0) or 0.0)
            unsupported_n = int(grounded.get("unsupported_count", 0) or 0)
            contradicted_n = int(grounded.get("contradicted_count", 0) or 0)
            if scored:
                u_rate = unsupported_n / scored
            else:
                u_rate = 0.0
            groundedness_rates.append(g_rate)
            unsupported_rates.append(u_rate)
            per_claim = list(grounded.get("claims", []) or [])
            # Breakdown record: entailed = scored − unsupported − contradicted
            # (the three exhaustive scored verdicts; clamped ≥0 defensively).
            groundedness_breakdown_records.append(
                {
                    "difficulty": str(q.get("difficulty") or "unknown"),
                    "question_type": str(q.get("question_type") or "unknown"),
                    "g_rate": g_rate,
                    "scored": scored,
                    "entailed": max(0, scored - unsupported_n - contradicted_n),
                }
            )
            # Scorer-v2 additive diagnostics (absent / 0 on a v1 report).
            computational_claim_count += int(grounded.get("computational_count", 0) or 0)
            filtered_claim_count += int(grounded.get("filtered_count", 0) or 0)
            # Numeric-grounding roll-up (present only when the opt-in check
            # ran; absent → contributes 0 → byte-stable on the default path).
            cnc = grounded.get("computational_numeric_check")
            if isinstance(cnc, dict):
                comp_numeric_claims_checked += int(
                    cnc.get("computational_claims_checked", 0) or 0
                )
                comp_numeric_ungrounded_claims += int(
                    cnc.get("claims_with_ungrounded_numeric", 0) or 0
                )
                comp_numeric_ungrounded_total += int(
                    cnc.get("ungrounded_numeric_count", 0) or 0
                )
            for claim in per_claim:
                if (
                    claim.get("verdict") == "entailed"
                    and claim.get("best_chunk_cited") is False
                ):
                    entailed_uncited_count += 1

        # --- E3 risk-coverage record ---------------------------------------
        # An ANSWERED question with an available groundedness report contributes
        # one {confidence, correct} pair to the selective-QA view: confidence =
        # per-answer groundedness rate (the "knows-when-it-doesn't" signal);
        # correct = the answer cited a gold-relevant chunk (an answer-quality
        # label INDEPENDENT of the groundedness confidence, so AURC/AUROC are
        # meaningful, not self-referential). Skipped when no report was scored.
        if status in _ANSWERED_STATUSES and g_rate is not None:
            risk_records.append(
                {"confidence": g_rate, "correct": relevant_here > 0}
            )

        # --- NLI-ADD shadow diagnostics (additive; present only when the arm ran)
        nli_add = _answer_field(answer, "nli_citation_add", None)
        if isinstance(nli_add, dict):
            would = [str(x) for x in (nli_add.get("would_add_ids") or [])]
            if would:
                nli_answers_with_would_adds += 1
                nli_total_would_adds += len(would)
                # WARNING-class, not a recovery win: would-adds on a
                # zero-citation (prune-to-empty) answer have audited as
                # question-induced FALSE adds. ON mode never adds to these; this
                # count only surfaces how often the criterion fires where it is
                # known-suspect.
                if n_cites == 0 or nli_add.get("zero_citation_answer"):
                    nli_zero_citation_answers_recovered += 1

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
                q_parts = _question_parts(q)
                part_cov = score_part_coverage(
                    answer_text_val,
                    q_parts,
                    chunks_by_part=_part_chunk_bodies(q_parts, chunks_by_id),
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
            # Persist EVERY emitted citation's chunk_id on EVERY per-question
            # row (not only inside the sampled review subset), so
            # citation_precision is auditable across the whole gold set: each id
            # can be joined to its chunk body to judge whether an
            # unpinned-but-cited passage is a genuine supporter. page_label
            # rides along when the pipeline carried it.
            "cited_chunk_ids": cited_ids,
            "cited_pages": [
                c.get("page_label")
                for c in cites
                if c.get("chunk_id")
            ],
        }
        if kp_cov is not None:
            row["key_point_coverage"] = kp_cov.to_dict()
        if part_cov is not None:
            row["part_coverage"] = part_cov.to_dict()
        if pop_break is not None:
            row["citation_population"] = pop_break.to_dict()
        # Persist the NLI-ADD shadow block per question (present only when the
        # arm ran). Headline counts alone cannot be hand-audited post-hoc, and
        # the shadow→on flip decision is gated on auditing exactly these rows.
        if isinstance(nli_add, dict):
            row["nli_citation_add"] = nli_add
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
        # Best-effort progress: cheap, already-computed signals only. unsupported
        # is the per-question claim count the groundedness axis already produced
        # (0 when the axis was unavailable); key-point counts ride along.
        _unsupported = 0
        if isinstance(grounded, dict) and grounded.get("available"):
            _unsupported = int(grounded.get("unsupported_count", 0) or 0)
        _progress_call(
            progress,
            "record_item",
            _PROGRESS_ARM_GROUNDED,
            qid,
            status,
            answered=(status in _ANSWERED_STATUSES),
            key_points_total=(kp_cov.total if kp_cov is not None else 0),
            key_points_covered=(kp_cov.covered if kp_cov is not None else 0),
            unsupported=_unsupported,
        )

    # --- refusal probes ---------------------------------------------------
    # REFUSAL-SAFETY axis (schema 1.3, additive): a probe is a question that
    # SHOULD be refused. refusal_recall/precision (UNCHANGED below) report the
    # share refused; this loop ALSO asks — for every probe the pipeline FAILED
    # to refuse (answered) — was that answer FABRICATED? It scores groundedness
    # on answered probes (with_groundedness threads the eval's flag, IDENTICAL
    # machinery to the gold loop) so a near-miss probe answered WITH a real
    # corpus citation (failed-to-refuse but grounded) is told apart from pure
    # invention. Per-probe rows are persisted under the top-level probe_results
    # key. NO change to refusal_recall / refusal_precision / n_probes.
    n_probes = len(probes)
    probe_refused = 0
    probe_answered = 0
    probe_composer_exhausted = 0
    # Refusal-safety aggregates over ANSWERED probes only.
    probe_answered_unsupported_rates: List[float] = []  # per-probe mean basis
    probe_answered_unsupported_count = 0  # claim-level numerator
    probe_answered_claims_scored = 0  # claim-level denominator
    probe_results: List[Dict[str, Any]] = []
    for probe in probes:
        pid = str(probe.get("probe_id", ""))
        category = str(probe.get("category", ""))
        ptext = str(probe.get("question_text", ""))
        # Schema 1.7 expected axis: an ill_posed probe whose expected_outcome is
        # correct_premise is a false-premise item; every other probe expects a
        # refusal (unanswerable axis).
        probe_axis = (
            _AXIS_FALSE_PREMISE
            if probe_expected_outcome(probe) == "correct_premise"
            else _AXIS_UNANSWERABLE
        )
        try:
            answer = pipeline(
                repo_root,
                course_slug,
                ptext,
                engine=engine,
                limit=limit,
                client=client,
                refusal_policy=refusal_policy,
                # Score groundedness on a probe the pipeline answers (the
                # refusal-safety axis); honors the eval's with_groundedness so a
                # deterministic-only run still produces refusal_recall exactly as
                # before (answered-probe groundedness simply reads null).
                with_groundedness=with_groundedness,
                capture=capture,
                chunkset_kind=gold_chunkset_kind,
            )
        except catchable:
            # A composer exhaustion on a probe is neither a clean refusal nor
            # an answer — it never reached a verdict, so it must NOT inflate
            # refusal_recall. Counted separately, surfaced in the report.
            probe_composer_exhausted += 1
            abstention_records.append(
                {
                    "item_id": pid,
                    "kind": "probe",
                    "expected_axis": probe_axis,
                    "outcome": _OUTCOME_COMPOSER_EXHAUSTED,
                }
            )
            probe_category_records.append(
                {"category": category, "refused": False}
            )
            probe_results.append(
                {
                    "probe_id": pid,
                    "category": category,
                    "refused": False,
                    "answered": False,
                    "composer_exhausted": True,
                    "unsupported_rate": None,
                }
            )
            _progress_call(
                progress,
                "record_item",
                _PROGRESS_ARM_GROUNDED,
                pid,
                _COMPOSER_EXHAUSTED,
            )
            continue
        status = str(_answer_field(answer, "status", "unknown"))
        is_refused = status in _REFUSED_STATUSES
        is_answered = status in _ANSWERED_STATUSES
        if is_refused:
            probe_refused += 1
        elif is_answered:
            probe_answered += 1
        # blocked_* on a probe is neither a clean refusal nor an answer; it is
        # surfaced via the blocked counters only (not double-counted here).

        # --- Schema 1.7 two-axis abstention record for this probe ---
        if is_refused:
            probe_outcome = _OUTCOME_REFUSED
        elif is_answered:
            probe_outcome = _OUTCOME_ANSWERED
        else:
            probe_outcome = _OUTCOME_BLOCKED
        p_abst_record = {
            "item_id": pid,
            "kind": "probe",
            "expected_axis": probe_axis,
            "outcome": probe_outcome,
        }
        abstention_records.append(p_abst_record)
        probe_category_records.append(
            {"category": category, "refused": is_refused}
        )
        # A false-premise (ill_posed) probe that ANSWERED is a judge candidate,
        # exactly like a false-premise gold item.
        if probe_axis == _AXIS_FALSE_PREMISE and is_answered:
            judge_candidates.append(
                {
                    "record": p_abst_record,
                    "item_id": pid,
                    "kind": "probe",
                    "question_text": ptext,
                    "answer_text": _answer_field(answer, "answer_text", None),
                    "cited_texts": [
                        c.get("text_quote")
                        for c in _citations_list(answer)
                        if c.get("text_quote")
                    ],
                }
            )

        # Refusal-safety: for an ANSWERED probe, pull its groundedness report's
        # unsupported signal (same shape the gold loop reads). null when the
        # probe was refused / NLI was unavailable / nothing scored — never a
        # fabricated 0.0.
        probe_unsupported_rate: Optional[float] = None
        if is_answered:
            p_grounded = _answer_field(answer, "groundedness", None)
            if isinstance(p_grounded, dict) and p_grounded.get("available"):
                p_scored = int(p_grounded.get("scored_count", 0) or 0)
                p_unsupported = int(p_grounded.get("unsupported_count", 0) or 0)
                if p_scored:
                    probe_unsupported_rate = p_unsupported / p_scored
                    probe_answered_unsupported_rates.append(
                        probe_unsupported_rate
                    )
                    probe_answered_unsupported_count += p_unsupported
                    probe_answered_claims_scored += p_scored
                else:
                    # Answered but all-computational/filtered → 0 scorable
                    # claims; credited as 0.0 in the per-probe mean (same
                    # dilution convention the gold loop uses), but contributes
                    # nothing to the claim-level denominator.
                    probe_unsupported_rate = 0.0
                    probe_answered_unsupported_rates.append(0.0)
        probe_results.append(
            {
                "probe_id": pid,
                "category": category,
                "refused": is_refused,
                "answered": is_answered,
                "unsupported_rate": probe_unsupported_rate,
            }
        )
        # Best-effort progress: a refused probe is the SAFE outcome (declined);
        # an answered probe is answered-not-refused. Status carries the raw
        # pipeline verdict for the tail line.
        _progress_call(
            progress,
            "record_item",
            _PROGRESS_ARM_GROUNDED,
            pid,
            status,
            answered=is_answered,
            declined=is_refused,
        )

    _progress_call(progress, "finish_arm", _PROGRESS_ARM_GROUNDED)

    # --- Premise-correction judge (schema 1.7) -----------------------------
    # Run the GEval-style false-premise judge ONLY on the false-premise items
    # that ANSWERED (a handful per run). Each verdict may upgrade that item's
    # abstention outcome answered → premise_corrected. Judge failure / timeout /
    # an unavailable backend degrades gracefully: the item stays 'answered' and
    # the run never crashes.
    judge_meta: Dict[str, Any] = {
        "enabled": False,
        "provider": None,
        "model": None,
        "candidates": len(judge_candidates),
        "judged": 0,
        "degraded": 0,
    }
    if judge_candidates:
        _client = judge_client
        _resolved = None
        if _client is None:
            _client, _resolved = _resolve_judge_client(capture)
        if _client is not None:
            judge_meta["enabled"] = True
            if _resolved is not None:
                judge_meta["provider"] = _resolved.provider_name
                judge_meta["model"] = _resolved.model_id
            elif judge_client is not None:
                judge_meta["provider"] = "injected"
            for cand in judge_candidates:
                verdict = _judge_premise_correction(_client, cand, capture)
                if verdict is None:
                    # Graceful degrade: outcome stays 'answered' (already set).
                    judge_meta["degraded"] += 1
                    continue
                judge_meta["judged"] += 1
                if verdict.corrected:
                    cand["record"]["outcome"] = _OUTCOME_PREMISE_CORRECTED
        else:
            # Judge backend unavailable → every candidate degrades to 'answered'
            # (its already-set outcome); surfaced via the degraded count.
            judge_meta["degraded"] = len(judge_candidates)

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
        # ADDITIVE headline (nli-add-flip plan PR-1, mirrors the prune arm's
        # *_preprune pattern): citation_precision recomputed over the citation
        # set BEFORE the NLI-ADD pass appended anything. When NLI-ADD is
        # off/shadow (the only modes shipping today) NO citation was added, so
        # this is byte-identical to citation_precision. When NLI-ADD is later
        # flipped ON, citation_precision_preadd stays comparable to today's
        # number (the model's own citations) while citation_precision reflects
        # the post-add set — so the ADD's denominator effect against a
        # one-passage gold is MEASURED, not assumed, and the flip run is not
        # misread as a precision regression.
        "citation_precision_preadd": (
            (n_relevant_citations_preadd / n_emitted_citations_preadd)
            if n_emitted_citations_preadd
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
            # REFUSAL-SAFETY axis (schema 1.3, additive): of the probes the
            # pipeline FAILED to refuse (answered), how fabricated were the
            # answers? Computed ONLY over answered probes — separates a base-
            # like pure invention from a grounded near-miss answered WITH a real
            # corpus citation (failed-to-refuse but materially less harmful).
            "answered_probe_count": probe_answered,
            # Per-probe mean unsupported-vs-corpus rate over answered probes.
            # None when zero answered / NLI unavailable — never a fabricated 0.0.
            "unsupported_answer_rate_on_answered_probes": (
                (
                    sum(probe_answered_unsupported_rates)
                    / len(probe_answered_unsupported_rates)
                )
                if probe_answered_unsupported_rates
                else None
            ),
            # Undiluted claim-level rate: total unsupported / total scored over
            # answered probes (does NOT credit an all-computational answer as
            # 0.0). None when no answered probe had a scorable claim.
            "claim_level_unsupported_rate_on_answered_probes": (
                (
                    probe_answered_unsupported_count
                    / probe_answered_claims_scored
                )
                if probe_answered_claims_scored
                else None
            ),
            "answered_probe_claims_scored": probe_answered_claims_scored,
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
        # Diagnostic, NOT pinned: multi_part coverage.
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
        # Scorer-v2 additive diagnostics (NOT pinned milestones — measure-then-
        # pin): claims the v2 groundedness scorer exempted as computational,
        # dropped as structural artifacts, and entailed via an UNCITED corpus
        # chunk (citation-selection signal, separate from fabrication-vs-corpus).
        # All sum over answered questions whose groundedness report was
        # available; 0 on a v1-scorer report (keys absent → counted as 0).
        "groundedness_scorer_v2": {
            "computational_claim_count": computational_claim_count,
            "filtered_claim_count": filtered_claim_count,
            "entailed_uncited_count": entailed_uncited_count,
            "_diagnostic": (
                "scorer-v2 counters (2026-06-12 audit response); diagnostics, "
                "not pinned milestones — re-pin unsupported_claim_rate after the "
                "first v2 run"
            ),
        },
        # Macro + micro groundedness sliced by difficulty stratum
        # AND by question_type over answered questions whose groundedness report
        # was available. Diagnostic breakdown (NOT a pinned milestone) — surfaces
        # where grounding is weak (which stratum / question shape) rather than
        # only the single headline mean.
        "groundedness_breakdown": _groundedness_breakdown(
            groundedness_breakdown_records
        ),
        # Schema 1.6 additive: per-learner-phrasing answerability slice
        # (canonical / colloquial / malformed). One row per phrasing value with
        # n / answered_count / answered_rate / blocked_count / refused_count over
        # the gold questions carrying that phrasing (a phrasing-less question
        # reads as canonical). Surfaced so an operator can see at a glance
        # whether the bare/colloquial + malformed-but-reasonable learner
        # questions (real GUI phrasings) pass — diagnostic, NOT a pinned metric.
        "phrasing_breakdown": _phrasing_breakdown(phrasing_records),
        # Numeric-grounding roll-up (all-zero on the default off path): the
        # computational-sentence numeric-grounding diagnostic aggregated over all
        # answered questions. WARNING-only — never fed any pinned metric.
        "computational_numeric_check": {
            "enabled": comp_numeric_claims_checked > 0
            or comp_numeric_ungrounded_total > 0,
            "computational_claims_checked": comp_numeric_claims_checked,
            "claims_with_ungrounded_numeric": comp_numeric_ungrounded_claims,
            "ungrounded_numeric_count": comp_numeric_ungrounded_total,
            "_diagnostic": (
                "W1.8 numeric-grounding of NLI-exempt computational sentences "
                "(ED4ALL_GROUNDEDNESS_COMPUTATIONAL); warning-only, never feeds "
                "a pinned metric — all-zero when the flag is off"
            ),
        },
        # NLI-ADD shadow diagnostics (NOT a pinned milestone — measure-then-
        # decide): what the NLI-based citation-ADD arm WOULD credit under its
        # composite criterion. Present (non-zero possible) only when the arm ran
        # (ED4ALL_ANSWER_NLI_ADD=shadow|on, with_groundedness on); all 0 on the
        # default off path where answers carry no nli_citation_add block.
        "shadow_nli_add": {
            "answers_with_would_adds": nli_answers_with_would_adds,
            "total_would_adds": nli_total_would_adds,
            "zero_citation_answers_recovered": nli_zero_citation_answers_recovered,
            "_diagnostic": (
                "NLI-ADD would-add counts (2026-06-12 under-citing "
                "investigation); shadow diagnostics, not a pinned milestone. "
                "zero_citation_answers_recovered is WARNING-class, not a win: "
                "would-adds on prune-to-empty answers were hand-judged 3/3 "
                "question-induced false adds; ON mode never adds to them"
            ),
        },
        "latency_ms": {
            "p50": _percentile(latencies, 50.0),
            "p95": _percentile(latencies, 95.0),
        },
    }

    # --- Schema 1.7 headline additions ------------------------------------
    # (5) MICRO groundedness promoted to a first-class headline field. Micro =
    # claim-weighted (total entailed / total scored over answered questions
    # whose groundedness report was available); None when nothing was scored.
    _g_scored = sum(r["scored"] for r in groundedness_breakdown_records)
    _g_entailed = sum(r["entailed"] for r in groundedness_breakdown_records)
    headline["groundedness_rate_micro"] = (
        (_g_entailed / _g_scored) if _g_scored else None
    )
    # (4) CITATION metrics. citation_precision is now the supported-over-CITED
    # per-question MACRO average (mean over questions that emitted >=1 citation).
    # The legacy POOLED number (relevant/emitted across the whole run) moves to
    # citation_precision_legacy for cross-run comparability. citation_recall is
    # the cited-over-required per-question macro average over answered questions
    # pinning >=1 relevant passage — a single-passage question contributes
    # 1.0-or-0.0 (all-or-nothing; stated honestly), so the multi-passage count
    # is surfaced alongside so single vs multi is legible.
    headline["citation_precision_legacy"] = (
        (n_relevant_citations / n_emitted_citations)
        if n_emitted_citations
        else 0.0
    )
    headline["citation_precision"] = (
        (sum(citation_precision_macro_terms) / len(citation_precision_macro_terms))
        if citation_precision_macro_terms
        else None
    )
    headline["citation_recall"] = (
        (sum(citation_recall_terms) / len(citation_recall_terms))
        if citation_recall_terms
        else None
    )
    headline["citation_recall_basis"] = {
        "answered_questions_with_pinned_passage": len(citation_recall_terms),
        "multi_passage_questions": multi_passage_question_count,
        "_note": (
            "citation_recall = macro mean of |cited ∩ required| / |required| over "
            "answered questions pinning >=1 relevant passage; single-passage "
            "questions contribute 1.0-or-0.0. citation_precision = macro mean of "
            "relevant/emitted over questions with >=1 citation; "
            "citation_precision_legacy is the run-pooled relevant/emitted number."
        ),
    }
    # (1) Two-axis abstention block.
    headline["abstention"] = _abstention_block(abstention_records, judge_meta)
    # (2) Per-category probe stats with Wilson CIs.
    headline["refusal"]["by_category"] = _probe_category_stats(
        probe_category_records
    )
    headline["refusal"]["_note"] = (
        "legacy single-axis refusal metrics (kept for comparability); see "
        "headline.abstention for the schema-1.7 two-axis abstention scorer that "
        "also credits premise-correction on false-premise items"
    )
    # (E2) Multi-turn REACHABILITY diagnostic. Surfaces how many gold questions
    # carried prior_turns and therefore actually drove the multiturn seam — the
    # harness now PASSES prior_turns (previously it never did, so multiturn was
    # structurally unreachable). All-zero on a single-turn gold set.
    headline["multiturn"] = {
        "gold_with_prior_turns": multiturn_gold_count,
        "answered_with_prior_turns": multiturn_answered_count,
        "prior_turns_total": multiturn_prior_turns_total,
        "reachable": multiturn_gold_count > 0,
        "_diagnostic": (
            "multi-turn reachability (E2): count of gold questions carrying "
            "prior_turns that drove answer_course_question(prior_turns=...); the "
            "ED4ALL_ANSWER_MULTITURN retrieval-rewrite fires only when that flag "
            "is on. Diagnostic, NOT a pinned milestone; all-zero single-turn."
        ),
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
        # REFUSAL-SAFETY axis (schema 1.3, additive): per-probe rows mirroring
        # the per-question `questions` rows so the answered-probe groundedness
        # is auditable post-hoc (probe_id / category / refused / answered /
        # unsupported_rate-or-null). The headline.refusal block carries the
        # roll-up; this is the row-level evidence.
        "probe_results": probe_results,
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
        # Schema 1.7 attribution stamp: the resolved ED4ALL_ANSWER_* + eval-judge
        # env config this run executed under, so a stored report is attributable
        # to its exact answer-path flag config (A/B + re-pin readability).
        "flag_config": _flag_config_stamp(),
        # Schema 1.8 (E3): selective-QA / risk-coverage view over the per-answer
        # groundedness scores (AURC, abstention AUROC, ECE, coverage curve). Pure
        # post-processing — no new model calls; basis=="none" when nothing scored.
        "risk_coverage": selective_qa_view(risk_records),
        "generated_at": _utcnow_iso(),
    }

    # Schema 1.8 (E2): OPTIONAL library-wide reachability slice. Present only when
    # a caller passes library_questions — drives answer_library_question over a
    # cross-course set; cleanly skips when the resolved library is single-course.
    # Absent by default → additive, byte-identical to the single-course report.
    if library_questions is not None:
        report["library_wide"] = run_library_eval(
            repo_root,
            course_slug,
            library_questions,
            engine=engine,
            limit=limit,
            client=client,
            with_groundedness=with_groundedness,
            capture=capture,
            course_slugs=library_course_slugs,
            answer_fn=library_answer_fn,
        )

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
    "REVIEW_SAMPLE_SCHEMA_VERSION",
    "MILESTONE_TARGETS",
    "MILESTONE_CEILINGS",
    "MILESTONE_TARGETS_PINNED_AT",
    "KEY_POINT_COVERAGE_DIAGNOSTIC_BASELINE",
    "probe_expected_outcome",
    "ENV_EVAL_JUDGE_PROVIDER",
    "ENV_EVAL_JUDGE_MODEL",
]
