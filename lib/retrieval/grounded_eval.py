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
    question_phrasing,
)

#: Bumped 1.0 -> 1.1 for the P4 additive per-question scoring fields
#: (key_point_coverage, part_coverage, population breakdown). Every v1.0
#: report field is preserved; the new fields are additive and present only
#: when the v1.1 gold question carries the matching authored data.
#: ``EVAL_SCHEMA_VERSION`` 1.2 → 1.3 (additive only — every 1.2 key unchanged):
#: the refusal-probe loop now scores groundedness on ANSWERED probes (probes the
#: pipeline failed to refuse), rolling the unsupported-vs-corpus signal into the
#: ``headline.refusal`` block (``answered_probe_count``,
#: ``unsupported_answer_rate_on_answered_probes``,
#: ``claim_level_unsupported_rate_on_answered_probes``,
#: ``answered_probe_claims_scored``) and persisting per-probe rows under a NEW
#: top-level ``probe_results`` key (probe_id / category / refused / answered /
#: unsupported_rate-or-null) — mirroring the per-question ``questions`` rows.
#: The existing ``refusal_recall`` / ``refusal_precision`` / ``n_probes`` fields
#: and every other headline metric are UNCHANGED.
#: ``EVAL_SCHEMA_VERSION`` 1.3 → 1.4 (additive only — every 1.3 key unchanged;
#: nli-add-flip plan PR-1 + gold-enrichment plan step 1): adds the
#: ``headline.citation_precision_preadd`` metric (citation_precision recomputed
#: over the citation set BEFORE the NLI-ADD pass appended anything — equal to
#: ``citation_precision`` whenever NLI-ADD is off/shadow, the only modes
#: shipping today) and persists per-question ``cited_chunk_ids`` + ``cited_pages``
#: on EVERY ``questions`` row (previously the cited chunk_ids lived only inside
#: the 25-question review sample). Both are prerequisites for measuring the
#: NLI-ADD flip + the multi-passage gold enrichment over all questions.
#: ``EVAL_SCHEMA_VERSION`` 1.4 → 1.5 (additive only — every 1.4 key unchanged;
#: W1.9): adds ``headline.groundedness_breakdown`` (macro + micro groundedness
#: sliced by difficulty stratum AND by question_type, over answered questions
#: whose groundedness report was available) and ``headline.computational_numeric_check``
#: (a roll-up of the W1.8 per-answer numeric-grounding diagnostic — all-zero on
#: the default path where ``ED4ALL_GROUNDEDNESS_COMPUTATIONAL`` is off). Both are
#: diagnostics, NOT pinned milestones.
#: ``EVAL_SCHEMA_VERSION`` 1.5 → 1.6 (additive only — every 1.5 key unchanged):
#: adds ``headline.phrasing_breakdown`` — a per-learner-phrasing answerability
#: slice (canonical / colloquial / malformed, from the v1.1 gold ``phrasing``
#: field, defaulting to canonical). For each phrasing value it reports n,
#: answered_count, answered_rate (status in the answered set), blocked_count, and
#: refused_count so an operator can see at a glance whether the bare/colloquial
#: and malformed-but-reasonable learner questions (surfaced by real GUI testing)
#: pass, not only the well-formed textbook phrasings. Diagnostic, NOT a pinned
#: milestone.
EVAL_SCHEMA_VERSION = "1.6"
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
#: Re-measured clean run (same day, scorer v2 + pinned ws3.v3 refusal policy
#: incl. the floor-rounding boundary fix, prune=on @ 0.444444):
#:   answer_rate 0.9351  citation_resolution_rate 1.0  citation_precision 0.4457
#:   groundedness_rate_mean 0.807  unsupported_claim_rate 0.0891
#:   refusal_recall 0.8235  refusal_precision 0.875  key_point_coverage 0.5482
#: Scorer-v1 vs scorer-v2 groundedness numbers are NOT comparable (see the
#: per-pin notes); refusal numbers moved with the policy pin, not a model
#: change.
#:
#: Gold v1.2 three-arm run (2026-06-12, the CURRENT pin basis): gold expanded
#: 77 → 117 questions (operator-approved definition + worked_example
#: templates), probes 34 → 44 (+10 pure off-topic), composer prompt ws3.v3
#: (synthesis/apply clauses), ED4ALL_ANSWER_NUM_CTX=8192 (production parity —
#: prior runs measured the 4096 config), prune=on @ 0.444444, NLI-ADD shadow:
#:   answer_rate 0.9727 (107/110)  citation_resolution_rate 1.0
#:   citation_precision 0.3314  groundedness_rate_mean 0.8045
#:   unsupported_claim_rate 0.0366  refusal_recall 0.6364  refusal_precision
#:   0.9655  key_point_coverage 0.6565  false_refusals_on_gold 1
#: Three-arm context (same scorer, same questions): BASE (qwen, no retrieval)
#: key_point_coverage 0.1521; RETRIEVAL extractive ceiling 0.9557 (hit@k
#: 0.9182); GROUNDED 0.6565. citation_precision moved 0.4457 → 0.3314 via gold
#: denominator semantics again (the 40 new template questions pin ONE relevant
#: passage each), not a pipeline change. Latency p50 8.5s → 24.0s: the 8192
#: serving window + 17 long-form worked_example questions; production runs the
#: same config, so this is the honest user-facing number for that mix.
#:
#: Gold v1.3 gap-closed run (2026-06-12, all coverage/validation warnings
#: cleared: 125q — 11 both-population, difficulty 54/58/13 in-band, 13/13
#: multi_part parts[] authored with passage refs, all expected_key_points
#: present; same probes/config as v1.2): answer_rate 0.976 (122/125)
#: citation_resolution_rate 1.0  citation_precision 0.3216
#: groundedness_rate_mean 0.8258  unsupported_claim_rate 0.0348
#: refusal_recall 0.6364  refusal_precision 0.9655  key_point_coverage 0.6420
#: part_coverage answered_rate 0.4643 (first real measurement — multi-part
#: drop-rate is now visible). Three-arm: BASE 0.1569 / RETRIEVAL ceiling
#: 0.9279 (hit@k 0.9040) / GROUNDED 0.6420. Every pin below holds on this
#: basis unchanged. Shadow NLI-ADD: 22 would-adds, hand-audited 22/22 genuine
#: at ent>=0.70 + cov>=0.80 + numerics + anchor — the flip-to-on evidence.
#:
#: CONFIG-LABEL CORRECTION (2026-06-13): the v1.2 + v1.3 runs above were
#: labeled "ED4ALL_ANSWER_NUM_CTX=8192 production parity" but actually ran
#: against a host-native ollama serving 4096 (the env var is only a local
#: prompt-budget hint; it does NOT set the server window — that needs
#: OLLAMA_CONTEXT_LENGTH or a Modelfile num_ctx PARAMETER). Additionally the
#: BASE arm reused the grounded pipeline's json_mode=True client, so raw qwen
#: emitted JSON the scorer-v2 artifact filter discarded — BASE hallucination
#: + key_point_coverage (0.1569) for v1.2/v1.3 are INVALID (filtered content).
#:
#: DEFINITIVE run (2026-06-13, gold v1.3 125q + 44 probes; TRUE 8192 via a
#: Modelfile-baked qwen2.5-7b-8k; BASE arm json_mode=False free text;
#: ED4ALL_NLI_DEVICE=cuda fp16 — grounded per-item latency 24s→5.8s):
#:   answer_rate 0.976  citation_resolution_rate 1.0  citation_precision 0.3627
#:   groundedness_rate_mean 0.899  unsupported_claim_rate 0.0395
#:   refusal_recall 0.8182  refusal_precision 0.9474  key_point_coverage 0.6883
#: Safety (44 probes): answered-not-refused BASE 1.00 → GROUNDED 0.182;
#:   probe_fabrication_rate BASE 1.00 → GROUNDED 0.011 (base fabricates on
#:   every refusable question; grounded ~1%). Hallucination-vs-corpus BASE
#:   0.0591 → GROUNDED 0.0395 (33% rel). Three-arm coverage BASE 0.40 (CORRECTED
#:   from the invalid 0.157 — free-text base is far more capable than the
#:   json-filtered measurement showed; the grounding coverage gain is ~1.7x,
#:   NOT the ~3.5x the broken base number implied) / RETRIEVAL 0.928 /
#:   GROUNDED 0.688. Every pin below holds unchanged on this definitive basis.
#:
#: citation_resolution_rate stays pinned at 1.0: every emitted citation resolved
#: on this run too, so the floor IS the measurement — any dip is a real
#: anchoring break, not noise.
MILESTONE_TARGETS: Dict[str, float] = {
    # FLOOR. Re-pinned 0.92 → 0.95 on the gold-v1.2 / ws3.v3 / 8192-ctx basis
    # (three-arm run 2026-06-12: measured 0.9727 = 107/110). The 0.92 pin's
    # attributed non-answers are RESOLVED on this basis: the ws3.v3
    # synthesis/apply prompt fixed the conceptual-synthesis false refusals
    # (4 → 1 false_refusals_on_gold) and the 8192 serving window stopped the
    # context budget from dropping a 4th-ranked gold passage whole (the
    # florist-LCM refusal — prior runs measured the 4096 config). Remaining
    # non-answers: 1 false refusal + 1 composer_exhausted + 1 citation-gate
    # fail-closed block. Single-corpus caveat applies.
    "answer_rate": 0.95,
    # FLOOR == measurement. Every emitted citation resolved on the 2026-06-12
    # run (and on all prior runs). Pinned at 1.0 exactly; any dip is a real
    # anchoring break.
    "citation_resolution_rate": 1.0,
    # FLOOR. measured 0.3495 (77q gold v1.1) and 0.3314 (117q gold v1.2,
    # 2026-06-12) — pinned to 0.30. DENOMINATOR ARTIFACT: each drafted gold
    # question pins exactly ONE relevant passage, so a multi-citation answer
    # mathematically deflates per-question citation_precision (the v1.2 dip
    # vs the 0.4457 clean run is the 40 new one-passage template questions,
    # not a pipeline change). NOT comparable to the old 0.45/10q-basis pin.
    # A richer multi-passage gold would raise it.
    "citation_precision": 0.30,
    # FLOOR. measured 0.572 under scorer v1 (single course 2026-06-12); the
    # first scorer-v2 run measured 0.805 (same day, same frozen gold — wider
    # gate-eligible evidence pool + windowed rescue + computational exemption
    # raise the rate by construction, so the two are NOT comparable). Floor
    # kept at 0.15 — the prior cross-course floor — because a single run does
    # not establish a credible cross-corpus bound (R4); tightening waits on a
    # second different-family course.
    "groundedness_rate_mean": 0.15,
    # CEILING. Re-pinned 0.25 → 0.10 on the scorer-v2 basis (2026-06-12:
    # 0.0904 on the first v2 run, 0.0891 on the clean run after the refusal
    # floor-rounding fix; single course, 77q frozen gold v1.1, hybrid-rrf, 7B,
    # prune=on @ 0.444444). Basis change: the v1 pin (measured 0.2326) counted
    # scorer noise the 2026-06-12 manual audit attributed to NLI false
    # negatives on glossary-style multi-topic chunks, novel-but-correct
    # computations, and claim-splitter artifacts (true fabrication ~5%);
    # scorer v2 (artifact filter + computational exemption + windowed rescue +
    # gate-eligible evidence pool) removes them, so v1/v2 values are NOT
    # comparable and the single-course never-tighten rule does not apply to a
    # basis-invalidated pin. Residual ~9% is dominated by question-induced
    # evaluative synthesis (gold questions that ask "which is better?") —
    # honest unsupported, not fabrication. Tightened 0.10 → 0.05 on the gold
    # v1.2 basis (measured 0.0366 over 117q — the definition/worked_example
    # templates ground cleanly, diluting the evaluative-synthesis tail).
    # Single-corpus caveat applies.
    "unsupported_claim_rate": 0.05,
    # FLOOR. Re-pinned 0.45 → 0.55 on the 44-probe gold-v1.2 basis
    # (2026-06-12: measured 0.6364 incl. the 10 new pure off-topic probes).
    # NOT comparable to the 34-probe or 9-probe bases. The remaining recall
    # gap is near-miss / adjacent-domain probes inside the score-overlap band
    # — MODEL-policy-owned, not a retrieval-threshold matter — see
    # lib/retrieval/refusal.py PINNED_POLICIES hybrid-rrf overlap block.
    "refusal_precision_floor_note": 0.0,  # placeholder removed below
    # FLOOR. measured 0.6364 (44-probe basis 2026-06-12), pinned 0.55.
    "refusal_recall": 0.55,
    # FLOOR. Re-pinned 0.85 → 0.90 (measured 0.9655 on the 44-probe gold-v1.2
    # basis; ws3.v3 cut false refusals on gold 4 → 1).
    "refusal_precision": 0.90,
}
# Remove the inline placeholder key used only to anchor the refusal_recall note.
MILESTONE_TARGETS.pop("refusal_precision_floor_note", None)

#: DIAGNOSTIC (unpinned). key_point_coverage baselines: 0.54 (77q gold v1.1),
#: 0.6565 (117q gold v1.2), 0.6420 (125q gold v1.3 gap-closed run, 2026-06-12
#: — same scorer measured BASE 0.1569 and the RETRIEVAL extractive ceiling
#: 0.9279, so grounded converts ~69% of what retrieval surfaces). Deliberately
#: NOT in MILESTONE_TARGETS until the metric is stable across a second
#: different-family course (plan risk R7 posture); the staleness test does NOT
#: gate on it.
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


def _breakdown_agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one slice of per-question groundedness records (W1.9).

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
    """Macro + per-stratum + per-question-type groundedness breakdown (W1.9).

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
    write: bool = True,
    progress: Optional[Any] = None,
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
    # W1.8 roll-up (additive; all-zero on the default off path where no answer
    # carries a ``computational_numeric_check`` block): computational sentences
    # whose numeric literals were verified against the cited chunks, and how
    # many carried an ungrounded (fabricated-looking) numeric literal.
    comp_numeric_claims_checked = 0
    comp_numeric_ungrounded_claims = 0
    comp_numeric_ungrounded_total = 0
    # W1.9 per-question groundedness records (difficulty stratum + question_type
    # + macro/micro basis). One row per answered question whose groundedness
    # report was available.
    groundedness_breakdown_records: List[Dict[str, Any]] = []
    # Per-learner-phrasing answerability records (schema 1.6). One row per gold
    # question: its resolved phrasing (canonical/colloquial/malformed) + final
    # status. Recorded for EVERY question incl. composer-exhausted ones so the
    # per-phrasing n counts the whole gold set.
    phrasing_records: List[Dict[str, Any]] = []
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
            phrasing_records.append(
                {"phrasing": phrasing, "status": _COMPOSER_EXHAUSTED}
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
            # W1.9 breakdown record: entailed = scored − unsupported − contradicted
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
            # W1.8 numeric-grounding roll-up (present only when the opt-in check
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

        # --- NLI-ADD shadow diagnostics (additive; present only when the arm ran)
        nli_add = _answer_field(answer, "nli_citation_add", None)
        if isinstance(nli_add, dict):
            would = [str(x) for x in (nli_add.get("would_add_ids") or [])]
            if would:
                nli_answers_with_would_adds += 1
                nli_total_would_adds += len(would)
                # WARNING-class, not a recovery win: would-adds on a
                # zero-citation (prune-to-empty) answer were hand-judged 3/3
                # question-induced false adds (2026-06-12 investigation). ON
                # mode never adds to these; this count surfaces how often the
                # criterion fires where it is known-suspect.
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
            # ADDITIVE (gold-enrichment plan step 1): persist EVERY emitted
            # citation's chunk_id on EVERY per-question row (previously these
            # ids lived only inside the 25-question review sample). This makes
            # citation_precision auditable + the multi-passage enrichment
            # measurable over all 125 questions — each id can be joined to its
            # chunk body to judge whether an unpinned-but-cited passage is a
            # genuine supporter. page_label rides along when the pipeline
            # carried it (cheap; the citation dict already exposes it).
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
        # Persist the NLI-ADD shadow block per question (additive; present only
        # when the arm ran). Without this the would-add audit set existed only
        # as headline counts — the 2026-06-12 three-arm run's 18/20 would-adds
        # could not be hand-audited post-hoc. The shadow→on flip decision is
        # gated on auditing exactly these rows.
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
        # misread as a precision regression (see plan §2.2/§2.3).
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
        # W1.9 additive: macro + micro groundedness sliced by difficulty stratum
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
        # W1.8 roll-up (additive; all-zero on the default off path): the
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
    "REVIEW_SAMPLE_SCHEMA_VERSION",
    "MILESTONE_TARGETS",
    "MILESTONE_CEILINGS",
    "MILESTONE_TARGETS_PINNED_AT",
    "KEY_POINT_COVERAGE_DIAGNOSTIC_BASELINE",
]
