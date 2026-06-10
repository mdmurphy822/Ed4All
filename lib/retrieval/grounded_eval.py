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

from lib.retrieval.gold_set import has_critical_issues, load_gold_set

EVAL_SCHEMA_VERSION = "1.0"
RETRIEVAL_EVAL_SUBDIR = "retrieval_eval"
REVIEW_SAMPLE_FILENAME = "groundedness_review_sample.json"

# --------------------------------------------------------------------------- #
# Milestone targets — measure-then-pin (floors CURRENT evidence meets)
# --------------------------------------------------------------------------- #

#: Date the targets below were pinned, from the three measured runs to date.
MILESTONE_TARGETS_PINNED_AT = "2026-06-10"

#: Headline-metric targets pinned against the three real-corpus eval runs.
#: Every target is FLOOR = min(measured) (a small float-noise margin below for
#: the >=-style metrics) — except ``unsupported_claim_rate``, which is a CEILING
#: = max(measured) rounded slightly up. All seven are evidence-met on every
#: artifact today, never aspirational. Re-pin (tighten) only after new measured
#: runs clear a stricter bound.
#:
#: Measured runs (model qwen2.5:14b-instruct-q4_K_M, prompt ws3.v1, local backend):
#:                         answer  cit_res  cit_prec  ground   unsup   ref_rec  ref_prec
#:   nvidiarag-101   sem   1.0     1.0      0.615     0.7      0.2     0.889    1.0   (2026-06-09)
#:   rdf-shacl-551-2 lex   0.5     1.0      0.5       0.6      0.2     0.667    1.0   (2026-06-10)
#:   openstax-alg-9  lex   0.9     1.0      0.6       0.167    0.222   1.0      0.9   (2026-06-10)
#:
#: citation_resolution_rate is pinned at 1.0: all three runs measured exactly
#: 1.0 (every emitted citation resolved), so the floor IS the measurement —
#: any regression below it is a real anchoring break, not noise.
MILESTONE_TARGETS: Dict[str, float] = {
    # FLOOR. min measured 0.5 (rdf-shacl-551-2 lexical 2026-06-10);
    # spread 0.5 (rdf-shacl) .. 1.0 (nvidiarag).
    "answer_rate": 0.50,
    # FLOOR == measurement. All three runs measured exactly 1.0 — nvidiarag-101
    # (sem, 2026-06-09), rdf-shacl-551-2 (lex, 2026-06-10), openstax-alg-9 (lex,
    # 2026-06-10). Pinned at 1.0 exactly; any dip is a real anchoring break.
    "citation_resolution_rate": 1.0,
    # FLOOR. min measured 0.5 (rdf-shacl-551-2 lexical 2026-06-10), small
    # float-noise margin; spread 0.5 (rdf-shacl) .. 0.615 (nvidiarag).
    "citation_precision": 0.45,
    # FLOOR. min measured 0.167 (openstax-alg-9 lexical 2026-06-10), small
    # margin. The wide spread 0.167 (openstax) .. 0.7 (nvidiarag) flags
    # prompt/corpus groundedness follow-up work (dated 2026-06-10); pinned
    # honestly to the openstax floor, NOT to an aspirational mid-range.
    "groundedness_rate_mean": 0.15,
    # CEILING. max measured 0.222 (openstax-alg-9 lexical 2026-06-10), rounded
    # up; spread 0.2 (nvidiarag/rdf-shacl) .. 0.222 (openstax). Lower is better.
    "unsupported_claim_rate": 0.25,
    # FLOOR. min measured 0.667 (rdf-shacl-551-2 lexical 2026-06-10), small
    # margin; spread 0.667 (rdf-shacl) .. 1.0 (openstax).
    "refusal_recall": 0.65,
    # FLOOR. min measured 0.9 (openstax-alg-9 lexical 2026-06-10), small
    # margin; spread 0.9 (openstax) .. 1.0 (nvidiarag/rdf-shacl).
    "refusal_precision": 0.85,
}

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


class PipelineUnavailable(NotImplementedError):
    """The grounded-answer pipeline (E6 wave B) has not landed yet.

    Raised (never silently degraded) when
    ``lib.retrieval.grounded_answer.answer_course_question`` is not importable,
    so an operator sees the exact missing dependency rather than a fabricated
    eval result.
    """


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
    }
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
    false_refusals_on_gold = 0
    groundedness_rates: List[float] = []
    unsupported_rates: List[float] = []

    # --- answerable gold questions ---------------------------------------
    for q in questions:
        qid = str(q.get("question_id", ""))
        qtext = str(q.get("question_text", ""))
        all_rel, primary_rel = _relevant_chunk_ids(q)

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
            if cid in all_rel:
                relevant_here += 1
            if cid in primary_rel:
                relevant_primary_here += 1

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

        per_question_report.append(
            {
                "question_id": qid,
                "status": status,
                "n_citations": n_cites,
                "citations_resolved": resolved_here,
                "citation_relevant_primary": relevant_primary_here,
                "groundedness_rate": g_rate,
                "latency_ms": latency,
            }
        )
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
    for probe in probes:
        ptext = str(probe.get("question_text", ""))
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
]
