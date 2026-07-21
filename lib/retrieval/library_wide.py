"""Library-wide grounded ask — union retrieval across course indexes (W4.2).

Today ``lib.retrieval.grounded_answer.answer_course_question`` answers a
question against ONE course. This module adds the opt-in
``ED4ALL_ANSWER_LIBRARY_WIDE`` mode: retrieval is unioned across MULTIPLE course
indexes (the W0.8 master-catalog set, or an explicit slug list), the composed
answer is grounded + citation-gated exactly like a single-course answer, and
**every citation keeps its source course's provenance** (the additive
``course_slug`` field on :class:`~lib.retrieval.grounded_answer.Citation`).

Contract (mirrors the single-course path's guarantees):

  * **Default OFF** — :func:`resolve_library_wide` reads
    ``ED4ALL_ANSWER_LIBRARY_WIDE`` (parse-with-fallback, default off). The
    single-course path is untouched and byte-identical.
  * **Local-only** — retrieval + the citation gate reuse the same
    ``grounded_answer`` machinery, whose compose step is the loopback-enforced
    ``ED4ALL_ANSWER_PROVIDER`` backend. No cloud call is introduced here.
  * **Per-course provenance** — each unioned passage is stamped with the course
    it came from; the citation gate resolves each cited passage against ITS OWN
    course dir + chunkset, so a citation can never be attributed to the wrong
    course.
  * **always-keep >= 1** — the union is score-sorted and trimmed to ``limit``
    but never below one passage when any course contributed.
  * **fail-open to single-course** — when the course set resolves to <= 1
    course (no catalog, only the home course archived, or every OTHER course's
    retrieval errored), the call degrades to
    :func:`answer_course_question` on the home course. A per-course retrieval
    error never fails the whole union — that course is skipped and logged.

Anti-fabrication: the union only ever surfaces REAL retrieved passages from
REAL course indexes; a citation's ``course_slug`` is the course it was actually
retrieved from, never inferred.
"""
from __future__ import annotations

import os
import time
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lib.retrieval.answer_composer import (
    ComposedAnswer,
    RetrievedPassage,
    compose_answer,
)
from lib.retrieval.citation_anchor import _RESOLVED_STATUSES, resolve_citation_anchor
from lib.retrieval.grounded_answer import (
    STATUS_ANSWERED,
    STATUS_ANSWERED_WITH_WARNINGS,
    STATUS_BLOCKED_CITATION_GATE,
    STATUS_BLOCKED_INVALID_CITATION,
    STATUS_REFUSED_LOW_CONFIDENCE,
    STATUS_REFUSED_NOT_IN_COURSE,
    Citation,
    GroundedAnswer,
    _answered,
    _blocked,
    _build_citation,
    _infer_chunkset_kind,
    _invoke_progress,
    _libv2_root,
    _passage_chunk_record,
    _query_sha,
    _refused,
    _result_chunk_type,
    _retrieve,
    _safe_log,
    _vector_index_chunkset_kind,
    answer_course_question,
    overfetch_for_exclude,
    resolve_anchor_containment,
    resolve_exclude_chunk_types,
)
from lib.retrieval.refusal import (
    REASON_LOW_CONFIDENCE,
    REASON_NOT_IN_COURSE_MODEL,
    RefusalPolicy,
    evaluate_confidence,
    resolve_policy,
    should_refuse,
)

__all__ = [
    "ENV_LIBRARY_WIDE",
    "DECISION_TYPE_LIBRARY_WIDE",
    "resolve_library_wide",
    "list_library_courses",
    "answer_library_question",
]

ENV_LIBRARY_WIDE = "ED4ALL_ANSWER_LIBRARY_WIDE"
DECISION_TYPE_LIBRARY_WIDE = "grounded_answer_library_wide"
DECISION_PHASE = "libv2-answer"

_TRUTHY = {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Flag + course-set resolution
# --------------------------------------------------------------------------- #


def resolve_library_wide(explicit: Optional[bool] = None) -> bool:
    """Resolve the library-wide toggle (``ED4ALL_ANSWER_LIBRARY_WIDE``).

    Default OFF. An explicit ``bool`` argument wins (the CLI/GUI flag). Otherwise
    read the env var truthily (``1``/``true``/``yes``/``on``). Parse-with-
    fallback: anything else (unset, ``0``, garbage) → off, mirroring the other
    answer-path knobs.
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(ENV_LIBRARY_WIDE)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def _course_has_retrievable_corpus(course_dir: Path) -> bool:
    """True when a course dir carries a chunkset the retriever can read.

    Guards the enumeration against catalog entries whose course was removed or
    never archived a chunkset (so we never widen the union to a course that
    would just error).
    """
    # Dual-READ: the canonical ``semantik_chunks`` dir is preferred over the
    # legacy ``dart_chunks`` fallback.
    for sub in ("semantik_chunks", "dart_chunks", "imscc_chunks", "corpus"):
        if (course_dir / sub / "chunks.jsonl").is_file():
            return True
    return False


def list_library_courses(
    libv2_root: Path, home_slug: str, *, explicit: Optional[Sequence[str]] = None
) -> List[str]:
    """Resolve the ordered, de-duplicated course set for a library-wide ask.

    Resolution order:

    * ``explicit`` (an operator-supplied slug list) wins verbatim — filtered to
      courses that actually carry a retrievable chunkset, with ``home_slug``
      always included first.
    * else the W0.8 master catalog (``catalog/master_catalog.json`` via
      :func:`LibV2.tools.libv2.catalog.load_master_catalog`) — every catalog
      slug carrying a retrievable chunkset.
    * else a filesystem scan of ``courses/*/`` (fail-open when the catalog is
      absent — a fresh LibV2 may have courses but no built catalog yet).

    ``home_slug`` is ALWAYS first in the returned list (it is the course the ask
    was launched from, and the fail-open single-course target). Ordering after
    it is deterministic (catalog order, else sorted slug). Never raises — any
    enumeration error degrades to ``[home_slug]`` (single-course fail-open).
    """
    courses_root = libv2_root / "courses"

    def _keep(slug: str) -> bool:
        return bool(slug) and _course_has_retrievable_corpus(courses_root / slug)

    ordered: List[str] = []
    seen: set = set()

    def _add(slug: str, *, force: bool = False) -> None:
        slug = str(slug or "").strip()
        if not slug or slug in seen:
            return
        if force or _keep(slug):
            ordered.append(slug)
            seen.add(slug)

    # home first (force — the fail-open target is the home course even if its
    # chunkset guard is momentarily unreadable; downstream retrieval fail-opens).
    _add(home_slug, force=True)

    try:
        if explicit is not None:
            for slug in explicit:
                _add(slug)
            return ordered

        catalog_slugs = _catalog_slugs(libv2_root)
        if catalog_slugs is not None:
            for slug in catalog_slugs:
                _add(slug)
            return ordered

        # Filesystem fallback (no catalog): scan course dirs.
        if courses_root.is_dir():
            for child in sorted(courses_root.iterdir()):
                if child.is_dir():
                    _add(child.name)
    except Exception:  # noqa: BLE001 — enumeration never fails the ask
        return ordered or [str(home_slug)]
    return ordered or [str(home_slug)]


def _catalog_slugs(libv2_root: Path) -> Optional[List[str]]:
    """Slugs from the W0.8 master catalog, or ``None`` when it is absent.

    ``None`` (not ``[]``) signals "no catalog" so the caller falls through to the
    filesystem scan; an empty catalog returns ``[]`` (an explicit empty set).
    """
    try:
        from LibV2.tools.libv2.catalog import load_master_catalog

        catalog = load_master_catalog(libv2_root)
    except Exception:  # noqa: BLE001 — catalog absence is expected; fall through
        return None
    if catalog is None:
        return None
    slugs: List[str] = []
    for entry in getattr(catalog, "courses", []) or []:
        slug = getattr(entry, "slug", None)
        if slug:
            slugs.append(str(slug))
    return slugs


# --------------------------------------------------------------------------- #
# Per-course chunkset kind (citation gate must resolve each course correctly)
# --------------------------------------------------------------------------- #


def _chunkset_kind_for(libv2_root: Path, slug: str, engine: str) -> str:
    """Resolve the chunkset kind for one course (index manifest, else dir)."""
    kind: Optional[str] = None
    if engine in ("semantic", "hybrid-rrf"):
        kind = _vector_index_chunkset_kind(libv2_root, slug)
    if kind is None:
        kind = _infer_chunkset_kind(libv2_root, slug)
    return kind


# --------------------------------------------------------------------------- #
# Union retrieval
# --------------------------------------------------------------------------- #


def _retrieve_union(
    libv2_root: Path,
    slugs: Sequence[str],
    query: str,
    *,
    engine: str,
    limit: int,
    exclude_types: Optional[frozenset] = None,
) -> Tuple[List[RetrievedPassage], Dict[str, int], List[str], int]:
    """Retrieve per course, stamp provenance, return the score-sorted union.

    Returns ``(passages, per_course_counts, failed_slugs, excluded_count)``. A
    course whose retrieval raises is SKIPPED (recorded in ``failed_slugs``) — a
    per-course error never fails the union (fail-open). Each surviving passage is
    stamped with its ``course_slug``. The union is stable-sorted by descending
    score.

    FIX A — when ``exclude_types`` is non-empty every course over-fetches
    ``overfetch_for_exclude(limit, ...)`` and drops candidates whose
    ``chunk_type`` is excluded BEFORE they enter the union (so the global trim
    downstream stays full); ``excluded_count`` totals the drops. When empty the
    fetch count + set are byte-identical to the legacy path.
    """
    exclude_types = exclude_types or frozenset()
    fetch_limit = overfetch_for_exclude(limit, exclude_types)
    passages: List[RetrievedPassage] = []
    per_course_counts: Dict[str, int] = {}
    failed: List[str] = []
    excluded_count = 0
    for slug in slugs:
        try:
            results = _retrieve(
                libv2_root, slug, query, engine=engine, limit=fetch_limit
            )
        except Exception:  # noqa: BLE001 — skip this course, keep the union
            failed.append(slug)
            continue
        n = 0
        for r in results:
            if exclude_types and _result_chunk_type(r) in exclude_types:
                excluded_count += 1
                continue
            p = RetrievedPassage.from_retrieval_result(r, engine=engine)
            passages.append(_dc_replace(p, course_slug=slug))
            n += 1
        per_course_counts[slug] = n
    # Stable sort: Python's sort is stable, so ties keep per-course retrieval
    # order (which is per-course rank order). Descending score.
    passages.sort(key=lambda p: -float(p.score))
    return passages, per_course_counts, failed, excluded_count


# --------------------------------------------------------------------------- #
# Per-course citation gate (each passage resolves against its own course dir)
# --------------------------------------------------------------------------- #


def _run_library_citation_gate(
    cited_passages: Sequence[RetrievedPassage],
    libv2_root: Path,
    *,
    chunkset_kind_by_course: Dict[str, str],
    answer_text: Optional[str],
    containment_threshold: float,
) -> Tuple[List[Citation], List[str], List[Tuple[str, str]]]:
    """Resolve every cited passage's anchor against ITS OWN course dir.

    Mirrors ``grounded_answer._run_citation_gate`` but routes each passage to
    ``courses/<passage.course_slug>/`` + that course's chunkset kind, and stamps
    the resulting :class:`Citation` with ``course_slug`` for provenance. A
    passage whose anchor is unresolved blocks emission (same fail-closed
    contract as the single-course gate).
    """
    citations: List[Citation] = []
    blocked: List[str] = []
    statuses: List[Tuple[str, str]] = []
    for passage in cited_passages:
        slug = passage.course_slug or ""
        course_dir = libv2_root / "courses" / slug
        chunkset_kind = chunkset_kind_by_course.get(slug) or "dart"
        anchor = resolve_citation_anchor(
            _passage_chunk_record(passage),
            course_dir,
            chunkset_kind=chunkset_kind,
            containment_threshold=containment_threshold,
        )
        statuses.append((passage.chunk_id, anchor.status.value))
        if anchor.status not in _RESOLVED_STATUSES:
            blocked.append(passage.chunk_id)
        citation = _build_citation(passage, anchor, answer_text)
        citations.append(_dc_replace(citation, course_slug=slug or None))
    return citations, blocked, statuses


# --------------------------------------------------------------------------- #
# Decision capture
# --------------------------------------------------------------------------- #


def _emit_library_wide(
    capture: Optional[Any],
    *,
    query_sha: str,
    home_slug: str,
    engine: str,
    slugs: Sequence[str],
    per_course_counts: Dict[str, int],
    failed_slugs: Sequence[str],
    union_size: int,
    kept: int,
    outcome: str,
    contributing: Sequence[str],
) -> None:
    """One capture per library-wide ask (dynamic, replayable rationale)."""
    if capture is None:
        return
    counts_str = ", ".join(
        f"{s}:{per_course_counts.get(s, 0)}" for s in slugs
    ) or "none"
    contrib_str = ", ".join(contributing) if contributing else "none"
    failed_str = ", ".join(failed_slugs) if failed_slugs else "none"
    rationale = (
        f"library-wide ask {outcome} for query {query_sha} "
        f"(home={home_slug}, engine={engine}); "
        f"courses={len(slugs)} [{counts_str}] "
        f"contributing=[{contrib_str}] failed=[{failed_str}]; "
        f"union_size={union_size} kept={kept}"
    )
    _safe_log(
        capture,
        decision_type=DECISION_TYPE_LIBRARY_WIDE,
        decision=f"library_wide:{outcome}",
        rationale=rationale,
        context=f"courses={len(slugs)} kept={kept}",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def answer_library_question(
    repo_root: Path,
    course_slug: str,
    query: str,
    *,
    engine: str = "lexical",
    limit: int = 8,
    client: Optional[Any] = None,
    course_slugs: Optional[Sequence[str]] = None,
    library_wide: Optional[bool] = None,
    refusal_policy: Optional[RefusalPolicy] = None,
    validate_citations: bool = True,
    with_groundedness: bool = False,
    capture: Optional[Any] = None,
    containment_threshold: Optional[float] = None,
    max_passages: int = 8,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> GroundedAnswer:
    """Answer a question grounded across the WHOLE library (opt-in), else one course.

    When library-wide mode is off (default) OR the course set resolves to a
    single course, this delegates verbatim to
    :func:`~lib.retrieval.grounded_answer.answer_course_question` on
    ``course_slug`` (the byte-identical single-course path). Otherwise it unions
    retrieval across the resolved course set, composes + citation-gates the
    answer, and returns a :class:`GroundedAnswer` whose every citation carries
    its source ``course_slug``.

    ``library_wide`` (explicit bool) wins over the ``ED4ALL_ANSWER_LIBRARY_WIDE``
    env; ``course_slugs`` (explicit list) wins over catalog enumeration.
    """
    start = time.monotonic()
    query_sha = _query_sha(query)
    libv2_root = _libv2_root(repo_root)

    # FIX B — resolve the citation-gate anchor-containment floor once (explicit
    # caller value wins; unset resolves ED4ALL_ANSWER_ANCHOR_CONTAINMENT, default
    # 0.85 → byte-identical). The resolved float is passed down the single-course
    # delegations AND used by the union gate below, so both paths share one floor.
    containment_threshold = resolve_anchor_containment(containment_threshold)
    # FIX A — the retrieval chunk_type exclusion set for the union path.
    exclude_types = resolve_exclude_chunk_types()

    enabled = resolve_library_wide(library_wide)

    # OFF (and no explicit course list) → single-course path, byte-identical.
    if not enabled and course_slugs is None:
        return answer_course_question(
            repo_root,
            course_slug,
            query,
            engine=engine,
            limit=limit,
            client=client,
            refusal_policy=refusal_policy,
            validate_citations=validate_citations,
            with_groundedness=with_groundedness,
            capture=capture,
            containment_threshold=containment_threshold,
            max_passages=max_passages,
            on_progress=on_progress,
        )

    slugs = list_library_courses(libv2_root, course_slug, explicit=course_slugs)

    # <= 1 course → nothing to union: fail-open to single-course.
    if len(slugs) <= 1:
        _emit_library_wide(
            capture, query_sha=query_sha, home_slug=course_slug, engine=engine,
            slugs=slugs, per_course_counts={}, failed_slugs=[], union_size=0,
            kept=0, outcome="fail_open_single_course", contributing=[],
        )
        return answer_course_question(
            repo_root, course_slug, query, engine=engine, limit=limit,
            client=client, refusal_policy=refusal_policy,
            validate_citations=validate_citations,
            with_groundedness=with_groundedness, capture=capture,
            containment_threshold=containment_threshold, max_passages=max_passages,
            on_progress=on_progress,
        )

    # 1) Union retrieval (per-course fail-open + provenance stamping).
    union, per_course_counts, failed, excluded_count = _retrieve_union(
        libv2_root, slugs, query, engine=engine, limit=limit,
        exclude_types=exclude_types,
    )
    contributing = sorted({p.course_slug for p in union if p.course_slug})

    # FIX A audit trail — how many candidates the exclusion filter dropped across
    # the union (empty on the default path); rides the answered/blocked warnings.
    union_warnings: List[str] = []
    if excluded_count:
        union_warnings.append(
            f"excluded_chunk_types:{','.join(sorted(exclude_types))}:{excluded_count}"
        )

    # Retrieval degraded to <= 1 contributing course → fail-open single-course.
    if len(contributing) <= 1:
        _emit_library_wide(
            capture, query_sha=query_sha, home_slug=course_slug, engine=engine,
            slugs=slugs, per_course_counts=per_course_counts, failed_slugs=failed,
            union_size=len(union), kept=0, outcome="fail_open_single_course",
            contributing=contributing,
        )
        return answer_course_question(
            repo_root, course_slug, query, engine=engine, limit=limit,
            client=client, refusal_policy=refusal_policy,
            validate_citations=validate_citations,
            with_groundedness=with_groundedness, capture=capture,
            containment_threshold=containment_threshold, max_passages=max_passages,
            on_progress=on_progress,
        )

    # 2) Trim to `limit` (always-keep >= 1).
    passages = union[: max(1, int(limit))]

    _emit_library_wide(
        capture, query_sha=query_sha, home_slug=course_slug, engine=engine,
        slugs=slugs, per_course_counts=per_course_counts, failed_slugs=failed,
        union_size=len(union), kept=len(passages), outcome="unioned",
        contributing=contributing,
    )

    # 3) Refusal policy. Library-wide unions across courses that may carry
    #    DIFFERENT embedding models, so a per-model cosine pin is meaningless
    #    for the mixed distribution — resolve the model-agnostic default per
    #    engine (embedding_model_id=None). An explicit override wins verbatim.
    policy = refusal_policy or resolve_policy(engine, None)

    verdict = evaluate_confidence(passages, policy)
    confidence_payload = verdict.to_dict()
    refusal_check = should_refuse(passages, policy)

    # L4 passages-first disclosure (union path parity): surface the unioned
    # passages + refusal short-circuit ONCE, after the gate, before compose.
    _invoke_progress(
        on_progress,
        passages,
        refused=refusal_check.refuse,
        status=STATUS_REFUSED_LOW_CONFIDENCE if refusal_check.refuse else None,
    )

    if refusal_check.refuse:
        return _refused(
            status=STATUS_REFUSED_LOW_CONFIDENCE,
            query=query,
            course_slug=course_slug,
            engine=engine,
            refusal=refusal_check.to_dict(),
            confidence=confidence_payload,
            model_id=None,
            prompt_version=None,
            start=start,
        )

    # 4) Compose (THE loopback-enforced LLM call site, owned by the composer).
    if client is None:
        from lib.retrieval.answer_backend import build_answer_client

        client = build_answer_client(capture=capture)

    composed: ComposedAnswer = compose_answer(
        query,
        passages,
        client=client,
        capture=capture,
        course_code=course_slug,
        max_passages=max_passages,
    )

    if composed.not_in_course:
        model_refusal = {
            "refuse": True,
            "reason_code": REASON_NOT_IN_COURSE_MODEL,
            "signals": dict(verdict.signals),
            "policy_version": policy.policy_version,
            "engine": engine,
            "embedding_model_id": policy.embedding_model_id,
        }
        return _refused(
            status=STATUS_REFUSED_NOT_IN_COURSE,
            query=query,
            course_slug=course_slug,
            engine=engine,
            refusal=model_refusal,
            confidence=confidence_payload,
            model_id=composed.model_id,
            prompt_version=composed.prompt_version,
            start=start,
        )

    if not composed.cited_chunk_ids:
        return _blocked(
            status=STATUS_BLOCKED_INVALID_CITATION,
            query=query,
            course_slug=course_slug,
            engine=engine,
            confidence=confidence_payload,
            model_id=composed.model_id,
            prompt_version=composed.prompt_version,
            warnings=["empty_citations"],
            start=start,
        )

    by_id = {p.chunk_id: p for p in passages}
    cited_passages = [
        by_id[cid] for cid in composed.cited_chunk_ids if cid in by_id
    ]

    # Per-course chunkset kinds for the citation gate.
    chunkset_kind_by_course = {
        slug: _chunkset_kind_for(libv2_root, slug, engine) for slug in contributing
    }

    # 5) Citation gate — each cited passage resolves against its OWN course dir.
    if not validate_citations:
        citations = []
        for p in cited_passages:
            slug = p.course_slug or ""
            anchor = resolve_citation_anchor(
                _passage_chunk_record(p),
                libv2_root / "courses" / slug,
                chunkset_kind=chunkset_kind_by_course.get(slug) or "dart",
                containment_threshold=containment_threshold,
            )
            citations.append(
                _dc_replace(
                    _build_citation(p, anchor, composed.answer_text),
                    course_slug=slug or None,
                )
            )
        return _answered(
            query=query,
            course_slug=course_slug,
            engine=engine,
            answer_text=composed.answer_text,
            citations=citations,
            confidence=confidence_payload,
            model_id=composed.model_id,
            prompt_version=composed.prompt_version,
            groundedness=None,
            warnings=list(union_warnings),
            start=start,
        )

    citations, blocked, _statuses = _run_library_citation_gate(
        cited_passages,
        libv2_root,
        chunkset_kind_by_course=chunkset_kind_by_course,
        answer_text=composed.answer_text,
        containment_threshold=containment_threshold,
    )

    if blocked:
        return _blocked(
            status=STATUS_BLOCKED_CITATION_GATE,
            query=query,
            course_slug=course_slug,
            engine=engine,
            confidence=confidence_payload,
            model_id=composed.model_id,
            prompt_version=composed.prompt_version,
            warnings=list(union_warnings)
            + [f"blocked_citation:{cid}" for cid in blocked],
            start=start,
        )

    return _answered(
        query=query,
        course_slug=course_slug,
        engine=engine,
        answer_text=composed.answer_text,
        citations=citations,
        confidence=confidence_payload,
        model_id=composed.model_id,
        prompt_version=composed.prompt_version,
        groundedness=None,
        warnings=list(union_warnings),
        start=start,
        status=STATUS_ANSWERED,
    )
