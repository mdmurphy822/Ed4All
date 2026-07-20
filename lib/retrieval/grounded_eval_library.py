"""Library-wide grounded-eval slice (E2 reachability).

The single-course grounded eval (:func:`lib.retrieval.grounded_eval.run_grounded_eval`)
drives ``answer_course_question`` only — so ``answer_library_question`` (the
cross-course grounded ask behind ``ED4ALL_ANSWER_LIBRARY_WIDE``) shipped with
ZERO harness coverage: structurally unreachable. This module closes that gap
with a small cross-course question slice that actually CALLS
``answer_library_question`` and records, per question, whether it answered /
refused and which COURSES its citations came from (the cross-course signal the
single-course path can never produce).

Cleanly SKIPPED when the resolved library holds a single course (the union has
nothing to union): the slice returns ``{"skipped": True, "reason":
"single_course", ...}`` rather than a fabricated cross-course number. The
pipeline callable is import-guarded + injectable (offline tests pass
``answer_fn``); no fake results are ever fabricated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: Additive report-section schema; tracked independently of EVAL_SCHEMA_VERSION.
LIBRARY_SLICE_SCHEMA_VERSION = "1.0"

_ANSWERED_STATUSES = frozenset({"answered", "answered_with_warnings"})
_REFUSED_STATUSES = frozenset(
    {"refused_low_confidence", "refused_not_in_course"}
)


def _import_library_pipeline() -> Any:
    """Import-guard ``answer_library_question`` (mirrors the single-course guard).

    Raises ``ImportError`` (surfaced by the caller as a skipped section, never a
    fabricated result) when the library-wide pipeline is not importable.
    """
    from lib.retrieval.library_wide import answer_library_question

    return answer_library_question


def _resolve_course_slugs(
    repo_root: Path,
    home_slug: str,
    explicit: Optional[Sequence[str]],
) -> List[str]:
    """Resolve the library course set (explicit list wins; else catalog).

    Best-effort: any failure enumerating the catalog degrades to ``[home_slug]``
    (a single course → the slice skips), never an exception into the eval.
    """
    if explicit is not None:
        return [str(s) for s in explicit if str(s).strip()]
    try:
        from lib.retrieval.library_wide import (
            _libv2_root,
            list_library_courses,
        )

        return list(
            list_library_courses(_libv2_root(repo_root), home_slug, explicit=None)
        )
    except Exception:  # noqa: BLE001 — catalog unavailable → single-course skip
        return [home_slug]


def _citation_courses(answer: Any, home_slug: str) -> List[str]:
    """Distinct source course_slugs across an answer's citations (order-stable).

    A single-course answer's citations carry no ``course_slug`` → they attribute
    to ``home_slug``. The library-wide union path stamps each citation's source
    course, so a genuinely cross-course answer surfaces >1 course here.
    """
    cites = getattr(answer, "citations", None)
    if cites is None and isinstance(answer, dict):
        cites = answer.get("citations")
    seen: List[str] = []
    for c in cites or []:
        if isinstance(c, dict):
            slug = c.get("course_slug")
        else:
            slug = getattr(c, "course_slug", None)
        slug = str(slug) if slug else home_slug
        if slug not in seen:
            seen.append(slug)
    return seen


def _answer_status(answer: Any) -> str:
    val = getattr(answer, "status", None)
    if val is None and isinstance(answer, dict):
        val = answer.get("status")
    return str(val) if val else "unknown"


def run_library_eval(
    repo_root: Path,
    home_slug: str,
    questions: Sequence[Dict[str, Any]],
    *,
    engine: str = "semantic",
    limit: int = 8,
    client: Optional[Any] = None,
    with_groundedness: bool = False,
    capture: Optional[Any] = None,
    course_slugs: Optional[Sequence[str]] = None,
    answer_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run a small cross-course question slice through ``answer_library_question``.

    ``questions`` = ``[{"question_id", "question_text", ...}]`` — a compact,
    course-agnostic probe set (the caller owns authoring). For each question the
    library-wide pipeline is driven in LIBRARY-WIDE mode over the resolved course
    set; the section records answered / refused counts and how many answers drew
    citations from >1 course (the cross-course reach signal).

    Returns a report section dict. When the resolved library has <=1 course the
    section is ``{"skipped": True, "reason": "single_course"}`` — the honest
    "nothing to union" outcome, never a fabricated cross-course metric. When the
    library pipeline is unimportable the section skips with
    ``reason == "pipeline_unavailable"``.
    """
    slugs = _resolve_course_slugs(repo_root, home_slug, course_slugs)
    section: Dict[str, Any] = {
        "schema_version": LIBRARY_SLICE_SCHEMA_VERSION,
        "home_slug": home_slug,
        "engine": engine,
        "resolved_course_count": len(slugs),
        "resolved_courses": list(slugs),
    }
    if len(slugs) <= 1:
        section.update(
            {
                "skipped": True,
                "reason": "single_course",
                "n_questions": len(questions),
            }
        )
        return section

    if answer_fn is not None:
        pipeline = answer_fn
    else:
        try:
            pipeline = _import_library_pipeline()
        except Exception as exc:  # noqa: BLE001 — surfaced as a skip, never a crash
            section.update(
                {
                    "skipped": True,
                    "reason": "pipeline_unavailable",
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_questions": len(questions),
                }
            )
            return section

    rows: List[Dict[str, Any]] = []
    answered = 0
    refused = 0
    cross_course_answers = 0
    for q in questions:
        qid = str(q.get("question_id", ""))
        qtext = str(q.get("question_text", ""))
        answer = pipeline(
            repo_root,
            home_slug,
            qtext,
            engine=engine,
            limit=limit,
            client=client,
            course_slugs=list(slugs),
            library_wide=True,
            with_groundedness=with_groundedness,
            capture=capture,
        )
        status = _answer_status(answer)
        is_answered = status in _ANSWERED_STATUSES
        is_refused = status in _REFUSED_STATUSES
        if is_answered:
            answered += 1
        elif is_refused:
            refused += 1
        courses = _citation_courses(answer, home_slug) if is_answered else []
        if len(courses) > 1:
            cross_course_answers += 1
        rows.append(
            {
                "question_id": qid,
                "status": status,
                "answered": is_answered,
                "refused": is_refused,
                "citation_courses": courses,
            }
        )

    n = len(questions)
    section.update(
        {
            "skipped": False,
            "n_questions": n,
            "answered_count": answered,
            "refused_count": refused,
            "answer_rate": (answered / n) if n else None,
            "cross_course_answer_count": cross_course_answers,
            "cross_course_rate": (
                (cross_course_answers / answered) if answered else None
            ),
            "questions": rows,
            "_diagnostic": (
                "Library-wide reachability slice (E2): drives "
                "answer_library_question over a cross-course question set; "
                "cross_course_* count answers whose citations span >1 course. "
                "Diagnostic, NOT a pinned milestone."
            ),
        }
    )
    return section


__all__ = [
    "LIBRARY_SLICE_SCHEMA_VERSION",
    "run_library_eval",
]
