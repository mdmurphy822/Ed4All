"""Gold-set coverage report (retrieval-answer-eval-set §1.3).

Deterministic, read-only, stdlib + repo-internal deps only (no LLM, no
network). Given a loaded gold set + its pinned chunkset index, builds the §1.3
coverage matrix and surfaces WARNINGS (never failures) for the gaps the
operator authoring sprint must close:

  * **Per-type**: question counts by question_type.
  * **Per-week**: every content-bearing week should get >= 3 questions
    (``COVERAGE_WEEK_UNDERFILLED``). Weeks are derived from each relevant
    passage's ``item_path`` (``week_NN/...``) — the only week signal on a
    union corpus.
  * **Per-objective (TO/CO)**: every TO covered by >= 2 questions
    (``COVERAGE_TO_UNDERFILLED``); every CO with chunk support covered by
    >= 1 (``COVERAGE_CO_UNCOVERED``). A question's objectives are the union
    of its authored ``objective_refs`` and the ``learning_outcome_refs`` of
    its cited chunks. COs with zero chunk support are reported *uncoverable*,
    not violations.
  * **Per-population (union corpora)**: >= 10 ``source``, >= 25 ``course``,
    >= 5 ``both`` (``COVERAGE_POPULATION_UNDERFILLED``). A question's
    population is derived from its primary chunk(s): a chunk is ``source``
    when it carries ``source.source_references[]`` (it is grounded in /
    cites the original document) AND its ``item_path`` is the original
    document; ``course`` when it is a generated course page; ``both`` when a
    question's passages span both populations.
  * **Difficulty**: easy/medium/hard ~ 40/40/20
    (``COVERAGE_DIFFICULTY_SKEWED``), only flagged at scale (>= 25 questions)
    so a small seed set isn't dinged.

The objectives universe (which TO/CO ids exist + which are terminal vs
chapter) is read, in priority order, from: an explicit ``objectives``
mapping passed by the caller; else derived from the union chunkset's
``learning_outcome_refs`` (terminal = ``to-*`` / ``TO-*``; chapter =
``co-*`` / ``CO-*``). The derived universe is honest about a course that
ships no separate objectives doc (the openstax union case).

The report is emitted to ``<course_dir>/retrieval_eval/gold_coverage_<ts>.json``
by :func:`write_coverage_report` and mirrors the ``GoldSetIssue`` warning
shape used by the loader.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.retrieval.gold_set import RETRIEVAL_EVAL_SUBDIR, _load_chunks_by_id

# ---------------------------------------------------------------- constants

# Per-week / per-objective / per-population / difficulty thresholds (§1.3).
_WEEK_MIN_QUESTIONS = 3
_TO_MIN_QUESTIONS = 2
_CO_MIN_QUESTIONS = 1
_POPULATION_MIN = {"source": 10, "course": 25, "both": 5}
_DIFFICULTY_TARGET = {"easy": 0.40, "medium": 0.40, "hard": 0.20}
_DIFFICULTY_SCALE_GATE = 25     # only flag difficulty skew at scale
_DIFFICULTY_TOLERANCE = 0.15    # allowed absolute fraction deviation

_WEEK_RE = re.compile(r"week[_-]?0*(\d+)", re.IGNORECASE)
_TERMINAL_RE = re.compile(r"^to-?\d+$", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"^co-?\d+$", re.IGNORECASE)


# ---------------------------------------------------------------- issue type


@dataclass(frozen=True)
class CoverageWarning:
    """A single coverage-matrix gap. Always warning-severity (the coverage
    report never fails a load — it surfaces authoring gaps)."""

    code: str
    message: str
    severity: str = "warning"


COVERAGE_WARNING_CODES = frozenset(
    {
        "COVERAGE_WEEK_UNDERFILLED",
        "COVERAGE_TO_UNDERFILLED",
        "COVERAGE_CO_UNCOVERED",
        "COVERAGE_CO_UNCOVERABLE",
        "COVERAGE_POPULATION_UNDERFILLED",
        "COVERAGE_DIFFICULTY_SKEWED",
    }
)


@dataclass
class CoverageReport:
    """The §1.3 coverage matrix + warnings for one gold set."""

    course_slug: str
    total_questions: int
    by_type: Dict[str, int] = field(default_factory=dict)
    by_week: Dict[str, int] = field(default_factory=dict)
    by_population: Dict[str, int] = field(default_factory=dict)
    by_difficulty: Dict[str, int] = field(default_factory=dict)
    by_objective: Dict[str, int] = field(default_factory=dict)
    uncoverable_objectives: List[str] = field(default_factory=list)
    warnings: List[CoverageWarning] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "course_slug": self.course_slug,
            "generated_at": self.generated_at,
            "total_questions": self.total_questions,
            "matrix": {
                "by_type": self.by_type,
                "by_week": self.by_week,
                "by_population": self.by_population,
                "by_difficulty": self.by_difficulty,
                "by_objective": self.by_objective,
                "uncoverable_objectives": self.uncoverable_objectives,
            },
            "warnings": [
                {"code": w.code, "severity": w.severity, "message": w.message}
                for w in self.warnings
            ],
        }


# ---------------------------------------------------------------- helpers


def _week_of_item_path(item_path: str) -> Optional[str]:
    """Return the zero-padded week number string for a ``week_NN/...``
    item_path, or None when the path carries no week signal."""
    if not isinstance(item_path, str):
        return None
    m = _WEEK_RE.search(item_path)
    if not m:
        return None
    return f"{int(m.group(1)):02d}"


def _normalize_lo_id(lo: str) -> str:
    """Lowercase + collapse a learning-outcome id for set membership
    (``TO-01`` and ``to-01`` are the same objective)."""
    return (lo or "").strip().lower().replace("to-", "to-").replace("co-", "co-")


def _is_terminal(lo: str) -> bool:
    return bool(_TERMINAL_RE.match((lo or "").strip()))


def _is_chapter(lo: str) -> bool:
    return bool(_CHAPTER_RE.match((lo or "").strip()))


def _population_of_chunk(chunk: Dict[str, Any]) -> str:
    """Classify a chunk as ``source`` (original document) or ``course``
    (generated course page).

    A chunk is ``source`` when it carries ``source.source_references[]`` (it is
    grounded in / cites the original document) OR its ``item_path`` is the
    original-document HTML (a top-level ``*.html`` that is not a ``week_NN/``
    course page). Everything else is a generated ``course`` page.
    """
    source = chunk.get("source") or {}
    if source.get("source_references"):
        return "source"
    item_path = source.get("item_path") or ""
    # A union corpus's original-document chunks land at a flat ``*.html``
    # item_path; generated course pages live under ``week_NN/`` or
    # ``course_overview/``.
    if (
        item_path.endswith(".html")
        and "week_" not in item_path
        and "course_overview" not in item_path
        and "/" not in item_path
    ):
        return "source"
    return "course"


def _objectives_from_chunkset(
    chunks_by_id: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Derive the objectives universe from the chunkset's
    ``learning_outcome_refs``.

    Returns ``(level_by_lo, chunk_support_by_lo)`` where ``level_by_lo`` maps a
    normalized lo id to ``"terminal"`` / ``"chapter"`` and
    ``chunk_support_by_lo`` counts how many chunks reference each lo (so a CO
    with zero chunk support reports as *uncoverable*, not a violation).
    """
    level_by_lo: Dict[str, str] = {}
    chunk_support: Dict[str, int] = {}
    for chunk in chunks_by_id.values():
        for lo in chunk.get("learning_outcome_refs") or []:
            if not isinstance(lo, str):
                continue
            key = _normalize_lo_id(lo)
            if _is_terminal(lo):
                level_by_lo[key] = "terminal"
            elif _is_chapter(lo):
                level_by_lo[key] = "chapter"
            else:
                level_by_lo.setdefault(key, "chapter")
            chunk_support[key] = chunk_support.get(key, 0) + 1
    return level_by_lo, chunk_support


def _objectives_from_doc(
    objectives: Dict[str, Any]
) -> Dict[str, str]:
    """Read the level-by-lo map from a synthesized objectives doc (Courseforge
    form ``terminal_objectives[]`` / ``chapter_objectives[]`` or a flat
    ``learning_outcomes[]`` with ``hierarchy_level``)."""
    level_by_lo: Dict[str, str] = {}
    for lo in objectives.get("terminal_objectives") or []:
        lid = lo.get("id") if isinstance(lo, dict) else None
        if isinstance(lid, str):
            level_by_lo[_normalize_lo_id(lid)] = "terminal"
    for lo in objectives.get("chapter_objectives") or []:
        lid = lo.get("id") if isinstance(lo, dict) else None
        if isinstance(lid, str):
            level_by_lo[_normalize_lo_id(lid)] = "chapter"
    for lo in objectives.get("learning_outcomes") or []:
        if not isinstance(lo, dict):
            continue
        lid = lo.get("id")
        if not isinstance(lid, str):
            continue
        level = lo.get("hierarchy_level")
        key = _normalize_lo_id(lid)
        if level in ("terminal", "chapter"):
            level_by_lo[key] = level
        else:
            level_by_lo.setdefault(key, "terminal" if _is_terminal(lid) else "chapter")
    return level_by_lo


# ---------------------------------------------------------------- builder


def build_coverage_report(
    gold: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    objectives: Optional[Dict[str, Any]] = None,
    is_union: bool = True,
) -> CoverageReport:
    """Build the §1.3 coverage matrix + warnings (pure function — no I/O).

    Args:
        gold: a loaded gold-set doc.
        chunks_by_id: the pinned chunkset indexed by id.
        objectives: optional synthesized objectives doc; when omitted the
            objectives universe is derived from the chunkset.
        is_union: when True (default for the openstax union course) the
            per-population coverage slice is enforced; set False for a
            single-population (non-union) chunkset to skip the population
            warnings.
    """
    questions = [q for q in (gold.get("questions") or []) if isinstance(q, dict)]
    report = CoverageReport(
        course_slug=gold.get("course_slug", ""),
        total_questions=len(questions),
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # Objectives universe: explicit doc first, else derive from the chunkset.
    chunk_support: Dict[str, int] = {}
    if objectives:
        level_by_lo = _objectives_from_doc(objectives)
        _, chunk_support = _objectives_from_chunkset(chunks_by_id)
    else:
        level_by_lo, chunk_support = _objectives_from_chunkset(chunks_by_id)

    # ---- accumulate the matrix ----
    questions_per_lo: Dict[str, int] = {}
    for q in questions:
        qtype = q.get("question_type", "unknown")
        report.by_type[qtype] = report.by_type.get(qtype, 0) + 1

        if q.get("difficulty"):
            d = q["difficulty"]
            report.by_difficulty[d] = report.by_difficulty.get(d, 0) + 1

        # Weeks + populations come from the cited chunks' item_paths.
        weeks_seen = set()
        pops_seen = set()
        lo_seen = set()
        for ref in q.get("objective_refs") or []:
            if isinstance(ref, str):
                lo_seen.add(_normalize_lo_id(ref))
        for p in q.get("relevant_passages") or []:
            if not isinstance(p, dict):
                continue
            cid = p.get("chunk_id")
            anchor = p.get("anchor") or {}
            item_path = anchor.get("item_path") or ""
            week = _week_of_item_path(item_path)
            if week:
                weeks_seen.add(week)
            chunk = chunks_by_id.get(cid) if isinstance(cid, str) else None
            if chunk is not None:
                pops_seen.add(_population_of_chunk(chunk))
                # fall back to the chunk's own item_path for the week signal
                if not week:
                    cw = _week_of_item_path((chunk.get("source") or {}).get("item_path", ""))
                    if cw:
                        weeks_seen.add(cw)
                for lo in chunk.get("learning_outcome_refs") or []:
                    if isinstance(lo, str):
                        lo_seen.add(_normalize_lo_id(lo))

        for w in weeks_seen:
            report.by_week[w] = report.by_week.get(w, 0) + 1
        # population: 'both' when a question spans source + course.
        if is_union:
            if pops_seen == {"source", "course"} or pops_seen == {"source", "course", "both"}:
                report.by_population["both"] = report.by_population.get("both", 0) + 1
            elif "source" in pops_seen:
                report.by_population["source"] = report.by_population.get("source", 0) + 1
            elif "course" in pops_seen:
                report.by_population["course"] = report.by_population.get("course", 0) + 1
            # honor an explicit expected_citation_population override
            expected = q.get("expected_citation_population")
            if expected in ("source", "course", "both"):
                # the authored intent is authoritative for the slice count
                pass
        for lo in lo_seen:
            questions_per_lo[lo] = questions_per_lo.get(lo, 0) + 1

    report.by_objective = dict(sorted(questions_per_lo.items()))

    # ---- warnings ----
    _emit_week_warnings(report, chunks_by_id)
    _emit_objective_warnings(report, level_by_lo, chunk_support, questions_per_lo)
    if is_union:
        _emit_population_warnings(report)
    _emit_difficulty_warnings(report)

    return report


def _emit_week_warnings(
    report: CoverageReport, chunks_by_id: Dict[str, Dict[str, Any]]
) -> None:
    # Universe of content-bearing weeks = weeks present in the chunkset.
    content_weeks = set()
    for chunk in chunks_by_id.values():
        w = _week_of_item_path((chunk.get("source") or {}).get("item_path", ""))
        if w:
            content_weeks.add(w)
    for week in sorted(content_weeks):
        n = report.by_week.get(week, 0)
        if n < _WEEK_MIN_QUESTIONS:
            report.warnings.append(
                CoverageWarning(
                    code="COVERAGE_WEEK_UNDERFILLED",
                    message=(
                        f"week {week} has {n} question(s) (< {_WEEK_MIN_QUESTIONS}); "
                        f"every content-bearing week should get >= {_WEEK_MIN_QUESTIONS}."
                    ),
                )
            )


def _emit_objective_warnings(
    report: CoverageReport,
    level_by_lo: Dict[str, str],
    chunk_support: Dict[str, int],
    questions_per_lo: Dict[str, int],
) -> None:
    for lo, level in sorted(level_by_lo.items()):
        n = questions_per_lo.get(lo, 0)
        support = chunk_support.get(lo, 0)
        if level == "terminal":
            if n < _TO_MIN_QUESTIONS:
                report.warnings.append(
                    CoverageWarning(
                        code="COVERAGE_TO_UNDERFILLED",
                        message=(
                            f"terminal objective {lo} covered by {n} question(s) "
                            f"(< {_TO_MIN_QUESTIONS})."
                        ),
                    )
                )
        else:  # chapter
            if support == 0:
                report.uncoverable_objectives.append(lo)
                report.warnings.append(
                    CoverageWarning(
                        code="COVERAGE_CO_UNCOVERABLE",
                        message=(
                            f"chapter objective {lo} has zero chunk support; "
                            f"uncoverable (not a violation)."
                        ),
                    )
                )
            elif n < _CO_MIN_QUESTIONS:
                report.warnings.append(
                    CoverageWarning(
                        code="COVERAGE_CO_UNCOVERED",
                        message=(
                            f"chapter objective {lo} has chunk support but 0 "
                            f"questions (< {_CO_MIN_QUESTIONS})."
                        ),
                    )
                )


def _emit_population_warnings(report: CoverageReport) -> None:
    for pop, floor in _POPULATION_MIN.items():
        n = report.by_population.get(pop, 0)
        if n < floor:
            report.warnings.append(
                CoverageWarning(
                    code="COVERAGE_POPULATION_UNDERFILLED",
                    message=(
                        f"population '{pop}' has {n} question(s) (< {floor}); "
                        f"union-corpus per-population slice underfilled."
                    ),
                )
            )


def _emit_difficulty_warnings(report: CoverageReport) -> None:
    if report.total_questions < _DIFFICULTY_SCALE_GATE:
        return
    total_tagged = sum(report.by_difficulty.values())
    if total_tagged == 0:
        return
    for level, target in _DIFFICULTY_TARGET.items():
        frac = report.by_difficulty.get(level, 0) / total_tagged
        if abs(frac - target) > _DIFFICULTY_TOLERANCE:
            report.warnings.append(
                CoverageWarning(
                    code="COVERAGE_DIFFICULTY_SKEWED",
                    message=(
                        f"difficulty '{level}' is {frac:.0%} of tagged questions "
                        f"(target ~{target:.0%} +/- {_DIFFICULTY_TOLERANCE:.0%})."
                    ),
                )
            )


# ---------------------------------------------------------------- I/O entry


def write_coverage_report(
    course_dir: Path,
    gold: Dict[str, Any],
    *,
    objectives: Optional[Dict[str, Any]] = None,
    is_union: bool = True,
) -> Tuple[CoverageReport, Path]:
    """Build the coverage report against the gold set's pinned chunkset and
    write it to ``<course_dir>/retrieval_eval/gold_coverage_<ts>.json``.

    Returns ``(report, written_path)``. Raises ``FileNotFoundError`` only when
    the pinned chunkset is missing (the report needs the chunkset to derive
    weeks / populations / objectives).
    """
    course_dir = Path(course_dir)
    chunks_rel = (gold.get("chunkset") or {}).get("chunks_path") or ""
    chunks_path = course_dir / chunks_rel
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"pinned chunkset not found at {chunks_path}; cannot build coverage report."
        )
    chunks_by_id = _load_chunks_by_id(chunks_path)
    report = build_coverage_report(
        gold, chunks_by_id, objectives=objectives, is_union=is_union
    )

    out_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = out_dir / f"gold_coverage_{ts}.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report, out_path


__all__ = [
    "CoverageWarning",
    "CoverageReport",
    "COVERAGE_WARNING_CODES",
    "build_coverage_report",
    "write_coverage_report",
]
