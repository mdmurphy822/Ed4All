"""FR-COURSE-03 — CumulativeAssessmentValidator.

Cumulative-retrieval breadth floor. The framework's B14 final graded
assessment is the course-level cumulative-retrieval surface: it must span
the breadth of the course's terminal objectives, not collapse onto one or
two TOs. A final graded assessment that touches only a single TO (while the
course defines many) is a narrow end-of-unit quiz masquerading as a
cumulative final.

One signal, warning-day-1 with a deferred critical-flip:

  * ``CUMULATIVE_RETRIEVAL_TOO_NARROW`` — the final graded assessment (B14)
    spans < 2 terminal objectives WHILE the course defines >= 4 TOs.

Strict no-op when the course has < 4 TOs (a short course has too few TOs for
a "breadth" expectation to be meaningful; the gate would only generate noise).

Mirrors ``AssessmentObjectiveAlignmentValidator``'s input shape
(``assessments_path`` + ``synthesized_objectives_path``) and posture: reads
the assessment payload's questions, resolves each graded question's
``objective_id`` to its terminal objective, and counts distinct TOs covered.

CO->TO resolution prefers the synthesized objectives' explicit
``terminal_id`` mapping; falls back to the canonical ``TO-``/``CO-`` id
hierarchy (a TO id is its own TO; an unmapped CO contributes no TO). NO
embeddings — pure id resolution.

Inputs (``inputs`` dict):

    assessments_path / assessment_path: str
        Path to the synthesized assessments JSON (questions[].objective_id).

    synthesized_objectives_path: str
        Path to ``synthesized_objectives.json`` (terminal_objectives +
        chapter_objectives, the TO universe + CO->TO mapping). Required —
        without it the TO count cannot be established, so the gate skips.

    decision_capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance. One
        ``assessment_quality`` event per validate (dynamic signals:
        n_tos, n_graded_questions, n_tos_spanned, verdict).

    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to
        ``"cumulative_assessment"``).

References:
    - FRAMEWORK.pdf — B14 final graded assessment (cumulative retrieval).
    - ``lib/validators/assessment_objective_alignment.py`` — sibling
      assessment-surface validator (input shape mirrored here).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

#: Minimum TO count for the breadth floor to apply. Below this the gate is a
#: strict no-op (short course — no meaningful cumulative-breadth expectation).
MIN_TOS_FOR_BREADTH: int = 4

#: Minimum distinct TOs the final graded assessment must span (when the
#: course has >= MIN_TOS_FOR_BREADTH TOs).
MIN_TOS_SPANNED: int = 2

#: Question-type markers that count as GRADED (the cumulative-retrieval
#: surface). Discussion / reflection / practice items are excluded — they are
#: not part of the B14 final graded assessment.
_NON_GRADED_TYPES: frozenset = frozenset(
    {"discussion", "reflection", "reflection_prompt", "practice",
     "formative", "self_check", "self_check_question"}
)


# ---------------------------------------------------------------------------
# Objective-universe loading
# ---------------------------------------------------------------------------


def _load_json(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("cumulative_assessment: failed to load %s: %s", path, exc)
        return None


def _terminal_ids_and_map(obj_doc: Any) -> "tuple[Set[str], Dict[str, str]]":
    """Return ``(terminal_ids, co_to_to)`` from a synthesized objectives doc.

    Accepts both the Courseforge synthesized form (``terminal_objectives`` /
    ``chapter_objectives``) and the LibV2 archive form (``terminal_outcomes``
    / ``component_objectives``). ``co_to_to`` maps a chapter/component
    objective id -> its parent terminal id (via the explicit ``terminal_id``
    field when present).
    """
    terminal_ids: Set[str] = set()
    co_to_to: Dict[str, str] = {}
    if not isinstance(obj_doc, dict):
        return terminal_ids, co_to_to

    to_list = (
        obj_doc.get("terminal_objectives")
        or obj_doc.get("terminal_outcomes")
        or []
    )
    if isinstance(to_list, list):
        for t in to_list:
            if isinstance(t, dict) and t.get("id"):
                terminal_ids.add(str(t["id"]))

    co_list = (
        obj_doc.get("chapter_objectives")
        or obj_doc.get("component_objectives")
        or []
    )
    if isinstance(co_list, list):
        for c in co_list:
            if not isinstance(c, dict):
                continue
            cid = c.get("id")
            tid = c.get("terminal_id")
            if cid and tid:
                co_to_to[str(cid)] = str(tid)

    # Fall back to the flat learning_outcomes list to discover TO ids when the
    # split lists are absent.
    if not terminal_ids:
        for lo in obj_doc.get("learning_outcomes") or []:
            if not isinstance(lo, dict):
                continue
            lid = str(lo.get("id") or "")
            if lid.startswith("TO-") or lo.get("hierarchy_level") == "terminal":
                terminal_ids.add(lid)

    return terminal_ids, co_to_to


def _resolve_to(obj_id: str, terminal_ids: Set[str], co_to_to: Dict[str, str]) -> Optional[str]:
    """Resolve an objective id to its terminal objective id, or None."""
    if not obj_id:
        return None
    if obj_id in terminal_ids:
        return obj_id
    mapped = co_to_to.get(obj_id)
    if mapped and mapped in terminal_ids:
        return mapped
    if mapped:
        return mapped
    # Canonical-id fallback: a TO-prefixed id is its own TO.
    if obj_id.startswith("TO-"):
        return obj_id
    return None


# ---------------------------------------------------------------------------
# Assessment-question extraction (mirror of AssessmentObjectiveAlignment)
# ---------------------------------------------------------------------------


def _extract_questions(data: Any) -> List[Dict[str, Any]]:
    """Pull the flat list of questions from an assessments payload.

    Supports ``{"questions": [...]}``, ``{"assessments": [{"questions": ...}]}``
    and a plain list.
    """
    if isinstance(data, list):
        return [q for q in data if isinstance(q, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("questions"), list):
            return [q for q in data["questions"] if isinstance(q, dict)]
        out: List[Dict[str, Any]] = []
        for a in data.get("assessments") or []:
            if isinstance(a, dict) and isinstance(a.get("questions"), list):
                out.extend(q for q in a["questions"] if isinstance(q, dict))
        return out
    return []


def _is_graded(q: Dict[str, Any]) -> bool:
    qtype = str(q.get("type") or q.get("question_type") or "").strip().lower()
    if qtype in _NON_GRADED_TYPES:
        return False
    # Explicit graded=False marker excludes; default is graded.
    if q.get("graded") is False:
        return False
    return True


def _question_objective_ids(q: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    oid = q.get("objective_id")
    if oid:
        out.append(str(oid))
    for x in q.get("objective_ids") or []:
        if x:
            out.append(str(x))
    return out


# ---------------------------------------------------------------------------
# Decision-capture emit
# ---------------------------------------------------------------------------


def _emit_decision(
    capture: Optional[Any],
    *,
    n_tos: int,
    n_graded_questions: int,
    n_tos_spanned: int,
    applied: bool,
    passed: bool,
) -> None:
    if capture is None:
        return
    verdict = (
        "cumulative_breadth_skipped" if not applied
        else ("cumulative_breadth_ok" if passed else "cumulative_breadth_too_narrow")
    )
    rationale = (
        f"cumulative_assessment audit: course defines n_tos={n_tos} terminal "
        f"objectives; final graded assessment has n_graded_questions="
        f"{n_graded_questions} spanning n_tos_spanned={n_tos_spanned} distinct "
        f"TOs; breadth floor applied={applied} -> {verdict}. The B14 final "
        f"graded assessment must span >= {MIN_TOS_SPANNED} TOs when the course "
        f"defines >= {MIN_TOS_FOR_BREADTH} TOs."
    )
    ml_features = {
        "n_tos": int(n_tos),
        "n_graded_questions": int(n_graded_questions),
        "n_tos_spanned": int(n_tos_spanned),
        "breadth_floor_applied": bool(applied),
        "passed": bool(passed),
    }
    try:
        capture.log_decision(
            decision_type="assessment_quality",
            decision=verdict,
            rationale=rationale,
            ml_features=ml_features,
        )
    except Exception as exc:  # noqa: BLE001 - audit emit is best-effort
        logger.debug(
            "DecisionCapture.log_decision raised on cumulative_assessment "
            "assessment_quality: %s",
            exc,
        )


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------


class CumulativeAssessmentValidator:
    """FR-COURSE-03 — cumulative-retrieval breadth floor for the B14 final.

    Validator-protocol-compatible class. Flags
    ``CUMULATIVE_RETRIEVAL_TOO_NARROW`` (warning) when the final graded
    assessment spans < 2 TOs while the course has >= 4 TOs. Strict no-op
    (passed=True) when the course has < 4 TOs.
    """

    name = "cumulative_assessment"
    version = "0.1.0"

    def __init__(self, *, decision_capture: Optional[Any] = None) -> None:
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or self._decision_capture

        objectives_path = inputs.get("synthesized_objectives_path")
        obj_doc = _load_json(objectives_path)
        if obj_doc is None:
            # Cannot establish the TO universe → structured skip (warning, not
            # critical — the gate is advisory day-1).
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="warning",
                        code="OBJECTIVES_UNAVAILABLE",
                        message=(
                            "synthesized_objectives_path missing / unreadable; "
                            "cannot establish the terminal-objective universe "
                            "for the cumulative-breadth check (skipped)."
                        ),
                    )
                ],
                action=None,
            )

        terminal_ids, co_to_to = _terminal_ids_and_map(obj_doc)
        n_tos = len(terminal_ids)

        # Strict no-op when the course has fewer than MIN_TOS_FOR_BREADTH TOs.
        if n_tos < MIN_TOS_FOR_BREADTH:
            _emit_decision(
                capture,
                n_tos=n_tos,
                n_graded_questions=0,
                n_tos_spanned=0,
                applied=False,
                passed=True,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="info",
                        code="CUMULATIVE_BREADTH_NA",
                        message=(
                            f"course defines {n_tos} terminal objective(s) "
                            f"(< {MIN_TOS_FOR_BREADTH}); cumulative-breadth "
                            f"floor not applicable (no-op)."
                        ),
                    )
                ],
                action=None,
            )

        assessments_path = (
            inputs.get("assessments_path") or inputs.get("assessment_path")
        )
        data = _load_json(assessments_path)
        if data is None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="warning",
                        code="ASSESSMENTS_UNAVAILABLE",
                        message=(
                            "assessments_path missing / unreadable; cannot "
                            "audit cumulative-assessment breadth (skipped)."
                        ),
                    )
                ],
                action=None,
            )

        questions = _extract_questions(data)
        graded = [q for q in questions if _is_graded(q)]
        spanned: Set[str] = set()
        for q in graded:
            for oid in _question_objective_ids(q):
                to = _resolve_to(oid, terminal_ids, co_to_to)
                if to:
                    spanned.add(to)
        n_spanned = len(spanned)

        passed = n_spanned >= MIN_TOS_SPANNED
        _emit_decision(
            capture,
            n_tos=n_tos,
            n_graded_questions=len(graded),
            n_tos_spanned=n_spanned,
            applied=True,
            passed=passed,
        )

        if passed:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
                action=None,
            )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=False,
            score=round(n_spanned / MIN_TOS_SPANNED, 4),
            issues=[
                GateIssue(
                    severity="warning",
                    code="CUMULATIVE_RETRIEVAL_TOO_NARROW",
                    message=(
                        f"final graded assessment spans only {n_spanned} "
                        f"terminal objective(s) across {len(graded)} graded "
                        f"question(s), but the course defines {n_tos} TOs "
                        f"(>= {MIN_TOS_FOR_BREADTH}). A B14 cumulative final "
                        f"must span >= {MIN_TOS_SPANNED} TOs."
                    ),
                    location=str(assessments_path or ""),
                    suggestion=(
                        "Broaden the final graded assessment to draw questions "
                        "from multiple terminal objectives so it functions as "
                        "a cumulative-retrieval surface, not a single-unit "
                        "quiz."
                    ),
                )
            ],
            action="regenerate",
        )


__all__ = [
    "CumulativeAssessmentValidator",
    "MIN_TOS_FOR_BREADTH",
    "MIN_TOS_SPANNED",
]
