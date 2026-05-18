"""Stage-2 chapter-objective coverage validator.

Three-stage large-LLM textbook synthesis architecture, Wave A — see
``plans/textbook-llm-synthesis-3stage-2026-05.md`` §8 row 3.

``ChapterObjectiveCoverageValidator`` gates the Stage-2 per-chapter
objective synthesis + reconciliation output produced by
``_plan_course_structure`` when ``TEXTBOOK_SYNTHESIS_PROVIDER`` is set.
Floor checks (plan §8):

1. every ``chapters[].id`` with non-empty text produced >=1 chapter
   objective (CO) — code ``CHAPTER_OBJ_MISSING``. Chapters listed in
   ``chapter_synthesis_failures`` are cross-checked: a failed chapter
   that produced no CO is the *expected* degraded state (§5.4 failure
   isolation), so it is surfaced as ``CHAPTER_OBJ_PARTIAL_COVERAGE``
   (informational warning) rather than ``CHAPTER_OBJ_MISSING``.
2. the reconciled ``terminal_objectives[]`` is non-empty — code
   ``RECONCILED_TO_EMPTY``.

**Graceful-degrade contract (the ``ABCD_MISSING`` pattern).** When the
Stage-2 enrichment signals are absent — a default-off run where
``TEXTBOOK_SYNTHESIS_PROVIDER`` is unset and the deterministic
``synthesize_objectives_from_topics`` ran — the validator returns a
no-op ``passed=True``. The Stage-2 signal that gates the check is the
``chapters[]`` list with ``chapter_text``: a default-off run does not
expose chapter-level coverage to audit, so the validator skips.

Day-1 severity is **warning**.

Inputs contract:

* ``inputs["textbook_structure"]`` / ``inputs["textbook_structure_path"]``
  — the source of ``chapters[]`` (each chapter dict carries ``id`` and,
  for Stage-2 runs, ``chapter_text``). Absence → graceful-degrade.
* ``inputs["synthesized_objectives"]`` /
  ``inputs["synthesized_objectives_path"]`` — the Stage-2 output dict
  (carries ``chapter_objectives`` + reconciled ``terminal_objectives``).
* ``inputs["chapter_synthesis_failures"]`` — optional list of chapter
  IDs whose per-chapter call degraded (§5.4). Used to downgrade a
  missing-CO finding to informational partial-coverage.
* ``inputs["decision_capture"]`` — optional ``DecisionCapture``; the
  validator emits one ``content_structure_check`` event per run.
* ``inputs["gate_id"]`` — optional gate-ID override.

Cross-references:

* ``lib/validators/abcd_objective.py`` — the ``_flatten_objectives``
  shape-tolerance rules are mirrored here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


#: Cap emitted issues so a uniformly-broken plan doesn't drown the
#: report.
_ISSUE_LIST_CAP: int = 50

#: Validator-capture decision_type — an EXISTING enum member.
_DECISION_TYPE: str = "content_structure_check"


def _emit_decision(
    capture: Any,
    *,
    decision: str,
    rationale: str,
    context: Optional[str] = None,
) -> None:
    """Emit one DecisionCapture event, swallowing capture-side errors."""
    if capture is None:
        return
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.warning(
            "ChapterObjectiveCoverageValidator: "
            "decision_capture.log_decision failed: %s",
            exc,
        )


def _load_json(
    path_raw: Any, *, code_prefix: str
) -> Tuple[Optional[Any], Optional[GateIssue]]:
    """Load + parse a JSON file from ``path_raw``.

    Returns ``(data, error_issue)``; ``error_issue`` is non-None only on
    a missing / unparseable path.
    """
    p = Path(path_raw)
    if not p.exists():
        return None, GateIssue(
            severity="critical",
            code=f"{code_prefix}_PATH_MISSING",
            message=f"{str(p)!r} does not exist.",
            location=str(p),
        )
    try:
        return json.loads(p.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, GateIssue(
            severity="critical",
            code=f"{code_prefix}_PATH_UNREADABLE",
            message=(
                f"Failed to parse {str(p)!r}: "
                f"{exc.__class__.__name__}: {exc}"
            ),
            location=str(p),
        )


def _coerce_structure(
    inputs: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[GateIssue]]:
    """Resolve the textbook_structure dict from ``inputs``."""
    explicit = inputs.get("textbook_structure")
    if explicit is not None:
        if isinstance(explicit, Mapping):
            return dict(explicit), None
        return None, GateIssue(
            severity="critical",
            code="CHAPTER_OBJ_STRUCTURE_MALFORMED",
            message=(
                f"inputs['textbook_structure'] is "
                f"{type(explicit).__name__!r}, not a mapping."
            ),
        )
    path_raw = inputs.get("textbook_structure_path")
    if not path_raw:
        return None, None
    data, err = _load_json(path_raw, code_prefix="CHAPTER_OBJ_STRUCTURE")
    if err is not None:
        return None, err
    if not isinstance(data, Mapping):
        return None, GateIssue(
            severity="critical",
            code="CHAPTER_OBJ_STRUCTURE_MALFORMED",
            message=(
                f"textbook_structure_path did not parse to a JSON "
                f"object (got {type(data).__name__!r})."
            ),
        )
    return dict(data), None


def _coerce_objectives(
    inputs: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[GateIssue]]:
    """Resolve the synthesized_objectives dict from ``inputs``."""
    explicit = inputs.get("synthesized_objectives")
    if explicit is not None:
        if isinstance(explicit, Mapping):
            return dict(explicit), None
        return None, GateIssue(
            severity="critical",
            code="CHAPTER_OBJ_OBJECTIVES_MALFORMED",
            message=(
                f"inputs['synthesized_objectives'] is "
                f"{type(explicit).__name__!r}, not a mapping."
            ),
        )
    path_raw = inputs.get("synthesized_objectives_path")
    if not path_raw:
        return None, None
    data, err = _load_json(path_raw, code_prefix="CHAPTER_OBJ_OBJECTIVES")
    if err is not None:
        return None, err
    if not isinstance(data, Mapping):
        return None, GateIssue(
            severity="critical",
            code="CHAPTER_OBJ_OBJECTIVES_MALFORMED",
            message=(
                f"synthesized_objectives_path did not parse to a JSON "
                f"object (got {type(data).__name__!r})."
            ),
        )
    return dict(data), None


def _flatten_chapter_objectives(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Pull a flat list of chapter-objective (CO) dicts.

    Tolerates ``chapter_objectives`` being a flat LO list OR a list of
    ``{"chapter": str, "objectives": [...]}`` groups (mirrors
    ``abcd_objective._flatten_objectives``).
    """
    out: List[Dict[str, Any]] = []
    for key in ("chapter_objectives", "component_objectives"):
        block = payload.get(key)
        if not isinstance(block, list):
            continue
        for entry in block:
            if not isinstance(entry, Mapping):
                continue
            if "objectives" in entry and isinstance(
                entry["objectives"], list
            ):
                out.extend(
                    o for o in entry["objectives"] if isinstance(o, Mapping)
                )
            else:
                out.append(dict(entry))
    return out


def _terminal_objectives(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Pull the reconciled terminal-objective list."""
    out: List[Dict[str, Any]] = []
    for key in ("terminal_objectives", "terminal_outcomes"):
        block = payload.get(key)
        if isinstance(block, list):
            out.extend(o for o in block if isinstance(o, Mapping))
    return out


def _chapter_co_ref(co: Mapping[str, Any]) -> Optional[str]:
    """Return the chapter ID a chapter-objective is attributed to.

    Tolerates the several keys a CO entry may carry the chapter
    back-pointer under.
    """
    for key in ("chapter_id", "chapterId", "chapter", "parent_chapter"):
        val = co.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # source_refs[].ref may carry the chapter ID (Stage-1 precedent).
    refs = co.get("source_refs")
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, Mapping):
                r = ref.get("ref")
                if isinstance(r, str) and r.strip():
                    return r.strip()
            elif isinstance(ref, str) and ref.strip():
                return ref.strip()
    return None


def _chapter_has_text(chapter: Mapping[str, Any]) -> bool:
    """Return True when a chapter dict carries non-empty chapter text."""
    text = chapter.get("chapter_text")
    return isinstance(text, str) and bool(text.strip())


class ChapterObjectiveCoverageValidator:
    """Stage-2 chapter-objective coverage gate.

    Wired into ``textbook_to_course::course_planning::
    chapter_objective_coverage`` (gate wiring is Wave D, not this wave).
    Skips-with-pass when the Stage-2 enrichment signals are absent so
    default-off runs are unaffected.
    """

    name = "chapter_objective_coverage"
    version = "0.1.0"  # Wave A — three-stage textbook synthesis

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

        structure, err = _coerce_structure(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )
        objectives, err = _coerce_objectives(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        # Graceful-degrade: a chapter list with chapter_text is the
        # Stage-2 signal. Absent → a default-off run produced no
        # chapter-level coverage to audit → no-op pass.
        chapters_raw = (structure or {}).get("chapters")
        chapters: List[Dict[str, Any]] = (
            [c for c in chapters_raw if isinstance(c, Mapping)]
            if isinstance(chapters_raw, list)
            else []
        )
        chapters_with_text = [c for c in chapters if _chapter_has_text(c)]

        if structure is None or objectives is None or not chapters_with_text:
            _emit_decision(
                capture,
                decision=(
                    "skip-with-pass: Stage-2 chapter-coverage signals "
                    "absent"
                ),
                rationale=(
                    "ChapterObjectiveCoverageValidator found no "
                    "chapters[] carrying `chapter_text` (and/or no "
                    "synthesized_objectives) — this is a default-off "
                    "run (TEXTBOOK_SYNTHESIS_PROVIDER unset) with no "
                    "Stage-2 chapter-level coverage to audit. Returning "
                    "a no-op pass per the ABCD_MISSING graceful-degrade "
                    "contract."
                ),
                context="stage2_signal_absent=true",
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        # Failed-chapter set (§5.4) — from inputs or the objectives dict.
        failures_raw = inputs.get("chapter_synthesis_failures")
        if not isinstance(failures_raw, list):
            failures_raw = objectives.get("chapter_synthesis_failures")
        failed_chapter_ids: set = {
            str(f).strip()
            for f in (failures_raw or [])
            if isinstance(f, (str,)) and str(f).strip()
        }

        chapter_objectives = _flatten_chapter_objectives(objectives)
        # Map each chapter ID → count of COs attributed to it.
        co_counts: Dict[str, int] = {}
        for co in chapter_objectives:
            ref = _chapter_co_ref(co)
            if ref:
                co_counts[ref] = co_counts.get(ref, 0) + 1

        issues: List[GateIssue] = []
        covered = 0

        for chapter in chapters_with_text:
            cid = chapter.get("id")
            if not isinstance(cid, str) or not cid.strip():
                continue
            cid = cid.strip()
            count = co_counts.get(cid, 0)
            if count > 0:
                covered += 1
                continue
            # Zero COs for a chapter with text.
            if cid in failed_chapter_ids:
                # Expected degraded state (§5.4) — informational only.
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="CHAPTER_OBJ_PARTIAL_COVERAGE",
                            message=(
                                f"Chapter {cid!r} produced no chapter "
                                f"objectives, but it is listed in "
                                f"chapter_synthesis_failures — this is "
                                f"the expected §5.4 per-chapter "
                                f"failure-isolation degraded state."
                            ),
                            location=f"chapters[id={cid}]",
                            suggestion=(
                                "Informational — re-run Stage 2 if this "
                                "chapter's objectives are needed."
                            ),
                        )
                    )
            else:
                # Unexpected gap — the chapter has text and did not
                # fail, yet produced no CO.
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="CHAPTER_OBJ_MISSING",
                            message=(
                                f"Chapter {cid!r} carries non-empty "
                                f"chapter_text and is not listed in "
                                f"chapter_synthesis_failures, yet "
                                f"produced 0 chapter objectives. "
                                f"Stage-2 coverage is incomplete."
                            ),
                            location=f"chapters[id={cid}]",
                            suggestion=(
                                "Stage 2 should emit >=1 CO per "
                                "chapter with text; investigate the "
                                "per-chapter synthesis call."
                            ),
                        )
                    )

        # Reconciled terminal objectives must be non-empty.
        terminal = _terminal_objectives(objectives)
        if not terminal:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="RECONCILED_TO_EMPTY",
                    message=(
                        "The reconciled terminal_objectives[] is empty "
                        "or absent; the §6 reconciliation step produced "
                        "no terminal objectives."
                    ),
                    location="terminal_objectives",
                    suggestion=(
                        "Re-run the reconciliation call — it should "
                        "emit at least one reconciled TO-NN."
                    ),
                )
            )

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        warning_count = sum(1 for i in issues if i.severity == "warning")

        _emit_decision(
            capture,
            decision=(
                f"Audited Stage-2 chapter-objective coverage: "
                f"{covered}/{len(chapters_with_text)} chapter(s) with "
                f"text produced >=1 CO; "
                f"{len(terminal)} reconciled terminal objective(s); "
                f"{warning_count} warning issue(s)."
            ),
            rationale=(
                f"ChapterObjectiveCoverageValidator ran the §8-row-3 "
                f"floor checks: every chapters[].id with non-empty "
                f"chapter_text produced >=1 CO (cross-checked against "
                f"{len(failed_chapter_ids)} chapter_synthesis_failures), "
                f"and the reconciled terminal_objectives[] is non-empty. "
                f"Coverage was {covered}/{len(chapters_with_text)}; "
                f"emitted {warning_count} warning issue(s); "
                f"passed={passed}."
            ),
            context=(
                f"chapters_with_text={len(chapters_with_text)}; "
                f"covered={covered}; "
                f"failed_chapters={len(failed_chapter_ids)}; "
                f"terminal={len(terminal)}; "
                f"warnings={warning_count}; passed={passed}"
            ),
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=(
                round(covered / len(chapters_with_text), 4)
                if chapters_with_text
                else 1.0
            ),
            issues=issues,
        )


__all__ = [
    "ChapterObjectiveCoverageValidator",
]
