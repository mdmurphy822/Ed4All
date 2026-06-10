"""Fail-fast terminal-objective coverage gate (course_planning).

This gate is the "can't reoccur silently" guarantee for the
``UNCOVERED_TERMINAL_OUTCOME`` archival failure. It catches an
internally-inconsistent objectives set **at the ``course_planning`` phase**
— where the operator can act on it — instead of letting the inconsistency
flow all the way to ``libv2_archival`` and trip the critical
``packet_integrity_strict`` gate
(``lib/validators/libv2/packet_integrity.py::_rule_to_has_teaching_and_assessment``)
many phases later.

Failure mode it gates: a terminal objective (``TO-NN``) that downstream
content generation cannot cover, so no teaching / assessment chunk ever
resolves to it. Two shapes (see ``lib/ontology/terminal_coverage.py`` for
the canonical roll-up model):

* **CO-bearing course** — a terminal with **zero** chapter objectives
  rolling up to it via ``parent_to`` / ``parent_terminal`` / ``terminal_id``.
  Code ``ORPHAN_TERMINAL_NO_CHAPTER_OBJECTIVE``. This is fully checkable
  from the objectives set alone, so it is **critical**.

* **CO-less course** — each terminal is its own teaching unit and carries
  a ``chapter`` back-pointer into the textbook structure. A terminal
  whose ``chapter`` does not resolve to a real chapter (code
  ``ORPHAN_TERMINAL_NO_CHAPTER``), or that carries no chapter back-pointer
  at all (code ``ORPHAN_TERMINAL_NO_CHAPTER_REF``), is unreachable →
  **critical**. Only adjudicated when ``textbook_structure`` is available;
  absent → graceful-degrade pass (see below).

**Graceful-degrade contract** (mirrors ``ChapterObjectiveCoverageValidator``
and the ``EMBEDDING_DEPS_MISSING`` pattern). The gate returns a no-op
``passed=True`` when:

* the objectives set is empty (no terminals to cover — other gates own the
  empty-objectives case), OR
* the synthesized_objectives input is unresolvable (the gate has nothing
  to inspect).

A CO-less course with **no** textbook_structure available is *not* a hard
failure — the gate emits a warning-severity ``TERMINAL_COVERAGE_UNVERIFIED``
note and passes, because the roll-up signal is genuinely absent. This
preserves backward compatibility with legacy / reuse runs that may not
re-stage the structure, while still fail-fast-blocking the CO-bearing
orphan case (which needs no extra signal) and the CO-less case whenever
the structure IS present.

Likewise, a **CO-bearing** course where ZERO chapter objectives carry a
parent-terminal back-pointer (``parent_to`` / ``parent_terminal`` /
``parent_terminal_id`` / ``terminal_id``) is *not* a hard failure: the CO
roll-up model is not actually in use (the deterministic synthesizer, the
Stage-2 provider, and ``_normalize_objective_entry`` all emit COs with no
parent key), so terminal orphanhood is unprovable from the objectives set
alone. Adjudicating it would false-flag **every** terminal as an orphan, so
the gate emits the same ``TERMINAL_COVERAGE_UNVERIFIED`` warning + passes;
the libv2_archival backstop still catches genuinely-uncovered terminals.

Day-1 severity is **critical** for the orphan codes (block) per the
"must not reoccur" requirement.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.ontology.terminal_coverage import (
    find_orphan_terminals,
    is_co_bearing,
    rolled_up_terminal_ids,
)

logger = logging.getLogger(__name__)

#: Validator-capture decision_type — an EXISTING enum member.
_DECISION_TYPE: str = "content_structure_check"

#: Cap emitted issues so a uniformly-broken plan doesn't drown the report.
_ISSUE_LIST_CAP: int = 50


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
            "TerminalObjectiveCoverageValidator: "
            "decision_capture.log_decision failed: %s",
            exc,
        )


def _load_json(
    path_raw: Any, *, code_prefix: str
) -> Tuple[Optional[Any], Optional[GateIssue]]:
    """Load + parse a JSON file. Returns ``(data, error_issue)``."""
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


def _coerce_dict(
    inputs: Mapping[str, Any],
    *,
    explicit_key: str,
    path_key: str,
    code_prefix: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[GateIssue]]:
    """Resolve a dict input either inline or via a path key."""
    explicit = inputs.get(explicit_key)
    if explicit is not None:
        if isinstance(explicit, Mapping):
            return dict(explicit), None
        return None, GateIssue(
            severity="critical",
            code=f"{code_prefix}_MALFORMED",
            message=(
                f"inputs[{explicit_key!r}] is "
                f"{type(explicit).__name__!r}, not a mapping."
            ),
        )
    path_raw = inputs.get(path_key)
    if not path_raw:
        return None, None
    data, err = _load_json(path_raw, code_prefix=code_prefix)
    if err is not None:
        return None, err
    if not isinstance(data, Mapping):
        return None, GateIssue(
            severity="critical",
            code=f"{code_prefix}_MALFORMED",
            message=(
                f"{path_key} did not parse to a JSON object "
                f"(got {type(data).__name__!r})."
            ),
        )
    return dict(data), None


class TerminalObjectiveCoverageValidator:
    """Fail-fast terminal-coverage gate at ``course_planning``.

    Wired into ``textbook_to_course::course_planning::
    terminal_objective_coverage`` (critical / block). Guarantees no orphan
    terminal escapes planning so the ``UNCOVERED_TERMINAL_OUTCOME``
    archival failure cannot reoccur silently.
    """

    name = "terminal_objective_coverage"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

        objectives, err = _coerce_dict(
            inputs,
            explicit_key="synthesized_objectives",
            path_key="synthesized_objectives_path",
            code_prefix="TERMINAL_COVERAGE_OBJECTIVES",
        )
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        if objectives is None:
            # Nothing to inspect — graceful-degrade pass.
            _emit_decision(
                capture,
                decision="skip-with-pass: no synthesized_objectives input",
                rationale=(
                    "TerminalObjectiveCoverageValidator received no "
                    "synthesized_objectives (inline or path); there is "
                    "nothing to adjudicate, so it returns a no-op pass "
                    "per the graceful-degrade contract."
                ),
                context="objectives_absent=true",
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        structure, err = _coerce_dict(
            inputs,
            explicit_key="textbook_structure",
            path_key="textbook_structure_path",
            code_prefix="TERMINAL_COVERAGE_STRUCTURE",
        )
        # A malformed structure is downgraded to "absent" — the CO-less
        # branch graceful-degrades on absence rather than hard-failing on
        # a structure we couldn't read. Surface the read error as a
        # warning so the operator sees it.
        structure_read_warning: Optional[GateIssue] = None
        if err is not None:
            structure_read_warning = GateIssue(
                severity="warning",
                code="TERMINAL_COVERAGE_STRUCTURE_UNREADABLE",
                message=(
                    "textbook_structure could not be read "
                    f"({err.code}); CO-less terminal coverage will be "
                    "left unverified."
                ),
                location=err.location,
            )
            structure = None

        terminals = (
            objectives.get("terminal_objectives")
            or objectives.get("terminal_outcomes")
            or []
        )
        terminals = [t for t in terminals if isinstance(t, Mapping)]
        chapter_objectives = (
            objectives.get("chapter_objectives")
            or objectives.get("component_objectives")
            or []
        )

        # Empty objectives → nothing to cover. Other gates own the
        # empty-objectives case.
        if not terminals:
            _emit_decision(
                capture,
                decision="skip-with-pass: no terminal objectives",
                rationale=(
                    "TerminalObjectiveCoverageValidator found zero "
                    "terminal objectives in the synthesized set; there "
                    "are no terminals to cover, so it returns a no-op "
                    "pass (the empty-objectives case is owned by other "
                    "course_planning gates)."
                ),
                context="terminal_count=0",
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        co_bearing = is_co_bearing(chapter_objectives)
        issues: List[GateIssue] = []
        if structure_read_warning is not None:
            issues.append(structure_read_warning)

        # CO-less course with no structure → cannot adjudicate roll-up.
        # Emit a warning + pass (preserves legacy/reuse-run behaviour).
        if not co_bearing and structure is None:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="TERMINAL_COVERAGE_UNVERIFIED",
                    message=(
                        f"{len(terminals)} terminal objective(s) use the "
                        "CO-less model (no chapter objectives) and no "
                        "textbook_structure was available to verify each "
                        "terminal resolves to a real chapter. Terminal "
                        "coverage is LEFT UNVERIFIED at planning; the "
                        "libv2_archival packet_integrity_strict gate is "
                        "the terminal backstop."
                    ),
                    location="terminal_objectives",
                    suggestion=(
                        "Re-stage textbook_structure.json into the "
                        "course_planning gate inputs to enable fail-fast "
                        "CO-less coverage checks."
                    ),
                )
            )
            _emit_decision(
                capture,
                decision=(
                    "pass-with-warning: CO-less course, structure absent"
                ),
                rationale=(
                    f"TerminalObjectiveCoverageValidator could not verify "
                    f"coverage for {len(terminals)} CO-less terminal(s) "
                    f"because no textbook_structure was supplied. Passing "
                    f"with TERMINAL_COVERAGE_UNVERIFIED warning rather "
                    f"than blocking, to preserve backward compatibility "
                    f"with reuse/legacy runs; the archival gate remains "
                    f"the terminal backstop."
                ),
                context=(
                    f"co_bearing=false; structure_absent=true; "
                    f"terminals={len(terminals)}"
                ),
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=issues,
            )

        # CO-bearing course where ZERO chapter objectives carry a
        # parent-terminal back-pointer ⟹ the roll-up model is not actually
        # in use (deterministic synthesizer / Stage-2 provider /
        # _normalize_objective_entry all emit COs with no parent key). The
        # objectives set alone cannot prove any terminal is an orphan, so
        # adjudicating would mark EVERY terminal orphaned (false positive).
        # Mirror the CO-less-no-structure graceful path: emit a
        # TERMINAL_COVERAGE_UNVERIFIED warning + pass rather than silently
        # returning no orphans; the libv2_archival gate is the backstop.
        if co_bearing and not rolled_up_terminal_ids(chapter_objectives):
            issues.append(
                GateIssue(
                    severity="warning",
                    code="TERMINAL_COVERAGE_UNVERIFIED",
                    message=(
                        f"{len(terminals)} terminal objective(s) ship with "
                        "chapter objectives, but ZERO of those chapter "
                        "objectives carry a parent-terminal back-pointer "
                        "(parent_to/parent_terminal/parent_terminal_id/"
                        "terminal_id). The CO roll-up model is therefore not "
                        "in use, so terminal orphanhood is unprovable from "
                        "the objectives set alone. Terminal coverage is LEFT "
                        "UNVERIFIED at planning; the libv2_archival "
                        "packet_integrity_strict gate is the terminal "
                        "backstop."
                    ),
                    location="chapter_objectives",
                    suggestion=(
                        "Add a parent-terminal back-pointer "
                        "(parent_to/parent_terminal/terminal_id) to each "
                        "chapter objective to enable fail-fast roll-up "
                        "coverage checks."
                    ),
                )
            )
            _emit_decision(
                capture,
                decision=(
                    "pass-with-warning: CO-bearing course, no "
                    "parent-terminal back-pointers"
                ),
                rationale=(
                    f"TerminalObjectiveCoverageValidator could not verify "
                    f"roll-up coverage for {len(terminals)} terminal(s): the "
                    f"course is CO-bearing but ZERO chapter objectives carry "
                    f"a parent-terminal key, so the roll-up model is not in "
                    f"use and orphanhood is unprovable. Passing with "
                    f"TERMINAL_COVERAGE_UNVERIFIED rather than false-flagging "
                    f"every terminal; the archival gate is the backstop."
                ),
                context=(
                    f"co_bearing=true; rolled_up=0; "
                    f"terminals={len(terminals)}"
                ),
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=issues,
            )

        # Adjudicate orphans via the shared roll-up model.
        orphans = find_orphan_terminals(
            terminals,
            chapter_objectives,
            textbook_structure=structure,
        )

        for orphan in orphans[:_ISSUE_LIST_CAP]:
            reason = orphan.get("reason")
            tid = orphan.get("id")
            if reason == "no_chapter_objective":
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="ORPHAN_TERMINAL_NO_CHAPTER_OBJECTIVE",
                        message=(
                            f"Terminal objective {tid!r} has ZERO chapter "
                            "objectives rolling up to it (no CO carries a "
                            "parent_to/parent_terminal/terminal_id pointing "
                            f"at {tid!r}). Downstream content generation "
                            "will produce no teaching/assessment chunk for "
                            "it, which trips UNCOVERED_TERMINAL_OUTCOME at "
                            "libv2_archival."
                        ),
                        location=f"terminal_objectives[id={tid}]",
                        suggestion=(
                            "Either author ≥1 chapter objective whose "
                            f"parent terminal is {tid!r}, or remove the "
                            "orphan terminal so the objectives set is "
                            "internally consistent."
                        ),
                    )
                )
            elif reason == "chapter_not_in_structure":
                chap = orphan.get("chapter")
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="ORPHAN_TERMINAL_NO_CHAPTER",
                        message=(
                            f"Terminal objective {tid!r} is attributed to "
                            f"chapter {chap!r}, which is not present in the "
                            "textbook_structure. No content source backs "
                            "this terminal, so no teaching/assessment "
                            "chunk will cover it (UNCOVERED_TERMINAL_OUTCOME "
                            "at libv2_archival)."
                        ),
                        location=f"terminal_objectives[id={tid}]",
                        suggestion=(
                            f"Point {tid!r} at a chapter that exists in "
                            "textbook_structure, or remove the terminal."
                        ),
                    )
                )
            else:  # no_chapter_ref
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="ORPHAN_TERMINAL_NO_CHAPTER_REF",
                        message=(
                            f"Terminal objective {tid!r} carries neither a "
                            "chapter back-pointer nor any rolling-up "
                            "chapter objective, so it is unreachable by "
                            "content generation (UNCOVERED_TERMINAL_OUTCOME "
                            "at libv2_archival)."
                        ),
                        location=f"terminal_objectives[id={tid}]",
                        suggestion=(
                            f"Attribute {tid!r} to a chapter (set its "
                            "'chapter' field) or author a chapter objective "
                            "that rolls up to it."
                        ),
                    )
                )

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        action = "pass" if passed else "block"
        covered = len(terminals) - len(orphans)

        _emit_decision(
            capture,
            decision=(
                f"Audited terminal-objective coverage "
                f"({'CO-bearing' if co_bearing else 'CO-less'} model): "
                f"{covered}/{len(terminals)} terminal(s) covered; "
                f"{len(orphans)} orphan(s); passed={passed}."
            ),
            rationale=(
                f"TerminalObjectiveCoverageValidator ran the fail-fast "
                f"orphan-terminal check over {len(terminals)} terminal "
                f"objective(s) using the "
                f"{'CO roll-up' if co_bearing else 'chapter-resolution'} "
                f"model. Found {len(orphans)} orphan(s) "
                f"({critical_count} critical); each orphan would trip "
                f"UNCOVERED_TERMINAL_OUTCOME at libv2_archival, so the "
                f"gate {'blocks' if not passed else 'passes'} at "
                f"course_planning. passed={passed}."
            ),
            context=(
                f"co_bearing={co_bearing}; terminals={len(terminals)}; "
                f"orphans={len(orphans)}; critical={critical_count}; "
                f"passed={passed}"
            ),
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=(
                round(covered / len(terminals), 4) if terminals else 1.0
            ),
            issues=issues,
            action=action,
        )


__all__ = [
    "TerminalObjectiveCoverageValidator",
]
