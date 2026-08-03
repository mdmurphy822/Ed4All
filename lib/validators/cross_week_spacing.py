"""C3-6 — course-level CROSS-WEEK distributed-practice (spacing) gate.

The framework's §6.5 spacing / distributed-practice axis wants a substantive
concept/objective REVISITED across weeks — taught in one week, then retrieved /
applied / consolidated again in a LATER week — rather than MASSED entirely
inside a single week and never seen again. Spacing is the strongest known
durability lever; a concept that is taught + assessed + closed all within one
calendar week, with no spaced revisit, is the massed-practice anti-pattern.

This is the COURSE-level companion to two existing, distinct surfaces:

  * the WITHIN-module IB7.5a ``ED4ALL_PLANNER_SPACING`` planner pass
    (``lib/generation/block_planner.py::_apply_spacing``) — that pass separates
    a check from the exposition that taught the SAME CO *inside one module*; it
    says nothing about whether the concept reappears in a later WEEK.
  * ``course_level_qa``'s ``COURSE_RETRIEVAL_RHYTHM_THIN`` — that signal checks
    the PRESENCE of retrieval/check blocks per module (are checks distributed
    across modules at all), NOT whether any individual concept is distributed
    across weeks. A course can pass the rhythm check (every module has a check)
    while every concept is still massed into its own single week.

This validator COMPOSES already-computed REAL signals — it does NOT load a
model and it does NOT re-score blocks. For each substantive objective (or, when
no objective ids resolve, each concept tag), it computes the SET of distinct
weeks the concept's blocks appear in (from the canonical ``week_NN_*`` page-id
grouping that every other course-level gate reuses) and flags the concept when
it is taught + assessed entirely within ONE week with no spaced revisit in a
later week.

Anti-fabrication:

  * A concept is only eligible for the spacing flag when it carries ENOUGH
    blocks to warrant spacing (``MIN_BLOCKS_FOR_SPACING``) AND at least one of
    those is a retrieval/check/assessment block (a concept that is merely
    mentioned once is not "massed practice" — there is no practice to space).
  * Week info that cannot be resolved (no ``week_NN`` page-ids on any block)
    yields a structured SKIP (``WEEK_INFO_UNRESOLVABLE``), never an invented
    verdict — the gate cannot measure cross-week distribution without weeks.
  * The course must span ≥2 distinct weeks at all; a single-week course (or a
    course whose blocks all collapse to one module) has no cross-week axis to
    measure and skips with ``SINGLE_WEEK_COURSE``.

Signals / codes
---------------

  * ``CONCEPT_MASSED_SINGLE_WEEK`` (warning) — a substantive concept/objective
    (≥ ``MIN_BLOCKS_FOR_SPACING`` blocks, ≥1 of them a retrieval/check) is
    taught AND assessed entirely within ONE week and never revisited in a later
    week. Distributed practice (§6.5) wants it spaced across ≥2 weeks.
  * ``WEEK_INFO_UNRESOLVABLE`` (warning, structured skip) — no block carried a
    resolvable ``week_NN`` page-id; cross-week distribution cannot be measured.
  * ``SINGLE_WEEK_COURSE`` (info, structured skip) — the course spans a single
    week, so there is no cross-week axis to audit.
  * ``RUBRIC_DISABLED`` (info) — ``ED4ALL_BLOCK_QUALITY_RUBRIC`` unset → no-op.

Posture
-------

Warning-day-1 (``passed=True`` unconditionally) with a ``# TODO(calibration)``
deferred critical-flip — mirroring ``course_level_qa`` and every IB6 /
FR-COURSE gate. Byte-stable no-op when ``ED4ALL_BLOCK_QUALITY_RUBRIC`` is unset
(this is course-level QA, riding the same flag as its sibling
``course_level_qa``), returning a ``RUBRIC_DISABLED`` info issue and writing
nothing. This is a course-level MEASUREMENT over existing blocks — it adds NO
Block field and never mutates blocks.

References:
    - FRAMEWORK.pdf — §6.5 distributed practice / spacing axis.
    - ``lib/generation/block_planner.py::_apply_spacing`` (IB7.5a) — the
      within-module spacing pass this gate is the cross-week companion to.
    - ``lib/validators/course_level_qa.py`` — sibling FR-COURSE course-level
      gate (the ``week_NN`` module grouping + objectives loading + warning-day-1
      posture mirrored here).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

_ISSUE_LIST_CAP = 50

#: Minimum number of blocks touching a concept before it is eligible for the
#: spacing flag. A concept mentioned once or twice is not "massed practice" —
#: there is no body of practice to distribute. Conservative to keep the gate
#: precision-leaning (anti-fabrication: only flag concepts substantial enough to
#: warrant spacing).
MIN_BLOCKS_FOR_SPACING: int = 3

#: Framework B-codes that constitute a retrieval / check / assessment surface —
#: the "practice" half of distributed PRACTICE. A massed concept is one whose
#: practice is all in one week (B07 knowledge-check / B11 reflection /
#: B14 graded — mirrors course_level_qa's _RETRIEVAL_CODES).
_RETRIEVAL_CODES: frozenset = frozenset({"B07", "B11", "B14"})


class CrossWeekSpacingValidator:
    """C3-6 — course-level cross-week distributed-practice (spacing) gate."""

    name = "cross_week_spacing"
    version = "0.1.0"

    def __init__(self, *, decision_capture: Optional[Any] = None) -> None:
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        from lib.validators._block_rubric_helpers import (
            block_attr,
            block_quality_rubric_enabled,
            framework_block_of,
        )

        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or self._decision_capture

        enabled = inputs.get("rubric_enabled")
        if enabled is None:
            enabled = block_quality_rubric_enabled()
        if not enabled:
            # Byte-stable no-op (course-level QA rides the same flag as its
            # sibling course_level_qa).
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="RUBRIC_DISABLED",
                        message=(
                            "ED4ALL_BLOCK_QUALITY_RUBRIC unset — cross-week "
                            "spacing gate skipped (no composition, byte-stable)."
                        ),
                    )
                ],
                action=None,
                metadata={"cross_week_spacing": {"enabled": False}},
            )

        blocks = inputs.get("blocks") or []
        if not blocks:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="warning",
                        code="BLOCKS_UNAVAILABLE",
                        message=(
                            "no blocks surfaced; cannot audit cross-week "
                            "distributed practice (skipped)."
                        ),
                    )
                ],
                action=None,
                metadata={"cross_week_spacing": {"enabled": True, "n_blocks": 0}},
            )

        # ------------------------------------------------------------------
        # Build per-concept week + retrieval-presence aggregates (REAL signal).
        # A "concept" is an objective id when block objective ids resolve, else
        # a concept tag — both are existing per-block fields, no new field.
        # ------------------------------------------------------------------
        # concept -> set of distinct week numbers it appears in
        concept_weeks: Dict[str, Set[int]] = {}
        # concept -> total blocks touching it
        concept_block_count: Dict[str, int] = {}
        # concept -> set of distinct weeks carrying a retrieval/check for it
        concept_retrieval_weeks: Dict[str, Set[int]] = {}

        all_weeks: Set[int] = set()
        n_blocks_with_week = 0

        for block in blocks:
            week = _week_of(block, block_attr)
            if week is None:
                continue
            n_blocks_with_week += 1
            all_weeks.add(week)
            is_retrieval = framework_block_of(block) in _RETRIEVAL_CODES
            for concept in _concepts_of(block, block_attr):
                concept_weeks.setdefault(concept, set()).add(week)
                concept_block_count[concept] = (
                    concept_block_count.get(concept, 0) + 1
                )
                if is_retrieval:
                    concept_retrieval_weeks.setdefault(concept, set()).add(week)

        # ---- Structured skips (anti-fabrication) ---------------------------
        if n_blocks_with_week == 0:
            return self._skip(
                gate_id,
                "WEEK_INFO_UNRESOLVABLE",
                "warning",
                (
                    "no block carried a resolvable week_NN page-id; cross-week "
                    "distributed practice cannot be measured (skipped — week "
                    "info unresolvable)."
                ),
                n_blocks=len(blocks),
                capture=capture,
            )
        if len(all_weeks) < 2:
            return self._skip(
                gate_id,
                "SINGLE_WEEK_COURSE",
                "info",
                (
                    f"course spans a single week ({sorted(all_weeks)}); there is "
                    f"no cross-week axis to audit for distributed practice "
                    f"(skipped)."
                ),
                n_blocks=len(blocks),
                capture=capture,
                severity_passed=True,
            )

        # ---- The spacing audit ---------------------------------------------
        issues: List[GateIssue] = []
        massed_concepts: List[str] = []
        n_substantive = 0
        n_spaced = 0

        for concept in sorted(concept_block_count):
            n_blocks = concept_block_count[concept]
            weeks = concept_weeks.get(concept, set())
            retrieval_weeks = concept_retrieval_weeks.get(concept, set())

            # Eligibility (anti-fabrication): enough blocks to warrant spacing,
            # AND at least one retrieval/check (there must be PRACTICE to space).
            if n_blocks < MIN_BLOCKS_FOR_SPACING or not retrieval_weeks:
                continue
            n_substantive += 1

            if len(weeks) >= 2:
                # Distributed across ≥2 weeks → spaced. Pass.
                n_spaced += 1
                continue

            # Taught + assessed entirely within ONE week, no spaced revisit.
            massed_concepts.append(concept)
            if len(issues) < _ISSUE_LIST_CAP:
                only_week = next(iter(weeks)) if weeks else None
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="CONCEPT_MASSED_SINGLE_WEEK",
                        message=(
                            f"concept {concept!r} ({n_blocks} blocks, "
                            f"retrieval/check present) is taught AND assessed "
                            f"entirely within a single week "
                            f"(week {only_week}) with no spaced revisit in a "
                            f"later week; this is the massed-practice anti-"
                            f"pattern (framework §6.5 distributed practice). "
                            f"Cross-week spacing is distinct from the within-"
                            f"module IB7.5a spacing pass and from "
                            f"course_level_qa's per-module retrieval rhythm."
                        ),
                        location=str(only_week),
                        suggestion=(
                            f"Revisit {concept!r} with a low-stakes retrieval / "
                            f"applied block in a LATER week so its practice is "
                            f"distributed across ≥2 weeks rather than massed."
                        ),
                    )
                )

        self._emit_decision(
            capture,
            n_weeks=len(all_weeks),
            n_substantive=n_substantive,
            n_spaced=n_spaced,
            n_massed=len(massed_concepts),
            massed_concepts=massed_concepts,
            concept_axis="objective" if any(
                _concept_is_objective(c) for c in concept_block_count
            ) else "concept_tag",
            issue_count=len(issues),
        )

        # Warning-day-1: never blocks. ``passed`` stays True so the gate cannot
        # alter ``final_status`` until the calibration-deferred critical-flip.
        #
        # TODO(calibration): when scripts/harness/calibration_harness.py confirms the
        # course-level FP rate of CONCEPT_MASSED_SINGLE_WEEK on >=2 corpora,
        # flip the gate row to ``severity: critical`` + ``behavior.on_fail:
        # block`` in config/workflows.yaml AND set ``passed`` to
        # ``not massed_concepts`` so a massed-practice course blocks promotion
        # (framework §6.5 distributed practice). WS3/W4 deferred-flip pattern;
        # IB3 is the single documented fastest-flip exception, this is NOT IB3.
        massed_rate = (
            round(len(massed_concepts) / n_substantive, 4)
            if n_substantive
            else 0.0
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=round(1.0 - massed_rate, 4),
            issues=issues,
            action=None,
            metadata={
                "cross_week_spacing": {
                    "enabled": True,
                    "n_blocks": len(blocks),
                    "n_weeks": len(all_weeks),
                    "weeks": sorted(all_weeks),
                    "n_substantive_concepts": n_substantive,
                    "n_spaced_concepts": n_spaced,
                    "n_massed_concepts": len(massed_concepts),
                    "massed_concepts": massed_concepts[:_ISSUE_LIST_CAP],
                    "massed_rate": massed_rate,
                }
            },
        )

    # ------------------------------------------------------------------
    def _skip(
        self,
        gate_id: str,
        code: str,
        severity: str,
        message: str,
        *,
        n_blocks: int,
        capture: Any,
        severity_passed: bool = True,
    ) -> GateResult:
        """Return a structured-skip GateResult (always passed=True)."""
        self._emit_skip_decision(capture, code=code, n_blocks=n_blocks)
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=severity_passed,
            issues=[GateIssue(severity=severity, code=code, message=message)],
            action=None,
            metadata={
                "cross_week_spacing": {
                    "enabled": True,
                    "n_blocks": n_blocks,
                    "skipped": code,
                }
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        n_weeks: int,
        n_substantive: int,
        n_spaced: int,
        n_massed: int,
        massed_concepts: List[str],
        concept_axis: str,
        issue_count: int,
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="validation_result",
                decision=(
                    f"cross_week_spacing weeks={n_weeks}, "
                    f"substantive_concepts={n_substantive}, "
                    f"massed={n_massed}, spaced={n_spaced}"
                ),
                rationale=(
                    "C3-6 course-level §6.5 distributed-practice audit: composed "
                    f"the per-{concept_axis} week-distribution from the existing "
                    f"week_NN block grouping (NO model load). Course spans "
                    f"{n_weeks} weeks; of {n_substantive} substantive concepts "
                    f"(>= {MIN_BLOCKS_FOR_SPACING} blocks with a retrieval/check), "
                    f"{n_spaced} are spaced across >=2 weeks and {n_massed} are "
                    f"massed into a single week ({massed_concepts[:10]}). "
                    f"{issue_count} issue(s). Warning-day-1: enumerated as issues "
                    "but does not yet block (calibration-deferred critical-flip)."
                ),
                ml_features={
                    "n_weeks": int(n_weeks),
                    "n_substantive_concepts": int(n_substantive),
                    "n_spaced_concepts": int(n_spaced),
                    "n_massed_concepts": int(n_massed),
                    "issue_count": int(issue_count),
                },
            )
        except Exception as exc:  # noqa: BLE001 — audit emit is best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on cross_week_spacing: %s",
                exc,
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _emit_skip_decision(capture: Any, *, code: str, n_blocks: int) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="validation_result",
                decision=f"cross_week_spacing skipped ({code})",
                rationale=(
                    "C3-6 course-level §6.5 distributed-practice audit skipped "
                    f"with structured reason {code!r} over {n_blocks} block(s): "
                    "cross-week distribution is unmeasurable without resolvable "
                    "weeks / a multi-week course (anti-fabrication — no invented "
                    "verdict). Warning-day-1; never blocks."
                ),
                ml_features={"skip_code": code, "n_blocks": int(n_blocks)},
            )
        except Exception as exc:  # noqa: BLE001 — audit emit is best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on cross_week_spacing "
                "skip: %s",
                exc,
            )


# ---------------------------------------------------------------------------
# Shape-tolerant block readers
# ---------------------------------------------------------------------------


def _week_of(block: Any, block_attr) -> Optional[int]:
    """Resolve the integer week number of a block from its ``week_NN`` page-id.

    Mirrors ``course_level_qa._module_key`` / ``_module_sort_key`` — the
    canonical ``week_NN_*`` page-id grouping every course-level gate reuses.
    Returns ``None`` when the page-id carries no resolvable ``week_NN`` prefix
    (graceful-skip signal — the concept cannot be placed on the week axis).
    """
    pid = block_attr(block, "page_id")
    if not isinstance(pid, str) or not pid:
        return None
    parts = pid.split("_")
    if len(parts) >= 2 and parts[0] == "week":
        try:
            return int(parts[1])
        except (TypeError, ValueError):
            return None
    return None


def _concept_is_objective(concept: str) -> bool:
    """True iff a concept key is an objective id (carries the ``::`` axis tag)."""
    return concept.startswith("objective::")


def _concepts_of(block: Any, block_attr) -> List[str]:
    """Resolve the concept keys a block touches (objective ids OR concept tags).

    Prefers the objective ids a block targets (the substantive teaching axis);
    when a block resolves no objective id, falls back to its ``concept_tags`` so
    a concept with no objective binding is still tracked across weeks. Keys are
    axis-namespaced (``objective::`` / ``concept::``) so the two axes never
    collide. Returns ``[]`` when neither resolves (the block contributes no
    spacing signal).
    """
    out: List[str] = []
    # Objective axis (preferred).
    for key in ("objective_id", "target_objective_id"):
        v = block_attr(block, key)
        if v:
            out.append(f"objective::{v}")
    for key in ("objective_ids", "target_co_ids", "learning_outcome_refs"):
        v = block_attr(block, key)
        if isinstance(v, (list, tuple)):
            out.extend(f"objective::{x}" for x in v if x)
    if out:
        return _dedup(out)

    # Concept-tag axis (fallback when no objective resolves).
    for key in ("concept_tags", "key_terms"):
        v = block_attr(block, key)
        if isinstance(v, (list, tuple)):
            out.extend(f"concept::{x}" for x in v if x)
        elif isinstance(v, str) and v:
            out.append(f"concept::{v}")
    return _dedup(out)


def _dedup(items: List[str]) -> List[str]:
    """Order-preserving de-dup."""
    seen: Set[str] = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


__all__ = [
    "CrossWeekSpacingValidator",
    "MIN_BLOCKS_FOR_SPACING",
]
