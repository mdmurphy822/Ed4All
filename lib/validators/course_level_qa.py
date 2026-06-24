"""FR-COURSE-01 — course-level §6.5 EMERGENT-quality QA gate.

The framework's §6.5 defines course-level EMERGENT quality (dimensions A-F)
that NO per-block gate captures, because the gap only exists at the whole-course
scope:

  * **A. Cumulative coverage** — every terminal objective is touched by the
    course's block set (no orphan TO with zero authored content).
  * **D. Spacing / retrieval rhythm** — retrieval/check blocks are distributed
    across modules, not massed into one. REUSED, not recomputed: the per-module
    spacing/sequence floors are owned by ``retrieval_presence`` (IB7.5b) and
    ``block_sequence_order`` (FR-COURSE-02); this gate only surfaces the
    COURSE-level rollup of how many modules carry a retrieval block.
  * **F. Interaction MIX (the keystone, OSCQR item 34)** — a course must offer a
    VARIETY of interaction types: student↔content (exposition / check),
    student↔student (discussion, B10), and student↔instructor (graded /
    feedback, B14). A course that delivers only student↔content blocks is a
    read-only course, not an interactive one.
  * **course-integration close** — the course must END with a summative /
    integration surface (a final assessment / cumulative summary on the last
    module), not trail off after the last unit.

This validator COMPOSES already-computed REAL signals — it does NOT re-score
blocks:

  * the **block→module→course rollup** (``lib/aggregators/block_quality_rollup``)
    — reconstructed self-sufficiently from the canonical IB6.1 rubric scorer,
    exactly as :class:`lib.validators.block_quality_rollup.BlockQualityRollupValidator`
    does (one scorer, one rollup owner);
  * the **per-page block-type distribution** of the rewrite-tier ``blocks`` (the
    framework B-codes that classify each block into an interaction mode and the
    page/module grouping that orders "course end");
  * the optional **06_assessments manifest** (a supplementary student↔instructor
    signal: a graded quiz / assignment is an instructor-interaction surface even
    when no B14 block was authored inline). Absent → the gate falls back to the
    block-derived signal only (post_rewrite_validation runs BEFORE
    assessment_synthesis, so the manifest is usually not yet on disk; the block
    set is the load-bearing signal).

Anti-fabrication: every flag is grounded in a REAL signal present in the input
surface. The gate never synthesises course data; an unresolvable signal yields a
structured skip, not an invented verdict.

Signals / codes
---------------

  * ``COURSE_INTERACTION_MIX_NARROW`` (Section F, keystone) — the course's
    block set spans < 2 of the 3 interaction modes (student-content,
    student-student, student-instructor). Discussion (B10) is the student↔
    student surface; graded/assessment (B14) OR a graded assessments-manifest
    item is the student↔instructor surface; exposition/check is student↔content.
  * ``COURSE_NO_INTEGRATION_CLOSE`` — the LAST content module carries no
    summative / integration block (B13 summary, B14 graded assessment, recap /
    checklist), so the course trails off without a cumulative close.
  * ``COURSE_TO_COVERAGE_GAP`` (Dimension A) — a terminal objective the course
    defines is touched by ZERO authored block (orphan TO). Only emitted when
    the objectives universe resolves; pure id resolution, no embeddings.
  * ``COURSE_RETRIEVAL_RHYTHM_THIN`` (Dimension D) — fewer than half the
    content-bearing modules carry a retrieval/check block. REUSES the rollup's
    per-module grouping; advisory companion to the per-module
    ``retrieval_presence`` floor (no recompute of the spacing logic).

Posture
-------

Warning-day-1 (``passed=True`` unconditionally) with a ``# TODO(calibration)``
deferred critical-flip — mirroring every IB6 / FR-COURSE gate. Byte-stable
no-op when ``ED4ALL_BLOCK_QUALITY_RUBRIC`` is unset (the rollup it composes
already rides that flag), returning a ``RUBRIC_DISABLED`` info issue and writing
nothing.

References:
    - FRAMEWORK.pdf — §6.5 emergent course-level quality (A-F).
    - OSCQR rubric item 34 — "Course includes a variety of interaction types
      (instructor-student, student-student, student-content)."
    - ``lib/validators/block_quality_rollup.py`` — sibling rollup gate (rollup
      reconstruction pattern mirrored here).
    - ``lib/validators/cumulative_assessment.py`` — sibling FR-COURSE assessment
      gate (objectives-universe loading mirrored here).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

_ISSUE_LIST_CAP = 50

#: Minimum distinct interaction modes (of the 3) for the mix to be "varied".
MIN_INTERACTION_MODES: int = 2

#: Framework B-codes by interaction mode (OSCQR item 34).
_STUDENT_CONTENT_CODES: frozenset = frozenset(
    {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09",
     "B11", "B12", "B13", "B15"}
)
_STUDENT_STUDENT_CODES: frozenset = frozenset({"B10"})  # discussion
_STUDENT_INSTRUCTOR_CODES: frozenset = frozenset({"B14"})  # graded assessment

#: Framework B-codes that constitute a summative / integration close.
_INTEGRATION_CLOSE_CODES: frozenset = frozenset({"B13", "B14"})  # summary / graded
#: Block-type names that ALSO count as an integration close (recap / checklist
#: render as B02/B08 but are end-of-course consolidation surfaces).
_INTEGRATION_CLOSE_TYPES: frozenset = frozenset(
    {"summary_takeaway", "recap", "checklist", "assessment_item"}
)

#: Framework B-codes that constitute a retrieval / check (Dimension D).
_RETRIEVAL_CODES: frozenset = frozenset({"B07", "B11", "B14"})  # check / reflect / graded


# ---------------------------------------------------------------------------
# Objectives-universe loading (mirror of cumulative_assessment)
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
        logger.debug("course_level_qa: failed to load %s: %s", path, exc)
        return None


def _terminal_ids(obj_doc: Any) -> Set[str]:
    """Return the set of terminal-objective ids from a synthesized objectives doc.

    Accepts both the Courseforge synthesized form (``terminal_objectives``) and
    the LibV2 archive form (``terminal_outcomes``); falls back to the flat
    ``learning_outcomes`` list (TO-prefixed ids / hierarchy_level == terminal).
    """
    out: Set[str] = set()
    if not isinstance(obj_doc, dict):
        return out
    to_list = (
        obj_doc.get("terminal_objectives")
        or obj_doc.get("terminal_outcomes")
        or []
    )
    if isinstance(to_list, list):
        for t in to_list:
            if isinstance(t, dict) and t.get("id"):
                out.add(str(t["id"]))
    if not out:
        for lo in obj_doc.get("learning_outcomes") or []:
            if not isinstance(lo, dict):
                continue
            lid = str(lo.get("id") or "")
            if lid.startswith("TO-") or lo.get("hierarchy_level") == "terminal":
                out.add(lid)
    return out


def _co_to_to(obj_doc: Any) -> Dict[str, str]:
    """Map each chapter/component objective id -> its parent terminal id."""
    mapping: Dict[str, str] = {}
    if not isinstance(obj_doc, dict):
        return mapping
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
                mapping[str(cid)] = str(tid)
    return mapping


def _resolve_to(obj_id: str, terminal_ids: Set[str], co_to_to: Dict[str, str]) -> Optional[str]:
    if not obj_id:
        return None
    if obj_id in terminal_ids:
        return obj_id
    mapped = co_to_to.get(obj_id)
    if mapped and mapped in terminal_ids:
        return mapped
    if mapped:
        return mapped
    if obj_id.startswith("TO-"):
        return obj_id
    return None


# ---------------------------------------------------------------------------
# Assessments-manifest extraction (optional student-instructor signal)
# ---------------------------------------------------------------------------


def _manifest_has_graded(path: Optional[str]) -> bool:
    """True iff the 06_assessments manifest carries >=1 graded surface.

    A quiz / assignment is a student↔instructor (graded) interaction surface.
    Reads the manifest shape-tolerantly: a ``manifest.json`` under the dir, or
    the dir's direct ``.json`` payload; counts any quiz / assignment resource.
    Absent / unreadable → False (the block-derived B14 signal stands alone).
    """
    if not path:
        return False
    p = Path(path)
    candidates: List[Path] = []
    if p.is_dir():
        candidates.append(p / "manifest.json")
    elif p.is_file():
        candidates.append(p)
    for c in candidates:
        data = _load_json(str(c)) if c.exists() else None
        if not isinstance(data, dict):
            continue
        # Common manifest shapes: {"assessments": [...]}, {"resources": [...]},
        # {"quizzes": [...], "assignments": [...]}.
        for key in ("assessments", "resources", "quizzes", "assignments"):
            items = data.get(key)
            if isinstance(items, list) and items:
                return True
    return False


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------


class CourseLevelQaValidator:
    """FR-COURSE-01 — course-level §6.5 emergent-quality QA gate."""

    name = "course_level_qa"
    version = "0.1.0"

    def __init__(self, *, decision_capture: Optional[Any] = None) -> None:
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        from lib.validators._block_rubric_helpers import (
            block_quality_rubric_enabled,
            block_type_of,
            framework_block_of,
        )

        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or self._decision_capture

        enabled = inputs.get("rubric_enabled")
        if enabled is None:
            enabled = block_quality_rubric_enabled()
        if not enabled:
            # Byte-stable no-op (the rollup it composes rides the same flag).
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
                            "ED4ALL_BLOCK_QUALITY_RUBRIC unset — course-level "
                            "QA gate skipped (no composition, byte-stable)."
                        ),
                    )
                ],
                action=None,
                metadata={"course_level_qa": {"enabled": False}},
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
                            "no blocks surfaced; cannot audit course-level "
                            "emergent quality (skipped)."
                        ),
                    )
                ],
                action=None,
                metadata={"course_level_qa": {"enabled": True, "n_blocks": 0}},
            )

        # ------------------------------------------------------------------
        # Compose the block→module→course rollup (REUSE, never re-score).
        # ------------------------------------------------------------------
        rollup_summary = self._compose_rollup(inputs)

        # ------------------------------------------------------------------
        # Per-page / per-module block-type distribution (REAL signal).
        # ------------------------------------------------------------------
        modes_present: Set[str] = set()
        codes_seen: Dict[str, int] = {}
        modules: Dict[str, List[Any]] = {}
        touched_objectives: Set[str] = set()

        for block in blocks:
            bcode = framework_block_of(block)
            if bcode:
                codes_seen[bcode] = codes_seen.get(bcode, 0) + 1
                if bcode in _STUDENT_STUDENT_CODES:
                    modes_present.add("student_student")
                elif bcode in _STUDENT_INSTRUCTOR_CODES:
                    modes_present.add("student_instructor")
                elif bcode in _STUDENT_CONTENT_CODES:
                    modes_present.add("student_content")
            module_key = _module_key(_block_page_id(block))
            modules.setdefault(module_key, []).append(block)
            for oid in _block_objective_ids(block):
                touched_objectives.add(oid)

        # Supplementary student↔instructor signal: a graded quiz / assignment in
        # the 06_assessments manifest is an instructor-interaction surface even
        # without an inline B14 block.
        manifest_graded = _manifest_has_graded(inputs.get("assessments_path"))
        if manifest_graded:
            modes_present.add("student_instructor")

        ordered_modules = _sorted_modules(modules.keys())

        issues: List[GateIssue] = []

        # ---- Section F (keystone): interaction MIX --------------------------
        if len(modes_present) < MIN_INTERACTION_MODES:
            missing = sorted(
                {"student_content", "student_student", "student_instructor"}
                - modes_present
            )
            issues.append(
                GateIssue(
                    severity="warning",
                    code="COURSE_INTERACTION_MIX_NARROW",
                    message=(
                        f"course spans only {len(modes_present)} of 3 interaction "
                        f"modes ({sorted(modes_present)}); missing: {missing}. "
                        f"OSCQR item 34 / framework §6.5 F requires a VARIETY of "
                        f"interaction types — student↔content (exposition/check), "
                        f"student↔student (discussion, B10), student↔instructor "
                        f"(graded/feedback, B14)."
                    ),
                    suggestion=(
                        "Add the missing interaction surface(s): a discussion "
                        "block (B10) for student↔student and/or a graded "
                        "assessment (B14) / instructor-feedback surface for "
                        "student↔instructor."
                    ),
                )
            )

        # ---- course-integration close --------------------------------------
        has_close, last_module = self._has_integration_close(
            ordered_modules, modules, block_type_of, framework_block_of
        )
        if ordered_modules and not has_close:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="COURSE_NO_INTEGRATION_CLOSE",
                    message=(
                        f"the last content module ({last_module!r}) carries no "
                        f"summative / integration block (B13 summary, B14 graded "
                        f"assessment, recap, or checklist); the course trails off "
                        f"without a cumulative close (framework §6.5)."
                    ),
                    location=str(last_module),
                    suggestion=(
                        "End the course with a summative summary (B13) or a "
                        "cumulative graded assessment (B14) so the final module "
                        "integrates the course's terminal objectives."
                    ),
                )
            )

        # ---- Dimension A: cumulative TO coverage ---------------------------
        n_orphan_tos = 0
        terminal_ids: Set[str] = set()
        obj_doc = _load_json(inputs.get("synthesized_objectives_path"))
        if obj_doc is not None:
            terminal_ids = _terminal_ids(obj_doc)
            co_to_to = _co_to_to(obj_doc)
            covered_tos: Set[str] = set()
            for oid in touched_objectives:
                to = _resolve_to(oid, terminal_ids, co_to_to)
                if to:
                    covered_tos.add(to)
            orphan_tos = sorted(terminal_ids - covered_tos)
            n_orphan_tos = len(orphan_tos)
            if orphan_tos and len(issues) < _ISSUE_LIST_CAP:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="COURSE_TO_COVERAGE_GAP",
                        message=(
                            f"{n_orphan_tos} terminal objective(s) are touched by "
                            f"ZERO authored block ({orphan_tos[:10]}); the course "
                            f"defines content the block set never delivers "
                            f"(framework §6.5 A — cumulative coverage)."
                        ),
                        suggestion=(
                            "Author at least one block targeting each orphan "
                            "terminal objective so every TO has delivered content."
                        ),
                    )
                )

        # ---- Dimension D: retrieval rhythm (advisory companion) ------------
        content_modules = [
            m for m in ordered_modules if _module_is_content_bearing(modules[m], framework_block_of)
        ]
        modules_with_retrieval = sum(
            1
            for m in content_modules
            if any(framework_block_of(b) in _RETRIEVAL_CODES for b in modules[m])
        )
        n_content_modules = len(content_modules)
        if n_content_modules >= 2 and modules_with_retrieval * 2 < n_content_modules:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="COURSE_RETRIEVAL_RHYTHM_THIN",
                    message=(
                        f"only {modules_with_retrieval}/{n_content_modules} "
                        f"content-bearing modules carry a retrieval/check block; "
                        f"retrieval is massed, not spaced across the course "
                        f"(framework §6.5 D). Per-module spacing is owned by "
                        f"retrieval_presence / block_sequence_order — this is the "
                        f"course-level rollup."
                    ),
                    suggestion=(
                        "Distribute low-stakes retrieval/check blocks across more "
                        "modules so retrieval rhythm spans the course."
                    ),
                )
            )

        self._emit_decision(
            capture,
            modes_present=modes_present,
            has_close=has_close,
            n_orphan_tos=n_orphan_tos,
            n_tos=len(terminal_ids),
            modules_with_retrieval=modules_with_retrieval,
            n_content_modules=n_content_modules,
            issue_count=len(issues),
            rollup_summary=rollup_summary,
        )

        # Warning-day-1: never blocks. ``passed`` stays True so the gate cannot
        # alter ``final_status`` until the calibration-deferred critical-flip.
        #
        # TODO(calibration): when scripts/calibration_harness.py confirms the
        # course-level FP rate on >=2 corpora, flip the gate row to
        # ``severity: critical`` + ``behavior.on_fail: block`` in
        # config/workflows.yaml AND set ``passed`` to
        # ``COURSE_INTERACTION_MIX_NARROW not in {i.code for i in issues}`` so a
        # narrow-interaction course blocks promotion (framework §6.5 F).
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=round(len(modes_present) / 3.0, 4),
            issues=issues,
            action=None,
            metadata={
                "course_level_qa": {
                    "enabled": True,
                    "n_blocks": len(blocks),
                    "interaction_modes_present": sorted(modes_present),
                    "n_interaction_modes": len(modes_present),
                    "framework_codes": codes_seen,
                    "has_integration_close": has_close,
                    "last_module": last_module,
                    "n_terminal_objectives": len(terminal_ids),
                    "n_orphan_terminal_objectives": n_orphan_tos,
                    "n_content_modules": n_content_modules,
                    "modules_with_retrieval": modules_with_retrieval,
                    "manifest_graded": manifest_graded,
                    "rollup_summary": rollup_summary,
                }
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _compose_rollup(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compose the block→module→course rollup summary (REUSE, no re-score).

        Mirrors :class:`BlockQualityRollupValidator` exactly: score the blocks
        with the canonical IB6.1 rubric scorer (single owner of the 0-3 logic),
        then roll up via the aggregator (single owner of the rollup math).
        Returns the rollup ``summary`` dict (or ``{}`` on any failure — the gate
        degrades to the block-derived signals only, never crashes).
        """
        try:
            from lib.validators.block_quality_rubric import (
                BlockQualityRubricValidator,
            )
            from lib.aggregators.block_quality_rollup import (
                BlockQualityRollupAggregator,
            )

            rubric_inputs = dict(inputs)
            rubric_inputs["rubric_enabled"] = True
            rubric_inputs.setdefault("gate_id", "block_quality_rubric")
            rubric_result = BlockQualityRubricValidator().validate(rubric_inputs)

            aggregator = BlockQualityRollupAggregator(
                phase_outputs={
                    "post_rewrite_validation": {"_gate_results": [rubric_result]}
                },
                course_code=str(
                    inputs.get("course_name") or inputs.get("course_code") or ""
                ),
                run_id=str(inputs.get("run_id") or ""),
            )
            report = aggregator.build()
            return report.get("summary") or {}
        except Exception as exc:  # noqa: BLE001 — composition is best-effort
            logger.debug("course_level_qa: rollup composition failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    @staticmethod
    def _has_integration_close(
        ordered_modules: List[str],
        modules: Dict[str, List[Any]],
        block_type_of,
        framework_block_of,
    ) -> "tuple[bool, str]":
        """Return ``(has_close, last_content_module)``.

        The LAST content-bearing module must carry a summative / integration
        block (B13 / B14, or a recap / checklist / summary_takeaway / assessment
        block-type). Returns ``(True, "")`` when there are no modules to audit.
        """
        if not ordered_modules:
            return True, ""
        # Walk from the last module backwards to the first content-bearing one.
        last_content = ""
        for m in reversed(ordered_modules):
            if _module_is_content_bearing(modules[m], framework_block_of):
                last_content = m
                break
        if not last_content:
            last_content = ordered_modules[-1]
        for b in modules[last_content]:
            if framework_block_of(b) in _INTEGRATION_CLOSE_CODES:
                return True, last_content
            if block_type_of(b) in _INTEGRATION_CLOSE_TYPES:
                return True, last_content
        return False, last_content

    # ------------------------------------------------------------------
    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        modes_present: Set[str],
        has_close: bool,
        n_orphan_tos: int,
        n_tos: int,
        modules_with_retrieval: int,
        n_content_modules: int,
        issue_count: int,
        rollup_summary: Dict[str, Any],
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="validation_result",
                decision=(
                    f"course_level_qa interaction_modes={len(modes_present)}/3, "
                    f"integration_close={has_close}, orphan_tos={n_orphan_tos}"
                ),
                rationale=(
                    "FR-COURSE-01 course-level §6.5 emergent-quality audit "
                    "(OSCQR item 34): composed the block→module→course rollup "
                    f"(quality_decision={rollup_summary.get('quality_decision')}) "
                    f"with the per-page block-type distribution. Interaction MIX "
                    f"spans {sorted(modes_present)} ({len(modes_present)}/3 modes); "
                    f"integration_close={has_close}; "
                    f"{n_orphan_tos}/{n_tos} terminal objectives orphaned; "
                    f"retrieval rhythm {modules_with_retrieval}/{n_content_modules} "
                    f"content modules; {issue_count} issue(s). Warning-day-1: "
                    "enumerated as issues but does not yet block (calibration-"
                    "deferred critical-flip)."
                ),
                ml_features={
                    "n_interaction_modes": len(modes_present),
                    "has_integration_close": bool(has_close),
                    "n_orphan_terminal_objectives": int(n_orphan_tos),
                    "n_terminal_objectives": int(n_tos),
                    "modules_with_retrieval": int(modules_with_retrieval),
                    "n_content_modules": int(n_content_modules),
                    "issue_count": int(issue_count),
                },
            )
        except Exception as exc:  # noqa: BLE001 — audit emit is best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on course_level_qa: %s",
                exc,
            )


# ---------------------------------------------------------------------------
# Shape-tolerant block readers / module ordering
# ---------------------------------------------------------------------------


def _block_page_id(block: Any) -> str:
    from lib.validators._block_rubric_helpers import block_attr

    pid = block_attr(block, "page_id")
    return str(pid) if isinstance(pid, str) else ""


def _block_objective_ids(block: Any) -> List[str]:
    """Resolve the objective ids a block targets (shape-tolerant)."""
    from lib.validators._block_rubric_helpers import block_attr

    out: List[str] = []
    for key in ("objective_id", "target_objective_id"):
        v = block_attr(block, key)
        if v:
            out.append(str(v))
    for key in ("objective_ids", "target_co_ids", "learning_outcome_refs"):
        v = block_attr(block, key)
        if isinstance(v, (list, tuple)):
            out.extend(str(x) for x in v if x)
    return out


def _module_key(page_id: str) -> str:
    """Resolve a module/week grouping key from a page_id (mirror of aggregator)."""
    if not page_id:
        return "module-1"
    parts = page_id.split("_")
    if len(parts) >= 2 and parts[0] == "week":
        return f"week_{parts[1]}"
    return page_id


def _module_sort_key(module_key: str) -> "tuple":
    """Sort modules into course order: numeric week prefix, then lexicographic."""
    parts = module_key.split("_")
    if len(parts) >= 2 and parts[0] == "week":
        try:
            return (0, int(parts[1]), module_key)
        except (TypeError, ValueError):
            return (1, 0, module_key)
    return (1, 0, module_key)


def _sorted_modules(keys) -> List[str]:
    return sorted(keys, key=_module_sort_key)


def _module_is_content_bearing(blocks: List[Any], framework_block_of) -> bool:
    """True iff the module carries >=1 pedagogical (non-chrome) block."""
    for b in blocks:
        if framework_block_of(b) is not None:
            return True
    return False


__all__ = [
    "CourseLevelQaValidator",
    "MIN_INTERACTION_MODES",
]
