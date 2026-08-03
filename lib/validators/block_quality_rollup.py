"""IB6.6 — block→module→course quality-rollup GATE (framework FR-07/13, §6.5).

Surfaces the :class:`lib.aggregators.block_quality_rollup.BlockQualityRollupAggregator`
``course_pass`` / per-block-fail signal as a workflow GateResult so the
block→module→course quality rollup can PARTICIPATE in gate evaluation instead
of only writing a best-effort report nothing acts on.

Pipeline relationship
---------------------

The aggregator already runs post-loop in
``MCP/core/workflow_runner.run_workflow`` as a best-effort artifact emitter
(failure logs a warning, NEVER changes ``final_status``) and writes
``<libv2_course>/block_quality_rollup_report.json``. That is the report. THIS
validator is the GATE: it runs INSIDE the ``post_rewrite_validation`` phase
(two-pass), recomputes the same per-block 8-dim rubric scores the aggregator
rolls up, and returns a GateResult whose ``passed`` mirrors the rollup's
``course_pass`` (the three hard gates: Accessibility==0 → block fail; assessment
Bloom < objective → Alignment cap 1; interaction-without-feedback →
Feedback+Coherence cap 1; plus the mean>=2.0 AND per-dimension-min-floor paths).

Why recompute instead of read the sibling rubric GateResult
-----------------------------------------------------------

Within a single phase, gates run sequentially via
``GateInputRouter.build`` and each phase's ``_gate_results`` chain is only
stashed into ``phase_outputs`` AFTER the phase completes — so an in-flight
sibling gate's result is not yet visible through ``phase_outputs``. This gate is
therefore SELF-SUFFICIENT: it scores the ``blocks`` surface itself (delegating
to :class:`lib.validators.block_quality_rubric.BlockQualityRubricValidator`,
the single source of the anchored 0-3 scoring logic — no second scorer), then
feeds the scored rows to the aggregator. The aggregator stays the single source
of the rollup math.

Severity / deferred critical-flip
---------------------------------

Wired **warning-day-1** (``severity: warning, on_fail: warn``) at
``post_rewrite_validation`` in ``course_generation`` + ``textbook_to_course``.
The gate computes ``course_pass`` and ENUMERATES the failing blocks/dimensions
in its issues, but stays warning — it does NOT yet change ``final_status``.

The critical-flip path (so a failing rollup blocks promotion, FR-07/13): flip
``severity: critical`` + ``behavior.on_fail: block`` at the gate's
``# TODO(calibration)`` marker in ``config/workflows.yaml`` AND set
``self.passed = course_pass`` to drive the block — once
``scripts/harness/calibration_harness.py`` confirms the eight-dimension rubric gates'
false-positive rate is acceptable on >=2 corpora (the anchored 0-3 scale must
be calibrated before the mean/min-floor hard gates can block early runs). Until
then the gate returns ``passed=True`` unconditionally so it never alters
``final_status`` (mirroring the warning posture of every IB6 gate).

Default-OFF byte-stability
--------------------------

The whole path is a no-op when ``ED4ALL_BLOCK_QUALITY_RUBRIC`` is unset: the
rubric validator no-ops (no scores), so this gate returns
``GateResult(passed=True)`` with a ``RUBRIC_DISABLED`` info issue and writes
nothing — byte-identical to a run without the gate.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

_ISSUE_LIST_CAP = 50


class BlockQualityRollupValidator:
    """IB6.6 — gate the block→module→course quality rollup's ``course_pass``."""

    name = "block_quality_rollup"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        from lib.validators._block_rubric_helpers import (
            block_quality_rubric_enabled,
        )

        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        enabled = inputs.get("rubric_enabled")
        if enabled is None:
            enabled = block_quality_rubric_enabled()
        if not enabled:
            # Byte-stable no-op: the rubric scorer itself no-ops when the
            # keystone flag is unset, so there is nothing to roll up.
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
                            "ED4ALL_BLOCK_QUALITY_RUBRIC unset — block-quality "
                            "rollup gate skipped (no scoring, no rollup, "
                            "byte-stable)."
                        ),
                    )
                ],
                action=None,
                metadata={"block_quality_rollup": {"enabled": False}},
            )

        # ------------------------------------------------------------------
        # 1) Score the blocks with the canonical IB6.1 rubric scorer (the
        #    single source of the anchored 0-3 logic). We pass the same input
        #    surface through so it reads ``blocks`` + ``gate_results_by_block``.
        # ------------------------------------------------------------------
        from lib.validators.block_quality_rubric import BlockQualityRubricValidator

        rubric_inputs = dict(inputs)
        rubric_inputs["rubric_enabled"] = True
        rubric_inputs.setdefault("gate_id", "block_quality_rubric")
        rubric_result = BlockQualityRubricValidator().validate(rubric_inputs)

        # ------------------------------------------------------------------
        # 2) Roll up via the canonical aggregator. The aggregator already
        #    knows how to reconstruct scored block rows from a stashed
        #    ``block_quality_rubric`` GateResult (its ``_blocks_from_phase_
        #    outputs`` path) — reuse it verbatim so the rollup math has a
        #    single owner. We hand it a synthetic phase_outputs carrying the
        #    rubric result we just produced.
        # ------------------------------------------------------------------
        from lib.aggregators.block_quality_rollup import BlockQualityRollupAggregator

        synthetic_phase_outputs = {
            "post_rewrite_validation": {"_gate_results": [rubric_result]}
        }
        aggregator = BlockQualityRollupAggregator(
            phase_outputs=synthetic_phase_outputs,
            course_code=str(inputs.get("course_name") or inputs.get("course_code") or ""),
            run_id=str(inputs.get("run_id") or ""),
        )
        report = aggregator.build()
        summary = report.get("summary") or {}
        course_pass = bool(summary.get("course_pass"))

        issues = self._build_issues(report)
        self._emit_decision(capture, summary=summary, course_pass=course_pass)

        # Warning-day-1: never blocks. ``passed`` stays True so the gate cannot
        # alter ``final_status`` (the post_rewrite_validation phase gates are
        # warn-only) until the calibration-deferred critical-flip lands.
        #
        # TODO(calibration): when scripts/harness/calibration_harness.py confirms the
        # rubric gates' FP rate on >=2 corpora, flip the gate row to
        # ``severity: critical`` + ``behavior.on_fail: block`` in
        # config/workflows.yaml AND change the line below to
        # ``passed=course_pass`` so a failing rollup blocks promotion
        # (framework FR-07/13).
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=(
                round(
                    summary.get("blocks_pass", 0) / summary.get("total_blocks", 1),
                    4,
                )
                if summary.get("total_blocks")
                else 1.0
            ),
            issues=issues,
            action=None,
            metadata={
                "block_quality_rollup": {
                    "enabled": True,
                    "course_pass": course_pass,
                    "quality_decision": summary.get("quality_decision"),
                    "total_blocks": summary.get("total_blocks", 0),
                    "blocks_pass": summary.get("blocks_pass", 0),
                    "total_modules": summary.get("total_modules", 0),
                    "modules_pass": summary.get("modules_pass", 0),
                    "accessibility_gate_failures": summary.get(
                        "accessibility_gate_failures", 0
                    ),
                    "alignment_orphan_blocks": summary.get(
                        "alignment_orphan_blocks", 0
                    ),
                    "course_dim_minimums": summary.get("course_dim_minimums"),
                    # Threaded so the post-loop aggregator / promotion-chain
                    # rollup can read the gate's verdict without recomputing.
                    "report": report,
                }
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_issues(report: Dict[str, Any]) -> List[GateIssue]:
        """Enumerate the failing blocks/modules/dimensions as gate issues."""
        issues: List[GateIssue] = []
        summary = report.get("summary") or {}

        # Course-level verdict issue first (the FR-07/13 rollup signal).
        if not summary.get("course_pass"):
            issues.append(
                GateIssue(
                    severity="warning",
                    code="COURSE_QUALITY_ROLLUP_FAIL",
                    message=(
                        "Block-quality course rollup FAILS "
                        f"(quality_decision={summary.get('quality_decision')}): "
                        f"{summary.get('accessibility_gate_failures', 0)} "
                        "Accessibility-gate block failure(s), "
                        f"{summary.get('alignment_orphan_blocks', 0)} "
                        "Alignment-orphan block(s); "
                        f"{summary.get('blocks_pass', 0)}/"
                        f"{summary.get('total_blocks', 0)} blocks and "
                        f"{summary.get('modules_pass', 0)}/"
                        f"{summary.get('total_modules', 0)} modules pass."
                    ),
                    suggestion=(
                        "Lift every block to Accessibility>=1 and every "
                        "pedagogical block to a resolved objective (Alignment>0); "
                        "each module needs mean>=2.0 AND every core-dim minimum "
                        ">=2.0 across its blocks."
                    ),
                )
            )

        # Per-failing-block issues (the localizable defects), capped.
        for row in report.get("per_block") or []:
            if row.get("block_pass"):
                continue
            if len(issues) >= _ISSUE_LIST_CAP:
                break
            code = (
                "BLOCK_ACCESSIBILITY_GATE_FAIL"
                if row.get("accessibility_gate_fail")
                else "BLOCK_QUALITY_ROLLUP_BELOW_FLOOR"
            )
            issues.append(
                GateIssue(
                    severity="warning",
                    code=code,
                    message=(
                        f"Block {row.get('block_id')!r} "
                        f"({row.get('framework_block')}) in module "
                        f"{row.get('module')} fails the rollup floor "
                        f"(mean={row.get('mean')}, core_ok={row.get('core_ok')}, "
                        f"accessibility_gate_fail="
                        f"{row.get('accessibility_gate_fail')}); weakest "
                        f"dimension: {row.get('weakest_dim')}."
                    ),
                    location=str(row.get("block_id")),
                    suggestion=(
                        "Raise the weakest dimension to >=2 (every core "
                        "dimension must clear 2.0); Accessibility must be >0."
                    ),
                )
            )

        # Per-failing-module rollup issues (the no-weak-block-hides path).
        for mod in report.get("per_module") or []:
            if mod.get("module_pass"):
                continue
            if len(issues) >= _ISSUE_LIST_CAP:
                break
            failing_dims = sorted(
                dim
                for dim, mn in (mod.get("module_dim_minimums") or {}).items()
                if isinstance(mn, int) and mn < 2
            )
            issues.append(
                GateIssue(
                    severity="warning",
                    code="MODULE_QUALITY_ROLLUP_FAIL",
                    message=(
                        f"Module {mod.get('module')} fails the rollup floor "
                        f"(module_mean={mod.get('module_mean')}, "
                        f"required_dim_minimum_ok="
                        f"{mod.get('required_dim_minimum_ok')}); "
                        f"below-floor dimension minimums: {failing_dims}."
                    ),
                    location=str(mod.get("module")),
                    suggestion=(
                        "No weak block may hide in a strong module average — "
                        "every core dimension's per-module minimum must clear "
                        "2.0 and the module mean must be >=2.0."
                    ),
                )
            )

        return issues

    # ------------------------------------------------------------------
    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        summary: Dict[str, Any],
        course_pass: bool,
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="validation_result",
                decision=(
                    f"block_quality_rollup course_pass={course_pass} "
                    f"({summary.get('blocks_pass', 0)}/"
                    f"{summary.get('total_blocks', 0)} blocks pass)"
                ),
                rationale=(
                    "IB6.6 block->module->course quality-rollup gate (FR-07/13, "
                    "framework S6.5): rolled up the eight-dimension rubric "
                    f"scores to course_pass={course_pass} "
                    f"(quality_decision={summary.get('quality_decision')}; "
                    f"{summary.get('accessibility_gate_failures', 0)} a11y-gate "
                    "block fails, "
                    f"{summary.get('alignment_orphan_blocks', 0)} alignment "
                    "orphans; "
                    f"{summary.get('modules_pass', 0)}/"
                    f"{summary.get('total_modules', 0)} modules pass). "
                    "Warning-day-1: enumerated as issues but does not yet block "
                    "(calibration-deferred critical-flip)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "block_quality_rollup gate: %s",
                exc,
            )


__all__ = ["BlockQualityRollupValidator"]
