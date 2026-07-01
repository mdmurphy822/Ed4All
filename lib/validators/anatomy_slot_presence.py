"""IB6.2 — anatomy slot-presence validator (QA-4 / coherence gate).

The framework's QA-4 item asserts a well-formed block instantiates all six
anatomy elements — ``heading · purpose_tag · body · interaction · feedback ·
transition`` (omitting any slot = malformed; a block missing its interaction
or transition is "the most common malformation", p.51). The Feedback gate
caps Feedback AND Coherence at 1 when an interactive block has practice but no
feedback slot (6.4).

This validator reads the IB1 first-class anatomy slots (``heading`` /
``purpose_tag`` / ``interaction`` / ``feedback`` / ``transition``; ``content``
IS the body slot per IB1.2) and asserts:

* **interaction present** on the interactive framework codes B07/B08/B10/B14;
* **feedback present** whenever interaction is present (the cap-1 trigger);
* **transition present** on every pedagogical block (B-code resolves).

It additionally asserts (FR-INT-01) that a **B08 guided-practice block in a
faded sequence** (one directly following a ``worked_example`` B05) carries a
``fade_state`` (the reused IB5 field) so the worked→completion→independent
ladder is auditable — ``ANATOMY_FADE_STATE_MISSING`` (warning-day-1, rides this
gate; no new gate row).

Emits ``ANATOMY_INTERACTION_MISSING`` / ``ANATOMY_FEEDBACK_MISSING_ON_INTERACTIVE``
/ ``ANATOMY_TRANSITION_MISSING`` / ``ANATOMY_FADE_STATE_MISSING``. The feedback-missing-on-interactive issue
carries the **Feedback+Coherence cap-1 semantics** consumed by the IB6.1
scorer / IB6.6 rollup: per offending block_id it stamps a
``caps_dims=["feedback","coherence"]`` entry on the GateResult ``metadata``.

Reuse / honest-scope contract: this checks slot PRESENCE only (the
``InteractionFeedbackValidator`` in ``interaction_feedback.py`` checks feedback
QUALITY / elaboration). It loads no model and re-parses no HTML beyond the
slot fields. Warning-day-1. Default-off byte-stable: when
``ED4ALL_BLOCK_QUALITY_RUBRIC`` is unset it is a no-op pass.

When ALL IB1 slots are ``None`` across the block set (the anatomy flag is off
upstream so slots were never derived), the validator gracefully skips with
``passed=True`` + an ``ANATOMY_SLOTS_ABSENT`` info issue — the framework's
"back-derivation lossy" posture from IB1.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

_ISSUE_LIST_CAP = 50

# The two dims an interaction-without-feedback block caps at 1 (framework 6.4).
_FEEDBACK_CAP_DIMS = ["feedback", "coherence"]


class AnatomySlotPresenceValidator:
    """IB6.2 — assert the six anatomy slots are present per the framework."""

    name = "anatomy_slot_presence"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        from lib.validators._block_rubric_helpers import (
            block_attr,
            block_quality_rubric_enabled,
            block_quality_scoring_active,
            block_type_of,
            framework_block_of,
            is_interactive_block,
        )

        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        enabled = inputs.get("rubric_enabled")
        if enabled is None:
            # W8.8 — also run under the shadow-collect flag (measurement only).
            enabled = block_quality_scoring_active()
        if not enabled:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="ANATOMY_PRESENCE_DISABLED",
                        message=(
                            "ED4ALL_BLOCK_QUALITY_RUBRIC unset — anatomy "
                            "slot-presence check skipped (byte-stable)."
                        ),
                    )
                ],
                metadata={"anatomy_slot_presence": {"enabled": False}},
            )

        blocks = inputs.get("blocks") or []

        # Graceful skip when no IB1 slot is populated anywhere (anatomy flag
        # off upstream → back-derivation never ran).
        any_slot = False
        for block in blocks:
            for slot in ("heading", "purpose_tag", "interaction", "feedback", "transition"):
                if _nonempty(block_attr(block, slot)):
                    any_slot = True
                    break
            if any_slot:
                break
        if blocks and not any_slot:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="ANATOMY_SLOTS_ABSENT",
                        message=(
                            "No IB1 anatomy slot populated on any block "
                            "(ED4ALL_BLOCK_ANATOMY off upstream / slots not "
                            "derived) — slot-presence check skipped."
                        ),
                    )
                ],
                metadata={"anatomy_slot_presence": {"enabled": True, "slots_absent": True}},
            )

        # FR-INT-03 — reflection-calibration gate (rides ED4ALL_REFLECTION_
        # CALIBRATION, default OFF → byte-stable no-op). When on, a B11
        # reflection's feedback slot should carry the calibration / benchmark
        # (the reveal_content / calibration_feedback). Optional override seam.
        reflection_calibration_enabled = inputs.get("reflection_calibration_enabled")
        if reflection_calibration_enabled is None:
            try:
                from lib.generation.reflection_calibration import (
                    resolve_reflection_calibration,
                )

                reflection_calibration_enabled = resolve_reflection_calibration()
            except Exception:  # noqa: BLE001 — never let the resolver break the gate
                reflection_calibration_enabled = False

        issues: List[GateIssue] = []
        audited = 0
        caps_by_block: Dict[str, List[str]] = {}
        per_block: Dict[str, Dict[str, Any]] = {}

        for idx, block in enumerate(blocks):
            bt = block_type_of(block)
            bcode = framework_block_of(block)
            if not bt or bcode is None:
                # chrome / unknown → non-pedagogical, no anatomy contract.
                continue
            audited += 1
            block_id = str(block_attr(block, "block_id") or f"block-{idx}")
            interactive = is_interactive_block(block)
            has_interaction = _nonempty(block_attr(block, "interaction"))
            has_feedback = _nonempty(block_attr(block, "feedback"))
            has_transition = _nonempty(block_attr(block, "transition"))

            block_issues: List[str] = []

            # FR-INT-01 — a B08 guided-practice block that is part of a FADED
            # sequence (it directly follows a worked_example B05) should carry a
            # ``fade_state`` (the reused IB5 field) so the worked→completion→
            # independent ladder is auditable. Warning-day-1 (rides this gate; no
            # new gate row). # TODO(calibration): flip to a hard check once the
            # fading planner pass is calibrated on >=2 corpora.
            prev_type = (
                block_type_of(blocks[idx - 1]) if idx > 0 else ""
            )
            if bcode == "B08" and prev_type == "worked_example":
                if not _nonempty(block_attr(block, "fade_state")):
                    block_issues.append("ANATOMY_FADE_STATE_MISSING")
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(
                            GateIssue(
                                severity="warning",
                                code="ANATOMY_FADE_STATE_MISSING",
                                message=(
                                    f"Guided-practice block {block_id!r} "
                                    f"({bt}/{bcode}) follows a worked_example "
                                    f"but carries no fade_state — the faded "
                                    f"guided-practice ladder "
                                    f"(worked→completion→independent) is not "
                                    f"auditable (FR-INT-01)."
                                ),
                                location=block_id,
                                suggestion=(
                                    "Stamp fade_state='completion' on the "
                                    "faded-practice block after a worked example."
                                ),
                            )
                        )

            if interactive and not has_interaction:
                block_issues.append("ANATOMY_INTERACTION_MISSING")
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="ANATOMY_INTERACTION_MISSING",
                            message=(
                                f"Interactive block {block_id!r} ({bt}/{bcode}) "
                                f"has no interaction slot — the most common "
                                f"malformation (p.51)."
                            ),
                            location=block_id,
                            suggestion="Author an interaction slot (the apply stage).",
                        )
                    )

            # Feedback gate: interaction present (or required) but no feedback
            # → caps Feedback + Coherence at 1.
            if (has_interaction or interactive) and not has_feedback:
                block_issues.append("ANATOMY_FEEDBACK_MISSING_ON_INTERACTIVE")
                caps_by_block.setdefault(block_id, [])
                for dim in _FEEDBACK_CAP_DIMS:
                    if dim not in caps_by_block[block_id]:
                        caps_by_block[block_id].append(dim)
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="ANATOMY_FEEDBACK_MISSING_ON_INTERACTIVE",
                            message=(
                                f"Block {block_id!r} ({bt}/{bcode}) has practice "
                                f"but no feedback slot — Feedback AND Coherence "
                                f"capped at 1 (framework 6.4 Feedback gate)."
                            ),
                            location=block_id,
                            suggestion="Add an elaborated feedback slot to the interaction.",
                        )
                    )

            if not has_transition:
                block_issues.append("ANATOMY_TRANSITION_MISSING")
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="ANATOMY_TRANSITION_MISSING",
                            message=(
                                f"Block {block_id!r} ({bt}/{bcode}) has no "
                                f"transition slot (the consolidate stage)."
                            ),
                            location=block_id,
                            suggestion="Add a transition that connects to the next block.",
                        )
                    )

            # FR-INT-03 — a B11 reflection's feedback slot should carry the
            # calibration / benchmark (the predict-then-reveal payload: the
            # reveal_content benchmark or calibration_feedback) so the
            # reflection's feedback is a benchmark to calibrate against, not just
            # an ack. Rides ED4ALL_REFLECTION_CALIBRATION (default OFF →
            # byte-stable). Warning-day-1; rides this gate (no new gate row).
            if reflection_calibration_enabled and bcode == "B11":
                has_benchmark = (
                    has_feedback
                    or _nonempty(block_attr(block, "reveal_content"))
                    or _nonempty(block_attr(block, "calibration_feedback"))
                )
                if not has_benchmark:
                    block_issues.append("ANATOMY_REFLECTION_NO_BENCHMARK")
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(
                            GateIssue(
                                severity="warning",
                                code="ANATOMY_REFLECTION_NO_BENCHMARK",
                                message=(
                                    f"Reflection block {block_id!r} ({bt}/B11) "
                                    f"feedback slot carries no calibration "
                                    f"benchmark (reveal_content / "
                                    f"calibration_feedback) — a reflection's "
                                    f"feedback should be a benchmark to calibrate "
                                    f"the learner's judgment against (FR-INT-03)."
                                ),
                                location=block_id,
                                suggestion=(
                                    "Populate the feedback slot with the "
                                    "calibration benchmark (the revealed answer "
                                    "+ how to compare your judgment to it)."
                                ),
                            )
                        )

            per_block[block_id] = {
                "framework_block": bcode,
                "interactive": interactive,
                "has_interaction": has_interaction,
                "has_feedback": has_feedback,
                "has_transition": has_transition,
                "issues": block_issues,
            }

        self._emit_decision(
            capture,
            audited=audited,
            caps=len(caps_by_block),
            issue_count=len(issues),
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1
            issues=issues,
            action=None,
            metadata={
                "anatomy_slot_presence": {
                    "enabled": True,
                    "blocks_audited": audited,
                    # block_id -> ["feedback","coherence"] cap directive
                    # consumed by IB6.1 / IB6.6.
                    "caps_dims_by_block": caps_by_block,
                    "per_block": per_block,
                }
            },
        )

    @staticmethod
    def _emit_decision(
        capture: Any, *, audited: int, caps: int, issue_count: int
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="content_structure_check",
                decision=f"anatomy_caps={caps}",
                rationale=(
                    f"IB6.2 anatomy slot-presence: audited {audited} "
                    f"pedagogical blocks for interaction / feedback / "
                    f"transition presence; {issue_count} slot-presence issues, "
                    f"{caps} interactive blocks missing feedback (Feedback+"
                    f"Coherence cap-1)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on anatomy_slot_presence: %s",
                exc,
            )


def _nonempty(value: Any) -> bool:
    """True iff ``value`` is a non-blank string."""
    return isinstance(value, str) and bool(value.strip())


__all__ = ["AnatomySlotPresenceValidator"]
