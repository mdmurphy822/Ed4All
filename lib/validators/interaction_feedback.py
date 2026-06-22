"""IB6.3 — universal interaction-feedback presence + elaboration gate.

Framework D5 / QA-8 / Invariant 2: every interaction returns immediate,
elaborated, misconception-targeted feedback ("why" not just "whether"); bare
right/wrong is insufficient; the Feedback gate caps Feedback + Coherence at 1
when an interactive block has no feedback (6.2-6.4, B07).

The companion to :class:`AnatomySlotPresenceValidator` (IB6.2). IB6.2 checks
slot *presence*; this checks feedback *quality / elaboration*. For every
interactive block (framework B07/B08/B10/B14), it asserts the feedback slot is:

* (a) present — else ``INTERACTION_NO_FEEDBACK`` (→ Feedback+Coherence cap-1);
* (b) elaborated — "why" not just "whether": the feedback visible text must
  exceed ``_ELABORATION_MIN_TOKENS`` tokens AND not be a bare correct/incorrect
  marker, else ``INTERACTION_BARE_RIGHT_WRONG`` / ``INTERACTION_FEEDBACK_THIN``;
* (c) misconception-targeted where a distractor cluster exists — REUSES the
  ``distractor_misconception_alignment`` signal threaded in via
  ``inputs["distractor_signals_by_block"]`` (does NOT recompute distractor
  quality; honest scope — the distractor engine stays untouched).

Honest delta (gap ``feedback-elaborated-misconception``, PARTIALLY MET):
misconception modeling + distractor faithfulness already exist for
``assessment_item`` (``lib.validators.distractor_*`` / ``padded_distractor``).
This adds ONLY the universal per-interaction presence/elaboration check + the
cap-1 hard-gate directive. Warning-day-1; default-off byte-stable.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

_ISSUE_LIST_CAP = 50

# Feedback under this many visible tokens is "thin" (a one-liner that asserts
# whether, not why). Deliberately low — recall-leaning detection, warning-day-1.
_ELABORATION_MIN_TOKENS = 8

# Bare right/wrong markers — feedback that is ONLY one of these (after strip)
# is "bare", not elaborated. Matched on the whole stripped feedback text.
_BARE_MARKER_RE = re.compile(
    r"^(?:correct|incorrect|right|wrong|true|false|yes|no|✓|✗|✔|✘)[\.\!\s]*$",
    re.IGNORECASE,
)

# The dims an interaction-without-feedback block caps at 1 (framework 6.4).
_FEEDBACK_CAP_DIMS = ["feedback", "coherence"]


class InteractionFeedbackValidator:
    """IB6.3 — universal per-interaction feedback presence + elaboration."""

    name = "interaction_feedback"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        from lib.validators._block_rubric_helpers import (
            block_attr,
            block_quality_rubric_enabled,
            block_type_of,
            framework_block_of,
            is_interactive_block,
            strip_html_text,
        )

        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        enabled = inputs.get("rubric_enabled")
        if enabled is None:
            enabled = block_quality_rubric_enabled()
        if not enabled:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="INTERACTION_FEEDBACK_DISABLED",
                        message=(
                            "ED4ALL_BLOCK_QUALITY_RUBRIC unset — interaction-"
                            "feedback check skipped (byte-stable)."
                        ),
                    )
                ],
                metadata={"interaction_feedback": {"enabled": False}},
            )

        blocks = inputs.get("blocks") or []
        # Optional reuse of the already-computed distractor/misconception signal
        # (block_id -> {"misconception_targeted": bool, ...}). Never recomputed.
        distractor_signals = inputs.get("distractor_signals_by_block") or {}

        issues: List[GateIssue] = []
        audited = 0
        caps_by_block: Dict[str, List[str]] = {}
        per_block: Dict[str, Dict[str, Any]] = {}

        for idx, block in enumerate(blocks):
            if not is_interactive_block(block):
                continue
            audited += 1
            bt = block_type_of(block)
            bcode = framework_block_of(block)
            block_id = str(block_attr(block, "block_id") or f"block-{idx}")
            fb_raw = block_attr(block, "feedback")
            fb_text = strip_html_text(fb_raw) if isinstance(fb_raw, str) else ""

            status = "elaborated"
            block_codes: List[str] = []

            if not fb_text:
                status = "missing"
                block_codes.append("INTERACTION_NO_FEEDBACK")
                caps_by_block.setdefault(block_id, [])
                for dim in _FEEDBACK_CAP_DIMS:
                    if dim not in caps_by_block[block_id]:
                        caps_by_block[block_id].append(dim)
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="INTERACTION_NO_FEEDBACK",
                            message=(
                                f"Interactive block {block_id!r} ({bt}/{bcode}) "
                                f"has no feedback slot — Feedback AND Coherence "
                                f"capped at 1 (framework D5/QA-8 Feedback gate)."
                            ),
                            location=block_id,
                            suggestion="Add elaborated, misconception-targeted feedback.",
                        )
                    )
            elif _BARE_MARKER_RE.match(fb_text):
                status = "bare"
                block_codes.append("INTERACTION_BARE_RIGHT_WRONG")
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="INTERACTION_BARE_RIGHT_WRONG",
                            message=(
                                f"Block {block_id!r} ({bt}/{bcode}) feedback is a "
                                f"bare right/wrong marker ({fb_text!r}) — the "
                                f"framework requires WHY, not just WHETHER."
                            ),
                            location=block_id,
                            suggestion="Explain the reasoning / target the misconception.",
                        )
                    )
            elif len(fb_text.split()) < _ELABORATION_MIN_TOKENS:
                status = "thin"
                block_codes.append("INTERACTION_FEEDBACK_THIN")
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="INTERACTION_FEEDBACK_THIN",
                            message=(
                                f"Block {block_id!r} ({bt}/{bcode}) feedback is "
                                f"thin ({len(fb_text.split())} tokens < "
                                f"{_ELABORATION_MIN_TOKENS}) — not elaborated "
                                f"enough to teach the 'why'."
                            ),
                            location=block_id,
                            suggestion="Elaborate the feedback beyond a single clause.",
                        )
                    )

            # (c) misconception-targeting reuse — advisory, never recomputed.
            sig = distractor_signals.get(block_id) if isinstance(distractor_signals, dict) else None
            misconception_targeted = None
            if isinstance(sig, dict):
                misconception_targeted = bool(sig.get("misconception_targeted"))

            per_block[block_id] = {
                "framework_block": bcode,
                "feedback_status": status,  # elaborated | thin | bare | missing
                "feedback_token_count": len(fb_text.split()) if fb_text else 0,
                "misconception_targeted": misconception_targeted,
                "issues": block_codes,
            }

        self._emit_decision(
            capture, audited=audited, caps=len(caps_by_block), issue_count=len(issues)
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1
            issues=issues,
            action=None,
            metadata={
                "interaction_feedback": {
                    "enabled": True,
                    "interactive_blocks_audited": audited,
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
                decision=f"interaction_feedback_caps={caps}",
                rationale=(
                    f"IB6.3 interaction-feedback: audited {audited} interactive "
                    f"blocks (B07/B08/B10/B14) for feedback presence + "
                    f"elaboration; {issue_count} issues, {caps} blocks with no "
                    f"feedback (Feedback+Coherence cap-1)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on interaction_feedback: %s",
                exc,
            )


__all__ = ["InteractionFeedbackValidator"]
