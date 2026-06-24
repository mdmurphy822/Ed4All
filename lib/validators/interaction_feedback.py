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
  quality; honest scope — the distractor engine stays untouched). FR-INT-05
  escalates that signal to a WARNING (``DISTRACTOR_NO_MISCONCEPTION_FEEDBACK``)
  when a B07/B14 block HAS distractors but neither the threaded signal nor the
  block's own ``option_feedback`` map covers them with per-option "why this is
  wrong" feedback — closing the distractor-naming gap.

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

        # FR-INT-03 — reflection-calibration gate (rides ED4ALL_REFLECTION_
        # CALIBRATION, default OFF → byte-stable). Optional override seam.
        reflection_calibration_enabled = inputs.get("reflection_calibration_enabled")
        if reflection_calibration_enabled is None:
            from lib.generation.reflection_calibration import (
                resolve_reflection_calibration,
            )

            reflection_calibration_enabled = resolve_reflection_calibration()

        issues: List[GateIssue] = []
        audited = 0
        caps_by_block: Dict[str, List[str]] = {}
        per_block: Dict[str, Dict[str, Any]] = {}

        for idx, block in enumerate(blocks):
            # FR-INT-03 — a B11 reflection is NOT in the interactive
            # (B07/B08/B10/B14) set, so audit it on its own arm before the
            # interactive short-circuit. No-op when the calibration flag is off.
            if (
                reflection_calibration_enabled
                and framework_block_of(block) == "B11"
            ):
                bt_r = block_type_of(block)
                block_id_r = str(block_attr(block, "block_id") or f"block-{idx}")
                refl_code = self._check_reflection_capture(
                    block=block,
                    bcode="B11",
                    calibration_enabled=True,
                )
                if refl_code is not None and len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code=refl_code,
                            message=(
                                f"Reflection block {block_id_r!r} ({bt_r}/B11) "
                                f"only asks — it carries no predict-then-reveal "
                                f"capture (prediction_prompt + reveal_content) and "
                                f"no calibration_feedback, so the learner never "
                                f"calibrates their judgment against a benchmark "
                                f"(FR-INT-03)."
                            ),
                            location=block_id_r,
                            suggestion=(
                                "Add a prediction_prompt (capture the learner's "
                                "judgment), a reveal_content benchmark, and "
                                "calibration_feedback comparing the two."
                            ),
                        )
                    )
                    per_block[block_id_r] = {
                        "framework_block": "B11",
                        "feedback_status": "reflection_no_capture",
                        "issues": [refl_code],
                    }
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

            # FR-INT-05 — escalate the threaded distractor_misconception_alignment
            # signal to a WARNING when a B07 knowledge-check's distractors lack
            # per-option misconception-targeted feedback. Closes the
            # distractor-naming gap: a knowledge-check whose wrong options carry no
            # "why this is wrong" feedback teaches nothing on a miss.
            misc_code = self._check_misconception_targeting(
                block=block,
                bcode=bcode,
                misconception_targeted=misconception_targeted,
            )
            if misc_code is not None:
                block_codes.append(misc_code)
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code=misc_code,
                            message=(
                                f"Knowledge-check block {block_id!r} ({bt}/{bcode}) "
                                f"has distractors but no per-option "
                                f"misconception-targeted feedback — the framework's "
                                f"Feedback dimension (D5/QA-8) requires each "
                                f"distractor explain WHY it is wrong."
                            ),
                            location=block_id,
                            suggestion=(
                                "Add per-option (option_feedback) misconception-"
                                "targeted feedback for each distractor."
                            ),
                        )
                    )

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
    def _check_misconception_targeting(
        *,
        block: Any,
        bcode: Any,
        misconception_targeted: Any,
    ) -> "str | None":
        """FR-INT-05 — return ``DISTRACTOR_NO_MISCONCEPTION_FEEDBACK`` when a
        B07 knowledge-check (or B14 graded assessment) carries distractors but
        no per-option misconception-targeted feedback; ``None`` otherwise.

        DETECTION precedence:

        * the threaded ``distractor_misconception_alignment`` signal
          (``misconception_targeted``) is AUTHORITATIVE when present — a truthy
          value clears the check (the distractor engine already confirmed
          misconception targeting), a falsey value (``False``, not ``None``)
          fires the warning.
        * with no threaded signal, fall back to the block's own
          ``option_feedback`` map: a knowledge-check that HAS distractors but
          whose distractor options carry no non-empty feedback fires.

        Scope: only blocks whose framework B-code is B07/B14 (the
        distractor-bearing types) AND that actually expose ≥1 distractor are
        audited; a check with no distractors (e.g. a free-response prompt)
        never fires (no distractor to target a misconception at).
        """
        from lib.validators._block_rubric_helpers import block_attr

        # Only the distractor-bearing knowledge/assessment types.
        if bcode not in ("B07", "B14"):
            return None

        distractors = InteractionFeedbackValidator._distractor_keys(block)
        if not distractors:
            # No distractors to target a misconception at — nothing to require.
            return None

        # Authoritative threaded signal wins (advisory, never recomputed).
        if misconception_targeted is True:
            return None
        if misconception_targeted is False:
            return "DISTRACTOR_NO_MISCONCEPTION_FEEDBACK"

        # No threaded signal → inspect the block's own per-option feedback map.
        raw = block_attr(block, "option_feedback")
        if isinstance(raw, dict) and raw:
            # At least one distractor must carry non-empty feedback.
            covered = any(
                isinstance(raw.get(d), str) and raw.get(d).strip()
                for d in distractors
            )
            if covered:
                return None
        return "DISTRACTOR_NO_MISCONCEPTION_FEEDBACK"

    @staticmethod
    def _check_reflection_capture(
        *,
        block: Any,
        bcode: Any,
        calibration_enabled: Any,
    ) -> "str | None":
        """FR-INT-03 — return ``REFLECTION_NO_CAPTURE`` when a B11 reflection
        only asks (no predict-then-reveal capture + no calibration feedback);
        ``None`` otherwise.

        A reflection CAPTURES when it carries BOTH a ``prediction_prompt`` and a
        ``reveal_content`` (the predict-then-reveal pair), OR a
        ``calibration_feedback`` benchmark. Scope: only B11 blocks, and only
        when ``calibration_enabled`` (rides ``ED4ALL_REFLECTION_CALIBRATION``;
        default OFF → byte-stable no-op).
        """
        from lib.validators._block_rubric_helpers import block_attr

        if not calibration_enabled or bcode != "B11":
            return None

        def _nonempty(v: Any) -> bool:
            return isinstance(v, str) and bool(v.strip())

        has_prediction = _nonempty(block_attr(block, "prediction_prompt"))
        has_reveal = _nonempty(block_attr(block, "reveal_content"))
        has_calibration = _nonempty(block_attr(block, "calibration_feedback"))

        # Captured when the predict-then-reveal pair is present, OR a calibration
        # benchmark is supplied.
        if (has_prediction and has_reveal) or has_calibration:
            return None
        return "REFLECTION_NO_CAPTURE"

    @staticmethod
    def _distractor_keys(block: Any) -> List[str]:
        """Return the distractor option identifiers for a knowledge-check block.

        A distractor is an option NOT marked correct. Reads the block's
        ``content`` dict ``options`` list (the canonical self-check / assessment
        shape ``{"text", "correct"}``). Returns the option text (or index-keyed
        fallback) for each non-correct option; ``[]`` when no options resolve.
        """
        from lib.validators._block_rubric_helpers import block_attr

        content = block_attr(block, "content")
        options = None
        if isinstance(content, dict):
            options = content.get("options")
        if not isinstance(options, list):
            return []
        keys: List[str] = []
        for i, opt in enumerate(options):
            if not isinstance(opt, dict):
                continue
            if opt.get("correct"):
                continue
            key = opt.get("text") or opt.get("key") or f"option_{i}"
            keys.append(str(key))
        return keys

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
