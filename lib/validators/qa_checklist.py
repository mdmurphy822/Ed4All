"""IB6.7 — 15-point per-block QA checklist surface.

A reusable per-block QA checklist (framework p.17) that composes already-scored
signals (IB1-IB6.6) into 15 measurable yes/no items + surfaces which item
fails. The QA surface is READ-ONLY over already-computed signals — it does NOT
recompute NLI / embeddings.

The 15 items (framework p.17):

  QA-1  single idea / <=200char       -> IB6.4 cognitive-load overflow
  QA-2  objective trace               -> block carries an objective_id ref
  QA-3  Bloom verb tag                 -> block carries bloom_verb / level
  QA-4  six elements present           -> IB6.2 anatomy slot-presence
  QA-5  constructive alignment         -> IB3 verb-triple status (aligned)
  QA-6  activation present             -> IB1 lifecycle activate-slot
  QA-7  interaction present            -> IB6.2 interaction slot (interactive)
  QA-8  elaborated feedback            -> IB6.3 interaction-feedback status
  QA-9  signaling restraint            -> NET-NEW: <=1 emphasis cluster/idea
  QA-10 dual coding + contiguity       -> >=2 representations (UDL) or media
  QA-11 coherence                      -> IB6.1 dim8 coherence score
  QA-12 WCAG 2.2 AA                     -> IB4 / rewrite_html_shape WCAG signal
  QA-13 UDL >=2 representations         -> NET-NEW: reuse IB4 n_representations
  QA-14 transition / transfer          -> IB6.2 transition slot
  QA-15 spacing                         -> NET-NEW: advisory until IB7 (unknown)

QA-9 / QA-13 / QA-15 are the 3 genuinely-net-new items; the other 12 map onto
upstream signals. ``passed = all 15 pass`` (warning-day-1 — "all pass before
publication" is the deferred-critical target). Default-off byte-stable.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

_ISSUE_LIST_CAP = 50

# QA-9 emphasis cluster detector — the "single strongest signal" rule. Over-
# signaling (many bold/callout/emphasis spans relative to idea-chunks) is noise.
_EMPHASIS_RE = re.compile(
    r"<(?:strong|b|em|mark)\b|data-cf-component=\"callout\"|class=\"[^\"]*callout",
    re.IGNORECASE,
)

# The 15 canonical QA item codes, ordered.
_QA_ITEMS: Tuple[Tuple[str, str], ...] = (
    ("QA-1", "single_idea_le_200char"),
    ("QA-2", "objective_trace"),
    ("QA-3", "bloom_verb_tag"),
    ("QA-4", "six_elements_present"),
    ("QA-5", "constructive_alignment"),
    ("QA-6", "activation_present"),
    ("QA-7", "interaction_present"),
    ("QA-8", "elaborated_feedback"),
    ("QA-9", "signaling_restraint"),
    ("QA-10", "dual_coding_contiguity"),
    ("QA-11", "coherence"),
    ("QA-12", "wcag_22_aa"),
    ("QA-13", "udl_two_representations"),
    ("QA-14", "transition_transfer"),
    ("QA-15", "spacing"),
)

# Net-new item failure codes.
_QA9_FAIL = "QA9_SIGNALING_EXCESS"
_QA13_FAIL = "QA13_SINGLE_REPRESENTATION"
_QA15_UNVERIFIED = "QA15_SPACING_UNVERIFIED"
_QA15_MASSED = "QA15_MASSED"


class QaChecklistValidator:
    """IB6.7 — compose upstream signals into a 15-item per-block QA checklist."""

    name = "qa_checklist"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        from lib.validators._block_rubric_helpers import (
            block_attr,
            block_quality_rubric_enabled,
            block_type_of,
            count_idea_chunks,
            framework_block_of,
            is_interactive_block,
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
                        code="QA_CHECKLIST_DISABLED",
                        message=(
                            "ED4ALL_BLOCK_QUALITY_RUBRIC unset — 15-point QA "
                            "checklist skipped (byte-stable)."
                        ),
                    )
                ],
                metadata={"qa_checklist": {"enabled": False}},
            )

        blocks = list(inputs.get("blocks") or [])
        signals = inputs.get("gate_results_by_block") or {}
        # Reuse the IB6.1 signal index so the checklist reads the same per-block
        # signal dicts the scorer composed (no recomputation).
        from lib.validators.block_quality_rubric import _index_signals

        per_block_signals = _index_signals(signals)
        # IB7 spacing signal (module-scope). Absent → "unknown" (advisory).
        spacing_by_block = inputs.get("spacing_by_block") or {}

        issues: List[GateIssue] = []
        per_block_report: Dict[str, Dict[str, Any]] = {}
        blocks_all_pass = 0
        audited = 0

        for idx, block in enumerate(blocks):
            bt = block_type_of(block)
            bcode = framework_block_of(block)
            if not bt or bcode is None:
                continue  # chrome / unknown — no QA contract
            audited += 1
            block_id = str(block_attr(block, "block_id") or f"block-{idx}")
            sig = per_block_signals.get(block_id, {})
            interactive = is_interactive_block(block)

            checklist = self._evaluate_block(
                block=block,
                block_id=block_id,
                framework_block=bcode,
                interactive=interactive,
                sig=sig,
                count_idea_chunks=count_idea_chunks,
                block_attr=block_attr,
                spacing=spacing_by_block.get(block_id),
            )
            all_pass = all(item["passed"] for item in checklist)
            if all_pass:
                blocks_all_pass += 1
            else:
                first_fail = next(
                    (it for it in checklist if not it["passed"]), None
                )
                if first_fail and len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code=first_fail["code"] or f"{first_fail['item']}_FAIL",
                            message=(
                                f"Block {block_id!r} ({bt}/{bcode}) fails "
                                f"{first_fail['item']} "
                                f"({first_fail['name']}); "
                                f"{sum(1 for it in checklist if not it['passed'])} "
                                f"of 15 QA items unmet."
                            ),
                            location=block_id,
                            suggestion="Address the failing QA item before publication.",
                        )
                    )
            per_block_report[block_id] = {
                "framework_block": bcode,
                "all_pass": all_pass,
                "checklist": checklist,
            }

        self._emit_decision(
            capture, audited=audited, all_pass=blocks_all_pass
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1
            score=(round(blocks_all_pass / audited, 4) if audited else 1.0),
            issues=issues,
            action=None,
            metadata={
                "qa_checklist": {
                    "enabled": True,
                    "blocks_audited": audited,
                    "blocks_all_pass": blocks_all_pass,
                    "per_block": per_block_report,
                }
            },
        )

    # ------------------------------------------------------------------
    def _evaluate_block(
        self,
        *,
        block: Any,
        block_id: str,
        framework_block: str,
        interactive: bool,
        sig: Dict[str, Any],
        count_idea_chunks,
        block_attr,
        spacing: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Return the 15 ``{item, name, passed, code}`` rows for one block."""

        def row(item: str, name: str, passed: bool, code: Optional[str] = None):
            return {"item": item, "name": name, "passed": bool(passed), "code": code}

        out: List[Dict[str, Any]] = []

        # QA-1 single idea / <=200char (IB6.4 overflow signal; pass when not overflow).
        overflow = _sig(sig, "cognitive_load_overflow", "block_cognitive_load.overflow")
        out.append(row("QA-1", "single_idea_le_200char", overflow is not True,
                       None if overflow is not True else "BLOCK_BODY_OVERFLOW"))

        # QA-2 objective trace.
        obj_ids = block_attr(block, "objective_ids")
        has_obj = bool(obj_ids)
        out.append(row("QA-2", "objective_trace", has_obj,
                       None if has_obj else "QA2_NO_OBJECTIVE_TRACE"))

        # QA-3 Bloom verb tag.
        has_bloom = bool(block_attr(block, "bloom_verb")) or bool(block_attr(block, "bloom_level"))
        out.append(row("QA-3", "bloom_verb_tag", has_bloom,
                       None if has_bloom else "QA3_NO_BLOOM_TAG"))

        # QA-4 six elements present (IB6.2 anatomy — pass when no slot-presence issue).
        anatomy_issues = _sig(sig, "anatomy_slot_presence.issues", default=[]) or []
        qa4 = not anatomy_issues
        out.append(row("QA-4", "six_elements_present", qa4,
                       None if qa4 else "QA4_SLOT_MISSING"))

        # QA-5 constructive alignment (IB3 verb-triple aligned / unverifiable→pass).
        verb_status = _sig(sig, "verb_triple_status", "alignment.verb_triple_status")
        qa5 = verb_status != "mismatch"
        out.append(row("QA-5", "constructive_alignment", qa5,
                       None if qa5 else "QA5_ALIGNMENT_MISMATCH"))

        # QA-6 activation present (IB1 lifecycle — heading/purpose_tag slot).
        has_activation = bool(block_attr(block, "heading")) or bool(block_attr(block, "purpose_tag"))
        out.append(row("QA-6", "activation_present", has_activation,
                       None if has_activation else "QA6_NO_ACTIVATION"))

        # QA-7 interaction present (only meaningful on interactive blocks).
        if interactive:
            has_interaction = bool(block_attr(block, "interaction"))
            out.append(row("QA-7", "interaction_present", has_interaction,
                           None if has_interaction else "ANATOMY_INTERACTION_MISSING"))
        else:
            out.append(row("QA-7", "interaction_present", True))

        # QA-8 elaborated feedback (IB6.3 status — interactive only).
        if interactive:
            fb_status = _sig(sig, "feedback_status", "interaction_feedback.feedback_status")
            qa8 = fb_status in ("elaborated", None)  # None = not threaded → don't fail day-1
            out.append(row("QA-8", "elaborated_feedback", qa8,
                           None if qa8 else "INTERACTION_FEEDBACK_THIN"))
        else:
            out.append(row("QA-8", "elaborated_feedback", True))

        # QA-9 signaling restraint (NET-NEW): <=1 emphasis cluster per idea-chunk.
        body = _block_body_html(block, block_attr)
        emphasis = len(_EMPHASIS_RE.findall(body)) if body else 0
        chunks = max(1, count_idea_chunks(block))
        qa9 = emphasis <= chunks
        out.append(row("QA-9", "signaling_restraint", qa9,
                       None if qa9 else _QA9_FAIL))

        # QA-10 dual coding + contiguity (>=2 representations OR a media/diagram block).
        n_repr = block_attr(block, "n_representations")
        n_repr = n_repr if isinstance(n_repr, int) else 0
        qa10 = n_repr >= 2 or framework_block in ("B04", "B06")
        out.append(row("QA-10", "dual_coding_contiguity", qa10,
                       None if qa10 else "QA10_SINGLE_CODE"))

        # QA-11 coherence (IB6.1 dim8 coherence score >= 2, else unknown→pass).
        coh = _sig(sig, "coherence_score", "coherence.score")
        qa11 = coh is None or (isinstance(coh, int) and coh >= 2)
        out.append(row("QA-11", "coherence", qa11,
                       None if qa11 else "QA11_INCOHERENT"))

        # QA-12 WCAG 2.2 AA (IB4 wcag_clean signal; unknown→pass day-1).
        wcag_clean = _sig(sig, "wcag_clean", "wcag.clean")
        qa12 = wcag_clean is not False
        out.append(row("QA-12", "wcag_22_aa", qa12,
                       None if qa12 else "QA12_WCAG_FAIL"))

        # QA-13 UDL >=2 representations (NET-NEW): reuse IB4 n_representations.
        qa13 = n_repr >= 2
        out.append(row("QA-13", "udl_two_representations", qa13,
                       None if qa13 else _QA13_FAIL))

        # QA-14 transition / transfer (IB6.2 transition slot).
        has_transition = bool(block_attr(block, "transition"))
        out.append(row("QA-14", "transition_transfer", has_transition,
                       None if has_transition else "ANATOMY_TRANSITION_MISSING"))

        # QA-15 spacing (NET-NEW): advisory until IB7 supplies the module-scope
        # spacing signal. Absent → unverified (warning, but never a hard fail).
        if spacing is None or spacing == "unknown":
            out.append(row("QA-15", "spacing", True, _QA15_UNVERIFIED))
        elif spacing == "massed":
            out.append(row("QA-15", "spacing", False, _QA15_MASSED))
        else:  # "spaced"
            out.append(row("QA-15", "spacing", True))

        return out

    @staticmethod
    def _emit_decision(capture: Any, *, audited: int, all_pass: int) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="content_selection",
                decision=f"qa_checklist_all_pass={all_pass}/{audited}",
                rationale=(
                    f"IB6.7 15-point QA checklist: audited {audited} "
                    f"pedagogical blocks by composing IB1-IB6.6 signals + the "
                    f"3 net-new items (signaling-restraint QA-9, "
                    f">=2-representations QA-13, spacing QA-15); {all_pass} "
                    f"blocks cleared all 15 items."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on qa_checklist: %s", exc
            )


def _sig(sig: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in sig and sig[k] is not None:
            return sig[k]
    return default


def _block_body_html(block: Any, block_attr) -> str:
    """Return the raw (un-stripped) body HTML for emphasis-cluster counting."""
    body = block_attr(block, "body")
    if isinstance(body, str) and body.strip():
        return body
    content = block_attr(block, "content")
    if isinstance(content, str):
        return content
    return ""


__all__ = ["QaChecklistValidator"]
