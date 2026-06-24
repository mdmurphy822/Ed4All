"""FR-INT-06 — B09DebriefValidator (mandatory case/scenario debrief).

A real B09 case/scenario/branching block is not complete when the learner
finishes the decision — the framework's B09 contract REQUIRES a DEBRIEF in the
block's transition/consolidate slot: a closing move that surfaces what the case
revealed and how to transfer it. A scenario that simply ends after the apply
prompt skips the consolidation that makes the situated practice stick.

This validator audits every ``scenario`` block (the Ed4All B09 type) and flags
``SCENARIO_DEBRIEF_MISSING`` when the block carries no debrief in its
transition/consolidate slot. The debrief is resolved from (in priority order):

* the Block ``transition`` slot (IB1 anatomy consolidate slot), OR
* dict content ``content["debrief"]`` / ``content["consolidate"]`` (outline
  tier), OR
* str content (rewrite-tier HTML) carrying a ``scenario-debrief`` /
  ``data-cf-purpose="debrief"`` marker or the word "debrief".

Gating (default OFF, byte-stable): a no-op (``passed=True`` + an info issue)
unless ``ED4ALL_NEW_BLOCK_TYPES`` is truthy — the same flag that makes the B09
scenario_mode / debrief renderer paths active, so a default run never sees this
gate fire (mirrors the B15 ``resource_link_purpose`` posture). When the flag is
on, fires WARNING-severity issues (``action="regenerate"``) — warning-day-1 with
a deferred critical-flip (``# TODO(calibration)`` at the gate in
``config/workflows.yaml``).

Inputs (``inputs`` dict):

    blocks: List[Block | dict]
        Outline- or rewrite-tier blocks. Only ``scenario`` blocks are audited.
    new_block_types_enabled: Optional[bool]
        Override the ``ED4ALL_NEW_BLOCK_TYPES`` resolution (testing seam).
    decision_capture / capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance; one
        ``content_structure_check`` event per validate.
    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to ``"b09_debrief"``).

References:
    - ``lib/validators/resource_link_purpose.py`` /
      ``lib/validators/b08_sequence.py`` — sibling block-type gates (structure
      mirrored: flag-off graceful no-op, ``_ISSUE_LIST_CAP``, capture-emit,
      resolve-override seam).
    - ``lib/generation/new_block_types.py`` — ``ED4ALL_NEW_BLOCK_TYPES`` gate.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

#: Cap on per-validate issue list to avoid runaway emit on dense corpora.
_ISSUE_LIST_CAP: int = 50

#: HTML markers that a rewrite-tier scenario block carries a debrief move.
_DEBRIEF_HTML_MARKERS = (
    "scenario-debrief",
    'data-cf-purpose="debrief"',
    "data-cf-purpose='debrief'",
)
_DEBRIEF_WORD_RE = re.compile(r"\bdebrief\b", re.IGNORECASE)


class B09DebriefValidator:
    """FR-INT-06 — mandatory case/scenario debrief gate for B09 blocks."""

    name = "b09_debrief"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        enabled = inputs.get("new_block_types_enabled")
        if enabled is None:
            from lib.generation.new_block_types import resolve_new_block_types

            enabled = resolve_new_block_types()
        if not enabled:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="B09_DEBRIEF_DISABLED",
                        message=(
                            "ED4ALL_NEW_BLOCK_TYPES unset — B09 scenario debrief "
                            "check skipped (byte-stable)."
                        ),
                    )
                ],
                metadata={"b09_debrief": {"enabled": False}},
            )

        blocks = inputs.get("blocks") or []
        issues: List[GateIssue] = []
        scenario_blocks = 0
        missing_debrief = 0

        for idx, block in enumerate(blocks):
            if _block_type_of(block) != "scenario":
                continue
            scenario_blocks += 1
            block_id = str(_block_attr(block, "block_id") or f"block-{idx}")
            if not _has_debrief(block):
                missing_debrief += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="SCENARIO_DEBRIEF_MISSING",
                            message=(
                                f"scenario block {block_id!r} has no debrief in "
                                f"its transition/consolidate slot — a B09 "
                                f"case/scenario MUST end with a debrief that "
                                f"surfaces what the case revealed and how to "
                                f"transfer it."
                            ),
                            location=block_id,
                            suggestion=(
                                "Add a debrief to the scenario's transition slot "
                                "(or content['debrief']) that consolidates the "
                                "case and names the transfer."
                            ),
                        )
                    )

        self._emit_decision(
            capture,
            scenario_blocks=scenario_blocks,
            missing_debrief=missing_debrief,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1
            issues=issues,
            action="regenerate" if missing_debrief else None,
            metadata={
                "b09_debrief": {
                    "enabled": True,
                    "scenario_blocks": scenario_blocks,
                    "scenario_missing_debrief": missing_debrief,
                }
            },
        )

    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        scenario_blocks: int,
        missing_debrief: int,
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="content_structure_check",
                decision=f"scenario_debrief_missing={missing_debrief}",
                rationale=(
                    f"B09 scenario debrief audit over {scenario_blocks} "
                    f"scenario block(s): {missing_debrief} missing a debrief in "
                    f"the transition/consolidate slot (case/scenario must "
                    f"consolidate with a debrief)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on b09_debrief: %s",
                exc,
            )


def _block_attr(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    if hasattr(block, key):
        return getattr(block, key)
    return None


def _block_type_of(block: Any) -> str:
    bt = _block_attr(block, "block_type")
    return bt.strip().lower() if isinstance(bt, str) else ""


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_debrief(block: Any) -> bool:
    """True iff the scenario block carries a debrief in its consolidate slot."""
    # IB1 anatomy transition (consolidate) slot — the canonical debrief seat.
    transition = _block_attr(block, "transition")
    if _nonempty_str(transition):
        return True

    content = _block_attr(block, "content")
    if isinstance(content, dict):
        if _nonempty_str(content.get("debrief")) or _nonempty_str(
            content.get("consolidate")
        ):
            return True
        return False
    if isinstance(content, str):
        lowered = content.lower()
        if any(marker.lower() in lowered for marker in _DEBRIEF_HTML_MARKERS):
            return True
        if _DEBRIEF_WORD_RE.search(content):
            return True
        return False
    return False


__all__ = ["B09DebriefValidator"]
