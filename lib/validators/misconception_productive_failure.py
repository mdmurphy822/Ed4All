"""misconception_rich — ``MisconceptionProductiveFailureValidator``.

A productive-failure misconception block NAMES its targeted faulty mental model
and follows a predict → reveal → reconcile structure (not a generic
claim → correction card). This validator audits every misconception block and
flags:

* ``MISCONCEPTION_NO_NAMED_CONCEPT`` — the block names no target concept
  (``Block.mc_named_concept`` absent on the outline path, or no
  ``misconception-predict`` / named-concept marker in the rewrite HTML).
* ``MISCONCEPTION_NO_PRODUCTIVE_FAILURE`` — the rewrite-HTML card carries no
  predict-then-reveal / reconcile structure (just a claim → correction).

Runs over BOTH block shapes (dict outline tier / str rewrite HTML), mirroring the
``CalloutStructureValidator`` dual-path posture. Non-misconception blocks ignored.

Gating (default OFF, byte-stable): a no-op (``passed=True`` +
``MISCONCEPTION_RICH_DISABLED`` info) unless ``ED4ALL_MISCONCEPTION_RICH`` is
truthy. When on, fires WARNING-severity issues (``action="regenerate"``) —
warning-day-1 with a ``# TODO(calibration)`` deferred critical-flip.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

_ISSUE_LIST_CAP: int = 50

#: The predict-then-reveal marker the productive-failure renderer / rewrite tier
#: emits (a <p class="misconception-predict"> or the mc-predict scaffold class).
_PREDICT_RE = re.compile(
    r"\b(?:misconception-predict|mc-predict|misconception-productive-failure)\b",
    re.IGNORECASE,
)
#: The reconcile marker (why the named model fails).
_RECONCILE_RE = re.compile(r"\b(?:misconception-reconcile|mc-reconcile)\b", re.IGNORECASE)
#: The named-concept marker.
_NAMED_RE = re.compile(r"\b(?:mc-named-concept|data-cf-concept-id|conceptId)\b", re.IGNORECASE)


def _block_attr(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _block_type_of(block: Any) -> str:
    bt = _block_attr(block, "block_type")
    return bt.strip().lower() if isinstance(bt, str) else ""


class MisconceptionProductiveFailureValidator:
    """misconception_rich — named-concept + productive-failure gate."""

    name = "misconception_productive_failure"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        enabled = inputs.get("misconception_rich_enabled")
        if enabled is None:
            try:
                from lib.generation.misconception_rich import (
                    resolve_misconception_rich,
                )

                enabled = resolve_misconception_rich()
            except Exception:  # noqa: BLE001
                enabled = False
        if not enabled:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="MISCONCEPTION_RICH_DISABLED",
                        message=(
                            "ED4ALL_MISCONCEPTION_RICH unset — productive-failure "
                            "check skipped (byte-stable)."
                        ),
                    )
                ],
                metadata={"misconception_productive_failure": {"enabled": False}},
            )

        blocks = inputs.get("blocks") or []
        issues: List[GateIssue] = []
        misconception_blocks = 0
        flagged = 0

        for idx, block in enumerate(blocks):
            if _block_type_of(block) != "misconception":
                continue
            misconception_blocks += 1
            block_id = str(_block_attr(block, "block_id") or f"block-{idx}")
            content = _block_attr(block, "content")
            named = _block_attr(block, "mc_named_concept")
            has_named = isinstance(named, str) and bool(named.strip())
            codes: List[str] = []

            if isinstance(content, str):
                html = content
                if not (has_named or _NAMED_RE.search(html)):
                    codes.append("MISCONCEPTION_NO_NAMED_CONCEPT")
                if not (_PREDICT_RE.search(html) and _RECONCILE_RE.search(html)):
                    codes.append("MISCONCEPTION_NO_PRODUCTIVE_FAILURE")
            else:
                # Outline-tier descriptor: the named-concept link must be
                # resolved (the productive-failure prose is authored downstream).
                if not has_named:
                    codes.append("MISCONCEPTION_NO_NAMED_CONCEPT")

            for code in codes:
                flagged += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(self._issue_for(code, block_id))

        self._emit_decision(
            capture, misconception_blocks=misconception_blocks, flagged=flagged
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1 (# TODO(calibration) deferred flip)
            issues=issues,
            action="regenerate" if flagged else None,
            metadata={
                "misconception_productive_failure": {
                    "enabled": True,
                    "misconception_blocks": misconception_blocks,
                    "flagged": flagged,
                }
            },
        )

    @staticmethod
    def _issue_for(code: str, block_id: str) -> GateIssue:
        messages = {
            "MISCONCEPTION_NO_NAMED_CONCEPT": (
                f"Misconception block {block_id!r} names no targeted concept — a "
                f"rich misconception targets a NAMED faulty mental model grounded "
                f"in the block's concept tags."
            ),
            "MISCONCEPTION_NO_PRODUCTIVE_FAILURE": (
                f"Misconception block {block_id!r} carries no predict → reveal → "
                f"reconcile structure — it is a generic claim → correction card."
            ),
        }
        suggestions = {
            "MISCONCEPTION_NO_NAMED_CONCEPT": (
                "Stamp mc_named_concept from the block's grounded concept_tags."
            ),
            "MISCONCEPTION_NO_PRODUCTIVE_FAILURE": (
                "Add a predict-then-reveal prompt before the correction and a "
                "reconcile step explaining why the named model fails."
            ),
        }
        return GateIssue(
            severity="warning",
            code=code,
            message=messages.get(code, code),
            location=block_id,
            suggestion=suggestions.get(code),
        )

    @staticmethod
    def _emit_decision(
        capture: Any, *, misconception_blocks: int, flagged: int
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="content_selection",
                decision=f"misconception_productive_failure_flagged={flagged}",
                rationale=(
                    f"misconception_rich productive-failure audit over "
                    f"{misconception_blocks} misconception block(s): {flagged} "
                    f"missing-named-concept / no-productive-failure issue(s)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "misconception_productive_failure: %s",
                exc,
            )


__all__ = ["MisconceptionProductiveFailureValidator"]
