"""FR-INT-02 — B08SequenceValidator (guided-practice sequence + fade-state).

The FR-INT-02 ``guided_practice`` block type is the framework's B08 first-class
faded-scaffold practice block (distinct from the B07 ``self_check_question`` and
the generic B08 ``activity`` / ``problem``). The framework's gradual-release
contract for guided practice is two-fold:

1. A guided_practice block should FOLLOW a worked_example (B05) — practice comes
   after a fully-worked demonstration (the worked → completion → independent fade
   ladder, framework p.139).
2. A guided_practice block should carry a ``fade_state`` (``worked`` |
   ``completion`` | ``independent``) marking which gradual-release stage it
   realizes.

This validator audits the ordered block list for both contracts:

* ``B08_PRACTICE_NOT_AFTER_WORKED`` — a guided_practice block with no
  worked_example anywhere before it in the page/module sequence.
* ``B08_PRACTICE_NO_FADE_STATE`` — a guided_practice block carrying no
  ``fade_state``.

Gating (default OFF, byte-stable): a no-op (``passed=True`` + an info issue)
unless ``ED4ALL_NEW_BLOCK_TYPES`` is truthy — the same flag that makes the B08
``guided_practice`` type selectable / renderable, so a default run never sees
this gate fire (mirrors the B15 ``resource_link_purpose`` posture). When the flag
is on, fires WARNING-severity issues (``action="regenerate"``) — warning-day-1
with a deferred critical-flip (``# TODO(calibration)`` at the gate in
``config/workflows.yaml``).

Inputs (``inputs`` dict):

    blocks: List[Block | dict]
        Outline- or rewrite-tier blocks, IN ORDER. Only ``guided_practice``
        blocks are audited; ``worked_example`` predecessors gate contract 1.
    new_block_types_enabled: Optional[bool]
        Override the ``ED4ALL_NEW_BLOCK_TYPES`` resolution (testing seam).
    decision_capture / capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance; one
        ``content_structure_check`` event per validate.
    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to ``"b08_sequence"``).

References:
    - ``lib/validators/resource_link_purpose.py`` — sibling block-type gate
      (structure mirrored: flag-off graceful no-op, ``_ISSUE_LIST_CAP``,
      capture-emit, resolve-override seam).
    - ``lib/generation/new_block_types.py`` — ``ED4ALL_NEW_BLOCK_TYPES`` gate.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

#: Cap on per-validate issue list to avoid runaway emit on dense corpora.
_ISSUE_LIST_CAP: int = 50

#: Valid B08 guided-practice fade-ladder stages (the reused IB5 fade_state).
_VALID_FADE_STATES: frozenset = frozenset({"worked", "completion", "independent"})


class B08SequenceValidator:
    """FR-INT-02 — B08 guided-practice sequence + fade-state gate."""

    name = "b08_sequence"
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
                        code="B08_SEQUENCE_DISABLED",
                        message=(
                            "ED4ALL_NEW_BLOCK_TYPES unset — B08 guided-practice "
                            "sequence check skipped (byte-stable)."
                        ),
                    )
                ],
                metadata={"b08_sequence": {"enabled": False}},
            )

        blocks = inputs.get("blocks") or []
        issues: List[GateIssue] = []
        practice_blocks = 0
        not_after_worked = 0
        missing_fade = 0
        seen_worked_example = False

        for idx, block in enumerate(blocks):
            bt = _block_type_of(block)
            if bt == "worked_example":
                seen_worked_example = True
                continue
            if bt != "guided_practice":
                continue
            practice_blocks += 1
            block_id = str(_block_attr(block, "block_id") or f"block-{idx}")

            # Contract 1 — must FOLLOW a worked_example.
            if not seen_worked_example:
                not_after_worked += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="B08_PRACTICE_NOT_AFTER_WORKED",
                            message=(
                                f"guided_practice block {block_id!r} appears with "
                                f"no worked_example before it — the framework's "
                                f"gradual-release contract puts faded practice "
                                f"AFTER a fully-worked demonstration (worked -> "
                                f"completion -> independent)."
                            ),
                            location=block_id,
                            suggestion=(
                                "Place a worked_example (B05) before the "
                                "guided_practice block on the same page/module."
                            ),
                        )
                    )

            # Contract 2 — must carry a fade_state.
            fade_state = _block_attr(block, "fade_state")
            fade_norm = (
                fade_state.strip().lower()
                if isinstance(fade_state, str)
                else ""
            )
            if fade_norm not in _VALID_FADE_STATES:
                missing_fade += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="B08_PRACTICE_NO_FADE_STATE",
                            message=(
                                f"guided_practice block {block_id!r} carries no "
                                f"valid fade_state (got {fade_state!r}) — a B08 "
                                f"block must mark its gradual-release stage "
                                f"(worked | completion | independent)."
                            ),
                            location=block_id,
                            suggestion=(
                                "Stamp fade_state on the guided_practice block "
                                "(worked | completion | independent)."
                            ),
                        )
                    )

        self._emit_decision(
            capture,
            practice_blocks=practice_blocks,
            not_after_worked=not_after_worked,
            missing_fade=missing_fade,
        )

        flagged = not_after_worked + missing_fade
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1
            issues=issues,
            action="regenerate" if flagged else None,
            metadata={
                "b08_sequence": {
                    "enabled": True,
                    "practice_blocks": practice_blocks,
                    "practice_not_after_worked": not_after_worked,
                    "practice_missing_fade_state": missing_fade,
                }
            },
        )

    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        practice_blocks: int,
        not_after_worked: int,
        missing_fade: int,
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="content_structure_check",
                decision=(
                    f"b08_sequence_flagged="
                    f"{not_after_worked + missing_fade}"
                ),
                rationale=(
                    f"B08 guided-practice sequence audit over {practice_blocks} "
                    f"guided_practice block(s): {not_after_worked} not after a "
                    f"worked_example, {missing_fade} missing a fade_state "
                    f"(worked/completion/independent)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on b08_sequence: %s",
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


__all__ = ["B08SequenceValidator"]
