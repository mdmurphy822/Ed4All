"""recall_self_check — ``RecallSelfCheckFormatValidator`` (free-recall / cloze).

A recall-marked ``self_check_question`` block (``Block.recall_format`` ∈
{``free_recall``, ``cloze``}) must be a PRODUCE-the-answer prompt — NOT
recognition. This validator audits every recall-marked self-check and flags:

* ``RECALL_SELF_CHECK_OPTIONS_VISIBLE`` — the block still pre-enumerates answer
  options (radio inputs on the rewrite-HTML path, or an ``options`` list on the
  outline-dict path), so the learner recognizes rather than recalls.
* ``RECALL_SELF_CHECK_ANSWER_INLINE`` — the answer is revealed in the VISIBLE
  body (not behind a ``<details>`` reveal), defeating the retrieval attempt.

Runs over BOTH block shapes (mirroring the ``Block*Validator`` /
``CalloutStructureValidator`` dual-path posture): dict content (outline tier) and
str content (rewrite-tier HTML). Non-recall self_check blocks AND every other
block type are ignored.

Gating (default OFF, byte-stable): a no-op (``passed=True`` + a
``RECALL_SELF_CHECK_DISABLED`` info issue) unless ``ED4ALL_RECALL_SELF_CHECK`` is
truthy — the same flag that makes the recall renderer / planner pass fire, so a
default run never sees this gate. When on, fires WARNING-severity issues
(``action="regenerate"``) — warning-day-1 with a ``# TODO(calibration)`` deferred
critical-flip.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

_ISSUE_LIST_CAP: int = 50

#: A radio input the legacy recognition renderer emits (the recognition signal).
_RADIO_RE = re.compile(r"<input[^>]*type\s*=\s*[\"']?radio", re.IGNORECASE)
#: A <details> reveal disclosure — the answer-hiding contract.
_DETAILS_RE = re.compile(r"<details\b", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
#: Everything inside a <details>…</details> (the legitimately-hidden answer).
_DETAILS_BLOCK_RE = re.compile(r"<details\b.*?</details>", re.IGNORECASE | re.DOTALL)


def _block_attr(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _block_type_of(block: Any) -> str:
    bt = _block_attr(block, "block_type")
    return bt.strip().lower() if isinstance(bt, str) else ""


def _recall_format_of(block: Any) -> Optional[str]:
    rf = _block_attr(block, "recall_format")
    if isinstance(rf, str) and rf.strip().lower() in {"free_recall", "cloze"}:
        return rf.strip().lower()
    return None


def _visible_text_outside_details(html: str) -> str:
    """Return the visible body text with any <details> block removed."""
    stripped = _DETAILS_BLOCK_RE.sub(" ", html)
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", stripped)).strip().lower()


class RecallSelfCheckFormatValidator:
    """recall_self_check — free-recall / cloze produce-the-answer gate."""

    name = "recall_self_check_format"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        enabled = inputs.get("recall_self_check_enabled")
        if enabled is None:
            try:
                from lib.generation.recall_self_check import (
                    resolve_recall_self_check,
                )

                enabled = resolve_recall_self_check()
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
                        code="RECALL_SELF_CHECK_DISABLED",
                        message=(
                            "ED4ALL_RECALL_SELF_CHECK unset — recall-self-check "
                            "format check skipped (byte-stable)."
                        ),
                    )
                ],
                metadata={"recall_self_check_format": {"enabled": False}},
            )

        blocks = inputs.get("blocks") or []
        issues: List[GateIssue] = []
        recall_blocks = 0
        flagged = 0

        for idx, block in enumerate(blocks):
            if _block_type_of(block) != "self_check_question":
                continue
            fmt = _recall_format_of(block)
            if fmt is None:
                continue  # non-recall self_check — ignored
            recall_blocks += 1
            block_id = str(_block_attr(block, "block_id") or f"block-{idx}")
            content = _block_attr(block, "content")
            codes: List[str] = []

            if isinstance(content, str):
                html = content
                if _RADIO_RE.search(html):
                    codes.append("RECALL_SELF_CHECK_OPTIONS_VISIBLE")
                # answer inline = a correct-option text present in the visible
                # body (outside any <details>) is hard to know without the
                # answer, so the conservative signal is: NO <details> reveal at
                # all on a recall block → the answer (if any) is not hidden.
                if not _DETAILS_RE.search(html):
                    codes.append("RECALL_SELF_CHECK_ANSWER_INLINE")
            elif isinstance(content, dict):
                # Outline-tier descriptor: a recall-marked block that still
                # carries enumerated options is recognition.
                options = content.get("options")
                if isinstance(options, (list, tuple)) and options:
                    codes.append("RECALL_SELF_CHECK_OPTIONS_VISIBLE")

            for code in codes:
                flagged += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(self._issue_for(code, block_id, fmt))

        self._emit_decision(
            capture, recall_blocks=recall_blocks, flagged=flagged
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1 (# TODO(calibration) deferred flip)
            issues=issues,
            action="regenerate" if flagged else None,
            metadata={
                "recall_self_check_format": {
                    "enabled": True,
                    "recall_blocks": recall_blocks,
                    "flagged": flagged,
                }
            },
        )

    @staticmethod
    def _issue_for(code: str, block_id: str, fmt: str) -> GateIssue:
        messages = {
            "RECALL_SELF_CHECK_OPTIONS_VISIBLE": (
                f"Recall self-check {block_id!r} (format={fmt}) still presents "
                f"pre-enumerated options — a recall item must be "
                f"produce-the-answer, not recognition."
            ),
            "RECALL_SELF_CHECK_ANSWER_INLINE": (
                f"Recall self-check {block_id!r} (format={fmt}) has no "
                f"<details> reveal — the answer must be hidden so the learner "
                f"attempts retrieval first."
            ),
        }
        suggestions = {
            "RECALL_SELF_CHECK_OPTIONS_VISIBLE": (
                "Drop the enumerated options; ask the learner to produce the "
                "answer (free recall) or fill a ____ blank (cloze)."
            ),
            "RECALL_SELF_CHECK_ANSWER_INLINE": (
                "Put the answer behind a <details><summary>Show answer</summary> "
                "reveal."
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
        capture: Any, *, recall_blocks: int, flagged: int
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="content_structure_check",
                decision=f"recall_self_check_flagged={flagged}",
                rationale=(
                    f"recall_self_check format audit over {recall_blocks} "
                    f"recall-marked self_check_question block(s): {flagged} "
                    f"options-visible / answer-inline issue(s)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on recall_self_check: %s",
                exc,
            )


__all__ = ["RecallSelfCheckFormatValidator"]
