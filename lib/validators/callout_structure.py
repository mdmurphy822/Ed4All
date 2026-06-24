"""FR-A11Y-03 — CalloutStructureValidator (typed B12 callouts, non-color coding).

The B12 ``callout`` block type highlights a note / tip / warning / example /
key-idea inline. WCAG 1.4.1 (Use of Color) requires the callout's CATEGORY be
conveyed by something other than color alone — a visible LABEL + icon, not just
a colored border. This validator audits every ``callout`` block and flags:

* ``CALLOUT_NO_KIND`` — the callout carries no ``callout_kind`` (untyped → the
  renderer cannot emit a redundant label/icon, so the category is implicit).
* ``CALLOUT_COLOR_ONLY`` — a str-content (rendered-HTML) callout whose markup
  carries a per-kind color/border class but NO visible label/icon row (the
  category is signalled by color alone — the 1.4.1 violation proper).
* ``CALLOUT_BODY_OVERFLOW`` — the callout body exceeds the ~200-char
  cognitive-load ceiling (reuses ``resolve_body_char_ceiling``; a callout is a
  single-idea highlight, not an everything-block).
* ``CALLOUT_MOTION`` — the callout markup carries motion/animation (a
  ``transition``/``animation``/``@keyframes`` / ``marquee`` / ``<blink>``)
  with no ``prefers-reduced-motion`` guard (WCAG 2.3.3 / 2.2.2).

It runs over BOTH block shapes (mirroring the ``Block*Validator`` /
``ResourceLinkPurposeValidator`` dual-path posture):

* dict content (outline tier) — reads ``content["items"]`` (the callout body
  text) for the overflow check; the color-only / motion checks are str-only
  (they need rendered markup).
* str content (rewrite-tier HTML) — strips tags for the overflow check and
  scans the raw markup for the color-only / motion signals.

Gating (default OFF, byte-stable): a no-op (``passed=True`` + an info issue)
unless ``ED4ALL_CALLOUT_TYPED`` is truthy — the same flag that makes the typed
callout renderer emit the redundant label/icon/border, so a default run never
sees this gate fire (mirrors the IB5 / B15 new-block-type no-op posture). When
the flag is on, fires WARNING-severity issues (``action="regenerate"``) —
warning-day-1 with a deferred critical-flip.

Inputs (``inputs`` dict):

    blocks: List[Block | dict]
        Outline- or rewrite-tier blocks. Only ``callout`` blocks are audited.
    callout_typed_enabled: Optional[bool]
        Override the ``ED4ALL_CALLOUT_TYPED`` resolution (testing seam).
    body_char_ceiling: Optional[int]
        Override the ~200-char body ceiling (testing seam).
    decision_capture / capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance; one
        ``content_structure_check`` event per validate.
    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to
        ``"callout_structure"``).

References:
    - WCAG 2.2 SC 1.4.1 Use of Color, SC 2.3.3 / 2.2.2 (motion).
    - ``lib/validators/resource_link_purpose.py`` — sibling block-type gate
      (structure mirrored: empty-input graceful pass, ``_ISSUE_LIST_CAP``,
      capture-emit, resolve-override seam).
    - ``lib/validators/_block_rubric_helpers.py::resolve_body_char_ceiling``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

#: Cap on per-validate issue list to avoid runaway emit on callout-dense corpora.
_ISSUE_LIST_CAP: int = 50

#: ED4ALL_CALLOUT_TYPED gate (mirrors the generate_course renderer flag).
_CALLOUT_TYPED_ENV = "ED4ALL_CALLOUT_TYPED"
_TRUTHY: frozenset = frozenset({"1", "true", "yes", "on"})

#: A per-kind color/border class the typed renderer emits (the color signal).
_KIND_CLASS_RE = re.compile(r"\bcallout-kind-[a-z-]+\b", re.IGNORECASE)

#: The visible label row the typed renderer emits alongside the color signal.
_LABEL_CLASS_RE = re.compile(r"\bcallout-(?:label|kind-label|icon)\b", re.IGNORECASE)

#: Motion / animation signals (WCAG 2.3.3 / 2.2.2).
_MOTION_RE = re.compile(
    r"(?:\btransition\s*:|\banimation\s*:|@keyframes\b|<marquee\b|<blink\b)",
    re.IGNORECASE,
)
#: A prefers-reduced-motion guard present anywhere in the markup clears motion.
_REDUCED_MOTION_RE = re.compile(r"prefers-reduced-motion", re.IGNORECASE)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_text(s: Any) -> str:
    """Strip HTML tags + collapse whitespace; ``""`` for non-string/empty."""
    if not isinstance(s, str) or not s:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", s)).strip()


def _callout_typed_enabled() -> bool:
    """True iff ``ED4ALL_CALLOUT_TYPED`` is truthy (read each call)."""
    return os.environ.get(_CALLOUT_TYPED_ENV, "").strip().lower() in _TRUTHY


def _block_attr(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    if hasattr(block, key):
        return getattr(block, key)
    return None


def _block_type_of(block: Any) -> str:
    bt = _block_attr(block, "block_type")
    return bt.strip().lower() if isinstance(bt, str) else ""


def _callout_body_text(block: Any) -> str:
    """Return the visible body text of a callout block (both shapes)."""
    content = _block_attr(block, "content")
    if isinstance(content, str):
        return _strip_text(content)
    if isinstance(content, dict):
        items = content.get("items")
        if isinstance(items, (list, tuple)):
            return _strip_text(
                " ".join(str(i) for i in items if isinstance(i, str))
            )
        # fall back to any string values
        parts = [v for v in content.values() if isinstance(v, str)]
        return _strip_text(" ".join(parts))
    return ""


class CalloutStructureValidator:
    """FR-A11Y-03 — typed B12 callout / non-color-coding gate."""

    name = "callout_structure"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        from lib.validators._block_rubric_helpers import resolve_body_char_ceiling

        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        enabled = inputs.get("callout_typed_enabled")
        if enabled is None:
            enabled = _callout_typed_enabled()
        if not enabled:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="CALLOUT_STRUCTURE_DISABLED",
                        message=(
                            "ED4ALL_CALLOUT_TYPED unset — callout-structure "
                            "check skipped (byte-stable)."
                        ),
                    )
                ],
                metadata={"callout_structure": {"enabled": False}},
            )

        ceiling = resolve_body_char_ceiling(inputs.get("body_char_ceiling"))

        blocks = inputs.get("blocks") or []
        issues: List[GateIssue] = []
        callout_blocks = 0
        flagged = 0
        per_block: Dict[str, Dict[str, Any]] = {}

        for idx, block in enumerate(blocks):
            if _block_type_of(block) != "callout":
                continue
            callout_blocks += 1
            block_id = str(_block_attr(block, "block_id") or f"block-{idx}")
            codes: List[str] = []

            kind = _block_attr(block, "callout_kind")
            content = _block_attr(block, "content")
            raw_html = content if isinstance(content, str) else ""

            # (a) CALLOUT_NO_KIND — untyped callout (no category to convey).
            if not (isinstance(kind, str) and kind.strip()):
                codes.append("CALLOUT_NO_KIND")

            # (b) CALLOUT_COLOR_ONLY — color/border class but no visible label.
            if raw_html and _KIND_CLASS_RE.search(raw_html):
                if not _LABEL_CLASS_RE.search(raw_html):
                    codes.append("CALLOUT_COLOR_ONLY")

            # (c) CALLOUT_BODY_OVERFLOW — single-idea highlight, not everything.
            body = _callout_body_text(block)
            if len(body) > ceiling:
                codes.append("CALLOUT_BODY_OVERFLOW")

            # (d) CALLOUT_MOTION — motion with no reduced-motion guard.
            if raw_html and _MOTION_RE.search(raw_html):
                if not _REDUCED_MOTION_RE.search(raw_html):
                    codes.append("CALLOUT_MOTION")

            for code in codes:
                flagged += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(self._issue_for(code, block_id, len(body), ceiling))

            per_block[block_id] = {
                "callout_kind": kind if isinstance(kind, str) else None,
                "body_chars": len(body),
                "issues": codes,
            }

        self._emit_decision(
            capture,
            callout_blocks=callout_blocks,
            flagged=flagged,
            ceiling=ceiling,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1
            issues=issues,
            action="regenerate" if flagged else None,
            metadata={
                "callout_structure": {
                    "enabled": True,
                    "callout_blocks": callout_blocks,
                    "flagged": flagged,
                    "body_char_ceiling": ceiling,
                    "per_block": per_block,
                }
            },
        )

    @staticmethod
    def _issue_for(
        code: str, block_id: str, body_chars: int, ceiling: int
    ) -> GateIssue:
        messages = {
            "CALLOUT_NO_KIND": (
                f"Callout block {block_id!r} carries no callout_kind — a typed "
                f"callout must declare its category (note/tip/warning/example/"
                f"key-idea) so the renderer can emit a redundant label + icon."
            ),
            "CALLOUT_COLOR_ONLY": (
                f"Callout block {block_id!r} signals its category by color/border "
                f"alone (a callout-kind-* class with no visible label/icon row) — "
                f"WCAG 1.4.1 requires a non-color cue."
            ),
            "CALLOUT_BODY_OVERFLOW": (
                f"Callout block {block_id!r} body is {body_chars} chars (> "
                f"{ceiling}) — a callout is a single-idea highlight, not an "
                f"everything-block."
            ),
            "CALLOUT_MOTION": (
                f"Callout block {block_id!r} carries motion/animation with no "
                f"prefers-reduced-motion guard — WCAG 2.3.3 / 2.2.2."
            ),
        }
        suggestions = {
            "CALLOUT_NO_KIND": "Set callout_kind to one of the typed kinds.",
            "CALLOUT_COLOR_ONLY": "Render a visible label + icon alongside the border.",
            "CALLOUT_BODY_OVERFLOW": "Trim the callout to a single idea.",
            "CALLOUT_MOTION": "Guard motion behind prefers-reduced-motion or remove it.",
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
        capture: Any,
        *,
        callout_blocks: int,
        flagged: int,
        ceiling: int,
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="content_structure_check",
                decision=f"callout_structure_flagged={flagged}",
                rationale=(
                    f"FR-A11Y-03 typed-callout audit over {callout_blocks} B12 "
                    f"callout block(s) (body ceiling {ceiling}): {flagged} "
                    f"non-color-coding / kind / overflow / motion issue(s)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on callout_structure: %s",
                exc,
            )


__all__ = ["CalloutStructureValidator"]
