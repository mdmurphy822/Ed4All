"""FR-A11Y-02 — InteractiveA11yValidator.

Interactive-block accessibility floor. Two framework-aligned WCAG checks the
per-block ``rewrite_html_shape`` a11y contract (IB4.1) does not cover, both
specific to INTERACTIVE block surfaces:

  * ``DRAG_ONLY_NO_KEYBOARD`` (WCAG 2.1.1 Keyboard / 2.5.7 Dragging Movements)
    — an interaction block that depends on a drag / pointer-gesture
    affordance (drag-and-drop, slider-drag, swipe, pinch) with NO
    keyboard-operable OR single-pointer alternative is unusable by keyboard /
    switch / assistive-tech users.
  * ``COLOR_ONLY_SIGNALING`` (WCAG 1.4.1 Use of Color) — an interaction /
    feedback block that conveys state, answer-correctness, or category by
    COLOR ALONE (no accompanying text label, icon/symbol, or pattern) is
    inaccessible to colour-blind / low-vision learners.

Different surface from ``UdlCoverageValidator`` (multiple-means coverage
fields) and ``RewriteHtmlShapeValidator._check_block_a11y_contract`` (the
per-block-type alt-text / name-role-value / link-text contract): this
validator audits ONLY the two interaction-specific WCAG signals above by
inspecting the rendered ``content`` HTML + the block's a11y annotations.

Rides ``ED4ALL_BLOCK_A11Y`` (the same flag that gates IB4's per-block a11y
emit). Default OFF → strict no-op: ``validate`` returns ``passed=True`` with
a single info-severity ``BLOCK_A11Y_FLAG_OFF`` issue and no audit, so a
default-off run is byte-stable. When the flag is on it audits every
interaction-bearing block.

Inputs (``inputs`` dict):

    blocks: List[Block]
        Rewrite-tier ``Courseforge.scripts.blocks.Block`` instances. Reads
        ``block_type``, ``content`` (rendered HTML str), ``block_id`` /
        ``page_id``.

    block_a11y_enabled: Optional[bool]
        Explicit override of the ``ED4ALL_BLOCK_A11Y`` resolution (the
        gate-input router threads this in, mirroring the rewrite_html_shape
        + udl_coverage Block surface). When absent, the env var is read.

    decision_capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance. One ``content_selection``
        event per audited interaction block (dynamic signals: block_id,
        block_type, drag_only, color_only, verdict).

    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to ``"interactive_a11y"``).

References:
    - WCAG 2.2 SC 2.1.1, 2.5.7, 1.4.1.
    - ``lib/validators/retrieval_presence.py`` — sibling structural validator
      (capture-emit + graceful-pass posture mirrored here).
    - ``lib/generation/block_a11y.py`` — the ``ED4ALL_BLOCK_A11Y`` resolver.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

# Import bridge for Block (mirror of retrieval_presence.py).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover - import guard
    from blocks import Block  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback for partial installs
    Block = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block-type sets
# ---------------------------------------------------------------------------

#: Interaction-bearing block types audited for the two WCAG signals. These are
#: the learner-input / feedback surfaces (members of the canonical
#: ``Courseforge.scripts.blocks.BLOCK_TYPES`` palette) where a drag affordance
#: or colour-only signal would actually impair operation.
INTERACTION_BLOCK_TYPES: frozenset = frozenset(
    {
        "self_check_question",
        "reflection_prompt",
        "assessment_item",
        "discussion_prompt",
        "activity",
        "problem",
        "scenario",
        "flip_card_grid",
    }
)

#: Cap on per-validate issue list to avoid runaway emit on many-block corpora.
_ISSUE_LIST_CAP: int = 50

# WCAG 2.1.1 / 2.5.7 — drag / pointer-gesture affordance markers in the
# rendered HTML or class/role attributes. Presence flags a drag-dependent
# interaction unless a keyboard / single-pointer alternative is also present.
_DRAG_MARKERS = (
    "draggable",
    "drag-and-drop",
    "drag and drop",
    "dragstart",
    "ondragstart",
    "ondrag",
    "ondrop",
    "drag-drop",
    "drag_drop",
    "swipe",
    "pinch",
    "data-drag",
)

# Keyboard / single-pointer alternative markers — presence of any clears the
# DRAG_ONLY check (a drag block with a documented keyboard path is fine).
_KEYBOARD_ALT_MARKERS = (
    "keyboard",
    "tabindex",
    "onkeydown",
    "onkeyup",
    "onkeypress",
    "aria-keyshortcuts",
    "single-pointer",
    "single pointer",
    "tap to",
    "click to",
    "select with",
    "use the arrow",
    "press enter",
)

# WCAG 1.4.1 — colour-only signalling markers. A bare colour word / hex /
# rgb() / CSS colour style used to convey state with no companion text/icon.
_COLOR_STYLE_RE = re.compile(
    r"(?:color\s*:\s*(?:red|green|amber|orange|yellow))"
    r"|(?:#[0-9a-fA-F]{3,6})"
    r"|(?:rgb\s*\()"
    r"|(?:\bclass\s*=\s*[\"'][^\"']*\b(?:correct|incorrect|wrong|right|"
    r"success|error|danger|valid|invalid)\b)",
    re.IGNORECASE,
)

# Colour words used in prose to signal runtime/state/answer/category.
_COLOR_SIGNAL_WORDS = (
    "red",
    "green",
    "amber",
    "yellow",
    "the red",
    "the green",
    "shown in red",
    "shown in green",
    "highlighted in",
    "marked in",
)

# Companion non-colour cues — presence of any clears the COLOR_ONLY check
# (the state is also conveyed by text / icon / pattern).
_NON_COLOR_CUE_MARKERS = (
    "correct",
    "incorrect",
    "right answer",
    "wrong answer",
    "✓",
    "✗",
    "check",
    "cross",
    "icon",
    "aria-label",
    "label",
    "pattern",
    "dashed",
    "dotted",
    "underline",
    "asterisk",
    "*",
    "true",
    "false",
)


# ---------------------------------------------------------------------------
# Flag resolution
# ---------------------------------------------------------------------------


def _resolve_block_a11y(override: Any) -> bool:
    """Resolve ``ED4ALL_BLOCK_A11Y`` (explicit override wins).

    Delegates to ``lib.generation.block_a11y.resolve_block_a11y`` so the
    truthy-token grammar is identical to the IB4 emit path. Never raises —
    a broken resolver import degrades to OFF (byte-stable).
    """
    if isinstance(override, bool):
        return override
    try:
        from lib.generation.block_a11y import resolve_block_a11y

        if override is not None:
            return resolve_block_a11y(override)
        return resolve_block_a11y()
    except Exception as exc:  # noqa: BLE001 - never let resolver break the gate
        logger.debug("interactive_a11y: block_a11y resolve failed: %s", exc)
        return False


def _content_text(block: Any) -> str:
    """Return the block's rendered content as a lowercased string.

    The rewrite-tier ``content`` is an HTML str; the outline-tier shape is a
    dict — flatten a dict to its concatenated str values so the audit never
    crashes on the wrong tier (it simply sees no markers).
    """
    raw = getattr(block, "content", "")
    if isinstance(raw, dict):
        parts: List[str] = []
        for v in raw.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, (list, tuple)):
                parts.extend(str(x) for x in v)
        raw = " ".join(parts)
    return str(raw or "").lower()


# ---------------------------------------------------------------------------
# Decision-capture emit
# ---------------------------------------------------------------------------


def _emit_decision(
    capture: Optional[Any],
    *,
    block_id: str,
    block_type: str,
    drag_only: bool,
    color_only: bool,
    passed: bool,
) -> None:
    """Emit one ``content_selection`` event per audited interaction block."""
    if capture is None:
        return
    verdict = "interactive_a11y_ok" if passed else "interactive_a11y_violation"
    rationale = (
        f"interactive_a11y audit of block_id={block_id!r} "
        f"(block_type={block_type!r}): drag_only_no_keyboard={drag_only}, "
        f"color_only_signaling={color_only} -> {verdict}. An interaction "
        f"block must offer a keyboard / single-pointer alternative to any "
        f"drag gesture (WCAG 2.1.1/2.5.7) and must not convey state by "
        f"colour alone (WCAG 1.4.1)."
    )
    ml_features = {
        "block_id": block_id,
        "block_type": block_type,
        "drag_only_no_keyboard": bool(drag_only),
        "color_only_signaling": bool(color_only),
        "passed": bool(passed),
    }
    try:
        capture.log_decision(
            decision_type="content_selection",
            decision=verdict,
            rationale=rationale,
            ml_features=ml_features,
        )
    except Exception as exc:  # noqa: BLE001 - audit emit is best-effort
        logger.debug(
            "DecisionCapture.log_decision raised on interactive_a11y "
            "content_selection: %s",
            exc,
        )


# ---------------------------------------------------------------------------
# Per-block checks
# ---------------------------------------------------------------------------


def _is_drag_only(text: str, block: Any) -> bool:
    """True iff the block depends on a drag gesture with no keyboard/SP alt."""
    # The drag signal comes from the rendered HTML markers (a drag/gesture
    # affordance is observable in the markup, never assumed from block_type —
    # no canonical palette type is unambiguously drag-only, so type-based
    # flagging would be a false positive).
    has_drag = any(m in text for m in _DRAG_MARKERS)
    if not has_drag:
        return False
    has_alt = any(m in text for m in _KEYBOARD_ALT_MARKERS)
    return not has_alt


def _is_color_only(text: str) -> bool:
    """True iff runtime/state/answer/category is conveyed by colour with no companion
    text / icon / pattern cue."""
    has_color = bool(_COLOR_STYLE_RE.search(text)) or any(
        w in text for w in _COLOR_SIGNAL_WORDS
    )
    if not has_color:
        return False
    has_non_color = any(m in text for m in _NON_COLOR_CUE_MARKERS)
    return not has_non_color


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------


class InteractiveA11yValidator:
    """FR-A11Y-02 — interaction-block WCAG 2.1.1/2.5.7 + 1.4.1 floor.

    Validator-protocol-compatible class. Audits every interaction-bearing
    block for the two WCAG signals. Rides ``ED4ALL_BLOCK_A11Y`` — strict
    no-op (passed=True + info issue) when the flag is off. Empty input / no
    interaction blocks pass gracefully when the flag is on.
    """

    name = "interactive_a11y"
    version = "0.1.0"

    def __init__(self, *, decision_capture: Optional[Any] = None) -> None:
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or self._decision_capture

        enabled = _resolve_block_a11y(inputs.get("block_a11y_enabled"))
        if not enabled:
            # Strict no-op — byte-stable default-off run.
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="info",
                        code="BLOCK_A11Y_FLAG_OFF",
                        message=(
                            "ED4ALL_BLOCK_A11Y is off; interactive_a11y "
                            "audit skipped (no-op, byte-stable)."
                        ),
                    )
                ],
                action=None,
            )

        raw = inputs.get("blocks")
        if raw is None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="MISSING_BLOCKS_INPUT",
                        message=(
                            "inputs['blocks'] is required; expected a list "
                            "of Courseforge Block instances."
                        ),
                    )
                ],
                action="regenerate",
            )
        if not isinstance(raw, list):
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="INVALID_BLOCKS_INPUT",
                        message=(
                            f"inputs['blocks'] must be a list; got "
                            f"{type(raw).__name__}."
                        ),
                    )
                ],
                action="regenerate",
            )

        blocks: List[Any] = list(raw)

        issues: List[GateIssue] = []
        audited = 0
        violations = 0
        for b in blocks:
            bt = getattr(b, "block_type", None)
            if bt not in INTERACTION_BLOCK_TYPES:
                continue
            audited += 1
            text = _content_text(b)
            block_id = str(getattr(b, "block_id", "") or "")
            page_id = str(getattr(b, "page_id", "") or "")
            drag_only = _is_drag_only(text, b)
            color_only = _is_color_only(text)
            passed_block = not (drag_only or color_only)

            _emit_decision(
                capture,
                block_id=block_id,
                block_type=str(bt),
                drag_only=drag_only,
                color_only=color_only,
                passed=passed_block,
            )

            if passed_block:
                continue
            violations += 1
            if len(issues) >= _ISSUE_LIST_CAP:
                continue
            if drag_only:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="DRAG_ONLY_NO_KEYBOARD",
                        message=(
                            f"interaction block block_id={block_id!r} "
                            f"(block_type={bt!r}) depends on a drag / "
                            f"pointer-gesture affordance with no keyboard or "
                            f"single-pointer alternative (WCAG 2.1.1 / 2.5.7)."
                        ),
                        location=block_id or page_id,
                        suggestion=(
                            "Provide a keyboard-operable OR single-pointer "
                            "(tap/click/arrow-key) alternative to the drag "
                            "interaction, or document it in the block content."
                        ),
                    )
                )
            if color_only and len(issues) < _ISSUE_LIST_CAP:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="COLOR_ONLY_SIGNALING",
                        message=(
                            f"interaction block block_id={block_id!r} "
                            f"(block_type={bt!r}) conveys state / answer / "
                            f"category by colour alone, with no companion "
                            f"text label, icon, or pattern (WCAG 1.4.1)."
                        ),
                        location=block_id or page_id,
                        suggestion=(
                            "Add a non-colour cue (text label, icon/symbol, "
                            "or pattern) alongside the colour so the state is "
                            "perceivable without colour vision."
                        ),
                    )
                )

        if audited == 0:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
                action=None,
            )

        passed = violations == 0
        score = round((audited - violations) / audited, 4) if audited else 1.0
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None if passed else "regenerate",
        )


__all__ = [
    "InteractiveA11yValidator",
    "INTERACTION_BLOCK_TYPES",
]
