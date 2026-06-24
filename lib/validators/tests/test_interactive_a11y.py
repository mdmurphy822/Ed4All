"""FR-A11Y-02 — tests for InteractiveA11yValidator.

Confirms the interaction-block WCAG floor:

  * flag OFF (default) → strict no-op (passed=True, BLOCK_A11Y_FLAG_OFF info,
    byte-stable even on a violating block);
  * a drag-only interaction with no keyboard alt → FLAGS
    ``DRAG_ONLY_NO_KEYBOARD``;
  * a drag interaction WITH a keyboard alt → PASS;
  * a colour-only signal with no companion cue → FLAGS
    ``COLOR_ONLY_SIGNALING``;
  * a colour signal WITH a text cue → PASS;
  * a clean interaction block → PASS;
  * empty input / no interaction blocks → graceful PASS;
  * the ``content_selection`` decision-capture fires.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from Courseforge.scripts.blocks import Block
from lib.validators.interactive_a11y import InteractiveA11yValidator


class _SpyCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self, *, decision_type: str, decision: str, rationale: str, **kw: Any
    ) -> None:
        self.events.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
                **kw,
            }
        )


def _block(
    block_type: str,
    *,
    content: str = "<p>body</p>",
    sequence: int = 0,
    page_id: str = "page1",
) -> Block:
    return Block(
        block_id=f"{page_id}#{block_type}_{sequence}",
        block_type=block_type,
        page_id=page_id,
        sequence=sequence,
        content=content,
    )


def _validate(
    blocks: List[Block],
    *,
    enabled: Optional[bool] = True,
    capture: Optional[_SpyCapture] = None,
):
    v = InteractiveA11yValidator()
    inputs: Dict[str, Any] = {"blocks": blocks}
    if enabled is not None:
        inputs["block_a11y_enabled"] = enabled
    if capture is not None:
        inputs["decision_capture"] = capture
    return v.validate(inputs)


# ---------------------------------------------------------------------------
# Flag OFF → strict no-op (byte-stable)
# ---------------------------------------------------------------------------


def test_flag_off_is_noop_even_with_violation(monkeypatch):
    monkeypatch.delenv("ED4ALL_BLOCK_A11Y", raising=False)
    # A drag-only block that WOULD fail when the flag is on.
    blocks = [
        _block(
            "activity",
            content="<div draggable='true'>drag the tiles into the bins</div>",
        )
    ]
    # enabled=None → resolver reads the (unset) env var → OFF.
    result = _validate(blocks, enabled=None)
    assert result.passed is True
    assert result.action is None
    codes = [i.code for i in result.issues]
    assert codes == ["BLOCK_A11Y_FLAG_OFF"]


def test_explicit_disabled_override_is_noop():
    blocks = [
        _block(
            "activity",
            content="<div class='draggable'>drag</div>",
        )
    ]
    result = _validate(blocks, enabled=False)
    assert result.passed is True
    assert [i.code for i in result.issues] == ["BLOCK_A11Y_FLAG_OFF"]


# ---------------------------------------------------------------------------
# DRAG_ONLY_NO_KEYBOARD
# ---------------------------------------------------------------------------


def test_drag_only_no_keyboard_flags():
    blocks = [
        _block(
            "activity",
            content="<div draggable='true'>drag the term onto its match</div>",
        )
    ]
    result = _validate(blocks)
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "DRAG_ONLY_NO_KEYBOARD" in codes
    issue = next(i for i in result.issues if i.code == "DRAG_ONLY_NO_KEYBOARD")
    assert issue.severity == "warning"


def test_drag_with_keyboard_alt_passes():
    blocks = [
        _block(
            "activity",
            content=(
                "<div draggable='true' tabindex='0'>drag the term, or use the "
                "arrow keys and press enter to place it</div>"
            ),
        )
    ]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


def test_swipe_gesture_without_alt_flags():
    # A swipe gesture in the markup with no keyboard/single-pointer path.
    blocks = [
        _block("activity", content="<div data-drag='true'>swipe to reorder</div>")
    ]
    result = _validate(blocks)
    assert result.passed is False
    assert "DRAG_ONLY_NO_KEYBOARD" in [i.code for i in result.issues]


# ---------------------------------------------------------------------------
# COLOR_ONLY_SIGNALING
# ---------------------------------------------------------------------------


def test_color_only_signaling_flags():
    blocks = [
        _block(
            "self_check_question",
            content="<p>Items shown in red are the ones to avoid.</p>",
        )
    ]
    result = _validate(blocks)
    assert result.passed is False
    codes = [i.code for i in result.issues]
    assert "COLOR_ONLY_SIGNALING" in codes
    issue = next(i for i in result.issues if i.code == "COLOR_ONLY_SIGNALING")
    assert issue.severity == "warning"


def test_color_with_text_cue_passes():
    blocks = [
        _block(
            "self_check_question",
            content=(
                "<p>The correct answer (marked ✓) is shown in green.</p>"
            ),
        )
    ]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# Clean / empty / non-interaction → PASS
# ---------------------------------------------------------------------------


def test_clean_interaction_block_passes():
    blocks = [
        _block(
            "self_check_question",
            content="<p>Which option is correct? Choose true or false.</p>",
        )
    ]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []
    assert result.score == 1.0


def test_empty_input_graceful_pass():
    result = _validate([])
    assert result.passed is True
    assert result.issues == []


def test_no_interaction_blocks_graceful_pass():
    # concept / explanation are not interaction blocks → not audited.
    blocks = [
        _block("concept", content="<p>shown in red</p>"),
        _block("explanation", content="<div draggable='true'>x</div>"),
    ]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# Decision capture fires
# ---------------------------------------------------------------------------


def test_decision_capture_fires_on_violation():
    capture = _SpyCapture()
    blocks = [
        _block(
            "activity",
            content="<div draggable='true'>drag</div>",
        )
    ]
    _validate(blocks, capture=capture)
    assert len(capture.events) == 1
    ev = capture.events[0]
    assert ev["decision_type"] == "content_selection"
    assert ev["decision"] == "interactive_a11y_violation"
    assert len(ev["rationale"]) >= 20
    feats = ev["ml_features"]
    assert feats["drag_only_no_keyboard"] is True
    assert feats["passed"] is False
