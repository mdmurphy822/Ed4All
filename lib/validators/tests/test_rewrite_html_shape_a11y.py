"""IB4.1 + IB4.7 — per-block WCAG 2.2 AA contract sub-check + decision capture.

Verifies the new ``REWRITE_BLOCK_A11Y_CONTRACT`` warning sub-check fires
correctly when ``block_a11y_enabled`` is on, is a complete no-op when off
(byte-stability), never flips the gate to failed (warning is non-blocking
day-1), and emits exactly one ``rewrite_block_a11y_check`` decision per audited
block. Also re-runs the existing four-part-shape contract module to confirm the
critical behavior is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest  # noqa: F401

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.rewrite_html_shape import RewriteHtmlShapeValidator  # noqa: E402


class _StubCapture:
    """Records every log_decision call for assertion."""

    def __init__(self) -> None:
        self.calls = []

    def log_decision(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _block(content, *, block_type="concept", block_id="page_01#concept_demo_0"):
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content=content,
    )


def _validate(blocks, *, a11y=False, capture=None):
    inputs = {"blocks": blocks, "block_a11y_enabled": a11y}
    if capture is not None:
        inputs["decision_capture"] = capture
    return RewriteHtmlShapeValidator().validate(inputs)


def _codes(result):
    return [i.code for i in result.issues]


def _critical(result):
    return [i for i in result.issues if i.severity == "critical"]


# --------------------------------------------------------------------------- #
# (a) non-keyboard-operable interactive component → warning, passed=True
# --------------------------------------------------------------------------- #


def test_self_check_div_no_keyboard_warns_but_passes() -> None:
    html = (
        '<div data-cf-block-id="page_01#self_check_question_q_0" '
        'data-cf-component="self-check" data-cf-purpose="formative-assessment" '
        'data-cf-bloom-level="remember"><p>Pick the right answer.</p></div>'
    )
    block = _block(
        html, block_type="self_check_question",
        block_id="page_01#self_check_question_q_0",
    )
    result = _validate([block], a11y=True)
    assert "REWRITE_BLOCK_A11Y_CONTRACT" in _codes(result)
    assert result.passed is True  # warning is non-blocking
    assert _critical(result) == []


# --------------------------------------------------------------------------- #
# (b) the same block with native <details>/<summary> → clean
# --------------------------------------------------------------------------- #


def test_self_check_details_summary_is_clean() -> None:
    html = (
        '<div data-cf-block-id="page_01#self_check_question_q_1" '
        'data-cf-component="self-check" data-cf-purpose="formative-assessment" '
        'data-cf-bloom-level="remember">'
        '<details><summary>Reveal answer</summary><p>The answer.</p></details>'
        "</div>"
    )
    block = _block(
        html, block_type="self_check_question",
        block_id="page_01#self_check_question_q_1",
    )
    result = _validate([block], a11y=True)
    assert "REWRITE_BLOCK_A11Y_CONTRACT" not in _codes(result)
    assert result.passed is True


def test_self_check_div_with_full_keyboard_contract_is_clean() -> None:
    html = (
        '<div data-cf-block-id="page_01#self_check_question_q_2" '
        'data-cf-component="self-check" data-cf-purpose="formative-assessment" '
        'data-cf-bloom-level="remember" tabindex="0" role="button" '
        'aria-label="Check your understanding"><p>q</p></div>'
    )
    block = _block(
        html, block_type="self_check_question",
        block_id="page_01#self_check_question_q_2",
    )
    result = _validate([block], a11y=True)
    assert "REWRITE_BLOCK_A11Y_CONTRACT" not in _codes(result)
    assert result.passed is True


# --------------------------------------------------------------------------- #
# (c) concept with alt-less <img> → warning; with alt → clean
# --------------------------------------------------------------------------- #


def _concept_with_img(img_tag):
    return (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        f"<p>prose</p>{img_tag}</section>"
    )


def test_concept_img_no_alt_warns() -> None:
    block = _block(_concept_with_img('<img src="a.png">'))
    result = _validate([block], a11y=True)
    assert "REWRITE_BLOCK_A11Y_CONTRACT" in _codes(result)
    assert result.passed is True


def test_concept_img_with_alt_is_clean() -> None:
    block = _block(_concept_with_img('<img src="a.png" alt="A scatter plot">'))
    result = _validate([block], a11y=True)
    assert "REWRITE_BLOCK_A11Y_CONTRACT" not in _codes(result)
    assert result.passed is True


def test_concept_img_decorative_is_clean() -> None:
    block = _block(_concept_with_img('<img src="a.png" role="presentation">'))
    result = _validate([block], a11y=True)
    assert "REWRITE_BLOCK_A11Y_CONTRACT" not in _codes(result)
    assert result.passed is True


# --------------------------------------------------------------------------- #
# (d) generic link text → warning
# --------------------------------------------------------------------------- #


def test_generic_link_text_warns() -> None:
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<p>See <a href="/docs">click here</a> for more.</p></section>'
    )
    block = _block(html)
    result = _validate([block], a11y=True)
    assert "REWRITE_BLOCK_A11Y_CONTRACT" in _codes(result)
    assert result.passed is True


def test_descriptive_link_text_is_clean() -> None:
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<p>See the <a href="/docs">federation setup guide</a>.</p></section>'
    )
    block = _block(html)
    result = _validate([block], a11y=True)
    assert "REWRITE_BLOCK_A11Y_CONTRACT" not in _codes(result)
    assert result.passed is True


# --------------------------------------------------------------------------- #
# Byte-stability: flag off → ZERO a11y issues regardless of markup
# --------------------------------------------------------------------------- #


def test_flag_off_emits_zero_a11y_issues() -> None:
    # alt-less img + generic link, both flag-on triggers.
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<p>See <a href="/docs">click here</a></p><img src="a.png"></section>'
    )
    block = _block(html)
    result = _validate([block], a11y=False)
    assert "REWRITE_BLOCK_A11Y_CONTRACT" not in _codes(result)
    assert result.passed is True


def test_flag_absent_key_emits_zero_a11y_issues() -> None:
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<p>x</p><img src="a.png"></section>'
    )
    block = _block(html)
    # No block_a11y_enabled key at all (the orchestrated default before IB4.6).
    result = RewriteHtmlShapeValidator().validate({"blocks": [block]})
    assert "REWRITE_BLOCK_A11Y_CONTRACT" not in _codes(result)


# --------------------------------------------------------------------------- #
# IB4.7 — decision capture fires exactly once per audited block (flag-on)
# --------------------------------------------------------------------------- #


def test_a11y_decision_fires() -> None:
    cap = _StubCapture()
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<p>x</p><img src="a.png"></section>'
    )
    block = _block(html)
    _validate([block], a11y=True, capture=cap)
    a11y_calls = [c for c in cap.calls if c.get("decision_type") == "rewrite_block_a11y_check"]
    assert len(a11y_calls) == 1
    rationale = a11y_calls[0]["rationale"]
    # Dynamic, replayable rationale fields.
    for token in ("block_type=", "n_representations=", "a11y_reason=",
                  "saw_img=", "interactive_role=", "link_text_flagged=",
                  "flag=ED4ALL_BLOCK_A11Y"):
        assert token in rationale
    assert len(rationale) >= 20


def test_no_a11y_decision_when_flag_off() -> None:
    cap = _StubCapture()
    block = _block(_concept_with_img('<img src="a.png">'))
    _validate([block], a11y=False, capture=cap)
    a11y_calls = [c for c in cap.calls if c.get("decision_type") == "rewrite_block_a11y_check"]
    assert a11y_calls == []


def test_a11y_decision_type_accepted_by_schema() -> None:
    import json

    schema_path = _REPO_ROOT / "schemas" / "events" / "decision_event.schema.json"
    schema = json.loads(schema_path.read_text())
    enum = schema["properties"]["decision_type"]["enum"]
    assert "rewrite_block_a11y_check" in enum


# --------------------------------------------------------------------------- #
# Keyboard operability (WCAG 2.1.1) — BLOCK_KEYBOARD_OPERABLE
# --------------------------------------------------------------------------- #


def test_clickonly_div_with_role_warns_keyboard_operable() -> None:
    """A click-only custom control with role+tabindex+name is FOCUSABLE but not
    OPERABLE (no native semantics, no key handler) → BLOCK_KEYBOARD_OPERABLE."""
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<div role="button" tabindex="0" aria-label="Reveal" '
        'onclick="reveal()">Reveal</div></section>'
    )
    block = _block(html)
    result = _validate([block], a11y=True)
    assert "BLOCK_KEYBOARD_OPERABLE" in _codes(result)
    assert result.passed is True  # warning-day-1
    assert _critical(result) == []


def test_clickonly_div_with_keydown_handler_is_clean() -> None:
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<div role="button" tabindex="0" aria-label="Reveal" '
        'onclick="reveal()" onkeydown="reveal(event)">Reveal</div></section>'
    )
    block = _block(html)
    result = _validate([block], a11y=True)
    assert "BLOCK_KEYBOARD_OPERABLE" not in _codes(result)


def test_clickonly_div_with_data_cf_keys_marker_is_clean() -> None:
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<div role="button" tabindex="0" aria-label="Reveal" '
        'onclick="reveal()" data-cf-keys="Enter Space">Reveal</div></section>'
    )
    block = _block(html)
    result = _validate([block], a11y=True)
    assert "BLOCK_KEYBOARD_OPERABLE" not in _codes(result)


def test_native_button_is_clean() -> None:
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<button onclick="reveal()">Reveal</button></section>'
    )
    block = _block(html)
    result = _validate([block], a11y=True)
    assert "BLOCK_KEYBOARD_OPERABLE" not in _codes(result)


def test_clickonly_div_rescued_by_details_summary_in_block() -> None:
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<div data-cf-component="reveal" onclick="x()">click</div>'
        '<details><summary>More</summary><p>body</p></details></section>'
    )
    block = _block(html)
    result = _validate([block], a11y=True)
    assert "BLOCK_KEYBOARD_OPERABLE" not in _codes(result)


def test_keyboard_operable_noop_when_flag_off() -> None:
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<div role="button" tabindex="0" aria-label="r" onclick="x()">r</div>'
        '</section>'
    )
    block = _block(html)
    result = _validate([block], a11y=False)
    assert "BLOCK_KEYBOARD_OPERABLE" not in _codes(result)
    assert result.passed is True


# --------------------------------------------------------------------------- #
# Visible focus (WCAG 2.4.7) — BLOCK_FOCUS_VISIBLE
# --------------------------------------------------------------------------- #


def test_outline_none_on_button_warns_focus_visible() -> None:
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<button style="outline:none">Go</button></section>'
    )
    block = _block(html)
    result = _validate([block], a11y=True)
    assert "BLOCK_FOCUS_VISIBLE" in _codes(result)
    assert result.passed is True
    assert _critical(result) == []


def test_outline_zero_on_link_warns_focus_visible() -> None:
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<a href="/x" style="color:red; outline: 0;">deep dive</a></section>'
    )
    block = _block(html)
    result = _validate([block], a11y=True)
    assert "BLOCK_FOCUS_VISIBLE" in _codes(result)


def test_outline_none_with_boxshadow_replacement_is_clean() -> None:
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<button style="outline:none; box-shadow:0 0 0 2px blue">Go</button>'
        '</section>'
    )
    block = _block(html)
    result = _validate([block], a11y=True)
    assert "BLOCK_FOCUS_VISIBLE" not in _codes(result)


def test_outline_none_on_non_interactive_element_is_clean() -> None:
    # A bare <p> is not interactive; outline:none on it is harmless.
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<p style="outline:none">prose</p></section>'
    )
    block = _block(html)
    result = _validate([block], a11y=True)
    assert "BLOCK_FOCUS_VISIBLE" not in _codes(result)


def test_focus_visible_noop_when_flag_off() -> None:
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" data-cf-key-terms="x">'
        '<button style="outline:none">Go</button></section>'
    )
    block = _block(html)
    result = _validate([block], a11y=False)
    assert "BLOCK_FOCUS_VISIBLE" not in _codes(result)


# --------------------------------------------------------------------------- #
# B04 time-based media per-piece codes (rides ED4ALL_NEW_BLOCK_TYPES)
# --------------------------------------------------------------------------- #


def _mm_block(html):
    return _block(
        html, block_type="multimedia",
        block_id="page_01#multimedia_x_0",
    )


def _validate_mm(blocks, *, enabled=True):
    return RewriteHtmlShapeValidator().validate(
        {"blocks": blocks, "new_block_types_enabled": enabled}
    )


_MM_FULL = (
    '<figure class="multimedia" data-cf-block-id="page_01#multimedia_x_0">'
    '<video controls><track kind="captions" srclang="en">'
    '<track kind="descriptions" srclang="en"></video>'
    '<details data-cf-transcript><summary>Transcript</summary><p>t</p></details>'
    '</figure>'
)


def test_mm_full_stack_passes_all_pieces() -> None:
    result = _validate_mm([_mm_block(_MM_FULL)])
    for code in (
        "MULTIMEDIA_CAPTIONS_MISSING", "MULTIMEDIA_AUDIO_DESC_MISSING",
        "MULTIMEDIA_TRANSCRIPT_MISSING", "MULTIMEDIA_CONTROLS_MISSING",
    ):
        assert code not in _codes(result)
    assert result.passed is True


def test_mm_missing_captions_flags_captions_only() -> None:
    html = _MM_FULL.replace('<track kind="captions" srclang="en">', "")
    result = _validate_mm([_mm_block(html)])
    assert "MULTIMEDIA_CAPTIONS_MISSING" in _codes(result)
    assert "MULTIMEDIA_AUDIO_DESC_MISSING" not in _codes(result)
    assert result.passed is True


def test_mm_missing_audio_desc_flags_audio_desc_only() -> None:
    html = _MM_FULL.replace('<track kind="descriptions" srclang="en">', "")
    result = _validate_mm([_mm_block(html)])
    assert "MULTIMEDIA_AUDIO_DESC_MISSING" in _codes(result)
    assert "MULTIMEDIA_CAPTIONS_MISSING" not in _codes(result)


def test_mm_audio_desc_via_class_marker_is_clean() -> None:
    # Matches the actual renderer skeleton: <p class="audio-description">.
    html = _MM_FULL.replace(
        '<track kind="descriptions" srclang="en">', ""
    ).replace(
        "</figure>",
        '<p class="audio-description">Audio description: …</p></figure>',
    )
    result = _validate_mm([_mm_block(html)])
    assert "MULTIMEDIA_AUDIO_DESC_MISSING" not in _codes(result)


def test_mm_missing_transcript_flags_transcript_only() -> None:
    html = (
        '<figure class="multimedia" data-cf-block-id="page_01#multimedia_x_0">'
        '<video controls><track kind="captions" srclang="en">'
        '<track kind="descriptions" srclang="en"></video></figure>'
    )
    result = _validate_mm([_mm_block(html)])
    assert "MULTIMEDIA_TRANSCRIPT_MISSING" in _codes(result)
    assert "MULTIMEDIA_CAPTIONS_MISSING" not in _codes(result)


def test_mm_missing_controls_flags_controls_only() -> None:
    html = _MM_FULL.replace("<video controls>", "<video>")
    result = _validate_mm([_mm_block(html)])
    assert "MULTIMEDIA_CONTROLS_MISSING" in _codes(result)
    assert "MULTIMEDIA_CAPTIONS_MISSING" not in _codes(result)


def test_mm_empty_stub_flags_every_piece() -> None:
    html = (
        '<figure class="multimedia" '
        'data-cf-block-id="page_01#multimedia_x_0"><video></video></figure>'
    )
    result = _validate_mm([_mm_block(html)])
    for code in (
        "MULTIMEDIA_CAPTIONS_MISSING", "MULTIMEDIA_AUDIO_DESC_MISSING",
        "MULTIMEDIA_TRANSCRIPT_MISSING", "MULTIMEDIA_CONTROLS_MISSING",
    ):
        assert code in _codes(result)
    assert result.passed is True  # warning-day-1


def test_mm_pieces_noop_when_flag_off() -> None:
    html = (
        '<figure class="multimedia" '
        'data-cf-block-id="page_01#multimedia_x_0"><video></video></figure>'
    )
    result = _validate_mm([_mm_block(html)], enabled=False)
    for code in (
        "MULTIMEDIA_CAPTIONS_MISSING", "MULTIMEDIA_AUDIO_DESC_MISSING",
        "MULTIMEDIA_TRANSCRIPT_MISSING", "MULTIMEDIA_CONTROLS_MISSING",
    ):
        assert code not in _codes(result)
