"""FR-A11Y-03 — CalloutStructureValidator."""
from __future__ import annotations

from Courseforge.scripts.blocks import Block
from lib.validators.callout_structure import CalloutStructureValidator


def _callout(content, *, kind=None, block_id="c1") -> Block:
    return Block(
        block_id=block_id,
        block_type="callout",
        page_id="week_01_content",
        sequence=1,
        content=content,
        callout_kind=kind,
    )


def test_disabled_is_noop_pass():
    block = _callout({"items": ["x"]})
    res = CalloutStructureValidator().validate(
        {"blocks": [block], "callout_typed_enabled": False}
    )
    assert res.passed is True
    assert {i.code for i in res.issues} == {"CALLOUT_STRUCTURE_DISABLED"}
    assert res.action is None


def test_no_kind_flagged():
    block = _callout({"items": ["A short note."]}, kind=None)
    res = CalloutStructureValidator().validate(
        {"blocks": [block], "callout_typed_enabled": True}
    )
    assert res.passed is True  # warning-day-1
    assert any(i.code == "CALLOUT_NO_KIND" for i in res.issues)
    assert res.action == "regenerate"


def test_typed_kind_short_body_passes_clean():
    block = _callout({"items": ["A short note."]}, kind="note")
    res = CalloutStructureValidator().validate(
        {"blocks": [block], "callout_typed_enabled": True}
    )
    assert not res.issues
    assert res.action is None
    assert res.metadata["callout_structure"]["callout_blocks"] == 1
    assert res.metadata["callout_structure"]["flagged"] == 0


def test_color_only_flagged():
    # Rendered HTML carrying a kind-class but no visible label/icon row.
    html = '<div class="callout callout-kind-warning">body text</div>'
    block = _callout(html, kind="warning")
    res = CalloutStructureValidator().validate(
        {"blocks": [block], "callout_typed_enabled": True}
    )
    assert any(i.code == "CALLOUT_COLOR_ONLY" for i in res.issues)


def test_color_with_label_not_flagged_color_only():
    html = (
        '<div class="callout callout-kind-warning">'
        '<p class="callout-label"><span class="callout-icon">!</span>'
        '<span class="callout-kind-label">Warning</span></p>body</div>'
    )
    block = _callout(html, kind="warning")
    res = CalloutStructureValidator().validate(
        {"blocks": [block], "callout_typed_enabled": True}
    )
    assert not any(i.code == "CALLOUT_COLOR_ONLY" for i in res.issues)


def test_body_overflow_flagged():
    long_body = "word " * 100  # ~500 chars >> 200 ceiling
    block = _callout({"items": [long_body]}, kind="note")
    res = CalloutStructureValidator().validate(
        {"blocks": [block], "callout_typed_enabled": True}
    )
    assert any(i.code == "CALLOUT_BODY_OVERFLOW" for i in res.issues)


def test_body_overflow_respects_ceiling_override():
    block = _callout({"items": ["a moderately sized note body here."]}, kind="note")
    res = CalloutStructureValidator().validate(
        {"blocks": [block], "callout_typed_enabled": True, "body_char_ceiling": 5}
    )
    assert any(i.code == "CALLOUT_BODY_OVERFLOW" for i in res.issues)


def test_motion_without_guard_flagged():
    html = (
        '<div class="callout callout-kind-tip" style="animation: pulse 1s;">'
        '<p class="callout-label">Tip</p>body</div>'
    )
    block = _callout(html, kind="tip")
    res = CalloutStructureValidator().validate(
        {"blocks": [block], "callout_typed_enabled": True}
    )
    assert any(i.code == "CALLOUT_MOTION" for i in res.issues)


def test_motion_with_reduced_motion_guard_not_flagged():
    html = (
        '<div class="callout callout-kind-tip">'
        '<style>@media (prefers-reduced-motion: reduce){.x{animation:none}}'
        '.x{animation: pulse 1s;}</style>'
        '<p class="callout-label">Tip</p>body</div>'
    )
    block = _callout(html, kind="tip")
    res = CalloutStructureValidator().validate(
        {"blocks": [block], "callout_typed_enabled": True}
    )
    assert not any(i.code == "CALLOUT_MOTION" for i in res.issues)


def test_non_callout_blocks_skipped():
    block = Block(
        block_id="x1",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>An idea.</p>",
    )
    res = CalloutStructureValidator().validate(
        {"blocks": [block], "callout_typed_enabled": True}
    )
    assert res.metadata["callout_structure"]["callout_blocks"] == 0
    assert not res.issues
