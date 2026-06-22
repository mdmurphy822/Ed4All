"""IB6.4 — BlockCognitiveLoadValidator (per-block D2 body ceiling)."""
from __future__ import annotations

from Courseforge.scripts.blocks import Block
from lib.validators.content import BlockCognitiveLoadValidator


def _concept(content: str, block_id: str = "b1") -> Block:
    return Block(
        block_id=block_id,
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content=content,
    )


def test_overflow_flags_everything_block():
    block = _concept("<p>" + ("word " * 200) + "</p>")  # ~1000 chars
    res = BlockCognitiveLoadValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert res.passed is True  # warning-day-1
    codes = {i.code for i in res.issues}
    assert "BLOCK_BODY_OVERFLOW" in codes
    meta = res.metadata["block_cognitive_load"]
    assert meta["blocks_overflow"] == 1
    assert meta["per_block"]["b1"]["overflow"] is True


def test_short_block_passes():
    block = _concept("<p>A short single idea about variables.</p>")
    res = BlockCognitiveLoadValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert res.passed is True
    assert not any(i.code == "BLOCK_BODY_OVERFLOW" for i in res.issues)
    assert res.metadata["block_cognitive_load"]["blocks_overflow"] == 0


def test_chrome_block_exempt():
    chrome = Block(
        block_id="c1",
        block_type="chrome",
        page_id="week_01_content",
        sequence=1,
        content="<div>" + ("x " * 500) + "</div>",
    )
    res = BlockCognitiveLoadValidator().validate(
        {"blocks": [chrome], "rubric_enabled": True}
    )
    assert res.passed is True
    # chrome never audited → no overflow recorded
    assert res.metadata["block_cognitive_load"]["blocks_audited"] == 0


def test_idea_chunk_ceiling():
    # 5 short <p> blocks → over the 4-chunk ceiling even though char count < 200.
    block = _concept("<p>a.</p><p>b.</p><p>c.</p><p>d.</p><p>e.</p>")
    res = BlockCognitiveLoadValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert any(i.code == "BLOCK_BODY_OVERFLOW" for i in res.issues)


def test_custom_ceiling_env(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_BODY_CHAR_CEILING", "50")
    block = _concept("<p>" + ("x" * 80) + "</p>")
    res = BlockCognitiveLoadValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert any(i.code == "BLOCK_BODY_OVERFLOW" for i in res.issues)


def test_disabled_is_noop_pass():
    block = _concept("<p>" + ("word " * 300) + "</p>")
    res = BlockCognitiveLoadValidator().validate(
        {"blocks": [block], "rubric_enabled": False}
    )
    assert res.passed is True
    assert {i.code for i in res.issues} == {"COGNITIVE_LOAD_DISABLED"}
    assert res.metadata["block_cognitive_load"]["enabled"] is False
