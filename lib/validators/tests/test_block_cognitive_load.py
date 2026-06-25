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


def _block(block_type: str, content: str, block_id: str = "b1") -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="week_01_content",
        sequence=1,
        content=content,
    )


def test_overflow_flags_everything_block():
    # A concept now gets the ~1000-char exposition budget, so to demonstrate
    # char-overflow it must clearly exceed it (~1200 chars of a single <p>).
    block = _concept("<p>" + ("word " * 240) + "</p>")  # ~1200 chars
    res = BlockCognitiveLoadValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert res.passed is True  # warning-day-1
    codes = {i.code for i in res.issues}
    assert "BLOCK_BODY_OVERFLOW" in codes
    meta = res.metadata["block_cognitive_load"]
    assert meta["blocks_overflow"] == 1
    assert meta["per_block"]["b1"]["overflow"] is True


def test_exposition_concept_within_budget_passes():
    # FIX 2: a single-idea ~800-char concept body (<=4 chunks) is WITHIN the
    # exposition per-type budget (~1000) and no longer overflows — proving the
    # per-type ceiling. Under the old type-blind 200 ceiling this overflowed.
    block = _concept("<p>" + ("word " * 160) + "</p>")  # ~800 chars, one <p>
    res = BlockCognitiveLoadValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert res.passed is True
    assert not any(i.code == "BLOCK_BODY_OVERFLOW" for i in res.issues)
    meta = res.metadata["block_cognitive_load"]
    assert meta["blocks_overflow"] == 0
    assert meta["per_block"]["b1"]["ceiling"] == 1000


def test_atomic_type_keeps_200_ceiling():
    # FIX 2: an ATOMIC type (key_idea) with a ~400-char body STILL overflows —
    # the 200 single-idea ceiling is preserved for non-exposition types.
    block = _block("key_idea", "<p>" + ("word " * 80) + "</p>")  # ~400 chars
    res = BlockCognitiveLoadValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert any(i.code == "BLOCK_BODY_OVERFLOW" for i in res.issues)
    assert res.metadata["block_cognitive_load"]["per_block"]["b1"]["ceiling"] == 200


def test_developed_types_within_1000_budget_pass():
    # FIX 3: the single-developed-idea types raised to ~1000 (hook / problem /
    # activity / explanation / misconception / multimedia / guided_practice) no
    # longer trip the char axis on a normally-sized body. A 547-char hook is a
    # normal activation prompt, not an "everything-block".
    for bt in (
        "hook",
        "problem",
        "explanation",
        "misconception",
        "guided_practice",
    ):
        block = _block(bt, "<p>" + ("word " * 100) + "</p>", block_id="b1")  # ~500 chars
        res = BlockCognitiveLoadValidator().validate(
            {"blocks": [block], "rubric_enabled": True}
        )
        assert not any(
            i.code == "BLOCK_BODY_OVERFLOW" for i in res.issues
        ), f"{bt} should not overflow at ~500 chars (budget 1000)"
        assert res.metadata["block_cognitive_load"]["per_block"]["b1"]["ceiling"] == 1000


def test_aggregating_types_within_1200_budget_pass():
    # FIX 3: structural roll-ups (prereq_set / recap / checklist / acronym /
    # reflection_prompt) get the ~1200 budget — a several-entry aggregate under
    # 1200 chars is normal, not over-stuffed. Each body is a single long <p> so
    # the orthogonal chunk axis stays quiet and we isolate the char axis.
    for bt in (
        "prereq_set",
        "recap",
        "checklist",
        "acronym",
        "reflection_prompt",
    ):
        block = _block(bt, "<p>" + ("word " * 200) + "</p>", block_id="b1")  # ~1000 chars
        res = BlockCognitiveLoadValidator().validate(
            {"blocks": [block], "rubric_enabled": True}
        )
        assert not any(
            i.code == "BLOCK_BODY_OVERFLOW" for i in res.issues
        ), f"{bt} should not overflow at ~1000 chars (budget 1200)"
        assert res.metadata["block_cognitive_load"]["per_block"]["b1"]["ceiling"] == 1200


def test_genuine_over_stuffer_still_fires_per_type():
    # FIX 3 must NOT blanket-raise to never fire: an acronym table at ~1400
    # chars (over the 1200 aggregating budget) is a genuine over-stuffer and
    # STILL trips the char axis.
    block = _block("acronym", "<p>" + ("word " * 280) + "</p>")  # ~1400 chars
    res = BlockCognitiveLoadValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert any(i.code == "BLOCK_BODY_OVERFLOW" for i in res.issues)
    assert res.metadata["block_cognitive_load"]["per_block"]["b1"]["ceiling"] == 1200


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
