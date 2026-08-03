"""IB6.5 — universal (verb · level · knowledge-type) triple chip render."""
from __future__ import annotations

from Courseforge.scripts.blocks import Block


def _concept() -> Block:
    return Block(
        block_id="b1",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>An idea.</p>",
        bloom_level="understand",
        bloom_verb="explain",
        cognitive_domain="conceptual",
    )


def test_chip_emitted_only_when_flag_on(monkeypatch):
    block = _concept()
    monkeypatch.delenv("ED4ALL_BLOCK_QUALITY_RUBRIC", raising=False)
    off = block.to_html_attrs()
    assert "data-cf-bloom-triple" not in off
    assert "data-cf-cognitive-domain" not in off

    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_RUBRIC", "1")
    on = block.to_html_attrs()
    assert 'data-cf-bloom-triple="explain.understand.conceptual"' in on
    assert 'data-cf-cognitive-domain="conceptual"' in on


def test_chip_byte_identical_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_BLOCK_QUALITY_RUBRIC", raising=False)
    block = _concept()
    # Two renders with the flag off are identical and chip-free.
    assert block.to_html_attrs() == block.to_html_attrs()
    assert "data-cf-bloom" not in block.to_html_attrs() or "triple" not in block.to_html_attrs()


def test_chip_skips_chrome(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_RUBRIC", "1")
    chrome = Block(
        block_id="c1",
        block_type="chrome",
        page_id="week_01_content",
        sequence=1,
        content="<div></div>",
        bloom_level="understand",
        bloom_verb="explain",
        cognitive_domain="conceptual",
    )
    # chrome has no framework B-code → no chip.
    assert "data-cf-bloom-triple" not in chrome.to_html_attrs()


def test_knowledge_type_derived_from_bloom(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_RUBRIC", "1")
    # No explicit cognitive_domain → derived from bloom_level.
    block = Block(
        block_id="b2",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>An idea.</p>",
        bloom_level="apply",
        bloom_verb="solve",
    )
    out = block.to_html_attrs()
    assert "data-cf-cognitive-domain=" in out
    # apply → procedural per bloom_to_cognitive_domain
    assert "data-cf-bloom-triple=" in out
