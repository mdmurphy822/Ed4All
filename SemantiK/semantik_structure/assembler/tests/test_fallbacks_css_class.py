"""Assembler emit — pedagogical ``css_class`` hint on demoted paragraphs.

The deterministic structure-correction pass demotes pedagogical headings to
paragraphs carrying ``payload['css_class']`` (e.g. ``"pedagogy-example"``).
``fallback_paragraph`` must emit ``<p class="…">`` for those, and stay
byte-identical (plain ``<p>``) for every paragraph WITHOUT the hint. CPU-only,
no model load.
"""

from __future__ import annotations

import pytest

from semantik_structure.assembler.fallbacks import emit_fallback, fallback_paragraph
from semantik_structure.structure_graph import Region
from semantik_structure.types import FeatureBlock, RawBlock


def _fb(text: str, page: int = 1) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=page,
        bbox=(0.0, 0.0, 10.0, 10.0),
        page_width=100.0,
        page_height=100.0,
    )
    return FeatureBlock(
        raw=raw,
        size_bucket="md",
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=1.0,
    )


def _para(text: str, idx: int, css_class: str | None = None) -> Region:
    payload: dict = {"text": text}
    if css_class is not None:
        payload["css_class"] = css_class
    return Region(
        kind="paragraph",
        feature_block_indices=(idx,),
        payload=payload,
        source_region_id=idx,
    )


def test_paragraph_without_class_is_plain_byte_identical():
    region = _para("EXAMPLE 1", 0)
    fbs = [_fb("EXAMPLE 1")]
    html = fallback_paragraph(region, fbs)
    assert html == "<p>EXAMPLE 1</p>"
    # The dispatch entry-point matches.
    assert emit_fallback(region, fbs) == "<p>EXAMPLE 1</p>"


@pytest.mark.parametrize(
    "css_class",
    [
        "pedagogy-example",
        "pedagogy-solution",
        "pedagogy-try-it",
        "pedagogy-how-to",
        "pedagogy-step",
        "pedagogy-objectives",
        "pedagogy-be-prepared",
        "pedagogy-practice",
    ],
)
def test_paragraph_with_class_emits_class_attr(css_class):
    region = _para("EXAMPLE 1", 0, css_class=css_class)
    fbs = [_fb("EXAMPLE 1")]
    html = fallback_paragraph(region, fbs)
    assert html == f'<p class="{css_class}">EXAMPLE 1</p>'
    assert emit_fallback(region, fbs) == f'<p class="{css_class}">EXAMPLE 1</p>'


def test_empty_or_whitespace_class_is_plain():
    # Defensive: an empty / whitespace css_class falls back to plain <p>.
    for bad in ("", "   "):
        region = _para("text", 0, css_class=bad)
        fbs = [_fb("text")]
        assert fallback_paragraph(region, fbs) == "<p>text</p>"


def test_text_is_escaped_with_class():
    region = _para("a < b & c", 0, css_class="pedagogy-example")
    fbs = [_fb("a < b & c")]
    html = fallback_paragraph(region, fbs)
    assert html == '<p class="pedagogy-example">a &lt; b &amp; c</p>'
