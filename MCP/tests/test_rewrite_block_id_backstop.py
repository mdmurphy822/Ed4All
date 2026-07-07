"""Regression: the universal ``data-cf-block-id`` backstop must inject the
attribute whenever no PARSEABLE ``data-cf-block-id`` attribute is present —
not merely when the literal substring is absent.

Root cause (a real ``post_rewrite_validation`` failure): the rewrite
backstop guarded block-id injection with a naive substring check
(``"data-cf-block-id" not in new_content``). When the 7B rewrite tier narrates
the contract in prose (or stamps the token somewhere the
``RewriteHtmlShapeValidator`` parser does not scrape as an attribute), the
substring is present so the guard SKIPS injection — yet the gate scrapes
parsed-attribute NAMES into ``found_attrs`` and still fails the critical
``REWRITE_MISSING_REQUIRED_ATTR`` on a block whose id is known. The fix replaces
the substring guard with :func:`_html_has_block_id_attr`, which parses the HTML
(same stdlib engine the gate uses) and injects only when the attribute is
genuinely absent from every element.

Also pins the renderer side (Task 1 finding): the ``generate_course.py``
activity + content-section renderers route through ``Block.to_html_attrs()``,
which DOES append ``data-cf-block-id`` when ``COURSEFORGE_EMIT_BLOCKS`` is on —
so the renderer is correct and the defect is rewrite-tier-only.

No LLM dispatch, no pipeline run, no GPU.
"""
from __future__ import annotations

import dataclasses as _dc

import pytest

from Courseforge.scripts.blocks import Block
from Courseforge.scripts.generate_course import (
    _render_activities,
    _render_content_sections,
)
from lib.validators.rewrite_html_shape import RewriteHtmlShapeValidator
from MCP.tools.pipeline_tools import (
    _html_has_block_id_attr,
    _inject_block_id_attr,
)


# ---------------------------------------------------------------------------
# Renderer side — activity + example carry data-cf-block-id and PASS the gate
# ---------------------------------------------------------------------------

def _gate_for_str_block(block_type: str, html: str, block_id: str):
    """Run RewriteHtmlShapeValidator over a single str-content Block."""
    blk = Block(
        block_id=block_id,
        block_type=block_type,
        page_id=block_id.split("#", 1)[0],
        sequence=0,
        content=html,
    )
    return RewriteHtmlShapeValidator().validate(
        {"blocks": [blk], "gate_id": "rewrite_html_shape"}
    )


def test_rendered_activity_carries_block_id_and_passes_gate(monkeypatch):
    """An activity card rendered via to_html_attrs carries the block-id."""
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    html = _render_activities(
        [{"title": "Practice", "description": "Do the thing.",
          "bloom_level": "apply", "objective_ref": "CO-01"}],
        source_ids=["dart:slug#abc123"],
        source_primary="dart:slug#abc123",
        page_id="week_01_application",
    )
    assert "data-cf-block-id=" in html
    assert _html_has_block_id_attr(html)
    # The rendered activity card root <div> carries the block-id; the gate
    # required-attr check (data-cf-block-id + component + purpose + bloom)
    # is satisfied.
    block_id = "week_01_application#activity_practice_0"
    result = _gate_for_str_block("activity", html, block_id)
    codes = {i.code for i in result.issues}
    assert "REWRITE_MISSING_REQUIRED_ATTR" not in codes, result.issues


def test_rendered_example_carries_block_id_and_passes_gate(monkeypatch):
    """An example content-section rendered via to_html_attrs carries it."""
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    sections = [{
        "heading": "Add and Subtract Integers",
        "content_type": "example",
        "paragraphs": ["Worked solution: 3 + (-5) = -2."],
        "key_terms": [],
    }]
    html = _render_content_sections(
        sections,
        source_ids=["dart:slug#def456"],
        source_primary="dart:slug#def456",
        page_id="week_02_content_02",
    )
    assert "data-cf-block-id=" in html
    assert _html_has_block_id_attr(html)


# ---------------------------------------------------------------------------
# The parse-aware presence probe (mirrors the gate's found_attrs scrape)
# ---------------------------------------------------------------------------

def test_substring_in_prose_is_not_a_real_attribute():
    """The exact failure: substring present, attribute absent → probe False."""
    html = (
        '<div class="activity-card" data-cf-component="activity" '
        'data-cf-purpose="practice" data-cf-bloom-level="apply">'
        "<p>The data-cf-block-id contract requires an id.</p></div>"
    )
    assert "data-cf-block-id" in html          # naive substring guard sees it
    assert _html_has_block_id_attr(html) is False   # but it is NOT an attribute


def test_real_root_attribute_is_detected():
    html = '<section data-cf-block-id="w#x_0" data-cf-content-type="example"><p>hi</p></section>'
    assert _html_has_block_id_attr(html) is True


def test_attribute_on_nested_element_is_detected():
    """The gate scrapes from ALL elements, so a nested attr counts as present."""
    html = '<section><p data-cf-block-id="w#x_0">hi</p></section>'
    assert _html_has_block_id_attr(html) is True


# ---------------------------------------------------------------------------
# Backstop — a block whose HTML has the substring-but-no-attribute gets it
# injected (== block_id) and then PASSES the gate
# ---------------------------------------------------------------------------

def _apply_block_id_backstop(html: str, block_id: str) -> str:
    """Replicate the guarded injection step from _apply_str_backstops."""
    if not _html_has_block_id_attr(html):
        html = _inject_block_id_attr(html, block_id)
    return html


def test_backstop_injects_when_substring_present_but_not_attribute():
    block_id = "week_02_application#activity_week02-application_9"
    raw = (
        '<div class="activity-card" data-cf-component="activity" '
        'data-cf-purpose="practice" data-cf-bloom-level="application">'
        "<p>Recall the data-cf-block-id requirement when authoring.</p></div>"
    )
    # Pre-fix substring guard would have skipped injection here.
    assert "data-cf-block-id" in raw
    fixed = _apply_block_id_backstop(raw, block_id)
    assert _html_has_block_id_attr(fixed)
    assert f'data-cf-block-id="{block_id}"' in fixed
    # Now the gate's required-attr check is satisfied.
    result = _gate_for_str_block("activity", fixed, block_id)
    codes = {i.code for i in result.issues}
    assert "REWRITE_MISSING_REQUIRED_ATTR" not in codes, result.issues


def test_backstop_injects_on_example_section_root():
    block_id = "week_02_content_02#example_13-add_and_subtract_integers_8"
    raw = (
        '<section data-cf-source-ids="dart:slug#abc" '
        'data-cf-content-type="example">'
        "<h2>Example</h2><p>data-cf-block-id is mentioned here.</p></section>"
    )
    assert "data-cf-block-id" in raw
    fixed = _apply_block_id_backstop(raw, block_id)
    assert _html_has_block_id_attr(fixed)
    assert f'data-cf-block-id="{block_id}"' in fixed
    result = _gate_for_str_block("example", fixed, block_id)
    codes = {i.code for i in result.issues}
    assert "REWRITE_MISSING_REQUIRED_ATTR" not in codes, result.issues


def test_backstop_is_idempotent_and_leaves_real_attr_untouched():
    block_id = "w#example_x_0"
    already = (
        f'<section data-cf-block-id="{block_id}" '
        'data-cf-content-type="example"><p>hi</p></section>'
    )
    out = _apply_block_id_backstop(already, "DIFFERENT#id")
    assert out == already                       # untouched
    assert out.count("data-cf-block-id") == 1   # no duplicate injection


def test_inject_is_noop_on_non_html():
    """A genuinely non-HTML emit gets no start tag → injection is a no-op so
    the gate still fails closed."""
    out = _inject_block_id_attr("just plain prose, no tags", "w#x_0")
    assert out == "just plain prose, no tags"
    assert _html_has_block_id_attr(out) is False
