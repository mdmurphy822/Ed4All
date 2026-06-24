"""Issue I6 — instruction-palette-v2 block types (table / acronym / key_idea).

CPU-pinned, no-GPU, no-model unit tests for the three first-class
instruction block types added by Issue I6:

- Registration: ``BLOCK_TYPES`` carries the three new members and the count
  is 24.
- Renderer (deterministic, LLM-free fallback) emits the WCAG-correct markup
  per type: a real ``<table>`` with ``<caption>`` + scoped ``<th>``; an
  ``acronym`` ``<dl>`` with matched ``<dt>`` / ``<dd>`` pairs; a ``key_idea``
  ``<aside>``.
- ``rewrite_html_shape`` validator passes the canonical markup and FAILS a
  shape-violating emit (missing caption, mismatched dl, non-aside key_idea).
- Schema round-trip: a Block of each new type validates against
  ``courseforge_jsonld_v1.schema.json::$defs.Block``.

All deterministic — no LLM dispatch, no embedding / NLI model load.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pytest  # noqa: E402

from blocks import Block, BLOCK_TYPES  # noqa: E402
from MCP.tools.pipeline_tools import _render_block_fallback_html  # noqa: E402
from lib.validators.rewrite_html_shape import (  # noqa: E402
    RewriteHtmlShapeValidator,
    _check_palette_v2_shape,
    _ShapeParser,
)

NEW_TYPES = ["table", "acronym", "key_idea"]


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("block_type", NEW_TYPES)
def test_new_type_registered(block_type):
    assert block_type in BLOCK_TYPES


def test_block_types_count_is_twenty_eight():
    # IB5 added 4 framework-aligned types to the 24-member palette; B15 added
    # `resources` -> 29; FR-INT-02 added `guided_practice` (B08) -> 30.
    assert len(BLOCK_TYPES) == 30


# --------------------------------------------------------------------------- #
# Renderer (deterministic fallback) WCAG markup
# --------------------------------------------------------------------------- #
def _render(block_type, content):
    blk = Block(
        block_id=f"week_01_content_01#{block_type}_x_0",
        block_type=block_type,
        page_id="week_01_content_01",
        sequence=0,
        content=content,
        source_ids=("dart:sample#b1",),
    )
    return _render_block_fallback_html(blk, source_ids=["dart:sample#b1"])


def test_table_renders_caption_and_scoped_th():
    html = _render(
        "table",
        {"key_claims": ["Fraction types", "Proper fraction", "Improper fraction"]},
    )
    assert "<table" in html
    assert "<caption>" in html
    assert 'scope="col"' in html or 'scope="row"' in html


def test_acronym_renders_dl_with_matched_pairs():
    html = _render(
        "acronym",
        {"key_claims": ["PEMDAS", "P: Parentheses", "E: Exponents", "M: Multiply"]},
    )
    assert "<dl" in html
    # Equal, non-zero dt/dd counts (one term per letter).
    assert html.count("<dt>") == html.count("<dd>")
    assert html.count("<dt>") >= 1


def test_key_idea_renders_aside():
    html = _render("key_idea", {"key_claims": ["The single most important rule."]})
    assert "<aside" in html
    assert "Key Idea" in html


# --------------------------------------------------------------------------- #
# rewrite_html_shape validator — passes canonical markup, fails violations
# --------------------------------------------------------------------------- #
def _shape_block(block_type, html):
    return Block(
        block_id=f"week_01_content_01#{block_type}_x_0",
        block_type=block_type,
        page_id="week_01_content_01",
        sequence=0,
        content=html,
    )


_VALID = {
    "table": (
        '<table data-cf-block-id="b#table_0" data-cf-content-type="table">'
        "<caption>Fraction types</caption>"
        '<thead><tr><th scope="col">Type</th><th scope="col">Definition</th></tr></thead>'
        "<tbody><tr><th scope=\"row\">Proper</th><td>num &lt; denom</td></tr></tbody>"
        "</table>"
    ),
    "acronym": (
        '<div class="acronym-card" data-cf-block-id="b#acronym_0" '
        'data-cf-content-type="acronym"><p><strong>PEMDAS</strong></p>'
        '<dl class="acronym-list"><dt>P</dt><dd>Parentheses</dd>'
        "<dt>E</dt><dd>Exponents</dd></dl></div>"
    ),
    "key_idea": (
        '<aside class="key-idea" data-cf-block-id="b#key_idea_0" '
        'data-cf-content-type="key_idea"><p><strong>Key Idea:</strong> '
        "Dividing by a common factor keeps the value.</p></aside>"
    ),
}


@pytest.mark.parametrize("block_type", NEW_TYPES)
def test_validator_passes_canonical_markup(block_type):
    validator = RewriteHtmlShapeValidator()
    result = validator.validate({"blocks": [_shape_block(block_type, _VALID[block_type])]})
    assert result.passed, [i.code for i in result.issues]


def test_validator_fails_table_without_caption():
    bad = (
        '<table data-cf-block-id="b#table_0" data-cf-content-type="table">'
        '<thead><tr><th scope="col">Type</th></tr></thead>'
        "<tbody><tr><td>Proper</td></tr></tbody></table>"
    )
    result = RewriteHtmlShapeValidator().validate({"blocks": [_shape_block("table", bad)]})
    assert not result.passed
    assert any(i.code == "REWRITE_BLOCK_SHAPE_INVALID" for i in result.issues)


def test_validator_fails_acronym_mismatched_pairs():
    bad = (
        '<div class="acronym-card" data-cf-block-id="b#acronym_0" '
        'data-cf-content-type="acronym"><dl><dt>P</dt><dt>E</dt><dd>Parentheses</dd>'
        "</dl></div>"
    )
    result = RewriteHtmlShapeValidator().validate({"blocks": [_shape_block("acronym", bad)]})
    assert not result.passed
    assert any(i.code == "REWRITE_BLOCK_SHAPE_INVALID" for i in result.issues)


def test_validator_fails_key_idea_not_aside():
    bad = (
        '<div class="key-idea" data-cf-block-id="b#key_idea_0" '
        'data-cf-content-type="key_idea"><p>Key Idea: a rule.</p></div>'
    )
    result = RewriteHtmlShapeValidator().validate({"blocks": [_shape_block("key_idea", bad)]})
    assert not result.passed
    assert any(i.code == "REWRITE_BLOCK_SHAPE_INVALID" for i in result.issues)


def test_check_palette_v2_shape_noop_for_other_types():
    # The shape check is a strict no-op for non-palette-v2 block types.
    parser = _ShapeParser()
    parser.feed("<p>hi</p>")
    assert _check_palette_v2_shape("concept", parser) is None
    assert _check_palette_v2_shape("objective", parser) is None


# --------------------------------------------------------------------------- #
# Schema round-trip — Block of each new type validates against $defs.Block
# --------------------------------------------------------------------------- #
def test_schema_roundtrip_block_validates():
    import jsonschema  # local — not a hot-path dep

    schema_path = (
        PROJECT_ROOT / "schemas" / "knowledge" / "courseforge_jsonld_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    block_schema = schema["$defs"]["Block"]
    # Resolve $defs-internal refs against the root document.
    resolver = jsonschema.RefResolver.from_schema(schema)
    for block_type in NEW_TYPES:
        entry = {
            "blockId": f"week_01_content_01#{block_type}_x_0",
            "blockType": block_type,
            "sequence": 0,
        }
        jsonschema.validate(entry, block_schema, resolver=resolver)
