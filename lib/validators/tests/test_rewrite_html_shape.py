"""Tests for RewriteHtmlShapeValidator (Plan §3.5 followup §3.2).

Eight cases per the followup plan:
1. Bare HTML body fragment with required attrs → pass.
2. JSON-wrapped (the recorded ``{"div": {...}}`` regression) → fail
   with ``REWRITE_JSON_WRAPPED_HTML``.
3. Markdown-fenced HTML (leading triple-backtick) → fail with
   ``REWRITE_NOT_HTML_BODY_FRAGMENT``.
4. Empty content → fail with ``REWRITE_HTML_PARSE_FAIL``.
5. Unbalanced tags (``<p>foo<p>bar``) → fail with
   ``REWRITE_HTML_PARSE_FAIL``.
6. Missing ``data-cf-block-id`` on assessment_item → fail with
   ``REWRITE_MISSING_REQUIRED_ATTR``.
7. Missing ``data-cf-objective-ref`` on assessment_item → fail with
   ``REWRITE_MISSING_REQUIRED_ATTR``.
8. Plain text without HTML tags → fail with
   ``REWRITE_NOT_HTML_BODY_FRAGMENT``.

Plus: outline-tier dict content is skipped silently; valid
short-form summary_takeaway emit passes (no body-tag requirement).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Repo root + scripts dir on path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.rewrite_html_shape import RewriteHtmlShapeValidator  # noqa: E402


def _make_block(content, *, block_type: str = "concept", block_id: str = None) -> Block:
    """Construct a Block with the given content and type."""
    if block_id is None:
        block_id = f"page_01#{block_type}_demo_0"
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content=content,
    )


def _validate(blocks):
    return RewriteHtmlShapeValidator().validate({"blocks": blocks})


# ---------------------------------------------------------------------- #
# Pass cases
# ---------------------------------------------------------------------- #


def test_bare_html_concept_fragment_passes() -> None:
    """A clean concept block with required attrs passes."""
    html = (
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" '
        'data-cf-key-terms="federation,trust">'
        '<h2>Federated Identity</h2>'
        '<p>Federation lets one identity provider vouch for another.</p>'
        '</section>'
    )
    block = _make_block(html, block_type="concept")
    result = _validate([block])
    assert result.passed is True
    assert result.action is None
    assert len([i for i in result.issues if i.severity == "critical"]) == 0


def test_summary_takeaway_short_form_passes() -> None:
    """summary_takeaway is short-form; the body-tag check is relaxed."""
    html = (
        '<li data-cf-block-id="page_01#summary_takeaway_recap_0" '
        'data-cf-content-type="summary_takeaway">Federation requires trust.</li>'
    )
    block = _make_block(html, block_type="summary_takeaway")
    result = _validate([block])
    assert result.passed is True
    assert result.action is None


def test_outline_tier_dict_content_skipped_silently() -> None:
    """Outline-tier blocks (dict content) skip the gate without issues."""
    block = _make_block({"key_claims": ["Federation works."]}, block_type="concept")
    result = _validate([block])
    assert result.passed is True
    assert result.action is None
    assert result.issues == []


# ---------------------------------------------------------------------- #
# Fail cases (the eight required from §3.5)
# ---------------------------------------------------------------------- #


def test_json_wrapped_html_fails_with_json_wrapped_code() -> None:
    """The recorded ``{"div": {...}}`` regression."""
    json_wrap = (
        '{"div": {"class": "assessment-item", '
        '"content": "<p>What are the three components of an RDF triple?</p>"}}'
    )
    block = _make_block(json_wrap, block_type="assessment_item")
    result = _validate([block])
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "REWRITE_JSON_WRAPPED_HTML" in codes


def test_markdown_fenced_html_fails() -> None:
    """Markdown-fenced HTML emits with leading triple-backtick."""
    fenced = "```html\n<p>Some content</p>\n```"
    block = _make_block(fenced, block_type="concept")
    result = _validate([block])
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "REWRITE_NOT_HTML_BODY_FRAGMENT" in codes


def test_empty_content_fails() -> None:
    """Empty content fails with parse error."""
    block = _make_block("", block_type="concept")
    result = _validate([block])
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "REWRITE_HTML_PARSE_FAIL" in codes


def test_unbalanced_tags_fails() -> None:
    """``<p>foo<p>bar`` opens two <p> tags without closing — unbalanced.

    The stdlib HTMLParser does not auto-close p tags, so the open-stack
    ends with two unclosed <p> elements.
    """
    block = _make_block(
        '<section data-cf-block-id="page_01#concept_demo_0" '
        'data-cf-content-type="concept" '
        'data-cf-key-terms="federation"><p>foo<p>bar</section>',
        block_type="concept",
    )
    result = _validate([block])
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "REWRITE_HTML_PARSE_FAIL" in codes


def test_missing_block_id_attr_on_assessment_item_fails() -> None:
    """assessment_item without data-cf-block-id fails."""
    html = (
        '<section data-cf-objective-ref="TO-01" '
        'data-cf-bloom-level="apply">'
        '<p>What is RDF?</p></section>'
    )
    block = _make_block(html, block_type="assessment_item")
    result = _validate([block])
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "REWRITE_MISSING_REQUIRED_ATTR" in codes
    msgs = [i.message for i in result.issues]
    assert any("data-cf-block-id" in m for m in msgs)


def test_missing_objective_ref_on_assessment_item_fails() -> None:
    """assessment_item without data-cf-objective-ref fails."""
    html = (
        '<section data-cf-block-id="page_01#assessment_item_q1_0" '
        'data-cf-bloom-level="apply">'
        '<p>What is RDF?</p></section>'
    )
    block = _make_block(html, block_type="assessment_item")
    result = _validate([block])
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "REWRITE_MISSING_REQUIRED_ATTR" in codes
    msgs = [i.message for i in result.issues]
    assert any("data-cf-objective-ref" in m for m in msgs)


def test_plain_text_without_tags_fails() -> None:
    """Plain text (no <) → fail with NOT_HTML_BODY_FRAGMENT."""
    block = _make_block("Just some prose, no markup.", block_type="concept")
    result = _validate([block])
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "REWRITE_NOT_HTML_BODY_FRAGMENT" in codes


def test_full_doctype_document_fails() -> None:
    """A <!DOCTYPE ...> preamble is not a body fragment."""
    html = (
        "<!DOCTYPE html>"
        '<html><body><p>Content</p></body></html>'
    )
    block = _make_block(html, block_type="concept")
    result = _validate([block])
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "REWRITE_NOT_HTML_BODY_FRAGMENT" in codes


def test_missing_blocks_input_fails() -> None:
    """No blocks key in inputs → critical fail."""
    result = RewriteHtmlShapeValidator().validate({})
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "MISSING_BLOCKS_INPUT" in codes


def test_decision_capture_emits_per_block() -> None:
    """When DecisionCapture is wired, a decision event fires per block."""

    class _StubCapture:
        def __init__(self) -> None:
            self.calls = []

        def log_decision(
            self,
            decision_type: str,
            decision: str,
            rationale: str,
            **kwargs,
        ) -> None:
            self.calls.append((decision_type, decision, rationale))

    capture = _StubCapture()
    blocks = [
        _make_block(
            '<section data-cf-block-id="page_01#concept_demo_0" '
            'data-cf-content-type="concept" '
            'data-cf-key-terms="x"><p>ok</p></section>',
            block_type="concept",
        ),
        _make_block("Plain text.", block_type="concept"),
    ]
    RewriteHtmlShapeValidator().validate({
        "blocks": blocks,
        "decision_capture": capture,
    })
    assert len(capture.calls) == 2
    assert all(c[0] == "rewrite_html_shape_check" for c in capture.calls)
    # First passes, second fails.
    assert capture.calls[0][1] == "passed"
    assert capture.calls[1][1].startswith("failed:")
    # Rationale ≥ 20 chars and references block_id.
    for _, _, rationale in capture.calls:
        assert len(rationale) >= 20
        assert "block_id=" in rationale
