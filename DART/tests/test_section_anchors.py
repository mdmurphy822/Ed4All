"""Wave 30 Gap 2 — #sec-X-Y anchor emission + TOC validation.

Pre-Wave-30, section / subsection heading templates emitted
``id="sec-<block_id>"`` (stable hash). TOC links generated from
``"X.Y Title"`` outline entries targeted ``#sec-X-Y`` — those hrefs
never matched the body ids, so 90+ TOC entries dead-ended on a real
textbook. Wave 30 Gap 2:

1. ``_tpl_section_heading`` / ``_tpl_subsection_heading`` emit
   ``id="sec-A-B[-C...]"`` whenever the classifier persisted a
   dotted-numeric hierarchy (``dotted_number`` / ``section_number``
   / ``number`` attribute, or scraped from the heading text).
2. ``_validate_toc_section_anchors`` runs as a post-assembly pass,
   mirroring Wave 25's ``_validate_toc_page_anchors``. Orphan
   ``#sec-X-Y`` hrefs get rewritten to the nearest ``chap-X`` anchor
   or demoted to ``<span aria-disabled="true">``.
3. Non-numbered subsection headings keep the stable-hash id so
   every anchor remains valid.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock
from DART.converter.block_templates import render_block
from DART.converter.cross_refs import resolve_cross_references


def _mk_section(
    text: str,
    *,
    block_id: str = "b1",
    number: str | None = None,
    role: BlockRole = BlockRole.SECTION_HEADING,
    heading_text: str | None = None,
    level: int | None = None,
) -> ClassifiedBlock:
    """Build a classified SECTION_HEADING / SUBSECTION_HEADING block."""
    attrs: dict = {}
    if heading_text is not None:
        attrs["heading_text"] = heading_text
    if number is not None:
        attrs["dotted_number"] = number
    if level is not None:
        attrs["level"] = level
    return ClassifiedBlock(
        raw=RawBlock(text=text, block_id=block_id),
        role=role,
        confidence=0.9,
        attributes=attrs,
    )


@pytest.mark.unit
def test_section_heading_with_numeric_pair_emits_sec_anchor():
    """1.1 Intro → id='sec-1-1' on the <section> wrapper."""
    block = _mk_section(
        "1.1 Intro",
        block_id="s_1_1",
        number="1.1",
        heading_text="Intro",
    )
    html = render_block(block)
    assert 'id="sec-1-1"' in html
    # <h2> still gets its own id to anchor the aria-labelledby target.
    assert "<h2 " in html


@pytest.mark.unit
def test_subsection_heading_depth_4_emits_sec_anchor():
    """4.8.1.1 Epistemological basis → id='sec-4-8-1-1' on the <hN>."""
    block = _mk_section(
        "4.8.1.1 Epistemological basis",
        block_id="sub_4_8_1_1",
        number="4.8.1.1",
        heading_text="Epistemological basis",
        role=BlockRole.SUBSECTION_HEADING,
        level=6,
    )
    html = render_block(block)
    assert 'id="sec-4-8-1-1"' in html
    # Level-6 promotion exercised.
    assert "<h6" in html


@pytest.mark.unit
def test_non_numbered_subsection_keeps_stable_hash_id():
    """Non-numbered subsection headings must NOT get a numeric sec-anchor
    (would silently collide) and must still emit a stable id so they are
    reachable + unique."""
    block = _mk_section(
        "Learning Outcomes",
        block_id="abc12345",
        heading_text="Learning Outcomes",
        role=BlockRole.SUBSECTION_HEADING,
    )
    html = render_block(block)
    # Non-numeric → stable-hash fallback (sub-<block_id>).
    assert 'id="sub-abc12345"' in html
    assert 'id="sec-' not in html  # no numeric anchor minted


@pytest.mark.unit
def test_numeric_scraped_from_heading_text_when_attribute_missing():
    """Even when the classifier forgot to persist dotted_number, the
    template must scrape the leading number out of heading_text so the
    anchor still resolves."""
    block = _mk_section(
        "2.3 Methods and Approach",
        block_id="scraped_2_3",
        heading_text="2.3 Methods and Approach",
        role=BlockRole.SUBSECTION_HEADING,
    )
    html = render_block(block)
    assert 'id="sec-2-3"' in html


@pytest.mark.unit
def test_toc_link_with_matching_anchor_preserved():
    """A TOC ``<a href='#sec-1-1'>`` with a matching ``id='sec-1-1'`` in
    the body must survive the validator pass unchanged — it is a valid
    link."""
    # Valid body: id="sec-1-1" really exists.
    doc = (
        "<!DOCTYPE html><html><head><title>T</title></head>"
        "<body>"
        "<nav><ol><li><a href='#sec-1-1'>1.1 Intro</a></li></ol></nav>"
        "<section id=\"sec-1-1\"><h2>Intro</h2></section>"
        "</body></html>"
    )
    out = resolve_cross_references(doc, [])
    assert "#sec-1-1" in out
    assert "aria-disabled" not in out


@pytest.mark.unit
def test_toc_link_orphan_demoted_to_aria_disabled():
    """A TOC ``<a href='#sec-99-99'>`` with NO matching body anchor and
    NO same-numbered chapter must be demoted to
    ``<span aria-disabled='true'>`` so screen readers don't announce
    the dead link."""
    doc = (
        "<!DOCTYPE html><html><head><title>T</title></head>"
        "<body>"
        "<nav><ol><li><a href=\"#sec-99-99\">99.99 Missing</a></li></ol></nav>"
        "</body></html>"
    )
    out = resolve_cross_references(doc, [])
    assert '<span aria-disabled="true">99.99 Missing</span>' in out
    assert 'href="#sec-99-99"' not in out


@pytest.mark.unit
def test_toc_link_rewritten_to_matching_chapter_anchor():
    """Orphan ``#sec-3-2`` gets rewritten to ``#chap-3`` when the body
    carries a ``chap-3`` anchor — preserves screen-reader jump
    semantics even when the section anchor didn't make it into the
    body."""
    doc = (
        "<!DOCTYPE html><html><head><title>T</title></head>"
        "<body>"
        "<nav><ol><li><a href=\"#sec-3-2\">3.2 Topic</a></li></ol></nav>"
        "<article id=\"chap-3\"><h2>Chapter 3</h2></article>"
        "</body></html>"
    )
    out = resolve_cross_references(doc, [])
    # The orphan href=#sec-3-2 is gone after rewrite.
    assert 'href="#sec-3-2"' not in out
    # The rewrite preserves the original label text and points at
    # the same-numbered chapter anchor.
    assert '<a href="#chap-3"' in out
    assert "3.2 Topic" in out


@pytest.mark.unit
def test_pagebreak_anchors_still_present_and_valid():
    """Wave 25 regression guard: the page-anchor validator pass (fix 6)
    must continue to fire alongside the Wave 30 section-anchor pass —
    the new pass must NOT swallow valid ``#page-N`` → doc-pagebreak
    links."""
    doc = (
        "<!DOCTYPE html><html><head><title>T</title></head>"
        "<body>"
        "<nav><ol>"
        "<li><a href=\"#page-42\">TOC entry to page 42</a></li>"
        "<li><a href=\"#sec-1-1\">1.1 Intro</a></li>"
        "</ol></nav>"
        "<span id=\"page-42\" role=\"doc-pagebreak\"></span>"
        "<section id=\"sec-1-1\"><h2>Intro</h2></section>"
        "</body></html>"
    )
    out = resolve_cross_references(doc, [])
    assert 'href="#page-42"' in out
    assert 'href="#sec-1-1"' in out
    assert "aria-disabled" not in out


@pytest.mark.unit
def test_dart_markers_validator_still_passes_on_emitted_html():
    """Regression guard for the Wave 6 ``dart_markers`` gate: the
    section template continues to carry ``data-dart-source`` +
    ``data-dart-block-id`` on the emitted ``<section>`` wrapper so the
    validator at lib/validators/dart_markers.py still scores 1.0."""
    block = _mk_section(
        "1.1 Scope",
        block_id="scope_1_1",
        number="1.1",
        heading_text="Scope",
    )
    html = render_block(block)
    # Provenance contract intact (Wave 19 P2 rule — attributes stop at
    # the wrapper, which is exactly this <section>).
    assert "data-dart-block-id" in html
    assert "data-dart-source" in html
    # And the new Wave 30 numeric id didn't displace the provenance.
    assert 'id="sec-1-1"' in html
