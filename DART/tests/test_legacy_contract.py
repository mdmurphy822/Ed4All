"""Wave 19 legacy-contract restoration tests.

Guards the re-emission of three pre-Wave-12 invariants the Waves 12-18
converter silently dropped:

* ``class="dart-document"`` stamped on the outer wrapping element.
* ``class="dart-section"`` stamped on every top-level
  ``<section>`` / ``<article>`` / ``<aside>`` wrapper.
* ``data-dart-source`` stamped on every wrapper (routing rules in
  :mod:`DART.converter.block_templates`).
* Leaf-level elements (``<p>``, ``<span>``, ``<h1>``, ``<h3>``,
  ``<cite>``, ``<a>``, ``<li>``, ``<figcaption>``) **never** carry
  ``data-dart-*`` attributes (Wave 8 P2 rule).

Also wires the ``dart_markers`` gate in at the final-HTML level to
lock in the restored contract.
"""

from __future__ import annotations

import re

import pytest

from DART.converter import convert_pdftotext_to_html
from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock
from DART.converter.block_templates import render_block
from lib.validators.dart_markers import DartMarkersValidator


_SHORT_RAW = (
    "Chapter 1: Foundations\n\n"
    "This is the introduction paragraph with substantive body text."
)

_LONG_RAW = (
    "Chapter 1: Learning to Teach\n\n"
    "Teaching is a complex practice requiring both pedagogical knowledge "
    "and subject-matter expertise.\n\n"
    "1.1 What is pedagogy?\n\n"
    "Pedagogy is the method and practice of teaching as an academic "
    "subject or theoretical concept.\n\n"
    "Chapter 2: Classroom Design\n\n"
    "Classroom design affects learner engagement and accessibility.\n\n"
    "2.1 Physical space\n\n"
    "Desks should be arranged to support multiple modes of learning.\n"
)


def _mk(role: BlockRole, text: str = "Body text.", **attrs) -> ClassifiedBlock:
    return ClassifiedBlock(
        raw=RawBlock(text=text, block_id="blk12345"),
        role=role,
        confidence=0.6,
        attributes=attrs,
    )


# ---------------------------------------------------------------------------
# Class re-emission (scope 1)
# ---------------------------------------------------------------------------


def test_dart_document_class_present_exactly_once():
    html = convert_pdftotext_to_html(_SHORT_RAW, title="Wave 19 Smoke")
    assert "dart-document" in html
    # The class appears once on the <main> wrapper only (document-level).
    main_open = re.search(r"<main\b[^>]*>", html)
    assert main_open is not None
    assert "dart-document" in main_open.group(0)
    # Body content should not re-emit the document class.
    body_only = html[main_open.end():]
    assert "dart-document" not in body_only


def test_dart_section_class_on_wrapper_templates():
    html = convert_pdftotext_to_html(_LONG_RAW, title="Wave 19 Wrappers")
    assert "dart-section" in html
    # Every <article> and <section> tag at document-body level carries
    # dart-section (modulo the <main> dart-document wrapper, which also
    # carries dart-document — dart-section isn't required on <main>).
    for match in re.finditer(r"<(article|section)\b[^>]*>", html):
        tag = match.group(0)
        # Skip the assembler's complementary <aside>, which is metadata,
        # not a dart-section wrapper.
        if "role=\"complementary\"" in tag:
            continue
        assert "class=" in tag, f"wrapper missing class attr: {tag}"
        assert "dart-section" in tag, f"wrapper missing dart-section: {tag}"


def test_dart_section_class_preserves_existing_classes():
    """Pullquote + callout templates merge their existing CSS classes
    with the new dart-section prefix (Wave 19 non-regression)."""
    pull = render_block(_mk(BlockRole.PULLQUOTE, "A great quote."))
    assert 'class="dart-section pullquote"' in pull

    callout = render_block(
        _mk(BlockRole.CALLOUT_INFO, "An info message.", title="Note")
    )
    assert 'class="dart-section callout callout-info"' in callout


# ---------------------------------------------------------------------------
# data-dart-source emission (scope 2)
# ---------------------------------------------------------------------------


def test_every_wrapper_emits_data_dart_source():
    html = convert_pdftotext_to_html(_LONG_RAW, title="Wave 19 Sources")
    for match in re.finditer(r"<(article|section|aside)\b[^>]*>", html):
        tag = match.group(0)
        # Skip non-DART aside (document metadata) + the default fallback
        # section when it doesn't carry dart-section (it should, but we
        # only care about DART-produced wrappers here).
        if 'role="complementary"' in tag:
            continue
        if "dart-section" not in tag:
            continue
        assert "data-dart-source=" in tag, f"wrapper missing source: {tag}"


def test_heuristic_classifier_source_is_dart_converter():
    html = convert_pdftotext_to_html(_SHORT_RAW, title="Wave 19 Heuristic")
    # When no LLM/extractor-hint blocks are involved every wrapper is
    # attributed to the default ``dart_converter`` source.
    assert 'data-dart-source="dart_converter"' in html


# ---------------------------------------------------------------------------
# Leaf-level strip (scope 3)
# ---------------------------------------------------------------------------


def test_paragraph_has_no_dart_attributes():
    """The leaf <p> template must emit a plain tag — no data-dart-*."""
    out = render_block(_mk(BlockRole.PARAGRAPH, "Hello world."))
    assert out.startswith("<p>")
    assert "data-dart-" not in out


def test_subsection_heading_is_leaf_h3():
    """Subsection heading is a leaf <h3 id> with no section wrapper."""
    out = render_block(
        _mk(BlockRole.SUBSECTION_HEADING, "Subsection Title")
    )
    assert "<h3" in out
    assert "<section" not in out
    assert "data-dart-" not in out


def test_citation_and_cross_reference_strip_attrs():
    cite = render_block(_mk(BlockRole.CITATION, "Doe (2023)"))
    assert cite.startswith("<cite>")
    assert "data-dart-" not in cite

    xref = render_block(
        _mk(BlockRole.CROSS_REFERENCE, "See Chapter 2", target_id="ch2")
    )
    assert xref.startswith("<a ")
    assert "data-dart-" not in xref


def test_document_level_paragraphs_have_no_dart_attributes():
    """End-to-end: scan the rendered HTML's <p> tags and confirm none
    carry data-dart-* attributes (the Wave 17 inflation mode)."""
    html = convert_pdftotext_to_html(_LONG_RAW, title="Wave 19 Leaves")
    for match in re.finditer(r"<p\b([^>]*)>", html):
        attrs = match.group(1)
        assert "data-dart-" not in attrs, (
            f"leaf <p> should not carry data-dart-*: <p{attrs}>"
        )


# ---------------------------------------------------------------------------
# End-to-end dart_markers gate (scope 1 + 2 acceptance)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,title",
    [
        (_SHORT_RAW, "Short Smoke"),
        (_LONG_RAW, "Longer Smoke"),
        ("One paragraph only.", "Single Para"),
    ],
)
def test_dart_markers_gate_passes_for_wave19_output(raw: str, title: str):
    """Wave 19 restoration: the dart_markers gate must pass (score >=
    0.90, no critical issues) on converter output for any meaningful
    input. Score 1.0 is expected when the default fallback wrapper is
    in play; rerunning against richer content also stays fully green."""
    html = convert_pdftotext_to_html(raw, title=title)
    v = DartMarkersValidator()
    result = v.validate({"html_content": html})
    critical = [i for i in result.issues if i.severity == "critical"]
    assert not critical, (
        f"unexpected critical issues: "
        f"{[(i.code, i.message) for i in critical]}"
    )
    assert result.passed, "dart_markers gate should pass post-Wave-19"
    assert result.score >= 0.90
