r"""FIX 2 — LaTeX-markup headings must not poison chunk section attribution.

Scan-OCR / VLM-fused extraction sometimes types a heading string as raw LaTeX
markup (``\textbf{EXAMPLE 1.58}``, ``$\checkmark \text{ Solution }$``) or even a
stray table row typed as a heading (``14 \text{ ft} &amp; \\ \hline``). The
chunker derives a chunk's ``section_heading`` from the emitted VISIBLE heading
text, so the adapter MUST sanitize every heading-text -> HTML seam and DEMOTE
table-row-shaped / empty-after-strip headings out of the ``<h*>`` stream.

Blocks are built synthetically (no corpus files).
"""
from __future__ import annotations

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _is_table_row_heading,
    _render_chapters,
    _render_html,
    _render_section,
    _sanitize_heading_text,
)


def _heading_block(heading_text: str, *, html: str = "", raw_block_index: int = 0):
    """A genuine heading block (region_kind='heading' -> visible <h3>)."""
    return _AdapterBlock(
        html=html,
        region_kind="heading",
        raw_block_index=raw_block_index,
        raw_text=heading_text,
        heading_text=heading_text,
    )


# ---------------------------------------------------------------------------
# _sanitize_heading_text — pure strip behavior.
# ---------------------------------------------------------------------------


def test_sanitize_checkmark_solution_part():
    raw = r"$\checkmark \text { Solution }$ (part N)"
    assert _sanitize_heading_text(raw) == "Solution (part N)"


def test_sanitize_textbf_example():
    assert _sanitize_heading_text(r"\textbf{EXAMPLE 1.58}") == "EXAMPLE 1.58"


def test_sanitize_textbf_square_root_notation():
    assert _sanitize_heading_text(r"\textbf{Square Root Notation}") == "Square Root Notation"


def test_sanitize_plain_heading_is_unchanged():
    # No markup -> byte-identical (no regression on ordinary headings).
    plain = "Square Root Notation"
    assert _sanitize_heading_text(plain) == plain


def test_sanitize_empty_and_none():
    assert _sanitize_heading_text("") == ""
    assert _sanitize_heading_text(None) == ""
    assert _sanitize_heading_text(r"\text{}") == ""


# ---------------------------------------------------------------------------
# _is_table_row_heading — demotion predicate.
# ---------------------------------------------------------------------------


def test_table_row_heading_detected():
    assert _is_table_row_heading(r"14 \text{ ft} &amp; \\ \hline") is True


def test_pipe_and_ampersand_and_hline_are_table_rows():
    assert _is_table_row_heading("a | b") is True
    assert _is_table_row_heading("x & y") is True
    assert _is_table_row_heading(r"row \hline") is True


def test_empty_after_strip_is_table_row():
    assert _is_table_row_heading(r"\text{}") is True
    assert _is_table_row_heading("") is True


def test_plain_heading_is_not_table_row():
    assert _is_table_row_heading("Square Root Notation") is False
    assert _is_table_row_heading(r"\textbf{EXAMPLE 1.58}") is False


# ---------------------------------------------------------------------------
# _render_section — <h3> emit + demotion.
# ---------------------------------------------------------------------------


def test_render_section_strips_latex_in_h3():
    block = _heading_block(r"\textbf{EXAMPLE 1.58}")
    html = _render_section(block, "sid1")
    assert "<h3 " in html
    assert "EXAMPLE 1.58</h3>" in html
    assert "textbf" not in html
    assert "\\" not in html


def test_render_section_checkmark_solution_h3():
    block = _heading_block(r"$\checkmark \text { Solution }$ (part N)")
    html = _render_section(block, "sid2")
    assert ">Solution (part N)</h3>" in html
    assert "checkmark" not in html
    assert "$" not in html


def test_render_section_plain_heading_unchanged_h3():
    block = _heading_block("Square Root Notation")
    html = _render_section(block, "sidP")
    assert ">Square Root Notation</h3>" in html


def test_render_section_table_row_heading_demoted():
    # Table-row-shaped heading -> NO <h3>; text preserved as a <p> body. B4:
    # the demoted content section anchors via a bare id, NOT an sr-only label.
    block = _heading_block(r"14 \text{ ft} &amp; \\ \hline")
    html = _render_section(block, "sidTR")
    assert "<h3" not in html
    assert 'class="sr-only"' not in html  # B4 — no generic-landmark label
    assert '<section class="dart-section" id="sidTR"' in html
    # Sanitized text is preserved as body so content isn't lost.
    assert "<p>14 ft amp;</p>" in html
    assert "\\hline" not in html
    assert "\\" not in html


def test_render_section_empty_after_strip_demoted_no_extra_body():
    # Heading that strips to empty -> demoted, and with no body html nothing
    # extra is emitted (the bare section id suffices; no sr-only label).
    block = _heading_block(r"\text{}")
    html = _render_section(block, "sidE")
    assert "<h3" not in html
    assert 'class="sr-only"' not in html
    assert 'id="sidE"' in html
    assert "<p>" not in html


def test_render_section_demoted_heading_with_existing_body_keeps_body():
    # If the demoted heading already has body html, keep it (don't overwrite).
    block = _heading_block(
        r"a & b", html="<p>real body</p>", raw_block_index=3
    )
    html = _render_section(block, "sidB")
    assert "<h3" not in html
    assert "<p>real body</p>" in html


# ---------------------------------------------------------------------------
# _render_chapters / _render_html — h2 / h1 seams.
# ---------------------------------------------------------------------------


def test_render_chapters_strips_latex_in_h2():
    chapter = _AdapterChapter(
        title=r"\textbf{Chapter 9 Roots}",
        blocks=[_heading_block("Intro")],
    )
    html = _render_chapters([chapter])
    assert "<h2>Chapter 9 Roots</h2>" in html
    assert "textbf" not in html


def test_render_html_strips_latex_in_h1_and_title():
    chapter = _AdapterChapter(title="Ch", blocks=[_heading_block("Intro")])
    doc = _render_html([chapter], title=r"\textbf{Algebra}", lang="en")
    assert "<h1>Algebra</h1>" in doc
    assert "<title>Algebra</title>" in doc
    assert "textbf" not in doc
