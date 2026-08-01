r"""Vendor-lane text-corruption regressions (2026-08-01).

Five verified defects on publisher-supplied accessible HTML, all reproduced here
as SMALL synthetic fixtures driven through the real ``ingest_vendor_html`` lane
(no course-data path, no network, no model):

1. Currency-``$`` inline-math region cascade — ``<td>25. $131.19</td> <td>27.
   <img …>`` was read as one ``$…$`` math region and the intervening MARKUP was
   entity-escaped into learner-visible ``&lt;td&gt;`` text; the non-cascading
   form ate the sigil outright (``$5 and $10`` → ``5 and \$10``).
2. ``&gt;`` deletion — the OCR gutter-glyph fold deleted every whitespace-bounded
   ``&gt;`` from vendor body text, so an inequality answer key lost its operators.
3. Orphaned ``data-alt`` figure descriptions — a wrapper with no recoverable
   image kept its description only in an attribute (invisible, unchunked).
4. Solution body dropped at a section seam — a TRY-IT answer that is a bare 2-4
   digit number was deleted as a "leaked printed folio".
5. Mis-nested block swallowing document structure — an unclosed publisher ``<dl>``
   absorbed a whole apparatus section, shipping a raw second ``<h1>``.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

from lib.semantik.vendor_ingest import ingest_vendor_html


def _convert(body: str) -> str:
    """Run a vendor HTML body through the real lane; return the emitted HTML."""
    doc = f"<html><body><h1>Chapter 1 Whole Numbers</h1>{body}</body></html>"
    out = ingest_vendor_html(doc, pdf_stem="fixture-ch01", doc_title="Fixture")
    assert out["success"] is True
    return out["html"]


# --- DEFECT 1: currency ``$`` never opens a math region -----------------------
_TABLE = (
    "<table><tbody>"
    "<tr><td>25. $131.19</td> "
    '<td>27. <img src="https://example.invalid/a.svg" alt="a=-5"/></td></tr>'
    "<tr>"
    '<td>45. <img src="https://example.invalid/b.svg" '
    'alt="s-\\frac{1}{12}=\\frac{1}{4}"/></td> '
    "<td>47. $32</td></tr>"
    "</tbody></table>"
)


def test_currency_cells_do_not_escape_intervening_markup():
    html = _convert(_TABLE)
    # The exemplar shape: the amount survives and the row markup is REAL markup.
    assert "25. $131.19</td>" in html
    assert "47. $32" in html
    assert "&lt;td&gt;" not in html
    assert "&lt;span" not in html
    # And the two figures still became accessible notation spans.
    assert html.count('class="semantik-figure-notation"') == 2


def test_currency_amounts_in_prose_survive_intact():
    html = _convert("<p>Jeannette has $5 and $10 bills in her wallet.</p>")
    assert "$5 and $10 bills" in html
    assert "\\$" not in html  # no literal escape litter in the artifact


def test_genuine_inline_math_still_pairs():
    # The guard is a currency test, not a "digit follows" test: a span whose
    # remainder carries math evidence still opens exactly as before.
    html = _convert("<p>Compute $5x + 3$ carefully.</p>")
    assert "$5x + 3$" in html


# --- DEFECT 2: ``&gt;`` is an operator, not OCR gutter debris -----------------
def test_greater_than_operators_survive_in_answer_key():
    html = _convert('<p id="ans">a) &gt; b) &lt; c) &gt; d) &gt;</p>')
    m = re.search(r'<p id="ans">(.*?)</p>', html, re.S)
    assert m is not None, "answer paragraph missing"
    assert m.group(1) == "a) &gt; b) &lt; c) &gt; d) &gt;"
    # The entities decode to the real operators for a reader.
    assert BeautifulSoup(m.group(0), "html.parser").get_text() == (
        "a) > b) < c) > d) >"
    )


def test_greater_than_survives_in_instructional_prose():
    html = _convert(
        "<p>The symbols &lt; and &gt; each have a smaller side.</p>"
    )
    assert "&lt; and &gt; each" in html


def test_no_double_escaped_entities():
    html = _convert("<p>Compare -1 &lt; 9 and 9 &gt; -1.</p>")
    assert "&amp;lt;" not in html and "&amp;gt;" not in html


# --- DEFECT 3: orphaned ``data-alt`` descriptions are surfaced ----------------
_ALT_A = (
    "The graph shows the x y-coordinate plane. The x- and y-axes each run "
    "from negative 6 to 6."
)
_ALT_B = (
    "An arrow starts at the origin and extends right to the number 2 on the "
    "x-axis."
)


def test_orphaned_media_alt_becomes_visible_figure_notation():
    html = _convert(
        f'<p><span data-type="media" data-alt="{_ALT_B}">&nbsp;</span></p>'
    )
    soup = BeautifulSoup(html, "html.parser")
    span = soup.find(class_="semantik-figure-notation")
    assert span is not None
    assert span.get("role") == "img"
    assert _ALT_B in span.get_text()  # VISIBLE body carries the description
    assert _ALT_B in (span.get("aria-label") or "")  # aria-label preserved


def test_nested_media_wrappers_become_siblings_not_nested_figures():
    html = _convert(
        f'<p><span data-type="media" data-alt="{_ALT_A}">'
        f'<span data-type="media" data-alt="{_ALT_B}">&nbsp;</span>'
        "</span></p>"
    )
    soup = BeautifulSoup(html, "html.parser")
    spans = soup.find_all(class_="semantik-figure-notation")
    assert len(spans) == 2
    assert not any(sp.find(class_="semantik-figure-notation") for sp in spans)
    text = soup.get_text(" ")
    assert _ALT_A in text and _ALT_B in text


def test_media_alt_with_recoverable_image_is_not_duplicated():
    # The image's own alt already reaches the learner through the placeholder;
    # promoting the wrapper's description too would double every figure.
    alt = "A graph plotting the points (-5, 4) and (-2, 3)."
    html = _convert(
        f'<p>See the plot: <span data-type="media" data-alt="{_ALT_A}">'
        f'<img src="https://example.invalid/g.png" alt="{alt}"/></span></p>'
    )
    assert html.count('class="semantik-figure-notation"') == 1
    # The image's alt is the visible description; the wrapper's alternate
    # ``data-alt`` stays an attribute rather than doubling the figure text.
    visible = BeautifulSoup(html, "html.parser").get_text(" ")
    assert alt in visible
    assert _ALT_A not in visible


# --- DEFECT 4: a numeric solution body is not a printed folio -----------------
_TRY_IT = (
    '<div class="textbox textbox--exercises"><div class="textbox__content">'
    '<div data-type="problem"><p id="q">Find the LCM of 24 and 32</p></div>'
    '<div data-type="solution"><div class="bc-details details">'
    '<div class="bc-summary summary">Show answer</div>'
    '<p id="a96">96</p></div></div></div></div>'
)


def test_bare_numeric_answer_survives_before_a_section_seam():
    html = _convert(_TRY_IT + "<h1>Key Concepts</h1><p>Recap.</p>")
    assert '<p id="a96">96</p>' in html
    assert "Show answer" in html


def test_bare_numeric_answer_survives_without_a_following_seam():
    html = _convert(_TRY_IT)
    assert '<p id="a96">96</p>' in html


# --- DEFECT 5: an unclosed block never swallows document structure ------------
def test_unclosed_dl_does_not_swallow_the_following_section():
    # The publisher's glossary <dl> is missing its </dd></dl>, so every parser
    # nests the rest of the document inside it.
    body = (
        '<div><h1>Glossary</h1><dl id="glossary"><dd><div class="textbox">'
        "<dl><dt>origin</dt><dd>The point labeled 0.</dd></dl>"
        "</div>"
        "<h1>Practice Makes Perfect</h1>"
        "<h2>Use Place Value</h2>"
        '<p id="pmp">In the following exercises, find the place value.</p>'
        "</div>"
    )
    html = _convert(body)
    # Exactly ONE <h1> in the document (the adapter's document title).
    assert len(re.findall(r"<h1\b", html)) == 1
    # The swallowed content is still present, and the glossary term is not lost.
    assert '<p id="pmp">' in html
    assert "The point labeled 0." in html
    assert "origin" in html
    # "Use Place Value" reached the heading path (a real <h*>, not raw markup).
    assert re.search(r"<h[2-6][^>]*>\s*Use Place Value\s*</h[2-6]>", html)
