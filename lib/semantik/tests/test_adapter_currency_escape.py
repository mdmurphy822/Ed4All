r"""Currency-``$`` handling: converted artifact vs. MathJax-rendered page.

Textbook money word-problems carry lone currency ``$`` before digits ("costs
$5 … and $3"). Two contracts, and they are DIFFERENT (2026-08-01 vendor-lane
text-corruption fix):

* **The converted accessible HTML keeps plain ``$5``.** It is the artifact the
  chunker, retrieval, Courseforge, and every non-MathJax consumer read, and the
  adapter used to escape currency there — shipping 601 literal ``\$`` tokens
  into a 9-chapter publisher algebra corpus ("Jeannette has \$5 and \$10").
* **The assembled end-user page escapes to ``\$``** — that page, and only that
  page, enables MathJax v3 with ``inlineMath [['$','$']]``, where two amounts in
  one paragraph would FALSE-PAIR into an italic span. ``processEscapes: true``
  renders the ``\$`` as a literal dollar.

Also covers the OPENER guards in ``math_fold``: a ``$`` that opens a currency
amount, and a candidate span that swallows HTML markup, never open a math
region — the cascade that turned ``<td>25. $131.19</td> <td>27. <img …>`` into
learner-visible ``25. $131.19&lt;/td&gt; &lt;td&gt;27. …`` literal text.

All synthetic IR — no course-data path, no model, no cascade run.
"""
from __future__ import annotations

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    normalize_cascade_to_ed4all,
)
from lib.semantik.math_fold import (
    escape_currency_dollars,
    escape_math_angle_brackets,
)


# --- unit: escape_currency_dollars --------------------------------------------
def test_two_currency_amounts_both_escaped():
    # (a) — two currency amounts in one paragraph: both dollars escaped.
    out = escape_currency_dollars("<p>It costs $5 to enter and $3 to ride.</p>")
    assert r"\$5" in out and r"\$3" in out
    assert "$5" not in out.replace(r"\$5", "")  # no UN-escaped $5 remains
    assert "$3" not in out.replace(r"\$3", "")


def test_paired_inline_math_untouched():
    # (b) — genuine inline math ``$5x + 3$`` is a paired span → never escaped.
    # The currency guard does NOT fire: the remainder after the leading ``5``
    # carries math evidence (``x``'s neighbouring ``+``).
    out = escape_currency_dollars("<p>Compute $5x + 3$ carefully.</p>")
    assert out == "<p>Compute $5x + 3$ carefully.</p>"
    assert r"\$" not in out


def test_digit_leading_arithmetic_span_untouched():
    # Companion to (b) — a span with NO letters at all (``$3 + 4 = 7$``) still
    # reads as math because the operators are evidence.
    out = escape_currency_dollars("<p>Then $3 + 4 = 7$ holds.</p>")
    assert out == "<p>Then $3 + 4 = 7$ holds.</p>"


def test_display_math_untouched():
    out = escape_currency_dollars("<p>Then $$3x = 9$$ follows.</p>")
    assert out == "<p>Then $$3x = 9$$ follows.</p>"
    assert r"\$" not in out


def test_mixed_currency_and_math_one_block():
    # (c) — currency ``$5`` escaped; the ``$3x + 2$`` math span left intact.
    out = escape_currency_dollars("<p>You pay $5 for $3x + 2$ items.</p>")
    assert r"\$5" in out
    assert "$3x + 2$" in out  # math span verbatim
    assert out == r"<p>You pay \$5 for $3x + 2$ items.</p>"


def test_answer_key_math_with_prose_word_untouched():
    # Regression — a genuine answer-key math span carrying a prose OPENER word
    # ("Solution") AND a LaTeX command must NOT have its opening ``$1139`` escaped
    # (doing so un-delimits the span and desyncs every downstream ``$…$`` pair,
    # surfacing the real math's ``\frac`` as phantom leakage in the scorecard).
    src = r"<p>Answer $1139 \text { Solution } (a)$ then divide.</p>"
    out = escape_currency_dollars(src)
    assert out == src  # byte-identical: the $…$ span is math, never escaped
    assert r"\$1139" not in out


def test_bracket_paren_math_untouched():
    out = escape_currency_dollars(r"<p>Given \(y = 2\) the fee is $7.</p>")
    assert r"\(y = 2\)" in out
    assert r"\$7" in out


def test_escape_is_idempotent():
    once = escape_currency_dollars("<p>Tickets $5 and $3 and math $x + 2$.</p>")
    assert escape_currency_dollars(once) == once


def test_noop_without_currency():
    assert escape_currency_dollars("<p>No dollars at all here.</p>") == (
        "<p>No dollars at all here.</p>"
    )
    # A lone non-currency ``$`` (not before a digit) is left as-is.
    assert escape_currency_dollars("<p>Var $x here.</p>") == "<p>Var $x here.</p>"


# --- DEFECT 1: currency never opens a math region -----------------------------
def test_currency_region_never_escapes_intervening_markup():
    """The exemplar cascade: two currency table cells with markup between.

    Source ``<td>25. $131.19</td> <td>27. <span …>a=-5</span></td> … <td>47.
    $32</td>``. Before the fix the ``$131.19`` opened a math region that closed
    at ``$32``, and the angle-bracket escape then rewrote every ``<``/``>`` in
    between — shipping ``25. $131.19&lt;/td&gt; &lt;td&gt;27. &lt;span …`` as
    literal learner-visible text.
    """
    src = (
        "<table><tbody><tr><td>25. $131.19</td> "
        '<td>27. <span class="semantik-figure-notation" role="img" '
        'aria-label="a=\\frac{1}{2} (image not recoverable)">a</span></td> '
        "</tr><tr><td>47. $32</td></tr></tbody></table>"
    )
    out = escape_math_angle_brackets(src)
    assert out == src  # byte-identical: no markup rewritten to entities
    assert "&lt;td&gt;" not in out
    assert "&lt;span" not in out


def test_currency_list_not_paired_into_math():
    # ``$5, $10, $15`` — a naive pairer glues ``$5, $`` into a math span. The
    # currency guard refuses, so every amount survives verbatim in the artifact.
    src = "<p>Prices are $5, $10, and $15 today.</p>"
    out = escape_math_angle_brackets(src)
    assert out == src


def test_markup_crossing_span_refused_even_without_currency():
    # The markup guard stands on its own: a ``$x$`` candidate whose body carries
    # an HTML tag is never one math run, whatever it opens on.
    src = "<p>$x</p><p>y$ and more</p>"
    assert escape_math_angle_brackets(src) == src


# --- adapter: the converted artifact keeps plain currency ---------------------
def _block(html: str, text: str) -> _AdapterBlock:
    return _AdapterBlock(
        html=html,
        region_kind="paragraph",
        raw_block_index=0,
        raw_text=text,
        heading_text=None,
        pages=[1],
    )


class _Result:
    def __init__(self, chapters):
        self.chapters = chapters
        self.exit_action = "ship_with_confidence"
        self.wcag_status = "passed"
        self.theta_score = 0.9
        self.flags = []
        self.lane_used = "x"
        self.lang = "en"


def test_normalize_keeps_plain_currency_in_html_and_sidecar():
    """The converted artifact carries ``$5`` — never a literal ``\\$``.

    Regression for the 601 ``\\$`` tokens the adapter-side escape shipped into
    the accessible HTML the chunker reads.
    """
    body = "Jeannette has $5 and $10 bills in her wallet."
    block = _block(f"<p>{body}</p>", body)
    ch = _AdapterChapter(title="Chapter 3 Fees", blocks=[block])
    out = normalize_cascade_to_ed4all(_Result([ch]), pdf_stem="doc-ch03")
    html = out["html"]
    assert "$5 and $10" in html
    assert r"\$" not in html
    sec_text = out["synthesized_sidecar"]["sections"][0]["data"]["text"]
    assert "$5" in sec_text and r"\$" not in sec_text
