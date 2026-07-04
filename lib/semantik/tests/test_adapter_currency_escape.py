r"""Currency-``$`` escape at the adapter seam (round-7b — MathJax false-pairing).

The OpenStax money word-problems carry lone currency ``$`` before digits ("costs
$5 … and $3"). The assembled end-user page enables MathJax v3 with
``inlineMath [['$','$']]``, so two such amounts in one paragraph FALSE-PAIR into
an italic inline-math span at render. ``math_fold.escape_currency_dollars`` + the
adapter ``_escape_currency_dollars`` pass rewrite each preserved currency ``$``
to ``\$`` (a literal dollar under the assembler's ``processEscapes: true``) in the
block HTML ONLY — ``raw_text`` / the sidecar keep plain ``$5`` for the chunker +
retrieval. Genuine ``$…$`` / ``$$…$$`` / ``\(…\)`` / ``\[…\]`` math is untouched.

All synthetic IR — no course-data path, no model, no cascade run.
"""
from __future__ import annotations

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _escape_currency_dollars,
    normalize_cascade_to_ed4all,
)
from lib.semantik.math_fold import escape_currency_dollars


# --- unit: escape_currency_dollars --------------------------------------------
def test_two_currency_amounts_both_escaped():
    # (a) — two currency amounts in one paragraph: both dollars escaped.
    out = escape_currency_dollars("<p>It costs $5 to enter and $3 to ride.</p>")
    assert r"\$5" in out and r"\$3" in out
    assert "$5" not in out.replace(r"\$5", "")  # no UN-escaped $5 remains
    assert "$3" not in out.replace(r"\$3", "")


def test_paired_inline_math_untouched():
    # (b) — genuine inline math ``$5x + 3$`` is a paired span → never escaped.
    out = escape_currency_dollars("<p>Compute $5x + 3$ carefully.</p>")
    assert out == "<p>Compute $5x + 3$ carefully.</p>"
    assert r"\$" not in out


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


# --- adapter pass: html-only, sidecar preserved -------------------------------
def _block(html: str, text: str) -> _AdapterBlock:
    return _AdapterBlock(
        html=html,
        region_kind="paragraph",
        raw_block_index=0,
        raw_text=text,
        heading_text=None,
        pages=[1],
    )


def test_pass_escapes_html_only_sidecar_plain():
    # (a) end-to-end contract: the adapter pass escapes ``block.html`` and leaves
    # ``raw_text`` / ``repaired_text`` (sidecar + chunk text) with plain ``$5``.
    plain = "It costs $5 to enter and $3 to ride."
    block = _block(f"<p>{plain}</p>", plain)
    block.repaired_text = plain
    ch = _AdapterChapter(title="Chapter 1 Money", blocks=[block])
    _escape_currency_dollars([ch])
    assert r"\$5" in block.html and r"\$3" in block.html
    # Sidecar / chunk text keeps the bare currency — never the ``\$`` escape.
    assert block.raw_text == plain
    assert block.repaired_text == plain
    assert r"\$" not in block.raw_text


def test_pass_leaves_math_block_untouched():
    # (b) a block whose body is genuine paired math is byte-identical after.
    html, text = "<p>Compute $5x + 3$ now.</p>", "Compute $5x + 3$ now."
    block = _block(html, text)
    ch = _AdapterChapter(title="Chapter 2 Algebra", blocks=[block])
    _escape_currency_dollars([ch])
    assert block.html == html
    assert block.raw_text == text


# --- adapter integration: full normalize --------------------------------------
class _Result:
    def __init__(self, chapters):
        self.chapters = chapters
        self.exit_action = "ship_with_confidence"
        self.wcag_status = "passed"
        self.theta_score = 0.9
        self.flags = []
        self.lane_used = "x"
        self.lang = "en"


def test_normalize_escapes_currency_in_html_not_sidecar():
    # Full adapter run: a single-currency block survives sanitize; its HTML
    # carries ``\$5`` while the sidecar text keeps plain ``$5``.
    body = "The ride costs $5 total."
    block = _block(f"<p>{body}</p>", body)
    ch = _AdapterChapter(title="Chapter 3 Fees", blocks=[block])
    out = normalize_cascade_to_ed4all(_Result([ch]), pdf_stem="doc-ch03")
    html = out["html"]
    assert r"\$5" in html
    sec_text = out["synthesized_sidecar"]["sections"][0]["data"]["text"]
    assert "$5" in sec_text and r"\$" not in sec_text
