r"""Raw ``<`` / ``>`` escape INSIDE math spans (round-8 — phantom-tag span break).

An OCR inequality glued to a letter (``\( a<b \)``) reaches the assembled learner
page as a LITERAL ``<``. The browser HTML tokenizer treats ``<`` immediately
before an ASCII letter as the start of a phantom tag and swallows the rest of the
``\(…\)`` span, so MathJax leaks the ``\(`` (a visible backslash-paren) and reds
the orphan ``\)`` — the exact "leaked ``\(`` + red ``\)`` + swallowed text"
signature. ``math_fold.escape_math_angle_brackets`` + the adapter
``_escape_math_angle_brackets`` pass rewrite each raw ``<`` / ``>`` inside a math
span to ``&lt;`` / ``&gt;`` in block HTML ONLY; the browser decodes the entity so
MathJax reads identical math and renders byte-identically, but no phantom tag ever
opens. ``raw_text`` / the sidecar keep plain ``x < 5`` for the chunker + retrieval.

All synthetic IR — no course-data path, no model, no cascade run.
"""
from __future__ import annotations

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _escape_math_angle_brackets,
)
from lib.semantik.math_fold import escape_math_angle_brackets


# --- unit: escape_math_angle_brackets -----------------------------------------
def test_inline_lt_glued_to_letter_escaped():
    # The dangerous case: ``<`` glued to a letter would open a phantom tag.
    out = escape_math_angle_brackets(r"<p>if \( a<b \) then done</p>")
    assert r"\( a&lt;b \)" in out
    assert "a<b" not in out  # no raw ``<`` survives inside the span


def test_inline_lt_gt_with_spaces_escaped():
    # ``\( x < 5 \) … \( x > -1 \)`` — the ch02 number-line inequalities.
    out = escape_math_angle_brackets(r"<p>(b) \( x < 5 \) (c) \( x > -1 \)</p>")
    assert r"\( x &lt; 5 \)" in out
    assert r"\( x &gt; -1 \)" in out


def test_dollar_span_angle_escaped():
    out = escape_math_angle_brackets("<p>compare $ a < b $ here</p>")
    assert "$ a &lt; b $" in out


def test_display_span_angle_escaped():
    out = escape_math_angle_brackets("<p>then $$ a > b $$ holds</p>")
    assert "$$ a &gt; b $$" in out


def test_real_html_tags_outside_math_untouched():
    # A ``<strong>`` tag OUTSIDE any math span is never entity-escaped.
    src = r"<p>Note <strong>key</strong> then \( x<y \) end</p>"
    out = escape_math_angle_brackets(src)
    assert "<strong>key</strong>" in out
    assert r"\( x&lt;y \)" in out


def test_no_angle_is_noop():
    src = r"<p>Check: Let \( y = -17 \). $$ -(y + 9) = 8 $$</p>"
    assert escape_math_angle_brackets(src) == src


def test_idempotent():
    src = r"<p>if \( a<b \) then \( c>d \) done</p>"
    once = escape_math_angle_brackets(src)
    assert escape_math_angle_brackets(once) == once


def test_currency_escape_not_treated_as_delimiter():
    # After the currency pass an escaped ``\$`` must not open a phantom span; a
    # raw ``<`` right after it (prose) is left alone (not inside real math).
    src = r"<p>pay \$5 when a < b in words</p>"
    out = escape_math_angle_brackets(src)
    # No math span → the prose ``<`` (already _esc_text'd upstream in real use)
    # is not our concern here; the currency ``\$`` is not a span opener.
    assert out == src


def test_exact_ch02_number_line_fixture_clean():
    # The exact ch02 Graph-on-the-number-line block: three inequalities, all
    # angle-escaped so the browser never phantom-tags a ``\(…\)`` span.
    src = (
        r"<p>Graph on the number line: (a) \( x \leq 1 \) (b) \( x < 5 \) "
        r"(c) \( x > -1 \) Solution (a) \( x \leq 1 \)</p>"
    )
    out = escape_math_angle_brackets(src)
    # Every raw inequality bracket inside a span is now an entity.
    assert "< 5" not in out and "> -1" not in out
    assert r"\( x &lt; 5 \)" in out and r"\( x &gt; -1 \)" in out
    # The ``\leq`` command word is NOT a raw angle bracket → untouched.
    assert r"\( x \leq 1 \)" in out


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
    plain = r"if \( a<b \) then \( x > 2 \) done"
    block = _block(f"<p>{plain}</p>", plain)
    block.repaired_text = plain
    ch = _AdapterChapter(title="Chapter 2 Inequalities", blocks=[block])
    _escape_math_angle_brackets([ch])
    assert r"\( a&lt;b \)" in block.html
    assert r"\( x &gt; 2 \)" in block.html
    # Sidecar / chunk text keeps the bare inequality — never the entity.
    assert block.raw_text == plain
    assert "&lt;" not in block.raw_text
    assert block.repaired_text == plain


def test_pass_leaves_angle_free_block_untouched():
    html = r"<p>Check: Let \( y = -17 \). $$ 8 = 8 \checkmark $$</p>"
    block = _block(html, "Check: Let y = -17. 8 = 8")
    ch = _AdapterChapter(title="Chapter 2 Solve", blocks=[block])
    _escape_math_angle_brackets([ch])
    assert block.html == html
