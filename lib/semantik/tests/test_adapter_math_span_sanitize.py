r"""Math-span content sanitizer (round-9 — MathJax ``mjx-merror`` typeset errors).

The headless render audit (``scripts/render_audit.py``) surfaced MathJax typeset
failures that NO text audit caught — they only appear after the browser runs.
Two OCR/VLM-emission families dominate, both INSIDE already-delimited math spans:

  (1) "Misplaced &" — the VLM pulled a textbook TABLE into a single ``$…$`` run,
      so a tabular column separator ``&`` (or its ``&amp;`` entity, decoded to a
      literal ``&`` before MathJax reads it) sits in math with NO alignment
      environment. Table-cell debris also appears as a lone ``$ & $`` span.
  (2) "Missing argument for \sqrt" (and the identical family ``\frac`` /
      ``\stackrel`` / trailing ``^``/``_`` / unclosed ``\begin{aligned}`` /
      unmatched ``\left``) — OCR truncated the operand.

``math_fold.sanitize_math_spans`` + the adapter ``_sanitize_math_spans`` pass
repair BOTH inside the delimited span content, conservatively: alignment ``&``
inside ``\begin{array}…`` and a valid ``\sqrt{x}`` / ``\sqrt 2`` are untouched;
``&lt;`` / ``&gt;`` entities stay real operators. HTML-only — ``raw_text`` / the
sidecar keep the plain fused text for the chunker + retrieval.

All synthetic IR — no course-data path, no model, no cascade run.
"""
from __future__ import annotations

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _sanitize_math_spans,
)
from lib.semantik.math_fold import sanitize_math_spans


# --- class 1: misplaced ``&`` -------------------------------------------------
def test_misplaced_amp_dropped_to_space():
    # A tabular ``&`` inside non-alignment inline math → space (no merror).
    out = sanitize_math_spans(r"<p>\( x & y \)</p>")
    assert out == r"<p>\( x y \)</p>"
    assert "&" not in out


def test_lone_amp_span_dropped_entirely():
    # The ``$ & $`` table-cell debris span sanitizes to empty → delimiters gone.
    assert sanitize_math_spans(r"$ & $") == ""


def test_amp_entity_in_math_dropped():
    # ``&amp;`` decodes to a literal ``&`` before MathJax → also a misplaced tab.
    out = sanitize_math_spans(r"<p>\( a &amp; b \)</p>")
    assert out == r"<p>\( a b \)</p>"


def test_aligned_env_amp_preserved():
    # ``&`` inside ``\begin{aligned}`` is a LEGITIMATE alignment tab — untouched.
    src = r"$$ \begin{aligned} & a = 1 \\ & b = 2 \end{aligned} $$"
    assert sanitize_math_spans(src) == src


def test_array_env_amp_preserved():
    src = (
        r"<p>\[ \begin{array}{ll} 4^{3}=64 & \sqrt[3]{64}=4 "
        r"\end{array} \]</p>"
    )
    assert sanitize_math_spans(src) == src


def test_lt_gt_entities_preserved_as_operators():
    # ``&lt;`` / ``&gt;`` decode to real ``<`` / ``>`` operators — NOT tabs.
    src = r"<p>\( a &lt; b \) and \( c &gt; d \)</p>"
    assert sanitize_math_spans(src) == src


def test_amp_outside_math_untouched():
    # A ``&`` in PROSE (outside any delimiter) is the HTML parser's business.
    src = "no math here x & y & z"
    assert sanitize_math_spans(src) == src


# --- class 2: dangling ``\sqrt`` (+ same-family truncations) -------------------
def test_dangling_sqrt_dropped():
    # The ch09 ``\text{ Simplify: } \sqrt`` — OCR truncated the radicand.
    out = sanitize_math_spans(r"\( \text{ Simplify: } \sqrt \)")
    assert out == r"\( \text{ Simplify: }\)"
    assert r"\sqrt" not in out


def test_dangling_sqrt_with_index_dropped():
    # ``\sqrt[3]`` (index but no radicand) at span end → dropped whole.
    assert sanitize_math_spans(r"\( \sqrt[3] \)") == ""


def test_valid_sqrt_braced_preserved():
    src = r"\( \sqrt{x} \)"
    assert sanitize_math_spans(src) == src


def test_valid_sqrt_indexed_preserved():
    src = r"\( \sqrt[3]{8} \)"
    assert sanitize_math_spans(src) == src


def test_valid_sqrt_single_token_preserved():
    # ``\sqrt 2`` — an unbraced SINGLE-token argument is valid LaTeX for a
    # single char, and ``\sqrt`` is not at the span end → kept.
    src = r"\( \sqrt 2 \)"
    assert sanitize_math_spans(src) == src


def test_dangling_stackrel_dropped():
    # The ch02 ``\stackrel`` truncation → dropped.
    assert sanitize_math_spans(r"\( \stackrel \)") == ""


def test_dangling_frac_one_arg_dropped():
    # ``\frac{1}`` — first arg present, second OCR-truncated → dropped whole.
    assert sanitize_math_spans(r"\( \frac{1} \)") == ""


def test_valid_frac_preserved():
    src = r"\( \frac{1}{2} \)"
    assert sanitize_math_spans(src) == src


def test_trailing_superscript_dropped():
    out = sanitize_math_spans(r"\( x^ \)")
    assert out == r"\( x\)"


# --- class 2 extras: unclosed env + unmatched \left --------------------------
def test_unclosed_aligned_env_span_dropped():
    # ``$$\begin{aligned} $$`` — opener with no ``\end`` → whole span dropped.
    assert sanitize_math_spans(r"$$\begin{aligned} $$") == ""


def test_orphan_end_and_hline_cleared():
    # ``\hline \end{array}`` (array lost its ``\begin``) → orphan env + rule gone.
    assert sanitize_math_spans(r"\( \hline \end{array} \)") == ""


def test_unmatched_left_stripped():
    # ``(a) \left(9p`` — a ``\left(`` with no ``\right`` → strip the sizing cmd,
    # the bare ``(`` glyph still renders.
    out = sanitize_math_spans(r"\( (a) \left(9p \)")
    assert out == r"\( (a) (9p \)"
    assert r"\left" not in out


def test_balanced_left_right_preserved():
    src = r"\( \left( x + 1 \right) \)"
    assert sanitize_math_spans(src) == src


def test_bare_row_break_preserved():
    # A bare ``\\`` line break (no array env) is legal — NOT dropped.
    src = r"\( a \\ b \)"
    assert sanitize_math_spans(src) == src


# --- invariants ---------------------------------------------------------------
def test_no_op_on_clean_math():
    src = r"<p>Let \( y = -17 \). $$ 3(2(-1) + 7) = 15 $$ holds.</p>"
    assert sanitize_math_spans(src) == src


def test_idempotent():
    src = r"<p>\( x & y \) then \( \text{done } \sqrt \) and $$\begin{aligned} $$</p>"
    once = sanitize_math_spans(src)
    assert sanitize_math_spans(once) == once


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


def test_pass_sanitizes_html_only_sidecar_plain():
    plain = r"cell \( x & y \) and \( \text{Simplify: } \sqrt \) done"
    block = _block(f"<p>{plain}</p>", plain)
    block.repaired_text = plain
    ch = _AdapterChapter(title="Chapter 9 Radicals", blocks=[block])
    _sanitize_math_spans([ch])
    # HTML: misplaced ``&`` folded + dangling ``\sqrt`` dropped.
    assert r"\( x y \)" in block.html
    assert r"\sqrt" not in block.html
    # Sidecar / chunk text is the chunker's business — left verbatim.
    assert block.raw_text == plain
    assert block.repaired_text == plain


def test_pass_leaves_clean_block_untouched():
    html = r"<p>Let \( y = -17 \). $$ \frac{1}{2} + \frac{1}{2} = 1 $$</p>"
    block = _block(html, "Let y = -17.")
    ch = _AdapterChapter(title="Chapter 9 Solve", blocks=[block])
    _sanitize_math_spans([ch])
    assert block.html == html
