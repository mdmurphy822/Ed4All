r"""TikZ / pgfplots figure-code graceful fallback (round-10 — final).

The round-10 headless render audit (``scripts/render_audit.py``) surfaced a NEW
MathJax ``mjx-merror`` family the round-9 span sanitizer does not touch: the VLM
transcribed coordinate-plane FIGURES as raw TikZ picture code INSIDE math
delimiters (``$$\begin{tikzpicture}…\end{tikzpicture}$$`` — MathJax reds it
"Undefined environment tikzpicture"; also the pgfplots ``\begin{axis}…``
sibling). ch04 shipped 12 such nodes.

``math_fold.strip_tikz_figures`` + the adapter ``_strip_tikz_figures`` pass
replace a PURE-figure span with the accessible ``.semantik-figure-notation``
placeholder (no visible TikZ source) and keep the math of a MIXED span,
conservatively: a span with no TikZ / pgfplots env (``\begin{aligned}`` etc.) is
untouched. HTML-only — ``raw_text`` / the sidecar keep the plain fused text for
the chunker + retrieval.

All synthetic IR — no course-data path, no model, no cascade run.
"""
from __future__ import annotations

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _strip_tikz_figures,
)
from lib.semantik.math_fold import strip_tikz_figures

_PLACEHOLDER = (
    '<span class="semantik-figure-notation" role="img" '
    'aria-label="Coordinate-plane figure (notation could not be rendered)">'
    "[coordinate-plane figure]</span>"
)


# --- pure figure span → placeholder ------------------------------------------
def test_tikz_display_span_replaced_with_placeholder():
    # The ch04 "Identify the x- and y-intercepts on a graph" exercise figure.
    src = (
        r"<p>139. $$\begin{tikzpicture} \draw[very thin,color=gray] "
        r"(-6,-6) grid (6,6); \draw[thick] plot coordinates "
        r"{(-2,4) (-1,3) (0,2) (1,1) (2,0)}; \end{tikzpicture}$$</p>"
    )
    out = strip_tikz_figures(src)
    assert out == f"<p>139. {_PLACEHOLDER}</p>"
    assert "tikzpicture" not in out
    assert r"\draw" not in out  # raw TikZ source never ships visibly
    assert "$$" not in out       # bare delimiters go with the figure


def test_tikz_with_nested_dollar_nodes_replaced():
    # node[right] {$x$} — nested single-$ math inside the figure code; the whole
    # display span (and the nested dollars) are consumed by the placeholder.
    src = (
        r"$$\begin{tikzpicture} \draw[->] (-6.5,0) -- (6.5,0) "
        r"node[right] {$x$}; \draw[->] (0,-6.5) -- (0,6.5) "
        r"node[above] {$y$}; \end{tikzpicture}$$"
    )
    assert strip_tikz_figures(src) == _PLACEHOLDER


def test_pgfplots_axis_env_replaced():
    # An OUTER tikzpicture wrapping a nested pgfplots \begin{axis}…\end{axis}:
    # the backreferenced env name matches tikzpicture to its OWN \end.
    src = (
        r"$$\begin{tikzpicture}\begin{axis}[xlabel=$x$] "
        r"\addplot coordinates {(0,0) (1,1)}; "
        r"\end{axis}\end{tikzpicture}$$"
    )
    out = strip_tikz_figures(src)
    assert out == _PLACEHOLDER
    assert "axis" not in out and "tikzpicture" not in out


def test_bare_axis_env_replaced():
    # A standalone pgfplots \begin{axis}…\end{axis} (no tikzpicture wrapper).
    src = r"\[ \begin{axis} \addplot {x^2}; \end{axis} \]"
    assert strip_tikz_figures(src) == _PLACEHOLDER


def test_inline_tikz_span_replaced():
    src = r"see \( \begin{tikzpicture} \draw (0,0) circle (1); \end{tikzpicture} \) here"
    assert strip_tikz_figures(src) == f"see {_PLACEHOLDER} here"


def test_html_entity_arrows_in_tikz_still_dropped():
    # The adapter angle-escape pass turns -> into -&gt; inside the figure; the
    # whole env is still stripped regardless of the entity content.
    src = (
        r"$$\begin{tikzpicture} \draw[-&gt;] (-6.5,0) -- (6.5,0) "
        r"node[right] {$x$}; \end{tikzpicture}$$"
    )
    assert strip_tikz_figures(src) == _PLACEHOLDER


# --- mixed span → strip only the figure, keep the math -----------------------
def test_mixed_span_strips_tikz_keeps_math():
    src = r"$ x = 5 \begin{tikzpicture} \draw (0,0) -- (1,1); \end{tikzpicture} $"
    out = strip_tikz_figures(src)
    assert out == r"$x = 5$"
    assert "tikzpicture" not in out
    assert "semantik-figure-notation" not in out  # NOT a pure figure → no placeholder


def test_mixed_span_math_before_and_after_tikz():
    src = (
        r"\[ a + b = c \begin{tikzpicture}\draw (0,0);\end{tikzpicture} d = e \]"
    )
    out = strip_tikz_figures(src)
    assert out == r"\[a + b = c d = e\]"
    assert r"\draw" not in out


# --- OCR-truncated orphan opener ---------------------------------------------
def test_orphan_tikz_opener_dropped():
    # \begin{tikzpicture} with the \end cut off by OCR truncation → figure debris
    # to the span end, dropped; nothing real remains → placeholder.
    src = r"$$\begin{tikzpicture} \draw[thick] plot coordinates {(0,0) (1,1)}$$"
    assert strip_tikz_figures(src) == _PLACEHOLDER


# --- no false fire -----------------------------------------------------------
def test_no_false_fire_on_aligned_env():
    src = r"$$\begin{aligned} & a = 1 \\ & b = 2 \end{aligned}$$"
    assert strip_tikz_figures(src) == src


def test_no_false_fire_on_array_env():
    src = r"\[ \begin{array}{ll} 4^{3}=64 & \sqrt[3]{64}=4 \end{array} \]"
    assert strip_tikz_figures(src) == src


def test_no_false_fire_on_plain_math():
    src = r"<p>Let \( y = -17 \). $$ 3(2(-1) + 7) = 15 $$ holds.</p>"
    assert strip_tikz_figures(src) == src


def test_no_op_on_text_without_tikz():
    src = "no math, no figures, just prose about tikzpicture the word"
    # 'tikzpicture' appears but NOT as \begin{tikzpicture} → fast guard no-op.
    assert strip_tikz_figures(src) == src


def test_tikz_outside_math_delimiters_untouched():
    # A bare (un-delimited) \begin{tikzpicture} is not our defect (it is not a
    # MathJax span); this pass only rewrites DELIMITED spans.
    src = r"<p>\begin{tikzpicture} \draw (0,0); \end{tikzpicture}</p>"
    assert strip_tikz_figures(src) == src


# --- invariants --------------------------------------------------------------
def test_idempotent():
    src = (
        r"<p>139. $$\begin{tikzpicture}\draw (0,0);\end{tikzpicture}$$ "
        r"and $ x=5 \begin{axis}\addplot{x};\end{axis} $</p>"
    )
    once = strip_tikz_figures(src)
    assert strip_tikz_figures(once) == once
    assert "tikzpicture" not in once and "\\begin{axis}" not in once


def test_multiple_figure_spans_all_replaced():
    src = (
        r"139. $$\begin{tikzpicture}\draw (0,0);\end{tikzpicture}$$ "
        r"140. $$\begin{tikzpicture}\draw (1,1);\end{tikzpicture}$$"
    )
    out = strip_tikz_figures(src)
    assert out == f"139. {_PLACEHOLDER} 140. {_PLACEHOLDER}"
    assert out.count("semantik-figure-notation") == 2


def test_empty_and_none():
    assert strip_tikz_figures("") == ""
    assert strip_tikz_figures(None) == ""  # type: ignore[arg-type]


# --- adapter pass: html-only, sidecar preserved ------------------------------
def _block(html: str, text: str) -> _AdapterBlock:
    return _AdapterBlock(
        html=html,
        region_kind="paragraph",
        raw_block_index=0,
        raw_text=text,
        heading_text=None,
        pages=[1],
    )


def test_pass_replaces_html_only_sidecar_plain():
    plain = r"139. $$\begin{tikzpicture}\draw (0,0);\end{tikzpicture}$$"
    block = _block(f"<p>{plain}</p>", plain)
    block.repaired_text = plain
    ch = _AdapterChapter(title="Chapter 4 Graphs", blocks=[block])
    _strip_tikz_figures([ch])
    # HTML: figure code replaced by the accessible placeholder.
    assert _PLACEHOLDER in block.html
    assert "tikzpicture" not in block.html
    # Sidecar / chunk text is the chunker's business — left verbatim (TikZ kept).
    assert block.raw_text == plain
    assert block.repaired_text == plain


def test_pass_leaves_clean_block_untouched():
    html = r"<p>Let \( y = -17 \). $$ \frac{1}{2} + \frac{1}{2} = 1 $$</p>"
    block = _block(html, "Let y = -17.")
    ch = _AdapterChapter(title="Chapter 4 Graphs", blocks=[block])
    _strip_tikz_figures([ch])
    assert block.html == html
