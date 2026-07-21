r"""Constrained TikZ → inline-SVG re-draw (real-figure wave — 2026-07).

Exercises :mod:`lib.semantik.tikz_draw` — the parser productions (grid, segment
paths with style opts, filled/unfilled circle + rectangle, ``\node`` labels,
``[scale=…]``, HTML-escaped arrows), golden SVG for representative corpus-shape
notations, the ``None`` fallbacks (pgfplots / ``plot coordinates`` /
``\includegraphics`` hallucination / unknown option / empty picture), and the
flag-gated adapter pass (``SEMANTIK_RENDER_TIKZ_FIGURES``) incl. the mixed-span
placeholder emission that closes the round-10 silent-drop gap.

All synthetic notation — no course-data path, no model, no cascade run.
"""
from __future__ import annotations

import pytest

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _render_tikz_figures,
)
from lib.semantik.math_fold import _TIKZ_FIGURE_PLACEHOLDER
from lib.semantik.tikz_draw import parse_tikz, render_svg, render_tikz_figures

_PLACEHOLDER = _TIKZ_FIGURE_PLACEHOLDER


# --- parser productions ------------------------------------------------------
def test_parse_grid_and_axes_and_point():
    # The canonical ch04 coordinate-plane figure (§1.4).
    spec = parse_tikz(
        r"\begin{tikzpicture}[scale=0.5] \draw[help lines] (-6,-6) grid (6,6); "
        r"\draw[->] (-7,0) -- (7,0) node[right] {$x$}; "
        r"\filldraw [gray] (1,4) circle (2pt); \end{tikzpicture}"
    )
    assert spec is not None
    assert spec.scale == 0.5
    kinds = [type(e).__name__ for e in spec.elements]
    # grid, path, inline x-label node, filled point.
    assert kinds == ["_Grid", "_Path", "_Node", "_Circle"]
    node = spec.elements[2]
    assert node.text == "x"          # $x$ math rendered as plain text
    assert node.x == 7.0 and node.y == 0.0   # attached to the path endpoint
    circ = spec.elements[3]
    assert circ.filled and circ.r_is_pt and circ.color == "gray"


def test_parse_segment_path_thick_and_node_at():
    spec = parse_tikz(
        r"\begin{tikzpicture} \draw[thick] (0,0) -- (0,4); "
        r"\node at (-1.5,2) {12 ft}; \end{tikzpicture}"
    )
    assert spec is not None
    path, node = spec.elements
    assert type(path).__name__ == "_Path" and path.thick is True
    assert path.points == ((0.0, 0.0), (0.0, 4.0))
    assert type(node).__name__ == "_Node" and node.text == "12 ft"


def test_parse_dashed_and_dotted_styles():
    spec = parse_tikz(
        r"\begin{tikzpicture} \draw[dashed] (0,0) -- (1,1); "
        r"\draw[dotted] (1,0) -- (0,1); \end{tikzpicture}"
    )
    assert spec is not None
    assert spec.elements[0].dashed is True
    assert spec.elements[1].dotted is True


def test_parse_unfilled_circle_and_filled_rectangle():
    spec = parse_tikz(
        r"\begin{tikzpicture} \draw (0,0) circle (1); "
        r"\fill[blue] (0,0) rectangle (2,3); \end{tikzpicture}"
    )
    assert spec is not None
    circ, rect = spec.elements
    assert type(circ).__name__ == "_Circle" and not circ.filled and not circ.r_is_pt
    assert circ.r == 1.0
    assert type(rect).__name__ == "_Rect" and rect.filled and rect.color == "blue"


def test_parse_multi_segment_path():
    spec = parse_tikz(
        r"\begin{tikzpicture} \draw (0,0) -- (1,1) -- (2,0) -- (3,1); \end{tikzpicture}"
    )
    assert spec is not None
    assert spec.elements[0].points == ((0, 0), (1, 1), (2, 0), (3, 1))


def test_parse_html_escaped_arrow():
    # Arrows arrive HTML-escaped inside block math spans (``->`` -> ``-&gt;``).
    spec = parse_tikz(
        r"\begin{tikzpicture} \draw[-&gt;] (-6.5,0) -- (6.5,0) "
        r"node[right] {$x$}; \end{tikzpicture}"
    )
    assert spec is not None
    assert spec.elements[0].arrow is True


# --- None fallbacks ----------------------------------------------------------
@pytest.mark.parametrize(
    "notation",
    [
        # pgfplots plot-coordinates (the round-10 test's own notation) — unsupported.
        r"\begin{tikzpicture} \draw[thick] plot coordinates {(-2,4) (-1,3)}; \end{tikzpicture}",
        # bare pgfplots axis env — no tikzpicture wrapper.
        r"\begin{axis} \addplot {x^2}; \end{axis}",
        # \addplot inside tikzpicture.
        r"\begin{tikzpicture}\begin{axis}[xlabel=$x$] \addplot coordinates {(0,0)}; \end{axis}\end{tikzpicture}",
        # the ch07 \includegraphics hallucination.
        r"\begin{tikzpicture}[overlay, remember picture] \includegraphics{foo.png}; \end{tikzpicture}",
        # unknown draw option → conservative fail.
        r"\begin{tikzpicture} \draw[wobble] (0,0) -- (1,1); \end{tikzpicture}",
        # unknown colour → fail.
        r"\begin{tikzpicture} \fill[chartreuse] (0,0) rectangle (1,1); \end{tikzpicture}",
        # empty picture → nothing to draw.
        r"\begin{tikzpicture}\end{tikzpicture}",
        # no tikzpicture env at all.
        r"$$ 3(2(-1)+7) = 15 $$",
        # single coordinate — not a path.
        r"\begin{tikzpicture} \draw (0,0); \end{tikzpicture}",
    ],
)
def test_parse_returns_none_outside_grammar(notation):
    assert parse_tikz(notation) is None


def test_parse_none_on_empty():
    assert parse_tikz("") is None
    assert parse_tikz(None) is None  # type: ignore[arg-type]


# --- golden SVG --------------------------------------------------------------
def test_golden_svg_unfilled_circle():
    spec = parse_tikz(r"\begin{tikzpicture} \draw (0,0) circle (1); \end{tikzpicture}")
    assert render_svg(spec) == (
        '<svg class="semantik-figure-svg" role="img" '
        'aria-label="Coordinate-plane figure" viewBox="0 0 84 84" '
        'xmlns="http://www.w3.org/2000/svg">'
        "<title>Coordinate-plane figure</title>"
        '<circle cx="42" cy="42" r="30" fill="none" stroke="currentColor" '
        'stroke-width="1"/></svg>'
    )


def test_golden_svg_filled_rectangle():
    spec = parse_tikz(
        r"\begin{tikzpicture} \fill[blue] (0,0) rectangle (2,3); \end{tikzpicture}"
    )
    assert render_svg(spec) == (
        '<svg class="semantik-figure-svg" role="img" '
        'aria-label="Coordinate-plane figure" viewBox="0 0 84 114" '
        'xmlns="http://www.w3.org/2000/svg">'
        "<title>Coordinate-plane figure</title>"
        '<rect x="12" y="12" width="60" height="90" fill="blue"/></svg>'
    )


def test_golden_svg_thick_segment_with_label():
    spec = parse_tikz(
        r"\begin{tikzpicture} \draw[thick] (0,0) -- (0,4); "
        r"\node at (-1.5,2) {12 ft}; \end{tikzpicture}"
    )
    assert render_svg(spec) == (
        '<svg class="semantik-figure-svg" role="img" '
        'aria-label="Coordinate-plane figure with labels: 12 ft" '
        'viewBox="0 0 69 144" xmlns="http://www.w3.org/2000/svg">'
        "<title>Coordinate-plane figure with labels: 12 ft</title>"
        '<polyline points="57,132 57,12" stroke="currentColor" '
        'stroke-width="2" fill="none"/>'
        '<text x="12" y="72" font-size="13" text-anchor="middle" '
        'dominant-baseline="middle" fill="currentColor">12 ft</text></svg>'
    )


def test_svg_structural_invariants_for_axes():
    spec = parse_tikz(
        r"\begin{tikzpicture}[scale=0.5] \draw[help lines] (-6,-6) grid (6,6); "
        r"\draw[->] (-7,0) -- (7,0) node[right] {$x$}; "
        r"\draw[->] (0,-7) -- (0,7) node[above] {$y$}; "
        r"\filldraw [gray] (1,4) circle (2pt); \end{tikzpicture}"
    )
    svg = render_svg(spec)
    assert svg.startswith('<svg class="semantik-figure-svg" role="img"')
    assert 'aria-label="Coordinate-plane figure with labels: x, y"' in svg
    assert svg.count("<title>") == 1
    assert 'viewBox="0 0' in svg
    assert svg.count(f'marker-end="url(#semantik-tikz-arrow)"') == 2  # both axes
    assert '<circle' in svg and 'fill="gray"' in svg
    assert svg.endswith("</svg>")


def test_xml_escape_in_label():
    spec = parse_tikz(
        r"\begin{tikzpicture} \node at (0,0) {a & b < c}; \end{tikzpicture}"
    )
    svg = render_svg(spec)
    assert "a &amp; b &lt; c" in svg
    assert "a & b <" not in svg


# --- adapter pass: flag OFF (default) → byte-identical no-op ------------------
def _block(html: str) -> _AdapterBlock:
    return _AdapterBlock(
        html=html,
        region_kind="paragraph",
        raw_block_index=0,
        raw_text="side",
        heading_text=None,
        pages=[1],
    )


def test_adapter_pass_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_RENDER_TIKZ_FIGURES", raising=False)
    html = (
        r"<p>139. $$\begin{tikzpicture} \draw (0,0) circle (1); \end{tikzpicture}$$</p>"
    )
    block = _block(html)
    ch = _AdapterChapter(title="Chapter 4 Graphs", blocks=[block])
    _render_tikz_figures([ch])
    assert block.html == html  # untouched — the strip pass owns the default path


# --- adapter pass: flag ON ---------------------------------------------------
def test_adapter_pass_renders_pure_figure_when_flag_on(monkeypatch):
    monkeypatch.setenv("SEMANTIK_RENDER_TIKZ_FIGURES", "1")
    html = (
        r"<p>139. $$\begin{tikzpicture} \draw (0,0) circle (1); \end{tikzpicture}$$</p>"
    )
    block = _block(html)
    block.repaired_text = "side"
    ch = _AdapterChapter(title="Chapter 4 Graphs", blocks=[block])
    _render_tikz_figures([ch])
    assert '<figure class="semantik-figure">' in block.html
    assert 'class="semantik-figure-svg"' in block.html
    assert "tikzpicture" not in block.html
    assert "$$" not in block.html          # delimiters consumed with the figure
    # HTML-only: sidecar / chunk text untouched.
    assert block.raw_text == "side"
    assert block.repaired_text == "side"


def test_adapter_pass_unparseable_left_for_strip(monkeypatch):
    monkeypatch.setenv("SEMANTIK_RENDER_TIKZ_FIGURES", "1")
    # ``plot coordinates`` is outside the grammar → span left untouched so the
    # downstream strip pass emits the placeholder.
    html = (
        r"<p>$$\begin{tikzpicture} \draw[thick] plot coordinates "
        r"{(-2,4) (-1,3)}; \end{tikzpicture}$$</p>"
    )
    block = _block(html)
    ch = _AdapterChapter(title="Ch", blocks=[block])
    _render_tikz_figures([ch])
    assert block.html == html  # unchanged — no <figure>, still has the env
    assert "<figure" not in block.html


def test_adapter_pass_mixed_span_emits_placeholder(monkeypatch):
    monkeypatch.setenv("SEMANTIK_RENDER_TIKZ_FIGURES", "1")
    # Mixed real-math + figure span: keep the math, but leave the placeholder for
    # the figure (round-10 silent-drop gap — previously vanished with no trace).
    html = r"<p>$ x = 5 \begin{tikzpicture} \draw (0,0) -- (1,1); \end{tikzpicture} $</p>"
    block = _block(html)
    ch = _AdapterChapter(title="Ch", blocks=[block])
    _render_tikz_figures([ch])
    assert "$x = 5$" in block.html
    assert _PLACEHOLDER in block.html
    assert "tikzpicture" not in block.html


def test_adapter_pass_idempotent_when_flag_on(monkeypatch):
    monkeypatch.setenv("SEMANTIK_RENDER_TIKZ_FIGURES", "1")
    html = (
        r"<p>$$\begin{tikzpicture} \draw (0,0) circle (1); \end{tikzpicture}$$ "
        r"and $ x=5 \begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture} $</p>"
    )
    block = _block(html)
    ch = _AdapterChapter(title="Ch", blocks=[block])
    _render_tikz_figures([ch])
    once = block.html
    _render_tikz_figures([ch])          # second pass — strict no-op
    assert block.html == once
    assert "tikzpicture" not in once


def test_render_tikz_figures_fast_guard_noop():
    # No TikZ begin → returned unchanged (fast guard).
    src = "<p>Let \\( y = -17 \\). $$ \\frac{1}{2} = 0.5 $$</p>"
    assert render_tikz_figures(src) == src


def test_render_tikz_figures_empty_and_none():
    assert render_tikz_figures("") == ""
    assert render_tikz_figures(None) == ""  # type: ignore[arg-type]
