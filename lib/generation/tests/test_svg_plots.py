"""Unit tests for the deterministic SVG math-plotter (``lib/generation/svg_plots``).

Covers the parser (good / ambiguous / malformed), each plot kind's SVG
structure + accessibility, byte-for-byte determinism, and the fail-closed path.
All equations are synthesized (no course slugs / corpus vocabulary).
"""

from __future__ import annotations

import xml.dom.minidom as minidom

import pytest

from lib.generation import svg_plots as sp

_SYMPY = sp._SYMPY_AVAILABLE
_needs_sympy = pytest.mark.skipif(not _SYMPY, reason="sympy not installed")


# --------------------------------------------------------------------------- #
# Parser — good inputs
# --------------------------------------------------------------------------- #
@_needs_sympy
@pytest.mark.parametrize(
    "text,boundary,direction,closed",
    [
        (r"\( x > -2 \)", -2.0, "right", False),
        (r"x \geq 3", 3.0, "right", True),
        ("-2 < x", -2.0, "right", False),
        ("2x - 4 <= 0", 2.0, "left", True),
        ("x < 5", 5.0, "left", False),
    ],
)
def test_parse_number_line(text, boundary, direction, closed):
    spec = sp.parse_plot_spec(text)
    assert spec is not None
    assert spec.kind == "number_line"
    assert spec.boundary == pytest.approx(boundary)
    assert spec.direction == direction
    assert spec.closed is closed


@_needs_sympy
@pytest.mark.parametrize(
    "text,slope,intercept",
    [
        (r"\( y = 2x + 3 \)", 2.0, 3.0),
        ("y = -x + 1", -1.0, 1.0),
        ("f(x) = 4x", 4.0, 0.0),
        ("y = 5", 0.0, 5.0),
    ],
)
def test_parse_line_slope_intercept(text, slope, intercept):
    spec = sp.parse_plot_spec(text)
    assert spec is not None
    assert spec.kind == "line_graph"
    assert spec.slope == pytest.approx(slope)
    assert spec.intercept == pytest.approx(intercept)


@_needs_sympy
def test_parse_line_standard_form():
    spec = sp.parse_plot_spec("2x + 3y = 6")
    assert spec is not None
    assert spec.kind == "line_graph"
    # 3y = 6 - 2x  ->  y = -2/3 x + 2
    assert spec.slope == pytest.approx(-2.0 / 3.0)
    assert spec.intercept == pytest.approx(2.0)


@_needs_sympy
@pytest.mark.parametrize(
    "text,a,b,c",
    [
        (r"\( y = x^2 - 4 \)", 1.0, 0.0, -4.0),
        ("y = 2x^2 + 3x - 1", 2.0, 3.0, -1.0),
        ("f(x) = -x^2", -1.0, 0.0, 0.0),
    ],
)
def test_parse_parabola(text, a, b, c):
    spec = sp.parse_plot_spec(text)
    assert spec is not None
    assert spec.kind == "parabola"
    assert spec.a == pytest.approx(a)
    assert spec.b == pytest.approx(b)
    assert spec.c == pytest.approx(c)


# --------------------------------------------------------------------------- #
# Parser — ambiguous + malformed all FAIL CLOSED (None)
# --------------------------------------------------------------------------- #
@_needs_sympy
@pytest.mark.parametrize(
    "text",
    [
        r"\( x > -2 \) and \( y = x \)",  # two fragments = ambiguous
        "y = x = 3",                       # two equalities
        "-2 < x < 3",                      # compound inequality
        "y = x^3 + 1",                     # degree 3 unsupported
        "x + y + z = 1",                   # extra free symbol
        "y = sin(x)",                      # non-polynomial (extra symbol residue)
        "the graph is nice",              # prose, no equation
        "y = ",                            # empty RHS
        "",                                # empty
        "   ",                             # whitespace
        r"\( a b c \)",                    # multi-symbol garbage
        "x != 4",                          # unsupported relation
    ],
)
def test_parse_fail_closed(text):
    assert sp.parse_plot_spec(text) is None


@_needs_sympy
def test_parse_none_for_non_string():
    assert sp.parse_plot_spec(None) is None  # type: ignore[arg-type]
    assert sp.parse_plot_spec(12345) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# SVG structure + accessibility
# --------------------------------------------------------------------------- #
def _parse_svg(svg: str):
    """Parse the SVG string (asserts well-formed XML) and return the DOM."""
    return minidom.parseString(svg)


@_needs_sympy
@pytest.mark.parametrize(
    "text,cls,viewbox",
    [
        ("x > -2", "svg-number-line", "0 0 480 120"),
        ("y = 2x + 3", "svg-line-graph", "0 0 480 360"),
        ("y = x^2 - 4", "svg-parabola", "0 0 480 360"),
    ],
)
def test_svg_structure_and_a11y(text, cls, viewbox):
    result = sp.svg_for_equation(text, id_prefix="blk")
    assert result is not None
    svg, alt = result
    dom = _parse_svg(svg)
    root = dom.documentElement
    assert root.tagName == "svg"
    assert root.getAttribute("viewBox") == viewbox
    assert root.getAttribute("class") == cls
    # role="img" + aria-labelledby pointing at an in-SVG title + desc.
    assert root.getAttribute("role") == "img"
    labelledby = root.getAttribute("aria-labelledby")
    assert labelledby == "blk-title blk-desc"
    title = dom.getElementsByTagName("title")
    desc = dom.getElementsByTagName("desc")
    assert len(title) == 1 and title[0].getAttribute("id") == "blk-title"
    assert len(desc) == 1 and desc[0].getAttribute("id") == "blk-desc"
    assert title[0].firstChild.data.strip()
    assert len(desc[0].firstChild.data.strip()) > 20  # a real long-description
    assert alt  # concise caption-style alt


@_needs_sympy
def test_number_line_has_boundary_circle_and_ray():
    svg, _ = sp.svg_for_equation("x >= 3", id_prefix="nl")
    dom = _parse_svg(svg)
    # A boundary circle (closed ⇒ filled with the plot color).
    circles = dom.getElementsByTagName("circle")
    assert len(circles) == 1
    assert circles[0].getAttribute("fill") == sp._C_PLOT  # closed circle filled
    # A shaded ray line carrying the arrow marker.
    lines = [
        ln for ln in dom.getElementsByTagName("line")
        if ln.getAttribute("marker-end")
    ]
    assert len(lines) == 1


@_needs_sympy
def test_number_line_open_circle_is_hollow():
    svg, _ = sp.svg_for_equation("x > 3", id_prefix="nl")
    dom = _parse_svg(svg)
    circ = dom.getElementsByTagName("circle")[0]
    assert circ.getAttribute("fill") == "#ffffff"  # open ⇒ hollow


@_needs_sympy
def test_line_graph_marks_intercepts():
    # y = 2x + 4 -> y-intercept (0,4), x-intercept (-2,0), both in window.
    svg, _ = sp.svg_for_equation("y = 2x + 4", id_prefix="lg")
    dom = _parse_svg(svg)
    texts = [t.firstChild.data for t in dom.getElementsByTagName("text") if t.firstChild]
    assert any("(0, 4)" in t for t in texts)
    assert any("(-2, 0)" in t for t in texts)
    # The line itself is at least one polyline.
    assert dom.getElementsByTagName("polyline")


@_needs_sympy
def test_parabola_marks_vertex_and_axis_of_symmetry():
    # y = x^2 - 4x + 3 -> vertex (2, -1), axis x = 2.
    svg, alt = sp.svg_for_equation("y = x^2 - 4x + 3", id_prefix="pb")
    dom = _parse_svg(svg)
    texts = [t.firstChild.data for t in dom.getElementsByTagName("text") if t.firstChild]
    assert any("vertex (2, -1)" in t for t in texts)
    # Dashed axis of symmetry present.
    dashed = [
        ln for ln in dom.getElementsByTagName("line")
        if ln.getAttribute("stroke-dasharray")
    ]
    assert len(dashed) == 1
    assert "vertex (2, -1)" in alt


# --------------------------------------------------------------------------- #
# Determinism — same input ⇒ byte-identical output
# --------------------------------------------------------------------------- #
@_needs_sympy
@pytest.mark.parametrize("text", ["x > -2", "y = 2x + 3", "y = x^2 - 4x + 3"])
def test_determinism_byte_identical(text):
    a = sp.svg_for_equation(text, id_prefix="d")
    b = sp.svg_for_equation(text, id_prefix="d")
    assert a is not None and b is not None
    assert a[0] == b[0]  # SVG bytes identical
    assert a[1] == b[1]  # alt identical


@_needs_sympy
def test_determinism_no_timestamp_or_object_id():
    svg, _ = sp.svg_for_equation("y = x^2 - 4", id_prefix="d")
    # No non-deterministic tokens leaked into the emit.
    assert "0x" not in svg  # no repr-style object id
    assert "20" + "26" not in svg  # no year-like timestamp for this input


# --------------------------------------------------------------------------- #
# id_prefix threads through for on-page uniqueness
# --------------------------------------------------------------------------- #
@_needs_sympy
def test_id_prefix_threads_through():
    svg, _ = sp.svg_for_equation("y = x^2", id_prefix="week03blk7")
    assert 'aria-labelledby="week03blk7-title week03blk7-desc"' in svg
    assert 'id="week03blk7-title"' in svg


# --------------------------------------------------------------------------- #
# svg_for_equation fail-closed contract
# --------------------------------------------------------------------------- #
@_needs_sympy
def test_svg_for_equation_fail_closed():
    assert sp.svg_for_equation("not an equation at all") is None
    assert sp.svg_for_equation("") is None


def test_render_unknown_kind_returns_none():
    spec = sp.PlotSpec(kind="pie_chart")
    assert sp.render_plot_svg(spec) is None
