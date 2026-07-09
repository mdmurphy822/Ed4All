"""Deterministic (no-LLM) SVG math-plotter for ``diagram`` (B06) blocks.

Motivation: an algebra course build shipped 2 images across 659 blocks — weeks
teaching graphing had ZERO figures. This module closes that dual-coding hole
DETERMINISTICALLY: given an equation already present in a content block, it
hand-rolls a labelled SVG figure (no matplotlib, no network, no randomness) for
three plot kinds:

* ``number_line`` — a linear inequality's solution ray (``x > -2``), with ticks,
  an open/closed boundary circle, and a shaded arrowed ray.
* ``line_graph``  — a slope-intercept / standard-form line (``y = mx + b`` /
  ``Ax + By = C``) drawn across a fixed window with gridlines + intercept dots.
* ``parabola``    — a quadratic (``y = ax^2 + bx + c``) drawn as a sampled
  polyline with the vertex marked + the axis of symmetry.

Design contract (mirrors the deterministic emitters in this package —
``faq_page`` / ``key_terms``):

* **Deterministic** — same input string ⇒ byte-identical SVG. All coordinates are
  formatted with a fixed ``%.2f``; every label uses the same deterministic
  number formatter; no timestamps, no ``id()``, no dict-ordering surprises.
* **Fail closed** — an ambiguous / malformed / unsupported equation yields
  ``None`` (the caller emits its existing placeholder). A WRONG plot is worse
  than no plot, so the parser is conservative: multiple math fragments, more
  than one equation, extra free symbols, or degree > 2 all resolve to ``None``.
* **Accessible** — every emitted ``<svg>`` carries ``role="img"`` +
  ``aria-labelledby`` pointing at an in-SVG ``<title>`` (short) and ``<desc>``
  (full spatial long-description), per the project's WCAG 2.2 AA posture.

sympy is the equation parser (a core dependency; it powers the worked-example
math gate). The import is guarded exactly like
``lib/validators/worked_example_math.py`` so the module still imports — and
:func:`parse_plot_spec` still returns ``None`` (fail closed) — when sympy is
absent.
"""

from __future__ import annotations

import html as _html
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = [
    "PlotSpec",
    "parse_plot_spec",
    "render_plot_svg",
    "svg_for_equation",
]

# --------------------------------------------------------------------------- #
# sympy graceful-degrade guard (mirrors worked_example_math.py)
# --------------------------------------------------------------------------- #
try:
    import sympy  # type: ignore
    from sympy.parsing.sympy_parser import (  # type: ignore
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    _SYMPY_TRANSFORMS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    _X = sympy.Symbol("x")
    _Y = sympy.Symbol("y")
    _SYMPY_AVAILABLE = True
except Exception:  # noqa: BLE001 — any import failure degrades to fail-closed
    sympy = None  # type: ignore
    parse_expr = None  # type: ignore
    _SYMPY_TRANSFORMS = ()  # type: ignore
    _X = _Y = None  # type: ignore
    _SYMPY_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Plot spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlotSpec:
    """A resolved, renderable plot description (the parser's output).

    ``kind`` is one of ``number_line`` / ``line_graph`` / ``parabola``. Only the
    fields relevant to the kind are populated; the rest keep their defaults.
    """

    kind: str
    #: number_line
    boundary: float = 0.0
    closed: bool = False
    direction: str = "right"  # "right" (x >) | "left" (x <)
    variable: str = "x"
    #: line_graph
    slope: float = 0.0
    intercept: float = 0.0
    #: parabola
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    #: normalised source equation (for the caption / description)
    equation: str = ""
    #: extra warnings surfaced for callers/tests (never affects rendering)
    notes: Tuple[str, ...] = field(default=())


# --------------------------------------------------------------------------- #
# Equation parsing
# --------------------------------------------------------------------------- #
_RE_LATEX_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_RE_LATEX_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_RE_DOLLAR = re.compile(r"\$(.+?)\$", re.DOTALL)
#: A leading ``y =`` / ``f(x) =`` / ``g(x) =`` function declaration.
_RE_FUNC_DECL = re.compile(
    r"^\s*(?:y|[a-eg-wz]\s*\(\s*x\s*\)|f\s*\(\s*x\s*\))\s*=\s*(.+)$",
    re.IGNORECASE,
)
_RE_RELATION = re.compile(r"(<=|>=|<|>)")


def _extract_fragment(text: str) -> Tuple[Optional[str], bool]:
    """Return ``(fragment, ambiguous)`` — the single math fragment to parse.

    Prefers a lone LaTeX ``\\( … \\)`` / ``\\[ … \\]`` / ``$ … $`` fragment; when
    the text carries MORE than one such fragment the intent is ambiguous, so
    ``ambiguous=True`` (⇒ fail closed). With no delimiters the whole (stripped)
    text is the candidate.
    """
    if not isinstance(text, str) or not text.strip():
        return None, False
    fragments = (
        _RE_LATEX_INLINE.findall(text)
        + _RE_LATEX_DISPLAY.findall(text)
        + _RE_DOLLAR.findall(text)
    )
    if len(fragments) > 1:
        return None, True
    if len(fragments) == 1:
        return fragments[0].strip(), False
    return text.strip(), False


def _clean(s: str) -> str:
    """Deterministically normalise a restricted-LaTeX / ascii fragment.

    Maps LaTeX / unicode relation + operator tokens to ascii, unifies the minus
    glyphs, drops residual ``\\command`` tokens and braces. ``^`` powers are left
    for sympy's ``convert_xor`` transform.
    """
    s = s.strip()
    # Relations first (longest token first so \geq is not shadowed by \ge).
    for a, b in (
        (r"\geq", ">="), (r"\leq", "<="),
        (r"\ge", ">="), (r"\le", "<="),
        (r"\gt", ">"), (r"\lt", "<"),
        ("≥", ">="), ("≤", "<="),
    ):
        s = s.replace(a, b)
    s = s.replace(r"\cdot", "*").replace("·", "*")
    s = s.replace(r"\times", "*").replace("×", "*")
    s = s.replace(r"\div", "/").replace("÷", "/")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-").replace("‐", "-")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\ ", " ")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)  # residual commands (\text, \displaystyle)
    s = s.replace("{", "(").replace("}", ")")
    return s.strip()


def _to_float(value) -> Optional[float]:
    """Coerce a sympy number to a finite python float, else ``None``."""
    if not _SYMPY_AVAILABLE:
        return None
    try:
        if not value.is_number:  # symbolic residue ⇒ unsupported
            return None
        f = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(f):
        return None
    return f


def _parse_number_line(cleaned: str) -> Optional[PlotSpec]:
    """Parse a single linear inequality in one variable → a number-line spec."""
    if not _SYMPY_AVAILABLE:
        return None
    # Exactly ONE relation operator (a compound ``-2 < x < 3`` is ambiguous here).
    if len(_RE_RELATION.findall(cleaned)) != 1:
        return None
    try:
        rel = parse_expr(  # type: ignore[misc]
            cleaned, transformations=_SYMPY_TRANSFORMS, evaluate=True
        )
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(rel, sympy.core.relational.Relational):
        return None
    syms = rel.free_symbols
    if len(syms) != 1:
        return None
    var = next(iter(syms))
    # ``lhs OP rhs`` ⇔ ``f OP 0`` where ``f = lhs - rhs``; f must be linear
    # (``k·var + m``) in the single variable — else it isn't a ray.
    try:
        poly = sympy.Poly(rel.lhs - rel.rhs, var)
    except Exception:  # noqa: BLE001
        return None
    if poly.degree() != 1:
        return None
    k = _to_float(poly.coeff_monomial(var))
    m = _to_float(poly.coeff_monomial(1))
    if k is None or m is None or k == 0.0:
        return None
    # ``k·var + m OP 0`` ⇒ ``var (op) -m/k``; a negative leading coeff flips OP.
    bound = -m / k
    op = rel.rel_op
    if k < 0:
        op = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[op]
    direction = "right" if op in (">", ">=") else "left"
    closed = op in (">=", "<=")
    var_name = str(var)
    eq = f"{var_name} {op} {_num(bound)}"
    return PlotSpec(
        kind="number_line",
        boundary=bound,
        closed=closed,
        direction=direction,
        variable=var_name,
        equation=eq,
    )


def _poly_in_x(cleaned: str) -> Optional["sympy.Expr"]:
    """Reduce an equality / bare expression to ``y = f(x)`` and return ``f(x)``.

    Handles ``y = …`` / ``f(x) = …`` declarations, ``Ax + By = C`` standard form
    (solved for ``y``), and a bare RHS expression. Returns a sympy expression in
    ``x`` only, or ``None`` when unsupported (extra symbols, no solution, …).
    """
    if not _SYMPY_AVAILABLE:
        return None
    m = _RE_FUNC_DECL.match(cleaned)
    if m:  # y = …  /  f(x) = …
        rhs = m.group(1)
        try:
            expr = parse_expr(
                rhs, transformations=_SYMPY_TRANSFORMS, evaluate=True
            )
        except Exception:  # noqa: BLE001
            return None
        if expr.free_symbols <= {_X}:
            return expr
        return None
    if "=" in cleaned:  # standard form Ax + By = C
        parts = cleaned.split("=")
        if len(parts) != 2:  # more than one '=' ⇒ ambiguous
            return None
        try:
            lhs = parse_expr(
                parts[0], transformations=_SYMPY_TRANSFORMS, evaluate=True
            )
            rhs = parse_expr(
                parts[1], transformations=_SYMPY_TRANSFORMS, evaluate=True
            )
        except Exception:  # noqa: BLE001
            return None
        combined = (lhs - rhs).free_symbols
        if not combined <= {_X, _Y} or _Y not in combined:
            return None
        try:
            sols = sympy.solve(sympy.Eq(lhs, rhs), _Y)
        except Exception:  # noqa: BLE001
            return None
        if len(sols) != 1 or not sols[0].free_symbols <= {_X}:
            return None
        return sols[0]
    # Bare expression f(x).
    try:
        expr = parse_expr(
            cleaned, transformations=_SYMPY_TRANSFORMS, evaluate=True
        )
    except Exception:  # noqa: BLE001
        return None
    if expr.free_symbols <= {_X}:
        return expr
    return None


def _parse_curve(cleaned: str) -> Optional[PlotSpec]:
    """Parse an equality / expression → a ``line_graph`` or ``parabola`` spec."""
    expr = _poly_in_x(cleaned)
    if expr is None:
        return None
    try:
        poly = sympy.Poly(expr, _X)
    except Exception:  # noqa: BLE001
        return None
    degree = poly.degree()
    if degree <= 1:
        slope = _to_float(expr.coeff(_X, 1))
        intercept = _to_float(expr.coeff(_X, 0))
        if slope is None or intercept is None:
            return None
        return PlotSpec(
            kind="line_graph",
            slope=slope,
            intercept=intercept,
            equation=f"y = {_expr_label(slope, intercept)}",
        )
    if degree == 2:
        a = _to_float(expr.coeff(_X, 2))
        b = _to_float(expr.coeff(_X, 1))
        c = _to_float(expr.coeff(_X, 0))
        if a is None or b is None or c is None or a == 0.0:
            return None
        return PlotSpec(
            kind="parabola",
            a=a,
            b=b,
            c=c,
            equation=f"y = {_quad_label(a, b, c)}",
        )
    return None  # degree > 2 unsupported ⇒ fail closed


def parse_plot_spec(text: str) -> Optional[PlotSpec]:
    """Parse ``text`` into a :class:`PlotSpec`, or ``None`` (fail closed).

    Accepts LaTeX-delimited (``\\( … \\)`` / ``\\[ … \\]`` / ``$ … $``) or plain
    equation text. Ambiguous input (multiple math fragments, more than one
    equation, extra free symbols, degree > 2, non-numeric coefficients) ⇒
    ``None`` — never a wrong plot.
    """
    fragment, ambiguous = _extract_fragment(text)
    if ambiguous or not fragment:
        return None
    cleaned = _clean(fragment)
    if not cleaned or not re.search(r"[0-9a-zA-Z]", cleaned):
        return None
    if _RE_RELATION.search(cleaned):
        return _parse_number_line(cleaned)
    return _parse_curve(cleaned)


# --------------------------------------------------------------------------- #
# Number / label formatting (deterministic)
# --------------------------------------------------------------------------- #
def _num(v: float) -> str:
    """Deterministic human label for a number (int when integral, else 2dp)."""
    v = float(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _px(v: float) -> str:
    """Deterministic SVG coordinate string — always fixed 2-decimal."""
    return f"{float(v):.2f}"


def _term(coeff: float, sym: str) -> str:
    """Format a ``coeff·sym`` term (``1x`` → ``x``, ``-1x`` → ``-x``)."""
    if abs(coeff - 1.0) < 1e-9:
        return sym
    if abs(coeff + 1.0) < 1e-9:
        return f"-{sym}"
    return f"{_num(coeff)}{sym}"


def _expr_label(slope: float, intercept: float) -> str:
    """``y = …`` right-hand-side label for a line (``2x + 3``, ``-x``, ``5``)."""
    if abs(slope) < 1e-12:
        return _num(intercept)
    out = _term(slope, "x")
    if abs(intercept) >= 1e-12:
        out += f" + {_num(intercept)}" if intercept > 0 else f" - {_num(-intercept)}"
    return out


def _quad_label(a: float, b: float, c: float) -> str:
    """``y = …`` right-hand-side label for a quadratic."""
    out = _term(a, "x^2")
    if abs(b) >= 1e-12:
        out += f" + {_term(b, 'x')}" if b > 0 else f" - {_term(-b, 'x')}"
    if abs(c) >= 1e-12:
        out += f" + {_num(c)}" if c > 0 else f" - {_num(-c)}"
    return out


# --------------------------------------------------------------------------- #
# SVG scaffolding
# --------------------------------------------------------------------------- #
# House palette (Courseforge CSS foundation — primary blue / ink / grid gray).
_C_AXIS = "#495057"
_C_GRID = "#dee2e6"
_C_PLOT = "#2c5aa0"
_C_MARK = "#dc3545"
_C_INK = "#212529"


def _svg_header(
    width: int, height: int, id_prefix: str, title: str, desc: str, css_class: str
) -> List[str]:
    tid = f"{id_prefix}-title"
    did = f"{id_prefix}-desc"
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'class="{css_class}" role="img" aria-labelledby="{tid} {did}" '
        f'width="100%" preserveAspectRatio="xMidYMid meet">',
        f'  <title id="{tid}">{_html.escape(title)}</title>',
        f'  <desc id="{did}">{_html.escape(desc)}</desc>',
    ]


def _text(x: float, y: float, s: str, *, anchor: str = "middle",
          size: int = 12, color: str = _C_INK) -> str:
    return (
        f'  <text x="{_px(x)}" y="{_px(y)}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{color}" '
        f'font-family="sans-serif">{_html.escape(s)}</text>'
    )


# --------------------------------------------------------------------------- #
# number_line
# --------------------------------------------------------------------------- #
_NL_W, _NL_H = 480, 120
_NL_AXIS_Y = 64
_NL_PAD = 36


def _render_number_line(spec: PlotSpec, id_prefix: str) -> str:
    b = spec.boundary
    lo = math.floor(b) - 5
    hi = math.ceil(b) + 5
    if hi - lo < 2:  # degenerate guard
        lo, hi = lo - 1, hi + 1
    span = hi - lo
    x0, x1 = _NL_PAD, _NL_W - _NL_PAD

    def sx(xv: float) -> float:
        return x0 + (xv - lo) / span * (x1 - x0)

    var = spec.variable
    rel = ">" if spec.direction == "right" else "<"
    rel += "=" if spec.closed else ""
    title = f"Number line: {var} {rel} {_num(b)}"
    circ = "closed" if spec.closed else "open"
    side = "right" if spec.direction == "right" else "left"
    desc = (
        f"A number line from {lo} to {hi}. A {circ} circle at {_num(b)} marks the "
        f"boundary, and the ray is shaded to the {side}, representing all values "
        f"where {var} {rel} {_num(b)}."
    )
    marker = f"{id_prefix}-arrow"
    parts = _svg_header(_NL_W, _NL_H, id_prefix, title, desc, "svg-number-line")
    parts.append(
        f'  <defs><marker id="{marker}" markerWidth="10" markerHeight="10" '
        f'refX="6" refY="3" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{_C_PLOT}"/></marker></defs>'
    )
    # Base axis.
    parts.append(
        f'  <line x1="{_px(x0)}" y1="{_px(_NL_AXIS_Y)}" x2="{_px(x1)}" '
        f'y2="{_px(_NL_AXIS_Y)}" stroke="{_C_AXIS}" stroke-width="1.5"/>'
    )
    # Ticks + labels.
    for t in range(lo, hi + 1):
        tx = sx(t)
        parts.append(
            f'  <line x1="{_px(tx)}" y1="{_px(_NL_AXIS_Y - 5)}" x2="{_px(tx)}" '
            f'y2="{_px(_NL_AXIS_Y + 5)}" stroke="{_C_AXIS}" stroke-width="1"/>'
        )
        parts.append(_text(tx, _NL_AXIS_Y + 20, _num(t), size=11, color=_C_AXIS))
    # Shaded ray with an arrowhead toward the open end.
    bx = sx(b)
    end = x1 if spec.direction == "right" else x0
    parts.append(
        f'  <line x1="{_px(bx)}" y1="{_px(_NL_AXIS_Y)}" x2="{_px(end)}" '
        f'y2="{_px(_NL_AXIS_Y)}" stroke="{_C_PLOT}" stroke-width="4" '
        f'marker-end="url(#{marker})"/>'
    )
    # Boundary circle (filled = closed, white = open).
    fill = _C_PLOT if spec.closed else "#ffffff"
    parts.append(
        f'  <circle cx="{_px(bx)}" cy="{_px(_NL_AXIS_Y)}" r="6" fill="{fill}" '
        f'stroke="{_C_PLOT}" stroke-width="2.5"/>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Cartesian scaffolding (line_graph + parabola)
# --------------------------------------------------------------------------- #
_CW, _CH = 480, 360
_C_ML, _C_MR, _C_MT, _C_MB = 40, 20, 20, 30


def _cart_mapper(xmin, xmax, ymin, ymax):
    pw = _CW - _C_ML - _C_MR
    ph = _CH - _C_MT - _C_MB

    def sx(xv: float) -> float:
        return _C_ML + (xv - xmin) / (xmax - xmin) * pw

    def sy(yv: float) -> float:
        return _C_MT + (ymax - yv) / (ymax - ymin) * ph

    return sx, sy


def _cart_frame(xmin, xmax, ymin, ymax, sx, sy) -> List[str]:
    """Gridlines + axes + tick labels for a Cartesian window."""
    out: List[str] = []
    # Gridlines (every integer, light).
    for gx in range(math.ceil(xmin), math.floor(xmax) + 1):
        out.append(
            f'  <line x1="{_px(sx(gx))}" y1="{_px(sy(ymax))}" '
            f'x2="{_px(sx(gx))}" y2="{_px(sy(ymin))}" stroke="{_C_GRID}" '
            f'stroke-width="1"/>'
        )
    for gy in range(math.ceil(ymin), math.floor(ymax) + 1):
        out.append(
            f'  <line x1="{_px(sx(xmin))}" y1="{_px(sy(gy))}" '
            f'x2="{_px(sx(xmax))}" y2="{_px(sy(gy))}" stroke="{_C_GRID}" '
            f'stroke-width="1"/>'
        )
    # Axes (x=0 / y=0 when in-window, else the frame edge).
    axis_x = 0.0 if xmin <= 0 <= xmax else xmin
    axis_y = 0.0 if ymin <= 0 <= ymax else ymin
    out.append(
        f'  <line x1="{_px(sx(xmin))}" y1="{_px(sy(axis_y))}" '
        f'x2="{_px(sx(xmax))}" y2="{_px(sy(axis_y))}" stroke="{_C_AXIS}" '
        f'stroke-width="1.5"/>'
    )
    out.append(
        f'  <line x1="{_px(sx(axis_x))}" y1="{_px(sy(ymin))}" '
        f'x2="{_px(sx(axis_x))}" y2="{_px(sy(ymax))}" stroke="{_C_AXIS}" '
        f'stroke-width="1.5"/>'
    )
    # A couple of endpoint tick labels so the scale is legible.
    out.append(_text(sx(xmax) - 4, sy(axis_y) - 4, _num(xmax),
                     anchor="end", size=10, color=_C_AXIS))
    out.append(_text(sx(axis_x) - 4, sy(ymax) + 10, _num(ymax),
                     anchor="end", size=10, color=_C_AXIS))
    return out


def _clip_polylines(pts: List[Tuple[float, float]], ymin, ymax, sx, sy) -> List[str]:
    """Emit one ``<polyline>`` per run of consecutive in-window sample points."""
    out: List[str] = []
    run: List[str] = []

    def flush():
        if len(run) >= 2:
            out.append(
                f'  <polyline points="{" ".join(run)}" fill="none" '
                f'stroke="{_C_PLOT}" stroke-width="2.5"/>'
            )
        run.clear()

    for xv, yv in pts:
        if ymin <= yv <= ymax:
            run.append(f"{_px(sx(xv))},{_px(sy(yv))}")
        else:
            flush()
    flush()
    return out


def _dot(sx, sy, xv, yv, label) -> List[str]:
    return [
        f'  <circle cx="{_px(sx(xv))}" cy="{_px(sy(yv))}" r="4" '
        f'fill="{_C_MARK}"/>',
        _text(sx(xv), sy(yv) - 8, label, size=11, color=_C_MARK),
    ]


# --------------------------------------------------------------------------- #
# line_graph
# --------------------------------------------------------------------------- #
def _render_line_graph(spec: PlotSpec, id_prefix: str) -> str:
    m, b = spec.slope, spec.intercept
    xmin, xmax, ymin, ymax = -10.0, 10.0, -10.0, 10.0
    sx, sy = _cart_mapper(xmin, xmax, ymin, ymax)
    title = f"Line graph: {spec.equation}"
    xint = None if abs(m) < 1e-12 else -b / m
    xint_txt = (
        f" and crosses the x-axis at {_num(xint)}"
        if xint is not None and xmin <= xint <= xmax
        else ""
    )
    desc = (
        f"A coordinate grid from {int(xmin)} to {int(xmax)} on both axes showing "
        f"the line {spec.equation}. It has slope {_num(m)} and crosses the y-axis "
        f"at {_num(b)}{xint_txt}."
    )
    parts = _svg_header(_CW, _CH, id_prefix, title, desc, "svg-line-graph")
    parts += _cart_frame(xmin, xmax, ymin, ymax, sx, sy)
    # Sample the line (straight, but sampling gives uniform clip handling).
    n = 200
    pts = [
        (xmin + i * (xmax - xmin) / n, m * (xmin + i * (xmax - xmin) / n) + b)
        for i in range(n + 1)
    ]
    parts += _clip_polylines(pts, ymin, ymax, sx, sy)
    # Intercept markers.
    if ymin <= b <= ymax:
        parts += _dot(sx, sy, 0.0, b, f"(0, {_num(b)})")
    if xint is not None and xmin <= xint <= xmax and abs(xint) > 1e-9:
        parts += _dot(sx, sy, xint, 0.0, f"({_num(xint)}, 0)")
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# parabola
# --------------------------------------------------------------------------- #
def _render_parabola(spec: PlotSpec, id_prefix: str) -> str:
    a, b, c = spec.a, spec.b, spec.c
    vx = -b / (2 * a)
    vy = c - b * b / (4 * a)
    xmin = math.floor(vx - 6)
    xmax = math.ceil(vx + 6)
    # y-window from sampled extremes, padded + integer-snapped, vertex included.
    edge = a * xmin * xmin + b * xmin + c
    edge2 = a * xmax * xmax + b * xmax + c
    ylo = min(vy, edge, edge2)
    yhi = max(vy, edge, edge2)
    pad = max(1.0, (yhi - ylo) * 0.1)
    ymin = math.floor(ylo - pad)
    ymax = math.ceil(yhi + pad)
    if ymax - ymin < 4:
        ymin, ymax = ymin - 2, ymax + 2
    sx, sy = _cart_mapper(float(xmin), float(xmax), float(ymin), float(ymax))
    opening = "upward" if a > 0 else "downward"
    title = f"Parabola: {spec.equation}"
    desc = (
        f"A coordinate grid showing the parabola {spec.equation}, opening "
        f"{opening}. Its vertex is at ({_num(vx)}, {_num(vy)}) and its axis of "
        f"symmetry is the vertical line x = {_num(vx)}."
    )
    parts = _svg_header(_CW, _CH, id_prefix, title, desc, "svg-parabola")
    parts += _cart_frame(float(xmin), float(xmax), float(ymin), float(ymax), sx, sy)
    # Axis of symmetry (dashed).
    if xmin <= vx <= xmax:
        parts.append(
            f'  <line x1="{_px(sx(vx))}" y1="{_px(sy(ymin))}" '
            f'x2="{_px(sx(vx))}" y2="{_px(sy(ymax))}" stroke="{_C_MARK}" '
            f'stroke-width="1" stroke-dasharray="4 3"/>'
        )
    # Sampled curve.
    n = 240
    pts = [
        (
            xmin + i * (xmax - xmin) / n,
            a * (xmin + i * (xmax - xmin) / n) ** 2
            + b * (xmin + i * (xmax - xmin) / n)
            + c,
        )
        for i in range(n + 1)
    ]
    parts += _clip_polylines(pts, float(ymin), float(ymax), sx, sy)
    # Vertex marker.
    if ymin <= vy <= ymax:
        parts += _dot(sx, sy, vx, vy, f"vertex ({_num(vx)}, {_num(vy)})")
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Public render entry points
# --------------------------------------------------------------------------- #
_RENDERERS = {
    "number_line": _render_number_line,
    "line_graph": _render_line_graph,
    "parabola": _render_parabola,
}


def render_plot_svg(spec: PlotSpec, *, id_prefix: str = "plot") -> Optional[str]:
    """Render a :class:`PlotSpec` to a deterministic, accessible SVG string.

    ``id_prefix`` seeds the in-SVG ``<title>`` / ``<desc>`` ids (and the
    number-line arrow marker id) so multiple plots on one page keep unique ids;
    pass the block id for page uniqueness. Returns ``None`` for an unknown kind.
    """
    renderer = _RENDERERS.get(spec.kind)
    if renderer is None:
        return None
    return renderer(spec, id_prefix)


def svg_for_equation(
    text: str, *, id_prefix: str = "plot"
) -> Optional[Tuple[str, str]]:
    """Parse ``text`` and render it, returning ``(svg, alt_text)`` or ``None``.

    Convenience one-shot for the ``diagram``-block wire-in: fails closed (returns
    ``None``) whenever :func:`parse_plot_spec` cannot unambiguously resolve a
    supported plot. ``alt_text`` is a concise caption-style description
    (distinct from the richer in-SVG ``<desc>``).
    """
    spec = parse_plot_spec(text)
    if spec is None:
        return None
    svg = render_plot_svg(spec, id_prefix=id_prefix)
    if svg is None:
        return None
    alt = _alt_text(spec)
    return svg, alt


def _alt_text(spec: PlotSpec) -> str:
    """Short caption-style alt text for the figure (fed to the caller)."""
    if spec.kind == "number_line":
        rel = (">" if spec.direction == "right" else "<") + ("=" if spec.closed else "")
        return f"Number line of the solution {spec.variable} {rel} {_num(spec.boundary)}."
    if spec.kind == "line_graph":
        return f"Line graph of {spec.equation}."
    if spec.kind == "parabola":
        vx = -spec.b / (2 * spec.a)
        vy = spec.c - spec.b * spec.b / (4 * spec.a)
        return (
            f"Parabola of {spec.equation} with vertex ({_num(vx)}, {_num(vy)})."
        )
    return "Math diagram."
