r"""Constrained TikZ → inline-SVG re-draw (real-figure wave — 2026-07).

The round-10 :func:`lib.semantik.math_fold.strip_tikz_figures` pass substitutes an
accessible ``.dart-figure-notation`` PLACEHOLDER for the coordinate-plane FIGURES
the VLM transcribed as raw TikZ picture code inside math delimiters (MathJax reds
them "Undefined environment tikzpicture"). This module goes one step further: it
DETERMINISTICALLY re-draws the narrow shape grammar the corpus actually emits
(grid / help-lines + ``\draw`` segment paths with thick/dashed/dotted/``->``
styles + ``\filldraw``/``\fill`` circle & rectangle + ``\node`` text labels +
``[scale=…]``) as a pure-string inline ``<svg role="img">``. Anything outside the
grammar — pgfplots ``axis`` / ``\addplot`` / ``plot coordinates`` / the ch07
``\includegraphics`` hallucination / any unrecognised option — makes
:func:`parse_tikz` return ``None`` so the caller falls back to the existing
placeholder (parse-with-fallback, never a hard failure, never a wrong figure).

No LaTeX toolchain, no model, no GPU: a hand-written recursive-descent-ish parser
over the observed grammar plus a string SVG emitter. HTML-only by contract (the
caller only rewrites rendered block HTML; ``raw_text`` / sidecar / chunk text keep
the plain fused TikZ for the chunker + retrieval). See
``plans/tikz-real-figures-2026-07.md`` for the feasibility scoping.
"""
from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Layout constants (fixed so the emitted SVG is byte-deterministic).
# ---------------------------------------------------------------------------
_UNIT = 30  # px per TikZ coordinate unit
_PAD = 12   # px viewBox padding
_FONT = 13  # px node-label font size
_ARROW_MARKER_ID = "dart-tikz-arrow"

# Recognised colour names (a bare colour token or ``color=/fill=/draw=`` value).
_COLORS = {
    "black", "gray", "grey", "darkgray", "lightgray", "white",
    "red", "green", "blue", "cyan", "magenta", "yellow", "orange",
    "purple", "violet", "brown", "pink", "teal", "olive", "lime",
}
# Draw-option tokens accepted but style-neutral (no effect on the emitted SVG).
_NEUTRAL_DRAW_OPTS = {
    "thin", "very thin", "ultra thin", "solid", "help lines",
    "densely dashed", "loosely dashed", "densely dotted", "loosely dotted",
}


# ---------------------------------------------------------------------------
# Figure element model.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Grid:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class _Path:
    points: Tuple[Tuple[float, float], ...]
    arrow: bool = False
    dashed: bool = False
    dotted: bool = False
    thick: bool = False
    color: Optional[str] = None


@dataclass(frozen=True)
class _Circle:
    cx: float
    cy: float
    r: float
    r_is_pt: bool
    filled: bool
    color: Optional[str] = None


@dataclass(frozen=True)
class _Rect:
    x0: float
    y0: float
    x1: float
    y1: float
    filled: bool
    color: Optional[str] = None


@dataclass(frozen=True)
class _Node:
    x: float
    y: float
    text: str


@dataclass
class FigureSpec:
    """Parsed, render-ready description of a constrained TikZ picture."""

    scale: float = 1.0
    elements: List[object] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grammar fragments.
# ---------------------------------------------------------------------------
_NUM = r"-?\d+(?:\.\d+)?"
_COORD = r"\(\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\)"
_COORD_RE = re.compile(_COORD)
# An inline ``node[opts]{text}`` attached to a path coordinate.
_INLINE_NODE_RE = re.compile(r"node\b\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")

_ENV_RE = re.compile(
    r"\\begin\s*\{\s*tikzpicture\s*\}\s*(?:\[(?P<opts>[^\]]*)\])?"
    r"(?P<body>.*?)"
    r"\\end\s*\{\s*tikzpicture\s*\}",
    re.DOTALL,
)
_SCALE_RE = re.compile(r"\bscale\s*=\s*(" + _NUM + r")")

# Statement dispatch.
_STMT_DRAW_RE = re.compile(
    r"^\\(draw|filldraw|fill)\b\s*(?:\[(?P<opts>[^\]]*)\])?\s*(?P<rest>.*)$", re.DOTALL
)
_STMT_NODE_RE = re.compile(
    r"^\\node\b\s*(?:\[[^\]]*\])?\s*at\s*" + _COORD + r"\s*\{(?P<txt>.*?)\}\s*$",
    re.DOTALL,
)
_GRID_RE = re.compile(r"^" + _COORD + r"\s*grid\s*" + _COORD + r"$")
_CIRCLE_RE = re.compile(
    r"^" + _COORD + r"\s*circle\s*\(\s*(" + _NUM + r")\s*(pt)?\s*\)$"
)
_RECT_RE = re.compile(r"^" + _COORD + r"\s*rectangle\s*" + _COORD + r"$")


def _norm_color(tok: str) -> Optional[str]:
    tok = tok.strip().lower()
    if tok == "grey":
        tok = "gray"
    return tok if tok in _COLORS else None


def _parse_draw_opts(opts: str) -> Optional[dict]:
    r"""Parse a ``\draw``/``\fill`` option list into a style dict, or ``None``.

    Conservative: an option token outside the recognised whitelist (a colour, a
    style keyword, an arrow tip, a neutral thickness, or a ``scale=``) fails the
    whole parse so the caller falls back to the placeholder rather than dropping
    an unsupported feature silently.
    """
    style = {"arrow": False, "dashed": False, "dotted": False,
             "thick": False, "color": None}
    if not opts:
        return style
    for raw in opts.split(","):
        tok = raw.strip()
        if not tok:
            continue
        low = tok.lower()
        if low in {"thick", "very thick", "semithick", "ultra thick"}:
            style["thick"] = True
        elif low == "dashed":
            style["dashed"] = True
        elif low == "dotted":
            style["dotted"] = True
        elif low in {"->", "<-", "<->", "-&gt;", "&lt;-", "&lt;->"}:
            style["arrow"] = True
        elif low in _NEUTRAL_DRAW_OPTS:
            continue
        elif low.startswith(("color=", "fill=", "draw=")):
            col = _norm_color(low.split("=", 1)[1])
            if col is None:
                return None
            style["color"] = col
        elif low.startswith("scale="):
            continue
        elif _norm_color(low) is not None:
            style["color"] = _norm_color(low)
        else:
            return None
    return style


def _parse_path(rest: str, style: dict) -> Optional[_Path]:
    """A ``coord -- coord [-- coord…]`` polyline with optional inline nodes."""
    coords = [(float(m.group(1)), float(m.group(2))) for m in _COORD_RE.finditer(rest)]
    if len(coords) < 2:
        return None
    # Purity: after removing coords, inline nodes, ``--`` joiners and whitespace,
    # nothing may remain (rejects ``plot coordinates {…}``, curves, etc.).
    residue = _COORD_RE.sub(" ", rest)
    residue = _INLINE_NODE_RE.sub(" ", residue)
    residue = residue.replace("--", " ")
    if re.sub(r"\s+", "", residue):
        return None
    return _Path(
        points=tuple(coords),
        arrow=style["arrow"],
        dashed=style["dashed"],
        dotted=style["dotted"],
        thick=style["thick"],
        color=style["color"],
    )


def _inline_nodes(rest: str) -> List[_Node]:
    """Extract ``node[..]{txt}`` labels attached to their preceding coordinate."""
    coords = list(_COORD_RE.finditer(rest))
    out: List[_Node] = []
    for nm in _INLINE_NODE_RE.finditer(rest):
        anchor = None
        for cm in coords:
            if cm.start() < nm.start():
                anchor = cm
            else:
                break
        if anchor is None:
            continue
        txt = _label_text(nm.group(1))
        if txt:
            out.append(_Node(float(anchor.group(1)), float(anchor.group(2)), txt))
    return out


def _label_text(raw: str) -> str:
    r"""Render a node label body as plain text (``$x$`` math → ``x``)."""
    txt = raw.strip()
    # Inline ``$…$`` math renders as its plain content (drop the delimiters).
    txt = re.sub(r"\$(.*?)\$", r"\1", txt)
    txt = txt.replace("$", "")
    return re.sub(r"\s+", " ", txt).strip()


def _parse_statement(stmt: str, spec: FigureSpec) -> bool:
    """Parse one ``;``-terminated statement into ``spec.elements``; False on miss."""
    stmt = stmt.strip()
    if not stmt:
        return True  # empty (trailing ``;``) — benign

    nm = _STMT_NODE_RE.match(stmt)
    if nm:
        txt = _label_text(nm.group(3))
        if txt:
            spec.elements.append(_Node(float(nm.group(1)), float(nm.group(2)), txt))
        return True

    dm = _STMT_DRAW_RE.match(stmt)
    if not dm:
        return False
    cmd = dm.group(1)
    style = _parse_draw_opts(dm.group("opts") or "")
    if style is None:
        return False
    rest = dm.group("rest").strip()
    filled = cmd in {"fill", "filldraw"}

    gm = _GRID_RE.match(rest)
    if gm:
        spec.elements.append(_Grid(*(float(gm.group(i)) for i in range(1, 5))))
        return True

    cm = _CIRCLE_RE.match(rest)
    if cm:
        spec.elements.append(
            _Circle(
                cx=float(cm.group(1)), cy=float(cm.group(2)),
                r=float(cm.group(3)), r_is_pt=bool(cm.group(4)),
                filled=filled, color=style["color"],
            )
        )
        return True

    rm = _RECT_RE.match(rest)
    if rm:
        spec.elements.append(
            _Rect(*(float(rm.group(i)) for i in range(1, 5)),
                  filled=filled, color=style["color"])
        )
        return True

    path = _parse_path(rest, style)
    if path is None:
        return False
    spec.elements.append(path)
    spec.elements.extend(_inline_nodes(rest))
    return True


def parse_tikz(notation: str) -> Optional[FigureSpec]:
    r"""Parse VLM-emitted TikZ ``notation`` into a :class:`FigureSpec`, or ``None``.

    ``notation`` is the content of a math span (HTML-escaped as it arrives inside
    block HTML — ``->`` reads as ``-&gt;``); it is unescaped first. Only a single
    ``\begin{tikzpicture}…\end{tikzpicture}`` environment made ENTIRELY of the
    supported statements (grid / segment-path / circle / rectangle / node) parses;
    anything else (pgfplots ``axis``, ``\addplot``, ``plot coordinates``,
    ``\includegraphics``, an unknown option, an empty picture) returns ``None``.
    """
    if not notation:
        return None
    text = html.unescape(notation)
    env = _ENV_RE.search(text)
    if not env:
        return None
    spec = FigureSpec()
    sm = _SCALE_RE.search(env.group("opts") or "")
    if sm:
        try:
            spec.scale = float(sm.group(1))
        except ValueError:
            spec.scale = 1.0
    body = env.group("body")
    for stmt in body.split(";"):
        if not _parse_statement(stmt, spec):
            return None
    if not spec.elements:
        return None
    return spec


# ---------------------------------------------------------------------------
# SVG emitter.
# ---------------------------------------------------------------------------
def _extents(spec: FigureSpec) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for el in spec.elements:
        if isinstance(el, _Grid):
            xs += [el.x0, el.x1]
            ys += [el.y0, el.y1]
        elif isinstance(el, _Path):
            xs += [p[0] for p in el.points]
            ys += [p[1] for p in el.points]
        elif isinstance(el, _Circle):
            rr = 0.0 if el.r_is_pt else el.r
            xs += [el.cx - rr, el.cx + rr]
            ys += [el.cy - rr, el.cy + rr]
        elif isinstance(el, _Rect):
            xs += [el.x0, el.x1]
            ys += [el.y0, el.y1]
        elif isinstance(el, _Node):
            xs.append(el.x)
            ys.append(el.y)
    if not xs:
        xs = [0.0, 1.0]
    if not ys:
        ys = [0.0, 1.0]
    return min(xs), min(ys), max(xs), max(ys)


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


def _accessible_name(spec: FigureSpec) -> str:
    labels = [el.text for el in spec.elements if isinstance(el, _Node) and el.text]
    name = "Coordinate-plane figure"
    if labels:
        name += " with labels: " + ", ".join(labels)
    return name


def render_svg(spec: FigureSpec) -> str:
    """Render a :class:`FigureSpec` as a self-contained inline ``<svg>`` string."""
    minx, miny, maxx, maxy = _extents(spec)
    if maxx <= minx:
        maxx = minx + 1
    if maxy <= miny:
        maxy = miny + 1
    width = round((maxx - minx) * _UNIT) + 2 * _PAD
    height = round((maxy - miny) * _UNIT) + 2 * _PAD

    def sx(x: float) -> int:
        return round((x - minx) * _UNIT) + _PAD

    def sy(y: float) -> int:
        return round((maxy - y) * _UNIT) + _PAD

    name = _accessible_name(spec)
    esc_name = _xml_escape(name)
    needs_arrow = any(isinstance(el, _Path) and el.arrow for el in spec.elements)

    parts: List[str] = [
        f'<svg class="dart-figure-svg" role="img" aria-label="{esc_name}" '
        f'viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        f"<title>{esc_name}</title>",
    ]
    if needs_arrow:
        parts.append(
            f'<defs><marker id="{_ARROW_MARKER_ID}" markerWidth="8" '
            'markerHeight="8" refX="7" refY="3" orient="auto" '
            'markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" '
            'fill="currentColor"/></marker></defs>'
        )

    for el in spec.elements:
        if isinstance(el, _Grid):
            parts.append(_render_grid(el, sx, sy))
        elif isinstance(el, _Path):
            parts.append(_render_path(el, sx, sy))
        elif isinstance(el, _Circle):
            parts.append(_render_circle(el, sx, sy))
        elif isinstance(el, _Rect):
            parts.append(_render_rect(el, sx, sy))
        elif isinstance(el, _Node):
            parts.append(_render_node(el, sx, sy))

    parts.append("</svg>")
    return "".join(parts)


def _render_grid(el: _Grid, sx, sy) -> str:
    x0, x1 = sorted((el.x0, el.x1))
    y0, y1 = sorted((el.y0, el.y1))
    lines: List[str] = ['<g class="dart-figure-grid" stroke="currentColor" '
                        'stroke-width="0.5" opacity="0.35">']
    gx = math.ceil(x0)
    while gx <= x1 + 1e-9:
        lines.append(f'<line x1="{sx(gx)}" y1="{sy(y0)}" x2="{sx(gx)}" y2="{sy(y1)}"/>')
        gx += 1
    gy = math.ceil(y0)
    while gy <= y1 + 1e-9:
        lines.append(f'<line x1="{sx(x0)}" y1="{sy(gy)}" x2="{sx(x1)}" y2="{sy(gy)}"/>')
        gy += 1
    lines.append("</g>")
    return "".join(lines)


def _stroke_attrs(color: Optional[str], thick: bool, dashed: bool, dotted: bool) -> str:
    attrs = [f'stroke="{color or "currentColor"}"',
             f'stroke-width="{2 if thick else 1}"', 'fill="none"']
    if dashed:
        attrs.append('stroke-dasharray="6 4"')
    elif dotted:
        attrs.append('stroke-dasharray="1 4"')
        attrs.append('stroke-linecap="round"')
    return " ".join(attrs)


def _render_path(el: _Path, sx, sy) -> str:
    pts = " ".join(f"{sx(x)},{sy(y)}" for x, y in el.points)
    attrs = _stroke_attrs(el.color, el.thick, el.dashed, el.dotted)
    marker = f' marker-end="url(#{_ARROW_MARKER_ID})"' if el.arrow else ""
    return f'<polyline points="{pts}" {attrs}{marker}/>'


def _render_circle(el: _Circle, sx, sy) -> str:
    r_px = max(2, round(el.r)) if el.r_is_pt else round(el.r * _UNIT)
    color = el.color or "currentColor"
    if el.filled:
        fill = f'fill="{color}"'
        stroke = ""
    else:
        fill = 'fill="none"'
        stroke = f' stroke="{color}" stroke-width="1"'
    return f'<circle cx="{sx(el.cx)}" cy="{sy(el.cy)}" r="{r_px}" {fill}{stroke}/>'


def _render_rect(el: _Rect, sx, sy) -> str:
    x = min(sx(el.x0), sx(el.x1))
    y = min(sy(el.y0), sy(el.y1))
    w = abs(sx(el.x1) - sx(el.x0))
    h = abs(sy(el.y1) - sy(el.y0))
    color = el.color or "currentColor"
    if el.filled:
        paint = f'fill="{color}"'
    else:
        paint = f'fill="none" stroke="{color}" stroke-width="1"'
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" {paint}/>'


def _render_node(el: _Node, sx, sy) -> str:
    return (
        f'<text x="{sx(el.x)}" y="{sy(el.y)}" font-size="{_FONT}" '
        f'text-anchor="middle" dominant-baseline="middle" '
        f'fill="currentColor">{_xml_escape(el.text)}</text>'
    )


# ---------------------------------------------------------------------------
# Span-level re-render pass (mirrors math_fold.strip_tikz_figures span walk).
# ---------------------------------------------------------------------------
def render_tikz_figures(text: str) -> str:
    r"""Re-draw parseable TikZ figure spans as inline SVG; else defer to the strip.

    Walks the same delimited math spans as
    :func:`lib.semantik.math_fold.strip_tikz_figures`. For a PURE-figure span
    (stripping the env leaves nothing) whose notation parses, the whole span is
    replaced by ``<figure class="dart-figure">…<svg…></figure>`` (delimiters +
    TikZ source gone). A pure-figure span that does NOT parse is left UNTOUCHED so
    the downstream strip pass emits the placeholder. A MIXED span (real math +
    embedded figure) keeps its surviving math and gains the accessible placeholder
    for the figure — closing the round-10 silent-drop gap (mixed spans previously
    vanished without a trace). Idempotent (emitted markup carries no
    ``\begin{tikz…}``); a fast guard returns text with no TikZ begin unchanged.
    """
    # Imported lazily to avoid any import cycle and to reuse the exact env/strip
    # regexes + placeholder the round-10 pass owns.
    from lib.semantik.math_fold import (
        _MATH_SPAN_ANGLE_RE,
        _TIKZ_ENV_RE,
        _TIKZ_FIGURE_PLACEHOLDER,
        _TIKZ_ORPHAN_RE,
        _TIKZ_PRESENT_RE,
        _split_math_delims,
    )

    if not text or not _TIKZ_PRESENT_RE.search(text):
        return text or ""

    def _fix(m: "re.Match[str]") -> str:
        span = m.group(0)
        o, inner, c = _split_math_delims(span)
        if not _TIKZ_PRESENT_RE.search(inner):
            return span  # no figure code in THIS span — leave it alone
        stripped = _TIKZ_ENV_RE.sub(" ", inner)
        stripped = _TIKZ_ORPHAN_RE.sub(" ", stripped)
        stripped = re.sub(r"[ \t]{2,}", " ", stripped).strip()
        if not stripped:
            # Pure-figure span → render if it parses, else defer to the strip.
            spec = parse_tikz(inner)
            if spec is None:
                return span  # unchanged → strip pass emits the placeholder
            return f'<figure class="dart-figure">{render_svg(spec)}</figure>'
        # Mixed span → keep the surviving math, but leave the placeholder for the
        # figure so it is no longer silently dropped.
        return f"{o}{stripped}{c}{_TIKZ_FIGURE_PLACEHOLDER}"

    return _MATH_SPAN_ANGLE_RE.sub(_fix, text)
