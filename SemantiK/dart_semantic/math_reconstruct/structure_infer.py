"""Deterministic 2-D glyph geometry -> math structure tree.

Consumes the per-glyph geometry recovered by :mod:`.glyph_extract` and
infers the structural relationships a flat reading-order string threw
away:

* **super / subscripts** — from a glyph's size shrink + baseline shift
  (geometry) OR from a unicode super/subscript codepoint (``x²``, ``H₂``).
* **fractions** — from a wide horizontal fraction-bar glyph with runs
  above (numerator) and below (denominator).
* **bracket / row grouping** — balanced ``( )`` / ``[ ]`` runs nest into
  an ``mrow``.

It deliberately recognises a *bounded* set of shapes and reports a
confidence + a ``can_reconstruct`` verdict. Shapes outside that set
(matrices, multi-line / aligned systems, big-operator limits, nested or
stacked fractions, roots-with-index) are flagged ``unsupported`` so
:mod:`.policy` routes them to the local QLoRA math specialist rather than
emitting a wrong structure. The output tree is consumed by
:mod:`.mathml_emit`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .glyph_extract import Glyph, normalize_subscript, normalize_superscript

# ---------------------------------------------------------------------------
# Structure tree nodes (presentation-MathML aligned, emitter-agnostic).
# ---------------------------------------------------------------------------


@dataclass
class Atom:
    """A leaf: identifier (mi), number (mn), operator (mo) or text (mtext)."""

    kind: str  # "mi" | "mn" | "mo" | "mtext"
    text: str


@dataclass
class Script:
    """A super- or sub-script: msup / msub."""

    kind: str  # "msup" | "msub"
    base: Any  # MNode
    script: Any  # MNode


@dataclass
class Frac:
    """A built-up fraction: mfrac(numerator, denominator)."""

    num: Any  # MNode
    den: Any  # MNode


@dataclass
class Row:
    """An ordered run: mrow."""

    children: list[Any] = field(default_factory=list)


@dataclass
class InferResult:
    root: Row
    confidence: float
    complexity: str  # "simple" | "scripts" | "fraction" | "unsupported"
    can_reconstruct: bool
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Character classification.
# ---------------------------------------------------------------------------

# Operators (incl. relations, arrows, calculus glyphs). Brackets handled
# separately for grouping.
_OPERATOR_CHARS = set(
    "+-−*×÷·/=≠≤≥<>±∓→⇒⇔↔∝≈≃∼∈∉⊂⊆∪∩∧∨¬∀∃,;:|"
)
_OPEN_BRACKETS = set("([{")
_CLOSE_BRACKETS = set(")]}")
# Big operators / shapes that need positional layout (munderover / mtable)
# we deliberately do NOT reconstruct deterministically.
_UNSUPPORTED_OPERATOR_CHARS = set("∑∏∫∮⋃⋂√∂∇⨋")

# Normalise look-alike operator codepoints for clean <mo> output.
_OPERATOR_NORMALISE = {
    "-": "−",   # hyphen-minus -> MINUS SIGN
    "*": "⋅",   # asterisk -> DOT OPERATOR
}


def _classify_char(ch: str) -> str:
    if ch.isdigit():
        return "mn"
    if ch.isalpha():
        return "mi"
    if ch in _OPERATOR_CHARS:
        return "mo"
    if ch in _OPEN_BRACKETS or ch in _CLOSE_BRACKETS:
        return "mo"
    return "mo"


# ---------------------------------------------------------------------------
# Geometry helpers.
# ---------------------------------------------------------------------------


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[len(s) // 2]


def _is_fraction_bar(g: Glyph, median_size: float) -> bool:
    """A wide horizontal bar glyph that could be a built-up fraction line."""
    if g.text not in ("-", "−", "—", "―", "─", "━", "⁄"):
        return False
    width = g.x1 - g.x0
    # A real fraction bar is markedly wider than a normal minus (which is
    # ~0.5em). Require width >= ~0.9 * median glyph size.
    return width >= max(1.0, 0.9 * median_size)


# ---------------------------------------------------------------------------
# Tokenisation: glyphs -> atoms, grouping consecutive digits into one mn.
# ---------------------------------------------------------------------------


def _coalesce_atoms(items: list[tuple[str, str]]) -> list[Atom]:
    """Merge adjacent mn runs (digits + interior '.') into single numbers
    and adjacent mi letters that form a multi-letter identifier are kept
    SEPARATE (standard MathML: each variable is its own <mi>)."""
    out: list[Atom] = []
    i = 0
    n = len(items)
    while i < n:
        kind, text = items[i]
        if kind == "mn":
            num = text
            j = i + 1
            while j < n and (
                items[j][0] == "mn"
                or (items[j][1] == "." and j + 1 < n and items[j + 1][0] == "mn")
            ):
                num += items[j][1]
                j += 1
            out.append(Atom("mn", num))
            i = j
            continue
        out.append(Atom(kind, _OPERATOR_NORMALISE.get(text, text) if kind == "mo" else text))
        i += 1
    return out


# ---------------------------------------------------------------------------
# Baseline-row clustering (detect multi-line / fraction stacks).
# ---------------------------------------------------------------------------


def _cluster_baselines(glyphs: list[Glyph], median_size: float) -> list[list[Glyph]]:
    """Group glyphs into horizontal rows by baseline (bottom) proximity.

    Two glyphs share a row when their baselines are within ~0.5*median_size.
    Rows are returned top-to-bottom. Used to detect built-up fractions and
    genuine multi-line systems.
    """
    if not glyphs:
        return []
    tol = max(1.0, 0.5 * median_size)
    rows: list[list[Glyph]] = []
    for g in sorted(glyphs, key=lambda x: x.baseline):
        placed = False
        for row in rows:
            if abs(g.baseline - row[0].baseline) <= tol:
                row.append(g)
                placed = True
                break
        if not placed:
            rows.append([g])
    for row in rows:
        row.sort(key=lambda x: x.x0)
    rows.sort(key=lambda r: r[0].baseline)
    return rows


# ---------------------------------------------------------------------------
# Script (super/subscript) inference within a single baseline row.
# ---------------------------------------------------------------------------


def _infer_scripts_in_row(glyphs: list[Glyph], median_size: float,
                          median_baseline: float) -> tuple[list[Any], bool]:
    """Walk one baseline-row left-to-right, attaching size-shrunk shifted
    glyphs as super/subscripts of the preceding base. Returns (nodes,
    used_geometry_scripts)."""
    nodes: list[Any] = []
    pending: list[tuple[str, str]] = []  # buffered (kind,text) for coalescing
    used_scripts = False
    size_floor = 0.8 * median_size
    shift_tol = 0.18 * median_size

    def flush_pending() -> None:
        if pending:
            nodes.extend(_coalesce_atoms(pending))
            pending.clear()

    for g in glyphs:
        ch = g.text
        # 1) Unicode super/subscript codepoints carry intent regardless of
        #    geometry — handle first.
        sup = normalize_superscript(ch)
        sub = normalize_subscript(ch)
        if sup is not None or sub is not None:
            flush_pending()
            if not nodes:
                # No base to attach to — treat as a normal atom.
                norm = sup if sup is not None else sub
                nodes.append(Atom(_classify_char(norm), norm))
                continue
            base = nodes.pop()
            norm = sup if sup is not None else sub
            script_atom = Atom(_classify_char(norm), norm)
            nodes.append(Script("msup" if sup is not None else "msub", base, script_atom))
            used_scripts = True
            continue

        # 2) Geometry: a shrunk glyph shifted off the baseline. Direction
        #    first, THEN flush the pending base run so the base atom is in
        #    ``nodes`` to attach to (the base usually sits in ``pending``).
        script_dir: str | None = None
        if (
            median_size > 0
            and g.size <= size_floor
            and median_baseline > 0
        ):
            if g.baseline <= median_baseline - shift_tol:
                script_dir = "msup"   # baseline raised (smaller bottom = higher)
            elif g.baseline >= median_baseline + shift_tol:
                script_dir = "msub"   # baseline lowered (larger bottom = lower)
        if script_dir is not None:
            flush_pending()
            if nodes:
                base = nodes.pop()
                nodes.append(Script(script_dir, base, Atom(_classify_char(ch), ch)))
                used_scripts = True
                continue
            # No base to attach to — treat the glyph as a normal atom.
            nodes.append(Atom(_classify_char(ch), ch))
            continue

        pending.append((_classify_char(ch), ch))

    flush_pending()
    return nodes, used_scripts


# ---------------------------------------------------------------------------
# Bracket grouping over a node list.
# ---------------------------------------------------------------------------


def _group_brackets(nodes: list[Any]) -> list[Any]:
    """Nest balanced ()/[]/{} runs into mrow children. Unbalanced brackets
    are left as flat mo atoms (still valid MathML)."""
    out: list[Any] = []
    stack: list[tuple[Atom, list[Any]]] = []
    for node in nodes:
        if isinstance(node, Atom) and node.kind == "mo" and node.text in _OPEN_BRACKETS:
            stack.append((node, []))
            continue
        if isinstance(node, Atom) and node.kind == "mo" and node.text in _CLOSE_BRACKETS:
            if stack:
                open_atom, inner = stack.pop()
                grp = Row([open_atom, *inner, node])
                target = stack[-1][1] if stack else out
                target.append(grp)
                continue
            # Unbalanced close — keep flat.
            (stack[-1][1] if stack else out).append(node)
            continue
        (stack[-1][1] if stack else out).append(node)
    # Any still-open brackets: flatten back so nothing is lost.
    while stack:
        open_atom, inner = stack.pop()
        target = stack[-1][1] if stack else out
        target.append(open_atom)
        target.extend(inner)
    return out


# ---------------------------------------------------------------------------
# Public entry.
# ---------------------------------------------------------------------------

# Glyph-density signals (from the math council) hinting that 2-D structure
# EXISTED. If we are running blind (no real geometry) and these fired but we
# recovered no scripts/fractions, we are likely missing structure -> low
# confidence -> route to the generative fallback.
_STRUCTURE_HINT_KEYS = ("subsup_count", "fraction_bar_count")


def infer_structure(
    glyphs: list[Glyph],
    *,
    glyph_density_features: dict[str, Any] | None = None,
    geometry_source: str = "pdfplumber",
) -> InferResult:
    """Infer a math structure tree from recovered glyphs.

    ``geometry_source`` is ``"pdfplumber"`` (real geometry) or
    ``"src_text"`` (blind / linearised). ``glyph_density_features`` is the
    math council's per-region signal dict (used only to lower confidence
    when we are blind but 2-D structure was detected upstream).
    """
    notes: list[str] = []
    feats = glyph_density_features or {}

    if not glyphs:
        return InferResult(Row([]), 0.0, "unsupported", False, ["no glyphs"])

    median_size = _median([g.size for g in glyphs])

    # Strip a trailing equation number "(N)" — it is layout furniture, not
    # part of the expression (the assembler can re-attach as <mtext> later).
    # Detect via a balanced trailing (...) of only digits/dots.
    # (Handled at row level below.)

    # --- Unsupported-shape gate (route to generative). -------------------
    expr_text = "".join(g.text for g in glyphs)
    if any(c in _UNSUPPORTED_OPERATOR_CHARS for c in expr_text):
        notes.append("big-operator/root present (needs positional layout)")
        return InferResult(Row([Atom("mtext", expr_text)]), 0.30,
                           "unsupported", False, notes)

    rows = _cluster_baselines(glyphs, median_size)

    # --- Built-up fraction: a wide bar with a row above and below. -------
    bar = None
    if geometry_source == "pdfplumber":
        for g in glyphs:
            if _is_fraction_bar(g, median_size):
                bar = g
                break

    if bar is not None:
        above = [g for g in glyphs if g is not bar and g.baseline < bar.top]
        below = [g for g in glyphs if g is not bar and g.top > bar.baseline]
        if above and below:
            # Reject NESTED / multiple bars (out of deterministic scope).
            other_bars = [
                g for g in glyphs
                if g is not bar and _is_fraction_bar(g, median_size)
            ]
            if other_bars:
                notes.append("multiple/nested fraction bars (unsupported)")
                return InferResult(Row([Atom("mtext", expr_text)]), 0.30,
                                   "unsupported", False, notes)
            num_res = _infer_simple_row(above, median_size)
            den_res = _infer_simple_row(below, median_size)
            root = Row([Frac(num_res, den_res)])
            notes.append("built-up fraction")
            return InferResult(root, 0.85, "fraction", True, notes)

    # --- Multi-line system (no fraction bar) -> unsupported. -------------
    if geometry_source == "pdfplumber" and len(rows) > 1:
        notes.append(f"{len(rows)} baseline rows w/o fraction bar (multiline)")
        return InferResult(Row([Atom("mtext", expr_text)]), 0.30,
                           "unsupported", False, notes)

    # --- Single-line: script inference + bracket grouping. ---------------
    line = rows[0] if rows else list(glyphs)
    baselines = [g.baseline for g in line]
    # Use the MAJORITY (full-size) baseline as the reference, not the median
    # of all glyphs (which scripts would skew).
    full = [g for g in line if g.size >= 0.85 * median_size]
    median_baseline = _median([g.baseline for g in (full or line)])

    nodes, used_scripts = _infer_scripts_in_row(line, median_size, median_baseline)
    nodes = _group_brackets(nodes)
    root = Row(nodes)

    complexity = "scripts" if used_scripts else "simple"
    confidence = 0.92 if used_scripts else 0.88

    # Blind-but-structured penalty: no geometry AND the council saw 2-D
    # signal we couldn't recover -> route to generative.
    if geometry_source != "pdfplumber" and not used_scripts:
        hinted = any(float(feats.get(k, 0) or 0) > 0 for k in _STRUCTURE_HINT_KEYS)
        if hinted:
            notes.append("blind linearise but upstream saw 2-D structure")
            return InferResult(root, 0.35, "unsupported", False, notes)
        # Pure linear expression recovered blind is still correct.
        confidence = 0.80

    if not nodes:
        return InferResult(root, 0.0, "unsupported", False, ["empty after parse"])

    return InferResult(root, confidence, complexity, True, notes)


def _infer_simple_row(glyphs: list[Glyph], median_size: float) -> Row:
    """Sub-row inference (no recursion into fractions) for fraction parts."""
    if not glyphs:
        return Row([])
    line = sorted(glyphs, key=lambda g: g.x0)
    full = [g for g in line if g.size >= 0.85 * median_size]
    median_baseline = _median([g.baseline for g in (full or line)])
    nodes, _ = _infer_scripts_in_row(line, median_size, median_baseline)
    nodes = _group_brackets(nodes)
    return Row(nodes)


__all__ = [
    "Atom",
    "Script",
    "Frac",
    "Row",
    "InferResult",
    "infer_structure",
]
