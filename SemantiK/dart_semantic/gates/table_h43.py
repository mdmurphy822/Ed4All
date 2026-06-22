"""Complex-table ``headers``/``id`` association — WCAG technique H43
(Plan 12 B1 / V3.3).

Simple tables (one header row, or one leading header column) are fully
served by ``<th scope>``. Complex tables — two-plus header rows, dual
axis (column headers AND row headers), or spanned header cells — make
``scope`` ambiguous; SC 1.3.1 then needs explicit ``<th id>`` +
``<td headers>`` pairing (H43).

Three entry points:

* :func:`analyze_table`  — grid model + complexity verdict.
* :func:`apply_h43`      — deterministic enrichment: generates ``id``
  on every ``<th>`` and ``headers`` on every ``<td>`` of a *complex*
  table. No-op (byte-identical input) for simple/non-table fragments.
  When a complex table already carries partial ``headers`` attributes
  they are regenerated — the grid arithmetic here is the authority,
  not the upstream Qwen emission.
* :func:`verify_h43`     — gate-side check: a complex table must carry
  resolvable id/headers pairing. Used by
  ``hard_region._check_table_structure``.

Uses lxml (already a hard dependency — the MathML gate raises without
it). Parsing failures raise — a fragment that survived the HTML5
well-formedness check but breaks the lxml parser is a bug to surface,
not to skip (feedback_no_silent_fallbacks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class H43Error(RuntimeError):
    """Raised when a table fragment cannot be parsed/enriched."""


def _lxml_html():
    try:
        from lxml import html as lxml_html
    except ImportError as exc:  # pragma: no cover — lxml is mandatory
        raise H43Error(
            "lxml is required for H43 table association (same hard dependency as the MathML gate)"
        ) from exc
    return lxml_html


# ---------------------------------------------------------------------------
# Grid model
# ---------------------------------------------------------------------------


@dataclass
class _Cell:
    el: Any
    is_th: bool
    row: int  # origin row in the expanded grid
    col: int  # origin column in the expanded grid
    rowspan: int
    colspan: int
    in_header_row: bool = False


@dataclass
class TableInfo:
    n_rows: int = 0
    n_cols: int = 0
    n_header_rows: int = 0
    has_row_headers: bool = False  # <th> in a non-header row
    has_spanned_th: bool = False
    is_complex: bool = False
    reasons: list[str] = field(default_factory=list)
    cells: list[_Cell] = field(default_factory=list)
    # grid[r][c] -> _Cell occupying that slot (spans expanded), or None
    grid: list[list[_Cell | None]] = field(default_factory=list)


def _int_attr(el: Any, name: str, default: int = 1) -> int:
    try:
        v = int(el.get(name, default))
        return max(1, v)
    except (TypeError, ValueError):
        return default


def _own_rows(table: Any) -> list[Any]:
    """``<tr>`` elements belonging to this table (not nested tables),
    in document order."""
    rows = []
    for tr in table.iter("tr"):
        anc = tr.getparent()
        while anc is not None and anc.tag != "table":
            anc = anc.getparent()
        if anc is table:
            rows.append(tr)
    return rows


def _build_info(table: Any) -> TableInfo:
    info = TableInfo()
    rows = _own_rows(table)
    if not rows:
        return info

    grid: list[list[_Cell | None]] = []

    def _ensure(r: int, c: int) -> None:
        while len(grid) <= r:
            grid.append([])
        while len(grid[r]) <= c:
            grid[r].append(None)

    for r, tr in enumerate(rows):
        in_thead = any(a.tag == "thead" for a in tr.iterancestors())
        cells_in_row = [ch for ch in tr if ch.tag in ("td", "th")]
        all_th = bool(cells_in_row) and all(ch.tag == "th" for ch in cells_in_row)
        header_row = in_thead or all_th
        c = 0
        for el in cells_in_row:
            _ensure(r, c)
            while grid[r][c] is not None:
                c += 1
                _ensure(r, c)
            cell = _Cell(
                el=el,
                is_th=(el.tag == "th"),
                row=r,
                col=c,
                rowspan=_int_attr(el, "rowspan"),
                colspan=_int_attr(el, "colspan"),
                in_header_row=header_row,
            )
            info.cells.append(cell)
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    _ensure(r + dr, c + dc)
                    grid[r + dr][c + dc] = cell
            c += cell.colspan
        if header_row:
            info.n_header_rows += 1

    info.grid = grid
    info.n_rows = len(grid)
    info.n_cols = max((len(r) for r in grid), default=0)
    info.has_row_headers = any(c.is_th and not c.in_header_row for c in info.cells)
    info.has_spanned_th = any(c.is_th and (c.rowspan > 1 or c.colspan > 1) for c in info.cells)

    if info.n_header_rows >= 2:
        info.reasons.append(f"{info.n_header_rows} header rows")
    if info.has_row_headers and info.n_header_rows >= 1:
        info.reasons.append("dual-axis headers (column AND row)")
    if info.has_spanned_th:
        info.reasons.append("spanned <th> cells")
    info.is_complex = bool(info.reasons)
    return info


def _parse_fragment(html: str):
    lxml_html = _lxml_html()
    try:
        root = lxml_html.fragment_fromstring(html, create_parent="div")
    except Exception as exc:
        raise H43Error(f"table fragment failed to parse: {exc}") from exc
    return root


def analyze_table(html: str) -> TableInfo:
    """Grid model + complexity verdict for the first ``<table>`` in
    the fragment. A fragment with no table returns an empty TableInfo
    (``is_complex=False``)."""
    root = _parse_fragment(html)
    table = next(root.iter("table"), None)
    if table is None:
        return TableInfo()
    return _build_info(table)


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def _governing_header_ids(cell: _Cell, info: TableInfo) -> list[str]:
    """ids of the <th> cells governing a data cell: column headers from
    header rows in the cell's spanned columns + row headers to its left
    in the cell's spanned rows. Document order, deduped."""
    seen: list[Any] = []
    for dc in range(cell.colspan):
        col = cell.col + dc
        for r in range(cell.row):
            if col < len(info.grid[r]):
                g = info.grid[r][col]
                if g is not None and g.is_th and g.in_header_row and g not in seen:
                    seen.append(g)
    for dr in range(cell.rowspan):
        row = cell.row + dr
        if row < len(info.grid):
            for c in range(cell.col):
                if c < len(info.grid[row]):
                    g = info.grid[row][c]
                    if g is not None and g.is_th and not g.in_header_row and g not in seen:
                        seen.append(g)
    return [g.el.get("id") for g in seen if g.el.get("id")]


def apply_h43(html: str, *, id_prefix: str = "t") -> str:
    """Add H43 ``id``/``headers`` association to a COMPLEX table.

    Simple tables and non-table fragments are returned byte-identical.
    For complex tables: every ``<th>`` gets a deterministic id
    (``{id_prefix}-r{row}c{col}``, origin coordinates; existing ids are
    kept), and every ``<td>``'s ``headers`` attribute is REGENERATED
    from the grid arithmetic.
    """
    root = _parse_fragment(html)
    table = next(root.iter("table"), None)
    if table is None:
        return html
    info = _build_info(table)
    if not info.is_complex:
        return html

    for cell in info.cells:
        if cell.is_th and not cell.el.get("id"):
            cell.el.set("id", f"{id_prefix}-r{cell.row}c{cell.col}")
    for cell in info.cells:
        if cell.is_th:
            continue
        ids = _governing_header_ids(cell, info)
        if ids:
            cell.el.set("headers", " ".join(ids))

    lxml_html = _lxml_html()
    return "".join(lxml_html.tostring(child, encoding="unicode") for child in root)


# ---------------------------------------------------------------------------
# Gate-side verification
# ---------------------------------------------------------------------------


def verify_h43(html: str) -> tuple[bool, str, dict]:
    """Gate check for complex tables: (passed, message, details).

    Simple tables pass trivially (``scope`` is their contract — checked
    elsewhere). Complex tables must carry an id on every ``<th>`` and a
    ``headers`` attribute on every governed ``<td>`` whose ids all
    resolve to ``<th>`` ids in the same table.
    """
    info = analyze_table(html)
    if not info.is_complex:
        return True, "not_complex", {"is_complex": False}

    details: dict = {
        "is_complex": True,
        "reasons": list(info.reasons),
        "rows": info.n_rows,
        "cols": info.n_cols,
    }
    th_ids = set()
    for cell in info.cells:
        if cell.is_th:
            ident = (cell.el.get("id") or "").strip()
            if not ident:
                return (
                    False,
                    f"complex table ({'; '.join(info.reasons)}): <th> at "
                    f"r{cell.row}c{cell.col} has no id= (H43 requires "
                    f"id/headers association)",
                    details,
                )
            th_ids.add(ident)
    for cell in info.cells:
        if cell.is_th:
            continue
        # Every <th> has an id by this point (checked above), so the
        # grid arithmetic fully determines which cells are governed.
        governed = bool(_governing_header_ids(cell, info))
        headers = (cell.el.get("headers") or "").split()
        if governed and not headers:
            return (
                False,
                f"complex table ({'; '.join(info.reasons)}): <td> at "
                f"r{cell.row}c{cell.col} has no headers= (H43)",
                details,
            )
        unresolved = [h for h in headers if h not in th_ids]
        if unresolved:
            return (
                False,
                f"complex table: <td> at r{cell.row}c{cell.col} headers= "
                f"references unknown ids {unresolved}",
                details,
            )
    return True, f"h43 ok ({'; '.join(info.reasons)})", details


__all__ = ["H43Error", "TableInfo", "analyze_table", "apply_h43", "verify_h43"]
