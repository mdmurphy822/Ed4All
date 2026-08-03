"""Build the Qwen-table LoRA adapter training dataset (Phase 1.2).

One row per HTML ``<table>`` element in an ar5iv pair's ``raw_source_html``.
The source side is the EXACT JSON envelope that
``semantik_structure.qwen_specialists.prompts.build_table_request`` emits at
inference time, given a ``Region`` derived from the same table. The
target side is an HTML5 ``<table>`` fragment, normalized to include
``<caption>`` (when ar5iv provides one via the wrapping
``<figure class="ltx_table">`` / ``<figcaption>``), ``<thead>``/``<tbody>``,
and ``<th scope="col"|"row">`` for header cells.

ar5iv conventions we adapt to:
    * ar5iv emits LaTeX ``tabular`` as ``<table class="ltx_tabular">``;
      LaTeX ``equation*`` /  ``eqnarray`` blocks are ALSO emitted as
      ``<table class="ltx_eqn_table ltx_equation">`` etc. — those are
      NOT data tables and are filtered out (they're handled by the
      math adapter, not the table adapter).
    * Real data-table captions are NOT in a ``<caption>`` child;
      ar5iv wraps the table in ``<figure class="ltx_table">`` with a
      sibling ``<figcaption class="ltx_caption">``. We hoist that into
      a generated ``<caption>`` in the target HTML5 fragment.
    * ar5iv ``<th>`` cells do NOT carry ``scope=`` attributes; instead
      they carry classes like ``ltx_th_column`` / ``ltx_th_row``.
      We translate those into ``scope="col"`` / ``scope="row"`` in
      the target HTML, and into the role enum in the request payload.

Splits use ``data.common._splits.stable_split_for_id`` so an arxiv_id that
trains the semantic_preservation head won't appear in this builder's
val/test (and vice versa). Cross-builder leakage would silently inflate
held-out scores once the two heads run together.

CPU-ONLY. Loads no ML weights, no tokenizers — token counts are
whitespace approximated. Safe to run while training is on the GPU.

Usage::

    python data/builders/build_table_qwen_data.py                    # full build
    python data/builders/build_table_qwen_data.py --max-pairs 30 \\
        --out-dir /tmp/qwen_table_smoke                     # smoke test
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from lxml import etree, html as lxml_html

# Live prompt builder — the contract surface this dataset is bound to.
# Any drift in prompts.py surfaces as failure in
# tests/test_qwen_table_dataset_contract.py.
from semantik_structure.qwen_specialists.prompts import build_table_request

from data.common._splits import stable_split_for_id


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_PAIRS_DIR = Path("data/pairs/arxiv")
DEFAULT_OUT_DIR = Path("data/qwen_table_dataset")

WS_RE = re.compile(r"\s+")

# Histogram bucket edges (right-exclusive); chosen to expose long-tail
# table sizes without 100s of empty bins.
TOKEN_HIST_EDGES = [0, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
ROW_HIST_EDGES = [0, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
COL_HIST_EDGES = [0, 2, 3, 4, 5, 6, 8, 10, 15, 20, 50]

# ar5iv class markers — used to discriminate data tables from
# equation-block tables. These are mutually exclusive in practice
# (a table either carries ltx_tabular OR an ltx_eqn* class).
LTX_DATA_TABLE_CLASS = "ltx_tabular"
LTX_EQUATION_CLASS_PREFIXES = ("ltx_eqn",)
LTX_EQUATIONGROUP_CLASS = "ltx_equationgroup"

# ar5iv key/value pseudo-tables that aren't real data tables and don't
# appear in the rendered PDF (the BERT extractor filters these too).
PSEUDO_TABLE_KEYWORDS = ("cite as:", "doi:", "subjects:")


# --------------------------------------------------------------------------- #
# Skip / counter bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class SkipCounters:
    not_data_table: int = 0          # table is an ltx_eqn* equation block
    pseudo_table: int = 0            # ar5iv "Cite as: / Subjects:" KV blob
    too_small: int = 0               # 1x1 or fewer (degenerate)
    all_empty_cells: int = 0         # every cell rendered to ""
    serialize_error: int = 0
    parse_error: int = 0
    duplicate_table: int = 0         # (arxiv_id, table_idx) seen before


@dataclass
class BuildStats:
    n_pairs_seen: int = 0
    n_pairs_with_html: int = 0
    n_pairs_with_arxiv_id: int = 0
    n_unique_arxiv_ids_seen: int = 0
    n_tables_inspected: int = 0
    n_rows_emitted: int = 0
    rows_per_split: Counter = field(default_factory=Counter)
    rows_per_arxiv_id: Counter = field(default_factory=Counter)
    n_with_caption: int = 0
    n_with_scope: int = 0
    n_with_rowspan: int = 0
    n_with_colspan: int = 0
    target_token_hist: Counter = field(default_factory=Counter)
    n_rows_hist: Counter = field(default_factory=Counter)
    n_cols_hist: Counter = field(default_factory=Counter)
    skip: SkipCounters = field(default_factory=SkipCounters)


def _bucket_for(n: int, edges: list[int]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= n < edges[i + 1]:
            return f"[{edges[i]},{edges[i+1]})"
    return f"[{edges[-1]},inf)"


def _ws_token_count(s: str) -> int:
    return len([t for t in WS_RE.split(s.strip()) if t])


# --------------------------------------------------------------------------- #
# ar5iv → normalized table extraction
# --------------------------------------------------------------------------- #


def _classes(elem: etree._Element) -> list[str]:
    return (elem.get("class") or "").split()


def _is_equation_table(elem: etree._Element) -> bool:
    """True if this ``<table>`` is an ar5iv equation-block wrapper.

    ar5iv emits both data tables AND LaTeX equation blocks as ``<table>``
    elements; the equation ones carry ``ltx_eqn*`` or
    ``ltx_equationgroup`` classes. Real data tables carry
    ``ltx_tabular``.
    """
    cls = _classes(elem)
    if any(c.startswith(LTX_EQUATION_CLASS_PREFIXES) for c in cls):
        return True
    if LTX_EQUATIONGROUP_CLASS in cls:
        return True
    return False


def _flat_text(elem: etree._Element) -> str:
    """Whitespace-collapsed text from an element subtree.

    Preserves math ``alttext`` when the cell contains a ``<math>``
    element (ar5iv's tabular cells often hold inline math; the alttext
    is the LaTeX source string). Without this, math-bearing cells
    would extract as empty.
    """
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        if isinstance(child.tag, str):
            tag_local = child.tag.split("}", 1)[-1]
            if tag_local == "math":
                alt = child.get("alttext") or ""
                if alt:
                    parts.append(alt)
                else:
                    parts.append(_flat_text(child))
            else:
                parts.append(_flat_text(child))
        if child.tail:
            parts.append(child.tail)
    return WS_RE.sub(" ", "".join(parts)).strip()


def _figure_caption_text(table_elem: etree._Element) -> str | None:
    """Hoist a caption from the wrapping ``<figure class="ltx_table">``.

    ar5iv prints data-table captions as a sibling ``<figcaption>`` of
    the ``<table>``, not as a child ``<caption>``. We walk up to the
    nearest ``ltx_table`` figure ancestor and pull the figcaption.
    Returns None if no caption is present.
    """
    for anc in table_elem.iterancestors():
        if anc.tag != "figure":
            continue
        cls = _classes(anc)
        if "ltx_table" not in cls:
            continue
        figcaps = anc.xpath('.//figcaption[contains(concat(" ", @class, " "), " ltx_caption ")]')
        if not figcaps:
            return None
        text = _flat_text(figcaps[0])
        return text or None
    return None


def _is_pseudo_kv_table(grid: list[list[dict[str, Any]]]) -> bool:
    """Detect ar5iv's "Cite as:" / "Subjects:" / "DOI:" key-value blob.

    These are ar5iv-injected metadata tables that don't appear in the
    rendered PDF and would contaminate the dataset. Originally we
    bounded by row count (≤4) as the BERT extractor does, but ar5iv
    occasionally emits 5+ row variants; switch to keyword-presence
    in the first column with no row-count cap.
    """
    first_col = " ".join(
        (row[0]["text"] if row and row[0].get("is_origin", True) else "")
        for row in grid
    ).lower()
    n_hits = sum(1 for k in PSEUDO_TABLE_KEYWORDS if k in first_col)
    # Two or more keywords → almost certainly the ar5iv metadata table;
    # one keyword could be a real data table that happens to mention
    # "DOI:" in passing, so require at least two for the broader filter.
    if n_hits >= 2:
        return True
    # Keep the original ≤4-row + single-keyword path for the small-table
    # case (preserves compatibility with the BERT extractor's filter).
    if len(grid) <= 4 and n_hits >= 1:
        return True
    return False


def _cell_role_from_classes(
    cell_elem: etree._Element, *,
    is_th: bool, rowspan: int, colspan: int,
    row_idx: int, col_idx: int,
) -> str:
    """Map an ar5iv cell's class soup to the BERT-TableSpecialist role.

    Mirror of ``data.builders.build_table_specialist_data.assign_cell_label`` but
    extended to consume ar5iv's ``ltx_th_column`` / ``ltx_th_row``
    classes (ar5iv ``<th>`` cells don't carry ``scope=`` so we have to
    read those classes).

    Roles (matches build_table_specialist_data):
        ``data``        plain ``<td>`` with no spans
        ``header_col``  ``<th>`` labeling a column (scope=col equivalent)
        ``header_row``  ``<th>`` labeling a row (scope=row equivalent)
        ``span``        ``<td>`` with rowspan/colspan > 1
    """
    spanned = (rowspan > 1) or (colspan > 1)
    if is_th:
        cls = _classes(cell_elem)
        # ar5iv signals: ltx_th_column → column header, ltx_th_row →
        # row header. A cell can carry BOTH (corner cell labeling both
        # axes) — collapse to header_col there since the live
        # build_table_request enum doesn't include header_both.
        has_col = "ltx_th_column" in cls
        has_row = "ltx_th_row" in cls
        scope = (cell_elem.get("scope") or "").lower()
        if scope in ("col", "colgroup"):
            return "header_col"
        if scope in ("row", "rowgroup"):
            return "header_row"
        if has_col:
            return "header_col"
        if has_row:
            return "header_row"
        # Position-based fallback (matches build_table_specialist_data):
        # first row → header_col; first col → header_row; mid → header_col.
        if row_idx == 0:
            return "header_col"
        if col_idx == 0:
            return "header_row"
        return "header_col"
    if spanned:
        return "span"
    return "data"


def _cell_scope_for_html(role: str) -> str | None:
    """Map our role enum back to an HTML5 ``scope`` value for the
    target HTML side. Spans / data don't get scope.
    """
    if role == "header_col":
        return "col"
    if role == "header_row":
        return "row"
    return None


def _expand_grid(table_elem: etree._Element) -> list[list[dict[str, Any]]]:
    """Materialise an ar5iv table into a rectangular grid with origin
    + replica cells, mirroring ``_TableExtractor`` in
    ``data/builders/build_table_specialist_data.py``.

    Each grid cell is::

        {
          "text": str,
          "tag": "th" | "td",          # source tag, before role mapping
          "rowspan": int,
          "colspan": int,
          "is_origin": bool,
          "role": str,                 # filled in after, by caller
          "elem": etree._Element|None, # origin element, for class lookup
        }

    Replica cells are inserted for colspan/rowspan so the grid is
    rectangular. Replicas reference the same ``elem`` as their origin
    so role assignment can read classes consistently.
    """
    grid: list[list[dict[str, Any]]] = []
    open_rowspans: dict[int, dict[str, Any]] = {}  # col -> replica template

    # Iterate <tr> in document order (skip nested table rows). We can't
    # use simple .//tr because that'd descend into nested <table>s; iterate
    # with a stack and skip subtrees rooted at a nested <table>.
    rows: list[etree._Element] = []
    def _walk(node: etree._Element) -> None:
        for child in node:
            if not isinstance(child.tag, str):
                continue
            tag = child.tag.split("}", 1)[-1]
            if tag == "table":
                continue  # nested table — its rows belong to that table
            if tag == "tr":
                rows.append(child)
            else:
                _walk(child)
    _walk(table_elem)

    for row_idx, tr in enumerate(rows):
        row: list[dict[str, Any]] = []
        # First, drop in any replicas from prior rowspans.
        if open_rowspans:
            max_col = max(open_rowspans.keys())
            new_open: dict[int, dict[str, Any]] = {}
            for col in range(max_col + 1):
                tmpl = open_rowspans.get(col)
                if tmpl is None:
                    continue
                # Replicas may interleave with same-row new cells; for
                # simplicity, append in column order. ar5iv emits
                # explicit rows that already account for spans, so this
                # is a defensive fallback.
                row.append(dict(tmpl, is_origin=False))
                rem = tmpl["rowspan"] - 1
                if rem >= 1:
                    new_open[col] = dict(tmpl, rowspan=rem)
            open_rowspans = new_open

        # Now consume this row's <td>/<th>.
        for cell_elem in tr:
            if not isinstance(cell_elem.tag, str):
                continue
            tag_local = cell_elem.tag.split("}", 1)[-1]
            if tag_local not in ("td", "th"):
                continue
            try:
                rowspan = int(cell_elem.get("rowspan") or "1")
            except ValueError:
                rowspan = 1
            try:
                colspan = int(cell_elem.get("colspan") or "1")
            except ValueError:
                colspan = 1
            rowspan = max(1, rowspan)
            colspan = max(1, colspan)
            text = _flat_text(cell_elem)
            base = {
                "text": text,
                "tag": tag_local,
                "rowspan": rowspan,
                "colspan": colspan,
                "is_origin": True,
                "elem": cell_elem,
            }
            row.append(base)
            for _ in range(colspan - 1):
                row.append(dict(base, is_origin=False))
            if rowspan > 1:
                start_col = len(row) - colspan
                for col_offset in range(colspan):
                    open_rowspans[start_col + col_offset] = dict(
                        base, is_origin=False, rowspan=rowspan - 1,
                    )
        grid.append(row)

    # Pad to rectangular shape.
    n_cols = max((len(r) for r in grid), default=0)
    for r in grid:
        while len(r) < n_cols:
            r.append({
                "text": "", "tag": "td", "rowspan": 1, "colspan": 1,
                "is_origin": True, "elem": None,
            })
    return grid


def _detect_header_rows(
    grid: list[list[dict[str, Any]]],
    table_elem: etree._Element,
) -> list[int]:
    """Decide which row indices are header rows.

    Strategy: a row is a header row iff every (non-empty) origin cell in
    it is a ``<th>``. ar5iv puts headers inside ``<thead>`` but our
    grid walker doesn't preserve thead structure — so we infer from the
    ``tag`` field. This matches what ``build_table_request`` would see
    via the structure_graph cell_grid.
    """
    out: list[int] = []
    for i, row in enumerate(grid):
        if not row:
            continue
        # Skip rows that are entirely replicas (no origin cells).
        origin_cells = [c for c in row if c.get("is_origin", True)]
        if not origin_cells:
            continue
        if all(c["tag"] == "th" for c in origin_cells):
            out.append(i)
    return out


def _is_bordered_arxiv(table_elem: etree._Element) -> bool:
    """Heuristic: ar5iv ``ltx_tabular`` carries no border attribute; the
    rendered LaTeX usually has hlines / vlines. We default to True for
    ar5iv data tables since LaTeX ``\\hline`` is overwhelmingly the
    convention; the field is informational only (not load-bearing in
    the trained adapter — it's an envelope key passed through).
    """
    return True


# --------------------------------------------------------------------------- #
# Target-HTML construction
# --------------------------------------------------------------------------- #


def _build_target_html(
    *, grid: list[list[dict[str, Any]]],
    caption_text: str | None,
    header_row_indices: list[int],
) -> str:
    """Build a normalized HTML5 ``<table>`` fragment.

    Output features:
        * ``<caption>`` first child if caption_text is non-None.
        * ``<thead>`` wrapping the contiguous header rows starting at
          row 0; ``<tbody>`` wrapping the rest. (ar5iv often has
          non-contiguous header rows — for those we emit no thead and
          dump everything into tbody, since HTML5 doesn't allow
          interleaved thead/tbody groups.)
        * ``<th scope="col"|"row">`` per role.
        * Origin cells get their own ``<td>``/``<th>``; replicas are
          dropped (the rowspan/colspan attribute on the origin already
          encodes them in HTML5).

    The output is then re-serialized via ``etree.tostring`` to ensure
    well-formedness.
    """
    table = etree.Element("table")
    if caption_text:
        cap = etree.SubElement(table, "caption")
        cap.text = caption_text

    # Determine if header rows form a contiguous prefix.
    contiguous_header = (
        header_row_indices == list(range(len(header_row_indices)))
        and len(header_row_indices) > 0
    )

    if contiguous_header:
        thead = etree.SubElement(table, "thead")
        tbody = etree.SubElement(table, "tbody")
    else:
        thead = None
        tbody = etree.SubElement(table, "tbody")

    header_set = set(header_row_indices)
    for i, row in enumerate(grid):
        target_parent = thead if (thead is not None and i in header_set) else tbody
        tr = etree.SubElement(target_parent, "tr")
        for cell in row:
            if not cell.get("is_origin", True):
                continue
            role = cell.get("role", "data")
            # Header cell: <th scope="...">.
            if role in ("header_col", "header_row"):
                el = etree.SubElement(tr, "th")
                scope = _cell_scope_for_html(role)
                if scope:
                    el.set("scope", scope)
            else:
                el = etree.SubElement(tr, "td")
            if cell["rowspan"] > 1:
                el.set("rowspan", str(cell["rowspan"]))
            if cell["colspan"] > 1:
                el.set("colspan", str(cell["colspan"]))
            el.text = cell["text"]

    return etree.tostring(table, encoding="unicode", with_tail=False)


# --------------------------------------------------------------------------- #
# Per-table row construction
# --------------------------------------------------------------------------- #


def _build_request_payload_via_live(
    *, grid: list[list[dict[str, Any]]],
    n_rows: int, n_cols: int,
    bordered: bool,
    header_row_indices: list[int],
    caption_text: str | None,
) -> dict[str, Any]:
    """Construct the inputs ``build_table_request`` expects, then call
    it and serialize its return value.

    ``build_table_request`` reads:
        - ``region.payload['cell_grid']``   (str text per cell)
        - ``region.payload['cell_roles']``  (role per cell)
        - ``region.payload['n_rows']``, ``n_cols``, ``bordered``,
          ``header_row_indices``, ``caption_fb_index``
        - ``feature_blocks[caption_fb_index].raw.text`` for the
          caption_text envelope field.

    We back the caption via a tiny stub: a one-element feature_blocks
    list whose ``raw.text`` carries the caption, plus
    ``caption_fb_index=0``. When caption_text is None we set
    caption_fb_index=None which forces ``_caption_text`` to return
    None — matching the live no-caption branch exactly.
    """
    cell_grid = [[c["text"] for c in row] for row in grid]
    cell_roles = [[c.get("role", "data") for c in row] for row in grid]

    if caption_text is not None:
        feature_blocks = (
            SimpleNamespace(raw=SimpleNamespace(text=caption_text)),
        )
        caption_fb_index = 0
    else:
        feature_blocks = ()
        caption_fb_index = None

    payload = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "bordered": bordered,
        "header_row_indices": list(header_row_indices),
        "cell_grid": cell_grid,
        "cell_roles": cell_roles,
        "caption_fb_index": caption_fb_index,
    }
    region = SimpleNamespace(
        payload=payload,
        feature_block_indices=(),
        aria_hints=(),
        kind="table",
    )
    request = build_table_request(region, feature_blocks)
    return {
        "adapter": request.adapter.value,
        "payload": request.payload,
        "request_id": request.request_id,
        "sampling_overrides": request.sampling_overrides,
    }


def _table_signature(grid: list[list[dict[str, Any]]]) -> tuple:
    """A small hashable signature for in-paper de-dup.

    Used to skip identical replicated tables within a single arxiv_id —
    e.g. some ar5iv pages repeat boilerplate tables. Per-arxiv_id
    dedup uses (arxiv_id, table_idx); but if the same paper has the
    same data table at two different table_idx slots, only the first
    survives this signature filter as well, by design.
    """
    return tuple(
        tuple(
            (c.get("text", ""), c.get("tag", ""), c.get("rowspan", 1),
             c.get("colspan", 1), c.get("is_origin", True))
            for c in row
        )
        for row in grid
    )


def build_row_for_table(
    *,
    table_elem: etree._Element,
    arxiv_id: str,
    source_pair: str,
    table_idx: int,
    skip: SkipCounters,
) -> dict[str, Any] | None:
    """Return a JSONL-serializable row, or ``None`` to skip."""

    # Filter equation tables (math adapter handles those).
    if _is_equation_table(table_elem):
        skip.not_data_table += 1
        return None

    grid = _expand_grid(table_elem)
    if not grid or not grid[0]:
        skip.too_small += 1
        return None
    n_rows = len(grid)
    n_cols = max((len(r) for r in grid), default=0)

    # Skip degenerate tables.
    if n_rows < 2 or n_cols < 2:
        skip.too_small += 1
        return None

    # Pseudo "Cite as:" tables.
    if _is_pseudo_kv_table(grid):
        skip.pseudo_table += 1
        return None

    # All-empty filter.
    n_non_empty = sum(
        1 for row in grid for c in row
        if c.get("is_origin", True) and (c.get("text") or "").strip()
    )
    if n_non_empty == 0:
        skip.all_empty_cells += 1
        return None

    # Assign per-cell role on origin cells; propagate to replicas.
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell.get("is_origin", True) and cell.get("elem") is not None:
                role = _cell_role_from_classes(
                    cell["elem"],
                    is_th=(cell["tag"] == "th"),
                    rowspan=cell["rowspan"],
                    colspan=cell["colspan"],
                    row_idx=i, col_idx=j,
                )
                cell["role"] = role
            elif not cell.get("is_origin", True):
                # Replicas inherit role from the row's prior origin
                # (best-effort; we'll resolve below in a second pass).
                cell.setdefault("role", "span")
            else:
                cell["role"] = "data"

    # Second pass: fix replica roles by walking each row and inheriting
    # from the most-recent origin to the left.
    for row in grid:
        last_role = "data"
        for cell in row:
            if cell.get("is_origin", True):
                last_role = cell.get("role", "data")
            else:
                cell["role"] = last_role if last_role != "data" else "span"

    header_row_indices = _detect_header_rows(grid, table_elem)
    caption_text = _figure_caption_text(table_elem)
    bordered = _is_bordered_arxiv(table_elem)

    has_rowspan = any(
        c.get("is_origin", True) and c.get("rowspan", 1) > 1
        for row in grid for c in row
    )
    has_colspan = any(
        c.get("is_origin", True) and c.get("colspan", 1) > 1
        for row in grid for c in row
    )
    n_th_cells = sum(
        1 for row in grid for c in row
        if c.get("is_origin", True) and c.get("tag") == "th"
    )
    n_origin_cells = sum(
        1 for row in grid for c in row if c.get("is_origin", True)
    )

    try:
        target_html = _build_target_html(
            grid=grid,
            caption_text=caption_text,
            header_row_indices=header_row_indices,
        )
    except Exception:
        skip.serialize_error += 1
        return None

    request_payload = _build_request_payload_via_live(
        grid=grid,
        n_rows=n_rows, n_cols=n_cols,
        bordered=bordered,
        header_row_indices=header_row_indices,
        caption_text=caption_text,
    )

    n_target_tokens = _ws_token_count(target_html)

    return {
        "source_pair": source_pair,
        "arxiv_id": arxiv_id,
        "table_idx": table_idx,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "request_payload": request_payload,
        "target_html": target_html,
        "n_target_tokens": n_target_tokens,
        "extraction_metadata": {
            "n_cells": n_origin_cells,
            "n_th_cells": n_th_cells,
            "has_caption": bool(caption_text),
            "has_rowspan": has_rowspan,
            "has_colspan": has_colspan,
        },
    }


# --------------------------------------------------------------------------- #
# Per-pair iteration
# --------------------------------------------------------------------------- #


def _iter_tables(html_str: str) -> Iterator[etree._Element]:
    """Yield every ``<table>`` element in document order."""
    try:
        root = lxml_html.fromstring(html_str)
    except Exception:
        return iter(())
    return iter(root.xpath(".//table"))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def iter_pair_files(pairs_dir: Path) -> Iterator[Path]:
    yield from sorted(pairs_dir.glob("*.json"))


def _open_split_files(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "train": (out_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (out_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (out_dir / "test.jsonl").open("w", encoding="utf-8"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="Stop after this many pair files (0 = all). "
                         "Useful for smoke tests.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=200,
                    help="Print a progress line every N pair files.")
    args = ap.parse_args()

    if not args.pairs_dir.exists():
        sys.exit(f"[fatal] pairs dir not found: {args.pairs_dir}")

    rng = random.Random(args.seed)  # reserved for any future shuffling
    _ = rng                         # silence the "unused" linter
    stats = BuildStats()
    handles = _open_split_files(args.out_dir)

    seen_arxiv_ids: dict[str, str] = {}            # id -> split
    # (arxiv_id, table_idx) dedup. We bucket-by-arxiv_id so even if a
    # paper's pair files were processed independently (they aren't —
    # we dedup at the arxiv_id level above) we'd still reject repeats.
    seen_table_keys: set[tuple[str, int]] = set()
    # Per-arxiv_id table signatures, to skip identical replicated tables
    # within the same paper (e.g. boilerplate appendices).
    seen_signatures: dict[str, set[tuple]] = defaultdict(set)

    try:
        for i, pair_path in enumerate(iter_pair_files(args.pairs_dir)):
            if args.max_pairs and i >= args.max_pairs:
                break
            stats.n_pairs_seen += 1
            try:
                d = json.loads(pair_path.read_text(encoding="utf-8"))
            except Exception as exc:
                sys.stderr.write(f"[parse-error] {pair_path}: {exc}\n")
                stats.skip.parse_error += 1
                continue
            arxiv_id = d.get("arxiv_id")
            html_str = d.get("raw_source_html")
            if not html_str:
                continue
            stats.n_pairs_with_html += 1
            if not arxiv_id:
                continue
            stats.n_pairs_with_arxiv_id += 1

            # Per-arxiv_id dedup: pair files are sectioned (~5x duplication).
            if arxiv_id in seen_arxiv_ids:
                continue
            split = stable_split_for_id(arxiv_id, 0.80, 0.10)
            seen_arxiv_ids[arxiv_id] = split
            stats.n_unique_arxiv_ids_seen += 1

            source_pair = pair_path.stem
            for table_idx, elem in enumerate(_iter_tables(html_str)):
                stats.n_tables_inspected += 1

                key = (arxiv_id, table_idx)
                if key in seen_table_keys:
                    stats.skip.duplicate_table += 1
                    continue

                row = build_row_for_table(
                    table_elem=elem,
                    arxiv_id=arxiv_id,
                    source_pair=source_pair,
                    table_idx=table_idx,
                    skip=stats.skip,
                )
                if row is None:
                    continue

                # Within-paper signature dedup (cheap, defensive).
                # We rebuild the grid here is wasteful; instead, derive
                # a coarse signature from request_payload to avoid a
                # second extraction.
                rp = row["request_payload"]["payload"]["prompt"]
                # The payload is "SYSTEM:...\nUSER: <json>"; the JSON
                # has cell_grid which fully determines content.
                sig = (row["n_rows"], row["n_cols"],
                       hash(rp))  # hash of prompt is content-stable
                if sig in seen_signatures[arxiv_id]:
                    stats.skip.duplicate_table += 1
                    continue
                seen_signatures[arxiv_id].add(sig)

                seen_table_keys.add(key)
                row["split"] = split
                handles[split].write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                handles[split].flush()
                stats.n_rows_emitted += 1
                stats.rows_per_split[split] += 1
                stats.rows_per_arxiv_id[arxiv_id] += 1
                if row["extraction_metadata"]["has_caption"]:
                    stats.n_with_caption += 1
                if row["extraction_metadata"]["n_th_cells"] > 0:
                    stats.n_with_scope += 1
                if row["extraction_metadata"]["has_rowspan"]:
                    stats.n_with_rowspan += 1
                if row["extraction_metadata"]["has_colspan"]:
                    stats.n_with_colspan += 1
                stats.target_token_hist[
                    _bucket_for(row["n_target_tokens"], TOKEN_HIST_EDGES)
                ] += 1
                stats.n_rows_hist[
                    _bucket_for(row["n_rows"], ROW_HIST_EDGES)
                ] += 1
                stats.n_cols_hist[
                    _bucket_for(row["n_cols"], COL_HIST_EDGES)
                ] += 1

            if (i + 1) % args.progress_every == 0:
                total_skips = (
                    stats.skip.not_data_table
                    + stats.skip.pseudo_table
                    + stats.skip.too_small
                    + stats.skip.all_empty_cells
                    + stats.skip.duplicate_table
                )
                print(
                    f"[progress] pairs={i+1} "
                    f"unique_arxiv={stats.n_unique_arxiv_ids_seen} "
                    f"tables_seen={stats.n_tables_inspected} "
                    f"rows={stats.n_rows_emitted} "
                    f"skips={total_skips}",
                    flush=True,
                )
    finally:
        for h in handles.values():
            h.close()

    # Coverage report.
    n = max(1, stats.n_rows_emitted)
    coverage = {
        "schema_version": 1,
        "seed": args.seed,
        "pairs_dir": str(args.pairs_dir),
        "out_dir": str(args.out_dir),
        "n_pairs_seen": stats.n_pairs_seen,
        "n_pairs_with_html": stats.n_pairs_with_html,
        "n_pairs_with_arxiv_id": stats.n_pairs_with_arxiv_id,
        "n_unique_arxiv_ids_seen": stats.n_unique_arxiv_ids_seen,
        "n_tables_inspected": stats.n_tables_inspected,
        "n_rows_emitted": stats.n_rows_emitted,
        "rows_per_split": dict(stats.rows_per_split),
        "n_unique_arxiv_ids_in_dataset": len(stats.rows_per_arxiv_id),
        "rows_per_arxiv_id_top20": dict(
            stats.rows_per_arxiv_id.most_common(20)
        ),
        "pct_with_caption": round(100 * stats.n_with_caption / n, 2),
        "pct_with_th_scope": round(100 * stats.n_with_scope / n, 2),
        "pct_with_rowspan": round(100 * stats.n_with_rowspan / n, 2),
        "pct_with_colspan": round(100 * stats.n_with_colspan / n, 2),
        "target_token_hist": dict(sorted(stats.target_token_hist.items())),
        "n_rows_hist": dict(sorted(stats.n_rows_hist.items())),
        "n_cols_hist": dict(sorted(stats.n_cols_hist.items())),
        "skip": {
            "not_data_table": stats.skip.not_data_table,
            "pseudo_table": stats.skip.pseudo_table,
            "too_small": stats.skip.too_small,
            "all_empty_cells": stats.skip.all_empty_cells,
            "serialize_error": stats.skip.serialize_error,
            "parse_error": stats.skip.parse_error,
            "duplicate_table": stats.skip.duplicate_table,
        },
        "notes": {
            "split_function": (
                "data.common._splits.stable_split_for_id (re-implementation of "
                "scripts.datasets.build_semantic_preservation_dataset's). "
                "train_frac=0.80, val_frac=0.10. Same arxiv_id -> "
                "same split across all qwen_* and "
                "semantic_preservation builders."
            ),
            "ar5iv_table_filter": (
                "ar5iv emits LaTeX equation blocks as <table> too "
                "(class='ltx_eqn_*'); those are filtered into "
                "skip.not_data_table — they're for the math adapter, "
                "not the table adapter."
            ),
            "ar5iv_caption_handling": (
                "ar5iv puts data-table captions in a sibling "
                "<figcaption class='ltx_caption'> of a wrapping "
                "<figure class='ltx_table'>, NOT in a child <caption>. "
                "We hoist that into a generated <caption> on the target "
                "HTML, and pass the same text into build_table_request "
                "via a stub feature_blocks list."
            ),
            "ar5iv_th_scope": (
                "ar5iv <th> cells carry classes ('ltx_th_column' / "
                "'ltx_th_row') instead of HTML scope= attributes. We "
                "translate those classes into scope='col'/'row' on the "
                "target HTML side and into the role enum "
                "(header_col/header_row) on the request_payload side."
            ),
            "target_format": (
                "lxml.etree.tostring(<table>) of a freshly-built HTML5 "
                "<table> with optional <caption>, <thead>/<tbody> when "
                "header rows form a contiguous prefix, and <th scope=>."
            ),
        },
    }
    (args.out_dir / "coverage_report.json").write_text(
        json.dumps(coverage, indent=2)
    )

    print()
    print(f"[done] rows={stats.n_rows_emitted} "
          f"(train={stats.rows_per_split.get('train', 0)} "
          f"val={stats.rows_per_split.get('val', 0)} "
          f"test={stats.rows_per_split.get('test', 0)})")
    print(f"[done] tables_seen={stats.n_tables_inspected} skips="
          f"eqn:{stats.skip.not_data_table} "
          f"pseudo:{stats.skip.pseudo_table} "
          f"small:{stats.skip.too_small} "
          f"empty:{stats.skip.all_empty_cells} "
          f"dup:{stats.skip.duplicate_table} "
          f"err:{stats.skip.serialize_error}")
    print(f"[save] coverage report -> {args.out_dir / 'coverage_report.json'}")


if __name__ == "__main__":
    main()
