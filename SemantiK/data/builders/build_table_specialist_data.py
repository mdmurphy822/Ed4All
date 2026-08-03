"""Build BERT-TableSpecialist (Phase 4 — task #52) training data.

Cell-level multi-head classifier on detector-confirmed table regions.
Sits downstream of Structure's ``table_region`` binary head: spans
flagged as ``table_region=1`` are aggregated into pdfplumber
``TableCandidate`` regions and then passed cell-by-cell to this BERT.

Heads (v1):
    * cell_role_scope     5-class    `data`, `header_col`, `header_row`,
                                     `header_both`, `span`
                                     Derived from HTML ground truth:
                                       <td>                       → data
                                       <th scope="col">           → header_col
                                       <th scope="row">           → header_row
                                       <th scope="colgroup">      → header_col
                                       <th scope="rowgroup">      → header_row
                                       <th> with no scope, in
                                         first column AND first row → header_both
                                       <th> with no scope, in
                                         first row only            → header_col
                                       <th> with no scope, in
                                         first col only            → header_row
                                       cell with rowspan/colspan>1
                                         AND not <th>              → span
                                     ar5iv: 100% of <th> carry a `scope`
                                     attribute (sample of 300 pairs:
                                     942/942 th, 28% caption coverage,
                                     0.7% spanned cells) so explicit
                                     `scope` rules dominate. The
                                     no-scope fallback is for OpenStax
                                     and other sources where authors
                                     elide the attribute.
    * caption_association  DEFER to v2 — needs caption-bbox + table
                          spatial proximity, not just per-cell.

Inputs (per cell):
    * text          — cell content (pdfplumber's extracted text for
                      that cell, NOT HTML — we want the inference-time
                      string, the HTML provides only the label).
    * layout        — small numeric vector:
                          row_idx_norm,        i / max(1, n_rows - 1)
                          col_idx_norm,        j / max(1, n_cols - 1)
                          n_rows_norm,         min(n_rows, 32) / 32
                          n_cols_norm,         min(n_cols, 16) / 16
                          is_first_row,        1 if i == 0 else 0
                          is_first_col,        1 if j == 0 else 0
                          is_last_row,         1 if i == n_rows-1 else 0
                          is_last_col,         1 if j == n_cols-1 else 0
                          text_len_log,        log1p(len(text)) / 6
                          starts_upper,        1 if text starts with [A-Z]
                          ends_period,         1 if text ends with `.`
                          has_digits,          1 if any digit in text
                          digit_frac,          fraction of chars that are digits
                          has_unit,            1 if matches \\b(kg|m|cm|s|%|ms|...)\\b
                                                — coarse signal that data cells often
                                                  carry numeric units
                          (15 dims, no font info — pdfplumber row-extract
                          does not retain per-cell font size; layout
                          features come from POSITION in the grid,
                          which is the strongest signal anyway.)
    * neighbors     — 4-tuple of (text_above, text_below, text_left,
                      text_right). Each truncated to 60 chars to keep
                      the input sequence tight.
    * cascade       — 8-dim Structure cascade (teacher-forced from a
                      run of the table_region-region's spans).
                      DEFERRED to a second-pass attach script —
                      mirrors how Phase 3c built Semantic data in two
                      phases. The v1 trainer can ignore the cascade
                      slot or zero-fill until the attach pass runs.

Source pipeline:
    Walk pair files. For each pair with a non-empty ``output_html``
    AND a usable ``local_pdf``:
        1. Parse HTML → list of HTMLTable objects, each a 2D grid
           of (cell_text, role, scope, rowspan, colspan).
        2. Run extract_shared_cached on the PDF → list of pdfplumber
           tables (each ``{"bbox": [...], "rows": [[...], ...]}``).
        3. Pair HTML tables to PDF tables greedily by row-count.
           If shapes mismatch (e.g., HTML 5x3 vs PDF 5x4), skip THAT
           pair but continue with other tables in the same doc.
        4. For each matched (HTML, PDF) pair, emit cells in row-major
           order with text from PDF and label from HTML.

Output layout:
    data/table_specialist_dataset/
        per_pair/<pair_stem>.jsonl
        train.jsonl  /  val.jsonl  /  test.jsonl
        coverage_report.json

Network usage: NONE — local pair files only.

Usage:
    python -m data.builders.build_table_specialist_data --workers 4
    python -m data.builders.build_table_specialist_data --pair-dirs data/pairs/arxiv \\
        --limit-per-source 100
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from html.parser import HTMLParser
from multiprocessing import Pool
from pathlib import Path

from data.common.balance import add_cap_args, apply_caps_and_report


# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------

CELL_ROLE_SCOPE_NAMES: tuple[str, ...] = (
    "data",  # 0
    "header_col",  # 1
    "header_row",  # 2
    "span",  # 3
)
CELL_ROLE_SCOPE_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CELL_ROLE_SCOPE_NAMES)}
# `header_both` (corner cell that labels both its row and its column)
# was in v0 but ar5iv tables always set explicit scope=col/row even on
# corner cells, and Wikipedia / OpenStax follow the same convention
# (sample build: 0/80,521 cells fell into this bucket). If a future
# source produces scopeless corner <th>, fall through to header_col by
# the position-based rule.

LAYOUT_FEATURE_NAMES: tuple[str, ...] = (
    "row_idx_norm",
    "col_idx_norm",
    "n_rows_norm",
    "n_cols_norm",
    "is_first_row",
    "is_first_col",
    "is_last_row",
    "is_last_col",
    "is_top_2_rows",
    "is_left_2_cols",
    "text_len_log",
    "starts_upper",
    "ends_period",
    "has_digits",
    "digit_frac",
    "has_unit",
    "is_empty",
)
LAYOUT_FEATURE_DIM = len(LAYOUT_FEATURE_NAMES)
assert LAYOUT_FEATURE_DIM == 17, LAYOUT_FEATURE_DIM


_UNIT_RE = re.compile(
    r"\b("
    r"kg|mg|ng|μg|ug|g|"
    r"km|cm|mm|nm|m|"
    r"hz|khz|mhz|ghz|"
    r"ms|μs|us|ns|s|min|hr|"
    r"k|mev|gev|ev|kev|"
    r"%|°|°c|°f"
    r")\b",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d")


def compute_cell_layout(
    *,
    row_idx: int,
    col_idx: int,
    n_rows: int,
    n_cols: int,
    text: str,
) -> list[float]:
    """Return the 17-dim layout vector for a cell."""
    text = text or ""
    text_stripped = text.strip()
    n_rows_safe = max(1, n_rows - 1)
    n_cols_safe = max(1, n_cols - 1)
    digit_count = len(_DIGIT_RE.findall(text_stripped))
    text_len = max(1, len(text_stripped))
    return [
        row_idx / n_rows_safe,
        col_idx / n_cols_safe,
        min(n_rows, 32) / 32.0,
        min(n_cols, 16) / 16.0,
        1.0 if row_idx == 0 else 0.0,
        1.0 if col_idx == 0 else 0.0,
        1.0 if row_idx == n_rows - 1 else 0.0,
        1.0 if col_idx == n_cols - 1 else 0.0,
        1.0 if row_idx <= 1 else 0.0,
        1.0 if col_idx <= 1 else 0.0,
        math.log1p(len(text_stripped)) / 6.0,
        (1.0 if text_stripped and text_stripped[0].isupper() else 0.0),
        1.0 if text_stripped.endswith(".") else 0.0,
        1.0 if digit_count > 0 else 0.0,
        digit_count / text_len,
        1.0 if _UNIT_RE.search(text_stripped) else 0.0,
        1.0 if not text_stripped else 0.0,
    ]


# ---------------------------------------------------------------------------
# HTML side — parse <table> into a 2D grid with role/scope per cell
# ---------------------------------------------------------------------------


class HTMLTable:
    """Materialised table grid: rows × cols of dicts.

    grid[i][j] = {
        "text": str,
        "role": "th" | "td",
        "scope": "col" | "row" | "colgroup" | "rowgroup" | None,
        "rowspan": int,
        "colspan": int,
        "is_origin": bool,        # True for the top-left of a span; False
                                  # for replicated cells
    }
    """

    def __init__(self) -> None:
        self.grid: list[list[dict]] = []
        self.caption: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self.grid)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.grid), default=0)

    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_cols)


class _TableExtractor(HTMLParser):
    """Pull tables out of an HTML stream. Best-effort — handles nested
    <table> by recording depth, collapses `<thead>`/`<tbody>`/`<tfoot>`,
    expands rowspan/colspan."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[HTMLTable] = []
        self._stack: list[HTMLTable] = []
        self._row: list[dict] | None = None
        self._cell: dict | None = None
        self._cell_text: list[str] = []
        self._in_caption = False
        self._caption_text: list[str] = []
        # Track open spans so we can replicate cells across rows
        self._open_rowspans: dict[int, dict] = {}  # col_idx → cell template

    def _current_table(self) -> HTMLTable | None:
        return self._stack[-1] if self._stack else None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        d = {k: v for k, v in attrs}
        t = self._current_table()
        if tag == "table":
            self._stack.append(HTMLTable())
        elif tag == "caption" and t is not None:
            self._in_caption = True
            self._caption_text = []
        elif tag == "tr" and t is not None:
            self._row = []
            self._open_rowspans_carry()
        elif tag in ("th", "td") and t is not None and self._row is not None:
            try:
                rowspan = int(d.get("rowspan", "1") or "1")
            except ValueError:
                rowspan = 1
            try:
                colspan = int(d.get("colspan", "1") or "1")
            except ValueError:
                colspan = 1
            scope = d.get("scope") or None
            scope = scope.lower() if isinstance(scope, str) else None
            self._cell = {
                "role": tag,
                "scope": scope
                if scope
                in (
                    "col",
                    "row",
                    "colgroup",
                    "rowgroup",
                )
                else None,
                "rowspan": max(1, rowspan),
                "colspan": max(1, colspan),
                "is_origin": True,
            }
            self._cell_text = []

    def handle_endtag(self, tag: str) -> None:
        t = self._current_table()
        if t is None and tag != "table":
            return
        if tag == "table" and self._stack:
            done = self._stack.pop()
            self.tables.append(done)
        elif tag == "caption":
            self._in_caption = False
            if t is not None:
                t.caption = " ".join(self._caption_text).strip() or None
        elif tag == "tr" and self._row is not None and t is not None:
            t.grid.append(self._row)
            self._row = None
        elif tag in ("th", "td") and self._cell is not None and self._row is not None:
            text = " ".join(self._cell_text).strip()
            base = dict(self._cell, text=text)
            colspan = base["colspan"]
            rowspan = base["rowspan"]
            self._row.append(base)
            for _ in range(colspan - 1):
                clone = dict(base, is_origin=False)
                self._row.append(clone)
            if rowspan > 1:
                # Anchor at next column
                start_col = len(self._row) - colspan
                for col_offset in range(colspan):
                    self._open_rowspans[start_col + col_offset] = dict(
                        base, is_origin=False, rowspan=rowspan - 1
                    )
            self._cell = None
            self._cell_text = []

    def handle_data(self, data: str) -> None:
        if self._in_caption:
            self._caption_text.append(data)
        elif self._cell is not None:
            self._cell_text.append(data)

    def _open_rowspans_carry(self) -> None:
        """At the start of a new <tr>, prepend any cells still open from
        a prior rowspan. Decrement remaining rowspan; drop when 0."""
        if not self._open_rowspans or self._row is None:
            return
        new_open: dict[int, dict] = {}
        max_col = max(self._open_rowspans.keys()) if self._open_rowspans else -1
        for col in range(max_col + 1):
            cell = self._open_rowspans.get(col)
            if cell is not None:
                self._row.append(dict(cell, is_origin=False))
                rem = cell["rowspan"] - 1
                if rem >= 1:
                    new_open[col] = dict(cell, rowspan=rem)
            else:
                # No carry; row will be filled by normal handle_starttag
                # for new <td>/<th> entries.
                # We append a placeholder which downstream code can
                # detect and skip if no real cell lands at this column.
                pass
        self._open_rowspans = new_open


def parse_html_tables(html: str) -> list[HTMLTable]:
    """Extract every <table> from an HTML stream. Returns an empty list
    if parsing fails."""
    if not html or "<table" not in html.lower():
        return []
    p = _TableExtractor()
    try:
        p.feed(html)
    except Exception:
        return []
    return [t for t in p.tables if t.n_rows > 0 and t.n_cols > 0]


# ---------------------------------------------------------------------------
# Label assignment — turn HTML cell role/scope into the 5-class label
# ---------------------------------------------------------------------------


def assign_cell_label(
    cell: dict,
    *,
    row_idx: int,
    col_idx: int,
    n_rows: int,
    n_cols: int,
) -> str:
    """Return one of CELL_ROLE_SCOPE_NAMES for an HTML cell.

    Spanned non-origin cells (replicas) propagate the origin's label.
    `span` is reserved for non-`<th>` cells with rowspan/colspan > 1.
    """
    if not cell.get("is_origin", True):
        # Non-origin: replicate label from origin handled by caller — for
        # safety, label as `span` if we got here without an origin pass.
        if cell.get("role") == "th":
            # Header span — collapse to whichever scope the origin had.
            scope = cell.get("scope")
            if scope in ("col", "colgroup"):
                return "header_col"
            if scope in ("row", "rowgroup"):
                return "header_row"
            return "header_col" if row_idx == 0 else "header_row"
        return "span"

    role = cell.get("role")
    scope = cell.get("scope")
    rowspan = cell.get("rowspan", 1)
    colspan = cell.get("colspan", 1)
    spanned = (rowspan > 1) or (colspan > 1)

    if role == "th":
        if scope in ("col", "colgroup"):
            return "header_col"
        if scope in ("row", "rowgroup"):
            return "header_row"
        # No-scope <th> — infer from position. Corner cell ([0,0]) and
        # rest-of-first-row → header_col; first-col-only → header_row;
        # mid-table scopeless <th> → conservative header_col.
        first_row = row_idx == 0
        first_col = col_idx == 0
        if first_row:
            return "header_col"
        if first_col:
            return "header_row"
        return "header_col"
    # <td>
    if spanned:
        return "span"
    return "data"


# ---------------------------------------------------------------------------
# PDF side — reuse extract_shared_cached for pdfplumber tables
# ---------------------------------------------------------------------------


def _pdf_tables_from_shared(shared: dict) -> list[dict]:
    """Flatten shared.pages[*].pdfplumber.tables into one ordered list,
    annotating each with its page number and a sequential index."""
    out: list[dict] = []
    for page in shared.get("pages", []):
        page_num = int(page.get("page_num", page.get("number", 0)) or 0)
        for t in page.get("pdfplumber", {}).get("tables", []) or []:
            rows = t.get("rows") or []
            if not rows:
                continue
            n_rows = len(rows)
            n_cols = max((len(r) for r in rows), default=0)
            if n_cols == 0:
                continue
            out.append(
                {
                    "page": page_num,
                    "bbox": t.get("bbox") or [],
                    "rows": rows,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                }
            )
    return out


# ---------------------------------------------------------------------------
# HTML ↔ PDF table alignment
# ---------------------------------------------------------------------------


def align_html_to_pdf_tables(
    html_tables: list[HTMLTable],
    pdf_tables: list[dict],
) -> list[tuple[HTMLTable, dict]]:
    """Greedy 1:1 alignment by exact (n_rows, n_cols) match first, then
    by (n_rows, |Δcols|<=1).

    Skips ar5iv metadata pseudo-tables (Comments:/Cite as:) by detecting
    the keyword 'Cite as:' in the first column of small (≤4-row) tables;
    those don't appear in the rendered PDF and would otherwise
    contaminate the dataset.
    """
    # Filter out ar5iv pseudo-tables from the HTML side
    html_real: list[HTMLTable] = []
    for ht in html_tables:
        if ht.n_rows <= 4:
            first_col_text = " ".join(
                (row[0]["text"] if row and row[0].get("is_origin", True) else "") for row in ht.grid
            ).lower()
            if (
                "cite as:" in first_col_text
                or "doi:" in first_col_text
                or "subjects:" in first_col_text
            ):
                continue
        html_real.append(ht)

    used_pdf: set[int] = set()
    aligned: list[tuple[HTMLTable, dict]] = []

    # Pass 1: exact shape match
    for ht in html_real:
        for k, pt in enumerate(pdf_tables):
            if k in used_pdf:
                continue
            if pt["n_rows"] == ht.n_rows and pt["n_cols"] == ht.n_cols:
                aligned.append((ht, pt))
                used_pdf.add(k)
                break

    # Pass 2: same n_rows, |Δcols| ≤ 1
    for ht in html_real:
        if any(ht is a[0] for a in aligned):
            continue
        for k, pt in enumerate(pdf_tables):
            if k in used_pdf:
                continue
            if pt["n_rows"] == ht.n_rows and abs(pt["n_cols"] - ht.n_cols) <= 1:
                aligned.append((ht, pt))
                used_pdf.add(k)
                break
    return aligned


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------


def _neighbor(rows: list[list[str]], i: int, j: int) -> str:
    if 0 <= i < len(rows) and 0 <= j < len(rows[i]):
        cell = rows[i][j] or ""
        return cell.strip()[:60]
    return ""


def process_pair(args_tuple: tuple[str, str]) -> dict:
    """Worker entry: extract tables from one pair, write per-pair JSONL.

    v1 walks HTML tables only — pdfplumber is too unreliable on
    LaTeX-typeset arxiv tables (most cells extract as empty), and the
    HTML side already contains everything we need for the labels and
    cell text. PDF/pdfplumber alignment can be added as a v2
    augmentation pass to cross-validate cell text with what the
    inference pipeline will see.
    """
    pair_path_str, examples_dir_str = args_tuple
    pair_path = Path(pair_path_str)
    examples_dir = Path(examples_dir_str)

    stats = {
        "pair": pair_path.stem,
        "tables_html": 0,
        "tables_kept": 0,
        "cells_emitted": 0,
        "tables_dropped_pseudo": 0,
        "tables_dropped_size": 0,
        "tables_dropped_sparse": 0,
        "error": None,
    }

    try:
        pair = json.loads(pair_path.read_text())
    except Exception as exc:  # noqa: BLE001
        stats["error"] = f"json: {exc}"
        return stats

    html = pair.get("output_html") or ""
    if not html:
        html = pair.get("raw_source_html") or ""
    if pair.get("raw_source_xml"):
        # PMC pairs store tables as JATS XML, not HTML — their output_html
        # carries no <table> at all. Normalize each <table-wrap> to the same
        # accessible HTML5 <table> the other sources emit (thead <td>→<th
        # scope=col>, graphic-only skipped) and append, so parse_html_tables
        # sees them in one HTML stream. Shared with the Qwen-table adapter
        # builder (Plans/06, plan b).
        from data.sources.jats_tables import jats_to_html5_tables

        jats = "\n".join(tbl for tbl, _cap in jats_to_html5_tables(pair["raw_source_xml"]))
        html = f"{html}\n{jats}" if html else jats
    if pair.get("raw_source_xml") and "gpotable" in pair["raw_source_xml"].lower():
        # CFR / Federal Register store tables as GPO GPOTABLE XML. Normalize
        # to the same accessible HTML5 (regulatory table_type). Shared with
        # the Qwen-table adapter builder (Plans/06).
        from data.sources.gpotable_tables import gpotable_to_html5_tables

        gpo = "\n".join(tbl for tbl, _cap in gpotable_to_html5_tables(pair["raw_source_xml"]))
        html = f"{html}\n{gpo}" if html else gpo

    html_tables = parse_html_tables(html)
    stats["tables_html"] = len(html_tables)
    if not html_tables:
        return stats

    examples: list[dict] = []
    for table_idx, ht in enumerate(html_tables):
        # Pseudo-table filter: ar5iv emits a "Cite as:" / "DOI:" /
        # "Subjects:" key-value table per paper that doesn't appear in
        # the rendered PDF. Drop those.
        if ht.n_rows <= 4:
            first_col_text = " ".join(
                (row[0]["text"] if row and row[0].get("is_origin", True) else "") for row in ht.grid
            ).lower()
            if any(
                k in first_col_text
                for k in (
                    "cite as:",
                    "doi:",
                    "subjects:",
                )
            ):
                stats["tables_dropped_pseudo"] += 1
                continue

        n_rows = ht.n_rows
        n_cols = ht.n_cols
        # Min size filter — a "table" carries cell-role signal only at
        # 2x2+. 1xN / Nx1 are typically figure groupings.
        if n_rows < 2 or n_cols < 2:
            stats["tables_dropped_size"] += 1
            continue

        # Sparse filter — drop tables where < 50% of cells have text
        non_empty = 0
        total_cells = 0
        for ii in range(n_rows):
            row_h = ht.grid[ii] if ii < len(ht.grid) else []
            for jj in range(min(n_cols, len(row_h))):
                total_cells += 1
                if (row_h[jj].get("text") or "").strip():
                    non_empty += 1
        if total_cells == 0 or non_empty / total_cells < 0.5:
            stats["tables_dropped_sparse"] += 1
            continue

        stats["tables_kept"] += 1
        for i in range(n_rows):
            html_row = ht.grid[i] if i < len(ht.grid) else []
            for j in range(n_cols):
                if j >= len(html_row):
                    continue
                cell_html = html_row[j]
                text = (cell_html.get("text") or "").strip()
                pdf_text = ""

                label = assign_cell_label(
                    cell_html,
                    row_idx=i,
                    col_idx=j,
                    n_rows=n_rows,
                    n_cols=n_cols,
                )
                layout = compute_cell_layout(
                    row_idx=i,
                    col_idx=j,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    text=text,
                )

                # Neighbors come from the HTML grid (matches the cell
                # text we're using for `text`).
                def _h(ii: int, jj: int) -> str:
                    if 0 <= ii < n_rows:
                        rr = ht.grid[ii] if ii < len(ht.grid) else []
                        if 0 <= jj < len(rr):
                            return (rr[jj].get("text") or "").strip()[:60]
                    return ""

                neighbors = {
                    "above": _h(i - 1, j),
                    "below": _h(i + 1, j),
                    "left": _h(i, j - 1),
                    "right": _h(i, j + 1),
                }

                examples.append(
                    {
                        "text": text[:300],
                        "layout": layout,
                        "neighbors": neighbors,
                        "labels": {
                            "cell_role_scope": CELL_ROLE_SCOPE_TO_ID[label],
                        },
                        "table_idx": table_idx,
                        "row_idx": i,
                        "col_idx": j,
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        "source": pair.get("source", "unknown"),
                        "pair": pair_path.stem,
                    }
                )

    if examples:
        examples_dir.mkdir(parents=True, exist_ok=True)
        out_path = examples_dir / f"{pair_path.stem}.jsonl"
        with out_path.open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    stats["cells_emitted"] = len(examples)
    return stats


# ---------------------------------------------------------------------------
# Stratified split + main
# ---------------------------------------------------------------------------


def run_in_pool(fn, items, *, workers: int):
    if workers <= 1:
        for it in items:
            yield fn(it)
        return
    with Pool(workers) as pool:
        for r in pool.imap_unordered(fn, items, chunksize=4):
            yield r


def stratified_split(
    examples: list[dict],
    *,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[list[dict], list[dict], list[dict]]:
    import random

    by_label: dict[int, list[dict]] = defaultdict(list)
    for ex in examples:
        by_label[ex["labels"]["cell_role_scope"]].append(ex)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for k, lst in by_label.items():
        rng.shuffle(lst)
        n = len(lst)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(lst[:n_train])
        val.extend(lst[n_train : n_train + n_val])
        test.extend(lst[n_train + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pair-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("data/pairs/arxiv"),
            Path("data/pairs/openstax"),
            Path("data/pairs/pmc"),
            Path("data/pairs/cfr"),
            Path("data/pairs/federal_register"),
            Path("data/pairs/nces_digest"),
            Path("data/pairs/wikipedia"),
            Path("data/pairs/forms"),
        ],
    )
    ap.add_argument(
        "--examples-dir", type=Path, default=Path("data/table_specialist_dataset/per_pair")
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/table_specialist_dataset"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-per-source", type=int, default=None)
    add_cap_args(ap)
    args = ap.parse_args()

    pair_paths: list[Path] = []
    for d in args.pair_dirs:
        if not d.exists():
            continue
        files = sorted(d.glob("*.json"))
        if args.limit_per_source:
            files = files[: args.limit_per_source]
        pair_paths.extend(files)
    print(f"[plan] {len(pair_paths)} pair files across {len(args.pair_dirs)} dirs", file=sys.stderr)

    existing = (
        {p.stem for p in args.examples_dir.glob("*.jsonl")} if args.examples_dir.exists() else set()
    )
    pair_paths = [p for p in pair_paths if p.stem not in existing]
    print(f"[plan] {len(pair_paths)} to process ({len(existing)} done)", file=sys.stderr)

    work_items = [(str(p), str(args.examples_dir)) for p in pair_paths]
    start = time.time()
    done = 0
    totals = {
        "tables_html": 0,
        "tables_kept": 0,
        "cells_emitted": 0,
        "errors": 0,
        "tables_dropped_pseudo": 0,
        "tables_dropped_size": 0,
        "tables_dropped_sparse": 0,
    }
    for stats in run_in_pool(process_pair, work_items, workers=args.workers):
        done += 1
        if stats.get("error"):
            totals["errors"] += 1
            if done % 25 == 1:
                print(
                    f"[{done}/{len(work_items)}] {stats['pair'][:50]} ERR {stats['error'][:60]}",
                    file=sys.stderr,
                )
        else:
            for k in (
                "tables_html",
                "tables_kept",
                "cells_emitted",
                "tables_dropped_pseudo",
                "tables_dropped_size",
                "tables_dropped_sparse",
            ):
                totals[k] += stats.get(k, 0)
        if done % 100 == 0 or done == len(work_items):
            print(
                f"[{done}/{len(work_items)}] "
                f"html={totals['tables_html']} kept={totals['tables_kept']} "
                f"cells={totals['cells_emitted']} "
                f"drop(pseudo={totals['tables_dropped_pseudo']} "
                f"size={totals['tables_dropped_size']} "
                f"sparse={totals['tables_dropped_sparse']}) "
                f"err={totals['errors']} "
                f"elapsed={(time.time() - start) / 60:.1f}min",
                file=sys.stderr,
            )

    # Merge per-pair output → train/val/test split
    # NB: read line-by-line via `open()` rather than splitlines() — table
    # cell text can contain Unicode line separators (U+2028) which
    # json.dumps(ensure_ascii=False) does NOT escape but splitlines()
    # DOES split on, corrupting the JSON.
    all_examples: list[dict] = []
    for p in sorted(args.examples_dir.glob("*.jsonl")):
        with p.open() as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.strip():
                    all_examples.append(json.loads(line))
    print(f"\n[merge] total cells: {len(all_examples)}", file=sys.stderr)

    if not all_examples:
        raise SystemExit("no cells produced")

    # Opt out of rare-label protection: the table_specialist trainer's
    # weight_cap=15 is tuned for the natural span~3.5% distribution, so
    # protecting rare cell_role_scope labels here would double-correct. A
    # --cap-protect-frac>0 now raises rather than being silently ignored.
    all_examples, cap_report = apply_caps_and_report(
        all_examples, args, label_key="cell_role_scope", allow_label_protection=False
    )

    train, val, test = stratified_split(all_examples, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[write] {path}  {len(rows)} rows", file=sys.stderr)

    # Coverage report
    label_counts = Counter(r["labels"]["cell_role_scope"] for r in all_examples)
    source_counts = Counter(r["source"] for r in all_examples)
    print("\n[balance] cell_role_scope:")
    for k, name in enumerate(CELL_ROLE_SCOPE_NAMES):
        n = label_counts.get(k, 0)
        print(f"  {name:14s} {n}")
    print("\n[balance] by source:")
    for s, n in source_counts.most_common():
        print(f"  {s:30s} {n}")

    coverage = {
        "n_cells_total": len(all_examples),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "cell_role_scope_counts": {CELL_ROLE_SCOPE_NAMES[k]: v for k, v in label_counts.items()},
        "source_counts": dict(source_counts),
        "source_caps": cap_report,
        "max_examples_per_source": args.max_examples_per_source,
        "layout_feature_dim": LAYOUT_FEATURE_DIM,
        "layout_feature_names": list(LAYOUT_FEATURE_NAMES),
        "cell_role_scope_names": list(CELL_ROLE_SCOPE_NAMES),
        "totals": totals,
    }
    (args.out_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2))
    print(f"[write] {args.out_dir / 'coverage_report.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
