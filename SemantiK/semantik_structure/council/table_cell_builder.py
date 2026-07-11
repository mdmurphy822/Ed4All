"""Runtime cell-builder for BERT-TableSpecialist v5 (17-dim layout).

Bridges Stage 2's :class:`~semantik_structure.region_detection.TableCandidate`
output to the cell-level input contract that
:mod:`semantik_structure.council.table_specialist` expects:

    * ``text``    — already-formatted bracketed neighbor string, byte-
                    equal to what the eval harness builds:
                    ``[ABOVE] {a} [LEFT] {l} [CELL] {c} [RIGHT] {r} [BELOW] {b}``
                    (see ``scripts/eval_table_specialist_per_source.py``
                    ``_format_input``).
    * ``layout``  — 17-float vector matching the v5 training contract
                    frozen in ``models/council/table_specialist/final_v5/``.

Why the layout compute is inlined here rather than imported from
``data.build_table_specialist_data``:

    The data builder evolves with each retrain. This module must mirror
    whatever layout contract the runtime adapter loaded by
    :mod:`semantik_structure.council.table_specialist` was trained against
    (``DEFAULT_ADAPTER_DIR`` → ``final_v5``, ``layout_dim=17``). The two
    features ``is_top_2_rows``/``is_left_2_cols`` were added in v4 and
    carried into v5; an earlier copy of this module froze the v3 15-dim
    layout and pointed at ``final/`` — that drifted out of sync with the
    loaded ``final_v5`` adapter and raised ``cell layout has wrong dim:
    15 != 17`` at inference. Keep this contract == the loaded adapter's
    ``layout_dim``; if ``table_specialist.DEFAULT_ADAPTER_DIR`` is
    repointed to a new retrain, update both in lockstep.

The runtime cell is rectangular: ragged rows are right-padded with
``""`` so every ``(row_idx, col_idx)`` position has a valid cell. This
diverges from training (``data.build_table_specialist_data`` skips
``j >= len(row_h)`` in its inner loop) but is necessary at inference
time because we still need to emit a well-defined prediction for every
row/col position the assembler will reference.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from ..region_detection import TableCandidate


# ---------------------------------------------------------------------------
# v5 layout contract — frozen (must match table_specialist final_v5 layout_dim)
# ---------------------------------------------------------------------------

LAYOUT_FEATURE_NAMES_V5: tuple[str, ...] = (
    "row_idx_norm", "col_idx_norm", "n_rows_norm", "n_cols_norm",
    "is_first_row", "is_first_col", "is_last_row", "is_last_col",
    "is_top_2_rows", "is_left_2_cols",
    "text_len_log", "starts_upper", "ends_period",
    "has_digits", "digit_frac", "has_unit", "is_empty",
)
LAYOUT_FEATURE_DIM_V5 = len(LAYOUT_FEATURE_NAMES_V5)
assert LAYOUT_FEATURE_DIM_V5 == 17, LAYOUT_FEATURE_DIM_V5

# Back-compat aliases (older callers/tests imported the V3 spelling).
LAYOUT_FEATURE_NAMES_V3 = LAYOUT_FEATURE_NAMES_V5
LAYOUT_FEATURE_DIM_V3 = LAYOUT_FEATURE_DIM_V5


# v5 contract — frozen. Mirrors data/build_table_specialist_data.py
# compute_cell_layout regexes; do NOT edit unless the trained adapter is rebuilt.
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


def _compute_cell_layout_v5(
    *,
    row_idx: int,
    col_idx: int,
    n_rows: int,
    n_cols: int,
    text: str,
) -> list[float]:
    """v5 contract — frozen. 17-dim layout vector for a cell.

    Body mirrors ``data/build_table_specialist_data.py:compute_cell_layout``
    verbatim (LAYOUT_FEATURE_DIM == 17), including the v4/v5 additions
    ``is_top_2_rows`` / ``is_left_2_cols`` (positions 9-10).
    """
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
        (
            1.0 if text_stripped and text_stripped[0].isupper() else 0.0
        ),
        1.0 if text_stripped.endswith(".") else 0.0,
        1.0 if digit_count > 0 else 0.0,
        digit_count / text_len,
        1.0 if _UNIT_RE.search(text_stripped) else 0.0,
        1.0 if not text_stripped else 0.0,
    ]


# ---------------------------------------------------------------------------
# Neighbor helpers — mirror data/build_table_specialist_data.py:506-512
# ---------------------------------------------------------------------------


def _neighbor(rows: list[list[str]], i: int, j: int) -> str:
    """Out-of-bounds → ``""``; otherwise stripped, truncated to 60 chars.

    Mirrors ``data/build_table_specialist_data.py:_neighbor``.
    """
    if 0 <= i < len(rows) and 0 <= j < len(rows[i]):
        cell = rows[i][j] or ""
        return cell.strip()[:60]
    return ""


def _format_neighbored_text(rows: list[list[str]], i: int, j: int) -> str:
    """Build the bracketed [ABOVE]/[LEFT]/[CELL]/[RIGHT]/[BELOW] string.

    Format string matches ``scripts/eval_table_specialist_per_source.py``
    ``_format_input`` exactly (lines 92-96). This string IS the model
    input — any drift breaks parity with the trained adapter.
    """
    above = _neighbor(rows, i - 1, j)
    below = _neighbor(rows, i + 1, j)
    left = _neighbor(rows, i, j - 1)
    right = _neighbor(rows, i, j + 1)
    cell = _neighbor(rows, i, j)
    return (
        f"[ABOVE] {above} [LEFT] {left} "
        f"[CELL] {cell} "
        f"[RIGHT] {right} [BELOW] {below}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableCell:
    """Runtime cell, ready for ``run_bert("table_specialist", cells)``."""

    text: str            # already-formatted bracketed neighbor string
    layout: list[float]  # 17-dim, v5-frozen
    region_id: int       # cell index within source TableCandidate (row-major)
    table_idx: int       # which TableCandidate this came from
    row_idx: int
    col_idx: int


def build_cells(
    table: TableCandidate,
    *,
    table_idx: int = 0,
    min_rows: int = 2,
    min_cols: int = 2,
) -> list[TableCell]:
    """Materialize :class:`TableCell` rows for one table candidate.

    Returns ``[]`` for tables smaller than ``min_rows`` × ``min_cols``
    (matches the training-time size filter at
    ``data/build_table_specialist_data.py:574``).

    Ragged rows are right-padded with ``""`` so the cell index space is
    rectangular — see module docstring for why this differs from the
    training pipeline.
    """
    cell_grid: list[list[Any]] = list(table.cell_grid or [])
    n_rows = len(cell_grid)
    n_cols = max((len(r) for r in cell_grid), default=0)
    if n_rows < min_rows or n_cols < min_cols:
        return []

    # Pad ragged rows so every (i, j) inside the rectangle has a string.
    rows: list[list[str]] = []
    for row in cell_grid:
        padded = [
            (str(c) if c is not None else "").strip()
            for c in row
        ]
        if len(padded) < n_cols:
            padded = padded + [""] * (n_cols - len(padded))
        rows.append(padded)

    cells: list[TableCell] = []
    for i in range(n_rows):
        for j in range(n_cols):
            cell_text = rows[i][j]
            layout = _compute_cell_layout_v5(
                row_idx=i, col_idx=j,
                n_rows=n_rows, n_cols=n_cols,
                text=cell_text,
            )
            text_in = _format_neighbored_text(rows, i, j)
            cells.append(TableCell(
                text=text_in,
                layout=layout,
                region_id=len(cells),
                table_idx=table_idx,
                row_idx=i,
                col_idx=j,
            ))

    assert all(len(c.layout) == LAYOUT_FEATURE_DIM_V5 for c in cells), (
        "v5 layout contract violated"
    )
    return cells


def build_cells_for_set(
    tables: list[TableCandidate],
    *,
    min_rows: int = 2,
    min_cols: int = 2,
) -> list[TableCell]:
    """Convenience: enumerate ``tables`` and concatenate their cells."""
    out: list[TableCell] = []
    for k, tc in enumerate(tables):
        out.extend(build_cells(
            tc, table_idx=k,
            min_rows=min_rows, min_cols=min_cols,
        ))
    return out


__all__ = [
    "LAYOUT_FEATURE_DIM_V5",
    "LAYOUT_FEATURE_NAMES_V5",
    "LAYOUT_FEATURE_DIM_V3",  # back-compat alias
    "LAYOUT_FEATURE_NAMES_V3",  # back-compat alias
    "TableCell",
    "build_cells",
    "build_cells_for_set",
]
