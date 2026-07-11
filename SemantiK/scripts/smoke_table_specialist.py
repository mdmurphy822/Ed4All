"""End-to-end smoke for the BERT-TableSpecialist runtime path.

Walks: PDF → ``extract_shared`` → ``detect_table_region_candidates`` →
``table_cell_builder.build_cells`` → ``run_bert("table_specialist", ...)``
and prints per-cell predictions plus an ASCII label grid per table.

Inline validation (4a + 4b in the spec):

    * Synthetic 3x3 grid → 9 cells, every layout 15-dim, every entry
      finite, ``(1,1).text`` contains ``[CELL]``.
    * Byte-equal parity with
      ``scripts.eval_table_specialist_per_source._format_input`` for
      cell ``(1, 1)`` of the synthetic grid.

Usage:
    .venv/bin/python scripts/smoke_table_specialist.py \\
        --pdf eval/side_by_side/tables_heavy_ml_v7/input.pdf
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path


# Repo root on sys.path so we can import semantik_structure.* and scripts.*
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


LABEL_LETTER = {
    "data": "d",
    "header_col": "H",
    "header_row": "R",
    "span": "S",
}


def _run_inline_validation() -> None:
    """4a + 4b: synthetic 3x3, parity check vs. eval _format_input."""
    from semantik_structure.council.table_cell_builder import (
        LAYOUT_FEATURE_DIM_V3, build_cells,
    )
    from semantik_structure.region_detection import TableCandidate
    from scripts.eval_table_specialist_per_source import _format_input

    grid = [
        ["", "Col A", "Col B"],
        ["Row 1", "1.0 kg", "2"],
        ["Row 2", "3", "4.5%"],
    ]
    fake_tc = TableCandidate(
        kind="table",
        bbox=(0.0, 0.0, 100.0, 50.0),
        pages=[1],
        evidence_features={"n_rows": 3, "n_cols": 3, "bordered": False},
        member_block_indices=[],
        cell_grid=grid,
        header_row_indices=[0],
        bordered=False,
    )
    cells = build_cells(fake_tc, table_idx=0)
    assert len(cells) == 9, f"expected 9 cells, got {len(cells)}"
    for c in cells:
        assert len(c.layout) == LAYOUT_FEATURE_DIM_V3, (
            f"cell ({c.row_idx},{c.col_idx}) layout dim "
            f"{len(c.layout)} != {LAYOUT_FEATURE_DIM_V3}"
        )
        for v in c.layout:
            assert math.isfinite(v), f"non-finite layout entry: {v}"
    middle = next(c for c in cells if c.row_idx == 1 and c.col_idx == 1)
    assert "[CELL]" in middle.text, middle.text

    # 4b: byte-equal parity with the eval harness for cell (1, 1).
    expected = _format_input({
        "text": "1.0 kg",
        "neighbors": {
            "above": "Col A",
            "below": "3",
            "left": "Row 1",
            "right": "2",
        },
    })
    if middle.text != expected:
        print("PARITY FAIL")
        print(f"  builder : {middle.text!r}")
        print(f"  expected: {expected!r}")
        raise SystemExit(1)
    print(
        f"[validate] 4a OK (9 cells, 15-dim, all finite); "
        f"4b OK (byte-equal parity with eval _format_input)"
    )


def _print_predictions(
    out, cells, *, max_print: int,
) -> None:
    label_counts: Counter = Counter()
    n = min(len(cells), max_print)
    print(f"  predictions (showing {n}/{len(cells)}):")
    for sig, cell in zip(out.signals[:n], cells[:n]):
        label = sig.top_k_labels[0]
        conf = sig.top_k_confidences[0]
        label_counts[label] += 1
        # Strip the bracket noise for human reading; we only show the
        # cell's own text.
        bare = cell.text.split("[CELL]", 1)[-1].split("[RIGHT]", 1)[0].strip()
        print(
            f"    ({cell.row_idx:2d},{cell.col_idx:2d}) "
            f"{bare[:40]!r:42s} -> {label:11s} {conf:.3f}"
        )
    # Always count the FULL set, not just printed.
    full = Counter(s.top_k_labels[0] for s in out.signals)
    print(f"  label counts (this table): {dict(full)}")


def _print_label_grid(out, cells) -> None:
    if not cells:
        return
    n_rows = max(c.row_idx for c in cells) + 1
    n_cols = max(c.col_idx for c in cells) + 1
    grid = [["?" for _ in range(n_cols)] for _ in range(n_rows)]
    for sig, cell in zip(out.signals, cells):
        letter = LABEL_LETTER.get(sig.top_k_labels[0], "?")
        grid[cell.row_idx][cell.col_idx] = letter
    print(f"  label grid ({n_rows}x{n_cols}):")
    for row in grid:
        print("    " + " ".join(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument(
        "--adapter-dir", type=Path,
        default=Path("models/council/table_specialist/final"),
    )
    ap.add_argument("--max-tables", type=int, default=5)
    ap.add_argument("--max-cells-print", type=int, default=30)
    args = ap.parse_args()

    print("=" * 70)
    print("inline validation")
    print("=" * 70)
    _run_inline_validation()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")
    if not args.adapter_dir.exists():
        raise SystemExit(f"adapter not found: {args.adapter_dir}")

    print()
    print("=" * 70)
    print(f"end-to-end smoke: {args.pdf}")
    print("=" * 70)

    from semantik_structure.council.runner import run_bert
    from semantik_structure.council.table_cell_builder import build_cells
    from semantik_structure.extract_shared import extract_shared_cached
    from semantik_structure.region_detection import detect_table_region_candidates

    print(f"[extract] {args.pdf}")
    shared = extract_shared_cached(args.pdf)
    n_pages = len(shared.get("pages", []) or [])
    print(f"[extract] {n_pages} pages")

    candidates = detect_table_region_candidates(shared)
    print(f"[detect] {len(candidates)} TableCandidate(s)")

    if not candidates:
        print("[smoke] no table candidates — nothing to run.")
        return

    total_cells = 0
    overall_labels: Counter = Counter()
    n_run = 0
    for k, tc in enumerate(candidates[:args.max_tables]):
        print(
            f"\n--- table {k}  page={tc.pages}  "
            f"shape={tc.evidence_features.get('n_rows')}x"
            f"{tc.evidence_features.get('n_cols')}  "
            f"bordered={tc.bordered} ---"
        )
        cells = build_cells(tc, table_idx=k)
        if not cells:
            print("  skipped (size filter: < 2x2)")
            continue
        n_run += 1
        out = run_bert("table_specialist", cells)
        total_cells += len(cells)
        for s in out.signals:
            overall_labels[s.top_k_labels[0]] += 1
        _print_predictions(out, cells, max_print=args.max_cells_print)
        _print_label_grid(out, cells)

    print()
    print("=" * 70)
    print("summary")
    print("=" * 70)
    print(f"  tables found              : {len(candidates)}")
    print(f"  tables run through bert   : {n_run}")
    print(f"  total cells predicted     : {total_cells}")
    print(f"  per-label counts (overall): {dict(overall_labels)}")


if __name__ == "__main__":
    main()
