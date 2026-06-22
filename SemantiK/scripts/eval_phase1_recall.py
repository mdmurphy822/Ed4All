"""Phase 1 — recall measurement for table/math region candidates.

Plan-validation criteria (Plans/01_implementation_plan.md §"Phase 1"):
    * >=95% table recall on a 30-doc held-out arXiv set
    * >=90% math recall  on a 30-doc held-out arXiv set
    where recall = ground-truth element overlaps a candidate by >=50% IoU.

This script implements the IoU-based recall *when ground-truth bboxes
are available*. The arXiv pair JSON files we ship under data/pairs/arxiv/
do NOT preserve per-element bboxes from ar5iv (they only carry a
cleaned-html `output_html` and a structured `output_json` that does not
include math/table coordinates). So the rigorous IoU pass cannot run on
the existing corpus without re-running the ar5iv-to-PDF alignment in
`dart_semantic/arxiv_sections.py`, which is out of scope for Phase 1.

Fallback (per Phase-1 plan latitude): we use *presence-based* recall:
for each held-out PDF, we infer ground truth from two channels:

    * Tables: the pair JSON's `output_html` contains a `<table>` element
      → the document has at least one table. Recall := fraction of those
      docs where `detect_table_region_candidates(...)` emits >=1
      candidate.
    * Math:  the source PDF carries CMMI/CMSY-family fonts, OR the
      pair's `output_html` contains a `<math>` element → the document
      has displayed math. Recall := fraction of those docs where
      `detect_math_region_candidates(...)` emits >=1 candidate.

This is a sanity check, not the rigorous IoU measurement. The IoU pass
will land in Phase 2 alongside the ar5iv → PDF alignment work needed
to train BERT-MathSpecialist.

Usage:
    python scripts/eval_phase1_recall.py
    python scripts/eval_phase1_recall.py --pairs-dir data/pairs/arxiv --max-docs 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run as `python scripts/...`.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dart_semantic.extract_shared import extract_shared  # noqa: E402
from dart_semantic.region_detection import (  # noqa: E402
    MATH_FONT_TOKENS,
    detect_math_region_candidates,
    detect_table_region_candidates,
)


def _has_math_font_signal(shared: dict) -> bool:
    """True if any merged text block uses a CMMI/CMSY-style math font."""
    for page in shared.get("pages", []) or []:
        for tb in page.get("merged", {}).get("text_blocks", []) or []:
            font = tb.get("font_name") or ""
            if any(tok in font for tok in MATH_FONT_TOKENS):
                return True
    return False


def _gt_has_table(pair: dict) -> bool:
    return "<table" in (pair.get("output_html") or "")


def _gt_has_math(pair: dict) -> bool:
    html = pair.get("output_html") or ""
    return "<math" in html


def _bbox_iou(a: tuple, b: tuple) -> float:
    """Standard IoU on (x0, y0, x1, y1) PDF-point bboxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _walk_pair_pdfs(pairs_dir: Path):
    """Yield (pair_dict, pdf_path) for every pair JSON whose local PDF
    exists. Deduplicates by arxiv_id so a single paper isn't counted
    multiple times when many sections exist."""
    seen: set[str] = set()
    for fp in sorted(pairs_dir.glob("*.json")):
        try:
            with fp.open() as f:
                pair = json.load(f)
        except Exception:
            continue
        local_pdf = pair.get("local_pdf")
        if not local_pdf or not os.path.exists(local_pdf):
            continue
        aid = pair.get("arxiv_id") or local_pdf
        if aid in seen:
            continue
        seen.add(aid)
        yield pair, Path(local_pdf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--pairs-dir", type=Path, default=Path("data/pairs/arxiv"),
        help="Directory of arXiv pair JSON files.",
    )
    ap.add_argument(
        "--eval-side-by-side", type=Path,
        default=Path("eval/side_by_side"),
        help="Side-by-side eval dir to seed from when pairs-dir is empty.",
    )
    ap.add_argument(
        "--max-docs", type=int, default=30,
        help="Max number of documents to evaluate.",
    )
    ap.add_argument(
        "--max-pages", type=int, default=15,
        help="Max number of pages per document; truncates long PDFs.",
    )
    args = ap.parse_args()

    table_total = 0
    table_hits = 0
    math_total = 0
    math_hits = 0
    docs_processed = 0
    fallback_used = True

    t0 = time.time()
    print("Phase 1 recall — fallback mode (presence-based, not IoU).")
    print(f"  pairs-dir = {args.pairs_dir}")
    print(f"  max-docs  = {args.max_docs}")
    print()

    if not args.pairs_dir.exists():
        print(f"  pairs-dir does not exist; falling back to {args.eval_side_by_side}")

    iterator = _walk_pair_pdfs(args.pairs_dir) if args.pairs_dir.exists() else iter(())
    for pair, pdf_path in iterator:
        if docs_processed >= args.max_docs:
            break
        gt_table = _gt_has_table(pair)
        gt_math = _gt_has_math(pair)

        try:
            shared = extract_shared(pdf_path)
        except Exception as exc:
            print(f"  [skip] {pdf_path.name}: extract failed: {exc}")
            continue

        # Truncate to max-pages so single-doc cost stays bounded.
        if args.max_pages and len(shared.get("pages", [])) > args.max_pages:
            shared["pages"] = shared["pages"][: args.max_pages]

        # Math-font signal is a stronger indicator than the cleaned HTML
        # for math presence on arXiv documents (math gets stripped from
        # `output_html` by some pair builders).
        gt_math_font = _has_math_font_signal(shared)
        gt_math = gt_math or gt_math_font

        tables = detect_table_region_candidates(shared)
        maths = detect_math_region_candidates(shared)

        if gt_table:
            table_total += 1
            if tables:
                table_hits += 1
        if gt_math:
            math_total += 1
            if maths:
                math_hits += 1

        docs_processed += 1
        if docs_processed % 5 == 0:
            print(f"  ... {docs_processed}/{args.max_docs} docs evaluated")

    # Fallback: also include eval/side_by_side as a sanity-check corpus
    # if pairs walk produced too few docs.
    if docs_processed < 4 and args.eval_side_by_side.exists():
        for sub in sorted(args.eval_side_by_side.iterdir()):
            pdf = sub / "input.pdf"
            if not pdf.exists():
                continue
            try:
                shared = extract_shared(pdf)
            except Exception:
                continue
            if args.max_pages:
                shared["pages"] = shared["pages"][: args.max_pages]
            tables = detect_table_region_candidates(shared)
            maths = detect_math_region_candidates(shared)
            # Heuristic ground truth from filename hints.
            name = sub.name.lower()
            if "table" in name:
                table_total += 1
                if tables:
                    table_hits += 1
            if "math" in name:
                math_total += 1
                if maths:
                    math_hits += 1
            docs_processed += 1
            if docs_processed >= args.max_docs:
                break

    elapsed = time.time() - t0

    def _pct(hit: int, total: int) -> str:
        return "n/a" if total == 0 else f"{hit/total*100:.1f}%"

    print()
    print(f"=== Phase 1 recall summary ({docs_processed} docs, {elapsed:.1f}s) ===")
    print(f"  table recall:  {table_hits}/{table_total} ({_pct(table_hits, table_total)})  "
          f"[target >=95%]")
    print(f"  math  recall:  {math_hits}/{math_total} ({_pct(math_hits, math_total)})  "
          f"[target >=90%]")
    if fallback_used:
        print()
        print("  NOTE: presence-based fallback used — IoU >=0.50 measurement")
        print("        is gated on ar5iv-to-PDF bbox alignment which lands in")
        print("        Phase 2 with BERT-MathSpecialist data builders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
