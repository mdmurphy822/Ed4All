"""Build per-block classifier training data with richer extractor features.

v1 (build_classifier_data.py) parsed the flat `input_ocr` string from each
pair file and fuzzy-aligned its lines to HTML-tag ground truth. The
features it captured were the five flat flags (size, gap, top, centered,
caps).

v2 re-runs extract_shared() on the pair's source PDF and matches the
resulting merged blocks to the HTML ground truth. This adds:

  provenance      which extractor(s) produced the block
                  (pypdfium2 / pdfplumber / tesseract / "+"-combos)
  in_table        block falls inside a pdfplumber-detected table bbox
  in_header_row   block is in a row that pdfplumber calls header-like
  in_widget       block's bbox overlaps a pikepdf AcroForm widget
  widget_kind     text | checkbox | radio | select (if in_widget)

These flags let the classifier condition on structural signals it cannot
derive from OCR typography alone. In-table blocks should be strongly
biased toward TABLE_DATA_CELL / TABLE_HEADER_CELL; in-widget blocks
toward FORM_FIELD / FORM_LABEL.

Source routing:
  pair["local_pdf"]       exists (arXiv, PDF-form pairs) — use as-is
  else                    re-render pair["output_html"] via Playwright

Labels derive from the output_html side by DOM-walking (same logic as v1):
  h1..h6   -> heading      p        -> paragraph     li       -> list_item
  th       -> table_header_cell     td       -> table_data_cell
  caption  -> table_caption         figcaption -> figure_caption
  blockquote -> blockquote          pre      -> code_block
  label    -> form_label            input    -> form_field

Output JSONL matches v1 shape for backward compat, with extra fields.

Usage:
    python data/build_classifier_data_v2.py --workers 4
    python data/build_classifier_data_v2.py --pair-dirs data/synthetic/forms
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

from semantik_structure.extract_shared import extract_shared, extract_shared_cached
from semantik_structure.types import FeatureBlock, RawBlock
from semantik_structure.features import _caps_pattern, feature_block_to_classifier_input
from semantik_structure.text_utils import jaccard_overlap
from semantik_structure.validate import HtmlValidator
from semantik_structure.worker_pool import run_in_pool


# ---------- ground-truth label extraction ----------

TAG_TO_ROLE = {
    "h1": "heading",
    "h2": "heading",
    "h3": "heading",
    "h4": "heading",
    "h5": "heading",
    "h6": "heading",
    "p": "paragraph",
    "li": "list_item",
    "th": "table_header_cell",
    "td": "table_data_cell",
    "caption": "table_caption",
    "figcaption": "figure_caption",
    "blockquote": "blockquote",
    "pre": "code_block",
    "code": "code_block",
    "dt": "definition",
    "dd": "definition",
    "legend": "form_label",
    "label": "form_label",
    "input": "form_field",
    "select": "form_field",
    "textarea": "form_field",
}

OUTER_TAG_ALLOW_NESTED = {"th", "td", "li", "caption", "figcaption", "legend", "label"}


def extract_html_blocks(html: str) -> list[tuple[str, str]]:
    """Return [(text, role), ...] in document order from output_html."""
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[str, str]] = []
    for el in soup.find_all(list(TAG_TO_ROLE.keys())):
        # Skip blocks nested inside another target block unless they're table
        # cells / list items (which legitimately nest inside tables / lists).
        if el.name not in OUTER_TAG_ALLOW_NESTED:
            has_outer = any(
                p.name in TAG_TO_ROLE and p is not el
                for p in el.parents if p.name is not None
            )
            if has_outer:
                continue
        text = " ".join((el.get_text() or "").split()).strip()
        if not text:
            continue
        out.append((text, TAG_TO_ROLE[el.name]))
    return out


# ---------- feature derivation from shared JSON ----------

def flags_from_block(page: dict, block: dict, *,
                     median_size: float, median_gap: float,
                     prev_bottom: float | None) -> dict:
    """Compute feature flags for one merged block from its per-page context."""
    bbox = block["bbox"]
    x0, y0, x1, y1 = bbox
    page_w = float(page.get("width", 612.0))
    page_h = float(page.get("height", 792.0))

    # Use tesseract dims when the whole page is OCR.
    if (page.get("tesseract_width")
            and all("tesseract" in b.get("provenance", [])
                    for b in page["merged"]["text_blocks"])):
        page_w = float(page["tesseract_width"])
        page_h = float(page["tesseract_height"])

    # Size bucket
    font_size = block.get("font_size") or median_size or 1.0
    ratio = font_size / median_size if median_size else 1.0
    if ratio >= 1.8:
        size_bucket = "xl"
    elif ratio >= 1.3:
        size_bucket = "lg"
    elif ratio < 0.8:
        size_bucket = "sm"
    else:
        size_bucket = "md"

    # Gap above
    if prev_bottom is not None and median_gap:
        gap = max(0, y0 - prev_bottom)
        gap_flag = "lg" if gap >= 2 * median_gap else None
    else:
        gap_flag = None

    is_top = y0 < 0.15 * page_h

    line_w = x1 - x0
    mid_x = (x0 + x1) / 2
    is_centered = (line_w < 0.5 * page_w
                   and abs(mid_x - page_w / 2) < 0.05 * page_w)

    text = block["text"]
    caps = _caps_pattern(text)

    # in_table: block's center is inside a pdfplumber table bbox
    in_table = False
    in_header_row = False
    mid_y = (y0 + y1) / 2
    for t in page.get("pdfplumber", {}).get("tables", []):
        tb = t.get("bbox") or []
        if len(tb) == 4 and tb[0] <= mid_x <= tb[2] and tb[1] <= mid_y <= tb[3]:
            in_table = True
            rows = t.get("rows") or []
            # Heuristic: the first non-empty row is likely the header row.
            # If the block's y is in the top 25% of the table bbox, call it header.
            if rows and (mid_y - tb[1]) / max(1, tb[3] - tb[1]) < 0.25:
                in_header_row = True
            break

    # in_widget: block's center is inside a pikepdf widget rect
    in_widget = False
    widget_kind = None
    for w in page.get("pikepdf", {}).get("widgets", []):
        wb = w.get("bbox") or []
        if len(wb) == 4 and wb[0] <= mid_x <= wb[2] and wb[1] <= mid_y <= wb[3]:
            in_widget = True
            ft = (w.get("field_type") or "").lstrip("/")
            # PDF FT codes: Tx=text, Btn=button/checkbox/radio, Ch=select, Sig=signature
            if ft == "Tx":
                widget_kind = "text"
            elif ft == "Btn":
                widget_kind = "button"
            elif ft == "Ch":
                widget_kind = "select"
            elif ft == "Sig":
                widget_kind = "signature"
            break

    # Provenance — which extractor(s) contributed this block
    prov = block.get("provenance") or ["unknown"]
    prov_tag = "+".join(prov)

    return {
        "size_bucket": size_bucket,
        "gap_above": gap_flag,
        "is_top_of_page": is_top,
        "is_centered": is_centered,
        "caps": caps,
        "in_table": in_table,
        "in_header_row": in_header_row,
        "in_widget": in_widget,
        "widget_kind": widget_kind,
        "provenance": prov_tag,
    }


# ---------- serialization for the classifier ----------

def features_to_input_string(text: str, flags: dict) -> str:
    """Same string format the classifier already consumes, extended with
    the new in_table / in_widget / provenance / widget_kind flags."""
    chunks = []
    if flags["size_bucket"] != "md":
        chunks.append(f"size={flags['size_bucket']}")
    if flags["gap_above"] == "lg":
        chunks.append("gap=lg")
    if flags["is_top_of_page"]:
        chunks.append("top")
    if flags["is_centered"]:
        chunks.append("centered")
    if flags["caps"] in ("all", "title"):
        chunks.append(f"caps={flags['caps']}")
    if flags["in_table"]:
        chunks.append("in_table")
        if flags["in_header_row"]:
            chunks.append("header_row")
    if flags["in_widget"]:
        chunks.append("in_widget")
        if flags["widget_kind"]:
            chunks.append(f"widget={flags['widget_kind']}")
    # Provenance is usually pypdfium2 or pypdfium2+pdfplumber; only include
    # when it's something notable (Tesseract fallback or others).
    if flags["provenance"] not in ("pypdfium2", "pypdfium2+pdfplumber",
                                    "pdfplumber+pypdfium2"):
        chunks.append(f"prov={flags['provenance']}")
    if chunks:
        return f"[{' '.join(chunks)}] {text}"
    return text


# ---------- worker ----------

def process_pair(validator, work: tuple) -> dict:
    """Worker: re-extract features from a pair + produce labeled examples."""
    pair_path_str, out_examples_dir_str = work
    pair_path = Path(pair_path_str)
    out_dir = Path(out_examples_dir_str)
    stats = {"pair": pair_path.name, "aligned": 0, "total_blocks": 0,
             "error": None}

    try:
        pair = json.loads(pair_path.read_text())
    except Exception as exc:
        stats["error"] = f"read pair: {exc}"
        return stats

    output_html = pair.get("output_html")
    if not output_html:
        stats["error"] = "no output_html"
        return stats

    # HTML-side ground truth
    html_blocks = extract_html_blocks(output_html)
    if not html_blocks:
        stats["error"] = "no html blocks"
        return stats

    # Source PDF: use local_pdf when available (arXiv, PDF-form pairs);
    # otherwise re-render the output_html to a temp PDF via Playwright.
    local_pdf = pair.get("local_pdf")
    with tempfile.TemporaryDirectory() as tmp:
        if local_pdf and Path(local_pdf).exists():
            pdf_path = Path(local_pdf)
            tmp_pdf = None
            # Same PDF often appears in many section pairs (arXiv). Cache
            # to disk so the heavy pypdfium2+pdfplumber work is done once.
            try:
                shared = extract_shared_cached(pdf_path)
            except Exception as exc:
                stats["error"] = f"extract (cached): {exc}"
                return stats
        else:
            tmp_pdf = Path(tmp) / "rendered.pdf"
            try:
                validator.render_pdf(output_html, tmp_pdf)
            except Exception as exc:
                stats["error"] = f"render: {exc}"
                return stats
            pdf_path = tmp_pdf
            try:
                # Temp PDFs are unique per call — no cache benefit.
                shared = extract_shared(pdf_path)
            except Exception as exc:
                stats["error"] = f"extract: {exc}"
                return stats

    # Walk each page, align merged blocks to HTML blocks, emit examples.
    examples: list[dict] = []
    html_cursor = 0  # sliding window into html_blocks
    window = 8

    for page in shared.get("pages", []):
        merged = page.get("merged", {}).get("text_blocks", [])
        if not merged:
            continue

        # Per-page medians for feature derivation
        sizes = [b["font_size"] for b in merged
                 if b.get("font_size") is not None]
        median_size = sorted(sizes)[len(sizes) // 2] if sizes else 1.0
        # Compute median vertical gap
        merged_sorted = sorted(merged, key=lambda b: (b["bbox"][1], b["bbox"][0]))
        gaps = []
        for prev, cur in zip(merged_sorted, merged_sorted[1:]):
            gap = max(0, cur["bbox"][1] - prev["bbox"][3])
            if gap > 0:
                gaps.append(gap)
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else median_size

        prev_bottom = None
        for block in merged_sorted:
            stats["total_blocks"] += 1
            flags = flags_from_block(
                page, block,
                median_size=median_size,
                median_gap=median_gap,
                prev_bottom=prev_bottom,
            )
            prev_bottom = block["bbox"][3]

            # Align to ground-truth HTML block by text overlap.
            text = block["text"]
            best_idx = -1
            best_score = 0.0
            for j in range(max(0, html_cursor - 2),
                           min(len(html_blocks), html_cursor + window)):
                s = jaccard_overlap(text, html_blocks[j][0])
                if s > best_score:
                    best_score = s
                    best_idx = j
            if best_idx < 0 or best_score < 0.3:
                continue  # unaligned — skip

            role = html_blocks[best_idx][1]
            html_cursor = max(html_cursor, best_idx + 1)

            # Shortcut: in_table blocks whose best HTML match is h2/p/etc
            # often belong to the table anyway — the HTML-block extractor
            # may have walked past the table as a single unit. Trust the
            # in_table signal over the match.
            if flags["in_table"] and role not in ("table_header_cell",
                                                   "table_data_cell",
                                                   "table_caption"):
                # Only override when overlap is weakish — high-confidence
                # matches override the table heuristic.
                if best_score < 0.6:
                    role = "table_header_cell" if flags["in_header_row"] else "table_data_cell"

            examples.append({
                "input": features_to_input_string(text, flags),
                "label": role,
                "source": pair.get("source", "unknown"),
                "pair": pair_path.stem,
                "flags": flags,  # retain raw flags for potential model-head variants
            })
            stats["aligned"] += 1

    # Write this pair's examples into a per-pair file so pool-wise merging is safe.
    if examples:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pair_path.stem}.jsonl"
        with out_path.open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return stats


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dirs", type=Path, nargs="+",
                    default=[Path("data/pairs/wikipedia"),
                             Path("data/pairs/openstax"),
                             Path("data/pairs/federal_register"),
                             Path("data/pairs/gutenberg"),
                             Path("data/pairs/forms"),
                             Path("data/pairs/arxiv")])
    ap.add_argument("--examples-dir", type=Path,
                    default=Path("data/classifier_dataset_v2/per_pair"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/classifier_dataset_v2"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-per-source", type=int, default=None,
                    help="Cap on pair files processed per source (for smoke tests)")
    args = ap.parse_args()

    # Enumerate pair files (with optional per-source cap).
    pair_paths: list[Path] = []
    for d in args.pair_dirs:
        if not d.exists():
            continue
        files = sorted(d.glob("*.json"))
        if args.limit_per_source:
            files = files[:args.limit_per_source]
        pair_paths.extend(files)
    print(f"[plan] {len(pair_paths)} pair files across {len(args.pair_dirs)} dirs",
          file=sys.stderr)

    # Skip pairs we've already processed.
    existing = {p.stem for p in args.examples_dir.glob("*.jsonl")} \
        if args.examples_dir.exists() else set()
    pair_paths = [p for p in pair_paths if p.stem not in existing]
    print(f"[plan] {len(pair_paths)} to process ({len(existing)} done)",
          file=sys.stderr)

    # Run the pool.
    work_items = [(str(p), str(args.examples_dir)) for p in pair_paths]
    start = time.time()
    done = 0
    totals = {"aligned": 0, "total_blocks": 0, "errors": 0}
    for stats in run_in_pool(process_pair, work_items, workers=args.workers):
        done += 1
        if stats.get("error"):
            totals["errors"] += 1
            if done % 25 == 1:
                print(f"[{done}/{len(work_items)}] {stats['pair'][:50]} "
                      f"ERR {stats['error'][:60]}", file=sys.stderr)
        else:
            totals["aligned"] += stats["aligned"]
            totals["total_blocks"] += stats["total_blocks"]
            if done % 50 == 0 or done == len(work_items):
                rate = totals["aligned"] / max(1, totals["total_blocks"]) * 100
                print(f"[{done}/{len(work_items)}] aligned={totals['aligned']}  "
                      f"align_rate={rate:.1f}%  "
                      f"elapsed={(time.time() - start)/60:.1f}min",
                      file=sys.stderr)

    # Merge all per-pair JSONL into train/val/test splits.
    all_examples: list[dict] = []
    for p in sorted(args.examples_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                all_examples.append(json.loads(line))
    print(f"\n[merge] total labeled examples: {len(all_examples)}",
          file=sys.stderr)

    if not all_examples:
        raise SystemExit("no examples produced")

    random.Random(args.seed).shuffle(all_examples)
    n = len(all_examples)
    n_test = max(1, int(n * args.test_frac))
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val - n_test
    splits = {
        "train": all_examples[:n_train],
        "val":   all_examples[n_train:n_train + n_val],
        "test":  all_examples[n_train + n_val:],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                # Drop the raw flags from the serialized example —
                # the classifier consumes only `input` and `label`.
                r = {"input": row["input"], "label": row["label"],
                     "source": row["source"], "pair": row["pair"]}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[write] {path}  {len(rows)} rows", file=sys.stderr)

    # Label + source distributions.
    print("\n[by-label]")
    for lbl, n in Counter(ex["label"] for ex in all_examples).most_common():
        print(f"  {lbl:22} {n}")
    print("\n[by-source]")
    for src, n in Counter(ex["source"] for ex in all_examples).most_common():
        print(f"  {src:22} {n}")


if __name__ == "__main__":
    main()
