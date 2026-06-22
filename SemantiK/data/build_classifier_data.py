"""[DEPRECATED] v1 per-block classifier training data builder.

This module is superseded by `data/build_classifier_data_v2.py`, which
re-extracts via the shared multi-extractor stack and emits richer
per-block features (in_table, in_widget, provenance).

Kept in the tree for historical data reproduction. New work should
target v2. Scheduled for deletion once v2 training proves out — see
docs/bloat_audit.md.

Walks existing pair files under data/pairs/ and data/synthetic/. Each pair
has:
    input_ocr    — layout-annotated OCR, one block per line
    output_html  — the ground-truth accessible HTML

For each pair, we:
  1. Walk output_html to extract (text, role) tuples from structural tags
     (h1 -> HEADING, p -> PARAGRAPH, li -> LIST_ITEM, th -> TABLE_HEADER_CELL,
      td -> TABLE_DATA_CELL, caption -> TABLE_CAPTION, figcaption ->
      FIGURE_CAPTION, blockquote -> BLOCKQUOTE, pre -> CODE_BLOCK, etc).
  2. Walk input_ocr to extract per-line (text, flags) tuples.
  3. Fuzzy-match each input line to the ground-truth block whose text is
     closest — assign that block's role as the label.
  4. Emit training examples: {text, flags..., label}.

Unmatched lines are skipped. A rough expected yield is ~10-30 labeled
blocks per pair file.

Output is chat-free — the classifier is a text-classification task, not a
chat-completion task. Each example becomes a serialized string like:
    "[size=xl top caps=title] Document Title"
paired with its role label.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup


# HTML tag → classify.Role mapping. Keys may not be unique to one role
# (h1..h6 all map to HEADING; TITLE is reserved for the first h1-ish thing
# on the document).
TAG_TO_ROLE = {
    "h1": "heading",         # stage 4 decides if it's the doc title
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


# Regexes for parsing the flat input_ocr lines.
PAGE_LINE = re.compile(r"^PAGE\s+\d+\s*$")
FLAGS_LINE = re.compile(r"^\[([^\]]*)\]\s+(.+)$")


def iter_pair_files(dirs: list[Path]):
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                yield p, json.loads(p.read_text())
            except json.JSONDecodeError:
                continue


# ---------- HTML side ----------

def extract_html_blocks(html: str) -> list[tuple[str, str]]:
    """Return [(text, role), ...] in document order.

    Walks the DOM for any tag in TAG_TO_ROLE. Empty/whitespace-only blocks
    are skipped. Text is collapsed whitespace.
    """
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[str, str]] = []
    for el in soup.find_all(list(TAG_TO_ROLE.keys())):
        # Skip blocks nested inside another target block we'll emit (we
        # want the outermost semantic unit for matching).
        has_outer = any(
            p.name in TAG_TO_ROLE and p is not el
            for p in el.parents
            if p.name is not None
        )
        if has_outer and el.name not in ("th", "td", "li", "caption", "figcaption", "legend", "label"):
            # Headings + paragraphs inside bigger blocks — skip
            continue

        text = " ".join((el.get_text() or "").split()).strip()
        if not text:
            continue
        out.append((text, TAG_TO_ROLE[el.name]))
    return out


# ---------- OCR side ----------

def extract_ocr_lines(input_ocr: str) -> list[tuple[str, dict]]:
    """Return [(text, flags_dict), ...] from the flat input_ocr string."""
    out: list[tuple[str, dict]] = []
    for line in input_ocr.splitlines():
        line = line.rstrip()
        if not line or PAGE_LINE.match(line):
            continue
        m = FLAGS_LINE.match(line)
        if m:
            flags_str, text = m.group(1), m.group(2)
            flags = _parse_flags(flags_str)
        else:
            # Body line with implicit size=md, no other flags
            flags = {"size": "md"}
            text = line
        text = text.strip()
        if not text:
            continue
        out.append((text, flags))
    return out


def _parse_flags(flags_str: str) -> dict:
    """Parse "size=xl top centered caps=title" -> {size:xl, top:True, ...}."""
    flags: dict = {"size": "md"}  # default
    for tok in flags_str.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            flags[k] = v
        else:
            flags[tok] = True
    return flags


# ---------- alignment ----------

from dart_semantic.text_utils import jaccard_overlap as _text_overlap


def align(ocr_lines: list[tuple[str, dict]],
          html_blocks: list[tuple[str, str]],
          *, min_overlap: float = 0.3) -> list[tuple[dict, str, str]]:
    """For each OCR line, return (flags, text, role) if we can match it.

    Uses a sliding-window over html_blocks because they're in document
    order too — we prefer nearby matches over globally-best matches to
    avoid cross-section confusion.
    """
    result = []
    html_cursor = 0
    window = 5

    for text, flags in ocr_lines:
        best_idx = -1
        best_score = 0.0
        for j in range(max(0, html_cursor - 2),
                       min(len(html_blocks), html_cursor + window)):
            score = _text_overlap(text, html_blocks[j][0])
            if score > best_score:
                best_score = score
                best_idx = j
        if best_idx >= 0 and best_score >= min_overlap:
            role = html_blocks[best_idx][1]
            result.append((flags, text, role))
            # Slide the cursor forward when we consume a block.
            html_cursor = max(html_cursor, best_idx + 1)
    return result


# ---------- serialization ----------

def example_to_input_string(text: str, flags: dict) -> str:
    """Serialize (text, flags) into the classifier's input string.

    Matches the format the features module's features_to_prompt uses for
    non-default lines: "[size=xl top caps=title] text".
    """
    chunks = []
    size = flags.get("size", "md")
    if size != "md":
        chunks.append(f"size={size}")
    if flags.get("gap") == "lg":
        chunks.append("gap=lg")
    if flags.get("top"):
        chunks.append("top")
    if flags.get("centered"):
        chunks.append("centered")
    caps = flags.get("caps")
    if caps in ("all", "title"):
        chunks.append(f"caps={caps}")
    if chunks:
        return f"[{' '.join(chunks)}] {text}"
    return text


def main() -> None:
    import sys
    print(
        "WARNING: data/build_classifier_data.py is DEPRECATED.\n"
        "         Use data/build_classifier_data_v2.py, which emits the\n"
        "         richer feature format the current classifier trains on.\n",
        file=sys.stderr,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dirs", type=Path, nargs="+",
                    default=[Path("data/synthetic/forms"),
                             Path("data/pairs/wikipedia"),
                             Path("data/pairs/arxiv")])
    ap.add_argument("--out-dir", type=Path, default=Path("data/classifier_dataset"))
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-overlap", type=float, default=0.3,
                    help="Minimum Jaccard for OCR/HTML text match to accept")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples: list[dict] = []
    stats = {"pair_files": 0, "ocr_lines": 0, "html_blocks": 0,
             "aligned": 0, "unaligned": 0}
    source_counts: Counter = Counter()

    for path, pair in iter_pair_files(args.pair_dirs):
        stats["pair_files"] += 1
        input_ocr = pair.get("input_ocr") or ""
        output_html = pair.get("output_html") or ""
        if not input_ocr or not output_html:
            continue

        ocr_lines = extract_ocr_lines(input_ocr)
        html_blocks = extract_html_blocks(output_html)
        stats["ocr_lines"] += len(ocr_lines)
        stats["html_blocks"] += len(html_blocks)

        aligned = align(ocr_lines, html_blocks, min_overlap=args.min_overlap)
        stats["aligned"] += len(aligned)
        stats["unaligned"] += len(ocr_lines) - len(aligned)

        source = pair.get("source", "unknown")
        for flags, text, role in aligned:
            examples.append({
                "input": example_to_input_string(text, flags),
                "label": role,
                "source": source,
                "pair": path.stem,
            })
            source_counts[source] += 1

    if not examples:
        raise SystemExit("no labeled examples produced")

    random.Random(args.seed).shuffle(examples)
    n = len(examples)
    n_test = max(1, int(n * args.test_frac))
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val - n_test
    splits = {
        "train": examples[:n_train],
        "val":   examples[n_train:n_train + n_val],
        "test":  examples[n_train + n_val:],
    }

    for name, rows in splits.items():
        p = args.out_dir / f"{name}.jsonl"
        with p.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[write] {p}  {len(rows)} rows")

    print("\n[stats]")
    for k, v in stats.items():
        print(f"  {k:16} {v}")
    print(f"  align_rate        {stats['aligned']/max(1,stats['ocr_lines'])*100:.1f}%")

    print("\n[by-source]")
    for src, n in sorted(source_counts.items()):
        print(f"  {src:20} {n}")

    print("\n[by-label]")
    label_counts = Counter(ex["label"] for ex in examples)
    for lbl, n in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {lbl:22} {n}")


if __name__ == "__main__":
    main()
