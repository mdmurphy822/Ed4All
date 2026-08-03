"""Rewrite v3 dataset targets so EVERY block has a label.

Problem: build_chunk_example dropped unaligned blocks from the assistant
target (median coverage ~3%). The system prompt says "label every block"
but the model only ever saw sparse targets, so at inference it doesn't
know what to emit for unaligned blocks and collapses.

Fix: for each training row, parse the user prompt's per-block `h=<code>`
hint (the DistilBERT prediction embedded in the input), and for every
block id not already present in the assistant target, add an entry with
the hint as the label. This matches the design where Qwen agrees with
DistilBERT by default and overrides when it has stronger evidence.

Input:  data/qwen_dataset_v3/{train,val,test}.jsonl  (sparse targets)
Output: data/qwen_dataset_v3d/{train,val,test}.jsonl  (dense targets)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Hint code -> full role name (inverse of reason_schema.HINT_CODES).
HINT_CODES = {
    "ti": "title",
    "hd": "heading",
    "pa": "paragraph",
    "ls": "list",
    "li": "list_item",
    "tb": "table",
    "tr": "table_row",
    "thc": "table_header_cell",
    "tdc": "table_data_cell",
    "tbc": "table_caption",
    "fg": "figure",
    "fgc": "figure_caption",
    "bq": "blockquote",
    "cb": "code_block",
    "ff": "form_field",
    "fl": "form_label",
    "rf": "reference",
    "fn": "footnote",
    "df": "definition",
    "mt": "metadata",
    "ph": "page_header",
    "pf": "page_footer",
}

BLOCK_RE = re.compile(r"^\s*\[[^\]]*\]")   # a block line starts with [...]
TEXT_AFTER_FLAGS_RE = re.compile(r"^\s*\[[^\]]*\]\s*(.*)$")
# TOC tail: long line ending in "... 24" or "5  Title  37" (text + trailing
# page number). Catches both leader-dot TOC formatting and bare "Foo 5"
# layouts.
TOC_TAIL_RE = re.compile(r"\.\s*\d+\s*$|\s\d+\s*$")
# Page running header: short title-cased phrase ending in a 1-3 digit number,
# typical of "Shades of Accessibility 1" / "Quantum 4" running headers. Has
# no length floor — the pattern itself is selective enough.
PAGE_RUN_HDR_RE = re.compile(r"^[A-Z][A-Za-z][A-Za-z\s]{1,40}\s\d{1,3}$")


def _block_text(line: str) -> str:
    """Extract the text portion of a block line (after the [flags])."""
    m = TEXT_AFTER_FLAGS_RE.match(line)
    return m.group(1) if m else ""


def _heading_sanity_demote(hint: str, text: str) -> str:
    """Return a safer label when the classifier hint is `heading` but the
    block text obviously isn't one.

    The v7 reasoner was trained against densified targets where the
    densify step took the classifier hint at face value. Classifier_v5
    over-predicts heading on body fragments (lowercase-start text,
    sentence-length text, very long text), so those filled-in heading
    targets propagated the bias into the v8 trainer too. This is a
    cheap deterministic safety net: when the classifier says heading
    but the text fails basic heading hygiene, demote to paragraph
    rather than poisoning the target.

    Categories demoted:
      - lowercase first letter
      - >= 100 chars (real headings are short)
      - ends with `.!?` after >60 chars (a complete sentence)
      - looks like a long TOC line: text + trailing page number
      - looks like a page running header: short title-cased phrase
        followed by a small page number (e.g. "Quantum 4")

    The page-running-header pattern is page_header rather than
    paragraph because the role exists in the schema and is more
    accurate; we still demote out of heading either way.
    """
    if hint != "heading":
        return hint
    t = text.strip()
    if not t:
        return hint
    if t[0].islower():
        return "paragraph"
    if len(t) >= 100:
        return "paragraph"
    if t.rstrip().endswith((".", "!", "?")) and len(t) > 60:
        return "paragraph"
    if PAGE_RUN_HDR_RE.match(t):
        return "page_header"
    if len(t) > 25 and TOC_TAIL_RE.search(t):
        return "paragraph"
    return hint


def _parse_user_hints(user_msg: str) -> list[tuple[str, str]]:
    """Return a list of (role, text) tuples, one per block, in chunk order.

    Walks the user_msg line by line. Stops at the OCR-regions sentinel
    so GLM-OCR annotation lines (which use the same `[...]` leading
    bracket but aren't part of the block stream) don't pollute the
    count. Non-block lines (chunk header, page separator) are skipped.
    For each block line, extracts the `h=<code>` token from the leading
    `[...]` flags. A missing hint defaults to 'paragraph'.

    The text portion (everything after the closing `]`) is returned
    alongside the hint so callers can apply role-aware sanity checks
    before using the hint as a target.
    """
    hints: list[tuple[str, str]] = []
    for line in user_msg.splitlines():
        if line.startswith("--- OCR regions ---"):
            break
        stripped = line.lstrip()
        if not stripped.startswith("["):
            continue
        if stripped.startswith("[chunk=") or stripped.startswith("--- page"):
            continue
        if stripped.startswith("[ocr "):
            continue
        m = re.match(r"\[([^\]]*)\]", stripped)
        if not m:
            continue
        flags = m.group(1)
        hint = "paragraph"
        for tok in flags.split():
            if tok.startswith("h="):
                code = tok.split("=", 1)[1]
                hint = HINT_CODES.get(code, "paragraph")
                break
        text = _block_text(stripped)
        hints.append((hint, text))
    return hints


def densify_row(row: dict) -> tuple[dict, Counter]:
    msgs = row["messages"]
    user = msgs[1]["content"]
    asst = msgs[2]["content"]

    hints = _parse_user_hints(user)
    n_blocks = len(hints)

    existing = json.loads(asst)
    by_id = {int(e["id"]): e for e in existing if isinstance(e, dict) and "id" in e}

    stats: Counter = Counter()
    new_entries = []
    for bid in range(n_blocks):
        if bid in by_id:
            new_entries.append(by_id[bid])
            continue
        hint, text = hints[bid]
        safe = _heading_sanity_demote(hint, text)
        if safe != hint:
            stats["heading_demoted"] += 1
        new_entries.append({"id": bid, "role": safe})

    out = {
        "messages": [
            msgs[0],
            msgs[1],
            {"role": "assistant", "content": json.dumps(new_entries, ensure_ascii=False)},
        ],
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("data/qwen_dataset_v3"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/qwen_dataset_v3d"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    row_stats: Counter = Counter()
    demote_stats: Counter = Counter()
    for split in ["train", "val", "test"]:
        in_path = args.in_dir / f"{split}.jsonl"
        out_path = args.out_dir / f"{split}.jsonl"
        if not in_path.exists():
            print(f"[skip] {in_path} not found")
            continue
        n = 0
        n_orig_entries = 0
        n_new_entries = 0
        n_demoted = 0
        with in_path.open() as f, out_path.open("w") as g:
            for line in f:
                row = json.loads(line)
                before = json.loads(row["messages"][2]["content"])
                new_row, row_demote = densify_row(row)
                after = json.loads(new_row["messages"][2]["content"])
                n_orig_entries += len(before)
                n_new_entries += len(after)
                n_demoted += row_demote.get("heading_demoted", 0)
                g.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                n += 1
        row_stats[split] = n
        demote_stats[split] = n_demoted
        print(f"[write] {out_path}  {n} rows  "
              f"entries: {n_orig_entries} -> {n_new_entries} "
              f"({n_new_entries/max(1,n_orig_entries):.1f}x)  "
              f"heading_hints_demoted: {n_demoted}")

    print(f"[done] rows={dict(row_stats)} "
          f"demoted_heading_hints={dict(demote_stats)}")


if __name__ == "__main__":
    main()
