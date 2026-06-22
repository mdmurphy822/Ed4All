"""Split each training chunk into smaller sub-chunks so the whole example
fits within a 2048-token training budget.

The v3d dataset has 60-block chunks with dense per-block targets — which
overshoots 2048 (asst p50 = 835 tokens + user p50 = 1412 = 2247). Cut each
chunk in half: two 30-block sub-chunks, both with properly renumbered
local block ids and sliced assistant arrays.

Input:  data/qwen_dataset_v3d/{train,val,test}.jsonl  (60-block chunks)
Output: data/qwen_dataset_v3ds/{train,val,test}.jsonl  (30-block chunks)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


CHUNK_HEADER_RE = re.compile(
    r"\[chunk=(\d+)/(\d+) pages=(\d+)-(\d+)\]"
)
PAGE_SEP_RE = re.compile(r"^--- page (\d+) ---$")
BLOCK_LINE_RE = re.compile(r"^\s*\[")
OCR_MARKER = "--- OCR regions ---"
OCR_LINE_RE = re.compile(r"^\s*\[ocr p=(\d+)\s")


def split_row(row: dict, sub_size: int = 30) -> list[dict]:
    msgs = row["messages"]
    system = msgs[0]
    user = msgs[1]["content"]
    asst = msgs[2]["content"]

    entries = json.loads(asst)
    entries_by_id = {int(e["id"]): e for e in entries}

    lines = user.splitlines()
    header_line = lines[0]
    m = CHUNK_HEADER_RE.search(header_line)
    if not m:
        return [row]   # unknown header shape, don't touch
    ch_idx, ch_total, p_start, p_end = map(int, m.groups())

    # Walk body: collect (page, block_line) tuples and track which page
    # each block lives on. Stop at the OCR-regions marker so its
    # `[ocr ...]` lines don't get counted as blocks.
    body = lines[1:]
    current_page = p_start
    blocks: list[tuple[int, str]] = []   # (page, block_line)
    ocr_lines: list[tuple[int, str]] = []  # (page, ocr_line)
    in_ocr = False
    for ln in body:
        if ln.startswith(OCR_MARKER):
            in_ocr = True
            continue
        if in_ocr:
            m_ocr = OCR_LINE_RE.match(ln)
            if m_ocr:
                ocr_lines.append((int(m_ocr.group(1)), ln))
            continue
        page_m = PAGE_SEP_RE.match(ln.strip())
        if page_m:
            current_page = int(page_m.group(1))
            continue
        if BLOCK_LINE_RE.match(ln):
            blocks.append((current_page, ln))
    n_blocks = len(blocks)
    if n_blocks <= sub_size:
        return [row]

    n_subs = (n_blocks + sub_size - 1) // sub_size
    new_rows: list[dict] = []
    for sub_i in range(n_subs):
        start = sub_i * sub_size
        end = min(start + sub_size, n_blocks)
        sub_blocks = blocks[start:end]
        # Reformat block lines: renumber the id/local tokens? No — the
        # build serializer uses a literal local_id that equals the
        # block's position in the chunk. Since block lines embed that
        # id implicitly through ordering, the block line TEXT stays as
        # is. The target's "id" field gets remapped to the new local id.
        sub_p_start = sub_blocks[0][0]
        sub_p_end = sub_blocks[-1][0]
        user_lines = [
            f"[chunk={sub_i}/{n_subs} pages={sub_p_start}-{sub_p_end}]"
        ]
        prev_page = -1
        for p, bln in sub_blocks:
            if p != prev_page:
                user_lines.append(f"--- page {p} ---")
                prev_page = p
            user_lines.append(bln)
        sub_user = "\n".join(user_lines)

        sub_entries = []
        for new_local, old_local in enumerate(range(start, end)):
            src = entries_by_id.get(old_local)
            if src is None:
                continue
            out_entry = dict(src)
            out_entry["id"] = new_local
            sub_entries.append(out_entry)
        sub_asst = json.dumps(sub_entries, ensure_ascii=False)

        # Filter OCR regions to the sub-chunk's page range.
        sub_ocr = [ln for p, ln in ocr_lines if sub_p_start <= p <= sub_p_end]
        if sub_ocr:
            user_lines.append(OCR_MARKER)
            user_lines.extend(sub_ocr)
            sub_user = "\n".join(user_lines)

        new_rows.append({
            "messages": [
                system,
                {"role": "user", "content": sub_user},
                {"role": "assistant", "content": sub_asst},
            ],
        })
    return new_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("data/qwen_dataset_v3d"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/qwen_dataset_v3ds"))
    ap.add_argument("--sub-size", type=int, default=30)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        in_path = args.in_dir / f"{split}.jsonl"
        out_path = args.out_dir / f"{split}.jsonl"
        if not in_path.exists():
            print(f"[skip] {in_path}")
            continue
        n_in = 0
        n_out = 0
        with in_path.open() as f, out_path.open("w") as g:
            for line in f:
                row = json.loads(line)
                n_in += 1
                for new_row in split_row(row, sub_size=args.sub_size):
                    g.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                    n_out += 1
        print(f"[write] {out_path}  {n_in} -> {n_out} rows")

    print(f"[done]")


if __name__ == "__main__":
    main()
