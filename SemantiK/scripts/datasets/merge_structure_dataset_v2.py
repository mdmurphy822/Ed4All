"""Merge synthesized table_region rows into the base structure dataset.

Plans/07 §5 step 2, "Balanced ~64K" variant. Takes
``data/structure_region_full/region_rows.jsonl`` (561K rows, 92% PMC,
86% positive) and folds a source-balanced, pair-aware subsample into the
base ``data/structure_dataset`` splits, writing ``data/structure_dataset_v2``.

Discipline:
  * PAIR-AWARE split — every row of a given ``pair`` lands in exactly one
    of train/val/test, so no table leaks across the boundary.
  * PER-SOURCE CAP — PMC capped (by whole pairs) so it can't dominate;
    all other sources kept in full. Restores source diversity.
  * Per-source 80/10/10 so each source is represented in every split
    (the ship gate wants a source-stratified table_region eval).
  * The base splits pass through untouched; synth is pure augmentation.

Deterministic (seeded). No silent fallbacks: asserts the row schema of
every synth row matches the base contract before writing.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(0)

BASE = Path("data/structure_dataset")
SYNTH = Path("data/structure_region_full/region_rows.jsonl")
OUT = Path("data/structure_dataset_v2")

# Balanced ~64K: cap PMC by whole pairs; keep all other sources.
SOURCE_ROW_CAP = {"pmc": 20000}
SPLIT_FRACS = (0.80, 0.10, 0.10)  # train / val / test

# Decouple the role objective from the table_region augmentation. Synth
# table-cell rows (table_region==1) are ~100% "paragraph" — the same
# non-authoritative content-shape convention the base uses (base table+
# rows are 97.4% paragraph too; structure.py:53-57 says structural_role
# does NOT encode table membership). Folding 3x more of them in only
# dilutes the rare prose roles and craters role macro-F1 (structure_v2
# regressed 0.866 -> 0.820). So we MASK structural_role + is_heading to
# the CE/metric ignore_index on synth table+ rows: the table_region head
# still gets the full hardening signal, the role head trains on the base
# distribution. Synth CONTEXT-NEGATIVE rows (table_region==0, real
# blockquote/code_block roles) stay supervised — they enrich rare roles.
MASK_INDEX = -100  # torch CrossEntropyLoss default ignore_index

TOP_KEYS = {"text", "layout", "labels", "html_tag", "source", "pair"}
LABEL_KEYS = {"structural_role", "is_heading", "table_region",
              "is_image_block", "list_nesting"}


def _validate(row: dict) -> None:
    if set(row) != TOP_KEYS:
        raise ValueError(f"top-key mismatch: {sorted(row)}")
    if set(row["labels"]) != LABEL_KEYS:
        raise ValueError(f"label-key mismatch: {sorted(row['labels'])}")
    if len(row["layout"]) != 20:
        raise ValueError(f"layout dim {len(row['layout'])} != 20")


def _read(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    # --- group synth rows by (source, pair) ---
    by_source_pair: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    n_synth = 0
    n_masked = 0
    with SYNTH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            _validate(row)
            # Mask role/is_heading on synth table-cell rows (see MASK_INDEX).
            if row["labels"]["table_region"] == 1:
                row["labels"]["structural_role"] = MASK_INDEX
                row["labels"]["is_heading"] = MASK_INDEX
                n_masked += 1
            by_source_pair[row["source"]][row["pair"]].append(row)
            n_synth += 1
    print(f"[synth] read {n_synth} rows across {len(by_source_pair)} sources "
          f"({n_masked} table+ rows role/is_heading masked to {MASK_INDEX})")

    # --- per-source cap (by whole pairs) + per-source 80/10/10 pair split ---
    split_rows: list[list[dict]] = [[], [], []]  # train, val, test
    kept_total = 0
    for source, pairs in sorted(by_source_pair.items()):
        pair_ids = list(pairs)
        random.shuffle(pair_ids)
        cap = SOURCE_ROW_CAP.get(source)
        kept_pairs, kept_rows = [], 0
        for pid in pair_ids:
            if cap is not None and kept_rows >= cap:
                break
            kept_pairs.append(pid)
            kept_rows += len(pairs[pid])
        # pair-aware split of the kept pairs
        n = len(kept_pairs)
        n_tr = int(n * SPLIT_FRACS[0])
        n_val = int(n * SPLIT_FRACS[1])
        buckets = (kept_pairs[:n_tr], kept_pairs[n_tr:n_tr + n_val], kept_pairs[n_tr + n_val:])
        src_counts = [0, 0, 0]
        for i, bucket in enumerate(buckets):
            for pid in bucket:
                split_rows[i].extend(pairs[pid])
                src_counts[i] += len(pairs[pid])
        kept_total += sum(src_counts)
        print(f"  {source:16s} pairs={n:5d} rows tr/val/test={src_counts[0]}/{src_counts[1]}/{src_counts[2]}")
    print(f"[synth] kept {kept_total} rows after caps")

    # --- merge with base, shuffle, write ---
    OUT.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(("train", "val", "test")):
        base = _read(BASE / f"{name}.jsonl")
        merged = base + split_rows[i]
        random.shuffle(merged)
        with (OUT / f"{name}.jsonl").open("w") as f:
            for row in merged:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # report
        pos = sum(r["labels"]["table_region"] for r in merged)
        src = defaultdict(int)
        for r in merged:
            src[r["source"]] += 1
        top_src = max(src.items(), key=lambda kv: kv[1])
        print(f"[{name}] base={len(base)} +synth={len(split_rows[i])} "
              f"= {len(merged)} | table_region pos={pos} ({pos/len(merged)*100:.1f}%) "
              f"| top source={top_src[0]} {top_src[1]/len(merged)*100:.1f}%")


if __name__ == "__main__":
    main()
