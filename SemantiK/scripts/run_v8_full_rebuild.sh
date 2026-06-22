#!/usr/bin/env bash
# v8 full rebuild — corrects heading-poisoning in v7 training data.
#
# v7 dataset audit found ~51% of `heading` targets in train.jsonl were
# not real headings (paragraph fragments, hyphenation tails, TOC
# entries with trailing page numbers, page running headers). Two fixes
# land together for v8:
#
#   data/build_qwen_data.py::align_to_ground_truth
#     - role-aware min_overlap: heading 0.6, default 0.3
#     - hyphenation-tail handling (block ending in "-" + next block
#       starting lowercase => inherit prev label)
#     - heading length-ratio gate: long PDF block ↔ short HTML heading
#       gets rejected even when jaccard scrapes by
#
#   scripts/densify_targets.py::_heading_sanity_demote
#     - filler-stage safety net: when classifier hint is `heading` but
#       text obviously isn't (lowercase start, sentence-ish, very long,
#       page-running pattern), demote to paragraph or page_header
#
# v8 reuses classifier_v5 — the classifier code path didn't change,
# only the qwen-dataset alignment + densify did. Skips steps 1-4 of
# v7's pipeline. ETA ~6-8h (mostly the qwen build + reasoner train).
set -euo pipefail
LOG_DIR="data/logs"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=".venv/bin/python"

mkdir -p "$LOG_DIR"

# ----- Step 1: clear stale downstream v8 outputs -----
# build_qwen_data writes per-pair staging files in data/qwen_dataset_v8/_partial/
# and resumes by skipping any pair whose staging file already exists. We
# DON'T rm data/qwen_dataset_v8 here — that lets a partial-failed prior run
# resume cheaply. To force a clean rebuild from scratch, rm it manually.
echo "[v8] 01 clear stale downstream v8 outputs" >&2
rm -rf data/qwen_dataset_v8d
rm -rf data/qwen_dataset_v8ds

# ----- Step 2: build qwen_dataset_v8 (4 shards, cache-only GLM-OCR) -----
# Shards run SEQUENTIALLY. Project memory: "On 8GB GPU, run 4 build_qwen
# shards sequentially; parallel poisons CUDA context." Each shard parent
# holds a DistilBERT classifier on GPU and runs batched inference; with 4
# parents fighting for VRAM the allocator thrashes and effective throughput
# drops to ~0.4 files/min total. Sequential gets us ~50 files/min/shard.
echo "[v8] 02 build qwen_dataset_v8 (4 shards, sequential)" >&2
for s in 0 1 2 3; do
    echo "[v8] 02.$s shard $s" >&2
    TOKENIZERS_PARALLELISM=false "$PY" data/build_qwen_data.py \
        --out-dir data/qwen_dataset_v8 \
        --classifier models/classifier_v5/final \
        --pages-per-chunk 4 --overlap 1 --max-blocks-per-chunk 60 \
        --glm-ocr --glm-ocr-skip-prepass \
        --workers 3 --num-shards 4 --shard "$s" \
        > "$LOG_DIR/v8_02_shard_$s.log" 2>&1
done

# ----- Step 3: assemble -----
echo "[v8] 03 assemble" >&2
TOKENIZERS_PARALLELISM=false "$PY" data/build_qwen_data.py \
    --out-dir data/qwen_dataset_v8 --assemble-only \
    > "$LOG_DIR/v8_03_assemble.log" 2>&1

# ----- Step 4: densify (with heading-sanity demote) -----
echo "[v8] 04 densify" >&2
"$PY" scripts/densify_targets.py \
    --in-dir data/qwen_dataset_v8 \
    --out-dir data/qwen_dataset_v8d \
    > "$LOG_DIR/v8_04_densify.log" 2>&1

# ----- Step 5: split into 30-block sub-chunks -----
echo "[v8] 05 split" >&2
"$PY" scripts/split_chunks.py \
    --in-dir data/qwen_dataset_v8d \
    --out-dir data/qwen_dataset_v8ds \
    > "$LOG_DIR/v8_05_split.log" 2>&1

# ----- Step 6: audit v8 vs v7 heading-poison rate -----
# This is the gate. If v8 still has high heading-suspect rate, stop
# and re-tune the alignment/densify rules before burning ~3h on the
# reasoner train. Manual review only — script logs the numbers and
# proceeds; user can ctrl-C if they look bad.
echo "[v8] 06 audit heading-poison rate" >&2
"$PY" - <<'PY' > "$LOG_DIR/v8_06_audit.log" 2>&1
import json, re
from collections import Counter
from pathlib import Path

def heading_suspect(text):
    t = text.strip()
    if not t: return None
    if (re.search(r"\.\s*\d+\s*$", t) or
        (re.search(r"\d+\s*$", t) and len(t) > 30)):
        return "toc_or_pagenum_tail"
    if re.match(r"^[A-Z][a-zA-Z ]{3,30}\s\d{1,3}$", t):
        return "page_running_header"
    if t[0].islower():
        return "lowercase_start"
    if len(t) >= 100:
        return "very_long"
    if t.rstrip().endswith((".","!","?")) and len(t) > 60:
        return "sentence_with_terminal"
    return None

block_re = re.compile(r"^\s*\[[^\]]*\]\s*(.*)$")
for split in ("train", "val", "test"):
    p = Path(f"data/qwen_dataset_v8ds/{split}.jsonl")
    if not p.exists(): continue
    n_h = 0
    suspect = Counter()
    with p.open() as f:
        for line in f:
            row = json.loads(line)
            try:
                entries = json.loads(row["messages"][2]["content"])
            except Exception:
                continue
            text_by_id = {}
            local_id = 0
            for ln in row["messages"][1]["content"].splitlines()[1:]:
                if ln.startswith("--- page") or ln.startswith("[chunk=") or ln.startswith("--- OCR"):
                    continue
                if ln.startswith("[ocr "): continue
                m = block_re.match(ln)
                if m:
                    text_by_id[local_id] = m.group(1)
                    local_id += 1
            for e in entries:
                if not isinstance(e, dict): continue
                if e.get("role") != "heading": continue
                n_h += 1
                cat = heading_suspect(text_by_id.get(e.get("id"), ""))
                if cat: suspect[cat] += 1
    print(f"  {split}: heading={n_h:6d}  suspect={sum(suspect.values()):5d} "
          f"({100*sum(suspect.values())/max(1,n_h):.1f}%)  {dict(suspect)}")
PY
echo "[v8] 06 audit results:" >&2
cat "$LOG_DIR/v8_06_audit.log" >&2

# ----- Step 7: train reasoner_v8 -----
echo "[v8] 07 train reasoner_v8" >&2
TOKENIZERS_PARALLELISM=false "$PY" train_reasoner.py \
    --dataset-dir data/qwen_dataset_v8ds \
    --output-dir models/reasoner_v8 \
    --subsample-train 3000 --eval-size 100 --epochs 1 \
    > "$LOG_DIR/v8_07_train.log" 2>&1

# ----- Step 8: side-by-side eval on the v7 family of PDFs -----
echo "[v8] 08 family eval (v8 adapter)" >&2
ADAPTER=models/reasoner_v8/final \
CLASSIFIER=models/classifier_v5/final \
    scripts/eval_v7_family.sh \
    > "$LOG_DIR/v8_08_eval.log" 2>&1 || true

echo "[v8] DONE" >&2
