#!/usr/bin/env bash
# Plan 12 — the remaining GPU queue, serialized (Plan 12 D1 / C1 / D3).
#
# USER-LAUNCHED ONLY. The 3070 is shared with other projects; this script
# refuses to start while anything else holds the device, and each stage's
# own training-lock guard re-checks. Run it when the GPU is yours:
#
#   nohup ./scripts/run_gpu_queue.sh > data/logs/gpu_queue.log 2>&1 &
#
# Order rationale (override with: ./scripts/run_gpu_queue.sh eval math gap):
#   1. eval — C1 real-runtime corpus eval (~10-25h, manifest_v2). Runs
#      FIRST because it informs everything else: the theta tau
#      re-calibration follow-up needs its score distribution, and it
#      measures the new strata (forms/scans) before any retrain.
#      NOTE: runs against the CURRENT adapters (math v1) — the report
#      records adapter versions in each doc's conformance audit, so a
#      post-math-v2 delta re-run stays attributable.
#   2. gap  — gap-fill v2 retrain (~hours; dataset 15,600 rows / 5 kinds,
#      citation capped, copyright/legal kinds included).
#   3. math — math v2 resume from checkpoint-1000 (LONGEST: ~61-80h at
#      max_len 2048). Touch models/qwen_specialists/math/v2/.checkpoint_now
#      for an on-demand save; .checkpoint_and_stop to stop cleanly.
#      Keep the host awake — the 2026-06-09 run died to a WSL reboot.
set -euo pipefail
cd "$(dirname "$0")/.."

USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$USED_MIB" -gt 1500 ]; then
  echo "[gpu-queue] REFUSING to start: ${USED_MIB} MiB VRAM in use by another process." >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv >&2
  exit 1
fi

ORDER=("${@:-eval}" )
if [ "$#" -eq 0 ]; then ORDER=(eval gap math); fi

for stage in "${ORDER[@]}"; do
  case "$stage" in
    eval)
      echo "[gpu-queue] === C1 real-runtime corpus eval (manifest_v2) ==="
      .venv/bin/python scripts/eval_full_cascade.py \
        --runtime real --isolate-per-pdf --max-regions-per-kind 60 --k 2 \
        --manifest data/eval_corpus/manifest_v2.json \
        --out data/eval_reports/full_cascade_real_c1_manifest_v2.json \
        --summary
      ;;
    gap)
      echo "[gpu-queue] === gap-fill v2 retrain ==="
      .venv/bin/python scripts/train_qwen_lora.py \
        --adapter gap_fill \
        --out-dir models/qwen_specialists/gap_fill/v2
      ;;
    math)
      echo "[gpu-queue] === math v2 resume (checkpoint-1000) ==="
      .venv/bin/python scripts/train_qwen_lora.py \
        --adapter math \
        --out-dir models/qwen_specialists/math/v2 \
        --resume-from models/qwen_specialists/math/v2/checkpoint-1000
      ;;
    *)
      echo "[gpu-queue] unknown stage '$stage' (want: eval|gap|math)" >&2
      exit 2
      ;;
  esac
  echo "[gpu-queue] stage '$stage' complete"
done
echo "[gpu-queue] ALL STAGES COMPLETE"
