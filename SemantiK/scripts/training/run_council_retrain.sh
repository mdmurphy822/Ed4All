#!/usr/bin/env bash
# Plan 15 — council retrain queue (OpenStax + legal), serialized & gated.
#
# USER-LAUNCHED ONLY. The 3070 is shared; this refuses to start while another
# process holds the device. Every stage is a single CUDA context — NEVER run a
# second one (no pytest, no build_qwen) while this is running
# (feedback_train_cuda_context_guard). Launch when the GPU is yours:
#
#   nohup ./scripts/training/run_council_retrain.sh > data/logs/council_retrain.log 2>&1 &
#
# Default order is the DEPENDENCY order and must not be reshuffled blindly:
#   structure -> semantic -> table_specialist
# Why: the Semantic dataset bakes in Structure's cascade AT BUILD TIME, so
# Structure must be retrained & accepted before Semantic is rebuilt (Plan 13).
# Override (e.g. table only):  ./scripts/training/run_council_retrain.sh table_specialist
# If you pass both structure and semantic, structure MUST precede semantic.
#
# Structure folds Plan 13 (OpenStax) + Plan 14 (legal) into ONE train via the
# combined dataset (--include-legal-pseudo) so Semantic is rebuilt only once.
#
# GATE: after each train, the built-in test macro-F1 from summary.json is
# compared against the backed-up previous model (final.prev_p15). A regression
# HARD-STOPS the queue so a bad Structure can never cascade into Semantic. This
# is the cheap first-line gate ONLY — the rigorous gates (real-runtime FRAGMENT
# on held-out opinions, per-source P/R, heading-count, theta recal) are manual
# and printed as reminders. Do not ship on the F1 gate alone (Plan 14 §6).
set -euo pipefail
cd "$(dirname "$0")/.."

PREV=final.prev_p15
TOL=0.005   # allow this much F1 slack below prev before calling it a regression

USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$USED_MIB" -gt 1500 ]; then
  echo "[council] REFUSING to start: ${USED_MIB} MiB VRAM in use by another process." >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv >&2
  exit 1
fi

# back up current adapters once (gate baseline + rollback target)
for h in structure semantic table_specialist; do
  if [ -d "models/council/$h/final" ] && [ ! -d "models/council/$h/$PREV" ]; then
    echo "[council] backing up models/council/$h/final -> $PREV"
    cp -r "models/council/$h/final" "models/council/$h/$PREV"
  fi
done

# gate(head, metric_key): exit 3 if new < prev - TOL. Prints both either way.
gate() {
  local head="$1" key="$2"
  .venv/bin/python - "$head" "$key" "$PREV" "$TOL" <<'PY'
import json, sys, os
head, key, prev, tol = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
new_p = f"models/council/{head}/final/summary.json"
prev_p = f"models/council/{head}/{prev}/summary.json"
new = json.load(open(new_p)).get(key)
if new is None:
    print(f"[gate:{head}] FAIL: '{key}' missing from {new_p}"); sys.exit(3)
old = None
if os.path.exists(prev_p):
    old = json.load(open(prev_p)).get(key)
if old is None:
    print(f"[gate:{head}] {key}={new:.4f} (no prev baseline — accepting, verify manually)"); sys.exit(0)
delta = new - old
status = "PASS" if new >= old - tol else "REGRESSION"
print(f"[gate:{head}] {key}: new={new:.4f} prev={old:.4f} delta={delta:+.4f} -> {status}")
sys.exit(0 if status == "PASS" else 3)
PY
}

ORDER=("$@")
if [ "$#" -eq 0 ]; then ORDER=(structure semantic table_specialist); fi
# guard the one ordering that matters
seen_struct=0
for s in "${ORDER[@]}"; do
  if [ "$s" = "structure" ]; then seen_struct=1; fi
  if [ "$s" = "semantic" ] && [ "$seen_struct" -eq 0 ]; then
    echo "[council] ERROR: 'semantic' requested before 'structure' in the same run." >&2
    echo "          Semantic bakes in Structure's cascade; train structure first." >&2
    exit 2
  fi
done

for stage in "${ORDER[@]}"; do
  case "$stage" in
    structure)
      echo "[council] === STRUCTURE (OpenStax + legal, combined) ==="
      # CPU build: combined dataset; legal appended AFTER source-cap (Plan 14 §4)
      PYTHONPATH=. .venv/bin/python -m data.build_structure_data \
        --max-examples-per-source 20000 --source-cap gutenberg=5000 \
        --include-legal-pseudo --workers 8 \
        --out-dir data/structure_dataset_v2
      # GPU train (Plan 14 §5 hyperparams — 12 epochs fixes is_heading undertraining)
      CUDA_VISIBLE_DEVICES=0 .venv/bin/python training/train_structure.py \
        --dataset-dir data/structure_dataset_v2 \
        --output-dir models/council/structure \
        --base-model answerdotai/ModernBERT-base \
        --epochs 12 --patience 4 --lr 2e-4 --batch-size 16 --max-length 192 \
        --lora-r 16 --lora-alpha 32 --weight-cap 30.0 --is-heading-sampler-cap 8.0 \
        --snapshot-policy weighted_macro
      gate structure test_role_macro_f1
      .venv/bin/python scripts/calibration/calibrate_structure_heads.py
      echo "[council] REMINDER: F1 gate is first-line only. Before shipping run the"
      echo "          REAL gates: real-runtime FRAGMENT on held-out opinions (frags->0,"
      echo "          NO heading loss: shades keeps 39, opinions keep DISCUSSION/I./A.),"
      echo "          is_heading_pos_f1 >= prev, and eval_cascade_structure_to_semantic."
      ;;
    semantic)
      echo "[council] === SEMANTIC (cascade vs NEW Structure) ==="
      # GPU build — NO --skip-cascade; uses the just-trained Structure (Plan 13 §2)
      PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m data.build_semantic_data \
        --max-examples-per-source 20000 --source-cap gutenberg=5000 --workers 8
      CUDA_VISIBLE_DEVICES=0 .venv/bin/python training/train_semantic.py \
        --dataset-dir data/semantic_dataset --output-dir models/council/semantic
      gate semantic test_doc_role_macro_f1
      ;;
    table_specialist)
      echo "[council] === TABLE SPECIALIST (independent) ==="
      # CPU build — uniform cap, NO --cap-protect-frac (Plan 13 §3)
      PYTHONPATH=. .venv/bin/python -m data.build_table_specialist_data \
        --max-examples-per-source 20000 --source-cap gutenberg=5000 --workers 8
      CUDA_VISIBLE_DEVICES=0 .venv/bin/python training/train_table_specialist.py \
        --dataset-dir data/table_specialist_dataset \
        --output-dir models/council/table_specialist/final
      gate table_specialist test_macro_f1
      ;;
    *)
      echo "[council] unknown stage '$stage' (want: structure|semantic|table_specialist)" >&2
      exit 2
      ;;
  esac
  echo "[council] stage '$stage' complete"
done
echo "[council] ALL STAGES COMPLETE"
echo "[council] Rollback any head:  rm -rf models/council/<h>/final && mv models/council/<h>/$PREV models/council/<h>/final"
echo "[council] On full acceptance:  rm -rf models/council/*/$PREV"
