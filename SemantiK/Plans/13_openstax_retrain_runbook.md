# Plan 13 — OpenStax-expansion council retrain (eval-after-each-head)

Status: **READY, GPU-GATED** (authored 2026-06-15). Do not start until the RTX 3070 is
free — every step below takes a CUDA context and must run **one at a time**
(see [[feedback_train_cuda_context_guard]]: a 2nd CUDA context — even `pytest` or a
parallel build — deadlocks a running train).

## Why
`data/pairs/openstax` grew 578 → 7,669 pairs. Only the builders that read OpenStax are
affected → retrain **Structure, Semantic, TableSpecialist** (3 of 5 council heads).
Skip (arxiv-only / probe-only): MergeOrSplit, MathSpecialist, Qwen prose/table/math,
Theta. Qwen **reasoner** deferred (separate, longer job).

## Decisions locked
- **Cap = source balance only**, uniform: `--max-examples-per-source 20000 --source-cap gutenberg=5000`.
  Class balance is the trainers' job (class-weighted CE + WeightedRandomSampler).
  Role-aware `--cap-protect-frac` stays OFF (default 0.0); **never enable for table_specialist**.
- **Gate = eval after each head** before proceeding to the next.
- **Ordering = Structure → Semantic** is hard-serial: Semantic's dataset bakes in
  Structure's cascade at build time, so Structure must be trained first, THEN the Semantic
  dataset is rebuilt against the new Structure. TableSpecialist is independent.

## Baselines to beat-or-match (current production summary.json)
| head | metric | baseline |
|---|---|---|
| Structure | `test_role_macro_f1` | **0.8536** (is_heading 0.8743, table_region 0.9625, is_image_block 0.7376, list_nesting_mae 0.0523) |
| Semantic | `test_doc_role_macro_f1` | **0.9470** (boilerplate_pos_f1 0.9661) |
| TableSpecialist | `test_macro_f1` | **0.9256** |

⚠️ **Comparability caveat:** each train script reports the NEW model on the NEW test split
(which now contains more OpenStax). That is not strictly apples-to-apples vs. the old
baseline (old model on old split). First-line gate = built-in test macro-F1 **plus per-class
F1** (watch the minor classes: legal/author/footer for Semantic; code/blockquote/form_label
for Structure; header/span for Table). Rigorous gate (recommended on any borderline result):
eval OLD vs NEW adapter on the SAME new test set via the per-head eval scripts below.

---

## Pre-flight (once, GPU idle)
```bash
cd /home/user/Projects/Semantic
# 1. confirm GPU idle
nvidia-smi
# 2. back up current models so a failed gate can roll back
for h in structure semantic table_specialist; do
  cp -r models/council/$h/final models/council/$h/final.prev_openstax
done
# 3. the _capped_preview dirs were inspection-only; real builds write to the live dirs below
```

---

## Step 1 — STRUCTURE  (CPU build, GPU train)
```bash
# 1a. rebuild dataset (alignment cached; merge+cap+split, CPU)
PYTHONPATH=. .venv/bin/python -m data.build_structure_data \
  --max-examples-per-source 20000 --source-cap gutenberg=5000 --workers 8
# 1b. train (GPU) — overwrites models/council/structure/final
.venv/bin/python train_structure.py \
  --dataset-dir data/structure_dataset --output-dir models/council/structure
# 1c. GATE
.venv/bin/python -c "import json;d=json.load(open('models/council/structure/final/summary.json'));print({k:v for k,v in d.items() if 'f1' in k or 'mae' in k})"
```
**Gate:** `test_role_macro_f1 ≥ 0.8536` and no minor-class collapse. PASS → Step 2.
FAIL → restore `final.prev_openstax`, diagnose (cap too tight? data poison?) before retrying.

## Step 2 — SEMANTIC  (GPU build → GPU train; only after Step 1 PASS)
```bash
# 2a. rebuild WITH cascade from the NEW Structure (GPU; NO --skip-cascade)
PYTHONPATH=. .venv/bin/python -m data.build_semantic_data \
  --max-examples-per-source 20000 --source-cap gutenberg=5000 --workers 8
# 2b. train (GPU)
.venv/bin/python train_semantic.py \
  --dataset-dir data/semantic_dataset --output-dir models/council/semantic
# 2c. GATE
.venv/bin/python -c "import json;d=json.load(open('models/council/semantic/final/summary.json'));print({k:v for k,v in d.items() if 'f1' in k})"
# optional: cascade drift check
.venv/bin/python -m scripts.eval_cascade_structure_to_semantic
```
**Gate:** `test_doc_role_macro_f1 ≥ 0.9470` (+ boilerplate ≈ 0.966), minor doc-roles intact.

## Step 3 — TABLE SPECIALIST  (CPU build, GPU train; independent)
```bash
# 3a. rebuild (uniform cap — do NOT add --cap-protect-frac; trainer's weight_cap=15 handles span)
PYTHONPATH=. .venv/bin/python -m data.build_table_specialist_data \
  --max-examples-per-source 20000 --source-cap gutenberg=5000 --workers 8
# 3b. train (GPU)
.venv/bin/python train_table_specialist.py \
  --dataset-dir data/table_specialist_dataset --output-dir models/council/table_specialist/final
# 3c. GATE
.venv/bin/python -c "import json;d=json.load(open('models/council/table_specialist/final/summary.json'));print({k:v for k,v in d.items() if 'f1' in k or 'macro' in k})"
# optional: per-source breakdown
.venv/bin/python -m scripts.eval_table_specialist_per_source \
  --adapter-dir models/council/table_specialist/final \
  --out-path data/eval_reports/table_specialist_per_source.json
```
**Gate:** `test_macro_f1 ≥ 0.9256`, per-source F1 healthy (incl. openstax).

---

## Step 4 — Final end-to-end (after all three PASS)
```bash
# definition-of-done / corpus eval
scripts/eval_v7_family.sh
# (or) full cascade dev
.venv/bin/python scripts/eval_full_cascade.py --max-arxiv 10 --out data/eval_reports/full_cascade_openstax.json
# optional: refresh Theta cross-domain probe (eval only, no retrain)
```
Confirms the retrained council didn't regress downstream WCAG/structure behaviour.

## Rollback
Any gate fails and isn't worth chasing → `rm -rf models/council/<h>/final && mv models/council/<h>/final.prev_openstax models/council/<h>/final`.
Clean up backups once satisfied: `rm -rf models/council/*/final.prev_openstax`.

## Deferred / out of scope here
- Qwen **reasoner** retrain (build_qwen_data → train_reasoner) — bigger job; do after council is proven.
- Last ~6 OpenStax catalog books (worsens balance; not needed).
- Siyavula (parked; ePUB math inaccessible).
