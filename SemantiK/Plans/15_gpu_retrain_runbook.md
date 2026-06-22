# Plan 15 — Sequenced GPU retrain runbook → broad corpus eval

**Status:** SCOPED 2026-06-18. Orders every gated GPU job against the
single-CUDA-context constraint and terminates in a broad-corpus definition-of-done
eval. Supersedes the ad-hoc ordering; folds in Plans 13 (OpenStax), 14 (legal),
and the Plan 12 queue (`scripts/run_gpu_queue.sh`: eval/gap/math).

## The two hard constraints that fix the order

1. **One CUDA context at a time** ([[feedback_train_cuda_context_guard]],
   [[feedback_qwen_build_serial]]). Every GPU step below runs **alone** — no
   `pytest tests/`, no second build that touches the GPU, no parallel shard.
   A second context deadlocks the running train. Guard CPU-only sidecar work
   with `CUDA_VISIBLE_DEVICES=""`.
2. **Structure → Semantic is hard-serial** (Plan 13). The Semantic dataset bakes
   in Structure's cascade *at build time*, and that build is itself a GPU job.
   So Structure must be fully retrained and accepted before Semantic is rebuilt.

**Linchpin decision — merge Plan 13 + Plan 14 into ONE Structure train.** Both
plans retrain the Structure head (13 = OpenStax breadth; 14 = legal hard-negatives).
Running them separately means retraining Structure twice *and* rebuilding the
GPU Semantic dataset twice. Build one combined Structure dataset
(`--include-legal-pseudo` on top of the OpenStax-expanded sources) and train once.

## What runs OFF the GPU critical path (do these in the background)

- **SigLIP figure labeling** (~650 figures, [[project_siglip_router_plan]]).
  Candidates already mined (`scripts/mine_figure_labels.py`). The labeling itself
  is CPU/manual — run it concurrently with any GPU train below. The 60 salvaged
  rows in-repo are the seed; this is the user-gated budget step.
- **CPU dataset builds** (Structure, TableSpecialist — see Phase 0). Pure CPU
  (alignment cached). The Semantic build is the exception: it is GPU (cascade)
  and CANNOT overlap a running train.

---

## Phase 0 — Off-GPU prep (no CUDA; do before claiming the card)

```bash
cd /home/user/Projects/Semantic
nvidia-smi   # confirm idle before any GPU phase; run_gpu_queue.sh refuses if >1500 MiB used

# 0a. Back up every adapter a gate might roll back to
for h in structure semantic table_specialist; do
  cp -r models/council/$h/final models/council/$h/final.prev_p15
done
cp -r models/qwen_specialists/gap_fill/final models/qwen_specialists/gap_fill/final.prev_p15 2>/dev/null || true

# 0b. Combined Structure dataset (CPU) — OpenStax sources + legal pseudo-labels,
#     legal appended AFTER source-capping so its 2,617 headings survive (Plan 14 §4).
#     Fresh out-dir preserves the live dataset until the retrain is accepted.
PYTHONPATH=. .venv/bin/python -m data.build_structure_data \
  --max-examples-per-source 20000 --source-cap gutenberg=5000 \
  --include-legal-pseudo --workers 8 \
  --out-dir data/structure_dataset_v2
#   Verify data/structure_dataset_v2/coverage_report.json:
#   - source `courtlistener_pseudo` present; legal heading count > 0
#   - blockquote/code/form counts NOT dropped vs current
#   - openstax volume up vs prior

# 0c. TableSpecialist dataset (CPU) — uniform cap, NO --cap-protect-frac (Plan 13 §3)
PYTHONPATH=. .venv/bin/python -m data.build_table_specialist_data \
  --max-examples-per-source 20000 --source-cap gutenberg=5000 --workers 8

# 0d. (background, no GPU) kick off the SigLIP ~650-figure labeling run

# 0e. (background, no GPU — long lead) FETCH fresh held-out docs for the broad
#     manifest_v3 and build it. The binding constraint is held-out PDF supply:
#     most non-arxiv pair sources are trained-on, so the breadth strata need
#     NEW documents disjoint from training (see "manifest_v3 stratum breakdown").
.venv/bin/python scripts/fetch_c2_eval_pdfs.py   # extend per-source targets to the v3 counts
.venv/bin/python scripts/build_eval_corpus_manifest.py \
  --out data/eval_corpus/manifest_v3.json        # target counts per the breakdown below
#   Verify manifest_v3 stratum_shortfalls == [] before relying on it.
```

---

## Phase 1 — (Optional, recommended) baseline corpus eval @ current adapters

Captures a *before* number **on the same broad `manifest_v3`** the final eval uses,
so the before/after delta is attributable. Skippable to save the wall-clock, but
then you lose the clean comparison — the report records adapter versions per doc
either way. (Do NOT use `run_gpu_queue.sh eval` here — that stage is pinned to
`manifest_v2`; run v3 directly so baseline and final share a corpus.)

```bash
nohup .venv/bin/python scripts/eval_full_cascade.py \
  --runtime real --isolate-per-pdf --max-regions-per-kind 60 --k 2 \
  --manifest data/eval_corpus/manifest_v3.json \
  --out data/eval_reports/full_cascade_real_p15_baseline.json --summary \
  > data/logs/eval_baseline_v3.log 2>&1 &
```

---

## Phases 2–4 — council retrain, ONE guarded command

Phases 2–4 are wired into `scripts/run_council_retrain.sh` — VRAM-refusal guard,
serial `structure → semantic → table_specialist` in dependency order, a fail-fast
F1 gate after each train (a regressed Structure HARD-STOPS before the GPU Semantic
build), and auto-backup/rollback hints. One user-gated launch:

```bash
nohup ./scripts/run_council_retrain.sh > data/logs/council_retrain.log 2>&1 &
# subsets: ./scripts/run_council_retrain.sh table_specialist
#          ./scripts/run_council_retrain.sh structure semantic   # (order enforced)
```

The script's per-head F1 gate is **first-line only**. The full gates below are
manual and MUST pass before shipping. The phase-by-phase detail (kept for the
gate definitions) follows.

## Phase 2 — STRUCTURE (combined OpenStax + legal) ← linchpin

Run by the queue script above (build → `train_structure.py` with Plan 14 §5
hyperparams → F1 gate → `calibrate_structure_heads.py`). Standalone equivalent:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_structure.py \
  --dataset-dir data/structure_dataset_v2 \
  --output-dir models/council/structure \
  --base-model answerdotai/ModernBERT-base \
  --epochs 12 --patience 4 --lr 2e-4 --batch-size 16 --max-length 192 \
  --lora-r 16 --lora-alpha 32 --weight-cap 30.0 --is-heading-sampler-cap 8.0 \
  --snapshot-policy weighted_macro
.venv/bin/python scripts/calibrate_structure_heads.py
```

**Gate (union of Plans 13 + 14 — ALL must hold):**
- `test_role_macro_f1` ≥ current production value. *Reconcile the baseline first:*
  Plan 13 records **0.8536**, Plan 14 §6 cites **0.906** — read the live
  `models/council/structure/final.prev_p15/summary.json` and gate against that.
- `is_heading_pos_f1` ≥ current (Plan 14 — do **not** trade heading recall away).
- No minor-class collapse (watch code / blockquote / form_label; legal headings).
- **Real-runtime FRAGMENT gate** on held-out legal opinions (Plan 14 §6.3): frags → 0
  with NO heading loss — shades keeps its 39, opinions keep DISCUSSION / I. / A.
- Regression: wiki/textbook per-source P/R/F1 not regressed
  (`scripts/eval_cascade_structure_to_semantic.py --mode endtoend`).

FAIL → `rm -rf models/council/structure/final && mv models/council/structure/final.prev_p15 models/council/structure/final`; diagnose before retry. **Do not proceed to Phase 3 on a failed Structure** — Semantic inherits it.

---

## Phase 3 — SEMANTIC (only after Phase 2 accepted)

```bash
# 3a. GPU build — cascade against the NEW Structure, NO --skip-cascade (Plan 13 §2)
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m data.build_semantic_data \
  --max-examples-per-source 20000 --source-cap gutenberg=5000 --workers 8
# 3b. train
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_semantic.py \
  --dataset-dir data/semantic_dataset --output-dir models/council/semantic
```
**Gate:** `test_doc_role_macro_f1 ≥ 0.9470` (boilerplate ≈ 0.966), minor doc-roles
(legal/author/footer) intact. Optional drift check:
`scripts/eval_cascade_structure_to_semantic.py`.
FAIL → restore `final.prev_p15`.

---

## Phase 4 — TABLE SPECIALIST (independent; dataset built in Phase 0)

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_table_specialist.py \
  --dataset-dir data/table_specialist_dataset \
  --output-dir models/council/table_specialist/final
```
**Gate:** `test_macro_f1 ≥ 0.9256`, per-source F1 healthy incl. openstax
(`scripts/eval_table_specialist_per_source.py`). FAIL → restore `final.prev_p15`.

*(Phases 2→3→4 land the full breadth win on the BERT council. The two Qwen jobs
and SigLIP below are independent of each other and of the council.)*

---

## Phase 5 — GAP-FILL v2 retrain

```bash
nohup ./scripts/run_gpu_queue.sh gap > data/logs/gpu_queue_gap.log 2>&1 &
# eval after:
.venv/bin/python scripts/eval_qwen_gap_fill_adapter.py   # per-slot-kind P/R
```
**Gate:** per-kind precision/recall not regressed; copyright/legal kinds covered.
Respect runtime output caps when reading the eval ([[feedback_qwen_runtime_output_caps]]).

---

## Phase 6 — SigLIP figure router (once Phase-0 labeling is done)

```bash
.venv/bin/python scripts/build_figure_embeddings.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_figure_router.py
.venv/bin/python scripts/eval_figure_router.py
```
**Gate:** 5-class router macro-F1 healthy on held-out; fail-closed `classify_subtype`
behavior preserved. Promote head only on pass. *(Flexible slot — runs whenever
labeling finished and the GPU is free; placed before math so the long job is last.)*

---

## Phase 7 — MATH v2 resume (LONGEST: ~61–80h; run LAST among retrains)

```bash
nohup ./scripts/run_gpu_queue.sh math > data/logs/gpu_queue_math.log 2>&1 &
# eval after:
.venv/bin/python scripts/eval_qwen_math_adapter.py
```
- Resumes from `models/qwen_specialists/math/v2/checkpoint-1000` at `max_len 2048`
  ([[project_math_adapter_caps]] — fixes the 32.5% length-censoring + truncation).
- **Keep the host awake** — the 2026-06-09 run died to a WSL reboot. On-demand save:
  `touch models/qwen_specialists/math/v2/.checkpoint_now` ([[feedback_qwen_ondemand_checkpoint]]).
- Last because it is by far the longest and fully independent: a death here does
  not block the council/gap/SigLIP wins already banked.
**Gate:** target-length distribution clears shipped `max_new_tokens` AND runtime
`n_ctx` ([[feedback_qwen_runtime_output_caps]]); MathML validity ≥ 95%.

---

## Phase 8 — Theta recalibration + BROAD corpus eval (definition-of-done)

The prior green eval was **n=3 arXiv PDFs** ([[project_r10_corpus_eval]]) — proves
wiring, not product. `manifest_v3` (built in Phase 0) is the broad corpus; see the
stratum breakdown below for target counts and the held-out-supply gaps it closes.

```bash
# 8a. Re-calibrate theta tau against the NEW adapter stack (calibration drifts
#     once Structure/Semantic/math/gap all moved).
.venv/bin/python scripts/calibrate_theta.py

# 8b. FINAL broad corpus eval — all retrained adapters live, SAME manifest_v3 as
#     the Phase-1 baseline → attributable before/after.
.venv/bin/python scripts/eval_full_cascade.py \
  --runtime real --isolate-per-pdf --max-regions-per-kind 60 --k 2 \
  --manifest data/eval_corpus/manifest_v3.json \
  --out data/eval_reports/full_cascade_real_p15_broad.json --summary
```

**Definition-of-done (the claim this whole runbook buys):**
- WCAG: zero critical/serious across the broad corpus, not just arxiv.
- `stage10_pass_rate` holds at scale; per-stratum (legal/forms/scans) not an outlier.
- theta `semantic_preservation` distribution healthy on the new strata.
- Compare against the Phase-1 baseline report for an attributable before/after.

---

## manifest_v3 stratum breakdown

`manifest_v2` = **56 PDFs / 13 strata**, but the non-arXiv strata are too thin to
defend a per-domain claim (legal 6, forms 5, govinfo 4, scans 3). The binding
constraint is **held-out PDF supply**: most pair sources are entirely trained-on
(`data/pairs/*` → trained), so widening means *fetching NEW documents disjoint
from training* — not resampling pairs. `manifest_v3` target ≈ **113 PDFs**, sized
so each stratum can resolve a failure rate to roughly ±10–15pp, weighted toward
the domains the retrains actually touch (openstax, legal).

| Stratum | v2 n | **v3 target** | Held-out source | Action to close the gap |
|---|---|---|---|---|
| `arxiv:*` (6 subjects × 5) | 24 | **30** | local Arxiv Repo, held-out by id | +6: pick more held-out ids (trivial; thousands available) |
| `openstax_prerender` | 5 | **18** | held-out OpenStax pairs → prerender | +13: render held-out pairs (no fetch). **Plan 13 domain — needs the most power** |
| `c2_courtlistener` (legal) | 6 | **15** | FRESH opinions, disjoint from the 532 trained pairs | **+9: fetch fresh (binding gap). Plan 14 domain.** Keep disjoint from Plan 14's training hold-out too |
| `wikipedia_prerender` | 5 | **10** | held-out Wikipedia pairs → prerender | +5: render held-out pairs (no fetch) |
| `c2_gov_forms` | 5 | **10** | fresh IRS fillable forms | +5: fetch (e.g. 941, 1099-series, Sched C, 1065, 990) |
| `c2_govinfo` (CFR/FR) | 4 | **8** | fresh CFR titles + Federal Register rules | +4: fetch |
| `scanned_ocr` | 3 | **8** | more GPO Statutes-at-Large / scanned PD | +5: fetch (hardest domain; OCR floor) |
| `side_by_side` (bench) | 4 | **4** | existing bench PDFs (`held_out=false`) | keep as comparability anchor; exclude from pass-rate denominators |
| **Total** | **56** | **≈113** | | |

Sizing rationale:
- **arXiv stays modest (30).** It is the proven domain; it anchors "no in-domain
  regression," not the breadth claim — over-investing here buys little.
- **openstax + legal get the most (18 / 15).** They are exactly where Plans 13 &
  14 retrain, so they need the statistical power to *prove the retrain worked* and
  to catch over-correction (e.g. Plan 14's #1 risk: heading suppression on legal).
- **forms / govinfo / scans (10 / 8 / 8)** are where Plan 12's C2 run already
  flagged real defects (label association, CFR hierarchy, OCR floor). n≥8 turns
  "anecdote" into "rate."

Held-out discipline (non-negotiable, enforced by the builder):
- Every v3 entry must be `held_out=true` with `pair_id=null` (the builder reuses
  `eval_full_cascade._build_trained_pair_id_set` so the manifest and driver agree
  on "trained-on"). Verify `manifest_v3.stratum_shortfalls == []` and
  `trained_pair_exclusion.union_size` covers the new datasets.
- Legal is the trap: the 532 `data/cache/courtlistener` PDFs are all trained
  pairs, and Plan 14 carves out ~5–9 more for its own gate. v3's 15 legal docs
  must be disjoint from **both**.
- All fetched docs stay commercial-OK ([[feedback_license_policy]]); the existing
  C2 sources are US-gov public domain.

Wall-clock: ~15–40 min/PDF on the 3070 with `--isolate-per-pdf` → **≈28–75h for
113 PDFs**. Plan for 3–5 overnight runs; the report flushes atomically per-PDF, so
`--resume` against the same out-path picks up where an interrupted run stopped.

### As-built 2026-06-18 — manifest_v3 (83 docs)

After the OpenStax/Wikipedia prerender pass, `manifest_v3.json`
(`--per-subject 5 --n-openstax 18 --n-wikipedia 10`) = **83 docs**:

| Stratum | as-built | target | status |
|---|---|---|---|
| arxiv (6×5) | **30** | 30 | ✅ full |
| openstax_prerender | **18** | 18 | ✅ full (49 held-out renders available) |
| wikipedia_prerender | **10** | 10 | ✅ full (16 held-out renders available) |
| c2_courtlistener (legal) | 9 | 15 | ⏸ deferred — anon v4 PDF-scarcity + hardcoded `max_pages=5` |
| c2_gov_forms | 5 | 10 | ⏸ deferred — static curated URL list |
| c2_govinfo | 4 | 8 | ⏸ deferred — static curated URL list |
| scanned_ocr | 3 | 8 | ⏸ deferred — static curated URL list |
| side_by_side | 4 | 4 | ✅ bench anchor |

**Breadth domains (arxiv/openstax/wiki) are at full target — the corpus now carries
real OpenStax generalization signal (the point of Plan 13).** The four C2 strata stay
below target per the 2026-06-18 "accept current, build now" decision; they need the
C2 fetcher changes (CL `--max-pages` + query breadth; new curated static gov URLs).

**How OpenStax coverage was restored (and the integrity caveat):**
- No reserve-and-rebuild was needed — the 20k-*example* cap already leaves **1,619
  OpenStax pairs naturally held-out** (zero spans in LIVE *or* v2 train/val), 1,345
  renderable. The hole was purely missing renders (HTML-hash-keyed cache; the
  578→7,669 expansion orphaned v2's 5 renders).
- Restored by `scripts/prerender_pairs.py` over **30** staged held-out OpenStax pairs
  (spread across 30 distinct books) + 10 Wikipedia → cache. Rendered 30, manifest
  uses 18 — the surplus is a deliberate buffer.
- **Integrity caveat — re-verify at Phase 8.** "Held-out" was computed vs the LIVE
  `data/structure_dataset` + `data/semantic_dataset` and vs `structure_dataset_v2`.
  But the **Semantic retrain (Phase 3) rebuilds `semantic_dataset` with its own 20k
  OpenStax cap** and could sample some of these pairs, contaminating them. Mitigation:
  the 18 are over-provisioned from 49 candidates, and the Phase-8 manifest rebuild
  re-runs `collect_prerendered_heldout` against the THEN-current trained set, so any
  contaminated pair is automatically dropped. Just re-confirm `openstax_prerender ≥ ~12`
  survives after the final Semantic.
- **Path note:** `eval_full_cascade._build_trained_pair_id_set` reads
  `data/structure_dataset` + `data/semantic_dataset` (NOT `_v2`). After the retrain is
  accepted, promote `structure_dataset_v2` → `data/structure_dataset` (or update
  `_DATASET_FILES`) so the manifest's trained-set matches what was actually trained.

Comparability note: rebuilding the manifest reshuffles picks and **breaks
comparability with v1/v2 reports** (per `data/eval_corpus/README.md`). That is
fine — v3 is a deliberately new, broader corpus. Run BOTH the Phase-1 baseline and
the Phase-8 final on v3 so the before/after is internally consistent; v2 reports
remain the historical record, not the comparison.

---

## One-glance order & GPU-occupancy

| # | Job | Build | Train | Gating dep | Notes |
|---|-----|-------|-------|-----------|-------|
| 0 | prep + backups + SigLIP labeling | CPU | — | — | off critical path |
| 1 | baseline corpus eval *(optional)* | — | GPU | — | before-number |
| 2 | **Structure** (OpenStax+legal merged) | CPU (P0) | GPU | — | linchpin |
| 3 | Semantic | **GPU** (cascade) | GPU | needs #2 | hard-serial after #2 |
| 4 | TableSpecialist | CPU (P0) | GPU | — | independent |
| 5 | Gap-fill v2 | — | GPU | — | `run_gpu_queue.sh gap` |
| 6 | SigLIP router | CPU emb | GPU | needs P0 labels | flexible slot |
| 7 | Math v2 resume | — | GPU | — | LONGEST, last; keep host awake |
| 8 | theta recal + **broad corpus eval** | CPU | GPU | needs #2–7 | definition-of-done |

Everything in the train/eval columns is **strictly serial** — never two at once.
Only Phase-0 CPU builds and the SigLIP *labeling* may overlap a running train.

## Open user decisions
- Run the optional Phase-1 baseline eval, or skip to save ~10–25h?
- SigLIP label budget — approve the ~650-figure run (gates Phase 6).
- GPU go/no-go per phase (each is a separate user-gated launch).
- Reconcile the Structure `role_macro_f1` baseline (0.8536 vs 0.906) against the
  live summary.json before locking the Phase-2 gate.
