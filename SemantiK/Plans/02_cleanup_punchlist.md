# Cleanup punchlist — 2026-05-03

> **STATUS (banner added 2026-06-17): EXECUTED — historical.** This 2026-05-03 punchlist was
> actioned; retained as a record. For the current bloat/cleanup state see the latest audit.

Auditor: cleanup-auditor agent (read-mostly + conservative archival pass).
Authoritative dispositions: `Plans/01_implementation_plan.md` §2.
Existing baseline: `docs/bloat_audit.md` (Phase 1 already closed, Phase 2 partly closed).

All "archive" actions move targets into `/home/user/Projects/Semantic/_archive_2026-05-03/` (ignored by git via a new `.gitignore` rule). The user can `rm -rf _archive_2026-05-03/` after a manual sanity check. **No data has been deleted outright** beyond `__pycache__` regeneration crud.

Two systemic facts that drove decisions:

1. The bulk of what's archived (`models/*`, `data/qwen_dataset_*`, `data/dataset/`, `eval/side_by_side/*`, `eval/results/*`) is already covered by `.gitignore`, so moves do **not** show up in `git status` and cannot be expressed as per-action commits. The single git-visible action this pass made is the `.gitignore` update that excludes `_archive_*/`.
2. `Plans/01_implementation_plan.md` keeps every v1 code module in-tree until v1 retires. So this pass touches **zero `dart_semantic/*.py` files** and **zero training entry points** (`train_classifier.py`, `train_reasoner.py`).

---

## Section A — Safely deleted / archived in this pass

### A.1 Bytecode crud — deleted outright

| Path | Size |
|---|---|
| `dart_semantic/__pycache__/` | 392 K |
| `data/__pycache__/` | 52 K |
| `scripts/__pycache__/` | 36 K |

`rm -rf` per directory. Regenerable; nothing references the .pyc files.

### A.2 Old reasoner adapters (current = `reasoner_v8`) — archived

Per `Plans/01_implementation_plan.md` §0.1: "Adapter at `models/reasoner_v8/final` is current." Verified by grep that no live script defaults to `reasoner_v1..v7`:

- `scripts/run_v7_full_rebuild.sh` builds v7 — but this is a self-contained historical script that retrains.
- `scripts/run_v8_full_rebuild.sh`, `scripts/eval_v7_family.sh`, `scripts/make_pdf_vs_html.py`, `scripts/infer_pdf.py` all default to `reasoner_v8` or accept the path via env/arg.

| Action | From | To | Size |
|---|---|---|---|
| `mv` | `models/reasoner_v1` | `_archive_2026-05-03/models/reasoner_v1` | 113 M |
| `mv` | `models/reasoner_v2` | `_archive_2026-05-03/models/reasoner_v2` | 113 M |
| `mv` | `models/reasoner_v3` | `_archive_2026-05-03/models/reasoner_v3` | 4 K (empty) |
| `mv` | `models/reasoner_v4` | `_archive_2026-05-03/models/reasoner_v4` | 113 M |
| `mv` | `models/reasoner_v5` | `_archive_2026-05-03/models/reasoner_v5` | 113 M |
| `mv` | `models/reasoner_v6` | `_archive_2026-05-03/models/reasoner_v6` | 113 M |
| `mv` | `models/reasoner_v7` | `_archive_2026-05-03/models/reasoner_v7` | 113 M |

Subtotal: ~678 M.

### A.3 Old classifier adapters (current = `classifier_v5`) — archived

`Plans/01_implementation_plan.md` §0.1: "Adapter at `models/classifier_v5/final` is current." Grep confirms every live consumer (`scripts/eval_v7_family.sh`, `scripts/infer_pdf.py`, `scripts/make_pdf_vs_html.py`, `scripts/run_v7_full_rebuild.sh`, `scripts/run_v8_full_rebuild.sh`, `Plans/01_implementation_plan.md`, `.claude/skills/cycle-reasoner`, `.claude/skills/promote-reasoner`) defaults to v5. References to `models/classifier/final` and `models/classifier_v2/final` only appear in the **example** docstring of `eval/compare_classifiers.py` (a CLI tool with no fixed default — runtime supplies paths) and the v1 README (untouched per rules).

| Action | From | To | Size |
|---|---|---|---|
| `mv` | `models/classifier` (v1) | `_archive_2026-05-03/models/classifier` | 1.8 G |
| `mv` | `models/classifier_v2` | `_archive_2026-05-03/models/classifier_v2` | 1.8 G |
| `mv` | `models/classifier_v3` | `_archive_2026-05-03/models/classifier_v3` | 1.8 G |
| `mv` | `models/classifier_v4` | `_archive_2026-05-03/models/classifier_v4` | 1.8 G |

Subtotal: ~7.2 G.

### A.4 Pre-DistilBERT trained Qwen runs — archived

`docs/refactor_plan.md` line 38 explicitly enumerates these as "all trained on the old (features → full JSON) task. Not reusable":

| Action | From | To | Size |
|---|---|---|---|
| `mv` | `models/runs/baseline` | `_archive_2026-05-03/models/runs/baseline` | 275 M |
| `mv` | `models/runs/epochs3` | `_archive_2026-05-03/models/runs/epochs3` | 275 M |
| `mv` | `models/runs/epochs5` | `_archive_2026-05-03/models/runs/epochs5` | 475 M |
| `mv` | `models/runs/preflight` | `_archive_2026-05-03/models/runs/preflight` | 275 M |
| `mv` | `models/runs/scaled_ctx2k` | `_archive_2026-05-03/models/runs/scaled_ctx2k` | 4 K (empty) |
| `mv` | `models/runs/scaled_ctx4k` | `_archive_2026-05-03/models/runs/scaled_ctx4k` | 4 K (empty) |

Subtotal (includes the parent `models/runs/`): ~1.3 G.

Also archived as part of the same era:

| Action | From | To | Size |
|---|---|---|---|
| `mv` | `models/qwen3-4b-structure` | `_archive_2026-05-03/models/qwen3-4b-structure` | 275 M |

This was the early Qwen-as-structure-decoder experiment (TRL-trained checkpoint + a single `checkpoint-23`). Predates the v1-v8 reasoner LoRA family. No script references it.

### A.5 Old Qwen training datasets (current = `qwen_dataset_v8`) — archived

`Plans/01_implementation_plan.md` §0.1: "Trained data at `data/qwen_dataset_v8`." Grep confirms only `scripts/run_v7_full_rebuild.sh` (historical) references v7 datasets. v8 rebuild script does not depend on v7.

| Action | From | To | Size |
|---|---|---|---|
| `mv` | `data/qwen_dataset_v7` | `_archive_2026-05-03/data/qwen_dataset_v7` | 303 M |
| `mv` | `data/qwen_dataset_v7d` | `_archive_2026-05-03/data/qwen_dataset_v7d` | 204 M |
| `mv` | `data/qwen_dataset_v7ds` | `_archive_2026-05-03/data/qwen_dataset_v7ds` | 260 M |

Subtotal: ~767 M.

### A.6 Orphan dataset — archived

| Action | From | To | Size |
|---|---|---|---|
| `mv` | `data/dataset/` | `_archive_2026-05-03/data/dataset/` | 4.2 M |

Three JSONLs (`train.jsonl`, `val.jsonl`, `test.jsonl`) from April 21. Zero references in any code, script, config, or doc. Per docstring of `scripts/run_v8_full_rebuild.sh` and `data/build_qwen_data.py`, the canonical dataset paths are `data/qwen_dataset_v*` — `data/dataset/` predates that convention and is orphan.

### A.7 Stale `eval/results/` — archived

Newer aggregated v7 family results stay; older per-pair classifier diffs are stale.

| Action | From | To | Size |
|---|---|---|---|
| `mv` | `eval/results/v1_vs_v2_5pdfs.json` | `_archive_2026-05-03/eval/results/` | 2.2 M |
| `mv` | `eval/results/v2_vs_v3_5pdfs.json` | `_archive_2026-05-03/eval/results/` | 2.2 M |

Both reference now-archived classifiers (`classifier`, `classifier_v2`, `classifier_v3`). Kept: `v7_family_20260426-110821.json`, `v7_family_20260426-220735.json` (current eval shape).

### A.8 Stale `eval/side_by_side/` runs — archived

Kept: anything tagged `_v7` (matches the current adapter family used by `eval_v7_family.sh`).

| Action | From | To | Size |
|---|---|---|---|
| `mv` | `2209.03909v1_Structured Negativity_..._20260423-1431` | archive | 528 K |
| `mv` | `2209.03909v1_v2adapter_20260423-2054` | archive | 4 K |
| `mv` | `2209.03909v1_v4adapter_20260424-1109` | archive | 4 K |
| `mv` | `2209.03909v1_v5adapter` | archive | 520 K |
| `mv` | `2209.03909v1_v6adapter` | archive | 524 K |
| `mv` | `diag_classv3_reasonerv6` | archive | 4 K |
| `mv` | `reasoner_v1_20260423-1417` | archive | 148 K |
| `mv` | `reasoner_v1_20260423-1518` | archive | 176 K |
| `mv` | `reasoner_v2_20260423-2125` | archive | 88 K |
| `mv` | `v6adapter_fixed_extract` | archive | 472 K |
| `mv` | `glm_ocr_*` (7 dirs total) | archive | ~1.9 M aggregate |

Subtotal: ~4.4 M.

### A.9 `.gitignore` update — committed

Single committed change: added `_archive_*/` rule so the cleanup holding directory does not pollute `git status`.

```
commit 1dc5a70 cleanup: gitignore _archive_*/ for cleanup-pass holding directory
```

(All other cleanup actions touched gitignored paths only and produced no per-file commits — there is nothing for git to track.)

---

## Section B — Code-level dead items already in `docs/bloat_audit.md` and now resolved

Verified the Phase 1 (30-min safe) items in `docs/bloat_audit.md` §1.1–§3.2 are **all already closed in tree** before this pass:

| bloat_audit item | Status | Note |
|---|---|---|
| §1.1 Mark `data/build_classifier_data.py` `[DEPRECATED]` | DONE pre-pass | Module-level docstring opens with `[DEPRECATED]`; `main()` prints a `WARNING:` line to stderr at line 220. |
| §1.2 Drop `NavigableString, Tag` from v1 builder | DONE pre-pass | Neither symbol present in current `build_classifier_data.py`. |
| §1.3 Delete `featurize_pdf_shared()` from `features.py` | DONE pre-pass | Not present in `dart_semantic/features.py`. |
| §3.1 Drop unused `pypdfium2`, `pytesseract` imports in `features.py` | DONE pre-pass | Comment at line 32 documents the deliberate non-import. |
| §3.2 `__init__.py` docstring parenthetical | DONE pre-pass | Current docstring lists eight stages cleanly with no stale "(currently in scripts/)" parenthetical. |
| §2.1 `feature_block_to_classifier_input` vs `features_to_input_string` dedup | OPEN (deferred per audit Phase 2) | v2 builder still has a parallel implementation; `bloat_audit.md` Phase 2 marks this as deferred until task #43 closes. Untouched. |
| §2.2 Centralize `_normalize` + `_text_overlap` | DONE | `data/build_classifier_data.py:158` already imports `jaccard_overlap` from `dart_semantic.text_utils`. Audit also self-marks DONE. |
| §2.3 Centralize `_caps_pattern` | DONE per audit | Audit self-marks DONE. |
| §5.1 Add comment to `experiments/configs.yaml` explaining v1/v2 mix | OPEN | Documentation-only; no risk. Untouched (see Section D). |
| §5.2 README references v1 builder as canonical | OPEN | The README is on the no-touch list; user has uncommitted edits already. Untouched. |
| Phase 2 #5 (delete `data/build_classifier_data.py`) | OPEN | Script still has the deprecation banner; deleting it would orphan the README v1-training instructions. See Section D. |

---

## Section C — Newly discovered dead code (beyond bloat_audit)

### C.1 `scripts/sweep.py` + `eval/compare.py` + `experiments/configs.yaml` form a zombie workflow

- `scripts/sweep.py:61` writes to `models/classifier_runs/{name}` — a directory that **does not exist and has never existed** (the actual sweep target was `models/runs/`, which is now archived in A.4).
- `eval/compare.py` is a thin diff over `models/classifier_runs/*/summary.json` that nothing produces.
- `experiments/configs.yaml` defaults `baseline`/`epochs5` to `data/classifier_dataset` (v1, kept around) and `v2_baseline`/`v2_epochs5` to `data/classifier_dataset_v2` (kept around).
- The whole sweep harness was designed for a hyperparameter sweep that was never re-run after the v3/v5 datasets landed.

`README.md` mentions the workflow at lines 78-79. Plan §2.1 is silent on it — falls under "ambiguous → human review" per task brief.

### C.2 Orphan diagnostic scripts

| Path | LOC | Last invocation context |
|---|---|---|
| `scripts/classifier_confusion.py` | 106 | Targets `classifier_v4` against `classifier_dataset_v3` (both archived in this pass). Built for the v4-era heading-bias hypothesis; that hypothesis is closed. |
| `scripts/validate_alignment_fix.py` | 192 | One-shot v7→v8 alignment-fix smoke check, per its docstring. v8 dataset is now stable. |
| `scripts/eval_reasoner_quick.py` | 0 grep refs | Standalone side-by-side eval helper for a reasoner adapter; superseded by `scripts/eval_v7_family.sh` + `scripts/_aggregate_v7_eval.py`. |
| `scripts/glm_ocr_smoke.py` | — | Smoke test for GLM-OCR region routing; output went into `eval/side_by_side/glm_ocr_*` which is archived in A.8. |

These are orphan but tracked in git and fairly small. Listed for human review (Section D), not auto-archived, because:

- they are *plausibly* useful to re-run as templates for future eval/diag work, and
- per the conservative deletion rule, "Plan or bloat_audit silent → leave alone."

### C.3 `train_classifier.py` default points at v1 paths

`train_classifier.py:150` defaults `--dataset-dir` to `data/classifier_dataset` (v1) and `--output-dir` to `models/classifier` (v1, now archived). Live invocations in `scripts/run_v7_full_rebuild.sh:56-58` override both flags, so this is not a runtime bug — but defaults will now error if invoked bare. **Not touched** per the rule "Never touch train_classifier.py."

### C.4 `data/build_qwen_data.py` default points at `data/qwen_dataset_v2`

`data/build_qwen_data.py:745` — same shape as C.3. Never invoked bare; both rebuild scripts override `--out-dir`. Plan §2.1 says "Phase out" not "edit". Untouched.

### C.5 `data/logs/` accumulating ad-hoc training logs

2.8 M of mixed v3-v8 era logs. Already covered by `*.log` in `.gitignore`. The directory is the active log destination (`scripts/run_v8_full_rebuild.sh:25`), so the path itself must stay. Pre-v7 era logs (`v3_*.log`, `v4_*.log`, `v5_*.log`, `v6_*.log`, plus `glm_ocr_*.log`, `infer_glm_ocr*.log`) reference workflows superseded by v8. Not auto-deleted; flagged in Section D.

### C.6 `eval/scorecards/` from old encoder runs

`eval/scorecards/{baseline,epochs3,epochs5}.{json,samples.jsonl}` — the trio matches `models/runs/{baseline,epochs3,epochs5}/` (archived in A.4). `docs/refactor_plan.md:38` explicitly says **"Keep `eval/scorecards/*.json` as historical data points."** Not touched.

---

## Section D — Needs human review

### D.1 `data/build_classifier_data.py` — bloat_audit Phase 2 says delete

12 K, 269 LOC. Already prints DEPRECATED warning at runtime. `bloat_audit.md` Phase 2 #5 says "Delete `build_classifier_data.py`" once task #43 closes. Plan §2.1 is silent. The README still calls it the canonical training-data builder (lines 71, 92, 119), so deleting it without a coordinated README update would leave broken instructions.

- **Why might be deletable:** v2 (`build_classifier_data_v2.py`) supersedes; v1 carries deprecated banner; no other module imports it.
- **Why might not:** README still references it; git history shows it was kept deliberately as a fallback.

### D.2 `scripts/sweep.py` + `eval/compare.py` (~10 K combined)

Dead workflow per C.1. Keeping them costs almost nothing. Removing them would also require a README cleanup (lines 78-79, 128).

- **Deletable:** writes to `models/classifier_runs/` which never existed; no caller; one of the configs (`baseline`/`epochs5` in `experiments/configs.yaml`) is duplicated.
- **Not deletable yet:** Phase 0 of `Plans/01_implementation_plan.md` DP-0.2 cites `experiments/configs.yaml` as "matches the YAML config style we want for council" — the file may be retained as a style reference. The two scripts however do not need to live alongside it.

### D.3 `experiments/configs.yaml` v1/v2 mix

Action recommended in `bloat_audit.md` §5.1: "Add a top-of-file comment explaining intent." Two-minute fix. Deferred here because docs Phase 1 was not in this pass's scope.

### D.4 `scripts/classifier_confusion.py` (106 LOC)

Built specifically for `classifier_v4` + `classifier_dataset_v3`. Both will be archived (v4) or are still resident (dataset_v3) after this pass. The "heading-bias" hypothesis it tests is closed.

- **Deletable:** narrowly scoped to a closed hypothesis; targeted classifier is archived.
- **Not deletable:** runtime takes flags so could be re-pointed at any classifier. May still be useful as an investigative template.

### D.5 `scripts/validate_alignment_fix.py` (192 LOC)

One-shot v7→v8 alignment regression check.

- **Deletable:** that regression has been validated and the v8 alignment ships.
- **Not deletable:** could be repurposed if v9 introduces alignment changes.

### D.6 `scripts/eval_reasoner_quick.py`

Side-by-side reasoner eval helper. Zero grep references but could be hand-invoked. Recommend keeping until target arch is mature enough that the council eval shape (per `Plans/01_implementation_plan.md` §5.1) replaces it.

### D.7 `scripts/glm_ocr_smoke.py`

Smoke test for the GLM-OCR region router. Output dirs (`eval/side_by_side/glm_ocr_*`) all archived. Probably keep until the GLM-OCR routing path is locked in.

### D.8 `data/classifier_dataset/` (12 M, v1)

Referenced by:
- `train_classifier.py:150` default
- `experiments/configs.yaml` (`baseline`/`epochs5`)
- `data/build_classifier_data.py:231` default

All three paths are themselves on the deprecation track. Once D.1, D.2, D.3 resolve, this dataset is archivable.

### D.9 `data/classifier_dataset_v2/` (54 M)

Intermediate stage between v1 and v3. `data/append_ocr_classifier_examples.py:127` reads it (`--v2-dir`), `experiments/configs.yaml` (`v2_baseline`/`v2_epochs5`) trains on it, and `data/build_classifier_data_v2.py` writes to it. v3 = v2 + OCR examples; if v3 is the only training target now, v2 may be reproducible from v2-builder + pairs. Currently a live input to the rebuild scripts so kept.

### D.10 `data/classifier_dataset_v3/` (28 M)

Current classifier training dataset (target of `data/append_ocr_classifier_examples.py`, consumed by `scripts/run_v7_full_rebuild.sh:57`). **Keep.**

### D.11 Pre-v7 era logs in `data/logs/`

About 2.0 M of `v3_*`, `v4_*`, `v5_*`, `v6_*`, `glm_ocr_*` logs from past rebuilds. `*.log` is already gitignored. Safe to delete on a manual sweep; not auto-deleted.

### D.12 README still describes v1 workflow

`README.md` references `data/build_classifier_data.py` (deprecated), `models/classifier/final` (archived in A.3), `scripts/sweep.py` + `eval/compare.py` (zombie per C.1). User has uncommitted edits already; per task rules, the README is on the no-touch list.

### D.13 `scripts/split_chunks.py` and `scripts/densify_targets.py` defaults reference v3

`scripts/split_chunks.py:126-127` defaults to `data/qwen_dataset_v3d`/`v3ds`. `scripts/densify_targets.py:14-15` defaults to `data/qwen_dataset_v3`/`v3d`. Live rebuild scripts always pass explicit `--in-dir`/`--out-dir`; defaults are stale but not broken. Not edited in this pass.

### D.14 `models/classifier_v5/checkpoint-7386` and `checkpoint-11079`

Checkpoints alongside `final/`. Per HF convention `final/` is the promoted snapshot; intermediate checkpoints are training artifacts. Archivable if the user wants to free another ~3.6 G but kept here because Phase 3b "subsumes" classifier_v5 (Plan §2.1) — keeping checkpoints lets a re-train resume from mid-run if needed.

Same for `models/reasoner_v8/checkpoint-375`.

---

## Section E — Disk-space summary

### Top-level directories (before / after)

| Path | Before (approx) | After |
|---|---|---|
| `models/` | ~12.8 G | 1.9 G |
| `data/` | ~3.5 G | 2.5 G |
| `eval/` | 13 M | 3.3 M |
| `_archive_2026-05-03/` | 0 | ~10 G (holding pen) |

(Repository-relative — does not include `.venv/`, `.git/`.)

### Net effect on live working tree

- **Live working tree:** went from ~16.3 G of models+data+eval to ~4.4 G (excluding `.git`, `.venv`).
- **Archive holding pen:** ~10 G in `_archive_2026-05-03/`. User can `rm -rf _archive_2026-05-03/` after spot-checking; that frees the full 10 G.
- **Outright deleted:** ~480 K of `__pycache__/`.

### Reclaimable on archive-purge

| Bucket | Size in archive |
|---|---|
| Old reasoner adapters (v1-v7) | ~678 M |
| Old classifier adapters (v1, v2, v3, v4) | ~7.2 G |
| `models/runs/*` + `models/qwen3-4b-structure/` | ~1.6 G |
| Old qwen datasets (v7) | ~767 M |
| `data/dataset/` orphan | 4.2 M |
| Old eval/results + side_by_side | ~6.6 M |
| **Total** | **~10.3 G** |

---

## Risks / surprises encountered

- **None of the live infer/eval/training scripts reference any of the archived paths as a default.** Verified via grep across `*.py`, `*.sh`, `*.yaml`, `*.json`, `*.md` (excluding `.git`, `.venv`, `__pycache__`, `eval/results`, `eval/side_by_side`, the new `_archive_*`).
- **`models/runs/` was both gitignored and explicitly enumerated as dead in `docs/refactor_plan.md`.** Highest-confidence archival of the pass.
- **`scripts/sweep.py` writes to `models/classifier_runs/` which has never existed in the tree.** That confirms it has not been run successfully against this layout — likely because the script was never updated when the `models/runs/` rename happened. Strong dead-code signal but plan-silent, so left for human review.
- **Per-action commits.** The brief asks for per-action git commits. In practice almost every cleanup action operates on gitignored content (matching the `.gitignore` rules for `models/`, `data/qwen_dataset_*/`, `data/classifier_dataset*/`, `data/dataset/`, `eval/results/`, `eval/side_by_side/`), so git produces nothing to commit. The single visible commit for this pass is the `.gitignore` update at `1dc5a70`.
- **No stale-reference broken imports detected.** Every `from dart_semantic.<module>` path in the tree resolves to a still-present module. No script references a model path that doesn't exist *and* is the runtime default. The closest near-miss was `scripts/sweep.py` writing to `models/classifier_runs/` — but that path was never the default for any other script.

---

## Suggested next pass (when v1 retires per Phase 3b validation)

1. Delete the archive directory: `rm -rf _archive_2026-05-03/`.
2. Resolve D.1-D.7 (orphan code modules).
3. Coordinated README rewrite to drop v1-era instructions (sweep, compare, v1 builder).
4. After Phase 3b ships: archive `data/classifier_dataset` and `data/classifier_dataset_v2` (D.8, D.9) and the v1 path of `train_classifier.py`.
