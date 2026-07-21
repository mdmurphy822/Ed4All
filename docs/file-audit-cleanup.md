# File-Audit Cleanup List

Actionable output of the repo-wide file-hygiene audit (read-only). Companion to
the [`FILE_MANIFEST.md`](FILE_MANIFEST.md) orientation map.

> **Nothing was deleted during the audit.** Every item below is a
> **recommendation only**. DELETION REQUIRES HUMAN CONFIRMATION.
>
> **A live pipeline may be writing** to `state/runs/`, `state/workflows/`, and
> `runtime/`. Do **not** delete any active-run artifact under those paths — prune
> only clearly-completed/stale entries, and never touch in-flight runs.

---

## 1. Data-leak fixes (gitignore) — highest priority

**None.** Across all 14 audited areas,
`git status --porcelain --untracked-files=all` returned **zero untracked,
non-ignored files** — i.e. no source, data, or output can currently leak to
GitHub. The `.gitignore` is comprehensive:

- `runtime/` wholesale; `ed4all.egg-info/`, `__pycache__/`, `.pytest_cache/`,
  `*.egg-info/` all confirmed ignored.
- `state/<dir>/*` per-dir rules **plus** the defensive `state/*/*` catch-all with
  `!state/*/.gitkeep`, so a new `state/` subdir cannot leak.
- Per-area blankets: `SemantiK/data/*`, `Courseforge/exports/*`,
  `Courseforge/inputs/textbooks/*`, `Trainforge/output/*`, `LibV2/courses/*`,
  `LibV2/catalog/*`, `inputs/*`, `training-captures/<subdir>/*`.
- `ci/` is **USED** (referenced by `.github/` workflows) — not flagged.

### Uncommitted new work (not leaks — should be `git add`ed)

Two well-formed **source/doc** files are untracked-and-not-ignored; they are new
work to commit, not hygiene risks:

| Path | What | Action |
|------|------|--------|
| `docs/architecture/hybrid-vision-extraction.md` | Legitimate architecture doc | `git add` and commit |
| `scripts/integration/vision_ocr_probe.py` | Vision-vs-OCR fidelity probe (companion to tracked `ocr_recall_ab.py` family), referenced by the doc above | `git add` and commit |

### One theoretical residual (no action)

A loose non-`.log`, non-`nvidia_*` file dropped directly at `state/` **root**
(e.g. `state/foo.json`) would not be caught by `state/*/*`. No such file exists
today; monitor only.

---

## 2. Trash / extra candidates

All items are **untracked + gitignored** (cannot leak to GitHub) — these are
**local disk-hygiene** items only. Sorted high → low confidence.

| Path | Tracked? | Reason | Confidence |
|------|:--------:|--------|:----------:|
| `runtime/fix_lo_refs.py` | no | One-off `fix_*` scratch script (Jun 8) in runtime output dir | **high** |
| `runtime/complete_content.py` | no | One-off `complete_*` scratch script, unreferenced | **high** |
| `runtime/complete_fresh.py` | no | One-off `complete_*` scratch script, obsolete patch-up | **high** |
| `runtime/_fres_content-generator-*.json` (13) + `_res_content-generator-*.json` (12) | no | Per-worker result blobs from a Jun 8 run; disposable | **high** |
| `runtime/result_course_planning.json`, `runtime/result_w01.json` | no | Stray single-phase result dumps (Jun 8) | **high** |
| `runtime/pipeline_*.log` (11 logs, incl. two ~945 KB) | no | Stray run logs (Jun 8), large + obsolete | **high** |
| `runtime/qwen_test/` | no | May 3 model-comparison scratch dir | **high** |
| `state/logs/*.log.contaminated` | no | Explicitly-named discarded/bad run log | **high** |
| `state/workflows/*.json.pre-remediation.bak` | no | Hand-made `.bak` of a workflow state file | **high** |
| `component_applier.log`, `imscc_extractor.log`, `remediation_validator.log` (repo root) | no | 0-byte stray logs (Jun 30) | **high** |
| `runtime/build_corpus_pdfs.py` | no | Ad-hoc corpus-PDF builder (tracked equiv under `scripts/`) | medium |
| `runtime/kg_prototype/` | no | Jun 8 KG prototype scratch, superseded, unwired | medium |
| `runtime/gui_course_corpus/` + `runtime/gui_course_corpus_pdf/` | no | Jun 8 generated demo-course corpus (regenerable) | medium |
| `state/logs/*.log` (old per-run diagnostic logs, some multi-MB) | no | Old per-run diagnostic logs | medium |
| `state/nvidia_authoring_contract.md`, `state/nvidia_page_manifest.json`, `state/nvidia_remaining_dispatch.tsv` | no | Operator working notes at `state/` root (Jun 9) | medium |
| `training-captures/courseforge/TEST_CHAIN/` | no | Decision captures under a test-run course code | medium |
| `inputs/**/*.log` (per-corpus run logs) | no | One-off pipeline run logs — **pipeline may still append** | low |
| conversion-output scratch (dated `backup_pre_rerender_*` / `rerender_*` rounds under a corpus's gitignored working tree) | no | Dated backup + re-render scratch rounds — **live conversion may write here** | low |
| `state/runs/` (accumulated timestamped dirs + a baseline snapshot `.md`) | no | Accumulated historical run-state — **LIVE pipeline writing here**; prune completed only | low |
| `.codex` (repo root) | no | 0-byte editor/tool marker | low |

---

## 3. Build artifacts

**All present build artifacts are already gitignored** — nothing to add:

- `__pycache__/` / `*.pyc` everywhere (SemantiK, Courseforge, Trainforge, LibV2,
  MCP, lib, cli, gui) — global `__pycache__/` rule.
- `.pytest_cache/` (Courseforge, root).
- `ed4all.egg-info/`, `.venv/` (root); `inputs/**/social_override.egg-info` —
  `*.egg-info/` rule.
- `scripts/shots/` (rendered PNGs); `Trainforge/output/` course artifacts;
  SemantiK `data/` caches.

No un-ignored build artifacts were found in any area.

---

## 4. Summary counts

| Category | Count | Notes |
|----------|:-----:|-------|
| **Real gitignore gaps (leaks)** | **0** | Every untracked path resolves to an ignore rule |
| Uncommitted source/docs to `git add` | 2 | `hybrid-vision-extraction.md`, `vision_ocr_probe.py` |
| Trash candidates — high confidence | 10 rows (~50 files) | Mostly `runtime/` scratch + 3 root 0-byte logs + 2 hand-made `.bak`/`.contaminated` |
| Trash candidates — medium confidence | 6 rows | runtime prototypes/corpora, old `state/logs`, nvidia notes, `TEST_CHAIN` captures |
| Trash candidates — low confidence | 5 rows | Active-write paths (`state/runs/`, conversion-output scratch, `inputs/**/*.log`) + `.codex` |
| Un-ignored build artifacts | 0 | All caches/egg-info/PNGs already ignored |

**Bottom line:** the repo has **no data-leak exposure**. All cleanup is optional
local disk hygiene, concentrated in `runtime/` scratch and a few stray root logs.
Defer anything under `state/runs/`, `runtime/`, and the conversion-output scratch
trees that the running pipeline may still touch.
