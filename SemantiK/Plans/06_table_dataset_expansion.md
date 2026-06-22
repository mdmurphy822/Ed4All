# Table Adapter Dataset Expansion — Multi-Source, Multi-Format, Multi-Type

> **STATUS (banner added 2026-06-17): FOLDED INTO Plan 12.** Now tracked as a sub-item of
> Plan 12 (§C/§D); see Plan 12 for current status. Historical detail retained here.

**Plan version:** 2026-05-26 (rev 2 — plan B + no-fetch finding)
**Branch:** `main`
**Status:** IN PROGRESS. Builders written + smoke-validated; full builds running.
Adopts **plan B** (dual-emit: one extraction feeds the Qwen adapter dataset AND
the BERT cell-role dataset) and the **no-fetch finding** (§0.1) — the external
PMC top-up is unnecessary, ~9K usable tables are already on disk.
**Self-contained** — do not need conversation history to execute.

---

## 0.1. No-fetch finding (2026-05-26) — supersedes the §3 fetch plan

Surveyed usable-markup table yield across on-disk pairs (skip graphic-only PMC,
skip layout/equation tables):

| Source | On disk | Usable tables | Notes |
|--------|--------:|-------------:|-------|
| arXiv (ar5iv) | 3,236 pairs | ~1,650 | existing dataset |
| **PMC (JATS)** | **4,993 pairs** | **~7,000** | 90% have real markup; only 10% graphic-only |
| OpenStax (HTML5) | 578 pairs | ~344 | ~510 layout tables correctly excluded |
| CFR / FedReg | 120 / 8 | a few hundred | optional |

**~9K usable tables without touching the network.** The PMC OA top-up that this
plan (§3) called the "biggest volume lever" is NOT needed — the 4,993 PMC docs
already fetched clear the 8–10K target. External fetches (§3) are deferred /
dropped. Gov-statistical remains a future option only if type-balance demands it.

---

## 0.2. Plan B — dual-emit (locked with user 2026-05-26)

The Qwen table adapter consumes `cell_roles` at inference from the **BERT
cell-role classifier** (`build_table_request` puts them in the prompt). That
BERT trains on `data/pairs/{arxiv, openstax, cfr, fedreg, wikipedia, forms}` —
**no PMC**. If we expand the adapter to PMC/gov tables but feed it BERT roles
trained only on the old mix, the adapter gets garbage roles on exactly the new
table types at inference and its improved generation can't help.

So one extraction feeds **both** datasets. The shared keystone is
`data/jats_tables.py` (`jats_to_html5_tables`): PMC JATS `<table-wrap>` →
accessible HTML5 `<table>`. Both builders ingest the SAME normalized HTML:

- `data/build_table_qwen_multi.py` — Qwen adapter dataset (parses HTML5 via lxml,
  reuses the ar5iv grid/role/target core).
- `data/build_table_specialist_data.py` — BERT cell-role dataset (extended to
  read PMC: appends JATS-normalized tables when a pair has `raw_source_xml`;
  `data/pairs/pmc` added to its default `--pair-dirs`).

**New required step (was missing from rev 1):** after the datasets rebuild,
**retrain the BERT cell-role classifier** on the PMC-inclusive set before/with
the adapter. `Structure`'s `table_region` head (PMC/CFR-absent) is a secondary,
lower-ROI extension (partly pdfplumber-driven on the PDF path) — track it but it
doesn't gate the adapter ship.

JATS quirk that bit us (now handled): PMC puts column headers in `<td>` inside
`<thead>`, not `<th>` — the normalizer promotes thead `<td>`→`<th scope="col">`,
else every PMC header reads as a data cell.

---

## 0. Why

The current table set (`data/qwen_table_dataset`) is **1,315 train rows, 100% arXiv/ar5iv
(LaTeXML)**. Two problems:

1. **Too small** for a generative LoRA — overfits at any epoch count. (Also interacts
   with the `total_step_cap` bug: cap 1500 vs 165 steps/epoch ⇒ ~9 epochs on 1.3K rows.
   See Plans/05 §… / Task #7.)
2. **Single source + single format + narrow type mix.** A table adapter trained only on
   LaTeXML `ltx_tabular` idioms generalizes poorly to the PDF/gov/scientific tables DART's
   buyers actually submit.

**Goal (locked with user 2026-05-25):** ~**8–10K** tables, diverse across **source**,
**format**, AND **table type**, fetching more data where needed.

## 1. Diversity axes

### 1a. Source / format (parser per format)
| Source | Format | On-disk yield | After fetch | License |
|--------|--------|------:|------:|--------|
| arXiv (have) | LaTeXML `ltx_tabular` | ~1,700 | — | ar5iv subset ✅ |
| PMC | JATS XML `<table-wrap>` | ~1.5–2.5K* | ~5K (fetch +3–4K OA) | CC‑BY ✅ |
| OpenStax | HTML5 `<table>` | ~700 | — | CC‑BY ✅ |
| CFR / Fed Register | gov CFR‑XML (`GPOTABLE`) | ~50 (10 docs) | few hundred (fetch) | US‑gov PD ✅ |
| **Gov statistical** (NEW) | Census/BLS/data.gov HTML | 0 (fetch) | ~1–2K | US‑gov PD ✅ |

\* PMC caveat: many `<table-wrap>` wrap a `<graphic>` (table-as-image), not markup —
those are unusable (would need OCR). Skip graphic-only; usable ≈ a fraction of the 3,473
raw `<table>`. Verify the markup fraction in stage 2.

**Wikipedia deliberately excluded** — CC‑BY‑SA is off the CC‑BY/CC0/ODC‑By allowlist
(see `feedback_license_policy`), and it's infobox/layout-heavy.

### 1b. Table TYPE (balance target — the axis the user flagged: "tables of datasets too")
Tag every row with a `table_type` and aim for a spread, not 90% one type:
- **simple/relational** — few columns, single header row (OpenStax textbook tables)
- **scientific results** — multi-level headers, footnotes, units (PMC, arXiv)
- **dense data / dataset** — many numeric rows, the "dataset table" type (gov statistical,
  arXiv data tables) ← explicitly sought
- **regulatory** — spanning cells, nested headers (CFR/FedReg)
- **matrix / cross-tab** — row+col headers (scientific)

## 2. The build — a multi-format table extractor

Today `build_table_qwen_data.py` is ar5iv-only (keys on `ltx_tabular` /
`<figure class="ltx_table">`). Generalize into per-format extractors that all normalize
to ONE accessible target shape:

1. **Per-source parsers** → emit a common intermediate `(caption, table_dom, source,
   table_type)`:
   - `ar5iv` (have): `ltx_tabular`, figure-caption wrapping.
   - `jats` (PMC): parse `<table-wrap>` → `<table>` + `<caption><title>`; **skip
     graphic-only**; map JATS `<thead>/<tbody>/<th>/<td>` (near-HTML already).
   - `html5` (OpenStax): plain `<table>` from `output_html`.
   - `cfr_xml` (CFR/FedReg): `GPOTABLE`/`BOXHD` → HTML table.
   - `gov_stat` (Census/BLS/data.gov): HTML `<table>` from fetched pages.
2. **Data-vs-layout filter** (critical for HTML sources): drop `role="presentation"`,
   infobox/navbox/layout tables, tables with no `<th>` and ≤1 data row, etc.
3. **Normalize to accessible target** (WCAG 2.2 AA): `<table>` with `<caption>`,
   `<thead>/<tbody>`, `<th scope=...>` (or `headers`/`id` for complex), strip
   presentational attrs. This is the single target contract the adapter learns.
4. **Type-tag** each row (§1b) for stratified reporting + balanced sampling.
5. **Dedup + split**: `data._splits.stable_split_for_id` (same doc → same split across all
   qwen_* builders), keyed on a per-source stable id.
6. **Tokenizer-aware length filter** at `max_len=1024` (mirror the prose/math builders —
   `--max-token-len` + wrapped-length drop; see Plans/05 outcome). Dense data tables can
   be long → expect meaningful drops; that's correct, not a bug.
7. **Coverage report**: rows per source / format / type / split, over_max drops, p50/p99
   token lengths.

## 3. Fetch steps (outward-facing — confirm before running)
- **PMC OA top-up**: `scripts/pair_from_pmc.py` — pull +3–4K more OA articles, CC‑BY/CC0
  license-gated (existing gate). Biggest volume lever.
- **Gov statistical**: NEW `scripts/pair_from_gov_stats.py` (or extend `pair_from_gov_forms`)
  — fetch Census/BLS/data.gov HTML data tables, public domain.
- CFR/FedReg top-up optional (lower ROI).

## 4. train_config implications (do alongside, not after)

**max_len = 2048 (locked with user 2026-05-26).** Tables are far longer than
prose/math: the prompt carries the full `cell_grid` JSON AND the target repeats
every cell as HTML, so wrapped length is ~2× cell content. Measured distribution
over the 9,027 unique tables: **p50=2,135, p90=6,755 tokens.** The prose/math
default of 1024 keeps only 17% (1,494 rows) — almost all tiny `simple` tables,
which defeats the expansion. Retention: 1024→1,494 / 1536→3,094 / **2048→4,337**
/ 3072→5,989 / 4096→6,957. 2048 keeps ~48% (4,337 rows, 3.3× the old arxiv-only
1.3K) with all four types >500, and is comfortably QLoRA-trainable on the 8GB
3070 (≈2× prose's seq len). Tables dropped at 2048 are mostly ungeneratable at
runtime anyway (target alone would exceed a sane `max_new_tokens`/`n_ctx`).

- **Runtime caps (carry-forward from [[qwen-runtime-output-caps]]):** before
  trusting the table eval, set the shipped `config.yaml` table `max_new_tokens`
  AND runtime `n_ctx` to clear the 2048-token target tail (n_ctx ≥ 4096,
  max_new_tokens ≥ ~1536). The math adapter shipped 960/1024 + n_ctx 4096;
  tables need more headroom on `max_new_tokens` since whole tables are emitted.
- With ~3,470 train rows and grad_accum 8 @ batch 1: ≈434 optimizer steps/epoch.
  For 3 epochs ≈ 1,300 steps. Set table `total_step_cap` ≥ 1,300 **and** fix the
  `max_steps = min(epochs×steps, cap)` semantics so the cap is a true ceiling
  (the bug that gave the old 1.3K set ~9 epochs).
- Reconsider `lora_r` (8→16) and `lora_dropout` now the set is 3.3× bigger and
  less overfit-prone (prose v1 used r=16/alpha=32/dropout=0.05).

## 5. Staging (each stage independently shippable)
1. ✅ **JATS keystone** — `data/jats_tables.py`. Validated: 0 ill-formed / 588
   tables in a 400-doc sample, thead `<td>`→`<th>` promotion confirmed.
2. ✅ **Multi-source adapter builder** — `data/build_table_qwen_multi.py`
   (arxiv + openstax + pmc). Smoke: 232 rows/360 pairs, contract test 14/14 green,
   36/36 targets pass axe wcag22aa (0 serious/critical). table_type spread:
   simple/dense_data/scientific/matrix all present.
3. ✅ **Plan-B cell-role extension** — `build_table_specialist_data.py` reads PMC
   via the JATS keystone. Smoke: 12,366 cells / 80 PMC docs, sane label dist.
4. ✅ **Full builds done.**
   - Adapter `data/qwen_table_dataset_multi` @ max_len 2048: **4,337 rows**
     (see §6). Contract 14/14, axe 65/65 clean.
   - Cell-role `data/table_specialist_dataset_pmc` (staging, v4 untouched):
     **799,495 cells** (train 639,594 / val 79,948 / test 79,953). Labels:
     data 666K / header_col 74K / span 48K / header_row 11K. By source:
     **pmc 662K (83%)** / arxiv 71K / wikipedia 54K / openstax 13K. CFR/FedReg
     emit 0 (GPOTABLE XML not yet parsed — deferred §6).
5. ◐ **Retrain BERT cell-role classifier** (plan B — see §0.2) on the
   PMC-inclusive set. **Heads-up:** the cell-role set is 83% PMC — even more
   skewed than the adapter. Downsample PMC and/or class-weight at train time so
   the BERT doesn't collapse to JATS conventions; `header_row` is rare (1.3%,
   mostly arxiv/wikipedia) so guard its recall. Then preflight + train the
   Qwen table adapter.
   - **STATUS 2026-05-27:** cell-role retrain RUNNING as `final_v5`
     (`models/council/table_specialist/final_v5`, dataset
     `data/table_specialist_dataset_v5`, weighted CE + sampler cap 3.0 for the
     PMC/`header_row` skew). Epoch 1/6: loss 0.126, val macro-F1 0.755. The
     **Qwen table adapter** launch is CHAINED behind it (serial GPU):
     `train_qwen_lora.py --adapter table --out-dir models/qwen_specialists/table/v1`
     reading `qwen_table_dataset_multi` @ max_len 2048. The adapter is
     independent of the cell-role BERT (trains on ground-truth cell roles from
     source markup, not BERT predictions), so it fires as soon as the GPU frees
     regardless of `final_v5`'s final metrics.
   - **PMC-skew decision (adapter):** ACCEPT 69% PMC as-is — option (a). A
     generative SFT adapter has no clean class-weight lever, and the dense gov
     tables that would dilute PMC are exactly the ones too long for the 2048
     cap (§4). `source` is recorded per row → measure quality **stratified by
     source** in the eval harness instead.
6. ◐ (optional) CFR-XML `GPOTABLE` parser; **`Structure` `table_region`
   extension → now scoped in [Plans/07_region_head_hardening.md](07_region_head_hardening.md)**
   (pilot validated 2026-05-27: render new HTML tables → PDF → same
   pdfplumber feature path → Structure-format `table_region` rows;
   `scripts/synthesize_region_features_from_html.py`). Bulk render + GPU
   retrain are the deferred follow-up there (serial-GPU-blocked behind
   `final_v5`).

**Deferred/dropped:** external PMC OA + gov-statistical fetches (§3) — see §0.1.

## 6. Acceptance gates — measured (2026-05-26, max_len 2048)

Final adapter dataset `data/qwen_table_dataset_multi`: **4,337 rows**
(train 3,448 / val 431 / test 458). caption 89.8%, th_scope 93.4%.

| Gate | Target | Actual | Verdict |
|------|--------|--------|---------|
| total rows | ≥ 8K (pre-cap pool) | 4,337 @ max_len 2048 | **revised** — 8K was the no-cap pool; tables are too long to keep all (§4). 4,337 is 3.3× the old set. |
| no single source > ~55% | | **pmc 69%** / arxiv 24% / openstax 7% | ⚠ **PMC-skewed** — see caveat below |
| each table_type ≥ 5% | | simple 64% / scientific 22% / matrix 9% / **dense_data 5.0%** | ✅ (dense_data just clears; the 2048 cap hits long dense tables hardest) |
| tokenizer filter (no row > 2048) | | 4,692 dropped at 2048 | ✅ enforced |
| targets well-formed accessible `<table>` | | contract 14/14; **65/65 axe wcag22aa pass**, 0 serious/critical | ✅ |
| no layout/graphic-only tables | | 723 layout + graphic-only skipped | ✅ |
| license provenance per row | | `source` field on every row | ✅ (CC-BY/PD gated at fetch) |

**PMC-skew caveat (open decision).** PMC is 69% of the set, over the ~55%
diversity target — because PMC is by far the largest on-disk source. Options if
skew matters for the adapter: (a) accept it (PMC is high-quality scientific
markup, and source is recorded for stratified eval); (b) cap PMC to ~55% by
downsampling (`--max-pairs` per source or a post-build subsample), trading ~900
rows. Not blocking; flagged for the train step. **dense_data** is also thin at
5% — the long dense tables we most wanted are exactly what the 2048 cap drops;
revisit if dense-table generation underperforms in eval.

## 7. Follow-up — gov-table coverage gap (open)

**Logged 2026-05-27.** The table *adapter* train/test sets cover scientific/
academic tables well (pmc 71% / arxiv 23% / openstax 5%) but have **near-zero
gov coverage**: `federal_register` and `nces_digest` are **absent from the
adapter dataset entirely**, and `cfr`/`regulatory` are n=1 in the test split.

**Why (not an oversight):** those dense gov-statistical/regulatory tables
exceed the max_len 2048 token cap (§4 — NCES median ~48K wrapped tokens), so
they were dropped from the *generative* adapter and kept only for the
*cell-role BERT* (cell-level, no length limit). The eval therefore says
nothing about how the adapter handles gov tables — and it likely never can,
because they're too long to generate token-by-token.

**Proposed direction (not the adapter):** serve long gov tables through the
**deterministic assembler** path driven by the cell-role BERT (`final_v5`) +
`Structure` `table_region` head — i.e. classify cell roles and *assemble* the
accessible `<table>` in code, rather than *generate* it with Qwen. This sidesteps
the length ceiling. Scope a separate eval for that path (the cell-role BERT is
already source-stratified incl. pmc/nces; the missing piece is an end-to-end
"cells → assembled accessible table" check on long gov tables). Cross-ref the
assembler passes (`dart_semantic/assembler/`) and `table_cell_builder.py`.

## Do-NOT list
- No CC‑BY‑SA / NC / ND sources (Wikipedia out). Only CC‑BY / CC0 / ODC‑By / US‑gov PD.
- No silent fallbacks in parsers — unknown table shape raises, not drops silently.
- Confirm before any external fetch.
- Serial GPU only when we later train (parallel poisons CUDA context).
