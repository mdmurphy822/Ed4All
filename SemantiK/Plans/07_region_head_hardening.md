# Structure `table_region` Head Hardening — Render-to-PDF Feature Synthesis

> **STATUS (banner added 2026-06-17): SUPERSEDED.** The render-to-PDF feature-synthesis
> approach was not adopted (`features.py::render_to_pdf` remains an intentional stub);
> `table_region` gating ships via the Structure head. Tracked under Plan 12. Historical.

**Plan version:** 2026-05-27 (rev 1)
**Branch:** `main`
**Status:** DATASET BUILT. Synthesis script validated on pilot, then bulk
render (steps 1–2 below) COMPLETED 2026-05-27: 7,971 tables → 561K synth
rows → balanced ~64K subsample merged into `data/structure_dataset_v2`
(train 149,427 / val 18,118 / test 19,802), validated through the real
`train_structure.load_split`. Only the **GPU retrain (step 3) remains** —
serial-GPU-blocked behind `final_v5`, PID 905. **Self-contained.**

Cross-links: this realizes the deferred item in
**Plans/06 §5.6** ("`Structure` `table_region` extension") and reuses the
same table corpus / normalizers (`data/jats_tables.py`,
`data/gpotable_tables.py`) that Plans/06 §0.2 (plan B) standardized on.

---

## 0. Rationale

`Structure`'s `table_region` binary head
(`dart_semantic/council/structure.py`:67,78,264 —
`TABLE_REGION_LABELS = ("not_table_region", "table_region")`,
`head_table_region`) is the gating signal that hands a span to the
TableSpecialist (the `is_heading→HeadingSpecialist` pattern). It is
trained by `train_structure.py` on `data/structure_dataset`, built by
`data/build_structure_data.py`.

**The gap.** That dataset is built from
`data/pairs/{wikipedia, openstax, federal_register, gutenberg, forms,
arxiv, synthetic_blockquote_code}`
(`build_structure_data.py:693-700`) — it is **PMC/CFR/NCES-table
absent**. Plans/06 brought ~9K new tables (PMC JATS, OpenStax HTML5,
NCES, CFR/FedReg GPOTABLE) into the corpus for the Qwen table adapter and
the cell-role BERT, but the Structure `table_region` head never saw them.
The head's current test pos-F1 is **0.924** (on the table-sparse mix);
it has had almost no exposure to the table geometry of the document types
DART's buyers actually submit, and the 20-dim layout side-channel never
learned the rendered-table signature from those sources.

**The fix.** Render each new HTML table to PDF and run the **same**
pdfplumber feature path the live pipeline uses, emitting Structure rows
where the table's blocks are `table_region=1` and synthetic surrounding
context is `not_table_region`. This hardens the head on the new sources
*at the same train/inference feature distribution*.

## 1. Approach — render → feature → label

Script: **`scripts/synthesize_region_features_from_html.py`**.

Per HTML `<table>` in the corpus:

1. **Table source.** `tables_from_pair()` mirrors
   `build_table_specialist_data.process_pair`'s source handling:
   - OpenStax / NCES: raw `<table>` substrings from `output_html`.
   - PMC: `data.jats_tables.jats_to_html5_tables(raw_source_xml)`.
   - CFR / FedReg: `data.gpotable_tables.gpotable_to_html5_tables`.
   Same 2×2+ and ≥50%-non-empty data-table gate as the cell-role builder
   (drops layout/pseudo tables). Normalizer failure raises
   `TableExtractionError` (no silent drop).
2. **Wrap** in a minimal accessible HTML5 doc (`lang`, `<title>`,
   `border-collapse` borders so pdfplumber's `find_tables` fires) with
   synthetic context blocks above/below: one `<h2>` heading + two `<p>`
   paragraphs. These are the genuine `not_table_region` negatives that
   share page geometry with the positives.
3. **Render** via `dart_semantic.validate.HtmlValidator.render_pdf`
   (`validate.py:130`) — the SAME Chromium `page.pdf()` path
   `build_structure_data.py:520` uses for synthetic HTML. **Rendering
   tooling decision:** reuse Playwright/Chromium (already a dep,
   `pyproject.toml`; Chromium verified launchable). No weasyprint added.
4. **Extract** via `dart_semantic.extract_shared.extract_shared`
   (`extract_shared.py:65`) — merged text blocks + pdfplumber table
   bboxes, identical to the live Stage-1 path.
5. **Features (REUSED, not reimplemented).** Per merged block:
   - `in_table` ← `data.build_structure_data._block_in_any_table`
     (bbox-center-in-table-bbox).
   - 20-dim layout ← `data.build_structure_data.compute_span_layout_features`
     — the exact function whose order
     `dart_semantic/council/structure.py:_compute_span_layout` mirrors.
6. **Label.** `table_region=1` if `in_table` OR the block text aligns
   (`jaccard_overlap ≥ 0.30`) to a known cell (borderless-table backstop,
   mirroring the `<table>`-ancestor backstop in
   `build_structure_data.py:608`). Context blocks → `0`. `structural_role`
   follows the builder: cell text → `paragraph`, context heading →
   `heading`. Row emitted in the canonical schema and validated.

## 2. Schema-match requirement (hard gate, no silent fallback)

Every emitted row passes `_validate_row()` before write; a mismatch
raises **`StructureSchemaMismatch`** (repo convention — same spirit as
the Stage-13 `StageThirteenStubRequired`; see
`feedback_no_silent_fallbacks`). The canonical row contract
(`build_structure_data.py:632`):

```json
{"text": str, "layout": [20 floats],
 "labels": {"structural_role": int, "is_heading": int, "table_region": int,
            "is_image_block": int, "list_nesting": int},
 "html_tag": str, "source": str, "pair": str}
```

The script also diffs its contract against a real
`data/structure_dataset/train.jsonl` row at startup (`--reference-row`).

## 3. Pilot results (2026-05-27)

Command:
```
PYTHONPATH=$PWD python scripts/synthesize_region_features_from_html.py \
    --limit-per-source 40 --max-tables-per-source 10 \
    --max-pos-rows-per-table 200 --out-dir data/structure_region_pilot
```

| Metric | Value |
|--------|------:|
| Sources spanned | 5 (pmc, openstax, nces_digest, cfr, federal_register) |
| Tables in | 36 |
| PDFs rendered OK | **36 / 36** (0 render failures) |
| Raw blocks extracted | 13,062 |
| Rows emitted (capped) | **3,467** (pos 2,399 / neg 1,068) |
| Positive rows capped | 9,595 (giant tables; see below) |
| Schema match vs real rows | **YES — byte-identical key/dtype shape** |
| Loads through `train_structure.load_split` | **YES** (3,467 rows, all 8 fields) |

Per-source (rows / pos / neg): pmc 1183 / 1089 / 94 · openstax 118 / 58 /
60 · nces_digest 1422 / 549 / 873 · cfr 22 / 16 / 6 · federal_register
722 / 687 / 35.

**Finding — giant tables.** One Federal Register table (a county-by-county
fee schedule) renders ~9,500 cells; uncapped it produced 9,997 rows and
swamped the positive distribution. Added `--max-pos-rows-per-table`
(default 200) to cap positives per table; negatives always kept. CFR is
table-sparse on disk (~3/60 pairs) — expected from Plans/06 §0.1; the
pilot still got one CFR table.

No human-decision blocker hit: the feature schema reproduces exactly from
rendered PDFs (verified through the real trainer loader).

## 4. Acceptance gates (for the bulk build, before retrain)

| Gate | Target |
|------|--------|
| PDF render success | ≥ 98% of attempted tables |
| Schema match | 100% rows pass `_validate_row`; `load_split` accepts file |
| Source spread | all 5 sources present; no single source > ~60% of rows |
| Positive/negative balance | per-table positive cap enforced; global pos ≤ ~70% |
| No layout/pseudo tables | 2×2+ and ≥50%-non-empty gate applied |
| License | sources are Plans/06-vetted (PMC OA, OpenStax CC-BY, NCES/CFR/FedReg US-gov PD). **No Wikipedia.** |

## 5. FOLLOW-UP — bulk build + GPU retrain (USER RUNS LATER)

**Serial-GPU dependency:** the only GPU (RTX 3070 8GB) is occupied by
`train_table_specialist.py … final_v5` (PID 905). Per
`feedback_qwen_build_serial`, do NOT start this retrain until that job
finishes (`gpu-status` shows the GPU clear). Steps 1–2 are CPU-only and
may run now; step 3 is the GPU step.

**Step 1 — bulk render (CPU).** Lift the pilot caps:
```
PYTHONPATH=$PWD python scripts/synthesize_region_features_from_html.py \
    --limit-per-source 100000 --max-tables-per-source 100000 \
    --max-pos-rows-per-table 200 \
    --out-dir data/structure_region_full
```

**Step 2 — merge into a retrain dataset (CPU). [DONE 2026-05-27]** A naive
`cat | shuf` was REJECTED: the raw 561K synth pool is 92% PMC / 86%
positive and 4.5× the base — blind-merging would invert the dataset and
bury the rare roles. Instead `scripts/merge_structure_dataset_v2.py`
(seeded, pair-aware) does the **Balanced ~64K** variant chosen by the
user:
  * PMC capped to 20K rows **by whole pairs** (276 pairs kept); all
    non-PMC sources (nces/openstax/cfr/fedreg) kept in full → 63,849
    synth rows.
  * Per-source 80/10/10 PAIR-aware split (whole `pair` → one split, no
    table leaks the boundary), then merged into the base splits.
Result (validated through `train_structure.load_split`):
  * train 149,427 (synth 50,631 = 33.9%) · val 18,118 · test 19,802
  * `table_region` positive 10.4% → **~32%** (all splits)
  * top source wikipedia ~30% (gate: <60% ✓); all 5 OOD sources present
    in the test split (source-stratified eval ready)
  * **rare roles 3/4/5 preserved unchanged** (synth is paragraph/heading
    only) — absolute counts identical to base; watch role-macro at
    retrain since role-0 share rose 54%→69%.

**Step 3 — retrain Structure (GPU; AFTER final_v5).**
```
python train_structure.py \
    --dataset-dir data/structure_dataset_v2 \
    --output-dir  models/council/structure_v2 \
    --epochs 6 --batch-size 16 --lora-r 16 --lora-alpha 32 \
    --weight-cap 30.0 --snapshot-policy weighted_macro --patience 3
```
Output: `models/council/structure_v2/final/{adapter_model.safetensors,
heads.pt, tokenizer/, summary.json}`. To promote, point
`dart_semantic/council/structure.py:DEFAULT_ADAPTER_DIR` (or the registry
path) at the new dir after validating metrics.

**What gates the retrain ship:**
- `summary.json` **`test_table_region_pos_f1` ≥ 0.924** (current baseline)
  — must not regress on the table head, and ideally gains on a
  source-stratified eval that includes pmc/nces/cfr/fedreg.
- `test_role_macro_f1` holds ≈ 0.866 (no collapse of the other heads — the
  synthesized rows are paragraph/heading only, so role balance must be
  watched; cap synthesized share if role-macro drops).
- All five heads still emit (heads.pt carries all 5 head state-dicts).

## Do-NOT list
- No GPU work while `final_v5` (PID 905) runs — serial GPU only.
- No bulk render in the pilot step (done; pilot was 36 tables).
- No Wikipedia (CC-BY-SA off-limits) — only the Plans/06-vetted sources.
- No silent fallbacks — schema mismatch raises `StructureSchemaMismatch`,
  normalizer failure raises `TableExtractionError`.
- Synthesized rows are `is_image_block=0`, `list_nesting=0`,
  `structural_role∈{paragraph,heading}` only — they harden `table_region`,
  they are NOT a balanced multi-head corpus. Use as augmentation, not a
  full retrain set.
```
