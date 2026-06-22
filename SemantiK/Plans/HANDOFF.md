# DART/Semantic — Handoff Note

> **SUPERSEDED (2026-06-09).** This is a 2026-05-04 snapshot. The "Next:
> Phase 3c" plan below is done — `train_semantic.py` and the rest of the
> council/Qwen/assembler/theta stack have all shipped (the v1 runtime
> landed; see `git log`). Treat this file as a historical record, not a
> task list. Current canonical docs: `architecture.md` and
> `Plans/10_completion_punchlist.md`.

**As of:** 2026-05-04
**Branch:** `main`
**Last commit:** `phase 3b: source blockquote + code_block — wikipedia v2 + synthetic`

---

## TL;DR

Phase 3b is **shipped**. BERT-Structure is the v1 replacement landmark — held-out test on the 4-head model: structural_role macro-F1 0.870, is_heading pos-F1 0.875, table_region pos-F1 0.927, list_nesting MAE 0.07. Adapter at `models/council/structure/final/` (v1 archived at `final_v1/`). Shadow-run on 4 held-out arXiv PDFs vs DistilBERT classifier_v5: 67.7% agreement on overlap classes, with v2 conservatively diverging on edge cases and adding three signals v1 didn't have (is_heading, table_region, list_nesting).

The architecture evolved during this session — the original 4-head plan (structural_role, is_heading, heading_level, list_nesting) became 4 different heads:

  1. **`structural_role`** trimmed from 21 → 6 active classes (paragraph, heading, list_item, form_label, blockquote, code_block) — non-authoritative recommendation; downstream specialists override
  2. **`is_heading`** binary — gates a future HeadingSpecialist for h1..h6 emission
  3. **`table_region`** binary (NEW) — gates the existing TableSpecialist plan; replaces what would have been a `Role.TABLE` class
  4. **`list_nesting`** 4-class (depth 0/1/2/3) — unchanged

`heading_level` and `figure_caption` were both dropped from Structure and pushed to specialists (HeadingSpecialist, ImageSpecialist Phase 3f — newly added to plan).

The next phase is **Phase 3c — BERT-Semantic**: doc-role (title/author/abstract/body/citation/footer/legal/metadata) + boilerplate flag, **cascaded from Structure** (teacher-forced on Structure labels with scheduled-sampling at end).

---

## What's done

### Design (canonical refs)

- `architecture.md` — 13-stage pipeline, locked architectural choices in §11. Read first.
- `Plans/01_implementation_plan.md` — phased build plan (Phases 0–11). Updated 2026-05-04 with Phase 3f ImageSpecialist spec.
- `Plans/03_theta_investigation.md` — theta evaluator (Stage 12) deep dive.
- `Plans/04_assembler_layer_investigation.md` — assembler / gap-fill / doc-gate / exits (Stages 9–13).
- `Plans/02_cleanup_punchlist.md` — pre-Phase-0 archive sweep audit.
- `docs/ontology.md` — WCAG / standards mapping.

### Locked architectural choices (from `architecture.md` §11)

- **7 BERTs (was 8 pre-2026-05-05, then 7 after TableDetector retirement)**: MergeOrSplit, Structure, Semantic, TableSpecialist, ImageSpecialist (added 2026-05-04), MathDetector (deferred), MathSpecialist. ~~TableDetector~~ retired 2026-05-05 — Structure's `table_region` binary head + pdfplumber `TableCandidate` aggregation does the gating job (P=R=F1=1.000 region-level on 170 arXiv held-out regions; see `scripts/eval_table_region_at_region_level.py`).
- **Council base encoder = ModernBERT-base.**
- **DeBERTa-v3-small** for cross-encoders + theta semantic-preservation scorer.
- **4 Qwen LoRA adapters**: prose, table, math, gap-fill. Strict routing, batched.
- **Qwen specialist runtime = llama.cpp.** Council BERTs stay on transformers + PEFT.
- **Page template owns body-scope `<header>`/`<footer>`.**
- **No scalar confidence number in v1.** `theta_score` is the only quality number on the wire.
- **Math wins matrix arbitration.**
- **No human escalation.** Four exit actions.
- **Gap-fill outputs go through the per-region hard gate.**
- **Merge-back is one-shot.**
- **Doc-level soft reranker v1 scope:** only fast-lane vs. offline-lane multi-assembly.
- **Theta is internal-diagnostic only.**
- **Theta retry policy capped at one offline retry.**

### Phases shipped

| Phase | Status | Key artifact |
|---|---|---|
| 0 — Scaffolding | ✓ | `dart_semantic/{council,qwen_specialists,gates,assembler,theta}/`, `pipeline_v2.py`, 36 unit tests |
| 1 — Layout/geometry | ✓ | `dart_semantic/region_detection.py` typed candidates; `featurize_with_regions`; math recall 100%, table recall 80% |
| 2 — Council infra + MathSpecialist | ✓ | `SharedBackbone` + `LoRAAdapter` + `MultiHeadModel` + `runner`; first-pass MathSpecialist macro-F1 0.766 |
| 2.5 — Augment ar5iv + retrain Math | ✓ | Cache 331 → 964 papers; rebuild 95K rows; **MathSpecialist macro-F1 0.860** (target 0.80 ✓) |
| 3a — MergeOrSplit (3-head + layout) | ✓ | `models/council/merge_or_split/final/`; same=0.778, join=0.815, hyphen=0.871 |
| 3b — Structure (4-head + layout) | ✓ | See "Phase 3b final results" below |

---

## Phase 3b final results (v2, 4-head + layout MLP)

### Adapter

`models/council/structure/final/`
- `adapter_model.safetensors` — LoRA weights on ModernBERT-base
- `heads.pt` — 4 heads (structural_role 6-class, is_heading 2, table_region 2, list_nesting 4) + LayerNorm + 64-dim layout MLP
- `summary.json` — full epoch log + final test metrics
- `tokenizer/` — saved alongside the adapter
- v1 archived at `models/council/structure/final_v1/`

### Held-out test (best epoch=6 of 6, snapshot score 0.887)

| Head | Metric | Value |
|---|---|---|
| `structural_role` (6-class) | macro-F1 | **0.870** |
| `is_heading` (binary) | pos-F1 | **0.875** |
| `table_region` (binary) | pos-F1 | **0.927** |
| `list_nesting` (4-class ordinal) | MAE | **0.040** → 0.070 (v2 vs v1; depth=2/3 still hard) |

Wall-clock 56.5 min, peak VRAM 2.74 GB on RTX 3070.

### Per-class structural_role detail

| Class | F1 | n (test) | Notes |
|---|---|---|---|
| paragraph | 0.896 | 6671 | dominant class, monotonic from v1 |
| heading | 0.881 | 2640 | high recall (0.95), modest precision (0.81) |
| list_item | 0.871 | 2531 | doubled from v1 (sourcing) |
| form_label | 0.901 | 117 | starved but stable |
| **blockquote** | **0.704** | 168 | up from v1's 0.323 (12 → 168 samples) |
| **code_block** | **0.966** | 227 | up from v1's 0.615 (7 → 227 samples) |

### Architectural choices ratified during Phase 3b

The original 4-head design (structural_role 21-class, is_heading, heading_level, list_nesting) iterated significantly:

1. **`heading_level` dropped → HeadingSpecialist.** Pushed to a future specialist gated on is_heading=1. Mirrors the Structure → TableSpecialist gating pattern. Lets the specialist train on a much narrower distribution (positives only).

2. **`table_region` promoted from class to binary head.** Originally `Role.TABLE` was one of the structural_role classes. Realized this artificially couples "this is a table cell" with "this is paragraph-shaped content" — a cell CAN be both. Splitting them lets the model learn each signal independently. table_region=1 hands the span to the TableSpecialist (Phase 3e) for cell-level role + scope; this head only DETECTS table membership.

3. **`figure_caption` dropped → ImageSpecialist (Phase 3f, new).** v1 trained-with-figure_caption had 0.42 precision / 0.86 recall on this class — over-fires on small italic captions that shape-resemble paragraphs. Same fix as table_region: detect "is_image_block" via a future Structure binary head, let an ImageSpecialist parse caption text on confirmed positives.

4. **`structural_role` trimmed 21 → 6 active classes.** Dropped 15 dead Role enum values that get zero training data (title, list, table_*, figure, figure_caption, form_field, reference, footnote, metadata, page_*). Page artifacts captured by the layout side-channel (top_5pct/bottom_5pct/is_artifact); cell-level table roles owned by TableSpecialist.

5. **Walker `_li_role_override`.** When a `<li>` directly wraps a `<blockquote>` or `<pre>`, relabel as that role. Wikiquote-style and technical-doc patterns were previously mislabeled list_item.

### Data sourcing (rare-class densification)

After v1 training revealed blockquote/code_block starvation (12 + 7 test samples → F1 0.32/0.61), added:
- **Wikipedia v2 seed expansion** (`scripts/seed_titles_v2.txt`) — 165 articles across CS topics, philosophy, literature, religion. wikipedia rows 23K → 56K (+143%).
- **Synthetic generator** (`scripts/pair_from_synthetic.py`) — 300 controlled-density public-domain pages (40 hardcoded quotes, 15 hardcoded code snippets across Python/SQL/shell/JS/Rust/C). 10,748 new training rows.

Yield: blockquote 120 → 1,675 (13.9×), code_block 62 → 2,264 (36.5×). Total dataset 79K → 123K (+55%).

Queued for future expansion if Phase 3c-onward shows we still need more rare-class accuracy:
- Wikiquote (#29) — same MediaWiki API as Wikipedia, structural quirk: uses `<li>` over `<blockquote>` so needs a preprocessing pass
- MDN web docs (#30) — high `<code>`/`<pre>` density, CC-BY-SA 2.5
- Python docs (#31) — Sphinx HTML, PSF license
- Gutenberg literary (#32) — filter for blockquote-rich semantic markup

### Shadow-run vs DistilBERT classifier_v5

`eval/results/structure_v2_shadow/summary.md`. 4 held-out arXiv PDFs (4 different subject areas — humanities, astro, math probability, cond-mat). 67.7% v1↔v2 agreement on overlap classes (paragraph/heading/list_item/blockquote/code_block/form_label).

Disagreement patterns:
- v2 fires more aggressively on heading (high recall) — 508 paragraph→heading flips on the largest paper
- Many disagreements are on PDF-extraction artifacts (single-character "blocks" `m`, `v`, `D`, `c`) — both models give arbitrary labels; right fix is upstream filtering
- v2's table_region head detected MORE table cells than v1's table_* family in every PDF, while still recovering 90%+ of v1's positives — likely catches borderless layout tables and equation arrays v1 missed

Ship-ready for council integration.

---

## Next: Phase 3c — BERT-Semantic

**Goal.** Multi-head: doc_role (title/author/abstract/body/citation/footer/legal/metadata) + boilerplate flag. **Cascaded from Structure** per architecture §7.2 — teacher-forced on Structure labels, scheduled-sampling at end. Validation targets: doc_role macro-F1 ≥ 0.75; boilerplate F1 ≥ 0.85.

### Open design questions to resolve before scaffolding

1. **Doc-role label vocabulary** — 8 classes proposed; do all 8 have enough training signal in our current corpus, or do we need to collapse some (e.g., legal/boilerplate could merge for v1)?

2. **Cascade input shape** — feed Structure top-k (k=3 ranked logit names + softmax confidences) as a serialized text prefix? Or as additional numeric features into the layout MLP? Phase 3a/3b layout-MLP pattern argues for the numeric path.

3. **Corpus mix** — current pair files: arXiv 51K rows / wikipedia 56K / synthetic 11K / gutenberg 2.6K / pdf_form 1.2K / openstax 829 / federal_register 289. Boilerplate/legal needs more govinfo-style content. Likely a sourcing pass before training.

4. **Scheduled-sampling schedule** — final 10% of epochs use predicted Structure top-k. Phase 3b ran 6 epochs, so the schedule is "epochs 1-5 teacher-forced, epoch 6 sampled". Confirm during scaffolding.

### Suggested execution order

1. Audit existing arXiv/wikipedia HTML for the 8 doc-role classes — quantify how many training rows each yields with simple heuristics (h1 = title, address = author, class="abstract" = abstract, p = body, ...).
2. Source more govinfo / IRS content if legal/boilerplate is starved.
3. Write `data/build_semantic_data.py` — walk pair files, run the trained Structure adapter to attach top-k cascade features per row, derive doc-role + boilerplate labels from HTML.
4. Write `train_semantic.py` mirroring `train_structure.py` skeleton (multi-head + layout MLP + scheduled-sampling teacher-forcing).
5. Write `dart_semantic/council/semantic.py` runtime.
6. Tests; train; shadow-run if a v1 doc-role classifier exists for comparison (none currently — Phase 3c is a new head, not a replacement).

### Time / risk

~3 weeks per the plan. Lower stakes than 3b (no v1 to replace), but the cascade pattern is novel — Phase 2/3a/3b all trained heads independently. Teacher-forcing + scheduled-sampling is a new training-loop concern.

---

## Phases after 3c (per `Plans/01_implementation_plan.md`)

3. ~~**Phase 3d — BERT-TableDetector**~~ — **RETIRED 2026-05-05**. Structure's `table_region` binary head + pdfplumber `TableCandidate` aggregation does the gating job (P=R=F1=1.000 region-level on 170 arXiv held-out regions). See `Plans/01_implementation_plan.md` Phase 3d notice.
4. **Phase 3e — BERT-TableSpecialist** (largest data ask; trains on Structure `table_region`-positive spans).
5. **Phase 3f — BERT-ImageSpecialist** (added 2026-05-04; figure_caption + caption_position + alt_candidate).
6. **Phase 4 — Cross-BERT reranker.**
7. **Phase 5 — Qwen prose specialist** (the llama.cpp pivot).
8. **Phase 6a/b/c — Qwen table / math / gap-fill specialists.**
9. **Phase 7 — Per-region hard gate.**
10. **Phase 8 — Per-region soft reranker.**
11. **Phase 9 — Deterministic assembler.**
12. **Phase 10 — Doc-level gates + reranker + exits.**
13. **Phase 11 — Theta evaluator.**

Each phase gates on measured lift on `eval_v7_family` per architecture §8.1.

---

## Hardware + runtime constraints

- RTX 3060 8 GB / RTX 3070 8 GB. ModernBERT-base + LoRA training comfortable at batch_size=16, max_length=192. Phase 3b peak VRAM 2.74 GB.
- Serial `build_qwen` discipline (parallel poisons CUDA on 8 GB) — applies to future Qwen specialist work.
- Wikipedia REST API rate-limits hard with workers > 1 — use workers=1 for any future re-fetch.
- No external LLMs at runtime. No human escalation. No API providers in v1.
- Gitignored: `data/ar5iv_html_cache/`, `data/math_specialist_dataset/`, `data/merge_or_split_dataset/`, `data/structure_dataset/`, `models/`, `_archive_*/`.

---

## Open questions / decisions surfaced but not yet resolved

- **Theta thresholds** (TAU_THETA_RETRY=0.70, TAU_THETA_SHIP=0.80, DELTA_THETA_IMPROVE=0.05) — placeholders; tune on held-out before any threshold ever ships (architecture §6.3).
- **Per-document-type theta thresholds** vs. global — currently global (DP-11.3).
- **Theta semantic-preservation cross-encoder** — DeBERTa-v3-small primary (DP-11.1).
- **llama.cpp loading strategy** — pre-merge GGUF per specialist (default) vs. base GGUF + hot LoRA swap. Decide on measured swap latency at Phase 5.
- **Offline-Qwen lane base model** — Qwen 3 4B + K=8 + temp 0.9 for v1. 7B Q4_K_M may fit; revisit at Phase 10.
- **Phase 3c doc-role label vocabulary** — 8 classes proposed; may need to collapse legal+boilerplate or split body further once corpus audit lands.

---

*If you're picking this up cold: read `architecture.md` and `Plans/01_implementation_plan.md` first. Then `git log --oneline -30` to see the path that got us here. Then start Phase 3c: `data/build_semantic_data.py` is the first concrete artifact to produce, but **first** audit the existing arXiv HTML for doc-role label density (Phase 3c "Open design questions" §1 above).*
