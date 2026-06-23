# SemantiK — Architecture (the 13-stage v2 cascade)

> SemantiK is Ed4All's **PDF → WCAG 2.2 AA accessible-HTML conversion
> engine** — a local-only pipeline built on the principle:
> **learned models are narrow candidate generators; deterministic code
> orchestrates, gates, and assembles.** The extraction path is built
> entirely on permissively-licensed PDF tooling (pypdfium2 + pdfplumber +
> pikepdf + Tesseract); no cloud LLM required at runtime (a hosted 70B endpoint
> is an opt-in seat, not a dependency); no human in the loop. SemantiK emits a
> stable source-provenance **wire contract** — the `data-dart-*` HTML block
> attributes and the `dart:{slug}#{block_id}` sourceId — so Ed4All consumers
> thread block-level provenance through the pipeline unchanged (see §12).
>
> This file is the cascade deep-dive. The subsystem guide (`CLAUDE.md`)
> links here; the wire contract + cross-venv bridge are §12–§14 below.

This document is the canonical reference for the structure of the pipeline.
For the WCAG / standards mapping that governs each output element, see
[`docs/ontology.md`](docs/ontology.md). For the historical eight-stage refactor
that this design supersedes, see [`docs/refactor_plan.md`](docs/refactor_plan.md).

---

## 1. Design principle

Every learned model in the pipeline is **scoped to a narrow decision
surface**. Composition, hierarchy enforcement, ARIA wiring, validation, and
final assembly are deterministic. This is what lets us claim WCAG 2.2 AA
conformance: the rules that produce conformance are visible, auditable code,
not weights.

Concretely:

- **BERTs classify.** Each BERT in the council owns one decision surface
  (structural role, semantic role, span-merge, table-detect, table-cells,
  math-detect, math-cells). Outputs are typed signals, not HTML.
- **Qwens generate.** Each Qwen adapter is a candidate generator for a
  narrow remediation task (prose, table, math, gap-fill). Multiple candidates
  per region; rerankers choose; validators gate.
- **Deterministic code orchestrates.** Layout extraction, candidate-graph
  construction, document assembly, hierarchy normalization, landmark wiring,
  hard gates, and exits are all rules.

Everything else flows from this principle.

---

## 2. End-to-end pipeline

```
PDF
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 1: Extraction (deterministic)                                 │
│  pikepdf + pypdfium2 + pdfplumber + Tesseract → unified per-page JSON│
└──────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 2: Layout / geometry (deterministic)                          │
│  Block features (size/weight/pos), column flow, reading-order        │
│  skeleton, table-region candidates, math-region candidates           │
└──────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 3: BERT council  (7 BERTs, strict routing, multi-head where   │
│                          it makes sense)                             │
│                                                                      │
│   table-candidates ──► Structure.table_region (binary head)          │
│                          ├─ confirmed → table track                  │
│                          └─ rejected  → demote to flat-text          │
│   (BERT-TableDetector retired 2026-05-05 — Structure's table_region  │
│    head + pdfplumber TableCandidate aggregation does the gating job; │
│    eval P=R=F1=1.000 region-level on 170 arXiv held-out regions)     │
│                                                                      │
│   math-candidates  ──► BERT-MathDetector                             │
│                          ├─ confirmed → math track                   │
│                          └─ rejected  → demote to flat-text          │
│                                                                      │
│   flat-text ──► BERT-MergeOrSplit                                    │
│             ──► BERT-Structure   (multi-head: role, h-level, list)   │
│             ──► BERT-Semantic    (uses Structure top-k as feature)   │
│                                                                      │
│   confirmed-tables ──► BERT-TableSpecialist                          │
│                       (uses Structure top-k as soft hint)            │
│                                                                      │
│   confirmed-math ──► BERT-MathSpecialist                             │
└──────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 4: Cross-BERT reranker                                        │
│  Combined typed signals + neighbor context → routing decision +      │
│  per-region structure call. Resolves disagreement (e.g. Structure    │
│  says heading@0.6, Semantic says abstract@0.7).                      │
└──────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 5: Structure-graph candidate generation (deterministic)       │
│  Typed signals → candidate region tree                               │
└──────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 6: Qwen region-specialist generation (3 LoRA adapters,        │
│           batched by adapter — never interleaved on 8GB VRAM)        │
│                                                                      │
│   Pass 1: load prose adapter   → K candidates per prose region       │
│   Pass 2: swap to table adapter → K candidates per table region      │
│   Pass 3: swap to math adapter  → K candidates per math region       │
│                                                                      │
│  (Gap-fill is the 4th adapter but does NOT fire here — it is         │
│   invoked from Stage 9 once the assembler has flagged gaps.)         │
└──────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 7: Per-region HARD gate (eliminating; no soft trade-offs)     │
│  axe-core pass · html5validator pass · text-preservation ≥ τ         │
│  · MathML validity (math regions only)                               │
│                                                                      │
│  Candidates that fail any check are dropped.                         │
└──────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 8: Per-region soft reranker (cross-encoder)                   │
│  Among gate-survivors, score:                                        │
│    heading hierarchy fit · table-semantics richness · diff-from-src  │
│    · neighbor consistency · ARIA restraint                           │
│  → top-1 candidate per region                                        │
└──────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 9: Deterministic document assembler                           │
│           (owns document-level orchestration)                        │
│                                                                      │
│  Phase 9a — first deterministic pass:                                │
│    · heading hierarchy: "promote-the-first, demote-forward,          │
│      never-skip" (force-claim H1 from Semantic.title once;           │
│      demote-to-fit, never promote)                                   │
│    · ARIA landmarks emitted by assembler: <main> always; <nav> on    │
│      TOC pattern; <aside class="legal|copyright"> for legal-or-      │
│      copyright; <address> for author                                 │
│    · body-scope <header> and <footer> are NOT emitted by assembler   │
│      — page template owns them (per docs/ontology.md §7). Semantic   │
│      `footer` doc-role drops to artifact (same as `metadata`).       │
│    · list-continuation: 4-clause merge (kind + marker + adjacency    │
│      + no-heading-interruption); figure interruptions tolerated      │
│    · reference resolution: 3-pass (build index → regex match →       │
│      resolve/leave/flag); flag GapSlot only on near-miss             │
│    · doc shell: doctype, <html lang> (langdetect on body),           │
│      <head><title>, skip-link to #main-content                       │
│    · reading-order DOM placement (from Stage 2 skeleton)             │
│    · gap detection: emit GapSlot(kind, context) for the 5 supported  │
│      kinds only; unsupported slots use deterministic fallbacks       │
│           ↓ flags gaps                                               │
│  Phase 9b — Qwen-GapFill pass (only if gaps flagged):                │
│    · load gap-fill adapter (4th and only invocation in pipeline)     │
│    · K candidates per flagged gap (K=4 fast, K=8 offline)            │
│    · slots: missing_title / citation_unresolved / author_block /     │
│             copyright_block / legal_disclaimer                       │
│    · each candidate passes through the per-region HARD gate          │
│      (Stage 7 checks); failures are dropped before scoring           │
│           ↓                                                          │
│  Phase 9c — deterministic merge back:                                │
│    · per-gap rule-based scoring (axe + html5 + kind-fit + length-    │
│      fit − diff-from-context); select top per gap                    │
│    · splice into document tree (per-kind splice semantics)           │
│    · ONE-SHOT re-run of heading hierarchy + reference resolution     │
│      if title or cross-refs changed; never iterate (cycle ruled out) │
│    · all-K-fail fallback: deterministic per-kind placeholder         │
└──────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 10: Document-level HARD gate                                  │
│  axe-core full doc · html5 validity · heading hierarchy validity     │
│  · <html lang> declared · <title> present · landmarks present        │
└──────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 11: Document-level soft reranker                              │
│                                                                      │
│  v1 scope: ONLY fast-lane vs. offline-lane multi-assembly.           │
│  Per-region top-2 fan-out and gap-fill K^M cross-product are         │
│  COMBINATORIAL TRAPS — deferred to v2.                               │
│                                                                      │
│  Rule-based scorer (DP-10.1 default, learned only on measurement):   │
│    composite = 0.30 · heading_tree_balance                           │
│              + 0.20 · landmark_coverage                              │
│              + 0.30 · ref_link_integrity                             │
│              + 0.20 · outline_cleanliness                            │
│  Tie-break: smaller diff-from-source by character count wins.        │
│  Emits per-axis scores for Stage 12 to consume.                      │
└──────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 12: ThetaEvaluator  (post-WCAG meaning-preservation score)    │
│                                                                      │
│  Runs ONLY on docs that pass Stage 10. Never overrides WCAG.         │
│  Composite over 8 dimensions (1 learned + 7 deterministic):          │
│    · semantic_preservation       (DeBERTa-v3-small cross-encoder,    │
│                                   per-section, length-weighted mean) │
│    · structural_coherence        (reuses Stage 11 axis)              │
│    · navigation_clarity          (reuses Stage 11 axis)              │
│    · context_continuity          (deterministic)                     │
│    · reference_integrity         (reuses Stage 11 axis)              │
│    · cognitive_load_reduction    (deterministic; emits risk enum)    │
│    · fragmentation_penalty       (deterministic)                     │
│    · hallucinated_structure_pen. (deterministic, gap-fill-aware)     │
│                                                                      │
│  Per-dimension floors trigger flags:                                 │
│    reference_integrity     < 0.80 → broken_refs_present              │
│    hallucinated_structure  < 0.85 → gap_fill_review_recommended      │
│    semantic_preservation   < 0.60 → meaning_preservation_low         │
│    cognitive_load_risk == "high"  → cognitive_load_high              │
│                                                                      │
│  Retry policy (capped at one offline retry):                         │
│    fast-lane theta < 0.75 AND not previously retried                 │
│      → rerun via offline-Qwen lane, re-evaluate, ship higher-theta   │
└──────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 13: Exit                                                      │
│    · ship-with-confidence  (theta ≥ 0.85, no floor breach)           │
│    · ship-with-flag        (theta 0.75–0.85, or floor breach)        │
│    · offline-Qwen lane     (gate or theta-low → one retry, same      │
│                             Qwen 3 4B + K=8 + temp 0.9)              │
│    · non-certified stamp   (gate failed both lanes → ship HTML +     │
│                             <meta dart-certification-status=         │
│                             "not-certified"> + visible <aside        │
│                             role="note"> at top of <main>)           │
│  No human escalation. theta is omitted on non-certified exit.        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. The BERT council

Seven BERTs total. **Multi-head where it makes sense** (one encoder, several
classification heads). **Specialist split (detector + cell-classifier)** for
tables and math, where the gating decision is binary and high-volume but the
specialist requires expensive cell-level labels.

| # | Model | Type | Heads | Input | Runs on |
|---|---|---|---|---|---|
| 1 | **BERT-MergeOrSplit** | multi-head | `same_logical_block` (binary) · `join_type` (space / newline_within_p / list_continuation) · `hyphen_repair` (binary) | text pair + 24-dim layout side-channel (font_size_a/b, font_size_delta, bold transition, y_gap_log1p, y_gap_lines, lhr, x0/x1_delta, width_ratio, x_overlap_frac, b_titlecase_frac, ...) + **deterministic features** (column_id, is_artifact) → LayerNorm → 64-dim MLP → concatenated with BERT pooled | flat-text + demoted regions |
| 2 | **BERT-Structure** | multi-head | structural role · `is_heading` (binary) · heading-level (h1..h6, conditional on `is_heading=1`) · list-nesting | text + layout features | merged flat-text blocks |
| 3 | **BERT-Semantic** | multi-head | doc-role (title/author/abstract/body/citation/footer/legal/metadata) · boilerplate flag | text + neighbor context + Structure top-k | merged flat-text blocks |
| 4 | **BERT-TableDetector** *(RETIRED 2026-05-05 — folded into Structure as the `table_region` binary head; pdfplumber TableCandidate aggregation supplies the region grouping. Eval evidence: `scripts/eval_table_region_at_region_level.py`, P=R=F1=1.000 region-level on 170 arXiv held-out regions.)* | — | — | — | — |
| 5 | **BERT-TableSpecialist** | multi-head | cell role + scope (header-col / header-row / both / data / span) · caption-association | layout + cell text + 2D neighbor cells + Structure top-k (soft hint) | detector-confirmed tables |
| 6 | **BERT-MathDetector** | binary | is this region actually math? (vs. italicized prose / aligned symbols) | layout + glyph features | math-candidate regions |
| 7 | **BERT-MathSpecialist** | multi-head | math-type (inline / display / numbered / multiline / matrix) · equation-number-association | text + glyph features (sub/sup, math-symbol density, fraction bars) | detector-confirmed math |

### 3.1 Feature-flow DAG (which BERT outputs feed which)

Pass distributions (top-k + confidences), not argmax.

```
MergeOrSplit ─┬─► Structure ─┬─► Semantic
              │              └─► TableSpecialist (soft hint)
              ├─► Structure.table_region ──► TableSpecialist
              │   (was BERT-TableDetector; retired 2026-05-05)
              └─► MathDetector  ──► MathSpecialist

{Structure, Semantic, TableSpecialist, MathSpecialist} ──► cross-BERT reranker
```

**Justified edges only.** MergeOrSplit redefines what "a block" is (structural,
not feature). Structure → Semantic disambiguates doc-role candidates.
Structure → TableSpecialist is a soft hint that heading-styled rows correlate
with `<th>`. Structure → MathSpecialist intentionally **omitted** — math is
its own subworld.

### 3.2 Arbitration rules

- **Math wins matrix conflicts.** When BERT-Table and BERT-Math disagree on a
  region (e.g. mathematical matrix expressions), output is MathML `<mtable>`,
  not HTML `<table>`. Hard rule in the cross-BERT reranker.
- **Detectors gate specialists.** A specialist never runs on a region the
  detector rejected. Rejected detector outputs demote the region into the
  flat-text path.
- **Ambiguous routing → prose.** When the cross-BERT reranker confidence
  falls below threshold, default to the prose specialist with a low-confidence
  flag the soft reranker downstream weighs.

### 3.3 Stage 5 — structure-graph candidate generation

The deterministic mapping from typed BERT signals to a flat list of
typed `Region` candidates. Implementation:
[`dart_semantic/structure_graph.py`](dart_semantic/structure_graph.py).

**Eleven region kinds** (Stage 6 specialists consume each):

- `paragraph` — a run of prose FeatureBlocks merged by MergeOrSplit.
- `heading` — single FB classified `is_heading` by Structure.
- `list` — consecutive `list_item`-role FBs grouped together.
- `definition_list` — reserved for a future detector (Stage 5 v1 emits none).
- `table` — passthrough of a Stage-4-confirmed `TableCandidate`, with `cell_grid` / `header_row_indices` / `bordered`.
- `math` — passthrough of a Stage-4-confirmed `MathCandidate`, with `src_text` and glyph-density features.
- `code_block` / `blockquote` — runs of the matching `structural_role` label.
- `figure` — a synthetic image FeatureBlock routed to the figure track (the
  default `SEMANTIK_DETECT_FIGURES` path), OR a single FB demoted by Stage 4's
  legacy `image_block_demoted` flag.
- `form` — runs of FBs with `in_widget=True` (pikepdf AcroForm overlap).
- `metadata_drop` — single FB whose Semantic `doc_role` is `footer` or `metadata` at high confidence; Stage 9 surfaces these as artifacts.

**Nine deterministic passes** (in order, sharing a `claimed: set[int]`):

1. Claim every `member_block_indices` of each `RoutingDecision` whose route is `table` or `math` so prose passes skip them.
1b. Drop deterministic page-furniture (running headers/footers).
2. Emit a heading region per FB whose `Structure.is_heading` top-1 is `heading` ≥ threshold.
3. Emit one `Region(kind="table"|"math")` per confirmed Stage-4 decision (overlapping math candidates dedup by FB ownership). Demoted candidates release their FBs back to prose.
3a. **Figure formation from `ImageCandidate`s (load-bearing).** An
    `ImageCandidate` routed `figure` by the deterministic arbiter forms ONE
    `figure` Region per candidate, claiming its synthetic image FB (the
    `is_image_block` head is a secondary confirmation only); a caption FB below
    is recorded for the alt path. Only runs on the `SEMANTIK_DETECT_FIGURES`
    path.
3b. Legacy `figure` formation for any FB Stage 4 flagged `image_block_demoted`.
4. Group maximal runs of unclaimed FBs whose `structural_role` top-1 is `list_item` into one `list` region.
5. Group maximal runs of unclaimed paragraph-role FBs into `paragraph` regions, splitting where MergeOrSplit's `same_logical_block` head emits `not_same` ≥ threshold.
6. Code/blockquote runs, single-FB metadata drops, form runs, and a single-FB-paragraph fallback for any FB still unclaimed.

**Figure detection path (`SEMANTIK_DETECT_FIGURES`, off by default).** When on,
Stage 1 enumerates PDF IMAGE page-objects via pypdfium2
(`_pypdfium2_page_images`, Y-flipped to top-left), filters spacer/rule images,
and synthesizes one image FeatureBlock per surviving candidate **interleaved
into the FB stream in reading order**. Each `ImageCandidate` is routed
DETERMINISTICALLY to the figure track (Pass-3a above). Stage 5c then renders the
figure Region bbox to a PNG and writes a deterministic sidecar
`{pdf_stem}_figures/fig-{first_fb_index}.png` next to the source PDF (per-region
fail-soft), stamping the relative `image_src` (`./{stem}_figures/fig-N.png`)
onto the region payload so the emitters fill the previously-empty `<img src="">`.
The flag is part of the extract disk-cache key, so a flip invalidates the cache.

**Boundary clarification.** Stage 5 produces a flat candidate region
list. Heading hierarchy normalization, list continuation, ARIA wiring,
reading-order placement, and HTML emission are all Stage 9
(assembler) responsibilities (§3.9 / lines 130-168). Stage 6 expands
each Region into HTML candidates; Stage 9 stitches the document.

**RESOLVED (2026-06-08) — Structure.is_heading over-fire.** The 2026-05-06
diagnostic showed the head over-firing 3.6× (619 emitted vs 171 reference
headings on the worst fixture). Fixed in three stages
(`Plans/11_heading_overdetection.md`): (1) a `_plausible_heading()` guard at
the Pass-2 promotion in `structure_graph.py` (drops empty/garbage/over-long
blocks; rejects fall through to prose); (2) post-hoc temperature scaling
T=1.6553 baked into `structure/final/heads.pt` plus `is_heading_threshold`
raised 0.7 → 0.8 (calibrated max-F1 P=0.926/R=0.902); (3) the extraction
root cause — pdfplumber's default 3pt `x_tolerance` glued LaTeX inter-word
gaps — fixed with font-scaled `x_tolerance_ratio=0.15` in
`extract_shared.py`. Validated on 1807.02622: 234 → 11 plausible headings,
document `heading_tree` gate failed → PASSED. Note `data/extract_cache/` is
mtime-keyed, so the extraction fix only applies to fresh extractions; the
R10 post-fix eval ran with a cleared cache. Historical measurement:
`data/eval_reports/stage5_heading_rate.json`.

---

## 4. The Qwen specialists

Four LoRA adapters. **Strict routing — one specialist per region, no fan-out.**
K candidates come from sampling diversity (temperature / top-p) within the
chosen specialist, not from running multiple specialists.

| # | Adapter | Scope | Output dialect |
|---|---|---|---|
| 1 | **prose** | flat-text remediation: paragraphs, headings, lists, blockquotes, code | HTML5 |
| 2 | **table** | confirmed table regions | HTML5 tables (`<thead>`/`<tbody>`/`<th scope>`/`<caption>`) |
| 3 | **math** | confirmed math regions | MathML 4.0 (with `alttext`) |
| 4 | **gap-fill** | narrow document-level slots: missing-title inference, footnote / citation cross-reference resolution, author / copyright / legal block remediation | HTML5 fragment per slot |

**Gap-fill scope is deliberately narrow.** It is **not** a generative document
assembler. It does **not** handle image alt-text (that is owned by the
Stage-6b figure captioner — SmolVLM2 in `figure_captioner.py`, emitted via
`assembler/fallbacks.fallback_figure` as `alt` + `aria-describedby`). It is
invoked by the deterministic assembler when a specific known-ambiguous slot
is flagged:

- No H1 in source → infer document title from first heading or doc metadata.
- Cross-reference like "Section 3.2" or "[12]" → resolve link target by text
  matching against the assembled document.
- Author byline / copyright notice / legal disclaimer detected by
  BERT-Semantic but not adequately remediated by the prose adapter (these
  blocks have specific structural conventions: `<address>`, `<footer>`,
  specific landmark wiring).

The prose adapter remediates body content. The gap-fill adapter handles the
narrow generative tasks the prose adapter is the wrong objective for.

### 4.1 Inference runtime — llama.cpp (not transformers, not Ollama)

Qwen specialists run via **llama.cpp** (`llama-cpp-python` bindings or the
`llama-server` subprocess), not via HuggingFace transformers + bitsandbytes,
and not via Ollama. Reasons:

- Better quantization formats (Q4_K_M / Q5_K_M) than transformers' bnb-NF4
  — meaningful VRAM headroom on 8 GB with Chromium/axe-core concurrent.
- Lower per-token latency for batched generation.
- Mature offline-local story; no Ollama daemon dependency.

**Training stays on transformers + PEFT.** LoRA adapters are produced by
HF/PEFT (`training/train_reasoner.py` recipe is reusable). Each trained adapter is
then exported to a llama.cpp-loadable form for inference. Two viable
implementation paths — choose at Phase 5 measurement, not now:

- **Pre-merge:** merge the LoRA into the base, export to one GGUF per
  specialist (4 GGUFs at ~2.5 GB each). Disk-heavy; inference is dead
  simple — load, generate, swap.
- **Hot LoRA swap:** keep one base GGUF resident, attach/detach LoRA
  adapter files via `--lora`. Disk-light; slightly more swap orchestration.

**Council BERTs stay on transformers + PEFT.** llama.cpp is decoder-only;
DeBERTa-v3-base + classification heads is encoder + heads, which llama.cpp
does not target. The two runtimes coexist: transformers for the council,
llama.cpp for the specialists.

### 4.2 VRAM discipline (8 GB constraint)

Adapter swapping is sequential, not interleaved. **Batch by adapter**, with
the four passes split across two pipeline stages:

```
Stage 6 — region-specialist passes (always run):
  Pass 1: load prose adapter   → process every prose region in the doc
  Pass 2: swap to table adapter → process every table region
  Pass 3: swap to math adapter  → process every math region

Stage 9 — gap-fill pass (only runs if assembler flagged gaps):
  Pass 4: swap to gap-fill      → process every flagged gap
```

Up to four adapter loads per document, not four per region. Adapters stay
resident through their full batch. The split exists because gap-fill is
*driven by the assembler's gap-detection output*, which only exists after the
per-region pipeline has finished. A document with no flagged gaps skips Pass 4
entirely.

This is consistent with the existing serial `build_qwen` constraint —
parallel adapter contexts poison CUDA on 8 GB.

### 4.3 Why gap-fill is invoked from Stage 9, not Stage 6

The principle is locked: **Qwens are narrow candidate generators;
deterministic code orchestrates, gates, and assembles.** Gap-fill cannot run
during Stage 6 because the inputs gap-fill needs (which slots are missing,
what surrounding context the assembled document provides for a title or
cross-reference) do not exist until the assembler has begun work in Stage 9.
Running gap-fill on raw region outputs would require the gap-fill adapter to
re-implement the assembler's job, which violates the principle and burns the
narrow-scope justification that earns the adapter its slot in the first
place.

---

## 5. Two-tier validation gate

Validation runs at **two tiers**, both with the same eliminating-vs-soft
discipline.

**Per-region hard gate (Stage 7).** Eliminating checks. A candidate that fails
any of these is dropped; it never reaches the soft reranker.

- axe-core pass on the region's HTML
- html5validator pass
- text preservation ≥ τ (against the source block's plain text)
- MathML validity (math regions only)

**Per-region soft reranker (Stage 8).** Among gate-survivors, learned
cross-encoder scores fit-quality:

- heading hierarchy fit (does h-level match document context?)
- table-semantics richness (`<th scope>`, `<caption>`, `<thead>` presence)
- diff-from-source size
- neighbor-block consistency
- ARIA restraint (penalize redundant roles per ARIA-in-HTML)

**Document-level hard gate (Stage 10).** After deterministic assembly:

- axe-core pass on full document
- html5 validity
- heading hierarchy validity (no skipped levels per `emit_html.py` policy)
- `<html lang>` declared
- `<title>` non-empty
- `<main>` landmark present

**Whole-document heading-contiguity normalization (Stage 9, this session
`3a71e57`).** Stage 9a's heading normalization only re-levels regions whose
`Region.kind == "heading"`, so an `<hN>` a Stage-6 specialist (e.g. the hosted
70B prose seat) emits *inside* a non-heading region's body bypasses it and
reaches the Stage-10 `heading_tree` gate verbatim (a prose-embedded `<h6>`
producing an `h2 → h6` skip → `wcag_status = "failed"` → the fast lane is
discarded for the offline lane). `assembler/heading_contiguity.py::normalize_
document_heading_levels`, wired into `assemble_document` after passes 9a/9c,
closes that gap: a deterministic, document-order pass re-levels EVERY `<hN>`
(structural + specialist-EMBEDDED + gap-filled) under the same
promote-first/demote-forward/never-skip rules so the Stage-10 gate sees a
contiguous hierarchy. Heading text/ids/attrs are preserved byte-for-byte (only
the level digit changes); idempotent (a contiguous doc returns byte-identical).

**Document-level soft reranker (Stage 11).** Rule-based composite (DP-10.1
default). **v1 scope is restricted to fast-lane vs. offline-lane** — only one
fast-lane assembly and at most one offline-lane assembly are ever scored
together. Per-region top-2 fan-out and gap-fill K^M cross-product are
combinatorial traps and are deferred to v2.

Composite formula:
`final = 0.30 · heading_tree_balance + 0.20 · landmark_coverage + 0.30 · ref_link_integrity + 0.20 · outline_cleanliness`

Tie-break: smaller diff-from-source by character count wins. Stage 11 also
emits its **per-axis scores** (heading_tree_balance, landmark_coverage,
ref_link_integrity, outline_cleanliness) so Stage 12 ThetaEvaluator can
reuse them without recomputing.

### 5.1 Why hard ≠ soft

Mixing eliminating signals with fit signals in one reranker lets the model
learn to trade them off — "this candidate has a small axe violation but
otherwise great." That is the wrong direction on the WCAG axis we are
competing on. Axe violations should eliminate, not penalize.

---

## 6. ThetaEvaluator — post-WCAG meaning-preservation score (Stage 12)

Theta is a **post-validation document-intelligence score**, not a compliance
score. It runs **after** the doc-level hard gate (Stage 10) and the
doc-level soft reranker (Stage 11) have already produced a chosen assembled
document. Theta evaluates **how well the remediated HTML preserves and
coordinates meaning** — across structure, navigation, context, and reader
burden — and emits a per-dimension report.

**Locked boundaries.**

- Theta **MUST NOT** override a failed WCAG hard gate. WCAG conformance
  remains the eliminating axis.
- Theta **MAY**: lower confidence, trigger a single offline-Qwen retry, or
  attach a `meaning-preservation review recommended` flag.
- Theta is a **developer-facing and consumer-facing internal diagnostic**,
  not a public marketing metric. The conformance statement template in
  [`docs/ontology.md`](docs/ontology.md) §6 must NOT include theta.
- Theta is **omitted** on the non-certified-stamp exit
  (`theta_score: null, wcag_status: "failed"`).

### 6.1 Composite shape (1 learned + 7 deterministic)

| # | Dimension | Mode | Source |
|---|---|---|---|
| 1 | semantic preservation | **learned** | DeBERTa-v3-small cross-encoder, section-level, length-weighted mean |
| 2 | structural coherence | deterministic | reuses Stage 11 `heading_tree_balance` |
| 3 | navigation clarity | deterministic | landmark coverage × heading-id density × ToC-resolution |
| 4 | context continuity | deterministic | section preservation rate + intra-section ref resolution rate |
| 5 | reference integrity | deterministic | reuses Stage 11 `ref_link_integrity` |
| 6 | cognitive-load reduction | deterministic | composite of paragraph-length distribution, heading density, list coverage; emits `risk` enum {low, medium, high} |
| 7 | fragmentation / ambiguity penalty | deterministic | inverse of source-paragraph-split rate + source-list-item-break rate |
| 8 | hallucinated-structure penalty | deterministic, **gap-fill-aware** | every gap-fill-emitted token must have a substring/paraphrase anchor in source |

Composite weights, exit taus, and floors are **calibrated** (Plan 12 A3,
2026-06-11): fitted on a 40-doc × 520-variant synthetic perturbation set
(`scripts/calibrate_theta.py`; report
`data/eval_reports/theta_calibration_v1.json`, fitted AUC 0.849 vs
uniform 0.834) and locked in `theta/config.yaml` (`theta-config-2.0`)
with full provenance. The numbers below are the calibrated values; the
config file is the source of truth — re-run the harness with
`--write-config` rather than hand-editing. Known caveat (recorded in the
report): synthetic clean fixtures sit above the real-pipeline theta
distribution, so the taus are clamped; re-calibrate against the C1
corpus distribution once the real-runtime eval at scale lands.

### 6.2 Per-dimension floors

Per-dimension floors trigger flags independently of the composite score:

| Dimension | Floor | Flag |
|---|---|---|
| `reference_integrity` | 0.80 | `broken_refs_present` (with broken-ref list) |
| `hallucinated_structure_penalty` | 0.85 | `gap_fill_review_recommended` (with span list) |
| `semantic_preservation` | 0.60 | `meaning_preservation_low` (and triggers retry per §6.3) |
| `cognitive_load_risk == "high"` | n/a | `cognitive_load_high` |

### 6.3 Retry policy (capped at one offline retry)

```
if exit_lane == "fast" and theta_score < TAU_THETA_RETRY (calibrated 0.75):
    if not previously_retried:
        rerun document through offline-Qwen lane
        re-evaluate theta on offline output
        if offline_theta > fast_theta + DELTA_THETA_IMPROVE (default 0.05):
            ship offline output (action = ship_with_confidence)
        else:
            ship the higher-theta output (action = ship_with_flag)
            flag = "meaning-preservation review recommended"
```

A theta retry is **one** retry only; the second pass cannot loop back. This
caps cost and matches the gate-failure retry policy.

### 6.4 Decoupling failures (false negatives are real)

Some valid WCAG remediations *legitimately* lower theta:

- Flattening a sidebar into linear flow is correct WCAG (per
  [`docs/ontology.md`](docs/ontology.md) §2.3) but raises fragmentation_penalty.
- Splitting a multi-column layout-merged paragraph into two `<p>` elements
  raises fragmentation_penalty.
- Emitting native MathML for an image-of-equation raises diff-from-source.

**Theta must not penalize correctness.** Concretely: fragmentation alone
never triggers retry; per-dimension floors are set above the noise floor.
Specific guards live in `theta/config.yaml`.

### 6.5 Output schema

```json
{
  "schema_version": "theta/1.0",
  "wcag_status": "passed",
  "lane": "fast",
  "theta_score": 0.87,
  "theta_version": "theta-config-1.0",
  "dimensions": { "...": "per-dimension scores with breakdowns" },
  "flags": ["meaning_preservation_borderline"],
  "action": "ship_with_flag"
}
```

Full schema in [`Plans/03_theta_investigation.md`](Plans/03_theta_investigation.md) §5.

### 6.6 v1 implementation status (Stage 12)

The Stage 12 evaluator (`dart_semantic/theta/evaluator.py`) implements
all 8 dimensions. Status as of 2026-06-11 — **no stubbed dimension
remains**:

* **`semantic_preservation` is LIVE (v8, 2026-06-08).** The full-FT
  DeBERTa-v3-small cross-encoder (cls pooling, BCE) shipped and loads
  without `DART_ALLOW_THETA_STUB`; it replaced the mode-collapsed v1.
  The `stub_v1` 0.7 fallback remains in code only as the documented
  loud-fail path when the model directory is absent.
* **Composite weights are CALIBRATED (Plan 12 A3, 2026-06-11).** The
  evaluator loads them from `theta/config.yaml` (`theta-config-2.0`,
  fitted on 520 perturbation variants, AUC 0.849 vs uniform 0.834;
  provenance in the config's `calibration:` block). A missing/invalid
  config raises `ThetaConfigError` — no silent default.
* **`hallucinated_structure_penalty` is REAL (Plan 12 A3, 2026-06-10):**
  token-level anchoring of each resolved gap's text (words + all
  numeric tokens) against the gap's extraction context plus full source
  text; worst per-gap ratio is the score, with the per-gap breakdown on
  the report. Pass 9c records `GapSlot.resolved_html` as provenance.

---

## 7. Exits — no human escalation (Stage 13)

Per [`feedback_no_external_llms.md`](../../.claude/projects/-home-mdmur-Projects-Semantic/memory/feedback_no_external_llms.md):
runtime is local-only. There is also no human-in-the-loop. The pipeline has
exactly four exit actions, decided by the combination of WCAG hard-gate
status and theta:

The tau values are calibrated and loaded from `theta/config.yaml`
(`theta-config-2.0`: TAU_THETA_RETRY = 0.75, TAU_THETA_CONFIDENCE =
0.85 — see §6.1 provenance):

| WCAG | Lane | Theta | Floor breach | Action |
|---|---|---|---|---|
| pass | fast | ≥ 0.85 | none | `ship_with_confidence` |
| pass | fast | 0.75–0.85 | none | `ship_with_flag` (`meaning_preservation_borderline`) |
| pass | fast | < 0.75 | (any) | `retry_offline` (once) |
| pass | fast | any | floor breach | `ship_with_flag` (specific flag) |
| pass | offline | ≥ 0.85 | none | `ship_with_confidence` |
| pass | offline | < 0.85 | none | `ship_with_flag` |
| pass | offline | any | floor breach | `ship_with_flag` |
| pass | (either) | stubbed | (any) | `ship_with_flag` (`theta_unverified_stub`) |
| **fail** | fast | n/a | n/a | `offline_qwen_lane` (existing) |
| **fail** | offline | n/a | n/a | `non_certified_stamp` (theta omitted) |

When theta is in **stub mode** (the `DART_ALLOW_THETA_STUB=1` mode-collapse
fallback substitutes a flat 0.7 placeholder), the placeholder is meaningless and
must NOT decide the exit. `theta_is_stubbed(report)` keys off the
`semantic_preservation` dimension's `method == "stub_v1"` (self-describing, no
env re-read; `theta/evaluator.py`); `offline_retry._needs_retry` then skips the
theta-`<TAU>` retry trigger (it STILL retries on a real `wcag=failed`), and
`exits.decide_exit` ships `ship_with_flag` with the explicit
`THETA_UNVERIFIED_STUB` flag instead of letting the 0.7 placeholder trip the
offline retry / non-certified path. Byte-stable when theta is real
(`theta/exits.py` + `theta/offline_retry.py`, this session `3a71e57`).

### 7.1 ship-with-confidence

The chosen assembled HTML, byte-for-byte, plus a sidecar JSON with
`{document_id, exit, gate_results, theta_report}`. The HTML carries
`<meta name="dart-certification-status" content="certified">`.

**v1 ships no scalar confidence number.** The closest equivalent is
`theta_report.theta_score` (Stage 12), which is the only quality number on
the wire. A heuristic `confidence` field would be uncalibrated; we wait for
calibration data before reporting one. Consumers wanting a single number
should read `theta_score`. Pass/fail of the WCAG hard gate is a boolean
(`gate_results.passed`).

### 7.2 ship-with-flag

Same as ship-with-confidence, plus a `flags: [...]` field naming each
triggered floor. Consumers can decide whether to surface flags to end-users
or use them for internal QA queues.

### 7.3 offline-Qwen lane

**Same Qwen 3 4B base + same adapters, but K=8 candidates with looser
sampling (temp 0.9 vs. 0.6, top-p 0.95).** No model upgrade — Qwen 3 7B
4-bit will OOM with axe-core's Chromium context concurrent on 8 GB.

The offline lane re-runs Stages 6–9 over only the regions whose fast-lane
gate failed (BERT outputs are cached; council and cross-BERT reranker do
not re-run). One re-entry only. If the offline lane also fails, fall to
non-certified stamp.

### 7.4 non-certified stamp

Two parts — both always emitted on this exit:

**Machine-readable.**
```html
<meta name="dart-certification-status" content="not-certified">
```
plus a sidecar JSON `{document_id, exit, failed_checks}`. Theta is omitted on this exit (`theta_score: null`).

**Human-visible** — first child of `<main>`:
```html
<aside role="note" class="dart-uncertified-banner"
       aria-label="Accessibility certification status">
  <p><strong>Accessibility notice:</strong> This document failed automated
  WCAG 2.2 AA validation and is <em>not certified accessible</em>. Some
  content may not be readable by assistive technology. Contact the document
  publisher for an alternative format.</p>
</aside>
```

Visible-not-just-machine is deliberate: a consumer with no provenance-aware
tooling sees nothing if the stamp is metadata-only. WCAG conformance is a
product claim; failure to meet it must be communicated to the actual reader.

**Consequence — gate threshold is a product decision, not a runtime decision.**
With no human in the loop, every false-positive at the hard gate is a
user-visible accessibility regression; every false-negative is a document
that gets stamped non-certified when it might have passed. Tune τ on a
held-out set before any threshold ever ships.

### 7.5 v1 implementation status (Stage 13)

The Stage 13 exit decider (`dart_semantic/theta/exits.py`) implements
the full §7 decision table. **The offline-Qwen lane is wired and live**
(`theta/offline_retry.py:maybe_offline_retry` re-runs the failing
regions through the offline lane and keeps the offline output iff
`offline_theta ≥ fast_theta + 0.05` and WCAG clears; `cascade.py` calls
it before `decide_exit`). Neither historical degradation flag
(`theta_low_no_retry`, `offline_lane_unavailable_v1`) is emitted
anywhere; six invariant tests in `tests/test_theta.py` lock the §7 rows
and the no-`*_v1`-flag invariant. The lane fired in the R10 real-runtime
eval (2 offline retries across 3 documents; fast lane retained both
times on the theta margin).

---

## 8. Training plan

### 8.1 Strict dependency order

Some BERTs feed others as input features. Training must respect this DAG.

```
1. BERT-MathSpecialist     (ar5iv-only labels — free; proves pipeline shape)
2. BERT-MergeOrSplit       (small data; partly synthesizable from PDF
                            span fragmentation patterns)
3. BERT-Structure          (must converge before Semantic training begins)
4. BERT-Semantic           (teacher-forced on Structure labels;
                            scheduled-sampling near end of training)
5. ~~BERT-TableDetector~~  (RETIRED 2026-05-05 — folded into Structure
                            as the `table_region` binary head; pdfplumber
                            TableCandidate aggregation supplies the
                            grouping. P=R=F1=1.000 region-level eval.)
6. BERT-TableSpecialist    (most expensive labels — last;
                            uses Structure once stable)
7. BERT-MathDetector       (DEFER — geometry is probably high-precision
                            enough; revisit only if MathSpecialist shows
                            false-positives on italicized prose)
```

Each BERT must demonstrate measured lift on `eval_v7_family` before the
next one starts training. **Do not pre-commit to training all seven** — gate
each on actual numbers.

### 8.2 Teacher-forcing for cascaded BERTs

Semantic depends on Structure outputs as input. Standard cascaded
structured-prediction pattern:

- **Training:** feed gold Structure labels alongside text. Pure teacher-forcing
  in early epochs.
- **End of training:** schedule a small fraction of "predicted-output mixing"
  in the final epochs to close the train/test gap.
- **Inference:** Structure runs first; its top-k feeds Semantic.

### 8.3 One label schema, multiple head-projections

Where labels can be derived from the same annotation pass, derive them.
Structure and Semantic can share an arXiv pair labeling pipeline; only Table
and Math need their own annotation paths. This keeps the data budget
tractable on the 8K-pair target.

### 8.4 Corpus mix per BERT

Different BERTs need different source mixes. Do **not** pipe a uniform corpus
into all five.

| BERT | Primary sources | Why |
|---|---|---|
| **MathSpecialist** | ar5iv | Only source with rich MathML supervision |
| **MergeOrSplit** | all sources, weighted toward Internet Archive scans | OCR-noisy sources are the hard cases |
| **Structure** | all five (ar5iv + OpenStax + govinfo + IRS forms + IA) | Cross-domain heading/list patterns; OpenStax is calibration gold |
| **Semantic** | govinfo + IRS + arXiv | Legal/boilerplate signal concentrated in regulatory corpus; arXiv contributes abstract / citation / author |
| ~~**TableDetector**~~ | *(retired 2026-05-05; gating signal now lives on Structure's `table_region` head — corpus mix inherited from Structure's row in this table)* | — |
| **TableSpecialist** | ar5iv + OpenStax + govinfo + IRS forms | Diverse table styles required — academic, educational, regulatory, form-style |
| **MathDetector** *(deferred)* | ar5iv + Internet Archive | Hard cases live in scanned scientific papers |

### 8.5 Cost asymmetry — detector vs. specialist

The detector-then-specialist cascade exploits a real cost asymmetry:

- **Detector:** binary task, abundant negatives, small distilled model.
  ~0.2× a full specialist to train.
- **Specialist:** trains *only* on detector-confirmed positives — distribution
  matches inference exactly. Specialists on filtered distributions reliably
  outperform mixed-distribution training.
- Total cost: ~1.2× a single combined model, not 2×.

This is the same logic that justified splitting BERT-Table into
TableDetector + TableSpecialist; in practice (eval committed 2026-05-05)
the detector half collapsed into Structure's `table_region` binary head
because pdfplumber `TableCandidate` regions are themselves high-precision
and Structure's head agreed with HTML truth on every aggregation
(P=R=F1=1.000 region-level on 170 arXiv held-out regions). The cost
asymmetry still applies — TableSpecialist still trains only on confirmed
positives — but the gate is now a head on Structure rather than a
standalone model. The same future split may still apply to BERT-Math if
measurement justifies it.

---

## 9. Hardware constraints

| Constraint | Source | Implication |
|---|---|---|
| RTX 3060 8 GB VRAM | hardware | No concurrent Qwen adapter contexts; batch-by-adapter |
| `build_qwen` must run serial | [`feedback_qwen_build_serial.md`](../../.claude/projects/-home-mdmur-Projects-Semantic/memory/feedback_qwen_build_serial.md) | 4 shards sequentially, never parallel |
| No external LLMs at runtime | [`feedback_no_external_llms.md`](../../.claude/projects/-home-mdmur-Projects-Semantic/memory/feedback_no_external_llms.md) | All inference is local |
| WSL2 Ubuntu 24.04, Python 3.12, CUDA 12.1+ | dev environment | Council BERTs run on transformers + PEFT (encoder+heads); Qwen specialists run on llama.cpp (decoder-only LLMs); cross-encoders / theta scorer use DeBERTa-v3-small via transformers |

**Council base encoder — ModernBERT-base** (~150 M params, MIT, 8K context,
modern flash attention). DeBERTa-v3-base was the originally-preferred choice
for its disentangled-attention classification strength, but transformers
5.5.4's tiktoken-extractor rejects DeBERTa-v3's SentencePiece `spm.model`
file (a real Phase-2 compatibility blocker). ModernBERT-base was the
documented DP-0.1 fallback and is now the locked choice. The 8K context
turns out to be a Phase-4 win — the cross-BERT reranker sees more neighbor
regions per pass without chunking. DeBERTa-v3-base remains a fallback only
if a future tokenizer-loading workaround (transformers pin or fast-tokenizer
config) lands and ModernBERT shows measurable shortfalls.

**Cross-encoders and theta scorer — DeBERTa-v3-small** (~44 M params, MIT).
Used for: per-region soft reranker (Stage 8), document-level soft reranker
(Stage 11) if promoted from rule-based, theta semantic-preservation scorer
(Stage 12). Small enough to keep multiple resident; cross-encoder regression
heads are cheap on top.

**Council VRAM strategy.** All council BERTs share the DeBERTa-v3-base
encoder with per-BERT LoRA adapters (rank 16, alpha 32 baseline). Swap
adapters between heads at inference; no need to keep all seven resident
simultaneously. Adapter swap is fast (LoRA matrices are small).

---

## 10. Relationship to the legacy 8-stage pipeline

The pre-existing layout described in [`README.md`](README.md) and
[`docs/refactor_plan.md`](docs/refactor_plan.md) was an **eight-stage
deterministic pipeline with one trained model** (a per-block role classifier
at stage 3). The model encoded in that doc is preserved in this new
architecture as a subset of BERT-Structure.

Mapping from the legacy stages to the new ones:

| Legacy stage | New location |
|---|---|
| 1. extract | Stage 1 (unchanged) |
| 2. features | Stage 2 (unchanged — feeds layout features into BERTs) |
| 3. classify (rules + DistilBERT) | Stage 3 BERT council (BERT-Structure subsumes the role classifier; six new BERTs added) |
| 4. hierarchy | Stage 9 (deterministic assembler — heading hierarchy normalization) |
| 5. ontology_map | Stages 5 + 9 (structure-graph generation + assembler — driven by [`docs/ontology.md`](docs/ontology.md)) |
| 6. enrich | Stage 6 Qwen region specialists (prose / table / math) + Stage 9 assembler-driven Qwen-GapFill |
| 7. validate | Stages 7 + 10 (per-region and document-level hard gates) |
| 8. escalate | Stage 12 ThetaEvaluator (new — post-WCAG quality score) + Stage 13 exits (ship-with-confidence / ship-with-flag / offline-Qwen lane / non-certified stamp; no human escalation) |

The ontology in [`docs/ontology.md`](docs/ontology.md) remains the
authoritative WCAG / standards mapping for every emitted element. The new
architecture is the *structural* layer that decides what to emit; the
ontology is the *standards* layer that decides how each emission must look.

---

## 11. Locked architectural choices

The following are decided. Changes require an explicit revision of this doc.

- **6 BERTs in the council** (was 7 pre-2026-05-05) — Structure,
  Semantic, MergeOrSplit, TableSpecialist, MathDetector (deferred
  build), MathSpecialist. **BERT-TableDetector retired 2026-05-05**
  — folded into Structure as the `table_region` binary head;
  pdfplumber `TableCandidate` aggregation supplies the region grouping.
  Eval evidence: `scripts/eval_table_region_at_region_level.py`,
  P=R=F1=1.000 region-level on 170 arXiv held-out regions
  (`data/eval/table_region_at_region.json`). With ImageSpecialist
  added 2026-05-04, the council target is **7 BERTs** total once
  ImageSpecialist ships (Structure, Semantic, MergeOrSplit,
  TableSpecialist, ImageSpecialist, MathDetector deferred,
  MathSpecialist).
- **Multi-head encoders** for Structure, Semantic, TableSpecialist,
  MathSpecialist, MergeOrSplit.
- **MergeOrSplit final shape (Phase 3a v4):** 3 heads (`same_logical_block`,
  `join_type` 3-class with `paragraph_break` dropped, `hyphen_repair`)
  + a 24-dim numeric layout side-channel fed through LayerNorm + 64-dim
  MLP, concatenated with the BERT pooled output before all heads. The
  pair-level `heading_body_boundary` head was attempted in v1/v2/v3b
  and dropped — heading detection moved to BERT-Structure as a
  span-level `is_heading` head (better positives, no adjacency
  conflation).
- **Council base encoder is ModernBERT-base** (locked Phase 2 after a transformers/spm.model compatibility issue blocked DeBERTa-v3-base; the 8K context is a Phase-4 cross-BERT reranker win). Cross-encoders and theta scorer use DeBERTa-v3-small.
- **One cross-BERT reranker**, not per-BERT rerankers.
- **3 directed BERT-feature edges**: MergeOrSplit → all, Structure → Semantic,
  Structure → TableSpecialist (soft hint).
- **4 Qwen LoRA adapters** — prose, table, math, gap-fill. Strict routing,
  no fan-out, batched by adapter on 8 GB VRAM.
- **Gap-fill scope is narrow**: missing-title inference, citation/footnote
  resolution, author/copyright/legal block remediation. **Not** image
  alt-text (owned by the Stage-6b figure captioner). **Not** a document
  assembler.
- **Gap-fill outputs go through the per-region hard gate** before merge-back;
  same axe + html5validator + text-preservation contract as prose/table/math.
- **Merge-back is one-shot** (re-run heading hierarchy + reference resolution
  once if title or refs changed); never iterate (cycle structurally ruled out).
- **Two-tier hard gate** with eliminating semantics; soft rerankers only
  among gate-survivors.
- **Doc-level soft reranker v1 scope**: only fast-lane vs. offline-lane
  multi-assembly. Per-region top-2 fan-out and gap-fill K^M cross-product
  are deferred to v2 (combinatorial trap).
- **Math wins matrix arbitration** in the cross-BERT reranker.
- **Qwen specialist runtime is llama.cpp** (not transformers, not Ollama). Council BERTs stay on transformers + PEFT because llama.cpp targets decoder-only models. Training uses HF/PEFT for both; only Qwen specialist *inference* is llama.cpp.
- **Page template owns body-scope `<header>`/`<footer>`** (per [`docs/ontology.md`](docs/ontology.md) §7). The assembler emits `<main>`, `<nav>`, `<aside>` (legal/copyright), and `<address>` (author); it does NOT wire body-scope `<header>` or `<footer>`. BERT-Semantic's `footer` doc-role is treated as a page artifact and dropped from visible content.
- **No scalar confidence number in v1.** A heuristic confidence would be uncalibrated; theta_score is the single quality number on the wire. Calibrated confidence deferred until held-out calibration data exists.
- **ThetaEvaluator (Stage 12)** is post-WCAG quality reporting, not a gate.
  One learned dimension (semantic preservation, DeBERTa-v3-small cross-
  encoder); seven deterministic dimensions. Three of the seven reuse
  Stage 11's emitted axis scores. Theta MUST NOT override a failed WCAG
  hard gate. Theta is internal-diagnostic; **conformance statements in
  [`docs/ontology.md`](docs/ontology.md) §6 must NOT include theta**.
- **Theta retry policy** is capped at one offline retry on
  `theta_score < TAU_THETA_RETRY` (calibrated 0.75, `theta-config-2.0`);
  no second retry.
- **No human escalation.** Four exit actions: `ship_with_confidence` /
  `ship_with_flag` / `offline_qwen_lane` / `non_certified_stamp`.
- **Training in dependency order**, gated on measured lift per BERT.

---

## 12. The output contract + the cross-venv bridge

SemantiK's output is a **stable, downstream-facing contract** — the HTML it
emits, and the way Ed4All references blocks inside it, are fixed so everything
downstream (Courseforge staging, source-mapping, the chunker, the Ask path)
reads one consistent shape across runs and versions.

### 12.1 The wire contract — block-provenance attributes + `dart:{slug}#{block_id}`

The cascade emits HTML; the **adapter seam** (`lib/semantik/adapter.py`,
`lib/semantik/cascade_ir.py`) normalizes that HTML into Ed4All's chapter IR,
wrapping each content block in a `<section class="dart-section">` carrying
the source-provenance attribute set (`adapter.py::_render_section`):

```html
<section class="dart-section"
         aria-labelledby="{sid}"
         data-dart-block-id="{sid}"          <!-- the sourceId block_id -->
         data-dart-source="synthesized"      <!-- or "vendor" via vendor_ingest -->
         data-dart-pages="1,3-5"             <!-- physical PDF pages -->
         data-dart-page-kind="physical"      <!-- honest; never "printed" -->
         data-dart-confidence="0.80"         <!-- 5-point band -->
         data-dart-block-role="section"
         data-dart-wcag="passed">            <!-- per-region Stage-7 verdict -->
  <h3 id="{sid}">…</h3>
  …content…
</section>
```

The **sourceId** is `dart:{slug}#{block_id}`:

- `{slug}` = the document slug (file stem, via `dart_slug_from_filename`).
- `{block_id}` (`sid`) = a **deterministic** block key minted by
  `adapter.py::_mint_sid`. Default key is the block's **first raw
  FeatureBlock index** (`b{raw_block_index}`) — never the post-model region
  order, so the id is stable across re-runs on the same PDF. Under
  `TRAINFORGE_CONTENT_HASH_IDS=1` it switches to a content hash of the raw
  text (stable iff the source text is unchanged).

This is the determinism contract Ed4All depends on:
**same PDF in → same sourceIds out**, so chunk `learning_outcome_refs[]`,
`source_module_map.json`, and citation deep-links resolve across re-runs.

Provenance honesty markers worth knowing: `data-dart-mock="true"` (only on
MockRuntime output — a real run never carries it), `data-dart-cell-roles=
"qwen-inferred"` (table cell roles guessed by the Qwen specialist, not
verified by BERT-TableSpecialist), and `data-dart-fabricated="title"` (a
Stage-9c gap-filled missing title — synthetic, not extracted).

### 12.2 `region_provenance`

The cascade emits a per-region provenance list in document (emission) order
(`cascade.py::_build_region_provenance`). Each entry records, per region:
`region_index`, `region_kind`, `role`, `confidence`, `wcag_status`
(`"passed"` / `"failed"` / `None`), `first_raw_block_index` (the §12.1
determinism key), `pages` (sorted 1-indexed physical PDF pages),
`heading_text` + `level` (headings/figures), `figure_alt` (the Stage-6b
caption), `raw_text` (the deterministic extracted text the sid hashes), and
an OPTIONAL `review` block (present only when Stage-5d ran and corrected a
heading: `corrected_from`/`corrected_to`/`level_from`/`level_to`/
`reason_code`/`reverted`/`note`). The adapter consumes this list to build
the chapter IR and apply the deterministic phantom-TOC / front-matter filter
(§14, `lib/semantik/toc_frontmatter_detector.py`) before chapters assemble.

### 12.3 The conformance audit

Alongside the HTML, the cascade emits a machine-readable
`*.conformance_audit.json` (`conformance_audit.py::build_conformance_audit`,
schema `conformance-audit/1.0`) recording: the final `exit` action +
`wcag_status` + lane used; the Stage-7 per-region gate log (per-candidate
verdicts + **skip counts**); the Stage-10 document axe summary (violations by
rule id); the `theta` report (`theta_score` + flags); the decision
`thresholds`; the `assembly` block (heading tree, `region_provenance`,
Stage-5d verdicts); and `wcag_coverage` (rule id → WCAG SC mapping). **Skips
are first-class** — a CheckOutcome with `skipped=True` means "no
measurement", not "verified safe", so a document whose text-preservation gate
skipped 80% of its regions is visibly different from one fully measured. This
is the anti-silent-degradation contract on the accessibility surface.

### 12.4 The cross-venv bridge

SemantiK's runtime pulls in **heavy ML deps** (torch, transformers, peft,
`llama-cpp-python` built against CUDA, sentence-transformers for theta) that
do NOT belong in Ed4All's MCP/orchestrator venv. So the cascade runs **out
of process**, in its own venv, behind a JSON bridge:

- `SemantiK/scripts/run_cascade_json.py` is the subprocess entry point: it
  takes a PDF path, runs `run_full_cascade`, and writes the HTML +
  `region_provenance` + conformance audit as JSON to stdout.
- `MCP/tools/pipeline_tools.py` invokes it. `SEMANTIK_PYTHON` is the absolute
  path to the SemantiK venv's python; `SEMANTIK_RUNTIME_DIR` is the SemantiK
  repo root used as the subprocess `cwd` (so model/cache dirs resolve). When
  the in-process SemantiK deps are absent and `SEMANTIK_PYTHON` is unset, the
  bridge **fails closed with operator guidance** (no silent stub).
- `SEMANTIK_BRIDGE_TIMEOUT_SECONDS` (default 3600s) caps the subprocess;
  `SEMANTIK_EXPANDABLE_SEGMENTS` opts the subprocess into
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (8 GB OOM mitigation).

The bridge is the seam that lets Ed4All stay lean while SemantiK keeps its
GPU-heavy ML stack isolated. Everything crossing it is the §12.1 wire
contract, so the rest of Ed4All consumes conversion output through one stable
interface.

---

## 13. Data flow (end to end)

```
                                  ┌─────────────────────────────────────────┐
   PDF                            │  SemantiK venv  (SEMANTIK_PYTHON)         │
    │   MCP/tools/pipeline_tools  │                                           │
    │   ──────────────────────►   │  run_cascade_json.py → run_full_cascade   │
    │      JSON bridge            │                                           │
    │   (SEMANTIK_RUNTIME_DIR cwd)│   1  extract   pikepdf/pypdfium2/         │
    │                             │                pdfplumber/Tesseract       │
    │                             │   2  features  font/geometry/columns      │
    │                             │   3  council   Structure · Semantic ·     │
    │                             │                MergeOrSplit · Table ·      │
    │                             │                Math  (LoRA adapter-swap)   │
    │                             │   4  cross-BERT reranker  (arbitrate)     │
    │                             │   5  structure_graph  → typed Regions     │
    │                             │   5b GLM-OCR table enrich (opt-in)        │
    │                             │   5c figure bbox → PNG bytes              │
    │                             │   5d 70B structure reviewer (OFF default) │
    │                             │   6  Qwen specialists  prose/table/math   │
    │                             │      (local GGUF | hosted 70B endpoint;   │
    │                             │       batched two-phase, by adapter)      │
    │                             │   6b figure captioner (SmolVLM2)          │
    │                             │   7  per-region HARD gate  (axe/html5/    │
    │                             │      text_preserve/mathml/table/heading)  │
    │                             │   8  per-region SOFT reranker  (pick top) │
    │                             │   9  assembler  role→HTML · heading tree ·│
    │                             │      gap-fill splice  (pass_9a/9b/9c)     │
    │                             │  10  document HARD gate  (axe/lang/title/ │
    │                             │      landmark/heading contiguity)         │
    │                             │  11  document SOFT reranker               │
    │                             │  12  theta  (DeBERTa-v3-small + LoRA       │
    │                             │      semantic-preservation cross-encoder) │
    │                             │  13  exit decider (+ one offline retry)   │
    │                             └─────────────────────────────────────────┘
    │                                              │
    ▼   JSON over the bridge ◄─────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│  lib/semantik/adapter.py + cascade_ir.py  (Ed4All venv — pure transform) │
│   · drop phantom-TOC / front-matter (toc_frontmatter_detector.py)        │
│   · wrap blocks in <section class="dart-section" data-dart-*=…>          │
│   · mint sourceIds  dart:{slug}#{block_id}                               │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  Accessible HTML  +  region_provenance  +  *.conformance_audit.json
  (consumed by Courseforge staging / source-mapping / the chunker — unchanged)
```

---

## 14. Known limitations

Honest constraints an operator should know going in:

- **Council VRAM on 8 GB.** The full cascade is **GPU-flaky on an 8 GB
  card.** The council BERTs share one ModernBERT-base backbone with a
  one-resident-LoRA-adapter discipline (§3, §9), and the Qwen specialists
  are batched **by adapter** rather than fanned out, precisely because
  parallel adapter contexts + a concurrent Chromium/axe-core process poison
  CUDA on 8 GB. This is mitigated, not eliminated — long math regions, the
  figure captioner (SmolVLM2), and theta all want the same card. The Spark
  deployment target (below) is the real fix; on a dev box, expect to gate
  GPU-heavy work and accept occasional OOM retries.

- **Structure quality is council-bound.** The block-ID quality of
  *pedagogical* elements (correctly segmenting a worked example vs. a
  definition vs. a checkpoint) is only as good as BERT-Structure's
  `structural_role` / `is_heading` heads. Heading over-detection on
  TOC/front-matter is a known failure mode; it is patched in **two**
  defensive layers — the off-by-default Stage-5d 70B reviewer
  (`SEMANTIK_STRUCTURE_REVIEW`) and the always-on **deterministic**
  front-matter / phantom-TOC detector at the adapter seam
  (`lib/semantik/toc_frontmatter_detector.py`, the page-density +
  monotonic-pagenum-run discriminator). The deterministic detector is the
  load-bearing one; the 70B reviewer is **conservative and off by default**
  (it may only correct headings under a strict text-conservation invariant,
  and fails closed to the unreviewed output on any token mismatch).

- **The structure reviewer is conservative by construction.** Stage-5d never
  touches text, never re-partitions FeatureBlocks, and reverts the whole
  region list on any document-level token-conservation violation. It will
  miss corrections it cannot make safely — that is the intended trade
  (no fabrication over more aggressive repair).

- **Deployment target is the Spark era.** SemantiK is built to run the local
  GGUF specialists on a dev box for development and the hosted 70B endpoint
  seat (`SEMANTIK_SPECIALIST_PROVIDER=nvidia`) for quality, but the intended
  production home is an NVIDIA DGX Spark-class box where the full council +
  specialists + theta fit resident without the 8 GB contention dance. Until
  then, the local lane is functional but VRAM-disciplined.

---

*Document version: 2026-06-23. §1–§11 are the target cascade architecture
(version 2026-05-03), superseding the "two trained models, Qwen as single
decision-maker" design; §12–§14 document the live output contract, cross-venv
bridge, and honest limitations of the conversion engine. WCAG / standards
mapping continues to live in [`docs/ontology.md`](docs/ontology.md).*
