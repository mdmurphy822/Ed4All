# DART/Semantic Implementation Plan: v1 → BERT-Council + Qwen-Specialist Architecture

> **STATUS (banner added 2026-06-17): LARGELY SHIPPED.** The v2 council + Qwen-specialist
> + theta + assembler architecture is model-complete (real-runtime corpus eval 2026-06-09).
> Retained as the foundational design record; **`architecture.md` is canonical** for current
> structure. See git log for what landed.

**Plan version:** 2026-05-03
**Plan author:** Architect agent (read-only planning pass)
**Authoritative target:** `architecture.md`
**Authoritative WCAG/standards mapping:** `docs/ontology.md`
**Self-contained.** Future sessions and execution agents do not need conversation history.

## 0. Orientation — what is, what target

### 0.1 What exists today (v1)

The v1 pipeline is an 8-stage deterministic flow with **two** trained models, not one as the README claims:

- **Stage 3a — DistilBERT block classifier** (`dart_semantic/classify.py`, `train_classifier.py`). 22 mutually-exclusive Role classes. Adapter at `models/classifier_v5/final` is current. Input string built by `features.feature_block_to_classifier_input`.
- **Stage 3b — Qwen 3 4B LoRA reasoner** (`dart_semantic/reason.py`, `train_reasoner.py`). One-shot per-block label re-emit over page-chunked context, with DistilBERT's label as a hint flag in the prompt. Adapter at `models/reasoner_v8/final`. Trained data at `data/qwen_dataset_v8`. Locked schema in `dart_semantic/reason_schema.py`.

Stages 1, 2, 4–8 are deterministic. `validate.py` already runs axe-core (`wcag22aa` ruleset) under Playwright; `escalate.py` already routes ship/llm_fallback/fail.

The pipeline produces zero-violation WCAG 2.2 AA HTML on four held-out arXiv papers. `architecture.md` is the *target*; v1 is what runs today.

### 0.2 What the target architecture changes

- Stage 3 splits from `(DistilBERT + Qwen)` into a **7-BERT council** (one of which subsumes today's DistilBERT).
- A new **Stage 4 cross-BERT reranker** is added.
- Stages 5+ are renumbered: structure-graph (5), Qwen specialists (6), per-region hard gate (7), per-region soft reranker (8), assembler (9), doc hard gate (10), doc soft reranker (11), exits (12).
- Qwen splits from a single labeling Qwen into **4 LoRA specialists** (prose / table / math / gap-fill), each batched, never interleaved on 8 GB.
- Two-tier hard gate (per-region, per-document) replaces the v1 single-pass axe gate.
- `escalate.py`'s tri-verdict becomes the formal v12 exit set: **ship-with-confidence / offline-Qwen lane / non-certified stamp**. No human escalation.
- `enrich.py` is mostly absorbed by the deterministic assembler (Stage 9) plus the gap-fill specialist.

### 0.3 Constraints reaffirmed

- RTX 3060 8 GB: never two Qwen adapter contexts simultaneously; serial `build_qwen`.
- No external LLMs at runtime (training-time labeling from ground-truth HTML stays as-is).
- One label schema, multiple head-projections — Structure and Semantic share an arXiv-pair labeling pipeline.
- Preserve v1 throughout phasing. New code lives alongside; v1 only retires after a target-arch run beats it on the four held-out arXiv papers and the broader eval corpus.
- Months of work. 11 phases (Phase 0–N+7 in the user's nomenclature).

### 0.4 Phase numbering convention

The user's "Phase N+X" pattern is preserved for clarity. Concretely the phases below are 0..10:

| User label | Plan label | Topic |
|---|---|---|
| Phase 0 | Phase 0 | Scaffolding |
| Phase 1 | Phase 1 | Layout/geometry upgrades |
| Phase 2 | Phase 2 | Council shell + first BERT (Math) |
| Phase 3+ | Phase 3 | Per-BERT rollout in DAG order |
| Phase N | Phase 4 | Cross-BERT reranker |
| Phase N+1 | Phase 5 | Qwen specialist scaffolding (prose-first) |
| Phase N+2 | Phase 6 | Qwen specialists (table, math, gap-fill) |
| Phase N+3 | Phase 7 | Per-region hard gate |
| Phase N+4 | Phase 8 | Per-region soft reranker |
| Phase N+5 | Phase 9 | Deterministic assembler |
| Phase N+6 | Phase 10 | Document-level hard gate + soft reranker + exits |
| (new)     | Phase 11 | ThetaEvaluator (post-WCAG meaning-preservation score) |

(User's "N+7 — Exit modes" folds into Phase 10 because the exit selector is a thin top-of-pipeline routing decision once the gates and rerankers exist; splitting it into its own phase produces unhelpfully short work.)

---

## 1. Phase plan

Each phase carries: **Goal · Inputs · Outputs · Hard prerequisites · Deliverables · Validation · Complexity · Decision points.**

Complexity scale: **S** ≈ 1–2 days; **M** ≈ 1 week; **L** ≈ 2–4 weeks; **XL** ≈ 4+ weeks. Total project on a single dev with this hardware: ~6–9 months.

---

### Phase 0 — Scaffolding (Complexity: **S**)

**Goal.** Add module skeletons, type definitions, config loading, and a parallel package directory for the new architecture without breaking v1.

**Inputs.**
- `dart_semantic/types.py` — current dataclasses (RawBlock, FeatureBlock, ClassifiedBlock, ResolvedBlock, Enrichment).
- `dart_semantic/pipeline.py` — current orchestrator.
- `dart_semantic/__init__.py` — current docstring naming the 8 stages.
- `architecture.md` §2 (12-stage pipeline) and §3 (BERT council).

**Outputs (new files).**

```
dart_semantic/
  council/
    __init__.py
    types.py              # TypedSignal, BertOutput, CouncilState, RegionCandidate
    base.py               # SharedBackbone abstract; LoRAAdapter loader
    registry.py           # Central registry of (bert_name -> adapter_path, head_spec)
    routing.py            # DAG: which BERTs run, which feed which (architecture §3.1)
    config.yaml           # Per-BERT hyperparameters, threshold τ, base-encoder choice
  qwen_specialists/
    __init__.py
    types.py              # SpecialistRequest, Candidate, AdapterID
    base.py               # AdapterSwap context manager (serial enforcement)
    config.yaml           # Per-adapter sampling parameters
  gates/
    __init__.py
    hard_region.py        # Stub: axe + html5validator + text-preserve interface
    hard_document.py      # Stub: full-doc axe + heading-tree + landmark checks
    soft_region.py        # Stub: cross-encoder reranker interface
    soft_document.py      # Stub: doc-level reranker interface
  assembler/
    __init__.py
    types.py              # GapSlot enum, AssembledDoc
    skeleton.py           # Empty deterministic assembler
  pipeline_v2.py          # New orchestrator behind a feature flag; calls v1 by default
  v2_config.py            # Config dataclass: which phase's models are loaded
```

**Hard prerequisites.** None.

**Deliverables checklist.**
- [ ] `council/`, `qwen_specialists/`, `gates/`, `assembler/`, `theta/` package skeletons exist with empty `__init__.py` and dataclass-only `types.py`.
- [ ] `council/types.py` defines `TypedSignal` (one BERT's output: head_name, top-k labels, top-k confidences, region_id, feature_provenance), `BertOutput` (collection of TypedSignals from one BERT), `CouncilState` (collection across all BERTs that ran).
- [ ] `council/base.py` defines `SharedBackbone` (abstract: load encoder weights once) and `LoRAAdapter` (load/swap LoRA matrices per BERT).
- [ ] `qwen_specialists/base.py` enforces serial adapter swap — context manager `with adapter_loaded("prose"): generate(...)` releases the adapter before another can be loaded.
- [ ] `assembler/types.py` defines `GapKind` (5-value enum: missing_title, citation_unresolved, author_block, copyright_block, legal_disclaimer), `GapSlot`, `CertificationStatus` (3-value enum), `AssembledDoc`, `DocCandidate`, `ExitDecision`. Concrete shapes per `Plans/04_assembler_layer_investigation.md` §10.
- [ ] `theta/types.py` defines `ThetaReport`, `ThetaConfig`, `ThetaDimension`, `CognitiveLoadRisk`, `ConfidenceAction`, `ThetaFlag`. Concrete shapes per `Plans/03_theta_investigation.md` §9. Plus `theta/config.yaml` with versioned weights/thresholds.
- [ ] `pipeline_v2.py` exists and, when called with the feature-flag off, delegates to `pipeline.run_pipeline` (v1 unchanged).
- [ ] `v2_config.py` carries a single dataclass that names which BERTs and which Qwen specialists are loaded — early phases set most to `None`.
- [ ] No new runtime-required dependency. The skeletons import only stdlib + already-listed deps.

**Validation criteria.**
- v1 invariant — `python scripts/infer_pdf.py path/to/arxiv.pdf` still runs and produces the same HTML, byte-for-byte, against a frozen pre-Phase-0 reference.
- `pipeline_v2.run(pdf, mode="v1")` produces identical output.
- Unit tests: `tests/test_council_skeleton.py` instantiates each new dataclass with stub data and serializes to JSON; `tests/test_adapter_serial.py` proves the adapter context manager raises if entered twice without exit.

**Decision points.**
- **DP-0.1: Council base encoder family. LOCKED in Phase 2: ModernBERT-base.** DeBERTa-v3-base was originally preferred (stronger classification empirics) but transformers 5.5.4's tiktoken-extractor rejects DeBERTa-v3's `spm.model` SentencePiece file. ModernBERT-base (~150 M params, MIT, 8 K context, modern flash attention) was the documented fallback and is now the actual choice — the 8K context is a Phase-4 cross-BERT reranker win. DeBERTa-v3-base stays a fallback only if a tokenizer workaround lands and ModernBERT shows measurable shortfalls. Cross-encoders and the theta semantic-preservation scorer use **DeBERTa-v3-small** (~44 M params).
- **DP-0.2: Configuration mechanism.** YAML in-tree (recommended; matches `experiments/configs.yaml`) vs. dataclass-with-defaults vs. argparse-only. Pick YAML for council/specialists; argparse for one-off training entry points.

**Time / risk.** ~3 days. Zero risk to v1 — purely additive.

---

### Phase 1 — Layout/geometry upgrades (Complexity: **M**)

**Goal.** Stage 2 emits, in addition to today's flat `FeatureBlock` stream, two new artifacts: **table-region candidates** and **math-region candidates**, both deterministic, both consumed by Stage 3 detector BERTs in later phases. No model work.

**Inputs.**
- `dart_semantic/features.py` — current `featurize_from_shared`, which already attaches `in_table` / `in_header_row` / `in_widget` flags from pdfplumber + pikepdf.
- `dart_semantic/region_detection.py` — already has `detect_math_regions` and `detect_table_regions`. These are mature and unit-test-able.
- `dart_semantic/extract_shared.py` — provides the merged JSON these regions are derived from.

**Outputs (modified + new files).**
- `dart_semantic/features.py` — add `featurize_with_regions(shared) -> FeatureSet` that returns the `FeatureBlock` stream **and** the `RegionCandidate` lists.
- `dart_semantic/region_detection.py` — extended:
  - `detect_table_region_candidates(shared) -> list[TableCandidate]` (typed, geometric-only — wraps current `detect_table_regions`).
  - `detect_math_region_candidates(shared) -> list[MathCandidate]` (wraps current `detect_math_regions`).
  - Add `RegionCandidate` typed dataclass with `kind`, `bbox`, `pages`, `evidence_features` (which heuristic flags fired), `member_block_indices` (indices into the FeatureBlock stream).
- `dart_semantic/types.py` — add `FeatureSet` aggregator: `feature_blocks: list[FeatureBlock]`, `table_candidates: list[TableCandidate]`, `math_candidates: list[MathCandidate]`.
- `tests/test_region_candidates.py` — golden tests on three synthetic shared-extract JSON files (one with a clear table, one with a displayed equation, one with neither).

**Hard prerequisites.** Phase 0 (for `council/types.py` to know what a region looks like). `RegionCandidate` is defined in `region_detection.py` but mirrored / re-exported via `council/types.py` so council consumers don't need to import from a deeper module.

**Deliverables checklist.**
- [ ] `RegionCandidate` typed dataclass shared between `region_detection.py` and `council/types.py`.
- [ ] `featurize_with_regions` returns the union artifact; old `featurize_from_shared` continues to exist and is the sole entrypoint v1 still uses.
- [ ] `MathCandidate` carries glyph-density features (CMMI/CMSY frequency, equation-number trailing pattern, fraction-bar count) precomputed.
- [ ] `TableCandidate` carries bbox + row count from pdfplumber + estimated col count + presence of detected header row (the `in_header_row` heuristic).
- [ ] Unit tests — at minimum 3 fixtures, 5 assertions each (presence, count, bbox stability, member-block correctness, evidence_features non-empty).
- [ ] Coverage report. The v1 path through `featurize_from_shared` hits every line it did before.

**Validation criteria.**
- **Recall on a 30-doc held-out arXiv set:** ≥95% of ground-truth tables (pulled from ar5iv `<table>`) overlap a `TableCandidate` bbox by ≥50% IoU.
- **Recall on a 30-doc held-out ar5iv math set:** ≥90% of `<math display="block">` elements overlap a `MathCandidate` bbox by ≥50% IoU.
- **Precision deferred** to Phase 2/3 (BERT-MathDetector and Structure's `table_region` binary head aggregating over pdfplumber `TableCandidate` regions — these *are* the precision improvers; the geometry pass is allowed to over-fire).
- v1 byte-for-byte regression on the held-out 4 arXiv papers still passes.

**Decision points.**
- **DP-1.1: How aggressive is candidate over-firing allowed to be?** Recommended: very aggressive. The detector BERTs in Phase 2/3 will reject false positives — recall is the geometric pass's only KPI.
- **DP-1.2: Should `RegionCandidate` carry the cell-grid for tables?** Yes — pdfplumber already produces it (`page["pdfplumber"]["tables"][i]["rows"]`). Forwarding it into the candidate avoids re-running pdfplumber when the TableSpecialist trains.

**Time / risk.** ~1 week. Low risk; no model.

---

### Phase 2 — Council shell + first BERT (BERT-MathSpecialist) (Complexity: **L**)

**Goal.** Build the council infrastructure (shared base encoder + LoRA adapter loading; typed-signal output schema) **and** the first BERT in dependency order. `architecture.md` §7.1 says BERT-MathSpecialist first because ar5iv labels are free. Confirmed: ar5iv produces `<math>` markup with display/inline distinction without any human labeling.

**Inputs.**
- `dart_semantic/parse_ar5iv.py` — already parses ar5iv HTML to legacy IR. Extend its label extractor to emit `MathSpan(bbox, mathml, alttext, math_type)` triples directly.
- Extracted ar5iv corpus on disk (training-side; not in repo). User has ~8K arXiv pairs target; ar5iv coverage of those pairs determines how much MathSpecialist data exists.
- `dart_semantic/region_detection.detect_math_regions` — gives Phase-1 candidates that supervised math labels align to.
- `council/types.py`, `council/base.py`, `council/registry.py` from Phase 0.

**Outputs (new files).**
- `dart_semantic/council/math_specialist.py` — module-local: `MathSpecialistHead` (multi-head: math-type {inline, display, numbered, multiline, matrix}; equation-number-association {none, attached_left, attached_right, separate}). Owns its own input serializer that consumes `MathCandidate` features.
- `data/build_math_specialist_data.py` — pair files + ar5iv `<math>` labels → JSONL with one row per math candidate.
- `train_math_specialist.py` — fine-tune ModernBERT-base + LoRA adapter, output `models/council/math_specialist/final/`. Adapter-only weights (~15 MB). Mirror the structure of `train_classifier.py` (label list → LABEL2ID, weighted loss, early stopping on macro-F1).
- `dart_semantic/council/runner.py` — common `run_bert(name, inputs) -> BertOutput`. Loads shared backbone once; swaps adapter; runs.
- `tests/test_math_specialist.py` — synthetic candidate batches; assert deterministic output shape, no GPU-only requirements.

**Hard prerequisites.**
- Phase 0 (council types + base.py).
- Phase 1 (`MathCandidate` exists).
- DP-0.1 resolved (base encoder fixed).

**Deliverables checklist.**
- [ ] `council/base.py`'s `SharedBackbone` actually loads ModernBERT-base once and keeps it warm; `LoRAAdapter.load(name)` swaps in adapter weights without re-loading the encoder. VRAM: encoder ~600 MB + adapter ~50 MB; verified with `gpu-status` skill before/after swap.
- [ ] `train_math_specialist.py` trains successfully on the available ar5iv math labels. Macro-F1 ≥ 0.80 on the math-type head, ≥ 0.70 on the equation-number-association head, on a held-out 10% split.
- [ ] `data/build_math_specialist_data.py` produces train/val/test JSONL splits with class-balance reporting.
- [ ] `BertOutput` JSON-serializable; round-trip test passes.
- [ ] Adapter swap test: load MathSpecialist, run, swap-out, swap-in fresh — same output bit-for-bit modulo float-tolerance. Confirms LoRA load/unload contract.
- [ ] **No regression to v1.** v1 paper-evaluation produces unchanged HTML.

**Validation criteria.**
- **Held-out F1.** BERT-MathSpecialist macro-F1 on math-type head ≥ 0.80; on equation-number-association head ≥ 0.70.
- **Recall lift on candidates.** Among Phase-1 `MathCandidate` regions that are *true* math (per ar5iv ground truth), ≥ 95% are emitted with a non-`unknown` math-type.
- **Latency.** On a 20-page arXiv paper: BERT-MathSpecialist runs in < 2 seconds wall (batched, GPU). Acceptable upper bound: 5 seconds.
- **Adapter-only checkpoint size.** ≤ 30 MB (LoRA r=16; rank can be tuned later if quality demands).

**Decision points.**
- **DP-2.1: Is BERT-MathSpecialist actually first, or BERT-MergeOrSplit?** `architecture.md` §7.1 lists MathSpecialist first because ar5iv labels are free *and* it proves pipeline shape. MergeOrSplit needs synthetic span-fragmentation labels which are also nearly-free. Recommend: **MathSpecialist** first because it touches the council infrastructure end-to-end (region candidate → detector input → specialist output → typed signal) more visibly than MergeOrSplit, which sits at flat-text. MergeOrSplit follows immediately as Phase 3a.
- **DP-2.2: LoRA rank.** r=16, alpha=32 works for `train_reasoner.py`; same for council. Empirically 8 may be enough — tune on first BERT and lock for the rest.
- **DP-2.3: What math-type label set?** `architecture.md` §3 names "inline / display / numbered / multiline / matrix". Confirm by sampling 100 ar5iv `<math>` blocks and checking the label coverage. If "matrix" is < 1% prevalence, drop it from the schema and let cross-BERT arbitration handle matrix-vs-table ambiguity directly. **Plan-side recommendation: keep all 5; matrix is rare but its presence in the schema is *load-bearing* for the math-wins-matrix arbitration rule (architecture §3.2).**

**Time / risk.** ~3 weeks (1 week labeling + builder, 1 week training + tuning, 1 week integration + tests).
**Risks.** ar5iv → arxiv-PDF bbox alignment is the single hardest sub-problem. `dart_semantic/arxiv_sections.py` already does this for sections; reuse its alignment logic.

---

### Phase 3 — Remaining BERTs in dependency order (Complexity: **XL**, broken into 3a–3f; 3d retired 2026-05-05)

Each sub-phase repeats Phase-2's pattern: build labeling pipeline, train adapter, integrate, validate, gate next BERT on measured lift. **No BERT graduates to "merged" until the project's eval harness shows lift.**

**Common deliverables for each sub-phase.**
- `dart_semantic/council/<bert_name>.py` (head module).
- `data/build_<bert_name>_data.py` (label extractor — shares arXiv-pair pipeline where possible per architecture §7.3).
- `train_<bert_name>.py` (top-level entry).
- `models/council/<bert_name>/final/` (adapter only).
- Held-out measurement on `eval_v7_family` (as architecture §7.1 demands).

**Common gating rule.** Do not start sub-phase N+1 until sub-phase N has its adapter checked in *and* `scripts/eval_v7_family.sh` shows non-trivial lift on at least one v7 metric vs. running pipeline-v2 with that BERT disabled.

#### Phase 3a — BERT-MergeOrSplit (M)

**Goal.** Multi-head classifier on adjacent span pairs. **4 learned heads + deterministic geometry features:**

| Head | Output | Purpose |
|---|---|---|
| `same_logical_block` | binary | primary merge decision |
| `join_type` | 4 classes (`space`, `newline_within_p`, `paragraph_break`, `list_continuation`) | conditional on positive primary; tells the assembler how to splice |
| `hyphen_repair` | binary | line-end "accessi-" + line-start "bility" → "accessibility"; ar5iv ground truth at line-wrap boundaries |
| `heading_body_boundary` | binary | safety: blocks the assembler from merging a heading into the next paragraph |

**Deterministic features fed alongside text** (computed at extract time, NOT learned): `column_id` (from x-range geometry), `is_artifact` (from `extract_shared` artifact heuristics — running headers / page numbers), plus font_size_a/b, weight, x_range, y_gap, line_height_ratio, ends_with_hyphen. These keep layout-driven decisions out of the learned head and free the model to focus on language signals.

- Inputs: `extract_shared` output's per-page `pypdfium2.text_blocks` plus the merged stream — disagreements between them are the natural training signal. Adjacent-span ground truth from ar5iv (`<p>`, `<li>`, `<th>`, `<h{1..6}>`, `<figcaption>`, `<blockquote>` boundaries plus tag-type for `heading_body_boundary` label).
- Multi-source per architecture §8.4: Internet Archive scans weighted heavily for the OCR-noisy hard cases.
- Heads NOT included (and why): `same_line_reflow` ≡ (`same_logical_block=yes` ∧ `join_type=space`); `paragraph_continuation` ≡ (`same_logical_block=yes` ∧ `join_type=newline_within_p`); `column_boundary` and `artifact_boundary` are layout-driven, modeled as input features.
- Validation: `same_logical_block` macro-F1 ≥ 0.85 on held-out (incl. OCR-noisy subset). Per-head macro-F1 reported separately.

#### Phase 3b — BERT-Structure (L) — *subsumes today's DistilBERT*

**Goal.** Multi-head: structural-role · is_heading · heading-level · list-nesting. **Replaces** `train_classifier.py`'s 22-class single-head with a 4-headed model on the shared backbone. Today's classifier becomes the structural-role head; the three new heads are added.

The `is_heading` head is new vs. the prior plan — it's a span-level binary `{not_heading, heading}` that absorbs the heading-detection responsibility from MergeOrSplit (where it lived as a pair-level `heading_body_boundary` head and didn't work — see Phase 3a closure note in `Plans/HANDOFF.md`). The pair-level framing was structurally wrong (~2.6% positives, conflated heading detection with adjacency). Span-level has 30-50× more positives. The `heading-level` head (h1..h6) is then **conditional** on `is_heading=1` — trained only on positives, with the loss masked elsewhere via `ignore_index=-100`. This is the same conditional-head pattern MergeOrSplit's `join_type` uses.

- Reuses the `data/build_classifier_data_v2.py` label extraction pipeline; widens it to also emit:
  - `is_heading` (binary, from whether the span aligns to an `<h1>`..`<h6>`).
  - `heading-level` (1..6, only for `is_heading=1` rows).
  - `list-nesting` (DOM depth of `<li>`, in {0, 1, 2, 3+}).
- Inherits the layout side-channel pattern from MergeOrSplit Phase 3a v4: 24-dim numeric vector → LayerNorm → 64-dim MLP → concatenated with BERT pooled before the four heads. The `is_heading` head specifically benefits from this — font_size_norm, bold flag, h_a/median_h, b_titlecase_frac are all directly relevant.
- This is the pivot phase. Once Structure ships, the v2 pipeline can begin replacing v1's stage-3 (DistilBERT). v1 isn't deprecated yet — the new model proves out on a 2-week shadow run.
- Validation:
  - Structural-role head macro-F1 ≥ that of `models/classifier_v5/final` (no regression).
  - `is_heading` head F1 ≥ 0.85 on held-out ar5iv pairs (the metric MergeOrSplit's `heading_body_boundary` failed at — span-level is the test that this redesign actually works).
  - `heading-level` head exact-match accuracy ≥ 0.80 on `is_heading=1` rows.
  - List-nesting head MAE ≤ 0.5.
  - Pipeline shadow run on the 4 arXiv papers: zero-violation HTML maintained.

#### Phase 3c — BERT-Semantic (L)

**Goal.** Multi-head: doc-role (title/author/abstract/body/citation/footer/legal/metadata) · boilerplate flag. **Cascaded from Structure** per architecture §7.2 (teacher-forced on Structure labels, scheduled-sampling at end).

- Corpus mix: govinfo + IRS + arXiv (architecture §7.4) — legal/boilerplate signal lives in the regulatory corpus; arXiv contributes abstract/citation/author.
- Training: feed gold Structure top-k as input feature; final 10% of epochs use scheduled-sampling with predicted Structure top-k.
- Validation: doc-role macro-F1 ≥ 0.75; boilerplate-flag F1 ≥ 0.85.

#### Phase 3d — BERT-TableDetector (M) — **RETIRED 2026-05-05**

**Status: RETIRED.** Standalone BERT-TableDetector is not being built. Structure's `table_region` binary head (added in Phase 3b, promoted from a role-class to a dedicated head — see task #26) plus pdfplumber `TableCandidate` aggregation already does the gating job that 3d was scoped to do.

**Eval evidence.** `scripts/eval_table_region_at_region_level.py` against `data/eval/table_region_at_region.json` (170 regions across 20 high-density arXiv held-out pairs): **P = R = F1 = 1.000 region-level**, span-level F1 = 0.986. `data/eval/table_region_low_density.json` (10 prose-heavy arXiv pairs) surfaced **0** pdfplumber candidates — no false-candidate population for a separate detector to filter. pdfplumber candidates are themselves high-precision; Structure's binary head agrees with HTML truth on every aggregation. A standalone detector model has nothing to learn beyond what 3b already encodes.

**Decision committed in `15dccc5` (5th head promotion) and `2473002` (region-level eval).**

**Budget reallocation.** ~3 weeks of saved scope go to Phase 3f (BERT-ImageSpecialist) scaffolding, where the `is_image_block` Structure head + specialist cascade pattern is the same shape as the retired detector but on a head that *did* show weakness in eval (`figure_caption` 0.42 precision in Structure v1).

**Section heading retained for anchor stability.** Future plan-doc references should point at `#phase-3d--bert-tabledetector-m--retired-2026-05-05` (or the original anchor) and read this notice.

#### Phase 3e — BERT-TableSpecialist (XL)

**Goal.** Multi-head per `architecture.md` §3: cell-role+scope · caption-association. Most expensive labels — last in the chain.

- Trains only on Structure's `table_region`-positive spans (aggregated within pdfplumber `TableCandidate` regions) — the same cost-asymmetry payoff in §7.5, with the gate now provided by Structure's binary head rather than a standalone detector.
- Uses Structure top-k as a soft hint per the 3-edge feature DAG in §3.1.
- Validation: cell-role+scope head macro-F1 ≥ 0.85; caption-association exact-match ≥ 0.80.

#### Phase 3f — BERT-ImageSpecialist (M) — *added 2026-05-04*

**Goal.** Mirror of TableSpecialist for images. Triggered when Structure emits `is_image_block=1` (a new binary head added to BERT-Structure during 3f scaffolding). Owns figure-caption emission + alt-text generation cues.

- **Why it's a specialist, not a Structure head.** When evaluating Structure v1, the `figure_caption` class fell out as the weakest — 0.42 precision / 0.86 recall on 285 test samples. The model over-fires on small italic captions that shape-resemble paragraphs, and stuffing the caption signal into a 7-class softmax forces it to compete against "paragraph" for probability mass. The fix is to detect "is image block" with a Structure binary gate (cleaner signal — image-block layout is strongly distinctive) and let an ImageSpecialist parse caption text + alt-text cues only on confirmed positives. Same architectural pattern as `is_heading→HeadingSpecialist` and `table_region→TableSpecialist`.
- Training input: confirmed image region + neighboring text block features + Structure top-k as soft hint.
- Heads:
  - `caption_role` — `{figure_caption, image_alt_text, decorative}` (3-class)
  - `caption_position` — `{above, below, beside_left, beside_right, none}` (5-class)
  - `is_alt_candidate` — binary (does the caption text describe the image well enough to seed alt-text?)
- Validation target: `caption_role` macro-F1 ≥ 0.85 on held-out arXiv + Wikipedia figure pairs.

**Phase 3 deferred BERT.** **BERT-MathDetector** is *deferred* per architecture §7.1: revisit only if MathSpecialist on the geometric candidate stream shows false-positives on italicized prose. Track this metric explicitly during the Phase-2 validation run.

**Time / risk for the whole of Phase 3.** ~3–4 months total (3d retirement on 2026-05-05 saved ~3 weeks; that budget reallocated to 3f scaffolding rather than shrinking the chain). The largest single subphase is 3e (TableSpecialist) at ~5–6 weeks because of the cell-level annotation cost; second-largest is 3b (Structure) because of the v1 replacement risk plus the `table_region` and `is_image_block` head additions; 3f (ImageSpecialist) is ~3–4 weeks.

---

### Phase 4 — Cross-BERT reranker (Complexity: **L**)

**Goal.** A single learned model that consumes the typed-signal output of all BERTs that have been built and produces the routing decision plus per-region structure call. Resolves disagreement (the architecture §3 example: Structure says heading@0.6, Semantic says abstract@0.7).

**Inputs.**
- `CouncilState` from running each BERT.
- Neighbor context (per-block typed signals from k=3 neighbors).
- Architecture §3.2 hard arbitration rules (math wins matrix; detectors gate specialists; ambiguous routing → prose) — encoded as **post-processing constraints** on the reranker's output, not learned.

**Outputs.**
- `dart_semantic/council/cross_reranker.py` — small DeBERTa-v3-small or ModernBERT-base + classification head over packed signals.
- `data/build_cross_reranker_data.py` — labels are the *gold final routing* per region (table-track, math-track, prose-track) from ar5iv ground truth.
- `train_cross_reranker.py`.
- `models/council/cross_reranker/final/`.

**Hard prerequisites.** At least Structure + Semantic + MathSpecialist must be live (3 BERTs minimum; 4+ ideal once ImageSpecialist or TableSpecialist lands). Structure's `table_region` head supplies the table-vs-flat-text gating signal that the retired BERT-TableDetector (Phase 3d) was originally scoped to provide.

**Deliverables checklist.**
- [ ] Reranker accepts `CouncilState` and emits `Routing(region_id, track, structure_role, confidence)`.
- [ ] Hard arbitration applied as a post-processing layer that overrides the learned head.
- [ ] Calibration check: predicted confidence vs. empirical accuracy on held-out — Brier score < 0.15.

**Validation criteria.**
- Routing accuracy on held-out ≥ 0.92 (3-class table/math/prose).
- Structure-role accuracy on held-out ≥ Structure head's stand-alone accuracy + 2 percentage points (the lift the reranker is supposed to provide).

**Decision points.**
- **DP-4.1: One reranker, multiple readers, or per-track rerankers?** Architecture §10 says one cross-BERT reranker. Stick.

**Time / risk.** ~3 weeks.

---

### Phase 5 — Qwen specialist scaffolding (prose-first) (Complexity: **L**)

**Goal.** Stand up the Qwen specialist runtime with the **prose** adapter alone. Repurpose what's reusable from `train_reasoner.py` and `dart_semantic/reason.py`.

**Inputs.**
- `train_reasoner.py` — the Qwen 4B + LoRA training scaffold, 4-bit NF4, completion-only loss. **Re-use 80% for training only.** What changes: input format and target format. **Training stays on HF/PEFT**; runtime moves to llama.cpp (see below).
- `dart_semantic/reason.py` — the Qwen runtime loader, currently transformers + bitsandbytes 4-bit. **Replaced** by a llama.cpp-based runtime per architecture §4.1. The transformers loader becomes the v1 fallback; v2 specialists load via `llama-cpp-python` against GGUF files.
- `qwen_specialists/` skeleton from Phase 0.

**Outputs.**
- `dart_semantic/qwen_specialists/prose.py` — adapter-specific input prompt + output parser. Output dialect = HTML5 (architecture §4).
- `data/build_prose_specialist_data.py` — input: `(text, surrounding-block flags, Structure+Semantic top-k typed signals)`; target: `(WCAG-conformant HTML fragment for the region)` from ar5iv / OpenStax ground truth. Per architecture §4.1, this is the prose adapter only; do not include table/math regions in this dataset.
- `train_prose_specialist.py` — clone of `train_reasoner.py` with the new dataset + new max-length (regions are smaller than chunks; 1024 tokens likely sufficient).
- `models/qwen_specialists/prose/final/` — LoRA adapter only.
- `dart_semantic/qwen_specialists/runtime.py` — `generate_candidates(region, adapter, k=N) -> list[Candidate]`. Sampling diversity via temperature/top-p. **Strict serial enforcement** of adapter swap.

**Hard prerequisites.** Phase 4 (Cross-BERT reranker — its routing decisions are how regions get assigned to "prose-track" so the prose specialist knows what to operate on).

**Deliverables checklist.**
- [ ] Prose adapter trains, saves, loads.
- [ ] `runtime.py` raises if a second adapter swap is attempted while one is loaded — enforces architecture §4.1's serial discipline. (Maps to existing `feedback_qwen_build_serial.md`.)
- [ ] Adapter swap measured: < 5 s under whichever llama.cpp loading strategy is picked (pre-merge GGUF per specialist, OR base GGUF + hot LoRA swap). If slower, profile and fix before Phase 6.
- [ ] Trained adapter exported to GGUF (or kept as PEFT LoRA file with a base GGUF) and loaded by `llama-cpp-python` end-to-end on `eval/side_by_side/math_heavy_short_v7/input.pdf`.
- [ ] K-candidate generation: diverse — measured by mean pairwise edit-distance ≥ X% across K=4 candidates per region (X chosen from initial calibration; ≥ 15% is a reasonable floor).

**Validation criteria.**
- On a held-out 50-prose-region set: ≥ 90% of generated top-1 candidates pass `axe-core` standalone (no full document yet).
- HTML fragment validity (html5validator) ≥ 95%.
- Text preservation (token-level Jaccard against source block) ≥ 0.95 for ≥ 90% of candidates.
- VRAM: peak ≤ 7.0 GB during generation (1 GB headroom from the 8 GB ceiling).

**Decision points.**
- **DP-5.1: Same Qwen 3 4B base or step up?** Architecture targets Qwen 3 4B given VRAM. Stick.
- **DP-5.2: K candidates per region — what K?** Architecture says "K candidates per region" without naming K. **Recommend K=4** for fast lane, K=8 for the offline-Qwen lane (Stage 13 exit 3). Tune empirically; gate on the gate's pass-through rate.
- **DP-5.3: llama.cpp loading strategy.** **(a) Pre-merge GGUF per specialist** — 4 GGUFs at ~2.5 GB each (~10 GB on disk), simple inference (load → generate → swap), no LoRA file management at runtime. **(b) Base GGUF + hot LoRA swap** — one base GGUF + 4 small LoRA files, smaller disk, slightly more orchestration. Decide based on measured swap latency and VRAM headroom on the 3060 in Phase 5. **Provisional default: pre-merge** (simpler, disk is cheap post-cleanup). Re-evaluate if pre-merge swap latency exceeds 5 s.

**Time / risk.** ~3 weeks.

---

### Phase 6 — Remaining Qwen specialists (Complexity: **L**, in three sub-phases)

**Sub-phases run in serial — adapter ordering matters because each one inherits training-data architecture from the previous.**

#### Phase 6a — Table specialist (M)

- Output dialect: HTML5 tables with `<thead>`/`<tbody>`/`<th scope>`/`<caption>` (architecture §4 dialect specification). **Compliance with `docs/ontology.md` §2.5 is a target, including the H43 `headers`/`id` pattern for complex tables — currently a `docs/ontology.md` §7 blocker.** Closing this gap is one of the explicit ROI items of the new architecture.
- Training input: confirmed table region + cell-grid + Structure top-k as soft hint + TableSpecialist BERT cell-role+scope output.
- Validation: ≥ 90% of generated tables pass axe + html5validator + text-preservation; complex tables (≥2 header rows) emit valid `headers`/`id` associations in ≥ 80% of cases.

#### Phase 6b — Math specialist (M)

- Output dialect: MathML 4.0 with `alttext` (architecture §4 + ontology §2.10 + §7 top gap #1). This closes the second `docs/ontology.md` §7 blocker.
- Training input: confirmed math region + glyph features + MathSpecialist BERT math-type output.
- Validation: MathML validity ≥ 95% (MathML validator); generated `alttext` non-empty for ≥ 99% of fragments; semantic equivalence to source ar5iv `<math>` ≥ 0.80 by tree-edit distance.

#### Phase 6c — Gap-fill specialist (M)

- Output dialect: HTML5 fragment per slot (architecture §4). Scope **deliberately narrow** per §4: missing-title inference, citation/footnote resolution, author/copyright/legal block remediation. **Not** image alt-text in v1; **not** a generative document assembler.
- Training input: a single `GapSlot` (kind, surrounding context, originating reason).
- Validation: per-slot-kind precision/recall measured separately. Title-inference accuracy ≥ 0.85 against ar5iv title ground truth; citation-resolution recall ≥ 0.90 on synthetic doc with known cross-references.

**Phase 6 hard prerequisites.** Phase 5 (prose specialist works end-to-end). Adapter-swap discipline already proven.

**Deliverables across 6a–6c.** Three new adapters under `models/qwen_specialists/`. Three new training scripts. Three new dataset builders. One updated `runtime.py` that knows about all four adapters (prose, table, math, gap-fill). Updated `qwen_specialists/config.yaml` with per-adapter sampling parameters (temperature varies by domain).

**Time / risk.** ~6–8 weeks. Math and table dataset construction are the long poles.

---

### Phase 7 — Per-region hard gate (Complexity: **M**)

**Goal.** Implement the eliminating-only check at Stage 7 of the target pipeline. Some logic exists in `dart_semantic/validate.py` (`HtmlValidator` over full HTML); the per-region variant runs the same axe-core ruleset on a fragment, plus html5validator and text-preservation.

**Inputs.**
- `dart_semantic/validate.py` — already has the `axe-playwright-python` runner. **Reuse the `HtmlValidator` class as-is.**
- `gates/hard_region.py` — Phase-0 stub.
- A list of `Candidate` HTML fragments from the Qwen specialists.

**Outputs.**
- `dart_semantic/gates/hard_region.py` — concrete:
  - `check_axe_pass(fragment) -> bool` (wraps existing validator on the fragment in a minimal HTML shell — the shell is part of the contract).
  - `check_html5_valid(fragment) -> bool` (subprocess to `html5validator`).
  - `check_text_preservation(fragment, source_text, tau) -> bool` (Jaccard or normalized edit distance from `text_utils.jaccard_overlap`).
  - `check_mathml_valid(fragment) -> bool` (only on math-track fragments).
  - `check_region(candidate, source) -> GateResult` (combines all four).
- `dart_semantic/text_utils.py` — extend with `text_preservation_score`.
- `pyproject.toml` — add `html5validator` dependency (Apache-2 → safe).

**Hard prerequisites.** Phase 5 (need real candidates to gate).

**Deliverables checklist.**
- [ ] All four checks implemented and unit-tested with both passing and failing fragments.
- [ ] Eliminating semantics: any single fail → drop. No soft scoring. (Architecture §5.1.)
- [ ] Threshold τ for text-preservation is configurable in `gates/config.yaml`.

**Validation criteria.**
- 100% of fragments containing axe-critical/serious violations are dropped.
- 0% of fragments that *do* pass axe-critical/serious are dropped (precision = 1).
- Latency per region ≤ 200 ms on the dev machine.

**Decision points.**
- **DP-7.1: Choose τ.** Architecture §5 says "tune τ on a held-out set before any threshold ever ships." Concrete recommendation: pick the value of τ in `[0.7, 0.9]` that maximizes downstream end-to-end document-level pass rate on a 30-doc held-out arXiv set. Re-tune per phase as the upstream candidate distribution shifts.

**Time / risk.** ~2 weeks.

---

### Phase 8 — Per-region soft reranker (Complexity: **L**)

**Goal.** Among gate-survivors, score fit-quality with a learned cross-encoder; pick top-1 per region. Architecture §5 lists the score axes: heading-hierarchy fit · table-semantics richness · diff-from-source · neighbor-consistency · ARIA restraint.

**Outputs.**
- `dart_semantic/gates/soft_region.py` — model loader + `rerank(survivors, region) -> Candidate`.
- `data/build_soft_reranker_data.py` — labels are the gold candidate among multiple gate-passing alternatives. For training, generate K candidates per region (using the prose/table/math specialists), gate-filter, then label by axe + html5 success on the full assembled doc.
- `train_soft_reranker.py` — small cross-encoder DeBERTa-v3-small.
- `models/gates/soft_region/final/`.

**Hard prerequisites.** Phase 7.

**Deliverables checklist.**
- [ ] Reranker outputs a calibrated score for each survivor.
- [ ] Top-1 chosen by score; ties broken by smaller diff-from-source.
- [ ] No mixing of eliminating-vs-fit signals: the gate has already eliminated; the reranker only ranks fit. Test: feed the reranker an axe-failing candidate and a passing one — the reranker should not be doing this work, the gate should have dropped the failing one upstream. Add an assertion in production to this effect.

**Validation criteria.**
- Top-1 selection matches ground-truth gold candidate ≥ 0.75 of the time on held-out.
- End-to-end document axe pass rate is ≥ that achieved by random selection from gate-survivors + 5 percentage points (the lift the reranker provides).

**Time / risk.** ~3 weeks.

---

### Phase 9 — Deterministic document assembler (Complexity: **L**)

**Goal.** Implement Stage 9 of the target. This is the *deterministic* document-level assembler that owns:

- heading hierarchy normalization — algorithm: **"promote-the-first, demote-forward, never-skip"** (force-claim H1 from Semantic.title once; demote-to-fit, never promote)
- ARIA landmarks emitted by assembler: `<main>` always, `<nav>` on TOC-pattern detection, `<aside class="legal|copyright">` for legal/copyright doc-role, `<address>` for author. **Body-scope `<header>` and `<footer>` are NOT emitted — page template owns them per `docs/ontology.md` §7.** Semantic's `footer` doc-role drops to artifact (same handling as `metadata`).
- list-continuation — 4-clause merge rule (kind + marker + adjacency + no-heading-interruption); figures tolerated as interruption, headings hard-block
- reference resolution — 3-pass (build index → regex match → resolve/leave/flag); flag GapSlot only on near-miss
- doc shell (doctype, `<html lang>` from langdetect on body, `<head><title>`, skip-link to `#main-content`)
- final reading-order DOM placement
- gap detection — 5 supported kinds only (`missing_title`, `citation_unresolved`, `author_block`, `copyright_block`, `legal_disclaimer`); unsupported slots use deterministic fallbacks
- gap-fill provenance — every gap-fill-emitted span MUST be tagged with `kind` and `source_anchor` (required by Phase 11 theta hallucinated-structure scoring)
- Phase 9b/9c — gap-fill outputs go through the per-region hard gate before merge-back; merge-back is **one-shot** (re-run heading + reference resolution once if title or refs changed); never iterate

> **Refined per `Plans/04_assembler_layer_investigation.md`** — read that investigation alongside this section for concrete algorithms, gap-detection table with per-kind context shapes, splice semantics per slot kind, and the all-K-fail fallback ladder. The investigation is the design-level supplement; this plan section is the build checklist.

This module **replaces** `dart_semantic/ontology_map.py` plus most of `dart_semantic/hierarchy.py`. Both are kept as v1 fall-through — they're not deleted in this phase, only superseded.

**Inputs.**
- `ontology_map.emit_html` — current rules and shell scaffolding. Reuse 50%.
- `hierarchy.resolve_hierarchy` — current font-stack heading depth + list nesting. Reuse 80%.
- `docs/ontology.md` §2 (governing standards) and §7 (gap analysis to address).
- The selected top-1 per region (Phase 8 output).

**Outputs.**
- `dart_semantic/assembler/builder.py` — main `assemble(top1_regions, council_state) -> AssembledDoc | list[GapSlot]`.
- `dart_semantic/assembler/heading_tree.py` — heading hierarchy normalization.
- `dart_semantic/assembler/landmarks.py` — ARIA landmark wiring (`<main>`, `<nav>`, `<aside>`, `<footer>`).
- `dart_semantic/assembler/references.py` — cross-reference resolution.
- `dart_semantic/assembler/lists.py` — list-continuation.
- `dart_semantic/assembler/shell.py` — doctype + `<html lang>` + `<title>` + skip-link.
- `dart_semantic/assembler/gaps.py` — `GapSlot` detection: enumerate the slot kinds the gap-fill specialist handles (architecture §4 narrow scope: missing-title, citation/footnote resolution, author/copyright/legal). Emit `GapSlot(kind, context)` instead of failing.

**Hard prerequisites.** Phase 8 (top-1 per region selected).

**Deliverables checklist.**
- [ ] All seven sub-modules implemented.
- [ ] Round-trip test: assembler over the same inputs produces deterministic byte-equal HTML.
- [ ] Gap-detection produces `GapSlot` items only for the **five** supported kinds (`missing_title`, `citation_unresolved`, `author_block`, `copyright_block`, `legal_disclaimer`); other missing slots use deterministic fallbacks (e.g., empty `<title>` → "Untitled document"; missing `lang` → langdetect; ambiguous reference → leave plain text).
- [ ] Gap-fill outputs pass the per-region hard gate (axe + html5validator + text-preservation) before merge-back; gate failures fall through to deterministic per-kind fallback.
- [ ] Merge-back is one-shot (one re-run of heading + reference normalization if title or refs changed); the assembler does NOT iterate.
- [ ] Gap-fill provenance: every span emitted by gap-fill carries `kind` and `source_anchor` for Phase 11 theta consumption.
- [ ] `AssembledDoc` populated per `Plans/04_assembler_layer_investigation.md` §10.4 (gaps_found/gaps_resolved/gaps_fallback, heading_tree, landmarks, anchors, region_provenance, sub_task_log).

**Validation criteria.**
- On the 4 held-out arXiv papers, the new assembler produces HTML that passes axe-core's full WCAG 2.2 AA ruleset with zero violations — same bar as v1 hits.
- Heading hierarchy: zero skipped levels.
- Landmarks: `<main>` always present; `<nav>` present when Semantic emitted any TOC region.
- Reference resolution: ≥ 80% of `(Section X.Y)` patterns in source produce valid `<a href="#sec-X-Y">`.

**Time / risk.** ~3 weeks. Risk: the v1 ontology_map.py is non-trivial; mistakes in the rewrite are easy and would silently break compliance. Heavy use of golden tests against v1 output.

---

### Phase 10 — Document-level gates + soft reranker + exits (Complexity: **L**)

**Goal.** Implement target Stages 10, 11, and 13. (Stage 12 is theta — its own Phase 11.)

> **Refined per `Plans/04_assembler_layer_investigation.md` §5–§7.** This phase implements concrete formulas, not pseudocode-as-design. Read that investigation alongside this section.

**10.1 Document-level hard gate (Stage 10).** Eliminating-only. Checks:
- axe-core full doc — zero `serious` or `critical` (warnings allowed).
- html5validator — zero errors (warnings allowed).
- heading hierarchy validity — no skipped levels per `emit_html.py` policy.
- `<html lang>` declared and non-empty.
- `<title>` present and non-empty.
- `<main>` landmark present (always).
- Conditional landmarks: `<nav>` if Semantic predicted any TOC region; `<footer>` if Semantic predicted any `legal`/`copyright` block (else not required).

**10.2 Document-level soft reranker (Stage 11).** **v1 scope is fast-lane vs. offline-lane only** — at most 2 candidate assemblies. Per-region top-2 fan-out and gap-fill K^M cross-product are combinatorial traps and are deferred to v2. Rule-based composite (DP-10.1):
```
final = 0.30 · heading_tree_balance
      + 0.20 · landmark_coverage
      + 0.30 · ref_link_integrity
      + 0.20 · outline_cleanliness
```
Tie-break: smaller diff-from-source. **Stage 11 must emit per-axis scores** so Phase 11 (theta) can reuse them without recomputing.

**10.3 Exits (Stage 13).** Four exit actions per the updated decision matrix in `architecture.md` §7. **No scalar confidence number in v1** — `theta_score` is the only quality number on the wire; sidecar JSON carries `gate_results.passed` boolean, not a heuristic confidence float.
- **ship-with-confidence** — gate passed, theta ≥ 0.80, no floor breach.
- **ship-with-flag** — gate passed, theta 0.70–0.80 OR floor breach. Specific flag attached.
- **offline-Qwen lane** — fast-lane gate failed OR fast-lane theta < 0.70. **Same Qwen 3 4B + K=8 + temp 0.9 + top-p 0.95** (NOT a model upgrade — Qwen 3 7B 4-bit OOMs with axe-core's Chromium concurrent on 8 GB; **also no API-provider escalation in v1**, offline-local is the only retry mechanism). Re-runs Stages 6–9 over only the regions whose fast-lane gate failed; council and cross-BERT reranker do NOT re-run.
- **non-certified stamp** — gate failed both lanes. Both machine-readable AND human-visible. Machine: `<meta name="dart-certification-status" content="not-certified">` plus sidecar JSON `{document_id, exit, failed_checks}`. Theta omitted (`theta_score: null`). Visible: `<aside role="note" class="dart-uncertified-banner">` as first child of `<main>` with the stamp text from `architecture.md` §7.4.

**Inputs.** All prior phases.

**Outputs.**
- `dart_semantic/gates/hard_document.py` — concrete.
- `dart_semantic/gates/soft_document.py` — rule-based scorer first (DP-10.1); promote to learned (DeBERTa-v3-small cross-encoder) only on measurement.
- `dart_semantic/exits.py` — `decide_exit(state) -> ExitDecision` with the four-action enum (matches `assembler/types.py::CertificationStatus` + `ConfidenceAction`).
- `dart_semantic/qwen_specialists/offline_lane.py` — re-runs Stages 6–9 over fast-lane-failed regions with K=8/temp=0.9; one re-entry only.
- `dart_semantic/non_certified_stamp.py` — both meta tag AND visible `<aside role="note">` injected.

**Hard prerequisites.** Phases 7–9.

**Deliverables checklist.**
- [ ] All four exit actions implemented and unit-tested.
- [ ] Offline-Qwen lane is re-entrant from a failed fast lane without re-running upstream BERTs (BERT outputs cached on document state).
- [ ] Non-certified stamp is BOTH visually unambiguous (visible aside) AND machine-discoverable (meta tag + sidecar JSON).
- [ ] Stage 11 emits per-axis scores in `AssembledDoc.soft_axes` for Phase 11 to consume.
- [ ] **`escalate.py`'s tri-verdict semantics now match the formal exits.** Old `escalate.py` is deprecated but kept until v1 is fully retired.

**Validation criteria.**
- On 30 held-out arXiv papers, 4 held-out OpenStax chapters, 4 held-out IRS forms:
  - ≥ 80% take **ship-with-confidence** OR **ship-with-flag** combined.
  - ≤ 5% take the **non-certified stamp** exit (after the offline-Qwen lane has run).
  - The remaining 15–20% take the offline lane and pass on retry.
- All ship-with-confidence outputs pass full-document axe wcag22aa with zero critical/serious violations.

**Decision points.**
- **DP-10.1: Soft doc reranker — learned or rule-based?** Start rule-based with the formula above. Promote to DeBERTa-v3-small cross-encoder only if rule-based shows poor calibration (>20% inversion vs. final axe outcome on a 30-doc held-out set).

**Time / risk.** ~3 weeks.

---

### Phase 11 — Theta evaluator (Complexity: **M**)

**Goal.** Implement Stage 12 — the post-WCAG meaning-preservation score per `architecture.md` §6 and `Plans/03_theta_investigation.md`. Theta is **not** a gate; it does not override WCAG. It runs on every WCAG-passing doc (both lanes), produces per-dimension scores plus a composite, and emits flags / retry decisions.

**Inputs.**
- Phase 9 output (assembled doc with gap-fill provenance — `gap_fill` spans tagged with `kind` and `source_anchor`).
- Phase 10 output — `AssembledDoc.soft_axes` (Stage 11 per-axis scores: heading_tree_balance, landmark_coverage, ref_link_integrity, outline_cleanliness).
- Source PDF text per top-level section (for the learned `semantic_preservation` scorer).
- `theta/types.py` and `theta/config.yaml` (Phase 0).

**Outputs.**
- `dart_semantic/theta/dimensions/semantic_preservation.py` — DeBERTa-v3-small cross-encoder, single regression head, section-level scoring.
- `dart_semantic/theta/dimensions/{structural_coherence,navigation_clarity,context_continuity,reference_integrity,cognitive_load,fragmentation,hallucinated_structure}.py` — deterministic scorers per `Plans/03_theta_investigation.md` §2.
- `dart_semantic/theta/composite.py` — weighted aggregator with per-dimension floors and flag emission.
- `data/build_theta_semantic_data.py` — synthetic perturbations of WCAG-clean ar5iv pairs (70%) + offline-lane bootstrap pairs (20%) + eval-signal proxy (10%).
- `train_theta_semantic.py` — fine-tune the DeBERTa-v3-small cross-encoder.
- `models/theta/semantic_preservation/final/`.
- Integration into `dart_semantic/exits.py` — `ConfidenceAction` is now driven by theta thresholds + per-dimension floors.

**Hard prerequisites.**
- Phase 9 (gap-fill provenance fields populated on `AssembledDoc`).
- Phase 10 (Stage 11 emits per-axis scores).
- Phase 0 (theta types + config landed).

**Deliverables checklist.**
- [ ] All 8 dimensions implemented; 7 deterministic, 1 learned.
- [ ] `ThetaReport` JSON-serializable; round-trip test passes.
- [ ] Per-dimension floors trigger their named flags (`broken_refs_present`, `gap_fill_review_recommended`, `meaning_preservation_low`, `cognitive_load_high`).
- [ ] Retry policy implemented: theta < 0.70 fast-lane → one offline retry → ship higher-theta output. Capped at one retry (no looping).
- [ ] `decide_exit` consumes `ThetaReport` and emits `ConfidenceAction` per the matrix in `architecture.md` §7.
- [ ] Theta is **omitted** on `non_certified_stamp` exit (`theta_score: null, wcag_status: "failed"`).
- [ ] Conformance statement template in `docs/ontology.md` §6 is **NOT** modified to include theta.
- [ ] `theta_version` field pins weights + thresholds + scorer-model versions; locked at release.

**Validation criteria.**
- Held-out semantic-preservation correlation against synthetic perturbation severity ≥ 0.70 Pearson.
- Per-dimension floor false-positive rate ≤ 5% on the 30-doc clean held-out set (i.e., a doc that passes all gates and has no real defects should not breach floors).
- On the same 30-doc held-out set: theta retry fires on ≤ 10% of fast-lane-passing docs (most should be high-theta on first pass).
- Theta evaluation latency < 1 second per doc on the dev GPU (DeBERTa-v3-small cross-encoder section-level inference).
- Theta does **not** penalize correct WCAG remediations: hand-curated 10-doc panel where each doc has a known correctness/fragmentation tradeoff (sidebar-flatten, multi-column-split, MathML-emit) — theta must not mark these as low.

**Decision points.**
- **DP-11.1.** Cross-encoder family for semantic preservation: DeBERTa-v3-small (≈140 MB, separate small model) vs. ModernBERT-base (≈600 MB, shared with council). Recommend **DeBERTa-v3-small** — separate small model avoids competing for council backbone slots.
- **DP-11.2.** Theta weighting strategy: hand-tuned (`Plans/03_theta_investigation.md` §2.9) vs. fit to held-out. Start hand-tuned; promote to fit-on-held-out only on uncalibration evidence.
- **DP-11.3.** Per-document-type thresholds vs. global threshold. **Start global.** Per-doc-type (academic vs. legal vs. educational) is v2-of-theta.

**Time / risk.** ~3 weeks. Breakdown: deterministic dimensions ~1 week, semantic-preservation training + calibration ~1.5 weeks, composite + integration ~3 days, cognitive-load calibration on OpenStax ~2 days.

---

## 2. Critical files map

### 2.1 Existing files — what becomes of each

| File | Disposition | Detail |
|---|---|---|
| `dart_semantic/extract.py` | **Keep, unchanged** | Stage 1 is unchanged in target architecture. |
| `dart_semantic/extract_shared.py` | **Keep, unchanged** | Stage 1 core. Phase 1 reads from it; does not modify. |
| `dart_semantic/features.py` | **Extend** | Phase 1 adds `featurize_with_regions`. v1 entry points stay. |
| `dart_semantic/region_detection.py` | **Extend** | Phase 1 wraps `detect_*_regions` with `RegionCandidate` typed output. |
| `dart_semantic/types.py` | **Extend** | Add `FeatureSet`, `RegionCandidate`, `TableCandidate`, `MathCandidate`. v1 dataclasses stay. |
| `dart_semantic/classify.py` | **Phase out** | v1 is the sole consumer until Phase 3b's BERT-Structure ships. After Structure passes shadow, v1 path stays only behind `pipeline_v2.run(mode="v1")`. Delete only when target arch is fully ratified on eval. |
| `dart_semantic/reason.py` | **Replaced by a llama.cpp runtime in qwen_specialists** | The transformers + bnb-NF4 loader is v1; v2 specialists load via `llama-cpp-python` against GGUF (per architecture §4.1). The chunk-based per-block-label use case becomes obsolete; replaced by per-region candidate generation. Keep `reason.py` until v1 is retired. |
| `dart_semantic/reason_chunking.py` | **Delete after Phase 5** | Region-based generation supersedes page-chunked labeling. |
| `dart_semantic/reason_schema.py` | **Repurpose then deprecate** | The locked short-form serialization is not used by region-based specialists. Some helpers (label codes) may still be needed in offline data builders; isolate, then drop. |
| `dart_semantic/hierarchy.py` | **Replace** | Phase 9 assembler subsumes its logic. v1 fallback only. |
| `dart_semantic/ontology_map.py` | **Replace** | Phase 9 assembler is the canonical emitter. v1 fallback only. |
| `dart_semantic/enrich.py` | **Replace** | Phase 9 assembler + Phase 6c gap-fill cover the same surface. v1 fallback only. |
| `dart_semantic/validate.py` | **Extend** | `HtmlValidator` is reused by Phase 7 (per-region) and Phase 10 (per-document). The class itself does not change. |
| `dart_semantic/escalate.py` | **Replace** | Phase 10's `exits.py` is the new formalization. Keep until v1 is retired. |
| `dart_semantic/pipeline.py` | **Keep as v1 path** | Phase 0 introduces `pipeline_v2.py`; v1 path stays callable indefinitely so the four-arXiv eval keeps working. |
| `dart_semantic/ir.py`, `dart_semantic/emit_html.py` | **Keep until ingester rewrite** | Used by `scripts/pair_from_*.py` for ground-truth HTML emission during data generation (per `docs/bloat_audit.md` §4.1). Don't touch in this work. |
| `train_classifier.py` | **Phase out after Phase 3b** | Replaced by `train_structure.py` (Phase 3b). Delete after target arch is ratified. |
| `train_reasoner.py` | **Repurpose to per-specialist trainers** | `train_prose_specialist.py` (Phase 5), `train_table_specialist.py` (Phase 6a), etc., all clone its skeleton. Delete the original after Phase 6 is shipped. |
| `dart_semantic/glm_ocr*.py` | **Keep, audit later** | Out of scope for this plan; deferred to a later cleanup pass. |
| `data/build_classifier_data_v2.py` | **Extend** | Becomes the basis for `data/build_structure_data.py` (Phase 3b). The label-extraction side is reusable; the heading-level + list-nesting heads add new labels. |
| `data/build_qwen_data.py` | **Phase out** | Replaced by `data/build_<specialist>_data.py` per Phase 5/6. |

### 2.2 New top-level files (cumulative)

```
dart_semantic/
  council/                    Phase 0 + filled in Phase 2/3
  qwen_specialists/           Phase 0 + filled in Phase 5/6
  gates/                      Phase 0 + filled in Phase 7/8/10
  assembler/                  Phase 0 + filled in Phase 9
  pipeline_v2.py              Phase 0; switches based on config
  v2_config.py                Phase 0
  exits.py                    Phase 10
  non_certified_stamp.py      Phase 10

data/
  build_math_specialist_data.py     Phase 2
  build_<bert_name>_data.py         Phases 3a–3e
  build_cross_reranker_data.py      Phase 4
  build_<specialist>_data.py        Phases 5, 6a–6c
  build_soft_reranker_data.py       Phase 8

train_math_specialist.py            Phase 2
train_<bert_name>.py                Phases 3a–3e
train_cross_reranker.py             Phase 4
train_<specialist>_specialist.py    Phases 5, 6a–6c
train_soft_reranker.py              Phase 8

models/
  council/<bert_name>/final/        Phases 2, 3a–3e
  council/cross_reranker/final/     Phase 4
  qwen_specialists/<name>/final/    Phases 5, 6a–6c
  gates/soft_region/final/          Phase 8
  gates/soft_document/final/        Phase 10 (if learned)
```

---

## 3. Risks and open questions

### 3.1 Known unknowns

- **Q1. Adapter swap latency under llama.cpp.** The architecture's serial `build_qwen` discipline assumes sub-5-second swaps. With llama.cpp's pre-merge strategy this is "load a fresh GGUF" cost; with hot LoRA swap it's "attach a LoRA file" cost. Either way, batching all regions for one adapter pass before swapping is critical. Measure both options in Phase 5 and lock DP-5.3.
- **Q2. Council base-encoder VRAM under multi-LoRA-resident training.** Phase 0 picks ModernBERT-base. If fine-tuning with multiple adapter copies in memory pushes past 8 GB, fall back to DeBERTa-v3-small. Measure by Phase 2 end.
- **Q3. ar5iv coverage of available arXiv pairs.** The plan assumes most of the 8K-pair target has ar5iv ground truth. If coverage is < 60%, MathSpecialist and Math-related labels are constrained. Measure during Phase 2 data build; document `data_coverage_report.json`.
- **Q4. TableSpecialist label cost.** Cell-level annotations across academic + educational + regulatory + form-style tables are the largest data ask in the project. If cost is prohibitive, fall back to a reduced label set (binary header/data + scope only) and accept ontology §7 H43 gap as partially closed instead of fully closed.
- **Q5. Gate threshold τ stability.** Architecture §5 calls out τ as a product decision tuned on a held-out set. Phase 7 establishes the first τ. Each subsequent phase that changes the upstream candidate distribution may invalidate τ. Plan for τ re-tuning at Phases 8, 9, and 10.
- **Q6. Cross-reranker — does it actually lift?** Phase 4 only ships if it shows ≥ 2-percentage-point lift over per-BERT argmax. If lift is null, it can be dropped — architecture says "one cross-BERT reranker", but that's a structural choice, not a quality guarantee.
- **Q7. Offline-Qwen lane base model.** The architecture says "larger local Qwen". Practical options on 8 GB **with llama.cpp**: Qwen 3 4B Q4_K_M with K=8 (no model upgrade — current default), Qwen 3 7B Q4_K_M (potentially fits with llama.cpp's better quantization headroom — needs measurement; transformers/bnb-NF4 OOMs but llama.cpp Q4_K_M may not), Qwen 3 14B Q4_K_M (definitely won't fit on 8 GB). **Recommended initial implementation: same Qwen 3 4B + larger K + looser sampling temperature.** Measure 7B Q4_K_M as a Phase-10 follow-up if 80% ship-with-confidence target isn't hit on 4B.
- **Q8. Are the four held-out arXiv papers a sufficient v1 regression bar?** Probably not for the target architecture. Add OpenStax + IRS + arXiv broader eval per architecture §7.4 corpus mix, ideally 30 documents minimum, before declaring target arch ready to retire v1.

### 3.2 Decision points consolidated (and recommended defaults)

| ID | Decision | Default | Phase |
|---|---|---|---|
| DP-0.1 | Council base encoder | ModernBERT-base (locked Phase 2) + DeBERTa-v3-small (cross-encoders/theta); DeBERTa-v3-base fallback | Phase 0 |
| DP-0.2 | Config mechanism | YAML | Phase 0 |
| DP-1.1 | Region candidate over-firing | Aggressive | Phase 1 |
| DP-1.2 | Table cell-grid forwarded | Yes | Phase 1 |
| DP-2.1 | First BERT | MathSpecialist | Phase 2 |
| DP-2.2 | LoRA rank | r=16, alpha=32 | Phase 2 |
| DP-2.3 | Math-type label set | Keep matrix label | Phase 2 |
| DP-4.1 | Reranker count | One cross-BERT reranker | Phase 4 |
| DP-5.1 | Qwen base | Qwen 3 4B | Phase 5 |
| DP-5.2 | K candidates per region | K=4 fast / K=8 offline | Phase 5 |
| DP-5.3 | llama.cpp loading strategy | Pre-merge GGUF per specialist (provisional; re-eval on swap latency) | Phase 5 |
| DP-7.1 | Gate τ choice | Tune on held-out | Phase 7 |
| DP-10.1 | Soft doc reranker | Rule-based first | Phase 10 |
| DP-11.1 | Theta semantic-preservation cross-encoder | DeBERTa-v3-small | Phase 11 |
| DP-11.2 | Theta weighting strategy | Hand-tuned (promote on uncalibration evidence) | Phase 11 |
| DP-11.3 | Theta thresholds per doc-type vs. global | Global | Phase 11 |

---

## 4. Phase 0 specifics — what to verify on landing

### 4.1 What Phase 0 produces

A package skeleton, type definitions, and a parallel orchestrator (`pipeline_v2.py`). Zero new model weights. Zero new model dependencies. A single new test file group. Documentation updates in `dart_semantic/__init__.py` to acknowledge the parallel target architecture.

### 4.2 How to verify Phase 0 landed without breaking v1

Concrete acceptance test:

1. **v1 byte-for-byte regression.** With Phase-0 changes merged:
   ```
   for pdf in $(ls data/test_arxiv/*.pdf); do
     python scripts/infer_pdf.py "$pdf" --save /tmp/v1_after.json
     diff /tmp/v1_pre.json /tmp/v1_after.json
   done
   ```
   All four held-out arXiv outputs must be byte-identical against the pre-Phase-0 snapshot.

2. **pipeline_v2 v1-mode equivalence.**
   ```
   python -c "from dart_semantic.pipeline_v2 import run; print(run('data/test_arxiv/x.pdf', mode='v1').html)"
   ```
   produces the same HTML as `pipeline.run_pipeline`.

3. **Imports stay clean.**
   ```
   python -c "import dart_semantic.council, dart_semantic.qwen_specialists, dart_semantic.gates, dart_semantic.assembler"
   ```
   succeeds without error and without loading any model.

4. **Adapter-serial test.**
   ```
   pytest tests/test_adapter_serial.py
   ```
   passes — proves the qwen_specialists context manager rejects nested entry.

5. **Type round-trip.**
   ```
   pytest tests/test_council_skeleton.py
   ```
   passes — proves all new dataclasses serialize/deserialize.

If any of the above fail, Phase 0 has not landed correctly. No Phase-1 work begins until all five pass.

### 4.3 What Phase 0 does NOT include

- No model code. The council and specialist modules are skeletons only.
- No new training data. No new training scripts.
- No `pipeline_v2.run(mode="v2")` working end-to-end. Mode `v2` raises `NotImplementedError`. Only `mode="v1"` works.
- No deletion of v1 code. Everything is additive.

### 4.4 Phase 0 ↔ remainder of plan handshake

Phase 0 delivers the *contract* — the typed signal shape, the registry, the orchestrator skeleton — that every subsequent phase plugs into. Subsequent phases should never add a new top-level package; they fill `council/`, `qwen_specialists/`, `gates/`, or `assembler/`. If a phase finds itself wanting to add a fifth top-level package, that is a signal that the architecture has drifted and Phase 0's contract should be revised before continuing.

---

## 5. Cross-cutting concerns

### 5.1 Eval infrastructure

The existing `scripts/eval_v7_family.sh` and `eval/results/v7_family_*.json` are the reasoner-quality eval. New evals are needed:

- `scripts/eval_council.sh` — runs each council BERT on a held-out set and emits per-head F1.
- `scripts/eval_specialists.sh` — runs each Qwen specialist on a held-out region set and emits axe + html5 + text-preservation.
- `scripts/eval_pipeline_v2.sh` — runs `pipeline_v2` end-to-end and emits the document-level pass-rate matrix.

The existing `eval-summary` skill already understands `eval_v7_family` shape; the new evals should follow the same JSON shape so the skill still applies.

### 5.2 v1 retirement criteria

The v1 pipeline retires only when:
1. All seven BERTs in the council are trained (or explicitly deferred per architecture §7.1).
2. All four Qwen specialists are trained.
3. Both gates and rerankers ship.
4. The Stage-9 deterministic assembler produces zero-violation HTML on the v1's four-arXiv held-out set + 30-arXiv broader set + 4 OpenStax + 4 IRS.
5. The exit distribution on the broader set hits the Phase-10 validation targets (80% ship-with-confidence, ≤5% non-certified stamp).

Until all five hold, v1 stays available behind `pipeline_v2.run(mode="v1")` and the four-arXiv test suite continues to gate every commit.

### 5.3 Documentation parallel-track

Each phase that adds a new module also adds the corresponding entry in `docs/ontology.md` §7 if it closes a gap. The math specialist (Phase 6b) closes gap #1 (MathML). The table specialist (Phase 6a) closes gap #2 (complex tables, headers/id). Future inline-`lang=` work (gap #3) is **not** in this plan; it requires source-side parser changes to populate language tags and is deferred.

The `architecture.md` document is the canonical target. Any deviation discovered during execution that requires changing the target requires an explicit revision of `architecture.md` (§10 Locked architectural choices). This plan does not request any such revision.

---

*End of plan.*

---

### Critical Files for Implementation

The single most load-bearing files for executing this plan, in order of how often they will be opened:

- /home/user/Projects/Semantic/architecture.md
- /home/user/Projects/Semantic/docs/ontology.md
- /home/user/Projects/Semantic/dart_semantic/pipeline.py
- /home/user/Projects/Semantic/dart_semantic/types.py
- /home/user/Projects/Semantic/dart_semantic/region_detection.py