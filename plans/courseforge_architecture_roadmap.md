# Courseforge Architecture Roadmap (Phases 1–7)

> Last refreshed: 2026-05-02. Authoritative cross-cutting plan for the
> Courseforge two-pass / LLM-agnostic / dual-chunkset rewrite. This file
> captures the parent conversation's architectural decisions and frames
> the seven-phase delivery surface in one place. Per-phase detailed plans
> live alongside this file (`plans/phase{1..5}_*_detailed.md` already
> landed; Phase 6 and Phase 7 detailed plans are authored after the
> follow-on investigation worker reports). Do not amend phase 1–5 plans
> from this document; that step happens after the investigation worker.

## Table of contents

1. [Pipeline architecture (canonical chain)](#1-pipeline-architecture-canonical-chain)
2. [Phase inventory](#2-phase-inventory)
   - 2.1 [Cross-cutting: env-var-audit cleanup](#21-cross-cutting-env-var-audit-cleanup-phase-3a3b3c)
3. [Architectural decisions log](#3-architectural-decisions-log)
4. [Provenance chain (PDF → adapter)](#4-provenance-chain-pdf--adapter)
5. [LibV2 archival layout (post-Phase-7)](#5-libv2-archival-layout-post-phase-7)
6. [Outstanding open questions](#6-outstanding-open-questions)
7. [Sequencing recommendation](#7-sequencing-recommendation)

---

## 1. Pipeline architecture (canonical chain)

The end-to-end Ed4All pipeline carries one input (raw PDF) and one output
(deployed adapter on Hugging Face) through eight transformations. Each
transformation is owned by a single phase or workflow stage; each emits a
content hash that the next stage records as its provenance handle.

```
PDF (sha256)
  └─> DART (version, config_hash) → HTML (sha256)
        └─> ed4all-chunker → DART chunks (sha256)         ← LibV2/courses/<slug>/dart_chunks/
              └─> Source analyzer + Concept extractor → concept_graph (sha256)
                    └─> Objective synthesizer (LARGE)     → synthesized_objectives.json (ABCD)
                          └─> Block generator (small)     → outline blocks
                                └─> Inter-tier validators → validated blocks
                                      └─> Block rewriter (LARGE) → final blocks
                                            └─> Post-rewrite validators → final-validated blocks
                                                  └─> IMSCC packager   → IMSCC (sha256)
                                                        └─> ed4all-chunker → IMSCC chunks (sha256)  ← LibV2/courses/<slug>/imscc_chunks/
                                                              └─> Trainforge synthesis → training pairs
                                                                    └─> Trainforge train → Adapter (sha256)
```

### 1.1 Per-stage breakdown

| Stage | Workflow phase | Status | Owner code path | Hash emitted |
|---|---|---|---|---|
| PDF → DART HTML | `dart_conversion` | DONE | `DART/` (multi-source synthesis); `MCP/tools/pipeline_tools.py::_raw_text_to_accessible_html` | `pdf.sha256`, `dart_html.sha256`, `dart.config_hash` |
| HTML → DART chunks | `chunking` (NEW Phase 7a) | NEW IN PHASE 7 | `ed4all-chunker` package (Phase 7a; lifted from `Trainforge/process_course.py:1432::_chunk_content` and `:1669::_chunk_text_block`) | `dart_chunks.sha256` |
| DART chunks → concept graph | `concept_extraction` (NEW Phase 6) | NEW IN PHASE 6 | `Trainforge/pedagogy_graph_builder.py::build_pedagogy_graph` (decoupled from `libv2_archival`) | `concept_graph.sha256` |
| concept graph → objectives | `course_planning` | DONE (Wave 24) | `MCP/tools/pipeline_tools.py::plan_course_structure` → `synthesized_objectives.json`; Phase 6 widens schema to ABCD | `objectives.sha256` |
| objectives → outline blocks | `content_generation_outline` | IN-FLIGHT (Phase 3) | `Courseforge/router/router.py::CourseforgeRouter.route_all` → `Courseforge/generators/_outline_provider.py` | `blocks_outline.sha256` |
| outline → validated blocks | inter-tier gate (Phase 3) + statistical tier (Phase 4) | PARTIAL (Phase 3 deterministic gates landed; Phase 4 statistical not yet) | `Courseforge/router/inter_tier_gates.py` (Phase 3); `lib/validators/courseforge_outline_shacl.py` + `lib/validators/objective_assessment_similarity.py` (Phase 4) | `validated_blocks.sha256` |
| validated → final blocks | `content_generation_rewrite` | IN-FLIGHT (Phase 3) | `Courseforge/generators/_rewrite_provider.py` (in-flight; one of three Phase-3a env-var fix targets) | `blocks_final.sha256` |
| final → final-validated blocks | `post_rewrite_validation` (NEW Phase 3.5) | NEW IN PHASE 3.5 | Same validator classes as inter-tier gate, shape-discriminating adapters; symmetric enforcement | `blocks_validated.sha256` |
| final → IMSCC | `packaging` | DONE | `Courseforge/scripts/package_multifile_imscc.py` | `imscc.sha256` |
| IMSCC → IMSCC chunks | `chunking` (NEW Phase 7b/c) | NEW IN PHASE 7 | `ed4all-chunker` (second invocation; same impl, different input) | `imscc_chunks.sha256` |
| IMSCC chunks → training pairs | `training_synthesis` | DONE | `Trainforge/synthesize_training.py::run_synthesis` | (no hash; pairs are the artifact) |
| training pairs → adapter | `trainforge_train` (post-import LibV2 stage, Wave 90) | DONE | `Trainforge/train_course.py` → `LibV2/courses/<slug>/models/<adapter_id>/` | `adapter.sha256` |

### 1.2 The dual chunker invocation

A central architectural shift in this roadmap: **the chunker fires twice
with the same canonical implementation.** The two invocations consume
different sources and feed different downstream consumers:

- **DART chunks** (`LibV2/courses/<slug>/dart_chunks/`) — chunks of the
  DART-converted PDF. Provenance for the IMSCC source. Archived alongside
  raw PDFs. Consumed by the source analyzer + concept extractor (Phase 6)
  to build the concept graph that feeds objective synthesis.
- **IMSCC chunks** (`LibV2/courses/<slug>/imscc_chunks/`, currently
  `corpus/chunks.jsonl` per `LibV2/CLAUDE.md:194`) — chunks of the
  generated IMSCC. Provenance for the trained adapter. Consumed by
  Trainforge synthesis (`Trainforge/synthesize_training.py`).

Both chunksets must be present before adapter promotion. The gate is
`LibV2ManifestValidator` (`lib/validators/libv2_manifest.py`); Phase 7c
extends it to require both `dart_chunks_sha256` and `imscc_chunks_sha256`
in `manifest.json`.

### 1.3 Symmetric validation enforcement

Both the outline tier (small model) and the rewrite tier (large model)
emit blocks that face the same validator chain. Neither tier is trusted
to bypass enforcement — the large model is not given a free pass merely
because it is bigger. Operationally:

```
Block source                 ↓ validator chain ↓
──────────────────────────────────────────────────────────
Outline tier (small) ──────→ inter_tier_validation ──┐
                                                      │
                              regen/escalate/block    │
                                                      ↓
Rewrite tier (large) ──────→ post_rewrite_validation ──→ packaging
                              (SAME validator classes,
                               different block.content shape)
```

The same four validator classes (`BlockCurieAnchoringValidator`,
`BlockContentTypeValidator`, `BlockPageObjectivesValidator`,
`BlockSourceRefValidator`) fire at both gates with shape-discriminating
adapters: when `block.content` is a `dict` (outline tier output), the
validator reads from the structured fields; when `block.content` is a
`str` (rewrite tier HTML output), the validator scans the rendered HTML
via the existing extraction regex (e.g. `lib/ontology/curie_extraction.py`)
and verifies presence. Same gate, two adapters.

Failure semantics on either gate are governed by the
`GateResult.action` contract (Phase 3 §6.5): `regenerate` re-rolls the
block within its budget; `escalate` promotes to the next tier (or to a
stronger model in the rewrite tier); `block` fails closed. The retry
budget defaults to **10 per tier** (a deliberate increase from the
prior default of 3) — failed attempts inject specific failure context
into the next prompt, mirroring Trainforge's
`_append_preserve_remediation` pattern.

---

## 2. Phase inventory

The roadmap covers seven delivery phases plus one cross-cutting cleanup.
The first five are already on disk as `plans/phase{1..5}_*.md`; phases 6
and 7 are introduced by this document and will get detailed plans
authored by the follow-on investigation worker. Phase 3.5 is a small
followup wave that lands the symmetric-validation post-rewrite gate
between Phase 3 finishing and Phase 4.

| Phase | Status | Scope summary | Key deliverables | Dependencies | Detailed plan |
|---|---|---|---|---|---|
| 1 | **DONE** | ToS unblock — `COURSEFORGE_PROVIDER` env-var routing for the content-generator surface (`Courseforge/generators/_provider.py::ContentGeneratorProvider`). Mirrors `Trainforge/generators/_curriculum_provider.py:158-263` line-for-line with re-used HTTP composition and decision-capture wiring. | `_provider.py`, `content_generator_call` decision_event enum, Wave-74 short-circuit guard at `MCP/core/executor.py:833`, docs sync (`docs/LICENSING.md`, root `CLAUDE.md`, `Courseforge/CLAUDE.md`). | None | `plans/phase1_tos_unblock.md` (high-level), `plans/phase1_tos_unblock_detailed.md` (landed). |
| 2 | **DONE** | Stable in-memory `Block` dataclass + `Touch` provenance + JSON Schema additions (`courseforge_jsonld_v1.schema.json`'s new `$defs/Block` + `$defs/Touch`) + SHACL shapes (`courseforge_v1.shacl.ttl`'s new `BlockShape` + `TouchShape`) + outline-mode CLI (`generate_course.py --emit-mode outline`) + `package_multifile_imscc.py --outline-only` + Trainforge consumer (`html_content_parser.py::_extract_blocks_from_jsonld:505`). Adds `validation_attempts` and `escalation_marker` fields preemptively for Phase 3. | `Courseforge/scripts/blocks.py` (Block + Touch + 16-value `BLOCK_TYPES`), `block_emitter.py`, schema additions, SHACL shape additions, `course_metadata.schema.json`, renderer migration B1-B6, JSON-LD builder migration, `_extract_blocks_from_jsonld`, `COURSEFORGE_EMIT_BLOCKS` env flag. | Phase 1 | `plans/phase2_intermediate_format.md` (high-level), `plans/phase2_intermediate_format_detailed.md` (landed). |
| 3 | **IN-FLIGHT (~75%)** | Two-pass router (outline + rewrite) + per-block-type model routing via `block_routing.yaml` + per-call kwargs + tier-default env vars (`COURSEFORGE_OUTLINE_*`, `COURSEFORGE_REWRITE_*`); inter-tier deterministic gates (`curie_anchoring`, `content_type`, `page_objectives`, `source_refs`); self-consistency dispatch (`route_with_self_consistency`); regen budget + escalation_marker primitive. Constrained-decoding amendment (commit `9b6a5e4`) repositions GBNF/JSON-Schema as primary structural gate, SHACL as secondary semantic. Statistical-tier filter is a no-op shim (Phase 4 plug-in). | `Courseforge/router/router.py`, `_outline_provider.py`, `_rewrite_provider.py`, `policy.py`, `inter_tier_gates.py`, two new phases in `config/workflows.yaml` (`content_generation_outline` + `content_generation_rewrite`), four new `decision_event` enum values. | Phase 1, Phase 2 | `plans/phase3_two_pass_router.md` (high-level; amended by `9b6a5e4` for constrained decoding + self-consistency + regen budget + GateResult action contract), `plans/phase3_two_pass_router_detailed.md` (~75% landed). |
| 3.5 | **NEW (this roadmap)** | Symmetric-validation post-rewrite gate. Adds `post_rewrite_validation` workflow phase between `content_generation_rewrite` and `packaging` (gated `enabled_when_env: COURSEFORGE_TWO_PASS=true`). Reuses the same four validator classes from `inter_tier_validation` with shape-discriminating adapters (block.content as dict vs str). Bumps `COURSEFORGE_OUTLINE_REGEN_BUDGET` default 3→10 + adds `COURSEFORGE_REWRITE_REGEN_BUDGET=10`. Generalizes the failure-feedback remediation builder from `RewriteProvider.generate_rewrite` (Worker 2E's CURIE-preservation gate) into a reusable `Courseforge/router/remediation.py` module. | `post_rewrite_validation` phase entry in `config/workflows.yaml`; shape-discriminating adapters on the four `Block*Validator` classes in `inter_tier_gates.py`; `Courseforge/router/remediation.py` (new); `route_with_self_consistency` calls `_append_remediation_for_gates(prompt, failures)` between iterations. | Phase 3 finish | NEW. Detailed plan authored by investigation worker. |
| 4 | **PLANNED** | Statistical-tier validators: SHACL wire-up (`outline_shacl`), three new embedding gates (`objective_assessment_similarity`, `concept_example_similarity`, `objective_roundtrip_similarity`), threshold calibration script. **BERT ensemble + k-reranker** lands here as the disagreement detector (3-5 BERT classifiers with confidence-weighted majority vote + dispersion penalty). Symmetric-validator surfaces both inter-tier and post-rewrite gates. | `lib/embedding/sentence_embedder.py`, `lib/validators/courseforge_outline_shacl.py`, `lib/validators/objective_assessment_similarity.py`, `lib/validators/concept_example_similarity.py`, `lib/validators/objective_roundtrip_similarity.py`, `lib/classifiers/bloom_bert_ensemble.py` (NEW — 3-5 BERT classifiers + k-reranker), `scripts/calibrate_phase4_thresholds.py`, gate registrations in `config/workflows.yaml`. | Phase 2 (`Block` dataclass), Phase 3 (router seam, `GateResult.action` contract), Phase 3.5 (symmetric-validation surface). | `plans/phase4_statistical_tier.md` (high-level only). Detailed plan authored by investigation worker. |
| 5 | **PLANNED** | Independent CLI subcommands per stage (`courseforge-outline`, `courseforge-validate`, `courseforge-classify`, `courseforge-rewrite`, `courseforge`); per-block re-execution via `--blocks` filter; `--escalated-only` for Phase-3 escalation-marked blocks; `_synthesize_outline_output` / `_synthesize_classification_output` runner additions. | Five new `SUPPORTED_WORKFLOWS` entries; `_synthesize_outline_output` + `_synthesize_classification_output` in `MCP/core/workflow_runner.py`; `target_block_ids` param routing in `config/workflows.yaml`; per-stage decision capture via existing `phase_*` directory convention. | Phase 2 (`Block.block_id`), Phase 3 (router + tiers), Phase 4 (validators + classifier seam). | `plans/phase5_independent_stages.md` (high-level only). Detailed plan after investigation worker. |
| 6 | **NEW (this roadmap)** | (a) ABCD framework — synthesizer emits ABCD as discrete schema-validated fields (`{audience, behavior: {verb, action_object}, condition, degree}`); Python parser composes canonical prose deterministically. (b) Verb-level mismatch validation **folded into** ABCD validator as a schema-enforced lookup (`verb in BLOOMS_VERBS[blooms_level]`); NOT a separate Phase 4 gate. (c) Concept extractor decoupling — extract `pedagogy_graph_builder.build_pedagogy_graph` from `libv2_archival` into its own workflow phase `concept_extraction` between `course_planning` and `content_generation_outline`. Reuse + independent validation + versionable artifact (concept-graph hash in provenance). | `synthesized_objectives.json` schema gains ABCD object; `lib/ontology/learning_objectives.py` gains `compose_abcd_prose(abcd) -> str`; new validator `lib/validators/abcd_objective.py` (replaces a hypothetical Phase-4 verb-level check); new workflow phase `concept_extraction` in `config/workflows.yaml`. **Prerequisite envvar fix:** DART hardcoded `claude-sonnet-4-20250514` (across 4 files in `DART/pdf_converter/`) is fixed as a Phase-6 prereq so concept-extractor inputs are reproducible. | Phase 2, Phase 3, Phase 4 (statistical seam). | NEW. Detailed plan to be authored by the follow-on investigation worker. |
| 7 | **NEW (this roadmap)** | (a) `ed4all-chunker` package — lift `_chunk_content` + `_chunk_text_block` from `Trainforge/process_course.py` (`:1432`, `:1669`) into a standalone Python package with versioned releases. Multi-consumer (Courseforge + Trainforge + future validators + retrieval indices + eval pipelines). Structural pass deterministic (no LLM); optional enrichment pass LLM-driven (env-var-controlled). (b) Dual-chunkset architecture — fire chunker twice. DART chunks at `LibV2/courses/<slug>/dart_chunks/`; IMSCC chunks at `LibV2/courses/<slug>/imscc_chunks/` (renamed from `corpus/`). Both required for adapter promotion. (c) `LibV2ManifestValidator` extension — manifest.json gains `dart_chunks_sha256` and `imscc_chunks_sha256`; gate fails closed when either is missing. | New `ed4all-chunker` package (PyPI? in-repo? — open question §6.2); `MCP/tools/pipeline_tools.py::_chunk_dart_html` (new) + `_chunk_imscc` (factor out existing); `LibV2/courses/<slug>/dart_chunks/manifest.json` schema; `LibV2/courses/<slug>/imscc_chunks/manifest.json` schema; `lib/validators/libv2_manifest.py` extension; `LibV2/CLAUDE.md` directory-tree update; concept extractor (Phase 6) consumes `dart_chunks/`. | Phase 6 (concept extractor consumes the chunker output) and Phase 5 (CLI subcommands likely reference `ed4all-chunker dart` / `ed4all-chunker imscc`). | NEW. Detailed plan to be authored by the follow-on investigation worker. Likely splits into Phase 7a (chunker package), Phase 7b (DART chunkset), Phase 7c (IMSCC chunkset + manifest gate). |

### 2.1 Cross-cutting: env-var-audit cleanup (Phase 3a/3b/3c)

The hardcoded-LLM audit found **six HIGH findings + three MEDIUM**.
Per the user's direction, env-var audit fixes are part of the new
planning process (the `investigate → plan → execute → validate` loop)
rather than landing piecemeal inside Phase 3 finishing work. Each
sub-batch is bundled into a phase whose other work touches the same
files:

- **Phase 3a (priority-zero, ~120 LOC, lands in Phase 3.5):** fix
  `Courseforge/config/block_routing.yaml`, `Courseforge/generators/_rewrite_provider.py`, and `Courseforge/router/router.py` to read tier-default env vars (`COURSEFORGE_REWRITE_MODEL`, `COURSEFORGE_OUTLINE_MODEL`) before falling back to a hardcoded ID. Lands in Phase 3.5 because Phase 3.5 already touches `_rewrite_provider.py` (the remediation-builder generalization) and `router.py` (`_append_remediation_for_gates`) — bundling avoids a redundant audit cycle.
- **Phase 3b (Trainforge surfaces, ~60 LOC, lands inside Phase 4 cleanup):**
  fix `Trainforge/align_chunks.py` (hardcoded `claude-haiku-4-5-20251001`) and `Trainforge/process_course.py` (hardcoded `target_models` list + hardcoded `align_chunks` model). The natural home is Phase 4 because the embedding-tier wiring touches the same files; combining the two saves a churn cycle.
- **Phase 3c (DART + eval-judge + MCP orchestrator, ~50 LOC, lands as Phase
  6 prerequisite):** fix `DART/pdf_converter/*.py` (4 files hardcoding `claude-sonnet-4-20250514`) and `MCP/orchestrator/llm_backend.py` (claims env-var override but doesn't implement). DART is the source of HTML the concept extractor consumes — fixing the env-var path is a prereq for reproducible concept-graph builds.

The audit's three MEDIUM findings are deferred to a single follow-up
wave; they don't gate Phase 3.5 / 4 / 5 / 6 / 7.

---

## 3. Architectural decisions log

Every architectural choice the parent conversation captured, in
deciding-factor order. Decisions affecting more than one phase are noted
under "Lands in" with the primary owner first.

### 3.1 ABCD authorship — synthesizer emits structured fields, parser composes prose

- **Decision:** The objective synthesizer emits ABCD as discrete
  schema-validated fields: `{audience, behavior: {verb, action_object}, condition, degree}`. A Python parser (`lib/ontology/learning_objectives.py::compose_abcd_prose`) composes the canonical prose deterministically.
- **Rationale:** ABCD is one of the few pedagogical contracts where the
  cognitive work (choosing the verb, naming the action object, writing
  the condition) is genuinely model-driven, but the surface form is
  100% mechanical. Letting the LLM emit the prose form means every
  audit/repair cycle has to re-parse English; emitting structured fields
  + composing deterministically lets the validator and the rewriter both
  operate on canonical Python objects. Cognitive work in synthesis,
  mechanical work in parser. Matches the meta-pattern: "Fold it in if
  the operation is mechanical and bounded."
- **Lands in:** Phase 6 (primary). Touches Phase 4 by absorbing what
  would otherwise be a separate `bloom_verb_mismatch` validator.
- **Affects:** `synthesized_objectives.json` schema (extended to carry
  `abcd` object per LO); `Courseforge/scripts/blocks.py::Block`
  optionally projects ABCD into JSON-LD; `MCP/tools/pipeline_tools.py::plan_course_structure` emit shape; `lib/validators/abcd_objective.py` (new).

### 3.2 BERT classifier — ensemble + k-reranker (off-the-shelf v1 → fine-tuned v2; skip few-shot)

- **Decision:** The Bloom-classifier-disagreement gate uses a **3-5
  BERT classifier ensemble + k-reranker**, not a single checkpoint.
  - **v1**: 3-5 HuggingFace off-the-shelf checkpoints (e.g.
    `kabir5297/bloom_taxonomy_classifier` + 2-4 alternative
    domain-tuned variants). Each emits Bloom level + confidence;
    k-reranker aggregates via confidence-weighted majority vote with
    a dispersion penalty.
  - **v2**: light fine-tune of 3-5 ensemble members on public Bloom's
    data + Claude paraphrases (~30 min each on a 3070 — total ~2
    hours for the ensemble). Distributional diversity across members
    is the load-bearing property; train each member with different
    data shuffles / paraphrase variants to maximize independence.
  - **Skip the few-shot LLM-as-judge path entirely** — using the
    generator model with a different prompt collapses the
    disagreement signal to self-disagreement (much less informative).
- **Disagreement signal fires when:**
  - Generator self-tag ≠ ensemble's consensus prediction (the original
    disagreement check), OR
  - **Ensemble's internal dispersion > threshold** (NEW — uncertainty
    signal independent of agreement; high-dispersion blocks get
    flagged for adjudication even when generator + ensemble happen to
    agree by chance).
- **Hardware budget on 3070**: 3-5 DistilBERT-base ≈ 750MB-1.25GB VRAM;
  well within headroom alongside the 7B generator. Per-block latency
  <50ms total — negligible. Quantize to int8 if VRAM contention.
- **Calibration**: temperature scaling + dispersion threshold tuning on
  holdout set, both live in the same calibration script Phase 4 was
  already going to author (`scripts/calibrate_phase4_thresholds.py`).
- **Rationale:** The disagreement signal is load-bearing only when
  classifiers are *independent* of the generator AND of each other.
  Single-classifier disagreement misses tail-end uncertainty; an
  ensemble surfaces both kinds (agreement-vs-generator AND
  internal-dispersion). Latency budget <50ms/block makes encoder
  inference the only viable architecture inside the inter-tier gate.
- **Lands in:** Phase 4 (primary).
- **Affects:** New `lib/classifiers/bloom_bert_ensemble.py` wrapping the
  3-5 checkpoints + k-reranker; new `lib/validators/bloom_classifier_disagreement.py`; `pyproject.toml` `[embedding]` extra grows to include `transformers`; gate registration in `config/workflows.yaml::content_generation_outline` AND `post_rewrite_validation` (symmetric).

### 3.3 Chunker as a standalone package — `ed4all-chunker`

- **Decision:** Lift `Trainforge/process_course.py:1432::_chunk_content`
  + `:1669::_chunk_text_block` (and the supporting boilerplate-detection helpers in `Trainforge/rag/boilerplate_detector.py`) into a versioned package `ed4all-chunker`. Multi-consumer (Courseforge + Trainforge + future validators + retrieval indices + eval pipelines). Structural pass is deterministic (no LLM); optional enrichment pass is LLM-driven and gated by an env var.
- **Rationale:** Apply the meta-pattern: "Make it a package if multiple
  tools consume it." Today the chunker lives inside Trainforge as a
  `CourseProcessor` method — single-consumer, mutually-coupled with the
  IMSCC parser. Phase 7's dual-chunkset architecture (DART chunks +
  IMSCC chunks) means at minimum two consumers; Phase 6's concept
  extractor adds a third; future eval pipelines and retrieval indices
  will bring more.
- **Lands in:** Phase 7a (primary). Phase 6 consumes the lifted package.
- **Affects:** New `ed4all-chunker/` package directory at the repo root
  (or external PyPI? — open question §6.2); refactor of
  `Trainforge/process_course.py::CourseProcessor._chunk_content`/`_chunk_text_block` to delegate; `pyproject.toml` adds the package as a workspace member; `Trainforge/CLAUDE.md` updates the chunker section to point at the package.

### 3.4 Concept extractor decoupling — its own workflow phase

- **Decision:** Extract `Trainforge/pedagogy_graph_builder.py::build_pedagogy_graph` (referenced from `Trainforge/process_course.py:3625-3628`) from the `libv2_archival` phase into its own workflow phase named `concept_extraction`. The new phase sits between `course_planning` and `content_generation_outline` (or, more strictly per the canonical chain, between `chunking` and `course_planning`).
- **Rationale:** Apply the meta-pattern: "Make it a phase if the
  artifact is reusable elsewhere." The concept graph is consumed by:
  (i) objective synthesizer (decides which concepts merit a TO/CO
  objective), (ii) inter-tier validator `concept_example_similarity`
  (Phase 4), (iii) abstention generator (`Trainforge/generators/abstention_generator.py`, Wave 124), (iv) Phase-6 BERT-ensemble-disagreement gate (uses concept tags as input feature).
- **Lands in:** Phase 6 (primary). Affects Phase 7 because the chunker
  package is the concept extractor's input.
- **Affects:** `config/workflows.yaml` — new `concept_extraction` phase
  under `textbook_to_course`; `Trainforge/process_course.py` — the
  extractor invocation moves out of the archival path; `LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json` becomes a first-class artifact; `lib/validators/concept_graph.py` (new) gates the artifact.

### 3.5 Dual-chunkset architecture — DART chunks + IMSCC chunks

- **Decision:** The chunker fires twice with the same canonical
  implementation:
  - `LibV2/courses/<slug>/dart_chunks/chunks.jsonl` — chunks of the
    DART HTML (the source).
  - `LibV2/courses/<slug>/imscc_chunks/chunks.jsonl` — chunks of the
    final IMSCC (currently lives at `corpus/chunks.jsonl` per
    `LibV2/CLAUDE.md:194`; renamed for symmetry).
  Both must be present before adapter promotion. The provenance chain
  is the triangle PDF → DART → DART chunks → synthesis → IMSCC → IMSCC
  chunks → adapter (see §4 for hash detail).
- **Rationale:** Today the only chunkset is `corpus/chunks.jsonl`,
  derived from the IMSCC. That makes the IMSCC the de-facto provenance
  unit for the trained adapter, but it severs the line back to the PDF
  source: nothing downstream can prove "this adapter was trained on
  pairs derived from concepts that the source PDF actually establishes."
  DART chunks fix that — they're a content-addressable representation
  of the source that the concept extractor and objective synthesizer
  consume directly.
- **Lands in:** Phase 7b/7c (primary). Phase 6 (concept extractor)
  consumes the DART chunkset.
- **Affects:** `LibV2/courses/<slug>/dart_chunks/` (new), `LibV2/courses/<slug>/imscc_chunks/` (renamed from `corpus/`), `manifest.json` schema gains `dart_chunks_sha256` + `imscc_chunks_sha256`, `lib/validators/libv2_manifest.py::LibV2ManifestValidator` extension, `LibV2/CLAUDE.md` directory-tree update, `Trainforge/synthesize_training.py` reads from `imscc_chunks/` (path migration).

### 3.6 Symmetric validation enforcement — both tiers face the same gates

- **Decision:** Both the outline tier (small model) and the rewrite
  tier (large model) emit blocks that face the same validator chain.
  The large model is not given a free pass merely because it is bigger.
  A new `post_rewrite_validation` workflow phase sits between
  `content_generation_rewrite` and `packaging`, gated
  `enabled_when_env: COURSEFORGE_TWO_PASS=true`. It reuses the same
  four validator classes from `inter_tier_validation` with
  shape-discriminating adapters (block.content as outline-tier dict vs
  rewrite-tier HTML string).
- **Rationale:** Validation enforcement is independent of model size.
  A large model that drops a CURIE or emits an out-of-taxonomy
  content_type is just as broken as a small model that does the same.
  The asymmetric design (validate outline, trust rewrite) silently
  permits regressions on the rewrite-tier surface — exactly the surface
  whose output ships in the IMSCC.
- **Lands in:** Phase 3.5 (primary; new phase introduced by this
  roadmap). Phase 4 extends the symmetric surface to its statistical
  validators.
- **Affects:** New `post_rewrite_validation` phase in
  `config/workflows.yaml`; shape-discriminating adapters on the four
  `Block*Validator` classes in `Courseforge/router/inter_tier_gates.py`; `_extract_curies_from_html` helper in `lib/ontology/curie_extraction.py` (extended for shape-discrimination); decision-capture extension of `block_validation_action` event with `tier="rewrite"` field.

### 3.7 Retry budget — bumped 3→10 + per-tier override

- **Decision:** Default retry budget for both tiers bumps from `3` to
  `10`. New env vars:
  - `COURSEFORGE_OUTLINE_REGEN_BUDGET` (default `10`; was `3`)
  - `COURSEFORGE_REWRITE_REGEN_BUDGET` (new, default `10`)
  - Per-block-type override via `block_routing.yaml` (existing schema
    accommodates this — Worker G's `regen_budget_by_block_type`
    fast-lookup map is already in place).
- **Rationale:** The user explicitly authorized "validation across
  inferencing is fine. up to 10 retries per call is also fine. failed
  tries should feed failures back into future prompts like what
  trainforge already does." Three retries is a conservative default
  that escalates too aggressively; ten retries with feedback injection
  is closer to the real cost/quality frontier for small-model outline
  generation. Escalation to the rewrite tier is still triggered at
  budget exhaustion — the bump just gives the small model more room
  to recover from transient parse failures.
- **Lands in:** Phase 3.5 (primary; same wave as the post-rewrite
  validation surface).
- **Affects:** Default constants in `Courseforge/router/router.py`
  (`_DEFAULT_REGEN_BUDGET = 10`); root `CLAUDE.md` flag table updated;
  `block_routing.schema.json` documents the override surface (no
  schema change needed — already an integer field).

### 3.8 Failure-feedback injection — generalize Trainforge's remediation pattern

- **Decision:** Failed validator passes inject specific failure context
  into the next prompt, mirroring Trainforge's
  `_append_preserve_remediation` pattern at
  `Trainforge/generators/_local_provider.py:548-583`. Worker 2E already
  ported this to `RewriteProvider.generate_rewrite` for CURIE
  preservation; Phase 3.5 generalizes it into a reusable module.
- **Concrete shape:**
  - New module `Courseforge/router/remediation.py` exposing
    `_append_remediation_for_gates(prompt: str, failures: List[GateResult]) -> str`. Each `GateResult.issues[]` carries a specific failure (e.g. "CURIE 'sh:NodeShape' was dropped"); the remediation builder formats these into a "Your previous attempt failed: [validator] flagged [issue]. Correct by: [directive]" prefix on the next prompt.
  - `route_with_self_consistency` (Worker H's deliverable) calls
    `_append_remediation_for_gates(prompt, failures)` between iterations
    instead of retrying with the same prompt.
  - Both tiers (outline retries with inter-tier-validator feedback;
    rewrite retries with post-rewrite-validator feedback) use the same
    builder; the validator-set is what differs.
  - Per-block-type remediation prompts get curated per failure mode
    (CURIE drop → "preserve all CURIEs verbatim"; verb-level mismatch
    → "use a verb from {valid set for declared Bloom level}";
    content-type miss → "select content_type from the canonical
    taxonomy: [list]").
- **Rationale:** The user named this explicitly: "failed tries should
  feed failures back into future prompts like what trainforge already
  does." Retrying with the same prompt wastes budget on the same
  failure; injecting specific failure context turns each retry into a
  remediation pass rather than a re-roll.
- **Lands in:** Phase 3.5 (primary).
- **Affects:** New `Courseforge/router/remediation.py`;
  `route_with_self_consistency` extended to call the builder between
  iterations; per-block-type remediation directives table (data file
  or constants module).

### 3.9 Env-var posture — every LLM call site reads the tier-default env var

- **Decision:** No LLM call site in any production code path may
  hardcode a model ID. Every call site reads the tier-appropriate env
  var (`COURSEFORGE_OUTLINE_MODEL` / `COURSEFORGE_REWRITE_MODEL` /
  `ANTHROPIC_SYNTHESIS_MODEL` / `LOCAL_SYNTHESIS_MODEL` / etc.) before
  falling back to a hardcoded default.
- **Rationale:** The hardcoded-LLM audit found nine call sites that
  silently hardcode a model ID — meaning an operator setting an
  env-var override was being silently ignored. That breaks both ToS
  posture (a license-clean run silently routes through Anthropic) and
  reproducibility (rebuilding the same course on a different
  operator's machine produces a different artifact).
- **Lands in:** Phase 3a (in-flight Phase 3 surfaces, lands in Phase
  3.5 alongside symmetric-validation work — see §2.1), Phase 4 cleanup
  (Trainforge surfaces — Phase 3b), Phase 6 prereq (DART + MCP
  orchestrator — Phase 3c).
- **Affects:** `Courseforge/config/block_routing.yaml`,
  `Courseforge/generators/_rewrite_provider.py`,
  `Courseforge/router/router.py`, `Trainforge/align_chunks.py`,
  `Trainforge/process_course.py`,
  `MCP/orchestrator/llm_backend.py`,
  `DART/pdf_converter/*.py` (4 files).

### 3.10 `escalation_marker` collision risk (Phase 2 followup)

- **Decision:** Phase 2 introduced
  `Block.escalation_marker: Optional[str]` validated against
  `_ESCALATION_MARKERS = {"outline_budget_exhausted", "structural_unfixable", "validator_consensus_fail"}`. Phase 3 sets `"outline_budget_exhausted"`; Phase 5 may set the others. The decision documents that the markers are an extensible enum and the SHACL shape (`schemas/context/courseforge_v1.shacl.ttl::BlockShape`) and JSON Schema (`schemas/knowledge/courseforge_jsonld_v1.schema.json::$defs.Block.escalationMarker`) BOTH need to be updated when a new marker is introduced.
- **Lands in:** Phase 2 followup (small wave; ideally before Phase 3.5
  finishes its escalation_marker plumbing).
- **Affects:** `schemas/context/courseforge_v1.shacl.ttl::BlockShape`,
  `schemas/knowledge/courseforge_jsonld_v1.schema.json::$defs.Block`,
  `Courseforge/scripts/blocks.py::_ESCALATION_MARKERS`, possibly
  `schemas/taxonomies/escalation_markers.json` (new).

### 3.11 `route_all` + `route_with_self_consistency` integration (Phase 3 followup)

- **Decision:** Make `route_all` invoke `route_with_self_consistency`
  per block by default (instead of the simpler `route(block, tier=...)` shape currently described). Without that, `route_all` doesn't benefit from the self-consistency loop and the regen-budget primitive is silently bypassed.
- **Lands in:** Phase 3 finishing (likely lands as part of Worker I's
  regen-budget batch or the Wave N+1 Batch 4 wrap-up).
- **Affects:** `Courseforge/router/router.py::CourseforgeRouter.route_all`, the integration test at `tests/integration/test_courseforge_two_pass_end_to_end.py` (Phase 3 plan §10 item 5).

### 3.12 Meta-pattern (operator's articulation)

The user articulated a four-rule heuristic for deciding whether a piece
of work is a phase, a fold-in, a package, or private detail:

- **Phase:** Make it a phase if the artifact is reusable elsewhere
  (chunks, concept graphs, objectives, validated blocks).
- **Fold:** Fold it in if the operation is mechanical and bounded
  (ABCD prose composition, verb-level validation, schema enforcement).
- **Package:** Make it a package if multiple tools consume it (chunker
  yes, concept extractor probably yes, validators yes).
- **Private:** Keep it private if it's incidental to one phase (prompt
  formatting, retrieval window construction, regeneration loops).

This roadmap applies the heuristic consistently. ABCD is a fold-in
(rationale: bounded, mechanical). Concept extraction is a phase
(rationale: artifact reused four ways). Chunker is a package
(rationale: multi-consumer, versionable). Verb-level validation is
folded into ABCD (rationale: schema-enforced lookup, not a separate
gate). Per-block prompt formatting stays private (incidental to outline
provider). Failure-feedback remediation builder is a private utility
that one phase (3.5) generalizes — but the resulting module is
reusable across both tiers, so it lives in `Courseforge/router/` as a
shared module rather than a separate package.

---

## 4. Provenance chain (PDF → adapter)

The provenance chain is a SHA-256 triangle: every transformation records
both its input hash and its output hash. The full chain is auditable end
to end; any operator can run `sha256sum LibV2/courses/<slug>/manifest.json` and trace forward and backward through the pipeline.

### 4.1 Per-artifact hash + storage + consumer

| Artifact | What gets hashed | Where the hash is recorded | Consumer that validates | Gate that fails closed on mismatch |
|---|---|---|---|---|
| `pdf.sha256` | Raw PDF bytes | `LibV2/courses/<slug>/source/pdfs/<name>.pdf.sha256` (sidecar) and `LibV2/courses/<slug>/manifest.json::source.pdfs[].sha256` | `MCP/tools/pipeline_tools.py::archive_to_libv2` | `lib/validators/libv2_manifest.py::LibV2ManifestValidator` (existing) |
| `dart.config_hash` | DART config payload (model, env vars, prompt versions) | `LibV2/courses/<slug>/source/dart_html/.dart_config.json::config_hash` | DART regression tests | none (advisory; helps reproducibility audit) |
| `dart_html.sha256` | DART-emitted HTML (per file or rolled up) | sidecar + `manifest.json::source.dart_html.sha256` | chunker (Phase 7b) | extension to `LibV2ManifestValidator` (Phase 7c) |
| `dart_chunks.sha256` (NEW Phase 7b) | The `chunks.jsonl` file content | `LibV2/courses/<slug>/dart_chunks/manifest.json::chunks_sha256` and rolled up to course manifest's `dart_chunks_sha256` | concept extractor (Phase 6); objective synthesizer (existing `course_planning`) | `LibV2ManifestValidator` extension (Phase 7c) — fail closed when `dart_chunks_sha256` missing |
| `concept_graph.sha256` (NEW Phase 6) | `concept_graph_semantic.json` | `LibV2/courses/<slug>/concept_graph/manifest.json::sha256` and rolled up | objective synthesizer; assessment generator; abstention generator | `lib/validators/concept_graph.py` (new, Phase 6) |
| `objectives.sha256` | `synthesized_objectives.json` | `Courseforge/exports/<project>/01_learning_objectives/manifest.json` and rolled up | content-generator (Phase 1 / Phase 3) | `lib/validators/page_objectives.py` (existing) |
| `block_outline.sha256` (NEW Phase 3) | `blocks_outline.jsonl` | `Courseforge/exports/<project>/01_outline/manifest.json` per Phase 5 §6 | inter-tier gates; rewrite tier | router-internal (Phase 3) |
| `block_validated.sha256` (NEW Phase 3.5) | `blocks_validated.jsonl` (post-rewrite-validation pass output) | `Courseforge/exports/<project>/04_validated/manifest.json` | packaging | `post_rewrite_validation` workflow phase (Phase 3.5) |
| `Block.content_hash` (Phase 2) | sha256 of the canonical Block payload (excludes `touched_by`, `sequence`) | `Block.content_hash` field on every Block; emitted in JSON-LD `blocks[].contentHash` | re-execution semantics (Phase 5); detects cross-tier drift | `Block` dataclass `__post_init__` (Phase 2) |
| `imscc.sha256` | `*.imscc` zip bytes | `Courseforge/exports/<project>/05_imscc/manifest.json` and rolled up | Trainforge `process_course.py`; LibV2 archival | `IMSCCValidator` (existing, `lib/validators/imscc.py`) |
| `imscc_chunks.sha256` (NEW Phase 7c) | `imscc_chunks/chunks.jsonl` | `LibV2/courses/<slug>/imscc_chunks/manifest.json::chunks_sha256` and rolled up to `manifest.json::imscc_chunks_sha256` | Trainforge synthesis; trainer | `LibV2ManifestValidator` extension (Phase 7c) |
| `adapter.sha256` | `adapter_model.safetensors` weights file | `LibV2/courses/<slug>/models/<adapter_id>/manifest.json::weights_sha256` (Wave 89) | promotion ledger (`models/_pointers.json`) | `lib/validators/libv2_model.py::LibV2ModelValidator` (existing) |

### 4.2 The triangle invariant

The dual-chunkset architecture turns the provenance chain into a
triangle. The invariant is: **for any deployed adapter, both the DART
chunkset and the IMSCC chunkset must hash-resolve back to the same
PDF.**

```
PDF.sha256
   ├── DART HTML.sha256 ─── DART chunks.sha256 ─── concept_graph.sha256 ─── objectives.sha256
   │                                                                              ↓
   │                                                                       Block emits
   │                                                                              ↓
   └── IMSCC.sha256       ─── IMSCC chunks.sha256 ─── adapter.sha256
```

`LibV2ManifestValidator` (Phase 7c extension) cross-checks the triangle:
the `pdf.sha256` listed under `source.pdfs[]` is the same value
referenced (transitively) by both chunksets' upstream `dart_html.sha256`
/ `imscc.sha256`. A mismatch is a hard failure; it means the IMSCC was
generated from a different source than the DART chunks the concept
graph was derived from, which would silently invalidate the training
provenance.

### 4.3 `Block.content_hash` re-execution semantics

`Block.content_hash` (Phase 2 deliverable, see
`Courseforge/scripts/blocks.py::Block.compute_content_hash`) deliberately excludes `touched_by` and `sequence` from its input. That makes the hash:

- **Stable across touches** — re-routing a block through a second model
  doesn't change `content_hash` until the content changes.
- **Stable across reorderings** — moving a block's `sequence` doesn't
  invalidate its hash.
- **Unstable across content changes** — a single character change in the
  content payload trips the hash.

This is the load-bearing property for Phase 5's `--blocks` re-execution
test: "every assessment block's `content_hash` may have changed; every
other block's hash is byte-identical."

### 4.4 Touch chain after symmetric validation lands (Phase 3.5)

`Block.touched_by[]` chain extends from `outline → rewrite` to
`outline → outline_val → rewrite → rewrite_val`. Each gate that fires
(passing OR failing) appends a Touch entry. Audit trail becomes the
canonical view of "this block was authored by X then validated by Y
then rewritten by Z then re-validated by Y'."

---

## 5. LibV2 archival layout (post-Phase-7)

Per-course directory tree after Phase 7 lands. Net additions: three new
top-level directories (`dart_chunks/`, `concept_graph/`, `imscc_chunks/`)
and a renamed-from-`corpus/` for symmetry. Models gain a manifest field
that carries both chunkset hashes for verification at promotion time.

```
LibV2/courses/<slug>/
├── source/
│   ├── pdfs/                       # raw PDFs (existing)
│   │   └── <name>.pdf.sha256       # sidecar (Phase 7b)
│   └── dart_html/                  # DART HTML output (existing)
│       └── .dart_config.json       # config_hash sidecar (Phase 7b extension)
├── dart_chunks/                    # NEW (Phase 7b)
│   ├── chunks.jsonl
│   ├── chunks_enriched.jsonl       # optional second-pass enrichment
│   └── manifest.json               # chunks_sha256, chunker_version, source_dart_html_sha256
├── concept_graph/                  # NEW (Phase 6 — extracted from libv2_archival embedding)
│   ├── concept_graph_semantic.json
│   └── manifest.json               # sha256, source_dart_chunks_sha256, builder_version
├── synthesized_objectives.json     # NEW shape (Phase 6 — ABCD-extended)
├── imscc/                          # final course package (existing — `source/` overlap clarified Phase 7c)
│   └── *.imscc
├── imscc_chunks/                   # IMSCC-derived chunks (renamed from corpus/ for clarity)
│   ├── chunks.jsonl
│   └── manifest.json               # chunks_sha256, chunker_version, source_imscc_sha256
├── training_specs/                 # SFT/DPO pairs (existing)
│   └── instruction_pairs.jsonl
├── concept_graph_semantic.json     # legacy location — Phase 6 deprecates in favour of concept_graph/
├── manifest.json                   # course manifest; Phase 7c adds dart_chunks_sha256 + imscc_chunks_sha256
├── course.json                     # learning_outcomes (existing, schema unchanged in Phase 6)
├── pedagogy/                       # pedagogy framework metadata (existing)
├── quality/                        # KG-quality reports (existing)
└── models/<adapter_id>/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── manifest.json               # carries dart_chunks_sha256 + imscc_chunks_sha256 (Phase 7c)
```

### 5.1 Path migrations

| Today | Post-Phase-7 | Migration |
|---|---|---|
| `LibV2/courses/<slug>/corpus/chunks.jsonl` | `LibV2/courses/<slug>/imscc_chunks/chunks.jsonl` | One-time rename in Phase 7c. `LibV2/tools/libv2/cli.py::retrieve` reads from both for one wave (back-compat); subsequent wave drops the legacy path. |
| `LibV2/courses/<slug>/concept_graph_semantic.json` (file at course root, for some courses) | `LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json` | Phase 6 standardises across all courses + adds `manifest.json` sidecar. |
| (no DART chunks today) | `LibV2/courses/<slug>/dart_chunks/` | Phase 7b creates fresh; no migration. |

### 5.2 Manifest schema additions (Phase 7c)

The course-level `manifest.json` (validated by `LibV2ManifestValidator`)
gains:

```json
{
  "dart_chunks_sha256": "<hex>",
  "imscc_chunks_sha256": "<hex>",
  "concept_graph_sha256": "<hex>",
  "chunker_version": "1.2.3"
}
```

All four are required for a course to clear the `libv2_manifest` gate
post-Phase-7. Legacy courses (pre-Phase-7) get a one-shot backfill
script (`LibV2/tools/libv2/scripts/backfill_dart_chunks.py`) that reads
DART HTML from `source/dart_html/`, runs the chunker, writes
`dart_chunks/`, and updates the manifest. Backfill is operator-driven
(not automatic) so the audit trail captures who initiated it.

---

## 6. Outstanding open questions

These decisions need explicit operator input before the follow-on
investigation worker can author a detailed plan for Phase 6 or Phase 7.

### 6.1 Concept-graph per-edge provenance

Should `concept_graph/concept_graph_semantic.json` carry per-edge
provenance (which DART chunk asserted the relationship)?

**Recommendation, pending operator input:** yes, add edge-level
provenance, gated behind a new env var
`TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE` (default off for legacy
corpora; flip on for Phase 6 emit).

### 6.2 `ed4all-chunker` packaging form

Should `ed4all-chunker` be a Python package shipped via PyPI, a CLI tool,
both, or an in-repo workspace member?

**Recommendation, pending operator input:** in-repo workspace member
first (Phase 7a delivery), with a TODO to promote to PyPI when an
external consumer (eval pipeline, third-party retriever) shows up.

### 6.3 Dual-chunkset gate posture

Should the dual-chunkset gate (`LibV2ManifestValidator` extension) be
advisory (warn-on-mismatch) or hard (fail-closed)?

**Recommendation, pending operator input:** hard, with a one-shot
backfill path for legacy courses. The whole point of the dual-chunkset
architecture is the triangle invariant; an advisory gate erodes it.

### 6.4 ABCD schema location

Should the ABCD schema (`{audience, behavior: {verb, action_object}, condition, degree}`) live in `schemas/knowledge/courseforge_jsonld_v1.schema.json::$defs.LearningObjective` (extending the existing LO def) or in a new top-level `schemas/knowledge/abcd_objective.schema.json`?

**Recommendation, pending operator input:** co-locate with an explicit
`$defs.AbcdObjective` sub-definition referenced from
`$defs.LearningObjective.properties.abcd`.

### 6.5 BERT ensemble member selection + checkpoint pinning

For the 3-5 BERT ensemble members in Phase 4, which checkpoints should
be the v1 ensemble? Pin by SHA?

**Recommendation, pending operator input:** Start with one
domain-tuned checkpoint (`kabir5297/bloom_taxonomy_classifier`) + 2-4
distilbert variants fine-tuned on different paraphrase splits during
v2. SHA-pin all members for reproducibility.

### 6.6 Concept extractor's input scope

Does the Phase 6 `concept_extraction` phase consume DART chunks
exclusively (clean source-of-truth provenance), or does it also see
`synthesized_objectives.json`?

**Recommendation, pending operator input:** two-stage — DART chunks
only, with an explicit linking pass
(`lib/ontology/concept_objective_linker.py`, new) between concept
extraction and content generation.

### 6.7 Symmetric-validation failure escalation policy

When the post-rewrite validation gate fails, the rewrite-tier block can
be (a) re-rolled with a remediation prompt, (b) escalated to a stronger
model (e.g. `claude-opus-4-7`), or (c) fail-closed at the workflow
level. Which sequence?

**Recommendation, pending operator input:** (a) → (b) → (c). Re-roll
once with remediation; escalate to stronger if remediation fails;
fail-closed only if escalation also fails.

---

## 7. Sequencing recommendation

The default ordering is **3.5 → 4 → 5 → 6 → 7** (each builds on prior).
The case for an alternative ordering is real: **3.5 → 4 → 7a → 6 →
7b/7c → 5** moves the chunker package up because it unblocks both
Phase 6 (concept extractor consumes the chunker package) and Phase 7
(dual-chunkset architecture is the whole point of Phase 7).

### 7.1 Recommended ordering: 3.5 → 4 → 7a → 6 → 7b/7c → 5

Rationale, in priority order:

1. **Phase 3.5 first.** Phase 3.5 is small (post-rewrite validation +
   retry budget bump + remediation generalization + Phase 3a env-var
   fix) and lands directly on top of in-flight Phase 3. Bundling
   avoids a redundant audit cycle on `_rewrite_provider.py` and
   `router.py`.
2. **Phase 4 second.** Phase 4 finishes the validator/gate seam Phase 3
   left open (the `GateResult.action` contract; the four embedding
   gates; the SHACL wire-up; the BERT ensemble + k-reranker disagreement
   detector). Phase 3b (Trainforge env-var fixes) naturally lands here.
3. **Phase 7a (chunker package) third.** The chunker as a standalone
   package unblocks two downstream consumers: the Phase 6 concept
   extractor and the Phase 7b/7c dual-chunkset work. Lifting the
   chunker first is the cheapest cut.
4. **Phase 6 fourth.** Phase 6 is the heaviest cognitive shift in the
   roadmap (ABCD authorship, concept extractor decoupling). Phase 3c
   env-var fixes (DART) bundle here as the prereq.
5. **Phase 7b/7c fifth.** Phase 7b creates the DART chunkset; Phase 7c
   renames `corpus/` → `imscc_chunks/` and extends
   `LibV2ManifestValidator`. With Phase 7a already landed, this is
   mostly directory layout + manifest schema changes.
6. **Phase 5 last.** Phase 5 is operator UX. Landing it last means the
   subcommands are designed against the final architecture, not an
   interim shape that would force a second-pass UX update.

**Recommendation:** 3.5 → 4 → 7a → 6 → 7b/7c → 5. Land the architectural
work before the operator-facing surface.

---

## Appendix A: Citations

Inline citations consolidated for the follow-on investigation worker:

- `Trainforge/process_course.py:1432-1748` — `_chunk_content` /
  `_chunk_text_block` (Phase 7a lift target).
- `Trainforge/process_course.py:3625-3628` — `pedagogy_graph_builder`
  invocation (Phase 6 decoupling target).
- `Trainforge/process_course.py:2333-2341` — `_extract_section_metadata`
  consumes JSON-LD `blocks[]` (Phase 2 deliverable, Phase 6 ABCD
  extension surface).
- `Courseforge/scripts/blocks.py:223-265` — `Block` dataclass (Phase 2).
- `Courseforge/scripts/blocks.py:77-96` — 16-value `BLOCK_TYPES` enum.
- `Courseforge/scripts/generate_course.py:2085-2098` — `_build_page_metadata` emits new Phase-2 fields when `COURSEFORGE_EMIT_BLOCKS=true`.
- `Courseforge/router/router.py` — Worker I in flight (regen budget).
- `Courseforge/generators/_rewrite_provider.py` — Phase 3a env-var fix
  target; also Worker 2E's CURIE-preservation gate (Phase 3.5
  remediation builder lift target).
- `Trainforge/generators/_local_provider.py:548-583` — Trainforge's
  `_append_preserve_remediation` pattern (Phase 3.5 generalization
  template).
- `Courseforge/config/block_routing.yaml` — Phase 3a env-var fix target.
- `Trainforge/align_chunks.py` — Phase 3b env-var fix target
  (hardcoded `claude-haiku-4-5-20251001`).
- `MCP/orchestrator/llm_backend.py` — Phase 3c env-var fix target
  (claims override but doesn't implement).
- `DART/pdf_converter/*.py` — Phase 3c env-var fix targets (4 files
  hardcoding `claude-sonnet-4-20250514`).
- `lib/validators/libv2_manifest.py` — Phase 7c extension target
  (`LibV2ManifestValidator`).
- `LibV2/CLAUDE.md:194` — `corpus/chunks.jsonl` (renames to
  `imscc_chunks/chunks.jsonl` in Phase 7c).
- `lib/ontology/learning_objectives.py` — Phase 6 home for
  `compose_abcd_prose` and `BLOOMS_VERBS` lookup.
- `schemas/knowledge/courseforge_jsonld_v1.schema.json` — Phase 6
  schema extension target (ABCD `$defs`).
- `schemas/context/courseforge_v1.shacl.ttl::BlockShape` — Phase 2
  followup (escalation_marker enum sync).
- `plans/phase3_two_pass_router.md` §2.1.1 — constrained decoding as
  primary structural gate (commit `9b6a5e4`).
- `plans/phase3_two_pass_router.md` §3.6 — self-consistency dispatch.
- `plans/phase3_two_pass_router.md` §3.7 — regen budget + escalation.
- `plans/phase3_two_pass_router.md` §6.5 — `GateResult.action`
  contract.
- `plans/phase4_statistical_tier.md` §3 — DistilBERT classifier
  deferral (REPLACED by Phase-4 BERT ensemble + k-reranker per this
  roadmap §3.2).
- `docs/LICENSING.md:46-60` — synthesis providers table.

## Appendix B: Update protocol

When this roadmap diverges from a phase's detailed plan, the detailed
plan wins (it's the implementation contract). When this roadmap
diverges from `CLAUDE.md` / `Trainforge/CLAUDE.md` / `LibV2/CLAUDE.md`,
the per-project CLAUDE.md wins (it's the operational contract). This
file is the architectural intent — it captures the *why*. Phase plans
capture the *how*. CLAUDE.md captures the *what's currently true*.
