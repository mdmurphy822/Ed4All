# Trainforge Architecture

Living architecture map for the Trainforge subsystem. Cross-references the project guide at `Trainforge/CLAUDE.md`, the root `CLAUDE.md` (validation-gate + behavior-flag tables), `schemas/ONTOLOGY.md`, and `docs/LICENSING.md` rather than restating their contents.

---

## Overview

Trainforge is the assessment-generation, knowledge-graph synthesis, training-pair synthesis, SLM training, and SLM evaluation subsystem of Ed4All. It consumes IMSCC packages (typically produced by Courseforge) and emits a LibV2-compatible RAG corpus plus, optionally, a course-pinned QLoRA adapter with a model card, an eval report, and an ablation report. It is the origin of every artifact a downstream retrieval consumer or trained-model consumer reads.

```
   IMSCC / HTML                                       (Courseforge output)
        |
        v
+-----------------------+
|  process_course.py    |   parsers/, rag/boilerplate, lib/ontology
|  (CPU-only, deterministic)
+-----------+-----------+
            |
            +--> imscc_chunks/chunks.jsonl  (Phase 7c rename of corpus/chunks.jsonl)
            +--> graph/concept_graph_semantic.json (+ .trig under TRAINFORGE_EMIT_TRIG)
            +--> graph/pedagogy_graph.json
            +--> graph/vocabulary.ttl
            +--> assessments.json
            +--> quality/quality_report.json
            +--> pedagogy/pedagogy_model.json
            |
            v
+-----------------------+
|  align_chunks.py      |   teaching-roles, prereq edges, LO refs
+-----------+-----------+
            |
            v
+-----------------------+
|  synthesize_training  |   generators/{anthropic,together,local,
|  .py                  |               claude_session,mock}
+-----------+-----------+
            |
            +--> training_specs/instruction_pairs.jsonl
            +--> training_specs/preference_pairs.jsonl
            +--> training_specs/.synthesis_cache.jsonl
            +--> training_specs/.synthesis_telemetry.jsonl
            +--> training_specs/pilot_report.md
            |
            |   (LibV2 import boundary -- post-import LibV2 stage)
            v
+-----------------------+
|  train_course.py      |   training/{runner,peft_trainer,
|  + training/runner.py |               base_models,compute_backend}
+-----------+-----------+
            |
            +--> models/<model_id>/adapter.safetensors
            +--> models/<model_id>/model_card.json   (7-hash provenance)
            +--> models/<model_id>/training_run.jsonl
            |
            v
+-----------------------+
|  eval/slm_eval_harness|   5 layers x 3 tiers + ablation runner
+-----------+-----------+
            |
            +--> eval/eval_report.json
            +--> eval/ablation_report.json (headline_delta block)
            +--> eval/eval_progress.jsonl
            |
            v
   EvalGatingValidator -> promotion via libv2 models promote
```

The corpus pipeline (`process_course.py`) is intentionally CPU-only and reproducible; the training pipeline (`train_course.py`) is GPU-bound and runs only after a course is imported into LibV2. They share schemas (`chunk_v4`, `courseforge_jsonld_v1`, `model_card`) and decision-capture infrastructure.

---

## Two Surfaces

### 1. Course-Processing Pipeline

Entry point: `Trainforge/process_course.py` (CLI: `python -m Trainforge.process_course --imscc … --course-code …`). Wired into the unified orchestrator as the `rag_training` workflow (`config/workflows.yaml::rag_training`).

Phases (logical, executed inside `CourseProcessor`):

| Phase | Output | Notes |
|-------|--------|-------|
| IMSCC unpack + HTML parse | per-page DOM, JSON-LD blocks, `data-cf-*` attrs | Priority chain: JSON-LD > `data-cf-*` > regex heuristics. |
| Chunking | `imscc_chunks/chunks.jsonl` (Phase 7c rename of `corpus/chunks.jsonl`) | `CHUNK_SCHEMA_VERSION = "v4"`. Optional content-hash IDs via `TRAINFORGE_CONTENT_HASH_IDS`. Optional shape enforcement via `TRAINFORGE_VALIDATE_CHUNKS`. |
| Boilerplate + WCAG canonicalization | filtered chunks | `Trainforge/rag/boilerplate_detector.py`, `Trainforge/rag/wcag_canonical_names.py`. |
| Pedagogy-graph build | `graph/pedagogy_graph.json` + `pedagogy/pedagogy_model.json` | `Trainforge/pedagogy_graph_builder.py`. |
| Concept-graph build (8 edge types) | `graph/concept_graph_semantic.json` | 3 taxonomic + 5 pedagogical edges (`schemas/knowledge/concept_graph_semantic.schema.json`). Optional TriG sibling via `TRAINFORGE_EMIT_TRIG`. |
| SHACL inference (optional) | additional `defined_by` edges | `Trainforge/rag/shacl_rule_runner.py`, gated by `TRAINFORGE_USE_SHACL_RULES`. |
| Assessment generation | `assessments.json` | `Trainforge/generators/assessment_generator.py` + `question_factory.py`. |
| Quality report | `quality/quality_report.json` | `Trainforge/rag/kg_quality_report.py::KGQualityReporter`. Includes the `assessments` dimension built by `Trainforge/generators/assessment_quality_report.py`. |
| Alignment pass | enriched chunks | `Trainforge/align_chunks.py`. Adds `teaching_role`, `prereq_concepts`, `learning_outcome_refs` via TF-IDF + optional LLM (`CurriculumAlignmentProvider`). |
| Training-pair synthesis | `training_specs/{instruction,preference}_pairs.jsonl` | `Trainforge/synthesize_training.py`. Provider-agnostic; see Provider Matrix below. |

### 2. Training Pipeline

Entry point: `Trainforge/train_course.py` (CLI: `python -m Trainforge.train_course --course-code <slug> --base-model <name>`). Wired into the unified orchestrator as the `trainforge_train` workflow.

Architectural call: training is a **post-import LibV2 stage**, not a step inside `process_course.py`. Re-training on a new base model never re-chunks the corpus -- the trainer reads `training_specs/*.jsonl` from the imported course and writes `models/<model_id>/` next to the existing `corpus/`, `graph/`, `pedagogy/`, `quality/`.

Steps inside `Trainforge/training/runner.py::TrainingRunner`:

| Step | Module | Output |
|------|--------|--------|
| Load course | runner | resolved paths to chunks / graphs / specs / vocabulary |
| Resolve base | `base_models.py::BaseModelRegistry` | HF repo + revision + chat template + recommended LoRA rank/alpha |
| Compose config | `configs/__init__.py::TrainingConfig` + per-base YAML | LoRA + DPO + LR knobs (see Trainforge/CLAUDE.md `TrainingConfig` table) |
| Hash provenance | runner | 7 SHA-256 hashes (see "7-hash provenance") |
| Dispatch trainer | `compute_backend.py::LocalBackend` -> `peft_trainer.py` | `adapter.safetensors` |
| Post-train eval | `eval/slm_eval_harness.py::SLMEvalHarness` | `eval_report.json`, `eval_progress.jsonl` |
| Ablation (default on) | `eval/ablation_runner.py` | `ablation_report.json` (skippable via `ED4ALL_SKIP_ABLATION`) |
| Emit model card | runner | `model_card.json` (validates against `schemas/models/model_card.schema.json` via `LibV2ModelValidator`) |
| Emit training_run.jsonl | `lib.decision_capture.DecisionCapture` | 4+ canonical training decision events |
| Gate enforcement | `lib/validators/eval_gating.py::EvalGatingValidator` | Critical fail blocks promotion (or advisory via `ED4ALL_GATE_ADVISORY`) |

Promotion workflow (separate, manual): `libv2 import-model … [--promote]` and `libv2 models promote` mutate `models/_pointers.json`. Schema: `schemas/models/model_pointers.schema.json`.

---

## Module Map

Top-level subdirectories under `Trainforge/`. Listed once -- detailed contracts live in `Trainforge/CLAUDE.md` and inline module docstrings.

### `parsers/`

IMSCC and HTML extraction. `imscc_parser.py` (zip + manifest), `qti_parser.py` (QTI 1.2 assessment parsing), `html_content_parser.py` (DOM + JSON-LD + `data-cf-*` extraction), `xpath_walker.py` (provenance helpers used by `chunks.jsonl::source.html_xpath`).

### `generators/`

Synthesis providers and the assessment generator.

| Module | Role |
|--------|------|
| `assessment_generator.py` | Top-level assessment orchestrator. |
| `question_factory.py`, `instruction_factory.py`, `preference_factory.py`, `summary_factory.py` | Type-specific factories. |
| `content_extractor.py` | Key-term / statement / relationship extraction from chunk text. |
| `assessment_quality_report.py` | Builds the `assessments` dimension of `quality_report.json`. |
| `_anthropic_provider.py` | Anthropic SDK paraphrase backend (default model `claude-sonnet-4-6`). |
| `_together_provider.py` | Together AI OpenAI-compatible paraphrase backend. |
| `_local_provider.py` | Local OpenAI-compatible (Ollama / vLLM / llama.cpp / LM Studio) paraphrase backend. |
| `_claude_session_provider.py` | Claude Code session-dispatch paraphrase backend (subagent). |
| `_curriculum_provider.py` | Teaching-role classification provider used by `align_chunks.py`. |
| `_openai_compatible_client.py` | Shared HTTP client (lenient JSON extraction, JSON mode, decision-capture hook). |
| `_session_budget.py` | `_BudgetTracker` + `_CircuitBreaker` for `claude_session` rebuild safety. |

### `eval/`

Five generic layers x three corpus-aware tiers. Composed by `slm_eval_harness.py::SLMEvalHarness`; ablation orchestrated by `ablation_runner.py`.

| Module | Layer / Role |
|--------|--------------|
| `holdout_builder.py` | Bloom-stratified holdout split + negative probes (Wave 108). Pins `holdout_graph_hash`. |
| `faithfulness.py` | Layer 1 -- held-out edge probes; surfaces `yes_rate`. |
| `invariants.py` | Layer 2 -- prereq ordering, Bloom level, misconception rejection. |
| `calibration.py` | Layer 3 -- confidence elicitation + ECE. |
| `baseline_compare.py` | Layer 4 -- paired-bootstrap CI of trained-vs-base. |
| `regression.py` | Layer 5 -- pointer-file-aware version-vs-version comparator. |
| `syntactic.py` | Tier-1 machine-verifiable checks (rdflib, pyshacl, SPARQL, predicate-URI strict mode). |
| `key_term_precision.py` | Tier-3 semantic (embedding / Jaccard) similarity. |
| `disambiguation.py` | Tier-3 `interferes_with`-anchored disambiguation. |
| `source_match.py` | Citation grounding -- multi-chunk ground-truth (Wave 108). |
| `negative_grounding.py` | Wave 108 yes-bias floor probe. |
| `property_eval.py` | Wave 109 per-property accuracy. |
| `qualitative_judge.py` | Optional judge-model rater for the headline-table `qualitative_score` column. |
| `lm_eval_wrapper.py` | LM Eval Harness shim. |
| `chunk_ids.py` | Wave 106 short-vs-full chunk-ID matching helpers. |
| `chunk_labels.py` | `ChunkLabelResolver` -- chunk ID -> human-readable label (Phase A 2026-04-30). Closes the leaked-ID-token-echo regression class. |
| `adapter_callable.py` | Wraps a trained adapter as a probe-callable using `eval_config.yaml` generation parameters. |
| `rag_callable.py` | Wraps base/adapter + LibV2 retriever as a RAG probe-callable. |
| `ablation_runner.py` | 4-row x 4-column headline table + 1x5 retrieval-method sweep. Emits `ablation_report.json`. |
| `headline_delta.py` | Wave 103 ED4ALL-Bench headline-sentence renderer. |
| `evidence_trace.py` | Per-probe trace emission (Wave 104). |
| `eval_config.py` | Loads per-course `LibV2/courses/<slug>/eval/eval_config.yaml`. |
| `hf_model_index.py` | Renders the README the HF upload publishes. |
| `verify_eval.py`, `reproducibility.py`, `diagnostics.py` | Auxiliary verification + reproducibility checks. |
| `configs/` | `rdf_shacl.yaml` (all three tiers on), `generic.yaml` (Tier 1 omitted). |

AblationRunner runs four model setups -- `base`, `base+rag`, `adapter`, `adapter+rag` -- and the `headline_delta.py` renderer extracts three procurement-grade numbers from the resulting table:

- `hallucination_reduction_pct = (base - adapter+rag) / base` over `hallucination_rate`
- `source_grounded_lift_x = adapter+rag / base` over `source_match`
- `accuracy_lift_x = adapter+rag / base` over `accuracy`

The `ChunkLabelResolver` indirection (Phase A 2026-04-30) exists because earlier eval probes interpolated raw chunk IDs (`shacl_551_chunk_NNNNN`) into question text, and adapters echoed the literal ID back instead of reasoning about content; the resolver maps each chunk ID to its `summary` (or first ~80 chars of `text`) so probe templates can substitute a semantically reasonable label.

`slm_eval_harness.py --smoke` runs the harness at N=3 prompts/evaluator with the real adapter, forces ablation off, and writes `smoke_eval_report.json` (sidecar, `smoke_mode: true`); `EvalGatingValidator` refuses to gate it. Wall-time target 2-5 min — verifies the eval pipeline before a 45-60 min full run.

### `training/`

| Module | Role |
|--------|------|
| `runner.py::TrainingRunner` | End-to-end orchestrator -- one course -> one model_id. Wave 90 contract surface. |
| `base_models.py::BaseModelRegistry` + `SUPPORTED_BASES` | The five short-name keys (`qwen2.5-1.5b`, `llama-3.2-1b`, `llama-3.2-3b`, `smollm2-1.7b`, `phi-3.5-mini`); pins HF repo + revision + chat template + LoRA defaults. `format_instruction()` resolves chatml / llama3 / phi3 templates. |
| `peft_trainer.py` | QLoRA SFT + DPO trainer. Loads in nf4 + double-quant when `use_4bit=true`; auto-selects `paged_adamw_8bit` + bf16/fp16. |
| `compute_backend.py` | `LocalBackend` (CUDA-required) + stubbed `RunPodBackend` for Wave 90 follow-up. |
| `configs/__init__.py::TrainingConfig` | Canonical config dataclass; per-base YAML in `configs/<short-name>.yaml` materializes production defaults. Schema mirrored in `schemas/models/model_card.schema.json::training_config`. |

### `rag/`

| Module | Role |
|--------|------|
| `kg_quality_report.py::KGQualityReporter` | Completeness / consistency / accuracy / coverage. Wrapped by `lib/validators/kg_quality.py::KGQualityValidator`. |
| `shacl_rule_runner.py` | Optional pyshacl-based inference path for the `defined_by` edge. Equivalence with the Python rule pinned by `tests/test_shacl_rules_defined_by.py`. |
| `boilerplate_detector.py` | Repeated-n-gram detection for chunk filtering. |
| `wcag_canonical_names.py` | Maps SC references in chunk text to canonical labels. |
| `libv2_bridge.py` | Cross-course RAG retrieval bridge. |
| `retrieval_benchmark.py` | Retrieval-precision benchmark used by Wave 102 ablation rows. |
| `typed_edge_inference.py`, `named_graph_writer.py` | Edge-type inference + `concept_graph_semantic.trig` named-graph writer. |
| `inference_rules/` | Rule package consumed by both the Python and SHACL paths. |

### `tests/`

Pytest suite (~140 files at the time of writing). Highest-traffic surfaces: `test_synthesize_training*.py`, `test_eval_*.py`, `test_*_synthesis_provider.py`, `test_pedagogy_graph_*.py`, `test_concept_graph_*.py`, `test_chunk_*.py`. The harness fixtures live in `tests/fixtures/` and `tests/_synthesis_fakes.py`.

### `agents/`

Markdown agent specs consumed by the orchestrator's subagent dispatch path. `training-synthesizer.md` is the spec the `claude_session` provider routes through (`config/agents.yaml::training-synthesizer.type: subagent`).

### `scripts/`

Operator CLIs. `pilot_synthesis.py` runs a small-N pilot synthesis pass and emits `pilot_report.md` with per-property coverage + template diversity (non-zero exit when any property is below floor). `pilot_report_helpers.py` factored shared report code so `synthesize_training.py` can regenerate the report incrementally during a long rebuild.

---

## Data Contracts

Trainforge emits and consumes the following schemas. Source of truth for all is `schemas/`.

| Schema | Path | Emitted by | Consumed by |
|--------|------|------------|-------------|
| Chunk v4 | `schemas/knowledge/chunk_v4.schema.json` | `process_course.py` | `align_chunks.py`, `synthesize_training.py`, `eval/`, `lib/validators/*` |
| Concept graph | `schemas/knowledge/concept_graph_semantic.schema.json` | `process_course.py` | `eval/`, `lib/validators/kg_quality.py` |
| Misconception | `schemas/knowledge/misconception.schema.json` | `process_course.py` (concept-graph build) | concept graph edges, instruction pairs |
| Source reference | `schemas/knowledge/source_reference.schema.json` | DART (originally) -> chunk + edge evidence arms | provenance audit |
| Courseforge JSON-LD | `schemas/knowledge/courseforge_jsonld_v1.schema.json` | Courseforge | Trainforge metadata extractor |
| Course | `schemas/knowledge/course.schema.json` | `process_course.py` (course.json) | LibV2 import |
| Objectives v1 | `schemas/knowledge/objectives_v1.schema.json` | upstream of `process_course.py` | `align_chunks.py` (LO refs) |
| Instruction pair | `schemas/knowledge/instruction_pair.schema.json` (+ `.strict.schema.json`) | `synthesize_training.py` | `peft_trainer.py` SFT, `LibV2ModelValidator` mock-corpus check |
| Preference pair | `schemas/knowledge/preference_pair.schema.json` | `synthesize_training.py` | `peft_trainer.py` DPO |
| Property manifest | `schemas/training/property_manifest.schema.json` (+ `rdf_shacl.yaml`, `generic.yaml.example`) | manifest authors | `lib/validators/property_coverage.py`, `eval/property_eval.py` |
| Model card | `schemas/models/model_card.schema.json` | `training/runner.py` | `lib/validators/libv2_model.py`, `libv2 import-model` |
| Model pointers | `schemas/models/model_pointers.schema.json` | `libv2 models promote` | `libv2 models list/eval` |
| Decision event | `schemas/events/decision_event.schema.json` | every LLM call site | strict-mode validation |

Full ontology map: `schemas/ONTOLOGY.md`.

---

## Provider Matrix

`Trainforge/synthesize_training.py` and `align_chunks.py` both consume LLM providers; the table below summarizes the synthesis surface. Operational details and licensing posture per provider live in `docs/LICENSING.md` (canonical reference) and `Trainforge/CLAUDE.md` (longer-form). One-line summary here only:

| Provider | Module | Use case | License-clean for training data? |
|----------|--------|----------|-----------------------------------|
| `mock` | factories under `generators/` | Plumbing tests only -- 30-template factory; trains a template-recognizer SLM. | N/A (no LLM call). Wave 107 `MOCK_PROVIDER_CORPUS` validator fails closed on promotion. |
| `anthropic` | `_anthropic_provider.py` | Highest-quality paraphrase via Anthropic SDK. | No -- Anthropic ToS restricts outputs from training-data use. |
| `claude_session` | `_claude_session_provider.py` | Paraphrase via the running Claude Code session (subagent dispatch). | No -- Anthropic Consumer Terms (Pro/Max). |
| `together` | `_together_provider.py` | Cloud OSS teacher (default `meta-llama/Llama-3.3-70B-Instruct-Turbo`) via OpenAI-compatible endpoint. | Yes -- Together ToS permits training-data use; underlying model license still applies. |
| `local` | `_local_provider.py` | Local OpenAI-compatible server (Ollama / vLLM / llama.cpp / LM Studio); default `qwen2.5:14b-instruct-q4_K_M` (Apache 2.0). | Yes -- recommended default for license-clean training data. |

The curriculum-alignment surface uses the same provider abstraction via `_curriculum_provider.py::CurriculumAlignmentProvider`, selected by `CURRICULUM_ALIGNMENT_PROVIDER`. The shared `_openai_compatible_client.py` backs both `together` and `local` and surfaces a single `llm_chat_call` decision capture per call.

Defaults: training-data synthesis prefers `--provider local` for an air-gapped clean corpus, or `--provider together` as the cloud fallback. Anthropic providers stay wired for backward compatibility but are not the recommended default for shippable training corpora.

---

## 7-Hash Provenance

`model_card.json::provenance` pins the run to seven canonical artifacts in the LibV2 course tree. Validated against `schemas/models/model_card.schema.json` on every emit; mismatch on `libv2 import-model` fails closed via `LibV2ModelValidator`.

| Hash field | Pins |
|------------|------|
| `chunks_hash` | `LibV2/courses/<slug>/imscc_chunks/chunks.jsonl` (Phase 7c rename of `corpus/chunks.jsonl`) |
| `pedagogy_graph_hash` | `LibV2/courses/<slug>/graph/pedagogy_graph.json` (or `pedagogy/pedagogy_graph.json` legacy) |
| `instruction_pairs_hash` | `LibV2/courses/<slug>/training_specs/instruction_pairs.jsonl` |
| `preference_pairs_hash` | `LibV2/courses/<slug>/training_specs/preference_pairs.jsonl` |
| `concept_graph_hash` | `LibV2/courses/<slug>/graph/concept_graph_semantic.json` |
| `vocabulary_ttl_hash` | `schemas/context/courseforge_v1.vocabulary.ttl` (Wave 96 fallback when a per-course TTL doesn't exist) |
| `holdout_graph_hash` | Bloom-stratified holdout split, SHA-256 over canonicalised payload (Wave 92). |

The seven hashes together make a run independently auditable: anyone with read access to the course tree can verify nothing under the model card has drifted since training.

---

## Eval Architecture

The eval harness composes five generic layers with three corpus-aware tiers. Two orthogonal axes; profiles in `Trainforge/eval/configs/` select which cells of the matrix run.

### Layers (generic)

| Layer | Module | Question answered |
|-------|--------|-------------------|
| 1 -- Faithfulness | `faithfulness.py` | Do trained generations align with held-out KG facts? |
| 2 -- Behavioral invariants | `invariants.py` | Prereq ordering, Bloom level, misconception rejection. |
| 3 -- Calibration | `calibration.py` | Confidence elicitation + ECE. |
| 4 -- Comparative delta | `baseline_compare.py` | Paired-bootstrap CI of trained-vs-base on the same prompts. |
| 5 -- Regression | `regression.py` | Pointer-aware version-vs-version comparator. |

### Tiers (corpus-aware)

| Tier | Coverage |
|------|----------|
| 1 -- Machine-verifiable | rdflib parses, pyshacl conformance, SPARQL syntax. Free, deterministic. |
| 2 -- Graph-derived | Holds out edges from the pedagogy graph; queries via prompt templates. |
| 3 -- Semantic | Embedding-or-Jaccard similarity (`key_term_precision.py`) + `interferes_with`-anchored `disambiguation.py`. |

Profiles: `eval/configs/rdf_shacl.yaml` (all three tiers on -- the canonical RDF/SHACL course profile) and `eval/configs/generic.yaml` (Tier 1 omitted -- not every domain has machine-verifiable surfaces).

### Ablation Runner

`eval/ablation_runner.py` produces two tables that ship with the model:

- **Headline table (4 rows x 4-5 columns)**: setups `base`, `base+rag`, `adapter`, `adapter+rag` x columns `accuracy`, `faithfulness`, `hallucination_rate`, `source_match` (+ optional `qualitative_score`). One retrieval method (`bm25`).
- **Retrieval-method sweep (1x5)**: setup pinned to `adapter+rag`, retrieval method varies across `bm25`, `bm25+intent`, `bm25+graph`, `bm25+tag`, `hybrid`. Columns `accuracy`, `faithfulness`, `source_match`, `mean_latency_ms`.

`eval/headline_delta.py` extracts the three procurement numbers from the headline table:

- `hallucination_reduction_pct`: `(base - adapter+rag) / base` on `hallucination_rate`.
- `source_grounded_lift_x`: ratio of `source_match` between `adapter+rag` and `base`.
- `accuracy_lift_x`: ratio of `accuracy` between `adapter+rag` and `base`.

…and renders the locked ED4ALL-Bench v1.0 marketing sentence at the top of the HF README.

A `+rag` row whose retrieved-chunks are empty for >50% of probes is stamped `setup.health = "rag_inert"` (Wave 105) and surfaces as a CRITICAL log line.

### Chunk-ID Indirection (Phase A 2026-04-30)

The `eval/chunk_labels.py::ChunkLabelResolver` was added after an audit found that `faithfulness.py`, `holdout_builder.py`, and `invariants.py` interpolated raw chunk-IDs (`shacl_551_chunk_NNNNN`) into probe text. Adapters echoed the literal ID back instead of reasoning about content -- 1441 chunk-id token matches in the cc07cc76 eval report; 0/22 correct on adapter+RAG faithfulness. The resolver maps each chunk ID to its `summary` (or first ~80 chars of `text`) so probes substitute a semantically reasonable label, and the adapter is tested on the question the eval was meant to ask.

The companion `eval/chunk_ids.py` (Wave 106) provides `is_chunk_id` / `normalize_chunk_id` / `chunk_ids_match` so short (`chunk_00270`) and full (`rdf_shacl_551_chunk_00270`) IDs compare equal across the harness, `source_match.py`, and `ablation_runner.py`.

### Holdout Builder + Negative Probes

`eval/holdout_builder.py` builds a Bloom-stratified holdout split and emits a `negative_probes[]` array (count balanced per-relation against the held-out positives) holding `(source, relation, target)` tuples that DON'T exist in the graph. `negative_grounding.py::NegativeGroundingEvaluator` scores no-rate against them and surfaces `negative_grounding_accuracy` -- the yes-bias floor that catches a "yes always" template-recognizer adapter (Wave 108).

---

## Validation Gates

Trainforge participates in three workflows -- `rag_training`, `textbook_to_course` (training_synthesis + libv2_archival phases), and `trainforge_train`. The full gate roster is in the **Active Gates** table in root `CLAUDE.md`; this section does not duplicate it.

Trainforge-specific validators (all under `lib/validators/`):

- `assessment.py`, `bloom.py`, `leak_check.py`, `content_facts.py`, `question_quality.py` -- assessment-quality gates on `rag_training::assessment_generation`.
- `assessment_objective_alignment.py` -- objective-coverage gate on `textbook_to_course::trainforge_assessment`.
- `min_edge_count.py`, `synthesis_diversity.py`, `synthesis_quota.py`, `property_coverage.py`, `synthesis_leakage.py` -- pre/post-synthesis gates on `textbook_to_course::training_synthesis`.
- `kg_quality.py` (wraps `Trainforge/rag/kg_quality_report.py::KGQualityReporter`) -- post-synthesis gate on `textbook_to_course::libv2_archival`. Wave 91 thresholds: completeness 0.95 / consistency 0.95 / accuracy 0.95 / coverage 0.5.
- `eval_gating.py::EvalGatingValidator` -- promotion gate on `trainforge_train::post_training_validation`. Critical-fails on faithfulness regression, yes-bias, no-bias, source-match drop, baseline regression, or per-property accuracy drop. Threshold table in `Trainforge/CLAUDE.md`.
- `libv2_model.py` -- model-card schema + weights file presence + hash agreement; wired as the `libv2_model` gate.
- `semantic_graph_rule_output.py` -- silent-zero-edge regression detector (Wave 82), enabled via `TRAINFORGE_VALIDATE_RULE_OUTPUTS`.

---

## Behavior Flags

Most-relevant Trainforge flags. The full table lives in root `CLAUDE.md` -- see "Opt-In Behavior Flags" there for every flag plus rationale.

| Flag | One-liner |
|------|-----------|
| `TRAINFORGE_VALIDATE_CHUNKS` | Enforce `chunk_v4.schema.json` on every chunk write; off = legacy-corpus tolerance. |
| `TRAINFORGE_CONTENT_HASH_IDS` | Chunk IDs become re-chunk-stable content hashes. |
| `TRAINFORGE_SCOPE_CONCEPT_IDS` | Concept node IDs become `{course_id}:{slug}` for cross-course disambiguation. |
| `TRAINFORGE_STRICT_EVIDENCE` | Drop the FallbackProvenance arm from the evidence discriminator. |
| `TRAINFORGE_SOURCE_PROVENANCE` | Edge evidence arms emit `source_references[]`. |
| `TRAINFORGE_USE_SHACL_RULES` | Run pyshacl-based `defined_by` inference instead of (or alongside) the canonical Python rule. |
| `TRAINFORGE_EMIT_TRIG` | Additionally write `concept_graph_semantic.trig` with per-rule named graphs. |
| `TRAINFORGE_SHACL_CLOSED_WORLD` | Merge closed-world shapes into the SHACL graph (Wave 88). |
| `TRAINFORGE_SEED_TECH_CONCEPTS` | Seed canonical W3C surface forms into `concept_tags` (Wave 82 Phase C). |
| `TRAINFORGE_VALIDATE_RULE_OUTPUTS` | Enable the `semantic_graph_rule_output` advisory gate (silent-zero detector). |
| `ANTHROPIC_SYNTHESIS_MODEL` | Override default Claude model used by `_anthropic_provider.py`. |
| `TOGETHER_API_KEY` / `TOGETHER_SYNTHESIS_MODEL` | Required key + optional model override for `--provider together`. |
| `LOCAL_SYNTHESIS_BASE_URL` / `LOCAL_SYNTHESIS_MODEL` / `LOCAL_SYNTHESIS_API_KEY` | Wire `--provider local` to a local OpenAI-compatible server. |
| `CURRICULUM_ALIGNMENT_PROVIDER` | Selects the LLM backend for `align_chunks.py` teaching-role classification. |
| `ED4ALL_SKIP_ABLATION` | Skip `ablation_runner.py` inside `TrainingRunner`. Set during fast iteration on the trainer surface. |
| `ED4ALL_GATE_ADVISORY` | Demote post-train eval gates to log-only; lets a regressing adapter through for diagnosis. Production runs leave this unset. |
| `TRAINFORGE_EVAL_PROGRESS_EVERY` | Cadence (default 25 calls) at which `slm_eval_harness.py::_EvalProgressTracker` logs progress to `eval_progress.jsonl`. |
| `TRAINFORGE_PROVENANCE_CORPUS` | Test-only -- absolute path to a regenerated `chunks.jsonl` for `tests/test_provenance.py`. |
| `DECISION_VALIDATION_STRICT` | Fail closed on unknown `decision_type` values. Enforced on every Trainforge LLM call site. |

The maintenance contract for these flags is in root `CLAUDE.md`: any new behavior flag that selects an LLM provider, model ID, or synthesis backend must land with a corresponding row in `docs/LICENSING.md`'s "Synthesis providers" table.

---

## CLI Entry Points

### Course processing (`rag_training` workflow)

```bash
# Unified CLI:
ed4all run rag_training --corpus course.imscc --course-name CHEM_101 --mode api

# Direct module invocation:
python -m Trainforge.process_course \
  --imscc /path/to/CHEM_101.imscc \
  --course-code CHEM_101 \
  --division ARTS --domain education --subdomain instructional-design \
  --output Trainforge/output/chem_101 \
  --import-to-libv2
```

### Training-pair synthesis (standalone)

```bash
# Default license-clean local provider (Ollama running qwen2.5:14b):
python -m Trainforge.synthesize_training \
  --course-dir LibV2/courses/rdf-shacl-551-2 \
  --provider local \
  --max-dispatches 2000

# Cloud OSS teacher (Together):
TOGETHER_API_KEY=… python -m Trainforge.synthesize_training \
  --course-dir LibV2/courses/rdf-shacl-551-2 \
  --provider together

# Smoke modes (Wave 120):
python -m Trainforge.synthesize_training --smoke-deterministic …  # ~0.7s, mock provider
python -m Trainforge.synthesize_training --smoke-paraphrase …    # ~10min, configured provider
```

### Training (`trainforge_train` workflow)

```bash
# Unified CLI:
ed4all run trainforge_train --course-code rdf-shacl-551-2 --base-model qwen2.5-1.5b

# Direct module invocation -- dry-run is CPU-only and produces a runner plan:
python -m Trainforge.train_course \
  --course-code rdf-shacl-551-2 \
  --base-model qwen2.5-1.5b \
  --dry-run

# Real run (requires pip install ed4all[training] + CUDA + HF_TOKEN for gated bases):
python -m Trainforge.train_course \
  --course-code rdf-shacl-551-2 \
  --base-model qwen2.5-1.5b \
  --backend local
```

### Promotion + introspection

```bash
libv2 import-model runtime/training/<run_id>/ --course rdf-shacl-551-2 [--promote]
libv2 models promote rdf-shacl-551-2 <model_id>
libv2 models list rdf-shacl-551-2          # stars current
libv2 models eval rdf-shacl-551-2 <model_id>
```

---

## Cross-References

- Project guide: `Trainforge/CLAUDE.md`
- Root protocols + Active Gates + full behavior-flag table: `/CLAUDE.md`
- Ontology + schema map: `schemas/ONTOLOGY.md`
- Licensing posture (per provider, per model): `docs/LICENSING.md`
- LibV2 import / model-pointers contract: `LibV2/CLAUDE.md`
- Courseforge JSON-LD output (Trainforge metadata input): `Courseforge/CLAUDE.md`
- DART source provenance (chunk `source.source_references[]` ancestry): `DART/CLAUDE.md`
