# lib/ — Shared Libraries

`lib/` is the cross-subsystem layer: everything SemantiK, Courseforge, Trainforge,
LibV2, `MCP/`, `cli/`, and `gui/` all need, but that belongs to none of them. It
holds 809 Python modules, of which 432 are tests under the per-package `tests/`
dirs. This file is a **navigation map** — which subpackage owns what,
which helpers are single-source-of-truth, and the two contracts a newcomer must
know before touching a validator.

Nothing here is a subsystem entry point. If you are looking for the pipeline
itself, start at the root `CLAUDE.md`; for a specific engine, its own
`CLAUDE.md` (`SemantiK/`, `Courseforge/`, `Trainforge/`, `LibV2/`).

---

## Subpackage map

| Package | Owns |
|---------|------|
| `validators/` | Every validation-gate implementation (114 top-level modules + the `alignment/`, `bloom/`, `libv2/`, `pair/`, `shacl/` sub-packages). The largest package by far. See § Validator pattern. |
| `ontology/` | Canonical taxonomy/identity loaders over `schemas/taxonomies/*.json` — Bloom, slugs, LO ids, teaching roles, CURIEs, concept ids, edge predicates. See § Single-source-of-truth helpers. |
| `retrieval/` | The grounded-answer path and its evaluation harness: `grounded_answer.py` (the single entry point), `groundedness.py`, `citation_anchor.py`, `answer_composer.py`, `refusal.py`, `reranker.py`, plus the `gold_*` / `grounded_eval*` / `probe_authoring.py` eval-set tooling. |
| `generation/` | Authoring-side helpers used by the content-generation phases: `block_planner.py`, `block_catalog.py`, `key_terms.py`, `faq_page.py`, `prereq_sequencer.py`, `technique_modes.py`, plus the two cross-cutting run-control modules `llm_checkpoint.py` (resume sidecars) and `stop_control.py` (graceful stop). |
| `objectives/` | TO/CO synthesis support: `chunk_window.py`, `objective_dedup.py`, `chapter_anchor.py`, `citation_reselect.py`, `citation_sanitize.py`, `bloom_relevel.py`, `bloom_complement.py`, `source_backfill.py`, `restructure.py` (backs `ed4all objectives restructure`). |
| `aggregators/` | Post-loop, read-only rollups written after the workflow phase loop. One class per report — see the Aggregators table in the root `CLAUDE.md` for the output paths + schemas. |
| `classifiers/` | The in-process ML classifiers and their VRAM/device policy: `nli_classifier.py` (DeBERTa entailment + device resolution, bucket batching, OOM handling), `nli_microbatch.py` (the coalescing dispatcher), `bloom_bert_ensemble.py`, `bloom_zero_shot.py`. |
| `embedding/` | Embedding backend registry (`providers.py`) + `sentence_embedder.py` (`SentenceEmbedder`, `EmbeddingCache`, `try_load_embedder`, `EmbeddingDepsMissing`). A new embedding backend is a registry entry in `_EMBEDDING_PROVIDERS`, never a subclass. |
| `llm/` | LLM plumbing that is not a provider: `endpoints.py` (loader for `config/endpoints.yaml`, the unified endpoint registry), `rate_limiter.py`, `truncation_guard.py`, `oom.py`, `vram_doctor.py`, `vram_reclaim.py`. |
| `governance/` | Promotion/status policy: `course_status.py` (`compose_course_status` + the 5-value enum), `calibration_gate.py` (severity-flip resolution for calibration-gated validators), `procurement_evidence.py`, `source_coverage.py`. |
| `licensing/` | `teacher_roster.py` — the machine-readable SFT teacher-license roster and its fail-closed guards (`assert_export_licenses`, `assert_checkpoint_license`, `assert_nemotron_pin`, `stamp_pair_license`, `provider_verdict_roster`). Prose posture lives in `docs/LICENSING.md`. |
| `diagnostics/` | The `ed4all doctor` check framework: `core.py` defines `CheckResult` / `CheckContext` / `register(group, fn)` / `run_checks` / `resolve_exit_code`; the sibling modules register the groups `environment`, `provider`, `postmortem`, `gpu_profile`, `gpu` (from `vram.py`), `window` (from `serving_window.py`), and `seat` (vLLM seat topology, from `seat_schedule.py`). `run_env.py` registers no group — it is the seat/provider-key + local-synthesis-topology resolution helper the other checks call (`resolve_local_synthesis_topology` makes the `environment`/`window` groups probe `/v1/models` on a vLLM-seat host instead of ollama). |
| `semantik/` | Ed4All-side adapters and helpers for the SemantiK cascade — `adapter.py` normalizes a cascade result into the downstream HTML + sidecar contract; also `heading_classifier.py`, `table_structure.py`, `latex_mathml.py`, `math_fold.py`, `vendor_ingest.py`, `toc_frontmatter_detector.py`. The cascade itself lives in `SemantiK/`. |
| `semantic_structure_extractor/` | `SemanticStructureExtractor` — staged HTML → `textbook_structure.json` (chapters/sections/blocks), plus `resegment.py` and the `core/`, `analysis/`, `formats/`, `transformers/` submodules. |
| `importers/` | `docs_corpus.py` + `_markdown.py` — the deterministic, LLM-free Markdown/docs-tree importer behind `ed4all import-docs`. |
| `bloom_labels/` | `harvester.py` — the deterministic artifact-asserted Bloom-label harvester behind `ed4all harvest-bloom-labels`. |
| `assessment/` | `irt_difficulty.py` — item-difficulty estimation helpers. |
| `utils/` | Small stdlib-only helpers: `hashing.py`, `jsonl.py`, `jsonschema.py` (cached Draft 2020-12 builders), `html_text.py`, `html_balance.py`, `stats.py`. |
| `testing/` | Test-harness guards, not production code: `no_network.py`, `reachability.py`. |
| `tests/` | Cross-cutting tests for the top-level `lib/*.py` modules. Package-scoped tests live beside their package (`lib/validators/tests/`, `lib/ontology/tests/`, …). |

---

## Top-level modules

Flat `lib/*.py` files are the cross-cutting primitives — capture, storage,
integrity, path/security discipline, and hardware lifecycle.

**Decision capture + training data**
`decision_capture.py` (`DecisionCapture`, the helper every LLM call site must
wire up), `streaming_capture.py`, `trainforge_capture.py`, `validation.py`
(schema validation of decision records), `quality.py` (capture quality scoring),
`constants.py`, `leak_checker.py` (answer-key leakage detection).

**Paths + security**
`paths.py` — the single source of truth for project paths; honors `ED4ALL_ROOT`
(code root) and `ED4ALL_HOME` (relocatable data root). Import paths from here
rather than recomputing them. Also `path_constants.py`, `secure_paths.py`
(traversal / Zip-Slip guards), `secrets_filter.py` (redaction before logging),
`write_facade.py` (validated + atomic writes with an audit trail).

**Integrity + run lifecycle**
`hash_chain.py` (tamper-evident append-only event logs), `provenance.py`,
`content_store.py` (content-addressed blobs), `run_finalizer.py`,
`replay_engine.py`, `state_manager.py` and `file_lock.py` (atomic /
`flock`-guarded state), `tool_registry.py`, `error_taxonomy.py` (the canonical
tool error envelope: `success` / `error_code` / `error`).

**LibV2 access**
`libv2_storage.py` (unified storage interface), `libv2_fsck.py` (integrity
check + repair), `course_identity.py` (one canonical slug + `course_id` per run).

**Hardware lifecycle**
`gpu_lifecycle.py` (phase-boundary GPU lease — load, work, release),
`vllm_container_lifecycle.py` (per-seat container lease, seat registries, the
seat-schedule reconciler).

**Text sanitation**
`chunk_heading_sanity.py`, `textbook_title_sanitize.py`, `page_label.py`
(the shared source-page citation formatter).

---

## Single-source-of-truth helpers (`lib/ontology/`)

Prefer these over hardcoded verb lists, slug regexes, or ID formats. The
underlying data lives in `schemas/taxonomies/*.json`, so the schema — not the
code — is authoritative.

| Module | Canonical API |
|--------|---------------|
| `bloom.py` | `get_verbs()`, `get_verb_objects()`, `detect_bloom_level(text)`, `detect_bloom_verbs(text)`, `bloom_to_cognitive_domain(level)`, `cognitive_domain_enum()`. |
| `slugs.py` | `canonical_slug(text)` — the one slug helper. |
| `taxonomy.py` | `load_taxonomy(name)` — generic cached JSON-taxonomy loader. |
| `teaching_roles.py` | `(component, purpose) → role` mapping. |
| `learning_objectives.py` | LO identity: `mint_lo_id`, `validate_lo_id`, `hierarchy_from_id`, `split_terminal_chapter`. The `^[A-Z]{2,}-\d{2,}$` pattern mirrors `schemas/knowledge/courseforge_jsonld_v1.schema.json`. |

The rest of `ontology/` is the knowledge-graph vocabulary: `concept_id.py`,
`concept_classifier.py`, `concept_tagging.py`, `concept_node_merge.py`,
`curie_discovery.py` / `curie_extraction.py`, `edge_kind.py`,
`edge_predicates.py`, `edge_slug_normalizer.py`, `relation_templates.py`,
`jsonld_context_loader.py`, `lo_backlink.py`, `terminal_coverage.py`,
`framework_blocks.py`, `content_types.py`, `labels.py`, `aliases.py`.

---

## Validator pattern

A validator is a plain class satisfying the `Validator` Protocol in
`MCP/hardening/validation_gates.py`:

```python
class Validator(Protocol):
    name: str
    version: str
    def validate(self, inputs: Dict[str, Any]) -> GateResult: ...
```

`GateResult` carries `passed` plus a list of `GateIssue`s; `GateSeverity` is
`critical` / `warning` / `info` and `GateBehavior` is `block` / `warn` /
`fail_closed`.

**How a gate gets wired** — three places, all of which must agree:

1. **Implementation** — a class under `lib/validators/`.
2. **Config** — a `validation_gates:` entry in `config/workflows.yaml` giving
   `gate_id`, a dotted `validator:` path (e.g. `lib.validators.wcag.WCAGValidator`),
   `severity`, `threshold`, and `behavior` (`on_fail` / `on_error`). The optional
   `config:` block is forwarded into the validator's input dict. `GateConfig`
   is built by `GateConfig.from_dict`, and `ValidationGateManager` imports the
   class by dotted path at run time — there is no registry to update.
3. **Inputs** — `MCP/hardening/gate_input_routing.py::GateInputRouter` builds the
   `inputs` dict a gate receives. A new gate that needs an input the router does
   not already produce needs a builder there too.

Of the 210 configured gates, 195 resolve to `lib.validators.*`; the remaining 15
are Courseforge inter-tier gates (`Courseforge.router.inter_tier_gates.*`). The
authoritative per-gate table is `docs/validation/gates.md`; the per-wave landing
history is `docs/validation/gate-history.md`.

**Feature cache.** `lib/validators/feature_cache.py::BlockFeatureCache` is one
instance per gate-chain invocation, built by the executor immediately before the
gate loop and passed to `GateInputRouter.build(cache=...)`, which threads it
into the builders that declare a `cache` parameter and exposes it to validators
as `inputs["feature_cache"]`. It memoizes the block hydration, chunk
parse, HTML strip, sentence splits (kept distinct per splitter id), resolved
passages, and batched embeddings so 50+ gates compute them once. It is
phase-scoped, in-memory, never persisted; per-block entries are keyed by
`(block_id, sha256(content))` so a re-rolled block self-invalidates. Gated by
`ED4ALL_VALIDATION_FEATURE_CACHE`; when off, every seam sees `cache=None` and
takes its legacy self-compute path.

---

## Graceful-degrade contract (embedding/NLI validators)

Statistical-tier validators depend on the optional `[embedding]` pyproject
extras. They must **not** hard-crash a build when those extras are absent:

- Load the model through `lib.embedding.sentence_embedder.try_load_embedder()`
  rather than importing `sentence_transformers` directly.
- When the deps are missing, emit a **single warning-severity** `GateIssue`
  (`EMBEDDING_DEPS_MISSING`, or `NLI_DEPS_MISSING` on the NLI path) and return
  `passed=True`. The distinct code is what keeps a degrade distinguishable from
  a real finding in downstream rollups.
- `TRAINFORGE_REQUIRE_EMBEDDINGS=true` flips this to fail-closed: raise instead
  of degrading. `lib.embedding.sentence_embedder.is_strict_mode()` resolves it.

This is a *dependency-availability* escape hatch, not a design-intent fallback:
when the backend is present but broken, fail loudly. Never silently downgrade a
real check into a pass.

Validators on this contract include `objective_assessment_similarity`,
`concept_example_similarity`, `objective_roundtrip_similarity`,
`bloom_classifier_disagreement`, `co_terminal_alignment`, `source_coverage`,
`claim_support`, and `block_prose_entailment`.

Related: CUDA OOM inside a gate surfaces as a `VALIDATOR_OOM` warning by
default; `ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM` makes it fail the gate closed.

---

## Conventions

- **Behavior flags default OFF and parse with fallback.** Resolution helpers
  live next to the feature (`resolve_*` functions in `lib/classifiers/`,
  `lib/embedding/`, `lib/vllm_container_lifecycle.py`, …). Garbage or
  non-positive values fall back to the documented default rather than raising.
  Every new cross-cutting flag needs a row in `docs/operations/behavior-flags.md`
  and the root `CLAUDE.md` index; a flag that selects an LLM provider, model, or
  synthesis backend additionally needs a row in `docs/LICENSING.md`.
- **Registry, not subclass.** New LLM endpoints are rows in
  `config/endpoints.yaml`; new embedding backends are entries in
  `lib/embedding/providers.py::_EMBEDDING_PROVIDERS`; new vLLM seats are entries
  in the `ED4ALL_VLLM_CONTAINERS` / `ED4ALL_SEAT_BASE_URLS` registries.
- **Stop + resume are per-call-site obligations.** A new long-running loop lands
  with a fingerprinted resume sidecar (`lib/generation/llm_checkpoint.py`) and a
  stop-sentinel poll at its unit boundary (`lib/generation/stop_control.py`,
  which raises `GracefulStopRequested`). Worst-case loss is one in-flight LLM call.
- **Every LLM call site logs a decision.** Instantiate `lib.decision_capture.DecisionCapture`
  and emit at least one decision per call (per batch when batched), with a
  rationale that interpolates dynamic signals — static boilerplate rationales are
  rejected. Precedents: `docs/architecture/decision-capture.md`.
- **Tests live beside the package.** `lib/<pkg>/tests/` for package-scoped tests;
  `lib/tests/` only for the flat top-level modules.

---

## Related docs

- `docs/validation/gates.md` — per-gate table (the authoritative wiring reference)
- `docs/validation/validators.md` — long-form per-validator detail
- `docs/architecture/aggregators.md` — per-aggregator detail
- `docs/architecture/decision-capture.md` — capture contract + call-site precedents
- `docs/operations/behavior-flags.md` — root-owned cross-cutting flag detail
- `schemas/ONTOLOGY.md` — ontology map
- `docs/LICENSING.md` — provider/model licensing posture
