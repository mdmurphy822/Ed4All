# DART Naming-Surface Inventory & Purge Plan

> **Context.** The DART (AGPL) conversion engine was already **retired** and replaced by
> **SemantiK**. Only the *naming* remains. The migration deliberately **preserved** the
> `data-dart-*` HTML-attribute contract and the `dart:{slug}#{block_id}` sourceId scheme so
> downstream consumers (Courseforge staging, source-mapping, the Trainforge chunker, the Ask
> path, LibV2 manifests) did not break. This document inventories every surviving `dart` token,
> classifies it by blast radius, and proposes an atomic-vs-incremental purge plan.
>
> **This is a plan. No code is changed by this document.**

---

## 1. Summary

### 1.1 By category

| Category | Risk | Files | Occurrences | What it is |
|----------|------|------:|------------:|------------|
| `contract-html-attr` | **breaking** | 107 | 836 | `data-dart-*` HTML provenance attributes (the wire contract) |
| `contract-sourceid` | **breaking** | 175 | 1207 | `dart:{slug}#{block_id}` sourceId / CURIE scheme |
| `schemas` | **breaking** | 45 | 428 | JSON-Schema patterns / enum-const values / SHACL predicates / manifest fields |
| `phase-and-config` | **breaking** | 89 | 421 | `dart_conversion` phase name + `run_dart_chunking` helper/tool + registry keys |
| `tests` | **breaking** | 388 | 4135 | Test fixtures/assertions mirroring all of the above (tripwire, not source of truth) |
| `symbols` | internal | 179 | 957 | Python symbols (`stage_dart_outputs`, `DartMarkersValidator`, `harvest_dart_source_refs`, …) |
| `output-paths` | internal | 26 | 104 | `DART/output`, `DART_OUTPUT_DIR`, `dart_output_dir()`, `--dart-output-dir` |
| `env-flags` | internal | 28 | 95 | `DART_*` env vars (theta/council + GUI vision knobs) |
| `agent-names` | internal | 22 | 73 | `dart-converter`, `dart-chunker`, `dart-automation-coordinator` |
| `docs` | cosmetic | 53 | 482 | CLAUDE.md / ARCHITECTURE.md / agent-md narrative mentions |

### 1.2 By risk tier

| Tier | Categories | Nature |
|------|-----------|--------|
| **BREAKING** | contract-html-attr, contract-sourceid, schemas, phase-and-config, tests | Wire/data contract + persisted state + phase routing. Producer + consumer + schema + tests must flip in **one atomic commit**; some also require an on-disk data migration. |
| **INTERNAL** | symbols, output-paths, env-flags, agent-names | Python symbols, registry keys, dir names, operator env vars. Safe to do incrementally; a partial rename only breaks in-repo dispatch, not persisted data. Small external-MCP + CLI-flag sub-surfaces noted below. |
| **COSMETIC** | docs | Pure narrative. MUST follow the code rename in the same commit — never rename docs independently (that *creates* drift). |

> **Counting caveat.** File/occurrence counts **overlap heavily across categories** — a single
> file (e.g. `MCP/tools/pipeline_tools.py`, `Trainforge/chunker/helpers.py`, or any
> `test_stage_dart_outputs.py`) appears in several rows. The `tests` category (388 files / 4135
> occ) in particular re-counts tokens already listed under the four other breaking categories.
> **Do not sum the columns** — they are per-lens tallies, not a partition. The distinct-file
> universe is on the order of ~450–500 tracked files; the true load-bearing break surface is
> **~5 regex/schema definition sites** plus their emitters.

---

## 2. The wire-contract surface (BREAKING — atomic or it half-breaks)

Three coupled contracts. A rename of any one that misses a producer, consumer, schema, CSS
selector, **or** already-persisted on-disk artifact leaves the contract half-broken (HTML emits
one token, parser expects another; or new code writes `semantik:` while old LibV2 chunks hold
`dart:`).

### 2.1 `data-dart-*` HTML attributes (22 distinct tokens)

Frequency (raw occ): `block-id` 206, `pages` 110, `unit` 75, `source` 75, bare `data-dart-` 53,
`opener` 51, `block-role` 43, `page-kind` 40, `subclass` 35, `flow` 29, `opener-group` 24,
`repair` 22, `wcag` 15, `demoted-role` 15, `confidence` 14, `fabricated` 9, `repair-count` 8,
`cell-roles` 6, `mock` 3, `strategy` 1, `sources` 1, `repair-annotated` 1.

**PRODUCERS (write side — emit the attrs):**
- `lib/semantik/adapter.py` — **the single canonical emitter** (lines 597–801 stamp
  block-id/source/pages/page-kind/confidence/block-role/wcag/demoted-role/repair-count/opener/
  flow/opener-group/unit). 53 refs.
- `SemantiK/dart_semantic/assembler/pass_9a.py`, `pass_9c.py` — cascade assembler emit.
- `SemantiK/dart_semantic/qwen_specialists/ocr_repair.py` — `data-dart-repair` / `-repair-count`.
- `lib/semantik/subclassifier.py` (`data-dart-subclass`), `opener_classifier.py`,
  `vendor_ingest.py`, `scripts/semantik_rerender.py`.
- `MCP/tools/_content_gen_helpers.py` — **MIXED**: parses source `data-dart-block-id`, then
  re-emits into generated content for source-id carry-through.

**CONSUMERS (read side — parse/select on the attrs):**
- `Trainforge/chunker/helpers.py` — primary harvest, regex constants `_DATA_DART_BLOCK_ID_RE`
  etc. (37 refs) + `chunker.py`.
- `Trainforge/parsers/html_content_parser.py` — `data-dart-unit` / `-flow` role reader.
- `lib/validators/dart_markers.py` — the gate validator (regex on source/block-id, `re.IGNORECASE`).
- `lib/objectives/lo_map_builder.py`, `citation_reselect.py`, `lib/page_label.py` (pages/page-kind).
- `lib/aggregators/provenance_resolution.py`, `lib/validators/content_grounding.py`,
  `_block_rubric_helpers.py`, `chunk_wcag_status.py`.
- `lib/semantic_structure_extractor/semantic_structure_extractor.py`.
- `gui/services/source_materials.py` (`soup.find_all(attrs={"data-dart-block-id": True})`),
  `gui/services/imscc_service.py`.
- `MCP/tools/pipeline_tools.py`, `MCP/hardening/gate_input_routing.py`.
- **`Courseforge/templates/accessibility/dart_content.css`** — 22 CSS attribute selectors
  (`[data-dart-opener=…]`, `[data-dart-block-role=…]`, `[data-dart-opener-group=…]`). **Easy to
  miss** — omitting it silently breaks block styling with zero test failure.

**SCHEMA:** `schemas/knowledge/chunk_v4.schema.json` descriptions name the attrs **and** the
harvesting regex-constant names (couples schema prose to `helpers.py`);
`schemas/aggregators/provenance_resolution.schema.json`;
`schemas/taxonomies/exercise_apparatus_lexicon.json` mirrors `data-dart-unit`/`-flow`.

### 2.2 `dart:{slug}#{block_id}` sourceId / CURIE scheme

**Canonical definition (single source of truth):**
`schemas/knowledge/source_reference.schema.json:12` — `"pattern": "^dart:[a-z0-9_-]+#[a-z0-9_-]+$"`.
Pointed at by `schemas/knowledge/courseforge_jsonld_v1.schema.json:602`.

**PRODUCERS (mint `f"dart:{slug}#..."`):**
`Trainforge/chunker/chunker.py` + `helpers.py` (canonical chunker mint),
`Trainforge/synthesize_training.py:441`,
`MCP/tools/_content_gen_helpers.py` (:1449, :2674),
`MCP/tools/pipeline_tools.py` (:3809 and others — **MIXED**, also consumer at :114
`_CANONICAL_DART_SOURCE_ID_RE`),
`lib/generation/faq_page.py:455`, `lib/validators/content_grounding.py` (:407, :430 — normalizes
to canonical form), `gui/services/source_pdf.py` (slug derivation).

**CONSUMERS (regex / `startswith("dart:")` parse):**
`lib/validators/source_refs.py:83` (`SOURCE_ID_RE`) + `objective_source_refs.py`,
`Courseforge/router/inter_tier_gates.py:74`,
`MCP/tools/pipeline_tools.py:114` (`_CANONICAL_DART_SOURCE_ID_RE`, rejects malformed at mint),
`lib/retrieval/grounded_answer.py`, `gui/services/answer_render.py` (:361/:385),
`gui/services/imscc_service.py:80`,
`lib/objectives/chunk_window.py:97` + `lo_map_builder.py:57` (slug extraction),
`lib/aggregators/provenance_resolution.py:86`, `scripts/calibration_harness.py:1068`.

**PERSISTED DATA (the migration blocker):** every already-built LibV2
`courses/<slug>/dart_chunks/chunks.jsonl`, `course_manifest.json`, and `source_references[]`
holds literal `dart:` tokens. A prefix rename is a **data migration**, not just a code edit —
unless a read-side alias accepts both prefixes.

**~18 FALSE POSITIVES to exclude:** the `skip_dart:` bool param and `if dart:` locals
(`cli/commands/run.py`, `MCP/core/workflow_runner.py:4632`, `pipeline_tools.py:6730`,
`test_course_completeness.py:44`) are unrelated to the scheme.

### 2.3 Schema fields carrying `dart` (persisted → BREAKING)

| Field / token | File | Note |
|---------------|------|------|
| `^dart:[a-z0-9_-]+#[a-z0-9_-]+$` | `source_reference.schema.json:12` | canonical sourceId pattern |
| `source_dart_html_sha256` (required) | `chunkset_manifest.schema.json`, `chunk_v4.schema.json:360` | chunk→source join key, persisted |
| `dart_chunks_sha256` (required) | `course_manifest.schema.json:29` | `LibV2ManifestValidator` fail-closes on absence |
| `chunkset_kind` enum `"dart"` | `chunkset_manifest.schema.json:26`, `vector_index_manifest.schema.json:72`, `gold_set.schema.json` | dart/imscc discriminator |
| `dart_chunk_count`, `dart_block_ids_dropped` | `chunkset_drift_report.schema.json` | drift report fields |
| `dart_source_check`, `"dart_disagreement"` | `pair_audit_fields.schema.json`, `training_pair.shacl.ttl` | SHACL `ed4all:DartSourceCheck` / `dartEntailment` |
| `"dart_html"` enum | `textbook_structure.schema.json:22` | source-kind discriminator |
| `"dart"` / `"dart-conversion"` / `"dart-validation"` | `decision_event.schema.json`, `session_annotation.schema.json` | component/tool enum |

Prose-only `DART` mentions inside schema `description` strings are cosmetic and can be reworded
independently.

---

## 3. Internal surface (safe-ish — incremental OK)

### 3.1 Output paths (`output-paths`, internal)

| Token | Where | User-facing? |
|-------|-------|--------------|
| `DART/output`, `DART/batch_output`, `DART/inputs` | on-disk dirs (all **gitignored** data) | dir names surface in CLI help + docs |
| `DEFAULT_DART_OUTPUT_DIR = "DART/output"` | `cli/commands/run.py:95`, `gui/services/run_service.py:74` | **yes** — default for `--dart-output-dir` |
| `DART_OUTPUT_DIR = PROJECT_ROOT/"DART"/"batch_output"` | `MCP/tools/pipeline_tools.py:45` | internal (note: `batch_output`, a *different* subdir) |
| `DART_PATH`, `dart_output_dir()`, `_DATA_DIR_KEYS` incl `"dart-output"` | `lib/paths.py` (SOURCE OF TRUTH) | internal |
| `--dart-output-dir`, workflow param `dart_output_dir` | `cli/commands/run.py`, backup, GUI | **yes** — user CLI flag + persisted resume param |
| `.gitignore` / `.dockerignore` rules | `DART/output/`, `DART/batch_output/`, `DART/inputs/` | internal |

**FROZEN exception:** the ED4ALL_HOME data-dir **key** `"dart-output"` is documented
"path name preserved" (`docs/operations/docker.md:88`) and is a backup-manifest key
(`cli/commands/backup.py`, `backup-restore.md`, `test_backup_command.py`). Renaming it needs a
data migration — **leave frozen** as back-compat.

**Separate surface (decide independently):** `MCP/tools/analysis_tools.py:32`
`DART_OUTPUT = TRAINING_DIR/"dart"` and `trainforge_tools.py:865` point at a
`training-captures/dart` analysis subdir, **not** `DART/output`.

### 3.2 Phase name & chunking helper (`phase-and-config`)

- **`dart_conversion` — a real PHASE NAME and a WIRE CONTRACT.** Declared once at
  `config/workflows.yaml:1402` (producer). Consumers route on the **exact string**:
  `workflow_runner.py` INPUTS_FROM / phase→gate maps (31 refs, keyed
  `("phase_outputs","dart_conversion",…)`), `cli/commands/run.py:1183`
  (`phase.name == "dart_conversion"`), `gate_input_routing.py`, plus `depends_on`/`inputs_from`
  cross-refs inside `workflows.yaml`. **Persisted:** phase-name string is embedded in
  `phase_outputs` dict keys, `state/runs/*/checkpoints/`, and resume sidecars → an in-flight
  `--resume` after a bare rename breaks without a state migration. Hence breaking, not internal.
- **`run_dart_chunking` / `_run_dart_chunking`** — NOT a phase (the phase is `chunking`). Helper
  + MCP-registry tool symbol. Defined/registered in
  `pipeline_tools.py::_build_tool_registry`; tool-schema entry `tool_schemas.py:745`; routing
  target `executor.py:186` (`"dart-chunker": "run_dart_chunking"`). The registry **key string**
  is a routing wire-contract → change `executor.py` + `tool_schemas.py` + `pipeline_tools.py`
  together, but no persisted data. Internal.
- On-disk artifact path `LibV2/courses/<slug>/dart_chunks/chunks.jsonl` (produced by
  `_run_dart_chunking`, consumed by LibV2 manifest validator + `backfill_dart_chunks.py`) is a
  **filesystem-path contract** — treat like the persisted schema fields.

### 3.3 Symbols (`symbols`, internal — external-MCP sub-surface)

`stage_dart_outputs`, `validate_dart_markers`, `_run_dart_chunking`, `DartMarkersValidator`,
`harvest_dart_source_refs`, `build_dart_block_offset_index`, `resolve_dart_refs_for_chunk`,
`parse_dart_pages_attr`, `_build_dart_markers`, etc.

- Registry/config **string keys** (`stage_dart_outputs`, `run_dart_chunking`, `dart-chunker`,
  `dart_markers`, `validate_dart_markers`) couple YAML + executor + tool registry → **one
  commit** or phase dispatch silently breaks. Internal (not persisted).
- **`stage_dart_outputs` + `validate_dart_markers` are `@mcp.tool()`-decorated** → visible to any
  external MCP client. Renaming is a **minor external-API break** (keep an alias if any client
  depends on them).
- Two symbol families bleed into §2 and must move with their contract, not this pass:
  `data_dart_*` attr readers / `parse_dart_pages_attr`, and the persisted manifest fields
  `source_dart_html_sha256` / `dart_chunks_sha256`.

### 3.4 Env flags (`env-flags`, internal — some dead)

Canonical env-flag set (9): `DART_ALLOW_THETA_STUB`, `DART_SEMANTIC_MODEL_DIR`,
`DART_STRUCTURE_ADAPTER_DIR`, `DART_THETA_DEVICE` (already a **back-compat alias** for
`SEMANTIK_THETA_DEVICE`), `DART_REQUIRE_THETA_MODEL` (**RETIRED** — only history-note mentions),
and four GUI knobs `DART_VISION_PROVIDER`, `DART_CLAUDE_MODEL`, `DART_LLM_CLASSIFICATION`,
`DART_VISION_MODEL`.

- Theta/council knobs are **operator-set** (consumers in `SemantiK/dart_semantic/theta/*`,
  `council/structure.py`; producer only in `scripts/calibrate_theta.py`).
- The four GUI knobs are **PRODUCER-ONLY** in the tracked tree (`gui/env_catalog.py` renders them;
  no in-repo consumer — their consumer was the retired AGPL DART engine).
  `DART_LLM_CLASSIFICATION` is documented **retired** (superseded by
  `SEMANTIK_SPECIALIST_PROVIDER`) yet still emitted — **likely dead → delete, not rename**.
  `DART_VISION_MODEL` is explicitly a non-var.
- User-facing: these appear as GUI settings + flag-table doc rows; the untracked operator profile
  `~/ed4all-spark.env` may set some → keep a back-compat read on rename.

### 3.5 Agent names (`agent-names`, internal)

`dart-converter`, `dart-chunker`, `dart-automation-coordinator`. Producer =
`config/agents.yaml` (declares keys + `source:` md pointers). Consumers = `executor.py`
`AGENT_TOOL_MAPPING`, `config/workflows.yaml` phase `agents:` lists,
`pipeline_tools.py:28423` fallback map, and ~6 test files asserting exact strings.
Renaming should also rename `Courseforge/agents/dart-automation-coordinator.md` (token in
filename) and the referenced `scripts/dart-batch-processor/` dir.

> **ARCHITECTURE.md / agents.yaml currently document these agent names as
> "deliberately kept for wire-contract continuity."** A full purge overrides that decision —
> confirm with the owner before flipping, since it contradicts a standing note.

---

## 4. Cosmetic surface (`docs`, 53 files / 482 occ)

All markdown; zero code. **Never rename independently** — sequence into the same commit as the
code contract or it *creates* drift. Highest-count files:

| File | occ | File | occ |
|------|----:|------|----:|
| `SemantiK/CLAUDE.md` | 69 | `Courseforge/CLAUDE.md` | 25 |
| `schemas/ONTOLOGY.md` | 45 | `CLAUDE.md` | 23 |
| `ARCHITECTURE.md` | 32 | `Courseforge/agents/dart-automation-coordinator.md` | 22 (+ filename) |
| `Courseforge/agents/textbook-ingestor.md` | 29 | `Courseforge/agents/source-router.md` | 19 |
| `SemantiK/architecture.md` | 28 | `docs/operations/pipeline-invocation.md` | 15 |

Plus ~30 long-tail files (1–5 hits each) and `Trainforge/CLAUDE.md` (14), `docs/validation/gates.md`
(11), `docs/operations/behavior-flags.md` (11), `LibV2/CLAUDE.md` (9), `gui/README.md` (9).

**Keep-as-is:** ~187 lines of bare narrative uppercase **DART** (retired-engine provenance —
"SemantiK replaced DART", "wire contract preserved from DART"). This is intentional history;
reword to "legacy DART" at most.

---

## 5. Recommended rename mapping

| Old token | Proposed new token | Compat shim to de-risk? |
|-----------|-------------------|--------------------------|
| `data-dart-*` (HTML attrs) | `data-semantik-*` (or `data-sk-*`) | **Yes** — dual-read in `Trainforge/chunker/helpers.py` regexes + `dart_markers.py` (accept old **and** new) so pre-existing HTML still chunks. Emit new only. |
| `dart:{slug}#{block_id}` (sourceId) | `semantik:{slug}#{block_id}` | **Yes (strongly)** — read-side accept `^(dart\|semantik):…` in `source_refs.py`, `pipeline_tools.py:114`, answer/imscc parsers; mint new-only. Avoids a full LibV2 data migration. |
| `^dart:[a-z0-9_-]+#…$` pattern | `^(dart\|semantik):…$` then `^semantik:…$` | Two-step: widen pattern first (accepts both), migrate data, then tighten. |
| `source_dart_html_sha256` | `source_html_sha256` | Migration or additive alias field; `LibV2ManifestValidator` reads either. |
| `dart_chunks_sha256` (required) | `chunks_sha256` | Additive: emit both keys one release, then drop old. |
| `chunkset_kind: "dart"` | `"semantik"` | Accept both in discriminator conditionals during transition. |
| `dart_conversion` (phase) | `conversion` (or `semantik_conversion`) | **Yes** — alias map in `workflow_runner` so old checkpoints/resume sidecars still resolve. |
| `run_dart_chunking` / `_run_dart_chunking` | `run_semantik_chunking` / `_run_semantik_chunking` | Registry alias key for one release. |
| `stage_dart_outputs` (MCP tool) | `stage_source_outputs` | **Yes** — keep old `@mcp.tool()` name as alias (external clients). |
| `validate_dart_markers` / `DartMarkersValidator` / `dart_markers` gate | `validate_source_markers` / `SourceMarkersValidator` / `source_markers` | change `workflows.yaml` gate id + validator dotted-path together. |
| `harvest_dart_source_refs` etc. | `harvest_source_refs` etc. | pure internal, no shim needed. |
| `DART/output`, `DART/batch_output`, `DART/inputs` | `conversion-output/…` (or `SemantiK/output/`) | dir move on deployed boxes; gitignored so no git churn. |
| `DEFAULT_DART_OUTPUT_DIR`, `DART_OUTPUT_DIR`, `DART_PATH`, `dart_output_dir()` | `…CONVERSION_OUTPUT_DIR`, `CONVERSION_PATH`, `conversion_output_dir()` | internal. |
| `--dart-output-dir` / param `dart_output_dir` | `--conversion-output-dir` / `conversion_output_dir` | **Yes** — accept old flag as hidden alias (user muscle memory + persisted resume params). |
| ED4ALL_HOME key `"dart-output"` | **KEEP** | frozen back-compat (documented "path name preserved"). |
| `dart_chunks/` (LibV2 dir) | `chunks/` (or `semantik_chunks/`) | filesystem contract → migrate with data or dual-read. |
| `dart-converter`, `dart-chunker`, `dart-automation-coordinator` | `semantik-converter`, `semantik-chunker`, `semantik-automation-coordinator` | rename `agents.yaml` + `executor.py` + `workflows.yaml` + `.md` file + `scripts/dart-batch-processor/` together. |
| `DART_ALLOW_THETA_STUB` | `SEMANTIK_ALLOW_THETA_STUB` | **Yes** — back-compat env read. |
| `DART_SEMANTIC_MODEL_DIR` | `SEMANTIK_THETA_MODEL_DIR` | back-compat read. |
| `DART_STRUCTURE_ADAPTER_DIR` | `SEMANTIK_STRUCTURE_ADAPTER_DIR` | back-compat read. |
| `DART_THETA_DEVICE` (alias) | **drop** (keep `SEMANTIK_THETA_DEVICE`) | already the alias; just remove. |
| `DART_VISION_PROVIDER/_CLAUDE_MODEL/_LLM_CLASSIFICATION/_VISION_MODEL` | **DELETE** (dead producer-only) | confirm no live consumer, then remove from `env_catalog.py`. |
| Narrative `DART` in docs | keep as "legacy DART" | historical provenance. |

---

## 6. Execution strategy

### 6.1 Must be ATOMIC (single commit each — contract half-breaks otherwise)

1. **`data-dart-*` rename** — all §2.1 producers + consumers + `dart_content.css` selectors +
   `chunk_v4.schema.json` prose + every test fixture/assertion that emits or matches the attrs.
   *De-risk:* land the dual-read regex first (separate prior commit) so the flip is emit-only.
2. **`dart:` sourceId rename** — `source_reference.schema.json` pattern + all §2.2 mint sites +
   all parse sites + schema tests. *De-risk:* widen pattern to `(dart|semantik):` first, then
   migrate LibV2 data, then flip mint + tighten.
3. **Persisted schema fields** (`source_dart_html_sha256`, `dart_chunks_sha256`, `chunkset_kind`,
   drift fields, SHACL predicates) — schema + emitter + validator + fixtures together, paired
   with the on-disk LibV2 migration.
4. **`dart_conversion` phase rename** — `workflows.yaml` + all `workflow_runner` routing maps +
   `run.py` equality gate + `gate_input_routing.py` + resume/checkpoint state migration + tests.
5. **Tests** — the ~388 test files are the tripwire; each must flip **in the same commit** as the
   producer/consumer it gates. They are not separable work.

### 6.2 Can be INCREMENTAL (safe to stage; partial only breaks in-repo dispatch)

- Internal Python symbols (`harvest_dart_source_refs`, `_build_dart_markers`, `DART_PATH`, …) —
  one PR each, run tests.
- Registry/config string keys (`run_dart_chunking`, `dart-chunker`, `dart_markers`,
  `stage_dart_outputs`) — one commit per key-family (executor + yaml + registry + tests together),
  but independent of the wire contract. Keep MCP-tool aliases for external clients.
- Env flags — add `SEMANTIK_*` with back-compat `DART_*` read; delete the 4 dead GUI knobs.
- Output paths — rename dirs + constants + CLI flag alias; move gitignored data on deployed boxes.
- Docs — **must ride along** with whichever code commit renames the token they describe (never
  standalone).

### 6.3 The gitignored-data nuance

`DART/output`, `DART/batch_output`, `DART/inputs`, and `LibV2/.../dart_chunks/` are all
**gitignored** — renaming them is **cosmetic for git** (no tracked bytes change). But:
- The **hardcoded path** `MCP/tools/pipeline_tools.py:45` `DART_OUTPUT_DIR = PROJECT_ROOT/"DART"/"batch_output"`
  (and the `DEFAULT_DART_OUTPUT_DIR` in `run.py` + `run_service.py`, and `lib/paths.py`) **must
  change in code**, and the real directories must be **physically moved on every deployed box**
  (incl. `ED4ALL_HOME` relocations). Code edit + ops migration are separate steps.
- The persisted `dart:` tokens and `*_dart_html_sha256` fields inside gitignored LibV2 corpora
  are the real migration cost — a code-only rename without a corpus rewrite (or read-side alias)
  makes every existing course fail its manifest/source-ref gates.

### 6.4 Suggested ordering

1. **Owner decision gate:** confirm the purge overrides the standing "kept for wire-contract
   continuity" notes (ARCHITECTURE.md, agents.yaml) and that the frozen ED4ALL_HOME `"dart-output"`
   key stays frozen.
2. **Land dual-read shims** (accept `dart` *and* `semantik` on every consumer regex; widen schema
   patterns). No behavior change, fully back-compat — merge freely.
3. **Migrate persisted data** (LibV2 chunks/manifests: rewrite `dart:` → `semantik:` and the sha
   field names) with a one-shot migration script + `libv2 fsck` verify.
4. **Flip emitters to new-only** (adapter, chunker mint, schema patterns tightened) — atomic per
   §6.1, tests in the same commits.
5. **Incremental internal renames** (symbols, paths, agents, env flags, registry keys) — batch by
   subsystem.
6. **Docs + narrative sweep** last, riding the code commits, then re-run
   `behavior-flag-doc-sync` and `doc-sanitation-reviewer` to confirm zero drift.
7. **Remove the shims** once no producer emits and no corpus holds the old token.

---

## 7. Single biggest risk

**The `dart:{slug}#{block_id}` sourceId prefix is baked into already-persisted, gitignored LibV2
corpora** (`chunks.jsonl`, `course_manifest.json`, `source_references[]`) and joined to HTML via
`source_dart_html_sha256`. A code-only rename that misses the data migration doesn't fail
loudly — it makes **every existing course** fail its source-ref / manifest gates at the next
validation, and desyncs the emitted `data-dart-*` HTML value from the parsed `dart:` prefix. The
mandatory mitigation is the **read-side dual-accept shim landed *before* any emitter flips**, so
old data keeps resolving while new data adopts `semantik:`.
