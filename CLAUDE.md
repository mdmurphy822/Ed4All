# Ed4All Hybrid Orchestrator

Unified orchestration system for SemantiK, Courseforge, Trainforge, and LibV2.

## Quick Start

### Canonical entry point

```bash
# Primary: run any workflow end-to-end via the unified CLI
ed4all run <workflow_name> --corpus <PATH> --course-name <NAME> [--mode local|api]

# Examples
ed4all run textbook-to-course --corpus textbook.pdf --course-name PHYS_101
ed4all run textbook-to-course --corpus ./pdfs/ --course-name BIO_201 --weeks 16
ed4all run rag_training --corpus course.imscc --course-name CHEM_101 --mode api
ed4all run textbook-to-course --corpus x.pdf --course-name T --dry-run   # plan only
ed4all run textbook-to-course --resume WF-20260420-abc12345               # resume

# --stop-after <phase>: halt the run cleanly AFTER the named phase
# completes, skipping every subsequent phase. The canonical "build a
# retrieval-ready course WITHOUT training synthesis" slice stops after
# imscc_chunking (before trainforge_assessment / training_synthesis /
# libv2_archival / finalization). The phase name is validated against
# the workflow's phase list (unknown name -> error). Default unset ->
# runs to completion.
ed4all run textbook-to-course --corpus pdfs/ --course-name PHYS_101 \
  --skip-training --stop-after imscc_chunking

# NVIDIA 70B-everywhere build profile (SETUP — gated on a later RUN
# discussion; nothing dispatches to NVIDIA by default). `--provider
# nvidia` (also via LLM_PROVIDER=nvidia) on a COURSEFORGE_TWO_PASS=true
# run redirects the block-routing YAML to
# Courseforge/config/block_routing.nvidia_large.yaml (rewrite tier on
# the hosted 70B `meta/llama-3.3-70b-instruct`; outline first draft
# stays local 7B), pins NVIDIA_LARGE_MODEL to the 70B (closes the
# 30B-nano registry-default leak), AND routes the textbook-synthesis
# seat (objective_extraction / course_planning / concept_extraction) to
# nvidia. The TRAINING seat (TRAINFORGE_SYNTHESIS_PROVIDER) is pinned
# LOCAL by this branch (licensing — the SLM training corpus must never
# route through Llama-3.3). All setdefault (explicit per-phase overrides
# win). The canonical cloud-model knob is NVIDIA_LARGE_MODEL / the YAML,
# NEVER COURSEFORGE_REWRITE_MODEL (dead on the cloud tier). Run --dry-run
# first for the "wired but not firing" routing preflight.
export COURSEFORGE_TWO_PASS=true
ed4all run textbook-to-course --provider nvidia --course-name PHYS_101 \
  --corpus slice.pdf --skip-dart --skip-training \
  --stop-after imscc_chunking --dry-run   # preflight: resolve+assert, NO dispatch

# Pin the course_planning phase to a previously-synthesized
# objectives JSON instead of re-dispatching the course-outliner
# subagent. Eliminates LLM-nondeterminism drift across re-runs that
# breaks chunk learning_outcome_refs continuity. Accepts both
# Courseforge synthesized form (terminal_objectives/chapter_objectives)
# and the LibV2 archive form (terminal_outcomes/component_objectives);
# the runner normalizes to the Courseforge form on disk before
# downstream phases consume it.
ed4all run textbook-to-course --corpus pdfs/ --course-name PHYS_101 \
  --reuse-objectives Courseforge/exports/PROJ-PHYS_101-.../01_learning_objectives/synthesized_objectives.json

# Reuse a prior SemantiK PDF→accessible-HTML conversion instead of
# re-running the (model-nondeterministic) SemantiK v2 cascade. When set
# AND prior `{stem}_accessible.html` + sidecars exist at the conversion
# output path, the cascade is skipped and the prior artifacts are reused
# (the re-run model-nondeterminism guarantee, analogous to
# --reuse-objectives). Mirrors the ED4ALL_REUSE_CONVERSION env var (the
# flag wins when both are set). See SemantiK/CLAUDE.md §3.3a.
ed4all run textbook-to-course --corpus pdfs/ --course-name PHYS_101 \
  --reuse-conversion

# Phase 5: stage-by-stage Courseforge two-pass subcommands. Re-run a
# single tier of the Courseforge two-pass pipeline against an existing
# project export. Pre-Courseforge phases (DART -> staging -> chunking
# -> objective_extraction -> source_mapping -> course_planning ->
# concept_extraction) pre-populate from disk via the synthesizer; non-
# whitelisted two-pass + post-Courseforge phases skip. See
# Courseforge/CLAUDE.md "Operator stage subcommands" for details.
export COURSEFORGE_TWO_PASS=true
ed4all run courseforge-outline --course-name PHYS_101              # outline tier only
ed4all run courseforge-validate --course-name PHYS_101             # validators only
ed4all run courseforge-rewrite --course-name PHYS_101 \
  --blocks assessment_item,objective                                # per-block-type rewrite
ed4all run courseforge --course-name PHYS_101 --force               # full two-pass slice
```

Modes:

- `--mode local` (default): uses the current Claude Code session as the LLM;
  no API key required. Phase workers are dispatched as subagents.
- `--mode api`: uses the Anthropic SDK directly (requires `ANTHROPIC_API_KEY`).
  Workers run as Python coroutines and call the SDK directly.

Environment toggles (override or supplement CLI flags):

| Env Var | Default | Purpose |
|---------|---------|---------|
| `LLM_MODE` | `local` | Chooses `local` or `api` if `--mode` isn't passed. |
| `LLM_PROVIDER` | `anthropic` | Provider in api mode (`anthropic` or `openai`; `openai` is stubbed, reserved for a later wave). |
| `LLM_MODEL` | per-provider | Model ID override (e.g., a specific Claude release). |
| `ANTHROPIC_API_KEY` | — | Required for api mode with Anthropic. |
| `OPENAI_API_KEY` | — | Reserved; OpenAI backend not yet implemented. |

### MCP Server
```bash
cd MCP
python server.py
```

### Control-Plane GUI

A no-stubs web control plane for the whole pipeline (upload→run, env/API keys,
per-task model routing incl. VLM/Ollama, course topics/objectives, retrieval +
adapter inference, Claude↔GUI activity bridge). Ships as the opt-in `gui` extra
(no heavy deps in the default install).

```bash
# One-click: build venv, install, serve, open browser (see gui/LAUNCH.md)
./run-gui.sh            # macOS / Linux   (run-gui.bat on Windows)

# Manual: install the extra, then launch
pip install -e '.[gui]'
ed4all gui              # serves http://127.0.0.1:8077
```

Full reference (six tabs, REST API, settings/secret persistence, model routing,
Claude integration): `gui/README.md`. Launcher flags + troubleshooting:
`gui/LAUNCH.md`. Containerized deploy (shared-netns `gui`↔`ollama` sidecar
compose, LibV2 bind-mounted at `/data/libv2` as the shared course library, GUI
image shipping `[gui,server,embedding]` on CPU torch): `docs/operations/docker.md`.

### Available Workflows
| Workflow | Description | Max Concurrent |
|----------|-------------|----------------|
| `textbook_to_course` | Full PDF → Course → Assessments pipeline | 10 |
| `course_generation` | Generate new course from objectives | 10 |
| `rag_training` | Trainforge assessment generation | 5 |
| `trainforge_train` | Train a course-pinned SLM adapter (post-import LibV2 stage) | 1 |
| `courseforge-outline` | Phase 5 stage subcommand — re-run only the outline tier (`content_generation_outline`) against an existing project export. Pre-Courseforge phases pre-populate from disk via `_synthesize_outline_output`; non-whitelisted two-pass + post-Courseforge phases skip via `courseforge_stage` whitelist. | 10 |
| `courseforge-validate` | Phase 5 stage subcommand — re-run only the inter-tier + post-rewrite validators against an existing project export (no LLM dispatch). Emits `02_validation_report/report.json` aggregating per-block pass/fail/escalated counts. | 1 |
| `courseforge-rewrite` | Phase 5 stage subcommand — re-run only the rewrite tier (`content_generation_rewrite` + `post_rewrite_validation`). Pairs with `--blocks <type1,type2>` for per-block-type re-execution scope; untouched blocks are byte-identical to the input. | 10 |
| `courseforge` | Phase 5 stage subcommand — re-run the full four-phase Courseforge two-pass slice (outline → inter-tier-validate → rewrite → post-rewrite-validate); skips post-Courseforge phases (packaging, libv2_archival, etc.). | 10 |

---

## Project Structure

```
Ed4All/
├── SemantiK/                # PDF to accessible HTML conversion (license-clean cascade)
├── Courseforge/             # Course content generation & packaging
├── Trainforge/              # Assessment-based RAG training (incl. canonical chunker)
├── LibV2/                   # Course content repository
├── MCP/                     # FastMCP server, orchestrator, IPC, tools
├── cli/                     # CLI commands (ed4all entry point)
├── lib/                     # Shared libraries & validators
├── config/                  # Workflow & agent configs
├── schemas/                 # JSON schemas for validation
├── state/                   # Shared state & progress tracking
├── training-captures/       # Decision capture output
├── ci/                      # CI integrity checks
└── .github/                 # CI/CD workflows
```

Per-subsystem layout lives in each subsystem's `CLAUDE.md` (see § Individual Project Guides).

### Test fixtures

Fixtures live with the code they exercise. A fixture that exercises only one
project lives under `<Project>/tests/fixtures/` (Trainforge mini-courses,
Courseforge sample HTML, schema-validation snapshots). A fixture that exercises
two or more projects (e.g. an end-to-end pipeline fixture that flows
DART → Courseforge → Trainforge → LibV2) lives under the top-level
`tests/fixtures/`. Schema-validation fixtures (snapshots that exist solely to
confirm a JSON Schema or SHACL shape accepts/rejects a known shape) live under
`schemas/tests/fixtures/<wave>/` and are wave-namespaced. Fixtures must NOT
cross from one project's `tests/fixtures/` into another project's test code; if
a Trainforge test needs a Courseforge IMSCC fixture, build the IMSCC at
fixture-load time from a corpus-fixture builder script (e.g.
`tests/fixtures/pipeline/build_fixture_pdf.py`) rather than pinning the
cross-project path. Fixture files are tracked in git; large binaries (e.g. PDFs
over 1MB) ship as regenerable scripts under `tests/fixtures/regen/` instead of
the bytes themselves. Active fixture roots: `tests/fixtures/` (cross-project
end-to-end), `Trainforge/tests/fixtures/` (mini-course corpora),
`Courseforge/scripts/tests/fixtures/` (sample HTML + IMSCC), and
`schemas/tests/fixtures/` (per-wave schema snapshots).

---

## Orchestrator Protocol

### Phase 1: Planning (NO EXECUTION)

Planning agent creates comprehensive todo list:
- Analyze requirements
- Break into discrete tasks
- Assign to appropriate agents
- **NO file creation, NO code execution**

### Phase 2: Load TodoWrite

TodoWrite is the **single source of truth**:
- All agents read from TodoWrite
- All agents update TodoWrite
- Status tracking: `pending` -> `in_progress` -> `completed`

### Phase 3: Batch Execution

Execute via parallel agent dispatch:
- **Maximum 10 simultaneous Task calls per batch**
- Wait for ALL batch completions before next batch
- Use `poll_task_completions()` to check status

### Phase 4: Quality Validation

Every artifact validated before finalization:
- SemantiK: WCAG compliance check
- Courseforge: IMSCC validation
- Trainforge: Assessment quality scoring

### Phase 5: Packaging

Final packaging and export:
- Update GENERATION_PROGRESS.md
- Export training captures
- Archive logs

---

## Decision Capture

### CRITICAL REQUIREMENT

**ALL LLM decisions MUST be logged** to `training-captures/` in JSONL format.

### Required Fields

Every decision event MUST include:
- `decision_type`: Category of decision (e.g., `content_selection`, `question_generation`, `form_data_backfill_session`, `family_completeness_decision`). Canonical enum: `schemas/events/decision_event.schema.json`.
- `decision`: The actual choice made
- `rationale`: Why this decision was made (**minimum 20 characters**)

### Using Decision Capture

Helper: `lib/decision_capture.py::DecisionCapture` — instantiate with `course_code`, `phase`, `tool`, then call `log_decision(decision_type, decision, rationale, alternatives_considered=[...])`. Output lands under `training-captures/<tool>/<COURSE_CODE>/phase_<phase>/decisions_*.jsonl`.

Canonical decision-event shape: `schemas/events/decision_event.schema.json`. Long-form rationale + LLM call-site precedents: `docs/architecture/decision-capture.md`.

---

## Individual File Protocol (MANDATORY)

### ONE Agent = ONE File

Each agent works on exactly ONE file at a time:
- No shared file editing
- No concurrent writes to same file
- Use file locking for state files

### Maximum Parallelism

```
Maximum 10 simultaneous Task calls per batch
```

### Batch Completion

Wait for ALL tasks in batch to complete:
```python
# CORRECT: Wait for batch
tasks = [dispatch_agent_task(...) for i in range(10)]
await poll_task_completions(workflow_id)  # Wait for all

# WRONG: Fire and forget
for i in range(50):
    dispatch_agent_task(...)  # No waiting!
```

---

## MCP Tool Reference

### Core File Tools

| Tool | Description |
|------|-------------|
| `list_directory` | List directory contents (READ_ONLY sandbox) |
| `read_file` | Read file contents (READ_ONLY sandbox) |
| `write_file` | Write to files (RESTRICTED sandbox: runtime/, state/) |
| `file_info` | Get file/directory metadata (READ_ONLY sandbox) |

**SemantiK tools** — see `SemantiK/CLAUDE.md` (PDF→accessible-HTML conversion; emits the Source-Provenance `data-dart-*` / `dart:{slug}#{block_id}` contract).

**Courseforge tools** — see `Courseforge/CLAUDE.md § MCP Tools` (includes Metadata Output contract: `data-cf-*` + JSON-LD).

### Orchestrator Tools

| Tool | Description |
|------|-------------|
| `create_workflow` | Create new workflow instance |
| `get_workflow_status` | Check workflow progress |
| `dispatch_agent_task` | Dispatch task to agent |
| `poll_task_completions` | Wait for task completions |
| `execute_workflow_task` | Execute a single workflow task |
| `complete_workflow_task` | Mark workflow task complete |
| `update_generation_progress` | Update progress file |
| `acquire_batch_lock` | Lock resource for batch |
| `release_batch_lock` | Release batch lock |

### GUI Tools

The Claude-interaction surface for the Control-Plane GUI. All nine operate on
the shared `state/gui/` store (`MCP/tools/gui_tools.py`), so a Claude session
and the GUI stay in sync. Full detail: `gui/README.md § Claude Code integration`.

| Tool | Description |
|------|-------------|
| `gui_get_settings` | Return the masked GUI settings doc |
| `gui_set_setting` | Deep-patch one setting at a dotted path (e.g. `model_routing.global.provider`) |
| `gui_list_runs` | List the GUI run registry (newest first) |
| `gui_get_run` | Return one run record |
| `gui_enqueue_run` | Write a run request (`status="requested"`) for the GUI to pick up |
| `gui_list_courses` | List Courseforge-export + LibV2 courses |
| `gui_get_objectives` | Return a course's synthesized objectives doc |
| `gui_post_event` | Append a `claude`-sourced event (shows in the Activity tab) |
| `gui_read_events` | Read activity events with `seq >= since` (sees human messages) |

**Trainforge tools** — see `Trainforge/CLAUDE.md § MCP Tools`.

### Pipeline Tools

| Tool | Description |
|------|-------------|
| `stage_dart_outputs` | Stage DART outputs for Courseforge |
| `get_pipeline_status` | Check pipeline progress |
| `validate_dart_markers` | Validate DART output markers |
| `archive_to_libv2` | Archive course artifacts to LibV2. Emits a top-level `chunker_version` field in `course_manifest.json` (resolved via `Trainforge.chunker.CHUNKER_SCHEMA_VERSION`) so LibV2 audits know which chunker shipped the corpus. |

**Pipeline-internal registry-only tools** (wired into `MCP/tools/pipeline_tools.py::_build_tool_registry` for workflow-phase dispatch; intentionally **not** decorated with `@mcp.tool()` — not reachable from external MCP clients):

| Tool | Phase | Purpose |
|------|-------|---------|
| `build_source_module_map` | `source_mapping` | TF-IDF-driven router that maps DART source blocks to Courseforge module pages. Output: `source_module_map.json`. |
| `extract_textbook_structure` | `objective_extraction` | Runs `SemanticStructureExtractor` over every staged DART HTML file and merges per-file chapter/section hierarchies into a single `textbook_structure.json`. |
| `plan_course_structure` | `course_planning` | Synthesizes canonical `TO-NN` / `CO-NN` learning objectives from the textbook structure and publishes `synthesized_objectives.json`. |

**Phase-name dispatch override** (`MCP/core/executor.py::_PHASE_TOOL_MAPPING`): six phases route by phase name, not agent name — `content_generation_outline` → `run_content_generation_outline`; `inter_tier_validation` → `run_inter_tier_validation`; `content_generation_rewrite` → `run_content_generation_rewrite`; `post_rewrite_validation` → `run_post_rewrite_validation`; `imscc_chunking` → `run_imscc_chunking`; `assessment_synthesis` → `run_assessment_synthesis`. Validator-only phases declare `agents: []` in `config/workflows.yaml`; `workflow_runner._create_phase_tasks` synthesizes a virtual `phase-handler` task only when the phase appears in this map. The mapping cannot be inferred from YAML.

### Analysis Tools

| Tool | Description |
|------|-------------|
| `analyze_training_data` | Analyze training capture data |
| `get_quality_distribution` | Get quality score distribution |
| `preview_export_filter` | Preview export filter results |

---

## Shared State

### GENERATION_PROGRESS.md

Location: `state/GENERATION_PROGRESS.md`

Central progress tracking file:
- Active workflows table
- Component status tables
- Batch locks table
- Error log

### File-Based IPC

Use `StatusTracker` for multi-terminal coordination:
```python
from MCP.ipc.status_tracker import StatusTracker

tracker = StatusTracker()  # defaults to state/status/
tracker.update_status("content_generator", "IN_PROGRESS",
                      worker_id="W001", details={"file": "Module_3.html"})
```

---

## Workflow Execution

### Course Generation Workflow

`planning → content_generation (batches of 10) → assessment_synthesis (W10, optional QTI/discussion/assignment emit) → packaging (IMSCC) → validation (QA + WCAG) → finalization`. Full phase shapes: `config/workflows.yaml::course_generation`.

### Other workflows

`rag_training` (extraction → indexing → assessment_generation → validation) — see `config/workflows.yaml` for canonical phase shapes.

### Textbook-to-Course Workflow

```
1. dart_conversion
   └── Convert PDF textbooks to accessible HTML (multi-source synthesis)

2. staging
   └── Stage DART outputs to Courseforge inputs

3. objective_extraction
   └── Parse staged DART HTML into textbook_structure.json (chapters,
       sections, content blocks); auto-scales duration_weeks to max(8,
       chapters) when --weeks is unset.

4. source_mapping
   └── Map DART source blocks to Courseforge module pages; emits
       source_module_map.json consumed by content_generation.

5. course_planning
   └── Synthesize canonical TO-NN / CO-NN learning objectives from
       textbook_structure; emits synthesized_objectives.json.
       WS5 §3.2: re-scales duration_weeks from the extractor's
       chapter-driven max(8, len(chapters)) to the TERMINAL-objective-
       driven max(8, num_tos) — one week-block per WS1-clustered TO —
       when --weeks is unset. The TO/cluster count is the authoritative
       pacing signal; COs then distribute WITHIN each TO's week-block
       via the §2.2 coverage-safe ceil-stride slicer (no CO dropped).
       Falls back to the legacy CO-count formula
       max(8, ceil(len(chapter_objectives) / WAVE18_COS_PER_WEEK)) only
       when no TOs are available. Skipped for --weeks and
       --reuse-objectives runs (operator's pacing decisions are
       preserved verbatim). The per-week CO placement cap is the
       separate, UNCONDITIONAL ED4ALL_COS_PER_WEEK_CAP (default auto).

6. content_generation
   └── Generate course content modules (parallel batches of 10). Every
       emitted sourceId must resolve against the DART staging manifest
       (source_refs gate).

7. assessment_synthesis (optional)
   └── W10 — synthesize grounded quizzes (multiple_choice /
       multiple_response / true_false / numeric fill_in_blank), short
       assignments, and discussion prompts from the DART chunkset and
       emit QTI 1.2 / imsdt / assignment XML + manifest.json into
       <export>/06_assessments/ so packaging picks them up as canonical
       IMS CC resource types (not inert HTML). Validator-only phase
       (agents: []) routed by phase NAME to run_assessment_synthesis
       via _PHASE_TOOL_MAPPING. Gated critical on qti_well_formed
       (QTI well-formedness/XSD/answer-key) + assessment_objective_alignment
       (grounding), warning on discussion_assignment_grounded. Runs
       before packaging; skipped via generate_assessments=false.

8. packaging
   └── Package course as IMSCC via the mature multi-file packager.

9. trainforge_assessment (optional)
   └── Generate assessments from the IMSCC package. Fails closed if any
       assessment objective_id isn't covered by a chunk's
       learning_outcome_refs.

10. training_synthesis (optional)
   └── Synthesize instruction + preference training pairs from the
       generated chunks + assessments. Routes via the
       `training-synthesizer` agent (tool: `synthesize_training`).
       Optional phase: skipped when no `ANTHROPIC_API_KEY` or when
       `--skip-training` is passed on the CLI. Emits per-pair resume
       sidecar at `training_specs/.synthesis_pairs_checkpoint.jsonl`
       so a mid-run crash on multi-hour local-LLM rebuilds resumes
       past every accepted pair; opt out via `--no-checkpoint`.

11. libv2_archival
   └── Archive course artifacts to LibV2 (raw PDFs, DART HTML, IMSCC,
       RAG corpus). Gated by libv2_manifest integrity checks.

12. vector_indexing (optional)
   └── Build the per-course on-device vector index (real embeddings +
       numpy exact-search index) from the LibV2-archived chunkset so a
       freshly-built course is askable (retrieval-ready) the moment the
       run completes. Routes via the `rag-indexer` agent
       (tool: `run_vector_indexing`) — the same handler as the
       `rag_training` `indexing` phase. Runs by default; skips cleanly
       with a logged reason when the `[embedding]` extras are absent
       UNLESS `TRAINFORGE_REQUIRE_EMBEDDINGS` is set, in which case the
       phase runs and fails closed on a broken/unavailable embedding
       backend (no file-count fallback).

13. finalization
   └── Final validation and training data export.
```

---

## Agent Registry

### Courseforge Agents

| Agent | Purpose |
|-------|---------|
| `course-outliner` | Create course structure |
| `requirements-collector` | Gather specifications & prerequisites |
| `content-generator` | Generate module content |
| `brightspace-packager` | Package for Brightspace LMS |
| `oscqr-course-evaluator` | OSCQR quality evaluation |
| `quality-assurance` | Pattern prevention & validation |

### Conversion / Remediation Agents

| Agent | Purpose |
|-------|---------|
| `dart-automation-coordinator` | Orchestrate PDF conversion |
| `dart-converter` | Drives the SemantiK v2 cascade for the `dart_conversion` phase (PDF → accessible HTML + source provenance) |
| `imscc-intake-parser` | Extract & inventory IMSCC packages |
| `content-analyzer` | Detect accessibility & quality gaps |
| `accessibility-remediation` | WCAG fixes, alt text, headings |
| `content-quality-remediation` | Educational depth & enhancement |
| `intelligent-design-mapper` | Component selection & styling |
| `remediation-validator` | Final QA & WCAG verification |
| `dart-chunker` | Emit `LibV2/courses/<slug>/dart_chunks/chunks.jsonl` from staged DART HTML via `Trainforge.chunker.chunk_content`; deterministic transformation (no LLM dispatch). Backed by `_run_dart_chunking` registered in `MCP/tools/pipeline_tools.py::_build_tool_registry`. |

### Textbook Pipeline Agents

| Agent | Purpose |
|-------|---------|
| `textbook-stager` | Stage DART outputs for Courseforge |
| `textbook-ingestor` | Parse DART HTML & extract objectives |
| `source-router` | Bind DART source blocks to Courseforge module pages (TF-IDF + confidence scoring) |
| `libv2-archivist` | Archive course artifacts to LibV2 |

### Trainforge Agents

| Agent | Purpose |
|-------|---------|
| `assessment-extractor` | Parse IMSCC & extract content |
| `rag-indexer` | Build vector embeddings & index (routes to `run_vector_indexing`; fails closed without an embedding backend) |
| `assessment-generator` | Generate questions & distractors |
| `assessment-validator` | Validate quality & Bloom's alignment |
| `training-synthesizer` | Synthesize instruction + preference training pairs from chunks + assessments (routes to `synthesize_training`). |

---

## Quality Standards

### Decision Rationale

Every decision rationale MUST:
- Be at least 20 characters
- Explain the "why" not just the "what"
- Reference alternatives when applicable

### LLM call-site instrumentation

Every LLM call site MUST wire up a `DecisionCapture` instance and emit at least one decision per call (per-batch when batched). Static boilerplate rationales are forbidden — rationale must interpolate dynamic signals specific to the call (block IDs, image hashes, page numbers, model + max_tokens, confidence distributions, etc.) so captures are replayable post-hoc. A regression test MUST assert that the capture fires on the call path.

Precedent call sites + regression tests: `docs/architecture/decision-capture.md`.

### Assessment Quality (Trainforge)

- Bloom's taxonomy alignment required
- Learning objective mapping required
- Distractor misconception targeting required

### Content Quality (Courseforge)

- WCAG 2.2 AA compliance
- Clear learning objectives per module
- Consistent formatting

### Conversion Quality (SemantiK)

- Semantic HTML structure
- Alt text for all images
- Proper heading hierarchy

---

## Error Handling

### Error Classification

Errors are classified to determine retry behavior:
- **Transient**: `timeout`, `rate_limit`, `connection_error`, `service_unavailable` → retryable
- **Permanent**: `validation_error`, `missing_input`, `permission_denied`, `schema_error` → no retry

### Retry Protocol

Failed tasks retry up to 3 times with exponential backoff:
1. First retry: After 5 seconds
2. Second retry: After 30 seconds
3. Third retry: After 120 seconds
4. After 3 failures: Log to error table, require manual intervention

### Poison Pill Detection

Stops a batch when the same error pattern repeats:
- Default threshold: 3 same-pattern failures within 5 minutes
- Prevents runaway batch failures from consuming resources

### Phase Checkpointing

Each phase completion creates a checkpoint in `state/runs/{run_id}/checkpoints/`:
- Enables crash recovery without re-running completed phases
- Checkpoints include phase outputs and state snapshots

### Error Logging

All errors logged to:
- GENERATION_PROGRESS.md error table
- Individual JSONL capture files

---

## Aggregators

Top-level workflow aggregators run post-loop in `WorkflowRunner.run_workflow` and roll up per-phase signals into a single operator-facing JSON. Best-effort: aggregator failure logs a warning but does not change `final_status`. Long-form per-aggregator detail: `docs/architecture/aggregators.md`.

| Aggregator | Output | Schema |
|------------|--------|--------|
| `CourseforgeValidationReport` | `<project>/courseforge_validation_report.json` | schema 1.1 |
| `TrainforgeAssessmentQualityReport` | `<libv2_course>/quality/trainforge_assessment_quality_report.json` | schema 1.0 |
| `CoverageMapAggregator` | `<libv2_course>/coverage_map.json` | `schemas/aggregators/coverage_map.schema.json` |
| `EdgeConsensusAggregator` (GPT-fb-12-may item 2) | `<libv2_course>/graph/edge_consensus_report.json` (sibling of `concept_graph_semantic.json`); also stamps per-edge `edge_status` + `consensus_signals[]` on the semantic graph via `apply_to_graph`. Attenuates `kg_quality.consistency` by `(1 - contradiction_rate)` inside `KGQualityValidator.validate`. | (helper, no separate file; surfaces in `schemas/knowledge/concept_graph_semantic.schema.json`) |
| `PromotionChainAggregator` (W3.G master) | `<libv2_course>/courseforge_promotion_chain_report.json` | `schemas/governance/promotion_chain.schema.json` |
| `BlockQualityRollupAggregator` (IB6.6) | `<libv2_course>/block_quality_rollup_report.json` (falls back to `<project_path>/...`) — block→module→course 8-dim quality rollup (BOTH mean ≥2.0 AND per-dimension minimum-floor paths + the 3 hard gates: Accessibility=0 block-fail, assessment-Bloom<objective Alignment-cap-1, interaction-without-feedback Feedback+Coherence-cap-1). Only runs when `ED4ALL_BLOCK_QUALITY_RUBRIC` is on. Emits `block_quality_rollup_aggregated`. | `schemas/aggregators/block_quality_rollup.schema.json` |
| `lib/governance/course_status.py::derive_course_status` | composes `course_status` enum on chain report | (helper, no separate file) |

`PromotionChainAggregator` supersedes the per-aggregator `final_promotion_decision` heuristics. `derive_course_status` returns the canonical 5-value enum (`failed | non_certified_archive | certified_accessible | certified_instructional | certified_trainable`); a missing per-stage report shorts to `course_status: failed` (anti-silent-degradation contract).

## Validation Gates

Validation gates run after workflow phases to enforce quality:

### Gate Configuration

```yaml
validation_gates:
  - gate_id: content_structure
    validator: lib.validators.content.ContentStructureValidator
    severity: critical     # critical | warning
    threshold:
      max_critical_issues: 0
    behavior:
      on_fail: block       # block | warn
      on_error: fail_closed # fail_closed | warn
```

### Severity Levels

| Severity | Behavior |
|----------|----------|
| `critical` | Blocks workflow progression on failure |
| `warning` | Logs warning, allows workflow to continue |

### Active Gates

Source of truth: `config/workflows.yaml::validation_gates`. Full per-gate table + per-validator Wave-reference paragraphs: `docs/validation/gates.md`. Severity-flip semantics for calibration-gated validators: `lib/governance/calibration_gate.py::resolve_severity_flip`.

Summary by workflow (counts derived from `config/workflows.yaml`):

| Workflow | Critical | Warning | Total |
|----------|---------:|--------:|------:|
| `course_generation` | 17 | 35 | 52 |
| `rag_training` | 4 | 3 | 7 |
| `textbook_to_course` | 39 | 79 | 118 |
| `trainforge_train` | 2 | 0 | 2 |
| **Total** | **62** | **117** | **179** |

Per-wave gate-landing history (additions, demotions, deferred severity flips, with the intermediate running subtotals at each wave): `docs/validation/gate-history.md`. The table above is the current authoritative count; the history file's per-wave subtotals are provenance-only and do not sum to the current total.

---

## Configuration Files

### workflows.yaml

Defines workflow phases and concurrency limits.
Location: `config/workflows.yaml`

### agents.yaml

Defines agent capabilities and project paths.
Location: `config/agents.yaml`

### workflows_meta.schema.json

Meta-schema that validates `config/workflows.yaml` at load time (phase routing, gate shape, `inputs_from` references).
Location: `schemas/config/workflows_meta.schema.json`

---

## Opt-In Behavior Flags

Environment-variable toggles gate opt-in strict / stable-ID / provenance / experimental-rule-graph behavior. All default off to preserve backward compatibility with legacy corpora. Full rationale per flag lives in the owning subsystem's flag table (named below).

Per-flag rows now live in subsystem CLAUDE.md files (one owner per prefix). Counting convention: **distinct flags, multi-flag rows expanded** — a single table row that documents several env vars (e.g. SemantiK's `SEMANTIK_MODEL_DIR / _CACHE_DIR / _DATA_DIR / _CONFIG_DIR` row = 4 flags, Courseforge's `COURSEFORGE_SELF_VERIFY / _REFINE_ROUNDS / _CHUNK_SCOPED` row = 3 flags) counts once per flag it documents, so the tally exceeds the printed row count.

| Prefix | Owner | Flag count |
|--------|-------|-----------:|
| `TRAINFORGE_*` / `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` / `CURRICULUM_ALIGNMENT_*` / `WAVE18_*` | [`Trainforge/CLAUDE.md § Opt-In Behavior Flags`](Trainforge/CLAUDE.md) | 51 |
| `NVIDIA_*` (hosted 70B/large cloud tier — `NVIDIA_API_KEY` / `NVIDIA_BASE_URL` / `NVIDIA_LARGE_MODEL`) | [`Trainforge/CLAUDE.md § Opt-In Behavior Flags`](Trainforge/CLAUDE.md) | 3 |
| `SEMANTIK_*` (DART replacement — SemantiK semantic-cascade converter; also honors the legacy `DART_THETA_DEVICE` compat env) | [`SemantiK/CLAUDE.md § Opt-In Behavior Flags`](SemantiK/CLAUDE.md) | 43 |
| `COURSEFORGE_*` / `COURSEPLANNER_*` / `TEXTBOOK_SYNTHESIS_*` | [`Courseforge/CLAUDE.md § Opt-In Behavior Flags`](Courseforge/CLAUDE.md) | 35 |
| `DECISION_*` / `ED4ALL_*` / `LOCAL_DISPATCHER_*` / `MCP_ORCHESTRATOR_*` / `LLM_*` (cross-cutting) | root (table below) | 83 |

### Cross-cutting flags (root-owned)

| Flag | Default | Purpose |
|------|---------|---------|
| `DECISION_VALIDATION_STRICT` | unset | Fails closed on unknown `decision_type` values in decision captures. |
| `ED4ALL_BLOCK_QUALITY_RUBRIC` | unset (off) | IB6 keystone — gates the eight-dimension 0-3 block-quality scoring pass (`lib/validators/block_quality_rubric.py::BlockQualityRubricValidator`), the anatomy slot-presence (`lib/validators/anatomy_slot_presence.py`) + interaction-feedback (`lib/validators/interaction_feedback.py`) + cognitive-load (`lib/validators/content.py::BlockCognitiveLoadValidator`) + QA-checklist (`lib/validators/qa_checklist.py`) validators, the universal `(verb·level·knowledge-type)` chip render (`Courseforge/scripts/blocks.py::_block_quality_rubric_emit_enabled` → `_bloom_triple_attrs`), the B01 both-axes assertion in `AbcdObjectiveValidator`, and the block→module→course rollup aggregator (`lib/aggregators/block_quality_rollup.py::BlockQualityRollupAggregator`). Default OFF → no rubric field written, no chip emitted, the B01 knowledge-type check is skipped, every scoring/rollup validator no-ops with `passed=True` + an info issue, no rollup file, snapshots byte-identical. When truthy (`1`/`true`/`yes`/`on`), the scoring pass COMPOSES already-computed validator signals (`objective_assessment_similarity`→Alignment, `instructional_depth`/200-char→Cognitive-load, `block_prose_entailment`→Coherence, IB4 WCAG/UDL→Accessibility/Engagement, IB3 verb-triple→Alignment, distractor cluster + IB6.3 feedback→Feedback) into the IB2 rubric container (NO new model load — reads upstream `GateResult` metadata), applies the three hard gates (Accessibility=0→block fail at rollup; assessment Bloom<objective→Alignment cap 1; interaction-without-feedback→Feedback+Coherence cap 1), and rolls up with BOTH mean (≥2.0) AND per-dimension-minimum-floor (≥2.0 on every required core dim). Gates wire warning-day-1 at `inter_tier_validation` + `post_rewrite_validation` with `# TODO(calibration)` markers; the hard-gate critical flip is DEFERRED until the anchored 0-3 scale is calibrated against ≥2 corpora. Generated PRODUCT-quality scoring, not training data → no `docs/LICENSING.md` row. Falsey / garbage → off (parse-with-fallback). Active only on the two-pass surface (`COURSEFORGE_TWO_PASS=true`). |
| `ED4ALL_BLOCK_BODY_CHAR_CEILING` | `200` (global override) / per-type default | IB6.4 per-block D2 cognitive-load body ceiling (`lib/validators/content.py::BlockCognitiveLoadValidator`; resolver `lib/validators/_block_rubric_helpers.py::resolve_body_char_ceiling`). The `BLOCK_BODY_OVERFLOW` ("everything-block anti-pattern") check fires when a block's visible body exceeds its ceiling (OR above the separate, UNTOUCHED >4-idea-chunk ceiling). Drives the D2 dimension score in the rubric. **This env is a GLOBAL override** — when set (and positive) it pins ONE ceiling for every block type (preserving the historical semantics: `50` makes even an exposition concept overflow). When UNSET, the ceiling is resolved PER BLOCK TYPE from `_BLOCK_BODY_CHAR_CEILING_BY_TYPE` reflecting each type's intended granularity (FIX 3, calibrated on the cal2 cohort's measured p50-p75 bodies — does NOT blanket-raise): **~200** atomic one-line micro-blocks (`vocab_card`/`callout`/`formula`/`key_idea`); **~1000** single-developed-idea exposition / answer-bearing types (`concept`/`example`/`worked_example`/`self_check_question`/`diagram`/`scenario`/`explanation`/`problem`/`activity`/`guided_practice`/`misconception`/`hook`/`multimedia`); **~1200** aggregating / structural roll-ups (`prereq_set`/`recap`/`checklist`/`reflection_prompt`/`acronym`/`table`/`discussion_prompt`/`resources`/`flip_card_grid`); any untabled type falls through to the **200** atomic default. `chrome`/`objective`/`summary_takeaway` are EXEMPT from the check entirely (`content.py::_COGNITIVE_LOAD_EXEMPT_TYPES` — structural/roll-up bodies, skipped before a ceiling is resolved, so they carry NO table row). Genuine over-stuffers (a concept over 1000, an acronym table over 1200) still trip the char axis; the FIX 3 calibration only rescued normally-sized composite blocks from the type-blind-200 FP that pinned the gate at its 50-issue cap. Garbage / non-positive env → falls through to the per-type default (parse-with-fallback, mirroring `ED4ALL_ANSWER_NUM_CTX`). Selects no provider/model → no `docs/LICENSING.md` row. No-op when `ED4ALL_BLOCK_QUALITY_RUBRIC` is off. |
| `MCP_ORCHESTRATOR_LLM_MODEL` | `claude-opus-4-7` | Pins the Anthropic model ID used by `MCP/orchestrator/llm_backend.py::DEFAULT_ANTHROPIC_MODEL`; per-run `LLM_MODEL` keeps higher precedence. |
| `LOCAL_DISPATCHER_ALLOW_STUB` | unset | Permits `LocalDispatcher` to emit a stubbed `PhaseOutput` when no `agent_tool` is wired. Tests / dry-run only. |
| `ED4ALL_CLOUD_RATE_LIMIT` | unset (off) | NVIDIA-70b-everywhere SETUP — master switch for the shared cloud-provider admission gate (`lib/llm/rate_limiter.py`: RPM token bucket + TPM token bucket (debit estimate up front, reconcile vs server-reported `usage`) + a concurrency semaphore). **Ships DARK.** Default OFF → `get_admission_gate` returns a no-op gate, so the hook at `Trainforge/generators/_openai_compatible_client.py::_post_with_retry` (the chokepoint above every COMPOSED authoring seat — Together / Local / Curriculum / Courseforge rewrite+outline / textbook synthesis / the NVIDIA large tier) is a pure no-op (no sleeps, no state, byte-identical to the legacy path; proven by the unchanged 60-test client suite). When truthy (`1`/`true`/`yes`/`on`) AND ≥1 ceiling env is set, a real `AdmissionGate` is built once process-wide and admits/blocks each request. Per-axis ceilings are operator-supplied at RUN time (no hardcoded guess): `ED4ALL_CLOUD_RPM` / `ED4ALL_CLOUD_TPM` / `ED4ALL_CLOUD_MAX_CONCURRENCY` (each unset by default → that axis is off; parse-with-fallback — garbage / non-positive → unset). Cross-process caveat (documented, not solved): an in-process singleton does NOT coordinate separate OS processes sharing one API key (concurrent `ed4all run` sessions each get their own gate) — a real multi-session deploy needs flock/IPC. KNOWN GAP: `MCP/orchestrator/llm_backend.py::AnthropicBackend` uses a raw `anthropic.Anthropic` client that BYPASSES `_post_with_retry`, so api-mode orchestration is NOT covered — a pure-nvidia build routes its authoring through the composed seats (all covered); harden the AnthropicBackend bypass only if/when api-mode runs cloud (documented TODO at the hook + at `llm_backend.py`). Selects no provider/model → no `docs/LICENSING.md` row. Falsey / garbage → off (parse-with-fallback). |
| `ED4ALL_AGENT_DISPATCH` | unset | Routes subagent-classified agents through `dispatcher.dispatch_task` instead of in-process tool registry. |
| `ED4ALL_AGENT_TIMEOUT_SECONDS` | `1800` | Per-task subagent dispatch mailbox timeout. |
| `ED4ALL_ALIGNMENT_VERB_TRIPLE` | unset (off) | IB3 constructive-alignment keystone. Default OFF → byte-identical: the verb-triple equality axis on `BlockObjectiveDeliveryValidator`, the evidence-form check on `AssessmentObjectiveAlignmentValidator`, the `AnchoredRubricValidator`, and the `TriangleCompletenessValidator` all no-op, and the `Block.anchored_rubric` field stays None + hash-excluded (snapshots unchanged). When truthy (`1`/`true`/`yes`/`on`): enforces `objective-verb = activity-verb = assessment-verb` within the Bloom band (`BLOCK_OBJECTIVE_VERB_TRIPLE_MISMATCH`), records the `alignment_cap_at_1` signal when assessment Bloom < objective Bloom (consumed by the IB6 rollup), bars recall MCQ on Apply+ objectives (`ASSESSMENT_EVIDENCE_FORM_TOO_LOW`), requires an exemplar-anchored published-first rubric on Evaluate/Create scored blocks, and asserts every objective has ≥1 activity + ≥1 band-aligned assessment (triangle completeness). All gates wire warning day-1 with an ACCELERATED fast-flip `# TODO(calibration)` marker (IB3 is the roadmap's documented exception — the keystone validity rule flips critical after ONE ≥2-corpus FP measurement). Reuses `lib/ontology/bloom.py` verb bands; needs IB1 slot-addressability (`interaction`/`feedback` slots). Generated PRODUCT content (course validation), not training data → no `docs/LICENSING.md` row. Falsey / garbage → off (parse-with-fallback). |
| `ED4ALL_ANSWER_PROVIDER` | `local` | Selects the grounded-answer backend (runtime Q&A inference) from the W-D12 `_OPENAI_COMPATIBLE_PROVIDERS` registry. **Loopback-only:** a resolved non-loopback `base_url` raises `AnswerProviderNotLocal` (Phase IA: no cloud arm on the answer path, ever). No escape-hatch env. Licensing row in `docs/LICENSING.md` § "Grounded-answer provider". |
| `ED4ALL_ANSWER_MODEL` | per-provider | Model ID override for the answer backend. Resolution chain: explicit arg > `ED4ALL_ANSWER_MODEL` > registry `model_env` (`local` → `LOCAL_SYNTHESIS_MODEL`) > registry default (`qwen2.5:7b-instruct-q4_K_M`, fits common 8GB GPUs fully resident — the canonical `DEFAULT_SYNTHESIS_MODEL` default and the `config/endpoints.yaml` local seat both use 7b). The larger Qwen2.5 sizes (14b/32b) remain valid `LOCAL_SYNTHESIS_MODEL` / `ED4ALL_ANSWER_MODEL` overrides for boxes with more VRAM. See `docs/operations/docker.md`. |
| `ED4ALL_ANSWER_TIMEOUT_SECONDS` | `120` | Answer-client HTTP timeout (long passages, slow local GPU). Garbage values fall back to the default. |
| `ED4ALL_ANSWER_NUM_CTX` | `4096` | Serving-window token budget for the grounded-answer prompt (`lib/retrieval/_prompts.py`). Honest about the common Ollama default (`num_ctx=4096`), which silently truncates the prompt HEAD when exceeded. Drives the **token-aware** context budget (math-safe 2.5 chars/token divisor) sized to fit system prompt + passages + question + allowed-id enumeration + remediation headroom + `max_tokens` inside the window; trailing passages are dropped whole before the question is ever truncated. Set this to the model server's ACTUAL window (e.g. `8192` for a long-context Modelfile, matching `OLLAMA_CONTEXT_LENGTH`) so the budget shrinks the prompt to fit. A post-call tripwire (`answer_composer._check_prompt_not_truncated`) compares the server-reported `usage.prompt_tokens` against the local estimate and raises `PromptTruncatedError` (fail-closed) on a large shortfall rather than letting silent head-truncation fabricate citations. Garbage / non-positive values fall back to the default. |
| `ED4ALL_ANSWER_CITATION_PRUNE` | `shadow` | Three-valued (`off` / `shadow` / `on`) governor of the claim-attribution citation **prune + add** pass at answer-composition time (`lib/retrieval/citation_attribution.py`, hooked in `grounded_answer.answer_course_question` strictly post-citation-gate). Attribution runs over ALL gate-eligible (renderer-included) passages. `on`: drop model-cited citations that back zero answer sentences (never below 1 remaining — all-prune is a logged no-op) AND credit an uncited high-support passage the model under-cited (anchor-must-resolve, capped at 2 per answer); final citations sorted strongest-supporter-first. `shadow` (code default): compute + emit the `grounded_answer_citation_prune` capture + warnings + `supporting_excerpt`/`supported_claim_count`, but mutate no citations. `off`: skip entirely. NEVER changes an answer verdict. The Docker compose stack sets this `on`. No licensing row (no provider/model). |
| `ED4ALL_ANSWER_PRUNE_MIN_OVERLAP` | `0.25` | Float support threshold for the PRUNE decision (`citation_attribution.resolve_min_overlap`; also the `prune_min_overlap` kwarg). A citation supports a claim iff its 4-shingle containment ≥ this OR its content-token coverage ≥ 0.80 (fixed secondary floor). Deliberately low (precision of the keep-vs-drop decision over recall). The separate ADD bar is `ED4ALL_ANSWER_ADD_MIN_SHINGLE` (default `0.50`, shingle-only) + a relative "out-support the kept set" gate. Out-of-range / garbage values fall back to the default. The Docker compose stack sets this `0.444444` (the 2026-06-12 single-course union-corpus precision-floored calibration recommendation; code default unchanged). No licensing row. |
| `ED4ALL_ANSWER_ADD_MIN_SHINGLE` | `0.50` | Float shingle floor for the ADD decision (`citation_attribution.resolve_add_min_shingle`; threaded through `grounded_answer._apply_citation_attribution`). An uncited gate-eligible passage is credited only if its 4-shingle containment ≥ this on some claim AND it out-supports the strongest kept citation on that claim (additions are precision-first, capped at 2/answer). Twice the prune default by design (the failure cost of a spurious added citation is a fabricated-looking source). The single-course union-corpus attribution calibration (2026-06-12) measured a median cited shingle of 0.000 — 0.500 below this bar, so additions rarely fire — warranting the KNOB without lowering the conservative default. Out-of-range / garbage values fall back to the default. No licensing row (no provider/model). |
| `ED4ALL_ANSWER_NLI_ADD` | `off` | Three-valued (`off` / `shadow` / `on`) governor of the **NLI-based citation-ADD** arm (`citation_attribution.resolve_nli_add_mode`; hooked in `grounded_answer.answer_course_question` strictly AFTER the lexical prune+add pass, step 7b). The entailment-driven successor to the shingle ADD arm, which the 2026-06-12 under-citing investigation measured unsalvageable (paraphrase answers never quote; median cited shingle 0.000 → zero adds even at bar 0.10). REUSES the groundedness scorer's per-claim NLI verdicts (never runs NLI twice; needs `with_groundedness`). Credits an uncited gate-eligible supporter only under a COMPOSITE criterion: NLI entailment ≥ 0.70 (windowed, scorer-v2) AND claim↔chunk content-token coverage ≥ 0.80 AND every numeric literal in the claim text present in the chunk (NLI is number-blind) AND the chunk anchors/resolves AND ≤ 2 adds/answer. **Default `off`** — unlike the lexical arm's `shadow` default, the NLI model is a ~750 MB lazy load that must NOT touch the default answer path. `shadow`: compute would-adds, emit the `grounded_answer_nli_citation_add` capture + the additive `nli_citation_add` diagnostics block (eval aggregates `shadow_nli_add`) + warnings, mutate NOTHING. `on`: actually add (anchor-resolved, sorted after existing citations, cap 2); ships dark. NEVER changes an answer verdict in any mode. The Docker compose stack sets this `shadow`. Garbage values fall back to `off`. No licensing row (no provider/model). |
| `ED4ALL_ANSWER_COMPLETENESS_RECHECK` | `on` | Governs the post-generation **completeness recheck** (`grounded_answer._resolve_completeness_recheck`; hooked in `answer_course_question` step 4b, BEFORE the citation gate). A small (7B-Q4) model non-deterministically drops a sub-question of a MULTI-part question even when the grounding is its top passage (observed 2026-06-18: "perimeter of a rectangle? circumference of a circle?" → only the rectangle half answered). When on (default), the answer is split into sub-questions (`lib/retrieval/answer_completeness.py`, pure-lexical — NO model load), and any part that is unaddressed by the answer BUT grounded in a retrieved passage triggers ONE bounded re-ask through the composer with the additive `COMPLETENESS_REMEDIATION_DIRECTIVE`; the re-ask's missing-part prose + citations are MERGED onto the original (the re-ask is itself not rechecked — single retry, no recursion). **Never regresses** — a re-ask that refuses / returns no citations / errors keeps the original answered response, and the answered-family verdict never becomes a refusal/block. Detection is deliberately UNDER-split + recall-leaning (a no-`?` search/statement query is single-part by construction; the grounding gate holds the precise 0.80 token-coverage floor so a re-ask only fires on a sub-question the corpus can answer). Emits the `grounded_answer_completeness_recheck` decision capture (dynamic, replayable rationale) every call. Falsey values (`0`/`false`/`no`/`off`) disable (parse-with-fallback, mirroring the other answer-path knobs). Selects no provider/model → no `docs/LICENSING.md` row. |
| `ED4ALL_BLOCK_ANATOMY` | unset (off) | IB1 six-slot anatomy contract emit gate (`Courseforge/scripts/blocks.py::_anatomy_emit_enabled`, consumed in `Block._minimal_block_jsonld`). Default OFF → no `anatomy` key in the JSON-LD `blocks[]` entry; every existing snapshot / `contentHash` stays byte-identical (the five new `Block` slot fields `heading`/`purpose_tag`/`interaction`/`feedback`/`transition` are Optional-default-None, hash-EXCLUDED, and the BODY slot is the existing `content`). When truthy (`1`/`true`/`yes`/`on`) AND `COURSEFORGE_EMIT_BLOCKS` is on, the entry gains a nested `anatomy` object carrying only the non-None slots plus the slot→stage `lifecycle` map (`activate→present→apply→check→consolidate`). Representation only — NO gate (IB6 owns the anatomy slot-presence validator), NO new `data-cf-*` HTML attr, NO LLM call. Generates PRODUCT metadata, not training data → no `docs/LICENSING.md` row. Falsey / garbage → off (parse-with-fallback, mirroring `COURSEFORGE_EMIT_BLOCKS`). |
| `ED4ALL_BLOCK_A11Y` | unset (off) | IB4 per-block WCAG 2.2 AA + UDL emit gate (`lib/generation/block_a11y.py::resolve_block_a11y`; emit reader `Courseforge/scripts/blocks.py::_block_a11y_emit_enabled`; threaded into the `rewrite_html_shape` + `udl_coverage` Block inputs by `MCP/hardening/gate_input_routing.py`). Default OFF → `Block`'s UDL fields (`n_representations`/`response_formats`/`engagement_affordance`) are NOT emitted to HTML/JSON-LD (Optional-default-empty, hash-EXCLUDED) and the per-block a11y sub-check in `RewriteHtmlShapeValidator._check_block_a11y_contract` is a no-op, so every existing snapshot stays byte-identical. When truthy (`1`/`true`/`yes`/`on`), the deterministic `_derive_udl_coverage` UDL fields are stamped (`data-cf-udl-*` HTML attrs + a `udlCoverage` JSON-LD sub-object) and the per-block-type WCAG contract (alt text 1.1.1 / keyboard-operable interaction + name/role/value 2.1.1+4.1.2 / descriptive link text 2.4.4 / captions+transcript for B04) is enforced as a WARNING (`REWRITE_BLOCK_A11Y_CONTRACT`) at `inter_tier_validation` + `post_rewrite_validation`. The `chunk_wcag_status` chunk-field gate, the `textbook_to_course` packaging `wcag_compliance` gate, and the `udl_coverage` validator run warning-day-1 regardless of this flag (they read existing data / reuse `WCAGValidator` / derive UDL on read). Falsey / garbage → off (parse-with-fallback). Generates PRODUCT content (course-page a11y attrs), not training data → no `docs/LICENSING.md` row. |
| `ED4ALL_CALLOUT_TYPED` | unset (off) | FR-A11Y-03 typed B12 callout emit + gate flag (renderer reader `Courseforge/scripts/generate_course.py::_callout_typed_enabled`; gate `lib/validators/callout_structure.py::CalloutStructureValidator`, wired warning-day-1 at `post_rewrite_validation` in BOTH two-pass workflows, builder `_build_block_input_rewrite`). Default OFF → the callout renderer emits its legacy markup byte-identical (no per-kind label/icon row, no `callout-kind-*` border class, no appended typed CSS in the page `<style>`) and the `callout_structure` gate is a strict no-op (`passed=True` + a `CALLOUT_STRUCTURE_DISABLED` info issue); the new `Block.callout_kind` field is Optional-default-None + hash-EXCLUDED (the `compute_content_hash` payload is a fixed 5-key allowlist), so every existing snapshot / `contentHash` stays byte-identical. When truthy (`1`/`true`/`yes`/`on`), a callout carrying a `callout_kind` (note / tip / warning / example / key-idea) renders a redundant non-color contract — a visible LABEL + icon + a per-kind border class (WCAG 1.4.1, never color-only) — and the gate fires WARNING issues (`CALLOUT_NO_KIND` / `CALLOUT_COLOR_ONLY` / `CALLOUT_BODY_OVERFLOW` reusing the ~200-char ceiling / `CALLOUT_MOTION`, `action=regenerate`). `# TODO(calibration)` deferred critical-flip (WS3/W4 deferred-flip pattern; NOT IB3). Generates PRODUCT content (course-page a11y markup), not training data → no `docs/LICENSING.md` row. Falsey / garbage → off (parse-with-fallback). Active only on the two-pass surface (`COURSEFORGE_TWO_PASS=true`). |
| `ED4ALL_COS_PER_WEEK_CAP` | `0` (auto) | WS5 §2.2 per-week chapter-objective placement cap for the single-sourced slicer `Courseforge/scripts/generate_course.py::_slice_cos_for_week` — consumed by BOTH the emit-side per-week slicer (`MCP/tools/pipeline_tools.py::_generate_course_content`) and the validator's allowed-set builder (`_slice_chapter_objectives_by_week` + `_plan_course_structure`'s §2.4(A) `"Week N"` group persistence), so emit-week-N ids == validator-allowed-week-N ids by construction. The slicer uses a CEIL stride `step = max(1, ceil(len(COs)/weeks))` (was the floor `len(COs)//weeks` + `[:2]` truncation that silently dropped grounded COs at any `duration_weeks < len(COs)`); each week claims `COs[(w-1)*step : w*step][:cap]`. Default `0` = auto = `step` (no truncation — every CO in the ceil-stride slice is placed, guaranteeing zero-drop coverage). A positive int pins a hard per-week ceiling (e.g. `3` to thin dense weeks). The cap-lift is UNCONDITIONAL — it applies even on an explicit `--weeks N` so a short course still places all COs (only the WS5 §3.2 TO-rescale is behind the override guards). Garbage / non-positive values fall back to `0`/auto (parse-with-fallback, mirroring `ED4ALL_ANSWER_NUM_CTX`). Selects no provider/model → no `docs/LICENSING.md` row. |
| `ED4ALL_CONTENT_PAGE_PER_CO` | unset (off) | Page-per-CO content-emit gate (resolver `lib/generation/content_page_budget.py::content_page_per_co_enabled`; call-site wrapper `MCP/tools/pipeline_tools.py::_content_page_per_co_enabled`, additionally gated on `COURSEFORGE_TWO_PASS`). Default OFF → byte-identical: the outline phase's content-page count stays `max(topic_count, 1)` per week (O(sections)), the two per-CO index builders run with their legacy unclamped / `require_id`-off signatures, every emitted descriptor / `contentHash` / snapshot stays byte-identical. When truthy (`1`/`true`/`yes`/`on`) AND the two-pass surface is on, the content-page count is driven by the week's CO count (the bounded axis) via `_resolve_content_page_count` — one content page per child CO with a NEVER-INCREASE guard (`min(max(co,1), topic_count or max(co,1))`, so a CO-rich/topic-thin week can never emit MORE pages than today), the page→CO map is 1:1 by construction (zero stranded COs), each CO's chunk universe consolidated onto its page is token-bounded by `cap_page_chunks` (union-then-top-K, always-keep-≥1, kept⊆union — anti-fabrication), and every CO group's resolved week is CLAMPED into `[1, duration_weeks]` (`_clamp_week`) so a chapter-numbered group past the last week is not silently stranded (the documented 15-chapter failure mode). The two index builders are made PREDICATE-IDENTICAL (both skip id-less objectives) with a per-week length-equality assertion (falls back to the OFF count on mismatch) + a post-loop coverage assertion (union of emitted CO ids == all valid CO ids; logs a coverage GAP error on any miss). Emits one `content_selection` decision event/run (pages emitted, chunks kept/dropped, num_ctx, token budget, max_chunks). Selects no provider/model → no `docs/LICENSING.md` row. Falsey / garbage → off (parse-with-fallback). Active only on the two-pass surface (`COURSEFORGE_TWO_PASS=true`). |
| `ED4ALL_CONTENT_PAGE_NUM_CTX` | `4096` (→ `ED4ALL_ANSWER_NUM_CTX` → 4096) | Authoring serving-window token budget for the page-per-CO per-page chunk cap (`lib/generation/content_page_budget.py::resolve_content_page_num_ctx`). Resolution chain: explicit arg > `ED4ALL_CONTENT_PAGE_NUM_CTX` > the shared `ED4ALL_ANSWER_NUM_CTX` > 4096. Drives `page_chunk_token_budget = num_ctx − (DYNAMICALLY measured rewrite system prompt) − user-fixed − reserve − max_tokens`; a negative budget (fixed cost alone overflows the window) is returned verbatim so the page caps to keep-≥1 rather than shipping chunks that guarantee head-truncation. Garbage / non-positive → falls back (parse-with-fallback). No-op when `ED4ALL_CONTENT_PAGE_PER_CO` is off. Selects no provider/model → no `docs/LICENSING.md` row. |
| `ED4ALL_CONTENT_PAGE_MAX_CHUNKS` | `5` | Hard top-K ceiling on chunks kept per CO page for the page-per-CO cap (`lib/generation/content_page_budget.py::resolve_content_page_max_chunks`; mirrors `ED4ALL_OBJECTIVE_MAX_CHUNKS_PER_OBJECTIVE`). The cap is `min(max_chunks, token-budget-allows)`; always-keep-≥1 supersedes both. Garbage / `< 1` → `5`. No-op when `ED4ALL_CONTENT_PAGE_PER_CO` is off. Selects no provider/model → no `docs/LICENSING.md` row. |
| `ED4ALL_EMBEDDING_PROVIDER` | `st` | Selects the retrieval-index embedding backend from `lib/embedding/providers.py::_EMBEDDING_PROVIDERS` (`st` in-process sentence-transformers / `local-openai` local `/v1/embeddings` server / `fake` deterministic test vectors). Registry entries, NOT subclasses. Not training-data synthesis; licensing row in `docs/LICENSING.md` § "Embedding providers". |
| `ED4ALL_EMBEDDING_MODEL` | per-provider | Model ID override for the embedding provider (e.g. `BAAI/bge-large-en-v1.5`). Resolution chain: explicit arg > env var > registry default. |
| `ED4ALL_EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | Base URL of the local OpenAI-compatible `/v1/embeddings` server (`local-openai` provider only; Ollama / vLLM / llama.cpp). |
| `ED4ALL_EMBEDDING_API_KEY` | `local` | Optional bearer token for the local embedding server (`local-openai` only; most local servers ignore it). |
| `ED4ALL_EMBEDDING_DEVICE` | `cpu` | Torch device for the in-process `st` provider. Default `cpu` for determinism; `cuda` allowed for speed (recorded in the index manifest so mixed-provenance comparisons are detectable). |
| `ED4ALL_EMBEDDING_BATCH_SIZE` | `16` | Encode batch size for the embedding client (replay parameter, recorded in the index manifest). |
| `ED4ALL_EMBEDDING_ALLOW_FAKE` | unset | **Anti-poisoning gate.** Permits a vector index built with the `fake` provider to be loaded in a production read path. Default off → a `fake`-provider index is refused at query time (mirrors `LOCAL_DISPATCHER_ALLOW_STUB`). |
| `ED4ALL_GATE_ADVISORY` | unset | **Safety-critical.** Flips post-training eval gates from blocking to advisory. Materially changes promotion semantics. |
| `ED4ALL_GENERATION_TECHNIQUE` | `C5` | W5 C0..C5 generation-technique selector resolved by `lib/generation/technique_modes.py::resolve_technique_mode`. Six cumulative modes — `C0` naive → `C1` chunk-scoped → `C2` free-text/thin-JSON envelope → `C3` self-verify → `C4` refine → `C5` best-of-N + NLI verifier (the keystone). `apply_mode_to_env` projects the mode onto the Courseforge router knobs (`COURSEFORGE_OUTLINE_N_CANDIDATES` / `COURSEFORGE_BEST_OF_N` / `COURSEFORGE_BEST_OF_N_SELECT_BY` / `COURSEFORGE_OUTLINE_GRAMMAR_MODE` / `COURSEFORGE_SELF_VERIFY` / `COURSEFORGE_REFINE_ROUNDS`) so one provider codebase runs every mode. Only `C5` projects the `entailment_argmax` best-of-N selector flip. Default `C5` (ship quality); set `C1` for fast dev. Unknown/garbage values fall back to `C5` (never crash a run). Drives the `Trainforge/eval/generation_curve_runner.py` per-arm curve. Selects no provider/model — no `docs/LICENSING.md` row (the synthesized course content is generated product, not training-data pairs; see `plans/finegrain/w5-7b-quality-harness.md` §9). |
| `ED4ALL_DYNAMIC_BLOCK_PLAN` | unset (off) | Wave-2 Part 3 (keystone) content-aware block planner gate (`lib/generation/block_planner.py::plan_week_blocks`, hooked in `MCP/tools/pipeline_tools.py::_run_content_generation_outline`). Default OFF → byte-identical fixed-plan behaviour: every week's pages are composed from the hardcoded `_PAGE_TYPE_BLOCK_PLAN` template via `_page_block_plan_for`. When truthy (`1`/`true`/`yes`/`on`), a 70B planner runs ONCE PER TERMINAL OBJECTIVE (week) BEFORE that TO's pages are authored — it is fed the TO statement, its child COs (id/statement/bloom), a bounded digest of the TO's grounded source-chunk text, and the machine-readable block catalog (`Courseforge/config/block_catalog.yaml`'s `use_when`/`bloom_fit` menu) and asked to select an ORDERED block sequence (within a 5-12 budget) that teaches THAT TO's content — picking the RIGHT block type per content shape (procedure→worked example/problem, key term→vocab_card, common error→misconception, real situation→scenario, checkpoint→self_check_question/reflection_prompt, recap→summary_takeaway/checklist) — so each week is content-shaped, not template-filled. Guardrails: every `block_type` must be in `Courseforge/scripts/blocks.py::BLOCK_TYPES` (the 30-value palette — each type declares its canonical framework B-code parent B01–B15 via the catalog's `framework_block` field; SoT loader `lib/ontology/framework_blocks.py`, IB2 reconciliation map in `Courseforge/CLAUDE.md` § "Framework B-code reconciliation") (unknown dropped); `page_type` repaired to one of the five canonical types; block count clamped to budget; every CO covered by ≥1 block (a dropped CO gets a default `concept` block). The planner's `{block_type, page_type, target_co_ids}` output feeds the SAME page-descriptor assembly (`_page_id_for` page-file grouping is preserved — only WHICH blocks land on each page changes). Defaults to the 70B NVIDIA rewrite seat (`meta/llama-3.3-70b-instruct`, key `NVIDIA_API_KEY`; model override `ED4ALL_DYNAMIC_BLOCK_PLAN_MODEL` > `NVIDIA_LARGE_MODEL`). **IB7.2 reachability reconciliation:** the seat is now operator-selectable via `ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER` (default `nvidia`) so the framework-aligned planner path is reachable WITHOUT the NVIDIA key — set it to `local` to run the license-clean Apache-2.0 Qwen seat (`LOCAL_SYNTHESIS_*`). The consumer passes the resolved seat explicitly to `build_planner_provider`, so the base's `env_provider_var` is honored only via that consumer read. Fail-safe: ANY LLM error / unparseable / empty response / missing key degrades to the deterministic fixed plan inside `plan_week_blocks` (never breaks the build). Emits one `block_plan` decision event per TO (chosen types, budget, coverage, model, fallback-used). Generates PRODUCT content (block-type selection), not training data → no `docs/LICENSING.md` row (mirrors the `ED4ALL_GENERATION_TECHNIQUE` / objective-review rationale). Falsey / garbage values → off (parse-with-fallback). No-op when `COURSEFORGE_TWO_PASS` is unset (the outline phase only runs in the two-pass pipeline). |
| `ED4ALL_DYNAMIC_BLOCK_PLAN_MODEL` | per-provider | Model-ID override for the `ED4ALL_DYNAMIC_BLOCK_PLAN` planner (`lib/generation/block_planner.py::_resolve_planner_model`). Resolution (high → low): `ED4ALL_DYNAMIC_BLOCK_PLAN_MODEL` > (for `nvidia`) `NVIDIA_LARGE_MODEL` > the `meta/llama-3.3-70b-instruct` 70B default; (for `local`, IB7.2) > `LOCAL_SYNTHESIS_MODEL` > the Apache-2.0 Qwen registry default. No-op when `ED4ALL_DYNAMIC_BLOCK_PLAN` is off. Generated product content, not training data → no `docs/LICENSING.md` row. |
| `ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER` | `nvidia` | IB7.2 planner-SEAT selector (`MCP/tools/pipeline_tools.py::_build_block_planner_provider`; the base's `BlockPlannerProvider` `env_provider_var`). Default `nvidia` (byte-stable). Set to `local` to route the content-aware block planner to the license-clean Apache-2.0 Qwen seat (`LOCAL_SYNTHESIS_*`) so the framework-aligned path is reachable WITHOUT the NVIDIA 70B key. The consumer passes the resolved seat explicitly to `build_planner_provider`. No-op when `ED4ALL_DYNAMIC_BLOCK_PLAN` is off. Generated PRODUCT content (block-type selection), not training data → no `docs/LICENSING.md` row (mirrors `ED4ALL_DYNAMIC_BLOCK_PLAN`). |
| `ED4ALL_PLANNER_BLOOM_CLIMB` | unset (off) | IB7.3 programmatic Bloom-climb re-sort (`lib/generation/block_planner.py::_apply_bloom_climb`, run inside `plan_week_blocks` after the page/P4 floors + palette-v2 + IB5 injection, before `_to_page_plan`). Default OFF ⇒ strict identity (byte-stable). When truthy (`1`/`true`/`yes`/`on`), the final block set is re-sorted onto the canonical 100%-frequency template — activation → exposition (vocabulary/pretraining FIRST) → worked-example → case → check → summary — with within-tier `target_bloom` ascending (lower-order scaffolds precede higher-order; the climb UP the pyramid). The climb is the FINAL ordering authority. Generated PRODUCT content, not training data → no `docs/LICENSING.md` row. Falsey / garbage → off (parse-with-fallback). No-op when `ED4ALL_DYNAMIC_BLOCK_PLAN` is off (the planner only runs in the two-pass outline phase). |
| `ED4ALL_PLANNER_LIFECYCLE` | unset (off) | IB7.4 lifecycle open/close guarantee + slot-edit escalation (`lib/generation/block_planner.py::_ensure_lifecycle_endpoints`). Default OFF ⇒ identity. When truthy, an Activate-stage block (`hook`/`objective`/`prereq_set`) is PREPENDED to a TO that lacks an opener and a Consolidate-stage block (`summary_takeaway`/`recap`/`checklist`/`reflection_prompt`) APPENDED to a TO that lacks a closer (anti-fabrication: targets the TO's first real CO id, mirroring `_inject_palette_v2`); an over-ceiling block FIRST gets a heavier `interaction`/`feedback` weight on its hash-excluded `anatomy_slot_weights` annotation BEFORE any IB7.6 type swap (framework p.139 Step 4 — differentiate by demand, not ornament). Generated PRODUCT content, not training data → no `docs/LICENSING.md` row. Falsey / garbage → off. No-op when `ED4ALL_DYNAMIC_BLOCK_PLAN` is off. |
| `ED4ALL_PLANNER_SPACING` | unset (off) | IB7.5a within-module temporal-spacing pass (`lib/generation/block_planner.py::_apply_spacing`). Default OFF ⇒ identity. When truthy, a deterministic bounded pass (ADDS NO blocks) separates a concept-touching check/reflection (`self_check_question`/`reflection_prompt`/`assessment_item`) from the exposition that taught the SAME CO — moving an adjacent (massed) check past the next intervening non-same-CO block so a checkpoint follows intervening material (spacing axis 3 / QA-15 / p.140). Pairs with the `retrieval_presence` gate (IB7.5b). Generated PRODUCT content, not training data → no `docs/LICENSING.md` row. Falsey / garbage → off. No-op when `ED4ALL_DYNAMIC_BLOCK_PLAN` is off. |
| `ED4ALL_PLANNER_BLOOM_CEILING` | unset (off) | IB7.6b per-type Bloom-range ceiling re-route (`lib/generation/block_planner.py::_apply_bloom_ceilings`). Default OFF ⇒ identity. When truthy, a block whose `target_bloom` exceeds its catalog `bloom_ceiling` (the advisory `bloom_fit` becomes a planner gate) is RE-ROUTED — after IB7.4's slot edit — to the first higher-order target (`scenario`/`problem`/`assessment_item`) whose own ceiling admits the demanded level (no "everything block"); the re-route changes `block_type`/`page_type` only and PRESERVES `target_co_ids` (anti-fabrication). Pairs with the `bloom_type_range` gate (IB7.6c). Generated PRODUCT content, not training data → no `docs/LICENSING.md` row. Falsey / garbage → off. No-op when `ED4ALL_DYNAMIC_BLOCK_PLAN` is off. |
| `ED4ALL_PLANNER_FADING` | unset (off) | FR-INT-01 B08 guided-practice **fading-sequence** planner pass (`lib/generation/block_planner.py::_apply_fading_sequence`, run inside `_apply_ib7_passes` alongside the IB7 passes). Default OFF ⇒ identity (byte-stable). When truthy (`1`/`true`/`yes`/`on`), after a `worked_example` (B05) block that is not already followed by a faded-practice block, a single B08 completion block (`problem`/`activity`/`checklist` — first present in the palette) is INJECTED and stamped `fade_state="completion"` (the reused IB5 field); the worked_example itself is stamped `fade_state="worked"` when it carries none — realizing the framework's worked→completion→independent guided-practice fade ladder (p.139). Anti-fabrication: the injected block inherits the worked_example's `target_co_ids` (or the TO's first real CO id); no CO id is invented. Pairs with the `anatomy_slot_presence` gate's `ANATOMY_FADE_STATE_MISSING` check (warning-day-1, # TODO(calibration); rides the existing anatomy gate — no new gate row). Generated PRODUCT content, not training data → no `docs/LICENSING.md` row. Falsey / garbage → off (parse-with-fallback). No-op when `ED4ALL_DYNAMIC_BLOCK_PLAN` is off (the planner only runs in the two-pass outline phase). |
| `ED4ALL_TRIANGLE_FLOOR` | unset (off) | GAP D (IB3 constructive-alignment triangle) per-CO floor (`lib/generation/block_planner.py::_apply_triangle_floor`, run via `_apply_alignment_floors` inside `plan_week_blocks` + `_fallback_plan` LAST — AFTER every IB7 pass (climb / lifecycle / spacing / bloom-ceiling / fading) and AFTER the retrieval-interleave floor, so no IB7 re-route/re-page can move the injected gate-closing blocks). Default OFF ⇒ identity (byte-stable; mirrors `ED4ALL_WORKED_EXAMPLE_FLOOR`). Closes the 2-corpus calibration's confirmed REAL gap — the planner emits ZERO `assessment_item` blocks and has no per-CO activity floor, so `TriangleCompletenessValidator`'s `OBJECTIVE_NO_ACTIVITY` + `OBJECTIVE_NO_ALIGNED_ASSESSMENT` fire on ~100% of objectives by construction. When truthy (`1`/`true`/`yes`/`on`), for every CO REFERENCED by ≥1 selected block (the gate's audit scope), if no activity-class block (`self_check_question`/`activity`/`problem`/`scenario` — the gate's `_ACTIVITY_BLOCK_TYPES`) targets it, APPEND one (first present-in-palette) on its default page; if no `assessment_item` (the ONLY type the gate's `_ASSESSMENT_BLOCK_TYPES` accepts) targets it, APPEND one stamped `target_bloom` = the CO's declared Bloom so the gate's band-alignment test passes. Because the retrieval floor runs FIRST and the triangle floor re-scans the referenced-CO set, a CO whose ONLY activity comes from the retrieval-injected `self_check_question` still gets its missing assessment arm (no redundant activity). **Render note:** on the two-pass surface this floor runs on, the injected `assessment_item` ships as a REAL, LLM-authored, VISIBLE in-page MCQ — the rewrite tier authors it (`Courseforge/generators/_rewrite_provider.py` `assessment_item` contract: stem + `<li data-cf-distractor-index="N">` options) and it renders as `<section>{content}</section>` (`MCP/tools/pipeline_tools.py` rewrite-emit; a retry-exhausted item falls to the `_render_block_fallback_html` deterministic distractor renderer, still gate-valid). It is INDEPENDENTLY GATED CRITICAL by `outline_assessment_item_payload` / `rewrite_assessment_item_payload` (`max_critical_issues: 0`), so a structurally-empty payload BLOCKS the run — it is NOT an empty placeholder and NOT filled by W10 (`assessment_synthesis` synthesizes scored QTI ORTHOGONALLY from chunks and never reads content-tier blocks). The injected block satisfies the alignment GATE (which audits the planned block set + emits a JSON-LD entry) AND is realised as authored, gated MCQ content. Anti-fabrication: every injected block targets a REAL CO id already referenced by a block (the fixed-plan path's all-empty `target_co_ids` ⇒ empty referenced set ⇒ no-op even when ON); no CO id invented; no double-inject for a CO that already carries both arms. Pairs with the IB3 `triangle_completeness` gate (gated by `ED4ALL_ALIGNMENT_VERB_TRIPLE`; warning-day-1 — no new gate row). Generated PRODUCT content, not training data → no `docs/LICENSING.md` row. Falsey / garbage → off (parse-with-fallback). No-op when `ED4ALL_DYNAMIC_BLOCK_PLAN` / the IB7 path is off (the planner only runs in the two-pass outline phase). |
| `ED4ALL_RETRIEVAL_INTERLEAVE` | unset (off) | GAP C (IB7.5b interleaved retrieval) per-content-page floor (`lib/generation/block_planner.py::_apply_retrieval_interleave_floor`, run via `_apply_alignment_floors` inside `plan_week_blocks` + `_fallback_plan` LAST — AFTER every IB7 pass, BEFORE the triangle floor). Default OFF ⇒ identity (byte-stable; mirrors `ED4ALL_WORKED_EXAMPLE_FLOOR`). Closes the 2-corpus calibration's confirmed REAL gap — the planner hard-routes every `self_check_question`/`reflection_prompt`/`assessment_item` to `page_type='self_check'` (`_BLOCK_TYPE_DEFAULT_PAGE`), so ALL retrieval siloes onto the dedicated end-of-week self-check page and 100% of content-bearing pages carry zero interleaved retrieval (and the IB7.5a spacing pass is inert because it cannot change page grouping), so `RetrievalPresenceValidator`'s `MODULE_NO_RETRIEVAL` fires on every content module. When truthy (`1`/`true`/`yes`/`on`), it scans EVERY page-type group (not just `content` — each `page_type` becomes a DISTINCT downstream `page_id` via `_page_id_for`, so a content-bearing `overview`/`application` page is its own audited module); for any group carrying ≥1 content block (`_RETRIEVAL_EXPOSITION_TYPES`, an EXACT mirror of the validator's `CONTENT_BLOCK_TYPES`) but no retrieval block, INJECT one `self_check_question` (first present-in-palette retrieval type) stamped with THAT `page_type` and placed AFTER the group's last block so it reads as SPACED retrieval-from-memory (index>0 on the page — not a pre-quiz). The cumulative end-of-week `self_check` page is PRESERVED (this ADDS interleaved checkpoints; it never moves/removes the self-check page — a self_check page carrying only retrieval blocks is not content-bearing, so it is never audited and never gets a duplicate). Anti-fabrication: the injected block targets the page's last content block's CO(s) (a real id already on the page), falling back to the TO's first real CO id; no CO id invented; no-op for a page that already has retrieval OR has no content block. Pairs with the IB7.5b `retrieval_presence` gate (warning-day-1 — no new gate row). Generated PRODUCT content, not training data → no `docs/LICENSING.md` row. Falsey / garbage → off (parse-with-fallback). No-op when `ED4ALL_DYNAMIC_BLOCK_PLAN` / the IB7 path is off (the planner only runs in the two-pass outline phase). |
| `ED4ALL_HOME` | unset (repo-relative) | **Relocatable data root.** When set, every mutable data dir defaults to `<ED4ALL_HOME>/<dirname>` (`state`, `libv2`, `exports`, `training-captures`, `dart-output`; `uploads` lands under the relocated `state/gui/`) instead of repo-relative — unblocks non-editable (site-packages) installs + tidy Docker volumes. Per-dir overrides below keep **higher** precedence (per-dir env > `ED4ALL_HOME` > repo-relative). Centralized in `lib/paths.py`; missing dirs are created on first use. Byte-stable to the repo-relative default when unset. |
| `ED4ALL_KEY_TERMS_PAGE` | unset (off) | Feature I5 — per-terminal-objective deterministic **"Key Terms" page** gate (`lib/generation/key_terms.py` builder, hooked in `MCP/tools/pipeline_tools.py::_run_content_generation_outline` via `_build_key_terms_blocks`). Default OFF → no key-terms page is emitted; every existing snapshot / chunk-count / Studio-nav list stays BYTE-IDENTICAL, and the five-type LLM page loop (`_WEEK_PAGE_TYPES`) is unchanged. When truthy (`1`/`true`/`yes`/`on`), a DETERMINISTIC post-pass (no 7B free-authoring — runs over an existing export with no GPU) authors a `week_NN_key_terms.html` page of vocab cards per week/TO: aggregates terms = union of (the TO's COs' chunk `concept_tags`) ∪ (`domain_concept_vocabulary` surface-form matches in the TO's grounded chunk text) ∪ (objective `keyConcepts`), dedups on canonical slug, sorts; resolves each term's DEFINITION preferring a source-chunk definition sentence, falling back to the vocabulary `definition_hint`, and OMITS the term when NEITHER resolves (anti-fabrication — never invents a definition); resolves a per-term source deep-link to the defining chunk's `item_path` + heading fragment via the FROZEN `heading_slug` algorithm (byte-identical to `gui/services/source_page.heading_slug` / `grounded_answer._fragment_for`). Each surviving term is a `Block(block_type="vocab_card")` (REUSES `vocab_card`; no new BLOCK_TYPE) carrying pre-rendered grounded HTML + `template_type="key_terms"` — the rewrite tier SHORT-CIRCUITS the LLM for these blocks (the definition is already grounded + verbatim). The new `key_terms` page type is recognized by `_PAGE_ID_TYPE_RE` / `_PAGE_TYPE_LABELS` / `_PAGE_TYPE_BLOCK_PLAN` and the packager `_iter_week_groups` (sorted before `summary`); the schema adds an OPTIONAL additive `sourceLink` on the JSON-LD `KeyTerm` shape (`schemas/knowledge/courseforge_jsonld_v1.schema.json`); the Studio render path is `Courseforge/scripts/generate_course.py::_render_key_terms_section`. Emits one `content_selection` decision event per week (candidate / emitted / omitted term counts). Falsey / garbage values → off (parse-with-fallback). Generates PRODUCT content (course pages), not training data → no `docs/LICENSING.md` row. Active only on the two-pass outline phase (`COURSEFORGE_TWO_PASS=true`). |
| `ED4ALL_NEW_BLOCK_TYPES` | unset (off) | IB5 gate for the four framework-aligned pedagogical block types — `hook` (B02 activation), `multimedia` (B04, the mandatory time-based-media a11y stack), `worked_example` (B05, subgoal labels + per-step Why + fade-state), `diagram` (B06, structured long-description + data-table equivalent). Canonical resolver `lib/generation/new_block_types.py::resolve_new_block_types`; emit reader `Courseforge/scripts/blocks.py::_new_block_types_emit_enabled`; threaded into the `rewrite_html_shape` Block inputs (as `new_block_types_enabled`) by `MCP/hardening/gate_input_routing.py`. Default OFF → byte-identical emit: the four tokens are never selected by the planner (`lib/generation/block_planner.py` — the IB5 content-shape nudges + the `_inject_ib5_types` deterministic injection are gated) and never rendered (`Courseforge/scripts/generate_course.py::_render_hook_section` / `_render_multimedia_section` / `_render_worked_example_section` / `_render_diagram_section`), so existing snapshots stay byte-stable. The four tokens ARE unconditionally valid `BLOCK_TYPES` members (the dataclass stays permissive — mirrors the I6 `table`/`acronym`/`key_idea` posture); only SELECTION + RENDER + field EMIT + the validator arm are flag-gated. When truthy (`1`/`true`/`yes`/`on`), the dynamic block planner can select them per content shape (procedure→worked_example, video→multimedia, spatial→diagram, TO-opener→hook), the renderers emit each type's a11y-contract HTML (multimedia ships the captions/AD/transcript/controls skeleton even when the corpus supplies no media URL; diagram ships a long-desc `<details>` + data-`<table>`), and the `rewrite_html_shape` validator gates B04's caption/AD/transcript/controls stack + B06's long-desc/data-table at WARNING (`REWRITE_IB5_A11Y_CONTRACT`, deferred critical-flip, `# TODO(calibration)` after a ≥2-corpus FP measurement). New Block fields (`fade_state`/`long_description`/`media_a11y`) are Optional + excluded from `compute_content_hash`. No-op when `COURSEFORGE_TWO_PASS` is unset (the new types are only selectable on the dynamic-planner / two-pass outline path). Generates PRODUCT content, not training data → no `docs/LICENSING.md` row (mirrors `ED4ALL_DYNAMIC_BLOCK_PLAN`). Falsey / garbage → off (parse-with-fallback). |
| `ED4ALL_REFLECTION_CALIBRATION` | unset (off) | FR-INT-03 gate for the B11 reflection **predict-then-reveal calibration** contract — a real reflection captures the learner's PREDICTION, REVEALS the benchmark, then gives CALIBRATION feedback comparing the two (not "just ask"). Canonical resolver `lib/generation/reflection_calibration.py::resolve_reflection_calibration`; emit reader `Courseforge/scripts/blocks.py::_reflection_calibration_emit_enabled`; threaded into the Block inputs (as `reflection_calibration_enabled`) by `MCP/hardening/gate_input_routing.py`. Default OFF → byte-identical: the three new `Block` fields (`prediction_prompt` / `reveal_content` / `calibration_feedback`, all Optional/None + excluded from `compute_content_hash`) are NOT projected to the `<details>` predict-then-reveal render scaffold (`Courseforge/scripts/generate_course.py::_render_reflection_calibration` returns `""`), and BOTH validator arms no-op — the `interaction_feedback` `REFLECTION_NO_CAPTURE` arm (a B11 that only asks, no capture + no calibration feedback → warning) and the `anatomy_slot_presence` `ANATOMY_REFLECTION_NO_BENCHMARK` arm (a B11 feedback slot carrying no calibration benchmark → warning). When truthy (`1`/`true`/`yes`/`on`), the render scaffold emits the predict-then-reveal `<details>` block and both arms fire WARNING-day-1. Both arms RIDE EXISTING gates (`interaction_feedback` + `anatomy_slot_presence`, themselves gated by `ED4ALL_BLOCK_QUALITY_RUBRIC`) — NO new gate row. Generates PRODUCT content (course-page reflection a11y/pedagogy), not training data → no `docs/LICENSING.md` row (mirrors `ED4ALL_NEW_BLOCK_TYPES`). Falsey / garbage → off (parse-with-fallback). |
| `ED4ALL_RICHER_VISUAL_SYSTEM` | unset (off) | Richer-visual-system Phase 0 (keystone) gate (resolver `Courseforge/scripts/generate_course.py::_richer_visual_system_enabled`, mirroring `_callout_typed_enabled`). Fixes a real a11y defect: the emitted course pages (a) hardcode hex in `COURSEFORGE_CSS` referencing ZERO `--cf-*` tokens AND (b) never inject the a11y themes — so `high_contrast.css` / `dyslexia_friendly.css` cannot reskin the output. Default OFF → byte-identical: the page `<style>` is exactly `COURSEFORGE_CSS` (`_page_style_block` composes `COURSEFORGE_CSS` + typed-callout + `_richer_style_segments`, the last being `""` when off), no `:root` token prelude is emitted, `contentHash` is unaffected (META-only 5-key allowlist), and the snapshot suite (`Courseforge/scripts/tests/test_css_component_baseline.py`) is untouched (it is substring/literal-hex/luminance based and the richer CSS is APPENDED, never an in-place `COURSEFORGE_CSS` rewrite). When truthy (`1`/`true`/`yes`/`on`), the page `<style>` gains, APPENDED after the baseline: (1) `COURSEFORGE_TOKENS_CSS` — a `:root{}` token prelude that is a CHECKED-IN MIRROR of `templates/_base/variables.css`'s `--cf-*` set (sync test `test_richer_token_prelude_in_sync`) plus the Phase-0 exact-value (`--cf-primary-tint #ebf8ff`/`--cf-ink #1a1a1a`/`--cf-heading*`)/role/accent (`--cf-accent-*` + `*-strong` ≥3:1 variants)/frame/lifecycle/fade tokens; (2) `COURSEFORGE_RICHER_CSS` — a token-consuming, VALUE-IDENTICAL re-statement of the existing `.objectives`/`.callout`/`.misconception-card`/… component rules (boxes become themeable WITHOUT changing default appearance); (3) when a theme is selected via the `theme` param on `_wrap_page` / `patch_css_in_html` (default none — the operator trigger is a deferred product decision; the seam + param are built), that theme's `:root` override block (`_THEME_OVERRIDE_CSS`) APPENDED LAST so cascade-by-source-order wins (no specificity war), finally letting the a11y themes reskin the emitted boxes. The Defect-D patch helpers (`patch_css_in_html` / `patch_css_in_export`) compose the same `_page_style_block`, so a post-hoc CSS patch run preserves the richer segments instead of stripping them. Generates PRODUCT content (course-page a11y CSS), not training data → no `docs/LICENSING.md` row (mirrors `ED4ALL_CALLOUT_TYPED`). Falsey / garbage → off (parse-with-fallback). |
| `ED4ALL_LIBV2_ROOT` | `<repo>/LibV2/` | Absolute path to the LibV2 root directory. Also honored by `lib/libv2_storage.py` (previously not consulted there). Wins over `ED4ALL_HOME`. |
| `ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS` | `60` at the client; `300` at the content-generation providers | Per-request HTTP timeout (float seconds) for local content-generation LLM calls so a 7B authoring multi-paragraph prose isn't capped at 60s. Honored by `Trainforge/generators/_openai_compatible_client.py::OpenAICompatibleClient` as its default request timeout when the caller passes no explicit `timeout` (resolution / precedence high → low: explicit per-call `timeout` arg > `ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS` > the `DEFAULT_TIMEOUT_SECONDS` 60s floor). The Courseforge outline + rewrite tiers (`Courseforge/generators/_outline_provider.py` / `_rewrite_provider.py`) source an explicit generous default from this env var, falling back to `300.0` (not the bare 60s client default) when unset — matching the `TEXTBOOK_SYNTHESIS_TIMEOUT_SECONDS` posture. Garbage / non-finite / non-positive values fall back to the relevant default (parse-with-fallback, mirroring `ED4ALL_ANSWER_TIMEOUT_SECONDS`). Does NOT touch the grounded-answer path (that has its own `ED4ALL_ANSWER_TIMEOUT_SECONDS`). Selects no provider/model → no `docs/LICENSING.md` row. |
| `ED4ALL_MAILBOX_BASE_DIR` | `<repo>/state/mailbox/` | Orchestrator task-mailbox base directory. |
| `ED4ALL_NLI_DEVICE` | `cpu` (code) / `cuda` (project default) | Torch device for the in-process NLI classifier (`lib/classifiers/nli_classifier.py`, MoritzLaurer DeBERTa-v3-base-mnli-fever-anli) that scores groundedness/eval entailment. **This project pins `cuda` as the operator default** via the committed `.claude/settings.json` `env` block (+ `~/.bashrc`) because CPU NLI grounding is too slow for real synthesis/eval runs (~20-50x slower); the in-CODE default stays `cpu` for CI hermeticity (byte-identical to the historical load — no `.to()`/`.half()`) and graceful GPU-less fallback. `cuda` / `cuda:N` casts the model to fp16 (`.half()`) to keep VRAM ~0.4GB (shares the card with a local ollama LLM); CPU stays fp32. Graceful fallback: `cuda` requested but `torch.cuda.is_available()` False → logs a warning and falls back to CPU (never crashes a CPU/CI box). Resolved device recorded on the `GroundednessReport` as `nli_device` for provenance. Determinism note: GPU softmax is non-associative so probabilities can differ ~1e-6 from a cpu pin — the 0.70/0.50 verdict thresholds are robust to that, so a cuda-scored run is NOT a regression vs a cpu-scored pin. Mirrors `ED4ALL_EMBEDDING_DEVICE`. No `docs/LICENSING.md` row (device knob, not a provider/model). |
| `ED4ALL_NLI_MIN_FREE_VRAM_MIB` | `1024` | Free-VRAM floor (MiB) gating the in-process NLI model onto CUDA (`lib/classifiers/nli_classifier.py::resolve_min_free_vram_mib`, consumed in `NliClassifier._place_model_on_device`). On an 8GB box shared with a resident local ollama 7B (~5.3GB), free VRAM sits at ~200MiB — enough to LOAD the ~0.4GB fp16 DeBERTa head but NOT enough for its batch-8×512 forward pass, whose activations spike several hundred MiB and raise an uncaught `RuntimeError: CUDA out of memory` mid-validation that the orchestrator's broad `except Exception` swallows tracebackless (the "silent death" of a pure-local `textbook_to_course` run: NLI loads during `concept_extraction` AFTER ollama is resident, then OOMs during `post_rewrite_validation`'s `block_prose_entailment` scoring → no `blocks_final.jsonl` consumer survives → no `block_quality_rollup_report.json`). When `ED4ALL_NLI_DEVICE` resolves to cuda AND cuda is available, the loader probes free VRAM via `probe_free_vram_mib` (NVML/`pynvml`-first, falling back to `torch.cuda.mem_get_info`); if free VRAM is below this floor it falls back to CPU (fp32) for NLI scoring so the heavy generation 7B keeps the GPU (NLI is comparatively light — the ~20-50x CPU slowdown is acceptable for the bounded validation fan-out). **NVML is load-bearing on WSL2:** `torch.cuda.mem_get_info` there does NOT account for a separate process's allocation (it reports ~5.4GB "free" on a full 8GB card with ollama resident; NVML correctly reports ~150MiB), so a `mem_get_info`-only probe would wave NLI onto a full GPU and OOM anyway. `0` disables the floor (preserves the historical load-time-only OOM guard). Garbage / negative / non-integer → `1024` (parse-with-fallback, mirroring `ED4ALL_NLI_DEVICE`). A probe failure (no NVML + no `mem_get_info`) returns None → degrades to the load-time OOM guard (never crashes the GPU path on a native-Linux box where the probe is merely unavailable). Selects no provider/model → no `docs/LICENSING.md` row. |
| `ED4ALL_NLI_EVICT_FOR_CUDA` | `true` (on) | Governs the VRAM-contention resolution strategy when `ED4ALL_NLI_DEVICE` resolves to cuda + cuda is available BUT free VRAM is below `ED4ALL_NLI_MIN_FREE_VRAM_MIB` (a resident local ollama generation model is holding the card). Resolver `lib/classifiers/nli_classifier.py::resolve_evict_for_cuda`, consumed in `NliClassifier._place_model_on_device`. The better design over the ab0ce44 pure-CPU-demotion guard: generation (ollama 7B) and validation (NLI) do NOT run simultaneously within a phase, so instead of demoting NLI to CPU, the loader FIRST evicts the resident ollama model (freeing the card) via `lib/llm/vram_reclaim.py::evict_local_llm` and RE-PROBES free VRAM (NVML-first again); if free VRAM is now ≥ the floor, NLI loads on cuda (fp16) as the user wants — the 7B lazy-reloads on its next generation HTTP request (ollama auto-loads on demand, so no explicit reload is needed). CPU fallback (fp32) is the LAST RESORT only when this flag is falsey (`0`/`false`/`no`/`off` → ab0ce44 pure-CPU-fallback behavior), eviction fails / is a no-op (ollama down, no resident model, HTTP error), or the re-probe is still below the floor. The reclaim helper resolves the ollama root from `LOCAL_SYNTHESIS_BASE_URL` (the same env the local synthesis client reads; strips the OpenAI-compatible `/v1` suffix), discovers the resident model(s) via `GET {root}/api/ps`, and unloads each via a `POST {root}/api/generate` with `keep_alive: 0`. Fully graceful: any reclaim error → eviction treated as failed → CPU fallback (never crashes). Default ON because this project pins cuda NLI for speed. Garbage values → on (parse-with-fallback; only the explicit falsey tokens disable). Selects no provider/model → no `docs/LICENSING.md` row. |
| `ED4ALL_OBJECTIVE_REVIEW_PROVIDER` | unset (off) | Grounding-safe **objective-review** pass gate (`lib/objectives/objective_review.py::review_objectives`, hooked in `MCP/tools/pipeline_tools.py::_plan_course_structure` AFTER the 7B assembles the synthesized objectives and BEFORE `synthesized_objectives.json` is written). Unset/empty → strict no-op (no client constructed, objectives byte-identical to the 7B output). `nvidia` → a strong hosted reviewer (NVIDIA Nemotron, via the `nvidia` content-gen seat — reuses `OpenAICompatibleClient` exactly as `Courseforge/generators/_base.py`'s `nvidia` branch; key from `NVIDIA_API_KEY`, never hardcoded) REVIEWS and ADJUSTS the objectives' QUALITY IN PLACE: improves `statement` clarity/specificity, repairs `bloom_level`/`bloom_verb` mismatches, completes/repairs `abcd`, and makes TO statements cover their child COs. Hard guardrails — **(1)** editable surface only (sends only `{id, statement, bloom_level, bloom_verb, abcd}`; `source_refs`/`chunk_ids` NEVER sent or mutated); **(2)** identity + the id SET are IMMUTABLE (no add/remove/reorder/merge/re-id — downstream chunk `learning_outcome_refs[]` continuity depends on it; a returned id outside the original set is dropped, an omitted id keeps its original verbatim); **(3)** grounding revert — every CO whose `statement` changed is re-scored with `cosine(adjusted_CO.statement, assigned_TO.statement)` (the SAME `co_terminal_alignment` 0.45 cosine floor / `backlink_cos_to_tos` signal) and REVERTED when it drops below the floor or materially worse than the original; a bloom edit is reverted when `abcd.behavior.verb ∉ BLOOMS_VERBS[bloom_level]` (the `abcd_verb_alignment` rule); TOs (no `source_refs`) accept statement/bloom edits after only the Bloom-verb check; **(4)** ANTI-FABRICATION — never synthesizes new `source_refs`/`chunk_ids`. Fully graceful: provider error / unset key / unparseable response → warning + keep the original 7B objectives (never fails the phase). Bounded to a single review call with a generous `max_tokens` (~16384). Emits one `objective_review` decision event per call (dynamic rationale: objectives reviewed, statements adjusted, reverted-for-grounding, bloom edits applied/rejected, model, provider). Generates PRODUCT content (synthesized course objectives), NOT training data — no `docs/LICENSING.md` row (mirrors the NVIDIA rewrite-tier rationale; the NVIDIA-key licensing row already lives in `Trainforge/CLAUDE.md`). Garbage / unknown provider → off (parse-with-fallback). |
| `ED4ALL_OBJECTIVE_REVIEW_MODEL` | per-provider | Model-ID override for the objective-review pass (`lib/objectives/objective_review.py::resolve_objective_review_model`). Resolution: explicit arg > `ED4ALL_OBJECTIVE_REVIEW_MODEL` > the provider default (for `nvidia`: `NVIDIA_LARGE_MODEL` > the nemotron `DEFAULT_SYNTHESIS_MODEL`). No-op when `ED4ALL_OBJECTIVE_REVIEW_PROVIDER` is unset. Generated product content, not training data → no `docs/LICENSING.md` row. |
| `ED4ALL_OBJECTIVE_CHUNK_RELEVANCE_FLOOR` | `0.30` | Fix 1A relevance floor for the objective-dedup union prune (`lib/objectives/objective_dedup.py::dedup_candidates` / `resolve_chunk_relevance_floor`, consumed via `MCP/tools/pipeline_tools.py::_run_stage2_window_synthesis`). When near-duplicate candidate objectives merge, every member's `source_chunk_ids` is unioned onto the representative; each unioned chunk is then ranked by cosine(rep statement, chunk text) and any below this floor is PRUNED (diffuse provenance is worse than thin — "no source rather than a misleading one"). **Anti-fabrication:** prune-only (kept set ⊆ union; never adds a chunk no member cited). The **always-keep-≥1** contract supersedes the floor (a merged CO never loses all provenance). Pairs with `ED4ALL_OBJECTIVE_MAX_CHUNKS_PER_OBJECTIVE` (the top-K cap). Graceful-degrade: no `chunks_by_id` or embed-client unavailable → legacy full union (logged, no crash). Out-of-range / garbage (not in `[0.0, 1.0]`) → `0.30`. Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_OBJECTIVE_DEDUP_THRESHOLD` | `0.88` | W2 §4.2 cosine clustering threshold for the in-synthesis objective-dedup pass (`lib/objectives/objective_dedup.py::dedup_candidates`, consumed by `MCP/tools/pipeline_tools.py::_run_stage2_window_synthesis` after the NLI-grounding filter). Two candidate objectives whose statement embeddings have cosine ≥ this collapse to one canonical CO (best-grounded representative; union of `source_chunk_ids`). **Advisory starting point** — Risk R6 mandates MEASURING `max_pairwise_cosine` / `near_dup_pairs` (both surfaced on every run + in the `objective_grounding_filter` decision event) on a real corpus before pinning; the calibration harness (out of W2 scope) consumes those measurements. Out-of-range / garbage values fall back to `0.88`. Selects no provider/model, so N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set (the path that runs the dedup pass). |
| `ED4ALL_OBJECTIVE_DISTINCT_SKILL_SPLIT` | unset (off) | I3 PRONG A — distinct-skill SPLIT gate for the objective-dedup pass (`lib/objectives/objective_dedup.py::dedup_candidates` / `resolve_distinct_skill_split` / `split_clusters_for_distinct_skills`, consumed via `MCP/tools/pipeline_tools.py::_run_stage2_window_synthesis`). Default OFF → byte-identical single-link behaviour (existing dedup tests stay green). When truthy (`1`/`true`/`yes`/`on`), AFTER single-link cosine clustering forms clusters, a post-pass re-partitions any cluster spanning ≥2 DISTINCT skill signatures (verb-object/keyphrase via `lib/ontology/bloom.py::get_all_verbs` minus stopwords + `concept_tags`) so each distinct named skill keeps its OWN representative statement + grounding — closing the single-link transitive-chaining loophole (e.g. "order of operations / PEMDAS" candidates chain into a "simplify expressions" cluster, lose their statement, and demote to supporting evidence). The 0.88 same-signature collapse for exact restatements is PRESERVED (Jaccard-overlap floor; near-restatements never split). **Anti-fabrication:** re-partitions already-synthesized candidate INDICES only — never invents a statement. Surfaces `clusters_split_for_distinct_skill` + `distinct_skill_count` on the run's `grounding_signals` and emits one `objective_distinct_skill_split` decision event per dedup pass that splits. CHEAP/DETERMINISTIC + embedding-OPTIONAL (pure string ops — not a new fail-closed dependency). Falsey / garbage → off (parse-with-fallback). Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_OBJECTIVE_SOURCE_BACKFILL` | unset (off) | I3 PRONG B — source-richness BACKFILL gate (`lib/objectives/source_backfill.py::backfill_uncovered_chunks` / `resolve_source_backfill`, consumed via `MCP/tools/pipeline_tools.py::_run_stage2_window_synthesis` after dedup+split, before CO-id minting). Default OFF → no-op (canonical CO set byte-identical). When truthy, the pass retains the PRE-DEDUP grounded candidate pool and, for any CONTENT-BEARING chunk left cited by NO canonical CO, PROMOTES the best-grounded DISCARDED pre-dedup candidate citing that chunk that names a DISTINCT un-named skill into a new CO. **Anti-fabrication (hard contract):** promotion only — NO re-synthesis; each promoted CO's `source_chunk_ids` are a STRICT SUBSET of the promoted candidate's; a candidate is eligible only if its skill signature is distinct from every existing canonical CO. COVERAGE TARGET = content-bearing chunks carrying a distinct teachable skill ONLY (figures / exercises / assessment_items / front-matter get NO CO — deterministic content-bearing filter `chunk_type ∈ {explanation,concept,content}` ∧ word_count ≥ 40). Surfaces `backfill_cos_promoted` / `backfill_uncovered_before` / `backfill_uncovered_after` / `backfill_content_chunks` on `grounding_signals` and emits one `objective_source_backfill` decision event. Falsey / garbage → off. Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_OBJECTIVE_BACKFILL_COVERAGE_TARGET` | `1.0` | I3 PRONG B coverage target (`lib/objectives/source_backfill.py::resolve_coverage_target`) — the minimum fraction of content-bearing chunks the backfill drives toward before it stops promoting discarded candidates. `1.0` (default) = cover every content-bearing chunk that has an eligible discarded distinct-skill candidate; a lower value caps how aggressively the pass promotes. Clamped to `[0.0, 1.0]`; out-of-range / garbage → `1.0`. No-op when `ED4ALL_OBJECTIVE_SOURCE_BACKFILL` is off. Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_OBJECTIVE_MAX_CHUNKS_PER_OBJECTIVE` | `5` | Fix 1A top-K cap on cited chunks per MERGED objective (`lib/objectives/objective_dedup.py::dedup_candidates` / `resolve_max_chunks_per_objective`, consumed via `MCP/tools/pipeline_tools.py::_run_stage2_window_synthesis`). After the dedup union, chunks are ranked by cosine to the representative's statement; at most K (above the `ED4ALL_OBJECTIVE_CHUNK_RELEVANCE_FLOOR`) are kept. Stops the unbounded-union grab-bag (a merged CO accreted 25+ mostly-off-topic chunks). **Anti-fabrication:** prune-only. **Always-keep-≥1** is enforced even if K is misconfigured. Surfaces `max_chunks_per_objective` + `pruned_chunk_total` onto the run's `grounding_signals` (calibration, mirroring `max_pairwise_cosine`/`near_dup_pairs`) and emits a per-objective `objective_chunk_prune` decision event. Garbage / `< 1` → `5`. Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_PRODUCTION` | `0` | When `1`, enables production-mode FastMCP server settings. |
| `ED4ALL_RESEGMENT_COLLAPSED` | `1` | WS6b collapse re-segmentation gate (`lib/semantic_structure_extractor/resegment.py::resegment_collapsed_structure`, hooked in `MCP/tools/pipeline_tools.py::_extract_textbook_structure` after the merge / before the `textbook_structure.json` write). When on (default), a structure that collapsed into a SINGLE chapter carrying more than `_STRUCTURE_COLLAPSE_SECTION_THRESHOLD` (40) sections — the DART heading-parser failure mode — is re-segmented into coherent contiguous pseudo-chapters via contiguity-constrained Ward clustering over section embeddings (chain-graph connectivity → contiguous segments). Each pseudo-chapter carries `chapter_text` (drives Stage-2 `chunks_for_chapter` order_fallback) and NO `source_file`. Stamps `structureDiagnostics.{resegmented,method,k,original_section_count}` for audit. Off → pass-through unchanged. Graceful: not collapsed, flag off, or sklearn/scipy/embeddings unavailable → return chapters unchanged (never crashes extraction). Falsey values (`0`/`false`/`no`/`off`) disable. Selects no provider/model → N/A for `docs/LICENSING.md`. |
| `ED4ALL_RESEGMENT_SECTIONS_PER_CHAPTER` | `13` | WS6b target sections-per-pseudo-chapter (`resegment_collapsed_structure`). Drives the cluster count `K = clamp(round(n_sections / this), 6, 20)` (141/13 ≈ 11 on the de-risked OpenStax structure). Garbage / `< 1` → `13`. Active only on the collapse-trigger path (1 chapter / >40 sections). Selects no provider/model → N/A for `docs/LICENSING.md`. |
| `ED4ALL_ROOT` | auto-detect | Absolute path to the Ed4All project root. |
| `ED4ALL_RUN_ID` | generated | Per-run identifier consumed by every artifact emitter. |
| `ED4ALL_SKIP_ABLATION` | unset | When set, skips the post-training ablation pass. |
| `ED4ALL_STAGE_MODE` | `symlink` | How `stage_dart_outputs` materialises DART HTML (`copy` / `symlink` / `hardlink`). |
| `ED4ALL_STATE_RUNS_DIR` | `<repo>/state/runs/` | State-runs directory. Wins over `ED4ALL_HOME` for the `runs/` subtree. |
| `ED4ALL_TO_BACKLINK_FLOOR` | `0.45` cosine / `0.10` token | WS2 dual weak-link floor for the deterministic CO→TO backlink (`lib/ontology/lo_backlink.py::backlink_cos_to_tos` / `resolve_to_backlink_floor`, consumed via `MCP/tools/pipeline_tools.py::_run_stage2_window_synthesis`). PURE MEASUREMENT — every CO is STILL argmax-assigned to its nearest TO (never-unset contract); a link whose best score falls BELOW this floor is additionally stamped `weak_terminal_link=True` + `weak_terminal_link_score`, and the run surfaces `weak_to_link_count` / `weak_to_link_rate` onto `grounding_signals` (auto-forwarded into the `objective_grounding_filter` decision event's `ml_features`). Dual floor: cosine (embeddings present) and token-overlap/Jaccard (embeddings absent) — a bare float overrides cosine only, `"<cos>,<tok>"` overrides both. Instrumentation, not a cure: ~0.87 of links are weak on the current collapsed-structure corpus (root-caused in WS1); WS1's bottom-up TO derivation should drop the rate toward ~0. A floor of `0.0` disables the stamp. Out-of-range / garbage (not in `[0.0, 1.0]`) → the relevant default. Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_TO_BACKLINK_REASSIGN` | unset (off) | M5 Fix A anti-junk-drawer reassignment + validator-parity scoring for the CO→TO backlink (`lib/ontology/lo_backlink.py::backlink_cos_to_tos` / `resolve_reassign_mode`, consumed via `MCP/tools/pipeline_tools.py::_run_stage2_window_synthesis`). Default OFF preserves the byte-stable WS2 PURE-MEASUREMENT pass (reconcile-hint-honored COs are honored verbatim and EXCLUDED from `scored_count`, so a cosine-0.20 hinted link is silently kept and never counted weak — the under-report that disagrees with the `co_terminal_alignment` validator). When truthy, the backlink (1) RE-SCORES every CO against its assigned TO — INCLUDING hint-honored ones — so `weak_to_link_*` reconciles with what the validator audits (it recomputes `cosine(co, assigned_to)` for every CO with a resolvable `terminal_id`); (2) RE-POINTS a below-`ED4ALL_TO_BACKLINK_FLOOR` CO to its strictly-better argmax TO among the EXISTING TO set (never fabricates a TO; preserves never-leave-a-CO-ungrouped), surfacing the count on `grounding_signals.weak_to_link_reassigned`; (3) stamps `weak_terminal_link` on the FINAL (post-reassign) link when still below floor — an honest count of irreducibly-poor fits. NEVER changes the never-unset contract (every CO keeps a `terminal_id`). Falsey / unset (`0`/`false`/`no`/`off`/garbage) → off (parse-with-fallback, mirroring `ED4ALL_TO_BACKLINK_FLOOR`). Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_TO_CLUSTER_K` | `0` (auto) | WS1.1 FIXED target-K for **bottom-up TO derivation** Ward agglomerative clustering (`lib/objectives/objective_dedup.py::resolve_to_cluster_k` / `cluster_to_target_k`, consumed by `MCP/tools/pipeline_tools.py::_derive_terminals_bottom_up`). WS1.1 supersedes WS1's single-link cosine-threshold clustering — a real 7B run proved single-link has NO good operating point (1 mega-cluster of 67 + 1 singleton at ≤0.70; dozens of singletons at ≥0.80). Calibration on the real 68-CO embeddings proved **Ward linkage (euclidean) on L2-normalized vectors at K≈12 gives balanced clusters** (sizes [12,11,8,6,5,5,5,4,4,3,3,2]) vs average/complete linkage or any threshold. `0` (default) = AUTO: K = `max(3, min(15, ceil(n / cos_per_cluster)))` where `cos_per_cluster` = `ED4ALL_TO_COS_PER_CLUSTER`. A positive value pins K (clamped to `[1, n]`). Resolution: explicit arg > `ED4ALL_TO_CLUSTER_K` (fixed) > auto. Garbage / `<= 0` → auto. Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_TO_CLUSTER_THRESHOLD` | `0.50` | WS1 cosine clustering threshold — **now governs ONLY the no-sklearn single-link FALLBACK path** for bottom-up TO derivation (`lib/objectives/objective_dedup.py::resolve_to_cluster_threshold` / `cluster_by_cosine`). WS1.1 replaced the primary TO-clustering path with TARGET-K Ward agglomerative clustering (`cluster_to_target_k`, see `ED4ALL_TO_CLUSTER_K`); when sklearn cannot be imported, `cluster_to_target_k` falls back to single-link `cluster_by_cosine` at THIS threshold (logged warning, never crashes). Also still drives the dedup pass's TO-cluster reference constant — but NOT the dedup clustering itself (that stays single-link at 0.88 via `ED4ALL_OBJECTIVE_DEDUP_THRESHOLD`). Deliberately LOWER than 0.88 — dedup collapses near-identical restatements, whereas TO-clustering groups RELATED COs into coarse themes. Out-of-range / non-float (not in `(0.0, 1.0]`) → default. Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_TO_COS_PER_CLUSTER` | `6` | WS1.1 AUTO-K divisor for bottom-up TO derivation (`lib/objectives/objective_dedup.py::resolve_to_cos_per_cluster`, consumed by `resolve_to_cluster_k`). Approximate number of chapter objectives per terminal-objective cluster — auto-K = `max(3, min(15, ceil(n / this)))` (e.g. 68 COs / 6 ≈ 12 clusters, matching the calibrated K≈12 balanced result). Only consulted when `ED4ALL_TO_CLUSTER_K` is `0`/auto. Garbage / `< 1` → default `6`. Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_TO_CLUSTER_GUARDS` | unset (off) | P1 (Bucket A) master gate for the post-cluster CONSOLIDATE pass on **bottom-up TO derivation** (`lib/objectives/objective_dedup.py::resolve_to_cluster_guards` / `consolidate_small_clusters` / `apply_cluster_guards`, consumed by `MCP/tools/pipeline_tools.py::_derive_terminals_bottom_up` after `cluster_to_target_k`). Default OFF → byte-identical to the prior Ward-target-K result (the consolidate pass is skipped; the run stays reproducible — the operator opts in). When truthy (`1`/`true`/`yes`/`on`), every cluster below `ED4ALL_TO_OUTLIER_MIN_SIZE` is folded SMALLEST-FIRST into its nearest-centroid-cosine neighbor (provided that neighbor is `>=` `ED4ALL_TO_OUTLIER_ABSORB_FLOOR` — it HAS a clear home), iterating until all clusters meet the floor or no eligible absorption remains. This catches the pathological singleton/runt case the fixed-K Ward result leaves behind: a real ~85-CO sample-course run hit auto-K=15 EXACTLY and gave a semantic-OUTLIER assessment-item CO (the "insurance word-problem" CO-71) its OWN singleton cluster → a wholesale-hallucinated course-wide "insurance" terminal objective (TO-15) woven through ~21 pages. With the guard on, that outlier is ABSORBED into its nearest real theme instead of seeding a garbage TO. Runs PRE-id-mint and re-partitions existing CO indices ONLY (PARTITION INVARIANT — never adds/drops/invents a CO), so downstream `learning_outcome_refs` continuity holds. The split arms are counted on `grounding_signals` as `to_guard_outliers_absorbed` (singletons) / `to_guard_undersize_merged` (multi-member runts) / `to_guard_clusters_before` / `to_guard_clusters_after`, and a `content_selection` decision event fires when any cluster is consolidated. The normal balanced ~12-15-cluster case is UNCHANGED (no sub-min-size clusters → no-op). Falsey / garbage → off (parse-with-fallback, mirroring `ED4ALL_OBJECTIVE_DISTINCT_SKILL_SPLIT`). Selects no provider/model → N/A for `docs/LICENSING.md`. **Calibration caveat:** the min-size + absorb-floor defaults are starting points; this guard needs a real-corpus objectives re-derive to confirm it removes the garbage TO without over-merging coherent thin themes. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_TO_OUTLIER_MIN_SIZE` | `3` | P1 (Bucket A) min-cluster-size floor for the CONSOLIDATE pass (`lib/objectives/objective_dedup.py::resolve_to_outlier_min_size`). A TO cluster with FEWER than this many COs is a merge/absorb candidate (size-1 singletons = the outlier-guard arm; size 2..min-1 = the runt-merge arm). Garbage / `< 1` → default `3`. Only consulted when `ED4ALL_TO_CLUSTER_GUARDS` is on (or the kwarg is passed). Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_TO_OUTLIER_ABSORB_FLOOR` | `0.20` | P1 (Bucket A) "has a clear home" centroid-cosine floor for the OUTLIER-absorb decision (`lib/objectives/objective_dedup.py::resolve_to_outlier_absorb_floor`). A sub-`ED4ALL_TO_OUTLIER_MIN_SIZE` cluster is folded into its nearest-centroid neighbor only when that neighbor's centroid cosine is `>=` this floor; below it the tiny cluster may REMAIN standing (a genuinely distinct competency with no good home — not every outlier is garbage). Deliberately LOW (a tiny cluster closest to SOME sibling almost always belongs there); `0.0` absorbs any tiny cluster into its nearest neighbor unconditionally. Out-of-range (not in `[0.0, 1.0]`) / garbage → default `0.20`. No-op when `ED4ALL_TO_CLUSTER_GUARDS` is off. Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_TO_MERGE_NEAR_DUP` | unset (off) | P1 (Bucket A) gate for the SECOND post-cluster guard — near-duplicate-TO merge (`lib/objectives/objective_dedup.py::resolve_to_merge_near_dup` / `merge_near_duplicate_clusters`, run by `apply_cluster_guards` AFTER the consolidate pass). Default OFF → byte-identical. When truthy (`1`/`true`/`yes`/`on`), any two TO clusters whose CENTROID cosine is `>=` `ED4ALL_TO_MERGE_COSINE` are UNIONED (transitively, via union-find over the centroid-cosine graph — the deterministic alternative to non-deterministic silhouette-K). Fixes the fixed-K-Ward over-split of a cosine-dense theme into 2-3 near-duplicate TOs (the real run split one number-line theme into ~3 TOs at centroid cosine ~0.85). Counted as `to_guard_near_dup_clusters_merged`. PARTITION INVARIANT preserved. Independent of `ED4ALL_TO_CLUSTER_GUARDS` (each guard arm gates separately). Falsey / garbage → off (parse-with-fallback). Selects no provider/model → N/A for `docs/LICENSING.md`. **Calibration caveat:** the merge-cosine default needs real-corpus validation before the critical flip. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_TO_MERGE_COSINE` | `0.85` | P1 (Bucket A) centroid-cosine merge floor for the near-duplicate-TO merge (`lib/objectives/objective_dedup.py::resolve_to_merge_cosine`). Two TO clusters whose centroids meet/exceed this are unioned into one TO. Out-of-range (not in `(0.0, 1.0]`) / garbage → default `0.85`. No-op when `ED4ALL_TO_MERGE_NEAR_DUP` is off. Selects no provider/model → N/A for `docs/LICENSING.md`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_TRAINING_CAPTURES_DIR` | `<repo>/training-captures/` | Overrides the legacy decision-capture mirror root; honored by `lib/paths.py::get_training_captures_dir`, `lib/decision_capture.py`, `lib/streaming_capture.py`. NOT governed by `ED4ALL_LIBV2_ROOT`. Wins over `ED4ALL_HOME`. |
| `ED4ALL_VRAM_DOCTOR` | unset (off) | VRAM-contention observability gate (resolver `lib/llm/vram_doctor.py::vram_doctor_enabled`). Default OFF → the per-phase trajectory hook in `MCP/core/workflow_runner.py::run_workflow` (`_vram_doctor_snapshot`) short-circuits BEFORE any probe, so the default run path is byte-identical and pays zero NVML/ollama-HTTP cost. When truthy (`1`/`true`/`yes`/`on`), the runner snapshots free VRAM (NVML-first, the WSL2-correct `lib/classifiers/nli_classifier.py::probe_free_vram_mib` reused) + resident ollama models (`/api/ps`) before and after EACH phase and appends rows to `state/runs/<run_id>/vram_trajectory.jsonl` (sibling of `checkpoints/`, same `get_state_runs_dir()` root), leaving a forensic timeline so a contended-OOM "silent death" on the shared 8GB box is diagnosable post-hoc. Best-effort: a snapshot/write error can NEVER perturb run control flow or `final_status` (helper is fully try/except-isolated). Independent of this flag: (a) the standalone `ed4all doctor` preflight CLI (`cli/commands/doctor.py` — `snapshot_vram`→`fit_check`→`format_doctor_report`; `--json`; exit 2=would-OOM / 1=cuda→CPU-fallback / 0=ok) always works on demand, and (b) the loud non-retryable CUDA-OOM diagnostic at `MCP/core/executor.py::_execute_with_retries` (`_is_cuda_oom`) always fires on a real OOM (no flag — a true OOM is already a broken run). Selects no provider/model → no `docs/LICENSING.md` row. Falsey / garbage → off (parse-with-fallback). |

The `LLM_*` env vars (`LLM_MODE`, `LLM_PROVIDER`, `LLM_MODEL`) are CLI runtime knobs documented in § Quick Start above.

**Other `ED4ALL_*` vars not in the table above (kept out to avoid table noise):** the GUI server vars `ED4ALL_GUI_HOST` / `ED4ALL_GUI_PORT` / `ED4ALL_GUI_LEARNER` / `ED4ALL_GUI_MODE` / `ED4ALL_GUI_TOKEN` (the full-mode operator shared-secret bearer token; required before any non-loopback operator deploy) are documented in `gui/README.md` (read in `gui/server.py` / `gui/app.py` / `gui/auth.py` / `cli/commands/gui_cmd.py`). The remaining `ED4ALL_*` knobs are test-only discovery / gating overrides documented inline at their read sites, not production code paths: `ED4ALL_RUN_FULL_ARCHIVE_TEST` (gates `Trainforge/tests/test_emit_pipeline_full_archive.py`), `ED4ALL_A11Y_SMOKE_OLLAMA` (gates the live-backend smoke in `gui/tests/test_learner_a11y_gate.py`), and the per-suite fixture-slug overrides `ED4ALL_ARCHIVE_FIXTURE_SLUG`, `ED4ALL_RDF_EXPORT_FIXTURE_SLUG`, `ED4ALL_INTENT_ROUTER_FIXTURE_SLUG`, `ED4ALL_TUTORING_FIXTURE_SLUG`, `ED4ALL_STUDY_PACK_FIXTURE_SLUG` (let a tier-2 test discover a specific course slug instead of auto-discovering one). Finally, two more `ED4ALL_*`-prefixed flags are documented in the subsystem file whose surface they gate (so the root cross-cutting count of 83 above, which tracks only the rows in this section's table, excludes them): the three rewrite-tier `ED4ALL_REWRITE_FIT_WINDOW` / `ED4ALL_REWRITE_NUM_CTX` / `ED4ALL_REWRITE_TRUNCATION_TRIPWIRE` flags live in [`Courseforge/CLAUDE.md § Opt-In Behavior Flags`](Courseforge/CLAUDE.md) alongside the Courseforge two-pass rewrite tier, and the W10 assessment prose-tier selector `ED4ALL_ASSESSMENT_PROSE_PROVIDER` lives in [`Trainforge/CLAUDE.md § Opt-In Behavior Flags`](Trainforge/CLAUDE.md) alongside the `assessment_synthesis` provider it builds.

---

## Licensing & ToS Posture

Canonical reference: **`docs/LICENSING.md`**. Read it before running any training-data synthesis pass.

The project distinguishes two surfaces with different licensing exposure:

- **Development tools** (Claude Code, OpenAI Codex) generate code, prose, and shell invocations. Their ToS restricts training-data routing, but on this project that restriction is moot — these tools never produce training data, so the dev tool you use has zero effect on the trained SLM's licensing.
- **LLM providers** (`--provider claude_session` / `together` / `local`) generate the paraphrased instruction / preference pairs that become training data. The trained model is a derivative work of those outputs, so the provider's ToS + the underlying model license decide whether the corpus is shippable.

Default posture: training-data synthesis routes to license-clean providers — `--provider local` with an Apache 2.0 model (Qwen2.5-7B/14B/32B) for an air-gapped clean corpus, or `--provider together` with a hosted OSS model as the cloud fallback. The `--provider anthropic` SDK training-pair path was **removed (Phase 4)** — `run_synthesis` fails closed on it unconditionally, so training-pair synthesis is license-clean by construction. The `claude_session` route stays wired behind the `TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS` acknowledgment gate but is not recommended for training data.

**Maintenance contract:** any new behavior flag in the table above that selects an LLM provider, model ID, or synthesis backend MUST land with a corresponding row in `docs/LICENSING.md`'s "Synthesis providers" table. Drift between this file's per-flag rows and `docs/LICENSING.md` is a documentation bug.

---

## Canonical Helpers

Long-form per-validator detail + BERT ensemble member detail + pyproject extras: `docs/validation/validators.md`.

Single-source-of-truth loaders (`lib/ontology/`):

- `lib/ontology/bloom.py` — Bloom verb / level / cognitive-domain detection.
- `lib/ontology/slugs.py::canonical_slug` — unified slug helper.
- `lib/ontology/teaching_roles.py` — `(component, purpose) → role` mapper.
- `lib/ontology/taxonomy.py::load_taxonomy(name)` — generic JSON-taxonomy loader (reads `schemas/taxonomies/`).
- `lib/ontology/learning_objectives.py` — single source of truth for LO identity (`mint_lo_id`, `validate_lo_id`, `hierarchy_from_id`, `split_terminal_chapter`). Pattern `^[A-Z]{2,}-\\d{2,}$` mirrors `schemas/knowledge/courseforge_jsonld_v1.schema.json`.

Validators (`lib/validators/`) — wiring in `docs/validation/gates.md`. Load-bearing thresholds:

- `kg_quality.py` — KG-quality report; thresholds **0.95 / 0.95 / 0.95 / 0.5** (completeness / consistency / accuracy / coverage). (As of the coverage-semantics redesign, the `coverage` floor thresholds chunk-anchored DomainConcept concept-node grounding — the share of concept nodes touched by ≥1 chunk-evidenced edge — NOT the old asserted/(asserted+derived) edge share; the legacy ratio survives unthresholded as `asserted_edge_share`.)
- `curie_anchoring` (binary per-pair anchoring sentinel) — default **`min_pair_anchoring_rate=0.95`**; supersedes deprecated `curie_preservation` shim (Wave 135c→135d migration).
- Statistical-tier embedding validators (`objective_assessment_similarity`, `concept_example_similarity`, `objective_roundtrip_similarity`, `bloom_classifier_disagreement`, `co_terminal_alignment`, `source_coverage`) graceful-degrade contract: missing `[embedding]` pyproject extras emit warning-severity `EMBEDDING_DEPS_MISSING` GateIssue with `passed=True` unless `TRAINFORGE_REQUIRE_EMBEDDINGS=true` flips to fail-closed.

`schemas/knowledge/course.schema.json` is the canonical shape for Trainforge-emitted `course.json` consumed by LibV2.

---

## Individual Project Guides

- **SemantiK**: `SemantiK/CLAUDE.md`
- **Courseforge**: `Courseforge/CLAUDE.md`
- **Trainforge**: `Trainforge/CLAUDE.md`
- **LibV2**: `LibV2/CLAUDE.md`
- **Chunker**: `Trainforge/chunker/` — canonical chunker shared by the conversion (SemantiK), IMSCC, and Trainforge synthesis paths. See `Trainforge/CLAUDE.md` § "Chunking" for the surface contract.
- **Ontology map + v0.2.0 changes**: `schemas/ONTOLOGY.md`

---

## Training Pipeline

SLM training is a post-import LibV2 stage, not a step in `Trainforge/process_course.py`. Top-level command: `ed4all run trainforge_train --course-code <slug> --base-model <name>`. Full deep-dive (base-model registry, provider config, 5×3 eval matrix, 7-hash provenance, promotion workflow, decision-capture contract): `Trainforge/CLAUDE.md § Training Pipeline`.

---

## Training Data Export

Formats: `jsonl` (default), `alpaca`, `openai`, `dpo`. CLI: `ed4all export-training <run_id> --format <fmt> --output <path>`. Full reference: `Trainforge/CLAUDE.md § Training Data Export`.

---

## Summary Checklist

Before starting any workflow:

- [ ] MCP server running
- [ ] TodoWrite initialized
- [ ] Decision capture configured
- [ ] GENERATION_PROGRESS.md cleared/ready
- [ ] Appropriate config loaded

During execution:

- [ ] Maximum 10 parallel tasks per batch
- [ ] All decisions logged with rationale (20+ chars)
- [ ] One agent per file
- [ ] Wait for batch completion before next batch

After completion:

- [ ] All tasks marked completed
- [ ] Training captures exported
- [ ] GENERATION_PROGRESS.md updated
- [ ] Errors reviewed and addressed
