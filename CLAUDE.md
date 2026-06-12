# Ed4All Hybrid Orchestrator

Unified orchestration system for DART, Courseforge, Trainforge, and LibV2.

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
| `intake_remediation` | Import and remediate IMSCC | 4 |
| `batch_dart` | Batch PDF to HTML conversion | 4 |
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
├── DART/                    # PDF to accessible HTML conversion
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
- DART: WCAG compliance check
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

**DART tools** — see `DART/CLAUDE.md § MCP Tools` (includes Source-Provenance Output contract).

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

**Phase-name dispatch override** (`MCP/core/executor.py::_PHASE_TOOL_MAPPING`): five phases route by phase name, not agent name — `content_generation_outline` → `run_content_generation_outline`; `inter_tier_validation` → `run_inter_tier_validation`; `content_generation_rewrite` → `run_content_generation_rewrite`; `post_rewrite_validation` → `run_post_rewrite_validation`; `imscc_chunking` → `run_imscc_chunking`. Validator-only phases declare `agents: []` in `config/workflows.yaml`; `workflow_runner._create_phase_tasks` synthesizes a virtual `phase-handler` task only when the phase appears in this map. The mapping cannot be inferred from YAML.

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

`planning → content_generation (batches of 10) → packaging (IMSCC) → validation (QA + WCAG) → finalization`. Full phase shapes: `config/workflows.yaml::course_generation`.

### Other workflows

`intake_remediation` (IMSCC parse → remediate → repackage) and `rag_training` (extraction → indexing → assessment_generation → validation) — see `config/workflows.yaml` for canonical phase shapes.

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
       Wave 1.8: re-scales duration_weeks from the extractor's
       chapter-driven max(8, len(chapters)) to the objective-driven
       max(8, ceil(len(chapter_objectives) / WAVE18_COS_PER_WEEK))
       when --weeks is unset. The objective count is the
       authoritative pacing signal (a textbook with 6 chapters but
       30 COs paces at 15 weeks, not 8). Skipped for
       --reuse-objectives runs (operator's pacing decisions are
       preserved verbatim).

6. content_generation
   └── Generate course content modules (parallel batches of 10). Every
       emitted sourceId must resolve against the DART staging manifest
       (source_refs gate).

7. packaging
   └── Package course as IMSCC via the mature multi-file packager.

8. trainforge_assessment (optional)
   └── Generate assessments from the IMSCC package. Fails closed if any
       assessment objective_id isn't covered by a chunk's
       learning_outcome_refs.

9. training_synthesis (optional)
   └── Synthesize instruction + preference training pairs from the
       generated chunks + assessments. Routes via the
       `training-synthesizer` agent (tool: `synthesize_training`).
       Optional phase: skipped when no `ANTHROPIC_API_KEY` or when
       `--skip-training` is passed on the CLI. Emits per-pair resume
       sidecar at `training_specs/.synthesis_pairs_checkpoint.jsonl`
       so a mid-run crash on multi-hour local-LLM rebuilds resumes
       past every accepted pair; opt out via `--no-checkpoint`.

10. libv2_archival
   └── Archive course artifacts to LibV2 (raw PDFs, DART HTML, IMSCC,
       RAG corpus). Gated by libv2_manifest integrity checks.

11. vector_indexing (optional)
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

12. finalization
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

### DART / Remediation Agents

| Agent | Purpose |
|-------|---------|
| `dart-automation-coordinator` | Orchestrate PDF conversion |
| `dart-converter` | Multi-source synthesis conversion |
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

### Conversion Quality (DART)

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
| `course_generation` | 16 | 2 | 18 |
| `intake_remediation` | 2 | 0 | 2 |
| `batch_dart` | 2 | 0 | 2 |
| `rag_training` | 4 | 3 | 7 |
| `textbook_to_course` | 38 | 38 | 76 |
| `trainforge_train` | 2 | 0 | 2 |
| **Total** | **64** | **43** | **107** |

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

Environment-variable toggles gate opt-in strict / stable-ID / provenance / experimental-rule-graph behavior. All default off to preserve backward compatibility with legacy corpora. See `schemas/ONTOLOGY.md` § 12 for full rationale per flag.

Per-flag rows now live in subsystem CLAUDE.md files (one owner per prefix):

| Prefix | Owner | Flag count |
|--------|-------|-----------:|
| `TRAINFORGE_*` / `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` / `CURRICULUM_ALIGNMENT_*` / `WAVE18_*` | [`Trainforge/CLAUDE.md § Opt-In Behavior Flags`](Trainforge/CLAUDE.md) | 46 |
| `DART_*` | [`DART/CLAUDE.md § Opt-In Behavior Flags`](DART/CLAUDE.md) | 6 |
| `COURSEFORGE_*` / `COURSEPLANNER_*` / `TEXTBOOK_SYNTHESIS_*` | [`Courseforge/CLAUDE.md § Opt-In Behavior Flags`](Courseforge/CLAUDE.md) | 17 |
| `DECISION_*` / `ED4ALL_*` / `LOCAL_DISPATCHER_*` / `MCP_ORCHESTRATOR_*` / `LLM_*` (cross-cutting) | root (table below) | 31 |

### Cross-cutting flags (root-owned)

| Flag | Default | Purpose |
|------|---------|---------|
| `DECISION_VALIDATION_STRICT` | unset | Fails closed on unknown `decision_type` values in decision captures. |
| `MCP_ORCHESTRATOR_LLM_MODEL` | `claude-opus-4-7` | Pins the Anthropic model ID used by `MCP/orchestrator/llm_backend.py::DEFAULT_ANTHROPIC_MODEL`; per-run `LLM_MODEL` keeps higher precedence. |
| `LOCAL_DISPATCHER_ALLOW_STUB` | unset | Permits `LocalDispatcher` to emit a stubbed `PhaseOutput` when no `agent_tool` is wired. Tests / dry-run only. |
| `ED4ALL_AGENT_DISPATCH` | unset | Routes subagent-classified agents through `dispatcher.dispatch_task` instead of in-process tool registry. |
| `ED4ALL_AGENT_TIMEOUT_SECONDS` | `1800` | Per-task subagent dispatch mailbox timeout. |
| `ED4ALL_ANSWER_PROVIDER` | `local` | Selects the grounded-answer backend (runtime Q&A inference) from the W-D12 `_OPENAI_COMPATIBLE_PROVIDERS` registry. **Loopback-only:** a resolved non-loopback `base_url` raises `AnswerProviderNotLocal` (Phase IA: no cloud arm on the answer path, ever). No escape-hatch env. Licensing row in `docs/LICENSING.md` § "Grounded-answer provider". |
| `ED4ALL_ANSWER_MODEL` | per-provider | Model ID override for the answer backend. Resolution chain: explicit arg > `ED4ALL_ANSWER_MODEL` > registry `model_env` (`local` → `LOCAL_SYNTHESIS_MODEL`) > registry default (`qwen2.5:14b-instruct-q4_K_M`). The Docker compose stack sets this to the lighter `qwen2.5:7b-instruct-q4_K_M` (fits common 8GB GPUs fully resident); the code-registry default stays 14b. See `docs/operations/docker.md`. |
| `ED4ALL_ANSWER_TIMEOUT_SECONDS` | `120` | Answer-client HTTP timeout (long passages, slow local GPU). Garbage values fall back to the default. |
| `ED4ALL_ANSWER_NUM_CTX` | `4096` | Serving-window token budget for the grounded-answer prompt (`lib/retrieval/_prompts.py`). Honest about the common Ollama default (`num_ctx=4096`), which silently truncates the prompt HEAD when exceeded. Drives the **token-aware** context budget (math-safe 2.5 chars/token divisor) sized to fit system prompt + passages + question + allowed-id enumeration + remediation headroom + `max_tokens` inside the window; trailing passages are dropped whole before the question is ever truncated. Set this to the model server's ACTUAL window (e.g. `8192` for a long-context Modelfile, matching `OLLAMA_CONTEXT_LENGTH`) so the budget shrinks the prompt to fit. A post-call tripwire (`answer_composer._check_prompt_not_truncated`) compares the server-reported `usage.prompt_tokens` against the local estimate and raises `PromptTruncatedError` (fail-closed) on a large shortfall rather than letting silent head-truncation fabricate citations. Garbage / non-positive values fall back to the default. |
| `ED4ALL_ANSWER_CITATION_PRUNE` | `shadow` | Three-valued (`off` / `shadow` / `on`) governor of the claim-attribution citation **prune + add** pass at answer-composition time (`lib/retrieval/citation_attribution.py`, hooked in `grounded_answer.answer_course_question` strictly post-citation-gate). Attribution runs over ALL gate-eligible (renderer-included) passages. `on`: drop model-cited citations that back zero answer sentences (never below 1 remaining — all-prune is a logged no-op) AND credit an uncited high-support passage the model under-cited (anchor-must-resolve, capped at 2 per answer); final citations sorted strongest-supporter-first. `shadow` (code default): compute + emit the `grounded_answer_citation_prune` capture + warnings + `supporting_excerpt`/`supported_claim_count`, but mutate no citations. `off`: skip entirely. NEVER changes an answer verdict. The Docker compose stack sets this `on`. No licensing row (no provider/model). |
| `ED4ALL_ANSWER_PRUNE_MIN_OVERLAP` | `0.25` | Float support threshold for the PRUNE decision (`citation_attribution.resolve_min_overlap`; also the `prune_min_overlap` kwarg). A citation supports a claim iff its 4-shingle containment ≥ this OR its content-token coverage ≥ 0.80 (fixed secondary floor). Deliberately low (precision of the keep-vs-drop decision over recall). The separate ADD bar is `ED4ALL_ANSWER_ADD_MIN_SHINGLE` (default `0.50`, shingle-only) + a relative "out-support the kept set" gate. Out-of-range / garbage values fall back to the default. The Docker compose stack sets this `0.444444` (the 2026-06-12 single-course union-corpus precision-floored calibration recommendation; code default unchanged). No licensing row. |
| `ED4ALL_ANSWER_ADD_MIN_SHINGLE` | `0.50` | Float shingle floor for the ADD decision (`citation_attribution.resolve_add_min_shingle`; threaded through `grounded_answer._apply_citation_attribution`). An uncited gate-eligible passage is credited only if its 4-shingle containment ≥ this on some claim AND it out-supports the strongest kept citation on that claim (additions are precision-first, capped at 2/answer). Twice the prune default by design (the failure cost of a spurious added citation is a fabricated-looking source). The single-course union-corpus attribution calibration (2026-06-12) measured a median cited shingle of 0.000 — 0.500 below this bar, so additions rarely fire — warranting the KNOB without lowering the conservative default. Out-of-range / garbage values fall back to the default. No licensing row (no provider/model). |
| `ED4ALL_ANSWER_NLI_ADD` | `off` | Three-valued (`off` / `shadow` / `on`) governor of the **NLI-based citation-ADD** arm (`citation_attribution.resolve_nli_add_mode`; hooked in `grounded_answer.answer_course_question` strictly AFTER the lexical prune+add pass, step 7b). The entailment-driven successor to the shingle ADD arm, which the 2026-06-12 under-citing investigation measured unsalvageable (paraphrase answers never quote; median cited shingle 0.000 → zero adds even at bar 0.10). REUSES the groundedness scorer's per-claim NLI verdicts (never runs NLI twice; needs `with_groundedness`). Credits an uncited gate-eligible supporter only under a COMPOSITE criterion: NLI entailment ≥ 0.75 (windowed, scorer-v2) AND claim↔chunk content-token coverage ≥ 0.65 AND every numeric literal in the claim text present in the chunk (NLI is number-blind) AND the chunk anchors/resolves AND ≤ 2 adds/answer. **Default `off`** — unlike the lexical arm's `shadow` default, the NLI model is a ~750 MB lazy load that must NOT touch the default answer path. `shadow`: compute would-adds, emit the `grounded_answer_nli_citation_add` capture + the additive `nli_citation_add` diagnostics block (eval aggregates `shadow_nli_add`) + warnings, mutate NOTHING. `on`: actually add (anchor-resolved, sorted after existing citations, cap 2); ships dark. NEVER changes an answer verdict in any mode. The Docker compose stack sets this `shadow`. Garbage values fall back to `off`. No licensing row (no provider/model). |
| `ED4ALL_EMBEDDING_PROVIDER` | `st` | Selects the retrieval-index embedding backend from `lib/embedding/providers.py::_EMBEDDING_PROVIDERS` (`st` in-process sentence-transformers / `local-openai` local `/v1/embeddings` server / `fake` deterministic test vectors). Registry entries, NOT subclasses. Not training-data synthesis; licensing row in `docs/LICENSING.md` § "Embedding providers". |
| `ED4ALL_EMBEDDING_MODEL` | per-provider | Model ID override for the embedding provider (e.g. `BAAI/bge-large-en-v1.5`). Resolution chain: explicit arg > env var > registry default. |
| `ED4ALL_EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | Base URL of the local OpenAI-compatible `/v1/embeddings` server (`local-openai` provider only; Ollama / vLLM / llama.cpp). |
| `ED4ALL_EMBEDDING_API_KEY` | `local` | Optional bearer token for the local embedding server (`local-openai` only; most local servers ignore it). |
| `ED4ALL_EMBEDDING_DEVICE` | `cpu` | Torch device for the in-process `st` provider. Default `cpu` for determinism; `cuda` allowed for speed (recorded in the index manifest so mixed-provenance comparisons are detectable). |
| `ED4ALL_EMBEDDING_BATCH_SIZE` | `16` | Encode batch size for the embedding client (replay parameter, recorded in the index manifest). |
| `ED4ALL_EMBEDDING_ALLOW_FAKE` | unset | **Anti-poisoning gate.** Permits a vector index built with the `fake` provider to be loaded in a production read path. Default off → a `fake`-provider index is refused at query time (mirrors `LOCAL_DISPATCHER_ALLOW_STUB`). |
| `ED4ALL_GATE_ADVISORY` | unset | **Safety-critical.** Flips post-training eval gates from blocking to advisory. Materially changes promotion semantics. |
| `ED4ALL_HOME` | unset (repo-relative) | **Relocatable data root.** When set, every mutable data dir defaults to `<ED4ALL_HOME>/<dirname>` (`state`, `libv2`, `exports`, `training-captures`, `dart-output`; `uploads` lands under the relocated `state/gui/`) instead of repo-relative — unblocks non-editable (site-packages) installs + tidy Docker volumes. Per-dir overrides below keep **higher** precedence (per-dir env > `ED4ALL_HOME` > repo-relative). Centralized in `lib/paths.py`; missing dirs are created on first use. Byte-stable to the repo-relative default when unset. |
| `ED4ALL_LIBV2_ROOT` | `<repo>/LibV2/` | Absolute path to the LibV2 root directory. Also honored by `lib/libv2_storage.py` (previously not consulted there). Wins over `ED4ALL_HOME`. |
| `ED4ALL_MAILBOX_BASE_DIR` | `<repo>/state/mailbox/` | Orchestrator task-mailbox base directory. |
| `ED4ALL_PRODUCTION` | `0` | When `1`, enables production-mode FastMCP server settings. |
| `ED4ALL_ROOT` | auto-detect | Absolute path to the Ed4All project root. |
| `ED4ALL_RUN_ID` | generated | Per-run identifier consumed by every artifact emitter. |
| `ED4ALL_SKIP_ABLATION` | unset | When set, skips the post-training ablation pass. |
| `ED4ALL_STAGE_MODE` | `symlink` | How `stage_dart_outputs` materialises DART HTML (`copy` / `symlink` / `hardlink`). |
| `ED4ALL_STATE_RUNS_DIR` | `<repo>/state/runs/` | State-runs directory. Wins over `ED4ALL_HOME` for the `runs/` subtree. |
| `ED4ALL_TRAINING_CAPTURES_DIR` | `<repo>/training-captures/` | Overrides the legacy decision-capture mirror root; honored by `lib/paths.py::get_training_captures_dir`, `lib/decision_capture.py`, `lib/streaming_capture.py`. NOT governed by `ED4ALL_LIBV2_ROOT`. Wins over `ED4ALL_HOME`. |

The `LLM_*` env vars (`LLM_MODE`, `LLM_PROVIDER`, `LLM_MODEL`) are CLI runtime knobs documented in § Quick Start above.

**Other `ED4ALL_*` vars not in the table above (kept out to avoid table noise):** the GUI server vars `ED4ALL_GUI_HOST` / `ED4ALL_GUI_PORT` / `ED4ALL_GUI_LEARNER` / `ED4ALL_GUI_MODE` / `ED4ALL_GUI_TOKEN` (the full-mode operator shared-secret bearer token; required before any non-loopback operator deploy) are documented in `gui/README.md` (read in `gui/server.py` / `gui/app.py` / `gui/auth.py` / `cli/commands/gui_cmd.py`). The remaining `ED4ALL_*` knobs are test-only discovery / gating overrides documented inline at their read sites, not production code paths: `ED4ALL_RUN_FULL_ARCHIVE_TEST` (gates `Trainforge/tests/test_emit_pipeline_full_archive.py`), `ED4ALL_A11Y_SMOKE_OLLAMA` (gates the live-backend smoke in `gui/tests/test_learner_a11y_gate.py`), and the per-suite fixture-slug overrides `ED4ALL_ARCHIVE_FIXTURE_SLUG`, `ED4ALL_RDF_EXPORT_FIXTURE_SLUG`, `ED4ALL_INTENT_ROUTER_FIXTURE_SLUG`, `ED4ALL_TUTORING_FIXTURE_SLUG`, `ED4ALL_STUDY_PACK_FIXTURE_SLUG` (let a tier-2 test discover a specific course slug instead of auto-discovering one).

---

## Licensing & ToS Posture

Canonical reference: **`docs/LICENSING.md`**. Read it before running any training-data synthesis pass.

The project distinguishes two surfaces with different licensing exposure:

- **Development tools** (Claude Code, OpenAI Codex) generate code, prose, and shell invocations. Their ToS restricts training-data routing, but on this project that restriction is moot — these tools never produce training data, so the dev tool you use has zero effect on the trained SLM's licensing.
- **LLM providers** (`--provider anthropic` / `claude_session` / `together` / `local`) generate the paraphrased instruction / preference pairs that become training data. The trained model is a derivative work of those outputs, so the provider's ToS + the underlying model license decide whether the corpus is shippable.

Default posture: training-data synthesis routes to license-clean providers — `--provider local` with an Apache 2.0 model (Qwen2.5-7B/14B/32B) for an air-gapped clean corpus, or `--provider together` with a hosted OSS model as the cloud fallback. Anthropic providers stay wired for backward compatibility but are not the recommended default for training data.

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
- Statistical-tier embedding validators (`objective_assessment_similarity`, `concept_example_similarity`, `objective_roundtrip_similarity`, `bloom_classifier_disagreement`) graceful-degrade contract: missing `[embedding]` pyproject extras emit warning-severity `EMBEDDING_DEPS_MISSING` GateIssue with `passed=True` unless `TRAINFORGE_REQUIRE_EMBEDDINGS=true` flips to fail-closed.

`schemas/knowledge/course.schema.json` is the canonical shape for Trainforge-emitted `course.json` consumed by LibV2.

---

## Individual Project Guides

- **DART**: `DART/CLAUDE.md`
- **Courseforge**: `Courseforge/CLAUDE.md`
- **Trainforge**: `Trainforge/CLAUDE.md`
- **LibV2**: `LibV2/CLAUDE.md`
- **Chunker**: `Trainforge/chunker/` — canonical chunker shared by DART, IMSCC, and Trainforge synthesis paths. See `Trainforge/CLAUDE.md` § "Chunking" for the surface contract.
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
