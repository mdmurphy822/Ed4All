# Ed4All Hybrid Orchestrator

Unified orchestration system for SemantiK, Courseforge, Trainforge, and LibV2.

## Quick Start

### Canonical entry point

> **Per-stage invocation** (stop-after / reuse / stage subcommands), **the timeout knobs
> that actually fire** (`ED4ALL_TASK_TIMEOUT_MINUTES` for slow in-process synthesis, not
> the batch/mailbox ones), the **outline-vs-rewrite naming trap**, and the **pure-local
> constrained-VRAM env recipe**: see [`docs/operations/pipeline-invocation.md`](docs/operations/pipeline-invocation.md).

```bash
# Primary: run any workflow end-to-end via the unified CLI
ed4all run <workflow_name> --corpus <PATH> --course-name <NAME> [--mode local|api]

# Examples
ed4all run textbook-to-course --corpus textbook.pdf --course-name PHYS_101
ed4all run textbook-to-course --corpus ./pdfs/ --course-name BIO_201 --weeks 16
ed4all run rag_training --corpus course.imscc --course-name CHEM_101 --mode api
ed4all run textbook-to-course --corpus x.pdf --course-name T --dry-run   # plan only
ed4all run textbook-to-course --resume WF-20260420-abc12345               # resume

# ed4all stop: graceful "checkpoint on command". Drops a stop sentinel; the run
# finishes its in-flight unit, checkpoints it, and pauses (exit code 3) —
# worst-case loss is one in-flight LLM call. Resume with a PLAIN --resume (never
# --force after a stop — force clears the resume sidecars). SIGTERM/Ctrl-C to a
# live `ed4all run` is the same request (signal again to hard-kill). Full
# runbook: docs/operations/pipeline-invocation.md § 7.
ed4all stop WF-20260420-abc12345    # pause ONE run at its next unit boundary
ed4all stop --all                   # global STOP_ALL — pause + BLOCK all runs
ed4all stop --clear-all             # remove STOP_ALL (operator-owned)

# --stop-after <phase>: halt cleanly AFTER the named phase, skipping all
# downstream. Canonical "retrieval-ready course, no training synthesis"
# slice stops after imscc_chunking. Phase name validated (unknown ->
# error). Full semantics: docs/operations/pipeline-invocation.md.
ed4all run textbook-to-course --corpus pdfs/ --course-name PHYS_101 \
  --skip-training --stop-after imscc_chunking

# Hosted large-model build profile (--provider nvidia = the vendor
# endpoint-registry key; SETUP — nothing dispatches to the cloud seat by
# default; gated on a later RUN discussion). Full routing detail (YAML
# redirect, seat pins, licensing caveat): see
# docs/operations/pipeline-invocation.md § 3.1. Run --dry-run first.
export COURSEFORGE_TWO_PASS=true
ed4all run textbook-to-course --provider nvidia --course-name PHYS_101 \
  --corpus slice.pdf --skip-dart --skip-training \
  --stop-after imscc_chunking --dry-run   # preflight: resolve+assert, NO dispatch

# --reuse-objectives: pin a prior objectives JSON instead of re-dispatching
# the course-outliner (kills re-run LLM-nondeterminism drift). Accepts both
# the Courseforge + LibV2 archive shapes; normalized on disk. Also valid on
# --resume (patches the persisted params before the resumed course_planning
# runs). See docs/operations/pipeline-invocation.md § 3.
ed4all run textbook-to-course --corpus pdfs/ --course-name PHYS_101 \
  --reuse-objectives Courseforge/exports/PROJ-PHYS_101-.../01_learning_objectives/synthesized_objectives.json

# ed4all objectives restructure: DETERMINISTICALLY (no LLM) rebuild an existing
# objectives doc — lexical dedup (E), vacuity annotate/drop (B),
# chapter-anchored TO re-derivation (A), sub-objective quality (D) — in minutes
# instead of a 7B re-roll. Writes <input>.restructured.json + restructure_report.json;
# feed the output straight back into --reuse-objectives (it round-trips that shape).
ed4all objectives restructure \
  Courseforge/exports/PROJ-PHYS_101-.../01_learning_objectives/synthesized_objectives.json \
  --course-name PHYS_101 --drop-vacuous

# --reuse-conversion: reuse a prior SemantiK conversion (skips the
# model-nondeterministic v2 cascade when prior artifacts exist). Mirrors
# ED4ALL_REUSE_CONVERSION (flag wins). See SemantiK/CLAUDE.md §3.3a.
ed4all run textbook-to-course --corpus pdfs/ --course-name PHYS_101 \
  --reuse-conversion

# Phase 5: stage-by-stage Courseforge two-pass subcommands — re-run a
# single tier against an existing export (upstream phases pre-populate
# from disk). See Courseforge/CLAUDE.md "Operator stage subcommands".
export COURSEFORGE_TWO_PASS=true
ed4all run courseforge-outline --course-name PHYS_101              # outline tier only
ed4all run courseforge-validate --course-name PHYS_101             # validators only
ed4all run courseforge-rewrite --course-name PHYS_101 \
  --blocks assessment_item,objective                                # per-block-TYPE rewrite
# I4 stage 2 — two ADDITIVE finer-grained rewrite-eviction scopes (both stack
# with --blocks; the rewrite tier consumes them). --block-ids: exact
# block-instance IDs (shape {page_id}#{block_type}_{slug}_{idx}). --pages: an
# exact page_id (e.g. week_01_content_02) OR a module prefix (e.g. week_01) for a
# whole week/module. All three unset => byte-identical failure-driven reuse; an
# unknown id / unmatched page fails the rewrite phase LOUDLY (never a silent no-op).
ed4all run courseforge-rewrite --course-name PHYS_101 \
  --block-ids 'week_01_content_02#example_derivative_03' \
  --pages week_01                                                    # instance + page/module scope
ed4all run courseforge --course-name PHYS_101 --force               # full two-pass slice

# --license-note / --attribution: optional corpus-provenance declarations
# recorded on the LibV2 course_manifest (license.note / attribution.statement,
# mirrored into the emitted NOTICE). See docs/operations/library-versioning.md
# + docs/operations/demo-course.md.
ed4all run textbook-to-course --corpus pdfs/ --course-name PHYS_101 \
  --license-note 'CC-BY-4.0' --attribution 'Access for free at openstax.org'

# Standalone verbs (no full pipeline run):
# ed4all convert — thin accessible-HTML remediation slice: PDF (or dir of PDFs,
# or dir of publisher HTML) → {stem}_accessible.html + sidecars, no course
# scaffolding / LibV2 / index. See docs/operations/convert-verb.md.
ed4all convert slice.pdf --output ./out/

# ed4all import-docs — deterministic, LLM-free Markdown/docs-tree → clean
# accessible-HTML corpus (honors mkdocs.yml nav) + import_manifest.json; feed
# the output dir straight to `ed4all run textbook-to-course --corpus <DIR>`.
ed4all import-docs ./docs-tree --output ./corpus/

# ed4all support-bundle — assemble a redacted .tar.gz of run state + doctor
# post-mortem for sharing (decision captures excluded unless --include-captures).
# See docs/operations/support-bundle.md.
ed4all support-bundle --output ./ed4all-support.tar.gz

# ed4all backup — full data-dir backup .tar.gz (0600, honors ED4ALL_HOME);
# --verify recomputes member sha256 vs manifest + runs LibV2 fsck. See
# docs/operations/backup-restore.md.
ed4all backup --output ./ed4all-backup.tar.gz
ed4all backup --verify ./ed4all-backup.tar.gz
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
| `LLM_PROVIDER` | `anthropic` | Provider in api mode: `anthropic`, `mock`, or any name in the OpenAI-compatible endpoint registry (`config/endpoints.yaml` → `_OPENAI_COMPATIBLE_PROVIDERS`, e.g. `local`, `together`); a new provider is a registry entry, never a subclass. `openai` is a deprecated alias for `local` (emits `DeprecationWarning`). |
| `LLM_MODEL` | per-provider | Model ID override (e.g., a specific Claude release). |
| `ANTHROPIC_API_KEY` | — | Required for api mode with Anthropic. |
| registry key envs | — | API keys for OpenAI-compatible registry providers come from each `config/endpoints.yaml` entry's `api_key_env` (e.g. `LOCAL_SYNTHESIS_API_KEY`, `TOGETHER_API_KEY`). |

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
| `courseforge-rewrite` | Phase 5 stage subcommand — re-run only the rewrite tier (`content_generation_rewrite` + `post_rewrite_validation`). Three ADDITIVE (stacking) eviction scopes over the tier's default failure-driven `blocks_final.jsonl` reuse, all consumed by the rewrite tier only: `--blocks <type1,type2>` (wired through `target_block_ids`, per-block-TYPE); `--block-ids <id1,id2>` (wired through `target_block_instance_ids`, exact block-instance IDs, shape `{page_id}#{block_type}_{slug}_{idx}`); `--pages <p1,p2>` (wired through `target_page_ids`, exact `page_id` OR module prefix e.g. `week_01`). Named blocks re-roll even after a prior success; every out-of-scope block keeps byte-identical cache reuse. All three unset → byte-identical failure-driven reuse. An unknown block-id / unmatched page token fails the rewrite phase LOUDLY (never a silent no-op). | 10 |
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
       textbook_structure; emits synthesized_objectives.json. Re-scales
       duration_weeks to the TO-driven max(8, num_tos) when --weeks is
       unset (WS5 §3.2 pacing — see docs/operations/pipeline-invocation.md
       § 2.1; skipped for --weeks / --reuse-objectives).

6. content_generation
   └── Generate course content modules (parallel batches of 10). Every
       emitted sourceId must resolve against the DART staging manifest
       (source_refs gate).

7. assessment_synthesis (optional)
   └── W10 — synthesize grounded quizzes, short assignments, and
       discussion prompts from the DART chunkset and emit QTI 1.2 /
       imsdt / assignment XML + manifest.json into <export>/06_assessments/
       (canonical IMS CC resource types). Validator-only phase routed by
       phase NAME to run_assessment_synthesis via _PHASE_TOOL_MAPPING;
       gated critical on qti_well_formed + assessment_objective_alignment.
       Runs before packaging; skipped via generate_assessments=false.

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
       Skipped when no `ANTHROPIC_API_KEY` or when `--skip-training`.
       Emits a per-pair resume sidecar at
       `training_specs/.synthesis_pairs_checkpoint.jsonl` (opt out via
       `--no-checkpoint`).

11. libv2_archival
   └── Archive course artifacts to LibV2 (raw PDFs, DART HTML, IMSCC,
       RAG corpus). Gated by libv2_manifest integrity checks.

12. vector_indexing (optional)
   └── Build the per-course on-device vector index from the LibV2-archived
       chunkset so a freshly-built course is askable. Routes via the
       `rag-indexer` agent (tool: `run_vector_indexing`). Runs by default;
       skips cleanly when the `[embedding]` extras are absent UNLESS
       `TRAINFORGE_REQUIRE_EMBEDDINGS` is set (then fails closed on a
       broken embedding backend).

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
- **Transient**: `rate_limit`, `connection_error`, `service_unavailable` → retryable
- **Permanent**: `validation_error`, `missing_input`, `permission_denied`, `schema_error` → no retry
- **`timeout` is no longer unconditionally transient-retry** (graceful-stop change): a **batch** timeout now writes the run-scoped stop sentinel and grace-drains to a checkpoint — it becomes a `paused` (not `TIMEOUT`), and only hard-cancels to `TIMEOUT` if the grace window also expires. A **task** timeout grace-drains the slow task, then keeps the existing `TIMEOUT` classification + transient-retry ladder (the resume sidecar makes the retry lossless). Detail: `docs/operations/pipeline-invocation.md` § 7.

### Retry Protocol

Failed tasks retry up to 3 times with exponential backoff:
1. First retry: After 5 seconds
2. Second retry: After 30 seconds
3. Third retry: After 120 seconds
4. After 3 failures: Log to error table, require manual intervention

### Graceful stop (checkpoint on command)

`ed4all stop <id>` / `--all` (and `SIGTERM`/`SIGINT` to a live `ed4all run`)
drops a filesystem stop sentinel that every long-running stage polls at its unit
boundaries. The in-flight unit finishes, checkpoints, and the phase pauses
(status `paused`, exit code **3**) — never `failed`, never retried, worst-case
loss one in-flight LLM call. Resume with a **plain** `ed4all run --resume <id>`
(never `--force` after a stop — `--force` clears the resume sidecars). The
operator-owned global `STOP_ALL` also **blocks new/resumed runs** until
`ed4all stop --clear-all`. Full semantics (per-phase worst-case-loss table,
timeout-to-pause grace windows, SemantiK chapter-seam granularity):
`docs/operations/pipeline-invocation.md` § 7.

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
| `EdgeConsensusAggregator` (GPT-fb-12-may item 2) | `<libv2_course>/graph/edge_consensus_report.json` (sibling of `concept_graph_semantic.json`); also stamps per-edge `edge_status` + `consensus_signals[]` on the graph via `apply_to_graph` and attenuates `kg_quality.consistency` by `(1 - contradiction_rate)`. | (helper, no separate file; surfaces in `schemas/knowledge/concept_graph_semantic.schema.json`) |
| `PromotionChainAggregator` (W3.G master) | `<libv2_course>/courseforge_promotion_chain_report.json` | `schemas/governance/promotion_chain.schema.json` |
| `BlockQualityRollupAggregator` (IB6.6) | `<libv2_course>/block_quality_rollup_report.json` (falls back to `<project_path>/...`) — block→module→course 8-dim quality rollup. Only runs when `ED4ALL_BLOCK_QUALITY_RUBRIC` is on. | `schemas/aggregators/block_quality_rollup.schema.json` |
| `ConceptCoverageAggregator` (W4.1) | `<libv2_course>/concept_coverage.json` (falls back to concept-graph dir) — concept-keyed coverage table. Only runs when `ED4ALL_CONCEPT_COVERAGE` is on (default OFF → no file). | `schemas/aggregators/concept_coverage.schema.json` |
| `IntelligenceLevelAggregator` (W4.6) | `<libv2_course>/intelligence_level_report.json` (falls back to `<trainforge_dir>/...`) — deterministic 0-5 capability rubric. Only runs when `ED4ALL_INTELLIGENCE_RUBRIC` is on (default OFF → no file). | `schemas/aggregators/intelligence_level.schema.json` |
| `AccessibilityConformanceAggregator` (roadmap T3) | `<libv2_course>/quality/accessibility_conformance.json` (falls back to `<trainforge_dir>/quality/...`) — inverts the gate WCAG issue stream into a per-success-criterion VPAT/WCAG-EM conformance table (`supports` / `partially_supports` / `does_not_support` / `not_evaluated`, with explicit `not_evaluated` rows for criteria outside automated static-HTML reach). | `schemas/aggregators/accessibility_conformance.schema.json` |
| `BuildCostAggregator` (roadmap OP2) | `<libv2_course>/build_cost_report.json` (falls back to `<trainforge_dir>/...`) — pure metering (no LLM): per-phase wall-clock (checkpoints), GPU residency (`vram_trajectory.jsonl`, section omitted when absent), and LLM calls/tokens (`llm_usage.jsonl` from the OP2 usage tap, section omitted when absent). | `schemas/aggregators/build_cost.schema.json` |
| `ProvenanceResolutionAggregator` | `<libv2_course>/quality/provenance_resolution_report.json` (falls back to `<project_path>/provenance_resolution_report.json`) | `schemas/aggregators/provenance_resolution.schema.json` |
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
| `course_generation` | 35 | 25 | 60 |
| `rag_training` | 4 | 3 | 7 |
| `textbook_to_course` | 64 | 68 | 132 |
| `trainforge_train` | 2 | 0 | 2 |
| **Total** | **105** | **96** | **201** |

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

Per-flag rows live in subsystem CLAUDE.md files (one owner per prefix); the root-owned cross-cutting rows live in [`docs/operations/behavior-flags.md`](docs/operations/behavior-flags.md) with a one-line index below. Counting convention for the **subsystem** rows: **distinct flags, multi-flag rows expanded** — a single table row that documents several env vars (e.g. SemantiK's `SEMANTIK_MODEL_DIR / _CACHE_DIR / _DATA_DIR / _CONFIG_DIR` row = 4 flags, Courseforge's `COURSEFORGE_SELF_VERIFY / _REFINE_ROUNDS / _CHUNK_SCOPED` row = 3 flags) counts once per flag it documents, so those tallies exceed the printed row count. The root-owned table is one row per flag, so its count equals its row count.

| Prefix | Owner | Flag count |
|--------|-------|-----------:|
| `TRAINFORGE_*` / `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` / `CURRICULUM_ALIGNMENT_*` / `WAVE18_*` | [`Trainforge/CLAUDE.md § Opt-In Behavior Flags`](Trainforge/CLAUDE.md) | 55 |
| `NVIDIA_*` (vendor endpoint-registry row for the hosted large-model seat — `NVIDIA_API_KEY` / `NVIDIA_BASE_URL` / `NVIDIA_LARGE_MODEL`) | [`Trainforge/CLAUDE.md § Opt-In Behavior Flags`](Trainforge/CLAUDE.md) | 3 |
| `SEMANTIK_*` (DART replacement — SemantiK semantic-cascade converter; also honors the legacy `DART_THETA_DEVICE` compat env) | [`SemantiK/CLAUDE.md § Opt-In Behavior Flags`](SemantiK/CLAUDE.md) | 86 |
| `COURSEFORGE_*` / `COURSEPLANNER_*` / `TEXTBOOK_SYNTHESIS_*` | [`Courseforge/CLAUDE.md § Opt-In Behavior Flags`](Courseforge/CLAUDE.md) | 36 |
| `DECISION_*` / `ED4ALL_*` / `LOCAL_DISPATCHER_*` / `MCP_ORCHESTRATOR_*` / `LLM_*` (cross-cutting) | root index (below) + [`docs/operations/behavior-flags.md`](docs/operations/behavior-flags.md) | 181 |

### Cross-cutting flags (root-owned)

**Full per-flag detail** (resolution chains, guardrails, calibration status, anti-fabrication contracts) lives in [`docs/operations/behavior-flags.md`](docs/operations/behavior-flags.md). The table below is a grep-able one-line index; every flag name stays searchable here.

| Flag | Default | One-line purpose |
|------|---------|------------------|
| `DECISION_VALIDATION_STRICT` | unset | Fails closed on unknown `decision_type` values in decision captures. |
| `ED4ALL_BLOCK_QUALITY_RUBRIC` | unset (off) | IB6 keystone — the 8-dim 0-3 block-quality scoring + rollup pass and its composing validators. |
| `ED4ALL_BLOCK_BODY_CHAR_CEILING` | `200` (global override) / per-type default | IB6.4 per-block D2 cognitive-load body ceiling |
| `ED4ALL_BLOCK_QUALITY_SHADOW` | unset (off) | W8.8 shadow-collect gate for the IB6 block-quality validators |
| `ED4ALL_COVERAGE_FLOOR` | `0.80` | W8.6 warning-day-1 source→objective coverage floor for the post-loop `PromotionChainAggregator` |
| `ED4ALL_COVERAGE_DROP_STRICT` | unset (off) | W8.6 opt-in stricter gating for the `COVERAGE_DROP` signal |
| `ED4ALL_KG_REAL_FLOORS` | unset (off) | Recompute REAL completeness+accuracy KG-quality scores before the per-dimension floor check. |
| `MCP_ORCHESTRATOR_LLM_MODEL` | `claude-opus-4-7` | Pins the Anthropic model ID for the MCP orchestrator LLM backend; per-run `LLM_MODEL` wins. |
| `LOCAL_DISPATCHER_ALLOW_STUB` | unset | Permits `LocalDispatcher` to emit a stubbed `PhaseOutput` when no `agent_tool` is wired; tests/dry-run only. |
| `ED4ALL_CLOUD_RATE_LIMIT` | unset (off) | Hosted-large build profile SETUP — master switch for the shared cloud-seat admission gate |
| `ED4ALL_AGENT_DISPATCH` | unset | Routes subagent-classified agents through `dispatcher.dispatch_task` instead of in-process tool registry. |
| `ED4ALL_AGENT_TIMEOUT_SECONDS` | `1800` | Per-task subagent dispatch mailbox timeout. |
| `ED4ALL_ALIGNMENT_VERB_TRIPLE` | unset (off) | IB3 constructive-alignment keystone |
| `ED4ALL_ANSWER_PROVIDER` | `local` | Selects the grounded-answer backend (W-D12 registry); loopback-only (non-loopback base_url raises). |
| `ED4ALL_ANSWER_MODEL` | per-provider | Model ID override for the answer backend |
| `ED4ALL_ANSWER_TIMEOUT_SECONDS` | `120` | Answer-client HTTP timeout |
| `ED4ALL_ANSWER_NUM_CTX` | `4096` | Serving-window token budget for the grounded-answer prompt |
| `ED4ALL_ANSWER_CITATION_PRUNE` | `shadow` | Three-valued governor of the claim-attribution citation **prune + add** pass at answer-composition time |
| `ED4ALL_ANSWER_ASSESSMENT_GUARD` | unset (off) | L2 three-valued (off/shadow/on) assessment-aware answering guard — matches a learner question to a course assessment stem and redirects-with-hint (never refuses) instead of doing the homework |
| `ED4ALL_ANSWER_ASSESSMENT_GUARD_THRESHOLD` | `0.75` | Float match floor (lexical containment / cosine) for the L2 assessment guard |
| `ED4ALL_ANSWER_PRUNE_MIN_OVERLAP` | `0.25` | Float support threshold for the PRUNE decision |
| `ED4ALL_ANSWER_ADD_MIN_SHINGLE` | `0.50` | Float shingle floor for the ADD decision |
| `ED4ALL_ANSWER_NLI_ADD` | `off` | Three-valued governor of the **NLI-based citation-ADD** arm |
| `ED4ALL_ANSWER_COMPLETENESS_RECHECK` | `on` | Governs the post-generation **completeness recheck** |
| `ED4ALL_ANSWER_LIBRARY_WIDE` | unset (off) | W4 library-wide grounded ask |
| `ED4ALL_ANSWER_INTENT_ROUTE` | unset (off) | W5.1 pre-retrieval intent-route bias on the grounded-answer path |
| `ED4ALL_ANSWER_MULTITURN` | unset (off) | W5.2 multi-turn antecedent query rewrite |
| `ED4ALL_ANSWER_DECOMPOSE` | unset (off) | W5.6 multi-part question decomposition |
| `ED4ALL_ANSWER_HYDE` | unset (off) | W5.7 hypothetical-document-embedding retrieval arm |
| `ED4ALL_ANSWER_GRAPH_EXPAND` | unset (off) | W5.3 concept-graph passage expansion on the grounded-answer path |
| `ED4ALL_ANSWER_GRAPH_EXPAND_MAX` | `4` | W5.3 per-answer cap on graph-reachable neighbors appended by `ED4ALL_ANSWER_GRAPH_EXPAND` |
| `ED4ALL_ANSWER_COMPLETENESS_RERETRIEVE` | unset (off) | W5.4 step-4b completeness recheck re-retrieves per uncovered sub-question before re-asking (verdict-safe). |
| `ED4ALL_ANSWER_HEDGE_TIER` | unset (off) | W5.5 confidence-graded HEDGE tier on the grounded-answer path |
| `ED4ALL_ANSWER_HEDGE_MARGIN` | `0.15` | W5.5 hedge band width for `ED4ALL_ANSWER_HEDGE_TIER` |
| `ED4ALL_ARCHIVE_REQUIRE_FULL_COURSE` | unset (off) | "True full course" archival-completeness strict-mode gate |
| `ED4ALL_RERANK_PROVIDER` | unset (off) | Cross-encoder reranker over the first-stage retrieval candidate pool on the grounded-answer path |
| `ED4ALL_BLOCK_ANATOMY` | unset (off) | IB1 six-slot anatomy contract emit gate |
| `ED4ALL_BLOCK_A11Y` | unset (off) | IB4 per-block WCAG 2.2 AA + UDL emit gate |
| `ED4ALL_CALLOUT_TYPED` | unset (off) | FR-A11Y-03 typed B12 callout emit + gate flag |
| `ED4ALL_COS_PER_WEEK_CAP` | `0` (auto) | WS5 §2.2 per-week chapter-objective placement cap for the single-sourced ceil-stride slicer. |
| `ED4ALL_WEEK_TO_GROUPS` | unset (off) | WS5 week-grouping override: when on AND `duration_weeks == num_tos`, per-week `"Week N"` groups are built by TO membership (week N = TO-N's `child_co_ids`) instead of the ceil-stride CO slice; else warns + falls back to ceil-stride. |
| `ED4ALL_CONCEPT_COVERAGE` | unset (off) | W4.1 read-only capability aggregator |
| `ED4ALL_CONCEPT_EXTRACTION_CHECKPOINT` | `on` | Site override for the concept_extraction Stage-3 per-window (`synthesize_concepts`) resume sidecar (beats `ED4ALL_GENERATION_CHECKPOINT`). |
| `ED4ALL_INTELLIGENCE_RUBRIC` | unset (off) | W4.6 read-only capability aggregator |
| `ED4ALL_CONTENT_PAGE_PER_CO` | unset (off) | Page-per-CO content-emit gate |
| `ED4ALL_CONTENT_PAGE_NUM_CTX` | `4096` (→ `ED4ALL_ANSWER_NUM_CTX` → 4096) | Authoring serving-window token budget for the page-per-CO per-page chunk cap |
| `ED4ALL_CONTENT_PAGE_MAX_CHUNKS` | `5` | Hard top-K ceiling on chunks kept per CO page for the page-per-CO cap |
| `ED4ALL_CHUNK_ROLE_DIVERSIFY` | unset (off) | Gap #11 — deterministic per-block-role rotation of a page's ranked chunk order so co-located blocks don't all lead with the same anchor example (chunk-universe remap; default off → byte-identical). |
| `ED4ALL_COURSE_IDENTITY_DEDUP` | unset (off) | W0.5 course-identity SPLIT-BRAIN guard |
| `ED4ALL_EMBEDDING_PROVIDER` | `st` | Selects the retrieval-index embedding backend (`st` / `local-openai` / `fake`). |
| `ED4ALL_EMBEDDING_MODEL` | per-provider | Model ID override for the embedding provider |
| `ED4ALL_EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | Base URL of the local OpenAI-compatible `/v1/embeddings` server (`local-openai` only). |
| `ED4ALL_EMBEDDING_API_KEY` | `local` | Optional bearer token for the local embedding server (`local-openai` only). |
| `ED4ALL_EMBEDDING_DEVICE` | `cpu` | Torch device for the in-process `st` provider |
| `ED4ALL_EMBEDDING_BATCH_SIZE` | `16` | Encode batch size for the embedding client (recorded in the index manifest). |
| `ED4ALL_EMBEDDING_ALLOW_FAKE` | unset | Anti-poisoning gate. Permits a `fake`-provider vector index to load in a production read path. |
| `ED4ALL_EVAL_CROSS_COURSE_NEGATIVES` | unset (off) | W4 eval arm mining REAL chunks from OTHER LibV2 courses as out-of-domain refusal probes. |
| `ED4ALL_GATE_ADVISORY` | unset | **Safety-critical.** Flips post-training eval gates from blocking to advisory |
| `ED4ALL_GENERATION_CHECKPOINT` | `on` | Family flag for the fingerprinted LLM unit-checkpoint resume sidecars; site flags override it. |
| `ED4ALL_GENERATION_TECHNIQUE` | `C5` | W5 C0..C5 generation-technique selector (naive → best-of-N + NLI verifier). |
| `ED4ALL_DYNAMIC_BLOCK_PLAN` | unset (off) | Wave-2 keystone content-aware large-model block planner gate (two-pass outline phase). |
| `ED4ALL_DYNAMIC_BLOCK_PLAN_MODEL` | per-provider | Model-ID override for the `ED4ALL_DYNAMIC_BLOCK_PLAN` planner |
| `ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER` | `nvidia` (endpoint-registry key) | IB7.2 planner-SEAT selector (hosted large seat by default) |
| `ED4ALL_PLANNER_BLOOM_CLIMB` | unset (off) | IB7.3 programmatic Bloom-climb re-sort |
| `ED4ALL_PLANNER_LIFECYCLE` | unset (off) | IB7.4 lifecycle open/close guarantee + slot-edit escalation |
| `ED4ALL_PLANNER_SPACING` | unset (off) | IB7.5a within-module temporal-spacing pass |
| `ED4ALL_PLANNER_BLOOM_CEILING` | unset (off) | IB7.6b per-type Bloom-range ceiling re-route |
| `ED4ALL_PLANNER_FADING` | unset (off) | FR-INT-01 B08 guided-practice **fading-sequence** planner pass |
| `ED4ALL_WORKED_EXAMPLE_FLOOR` | unset (off) | P4 worked-example DENSITY floor: ≥1 example/problem block per procedural (apply/create) CO. |
| `ED4ALL_BLOOM_SPREAD_FLOOR` | unset (off) | P4 Bloom-spread floor: ≥1 analyze-or-higher block per week (widens the apply-heavy Bloom mix). |
| `ED4ALL_TRIANGLE_FLOOR` | unset (off) | GAP D (IB3 alignment triangle) per-CO activity + assessment floor for the block planner. |
| `ED4ALL_RETRIEVAL_INTERLEAVE` | unset (off) | GAP C (IB7.5b interleaved retrieval) per-content-page retrieval-block floor. |
| `ED4ALL_HOME` | unset (repo-relative) | Relocatable data root — sets every mutable data dir under it instead of repo-relative. |
| `ED4ALL_KEY_TERMS_PAGE` | unset (off) | Feature I5 — per-terminal-objective deterministic **"Key Terms" page** gate |
| `ED4ALL_NEW_BLOCK_TYPES` | unset (off) | IB5 gate for four framework block types: hook, multimedia, worked_example, diagram. |
| `ED4ALL_REFLECTION_CALIBRATION` | unset (off) | FR-INT-03 gate for the B11 reflection predict-then-reveal calibration contract. |
| `ED4ALL_REASONING_THINKING_OFF` | unset (off) | Injects the Nemotron "detailed thinking off" system directive + `chat_template_kwargs.enable_thinking=false` on every composed OpenAI-compatible call so reasoning-token output doesn't trip the finish_reason=length truncation guard. |
| `ED4ALL_RECALL_SELF_CHECK` | unset (off) | Free-recall / cloze self-check variant gate |
| `ED4ALL_MISCONCEPTION_RICH` | unset (off) | Named subject-specific misconception + productive-failure gate for the B03/B12 `misconception` block |
| `ED4ALL_MAYER_CTML` | unset (off) | Mayer CTML 12-principles structural check enriching the UDL/multimedia surface |
| `ED4ALL_BLOOM_DISTRIBUTION` | unset (off) | Course-level Bloom-distribution-vs-target-curve gate |
| `ED4ALL_BLOOM_DISTRIBUTION_TARGET` | unset (canonical default) | Operator override for the target Bloom curve consumed by `BloomDistributionValidator` |
| `ED4ALL_BLOOM_DISTRIBUTION_TOLERANCE` | `0.20` | Float L1-deviation tolerance for the `BLOOM_DISTRIBUTION_OFF_TARGET` decision |
| `ED4ALL_BLOOM_DISTRIBUTION_MIN_LOS` | `6` | Small-N objective floor for `BloomDistributionValidator` |
| `ED4ALL_PREREQ_SEQUENCING` | unset (off) | Prerequisite-DAG-driven content sequencing |
| `ED4ALL_PREREQ_TRANSITIVE_REDUCTION` | unset (off) | W3.3 deterministic stdlib DFS transitive reduction of the projected TO→TO prereq graph |
| `ED4ALL_PREREQ_CENTRALITY_TIEBREAK` | unset (off) | W3.4 concept-centrality stable tie-break for the TO topological sort |
| `ED4ALL_PREREQ_CENTRALITY_METHOD` | `in_degree` | W3.4 satellite selecting the centrality method consumed by `ED4ALL_PREREQ_CENTRALITY_TIEBREAK` |
| `ED4ALL_KG_PREREQ_HEALTH` | unset (off) | W3.2/W3.4/W3.6 prereq-DAG health signals on the `kg_quality_report` validator |
| `ED4ALL_RICHER_VISUAL_SYSTEM` | unset (off) | Richer-visual-system Phase 0 gate |
| `ED4ALL_LIBV2_ROOT` | `<repo>/LibV2/` | Absolute path to the LibV2 root directory |
| `ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS` | `60` at the client; `300` at the content-generation providers | Per-request HTTP timeout (s) for local content-generation LLM calls (7B prose authoring). |
| `ED4ALL_MAILBOX_BASE_DIR` | `<repo>/state/mailbox/` | Orchestrator task-mailbox base directory. |
| `ED4ALL_NLI_DEVICE` | `cpu` (code) / `cuda` (project default) | Torch device for the in-process NLI classifier that scores groundedness/eval entailment |
| `ED4ALL_NLI_MIN_FREE_VRAM_MIB` | `1024` | Free-VRAM floor gating the in-process NLI model onto CUDA |
| `ED4ALL_NLI_EVICT_FOR_CUDA` | `true` (on) | VRAM-contention strategy: evict the resident ollama model to free the card for NLI on CUDA. The `ED4ALL_GPU_LIFECYCLE` phase-boundary sweep makes this largely moot (it hands the card over before NLI loads); kept as the in-phase fallback (generation + NLI in one phase, or lifecycle opted off). |
| `ED4ALL_OBJECTIVE_REVIEW_PROVIDER` | unset (off) | Grounding-safe **objective-review** pass gate |
| `ED4ALL_OBJECTIVE_REVIEW_MODEL` | per-provider | Model-ID override for the objective-review pass |
| `ED4ALL_OBJECTIVE_CHUNK_RELEVANCE_FLOOR` | `0.30` | Fix 1A relevance floor for the objective-dedup union prune |
| `ED4ALL_OBJECTIVE_CITATION_RESELECT` | unset (off) | Post-hoc CO citation re-selection: re-cite the best in-window supporter by cosine. |
| `ED4ALL_OBJECTIVE_RESELECT_EXERCISE_DEMOTE` | on (when reselect on) | Demotes exercise/answer-list chunks below instructional prose in the citation re-selection rank (opt-out). |
| `ED4ALL_OBJECTIVE_RESELECT_KEEP_ORIGINAL` | on (when reselect on) | Keep-original union guard for citation re-selection: never STRIP a synthesis citation — unions every above-floor, non-exercise original into the kept set so the cosine top-K can only ADD supporters (fixes an entailment-gate regression; opt-out). |
| `ED4ALL_OBJECTIVE_ENTAILMENT_MATH_FOLD` | unset (off) | Opt-in LaTeX/unicode-math folding of premise + hypothesis before NLI in the `objective_entailment` gate (measured net-neutral on a math-scan corpus — see behavior-flags.md; deferred-flip candidate). |
| `ED4ALL_OBJECTIVE_DEDUP_THRESHOLD` | `0.88` | W2 §4.2 cosine clustering threshold for the in-synthesis objective-dedup pass |
| `ED4ALL_OBJECTIVE_DISTINCT_SKILL_SPLIT` | unset (off) | I3 PRONG A — distinct-skill SPLIT gate for the objective-dedup pass |
| `ED4ALL_OBJECTIVE_DEDUP_LEXICAL` | unset (off) | W2 Defect E — cross-window lexical-dedup SECOND PASS (complete-linkage merge of near-restatement clusters after single-link cosine, before the PRONG-A split). Default off → byte-identical DedupResult. |
| `ED4ALL_OBJECTIVE_DEDUP_LEXICAL_COSINE` | `0.78` | W2 Defect E satellite — centroid-cosine floor for a lexical merge edge (below the 0.88 single-link dedup threshold). |
| `ED4ALL_OBJECTIVE_DEDUP_LEXICAL_JACCARD` | `0.60` | W2 Defect E satellite — best-grounded skill-signature Jaccard floor for a lexical merge edge (above PRONG-A's <0.34 distinctness band). |
| `ED4ALL_OBJECTIVE_SPECIFICITY` | unset (off) | W2 Defect B — opt-in gate for the CO-statement specificity/vacuity validator (`objective_specificity` at course_planning; V1 content-residual vacuity + V2 vague-object + V3 source-token recall). Default off → byte-identical skip-with-pass. |
| `ED4ALL_OBJECTIVE_SEED_SANITIZE` | unset (off) | W4 Defect C — exercise-apparatus seed sanitation (`lib/objectives/chunk_window.py::resolve_seed_sanitize`): strips apparatus lines/sentences from the RENDERED Pass-B window body + drops Pass-C survivors whose STATEMENT matches an apparatus marker (chunk_ids / citability untouched). Default off → byte-identical windows. **Operator note:** flipping this mid-course changes the window-render fingerprint, so window resume sidecars invalidate and those windows re-run on `--resume`. |
| `ED4ALL_OBJECTIVE_SOURCE_BACKFILL` | unset (off) | I3 PRONG B — source-richness BACKFILL gate |
| `ED4ALL_OBJECTIVE_BACKFILL_COVERAGE_TARGET` | `1.0` | I3 PRONG B coverage target: min fraction of content-bearing chunks the backfill drives toward. |
| `ED4ALL_OBJECTIVE_BLOOM_RELEVEL` | unset (off) | Feature 1 — deterministic Bloom-level relevel (re-derive a mislabelled CO/TO `bloom_level` from its main verb's canonical level; statements never change). |
| `ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT` | unset (off) | Feature 2 / PRONG C — LLM-assisted grounded analyze/evaluate complement synthesis when the higher-order Bloom share is too thin. |
| `ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MIN_SHARE` | `0.15` | PRONG C satellite — analyze+evaluate+create share floor below which complements are synthesized. |
| `ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MAX` | `8` | PRONG C satellite — hard cap on complement COs added per pass (0 = measure-only). |
| `ED4ALL_OBJECTIVE_LIBRARY_EXEMPLARS` | unset (off) | W4.3 cross-course objective-library EXEMPLARS master gate |
| `ED4ALL_OBJECTIVE_LIBRARY_EXEMPLAR_LIMIT` | `8` | W4.3 top-K cap on surfaced exemplars |
| `ED4ALL_OBJECTIVE_LIBRARY_EXEMPLAR_MIN_OVERLAP` | `0.05` | W4.3 Jaccard floor for an exemplar to be surfaced |
| `ED4ALL_OBJECTIVE_MAX_CHUNKS_PER_OBJECTIVE` | `5` | Fix 1A top-K cap on cited chunks per MERGED objective |
| `ED4ALL_OBJECTIVE_SYNTHESIS_CHECKPOINT` | `on` | Site override for the stage-2 window + cluster resume sidecars (beats `ED4ALL_GENERATION_CHECKPOINT`). |
| `ED4ALL_REQUIRE_ARCHIVED_OBJECTIVES` | unset (off) | W2.3 fail-closed for the archive_to_libv2 objectives→objectives.json plumbing. |
| `ED4ALL_PRODUCTION` | `0` | When `1`, enables production-mode FastMCP server settings. |
| `ED4ALL_RESEGMENT_COLLAPSED` | `1` | WS6b collapse re-segmentation gate |
| `ED4ALL_RESEGMENT_SECTIONS_PER_CHAPTER` | `13` | WS6b target sections-per-pseudo-chapter |
| `ED4ALL_ROOT` | auto-detect | Absolute path to the Ed4All project root. |
| `ED4ALL_RUN_ID` | generated | Per-run identifier consumed by every artifact emitter. |
| `ED4ALL_SKIP_ABLATION` | unset | When set, skips the post-training ablation pass. |
| `ED4ALL_STAGE_MODE` | `symlink` | How `stage_dart_outputs` materialises DART HTML (copy / symlink / hardlink). |
| `ED4ALL_STATE_RUNS_DIR` | `<repo>/state/runs/` | State-runs directory |
| `ED4ALL_STRUCTURE_EXTRACT_GUARDS` | unset (off) | SemantiK structure-fidelity Package 1+3 — DPUB-ARIA article-path continuation-merge / headingless-wrapper grouping / noncontent+numbered-apparatus heading filter / structureDiagnostics sanity on the extractor (byte-identical off). |
| `ED4ALL_STRUCTURE_OUTLINE_ANCHOR` | on-when-guards-on | SemantiK extractor outline-anchored section alignment (`lib/semantic_structure_extractor/semantic_structure_extractor.py`): aligns built sections to the chapter-outline / ToC `N.M` declarations with an ordinal-union harvest so scan-split sections re-fuse to their declared section. Opt-out: inert unless `ED4ALL_STRUCTURE_EXTRACT_GUARDS` is on (only the guarded path reaches it), then default ON → set falsey for byte-identical Package-1 chapters. |
| `ED4ALL_TO_BACKLINK_FLOOR` | `0.45` cosine / `0.10` token | WS2 dual weak-link floor for the deterministic CO→TO backlink |
| `ED4ALL_TO_BACKLINK_REASSIGN` | unset (off) | M5 Fix A anti-junk-drawer reassignment + validator-parity scoring for the CO→TO backlink |
| `ED4ALL_TO_CLUSTER_K` | `0` (auto) | WS1.1 FIXED target-K for **bottom-up TO derivation** Ward agglomerative clustering |
| `ED4ALL_TO_CLUSTER_THRESHOLD` | `0.50` | WS1 cosine threshold — now governs ONLY the no-sklearn single-link TO-clustering fallback. |
| `ED4ALL_TO_COS_PER_CLUSTER` | `6` | WS1.1 AUTO-K divisor for bottom-up TO derivation |
| `ED4ALL_TO_CLUSTER_GUARDS` | unset (off) | P1 master gate for the post-cluster CONSOLIDATE pass on **bottom-up TO derivation** |
| `ED4ALL_TO_OUTLIER_MIN_SIZE` | `3` | P1 min-cluster-size floor for the CONSOLIDATE pass |
| `ED4ALL_TO_OUTLIER_ABSORB_FLOOR` | `0.20` | P1 "has a clear home" centroid-cosine floor for the OUTLIER-absorb decision |
| `ED4ALL_TO_MERGE_NEAR_DUP` | unset (off) | P1 gate for the SECOND post-cluster guard — near-duplicate-TO merge |
| `ED4ALL_TO_MERGE_COSINE` | `0.85` | P1 centroid-cosine merge floor for the near-duplicate-TO merge |
| `ED4ALL_TO_ALLOW_SINGLETON_TO` | unset (off → singleton TOs dissolved) | Anti-hallucinated-TO backstop — the OPT-OUT for the unconditional `dissolve_singletons` pass |
| `ED4ALL_TO_MIN_CLUSTERS` | `3` | Tiny-course floor for the `dissolve_singletons` backstop |
| `ED4ALL_TO_SOURCE_GROUNDING` | unset (off) | W7.5 opt-in gate for the TERMINAL-objective source-grounding validator |
| `ED4ALL_TO_CHAPTER_ANCHOR` | unset (off) | W3 Defect A master gate — chapter-anchored TO derivation (one DART module → one terminal objective by cited-chunk plurality) instead of bottom-up statement clustering. Default off → bottom-up path unchanged. |
| `ED4ALL_TO_CHAPTER_ANCHOR_REORDER` | on-when-master-on | W3 Defect A §6 satellite — stable book-order CO re-sort (by module order + in-module position) BEFORE the week slice, so ceil-stride weeks are chapter-contiguous even without `ED4ALL_WEEK_TO_GROUPS`. Only the falsey tokens disable it. |
| `ED4ALL_TO_CHAPTER_MIN_MODULES` | `2` | W3 Defect A satellite — module-count floor below which anchor mode degrades to bottom-up (a monolithic single-HTML corpus can't be anchored). |
| `ED4ALL_TO_CHAPTER_MIN_CO_COVERAGE` | `0.80` | W3 Defect A satellite — min fraction of COs that must resolve ≥1 module from their own cited chunks for anchor mode to fire (else degrade to bottom-up). |
| `ED4ALL_TRAINING_CAPTURES_DIR` | `<repo>/training-captures/` | Overrides the legacy decision-capture mirror root (`training-captures/`). |
| `ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM` | unset (off) | W2.1 in-gate CUDA OOM surfaces as a `VALIDATOR_OOM` warning; truthy ⇒ the OOM fails the gate closed. |
| `ED4ALL_CALIB_EXTRA_CORPORA` | unset (off) | W8.3 multi-corpus discovery pointer for the calibration harness |
| `ED4ALL_VRAM_DOCTOR` | unset (off) | VRAM-contention observability gate |
| `ED4ALL_GPU_LIFECYCLE` | `on` (default ON — deviation) | Deterministic GPU-lifecycle LEASE: every model loads, runs, releases the card, hands it to the next stage. Phase-boundary sweep in `workflow_runner.run_workflow` (ollama `keep_alive:0` + torch `empty_cache`, best-effort, off-loop) + SemantiK cascade stage seams (`cascade._gpu_lifecycle_release` via the cross-venv twin). Residency/timing only — never an output byte (default-ON justified). Opt-out `=0` for perf. |
| `ED4ALL_BIG_MEMORY_MIN_MIB` | `49152` (48 GiB) | Total-VRAM threshold above which the `ed4all doctor` `gpu_profile` group treats the box as a big-memory concurrent-serving host and emits ADVISORY warns for each small-box default still on (`ED4ALL_GPU_LIFECYCLE`, `ED4ALL_NLI_EVICT_FOR_CUDA`, `ED4ALL_CLOUD_RATE_LIMIT`, low `ED4ALL_GPU_MAX_USED_MB`); below threshold / unprobeable GPU → silent no-op. |
| `ED4ALL_PLANNER_INTERLEAVE` | unset (off) | W6.2 TRUE cross-CO practice interleaving for the dynamic block planner |
| `ED4ALL_PLANNER_FAR_TRANSFER` | unset (off) | W6.3 per-TO novel-context far-transfer floor for the dynamic block planner |
| `ED4ALL_PLANNER_DUAL_CODING` | unset (off) | W6.5 per-CO dual-coding floor for the dynamic block planner |
| `ED4ALL_PLANNER_INTEGRATION` | unset (off) | Integration floor — turns a weak-membership ("oddball") CO into cross-CO integrated practice for the dynamic block planner |
| `ED4ALL_PLANNER_INTEGRATION_MAX_OVERLAP` | `0.10` | Integration-floor satellite — token-overlap floor below which a CO is weak-membership (aligned with the CO→TO backlink token floor) |
| `ED4ALL_PLANNER_INTEGRATION_MAX_PER_WEEK` | `2` | Integration-floor satellite — hard cap on integrated-practice injections per week (weakest-first) |
| `ED4ALL_PLANNER_CROSS_WEEK_RETRIEVAL` | unset (off) | W6.1 cross-week cumulative retrieval for the dynamic block planner (needs prior-week context). |
| `ED4ALL_FAQ_PAGE` | unset (off) | W6.4 deterministic per-week "FAQ" page gate |
| `ED4ALL_FAQ_MAX_PER_PAGE` | `12` | W6.4 satellite — hard cap on FAQ cards emitted per week page |
| `ED4ALL_OBJECTIVE_OBSERVABLE_VERB` | unset (off) | W1.1 non-observable (fuzzy) main-verb scan on the objective statement (legacy no-ABCD path). |
| `ED4ALL_OBJECTIVE_INFER_BLOOM` | unset (off) | W1.4 infer a null LO bloom_level from its declared ABCD behavior.verb instead of skipping. |
| `ED4ALL_CHUNK_COVERAGE_FLOOR` | unset (off) | W1.2 import-coverage gate floor on the existing `chunkset_manifest` gate |
| `ED4ALL_MIN_CHUNKS` | unset (off) | W1.2 thin-chunkset floor on the `chunkset_manifest` gate (CHUNKSET_TOO_THIN warning). |
| `ED4ALL_KEYTERM_DEF_QUALITY` | unset (off) | W1.5 glossary definition-quality gate (circular / too-long / not-distinct / missing-math-condition) — critical gate (flip-wave-2) emitting warning-severity issues; audits key-terms vocab cards AND inline `<div class="definition-box">` blocks parsed out of concept/explanation HTML. |
| `ED4ALL_PAGE_EST_MINUTES` | unset (off) | W1.6 per-page estimated learning-time emit gate |
| `ED4ALL_PAGE_WPM` | `200` | W1.6 satellite — reading-speed divisor for the `ED4ALL_PAGE_EST_MINUTES` estimate |
| `ED4ALL_PAGE_INTERACTION_MINUTES` | `1.0` | W1.6 satellite — per-interaction minute cost for the `ED4ALL_PAGE_EST_MINUTES` estimate |
| `ED4ALL_GROUNDEDNESS_COMPUTATIONAL` | unset (off) | W1.8 numeric-grounding of NLI-exempt computational sentences on the grounded-answer path |
| `ED4ALL_EMBED_OVERFLOW_GUARD` | unset (off) | W1b.2 embedding serving-window overflow guard |
| `ED4ALL_EMBED_OVERFLOW_SPLIT` | unset (off) | W1b.2 satellite of `ED4ALL_EMBED_OVERFLOW_GUARD` |
| `ED4ALL_EMBED_MAX_SEQ_TOKENS` | `512` | W1b.2 serving-window token ceiling driving the embed model max_seq_length pin + overflow. |
| `ED4ALL_CHUNK_CODE_SPLIT` | unset (off) | W1b.3 code-fence-aware chunk splitting |
| `ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR` | `0` (off) | W1b.4 runt-fragment merge floor |
| `ED4ALL_CHUNK_SECTION_HARD_BREAK` | unset (off) | Forces a chunk break at textbook SECTION-heading boundaries (anti cross-section fusion) |
| `ED4ALL_CHUNK_LO_HEURISTIC` | unset (off) | W1b.5 heuristic LO-backfill arm |
| `ED4ALL_CROSS_COURSE_DEDUP` | unset (off) | W1b.1 cross-course boilerplate dedup for multi-course batch imports |

The `LLM_*` env vars (`LLM_MODE`, `LLM_PROVIDER`, `LLM_MODEL`) are CLI runtime knobs documented in § Quick Start above. Other `ED4ALL_*` vars kept out of this index — the GUI server vars (`ED4ALL_GUI_HOST` / `_PORT` / `_LEARNER` / `_MODE` / `_TOKEN`), the test-only fixture/gating overrides, and the three rewrite-tier `ED4ALL_REWRITE_*` + the W10 `ED4ALL_ASSESSMENT_PROSE_PROVIDER` flags owned by subsystem files — are enumerated verbatim in [`docs/operations/behavior-flags.md`](docs/operations/behavior-flags.md).

---

## Licensing & ToS Posture

Canonical reference: **`docs/LICENSING.md`**. Read it before running any training-data synthesis pass.

The project distinguishes two surfaces with different licensing exposure:

- **Development tools** (Claude Code, OpenAI Codex) generate code, prose, and shell invocations. Their ToS restricts training-data routing, but on this project that restriction is moot — these tools never produce training data, so the dev tool you use has zero effect on the trained SLM's licensing.
- **LLM providers** (`--provider claude_session` / `together` / `local`) generate the paraphrased instruction / preference pairs that become training data. The trained model is a derivative work of those outputs, so the provider's ToS + the underlying model license decide whether the corpus is shippable.

Default posture: training-data synthesis routes to license-clean providers — `--provider local` with an Apache 2.0 model (Qwen2.5-7B/14B/32B) for an air-gapped clean corpus, or `--provider together` with a hosted OSS model as the cloud fallback. The `--provider anthropic` SDK training-pair path was **removed (Phase 4)** — `run_synthesis` fails closed on it unconditionally, so training-pair synthesis is license-clean by construction. The `claude_session` route stays wired behind the `TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS` acknowledgment gate but is not recommended for training data.

**Maintenance contract:** any new behavior flag (a `docs/operations/behavior-flags.md` row for the root-owned cross-cutting prefixes, or a subsystem CLAUDE.md row) that selects an LLM provider, model ID, or synthesis backend MUST land with a corresponding row in `docs/LICENSING.md`'s "Synthesis providers" table. Drift between those per-flag rows and `docs/LICENSING.md` is a documentation bug.

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
