# Ed4All Hybrid Orchestrator

Unified orchestration system for SemantiK, Courseforge, Trainforge, and LibV2.

## Quick Start

### Canonical entry point

> **Per-stage invocation** (stop-after / reuse / stage subcommands), **the timeout knobs
> that actually fire** (`ED4ALL_TASK_TIMEOUT_MINUTES` for slow in-process synthesis, not
> the batch/mailbox ones), the **outline-vs-rewrite naming trap**, and the **pure-local
> constrained-VRAM env recipe**: see [`docs/operations/pipeline-invocation.md`](docs/operations/pipeline-invocation.md).

> **Full production run** (the end-to-end multi-hour build: seat topology and
> ordering, per-phase artifacts, gate-failure triage, resume/stop procedure,
> what to check between phases): see
> [`docs/operations/full-run-playbook.md`](docs/operations/full-run-playbook.md).
> Its environment template is [`docs/operations/run-env.example.sh`](docs/operations/run-env.example.sh) —
> sanitized, sectioned into PORTABLE / HARDWARE-PROFILE / VALIDATED
> MEASUREMENTS. The hardware-profile values (concurrency, NLI batch sizes,
> `ED4ALL_GPU_LIFECYCLE=0`, `ED4ALL_NLI_EVICT_FOR_CUDA=0`) are tuned for a
> single-GPU large-unified-memory host and will OOM a small card unedited.
> Canonical full-run invocation:
>
> ```bash
> # _SEAT_MAIN_MODEL must match your seat's --served-model-name; the template
> # warns loudly at source time if it is unset. Edit §2 for your hardware first.
> _SEAT_MAIN_MODEL=<served-model-name> source ./docs/operations/run-env.example.sh
> ed4all run textbook-to-course --corpus ./inputs/<DIR> --course-name <NAME> \
>   --skip-training
> ```
>
> Pipeline-owned vLLM seat lifecycle is configured separately — source
> [`docs/operations/seat-schedule.env.example`](docs/operations/seat-schedule.env.example)
> alongside the run env (never enable it mid-build).

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

# --with-training: OPT IN to the in-build training tail of
# textbook_to_course (training -> post_training_validation -> evaluation,
# between vector_indexing and finalization). OFF by default — a training
# run is multi-hour and owns the whole card, so it never attaches to a
# build implicitly. --skip-training WINS when both are passed. Also valid
# on --resume (patches the persisted params before the resumed phases
# run). Training an ALREADY-archived course without rebuilding stays
# `ed4all run trainforge_train --course-name <slug>`.
ed4all run textbook-to-course --corpus pdfs/ --course-name PHYS_101 \
  --with-training

# Hosted large-model build profile (--provider nvidia = the vendor
# endpoint-registry key; SETUP — nothing dispatches to the cloud seat by
# default; gated on a later RUN discussion). Full routing detail (YAML
# redirect, seat pins, licensing caveat): see
# docs/operations/pipeline-invocation.md § 3.1. Run --dry-run first.
export COURSEFORGE_TWO_PASS=true
ed4all run textbook-to-course --provider nvidia --course-name PHYS_101 \
  --corpus slice.pdf --skip-conversion --skip-training \
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

# ed4all harvest-bloom-labels — deterministic (no-LLM) harvester: walks a
# Courseforge project/export (+ optional --course-path LibV2 dir) and collects
# every artifact-asserted Bloom label (objectives / blocks / assessment items)
# into state/bloom_labels/labels.jsonl — the corpus behind the re-founded
# bloom_classifier_disagreement voter 1 (ED4ALL_BLOOM_TRIVOTE). --dry-run counts
# only. Also runs post-build under ED4ALL_HARVEST_BLOOM_LABELS.
ed4all harvest-bloom-labels ./Courseforge/exports/PROJ-... --dry-run

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
SemantiK → Courseforge → Trainforge → LibV2) lives under the top-level
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

**SemantiK tools** — see `SemantiK/CLAUDE.md` (PDF→accessible-HTML conversion; emits the Source-Provenance `data-semantik-*` / `semantik:{slug}#{block_id}` contract).

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

### Assistant Tools

External MCP surface for the operator-assistant's BOUNDED campaign harness
(`MCP/tools/assistant_tools.py`). Thin, non-widening wrappers that delegate to
`lib/assistant/campaign_tools` (every mutating tool routes through the validated
`dispatch_campaign_tool` choke point) + `lib/assistant/client`; path
confinement, the flag allowlist, plain-resume-only (no `--force`), and the
single-owner preflight are all inherited from the underlying implementation.
`assistant_ask` runs a READONLY `campaign-tick` engine turn (observe + report),
so an MCP client cannot mutate the campaign via the LLM — direct mutations go
through the explicit tools. Full contract: `lib/assistant/` module docstrings.

| Tool | Description |
|------|-------------|
| `assistant_campaign_queue` | Return the book/corpus queue + manifest states (read-only) |
| `assistant_campaign_run_status` | Active/recent run states; optional `wf_id` for one record + log tail |
| `assistant_campaign_prepare_run` | Validate corpus + flag overrides, then write a fixed-shape env overlay (no script) |
| `assistant_campaign_launch_run` | Launch a prepared overlay by name with the full single-owner preflight |
| `assistant_campaign_resume_run` | Resume a paused run with a PLAIN `--resume` (never `--force`) |
| `assistant_campaign_stop_run` | Gracefully stop one run (drops the stop sentinel) |
| `assistant_campaign_prepare_training` | Validate a book's Stage-B LoRA-training readiness (manifest status, pairs, base model, reviewer approval marker, training env, single-owner) — mutates nothing |
| `assistant_campaign_launch_training` | Re-validate everything, `docker stop` every registered vLLM seat + verify the card is free, then launch `ed4all run trainforge_train` detached (fixed argv) |
| `assistant_campaign_training_status` | Recent/active training runs (launched-runs `kind:"training"` rows) + WF status + bounded log tail + env-readiness (read-only) |
| `assistant_campaign_report` | File a structured review-queue report for a human reviewer |
| `assistant_seat_status` | Dynamic seat-resolution view: which seat/model would answer now (read-only probe) |
| `assistant_ask` | One-shot READONLY `campaign-tick` engine turn (observe + report only) |

**Trainforge tools** — see `Trainforge/CLAUDE.md § MCP Tools`.

### Pipeline Tools

| Tool | Description |
|------|-------------|
| `stage_semantik_outputs` | Stage SemantiK outputs for Courseforge |
| `get_pipeline_status` | Check pipeline progress |
| `validate_semantik_markers` | Validate SemantiK output markers |
| `archive_to_libv2` | Archive course artifacts to LibV2. Emits a top-level `chunker_version` field in `course_manifest.json` (resolved via `Trainforge.chunker.CHUNKER_SCHEMA_VERSION`) so LibV2 audits know which chunker shipped the corpus. |

**Pipeline-internal registry-only tools** (wired into `MCP/tools/pipeline_tools.py::_build_tool_registry` for workflow-phase dispatch; intentionally **not** decorated with `@mcp.tool()` — not reachable from external MCP clients):

| Tool | Phase | Purpose |
|------|-------|---------|
| `build_source_module_map` | `source_mapping` | TF-IDF-driven router that maps SemantiK source blocks to Courseforge module pages. Output: `source_module_map.json`. |
| `extract_textbook_structure` | `objective_extraction` | Runs `SemanticStructureExtractor` over every staged SemantiK HTML file and merges per-file chapter/section hierarchies into a single `textbook_structure.json`. |
| `plan_course_structure` | `course_planning` | Synthesizes canonical `TO-NN` / `CO-NN` learning objectives from the textbook structure and publishes `synthesized_objectives.json`. |

**Phase-name dispatch override** (`MCP/core/executor.py::_PHASE_TOOL_MAPPING`): nine phases route by phase name, not agent name — `content_generation_outline` → `run_content_generation_outline`; `inter_tier_validation` → `run_inter_tier_validation`; `content_generation_rewrite` → `run_content_generation_rewrite`; `post_rewrite_validation` → `run_post_rewrite_validation`; `imscc_chunking` → `run_imscc_chunking`; `assessment_synthesis` → `run_assessment_synthesis`; `heading_judge` → `run_heading_judge` (the SEMANTIK_HEADING_JUDGE post-conversion Super heading-level judge — default ON, explicit falsey token opts out; skip-with-pass when explicitly off / born-digital, per-chapter fail-open copy-back); `training` → `run_training` (wraps `Trainforge.train_course` — the standalone `trainforge_train` workflow and the opt-in `textbook_to_course` tail share this one handler); `evaluation` → `run_evaluation` (held-out harness + grounded-answer arms, one verdict derived by calling `EvalGatingValidator` rather than re-implementing its thresholds). `run_training` / `run_evaluation` additionally sit in a deterministic-tool set keyed on the resolved tool name, so they execute in-process even under `ED4ALL_AGENT_DISPATCH` (the subagent fork happens before the registry lookup and cannot produce an adapter). Validator-only phases declare `agents: []` in `config/workflows.yaml`; `workflow_runner._create_phase_tasks` synthesizes a virtual `phase-handler` task only when the phase appears in this map. The mapping cannot be inferred from YAML.

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

`planning → content_generation (batches of 10) → assessment_synthesis (W10, optional QTI/discussion/assignment emit) → packaging (IMSCC) → validation (QA + WCAG) → finalization`. Like `textbook_to_course`, this workflow also declares the four `COURSEFORGE_TWO_PASS=true` tiers (`content_generation_outline` → `inter_tier_validation` → `content_generation_rewrite` → `post_rewrite_validation`) between `content_generation` and `packaging`; `content_generation` itself is `enabled_when_env: COURSEFORGE_TWO_PASS!=true`, so exactly one of the two paths runs. Full phase shapes: `config/workflows.yaml::course_generation`.

### Other workflows

`rag_training` (extraction → indexing → assessment_generation → validation) — see `config/workflows.yaml` for canonical phase shapes.

### Textbook-to-Course Workflow

```
1. semantik_conversion
   └── Convert PDF textbooks to accessible HTML (multi-source synthesis).
       The legacy phase name is accepted on READ (checkpoint alias +
       phase_outputs resume-normalization) so old paused runs still
       `--resume`.

2. heading_judge (SEMANTIK_HEADING_JUDGE, default ON — deviation; opt out
   with an explicit falsey token 0/false/no/off)
   └── Super heading-level judge over the GLM-OCR lane's
       {stem}.glmocr_layout.json sidecars: re-levels heading_level_pending
       headings via the standalone judge subprocess, then copies judged
       {stem}_accessible.html + corrected escalations back over the
       conversion output (.prejudge.bak / .bak kept; the layout sidecar is
       never overwritten). Skip-with-pass when explicitly opted out or the
       corpus is born-digital; per-chapter FAIL-OPEN (never blocks a build).

3. staging
   └── Stage SemantiK outputs to Courseforge inputs

4. chunking
   └── Emit the SemantiK chunkset from the staged HTML via the
       `semantik-chunker` agent (deterministic, no LLM dispatch).
       Outputs: semantik_chunks_path + semantik_chunks_sha256. Gated on
       chunkset_manifest + chunk_wcag_status.

5. objective_extraction
   └── Parse staged SemantiK HTML into textbook_structure.json (chapters,
       sections, content blocks); auto-scales duration_weeks to max(8,
       chapters) when --weeks is unset.

6. source_mapping
   └── Map SemantiK source blocks to Courseforge module pages; emits
       source_module_map.json consumed by content_generation.

7. course_planning
   └── Synthesize canonical TO-NN / CO-NN learning objectives from
       textbook_structure; emits synthesized_objectives.json. Re-scales
       duration_weeks to the TO-driven max(8, num_tos) when --weeks is
       unset (WS5 §3.2 pacing — see docs/operations/pipeline-invocation.md
       § 2.1; skipped for --weeks / --reuse-objectives).

8. concept_extraction
   └── Build the pedagogy/concept graph from the staged output + chunkset
       via the `pedagogy-graph-builder` agent. Depends on course_planning
       + chunking; emits concept_graph_path, concept_graph_sha256, and
       domain_concept_vocabulary_path. Gated on concept_graph +
       domain_concept_vocabulary.

9. content_generation
   └── Generate course content modules (parallel batches of 10). Every
       emitted sourceId must resolve against the SemantiK staging manifest
       (source_refs gate). SINGLE-PASS path only — declares
       `enabled_when_env: COURSEFORGE_TWO_PASS!=true`, so it is skipped
       whenever the two-pass tiers (10-12, 14) are active.

10. content_generation_outline (COURSEFORGE_TWO_PASS=true)
   └── Two-pass Phase 3 — outline-tier Block emit (terse provider, no HTML
       body). Routed by phase NAME to run_content_generation_outline via
       _PHASE_TOOL_MAPPING. Emits blocks_outline_path.

11. inter_tier_validation (COURSEFORGE_TWO_PASS=true)
   └── Two-pass Phase 3 — validator-only phase (`agents: []`) running the
       structural validators over the outline tier; routed by phase NAME
       to run_inter_tier_validation. Emits blocks_validated_path +
       blocks_failed_path.

12. content_generation_rewrite (COURSEFORGE_TWO_PASS=true)
   └── Two-pass Phase 3 — rewrite-tier HTML emit consuming the validated
       Block outlines; routed by phase NAME to
       run_content_generation_rewrite. Emits blocks_final_path +
       content_paths; gated on content_grounding.

13. assessment_synthesis (optional)
   └── W10 — synthesize grounded quizzes, short assignments, and
       discussion prompts from the SemantiK chunkset and emit QTI 1.2 /
       imsdt / assignment XML + manifest.json into <export>/06_assessments/
       (canonical IMS CC resource types). Validator-only phase routed by
       phase NAME to run_assessment_synthesis via _PHASE_TOOL_MAPPING;
       gated critical on qti_well_formed + assessment_objective_alignment.
       Runs before packaging; skipped via generate_assessments=false.

14. post_rewrite_validation (COURSEFORGE_TWO_PASS=true)
   └── Two-pass Phase 3.5 — validator-only phase (`agents: []`) re-running
       the inter-tier validators against the rewrite-tier blocks; routed
       by phase NAME to run_post_rewrite_validation. Under
       COURSEFORGE_TWO_PASS it additionally depends on
       assessment_synthesis (depends_on_when_env), so it observes the
       emitted assessments. Carries the largest gate set of any phase
       (including block_prose_entailment, claim_support, and the
       block-quality rollup).

15. packaging
   └── Package course as IMSCC via the mature multi-file packager.

16. imscc_chunking
   └── Emit the post-packaging IMSCC chunkset from the packaged course via
       the `semantik-chunker` agent. Outputs imscc_chunks_path +
       imscc_chunks_sha256; gated on chunkset_manifest +
       chunk_wcag_status. This is the phase `--stop-after imscc_chunking`
       names for the "retrieval-ready course, no training synthesis" slice.

17. trainforge_assessment (optional)
   └── Generate assessments from the IMSCC package. Fails closed if any
       assessment objective_id isn't covered by a chunk's
       learning_outcome_refs.

18. training_synthesis (optional)
   └── Synthesize instruction + preference training pairs from the
       generated chunks + assessments. Routes via the
       `training-synthesizer` agent (tool: `synthesize_training`).
       Skipped ONLY via `--skip-training`; the provider resolves through
       `TRAINFORGE_SYNTHESIS_PROVIDER` (license-clean local/together —
       the anthropic SDK path fails closed; no `ANTHROPIC_API_KEY`
       involvement).
       Emits a per-pair resume sidecar at
       `training_specs/.synthesis_pairs_checkpoint.jsonl` (opt out via
       `--no-checkpoint`).

19. libv2_archival
   └── Archive course artifacts to LibV2 (raw PDFs, SemantiK HTML, IMSCC,
       RAG corpus). Gated by libv2_manifest integrity checks.

20. vector_indexing (optional)
   └── Build the per-course on-device vector index from the LibV2-archived
       chunkset so a freshly-built course is askable. Routes via the
       `rag-indexer` agent (tool: `run_vector_indexing`). Runs by default;
       skips cleanly when the `[embedding]` extras are absent UNLESS
       `TRAINFORGE_REQUIRE_EMBEDDINGS` is set (then fails closed on a
       broken embedding backend).

21. training (optional, --with-training)
   └── Train the course-pinned SLM adapter in-build against the
       just-archived LibV2 course. Validator-only-style phase
       (`agents: []`) routed by phase NAME to `run_training` via
       _PHASE_TOOL_MAPPING; wraps `Trainforge.train_course` (same
       LocalBackend + TrainingRunner construction as the standalone
       `trainforge_train` workflow) and asserts the base model through
       BaseModelRegistry BEFORE constructing a runner, so an unknown
       name returns the supported list instead of silently substituting.
       Declares `seats: []` (the trainer wants the whole card, so every
       vLLM seat must be down) and a 720-minute ceiling. Emits run_dir +
       model_card_path + adapter_path. Skipped unless `--with-training`;
       `--skip-training` always wins.

22. post_training_validation (optional, --with-training)
   └── Validator-only phase (`agents: []`, no handler) carrying the two
       gates copied verbatim from `trainforge_train`: eval_gating (reads
       <model_dir>/eval/eval_report.json; fails closed on faithfulness
       regression / yes-bias / no-bias / source-match drop / baseline
       regression) and family_completeness (fails closed on a partially
       complete CURIE family). Both critical.

23. evaluation (optional, --with-training)
   └── Evaluate the trained adapter — held-out harness + grounded-answer
       arms — and derive ONE verdict by calling EvalGatingValidator
       itself, so the phase can never disagree with the gate above it.
       Routed by phase NAME to `run_evaluation`; its report merges
       ADDITIVELY into the same eval_report.json path that gate reads
       (harness-owned keys preserved byte-for-byte). Ordered after
       post_training_validation on purpose — no eval wall-clock is spent
       on an adapter the gate already blocked. model_dir is threaded
       from the training phase's run_dir.

24. finalization
   └── Final validation and training data export. Depends on
       `evaluation`, so it is genuinely LAST; a default build (no
       --with-training) still reaches it because a skipped phase stamps
       `_completed` and the dependency check reads only that.
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
| `semantik-automation-coordinator` | Orchestrate PDF conversion |
| `semantik-converter` | Drives the SemantiK v2 cascade for the `semantik_conversion` phase (PDF → accessible HTML + source provenance). The legacy agent name still resolves as an `AGENT_TOOL_MAPPING` dispatch alias (read-compat only). |
| `imscc-intake-parser` | Extract & inventory IMSCC packages |
| `content-analyzer` | Detect accessibility & quality gaps |
| `accessibility-remediation` | WCAG fixes, alt text, headings |
| `content-quality-remediation` | Educational depth & enhancement |
| `intelligent-design-mapper` | Component selection & styling |
| `remediation-validator` | Final QA & WCAG verification |
| `semantik-chunker` | Emit `LibV2/courses/<slug>/semantik_chunks/chunks.jsonl` from staged SemantiK HTML via `Trainforge.chunker.chunk_content`; deterministic transformation (no LLM dispatch). Backed by `_run_dart_chunking` registered in `MCP/tools/pipeline_tools.py::_build_tool_registry` (the registry key keeps its legacy DART name for checkpoint/read-compat). Canonical agent name for both the `chunking` and `imscc_chunking` phases in `config/workflows.yaml`; the legacy agent name survives only as an `AGENT_TOOL_MAPPING` dispatch alias (read-compat). | <!-- legacy-token: allow -->

### Textbook Pipeline Agents

| Agent | Purpose |
|-------|---------|
| `textbook-stager` | Stage SemantiK outputs for Courseforge |
| `textbook-ingestor` | Parse SemantiK HTML & extract objectives |
| `source-router` | Bind SemantiK source blocks to Courseforge module pages (TF-IDF + confidence scoring) |
| `pedagogy-graph-builder` | Build the pedagogy/concept graph backing the `concept_extraction` phase; routes to `run_concept_extraction` via `MCP/core/executor.py::AGENT_TOOL_MAPPING`. Declared in `config/workflows.yaml` only (no `config/agents.yaml` entry). |
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
| `procurement_evidence` exporter (`lib/governance/procurement_evidence.py`, backlog E4/E5/D5; wired post-loop in `workflow_runner._maybe_write_procurement_evidence`) | `<libv2_course>/retrieval_eval/procurement_evidence_bundle.json` — rolls the newest `grounded_answer_eval_*.json` into a versioned ADVISORY evidence bundle (pinned headline + phrasing/abstention/refusal breakdowns + flag-config stamp + Wilson/PPI CIs + blocking-flip readiness). Keyed to the promotion-chain report by `chain_hash`; never mutates it or `course_status`. Missing report → explicit `not_evaluated`. Best-effort (never alters `final_status`). | in-module `EVIDENCE_SCHEMA_VERSION` 1.0 |
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
| `course_generation` | 34 | 30 | 64 |
| `rag_training` | 4 | 3 | 7 |
| `textbook_to_course` | 65 | 74 | 139 |
| `trainforge_train` | 2 | 0 | 2 |
| **Total** | **105** | **107** | **212** |

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
| `TRAINFORGE_*` / `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` / `CURRICULUM_ALIGNMENT_*` / `WAVE18_*` | [`Trainforge/CLAUDE.md § Opt-In Behavior Flags`](Trainforge/CLAUDE.md) | 75 |
| `NVIDIA_*` (vendor endpoint-registry row for the hosted large-model seat — `NVIDIA_API_KEY` / `NVIDIA_BASE_URL` / `NVIDIA_LARGE_MODEL`) | [`Trainforge/CLAUDE.md § Opt-In Behavior Flags`](Trainforge/CLAUDE.md) | 3 |
| `SEMANTIK_*` (SemantiK semantic-cascade converter; also honors the single legacy `DART_THETA_DEVICE` compat env, aliased to `SEMANTIK_THETA_DEVICE`) <!-- legacy-token: allow --> | [`SemantiK/CLAUDE.md § Opt-In Behavior Flags`](SemantiK/CLAUDE.md) | 164 |
| `COURSEFORGE_*` / `COURSEPLANNER_*` / `TEXTBOOK_SYNTHESIS_*` | [`Courseforge/CLAUDE.md § Opt-In Behavior Flags`](Courseforge/CLAUDE.md) | 45 |
| `DECISION_*` / `ED4ALL_*` / `LOCAL_DISPATCHER_*` / `MCP_ORCHESTRATOR_*` / `LLM_*` (cross-cutting) | root index (below) + [`docs/operations/behavior-flags.md`](docs/operations/behavior-flags.md) | 257 |

### Cross-cutting flags (root-owned)

**Full per-flag detail** (resolution chains, guardrails, calibration status, anti-fabrication contracts) lives in [`docs/operations/behavior-flags.md`](docs/operations/behavior-flags.md). The table below is a grep-able one-line index; every flag name stays searchable here.

| Flag | Default | One-line purpose |
|------|---------|------------------|
| `DECISION_VALIDATION_STRICT` | unset | Fails closed on unknown `decision_type` values in decision captures. |
| `ED4ALL_CAPTURE_BUFFER` | unset (off) | Coalesces the per-decision write+flush+fsync across the three JSONL capture mirrors into one batched write every N rows (telemetry-only crash-loss window; drains on flush/close/atexit). |
| `ED4ALL_CAPTURE_BUFFER_ROWS` | `50` | Satellite of `ED4ALL_CAPTURE_BUFFER` — buffered-row batch size / worst-case telemetry-loss window (garbage / ≤0 → 50). |
| `ED4ALL_BLOCK_QUALITY_RUBRIC` | unset (off; **A5 auto-on for pipeline runs**) | IB6 keystone — the 8-dim 0-3 block-quality scoring + rollup pass and its composing validators. |
| `ED4ALL_BLOCK_BODY_CHAR_CEILING` | `200` (global override) / per-type default | IB6.4 per-block D2 cognitive-load body ceiling |
| `ED4ALL_BLOCK_QUALITY_SHADOW` | unset (off; **A5 auto-on for pipeline runs**) | W8.8 shadow-collect gate for the IB6 block-quality validators |
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
| `ED4ALL_ANSWER_EXCLUDE_CHUNK_TYPES` | unset (off) | FIX A — comma-separated `chunk_type` values to exclude from the grounded-answer retrieval candidate pool (drops e.g. QTI-harvested `assessment_item` chunks that fail the anchor gate by construction; over-fetches to keep top-`limit` full). Default unset → byte-identical. |
| `ED4ALL_ANSWER_ANCHOR_CONTAINMENT` | `0.85` | FIX B — float citation-gate anchor containment floor for the answer path (clamped `[0.5, 1.0]`; garbage → 0.85); does not change `citation_anchor.py`'s own default. |
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
| `ED4ALL_ARCHIVE_REQUIRE_FULL_COURSE` | unset (off; **A5 auto-on for pipeline runs**) | "True full course" archival-completeness strict-mode gate |
| `ED4ALL_ASSISTANT_BASE_URL` | `http://localhost:8004/v1` | Base URL of the local nano vLLM seat behind `ed4all assistant`; loopback-only (non-loopback host → raise, mirroring the grounded-answer guard). |
| `ED4ALL_ASSISTANT_MODEL` | `nemotron-3-nano` | Model ID for the assistant seat (must match its `--served-model-name`); honors `ED4ALL_REASONING_THINKING_OFF`. |
| `ED4ALL_ASSISTANT_AUTOSTART` | unset (off) | When truthy and the seat is down, `ed4all assistant` lazy-starts the `spark-nano` seat via `start_seat_coherent` (liveness ceiling + coherence probes); `--no-seat-start` forbids it. |
| `ED4ALL_ASSISTANT_MAX_TOKENS` | `1024` | Per-reply generation cap for the assistant seat (garbage / non-positive → 1024). |
| `ED4ALL_ASSISTANT_TIMEOUT_SECONDS` | `120` | Assistant-seat HTTP timeout; a transport failure raises `AssistantSeatUnavailable` (never a canned fallback). |
| `ED4ALL_ASSISTANT_SEAT` | `spark-nano` | Logical registry seat name the assistant autostart path targets (model-agnostic — nano is only the default deployment; point at any registered local seat). |
| `ED4ALL_ASSISTANT_DEBUG_ON_FAILURE` | unset (off) | Campaign driver writes `<campaign dir>/last_failure.json` on a failed book + prints the `ed4all assistant --debug --run <id>` command; `build_debug_context(None)` prefers that pointer. |
| `ED4ALL_ASSISTANT_SEAT_PRIORITY` | `spark-super,spark-nano` | Ordered logical-seat priority the `ed4all assistant` dynamic resolver walks (registry-driven via `ED4ALL_SEAT_BASE_URLS`, loopback-enforced, served model read from `/v1/models`); first LIVE seat answers, so the assistant rides Super while the pipeline serves it, else nano. Priority seats are never autostarted (only fallback nano is). |
| `ED4ALL_CAMPAIGN_DIR` | `plans/campaign` (repo-relative) | Operator campaign-harness directory (`lib/paths.py::campaign_dir`) — the single site-configurable root every campaign path derives from (manifest, per-book logs, prepared overlays, review queue, pilot driver). Absolute wins; relative resolves against `PROJECT_ROOT`; blank → default. |
| `ED4ALL_PILOT_TICK_SECONDS` | `600` | Loop interval (s) for the deterministic campaign monitor driver `<campaign dir>/pilot.py` (positive int; garbage → 600). |
| `ED4ALL_PILOT_MAX_AUTO_RESUMES` | `2` | Per-run cap on pilot plain `--resume` auto-retries for a PAUSED run before it files a `run_paused` review report (never `--force`; non-negative int, 0 = report immediately, garbage → 2). |
| `ED4ALL_PILOT_STALL_MINUTES` | `45` | Phase-stall window (min) — a RUNNING run whose newest checkpoint/sidecar mtime is older is flagged `stalled` in the pilot snapshot (report-only; positive int, garbage → 45). |
| `ED4ALL_PILOT_MAX_AUTO_RECOVERIES` | `1` | Per-run cap on the campaign pilot's scheduler-only SELF-RECOVERY of a FAILED run: an auto `--resume` fires only when the failure's UNDERLYING cause classifies TRANSIENT (a seat stopped out from under the run → connection refused → `POISON_PILL`, classified via the production `ErrorClassifier`, not the surface status) AND a self-heal gate confirms the needed seats serve now / no `STOP_ALL` / disk not full; content-gate / schema / genuine-poison / unknown / Stage-B failures never recover (they halt for human). Persisted separately from the paused-run auto-resume counter; 0 = never recover, garbage/negative → 1. |
| `ED4ALL_PILOT_SCHEDULE` | unset (off) | Opt-in campaign SCHEDULER in the pilot tick loop (also `--schedule`): reconcile manifest vs workflow records, then at most ONE action per idle tick — next pending book's Stage-A build, or Stage-B `trainforge_train` for a built+pairs+approved book (approval marker written only by the reviewer). Off → monitor-only, byte-identical. |
| `ED4ALL_PILOT_MAX_CONSECUTIVE_FAILURES` | `2` | Scheduler circuit breaker — consecutive failed books before the campaign halts with a `campaign_halted` report (success resets; persisted across pilot restarts; training-launch refusals never count). |
| `ED4ALL_PILOT_NO_REVIEW_GATES` | unset (off) | Full-auto bypass of the per-book `training-approved` reviewer marker gating Stage-B launches (default keeps the review-protocol contract). |
| `ED4ALL_CAMPAIGN_BASE_MODEL` | `nemotron3-nano-30b` | Campaign Stage-B base-model selector — must resolve in `Trainforge/training/base_models.py::BaseModelRegistry`; unknown name = loud error, never a fallback model. Has a `docs/LICENSING.md` row (Nemotron license pin guard). |
| `ED4ALL_RERANK_PROVIDER` | unset (off) | Cross-encoder reranker over the first-stage retrieval candidate pool on the grounded-answer path |
| `ED4ALL_BLOCK_ANATOMY` | unset (off) | IB1 six-slot anatomy contract emit gate |
| `ED4ALL_BLOCK_A11Y` | unset (off) | IB4 per-block WCAG 2.2 AA + UDL emit gate |
| `ED4ALL_CALLOUT_TYPED` | unset (off) | FR-A11Y-03 typed B12 callout emit + gate flag |
| `ED4ALL_COS_PER_WEEK_CAP` | `0` (auto) | WS5 §2.2 per-week chapter-objective placement cap for the single-sourced ceil-stride slicer. |
| `ED4ALL_WEEK_TO_GROUPS` | unset (off) | WS5 week-grouping override: when on AND `duration_weeks == num_tos`, per-week `"Week N"` groups are built by TO membership (week N = TO-N's `child_co_ids`) instead of the ceil-stride CO slice; else warns + falls back to ceil-stride. |
| `ED4ALL_CONCEPT_COVERAGE` | unset (off; **A5 auto-on for pipeline runs**) | W4.1 read-only capability aggregator |
| `ED4ALL_CONCEPT_EXTRACTION_CHECKPOINT` | `on` | Site override for the concept_extraction Stage-3 per-window (`synthesize_concepts`) resume sidecar (beats `ED4ALL_GENERATION_CHECKPOINT`). |
| `ED4ALL_INTELLIGENCE_RUBRIC` | unset (off; **A5 auto-on for pipeline runs**) | W4.6 read-only capability aggregator |
| `ED4ALL_CONTENT_PAGE_PER_CO` | unset (off) | Page-per-CO content-emit gate |
| `ED4ALL_CONTENT_PAGE_NUM_CTX` | `4096` (→ `ED4ALL_ANSWER_NUM_CTX` → 4096) | Authoring serving-window token budget for the page-per-CO per-page chunk cap |
| `ED4ALL_CONTENT_PAGE_MAX_CHUNKS` | `5` | Hard top-K ceiling on chunks kept per CO page for the page-per-CO cap |
| `ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED` | unset (off) | TRUE one-page-per-CO opt-in: lifts the page-per-CO NEVER-INCREASE topic cap so a CO-rich week emits one content page per CO (1:1 binding; O(Σ COs) authoring cost) |
| `ED4ALL_CHUNK_ROLE_DIVERSIFY` | unset (off; **A5 auto-on for pipeline runs**) | Gap #11 — deterministic per-block-role rotation of a page's ranked chunk order so co-located blocks don't all lead with the same anchor example (chunk-universe remap; bare library calls keep default-off → byte-identical). |
| `ED4ALL_COURSE_IDENTITY_DEDUP` | unset (off) | W0.5 course-identity SPLIT-BRAIN guard |
| `ED4ALL_EMBEDDING_PROVIDER` | `st` | Selects the retrieval-index embedding backend (`st` / `local-openai` / `fake`). |
| `ED4ALL_EMBEDDING_MODEL` | per-provider | Model ID override for the embedding provider |
| `ED4ALL_EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | Base URL of the local OpenAI-compatible `/v1/embeddings` server (`local-openai` only). |
| `ED4ALL_EMBEDDING_API_KEY` | `local` | Optional bearer token for the local embedding server (`local-openai` only). |
| `ED4ALL_EMBEDDING_DEVICE` | `cpu` | Torch device for the in-process `st` provider |
| `ED4ALL_EMBEDDING_BATCH_SIZE` | `16` | Encode batch size for the embedding client (recorded in the index manifest). |
| `ED4ALL_EMBED_BATCH_TUNE` | unset (off) | Builder-C entailment-gate embed-throughput knob: the two hot in-process `SentenceEmbedder` callers (`feature_cache.BlockFeatureCache.embed` + the groundedness S1 embed path) pass `batch_size=256, length_sort=True` into `encode_batch` / `encode_batch_cached` (bigger GPU batches + longest-first packing to cut padding waste; input order restored). Off → today's exact `encode_batch(texts)`, byte-identical. |
| `ED4ALL_EMBED_PERSIST_CACHE` | unset (off) | Builder-C disk-persisted embedding cache: those same callers route through `SentenceEmbedder.encode_batch_cached` (content + model-addressed `state/embedding_cache.batch.<model>.jsonl`; batch-encodes only misses, appends incrementally, returns input order). Off → in-memory-only exactly as today. |
| `ED4ALL_EMBED_FP16` | unset (off) | Builder-C fp16 embed cast: a freshly-loaded `SentenceTransformer` that landed on CUDA is cast via `model.half()` (CPU models left fp32; best-effort — any failure keeps fp32). Cosine shifts only in low-order bits (~1e-3), well under every downstream threshold. Off → fp32, byte-identical. |
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
| `ED4ALL_WORKED_EXAMPLE_FLOOR` | unset (off; **A5 auto-on for pipeline runs**) | P4 worked-example DENSITY floor: ≥1 example/problem block per procedural (apply/create) CO. |
| `ED4ALL_BLOOM_SPREAD_FLOOR` | unset (off; **A5 auto-on for pipeline runs**) | P4 Bloom-spread floor: ≥1 analyze-or-higher block per week (widens the apply-heavy Bloom mix). |
| `ED4ALL_TRIANGLE_FLOOR` | unset (off; **A5 auto-on for pipeline runs**) | GAP D (IB3 alignment triangle) per-CO activity + assessment floor for the block planner. |
| `ED4ALL_RETRIEVAL_INTERLEAVE` | unset (off; **A5 auto-on for pipeline runs**) | GAP C (IB7.5b interleaved retrieval) per-content-page retrieval-block floor. |
| `ED4ALL_HOME` | unset (repo-relative) | Relocatable data root — sets every mutable data dir under it instead of repo-relative. |
| `ED4ALL_IMSCC_MODULE_TITLES` | unset (legacy) | IMSCC packager module-title mode: exact token `to` titles org groups "Module N: <TO topic>" from `terminal_objectives[N-1]` (anchor_module_title → truncated statement → legacy fallback); anything else → legacy "Week N" titles. Operator-set alongside `ED4ALL_WEEK_TO_GROUPS`. |
| `ED4ALL_KEY_TERMS_PAGE` | unset (off; **A5 auto-on for pipeline runs**) | Feature I5 — per-terminal-objective deterministic **"Key Terms" page** gate |
| `ED4ALL_NEW_BLOCK_TYPES` | unset (off) | IB5 gate for four framework block types: hook, multimedia, worked_example, diagram. |
| `ED4ALL_REFLECTION_CALIBRATION` | unset (off) | FR-INT-03 gate for the B11 reflection predict-then-reveal calibration contract. |
| `ED4ALL_REASONING_THINKING_OFF` | unset (off) | Injects the Nemotron "detailed thinking off" system directive + `chat_template_kwargs.enable_thinking=false` on every composed OpenAI-compatible call so reasoning-token output doesn't trip the finish_reason=length truncation guard. |
| `ED4ALL_REASONING_LOW_EFFORT` | unset (off) | Enables compatible servers' low-effort reasoning mode; takes precedence over thinking-off without selecting a provider or model. |
| `ED4ALL_RECALL_SELF_CHECK` | unset (off) | Free-recall / cloze self-check variant gate |
| `ED4ALL_MISCONCEPTION_RICH` | unset (off) | Named subject-specific misconception + productive-failure gate for the B03/B12 `misconception` block |
| `ED4ALL_MAYER_CTML` | unset (off) | Mayer CTML 12-principles structural check enriching the UDL/multimedia surface |
| `ED4ALL_BLOOM_DISTRIBUTION` | unset (off; **A5 auto-on for pipeline runs**) | Course-level Bloom-distribution-vs-target-curve gate |
| `ED4ALL_BLOOM_DISTRIBUTION_TARGET` | unset (canonical default) | Operator override for the target Bloom curve consumed by `BloomDistributionValidator` |
| `ED4ALL_BLOOM_DISTRIBUTION_TOLERANCE` | `0.20` | Float L1-deviation tolerance for the `BLOOM_DISTRIBUTION_OFF_TARGET` decision |
| `ED4ALL_BLOOM_DISTRIBUTION_MIN_LOS` | `6` | Small-N objective floor for `BloomDistributionValidator` |
| `ED4ALL_BLOOM_TRIVOTE` | unset (off) | Re-founds the `bloom_classifier_disagreement` gate on THREE interpretable voters — (1) the generator's OWN asserted `bloom_level` read from the artifact metadata, (2) zero-shot DeBERTa entailment of per-Bloom-level hypothesis templates on the ALREADY-LICENSED `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` process-singleton (no second model load), (3) the deterministic verb-based level from `lib/ontology/bloom.py` — retiring the unlicensed/undocumented `cip29/bert-blooms-taxonomy-classifier` + the sentiment member (`distilbert-sst-2`) from `BloomBertEnsemble`. Gate meaning becomes "does the generator's own Bloom claim survive independent checks". Zero new model weights (HF-offline). Default off → byte-identical legacy 3-member-ensemble path. |
| `ED4ALL_HARVEST_BLOOM_LABELS` | unset (off) | Post-build `workflow_runner` hook — deterministically (no LLM) walks the resolved Courseforge export (+ the LibV2 course dir when archival ran) and appends every de-duplicated artifact-asserted Bloom claim (objectives / outline+rewrite blocks / assessment items) to the shared `state/bloom_labels/labels.jsonl` store (the corpus behind `ED4ALL_BLOOM_TRIVOTE` voter 1). Best-effort (never alters `final_status`). When off the hook short-circuits BEFORE any path resolution → byte-identical (nothing written). Also drivable standalone via `ed4all harvest-bloom-labels`. |
| `ED4ALL_PREREQ_SEQUENCING` | unset (off) | Prerequisite-DAG-driven content sequencing |
| `ED4ALL_PREREQ_TRANSITIVE_REDUCTION` | unset (off) | W3.3 deterministic stdlib DFS transitive reduction of the projected TO→TO prereq graph |
| `ED4ALL_PREREQ_CENTRALITY_TIEBREAK` | unset (off) | W3.4 concept-centrality stable tie-break for the TO topological sort |
| `ED4ALL_PREREQ_CENTRALITY_METHOD` | `in_degree` | W3.4 satellite selecting the centrality method consumed by `ED4ALL_PREREQ_CENTRALITY_TIEBREAK` |
| `ED4ALL_KG_PREREQ_HEALTH` | unset (off; **A5 auto-on for pipeline runs**) | W3.2/W3.4/W3.6 prereq-DAG health signals on the `kg_quality_report` validator |
| `ED4ALL_RICHER_VISUAL_SYSTEM` | unset (off) | Richer-visual-system Phase 0 gate |
| `ED4ALL_LIBV2_ROOT` | `<repo>/LibV2/` | Absolute path to the LibV2 root directory |
| `ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS` | `60` at the client; `300` at the content-generation providers | Per-request HTTP timeout (s) for local content-generation LLM calls (7B prose authoring). |
| `ED4ALL_LLM_OMIT_OLLAMA_FORMAT` | unset (off) | Omits the Ollama-only top-level `format` field for strict OpenAI servers while retaining standard `response_format`. |
| `ED4ALL_MAILBOX_BASE_DIR` | `<repo>/state/mailbox/` | Orchestrator task-mailbox base directory. |
| `ED4ALL_NLI_DEVICE` | `cpu` (code) / `cuda` (project default) | Torch device for the in-process NLI classifier that scores groundedness/eval entailment |
| `ED4ALL_NLI_MIN_FREE_VRAM_MIB` | `1024` | Free-VRAM floor gating the in-process NLI model onto CUDA |
| `ED4ALL_NLI_EVICT_FOR_CUDA` | `true` (on) | VRAM-contention strategy: evict the resident ollama model to free the card for NLI on CUDA. The `ED4ALL_GPU_LIFECYCLE` phase-boundary sweep makes this largely moot (it hands the card over before NLI loads); kept as the in-phase fallback (generation + NLI in one phase, or lifecycle opted off). |
| `ED4ALL_NLI_MICROBATCH` | unset (off) | Opt-in NLI micro-batching dispatcher: concurrent Courseforge best-of-N workers (`score_candidate`) bypass the `_NLI_SCORE_LOCK` mutex and route (premise, hypothesis) pairs through one background scorer thread that coalesces batched forward passes (defeats the ~28-in-flight lock ceiling at 64 rewrite threads). Scoped to `score_candidate`; the router's in-loop `_run_validator_chain` `_NLI_SCORE_LOCK` is retained (it guards the non-NLI sentence-transformer + Bloom BERT singletons the NLI-only dispatcher does not own). The standalone `block_prose_entailment` gate has its OWN independent opt-in (`ED4ALL_NLI_MICROBATCH_VALIDATORS`) that uses a spawn PROCESS pool (not this thread dispatcher). Batched vs single-pair scores differ only in low-order bits (proper attention masking → semantically identical). |
| `ED4ALL_NLI_MICROBATCH_MAX_PAIRS` | `64` | Satellite of `ED4ALL_NLI_MICROBATCH` — max pairs coalesced into one drain / batched forward pass (the `score_candidate` dispatcher path). |
| `ED4ALL_NLI_MICROBATCH_WINDOW_MS` | `10` | Satellite of `ED4ALL_NLI_MICROBATCH` — collection window (ms) the `score_candidate` scorer waits for more concurrent requests before scoring. |
| `ED4ALL_NLI_CROSSBLOCK` | unset (off) | Single-process cross-block NLI concurrency for the standalone `block_prose_entailment` gate — fans SCORABLE cache-miss blocks across a `ThreadPoolExecutor` whose workers run the UNCHANGED `score_groundedness` against the shared coalescing dispatcher (one scorer thread owns the GPU model; NO multiprocessing / extra CUDA contexts). Supersedes the dormant `ED4ALL_NLI_MICROBATCH_VALIDATORS` spawn-pool (the measured livelock). Verdict-identical (low-order softmax bits only); OOM-safe (dispatcher halves the drain on CUDA OOM); resume/stop-cooperative. Default off → serial, byte-identical. |
| `ED4ALL_NLI_CROSSBLOCK_THREADS` | `16` | Satellite of `ED4ALL_NLI_CROSSBLOCK` — ThreadPoolExecutor worker count fanning cache-miss blocks at the dispatcher (capped at the scorable-miss count). Pairs with `ED4ALL_NLI_MICROBATCH_MAX_PAIRS` (raise to 128 for the cross-block population) + `_WINDOW_MS`. |
| `ED4ALL_VALIDATION_FEATURE_CACHE` | unset (off) | Shared per-block feature cache for the validation-gate suite (`lib/validators/feature_cache.py::BlockFeatureCache`) — compute-once, phase-scoped, thread-safe. Built ONCE before the gate loop in `executor.py` and threaded into every input builder (`GateInputRouter.build(cache=)`) + validator (`inputs["feature_cache"]`); memoizes the ~424-block hydration, `chunks.jsonl` parse, objectives flatten, HTML strip, sentence splits (kept DISTINCT by splitter id), resolved passages, per-chunk windows, and batched embeddings so the 52-gate suite computes them once instead of per gate. Verdict-identical (every accessor delegates to the same self-compute function); content-sha per-block keying self-invalidates a re-roll; never persisted (the `ED4ALL_VALIDATION_CHECKPOINT` sidecar still owns NLI-report persistence). Default off → every seam sees `cache=None` (byte-identical). |
| `ED4ALL_NLI_MICROBATCH_VALIDATORS` | unset (off) | Opt-in (REPURPOSED) — shards the standalone `block_prose_entailment` gate's per-block NLI entailment scoring (`post_rewrite_validation`, `lib/validators/block_prose_entailment.py`) across a spawn `ProcessPoolExecutor` (`ED4ALL_NLI_VALIDATORS_PROCS` workers) so ~424 blocks score in genuine parallel instead of the ~50-min single-stream serial loop. Each persistent worker loads its OWN NLI once via a picklable module-level factory (`default_nli_factory`; the MODEL is never pickled, only a dotted factory string crosses the process boundary) and runs the UNCHANGED `score_groundedness` per block end-to-end. This REPLACES the earlier in-process thread-dispatcher design (a thread pool could not defeat the single-GPU forward-pass serialization; a process pool gives parallel forward passes on distinct CUDA contexts). DELIBERATELY separate from `ED4ALL_NLI_MICROBATCH` (that gates only `score_candidate`): this is a load-bearing quality gate AND the change adds NEW cross-block parallelism, so it earns independent default-OFF control. Results are keyed by block index; issue-list assembly, the 50-issue cap, counters, and decision-capture stay SERIAL in the parent in original block order, so the GateResult is verdict-identical + order-stable vs serial (only the documented GPU-softmax low-order-bit difference, well under the 0.60/0.70/0.50 thresholds). Any pool-start / worker failure degrades gracefully to the serial path. Default OFF → byte-identical serial path (no pool constructed). Garbage / falsey → off (parse-with-fallback). |
| `ED4ALL_NLI_VALIDATORS_PROCS` | `4` | Satellite of `ED4ALL_NLI_MICROBATCH_VALIDATORS` — the number of persistent NLI-scoring worker processes (spawn `ProcessPoolExecutor` size). Default 4 (memory-budget rec: 4 workers ≤ ~6 GB worst-case at batch-8×512 DeBERTa activations beside a resident vLLM seat; safe ceiling ~12-14). Always capped at the number of scorable blocks. Garbage / non-positive → `4`. No-op when the master flag is off. |
| `ED4ALL_NLI_BUCKET_BATCHING` | unset (off) | Token-length bucketed batching for the NLI forward pass (`lib/classifiers/nli_classifier.py::_run_forward`) — the already-length-sorted pairs are partitioned into four `<=128/256/384/512`-token buckets (tokenizer-free `ceil((len(p)+len(h))/3.5)` estimate) and each bucket runs at its own batch size from `ED4ALL_NLI_BUCKET_BATCH`, so short window pairs batch big (~350-650 pairs/s) and long ~512-tok pairs batch small (bounded O(L²) activation, ~26-45 pairs/s). Verdict-identical (per-pair logits independent of batch composition; only low-order float bits move), results restored to input order. Off → single-global-batch-size path, byte-identical. |
| `ED4ALL_NLI_BUCKET_BATCH` | `256,128,64,32` | Satellite of `ED4ALL_NLI_BUCKET_BATCHING` — per-bucket forward-pass batch sizes, comma list aligned to the four `<=128/256/384/512`-token buckets shortest-first. Must parse to exactly four positive ints or the whole default list is used (a partly-garbage override never yields a half-sized plan). No-op when the master flag is off. |
| `ED4ALL_GROUNDEDNESS_FRONTIER` | unset (off) | Frontier-batched early-stop stage-1 for the NLI groundedness scorer (`lib/retrieval/groundedness.py::score_groundedness`) — instead of the whole-passage grid (every claim × every ~700-word passage = the block-prose-entailment monster), each claim is scored against its pool in a deterministic PRIORITY order (direct-cited ids → lexical anchor → MiniLM-cosine tiebreak → stable index) and RETIRES the instant a premise clears the entailment floor. Premises are the SAME whole-passage texts, so verdicts are exact-parity (entailed retires early; non-entailed exhausts its whole pool → bitwise-identical argmax + unchanged stage-2 rescue). Mutually exclusive with the REFUTED `ED4ALL_GROUNDEDNESS_S1_TOPK` cosine-preselect (which EXCLUDES passages, ~50% verdict flips); when both set, FRONTIER wins (one-time warning). Folded into the prose-entailment sidecar fingerprint only when active. Off → legacy grid, byte-identical. |
| `ED4ALL_GROUNDEDNESS_FRONTIER_WIDTH` | `4` | Satellite of `ED4ALL_GROUNDEDNESS_FRONTIER` — per-round frontier width: each still-active claim contributes its next `W` candidate passages per round, all combined into one coalesced `score_batch`. Wider `W` = fewer rounds (less scheduling overhead) but scores more past a claim's first entailing premise before it retires. No-op when the master flag is off; garbage / non-positive → `4`. |
| `ED4ALL_VALIDATION_CHECKPOINT` | `on` (site flag under `ED4ALL_GENERATION_CHECKPOINT`) | Per-block resume sidecar for the slow `block_prose_entailment` gate (`lib/validators/block_prose_entailment.py`). Each scored block's `GroundednessReport` is content-addressed (sha256 of prose + sorted cited chunk_ids/texts + floors + scorer_version + NLI model revision + device-class) and persisted to a fingerprinted JSONL sidecar next to the rewrite export (`<export>/.prose_entailment_cache/`, the `ED4ALL_GENERATION_CHECKPOINT` family contract), so a killed / `ed4all stop`-ed / timed-out validation RESUMES nearly free (already-scored blocks are cache HITs that skip the DeBERTa forward pass). A degraded / unavailable report is NEVER cached. Precedence: this site flag when set > the `ED4ALL_GENERATION_CHECKPOINT` family (falsey `0`/`false`/`no`/`off` → off; unset / garbage → on). Byte-identical to the legacy path when the flag is falsey OR no cache dir is resolvable. The gate also polls the run-scoped stop sentinel between blocks (closing the "validator phases ignore `ed4all stop`" gap) and holds NLI + the MiniLM embedder as process singletons so neither reloads mid-phase. |
| `ED4ALL_OBJECTIVE_REVIEW_PROVIDER` | unset (off) | Grounding-safe **objective-review** pass gate |
| `ED4ALL_OBJECTIVE_REVIEW_MODEL` | per-provider | Model-ID override for the objective-review pass |
| `ED4ALL_OBJECTIVE_CHUNK_RELEVANCE_FLOOR` | `0.30` | Fix 1A relevance floor for the objective-dedup union prune |
| `ED4ALL_OBJECTIVE_CITATION_RESELECT` | unset (off; **A5 auto-on for pipeline runs**) | Post-hoc CO citation re-selection: re-cite the best REAL window/chapter chunk by cosine, INDEPENDENTLY of the (possibly fabricated) cited ids — re-grounds a wholly-fabricated / zero-citation CO instead of leaving ~0 real grounding. Bare library calls keep default-off. |
| `ED4ALL_OBJECTIVE_RESELECT_EXERCISE_DEMOTE` | on (when reselect on) | Demotes exercise/answer-list chunks below instructional prose in the citation re-selection rank (opt-out). |
| `ED4ALL_OBJECTIVE_RESELECT_KEEP_ORIGINAL` | on (when reselect on) | Keep-original union guard for citation re-selection: never STRIP a synthesis citation — unions every above-floor, non-exercise original into the kept set so the cosine top-K can only ADD supporters (fixes an entailment-gate regression; opt-out). |
| `ED4ALL_OBJECTIVE_SANITIZE_CITATIONS` | on (default) | Write-time citation SANITIZER backstop — deterministically DROPS any `source_refs` / `source_chunk_ids` entry that resolves against NOTHING in the current chunkset/textbook-structure universe (fabricated topic-label / statement-echo citations the local synthesizer emits) just before `synthesized_objectives.json` is written, converting a would-be `objective_source_refs` ORPHANED_CITATIONS critical block into the benign `OBJECTIVE_NO_GROUNDING_SOURCE` warning path. Set-membership REMOVAL only (never invents/re-points an id — the complement of `ED4ALL_OBJECTIVE_CITATION_RESELECT`). No-op (byte-identical) on a healthy corpus, a legacy/no-chunkset run, or when set falsey. |
| `ED4ALL_OBJECTIVE_ENTAILMENT_MATH_FOLD` | unset (off) | Opt-in LaTeX/unicode-math folding of premise + hypothesis before NLI in the `objective_entailment` gate (measured net-neutral on a math-scan corpus — see behavior-flags.md; deferred-flip candidate). |
| `ED4ALL_OBJECTIVE_DEDUP_THRESHOLD` | `0.88` | W2 §4.2 cosine clustering threshold for the in-synthesis objective-dedup pass |
| `ED4ALL_OBJECTIVE_DISTINCT_SKILL_SPLIT` | unset (off) | I3 PRONG A — distinct-skill SPLIT gate for the objective-dedup pass |
| `ED4ALL_OBJECTIVE_DEDUP_LEXICAL` | unset (off; **A5 auto-on for pipeline runs**) | W2 Defect E — cross-window lexical-dedup SECOND PASS (complete-linkage merge of near-restatement clusters after single-link cosine, before the PRONG-A split). Bare library calls keep default-off → byte-identical DedupResult. |
| `ED4ALL_OBJECTIVE_DEDUP_LEXICAL_COSINE` | `0.78` | W2 Defect E satellite — centroid-cosine floor for a lexical merge edge (below the 0.88 single-link dedup threshold). |
| `ED4ALL_OBJECTIVE_DEDUP_LEXICAL_JACCARD` | `0.60` | W2 Defect E satellite — best-grounded skill-signature Jaccard floor for a lexical merge edge (above PRONG-A's <0.34 distinctness band). |
| `ED4ALL_OBJECTIVE_SPECIFICITY` | unset (off; **A5 auto-on for pipeline runs**) | W2 Defect B — opt-in gate for the CO-statement specificity/vacuity validator (`objective_specificity` at course_planning; V1 content-residual vacuity + V2 vague-object + V3 source-token recall). Default off → byte-identical skip-with-pass. |
| `ED4ALL_OBJECTIVE_WINDOW_PER_SECTION` | unset (off) | Vendor-depth per-SECTION stage-2 map units — partitions each chapter's chunks along the `textbook_structure.json` chapter→section tree (ordered walk on `section_heading` provenance; <3-chunk sections coalesce with a sibling) so a 10-chapter/71-section corpus synthesizes ~64-71 section-scoped windows instead of ~23 giant packed windows (the measured 52-vs-797-CO depth gap). Build mode + section label fold into the window fingerprint, so a mid-course flip invalidates the window resume sidecars (those windows re-run on `--resume`). Default off → byte-identical chapter-packed windows. |
| `ED4ALL_OBJECTIVE_WINDOW_MAX_CANDIDATES` | unset (legacy 1-3) / `12` in per-section mode | Per-window candidate-objective budget for Stage-2 window synthesis — ONE resolved value drives the prompt ("Synthesize 1-N"), the parse-side schema `maxItems`, AND the grammar payload so all three agree; folded into the window fingerprint (a budget change re-rolls affected windows on `--resume`). Positive-int env > `12` when `ED4ALL_OBJECTIVE_WINDOW_PER_SECTION` is on > legacy `None` (byte-identical "1-3" prompt + untouched schema). Garbage / non-positive → next tier (parse-with-fallback). |
| `ED4ALL_OBJECTIVE_SEED_SANITIZE` | unset (off) | W4 Defect C — exercise-apparatus seed sanitation (`lib/objectives/chunk_window.py::resolve_seed_sanitize`): strips apparatus lines/sentences from the RENDERED Pass-B window body + drops Pass-C survivors whose STATEMENT matches an apparatus marker (chunk_ids / citability untouched). Default off → byte-identical windows. **Operator note:** flipping this mid-course changes the window-render fingerprint, so window resume sidecars invalidate and those windows re-run on `--resume`. |
| `ED4ALL_SYNTHESIS_SKELETON` | unset (off) | Structure-aware objective synthesis — inject a compact CONTEXT-ONLY document HEADING SKELETON of the window's chapter (from `textbook_structure.json` chapters→sections→subsections) into the Stage-2 per-window synthesis prompt so TO/CO derivation is structure-aware. Capped ~1.5k tokens/window, deepest heading levels dropped first. Objectives still cite chunk ids (citation path unchanged). Default off → byte-identical prompt. **Operator note:** the skeleton is folded into the window-render fingerprint, so flipping this mid-course invalidates the window resume sidecars and those windows re-run on `--resume`. Active only when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. |
| `ED4ALL_OBJECTIVE_SOURCE_BACKFILL` | unset (off) | I3 PRONG B — source-richness BACKFILL gate |
| `ED4ALL_OBJECTIVE_BACKFILL_COVERAGE_TARGET` | `1.0` | I3 PRONG B coverage target: min fraction of content-bearing chunks the backfill drives toward. |
| `ED4ALL_OBJECTIVE_BLOOM_RELEVEL` | unset (off; **A5 auto-on for pipeline runs**) | Feature 1 — deterministic Bloom-level relevel (re-derive a mislabelled CO/TO `bloom_level` from its main verb's canonical level; statements never change). |
| `ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT` | unset (off) | Feature 2 / PRONG C — LLM-assisted grounded analyze/evaluate complement synthesis when the higher-order Bloom share is too thin. |
| `ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MIN_SHARE` | `0.15` | PRONG C satellite — analyze+evaluate+create share floor below which complements are synthesized. |
| `ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MAX` | `8` | PRONG C satellite — hard cap on complement COs added per pass (0 = measure-only). |
| `ED4ALL_OBJECTIVE_LIBRARY_EXEMPLARS` | unset (off) | W4.3 cross-course objective-library EXEMPLARS master gate |
| `ED4ALL_OBJECTIVE_LIBRARY_EXEMPLAR_LIMIT` | `8` | W4.3 top-K cap on surfaced exemplars |
| `ED4ALL_OBJECTIVE_LIBRARY_EXEMPLAR_MIN_OVERLAP` | `0.05` | W4.3 Jaccard floor for an exemplar to be surfaced |
| `ED4ALL_OBJECTIVE_MAX_CHUNKS_PER_OBJECTIVE` | `5` | Fix 1A top-K cap on cited chunks per MERGED objective |
| `ED4ALL_OBJECTIVE_SYNTHESIS_CHECKPOINT` | `on` | Site override for the stage-2 window + cluster resume sidecars (beats `ED4ALL_GENERATION_CHECKPOINT`). |
| `ED4ALL_PLANNING_GATE_RETRIES` | `0` (off) | Graceful course_planning gate-failure retry budget — bounds how many times a nondeterministic TO/CO synthesis re-roll may be attempted before the phase stops blocking the build (per attempt: evict the cluster sidecar, keep the window/CO cache, salted TO re-roll); after exhaustion the phase FAIL-OPENS complete-with-warning (`PLANNING_GATE_RETRIES_EXHAUSTED`) and the workflow continues. |
| `ED4ALL_PLANNING_REROLL_SALT` | unset (runner-managed) | Per-attempt `attempt-N` re-roll salt the gate-retry loop sets (not operator-set); folded into the synthesis system prompt + the cluster sidecar fingerprint so each retry attempt genuinely differs. Sibling: `ED4ALL_PLANNING_REROLL_FEEDBACK` (next row) rides the same set/pop lifecycle. |
| `ED4ALL_PLANNING_REROLL_FEEDBACK` | unset (runner-managed) | Failure-DIRECTED sibling of the re-roll salt (not operator-set): the gate-retry loop serializes the failing critical gates' issues into a compact per-issue remediation digest (objective id + issue code + code→directive line, ≤1200 chars) and the synthesis provider folds it into the cluster/TO-minting system prompt as a delimited REMEDIATION section + into the cluster sidecar fingerprint (a changed digest never reuses a stale cached TO). |
| `ED4ALL_REQUIRE_ARCHIVED_OBJECTIVES` | unset (off) | W2.3 fail-closed for the archive_to_libv2 objectives→objectives.json plumbing. |
| `ED4ALL_PRODUCTION` | `0` | When `1`, enables production-mode FastMCP server settings. |
| `ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE` | unset (off; **A5 auto-on for pipeline runs**) | Gate-side provenance resolution for `block_prose_entailment` — when a rewrite block's cited `semantik:{slug}#anchor` refs resolve to nothing in `source_chunks`, map them through a `{ref -> [chunk_id]}` index (section-level ref → ALL section chunks) to recover the premise set. ADD-only, anti-fabrication (existing refs → existing chunks); bare library calls keep default-off → byte-identical NO_GROUNDING path. |
| `ED4ALL_RESEGMENT_COLLAPSED` | `1` | WS6b collapse re-segmentation gate |
| `ED4ALL_RESEGMENT_SECTIONS_PER_CHAPTER` | `13` | WS6b target sections-per-pseudo-chapter |
| `ED4ALL_ROOT` | auto-detect | Absolute path to the Ed4All project root. |
| `ED4ALL_RUN_ID` | generated | Per-run identifier consumed by every artifact emitter. |
| `ED4ALL_SKIP_ABLATION` | unset | When set, skips the post-training ablation pass. |
| `ED4ALL_STAGE_MODE` | `symlink` | How `stage_semantik_outputs` materialises SemantiK HTML (copy / symlink / hardlink). |
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
| `ED4ALL_TO_SOURCE_GROUNDING` | unset (off; **A5 auto-on for pipeline runs**) | W7.5 opt-in gate for the TERMINAL-objective source-grounding validator |
| `ED4ALL_TO_CHAPTER_ANCHOR` | unset (off) | W3 Defect A master gate — chapter-anchored TO derivation (one SemantiK module → one terminal objective by cited-chunk plurality) instead of bottom-up statement clustering. Default off → bottom-up path unchanged. |
| `ED4ALL_TO_CHAPTER_ANCHOR_REORDER` | on-when-master-on | W3 Defect A §6 satellite — stable book-order CO re-sort (by module order + in-module position) BEFORE the week slice, so ceil-stride weeks are chapter-contiguous even without `ED4ALL_WEEK_TO_GROUPS`. Only the falsey tokens disable it. |
| `ED4ALL_TO_CHAPTER_MIN_MODULES` | `2` | W3 Defect A satellite — module-count floor below which anchor mode degrades to bottom-up (a monolithic single-HTML corpus can't be anchored). |
| `ED4ALL_TO_CHAPTER_MIN_CO_COVERAGE` | `0.80` | W3 Defect A satellite — min fraction of COs that must resolve ≥1 module from their own cited chunks for anchor mode to fire (else degrade to bottom-up). |
| `ED4ALL_TRAINING_CAPTURES_DIR` | `<repo>/training-captures/` | Overrides the legacy decision-capture mirror root (`training-captures/`). |
| `ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM` | unset (off) | W2.1 in-gate CUDA OOM surfaces as a `VALIDATOR_OOM` warning; truthy ⇒ the OOM fails the gate closed. |
| `ED4ALL_CALIB_EXTRA_CORPORA` | unset (off) | W8.3 multi-corpus discovery pointer for the calibration harness |
| `ED4ALL_VRAM_DOCTOR` | unset (off) | VRAM-contention observability gate |
| `ED4ALL_GPU_LIFECYCLE` | `on` (default ON — deviation) | Deterministic GPU-lifecycle LEASE: every model loads, runs, releases the card, hands it to the next stage. Phase-boundary sweep in `workflow_runner.run_workflow` (ollama `keep_alive:0` + torch `empty_cache`, best-effort, off-loop) + SemantiK cascade stage seams (`cascade._gpu_lifecycle_release` via the cross-venv twin). Residency/timing only — never an output byte (default-ON justified). Opt-out `=0` for perf. |
| `ED4ALL_BIG_MEMORY_MIN_MIB` | `49152` (48 GiB) | Total-VRAM threshold above which the `ed4all doctor` `gpu_profile` group treats the box as a big-memory concurrent-serving host and emits ADVISORY warns for each small-box default still on (`ED4ALL_GPU_LIFECYCLE`, `ED4ALL_NLI_EVICT_FOR_CUDA`, `ED4ALL_CLOUD_RATE_LIMIT`, low `ED4ALL_GPU_MAX_USED_MB`); below threshold / unprobeable GPU → silent no-op. |
| `ED4ALL_VLLM_CONTAINER_LIFECYCLE` | unset (off) | Task #10 vLLM per-seat container-lifecycle LEASE (`lib/vllm_container_lifecycle.py`) — `docker start`/`/v1/models`-poll a seat's container on `ensure_serving`, `docker stop` on `release_all` at the workflow-end boundary. Data-driven seat map via `ED4ALL_VLLM_CONTAINERS`; all functions best-effort / never-raise (docker missing / no perms / wedged → off-sentinel). Default OFF → every function a no-op (byte-identical control flow). |
| `ED4ALL_VLLM_CONTAINERS` | unset (`{}`) | Task #10 comma-separated `base_url=container` seat registry consumed by `ED4ALL_VLLM_CONTAINER_LIFECYCLE` (`lib/vllm_container_lifecycle.py::parse_container_registry`), e.g. `http://localhost:8000=vllm-omni,http://localhost:8001=vllm-embed` — the single source of truth for the seat map (a new vLLM seat is a registry entry, never a subclass). Base URLs normalized (trailing `/` stripped); a malformed token is SKIPPED with a one-time warning (partly-garbage registry still yields its valid pairs). No-op when the lifecycle flag is off. |
| `ED4ALL_SEAT_SCHEDULE` | unset (off) | Declarative per-phase vLLM SEAT SCHEDULE enforcement (`lib/vllm_container_lifecycle.py::resolve_seat_schedule_mode`; wired in `MCP/core/workflow_runner.py::run_workflow`) — at each phase boundary reconciles resident seats to the phase's `seats:` annotation in `config/workflows.yaml` (stop unneeded / start newly-needed via a two-phase health check: LIVENESS poll of `/v1/models` up to the `ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS` ceiling, then a BOUNDED coherence check of `ED4ALL_SEAT_COHERENCE_ATTEMPTS` tries — coherence is NOT ceiling-bound, a mode-collapse is caught in seconds). SELF-HEAL: a warm seat that comes up live but INCOHERENT (mode-collapse) is AUTOMATICALLY cold-recreated IMMEDIATELY (`docker rm -f` + relaunch via `ED4ALL_SEAT_LAUNCH_SPECS`) and re-checked once (emits a `seat_cold_recreate` DecisionCapture) — self-heals in ~30-45s, not ~20 min. Only a seat that still cannot come up coherently — or has no launch spec to self-heal — raises `SeatScheduleProbeError` (loud, anti mode-collapse); everything else best-effort. Default OFF → no-op. Full detail: `docs/operations/behavior-flags.md`. |
| `ED4ALL_SEAT_BASE_URLS` | unset (`{}`) | Comma-separated `seat_name=base_url` registry (`lib/vllm_container_lifecycle.py::parse_seat_registry`) mapping the LOGICAL seat names in the `config/workflows.yaml` `seats:` annotations to vLLM base URLs; resolves onward to a container via `ED4ALL_VLLM_CONTAINERS`. Fail-soft (bad token skipped with one-time warn). No-op when `ED4ALL_SEAT_SCHEDULE` is off. |
| `ED4ALL_SEAT_LAUNCH_SPECS` | unset (`{}`) | Data-driven per-seat launch-spec registry (`seat_name=<launch script path or command>`; `lib/vllm_container_lifecycle.py::parse_seat_launch_specs`) that lets `ED4ALL_SEAT_SCHEDULE` COLD-RECREATE a seat (`docker rm -f` + relaunch) rather than only warm `docker start` one — the self-heal path for a mode-collapsed seat. Prefer an absolute launch-SCRIPT path per seat (`spark-super=/opt/seats/launch-super.sh`); tokens split on `;` (then `,`), spec keeps its `=`. Fail-soft (bad token skipped with one-time warn). A seat with no spec cannot self-heal (that phase fails loudly). No-op when `ED4ALL_SEAT_SCHEDULE` is off. |
| `ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS` | `1200` (20 min) | Env-tunable CEILING (seconds) on the seat **LIVENESS** load-wait poll (`lib/vllm_container_lifecycle.py::resolve_seat_load_timeout`) for `start_seat` / `start_seat_coherent` / `recreate_seat`. Bounds ONLY the wait for `/v1/models` to answer 200; a CEILING not a sleep (returns the instant the seat is live). Does NOT bound coherence — that is the separate `ED4ALL_SEAT_COHERENCE_ATTEMPTS` gate. 1200s covers a 120B NVFP4 cold load (~8-10 min, old 600s too tight). Parse-with-fallback: positive int/float wins; garbage / non-positive → 1200. No-op when `ED4ALL_SEAT_SCHEDULE` is off. |
| `ED4ALL_SEAT_COHERENCE_ATTEMPTS` | `3` | Env-tunable COUNT of content-coherence probe attempts once a seat is LIVE (`lib/vllm_container_lifecycle.py::resolve_seat_coherence_attempts`; `_coherence_check` in `start_seat_coherent`). Coherence is BOUNDED, not ceiling-bound — a mode-collapsed seat is live-but-incoherent and caught in seconds: `coherence_probe` is retried ≤ this many times (~8s apart), returning on first pass; if still incoherent the schedule cold-recreates IMMEDIATELY (self-heal ~30-45s, not the 1200s liveness ceiling). Parse-with-fallback: positive int wins; garbage / non-positive → 3. No-op when `ED4ALL_SEAT_SCHEDULE` is off. |
| `ED4ALL_LLM_TTFT_METER` | unset (off) | Task #10 time-to-first-token metering for the OpenAI-compatible content-generation client (`Trainforge/generators/_openai_compatible_client.py`, env `ENV_TTFT_METER`). Truthy → the client STREAMS the completion and records `ttft_ms` on its usage row (surfaced as p50/p95 by `BuildCostAggregator`, additive — no schema bump); any streaming-specific failure falls back to the proven non-streaming path with no ttft recorded (metering never fails a real call). Default OFF → byte-identical to the legacy non-streaming path (no `stream` key on the wire). Parse-with-fallback truthy `1`/`true`/`yes`/`on`. |
| `ED4ALL_PLANNER_INTERLEAVE` | unset (off) | W6.2 TRUE cross-CO practice interleaving for the dynamic block planner |
| `ED4ALL_PLANNER_FAR_TRANSFER` | unset (off) | W6.3 per-TO novel-context far-transfer floor for the dynamic block planner |
| `ED4ALL_PLANNER_DUAL_CODING` | unset (off) | W6.5 per-CO dual-coding floor for the dynamic block planner |
| `ED4ALL_PLANNER_INTEGRATION` | unset (off) | Integration floor — turns a weak-membership ("oddball") CO into cross-CO integrated practice for the dynamic block planner |
| `ED4ALL_PLANNER_INTEGRATION_MAX_OVERLAP` | `0.10` | Integration-floor satellite — token-overlap floor below which a CO is weak-membership (aligned with the CO→TO backlink token floor) |
| `ED4ALL_PLANNER_INTEGRATION_MAX_PER_WEEK` | `2` | Integration-floor satellite — hard cap on integrated-practice injections per week (weakest-first) |
| `ED4ALL_PLANNER_CROSS_WEEK_RETRIEVAL` | unset (off) | W6.1 cross-week cumulative retrieval for the dynamic block planner (needs prior-week context). |
| `ED4ALL_FAQ_PAGE` | unset (off; **A5 auto-on for pipeline runs**) | W6.4 deterministic per-week "FAQ" page gate |
| `ED4ALL_FAQ_MAX_PER_PAGE` | `12` | W6.4 satellite — hard cap on FAQ cards emitted per week page |
| `ED4ALL_OBJECTIVE_OBSERVABLE_VERB` | unset (off) | W1.1 non-observable (fuzzy) main-verb scan on the objective statement (legacy no-ABCD path). |
| `ED4ALL_OBJECTIVE_INFER_BLOOM` | unset (off) | W1.4 infer a null LO bloom_level from its declared ABCD behavior.verb instead of skipping. |
| `ED4ALL_CHUNK_COVERAGE_FLOOR` | unset (off) | W1.2 import-coverage gate floor on the existing `chunkset_manifest` gate |
| `ED4ALL_MIN_CHUNKS` | unset (off) | W1.2 thin-chunkset floor on the `chunkset_manifest` gate (CHUNKSET_TOO_THIN warning). |
| `ED4ALL_CHUNK_HEALTH_GATE` | unset (off) | Opt-in master switch for the pre-synthesis `chunk_health` gate (`lib/validators/chunk_health.py::ChunkHealthValidator`) at `textbook_to_course::objective_extraction` — audits the emitted chunkset + `textbook_structure.json` for synthesis-poisoning defects (phantom-chapter/resegment/section-explosion/instructional-starvation/empty-chunk → critical/block; OCR mojibake/furniture/reading-order → warning) BEFORE `course_planning`. Default OFF → skip-with-pass. C2 counts WORKED examples (`example`/`worked_example` chunks with a `Solution`/`Try It`/`Step N` marker) as instructional so a worked-example workbook is not false-starved; C10 evaluates only non-apparatus PROSE chunks. Thresholds env-overridable (`ED4ALL_CHUNK_HEALTH_{CHAPTER_RATIO_FAIL,CHAPTER_RATIO_WARN,SECTIONS_PER_CHAPTER,INSTRUCTIONAL_FAIL,INSTRUCTIONAL_WARN,APPARATUS_FAIL,APPARATUS_WARN,MIN_CHUNKS,TINY_WORDS,MEGA_WORDS,MOJIBAKE_RATE,NUMDUMP_RATE,WORKED_EXAMPLE_INSTRUCTIONAL}`; parse-with-fallback). |
| `ED4ALL_KEYTERM_DEF_QUALITY` | unset (off; **A5 auto-on for pipeline runs**) | W1.5 glossary definition-quality gate (circular / too-long / not-distinct / missing-math-condition) — critical gate (flip-wave-2) emitting warning-severity issues; audits key-terms vocab cards AND inline `<div class="definition-box">` blocks parsed out of concept/explanation HTML. |
| `ED4ALL_PAGE_EST_MINUTES` | unset (off) | W1.6 per-page estimated learning-time emit gate |
| `ED4ALL_PAGE_WPM` | `200` | W1.6 satellite — reading-speed divisor for the `ED4ALL_PAGE_EST_MINUTES` estimate |
| `ED4ALL_PAGE_INTERACTION_MINUTES` | `1.0` | W1.6 satellite — per-interaction minute cost for the `ED4ALL_PAGE_EST_MINUTES` estimate |
| `ED4ALL_GROUNDEDNESS_COMPUTATIONAL` | unset (off) | W1.8 numeric-grounding of NLI-exempt computational sentences on the grounded-answer path |
| `ED4ALL_EMBED_OVERFLOW_GUARD` | unset (off; **A5 auto-on for pipeline runs**) | W1b.2 embedding serving-window overflow guard |
| `ED4ALL_EMBED_OVERFLOW_SPLIT` | unset (off) | W1b.2 satellite of `ED4ALL_EMBED_OVERFLOW_GUARD` |
| `ED4ALL_EMBED_MAX_SEQ_TOKENS` | `512` | W1b.2 serving-window token ceiling driving the embed model max_seq_length pin + overflow. |
| `ED4ALL_CHUNK_CODE_SPLIT` | unset (off) | W1b.3 code-fence-aware chunk splitting |
| `ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR` | `0` (off) | W1b.4 runt-fragment merge floor |
| `ED4ALL_CHUNK_SECTION_HARD_BREAK` | unset (off) | Forces a chunk break at textbook SECTION-heading boundaries (anti cross-section fusion). Size-guarded successor: `ED4ALL_CHUNK_SUBSECTION_BREAK`. |
| `ED4ALL_CHUNK_SUBSECTION_BREAK` | unset (off) | Size-guarded successor to `ED4ALL_CHUNK_SECTION_HARD_BREAK` — breaks at a genuine content sub-section heading (incl. non-numbered h3/h4) only once the buffer clears `ED4ALL_CHUNK_SUBSECTION_MIN_WORDS`, closing the CO-depth gap without a runt cascade. |
| `ED4ALL_CHUNK_SUBSECTION_MIN_WORDS` | `250` | W-satellite — accumulation floor (words) a buffer must reach before a sub-section heading may flush it under `ED4ALL_CHUNK_SUBSECTION_BREAK`. |
| `ED4ALL_CHUNK_LO_HEURISTIC` | unset (off) | W1b.5 heuristic LO-backfill arm |
| `ED4ALL_CROSS_COURSE_DEDUP` | unset (off) | W1b.1 cross-course boilerplate dedup for multi-course batch imports |
| `ED4ALL_WITH_ASSESSMENT_SFT` | unset (off) | SFT program S1 — appends deterministic open-book assessment→SFT pairs to `instruction_pairs.jsonl` (OR'd with the `with_assessment_sft` kwarg). |
| `ED4ALL_WITH_GRAPH_SFT` | unset (off) | SFT program S5 — appends deterministic open-book concept-graph→SFT pairs from the holdout-reduced consensus-filtered graph (OR'd with `with_graph_sft`). |
| `ED4ALL_ASSESSMENT_APPLY_ARM` | unset (off) | A1 LLM apply-arm — routes apply word-problems + misconception prose through a roster-license-filtered local seat behind a mandatory sympy→groundedness→trivote verify chain. |
| `ED4ALL_ASSESSMENT_APPLY_ARM_MAX` | `4` | Satellite of `ED4ALL_ASSESSMENT_APPLY_ARM` — bounded per-quiz LLM draft budget (garbage / non-positive → 4). |
| `ED4ALL_ASSESSMENT_ITEM_TRIVOTE` | unset (off) | A2 Bloom-trivote seam in the item-writing linter (asserted vs verb-ontology + injected zero-shot voter; `ITEM_BLOOM_TRIVOTE_UNSUPPORTED` warning). |
| `ED4ALL_ASSESSMENT_NUMERIC_RECOVERY` | unset (off) | A3 apparatus-guard numeric-recovery — re-admits Solution/Check/Step-N regions for the numeric-FIB extractor ONLY (still sympy-verified); guard intact elsewhere. |
| `ED4ALL_ASSESSMENT_APPARATUS_STRICT` | unset (off; **A5 auto-on for pipeline runs**) | Widened GENERIC apparatus markers on the assessment harvest paths — colon-less `Solution`/`Check`, all-caps `HOW TO` banners, leading `Figure|Table|Example N.N` captions, and glyph alt-text — closing the OCR'd-scan leak the legacy colon-anchored set misses. |
| `ED4ALL_ASSESSMENT_CLEAN_PROSE` | unset (off; **A5 auto-on for pipeline runs**) | Prose-only MINING VIEW for `assessment_synthesis` — re-derives each chunk's text at mining time by masking out every span that came from a SemantiK non-prose region (`data-semantik-block-role` ∈ figure/caption/apparatus/exercise/solution/try_it/math/…) or a generic structural carrier (`<table>`, `<figcaption>`, `img/@alt`), so figure alt-text, worked solutions and flattened tables never reach the generator as distractors or correct answers. Chunks on disk are untouched → `semantik_chunks_sha256` stays stable and no upstream phase re-runs. Complements `ED4ALL_ASSESSMENT_APPARATUS_STRICT` (that filters strings the generator already picked; this removes them from the pool). See `lib/assessment/source_prose.py`. |
| `ED4ALL_ASSESSMENT_ITEM_BANK` | unset (off; **A5 auto-on for pipeline runs**) | Emits the QTI 1.2 `<objectbank>` question-LIBRARY sidecar `06_assessments/item_bank.xml` (every item + queryable `ed4all_*` selection `qtimetadata`), recorded under its own `item_bank` manifest key so the packager never ships the bank as an exam. |
| `ED4ALL_ASSESSMENT_ITEMS_PER_OBJECTIVE` | `1` | Expansive item-bank scaling — multiplies the per-objective item floor in `assessment_synthesis` (`max(question_count, n_objectives x N)`) so a course emits N items per objective instead of the 1-per-objective exam minimum. Folded into the quiz-unit fingerprint. Garbage / non-positive falls back to `1` (never 0 — that would collapse the archival-gate coverage floor). |
| `ED4ALL_TRAINFORGE_ASSESSMENT_HARVEST` | unset (off) | `trainforge_assessment` HARVESTS the already-emitted QTI out of the packaged IMSCC and re-keys each item onto the IMSCC chunks whose `learning_outcome_refs` carry its `objective_id`, instead of running a SECOND generation pass over the same content. Corpus + graph build unchanged. Fails LOUD on an empty harvest (never a silent fallback). |
| `ED4ALL_DISCUSSION_GROUNDING_NLI` | unset (off) | A5 text-grounded NLI arm for `discussion_assignment_grounded` — flips refs-only Jaccard to authoritative text-entailment where runnable (only tightens; legacy fallback). |
| `ED4ALL_EVAL_COMPOSER_PROVIDER` | unset (absent) | E7a diagnostic-composer arm — composes eval answers on a stronger local seat while retrieval + gates stay byte-identical (separates retrieval vs composition failures). |
| `ED4ALL_EVAL_COMPOSER_MODEL` | per-provider | Satellite of `ED4ALL_EVAL_COMPOSER_PROVIDER` — model-ID override for the diagnostic composer seat. |

The `LLM_*` env vars (`LLM_MODE`, `LLM_PROVIDER`, `LLM_MODEL`) are CLI runtime knobs documented in § Quick Start above. Other `ED4ALL_*` vars kept out of this index — the GUI server vars (`ED4ALL_GUI_HOST` / `_PORT` / `_LEARNER` / `_MODE` / `_TOKEN`), the test-only fixture/gating overrides (incl. `ED4ALL_NLI_VALIDATORS_FACTORY`, the picklable-factory dotted-path seam the `ED4ALL_NLI_MICROBATCH_VALIDATORS` process pool uses to build each worker's NLI — default the production singleton loader), and the three rewrite-tier `ED4ALL_REWRITE_*` + the W10 `ED4ALL_ASSESSMENT_PROSE_PROVIDER` flags owned by subsystem files — are enumerated verbatim in [`docs/operations/behavior-flags.md`](docs/operations/behavior-flags.md).

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
- `lib/licensing/teacher_roster.py` — machine-readable SFT teacher-license roster + fail-closed export/ingest/Nemotron-pin guards (`assert_export_licenses` / `assert_checkpoint_license` / `assert_nemotron_pin` / `stamp_pair_license` / `provider_verdict_roster`); canonical prose posture in `docs/LICENSING.md § SFT teacher roster`.

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
- **Universal block-label ontology** (SemantiK structure labels — 16 DocLayNet-mapped structural kinds + genre-role profiles + publisher lexicons, all data under `schemas/taxonomies/block_kinds.json` / `block_relations.json` / `genre_profile_*.json` / `*_lexicon.json`): `docs/architecture/block-ontology.md`

---

## Training Pipeline

SLM training is a post-import LibV2 stage, not a step in `Trainforge/process_course.py`. Top-level command: `ed4all run trainforge_train --course-name <slug> --base-model <name>` (`--course-name` is the CLI flag; `course_code` is only the handler-side param alias declared in `config/workflows.yaml::training`'s `inputs_from` block). `--base-model` populates `workflow_params.base_model` — the route that phase reads — and is validated at parse time against `Trainforge/training/base_models.py::BaseModelRegistry`, so an unknown name exits 2 with the supported list instead of silently training another base. Precedence: `--base-model` > `ED4ALL_CAMPAIGN_BASE_MODEL` > the registry default (`nemotron3-nano-30b`). The same flag pins the base for the in-build `--with-training` tail, and re-pins it on `--resume`. Full deep-dive (base-model registry, provider config, 5×3 eval matrix, 7-hash provenance, promotion workflow, decision-capture contract): `Trainforge/CLAUDE.md § Training Pipeline`.

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
