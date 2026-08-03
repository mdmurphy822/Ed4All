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
ed4all run textbook-to-course --corpus textbook.pdf --course-name <course-name>
ed4all run textbook-to-course --corpus ./pdfs/ --course-name <course-name> --weeks 16
ed4all run rag_training --corpus course.imscc --course-name <course-name> --mode api
ed4all run textbook-to-course --corpus x.pdf --course-name T --dry-run   # plan only
ed4all run textbook-to-course --resume <RUN_ID>                            # resume

# ed4all stop: graceful "checkpoint on command". Drops a stop sentinel; the run
# finishes its in-flight unit, checkpoints it, and pauses (exit code 3) —
# worst-case loss is one in-flight LLM call. Resume with a PLAIN --resume (never
# --force after a stop — force clears the resume sidecars). SIGTERM/Ctrl-C to a
# live `ed4all run` is the same request (signal again to hard-kill). Full
# runbook: docs/operations/pipeline-invocation.md § 7.
ed4all stop <RUN_ID>                 # pause ONE run at its next unit boundary
ed4all stop --all                   # global STOP_ALL — pause + BLOCK all runs
ed4all stop --clear-all             # remove STOP_ALL (operator-owned)

# --stop-after <phase>: halt cleanly AFTER the named phase, skipping all
# downstream. Canonical "retrieval-ready course, no training synthesis"
# slice stops after imscc_chunking. Phase name validated (unknown ->
# error). Full semantics: docs/operations/pipeline-invocation.md.
ed4all run textbook-to-course --corpus pdfs/ --course-name <course-name> \
  --skip-training --stop-after imscc_chunking

# --with-training: OPT IN to the in-build training tail of
# textbook_to_course (training -> post_training_validation -> evaluation,
# between vector_indexing and finalization). OFF by default — a training
# run is multi-hour and owns the whole card, so it never attaches to a
# build implicitly. --skip-training WINS when both are passed. Also valid
# on --resume (patches the persisted params before the resumed phases
# run). Training an ALREADY-archived course without rebuilding stays
# `ed4all run trainforge_train --course-name <slug>`.
ed4all run textbook-to-course --corpus pdfs/ --course-name <course-name> \
  --with-training

# Hosted large-model build profile (--provider nvidia = the vendor
# endpoint-registry key; SETUP — nothing dispatches to the cloud seat by
# default; gated on a later RUN discussion). Full routing detail (YAML
# redirect, seat pins, licensing caveat): see
# docs/operations/pipeline-invocation.md § 3.1. Run --dry-run first.
export COURSEFORGE_TWO_PASS=true
ed4all run textbook-to-course --provider nvidia --course-name <course-name> \
  --corpus slice.pdf --skip-conversion --skip-training \
  --stop-after imscc_chunking --dry-run   # preflight: resolve+assert, NO dispatch

# --reuse-objectives: pin a prior objectives JSON instead of re-dispatching
# the course-outliner (kills re-run LLM-nondeterminism drift). Accepts both
# the Courseforge + LibV2 archive shapes; normalized on disk. Also valid on
# --resume (patches the persisted params before the resumed course_planning
# runs). See docs/operations/pipeline-invocation.md § 3.
ed4all run textbook-to-course --corpus pdfs/ --course-name <course-name> \
  --reuse-objectives Courseforge/exports/<project-export>/01_learning_objectives/synthesized_objectives.json

# ed4all objectives restructure: DETERMINISTICALLY (no LLM) rebuild an existing
# objectives doc — lexical dedup (E), vacuity annotate/drop (B),
# chapter-anchored TO re-derivation (A), sub-objective quality (D) — in minutes
# instead of a 7B re-roll. Writes <input>.restructured.json + restructure_report.json;
# feed the output straight back into --reuse-objectives (it round-trips that shape).
ed4all objectives restructure \
  Courseforge/exports/<project-export>/01_learning_objectives/synthesized_objectives.json \
  --course-name <course-name> --drop-vacuous

# --reuse-conversion: reuse a prior SemantiK conversion (skips the
# model-nondeterministic v2 cascade when prior artifacts exist). Mirrors
# ED4ALL_REUSE_CONVERSION (flag wins). See SemantiK/CLAUDE.md §3.3a.
ed4all run textbook-to-course --corpus pdfs/ --course-name <course-name> \
  --reuse-conversion

# --instruction-variants-per-chunk N: how many INSTRUCTION units the
# training_synthesis phase synthesizes per chunk. Default 1 (unset = the
# key is not even recorded, so behavior is byte-identical). Raise to 2+
# when reject-mined DPO negatives are wanted — mined yield is
# STRUCTURALLY ZERO at 1, since a chunk holding one instruction unit can
# never hold both an accepted anchor and a rejected unit to pair it
# against. Routed via workflow_params ->
# config/workflows.yaml::training_synthesis.inputs_from -> run_synthesis.
ed4all run textbook-to-course --corpus pdfs/ --course-name <course-name> \
  --instruction-variants-per-chunk 2

# Phase 5: stage-by-stage Courseforge two-pass subcommands — re-run a
# single tier against an existing export (upstream phases pre-populate
# from disk). See Courseforge/CLAUDE.md "Operator stage subcommands".
export COURSEFORGE_TWO_PASS=true
ed4all run courseforge-outline --course-name <course-name>              # outline tier only
ed4all run courseforge-validate --course-name <course-name>             # validators only
ed4all run courseforge-rewrite --course-name <course-name> \
  --blocks assessment_item,objective                                # per-block-TYPE rewrite
# I4 stage 2 — two ADDITIVE finer-grained rewrite-eviction scopes (both stack
# with --blocks; the rewrite tier consumes them). --block-ids: exact
# block-instance IDs (shape {page_id}#{block_type}_{slug}_{idx}). --pages: an
# exact page_id (e.g. week_01_content_02) OR a module prefix (e.g. week_01) for a
# whole week/module. All three unset => byte-identical failure-driven reuse; an
# unknown id / unmatched page fails the rewrite phase LOUDLY (never a silent no-op).
ed4all run courseforge-rewrite --course-name <course-name> \
  --block-ids 'week_01_content_02#example_derivative_03' \
  --pages week_01                                                    # instance + page/module scope
ed4all run courseforge --course-name <course-name> --force               # full two-pass slice

# --license-note / --attribution: optional corpus-provenance declarations
# recorded on the LibV2 course_manifest (license.note / attribution.statement,
# mirrored into the emitted NOTICE). See docs/operations/library-versioning.md
# + docs/operations/demo-course.md.
ed4all run textbook-to-course --corpus pdfs/ --course-name <course-name> \
  --license-note 'CC-BY-4.0' --attribution 'Provided by the source publisher'

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
# into runtime/state/bloom_labels/labels.jsonl — the corpus behind the re-founded
# bloom_classifier_disagreement voter 1 (ED4ALL_BLOOM_TRIVOTE). --dry-run counts
# only. Also runs post-build under ED4ALL_HARVEST_BLOOM_LABELS.
ed4all harvest-bloom-labels ./Courseforge/exports/<project-export> --dry-run

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

Per-subsystem layout lives in each subsystem's `CLAUDE.md` (see § Individual Project Guides).

Placement rules at every level — which zone a new file/dir belongs to, the
closed top-level allowlist, the `scripts/` and `docs/` taxonomies, the
subsystem *interior* schema and its per-directory flat-file caps (§7), and what
is deliberately never reorganized — live in
[`docs/architecture/repo-organization.md`](docs/architecture/repo-organization.md)
(enforced by `ci/layout_guard.py`, five checks).

Two rules bite most often when adding a file: a new flat `lib/*.py` is a
violation (new cross-cutting code goes in a `lib/<topic>/` subpackage), and a
capped interior directory cannot grow a new loose code file — check
`ci/layout_allowlist.txt` for its `flatcap:` line. Both are ratchets: the
numbers only ever go down, and raising one is an exception to justify in the
same PR.

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

## Orchestrator protocol

`config/workflows.yaml` is the source of truth for phase order, dependencies,
concurrency, and validation gates. `config/agents.yaml` owns the pipeline-agent
registry, while `MCP/core/executor.py` owns dispatch overrides that cannot be
inferred from YAML alone.

Agents execute the task state supplied by the active harness. Do not create a
second planning ledger or require a harness-specific todo API. Respect the
repository-wide ten-task batch ceiling, wait for a batch before starting the
next one, and keep one writer per file. A workflow advances only through its
configured validation gates; stop at the first blocking failure and fix the
artifact rather than weakening the gate.

---

## Decision Capture

### CRITICAL REQUIREMENT

**ALL LLM decisions MUST be logged** to `runtime/training-captures/` in JSONL format.

### Required Fields

Every decision event MUST include:
- `decision_type`: Category of decision (e.g., `content_selection`, `question_generation`, `form_data_backfill_session`, `family_completeness_decision`). Canonical enum: `schemas/events/decision_event.schema.json`.
- `decision`: The actual choice made
- `rationale`: Why this decision was made (**minimum 20 characters**)

### Using Decision Capture

Helper: `lib/decision_capture.py::DecisionCapture` — instantiate with `course_code`, `phase`, `tool`, then call `log_decision(decision_type, decision, rationale, alternatives_considered=[...])`. Output lands under `runtime/training-captures/<tool>/<COURSE_CODE>/phase_<phase>/decisions_*.jsonl`.

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

The `@mcp.tool()`-decorated surface is enumerated in `MCP/tools/` — read those
modules for the current tool list and signatures. Documented below is only what
the source does NOT make obvious: the sandbox tiers, the registry-only tools,
and the phase-name dispatch override.

**Core file tools** (`MCP/server.py` — the `@mcp.tool()` defs and their
`ToolCapability` registrations both live there; there is no
`MCP/tools/file_tools.py`) — `list_directory` / `read_file` / `file_info` run
in a READ_ONLY sandbox rooted at `ALLOWED_FILE_ROOT`; `write_file` in a
RESTRICTED sandbox whose writable `allowed_paths` are confined to `runtime/`.

**SemantiK tools** — see `SemantiK/CLAUDE.md` (PDF→accessible-HTML conversion; emits the Source-Provenance `data-semantik-*` / `semantik:{slug}#{block_id}` contract).

**Courseforge tools** — see `Courseforge/CLAUDE.md § MCP Tools` (includes Metadata Output contract: `data-cf-*` + JSON-LD).

**Orchestrator tools** (`MCP/tools/orchestrator_tools.py`) — workflow lifecycle
+ agent dispatch + batch locking.

**GUI tools** (`MCP/tools/gui_tools.py`) — the Claude-interaction surface for the
Control-Plane GUI. All operate on the shared `runtime/state/gui/` store, so a Claude
session and the GUI stay in sync. Full detail:
`gui/README.md § Claude Code integration`.

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

`assistant_campaign_launch_training` is the one tool with a hardware
side effect beyond the harness: it `docker stop`s every registered vLLM seat and
verifies the card is free before launching `ed4all run trainforge_train`
detached with a fixed argv.

**Trainforge tools** — see `Trainforge/CLAUDE.md § MCP Tools`.

### Pipeline Tools

`archive_to_libv2` emits a top-level `chunker_version` field in
`course_manifest.json` (resolved via `Trainforge.chunker.CHUNKER_SCHEMA_VERSION`)
so LibV2 audits know which chunker shipped the corpus.

**Pipeline-internal registry-only tools** (wired into `MCP/tools/pipeline_tools.py::_build_tool_registry` for workflow-phase dispatch; intentionally **not** decorated with `@mcp.tool()` — not reachable from external MCP clients):

| Tool | Phase | Purpose |
|------|-------|---------|
| `build_source_module_map` | `source_mapping` | TF-IDF-driven router that maps SemantiK source blocks to Courseforge module pages. Output: `source_module_map.json`. |
| `extract_textbook_structure` | `objective_extraction` | Runs `SemanticStructureExtractor` over every staged SemantiK HTML file and merges per-file chapter/section hierarchies into a single `textbook_structure.json`. |
| `plan_course_structure` | `course_planning` | Synthesizes canonical `TO-NN` / `CO-NN` learning objectives from the textbook structure and publishes `synthesized_objectives.json`. |

**Phase-name dispatch override** (`MCP/core/executor.py::_PHASE_TOOL_MAPPING`): nine phases route by phase name, not agent name — `content_generation_outline` → `run_content_generation_outline`; `inter_tier_validation` → `run_inter_tier_validation`; `content_generation_rewrite` → `run_content_generation_rewrite`; `post_rewrite_validation` → `run_post_rewrite_validation`; `imscc_chunking` → `run_imscc_chunking`; `assessment_synthesis` → `run_assessment_synthesis`; `heading_judge` → `run_heading_judge` (the SEMANTIK_HEADING_JUDGE post-conversion Super heading-level judge — default ON, explicit falsey token opts out; skip-with-pass when explicitly off / born-digital, per-chapter fail-open copy-back); `training` → `run_training` (wraps `Trainforge.train_course` — the standalone `trainforge_train` workflow and the opt-in `textbook_to_course` tail share this one handler); `evaluation` → `run_evaluation` (held-out harness + grounded-answer arms, one verdict derived by calling `EvalGatingValidator` rather than re-implementing its thresholds). `run_training` / `run_evaluation` additionally sit in a deterministic-tool set keyed on the resolved tool name, so they execute in-process even under `ED4ALL_AGENT_DISPATCH` (the subagent fork happens before the registry lookup and cannot produce an adapter). Validator-only phases declare `agents: []` in `config/workflows.yaml`; `workflow_runner._create_phase_tasks` synthesizes a virtual `phase-handler` task only when the phase appears in this map. The mapping cannot be inferred from YAML.

**Analysis tools** (`MCP/tools/analysis_tools.py`) — training-capture analysis,
quality distribution, export-filter preview.

---

## Shared State

### GENERATION_PROGRESS.md

Location: `runtime/state/GENERATION_PROGRESS.md`

Central progress tracking file:
- Active workflows table
- Component status tables
- Batch locks table
- Error log

### File-Based IPC

Use `StatusTracker` for multi-terminal coordination:
```python
from MCP.ipc.status_tracker import StatusTracker

tracker = StatusTracker()  # defaults to runtime/state/status/
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
       Outputs: semantik_chunks_path + semantik_chunks_sha256, plus a
       per-file `source_html_parse_outcomes` histogram. The staged-HTML
       parse runs across a `spawn` process pool sized by
       `ED4ALL_HTML_PARSE_WORKERS` (0/1 = the byte-identical serial
       path); pooled and serial emit are identical by construction.
       Files that parse to nothing are recorded under named
       `source_coverage.drop_reasons` keys, never silently dropped.
       Gated on chunkset_manifest + chunk_wcag_status.

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
       Reads `imscc_chunks/chunks.jsonl` (via
       `lib/libv2_storage.py::resolve_imscc_chunks_path`; legacy
       `corpus/chunks.jsonl` is a fallback only) plus `objectives.json`;
       writes `training_specs/instruction_pairs.jsonl` +
       `preference_pairs.jsonl`.
       Runs under the `staged-v4` synthesis contract —
       `workflow_runner` setdefaults `TRAINFORGE_STAGED_SYNTHESIS_V4=true`
       and passes no CLI selector. `--synthesis-contract` is a
       `python -m Trainforge.synthesize_training` flag, NOT a pipeline
       route. `micro-v1` is selectable but cannot complete a production
       run today (its Stage A needs an `action_object` the production
       objective loader strips) — see
       `Trainforge/CLAUDE.md § Training-pair synthesis`.
       The phase routes only `course_code` / `provider` / `seed` /
       `required_training` / `instruction_variants_per_chunk` + the four
       deterministic generators. The assessment-SFT and graph-SFT
       programs are reachable ONLY via `ED4ALL_WITH_ASSESSMENT_SFT` /
       `ED4ALL_WITH_GRAPH_SFT`; `include_dpo_from_misconceptions` has no
       env or `inputs_from` route at all and is CLI-only.
       Emits a per-pair resume sidecar at
       `training_specs/.synthesis_pairs_checkpoint.jsonl` (opt out via
       `--no-checkpoint`).
       NOTE: preference-pair ELIGIBILITY is not DPO ADMISSIBILITY. The
       trainer's default `dpo_preference_filter=editorial_or_misconception`
       admits only pairs carrying `misconception_id` or a
       misconception/mined `source`; a chunkset with no structured
       `misconceptions[]` emits all-`rule_synthesized` pairs, which count
       zero against `min_dpo_pairs` and fail the run closed.

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

The agent roster and its capabilities live in `config/agents.yaml`; per-agent
tool routing lives in `MCP/core/executor.py::AGENT_TOOL_MAPPING`. Only the
entries that those two files do NOT make self-evident are documented here.

| Agent | What the config doesn't tell you |
|-------|----------------------------------|
| `semantik-converter` | Drives the SemantiK v2 cascade for the `semantik_conversion` phase (PDF → accessible HTML + source provenance). The legacy agent name still resolves as an `AGENT_TOOL_MAPPING` dispatch alias (read-compat only). |
| `semantik-chunker` | Emit `LibV2/courses/<slug>/semantik_chunks/chunks.jsonl` from staged SemantiK HTML via `Trainforge.chunker.chunk_content`; deterministic transformation (no LLM dispatch). Backed by `_run_dart_chunking` registered in `MCP/tools/pipeline_tools.py::_build_tool_registry` (the registry key keeps its legacy DART name for checkpoint/read-compat). Canonical agent name for both the `chunking` and `imscc_chunking` phases in `config/workflows.yaml`; the legacy agent name survives only as an `AGENT_TOOL_MAPPING` dispatch alias (read-compat). | <!-- legacy-token: allow -->
| `pedagogy-graph-builder` | Builds the pedagogy/concept graph backing the `concept_extraction` phase; routes to `run_concept_extraction` via `AGENT_TOOL_MAPPING`. Declared in `config/workflows.yaml` only — there is **no** `config/agents.yaml` entry. |
| `rag-indexer` | Builds vector embeddings & index (routes to `run_vector_indexing`); fails closed without an embedding backend. |

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

**Exception — the pooled staged-HTML parse.** When `ED4ALL_HTML_PARSE_WORKERS`
resolves above 1, the `chunking` phase parses across a process pool, so a stop
request drains roughly `3 × worker_count` in-flight files rather than one unit.
Those files are re-parsed on resume (parsing is deterministic and cheap — no
LLM call is lost), but the "worst-case loss is one unit" phrasing above does
not apply to that phase. `ED4ALL_HTML_PARSE_WORKERS=1` restores single-unit
granularity on the byte-identical serial path.

### Poison Pill Detection

Stops a batch when the same error pattern repeats:
- Default threshold: 3 same-pattern failures within 5 minutes
- Prevents runaway batch failures from consuming resources

### Phase Checkpointing

Each phase completion creates a checkpoint in `runtime/state/runs/{run_id}/checkpoints/`:
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
| `course_generation` | 33 | 33 | 66 |
| `rag_training` | 4 | 3 | 7 |
| `textbook_to_course` | 64 | 79 | 143 |
| `trainforge_train` | 2 | 0 | 2 |
| **Total** | **103** | **115** | **218** |

The table above is the current authoritative count. Per-wave landing history is
operator-local release evidence rather than part of the public documentation.

---

## Opt-In Behavior Flags

Environment-variable toggles gate opt-in strict / stable-ID / provenance / experimental-rule-graph behavior. All default off to preserve backward compatibility with legacy corpora. Full rationale per flag lives in the owning prefix's reference doc (named below).

Per-flag rows live in per-subsystem reference docs under `docs/operations/` (one owner per prefix), NOT in the CLAUDE.md files — see the table below for which doc owns which prefix. Counting convention for the **subsystem** rows: **distinct flags, multi-flag rows expanded** — a single table row that documents several env vars (e.g. SemantiK's `SEMANTIK_MODEL_DIR / _CACHE_DIR / _DATA_DIR / _CONFIG_DIR` row = 4 flags, Courseforge's `COURSEFORGE_SELF_VERIFY / _REFINE_ROUNDS / _CHUNK_SCOPED` row = 3 flags) counts once per flag it documents, so those tallies exceed the printed row count. The root-owned table is one row per flag, so its count equals its row count.

| Prefix | Owner | Flag count |
|--------|-------|-----------:|
| `TRAINFORGE_*` / `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` / `CURRICULUM_ALIGNMENT_*` / `WAVE18_*` | [`docs/operations/behavior-flags-trainforge.md`](docs/operations/behavior-flags-trainforge.md) | 77 |
| `NVIDIA_*` (vendor endpoint-registry row for the hosted large-model seat — `NVIDIA_API_KEY` / `NVIDIA_BASE_URL` / `NVIDIA_LARGE_MODEL`) | [`docs/operations/behavior-flags-trainforge.md`](docs/operations/behavior-flags-trainforge.md) | 3 |
| `SEMANTIK_*` (SemantiK semantic-cascade converter; also honors the single legacy `DART_THETA_DEVICE` compat env, aliased to `SEMANTIK_THETA_DEVICE`) <!-- legacy-token: allow --> | [`docs/operations/behavior-flags-semantik.md`](docs/operations/behavior-flags-semantik.md) | 226 |
| `COURSEFORGE_*` / `COURSEPLANNER_*` / `TEXTBOOK_SYNTHESIS_*` | [`docs/operations/behavior-flags-courseforge.md`](docs/operations/behavior-flags-courseforge.md) | 45 |
| `DECISION_*` / `ED4ALL_*` / `LOCAL_DISPATCHER_*` / `MCP_ORCHESTRATOR_*` / `LLM_*` (cross-cutting) | [`docs/operations/behavior-flags.md`](docs/operations/behavior-flags.md) | 272 |

### Cross-cutting flags (root-owned)

Every root-owned flag — name, default, purpose, resolution chain, guardrails,
calibration status, anti-fabrication contract — lives in
[`docs/operations/behavior-flags.md`](docs/operations/behavior-flags.md), one
row per flag. That file is the index; grep it by flag name:

```bash
grep -n 'ED4ALL_NLI_CROSSBLOCK' docs/operations/behavior-flags.md
```

The `LLM_*` env vars (`LLM_MODE`, `LLM_PROVIDER`, `LLM_MODEL`) are CLI runtime knobs documented in § Quick Start above. Other `ED4ALL_*` vars — the GUI server vars (`ED4ALL_GUI_HOST` / `_PORT` / `_LEARNER` / `_MODE` / `_TOKEN`), the test-only fixture/gating overrides (incl. `ED4ALL_NLI_VALIDATORS_FACTORY`, the picklable-factory dotted-path seam the `ED4ALL_NLI_MICROBATCH_VALIDATORS` process pool uses to build each worker's NLI — default the production singleton loader), and the four rewrite-tier `ED4ALL_REWRITE_*` + the two W10 assessment flags `ED4ALL_ASSESSMENT_PROSE_PROVIDER` / `ED4ALL_ASSESSMENT_DIVERSIFIED` owned by the subsystem flag docs — are enumerated verbatim in [`docs/operations/behavior-flags.md`](docs/operations/behavior-flags.md).

---

## Licensing & ToS Posture

Canonical reference: **`docs/LICENSING.md`**. Read it before running any training-data synthesis pass.

The project distinguishes two surfaces with different licensing exposure:

- **Development tools** (Claude Code, OpenAI Codex) generate code, prose, and shell invocations. Their ToS restricts training-data routing, but on this project that restriction is moot — these tools never produce training data, so the dev tool you use has zero effect on the trained SLM's licensing.
- **LLM providers** (`--provider claude_session` / `together` / `local`) generate the paraphrased instruction / preference pairs that become training data. The trained model is a derivative work of those outputs, so the provider's ToS + the underlying model license decide whether the corpus is shippable.

Default posture: training-data synthesis routes to license-clean providers — `--provider local` with an Apache 2.0 model (Qwen2.5-7B/14B/32B) for an air-gapped clean corpus, or `--provider together` with a hosted OSS model as the cloud fallback. The `--provider anthropic` SDK training-pair path was **removed (Phase 4)** — `run_synthesis` fails closed on it unconditionally, so training-pair synthesis is license-clean by construction. The `claude_session` route stays wired behind the `TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS` acknowledgment gate but is not recommended for training data.

**Maintenance contract:** any new behavior flag (a row in whichever `docs/operations/behavior-flags*.md` owns its prefix) that selects an LLM provider, model ID, or synthesis backend MUST land with a corresponding row in `docs/LICENSING.md`'s "Synthesis providers" table. Drift between those per-flag rows and `docs/LICENSING.md` is a documentation bug.

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
- Statistical-tier embedding validators — the eight that load a `SentenceEmbedder` via `try_load_embedder`: `objective_assessment_similarity`, `concept_example_similarity`, `objective_roundtrip_similarity`, `co_terminal_alignment`, `source_coverage`, `rewrite_source_grounding`, `terminal_objective_source_grounding`, `distractor_misconception_alignment` (13 gate wirings in `config/workflows.yaml`). Two DISTINCT contracts:
  - **Missing `[embedding]` extras** → warning-severity `EMBEDDING_DEPS_MISSING` GateIssue with `passed=True`, unless `TRAINFORGE_REQUIRE_EMBEDDINGS=true` flips it fail-closed (that flip then wins over `on_error: warn`). Unchanged.
  - **Extras present, requested `ED4ALL_EMBEDDING_DEVICE` absent** (default `cuda`, no CUDA→CPU fallback) → `EmbeddingModelUnavailable`, a type unrelated to `EmbeddingDepsMissing`. Always fatal: the validators `preload()` before their audit loop and re-raise it past their per-encode handlers, and `ValidationGateManager.run_gate` carries a typed passthrough that returns `passed=False` + critical `EMBEDDING_MODEL_UNAVAILABLE` **without consulting `behavior_on_error`**, so `on_error: warn` can no longer rewrite it to a pass. Pin `ED4ALL_EMBEDDING_DEVICE=cpu` on GPU-less hosts — that explicit opt-out is the only supported downgrade.
  - Caveats that survive: all 13 wirings are declared `severity: warning`, so a device failure is a recorded FAILED gate but does not by itself halt the phase; a *typo'd* device token raises a plain `ValueError` and still warn-passes; `distractor_misconception_alignment` does not honor `TRAINFORGE_REQUIRE_EMBEDDINGS` on missing extras (pre-existing, deliberate). `bloom_classifier_disagreement` is NOT on this contract — it shares the `[embedding]` extras group but loads a BERT ensemble (`BertEnsembleDepsMissing` / `TRAINFORGE_REQUIRE_BERT_ENSEMBLE`). Full detail + the residual holes: `docs/architecture/validation-architecture.md` § 4.1-4.3.

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

SLM training is a post-import LibV2 stage, not a step in `Trainforge/process_course.py`. Top-level command: `ed4all run trainforge_train --course-name <slug> --base-model <name>` (`--course-name` is the CLI flag; `course_code` is only the handler-side param alias declared in `config/workflows.yaml::training`'s `inputs_from` block). `--base-model` populates `workflow_params.base_model` — the route that phase reads — and is validated at parse time against `Trainforge/training/base_models.py::BaseModelRegistry`, so an unknown name exits 2 with the supported list instead of silently training another base. Precedence: `--base-model` > `ED4ALL_CAMPAIGN_BASE_MODEL` > the registry default (`nemotron3-nano-30b`). The same flag pins the base for the in-build `--with-training` tail, and re-pins it on `--resume`. `--config-overrides <path-or-inline>` is the sibling route for per-run `TrainingConfig` fields — it populates `workflow_params.config_overrides` (the `config_overrides` route on the same `inputs_from` block), accepts a YAML/JSON file path, an inline JSON object, or inline `key=value[,key=value]` pairs (list fields use `|` between items), and is likewise re-applied on `--resume`. It is validated at parse time against the real `TrainingConfig` field set via the one canonical parser `Trainforge/training/configs/__init__.py::parse_config_overrides`, so an unknown key, a bad type, or an out-of-range value exits 2 (naming the supported field list on an unknown key) rather than being dropped or discovered mid-run; `base_model` is rejected there (use `--base-model`). This is the ONLY pipeline route to a field the checked-in per-base YAML deliberately leaves unset — `Trainforge/training/configs/nemotron3-nano-30b.yaml` ships `dpo_learning_rate: null` and `Trainforge/training/peft_trainer.py` RAISES rather than reusing the SFT rate, so Nemotron Nano DPO is unstartable without it. The supplied override set is recorded verbatim on `model_card.json::config_overrides` (and the effective `dpo_learning_rate` on `training_config`), because an adapter trained at a hand-picked rate is otherwise unreproducible. A real fit runs through the repository-managed training environment — `scripts/ops/bootstrap-training-env.sh` then `scripts/ops/ed4all-training`, which fails on a version-band violation BEFORE loading weights — not a bare `pip install ed4all[training]` (that extra is the portable band for dry-run planning, eval, and CI). Nemotron Nano additionally requires the documented canary preflight (`max_steps: 1` + `dpo_learning_rate: 1.0e-6` via `--config-overrides`, inspected for loss / peak GPU memory / wall time before the measured DPO rate is pinned): [`docs/operations/nemotron-lora-canary.md`](docs/operations/nemotron-lora-canary.md). No canary command runs automatically. Full deep-dive (base-model registry, provider config, 5×3 eval matrix, 7-hash provenance, promotion workflow, decision-capture contract): `Trainforge/CLAUDE.md § Training Pipeline`; the synthesis path that produces its inputs: `Trainforge/CLAUDE.md § Training-pair synthesis — what actually runs`.

---

## Training Data Export

Formats: `jsonl` (default), `alpaca`, `openai`, `dpo`. CLI: `ed4all export-training <run_id> --format <fmt> --output <path>`. Full reference: `Trainforge/CLAUDE.md § Training Data Export`.
