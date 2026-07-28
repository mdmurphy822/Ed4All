# MCP — Orchestration Engine

Navigation guide for the `MCP/` package: the FastMCP server surface plus the
in-process workflow engine that actually runs every `ed4all run` pipeline.

This file documents **what lives where** and the **non-obvious contracts**.
Workflow phase lists, agent registries, gate tables, and behavior-flag rationale
live in the root `CLAUDE.md` and `docs/` — not here.

---

## Layout

| Dir | Contents |
|-----|----------|
| `server.py` | FastMCP server instance (`mcp = FastMCP("ed4all-orchestrator")`), the four sandboxed file tools, tool-registry init/snapshot, sandbox + capability checks, `startup_hardening()`. |
| `core/` | The engine: `workflow_runner.py`, `executor.py`, `config.py`, `schemas.py`, `param_mapper.py`, `tool_schemas.py`. |
| `tools/` | Tool implementations. `pipeline_tools.py` is the pipeline phase-handler monolith; the rest are `register_*_tools(mcp)` modules. |
| `orchestrator/` | Mode-aware front controller + dispatchers + LLM backends + the file-based task mailbox. |
| `hardening/` | Checkpointing, error classification / poison pill / retry policy, validation-gate manager, gate-input routing, config lockfiles. |
| `ipc/` | `status_tracker.py` — file-locked multi-terminal status/lock/log files under `state/`. |
| `tests/`, `tools/tests/`, `core/tests/`, `hardening/tests/` | Tests colocated per layer (175 / 39 / 3 / 4 files respectively). |

Module sizes matter for navigation: `tools/pipeline_tools.py` is ~31k lines and
`core/workflow_runner.py` ~7.4k. Grep for the symbol; never read them whole.

---

## Entry points

Two, and they are different surfaces:

1. **`MCP/server.py`** — the external MCP surface. `python MCP/server.py` serves
   FastMCP. Only tools registered here (or via a `register_*_tools(mcp)` call it
   makes) are reachable from an external MCP client.
2. **`core/workflow_runner.py::WorkflowRunner.run_workflow`** — the pipeline
   engine. This is what `ed4all run` drives (`cli/commands/run.py` constructs
   `WorkflowRunner(executor=..., config=...)` directly). The FastMCP server is
   **not** in the path of a normal pipeline run.

`orchestrator/pipeline_orchestrator.py::PipelineOrchestrator` is the mode-aware
front controller layered above `WorkflowRunner`: it picks a dispatcher
(`LocalDispatcher` for `--mode local`, `APIDispatcher` for `--mode api`) and
hands it an `llm_factory`. Design invariant stated in the class docstring: the
orchestrator never calls an LLM itself — it only chooses a dispatcher and feeds
it a factory.

### WorkflowRunner responsibilities

`run_workflow` owns the phase loop: topological sort of phases
(`_topological_sort` / `_dependencies_met` / `_effective_depends_on`), skip
decisions (`_should_skip_phase`), parameter routing (`_route_params` driven by
`_get_phase_param_routing` read out of `config/workflows.yaml`), task creation
(`_create_phase_tasks`), delegation to `TaskExecutor.execute_phase`, and output
extraction (`_extract_phase_outputs` / `_get_phase_output_keys`).

`final_status` is one of `COMPLETE` / `FAILED` / `PAUSED`. `PAUSED` is the
graceful-stop terminal state — never `FAILED`.

Post-loop, `run_workflow` calls the `_maybe_write_*` aggregator methods
(`_maybe_write_promotion_chain_report`, `_maybe_write_coverage_map`,
`_maybe_write_build_cost_report`, `_maybe_harvest_bloom_labels`, and ~10 more).
All are best-effort by contract: an aggregator failure logs and continues, and
can never change `final_status`.

### TaskExecutor responsibilities

`core/executor.py::TaskExecutor` executes tasks. Key methods: `execute_task` →
`_execute_with_retries` → `_invoke_tool`; `execute_phase` (checkpoint at phase
start, checkpoint per task completion, run validation gates at phase end,
returns `(results, gates_passed, gate_results)`); `_execute_parallel` /
`_execute_sequential`; `validate_tool_registry` / `get_missing_tools`.

---

## Dispatch: three routing layers

This is the single most non-obvious part of the package. A task's handler is
chosen by walking these in order:

**1. `_PHASE_TOOL_MAPPING` (phase-name override, `core/executor.py`)**

Checked **before** the agent mapping. Nine phases route by phase NAME
regardless of which agent is threaded through the task:

```
content_generation_outline  -> run_content_generation_outline
inter_tier_validation       -> run_inter_tier_validation
content_generation_rewrite  -> run_content_generation_rewrite
post_rewrite_validation     -> run_post_rewrite_validation
imscc_chunking              -> run_imscc_chunking
assessment_synthesis        -> run_assessment_synthesis
heading_judge               -> run_heading_judge
training                    -> run_training
evaluation                  -> run_evaluation
```

`training` / `evaluation` are shared by the standalone `trainforge_train`
workflow and the opt-in `textbook_to_course` training tail (`--with-training`).
Phase-name routing alone is not sufficient for them: `training-synthesizer` is
in `AGENT_SUBAGENT_SET` and the subagent fork happens **before** the registry
lookup, so a deterministic-tool set keyed on the resolved tool name
(`run_training` / `run_evaluation`) forces in-process execution even under
`ED4ALL_AGENT_DISPATCH`.

This mapping **cannot be inferred from YAML**. It also gates virtual-task
synthesis: `workflow_runner._create_phase_tasks` synthesizes a single
`agent_type="phase-handler"` task for a phase that produced no agent tasks
**only if** `_PHASE_TOOL_MAPPING.get(phase.name)` is truthy. That is how validator-only phases declaring `agents: []` in `config/workflows.yaml`
still execute.

`imscc_chunking` illustrates why the override exists: it reuses the same chunker
agent as the content-side `chunking` phase, but must emit to `imscc_chunks/` with
`chunkset_kind="imscc"` instead of `semantik_chunks/` — a phase-name override picks
the right helper without forking the agent registry.

**2. `AGENT_TOOL_MAPPING` (agent-name fallback, `core/executor.py`)**

Maps agent names to registry tool names (`content-generator` →
`generate_course_content`, `textbook-ingestor` → `extract_textbook_structure`,
…). Renamed agents keep read-compat aliases here (e.g. the chunker agent is
`semantik-chunker`; a legacy chunker-agent key is retained as a dispatch
alias so old resume states still route).

**3. Subagent fork (`AGENT_SUBAGENT_SET` + `ED4ALL_AGENT_DISPATCH`)**

In `_invoke_tool`, if `ED4ALL_AGENT_DISPATCH` is truthy AND a `dispatcher` was
injected AND the task's `agent_type` is in `AGENT_SUBAGENT_SET` (the 12
reasoning agents — outliner, content-generator, remediation agents, assessment
agents, training-synthesizer), the call goes to `dispatcher.dispatch_task`
instead of the in-process registry. This fork happens **before** the registry
lookup, so an agent with no Python tool backing it does not trip the
"tool not registered" guard.

Per-agent escape hatch: `AGENT_PROVIDER_ENV_MAP` short-circuits the subagent
fork back to the in-process provider lattice when the agent's provider env var
is set — `content-generator`/`COURSEFORGE_PROVIDER`,
`course-outliner`/`COURSEPLANNER_PROVIDER`,
`assessment-generator`/`TRAINFORGE_ASSESSMENT_PROVIDER`.
`AGENT_AUTHORING_PROVIDER_ENV_MAP` is the wider map consumed by
`workflow_runner._enforce_authoring_provider_route`, which **fails fast** when
an LLM-needing agent has neither a
provider env, nor a serviced mailbox (`ED4ALL_AGENT_DISPATCH=true` +
`ED4ALL_MAILBOX_SERVICED=1`), nor the explicit test-only stub opt-in
(`LOCAL_DISPATCHER_ALLOW_STUB`) — so a run cannot hang on an unserviced mailbox
or silently degrade to a templated stub. Subagent-classified agents absent from
that map (`oscqr-course-evaluator`, `quality-assurance`, `content-analyzer`,
`accessibility-remediation`, `content-quality-remediation`,
`intelligent-design-mapper`, `assessment-extractor`, `assessment-validator`)
have no provider short-circuit at all — they are session-only.

---

## Tool surfaces: two registries, deliberately split

**External (`@mcp.tool()`)** — reachable from any MCP client. Registered either
directly in `server.py` (`list_directory`, `read_file`, `write_file`,
`file_info`) or via a `register_*_tools(mcp)` closure. The registration calls
are **module-level** in `server.py` (each wrapped in its own `try/except
ImportError` that appends to `_loaded_modules` / `_failed_modules`), so they run
at import time — not inside `startup_hardening()`, which only initializes the
capability registry and writes a snapshot.

| Module | External tools |
|--------|----------------|
| `tools/orchestrator_tools.py` | `create_workflow`, `get_workflow_status`, `dispatch_agent_task`, `poll_task_completions`, `execute_workflow_task`, `complete_workflow_task`, `update_generation_progress`, `acquire_batch_lock`, `release_batch_lock` |
| `tools/courseforge_tools.py` | `create_course_project`, `generate_course_content`, `get_courseforge_status`, `intake_imscc_package`, `package_imscc`, `remediate_course_content` |
| `tools/trainforge_tools.py` | `analyze_imscc_content`, `generate_assessments`, `validate_assessment`, `export_training_data`, `get_trainforge_status`, `analyze_teaching_role_alignment` |
| `tools/pipeline_tools.py` | `stage_semantik_outputs`, `archive_to_libv2`, `get_pipeline_status`, `validate_semantik_markers`, `synthesize_training` |
| `tools/analysis_tools.py` | `analyze_training_data`, `get_quality_distribution`, `preview_export_filter` |
| `tools/gui_tools.py` | the nine `gui_*` tools |
| `tools/assistant_tools.py` | the twelve `assistant_*` operator-assistant campaign tools (thin wrappers delegating to `lib/assistant`; every mutating tool routes through `campaign_tools.dispatch_campaign_tool`, so the external surface is bounded, not widened) |

**Internal (registry-only)** — `tools/pipeline_tools.py::_build_tool_registry()`
returns a `{tool_name: async callable}` dict consumed by `TaskExecutor` for
workflow-phase dispatch. These are **intentionally not** `@mcp.tool()`-decorated,
so they are unreachable from external MCP clients. Current keys:

`analyze_imscc_content`, `archive_to_libv2`, `build_source_module_map`,
`create_course_project`, `extract_and_convert_pdf`, `extract_textbook_structure`,
`generate_assessments`, `generate_course_content`, `get_courseforge_status`,
`intake_imscc_package`, `package_imscc`, `plan_course_structure`,
`remediate_course_content`, `run_assessment_synthesis`, `run_concept_extraction`,
`run_content_generation_outline`, `run_content_generation_rewrite`,
`run_evaluation`, `run_heading_judge`, `run_imscc_chunking`,
`run_inter_tier_validation`, `run_post_rewrite_validation`, `run_training`,
`run_dart_chunking`, `run_vector_indexing`, `stage_semantik_outputs`, <!-- legacy-token: allow -->
`synthesize_training`,
`validate_assessment`.

Several names appear in **both** surfaces (`archive_to_libv2`, `package_imscc`,
`stage_semantik_outputs`, `generate_assessments`, `synthesize_training`,
`generate_course_content`). They are separate implementations, and **parity
between the two variants is a tested contract** — see
`tests/test_archive_to_libv2_mcp_tool_parity.py`,
`tests/test_package_imscc_mcp_tool_parity.py`,
`tests/test_tool_registry_schema_parity.py`,
`tests/test_missing_registry_stubs.py`. Change one variant, change both.

`core/tool_schemas.py` holds the declarative per-tool schema (required/optional
params, param name mapping, defaults) that `core/param_mapper.py::TaskParameterMapper`
uses to translate a generic task dict into tool-specific kwargs. A tool whose
schema drifts from its `@mcp.tool()` signature fails the parity test.

Three modules under `tools/` are plain Python helpers with **no** MCP
registration and no `register_*` function: `tools/tutoring_tools.py`
(misconception index / matching, consumed by `LibV2/tools/intent_router.py`),
`tools/intent_dispatch_tool.py`, and `tools/quiz_generator.py` (the
deterministic LLM-free engine behind `ed4all libv2 generate-quiz`, imported by
`cli/commands/libv2_generate_quiz.py`). `tools/_content_gen_helpers.py` is
likewise a private helper module for `pipeline_tools`, not a tool surface.

### Where phase handlers actually live

Nearly all of them are in `tools/pipeline_tools.py` as module-level
`async def _run_*` / `_generate_*` functions wrapped into the registry by
`_build_tool_registry`. The heavyweight ones:
`_run_content_generation_outline`, `_run_inter_tier_validation`,
`_run_content_generation_rewrite`, `_run_post_rewrite_validation`,
`_run_heading_judge`, `_run_stage2_window_synthesis`, plus the conversion seams
`_run_semantik_v2_conversion`, `_run_vendor_ingest_conversion`, and
`_run_semantik_bridge_subprocess` (the cross-venv SemantiK bridge).

Conversion dispatch is **phase-scoped**: `extract_and_convert_pdf` only routes
to the SemantiK cascade when `kwargs["phase"]` is `semantik_conversion` (or the
legacy conversion-phase name still accepted on read); anything else fails closed
with an explicit reason rather than silently falling through. Input type is
detected by `_detect_conversion_input_type` (PDF wins a mixed directory;
unknown input fails closed).

---

## Dispatcher / mailbox model

`orchestrator/local_dispatcher.py::LocalDispatcher` has two dispatch paths:

- `_dispatch_task_via_callable` — a directly injected callable.
- `_dispatch_task_via_mailbox` / `_dispatch_via_mailbox` — writes a task spec to
  the file mailbox and waits for a completion envelope. This is how a Claude
  Code session (or `scripts/mailbox_servicer.py`) services subagent work.

`orchestrator/task_mailbox.py::TaskMailbox` is the file protocol. Root is
`state/runs/{run_id}/mailbox/`, with the `state/runs` base overridable via
`ED4ALL_STATE_RUNS_DIR`. (`ED4ALL_MAILBOX_BASE_DIR` is a separate knob, read by
`MailboxBrokeredBackend` in `orchestrator/llm_backend.py`, not by `TaskMailbox`.)
Three state dirs —
`pending/`, `in_progress/`, `completed/` — and the transitions are
`put_pending` → `claim` (raises `TaskClaimConflict` on a double claim) →
`complete` → `read_completion` / `wait_for_completion` /
`await_completion_async` → `cleanup`.

Timeouts: `_DEFAULT_MAILBOX_TIMEOUT` 600s; agent-task timeout defaults to 1800s,
overridable by `ED4ALL_AGENT_TIMEOUT_SECONDS`. On timeout the dispatcher does
not hard-kill immediately — `_mailbox_grace_seconds` (10% of the timeout, capped
at 600s) buys a grace drain via `_await_completion_within_grace`.

`orchestrator/api_dispatcher.py::APIDispatcher` is the `--mode api` counterpart
(`dispatch_phase` / `dispatch_task` / `dispatch_batch`) and refuses to stub
unless explicitly allowed (`APIDispatcherStubNotAllowed`).

`orchestrator/llm_backend.py` holds the `LLMBackend` protocol and its
implementations — `AnthropicBackend`, `LocalBackend`, `OpenAICompatibleBackend`,
`MailboxBrokeredBackend`, `MockBackend` — plus `build_backend(BackendSpec)`,
`resolve_openai_compatible_backend`, and `license_metadata_for_provider`. A new
OpenAI-compatible provider is a `config/endpoints.yaml` registry entry resolved
here, never a new subclass. `_CaptureMixin` is what wires DecisionCapture into
every backend call; `_redact_base_url_for_capture` keeps host detail out of
captures.

`orchestrator/content_prompts.py` builds the prompts handed to mailbox-brokered
subagents. Its stated design rule: prompts must not leak corpus-specific
identifiers.

`orchestrator/worker_contracts.py` defines the JSON-serializable language every
dispatcher speaks: `PhaseInput`, `PhaseOutput`, `GateResult`. Note this
`GateResult` is the *worker-contract* dataclass and is distinct from
`hardening/validation_gates.py::GateResult` — same name, different layer.

---

## Checkpointing and graceful stop

`hardening/checkpoint.py::CheckpointManager` writes
`state/runs/{run_id}/checkpoints/{phase}_checkpoint.json`
(`start_phase` → `complete_task` → completion). `get_resume_point(run_path)` is
the resume entry. Non-obvious: `_PHASE_NAME_ALIASES` maps the conversion phase's
legacy name ↔ `semantik_conversion` both directions, so a checkpoint written
under one phase name is still found under the other across the rename — a resume
must not break on the rename.

**Resume accounting (c7339ac1).** A phase that spans N resume segments is
reported as ONE phase, not as its last segment. `start_phase` opens a segment
but no longer rewrites `started_at`: the checkpoint keeps `first_started_at`,
accumulates `elapsed_seconds` across segments, and counts `segments`;
`complete_phase` / `fail_phase` / `pause_phase` each close their segment. A
legacy checkpoint predating those fields is folded in from its timestamps rather
than discarded. `pause_phase` also writes `status="paused"` — a pause has to be
distinguishable from a finish, or a phase that stopped after ~2% of its units
reports "completed". Consumers should prefer the cumulative `elapsed_seconds`
over the single-segment span (`gui/services/progress_service.py` does).

Graceful stop is a filesystem sentinel polled at unit boundaries. The shared
primitives live outside this package in `lib/generation/stop_control.py`
(`stop_requested`, `GracefulStopRequested`); MCP consumes them at:

- `core/executor.py` — between parallel/sequential task dispatches.
- `core/workflow_runner.py` — `_stop_requested_now()` at the phase-loop boundary;
  sets `final_status = "PAUSED"`.
- `orchestrator/task_mailbox.py` — inside the completion wait loop.
- `tools/pipeline_tools.py` — inside the long-running generation/validation
  loops; the SemantiK bridge subprocess gets the sentinel forwarded through the
  env (`_semantik_stop_sentinel_env_value`).

Timeout semantics differ by scope: a **batch** timeout grace-drains to a
checkpoint and becomes `PAUSED`; a **task** timeout grace-drains then keeps the
`TIMEOUT` classification and the transient-retry ladder. `_grace_seconds` in the
executor computes the window. Per-phase batch timeout precedence: the phase's
YAML `batch_timeout_minutes` wins over `ED4ALL_BATCH_TIMEOUT_MINUTES` /
the executor-wide default, for that phase only. Likewise, the phase's
`timeout_minutes` is forwarded as its phase-local per-task deadline and wins
over `ED4ALL_TASK_TIMEOUT_MINUTES` / the executor default without mutating
other phases. Resumes reload both values from the workflow registry; timeout
values in an old task/checkpoint cannot pin the resumed phase to a stale
deadline. A multi-day phase must declare both values because its single task
and its whole batch are independent safety deadlines.

`hardening/error_classifier.py` supplies `ErrorClassifier` (transient vs
permanent), `PoisonPillDetector` (same-pattern failure threshold stops a batch),
and `RetryPolicy`.

`hardening/lockfile.py` pins config integrity per run
(`create_run_lockfile` / `verify_run_lockfile`), so a resumed run can detect that
its config changed underneath it.

---

## Validation gates

`hardening/validation_gates.py::ValidationGateManager` loads validators by
dotted path from `config/workflows.yaml`. Security contract: `load_validator`
enforces `ALLOWED_VALIDATOR_PREFIXES` = `lib.validators.`, `lib.leak_checker`,
`Courseforge.router.` — an arbitrary module path in YAML raises `ImportError`
rather than importing it.

`hardening/gate_input_routing.py::GateInputRouter` maps a validator's dotted
path to a builder that assembles that validator's inputs from `phase_outputs` +
`workflow_params`. `default_router()` registers ~106 builders. Two contracts
worth knowing:

- **Unknown validator → fallthrough, logged.** A validator with no registered
  builder returns `({}, ["__no_builder_registered__"])` and logs a warning, so
  the drift is observable rather than silent. Same for a raising builder
  (`__builder_error__`) — builders never raise by contract.
- **`cache` is opt-in and signature-sniffed.** `_builder_accepts_cache(fn)`
  inspects the builder; the shared `BlockFeatureCache` is threaded only into
  builders that declare a `cache` parameter, so builders without it are called
  with the exact legacy signature (flag off ⇒ byte-identical).

Adding a validator is a one-line `r.register(...)` entry plus its YAML gate —
no executor edits.

---

## Courseforge stage subcommands

`ed4all run courseforge-outline|validate|rewrite|courseforge` reuse the full
`textbook_to_course` phase list but restrict execution via
`WorkflowRunner._COURSEFORGE_STAGE_ACTIVE_PHASES` (resolved by
`_resolve_courseforge_stage_active_phases`, applied by
`_should_skip_for_courseforge_stage`). Semantics: an unrecognized stage returns
`None` and skips nothing (do not skip on behalf of a typo); phases inside the
four-phase two-pass surface but outside the stage's whitelist are skipped; every
phase outside that surface is skipped entirely — pre-Courseforge state is
instead pre-populated from disk by `_synthesize_outline_output`.

Related synthesis-from-disk helpers on `WorkflowRunner`:
`_synthesize_conversion_skip_output` (`--skip-conversion`),
`_synthesize_course_planning_reuse_output` (`--reuse-objectives`, with
`_normalize_to_courseforge_form` accepting both the Courseforge and LibV2
archive shapes), `_restore_resume_phase_outputs` / `_reconstruct_resume_outputs`
(resume), and `_resume_reusable_conversion_stems` (`--reuse-conversion`).

---

## Gotchas

- **Two `GateResult` classes.** `hardening/validation_gates.py` vs
  `orchestrator/worker_contracts.py`. Check the import.
- **Legacy phase/agent names are dual-READ, not rewritten.** The conversion
  phase and the chunker agent both keep their pre-SemantiK names accepted on read
  (checkpoint aliases, `AGENT_TOOL_MAPPING` alias, `_CONVERSION_PHASE_NAMES`) so
  paused runs resume across the rename. Don't "clean up" a legacy read-compat key
  without checking the resume path.
- **`server.py` is not in the pipeline path.** Debugging a phase means
  `workflow_runner` → `executor` → `pipeline_tools`, not the FastMCP server.
- **Registry-only tools are invisible to MCP clients by design.** If you want a
  pipeline helper reachable externally, that is a deliberate surface expansion
  with a parity test, not a decorator addition.
- **Seat scheduling / GPU lifecycle hooks are best-effort and off-loop.**
  `_gpu_lifecycle_sweep`, `_vram_doctor_snapshot`, `_apply_seat_schedule` run at
  phase boundaries and are documented in-code as unable to perturb
  `final_status` — except `SeatScheduleProbeError`, which is deliberately loud
  (a seat that cannot come up coherently must not silently produce output).
