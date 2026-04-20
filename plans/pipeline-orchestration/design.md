# Design: Pipeline orchestration feature (Wave 7 on dev-v0.2.0)

## Goal

One canonical end-to-end entry point to run any Ed4All workflow, with a **dual-mode LLM backend**:

- **Local mode** — Claude Code session acts as the orchestrator; phase workers are subagent dispatches. No API key required. Default for interactive use.
- **API mode** — Python orchestrator process; phase workers invoke Anthropic/OpenAI SDKs directly. Requires API key. Supports token streaming. For headless/batch runs.

Cleanup goal: retire the three-way invocation maze (CLI that doesn't run, MCP tools that create-but-not-execute, direct Python) in favor of a single `ed4all run <workflow>` that Just Works.

## Current state (snapshot)

### Three invocation paths, all incomplete

1. **CLI** — `ed4all textbook-to-course <pdf>` at `cli/main.py:401`. Calls `create_textbook_pipeline()` but not `run_textbook_pipeline()`. Exits after workflow creation. User told to "Monitor progress with `ed4all summarize-run`" — but there's no `run` step before monitoring. De facto unusable for full pipeline runs.

2. **MCP tools** — `create_textbook_pipeline_tool` at `MCP/tools/pipeline_tools.py:195` + `run_textbook_pipeline_tool` at L573. Both work but must be called sequentially via an MCP client. Not a first-class user surface.

3. **Direct Python** — `create_textbook_pipeline()` + `run_textbook_pipeline()` as importable functions. Used by CLI + MCP. Works but requires knowing the internals.

Supporting infrastructure that's already solid (keep):

- `MCP/core/workflow_runner.py::WorkflowRunner.run_workflow()` — topological phase ordering, crash recovery via `phase_outputs`, optional phase skipping. Wave 6 added YAML-validated phase routing. This is the engine.
- `MCP/core/executor.py::TaskExecutor` — per-phase task execution.
- `config/workflows.yaml` — declarative phase/gate config.

### No LLM abstraction

Three places use `anthropic.Anthropic()` directly:

- `DART/pdf_converter/claude_processor.py:242` — secondary structure-analysis path.
- `DART/pdf_converter/alt_text_generator.py:61` — image alt-text generation.
- `Trainforge/align_chunks.py:606` — chunk-alignment LLM call.

Zero abstraction. Each hardcodes `import anthropic` + reads `ANTHROPIC_API_KEY`. No way to swap backends or run LLM-free.

### No orchestrator-level worker dispatch

`WorkflowRunner.run_workflow()` walks phases sequentially. Each phase's "task" is an in-process call through the `TaskExecutor`. There is no notion of "dispatch this phase to a separate worker process / subagent." All work happens in the caller's process.

That's fine for MCP tool calls in a server, but it's why there's no natural place to sit a Claude Code subagent as a phase executor.

## Design

### Canonical entry point

New CLI command at `cli/commands/run.py` (replaces `cli/main.py::textbook-to-course`):

```
ed4all run <workflow_name> [options]

Options:
  --corpus PATH           Input material (PDF, directory of PDFs, IMSCC)
  --course-name NAME      Required; course identifier
  --mode MODE             local | api (default: local)
  --api-provider PROVIDER anthropic | openai (default: anthropic; api mode only)
  --model MODEL           Model ID; default per provider
  --weeks N               Course duration (workflow-dependent)
  --no-assessments        Skip trainforge phase
  --resume RUN_ID         Resume a prior run from last checkpoint
  --dry-run               Show plan without executing
  --watch                 Stream phase transitions + LLM output to stdout
  --json                  Machine-readable output
```

Under the hood:
1. CLI constructs the workflow params (as today).
2. CLI instantiates the **PipelineOrchestrator** (new).
3. Orchestrator creates the workflow state (reuses `create_*_pipeline` logic).
4. Orchestrator dispatches phase workers according to `--mode`.
5. Orchestrator streams progress / final result back.

Legacy `ed4all textbook-to-course` stays for one cycle with a deprecation warning redirecting to `ed4all run textbook-to-course`.

### PipelineOrchestrator

New package: `MCP/orchestrator/`.

- `pipeline_orchestrator.py` — mode-agnostic entry point. Wraps `WorkflowRunner` but replaces its in-process phase execution with **worker dispatch** via a mode-specific dispatcher.
- `llm_backend.py` — `LLMBackend` Protocol + `LocalBackend`, `AnthropicBackend`, `OpenAIBackend` implementations.
- `worker_contracts.py` — `PhaseInput`, `PhaseOutput`, `GateResult` dataclasses. Shared across all workers regardless of mode.
- `local_dispatcher.py` — dispatches phase workers as Claude Code `Agent` subagent calls.
- `api_dispatcher.py` — dispatches phase workers as Python subprocesses (or coroutines) that use an `AnthropicBackend` / `OpenAIBackend`.

Orchestrator responsibilities:
- Load workflow config + state.
- Topological phase order (reuse `WorkflowRunner._topological_sort`).
- For each phase, construct `PhaseInput` (params routed from prior phases per Wave 6 YAML config).
- Dispatch to mode-specific worker.
- Await `PhaseOutput`; run post-phase validation gates (already wired via `WorkflowRunner.run_validation_gates`).
- Persist phase output + checkpoint.
- On failure: classify error (transient vs permanent via existing hardening layer), retry per workflow retry policy, or hard-fail with a diagnosable error surface.
- Emit decision captures end-to-end into `training-captures/`.

### LLM backend abstraction

```python
# MCP/orchestrator/llm_backend.py

class LLMBackend(Protocol):
    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | AsyncIterator[str]: ...

class LocalBackend:
    """LLM backend that uses the current Claude Code session.
    Implemented by dispatching Agent subagent calls whose prompt IS the
    LLM prompt. The subagent's session context does the completion.
    Non-streaming (the Agent tool returns final result only)."""

class AnthropicBackend:
    def __init__(self, api_key: str, default_model: str = "claude-opus-4-7"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.default_model = default_model
    async def complete(self, system, user, *, model=None, max_tokens=4096,
                       temperature=0.7, stream=False):
        # Direct SDK call; streaming support when stream=True.
        ...

class OpenAIBackend:
    """Optional; parallel to AnthropicBackend. Ship later if needed."""
```

Refactor the three direct-SDK sites to consume an injected `LLMBackend`:

- `DART/pdf_converter/claude_processor.py::ClaudeProcessor.__init__` — accept `llm: LLMBackend` instead of `api_key`. Remove the lazy `anthropic.Anthropic()` construction.
- `DART/pdf_converter/alt_text_generator.py` — same.
- `Trainforge/align_chunks.py` — same.

Each call site becomes `await self.llm.complete(system=..., user=..., model=...)`. No direct `anthropic` imports in domain code — only in `llm_backend.py`.

### Local-mode dispatch (Claude Code)

When `--mode local`:

- The running Claude Code session is the orchestrator.
- `LocalDispatcher.dispatch_phase(phase_input)` uses the `Agent` tool to spawn a subagent with:
  - A prompt that describes the phase's responsibilities (from the agent spec in `Courseforge/agents/` / `Trainforge/agents/` / `DART/agents/`).
  - The phase's input params.
  - The `LLMBackend` resolver — the subagent passes its LLM needs back to the orchestrator via `LocalBackend`, which dispatches *another* subagent for that completion. This is recursive but bounded (subagents don't recurse indefinitely; each LLM call is one subagent level deep).
- Subagent returns `PhaseOutput` as JSON; orchestrator validates.

Trade-off: fan-out of subagents is real. For a 12-week course with 60 generated modules, that's 60+ subagent calls per content-generation phase. Claude Code can handle this (the existing `max_concurrent: 10` batching in workflows.yaml applies). Cost: each subagent invocation is a session context; context is small because inputs are narrow (LO + source chunks + template).

### API-mode dispatch (SDK)

When `--mode api`:

- Orchestrator is a Python process (the CLI's own process, or a long-lived daemon).
- `APIDispatcher.dispatch_phase(phase_input)` runs the phase as a Python coroutine (or subprocess for heavier phases that benefit from isolation).
- Workers instantiate `AnthropicBackend(api_key=...)` and call it directly.
- Supports `--watch` token streaming: LLM output streams to stdout as it generates.

Trade-off: API costs real money; streaming needs backpressure handling. But for headless batch runs this is the right shape.

### Phase worker contracts

```python
@dataclass
class PhaseInput:
    run_id: str
    workflow_type: str            # e.g., "textbook_to_course"
    phase_name: str               # e.g., "content_generation"
    phase_config: PhaseConfig     # loaded from config/workflows.yaml
    params: Dict[str, Any]        # routed per workflows.yaml inputs_from
    mode: Literal["local", "api"]
    llm_factory: Callable[[], LLMBackend]  # workers call to get backend
    project_root: Path
    state_dir: Path               # state/runs/{run_id}/
    captures_dir: Path            # training-captures/{tool}/{course}/phase_{name}/

@dataclass
class PhaseOutput:
    run_id: str
    phase_name: str
    outputs: Dict[str, Any]       # matches phase_config.outputs declaration
    artifacts: List[Path]
    gate_results: Dict[str, GateResult]
    decision_captures_path: Path
    status: Literal["ok", "warn", "fail"]
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
```

Every phase worker — whether a Claude Code subagent or a Python subprocess — produces this same shape. The orchestrator doesn't care how it was produced.

### Cleanup: what goes away, what stays

**Deprecated (with warnings, one cycle):**
- `ed4all textbook-to-course` → redirect to `ed4all run textbook-to-course`.
- `create_textbook_pipeline_tool` + `run_textbook_pipeline_tool` as top-level MCP tools → merge into a single `run_workflow_tool` that takes `workflow_name` + params. (They stay as internal `create_*_pipeline` / `run_*_pipeline` functions.)
- Any direct `import anthropic` in domain code.

**Stays as building blocks (low-level, documented):**
- `WorkflowRunner` — orchestrator sits on top of it, not around it.
- `TaskExecutor`.
- Per-phase MCP tools (`convert_pdf_multi_source`, `stage_dart_outputs`, `generate_course_content`, `process_course`, etc.) — useful for phase-by-phase development and debugging.
- Direct script invocations (`Courseforge/scripts/generate_course.py`, `Trainforge/process_course.py`) — low-level.

**New primary surface:**
- `ed4all run <workflow>` (CLI).
- `MCP/orchestrator/PipelineOrchestrator` (programmatic).
- `LLMBackend` (swap-in backends).

### Error handling + decision capture

Existing hardening (Phase 0: error classifier, validation gates, checkpointing) is preserved verbatim — orchestrator calls it exactly as `WorkflowRunner` does. Decision capture (rationales ≥20 chars, strict-mode unknown-type checks) continues to write JSONL under `training-captures/`.

What's new: the orchestrator itself logs decision events for dispatch choices ("dispatched phase X to local worker"; "retried after transient error"; "hit poison pill threshold"). These are orchestrator-layer decisions, not domain decisions.

### Configuration

New env vars (all optional, with sensible defaults):

- `LLM_MODE=local|api` (default: `local`)
- `LLM_PROVIDER=anthropic|openai` (default: `anthropic`; api mode only)
- `LLM_MODEL=<model_id>` (default: per-provider)
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — standard SDK env vars; required for api mode

CLI flags override env vars. Env vars override defaults.

## Wave structure

### Wave 7 — Pipeline orchestration foundation

**One worker, one wave, merged into `dev-v0.2.0` via PR.** Same pattern as Waves 1–6.

Files touched:

**New:**
- `MCP/orchestrator/__init__.py`
- `MCP/orchestrator/pipeline_orchestrator.py` — `PipelineOrchestrator` class
- `MCP/orchestrator/llm_backend.py` — `LLMBackend` protocol + 2–3 implementations
- `MCP/orchestrator/worker_contracts.py` — `PhaseInput`, `PhaseOutput`, `GateResult`
- `MCP/orchestrator/local_dispatcher.py` — Claude Code subagent dispatch
- `MCP/orchestrator/api_dispatcher.py` — Python coroutine/subprocess dispatch
- `cli/commands/run.py` — new canonical CLI entry
- `MCP/tools/orchestrator_tools.py::run_workflow_tool` — unified MCP tool (internal)
- Tests under `MCP/tests/test_pipeline_orchestrator.py` + `lib/tests/test_llm_backend.py`

**Refactored:**
- `DART/pdf_converter/claude_processor.py` — consume `LLMBackend`
- `DART/pdf_converter/alt_text_generator.py` — consume `LLMBackend`
- `Trainforge/align_chunks.py` — consume `LLMBackend`
- `cli/main.py::textbook-to-course` — deprecation warning + redirect
- `CLAUDE.md` (root) — document `ed4all run` + LLM_MODE env var; add to opt-in flag section

**Not touched:**
- `config/workflows.yaml` — no phase config changes needed.
- `MCP/core/workflow_runner.py` — used as-is by orchestrator.
- Per-phase MCP tools — unchanged.

### Why Wave 7 (before source-provenance waves)?

Source-provenance Waves 8–11 add significant emit-side complexity. Testing those changes requires running the pipeline end-to-end. With the orchestration feature in place, that test story is simply `ed4all run textbook-to-course --corpus deans_for_impact.pdf --mode local`. Without it, testing source-provenance means stitching together CLI-create + MCP-run by hand, which we just determined is broken.

Landing orchestration first means:
- Source-provenance waves can include a 1-command smoke test as part of their exit criteria.
- The pipeline-run test we deferred earlier becomes trivially repeatable.
- API-key users (future) get the same testing surface as local users.

### Why one worker not multiple?

The orchestration work is tightly coupled — `PipelineOrchestrator`, `LLMBackend`, and the three refactored call sites all have to land together for `ed4all run` to work end-to-end. Splitting across workers creates either merge hell or a half-landed feature. One worker, one PR, all-or-nothing.

## Decisions (confirmed)

| # | Decision | Confirmed |
|---|---|---|
| O1 | Mode default | `local` (works without API key) |
| O2 | OpenAI backend in Wave 7 | `AnthropicBackend` only; stub `OpenAIBackend` for a later wave |
| O3 | Token streaming in API mode | Non-streaming in Wave 7; `--watch` streaming added later |
| O4 | Wave ordering | Wave 7 orchestration first, then 8–11 provenance sequential |
| O5 | Deprecation window for `ed4all textbook-to-course` | One cycle: warn in Wave 7, remove in next cleanup merge-train |
| O6 | Local dispatcher recursion | Accept fan-out; no batching speculation. Revisit if real runs surface pain |

See `plans/source-provenance/design.md` for provenance decisions P1–P5.

## Risks

1. **Local mode scale.** 60 subagents for content generation is realistic for small corpora but grows linearly. Need to confirm Claude Code's subagent-within-subagent behavior handles this without context issues.
2. **API mode cost surprise.** Easy to launch a 500pp corpus without realizing it'll cost $N. Add a `--estimated-cost` dry-run check that prints projected token counts + a confirmation prompt.
3. **Refactor blast radius.** Three call sites hardcode `anthropic.Anthropic()`. Refactoring changes their constructors — any code that instantiates them with `api_key=` positional args breaks. Low risk (they're internal), but needs a grep sweep.
4. **Test coverage.** API mode is hard to test without hitting real APIs. Need a `MockBackend` that records calls and returns fixtures. Add to `lib/tests/fixtures/llm_responses/`.
5. **State directory growth.** Every phase writes checkpoints + decision captures. For long corpora this balloons. Not a Wave 7 problem but flag for Wave 12+ (retention policy).

## Non-goals

- No API/web UI. CLI + programmatic only.
- No multi-host distributed execution. Local machine only.
- No LLM response caching in Wave 7 (separate optimization; DART already has one in `claude_processor.py::FileCache`).
- No mode-switching mid-run. Choose `local` or `api` at run start; can't flip.
- No automatic fallback between Anthropic and OpenAI. Provider is user-chosen.

## Relationship to source-provenance design

Source-provenance waves renumber:

- Wave 8 → DART provenance emit (was Wave 7 in `plans/source-provenance/design.md`)
- Wave 9 → Courseforge (was Wave 8)
- Wave 10 → Trainforge chunk + node (was Wave 9)
- Wave 11 → Trainforge evidence arms (was Wave 10)

Source-provenance implementation is unblocked by orchestration (the subsystems are disjoint). But *testing* source-provenance end-to-end benefits hugely from the orchestrator being in place.

I'll update `plans/source-provenance/design.md` to reflect the renumber once you approve this wave ordering.

## Ready-to-go checklist

- [x] User decisions confirmed (O1–O6 above).
- [x] Source-provenance design doc renumbered (Wave 8–11).
- [x] Wave 7 worker prompt drafted (with worktree isolation Step 0 guardrail).
- [x] User approves Wave 7 worker dispatch.
- [x] Wave 7 landed: `ed4all run`, `PipelineOrchestrator`, `LLMBackend`
      abstraction + refactors (claude_processor, alt_text_generator,
      align_chunks) + deprecation warning on `ed4all textbook-to-course`.
      CLAUDE.md + README.md updated to reflect `ed4all run` as canonical entry.
