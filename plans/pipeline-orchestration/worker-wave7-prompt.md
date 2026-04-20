# Wave 7 Worker Prompt — Pipeline Orchestration Foundation

> This is the prompt text to dispatch when the user approves Wave 7 kickoff. Copy-paste into the `Agent` tool call, `subagent_type: general-purpose`, `isolation: worktree`.

---

## Step 0 — WORKTREE ISOLATION (MANDATORY, do this before anything else)

You are running with `isolation: worktree`. A dedicated git worktree has been created for you off `dev-v0.2.0`. **Operate exclusively inside your worktree.** Do not `cd` out of it; do not use absolute paths that point at the user's main checkout at `/home/mdmur/Projects/Ed4All`; do not touch `main`.

Confirm your CWD is a worktree path (it will look like `/tmp/claude-worktrees/...` or similar), run `git status` + `git branch --show-current` to confirm you're on a fresh `worker-wave7-orchestration` (or similar) branch. If anything looks wrong, stop and report.

All file paths below are relative to your worktree root.

## Context

Ed4All is a multi-subsystem educational pipeline (DART PDF→HTML, Courseforge course gen, Trainforge assessment + knowledge graph). Waves 1–6 landed v0.2.0 KG-quality work on `dev-v0.2.0`. 1025 tests pass, 8/8 integrity checks.

Root docs you must read before coding:
- `CLAUDE.md` (root) — project protocols, MCP tool registry, opt-in flag table
- `plans/pipeline-orchestration/design.md` — **the design spec this wave implements**; all 6 decisions (O1–O6) are locked
- `MCP/core/workflow_runner.py` — existing WorkflowRunner you build on top of
- `config/workflows.yaml` — declarative phase config (Wave 6 YAML routing intact)
- `cli/main.py` — existing CLI (the `textbook-to-course` command you're deprecating lives at L401)

## Goal

Land the **Pipeline Orchestration foundation**: a canonical `ed4all run <workflow>` CLI + `PipelineOrchestrator` class + `LLMBackend` abstraction that lets every LLM call in the codebase go through a swappable backend (`local` = Claude Code session, `api` = Anthropic SDK).

One worker (you), one PR, merged to `dev-v0.2.0`. Main untouched.

## Deliverables

### New files

1. **`MCP/orchestrator/__init__.py`** — package init; exports `PipelineOrchestrator`, `LLMBackend`, `PhaseInput`, `PhaseOutput`.

2. **`MCP/orchestrator/worker_contracts.py`** — dataclasses shared across modes:
   - `PhaseInput` (run_id, workflow_type, phase_name, phase_config, params, mode, llm_factory, project_root, state_dir, captures_dir)
   - `PhaseOutput` (run_id, phase_name, outputs, artifacts, gate_results, decision_captures_path, status, error, metrics)
   - `GateResult` (gate_id, severity, passed, issues, details)
   - All dataclasses must be JSON-serializable (for state persistence + subagent handoff).

3. **`MCP/orchestrator/llm_backend.py`** — `LLMBackend` Protocol + implementations:
   - `LLMBackend` Protocol: `async def complete(system, user, *, model=None, max_tokens=4096, temperature=0.7, stream=False) -> str | AsyncIterator[str]`
   - `LocalBackend`: uses the current Claude Code session. Implementation: calls `Agent` tool via MCP with the prompt as subagent input, returns subagent result. Non-streaming (Agent tool is non-streaming).
   - `AnthropicBackend`: `import anthropic; self.client = anthropic.Anthropic(api_key=...)`; direct SDK call; streaming when `stream=True`. Default model: `claude-opus-4-7` (per root CLAUDE.md).
   - `OpenAIBackend`: **stub only** — class definition with `NotImplementedError` in `complete`. Docstring says "reserved for future wave". Per decision O2.
   - `MockBackend` (for tests): records calls, returns fixture-driven responses. Place under `lib/tests/fixtures/llm_responses/`.

4. **`MCP/orchestrator/pipeline_orchestrator.py`** — `PipelineOrchestrator` class:
   - Constructor takes `config: OrchestratorConfig`, `mode: Literal["local", "api"]`, `llm_factory: Callable[[], LLMBackend]`.
   - Primary method `async def run(workflow_id: str) -> Dict[str, Any]`:
     - Load workflow state (reuse pattern from `MCP/core/workflow_runner.py::WorkflowRunner.run_workflow`).
     - Topologically sort phases (call into existing `WorkflowRunner._topological_sort`).
     - For each phase: build `PhaseInput`, dispatch via mode-specific dispatcher, await `PhaseOutput`, run validation gates (reuse `WorkflowRunner.run_validation_gates`), persist checkpoint.
     - Emit orchestrator-level decision captures for dispatch choices + retries + gate outcomes.
   - Does NOT replace WorkflowRunner wholesale — sits on top of it; reuses its phase-ordering, state-persistence, gate-running logic. Think of PipelineOrchestrator as a front controller; WorkflowRunner is still the engine.

5. **`MCP/orchestrator/local_dispatcher.py`** — `LocalDispatcher`:
   - `async def dispatch_phase(phase_input: PhaseInput) -> PhaseOutput`
   - Constructs a subagent prompt from the phase's agent spec (look up via `phase_config.agents[0]` → read corresponding `agents/*.md` file).
   - Invokes the subagent via the MCP `Agent` tool (you'll need to figure out how to do this cleanly — study `MCP/tools/orchestrator_tools.py::dispatch_agent_task` for how the existing code does it).
   - Parses subagent result JSON → `PhaseOutput`.

6. **`MCP/orchestrator/api_dispatcher.py`** — `APIDispatcher`:
   - `async def dispatch_phase(phase_input: PhaseInput) -> PhaseOutput`
   - Runs the phase as a Python coroutine in-process (not subprocess — keep Wave 7 scope narrow).
   - Uses the provided LLM backend for any LLM calls.
   - For phases that need per-page parallelism (content_generation), respects `phase_config.max_concurrent`.

7. **`cli/commands/__init__.py`** + **`cli/commands/run.py`** — new Click command:
   - `ed4all run <workflow_name>` with flags per design doc §"Canonical entry point" (--corpus, --course-name, --mode, --api-provider, --model, --weeks, --no-assessments, --resume, --dry-run, --watch, --json)
   - Instantiate `PipelineOrchestrator`, call `.run()`, stream or return result per `--watch` / `--json`
   - Add command to the CLI group — wire into `cli/main.py`'s `@cli.group()` entrypoint

8. **Tests:**
   - `MCP/tests/test_pipeline_orchestrator.py` — orchestrator end-to-end with `MockBackend`; crash recovery; gate failure handling
   - `lib/tests/test_llm_backend.py` — protocol conformance; MockBackend behavior; AnthropicBackend contract (with mocked SDK — don't hit real API)
   - `lib/tests/test_local_dispatcher.py` + `lib/tests/test_api_dispatcher.py` — phase dispatch logic with fixtures
   - `cli/tests/test_run_command.py` — CLI flag parsing, dry-run output, --resume behavior

### Refactors (behavior-preserving)

9. **`DART/pdf_converter/claude_processor.py`** — constructor accepts `llm: Optional[LLMBackend] = None` in addition to existing `api_key`. If `llm` provided, use it; else fall back to current `anthropic.Anthropic()` path (backward compat for existing direct users). `_call_claude` internal method routes through the backend if available.

10. **`DART/pdf_converter/alt_text_generator.py`** — same refactor shape.

11. **`Trainforge/align_chunks.py`** — same refactor shape. Note this one has a function-local `import anthropic` at L599; change to accept an injected backend parameter.

All three refactors are **additive** — existing callers pass no `llm=` and get unchanged behavior; new orchestrator-dispatched callers pass `llm=` and route through the abstraction. No breaking changes.

### Deprecations

12. **`cli/main.py::textbook_to_course`** (L401) — add a deprecation warning at the top of the command body: `click.secho("DEPRECATED: Use 'ed4all run textbook-to-course' instead. This command will be removed in the next cleanup cycle.", fg='yellow')`. Keep the command working; just warn.

13. **`MCP/tools/pipeline_tools.py::create_textbook_pipeline_tool` + `run_textbook_pipeline_tool`** — no behavior change; add docstring note that these are internal and users should go through `ed4all run`. Not removed in this wave.

### Documentation

14. **Root `CLAUDE.md`** — add to the "Quick Start" section: document `ed4all run <workflow> --corpus PATH --mode local|api`. Add to the opt-in flag table: `LLM_MODE=local|api` (default local), `LLM_PROVIDER=anthropic|openai` (api mode only; openai stubbed). Document which env vars matter per mode.

15. **Root `README.md`** — update Quick Start example + the CLI section to show `ed4all run textbook-to-course ...` as the primary/recommended command. Keep `ed4all textbook-to-course` visible only as "Deprecated, use `ed4all run` instead" one-liner (or remove from CLI section entirely if it feels cleaner — your call, optimize for end-customer clarity). **CRITICAL: do NOT add any version callouts, "v0.2.0 changes" sections, wave summaries, or changelog-style content to README.md. It is for end customers only — project purpose + getting started only.** This is a hard requirement per user's explicit feedback.

16. **`plans/pipeline-orchestration/design.md`** — mark the "Ready-to-go checklist" Wave 7 item as complete after your implementation.

## Constraints

- **Local-only run** — no `git push`, no `gh` commands, no PRs created by you. When done, commit to your worker branch and STOP; orchestrator (the user) will open the PR.
- No schema changes. No workflow config changes (`config/workflows.yaml` untouched).
- No modifications to `WorkflowRunner.run_workflow` or `run_validation_gates` beyond what's strictly needed to make them reusable from `PipelineOrchestrator`.
- No new external dependencies beyond `anthropic` (already in requirements.txt per grep).
- Keep `AnthropicBackend` + `OpenAIBackend` stub + `MockBackend` as the full backend roster for Wave 7. No extra backends.
- No streaming in Wave 7 (per decision O3). `stream=True` param exists in the Protocol signature (for future use) but `AnthropicBackend.complete(stream=True)` raises `NotImplementedError` with a clear message.
- No multi-host / distributed execution.
- No LLM response caching (DART's existing `FileCache` stays as-is; don't port into the backend abstraction).
- Match existing code style: type hints everywhere, dataclasses for structured I/O, `logger = logging.getLogger(__name__)` convention.

## Acceptance criteria

Before declaring done:

1. `python3 -m ci.integrity_check` passes 8/8.
2. Full test suite passes (expected baseline 1025 + your added tests ≈ 1040+).
3. `ed4all run --help` shows the new command with all flags.
4. `ed4all run textbook-to-course --dry-run --corpus=inputs/pdfs/fake.pdf --course-name=TEST` prints the planned phases without error (even with fake PDF — dry-run shouldn't touch it).
5. `ed4all textbook-to-course --help` still works but prints the deprecation warning.
6. Running the full test suite with `LLM_MODE=local` and with `LLM_MODE=api` + mocked Anthropic both pass.
7. Zero direct `import anthropic` outside `MCP/orchestrator/llm_backend.py` (grep-verified; refactored sites import from the abstraction).
8. `MockBackend` fixture responses exist for at least: `claude_processor` structure-analysis prompt, `alt_text_generator` prompt, `align_chunks` prompt. Test coverage for the three refactored call sites.
9. No new `TODO` / `FIXME` / `HACK` comments introduced.

## What to hand back to the orchestrator

On completion, return a JSON summary:

```json
{
  "status": "ok" | "partial" | "failed",
  "branch": "<your-worker-branch>",
  "commit_sha": "<final commit sha>",
  "files_added": [...],
  "files_modified": [...],
  "tests_added": <int>,
  "tests_passing": <int>,
  "tests_total": <int>,
  "integrity_check": "8/8" | "<details>",
  "deprecation_warnings_wired": true | false,
  "direct_anthropic_imports_remaining": <int, should be 1 — llm_backend.py only>,
  "open_issues": [...],   // any gotchas or half-finished pieces
  "next_wave_ready": true | false,
  "summary": "<2-3 sentences of what shipped>"
}
```

If anything is blocked, status=`partial` with clear open_issues; don't fake completion.

## Out of scope (explicit)

- OpenAI backend implementation (stub only per O2)
- Token streaming (per O3)
- Source-provenance schema changes (those are Waves 8–11)
- New validation gates
- Phase-level MCP tool consolidation (keep create_*_pipeline tools working)
- Real API calls in tests (always mock)
- Performance optimization / caching
- UI / web surfaces

## If you get stuck

- Orchestrator (the user) is running this as a background agent. If you hit a genuine blocker (unclear contract, breaking compat requirement surfaces, schema drift), **stop and return with `status: partial` and a clear `open_issues` description**. Don't improvise schema changes.
- If a refactor reveals that a call site has more direct-SDK coupling than expected (e.g., uses streaming or uses an uncommon model param we didn't plan for), document it in open_issues rather than cargo-culting through.
