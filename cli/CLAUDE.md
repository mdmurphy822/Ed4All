# CLI — the `ed4all` Entry Point

Agent-facing contract map for `cli/`. Operator-facing invocation recipes are not
duplicated here:

- [`docs/operations/pipeline-invocation.md`](../docs/operations/pipeline-invocation.md)
  — per-stage invocation, timeout knobs, stop/resume runbook, seat pins.
- Root [`CLAUDE.md`](../CLAUDE.md) § Quick Start — canonical command examples.

---

## Entry point + registration

`pyproject.toml` declares `[project.scripts] ed4all = "cli.main:main"`.
`cli/main.py` builds a `click.Group` and each command attaches itself through a
`register_*_command(cli_group)` function exported by `cli/commands/__init__.py`.

**Every registration is wrapped in its own `try/except ImportError`** that logs a
warning and continues. That is deliberate: a command whose optional dependency
tree is absent (embeddings, SemantiK, bs4, torch) must not take the whole CLI
down — `ed4all doctor` has to stay reachable precisely when something is broken.
A new command follows the same shape: `@click.command(...)` +
`register_x_command(cli_group)` + an entry in `cli/commands/__init__.py.__all__`
+ a guarded registration block in `cli/main.py`.

`cli/main.py` also owns log-level control: `--verbose` (or `ED4ALL_LOG_LEVEL`,
name or number) attaches a stderr `basicConfig` handler. Without it the root
logger drops everything below WARNING, silencing the per-gate wall-clock /
NLI-throughput INFO instrumentation.

---

## Layout

```
cli/main.py             click group, --verbose/log-level, guarded registrations,
                        plus 7 commands defined inline (see table below)
cli/commands/           one module per command / command group
cli/validators/         run_validator.py   → RunValidator (validate-run)
cli/reporters/          run_summarizer.py  → RunSummarizer (summarize-run)
cli/comparators/        run_diff.py        → RunDiff (diff-runs)
cli/exporters/          training_exporter.py → TrainingExporter (export-training)
cli/tests/              21 test_*.py
```

Commands defined **inline in `cli/main.py`** (they only need `lib/` helpers, so
they carry no import risk): `validate-run`, `summarize-run`, `diff-runs`,
`export-training`, `fsck`, `list-runs`, `verify-chain`.

---

## Command surface

| Command / group | Module | Notes |
|---|---|---|
| `run` | `commands/run.py` | Canonical workflow entry point (below). |
| `stop [TARGET] \| --all \| --clear-all` | `commands/stop.py` | Writes a stop sentinel. Exactly one of target/`--all`/`--clear-all`; otherwise exit 2. |
| `objectives restructure` | `commands/objectives_cmd.py` | Deterministic (no-LLM) objectives rebuild; output round-trips `--reuse-objectives`. |
| `libv2 validate-packet \| query \| generate-quiz \| generate-study-pack \| ask` | `commands/libv2_*.py` | Five subcommands assembled onto one `libv2` group by `register_libv2_command`. |
| `tutor diagnose \| inventory \| guardrails` | `commands/tutor.py` | Misconception-aware tutoring tools over LibV2 archives. |
| `state prune` | `commands/state_prune.py` | GC for `state/runs` + `state/workflows`. |
| `mailbox watch` | `commands/mailbox_watch.py` | Outer-session watcher for the `LocalDispatcher` task mailbox. |
| `mailbox-bridge peek \| complete \| peek-agent \| complete-agent` | `commands/mailbox_bridge.py` | Hidden group (`hidden=True`); operator plumbing for the brokered LLM bridge. |
| `gui` | `commands/gui_cmd.py` | Launches the control-plane server; see [`gui/CLAUDE.md`](../gui/CLAUDE.md). |
| `doctor` | `commands/doctor.py` | Preflight/post-mortem. `-g/--group` repeatable; default groups `gpu`/`gpu_profile`/`window`/`environment` (`_DEFAULT_GROUPS`), `provider` is opt-in via `--run`/`--ping`/`-g provider`, `seat` (vLLM seat topology) is opt-in via `-g seat`/`--run`/a configured seat registry. The `environment` + `window` groups are topology-aware: on a vLLM-seat host (LOCAL_SYNTHESIS_BASE_URL → a registered seat) they probe `/v1/models` instead of the ollama `/api/tags` + `/api/show` (no false "model not pulled / served window unknown" DEGRADED). |
| `convert` | `commands/convert.py` | Thin PDF/HTML → `{stem}_accessible.html` slice; `--output` required. |
| `import-docs` | `commands/import_docs.py` | Deterministic Markdown/docs-tree → accessible-HTML corpus. |
| `harvest-bloom-labels` | `commands/harvest_bloom_labels.py` | No-LLM Bloom-label harvester. |
| `support-bundle` | `commands/support_bundle.py` | Redacted diagnostics tarball. |
| `backup` | `commands/backup.py` | Data-dir backup + `--verify`. |
| `validate-run` `summarize-run` `diff-runs` `export-training` `fsck` `list-runs` `verify-chain` | `cli/main.py` | Run-integrity + reporting verbs. |

---

## How `ed4all run` maps to a workflow

`cli/commands/run.py` is the single recommended entry point. Flow:

1. **Normalize** — `_normalize_workflow` lowercases and maps `-` → `_`, so
   `textbook-to-course` and `textbook_to_course` are the same workflow.
2. **Validate** — the normalized name must be in `SUPPORTED_WORKFLOWS`
   (`textbook_to_course`, `course_generation`, `rag_training`,
   `trainforge_train`, plus the four `courseforge*` stage aliases); otherwise
   exit 2 with the valid list.
3. **Stage aliasing** — a name in `COURSEFORGE_STAGE_SUBCOMMANDS`
   (`courseforge`, `courseforge_outline`, `courseforge_validate`,
   `courseforge_rewrite`) is recorded as the `courseforge_stage` param and the
   workflow is then rewritten to `textbook_to_course`. The runner reads
   `courseforge_stage` to decide which Phase 3 tier to re-execute while the
   others pre-populate from disk.
4. **Params** — `_build_workflow_params` assembles the params dict from the flags.
5. **Create** — `textbook_to_course` (incl. stage aliases) goes through
   `_create_textbook_workflow` → `MCP.tools.pipeline_tools.create_textbook_pipeline`;
   every other workflow goes through `_create_generic_workflow` →
   `MCP.tools.orchestrator_tools.create_workflow_impl`.
6. **Run** — `_build_orchestrator` constructs `MCP.orchestrator.PipelineOrchestrator`
   with the resolved mode + `BackendSpec`, then `await orchestrator.run(workflow_id)`.

Mode/provider resolution: `--mode` → `LLM_MODE` → `local`;
`--api-provider`/`--provider` → `LLM_PROVIDER` → `anthropic`.

`--dry-run` prints the planned phase sequence (`_dry_run_plan` /
`_print_dry_run_plan`) and, for the hosted-seat profile, runs
`_cloud_seat_preflight` — resolve + assert only, **no dispatch**.

`--resume <run_id>` takes the `_resume_workflow` path. `--stop-after`,
`--reuse-objectives` and `--with-training` have dedicated resume-override
helpers (`_apply_resume_stop_after_override`,
`_apply_resume_reuse_objectives_override`,
`_apply_resume_with_training_override`) that patch the persisted params before
the resumed phase runs — the runner reads those decisions from persisted state,
so without the patch the flag would be a silent no-op on a resume.
`_apply_resume_with_training_override` also takes the `--skip-training` value,
because `--skip-training` wins over `--with-training` on the resume path exactly
as it does on the creation path.

### Exit codes for `ed4all run`

| Code | Meaning |
|------|---------|
| 0 | Completed (no gate failure). |
| 1 | Failed / a critical validation gate blocked. |
| 2 | Usage error — unknown workflow, bad flag combination, invalid `--skip-conversion` inputs, malformed `--reuse-objectives`. |
| 3 | **Paused** at a checkpoint (graceful stop). `_paused_exit_code` returns 3 and prints a resume hint. A paused resume that pauses again is 3 again, never a failure. |

Exit 3 wins over the gate/status collapse — a graceful stop is not a failure.

### Graceful-stop signal handling

`_install_stop_signals` registers handlers for `SIGTERM` and `SIGINT` via
`signal.signal`, **not** `loop.add_signal_handler` (the latter raises
`NotImplementedError` on Windows event loops; precedent:
`cli/commands/mailbox_watch.py`). A second signal hard-kills, and
`_best_effort_mark_interrupted` stamps a still-`RUNNING` workflow `paused` so the
staleness heuristic and the resume path see the truth.

---

## Notable parse-time contracts in `run.py`

- **`VALID_BLOCK_TYPES`** — a flat tuple mirroring the canonical `BLOCK_TYPES`
  enum in `Courseforge/scripts/blocks.py`, held locally so `ed4all run --help`
  does not import the Courseforge renderer dependency tree. A regression test in
  `cli/tests/test_run_command.py` re-validates it against the canonical enum;
  **if you add a block type there, that test is the tripwire.**
- **Three additive rewrite-eviction scopes**, all consumed by the rewrite tier
  only and all stacking: `--blocks` → `target_block_ids` (block *type*),
  `--block-ids` → `target_block_instance_ids` (exact instance, shape
  `{page_id}#{block_type}_{slug}_{idx}`), `--pages` → `target_page_ids` (exact
  `page_id` or a module prefix). All three unset → byte-identical failure-driven
  cache reuse. An unknown id / unmatched page fails the rewrite phase **loudly** —
  never a silent no-op.
- **Fail-fast validation before state creation** — the `--skip-conversion`
  inputs and `--reuse-objectives` (`_validate_reuse_objectives_file`) are
  checked at parse time so a typo does not leave orphaned workflow state on
  disk. `--skip-conversion` keeps a `hidden=True` deprecated back-compat alias;
  the persisted skip run-param key is read-normalized so a run paused under the
  old flag name still `--resume`s.
- **`--semantik-output-dir`** is the canonical staged-HTML directory flag; a
  `hidden=True` deprecated back-compat alias is retained. Both coalesce into the
  same param — the canonical flag wins if both are somehow passed.

---

## Keeping the CLI and GUI in sync

`gui/services/run_service.py` carries its own `SUPPORTED_WORKFLOWS` and
`COURSEFORGE_STAGE_SUBCOMMANDS` mirroring the ones here. A workflow added to
`cli/commands/run.py` must be added there too, or it is launchable from the CLI
and rejected by the GUI.

---

## When changing this subsystem

- New command → guarded registration in `cli/main.py`, module in
  `cli/commands/`, export in `cli/commands/__init__.py`, test in `cli/tests/`.
- Keep heavy imports **inside** the command body (the inline commands in
  `main.py` all import their implementation lazily), so `ed4all --help` stays fast
  and a broken optional dep never blocks unrelated commands.
- New flag on `ed4all run` → thread it through `_build_workflow_params` and, if it
  must survive a resume, add a `_apply_resume_*_override` helper; validate it at
  parse time rather than mid-run.
- A flag selecting an LLM provider / model / synthesis backend also needs a row in
  [`docs/LICENSING.md`](../docs/LICENSING.md).
- No silent degradation: prefer a loud exit-2 usage error over a quiet fallback.
