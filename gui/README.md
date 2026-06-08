# Ed4All Control-Plane GUI

A no-stubs web control plane for the whole Ed4All pipeline. Every panel and
endpoint calls a real backend function and returns real data — there are no
fabricated successes. When a capability cannot be wired (e.g. ML deps absent for
adapter inference), the GUI surfaces a typed error rather than faking a result.

The GUI lets a human:

- **Upload a corpus and launch a run** — drag-drop PDFs / IMSCC, pick a full
  workflow (or a single phase), set course name + options, and watch the live
  log stream + validation-gate results.
- **Manage env / API keys** — every env knob the pipeline reads, rendered from a
  declarative catalog; secrets are write-only and masked on read.
- **Route models per task** — set provider + model independently for DART,
  Courseforge outline/rewrite, course planning, textbook synthesis, Trainforge
  synthesis/assessment, plus a vision/VLM lane (incl. local Ollama).
- **Author course topics / subjects / objectives** — edit terminal/chapter
  learning objectives and LibV2 classification; saved as a reuse-compatible
  `synthesized_objectives.json`.
- **Run retrieval + adapter inference** — BM25, multi-query, or LLM-rerank
  retrieval against a LibV2 course; run a trained adapter against a prompt.
- **Bridge with Claude Code** — a shared `state/gui/` store + an append-only
  activity log keep a Claude session and the GUI in sync in both directions.

The GUI ships as an opt-in `gui` extra — it adds no heavy deps to the default
install. Retrieval and adapter inference reuse the existing optional extras.

---

## Quick start

### One-click (recommended)

Go from a fresh checkout to the GUI in your browser with a single double-click —
no manual venv or `pip` steps. The launcher builds its own `.venv-gui/`,
installs the `gui` + `server` extras, starts the server on a free port, and
opens your browser.

| Platform | Do this |
|----------|---------|
| **macOS / Linux** | Double-click **`run-gui.sh`** (or `./run-gui.sh` from the repo root). |
| **Windows** | Double-click **`run-gui.bat`**. |
| **Any (manual launcher)** | `python3 gui/launch.py` from the repo root. |

Full launcher reference (flags `--host` / `--port` / `--no-browser` /
`--no-install` / `--reinstall`, requirements, troubleshooting): see
[`gui/LAUNCH.md`](LAUNCH.md).

### Manual (already have an environment)

If you manage your own virtualenv:

```sh
pip install -e '.[gui]'
ed4all gui                              # serves http://127.0.0.1:8077
ed4all gui --host 0.0.0.0 --port 9000   # bind elsewhere
ed4all gui --reload                     # uvicorn autoreload (dev only)
```

`ed4all gui` runs uvicorn against the FastAPI app factory `gui.app:create_app`.
You can also run the server module directly: `python -m gui.server [--host ...]
[--port ...] [--reload]`.

**Default URL:** <http://127.0.0.1:8077>. The SPA is served at `/`; the REST API
lives under `/api`; the run-log WebSocket is `/ws/runs/{run_id}`. A liveness
probe is `GET /api/health`.

---

## The six tabs

The frontend is a vanilla-JS single-page app (no build step, no framework). The
left nav has six tabs:

| Tab | What it does |
|-----|--------------|
| **Upload & Run** | Drag-drop a corpus (PDF / IMSCC) → pick a workflow *or* a single phase → set course name + options → Launch. A live log panel streams over WebSocket and a gate-results table shows per-phase pass/fail. |
| **Settings / API Keys** | Renders the credentials + global-routing knobs from the env catalog. Secret fields show `set` / `unset` only; saving writes the raw value. A "Test" button per provider runs a real reachability probe. |
| **Model Routing** | Per-task provider + model selectors (dropdowns sourced from the live provider registry, base-model list, and live Ollama model discovery). Includes the two-pass toggle and the vision/VLM lane. Maps to the `model_routing` block. |
| **Courses & Topics** | Course list (Courseforge exports + LibV2 courses). Edit terminal/chapter objectives, subjects, and tags; save writes a reuse-compatible objectives JSON. Edit LibV2 classification where a manifest exists. |
| **Retrieval** | Course picker, query box, and a mode switch (BM25 / multi / LLM-rerank). Results show scores + per-item rationale. An adapter picker + prompt runs real adapter inference. |
| **Activity (Claude ↔ GUI)** | Tails `events.jsonl`. Post a message to Claude (writes a `gui`-sourced event); see `claude`-sourced events. This is the human-visible side of the Claude↔GUI bridge. |

---

## Settings & API keys

### Where settings persist

The canonical settings doc is `state/gui/settings.json` (versioned, schema v1).
It carries four sections:

- `env` — raw env-var values (including secrets).
- `model_routing` — per-task `{provider, model[, mode][, vision_provider]}`.
- `retrieval` — `top_k`, `min_grounding_cosine`, `require_embeddings`.
- `flags` — boolean behavior flags (`COURSEFORGE_TWO_PASS`,
  `DART_LLM_CLASSIFICATION`, `TRAINFORGE_REQUIRE_EMBEDDINGS`).

`gui/settings_store.py` is the single source of truth: `load_settings()`
overlays the on-disk doc onto catalog defaults; `save_settings()` /
`update_settings()` validate (unknown provider, bad `LLM_MODE` → `ValueError`)
and atomically persist. The settings UI never hardcodes knobs — it renders from
the declarative catalog in `gui/env_catalog.py`.

### Secret masking (write-only)

Secrets (`*_API_KEY` / `*_KEY`) are **write-only**. Every GET response runs
through `mask_secrets()`, which replaces a populated secret with the literal
string `"set"` and an empty one with `null`. The raw value is never returned by
any endpoint, and the `.env.rendered` artifact masks secrets too.

### How env gets applied

`apply_env(doc)` renders the settings doc to an env-var map and writes it into
`os.environ` for the current process, so any run launched afterwards inherits
the configuration. Rendering precedence (later wins):

1. `flags` → boolean env vars (`"true"` / `"false"`).
2. `retrieval.require_embeddings` → `TRAINFORGE_REQUIRE_EMBEDDINGS`.
3. `model_routing.<task>.<field>` → the canonical env var (via
   `env_catalog.routing_to_env`).
4. `env` → raw passthrough of catalog-known keys.

Apply also writes `state/gui/.env.rendered` — an informational dotenv-style
dump of the resolved vars (secrets masked). It is **not** sourced by anything;
the backend applies vars in-process. It exists so an operator can see exactly
what was injected.

A launch additionally overlays per-request `mode` / `provider` / `model` into
`os.environ` (mapped to `LLM_MODE` / `LLM_PROVIDER` / `LLM_MODEL`) so a one-off
run can override the saved defaults without mutating `settings.json`.

---

## Model routing (incl. VLM + Ollama)

The provider registry mirrors `_OPENAI_COMPATIBLE_PROVIDERS` in
`MCP/orchestrator/llm_backend.py` plus the native `anthropic` and `mock`
backends. No providers are invented; adding one is a registry change there,
mirrored in `env_catalog.PROVIDERS`. Current providers:

| Provider | Label | API key env | Default base URL | Vision |
|----------|-------|-------------|------------------|:------:|
| `anthropic` | Anthropic (Claude) | `ANTHROPIC_API_KEY` | — | ✓ |
| `local` | Ollama (local) | `LOCAL_SYNTHESIS_API_KEY` (optional) | `http://localhost:11434/v1` | ✓¹ |
| `together` | Together AI (text) | `TOGETHER_API_KEY` | `https://api.together.xyz/v1` | — |
| `together-vision` | Together AI (vision) | `TOGETHER_API_KEY` | `https://api.together.xyz/v1` | ✓ |
| `groq` | Groq | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` | — |
| `fireworks` | Fireworks | `FIREWORKS_API_KEY` | `https://api.fireworks.ai/inference/v1` | — |
| `deepseek` | DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/v1` | — |
| `mock` | Mock (no network) | — | — | — |

`groq` / `fireworks` / `deepseek` are flagged `unverified` in the registry.

Per-task routing slots (each → its canonical env var on render): `global`
(`LLM_MODE` / `LLM_PROVIDER` / `LLM_MODEL`), `dart`, `vision`,
`courseforge_outline`, `courseforge_rewrite`, `courseplanner`,
`textbook_synthesis`, `trainforge_synthesis`, `trainforge_assessment`.

### Ollama = the `local` provider

The `local` provider **is Ollama** by default: its base URL points at Ollama's
OpenAI-compatible port (`http://localhost:11434/v1`). The registry name stays
`local` because DART / Trainforge / the provider resolver all key on that
literal; the GUI shows the human label "Ollama (local)". Point
`LOCAL_SYNTHESIS_BASE_URL` at any OpenAI-compatible server (vLLM / llama.cpp /
LM Studio) to use it instead.

**Live model discovery:** `GET /api/settings/ollama-models` issues a real
`GET <host>/api/tags` against the local server (stripping the `/v1` OpenAI-compat
suffix to reach Ollama's native tags API) and returns the installed model names.
This powers the live model dropdowns in Model Routing — a real probe, never a
hardcoded list. It degrades gracefully (`available: false`, empty `models`) when
Ollama is offline.

### Vision / VLM

The vision lane drives DART's image / alt-text calls. The vision provider
dropdown is filtered to vision-capable backends only: `anthropic`,
`together-vision`, and `local` (Ollama with a vision model). Selecting a
vision model:

- **`together-vision`** — set the model; it renders to `TOGETHER_VISION_MODEL`.
- **`local` (Ollama)** — set `LOCAL_SYNTHESIS_MODEL` to a vision model (e.g.
  `llama3.2-vision`, `llava:13b`, `qwen2.5-vl`) **and** flip
  `LOCAL_VISION_CAPABLE=true`. There is no separate `LOCAL_VISION_MODEL` var.
- **`anthropic`** — the model comes from `DART_CLAUDE_MODEL` / `LLM_MODEL`.

`vision.provider` maps to `DART_VISION_PROVIDER` (the real DART knob).
`vision.model` is provider-conditional and only renders a dedicated env var for
`together-vision`.

### Provider reachability test

`POST /api/settings/test-provider` runs a **real** probe per provider family:

- `anthropic` — key presence, then a minimal 1-token ping when `LLM_MODE=api`
  and the `anthropic` package is importable; otherwise key-presence only.
- `local` — HTTP `GET <base_url>/models` with a short timeout; `ok` on HTTP 200.
- `together` / `groq` / `fireworks` / `deepseek` — API-key-presence only (no
  billable call).
- `mock` — always reachable.

Returns `{provider, ok, status, detail}`.

---

## REST API reference

All routes are JSON in/out under `/api`, except the WebSocket. Errors return a
typed `{error, detail}` body (uploads/runs use a `{detail: {error, detail}}`
HTTPException shape) with the right status code: 422 validation, 503 deps
missing / backend unavailable, 404 not found, 500 unexpected. Never a fabricated
success.

### Settings — `/api/settings`

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/settings` | — | masked settings doc + `catalog` + `providers` + `base_models` |
| PUT | `/api/settings` | full settings doc | masked stored doc (422 on invalid) |
| PATCH | `/api/settings` | partial patch (`env` / `flags` / `model_routing` / `retrieval`) | masked stored doc (deep-merged) |
| POST | `/api/settings/apply` | — | `{applied: [<env key names>]}` (names only, never values) |
| GET | `/api/settings/ollama-models` | — | `{available, models, detail, host}` (live Ollama probe) |
| POST | `/api/settings/test-provider` | `{provider}` | `{provider, ok, status, detail}` (real probe) |

### Uploads — `/api/uploads`

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `/api/uploads` | multipart `files` (`.pdf` / `.imscc`) | `{upload_id, files: [{name, path, size, kind}]}` |
| GET | `/api/uploads` | — | `{uploads: [{upload_id, files: [...]}]}` |
| DELETE | `/api/uploads/{upload_id}` | — | `{deleted: <upload_id>}` |

Saved file paths feed `--corpus` on a launch. Extensions and resolved
destinations are validated (path-traversal guard); a rejected batch leaves no
partial directory.

### Runs + workflows — mounted at `/api`

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/workflows` | — | `{workflows: [{name, description, phases:[{name, agents, depends_on, validation_gates_count, optional}], stage_subcommand}]}` |
| POST | `/api/runs` | `{workflow, course_name, corpus?, weeks?, mode?, provider?, model?, options{}}` | `{run_id, workflow_id, status}` (422 on launch failure) |
| POST | `/api/runs/phase` | `{workflow, phase, course_name?, project_id?, mode?, provider?, model?, options{}}` | `{run_id, status, tasks?, gate_results?}` (422 on failure) |
| GET | `/api/runs` | — | `{runs: [<run record>...]}` (newest first) |
| GET | `/api/runs/{run_id}` | — | the run record (404 if unknown) |
| POST | `/api/runs/{run_id}/cancel` | — | `{run_id, status}` (404 if unknown) |

The launchable workflows are `textbook_to_course`, `course_generation`,
`intake_remediation`, `batch_dart`, `rag_training`, `trainforge_train`, plus the
Courseforge stage subcommands (`courseforge`, `courseforge_outline`,
`courseforge_validate`, `courseforge_rewrite`) surfaced as
`textbook_to_course` aliases.

### Run log WebSocket

| Path | Frames |
|------|--------|
| `WS /ws/runs/{run_id}` | `{type:"line", line}` per log chunk; `{type:"status", status, gates, error}` at a terminal state, then close; `{type:"error", ...}` if the run is unknown. |

The socket tails the run's log file by byte offset and polls the run record
until it reaches `completed` / `failed` / `cancelled`.

### Courses — `/api/courses`

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/courses` | — | `[{course_name, project_id, slug, duration_weeks, terminal_count, chapter_count, topics, subdomains}]` |
| GET | `/api/courses/{id}` | — | one course summary (404 if unknown; resolves by project_id / course_name / slug) |
| GET | `/api/courses/{id}/objectives` | — | objectives doc (terminal/chapter normalized to `id` + `text`) |
| PUT | `/api/courses/{id}/objectives` | `{terminal_objectives:[{id,text,...}], chapter_objectives:[...], mint_method?, duration_weeks?}` | reuse-compatible `synthesized_objectives.json` (422 invalid LO id, 404 no export) |
| GET | `/api/courses/{id}/classification` | — | LibV2 manifest classification block (404 if no manifest) |
| PATCH | `/api/courses/{id}/classification` | `{division?, primary_domain?, secondary_domains?, subdomains?, topics?, subtopics?, tags?}` | merged classification (422 schema violation, 404 no manifest) |

Saving objectives stamps `mint_method="user_supplied_objectives_json"` so
`ed4all run --reuse-objectives` accepts the result. LO ids are validated against
the canonical `TO-NN` / `CO-NN` pattern via `lib.ontology.learning_objectives`.

### Retrieval — `/api/retrieval`

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/retrieval/courses` | — | `[{slug, chunk_count}]` (LibV2 courses) |
| POST | `/api/retrieval/query` | `{slug, query, top_k=5, filters{}, mode}` (`mode` ∈ `bm25` / `multi` / `llm_rerank`) | `{results}` (bm25 / llm_rerank) or `{results, decomposition}` (multi) |
| GET | `/api/retrieval/{slug}/adapters` | — | `[{model_id, base_model?, eval?}]` |
| POST | `/api/retrieval/{slug}/infer` | `{model_id, prompt, max_new_tokens=256}` | `{generation, base_model}` (503 `training_deps_missing` when ML deps absent) |

`mode=llm_rerank` retrieves BM25 candidates then runs a real LLM rerank using
the active settings provider/model, returning a reranked order with per-item
`rationale`. `top_k` is capped at 50. Adapter inference lazy-imports the
training stack; a missing-deps state returns a typed 503, never a fabricated
generation.

### Activity bridge — `/api/activity`

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/activity/events` | `?since=<seq>` | `{events: [...]}` with `seq >= since` |
| POST | `/api/activity/post` | `{kind, payload{}}` | `{event: <stored record>}` (always `source="gui"`) |

### Health

| Method | Path | Response |
|--------|------|----------|
| GET | `/api/health` | `{status: "ok"}` |

---

## Claude Code integration

The GUI and a Claude Code session share a single on-disk store, `state/gui/`,
written via the web-dep-free `gui/shared_state.py`. The same store backs nine
`gui_*` MCP tools (`MCP/tools/gui_tools.py`, registered in `MCP/server.py`). A
Claude session drives and observes exactly what the GUI shows, and vice versa —
symmetric to the existing StatusTracker / TaskMailbox file-IPC idiom.

```
state/gui/
├── settings.json     # canonical settings doc (gui_get_settings / gui_set_setting)
├── .env.rendered     # informational rendered env (secrets masked)
├── runs/             # run registry: <run_id>.json (gui_list_runs / gui_get_run / gui_enqueue_run)
├── logs/             # <run_id>.log (tailed by the WebSocket)
├── uploads/          # uploaded corpora (one dir per upload_id)
└── events.jsonl      # append-only Claude ↔ GUI bridge (gui_post_event / gui_read_events)
```

The nine MCP tools:

| Tool | What it does |
|------|--------------|
| `gui_get_settings()` | Return the masked settings doc (what the Settings tab shows). |
| `gui_set_setting(path, value)` | Deep-patch one setting at a dotted path (e.g. `model_routing.global.provider`); returns the masked doc. |
| `gui_list_runs()` | List the run registry (newest first). |
| `gui_get_run(run_id)` | Return one run record. |
| `gui_enqueue_run(workflow, course_name, corpus, options_json)` | Write a run request with `status="requested"` for the GUI side to pick up. |
| `gui_list_courses()` | List Courseforge-export + LibV2 courses (lazy course-service import). |
| `gui_get_objectives(course)` | Return a course's synthesized objectives doc. |
| `gui_post_event(kind, payload_json)` | Append a `claude`-sourced event (shows in the Activity tab). |
| `gui_read_events(since)` | Read events with `seq >= since` (so Claude sees human messages). |

The tool module imports only `gui.shared_state` + `gui.settings_store` at module
level (no FastAPI), so it is import-safe without the `gui` extra; the two course
tools lazy-import the course service and return a typed
`course_service_unavailable` error if it can't load.

**Bidirectional sync, in practice:** a `gui_set_setting` from Claude changes
what the GUI's Settings tab renders on its next poll. A `gui_post_event` from
Claude appears in the Activity tab; a human message posted from the Activity tab
(`POST /api/activity/post`, `source="gui"`) is visible to `gui_read_events`.
`gui_enqueue_run` writes a `status="requested"` record for the GUI to launch.

---

## Architecture note (no stubs)

The services layer wraps the **same** backend functions the `ed4all run` CLI
drives — nothing is re-implemented:

- **Workflow launch** — `run_service.launch_pipeline` applies the settings env,
  then calls `create_textbook_pipeline` (textbook_to_course + Courseforge stage
  aliases) or `create_workflow_impl` (other workflows). A real workflow is
  created under `state/workflows/` and driven by `PipelineOrchestrator.run` in a
  background asyncio task, streaming status/log to `state/gui/logs/<run_id>.log`.
- **Single phase** — `run_service.launch_phase` uses the documented phase
  pathway: `OrchestratorConfig` → pick the phase → `WorkflowRunner._route_params`
  → `_create_phase_tasks` → `TaskExecutor.execute_phase`, with upstream
  `phase_outputs` pre-populated from an existing export via
  `_synthesize_outline_output`.
- **Workflow list** — read from `OrchestratorConfig.load()` (the real
  `config/workflows.yaml`), never a hardcoded list.
- **Retrieval** — `retrieval_service` wraps `retrieve_chunks` /
  `MultiRetriever` and reads adapter `model_card.json` + `eval_report.json`.
- **Adapter inference** — lazy-imports `Trainforge.eval.adapter_callable.
  AdapterCallable`; a missing-deps state returns a typed 503.

Decision capture is unaffected: the GUI sets env / params and launches the same
code paths, so existing `DecisionCapture` wiring stays intact.

---

## Troubleshooting

- **Retrieval returns 0 BM25 hits (known data caveat).** An empty
  `imscc_chunks/` directory can *shadow* a legacy `corpus/chunks.jsonl`: the
  chunk-path resolver prefers `imscc_chunks/` and, if it exists but is empty,
  retrieval finds nothing. Fix by either running the backfill —
  `python LibV2/tools/libv2/scripts/backfill_dart_chunks.py` — or removing the
  empty `imscc_chunks/` directory so the legacy `corpus/chunks.jsonl` resolves.
- **Local providers fail / Ollama models don't list.** Ollama (or whichever
  OpenAI-compatible server `LOCAL_SYNTHESIS_BASE_URL` names) must be running with
  the relevant model pulled. Use Settings → "Test" (`local`) for a live
  `GET /models` probe, and the Model Routing dropdowns reflect
  `GET /api/settings/ollama-models`.
- **Adapter inference returns 503 `training_deps_missing`.** The training extras
  aren't installed in the GUI's environment. Install them (or run inference from
  an environment that has the Trainforge eval stack). The 503 is intentional —
  the GUI never fakes a generation.
- **`gui` extra missing.** `ed4all gui` raises an actionable error
  (`pip install 'ed4all[gui]'`). The one-click launcher installs it for you.
- **Port in use.** The one-click launcher auto-advances to the next free port;
  with `ed4all gui` pass `--port <n>`.

The launcher-specific troubleshooting (venv, `python3-venv`, browser open) lives
in [`gui/LAUNCH.md`](LAUNCH.md).
