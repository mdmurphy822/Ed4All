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
- **Route models per task** — set provider + model independently for the
  SemantiK conversion lane, Courseforge outline/rewrite, course planning,
  textbook synthesis, Trainforge synthesis/assessment, plus a vision/VLM lane
  (incl. a local OpenAI-compatible server — vLLM, Ollama, llama.cpp).
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

### Docker (deployable stack)

For a containerized deploy that serves Studio on `:8077` with a local Ollama
backend and a single `/data` volume for all course/state data, see
[`docs/operations/docker.md`](../docs/operations/docker.md) — `docker compose up`
builds the GUI image (`Dockerfile.gui`), pull the answer model once, open the
browser.

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

The bind address resolves from two env vars (CLI flags override them when both
are passed): `ED4ALL_GUI_HOST` (default `127.0.0.1`) and `ED4ALL_GUI_PORT`
(default `8077`). Set them to relocate the server without passing `--host` /
`--port` on every launch — e.g. `ED4ALL_GUI_HOST=0.0.0.0 ED4ALL_GUI_PORT=9000
ed4all gui` to bind all interfaces on port 9000.

**Default URL:** <http://127.0.0.1:8077>. The SPA is served at `/`; the REST API
lives under `/api`; the run-log WebSocket is `/api/ws/runs/{run_id}`. A liveness
probe is `GET /api/health`.

---

## The six tabs

The frontend is a vanilla-JS single-page app (no build step, no framework). The
left nav has six tabs:

| Tab | What it does |
|-----|--------------|
| **Upload & Run** | Drag-drop a corpus (PDF / IMSCC) → pick a workflow *or* a single phase → set course name + options → Launch. A live log panel streams over WebSocket and a gate-results table shows per-phase pass/fail. |
| **Settings / API Keys** | Renders the credentials + global-routing knobs from the env catalog. Secret fields show `set` / `unset` only; saving writes the raw value. A "Test" button per provider runs a real reachability probe. |
| **Model Routing** | Per-task provider + model selectors (dropdowns sourced from the live provider registry, base-model list, and live local model-server discovery). Includes the two-pass toggle and the vision/VLM lane. Maps to the `model_routing` block. |
| **Courses & Topics** | Course list (Courseforge exports + LibV2 courses). Edit terminal/chapter objectives, subjects, and tags; save writes a reuse-compatible objectives JSON. Edit LibV2 classification where a manifest exists. |
| **Retrieval** | Course picker, query box, and a mode switch (BM25 / multi / LLM-rerank). Results show scores + per-item rationale. An adapter picker + prompt runs real adapter inference. |
| **Activity (Claude ↔ GUI)** | Tails `events.jsonl`. Post a message to Claude (writes a `gui`-sourced event); see `claude`-sourced events. This is the human-visible side of the Claude↔GUI bridge. |

---

## Settings & API keys

### Where settings persist

The canonical settings doc is `state/gui/settings.json` (versioned, schema v1).
It carries five sections:

- `env` — raw env-var values (including secrets).
- `model_routing` — per-task `{provider, model[, mode][, vision_provider]}`.
- `retrieval` — `top_k`, `min_grounding_cosine`, `require_embeddings`.
- `flags` — boolean behavior flags (e.g. `COURSEFORGE_TWO_PASS`,
  `TRAINFORGE_REQUIRE_EMBEDDINGS`).
- `assistant` — the Studio Assistant chat panel: `autostart` (bool, default
  `false`) governs whether a chat request may *attempt* the engine's lazy
  seat-start path when the local assistant seat is down (the engine's own
  `ED4ALL_ASSISTANT_AUTOSTART` policy still decides whether a start actually
  occurs). Patchable via the usual deep-patch paths, e.g.
  `gui_set_setting("assistant.autostart", true)` from the MCP bridge.

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

### Blessed authoring route (turnkey, no Claude session)

A GUI-launched run is **headless** — there is no Claude Code session draining
the orchestrator mailbox. Every phase that needs LLM generation
(content-generator, course-outliner, assessment-generator,
training-synthesizer) must therefore resolve its generation through the
**in-process provider lattice** (the OpenAI-compatible provider registry),
never through a Claude-session subagent. To guarantee this, `launch_pipeline`
/ `launch_phase` set each agent's provider env
(`COURSEFORGE_PROVIDER` / `COURSEPLANNER_PROVIDER` /
`TRAINFORGE_ASSESSMENT_PROVIDER` / `TRAINFORGE_SYNTHESIS_PROVIDER`) by default
for the enqueued run. Resolution per env var (only when not already set via
`model_routing`): per-request `provider` → global `LLM_PROVIDER` → `local`
(an air-gapped Ollama/vLLM provider that needs no key). A per-task provider
you configured in **Model Routing** is preserved — it is never overwritten by
this default.

If a launched run somehow reaches a phase that would enqueue a mailbox
`agent_task` with no provider env (and no servicer attached), the orchestrator
**fails fast** with an actionable error
(`AuthoringProviderRouteError`) instead of hanging forever — the message names
the exact env var to set, or tells you to run inside a Claude session with
`ED4ALL_AGENT_DISPATCH=true` + `ED4ALL_MAILBOX_SERVICED=1`. This is a
guardrail, not the normal path: with the blessed-route defaulting above, a GUI
run never trips it.

---

## Model routing (incl. VLM + local model server)

The provider registry mirrors `_OPENAI_COMPATIBLE_PROVIDERS` in
`MCP/orchestrator/llm_backend.py` plus the native `anthropic` and `mock`
backends. No providers are invented; adding one is a registry change there,
mirrored in `env_catalog.PROVIDERS`. Current providers:

| Provider | Label | API key env | Default base URL | Vision |
|----------|-------|-------------|------------------|:------:|
| `anthropic` | Anthropic (Claude) | `ANTHROPIC_API_KEY` | — | ✓ |
| `local` | Local model server (OpenAI-compatible: TRT-LLM, vLLM, llama.cpp) | `LOCAL_SYNTHESIS_API_KEY` (optional) | `http://localhost:8000/v1` | ✓¹ |
| `together` | Together AI (text) | `TOGETHER_API_KEY` | `https://api.together.xyz/v1` | — |
| `together-vision` | Together AI (vision) | `TOGETHER_API_KEY` | `https://api.together.xyz/v1` | ✓ |
| `groq` | Groq | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` | — |
| `fireworks` | Fireworks | `FIREWORKS_API_KEY` | `https://api.fireworks.ai/inference/v1` | — |
| `deepseek` | DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/v1` | — |
| `mock` | Mock (no network) | — | — | — |

`groq` / `fireworks` / `deepseek` are flagged `unverified` in the registry.

Per-task routing slots (each → its canonical env var on render): `global`
(`LLM_MODE` / `LLM_PROVIDER` / `LLM_MODEL`), the SemantiK conversion lane,
`vision`, `courseforge_outline`, `courseforge_rewrite`, `courseplanner`,
`textbook_synthesis`, `trainforge_synthesis`, `trainforge_assessment`.

### The `local` provider = any local OpenAI-compatible server

The `local` provider is a strict OpenAI-compatible model server — TRT-LLM,
vLLM, llama.cpp, or LM Studio. Its default base URL is the canonical local
seat at `http://localhost:8000/v1`; there is no implicit Ollama endpoint or
lifecycle fallback. Point `LOCAL_SYNTHESIS_BASE_URL` at another compatible
seat when needed. The registry name
stays `local` because SemantiK / Trainforge / the provider resolver all key on
that literal; the GUI shows the human label "Local model server (OpenAI-compatible:
TRT-LLM, vLLM, llama.cpp)".

**Live model discovery (protocol-first):** `GET /api/settings/local-models`
probes the OpenAI-compatible `GET <root>/v1/models` endpoint FIRST (vLLM, Ollama,
llama.cpp, and LM Studio all serve it) and falls back to Ollama's native
`GET <root>/api/tags` only when the OpenAI path fails or returns nothing. The
`/v1` suffix on `LOCAL_SYNTHESIS_BASE_URL` is detected and never doubled into
`/v1/v1/models`. The response is `{available, models, detail, host, backend}`,
where `backend` names what actually answered (`openai-compatible` | `ollama` |
`null`). This powers the live model dropdowns in Model Routing — a real probe,
never a hardcoded list. It degrades gracefully (`available: false`, empty
`models`, `backend: null`, vendor-neutral `detail`) when the local server is
offline.

> **Deprecated alias:** `GET /api/settings/ollama-models` still resolves and
> delegates to the same protocol-first `list_local_models` implementation, for
> backward compatibility. Prefer `/api/settings/local-models`.

### Vision / VLM

The vision lane drives the SemantiK conversion engine's image / alt-text calls.
The vision provider dropdown is filtered to vision-capable backends only:
`anthropic`,
`together-vision`, and `local` (a local OpenAI-compatible vision seat with a
vision model — e.g. Ollama or vLLM). Selecting a vision model:

- **`together-vision`** — set the model; it renders to `TOGETHER_VISION_MODEL`.
- **`local`** — set `LOCAL_SYNTHESIS_MODEL` to a vision model (e.g.
  `qwen2.5-vl`, `llama3.2-vision`, `llava:13b`) **and** flip
  `LOCAL_VISION_CAPABLE=true`. There is no separate `LOCAL_VISION_MODEL` var.
- **`anthropic`** — the model comes from the SemantiK conversion lane's
  Anthropic model env knob (falling through to `LLM_MODEL`).

`vision.provider` maps to the SemantiK conversion engine's vision-provider env
knob (the real conversion-lane vision selector). `vision.model` is
provider-conditional and only renders a dedicated env var for `together-vision`.

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
| GET | `/api/settings/local-models` | — | `{available, models, detail, host, backend}` (protocol-first local model-server probe: `/v1/models` then Ollama `/api/tags`) |
| GET | `/api/settings/ollama-models` | — | **DEPRECATED** alias for `/local-models` (same protocol-first response) |
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
| GET | `/api/runs/{run_id}` | — | the run record (404 if unknown). On a failed run the record carries `failed_phase` + `failure_reason`. |
| GET | `/api/runs/{run_id}/progress` | — | Stage-tracker payload for the pipeline rail + live stats band: `{run_id, workflow_id, workflow, status, phases:[{name, index, state, group, label, wallclock_s, telemetry?}], current_phase, failed_phase, failure_reason, stats:{tok_s, streams?, calls, prompt_tokens, completion_tokens, ttft_p50_ms, phase_elapsed_s, phase_telemetry?, seat:{name, url, model}\|null}, updated_at}`. Phase `state` is `done \| current \| paused \| pending \| failed \| skipped`; the phase list is the run's own `config/workflows.yaml` plan (workflow-agnostic, cached — when the plan carries both sides of an `enabled_when_env` branch, the not-taken side is omitted from the list: a branchy row that resolved skipped, or the negative-clause fallback row with no evidence of running yet), merged with the orchestrator workflow state, checkpoint wall-clocks, optional versioned atomic `state/runs/<run_id>/telemetry/<phase>.json` snapshots, and a bounded tail of the OP2 `llm_usage.jsonl` tap. Telemetry is run/phase/schema/type validated fail-safe; invalid documents are omitted, while valid completed metrics remain visible after the phase advances. (`tok_s` is the AGGREGATE seat throughput — sliding-window completion tokens over the window's wall span; `streams` is the estimated in-flight request count over that window, omitted when not computable; the per-stream figure is `stats.detail.throughput.per_stream_tok_s`; nulls when no usage yet). `seat` is a TTL-cached `/v1/models` probe over the `ED4ALL_SEAT_BASE_URLS` registry (null when no seat registry / terminal run). Cheap enough to poll at 2–5s. Accepts a GUI run id or a bare orchestrator workflow id; typed 404 otherwise. |
| GET | `/api/runs/{run_id}/output-tail` | — | Live-output tail for the run's CURRENT phase (the Studio "Live output" panel): `{run_id, phase, source, label, row_count, rows:[{seq, label, text}]}` — the last ≤15 complete rows of the phase's per-unit resume-checkpoint sidecar (or, for `heading_judge`, the newest per-chapter judgment files — a growing-directory source; `training_synthesis` tails its per-pair `training_specs/.synthesis_pairs_checkpoint.jsonl`), each mapped to a bounded display record (`text` HTML-stripped + truncated to ~500 chars with a trailing `…`; rows without a content field render a compact JSON of their payload minus fingerprint/schema bookkeeping). Bounded seek-from-end read (~2 MB cap); corrupt / mid-append partial rows are skipped; absent sidecar or unmapped phase → honest `rows: []` with `source: null`. Typed 404 when the run is unknown. |
| GET | `/api/runs/{run_id}/validation-report` | — | `{run_id, report, report_path, failed_gates:[{phase, gate_id, severity, message, issues_count}], failed_phase, failure_reason}`. `report` is the `courseforge_validation_report.json` body when present, else `null` with an explanatory `note`. 404 if the run is unknown. |
| POST | `/api/runs/{run_id}/cancel` | — | `{run_id, status}` (404 if unknown) |

The launchable workflows are `textbook_to_course`, `course_generation`,
`rag_training`, `trainforge_train`, plus the
Courseforge stage subcommands (`courseforge`, `courseforge_outline`,
`courseforge_validate`, `courseforge_rewrite`) surfaced as
`textbook_to_course` aliases.

### Run log WebSocket

| Path | Frames |
|------|--------|
| `WS /api/ws/runs/{run_id}` | `{type:"line", line}` per log chunk; `{type:"status", status, gates, error}` at a terminal state, then close; `{type:"error", ...}` if the run is unknown. |

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
| GET | `/api/courses/{id}/attestation` | — | the LibV2 human-review **attestation** block (empty `{}` for an un-attested course so the SPA can render the "not yet reviewed" affordance; 404 only when no manifest exists) |
| PATCH | `/api/courses/{id}/attestation` | `{reviewed_by, scope, ...}` | schema-validated attestation stamped onto the manifest (`reviewed_at` server-stamped when omitted); 422 schema violation, 404 no manifest. Backs the Studio **attestation editor**. |

Saving objectives stamps `mint_method="user_supplied_objectives_json"` so
`ed4all run --reuse-objectives` accepts the result. LO ids are validated against
the canonical `TO-NN` / `CO-NN` pattern via `lib.ontology.learning_objectives`.

### Retrieval — `/api/retrieval`

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/retrieval/courses` | — | `[{slug, chunk_count}]` (LibV2 courses) |
| QUERY / POST | `/api/retrieval/query` | `{slug, query, top_k=5, filters{}, mode}` (`mode` ∈ `bm25` / `multi` / `llm_rerank`) | `{results}` (bm25 / llm_rerank) or `{results, decomposition}` (multi) |
| GET | `/api/retrieval/{slug}/adapters` | — | `[{model_id, base_model?, eval?}]` |
| POST | `/api/retrieval/{slug}/infer` | `{model_id, prompt, max_new_tokens=256}` | `{generation, base_model}` (503 `training_deps_missing` when ML deps absent) |

`mode=llm_rerank` retrieves BM25 candidates then runs a real LLM rerank using
the active settings provider/model, returning a reranked order with per-item
`rationale`. `top_k` is capped at 50. Adapter inference lazy-imports the
training stack; a missing-deps state returns a typed 503, never a fabricated
generation.

**HTTP method — QUERY (canonical) + POST (deprecated alias).** The retrieval
query is *safe and idempotent* yet needs a request body, so its canonical method
is the IETF **QUERY** method ([RFC 10008](https://www.rfc-editor.org/rfc/rfc10008),
"The HTTP QUERY Method", Standards Track, June 2026). **POST** remains a
back-compat alias that returns a `Deprecation: true` response header (plus a
`Link; rel="successor-version"` pointing at the QUERY method); QUERY responses
carry neither. The GUI SPA sends QUERY and transparently downgrades **once** to
POST for the session if a client/proxy rejects the method. QUERY is a
non-safelisted method, so a **cross-origin** caller incurs a CORS preflight (the
open `allow_methods=["*"]` policy already permits it); the **same-origin** SPA is
unaffected. `GET`/other methods on this path return 405.

### Assistant — `/api/assistant`

The Studio **Assistant** panel's backend — a pure chat front-end over the
sandboxed `lib.assistant.AssistantEngine` (loopback-only local seat, typed tool
whitelist, 6-round tool cap, decision capture). The GUI adds no capability on
top of the engine. The engine is built lazily and cached per server process;
every request is stateless — the conversation `history` round-trips through
the client.

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `/api/assistant/chat` | `{message, history?, debug?, run_id?}` (`history` = the previous response's `history`, or `null` for a fresh conversation; `debug: true` + optional `run_id` run the turn on the debug-mode engine) | `{reply, history, tool_calls: [{tool, arguments, result}], rounds}` (+ `debug: {run_id, failed_phase, summary}` on debug turns) |
| GET | `/api/assistant/debug-context` | optional `?run_id=WF-YYYYMMDD-xxxxxxxx` | `{available: bool, run_id, course, failed_phase, summary}` — 404-free: no failed run → `available: false` + `reason` |
| GET | `/api/assistant/status` | — | `{seat_serving: bool, model: str\|null}` (a misconfigured seat adds a `detail` string) — powers the panel's status pill |
| GET | `/api/assistant/seat/status` | — | `{live_seat: str\|null, seats: {name: bool}, starting: bool, last_seat_error: str\|null}` (a misconfigured seat adds a `detail` string) — powers the "Seat model" button |
| POST | `/api/assistant/seat` | — (no body) | `{ok: true, state: "starting", seat, error: null}` — starts the assistant's own seat in the background |

**Debug mode** consumes `lib.assistant.debug_context.build_debug_context`: the
summarized, secret-filtered failed-run snapshot (run report + failing-phase
gate report + log tail + doctor post-mortem, ≤ ~6k chars) for a given `WF-` id
or the auto-resolved most recent failure. The server builds it **once per
failed run per process** (the probe and the debug conversation share the
cache) and constructs an `AssistantEngine(mode="debug", debug_context=...)`
for those requests — same tool whitelist, no extra capability. The client
re-sends `debug` + `run_id` with the round-tripped `history` on every turn of
a debug conversation.

Errors are typed `{error, detail}`: **503 `assistant_seat_unavailable`** when
the local assistant seat is down — `detail` carries the seat-start
instructions verbatim (when the `assistant.autostart` setting is true the
server attempts the engine's `ensure_seat` lazy-start once before failing);
**500 `assistant_config_error`** when the resolved seat URL is non-loopback
(the engine's construction-time guard); **422 `empty_message`** on a blank
message and **422 `invalid_run_id`** on a malformed `run_id` (both endpoints);
**409 `debug_context_unavailable`** when a debug chat is requested but no
failed run resolves. Each exchange also appends one `assistant`-sourced event
to the activity bridge (see below) so assistant usage shows in the Activity
tab (debug exchanges are marked `mode: "debug"`).

**Seat model** (`POST /api/assistant/seat`) is the panel's explicit
human-triggered seat start. It seats the assistant's **own** seat — the logical
name from `ED4ALL_ASSISTANT_SEAT` — through exactly the lifecycle path the
engine's autostart uses (`lib.assistant.client.autostart_seat` →
`lib.vllm_container_lifecycle.start_seat_coherent`: liveness ceiling + bounded
coherence probes + cold-recreate self-heal). The launch script comes from that
seat's `ED4ALL_SEAT_LAUNCH_SPECS` registry entry — the GUI never names a script
path. Because it is an explicit action it does **not** require
`ED4ALL_ASSISTANT_AUTOSTART` (that flag governs only the implicit lazy-start on
a chat turn).

A cold start takes minutes, so the call returns `{"ok": true, "state":
"starting"}` immediately and the work runs on one background thread;
`GET /api/assistant/seat/status` reports `starting: true` while it runs and
surfaces a failed start as `last_seat_error` afterwards. The start is
**single-flight**: a second POST while one is in flight returns the same
`starting` payload and never launches a second start. Refusals have **no side
effect**: **409 `assistant_seat_already_live`** when any candidate seat already
answers `/v1/models` (the detail names it), **500
`assistant_seat_no_launch_spec`** when the resolved seat has no
`ED4ALL_SEAT_LAUNCH_SPECS` entry (the detail is that remediation message
verbatim), and **500 `assistant_config_error`** on a non-loopback seat config.

### Activity bridge — `/api/activity`

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/activity/events` | `?since=<seq>` | `{events: [...]}` with `seq >= since` |
| POST | `/api/activity/post` | `{kind, payload{}}` | `{event: <stored record>}` (always `source="gui"`) |

Event `source` values are `gui` | `claude` | `assistant` — the third is
appended server-side by the Assistant panel (one `assistant_exchange` event per
chat exchange: question summary + tools used), so assistant activity shows in
the Activity tab alongside the Claude↔GUI events.

### Health

| Method | Path | Response |
|--------|------|----------|
| GET | `/api/health` | `{status: "ok"}` — the unauthenticated Docker liveness probe. |
| GET | `/api/health/doctor` | Cached in-process `ed4all doctor` preflight: `{generated_at, exit_state, exit_code, verdict, groups: [{group, checks: [CheckResult]}], summary}`. `exit_state` is `healthy` \| `degraded` \| `critical` (the worst-severity banner verdict); each `check` is a `{name, group, severity, summary, detail, remediation, data}` dict. **Group-agnostic** — the default check groups (`gpu`, `gpu_profile`, `window`, `environment`) plus any opt-in group whose env prerequisite is present (e.g. `seat` when a seat-registry env is set) are discovered from the live diagnostics registry, never hardcoded; the run-preflight `provider` + run-scoped `postmortem` groups are excluded. Results are cached ~30 s. |
| POST | `/api/health/doctor/refresh` | Re-runs the preflight now (bypasses the TTL cache); same shape. |
| GET | `/api/health/doctor/run/{run_id}` | Cached **run-scoped post-mortem** (the `ed4all doctor --run-id` equivalent): the global shape plus `{run_id, orchestrator_run_id, effective_status, usage: {present, rows}}`. `run_id` accepts a GUI run id, a `WF-*` orchestrator workflow id, or a bare orchestrator run id (resolved to `state/runs/<run_id>/` like the run/progress services). Unknown run → 404 `run_not_found`. |
| POST | `/api/health/doctor/run/{run_id}/refresh` | Re-runs the run-scoped post-mortem now; same shape (404 for an unknown run). |

The `/api/health/*` family is deliberately **open** (mirroring `/api/health`): it
carries no secret and rides the open Studio Dashboard. In-process, no subprocess;
heavy checks run in the request threadpool so the event loop never blocks.
Service: `gui.services.health_service`; router: `gui.routers.health`.

### Model seats — `/api/seats`

| Method | Path | Response |
|--------|------|----------|
| GET | `/api/seats` | The **global** vLLM seat monitor: `{generated_at, registry_configured, docker_available, seats: [{name, base_url, live, state, container, model, since, since_ms}]}`. The seat list is exactly the `ED4ALL_SEAT_BASE_URLS` registry, **in registry order** — model-agnostic, no seat name is hardcoded anywhere in this surface. Cached ~10 s. |
| GET | `/api/seats/run/{run_id}` | The same payload **plus the run's phase context**: `{run_id, workflow, status, effective_status, current_phase, expected_seats, mismatch: [{seat, state, base_url}]}`, and an `expected: bool` flag on every seat. `run_id` accepts a GUI run id or a `WF-*` workflow id (same resolution as `/api/runs/{id}/progress`); unknown run → 404 `run_not_found`. |

`state` is one of:

| state | meaning |
|-------|---------|
| `live` | `GET {base_url}/v1/models` answered 2xx (`model` carries the served id). |
| `loading` | the seat's registered container **is running** but the endpoint has not answered yet — a large seat's cold start takes ~9 minutes, so this is progress, **not** an alarm. |
| `down` | the seat's registered container is verifiably **not running**. |
| `unknown` | we genuinely cannot tell: docker is unavailable (CLI absent / no perms / timeout) or the seat has no `ED4ALL_VLLM_CONTAINERS` entry. Never a guessed `down`. |

`expected_seats` mirrors the current phase's `seats:` annotation in
`config/workflows.yaml` and honours all three cases: a **name list** (an
expectation), **`[]`** (declared seat-free), and **`null`** (the key is absent —
no opinion, so no mismatch is ever reported). `mismatch` names an expected seat
that is neither `live` nor `loading` — the "this phase needs *X*, *X* is down"
signal. `since` / `since_ms` are the age of the seat's **current** state as
observed by this server process (`null` on the first observation — there is no
prior observation to compare against); they are not a claim about the seat's
real history.

Open (mirroring `/api/health/*`) — it carries no secret and rides the open
Studio Dashboard. Read-only: it probes `/v1/models` and reads `docker ps`, and
never starts or stops anything. Service: `gui.services.seat_service`; router:
`gui.routers.seats`.

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
  retrieval finds nothing. Fix by removing the empty `imscc_chunks/` directory
  so the legacy `corpus/chunks.jsonl` resolves, or by running the legacy
  chunk-backfill helper under `LibV2/tools/libv2/scripts/`.
- **Local providers fail / local models don't list.** The local model server
  `LOCAL_SYNTHESIS_BASE_URL` names — a vLLM seat, Ollama, llama.cpp, LM Studio,
  etc. — must be running with the relevant model served. Use Settings → "Test"
  (`local`) for a live `GET /models` probe, and the Model Routing dropdowns
  reflect `GET /api/settings/local-models` (which probes `/v1/models` first, then
  Ollama's `/api/tags`).
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

---

## Learner surface (`/learn/`)

The same FastAPI process also serves a **learner-facing answer UI** — a
disjoint, accessibility-first surface for the query → grounded-answer → citation
→ source-page loop. It is rendered to WCAG 2.2 AA from the first build and is
deliberately separate from the operator SPA: a learner must never see env vars,
API keys, run launching, or the Claude bridge.

### Routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/learn/` | The learner page (static shell; auto-served by the existing `StaticFiles` mount). In `--learner` mode it is also served at `/`. |
| GET | `/api/learn/courses` | List answerable courses (`[{slug, chunk_count}]`). |
| QUERY / POST | `/api/learn/ask` | `{slug, query, engine="auto"}` → `{answer, html}`. Returns **200 for all answer outcomes** (answered / answered-with-warnings / both refusal states / both blocked states); typed backend failures map to 503/502/500. The `html` field is the server-rendered answer fragment the page swaps in (single rendering path — no client-side re-render). **Canonical method: QUERY** ([RFC 10008](https://www.rfc-editor.org/rfc/rfc10008)); **POST** is a deprecated alias carrying a `Deprecation` header. The learner page sends QUERY and downgrades once to POST if a client/proxy rejects the method. |
| GET | `/api/learn/source/{slug}?item_path=…[&fragment=…]` | Serve the archived source page a citation links to, sanitized and wrapped, with a restrictive CSP. Path-traversal attempts are rejected (422) before any filesystem access. |
| POST | `/api/learn/ask-jobs` | Enqueue a **durable async** ask (L4); returns `{ask_id, status, queue_position}`. Survives a refresh — the job record persists retrieved passages onto the running record so the drawer can disclose them **passages-first** during the compose window. |
| GET | `/api/learn/ask-jobs/{ask_id}` | Poll one async ask job. While running it can already carry `passages` (+ `passages_refused`) for passages-first disclosure; on completion it carries the rendered answer. |
| GET | `/api/learn/ask-capabilities` | Ask-drawer capability probe (L3): `{indexed_course_count, library_wide_eligible, library_wide_default}` — gates the **cross-course ("search all courses") toggle**. |
| POST | `/api/learn/feedback` | Record thumbs up/down (+ optional trimmed comment) on a completed answer (I6). Fans out to the event log (`answer_feedback`) + a per-course `feedback.jsonl`. 422 bad verdict, 429 flood. |
| GET | `/api/learn/quiz/{slug}` | List the course's playable assessments (quiz player, L1/Q6). |
| GET | `/api/learn/quiz/{slug}/{assessment_id}` | One assessment as learner JSON — **stems only**, the answer key stripped by contract. |
| POST | `/api/learn/quiz/{slug}/{assessment_id}/grade` | Grade a submitted attempt server-side and return per-item correctness + score; `GET …/attempts` lists prior attempts. |

The answer engine is **synchronous and non-streaming**; the UI shows a polite
busy state and a visual-only elapsed counter rather than faking token streaming.
The **Ask drawer** rides the durable async `ask-jobs` path so a refresh
re-attaches to an in-flight ask; it discloses retrieved **passages first** while
the answer composes, surfaces a per-answer **feedback** (thumbs up/down) control,
and — when `ask-capabilities` reports the course library is eligible — a
**cross-course ("search all courses") toggle** (`library_wide`). The learner page
also mounts an **interactive quiz player** (served answer-key-free; graded
server-side). Groundedness/NLI is never loaded on the learner ask path
(operator-only, advisory).

### Learner-only serve mode

For a moderated pilot session, run the appliance so that **only** the learner
surface is mounted — the operator routers (settings, uploads, runs, courses,
retrieval) and the operator SPA are not even registered:

```bash
ed4all gui --learner                 # learner surface only; /  IS the learner page
ED4ALL_GUI_LEARNER=1 ed4all gui      # env fallback (same effect)
python -m gui.server --learner       # direct uvicorn entrypoint
```

In learner-only mode:

- **Reachable:** `/` and `/learn/` (the learner page), `/api/learn/*`,
  `/api/health`.
- **Not mounted (404):** `/api/settings`, `/api/uploads`, `/api/runs`,
  `/api/courses`, `/api/retrieval`, `/api/assistant`, the `/ws/runs/*` log
  stream, and the operator SPA.

Without `--learner`, the full app serves both surfaces: the facilitator uses the
operator SPA while a participant opens `/learn/` — appropriate for dev/demo on a
trusted machine, not for a shared/exposed deployment.

### Embedding the ask widget in an LMS (`iframe`)

The learner page doubles as an **embeddable "Ask the Course" widget** so a
grounded ask box can live *inside* an LMS course page (the OpenOLAT retrieval
demo frames it from an OpenOLAT *External Page* / *LTI Page* course element). The
surface is **cookieless** (the anonymous `X-Learner-Id` lives in the framed
origin's `localStorage`, not a cookie — no `SameSite` third-party-cookie
breakage) and needs no server-side session, so an `iframe` embed works with zero
new auth plumbing.

**Widget URL shape** (this is the exact contract a host LMS configures):

```
http://<gui-host>:8077/learn/?course=<slug>&embed=1
```

| Query param | Effect |
|-------------|--------|
| `course=<slug>` | Pins the widget to ONE course: the course `<select>` is pre-selected and **disabled** (a learner can't switch it). The slug is passed to `/api/learn/ask` exactly as a manual pick would be; the server still validates it (an unknown slug returns an honest 404 fragment, never a wrong course). |
| `embed=1` | **Compact widget mode** (truthy: `1`/`true`/`yes`/`on`): hides the course-picker field and the Quizzes panel, leaving a pure ask box that sits flush in its frame. Presentational only — the ask form, live region, and answer surface are untouched. |

With neither param the page is **byte-identical** to the standalone learner
surface.

**Frame-ancestors allowlist (`ED4ALL_GUI_FRAME_ANCESTORS`)** — opt-in control
over *who may frame the GUI*:

```bash
# Only this OpenOLAT origin may iframe the ask widget:
ED4ALL_GUI_FRAME_ANCESTORS='http://lms.example:8080' ed4all gui
# Multiple origins / 'self' are space-separated (a CSP frame-ancestors source list):
ED4ALL_GUI_FRAME_ANCESTORS="'self' http://lms.example:8080 https://lms.example"
```

- **Unset (default):** the GUI sends **no** `frame-ancestors` header — the ask
  surface stays framable by any origin, matching the cookieless embed design.
  This is the byte-identical, zero-regression default.
- **Set:** every HTTP response that does not already carry a
  `Content-Security-Policy` gains
  `Content-Security-Policy: frame-ancestors <allowlist>`, so only the listed
  origins can frame the GUI. The archived-source viewer's own restrictive CSP is
  never clobbered. The value is resolved per-request (an edit takes effect
  without a restart); CR/LF are scrubbed and interior whitespace collapsed so a
  hostile value can't smuggle a second header. Applies in **all** serve modes
  (`full` / `studio` / `learner`) since the widget can be served from any of
  them.

Modern browsers honour CSP `frame-ancestors`; it supersedes the legacy
`X-Frame-Options` header (which cannot express a multi-origin allowlist), so the
GUI emits **only** the CSP form.

**Auth scoping (structural, no embed token needed).** The ask surface
(`/api/learn/*`) is **open by design** — a learner never presents a token. In
`full` mode the operator token gate (`ED4ALL_GUI_TOKEN`) covers the operator
surface only; `/api/learn/*` is deliberately un-gated, so the embed widget keeps
working with the token armed. In `studio` / `learner` mode the operator routers
are not mounted at all. There is therefore no separate "embed-scope" token to
issue.

**CORS fallback (no server change).** The primary route is the `iframe` embed
above (CORS is irrelevant inside an iframe). A host that would rather render
answers in its **own** LMS-themed DOM can instead call `/api/learn/ask`
cross-origin directly — the GUI already sends a wildcard `Access-Control-Allow-Origin`
and the ask surface is unauthenticated, so no new CORS env is required. (LTI 1.3
is intentionally **not** implemented — the iframe route needs none of its
OIDC/JWKS/launch-JWT machinery; the seam is noted for a future signed-context
handoff if a customer ever requires it.)

## Studio surface (`/studio/`)

The **Studio** is the end-user surface for *browsing archived courses* — a
Library card grid plus an IMS Common Cartridge viewer (an ARIA manifest tree
beside a sandboxed content pane with a prev/next pager that follows the manifest
order). It serves the packaged IMSCC living at
`LibV2/courses/<slug>/source/imscc/*.imscc` (courses discovered dynamically —
never hardcoded). Built to **WCAG 2.2 AA** from the first build (semantic
landmarks + skip link, full keyboard operability of the tree + pager, ARIA tree
pattern with roving `tabindex`, programmatic focus management, sandboxed iframe
for untrusted archive HTML), gated by `gui/tests/test_studio_a11y_gate.py`.

The frontend is vanilla ES modules (no build step): `gui/static/studio/` imports
the reusable toolkit factored into `gui/static/shared/` (`api.js`, `dom.js`,
`toast.js`, `router.js`). Archived cartridge HTML is **untrusted** — every served
page is run through the same audited active-content scrub the source-viewer uses
(`gui.services.source_page.sanitize_soup`: drops `<script>`, inline `on*`
handlers, `javascript:`/`data:` URLs) and served with a restrictive CSP +
`nosniff`. Generated pages ship as zero-JS self-contained HTML whose interactive
components (self-check reveal buttons, flip-card grids) rely on inline handlers
the sanitiser strips; a small **viewer shim** (`gui/static/studio/viewer-shim.js`)
re-binds them accessibly *inside the iframe sandbox* (`allow-scripts` only — no
`allow-same-origin`, so it can't reach the parent origin).

### Routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/studio/` | The Studio shell (static; in `--mode studio` the bare `/` redirects here). |
| GET | `/api/library` | Course cards for every archived course carrying an IMSCC cartridge, each stamped with a **certification badge** (the derived `course_status` / promotion tier) so the Library grid shows trust at a glance. |
| GET | `/api/library/{course_id}/scorecard` | The per-course **quality scorecard** (T2): a section-per-artifact rollup composed request-time from the course's governance / eval artifacts. An un-evaluated course still returns 200 with every section marked *not yet evaluated* (never a fabricated number). 404 unknown course, 422 malformed slug. |
| GET | `/api/courses/{course_id}/manifest` | The course organization tree: `{slug, title, items: [...]}` (recursive module/unit/page nodes). |
| GET | `/api/courses/{course_id}/page?item=<id>` | One sanitized cartridge page, resolved via its manifest **resource id** (never a raw path), with a restrictive CSP. |
| GET | `/api/courses/{course_id}/asset?path=<rel>` | A whitelisted static asset (image/css/font) from the cartridge. **Path-traversal-safe:** the path is pre-rejected (no `..`/absolute/NUL/backslash) before any zip read, the extension must be in the content-type whitelist (never `.html` / active types), and the member must literally exist in the cartridge. Unknown course / cartridge / member → 404. |

Backend service: `gui.services.imscc_service` (org-tree walk factored from
`Courseforge/scripts/imscc-extractor/imscc_extractor.py`, read in-memory from the
cartridge zip — no extract-to-disk). Router: `gui.routers.library`.

### Create wizard + Studio settings (C3)

Studio is not only for *browsing* finished courses — a non-developer can author
one. Two extra hash routes ride the Studio shell (`gui/static/studio/`):

| Route | Module | Purpose |
|-------|--------|---------|
| `#/create` | `create.js` | The **Create wizard** — a 3-step flow to build a course from a textbook. |
| `#/create/<run_id>` | `create.js` | The **progress view** for a launched build (re-attaches on refresh via the run registry + WS). |
| `#/create/<run_id>/log` | `create.js` | The raw **build log** for a run (replayed over the WS), surfaced on failure. |
| `#/live` | `create.js` | **Live run** — thin resolver: finds the currently running workflow in the merged `/api/runs` list (running/paused preferred, CLI-launched runs included) and delegates to the same build-progress view; honest "No live run" empty state (CTA to Run history) when nothing is building. |
| `#/settings` | `settings.js` | The simplified **Studio settings** page (provider/model + answer backend + cloud API keys). |
| `#/assistant` | `assistant.js` | The **Assistant** chat panel — see below. |

**Create wizard flow** (3 steps as a labelled `<ol aria-label>` progress
structure with `aria-current="step"` on the active step):

1. **Upload** — drag-drop *or* file-picker for one or more PDFs, with
   client-side type (`.pdf`) + size (≤ 200 MB) validation; files are saved via
   `POST /api/uploads` when the user advances, yielding an `upload_id`.
2. **Configure** — course name (validated against the slug rule
   `^[A-Za-z0-9_-]{2,}$` *client-side hint only*; the server is authoritative),
   optional weeks, a provider/model summary read from `GET /api/settings/studio`
   with a link to the Studio settings page, and an **Advanced** `<details>`
   block (skip assessments / skip training synthesis) collapsed by default.
3. **Launch** — `POST /api/runs` (`workflow: textbook_to_course`), then a
   **phase-checklist** progress view driven by the existing `/api/ws/runs/{run_id}`
   socket. Each phase renders with a friendly name (from the `label` field on
   `/api/workflows` phases — see `run_service.PHASE_LABELS`) and a
   pending / running / done / skipped / failed state, plus an elapsed timer. On
   success → an **Open course** link into the viewer; on failure → an actionable
   error + a **View build log** link. Launching navigates to `#/create/<run_id>`
   so a browser refresh **re-attaches** to the running build (state survives via
   the run registry, never client-only).

**Pipeline stage rail + live stats band** — under the Build Console the progress
view renders `stage-rail.js` (shared kit component), fed by a 3s poll of
`GET /api/runs/{run_id}/progress` that stops at a terminal state (a finished /
failed run renders statically — no pulse, no polling). The rail is a horizontal
connected chain of phase nodes (wraps on narrow widths), visually grouped by the
server-derived stage (conversion / planning / generation / validation /
packaging / archive / training / finalization — name-derived, never phase
indices; the map is `progress_service.py::_EXACT_GROUPS`, and because the rail
buckets by FIRST occurrence a group must span execution-contiguous phases, so
the post-build training tail and finalization render last). Node states: done ✓,
current (CSS pulse; a static highlight ring under `prefers-reduced-motion` —
motion lives in `tokens.css`), paused Ⅱ, pending ○, failed ✗, skipped – (dimmed, e.g.
`enabled_when_env`-disabled tiers or `--skip-training`). Completed nodes show
their checkpoint wall-clock under the node. The stats band shows tokens/sec
(AGGREGATE seat throughput: sliding-window completion tokens over the window's
wall span), the estimated in-flight request count ("N in flight" — the seat is
serving many requests concurrently, so per-request speed is roughly
aggregate ÷ N; the per-stream figure lives in the detail disclosure), total LLM calls,
prompt/completion token totals, TTFT p50 (when the run streams under
`ED4ALL_LLM_TTFT_METER`), elapsed time in the current phase, and the currently
serving vLLM seat/model. A valid run-scoped phase telemetry snapshot adds the
node's completed/total chunk units and generic detail rows for pair
accepted/rejected and SFT/DPO counts, active/maximum workers, queued/in-flight
units, generation results versus cache replays, transient lifecycle,
throughput, ETA, gate readiness, provider/model, and rejection reasons. The
reader rejects impossible relationships (`terminal > completed > total`,
`active + queued != in_flight`, or concurrency above maximum); metrics remain
visible on completed phases. Every state pairs its color with a glyph +
visually-hidden text (never color-only), and the rail adds no live region — the
console's single `role=status` line stays the only announcer.

The progress checklist's live per-phase signal comes from
`run_service._poll_phase_progress`: a best-effort poller that watches the
workflow state file's `phase_outputs` `_completed`/`_skipped` markers and
appends `[phase] <name> <state> — <label>` log lines as each phase finishes;
those lines stream over the existing WS (the orchestrator itself only logs a
phase *summary* at the very end). The poller is pure observation — it never
mutates workflow state and is cancelled when the run ends.

**Failure UX** — when a build fails a validation gate (or a phase errors), the
progress view replaces the raw error string with an actionable failure panel
(`create.js::renderFailurePanel`, `role="alert"`):

- The failing phase is marked from a **structured** `[phase] <name> failed —
  <label>: <reason>` log line (emitted by `run_service._drive_pipeline` /
  `_run_single_phase`), not the old "last running phase" guess. The runner
  surfaces `failed_phase` + `failure_reason` (`WorkflowRunner.run_workflow` →
  `OrchestratorResult`); they are persisted on the run record and replayed on
  refresh.
- The panel renders a **severity-badged failed-gate table** (from the
  `failed_gates` digest of `GET /api/runs/{run_id}/validation-report`) and, when
  a `courseforge_validation_report.json` exists for the run, a **per-block-type
  pass/fail/escalated summary** rolled up from `per_block_results[]`.
- Below that summary the panel renders a **per-page / per-block rewrite picker**
  (`create.js::renderRewritePicker`, I4 stage 2): one collapsible group per
  failing `page_id` (rows with no page fall under an "Unassigned" group), each
  with a whole-page checkbox (`pages` scope) and a labelled checkbox per failing
  block (`block_ids` scope, block id code-styled). Ticking a whole page
  checks+disables its child blocks; the page checkbox goes indeterminate on a
  partial selection. A polite live-region summary bar counts the selection and a
  single *Rewrite selected* CTA enqueues `courseforge_rewrite` with `block_ids` +
  `pages` posted as raw arrays. The JS stays dumb — all selection composition,
  dedup, and subsumption (dropping a block whose page is also selected) happen
  server-side in `run_service._normalize_blocks_param`; unknown ids/pages fail
  loudly at rewrite run time, not at enqueue.
- **What-to-do-next affordances** (each confirms before enqueuing, then links to
  the new run's progress view), shown only when applicable:
  *Re-run validation* (`courseforge_validate`) and *Rewrite failing blocks*
  (`courseforge_rewrite` with `--blocks` prefilled from the report's failing
  block types) appear only when a validation report is present; *Re-run failed
  step* (`POST /api/runs/phase` for the failed phase) appears when a phase is
  known; *View / download build log* is always offered.

**Pause for objectives review (I1)** — the wizard can launch a build with a
`--stop-after` intent so it halts cleanly after course planning. The orchestrator
returns `status="paused"` (exit-code-3 semantics), which `run_service` persists as
a `paused` run status carrying the `paused_phase`. The progress view then renders
an **objectives-review + resume panel** (instead of the failure panel): the
operator edits the synthesized objectives via the existing
`PUT /api/courses/{id}/objectives`, then resumes the SAME workflow by relaunching
with `resume_run_id` (the documented `--resume WF-…` pathway — all other launch
params are ignored on a resume). The persisted `--stop-after` intent is replayed
so a bare resume doesn't silently drop the halt.

**Studio settings page** — a strict subset of the operator settings catalog,
served by `GET /api/settings/studio` (`settings_service.build_studio_settings_payload`):
only the `credentials` / `global` / `answer` / `local` env-catalog categories
(the AI provider + model, the grounded-answer backend, and cloud-provider API
keys), plus a **read-only** GUI host/port echo. Secrets stay masked (`"set"` /
`null`); the full operator catalog (SemantiK conversion / per-tier Courseforge /
Trainforge / embedding knobs) is never returned here. Writes go through the existing
`PATCH /api/settings` (deep-merge of `model_routing.*` dotted paths + `env`
keys); a blank key field keeps the saved value. A **Test provider** button hits
the existing `POST /api/settings/test-provider` (a cheap reachability check — no
model generation; the `local` arm is a short-timeout `GET <base_url>/models`,
loopback-respecting for the answer provider) and reports the result in plain
language: connected / ready / key-present / **not authorized** (missing key) /
**unreachable** / error.

**Assistant panel** (`#/assistant`, `assistant.js`) — a chat surface over the
sandboxed operator assistant (`POST /api/assistant/chat`; see the REST
reference above). It is a *pure front-end*: the engine server-side owns the
tool whitelist, the loopback guard, and the round cap — the panel only renders
the conversation. It carries a chat message list, a send-on-Enter input
(Shift+Enter for a newline), a busy indicator, a **status pill** driven by
`GET /api/assistant/status` (seat serving + model / seat down / misconfigured),
a collapsible per-message **tool-call trace** (tool name, arguments, result),
and a "New conversation" reset. When the seat is down the 503's start
instructions render in-line as the assistant's reply. Reachable from the
Studio primary nav ("Assistant").

A **"Debug last failure"** button sits in the panel header: always visible,
enabled by a single mount-time probe of `GET /api/assistant/debug-context`
(disabled with the probe's `reason` as tooltip when no failed run exists — no
polling). Clicking starts a *fresh* conversation seeded in debug mode: the
failure summary renders as a distinct system-style banner bubble, followed by
the model's opening diagnosis; subsequent messages in that conversation keep
the debug context (the panel re-sends `debug` + `run_id` each turn). The deep
link `#/assistant/debug[/WF-...]` triggers the same flow on mount, so other
views can link a failed run straight into a debug conversation.

A **"Seat model"** button sits beside the status pill. It is a real `<button>`
rendered only when `GET /api/assistant/seat/status` reports that **no**
candidate seat is live and no start is in flight — when a seat is up, it is
hidden. Clicking POSTs `/api/assistant/seat` (no body: the server resolves the
seat and its registered launch spec), switches the button to a busy state
(`aria-busy`, text "Seating nano…") and polls the same status endpoint every
10s until a seat comes up (pill refreshes, button hides) or the start fails
(toast + an inline `role="status"` message carrying the server's detail — the
signal is always text, never color alone).

**System health section** (Dashboard `#/`, `health.js`) — a self-fetching
"System health" zone on the Author Dashboard over `GET /api/health/doctor` (see
the REST reference above). It renders an overall **verdict banner**
(healthy / degraded / critical, from the worst check severity) with a
last-checked timestamp and a **Refresh** button (`POST .../refresh`), then one
**card per check group** — each check a row with a shared **statusPill**
(glyph + text, never color-only) and its summary, plus an expandable
`<details>` carrying the remediation + detail. It is **group-agnostic** (renders
whatever groups the backend reports) and **on-demand** (fetched once on mount,
re-fetched only on Refresh — no polling loop). The run-scoped
`GET /api/health/doctor/run/{run_id}` post-mortem backs the build console's
per-run health/debug surface (payload shaped for that follow-up view).

**Model seats monitor** (`seats.js`) — the always-on vLLM seat panel, mounted on
**both** the Author Dashboard (a "Model seats" zone below System health, over
`GET /api/seats`) and the build-progress page (a compact strip below the
run-health strip, over `GET /api/seats/run/{run_id}`). One chip per **registered**
seat — registry order, no seat name in the code — carrying a shared statusPill
plus the seat name, a plain-language state word (`serving` / `starting up` /
`down` / `unknown`) and how long it has been in that state ("serving · 12m"), so
a flapping seat is visible. Never color-only.

It exists because of a real incident: two concurrent runs, one run's seat
schedule stopped the seat the other was mid-dispatch to, and a 2.5-hour build
failed with nothing in the GUI showing it. So on a build page the panel's
headline is **expected-vs-actual** — when the current phase declares a seat that
is not serving, a warning line names both: *"Phase heading_judge needs
spark-super — not serving."* A seat still **loading** (a ~9-minute cold start)
never raises that warning.

Both mounts poll every 15 s while on screen and stop on unmount (the dashboard
route returns a teardown; the build page latches on a terminal run after one
final snapshot). The service holds a single ~10 s cache, so a dashboard and a
build page open at once do **not** double the probe rate.

### Serve modes (`--mode` / `ED4ALL_GUI_MODE`)

The serve mode is now a three-way choice — `full` (default) | `studio` |
`learner`:

```bash
ed4all gui                           # full: operator + Studio + learner
ed4all gui --mode studio             # Studio + learner only; operator NOT mounted
ed4all gui --mode learner            # learner answer surface only
ed4all gui --learner                 # legacy alias for --mode learner
ED4ALL_GUI_MODE=studio ed4all gui    # env fallback (factory/reload path)
```

In **studio mode**:

- **Reachable:** `/` (→ `/studio/`), `/studio/*`, `/shared/*`, `/learn/*` (the
  C2 ask drawer rides this), `/api/library`, `/api/courses/{id}/*`,
  `/api/learn/*`, `/api/health`, `/api/seats` (the model-seat monitor),
  **plus the C3 Create-wizard surface**:
  `/api/uploads` (PDF intake), `/api/runs` + `/api/workflows` + `/ws/runs/*`
  (launch + run registry + the progress stream), and `/api/settings` +
  `/api/settings/studio` + `/api/settings/test-provider` (the scoped Studio
  settings page), and `/api/assistant/*` (the Assistant chat panel). The
  settings router is mounted in full so `PATCH` / `test-provider` work, but
  the Studio settings UI only surfaces the `/api/settings/studio` subset.
- **Not mounted (404):** the operator `/api/retrieval` router and the operator
  `/api/courses` listing router (the Studio `/api/courses/{id}/manifest|page|asset`
  viewer endpoints come from `gui.routers.library`), plus the operator SPA.
- **Still excluded from learner mode:** none of the above operator routes are
  mounted in `--mode learner` (answer-only); `/api/settings/studio` 404s there.

**Precedence** (`gui.app._resolve_mode`, high → low): explicit `mode=` kwarg >
`learner_only=True` kwarg > `ED4ALL_GUI_MODE` env > legacy `ED4ALL_GUI_LEARNER`
env (truthy) > default `full`. `ED4ALL_GUI_MODE` **wins** over the legacy
`ED4ALL_GUI_LEARNER` (a deliberate `studio` request is never silently downgraded
to learner by a stale learner env). The full app also mounts the Studio
Library/viewer API so a facilitator can preview `/studio/` without a second
process.

### Operator auth (`ED4ALL_GUI_TOKEN`)

The **full** mode operator surface supports a minimum-viable shared-secret
bearer token so it can be exposed beyond loopback without leaving env / API-key
management and run-launching wide open. Set `ED4ALL_GUI_TOKEN` (the
Docker-friendly primary channel; the settings store's `secrets.gui_token` is
also accepted, with **env precedence**):

```bash
ED4ALL_GUI_TOKEN=$(openssl rand -hex 32) ed4all gui --host 0.0.0.0
```

When the token is set, an ASGI middleware gates the **operator-classified**
paths — anything below requires `Authorization: Bearer <token>` (a constant-time
compare; a miss returns `401` with a `WWW-Authenticate: Bearer` challenge):

- the six-tab operator **SPA root** (`/`, `/index.html`, `/app.js`),
- the OpenAPI docs surface (`/docs`, `/redoc`, `/openapi.json`),
- the operator-only API routers Studio never calls: `/api/courses`,
  `/api/retrieval`, `/api/activity`,
- the operator **run-log WebSocket** `/api/ws/runs/*` — browser JS can't set WS
  request headers, so the token rides the `?token=<token>` **query param**
  (constant-time compared server-side; a miss closes the socket with app code
  `4401`).

Deliberately **left open** even when the token is set (so the Docker
healthcheck + the Studio Create/Settings flows keep working): `/api/health`,
`/api/seats`, the Studio-shared API routers (`/api/settings`, `/api/uploads`, `/api/runs`
REST, `/api/learn`, `/api/library`), and the static `shared` / `studio` /
`learn` / `styles.css` assets. Studio & learner serve modes don't mount the
operator surface and install **no** gate, so the token is full-mode-only.

The operator SPA carries a minimal **token-entry overlay**: the token is held in
`sessionStorage` (cleared on tab close), attached to every `fetch` by the shared
API wrapper, and a `401` pops the overlay to (re)enter it (one retry with the new
token). No token configured → the overlay never appears (current open behaviour).

When `ED4ALL_GUI_TOKEN` is **unset** the surface is fully open (the LAN/loopback
default); binding a non-loopback host in full mode with no token logs a startup
**WARNING** naming the env var.

### Access posture (honest)

The control-plane GUI's only built-in auth is the optional operator token above;
with **no token** set there is no login, no user accounts; CORS is
`allow_origins=["*"]`. The learner / Studio surfaces are open by design. Access
control for the Phase IA pilot is therefore **operational** unless the token is
configured:

- **Default loopback bind.** The server binds `127.0.0.1:8077` by default
  (`ED4ALL_GUI_HOST`/`ED4ALL_GUI_PORT`, or `--host`/`--port`). Do **not** bind
  `0.0.0.0` (or a routable host) while the **full** app is running where a
  learner can reach it **without setting `ED4ALL_GUI_TOKEN`** — that would expose
  the operator settings/API-key surface to anyone on the network (the startup
  WARNING flags exactly this).
- **Non-full serve modes are the access boundary.** When the surface must be
  reached by a participant who should only *consume* courses, run
  `--mode learner` (answer-only) so no operator route is mounted at all. This —
  plus the loopback default and a facilitator-moderated, single-machine session —
  *is* the access control for the pilot. **`--mode studio` is an authoring
  surface, not a locked-down consumer surface:** as of C3 it mounts the Create
  wizard's upload + run + (scoped) settings routes so a non-developer can build a
  course, so it carries the same "no auth — keep it on loopback / a trusted
  machine" caveat as the full operator app for those routes. Studio still does
  **not** expose the per-tier Courseforge / Trainforge / embedding operator knobs
  or the operator `/api/retrieval` surface.
- **Moderated sessions.** Pilot sessions are run on a single facilitator-operated
  machine. There is no multi-user isolation; sessions are supervised.
- **Minimum-viable auth is available** via `ED4ALL_GUI_TOKEN` (operator surface
  only; see above) and is **required before any non-loopback deploy** of the full
  app. Richer multi-user authentication (login, accounts) is still deferred to a
  later hardening wave; treat the current posture as pilot-only.

### Privacy note

Learner queries are logged locally to `training-captures/` (JSONL, via
`DecisionCapture`) on the **same device** — there is no telemetry or
network-egress path. Disclose this in the session consent language.

### Accessibility

The learner surface is built to **WCAG 2.2 AA**: semantic landmarks + skip
link, a single `<h1>` in `<main>`, label-paired form controls, a single
`aria-live="polite"` status region (announced once on busy + once on arrival,
no per-second chatter), focus moved to the answer heading on arrival, a real
`:focus-visible` outline (no `outline:none`-only patterns) for Windows
High-Contrast compatibility, ≥24×24 px interactive targets, ≥4.5:1 text
contrast, no color-only state, and a `prefers-reduced-motion` guard.

Conformance is enforced two ways:

- **Automated, every CI run** — a pytest gate runs the bs4-based
  `WCAGValidator` over every server-rendered page variant (idle, busy, all six
  answer statuses, error copies, and a wrapped source page) and fails on any
  Level A/AA finding.
- **Manual, each cycle** — the screen-reader + keyboard-only walkthrough in
  [`docs/operations/learner-ui-manual-pass.md`](../docs/operations/learner-ui-manual-pass.md)
  (NVDA + Firefox, VoiceOver + Safari, keyboard-only tab order, axe DevTools
  sweep, 200 % zoom / High-Contrast / target-size spot checks), with a
  pass/fail sign-off log.
