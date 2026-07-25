# GUI — Ed4All Control-Plane Web Server

Agent-facing contract map for `gui/`. **Operator/user documentation lives elsewhere
and is not duplicated here:**

- [`gui/README.md`](README.md) — six-tab walkthrough, REST API reference, model
  routing, Claude-integration, learner + Studio surfaces, access posture.
- [`gui/LAUNCH.md`](LAUNCH.md) — one-click launcher (`run-gui.sh` / `run-gui.bat` /
  `python3 gui/launch.py`), flags, troubleshooting.
- [`docs/operations/docker.md`](../docs/operations/docker.md) — containerized deploy.

This file is the map an agent needs *before editing code here*: layer boundaries,
serve-mode wiring, the auth classification, and the invariants a change must not break.

---

## Layering (hard rule)

```
gui/shared_state.py   state/gui/ paths + atomic writes + run registry + event log
gui/env_catalog.py    declarative env-knob catalog + provider list + routing→env
gui/settings_store.py settings.json load/save/patch, secret split, env render
gui/models.py         pydantic request/response models
        ↑ FOUNDATION — must import WITHOUT fastapi/uvicorn installed
gui/app.py            create_app(): middleware, router mounts, static mounts
gui/auth.py           OperatorTokenMiddleware + path classification
gui/server.py         argparse + uvicorn entry (`python -m gui.server`)
gui/routers/*.py      HTTP surface only — thin, delegate to services
gui/services/*.py     real backend work (runs, retrieval, answers, quizzes, ...)
```

**Invariant:** the four foundation modules must import WITHOUT the `gui` extra —
`shared_state` is pure stdlib (it imports `lib.paths.STATE_PATH` lazily inside
`_state_root`), `settings_store` adds only `gui.env_catalog` + `gui.shared_state`,
`env_catalog` adds `lib.llm.endpoints` (stdlib + `yaml` + `jsonschema`), and
`models` adds only `pydantic`. No `fastapi`/`uvicorn` above that line:
`MCP/tools/gui_tools.py` (9 `gui_*` MCP tools) imports `gui.shared_state` +
`gui.settings_store` directly, so a web dep leaking downward breaks MCP in a
default install. Web deps are imported in `gui/app.py` and below.

**Routers are thin.** A router does argument shaping, calls a service, maps typed
service errors to HTTP status codes. Business logic belongs in `gui/services/`.

---

## Serve modes

Resolved by `gui/app.py::_resolve_mode`. Precedence, high → low:
`create_app(mode=...)` → `learner_only=True` kwarg → `ED4ALL_GUI_MODE`
(`full|studio|learner`) → legacy `ED4ALL_GUI_LEARNER` truthy → `full`.
`ED4ALL_GUI_MODE` deliberately wins over the legacy env so a stale learner env
cannot silently downgrade a requested `studio`.

| Mode | Routers mounted | `/` serves |
|------|-----------------|-----------|
| `full` (default) | `settings`, `uploads`, `runs` (at bare `/api`), `courses`, `retrieval`, `learn`, `assistant`, plus `library` | Studio shell (open); operator SPA + dev console move behind the token gate at `/advanced/` |
| `studio` | `library`, `learn`, `uploads`, `runs`, `settings`, `assistant` | redirect to `/studio/` |
| `learner` | `learn` only | learner page (`gui/static/learn/`) |

Mount-order matters in `full`: `/advanced`, `/studio`, `/learn`, `/shared` are
mounted **before** the bare `/` Studio-root mount so `/` never shadows them or the
`/api/...` routers.

Static subtrees are served through `_NoCacheStaticFiles` (`Cache-Control: no-cache`)
— the SPA has no build step and no fingerprinted filenames, so without it a
redeployed container serves code returning browsers never fetch.

---

## Auth (`gui/auth.py`)

Single shared bearer token. Resolution: `ED4ALL_GUI_TOKEN` env, else the settings
store's `secrets.gui_token` (env wins). `OperatorTokenMiddleware` is installed in
**`full` mode only** — `studio`/`learner` don't mount the operator routers at all,
so their scoping is *structural*, not token-based. With no token configured the
middleware is a pass-through; `gui/server.py::_warn_if_exposed_without_token`
logs a startup WARNING on a non-loopback bind with no token.

Operator-classified (gated when a token is set):
`/api/courses`, `/api/retrieval`, `/api/activity`, the `/api/ws/runs` WebSocket
(token via `?token=` — browser JS cannot set WS headers), `/docs` `/redoc`
`/openapi.json`, the `/advanced/...` SPA + `/advanced/dev/` console, and
`DELETE /api/library` (method-gated destructive route on an otherwise-open family).

Deliberately open: the bare `/` (Studio shell = the product), `/api/health`
(Docker healthcheck), the Studio-shared routers `/api/settings` `/api/uploads`
`/api/runs` `/api/learn` `/api/library` `/api/assistant` (the Assistant panel
rides the Studio shell; its power is bounded by the engine's tool whitelist,
mirroring the open `/api/runs` posture), and the `/shared/` `/studio/`
`/learn/` static subtrees plus `/advanced/styles.css`.

Token comparison uses `hmac.compare_digest` (`tokens_match`). Do not add a
plain `==` compare.

---

## HTTP QUERY (`gui/routers/http_query.py`)

QUERY (**RFC 10008**, Standards Track, June 2026) is safe + idempotent yet carries
a body — the correct method for retrieval-shaped reads that ride POST only because
GET cannot carry a query body. This module is the **single** declaration of the
contract; every retrieval-shaped route imports it so the method set never drifts.

- `QUERY_METHODS = ["QUERY", "POST"]` — wired into `@router.api_route(methods=...)`.
- QUERY is canonical. POST is a **deprecated alias**: `apply_deprecation_if_post`
  stamps `Deprecation: true` + a `Link; rel="successor-version"` header (RFC 9745).
- Current call sites: `POST|QUERY /api/learn/ask`, `POST|QUERY /api/retrieval/query`.
- Cross-origin QUERY triggers a CORS preflight (non-safelisted method); the
  same-origin SPA is unaffected. Clients send QUERY and downgrade once per session
  on a 405/501 or network-layer method block.

A new retrieval-shaped endpoint must use `QUERY_METHODS` + the helper, not bare POST.

---

## Settings + secrets (`gui/settings_store.py`)

Canonical store: `state/gui/settings.json` (schema `version: 1`; keys `env`,
`model_routing`, `retrieval`, `flags`, `assistant`). Split-file secret handling:

- `_is_secret_key` = any env key ending `_API_KEY` or `_KEY`.
- Secrets are written to a sibling `secrets.json`, never into `settings.json`.
- `mask_secrets` renders a secret as `"set"` or `None` — **write-only**; a GET
  never returns a credential value.
- `render_env` / `apply_env` project the doc (plus `env_catalog.routing_to_env`)
  into process env + `state/gui/.env.rendered`.

`gui/shared_state.py` owns every `state/gui/` path (`settings.json`,
`.env.rendered`, `runs/`, `logs/<run_id>.log`, `uploads/`, `events.jsonl`) and all
writes go through `_atomic_write_text` / `_atomic_write_json` (tmpfile +
`os.replace`). Never open a state file and write it in place. `_state_root()`
honors `ED4ALL_STATE_RUNS_DIR` (its parent) so tests redirect into `tmp_path`.

The append-only `events.jsonl` (sources `gui` | `claude` | `assistant`) is the
Claude↔GUI activity bridge read/written by both `gui/routers/runs.py` and the
MCP `gui_*` tools; `gui/services/assistant_service.py` appends one
`assistant`-sourced event per chat exchange.

---

## Runs

`gui/services/run_service.py` creates a **real** workflow (no stubs): full
workflows via `MCP.tools.pipeline_tools.create_textbook_pipeline` /
`MCP.tools.orchestrator_tools.create_workflow_impl`, then drives it through
`PipelineOrchestrator`. Its `SUPPORTED_WORKFLOWS` + `COURSEFORGE_STAGE_SUBCOMMANDS`
tuples mirror `cli/commands/run.py` — **keep both in sync when a workflow lands.**

`gui/services/progress_service.py` backs `GET /api/runs/{run_id}/progress` (the
Studio stage-tracker rail + live stats band): a READ-ONLY merge of the run's
`config/workflows.yaml` phase plan (mtime-cached; workflow-agnostic — never a
hardcoded phase list) with the orchestrator workflow state's
`_completed`/`_skipped` markers, `state/runs/<id>/checkpoints/` wall-clocks, a
bounded incremental tail of the OP2 `llm_usage.jsonl` tap, a TTL-cached
`/v1/models` probe over the `ED4ALL_SEAT_BASE_URLS` seat registry, and real
in-phase unit progress (`stats.phase_units`) counted from the pipeline's own
per-unit resume-checkpoint sidecars (`_PHASE_UNIT_SIDECARS`, a data-driven
phase→sidecar map whose names are pinned to the writing code; sidecar dirs are
resolved from the run's real `project_path`/chunkset outputs, never a
hardcoded export pattern), plus the comprehensive `stats.detail` matrix
(run totals, cumulative avg tok/s, TTFT p50/p95, duration mean/median,
`finish_reason=length` truncation + `stream_usage_present:false` health
tripwires, per-(provider, model) and per-phase token breakdowns — the
per-phase join buckets usage-row epochs against the checkpoint wall-clock
windows with an explicit `unattributed` bucket for rows outside every
window — and the latest `vram_trajectory.jsonl` sample). Page-metered
usage rows (the SemantiK GLM-OCR lane: honest zero tokens, progress in a
positive-int `pages` field + `duration_ms`) aggregate into always-present
`stats.pages` / `pages_rows` / `pages_per_hr` (0/None defaults — shape
back-compat); the rate prefers the WALL SPAN of the page rows' own
timestamps (`ts` is completion-stamped, so earliest `ts − duration` →
latest `ts`; concurrent batches make a summed-duration rate understate
wall-clock throughput), falling back to summed durations when no ts
parses — never this process's wall clock. `create.js::applyPagesStat`
renders it as a "pages" band row and, on an all-page-metered run
(calls > 0, zero tokens), drops the empty token rows so the pages stat
carries the band. It never
mutates run state and never fabricates: unknown workflow → the observed
`phase_outputs` order; no usage rows → null/zero stats; empty seat registry →
no probe at all; no sidecar on disk → `phase_units` omitted entirely, and no
totals are ever estimated. Four honesty contracts on the rail payload: an
env-conditional phase with NO observed marker evidence renders `pending`,
never a skip guessed from the serving process env; when the plan carries BOTH
sides of an `enabled_when_env` variable (single-pass `content_generation` vs
the two-pass tiers), the not-taken side is HIDDEN from the payload — a
branchy row that resolved skipped, or the negative-clause (fallback) row
while it has no evidence of running — so the generation group starts at the
outline tier on a two-pass run and at `content_generation` on a single-pass
one; phase `group`s consolidate the whole authoring slice (single-pass +
two-pass tiers + inter-tier validators + assessment synthesis) into ONE
`generation` section so each rail header renders once; a near-zero (≤0.5s,
i.e. restored-from-checkpoint) wall-clock is suppressed rather than shown as
"0s". The group map is `progress_service.py::_EXACT_GROUPS` — eight groups,
`conversion` → `planning` → `generation` → `validation` → `packaging` →
`archive` → `training` (`training` / `post_training_validation` /
`evaluation`) → `finalization`. The rail buckets phases by FIRST occurrence,
so a group's section lands wherever its earliest member sits and a group must
therefore span execution-contiguous phases: `finalization` and the post-build
training tail each own their own key rather than folding into `archive` /
`generation`, and the `_KEYWORD_GROUPS` fallbacks mirror the exact map
(`eval`/`training` → `training`, `final` → `finalization`) so a future phase name
cannot fall back into an earlier group. Known, pre-existing, untouched: in
`course_generation` the trailing `validation` phase folds into the earlier
validation section and renders before packaging — same defect class, needs a
distinct group key. The payload also carries the run's LIVE course/book identity
(`course_name` + `display_title`): the workflow state's `params` win over the
GUI record's creation-time name, because an `--auto-name` run rebinds
`params.course_name` mid-run (`workflow_runner._maybe_apply_auto_name`) —
unknown → `null`, never fabricated. The build page (`create.js`) re-syncs its
header/meta from it on every poll, and mounts a run-health strip below the
rail (60s poll of `GET /api/health/doctor/run/{id}`) that expands into a
Debug panel — findings FAIL-first with remediation, the effective_status
explanation, and a copyable `ed4all assistant --debug --run <id>` command —
on a paused/stopping/incomplete/stalled?/failed run. The same module's
`output_tail` backs `GET /api/runs/{run_id}/output-tail` (the Studio "Live
output" panel): a bounded seek-from-end tail of the CURRENT phase's per-unit
resume sidecar (or, for `heading_judge`, the newest per-chapter judgment
files under `state/runs/<run_id>/heading_judge/` — a growing-directory
source; `training_synthesis` tails its per-pair
`training_specs/.synthesis_pairs_checkpoint.jsonl`; the `trainforge_train`
`training` phase tails the NEWEST
`models/<model_id>/eval/eval_progress.jsonl` eval-harness stream, its
`training_run.jsonl` being a run-end whole-file mirror, not incremental)
mapped to truncated,
HTML-stripped display rows — absent sidecar → `rows: []`, and phases whose
artifacts are atomic whole-file emits (chunking, packaging, vector_indexing,
semantik_conversion, …) are deliberately unmapped rather than fabricated.
Tests: `gui/tests/test_run_progress.py`.

`gui/services/liveness.py` derives an honest `effective_status` (ADDITIVE — it
never mutates `status`) so a stale CLI `RUNNING` record no longer renders
"Building" forever. It is stdlib-only + read-only + import-light (imported by
BOTH `run_service.list_runs` and `progress_service.run_progress`; never writes
`state/`). Signals: a `/proc` scan for live `ed4all run` processes matched on
ADJACENT `ed4all`/`run` argv tokens (never `pgrep -f`, which self-matches the
wrapper shell), the graceful-stop sentinel
`state/runs/<orch_run_id>/control/STOP_REQUESTED`, and workflow-file mtime.
A run ATTRIBUTES to a live process when any of its identifier tokens
(corpus / project / course name / orchestrator run id / workflow `WF-<id>`)
is a substring of that process's argv — the workflow id is what lets a bare
`ed4all run … --resume WF-<id>` process self-attribute (its argv carries
NEITHER the corpus/project — loaded from persisted state — NOR the
`orch_run_id`, which differs from the `WF-<id>`); without it a long
single-phase resume false-positives as `stalled?`. All three callers
(`run_service`, `progress_service`, `health_service`) thread the run record's
`workflow_id` into `attribution_tokens_from_params(..., wf_id=…)`.
Vocabulary: `paused` (resumable) · `stopping` (sentinel present, draining) ·
`incomplete` (CLI process gone without reaching completion, record stale) ·
`stalled?` (live-but-unattributable
process + >10 min stale file) · `building` (attributed/GUI/fresh) · terminal
passthrough. The `/proc`-liveness branch (`incomplete`/`stalled?`) applies to
CLI-launched runs ONLY — a GUI run is driven in-process (no external process),
so its non-terminal status is left unchanged bar the PAUSED / stop-sentinel
signals. Frontend: `pill.js` carries the new run keys (glyph+text, never
color-only); `run-history.js` + `create.js` prefer `effective_status` for the
pill / heading / meta line (a paused/stopping/incomplete run gets an honest state
line, not "Building your course"). Tests: `gui/tests/test_run_liveness.py`.

Background driver tasks are held in a module-level dict so asyncio's weak
reference doesn't collect them mid-flight and so `cancel_run` can find them. Those
tasks are in-process only, so `create_app` registers a startup hook calling
`run_service.reconcile_orphans()` — a uvicorn restart would otherwise leave
`queued`/`running` runs stuck forever with no driver.

---

## Seat monitor (`gui/services/seat_service.py`)

`seat_overview(run_id=None)` backs `GET /api/seats` (global) and
`GET /api/seats/run/{run_id}` (phase-aware), rendered by
`gui/static/studio/seats.js` on BOTH the Dashboard ("Model seats", always
visible) and the build page (a strip below the run-health strip). It exists
because a run's seat schedule once stopped the seat another live run was
dispatching to — connection refused → poison pill → a failed 2.5-hour build —
with nothing in the GUI showing it. Five invariants a change must not break:

- **Registry-driven, never a roster.** Seats are exactly
  `lib.vllm_container_lifecycle.parse_seat_registry()` (`ED4ALL_SEAT_BASE_URLS`)
  and containers `parse_container_registry()` (`ED4ALL_VLLM_CONTAINERS`). No
  seat name appears in `seat_service.py`, `seats.js`, or the CSS — a new seat is
  a registry entry. A test greps for that.
- **`loading` ≠ `down`.** A large seat's cold start takes ~9 minutes (container
  up, `/v1/models` silent) and must not render as an alarm. `down` is asserted
  ONLY when the registered container is verifiably not running; when docker is
  unavailable or the seat has no container entry the state is `unknown` — never
  a guessed `down`. The `docker ps` read is bounded, never-raising, and retries
  once through `sg docker -c` (the Spark docker-group wrapping), mirroring
  `lib/diagnostics/seat_schedule.py`. It is READ-ONLY: nothing here ever starts
  or stops a container.
- **Expectations are DECLARED.** `expected_seats` comes from the current phase's
  `seats:` annotation in `config/workflows.yaml`, and the three cases are
  distinct: names (an expectation), `[]` (declared seat-free), ABSENT (`null` —
  no opinion, so no mismatch can ever be reported). This is why the service
  re-reads the raw YAML instead of using `progress_service.phase_plan`, whose
  `seats` key coerces absent → `[]`.
- **One shared cache.** A single module-level ~10 s TTL cache serves the
  dashboard AND every build page, so N pollers cost ONE probe round; probes run
  OUTSIDE the lock. Never raises — a broken registry yields an empty seat list
  with a `detail`, a failed probe yields `unknown`.
- **`since` is process-scoped and honest.** The per-seat transition markers
  record when THIS PROCESS first observed the current state (reset on change,
  preserved when unchanged, pruned when a seat leaves the registry); the first
  observation reports `null` because there is nothing to compare against. It is
  never presented as the seat's real history.

Frontend contract: chips are statusPill + seat name + a plain state word + age
(text, never color-only); the mismatch line names the phase and the missing
seat. Both mounts poll 15 s and stop on teardown (`renderDashboard` now returns
a teardown; the build page latches on a terminal run). Tests:
`gui/tests/test_seat_service.py`.

## Assistant seat start

`gui/services/assistant_service.py` owns the Studio Assistant adapter. Beyond
the chat/debug turn plumbing it exposes the panel's **"Seat model"** action —
`seat_status()` + `seat_nano()`, surfaced as `GET /api/assistant/seat/status` +
`POST /api/assistant/seat`. Four invariants a change must not break:

- **No script path in the GUI.** The seat name comes from
  `lib.assistant.client.resolve_assistant_seat()` (`ED4ALL_ASSISTANT_SEAT`) and
  the launch command from that seat's `ED4ALL_SEAT_LAUNCH_SPECS` registry entry;
  the start itself goes through `client.autostart_seat()` →
  `lib.vllm_container_lifecycle.start_seat_coherent`. Never subprocess a script
  here. A seat with no registered launch spec is a typed refusal whose detail is
  the remediation message verbatim — never a silent no-op or a guessed path.
- **Explicit action ≠ autostart policy.** `ED4ALL_ASSISTANT_AUTOSTART` gates
  only the *implicit* lazy-start inside `AssistantEngine.ensure_seat`; a human
  clicking the button is not subject to it. Everything else (loopback guard,
  coherence probes, self-heal) stays engine/lifecycle-owned.
- **Refusals have no side effect.** If the priority walk
  (`resolve_active_seat`) reports any live seat, the start is refused with that
  seat named and the lifecycle is never called.
- **Single-flight, background.** A cold start takes minutes, so it runs on one
  daemon thread behind a module-level lock + state (`_SEAT_START`); the POST
  returns `state="starting"` immediately, a second click observes the in-flight
  start, and a failure surfaces on the next `seat_status()` as
  `last_seat_error`. The worker never raises.

`GET /api/assistant/status` (the pill's three-key `seat_serving`/`model`/`seat`
shape) is deliberately left untouched — the button has its own endpoint.

## Embeddable ask widget

The learner ask surface is designed to be iframed by an LMS: cookieless, and open
even when the operator token gate is armed (scoping is structural). The learner page
honors `?course=` (course pin) and `?embed=1` (compact mode).

Framing is governed by `ED4ALL_GUI_FRAME_ANCESTORS` (`gui/app.py`):
`resolve_frame_ancestors` + `FrameAncestorsMiddleware`, installed in **all** serve
modes. Unset/blank → **no header at all** (framed-by-anyone, the default). Set →
`Content-Security-Policy: frame-ancestors <sources>` appended to every HTTP
response that does *not* already carry a CSP, so the archived-source viewer's own
restrictive CSP is never clobbered. CR/LF are scrubbed and whitespace collapsed
(header-injection safety); a value that scrubs empty resolves to `None`.

---

## Environment variables

| Var | Read by | Purpose |
|-----|---------|---------|
| `ED4ALL_GUI_HOST` / `ED4ALL_GUI_PORT` | `gui/server.py` | Bind host/port (defaults `127.0.0.1` / `8077`, from `gui/__init__.py`). |
| `ED4ALL_GUI_MODE` | `gui/app.py` | `full` \| `studio` \| `learner`; wins over the legacy learner env. |
| `ED4ALL_GUI_LEARNER` | `gui/app.py`, `gui/server.py` | Legacy truthy learner-mode toggle (back-compat only). |
| `ED4ALL_GUI_TOKEN` | `gui/auth.py` | Operator bearer token; falls back to `secrets.gui_token`. |
| `ED4ALL_GUI_FRAME_ANCESTORS` | `gui/app.py` | CSP `frame-ancestors` source list for iframe embedding. |
| `ED4ALL_STATE_RUNS_DIR` | `gui/shared_state.py` | Its *parent* is the resolved `state/` root. |

---

## Tests

`gui/tests/` (52 `test_*.py`). Every module that needs the web stack starts with
`pytest.importorskip("fastapi")` so the suite is a clean skip on a default install
without the `gui` extra. The suite includes per-surface a11y gates
(`test_studio_a11y_gate.py`, `test_learner_a11y_gate.py`,
`test_operator_settings_a11y_gate.py`, `test_dev_console_a11y_gate.py`,
`test_component_gallery_a11y_gate.py`), serve-mode mount assertions
(`test_*_serve_mode.py`), and `test_query_method.py` for the QUERY⇄POST contract.
Client-side JS that can't be driven headless is asserted against the served static
assets (see `test_embed_widget.py`).

---

## When changing this subsystem

- Adding an endpoint → router stays thin; logic to `gui/services/`; register the
  path in `gui/auth.py` if it is operator-classified; document it in `gui/README.md`.
- Retrieval-shaped read → use `QUERY_METHODS` + `apply_deprecation_if_post`.
- New env knob the pipeline reads → add it to `gui/env_catalog.py`, not ad-hoc
  `os.environ` reads in a router.
- New provider/model selector → `config/endpoints.yaml` registry entry, never a
  subclass; a flag selecting an LLM provider/model/synthesis backend also needs a
  row in [`docs/LICENSING.md`](../docs/LICENSING.md).
- No stubs, no fabricated successes: an unwireable capability returns a typed error.
