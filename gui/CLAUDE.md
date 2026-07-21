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
| `full` (default) | `settings`, `uploads`, `runs` (at bare `/api`), `courses`, `retrieval`, `learn`, plus `library` | Studio shell (open); operator SPA + dev console move behind the token gate at `/advanced/` |
| `studio` | `library`, `learn`, `uploads`, `runs`, `settings` | redirect to `/studio/` |
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
`/api/runs` `/api/learn` `/api/library`, and the `/shared/` `/studio/` `/learn/`
static subtrees plus `/advanced/styles.css`.

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
`model_routing`, `retrieval`, `flags`). Split-file secret handling:

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

The append-only `events.jsonl` (sources `gui` | `claude`) is the Claude↔GUI
activity bridge read/written by both `gui/routers/runs.py` and the MCP `gui_*` tools.

---

## Runs

`gui/services/run_service.py` creates a **real** workflow (no stubs): full
workflows via `MCP.tools.pipeline_tools.create_textbook_pipeline` /
`MCP.tools.orchestrator_tools.create_workflow_impl`, then drives it through
`PipelineOrchestrator`. Its `SUPPORTED_WORKFLOWS` + `COURSEFORGE_STAGE_SUBCOMMANDS`
tuples mirror `cli/commands/run.py` — **keep both in sync when a workflow lands.**

Background driver tasks are held in a module-level dict so asyncio's weak
reference doesn't collect them mid-flight and so `cancel_run` can find them. Those
tasks are in-process only, so `create_app` registers a startup hook calling
`run_service.reconcile_orphans()` — a uvicorn restart would otherwise leave
`queued`/`running` runs stuck forever with no driver.

---

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

`gui/tests/` (46 `test_*.py`). Every module that needs the web stack starts with
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
