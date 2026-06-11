# Docker deployment

A two-service `docker compose` stack that serves the Ed4All **Studio** control
plane on `http://localhost:8077` on a clean machine. CPU-only by default.

- **gui** — FastAPI/uvicorn control plane (built from `Dockerfile.gui`).
- **ollama** — local LLM server, backs the grounded-answer path and (optionally)
  the local embedding provider.

## Quickstart

```bash
# 1. Build + start the stack (first build pulls base images + installs deps).
docker compose up -d

# 2. Pull the answer model into Ollama (one-shot; persists on the named volume).
docker compose exec ollama ollama pull qwen2.5:14b-instruct-q4_K_M

# 3. Open Studio.
open http://localhost:8077        # or just browse to it

# 4. Upload a PDF from the Studio UI and run the pipeline.
```

The embedding model (`bge-large` via sentence-transformers) is **not** pulled
through Ollama — it downloads into the GUI container's Hugging Face cache the
first time you build a retrieval index. That cache lives at `HF_HOME=/data/hf-cache`,
which is on the `/data` volume, so it survives container restarts and is only
downloaded once.

## Loopback constraint (read this before changing the network)

The grounded-answer backend (`lib/retrieval/answer_backend.py`) **refuses any
non-loopback answer-provider base URL by design** (Phase IA: no cloud answer
path, ever). The GUI process must therefore reach Ollama at a genuine loopback
address (`localhost` / `127.0.0.0/8` / `[::1]`).

### Linux / WSL2 (default)

Both services run with `network_mode: host`, sharing the host's network
namespace. The GUI reaches Ollama at `http://localhost:11434` — a real loopback
address from the GUI process's perspective. The policy holds with **zero**
weakening. This is the shipped default.

### Docker Desktop / macOS

Host networking does not share `localhost` the same way (the engine runs inside
a VM). A bridged sidecar reachable at `http://ollama:11434` is **not** loopback,
so the answer backend refuses it. Making a bridged sidecar work would require
relaxing the loopback policy — that is a separate product decision and is **not**
made here. On these platforms, run Ollama natively on the host (so the GUI
container talks to a loopback-checked URL) or run the whole stack inside a Linux
VM where host networking behaves as above.

## Operator auth (`ED4ALL_GUI_TOKEN`)

The shipped stack runs `ED4ALL_GUI_MODE=studio`, which mounts **no** operator
surface, so it needs no token. If you switch the `gui` service to
`ED4ALL_GUI_MODE=full` to reach the operator control plane remotely, you **must**
set a shared-secret operator token — the full app exposes env / API-key
management and run-launching, and binding it beyond loopback without a token
leaves those open. Add a compose env line (a commented placeholder ships in
`docker-compose.yml`):

```yaml
    environment:
      ED4ALL_GUI_MODE: full
      ED4ALL_GUI_TOKEN: "change-me-to-a-long-random-secret"   # openssl rand -hex 32
```

With the token set, the operator SPA + its operator-only routes (and the
operator run-log WebSocket, via a `?token=` query param) require
`Authorization: Bearer <token>`; `/api/health` stays open for the healthcheck.
See `gui/README.md` § "Operator auth" for the full route classification.

## Volume layout

| Volume | Mount | Holds |
|--------|-------|-------|
| `ed4all-data` | `/data` (= `ED4ALL_HOME`) | All mutable data: state, LibV2 courses, Courseforge exports, training captures, DART output, and the HF embedding cache (`/data/hf-cache`). |
| `ollama-models` | `/root/.ollama` | Pulled Ollama models (so a restart doesn't re-download multi-GB weights). |

`ED4ALL_HOME=/data` makes every data dir resolve under the single mounted volume
(`lib/paths.py`). The repo itself is **not** bind-mounted — the image already
contains the code; the volume holds only data.

## GPU (optional)

The stack is CPU-only by default. To run Ollama on an NVIDIA GPU:

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   on the host.
2. Uncomment the `deploy.resources.reservations.devices` block under the
   `ollama` service in `docker-compose.yml`.
3. `docker compose up -d`.

CPU inference works without any of this (just slower).

## Troubleshooting

- **`AnswerProviderNotLocal` / answer requests fail** — the GUI resolved a
  non-loopback Ollama URL. Confirm both services use `network_mode: host` and
  that the model-routing base URL is `http://localhost:11434/v1` (or
  `http://127.0.0.1:11434`). See the loopback section above; on Docker
  Desktop/macOS run Ollama on the host.
- **Healthcheck never goes healthy** — check `docker compose logs gui`. The
  probe hits `GET /api/health`; a `503`/connection-refused means uvicorn hasn't
  bound yet (give it the 40s `start_period`) or crashed on import.
- **Model pull is slow / interrupted** — `docker compose exec ollama ollama pull
  qwen2.5:14b-instruct-q4_K_M` is resumable; re-run it. Pulled weights persist
  on `ollama-models`.
- **Embedding model re-downloads every restart** — confirm `HF_HOME` resolves
  under `/data` and the `ed4all-data` volume is mounted (not an anonymous
  volume).

## Related

- MCP-server image: repo-root `Dockerfile` (separate service; not part of this
  compose stack).
- GUI feature reference: `gui/README.md`.
- Launcher (non-Docker): `gui/LAUNCH.md`.
