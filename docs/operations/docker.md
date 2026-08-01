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
#    7B Q4 is the compose default (fits an 8GB GPU fully; OK-ish on CPU).
docker compose exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M

# 3. Open Studio.
open http://localhost:8077        # or just browse to it

# 4. Upload a PDF from the Studio UI and run the pipeline.
```

The embedding model (`bge-large` via sentence-transformers) is **not** pulled
through Ollama — it downloads into the GUI container's Hugging Face cache the
first time you build a retrieval index. That cache lives at `HF_HOME=/data/hf-cache`,
which is on the `/data` volume, so it survives container restarts and is only
downloaded once.

## Required: pin the embedding device to CPU in this image

`ED4ALL_EMBEDDING_DEVICE` defaults to **`cuda`** in code — for index builds,
query encoding, and the statistical-tier validator embedder alike — and there is
**no automatic CUDA→CPU fallback** (auto-detection is silent degradation, which
this project forbids). The `gui` image ships `[gui,server,embedding]` on **CPU
torch**, so it must select CPU explicitly or every embedding path raises
`EmbeddingBackendUnavailable` / `EmbeddingModelUnavailable` the first time an
index is built or a semantic query runs:

```yaml
services:
  gui:
    environment:
      # CPU-torch image: the code default is cuda and never falls back.
      ED4ALL_EMBEDDING_DEVICE: cpu
```

Notes:

- This is the **supported** portable configuration, not a degraded one. CPU
  remains a first-class explicit operator selection.
- Do **not** set `ED4ALL_EMBEDDING_DTYPE` to `bf16`/`fp16` here — half precision
  is CUDA-only for these encoders and is refused (rather than silently
  downgraded) when the device is `cpu`.
- `ED4ALL_NLI_DEVICE` behaves differently: it still degrades `cuda`→CPU with a
  warning. Only the embedding knob is fail-loud.
- A **missing** `[embedding]` extra is a separate, unchanged contract: the
  statistical-tier validators emit a warning-severity `EMBEDDING_DEPS_MISSING`
  and pass, unless `TRAINFORGE_REQUIRE_EMBEDDINGS=true`. The device pin above
  does not affect that path.
- The same pin is required on any GPU-less CI runner or release job that
  exercises an embedding path.
- If you later run the GUI image on a CUDA host with a GPU-enabled torch build,
  delete the pin rather than switching it to `auto` — `auto` is not a
  recognized token. Accepted values are `cpu`, `cuda`, `cuda:N`.

## Loopback constraint (read this before changing the network)

The grounded-answer backend (`lib/retrieval/answer_backend.py`) **refuses any
non-loopback answer-provider base URL by design** (Phase IA: no cloud answer
path, ever). The GUI process must therefore reach Ollama at a genuine loopback
address (`localhost` / `127.0.0.0/8` / `[::1]`).

### Shared network namespace (default, all platforms)

The `gui` service joins the `ollama` service's network namespace
(`network_mode: "service:ollama"` — the pod-style sidecar pattern). The two
containers share one `localhost`, so the GUI reaches Ollama at
`http://localhost:11434` — a real loopback address from the GUI process's
perspective. The policy holds with **zero** weakening. Studio's `:8077` is
published as a normal bridged port on the namespace-owning `ollama` service,
so ingress works identically on native Linux, Docker Desktop (Windows/WSL2),
and macOS.

### Why not `network_mode: host`?

An earlier draft used host networking for both services. That only behaves as
intended on native-Linux Docker. On Docker Desktop, "host" means the Desktop
VM's namespace: the in-stack loopback hop still works, but there is **no
ingress path at all** — a host-network service is unreachable from Windows
`localhost` *and* from the WSL distro (verified empirically on Docker Desktop
29.x with host networking enabled). The shared-netns sidecar keeps the
loopback hop and restores normal published-port ingress everywhere. A bridged
sidecar reachable at `http://ollama:11434` would **not** be loopback and the
answer backend would refuse it — that is exactly what the shared namespace
avoids without relaxing the policy.

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
| `ed4all-data` (named volume) | `/data` (= `ED4ALL_HOME`) | Mutable data: state, Courseforge exports, training captures, conversion (SemantiK) output (`semantik-output/`; a legacy `dart-output/` basename is still read for backward compatibility), uploads, and the HF embedding cache (`/data/hf-cache`). <!-- legacy-token: allow --> |
| `./LibV2` (repo bind mount) | `/data/libv2` | **The course library.** The repo checkout's LibV2 is one shared store: courses archived by host (non-Docker) runs appear in the containerized Studio, and pipeline runs launched from the container archive back into the same place. |
| `ollama-models` (named volume) | `/root/.ollama` | Pulled Ollama models (so a restart doesn't re-download multi-GB weights). |

`ED4ALL_HOME=/data` makes every data dir resolve under the single mounted volume
(`lib/paths.py`), with LibV2 overridden onto the repo checkout. The repo's
**code** is not bind-mounted — the image already contains it.

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
  non-loopback Ollama URL. Confirm the `gui` service still has
  `network_mode: "service:ollama"` and that the model-routing base URL is
  `http://localhost:11434/v1` (or `http://127.0.0.1:11434`). See the loopback
  section above.
- **Studio unreachable on :8077** — confirm the `8077:8077` mapping lives on
  the `ollama` service (the namespace owner), not on `gui`; Docker silently
  ignores/rejects port mappings on a container that joined another service's
  namespace.
- **Healthcheck never goes healthy** — check `docker compose logs gui`. The
  probe hits `GET /api/health`; a `503`/connection-refused means uvicorn hasn't
  bound yet (give it the 40s `start_period`) or crashed on import.
- **Model pull is slow / interrupted** — `docker compose exec ollama ollama pull
  qwen2.5:7b-instruct-q4_K_M` is resumable; re-run it. Pulled weights persist
  on `ollama-models`.
- **Embedding model re-downloads every restart** — confirm `HF_HOME` resolves
  under `/data` and the `ed4all-data` volume is mounted (not an anonymous
  volume).
- **`EmbeddingBackendUnavailable` / `EmbeddingModelUnavailable` on the first
  index build or semantic query** — the container is on CPU torch but the code
  default device is `cuda`, and nothing falls back. Set
  `ED4ALL_EMBEDDING_DEVICE: cpu` on the `gui` service (see the section above)
  and recreate it.

## Related

- MCP-server image: repo-root `Dockerfile` (separate service; not part of this
  compose stack).
- GUI feature reference: `gui/README.md`.
- Launcher (non-Docker): `gui/LAUNCH.md`.
