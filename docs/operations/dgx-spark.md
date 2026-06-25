# DGX Spark deployment

The NVIDIA DGX Spark (GB10, 128GB unified memory) is Ed4All's production
target: a single Spark serves a **70B-class local model** for the whole course
build. This page is the unboxing → building-a-70B-course checklist. The goal is
**one command** once the Spark is serving a model.

**This is a config flip, not new code.** The routing switch
(`MCP/core/workflow_runner.py::_apply_authoring_route_env`) and the
window-parametric rewrite-overflow fix
(`Courseforge/generators/_rewrite_fit_window.py`, `ED4ALL_REWRITE_NUM_CTX`)
already shipped on `dev-v0.3.1`. The Spark serves the 70B **locally** — no
hosted endpoint, no API key, loopback-clean.

> **What we know vs. what you confirm.** We have never run a Spark. The
> serving-stack choice and the exact model artifact are genuinely
> Spark-specific — they are marked **OPERATOR FILL-IN** below with a
> verification command, rather than fabricated. Everything else (the env
> profile, the window math, the one-command invocation) is proven on the
> hosted-70B and 7B paths and carries over.

---

## Checklist (top to bottom)

- [ ] **(a)** Install Ed4All + extras on the Spark.
- [ ] **(b)** Serve a 70B locally (OPERATOR FILL-IN: stack + model artifact).
- [ ] **(c)** Point the launch profile at the Spark endpoint.
- [ ] **(d)** Size the window (`ED4ALL_REWRITE_NUM_CTX≈64000` = server ctx).
- [ ] **(e)** Preflight (`DRY=1` → `--dry-run`); read every `pass`.
- [ ] **(f)** Run the one command.

---

## (a) Prerequisites — install + extras

```bash
git clone <repo> ed4all && cd ed4all
git checkout dev-v0.3.1
python -m venv .venv && . .venv/bin/activate
# Course build needs embeddings (retrieval index) + the server extras.
pip install -e '.[embedding,server]'
ed4all --help          # confirms the CLI is on PATH
```

If you also reuse the host's existing accessible HTML (the `--skip-dart`
path), no `[semantik]` extra is needed — SemantiK only runs for fresh PDF
conversion.

---

## (b) Serve the 70B on the Spark — **OPERATOR FILL-IN**

Ed4All attaches to an **OpenAI-compatible `/v1` endpoint** (the `local`
endpoint row in `config/endpoints.yaml`). Any server that exposes
`POST /v1/chat/completions` works. Two realistic stacks:

| Stack | Default `base_url` | Context-length knob | Notes |
|-------|--------------------|---------------------|-------|
| **Ollama** | `http://localhost:11434/v1` | `num_ctx` (per-request) / `OLLAMA_CONTEXT_LENGTH` (server) | Simplest. Single-stream by default → pace the rewrite fan-out (see troubleshooting). Pull a 70B tag (`ollama pull llama3.3:70b-instruct-q8_0`). |
| **vLLM** | `http://localhost:8000/v1` | `--max-model-len` (server start) | Higher throughput, real concurrency. Start with `--max-model-len 65536`. |

**Decision: Ollama for first-boot bring-up, vLLM for throughput.** Ollama is
the path of least resistance to a *first* 70B course (one `pull`, one `serve`),
and it matches the `local` endpoint's default `base_url`. Move to vLLM when you
want the rewrite fan-out to run concurrently rather than serialized. The
pipeline is indifferent — it only sees a `/v1` URL + a model id.

**Confirm two things on first boot (do not trust this doc's example strings):**

```bash
# 1. WHICH stack + that it answers:
curl -s http://localhost:11434/v1/models | head        # Ollama (or :8000 for vLLM)
# 2. The EXACT model id the server exposes — this string is SPARK_MODEL below.
curl -s http://localhost:11434/v1/models | python -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])'
```

The model-id string from that command is the one and only `SPARK_MODEL` value
the rest of this doc references. (Ollama returns its tag, e.g.
`llama3.3:70b-instruct-q8_0`; vLLM returns the served HF id, e.g.
`meta-llama/Llama-3.3-70B-Instruct`.)

---

## (c) Env profile — point every build tier at the Spark

The cleanest wiring re-homes the existing **`local`** lattice endpoint onto the
Spark by overriding its `base_url`/`model` env vars, then sets `provider=local`
on every build seat. No `endpoints.yaml` edit is needed (see the optional
`spark` row below if you'd rather pin it).

Why `local` and not `--provider nvidia`: the Spark serves the model itself, so
there is no hosted endpoint. The `local` row's `base_url_env`
(`LOCAL_SYNTHESIS_BASE_URL`) and `model_env` (`LOCAL_SYNTHESIS_MODEL`) make it
fully env-overridable, so one base-URL change re-points the whole build —
synthesis, outline, and rewrite tiers all read `LOCAL_SYNTHESIS_*`.

```bash
# (1) Re-home the `local` endpoint onto the Spark server.
export LOCAL_SYNTHESIS_BASE_URL="http://localhost:11434/v1"   # OPERATOR FILL-IN
export LOCAL_SYNTHESIS_MODEL="llama3.3:70b-instruct-q8_0"     # OPERATOR FILL-IN (from step b)

# (2) Per-phase provider seats = local (= the Spark). `--provider local` fills
#     the four authoring envs + tier providers but NOT TEXTBOOK_SYNTHESIS_PROVIDER,
#     so set that one explicitly (reaches objective/planning/concept synthesis).
export COURSEPLANNER_PROVIDER=local       COURSEPLANNER_MODEL="$LOCAL_SYNTHESIS_MODEL"
export TEXTBOOK_SYNTHESIS_PROVIDER=local  TEXTBOOK_SYNTHESIS_MODEL="$LOCAL_SYNTHESIS_MODEL"
export COURSEFORGE_OUTLINE_PROVIDER=local COURSEFORGE_OUTLINE_MODEL="$LOCAL_SYNTHESIS_MODEL"
export COURSEFORGE_REWRITE_PROVIDER=local COURSEFORGE_REWRITE_MODEL="$LOCAL_SYNTHESIS_MODEL"

# (3) Required surfaces.
export COURSEFORGE_TWO_PASS=true   # the all-local routing redirect only fires here
export ED4ALL_AGENT_DISPATCH=true  # in-process lattice dispatch for --mode local
export ED4ALL_ANSWER_PROVIDER=local ED4ALL_ANSWER_MODEL="$LOCAL_SYNTHESIS_MODEL"
```

This mirrors the proven `inputs/nvidia70b/launch-slice.sh` seat set, pointed at
the LOCAL Spark instead of the hosted NVIDIA API. The packaged form lives in
`inputs/spark/launch-build.sh` (below).

### Optional: a `spark` endpoints row (turnkey alternative)

If you prefer naming the Spark as a first-class endpoint instead of overriding
`local`, add this **additive, default-inert** row to `config/endpoints.yaml`
(it changes nothing until a seat resolves `provider: spark`):

```yaml
  spark:
    kind: openai_compatible
    base_url: http://localhost:11434/v1
    base_url_env: SPARK_BASE_URL          # operator overrides per-box
    api_key_env: SPARK_API_KEY
    api_key_default: local                # local server ignores the key
    api_key_required: false
    model_env: SPARK_MODEL
    default_model: llama3.3:70b-instruct-q8_0   # OPERATOR FILL-IN
    loopback_only: true                   # Spark is on-box; never cloud
    provenance_provider: local            # maps onto the frozen `local` value
```

After adding it, run the provenance codegen and select it with
`provider=spark` per seat. **The override-`local` path above is recommended for
first boot** — it needs zero YAML edit and zero codegen, so there is less to get
wrong before the Spark's first course. Promote to a named `spark` row once the
serving setup is stable.

---

## (d) Window sizing — `ED4ALL_REWRITE_NUM_CTX ≈ 64000`

The Courseforge rewrite tier **head-truncates on the 8192-window 7B** (100% of
calls — it silently drops the system-prompt contract). The fix
(`ED4ALL_REWRITE_FIT_WINDOW`, commit `9f81e00`) is **window-parametric**: enable
it and size the window to the server's headroom.

```bash
export ED4ALL_REWRITE_FIT_WINDOW=on
export ED4ALL_REWRITE_NUM_CTX=64000      # MUST equal the server context length
```

…and set the **server** to match (this is the half the pipeline can't set):

```bash
# Ollama: build a Modelfile with `PARAMETER num_ctx 65536`, or export
#   OLLAMA_CONTEXT_LENGTH=65536 before `ollama serve`.
# vLLM:   start with `--max-model-len 65536`.
```

**Why ~48-64k, not 128k.** Our rewrite prompts max ~35k tokens, so a 64k window
clears them with headroom; 128k just burns KV-cache VRAM. The KV/VRAM math on
the Spark's 128GB unified memory:

| Term | Estimate |
|------|----------|
| Llama-3.3-70B weights (FP8) | ≈ 70 GB |
| KV-cache (FP16, ≈320 KB/token) at 64k | ≈ 20 GB |
| **Total + headroom** | fits 128 GB comfortably |

64k is the operational sweet spot: above the ~35k prompt ceiling, well under the
128GB budget. If you ever exceed it (very large grounding sets), the tripwire
fires loudly and escalates the block — it never silently corrupts.

---

## (e) Preflight / verification — `--dry-run`

Validate routing, window, and slug **without any model call** using the launch
script's dry path:

```bash
DRY=1 bash inputs/spark/launch-build.sh
```

This runs `ed4all run … --dry-run`, printing the planned phases, the
`--stop-after imscc_chunking` halt point, and (for the `local` path) the
resolved seats. Read every line before the real run. A `curl … /v1/models`
returning the expected model id (step b) is the only network check you need —
the dry-run itself dispatches nothing.

---

## (f) The one command — build a retrieval-ready 70B course

Fill in the four operator values at the top of `inputs/spark/launch-build.sh`
(`SPARK_BASE_URL`, `SPARK_MODEL`, `CORPUS_DIR`, `COURSE_NAME`), then:

```bash
bash inputs/spark/launch-build.sh
```

which runs the canonical invocation the doc lands on:

```bash
ed4all run textbook-to-course \
  --provider local --mode local \
  --corpus "$CORPUS_DIR" --course-name "$COURSE_NAME" \
  --skip-dart --dart-output-dir "$CORPUS_DIR" \
  --skip-training --stop-after imscc_chunking
```

- `--skip-dart --dart-output-dir` — reuse existing accessible HTML (no SemantiK
  rerun). Omit both to convert PDFs fresh (needs the `[semantik]` extra).
- `--skip-training --stop-after imscc_chunking` — stop at a **retrieval-ready**
  course (askable, no training synthesis → licensing-safe; see
  `docs/LICENSING.md`). Drop both to run through `libv2_archival` +
  `vector_indexing` to a fully archived course.

---

## Troubleshooting

Seeded with what we learned the hard way bringing the 70B-everywhere path up:

- **Grounding silently empty / `chapter_fallback` everywhere** — the
  `--course-name` was not a canonical lowercase-hyphen slug. A mixed-case name
  (e.g. `PHYS_101`) makes chunking write to `LibV2/courses/phys-101/` while
  `plan_course_structure` looks for the raw name → empty grounding. **Use a
  lowercase-hyphen slug** (`phys-101`).
- **Rewrite blocks escalate with no prose / tripwire fires** — the window is
  too small for the grounding. Confirm `ED4ALL_REWRITE_NUM_CTX` equals the
  **server** context length (Ollama `num_ctx`/`OLLAMA_CONTEXT_LENGTH`, vLLM
  `--max-model-len`). The overflow fix is window-parametric: it detects loudly
  rather than corrupting, so a mismatch shows up as escalation, not bad output.
- **Build hangs / `max_retries_exceeded` under load** — the server has limited
  concurrency (a single-stream Ollama serializes). Pace the rewrite fan-out:
  `export COURSEFORGE_REWRITE_CONCURRENCY=1`, or move to vLLM for real
  batching.
- **Synthesis phases ran on the wrong/old model** — `--provider local` does NOT
  fill `TEXTBOOK_SYNTHESIS_PROVIDER`. Export it explicitly (the launch script
  does). Symptom: objective/planning/concept captures show the dev 7B, not the
  Spark 70B.
- **Model id mismatch (404 from the server)** — `SPARK_MODEL` must be the EXACT
  string from `GET /v1/models` (step b), not a guess. Ollama tags and vLLM HF
  ids differ.
- **OOM on the 70B** — drop the window first (`ED4ALL_REWRITE_NUM_CTX=48000` +
  matching server ctx) before changing quantization; KV-cache is the elastic
  term.

---

## Related

- The committed 70B-everywhere routing switch:
  `MCP/core/workflow_runner.py::_apply_authoring_route_env`,
  `cli/commands/run.py` (`--provider`, `--stop-after`, `_nvidia_preflight`).
- The window-parametric overflow fix:
  `Courseforge/generators/_rewrite_fit_window.py`, `ED4ALL_REWRITE_NUM_CTX`.
- Endpoint registry: `config/endpoints.yaml` (the `local` row this profile
  re-homes; the optional `spark` row above).
- Launch profile: `inputs/spark/launch-build.sh` (gitignored — fill in the four
  operator values).
- Hosted-API sibling (paused): `inputs/nvidia70b/launch-slice.sh` — the same
  seat set pointed at the NVIDIA API instead of a local Spark.
- Containerized deploy: `docs/operations/docker.md`.
- Licensing posture (why `--stop-after imscc_chunking`): `docs/LICENSING.md`.
```
