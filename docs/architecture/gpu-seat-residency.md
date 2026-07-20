# GPU residency and the seat lease model

**Status:** describes shipping behavior as of 2026-07-20.
**Code:** `lib/gpu_lifecycle.py`, `lib/vllm_container_lifecycle.py`,
`MCP/core/workflow_runner.py` (`_gpu_lifecycle_sweep`, `_apply_seat_schedule`,
`_maybe_report_seat_schedule`), `SemantiK/semantik_structure/cascade.py`
(`_gpu_lifecycle_release`).
**Config:** the `seats:` annotation on each phase in `config/workflows.yaml`.
**Relates to:** `docs/operations/behavior-flags.md` (per-flag detail),
`docs/operations/pipeline-invocation.md`, `docs/operations/nemotron-spark-serving.md`.

A full `textbook_to_course` build runs more distinct models than fit on one card at once: an OCR/layout
stack, a vision captioner, a large reasoning seat, a 7B authoring seat, a DeBERTa NLI classifier, a
sentence-transformer embedder, and a Bloom classifier. This document describes how the pipeline decides
which of them may hold VRAM at any moment.

The governing idea is a **lease**, not a pool: a model borrows the card for a bounded span, then hands it
back. Nothing in the system negotiates for memory, waits on a lock, or falls back to CPU when the card is
busy. Two independent mechanisms implement this at two different granularities.

---

## 1. Why not a scheduler or a lock

The alternatives were considered and rejected for reasons worth stating, because both are the obvious first
instinct:

- **A VRAM lock / semaphore.** Workers would contend for a shared resource, which means workers block on each
  other, which means throughput is set by the slowest lease-holder and deadlock becomes possible. It also
  makes residency a runtime race rather than a property you can read off the config.
- **CPU fallback under pressure.** This is the silent-degradation failure mode the codebase rejects
  generally. A stage that quietly runs a classifier on CPU because the card was busy produces the same
  artifact, hours later, with no signal that anything went wrong. Misconfiguration should be loud.

The lease model instead makes residency **declarative and deterministic**: the phase sequence in
`config/workflows.yaml` determines the residency sequence, and every hand-off happens at a stage or phase
boundary where it is observable.

---

## 2. Mechanism 1 — the lifecycle sweep (in-process + ollama)

`ED4ALL_GPU_LIFECYCLE`, default **ON**. This is one of very few default-on flags in the tree; it earns that
because it touches residency and timing only, never an output byte.

Resolution is parse-with-fallback biased toward the default: only the explicit falsey tokens
(`0` / `false` / `no` / `off`, case-insensitive) disable it. Unset, blank, garbage, or truthy all keep the
lease behavior. Opt out with `ED4ALL_GPU_LIFECYCLE=0` only when you are deliberately trading residency
churn for speed on a card with headroom.

### Two arms

`lib/gpu_lifecycle.py` exposes two independent release arms:

- **`release_ollama_models(base_url=None, stage=None)`** — enumerates resident models via
  `GET {root}/api/ps` and unloads each with a `keep_alive:0` generate request. Returns the list of names it
  successfully asked to unload. It delegates to the existing `lib/llm/vram_reclaim` primitives rather than
  forking that parser. Evicted models lazy-reload on the next generation request, so eviction costs a reload,
  never correctness.
- **`release_torch(stage=None)`** — runs every unloader registered through `register_releaser`, then `gc` +
  `torch.cuda.empty_cache()`. In-process singletons (NLI, the sentence embedder, the Bloom classifier)
  register themselves here, which is what lets a validator phase hand the card to the next phase without
  the caller knowing which singletons happened to be loaded.

Both are **best-effort and never raise**. A missing `httpx`, a down ollama server, a malformed `/api/ps`
response, or a failed unload all degrade to an empty or partial result. A hand-off that fails leaves a model
resident — recoverable — whereas a hand-off that raises would convert a working build into a failed one.

### Where it fires

- **Phase boundaries.** `WorkflowRunner._gpu_lifecycle_sweep(phase_name)` runs the blocking sweep off the
  event loop via `asyncio.to_thread` at each phase transition. It is called at three sites in `run_workflow`
  and is itself wrapped so a sweep failure logs a warning and never alters `final_status`.
- **SemantiK cascade stage seams.** `cascade.py::_gpu_lifecycle_release` fires at named intra-conversion
  seams — after the Stage-6b captioner, between Stage-5e and Stage-6, after Stage-6, after the Stage-12 theta
  pass, and after the second-pass/OCR-repair stage. Conversion loads several models in sequence inside a
  single pipeline phase, so phase-boundary sweeps alone would be too coarse.

### `stage_lease` — the context-manager form

```python
with stage_lease("vlm-extract", ollama=True, torch=True):
    ...  # stage body
```

Yields immediately; releases the selected arms on exit **including when the body raises**. The body's own
exception still propagates afterward — the lease never substitutes its own failure for the stage's. When the
lifecycle mode is off, this is a pure no-op: the body runs and no release fires, so the flag-off path is
byte-identical control flow. The `ollama` / `torch` selectors let a stage skip a pointless probe (a captioner
that only touched torch weights passes `ollama=False`).

---

## 3. Mechanism 2 — the seat schedule (containerized vLLM seats)

`ED4ALL_SEAT_SCHEDULE`, default **off**. Where mechanism 1 evicts models from a running server, this one
starts and stops the servers themselves.

A **seat** is a logical name (`spark-super`, `spark-glm`, `spark-qwen`) for a vLLM server. Three registries,
all env-driven so a new seat is a config entry and never a code change:

| Registry | Env var | Shape | Purpose |
|---|---|---|---|
| Seat → base URL | `ED4ALL_SEAT_BASE_URLS` | `seat=url,seat=url` | Resolves a logical name to an endpoint |
| Base URL → container | `ED4ALL_VLLM_CONTAINERS` | `url=container,url=container` | Resolves an endpoint to a docker container |
| Seat → launch spec | `ED4ALL_SEAT_LAUNCH_SPECS` | `seat=/path/to/launch.sh` | Enables cold recreate (self-heal) |

Every parser is fail-soft: a malformed token is skipped with a one-time warning, so a partly-garbage registry
still yields its valid pairs.

### Declaring residency in `config/workflows.yaml`

Each phase may carry a `seats:` annotation, validated by `schemas/config/workflows_meta.schema.json`. Three
states, and the distinction between the last two is the one people get wrong:

| Annotation | Meaning |
|---|---|
| `seats: [spark-super]` | This phase needs exactly this seat resident |
| `seats: []` | This phase explicitly needs **no** seat — free the card |
| *(key absent)* | **No opinion.** Seat state carries over from the prior phase unchanged |

`_phase_seats(workflow_type, phase_name)` returns `None` for the absent case and a list otherwise. It is
deliberately **workflow-scoped**, unlike the older `_phase_yaml_block` helper, which returns the first phase
matching a name across all workflows and therefore collides on names shared between `course_generation` and
`textbook_to_course` — `assessment_synthesis` exists in both.

`textbook_to_course` is the workflow that actually uses this today. Its shape: conversion holds
`spark-glm` + `spark-qwen`; the heading judge, course planning, concept extraction, and the whole two-pass
generation and validation span hold `spark-super`; and from `post_rewrite_validation` onward every phase
declares `seats: []`, because packaging, chunking, archival, and indexing want the card free for the
embedding and NLI work rather than held by an idle 120B seat.

### Reconciliation at the phase boundary

`_apply_seat_schedule_blocking(phase_name, desired, current_seats, run_dir)` computes the transition:

1. **First scheduled phase.** When `current_seats` is `None`, seat state is unknown, so *every registered
   seat* not in `desired` becomes a stop candidate. The run begins from a clean, known state rather than
   trusting whatever a previous run left resident.
2. **Stop** `known - desired`, best-effort. A stop failure logs and continues — the worst case is a seat
   holding VRAM it should have freed, which is a performance problem, not a correctness one.
3. **Start** `desired - current`, via `start_seat_coherent`. A seat name absent from `ED4ALL_SEAT_BASE_URLS`
   is a config gap: warn and skip, never a run-failing error. If the seat is genuinely required, the
   pipeline's own provider call surfaces a hard error later.

### The two-phase health check

This is the part that exists because `/v1/models` answering 200 is **not** evidence a seat works. A vLLM seat
that was warm-started via `docker start` can come up live and still emit null or degenerate content — mode
collapse. A liveness-only check hands that seat a whole phase of work and produces garbage.

So a start is two bounded checks with different budgets:

- **Liveness** — poll `/v1/models` until it answers, ceiling `ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS` (default
  1200s / 20 min). A ceiling, not a sleep: it returns the instant the seat is live. 1200s covers a 120B
  NVFP4 cold load.
- **Coherence** — once live, run `coherence_probe` up to `ED4ALL_SEAT_COHERENCE_ATTEMPTS` times (default 3,
  roughly 8s apart), returning on first pass. Deliberately **not** ceiling-bound: a mode-collapsed seat is
  caught in seconds rather than after the 20-minute liveness budget.

`_looks_coherent` is intentionally permissive about *what* the model said and strict only about whether it
said anything at all. It rejects `None`, empty or whitespace-only content, content with no alphanumeric
character, and single-glyph repetition soup (`"!!!!!!"`). It is not a quality judgment — only a null/soup
detector.

### Self-heal, then fail loudly

A warm seat that comes up live but incoherent is **automatically cold-recreated** — `docker rm -f` plus a
relaunch through its `ED4ALL_SEAT_LAUNCH_SPECS` entry — and re-checked once, emitting a `seat_cold_recreate`
decision capture. This turns a manual cold-restart into pipeline behavior and heals in roughly 30-45s.

If the seat still cannot come up coherently, or has no launch spec to recreate from,
`SeatScheduleProbeError` fails the phase **loudly**. The error messages name the reason explicitly
(`start_failed`, `recreate_failed`, `warm_incoherent_no_spec`, `still_incoherent_after_recreate`) and say why
failing beats continuing: *"Failing the phase LOUDLY rather than running it seat-starved."* Running a
generation phase against a collapsed seat produces artifacts that pass structural gates while being
semantically empty — the expensive failure this check exists to prevent.

### Observability

`_maybe_report_seat_schedule` logs the full phase → seat plan once at workflow start: every phase in dispatch
order with its declared seats, `[]` shown as `NO seat — GPU free`, and an absent annotation shown as
`(no opinion → <carried seats>)` so the carry-forward is visible rather than inferred. It also lists any
declared seat name unmapped in `ED4ALL_SEAT_BASE_URLS`, surfacing a config gap before the run reaches the
phase that needs it. Pure observability — no docker, no network — and a no-op unless the flag is on.

---

## 4. How the two mechanisms compose

They operate at different layers and do not coordinate:

- The **seat schedule** decides which vLLM *servers* exist during a phase.
- The **lifecycle sweep** evicts models from *within* a running server (ollama) and releases in-process torch
  singletons.

In practice the seat schedule makes much of the in-phase contention handling moot. `ED4ALL_NLI_EVICT_FOR_CUDA`
(default on) exists to evict a resident ollama model so NLI can take the card; with the seat schedule
enforcing the hand-off at the phase boundary, the card is usually already free before NLI loads. That flag is
retained as the in-phase fallback for the case where generation and NLI share one phase, or where the
lifecycle sweep is opted off.

---

## 5. Metering

Residency is measured, not assumed:

- `ED4ALL_VRAM_DOCTOR` writes `state/runs/<run_id>/vram_trajectory.jsonl`, one row per sample carrying `ts`,
  `phase`, and `resident_models`. The lifecycle sweep contributes a `lifecycle_sweep` row at each hand-off,
  best-effort.
- `BuildCostAggregator` (`lib/aggregators/build_cost.py`) joins those samples to the per-phase wall-clock
  windows from `state/runs/<run_id>/checkpoints/*.json` and reports a per-phase residency span and peak
  resident VRAM. **An absent trajectory file omits the GPU section entirely** rather than reporting zeros —
  a run without `ED4ALL_VRAM_DOCTOR` produced no measurement, and saying so is honest where reporting 0 MiB
  would not be.
- `ed4all doctor`'s `gpu_profile` group treats a box above `ED4ALL_BIG_MEMORY_MIN_MIB` (default 49152, 48 GiB)
  as a concurrent-serving host and emits advisory warnings for small-box defaults left on — including
  `ED4ALL_GPU_LIFECYCLE` and `ED4ALL_NLI_EVICT_FOR_CUDA`. On a card with room for several resident seats, the
  lease's eviction churn costs reload time and buys nothing. Below the threshold, or when the GPU cannot be
  probed, the group is a silent no-op.

---

## 6. Invariants a change here must preserve

1. **Residency never changes an output byte.** Every mechanism here is timing and memory only. This is what
   justifies `ED4ALL_GPU_LIFECYCLE` defaulting on and what any new arm must also satisfy.
2. **Flag-off is byte-identical control flow.** Not merely equivalent output — the release path must not run
   at all.
3. **Hand-off failures degrade; starvation fails loudly.** A failed *release* logs and continues (worst case:
   memory not freed). A failed seat *acquisition* raises (worst case avoided: a phase running against a dead
   or collapsed seat).
4. **Liveness is never taken as health.** Any new seat type needs a content probe, not just an endpoint
   check.
5. **Seats are registry data, never subclasses.** Adding a seat is an entry in `ED4ALL_SEAT_BASE_URLS`,
   `ED4ALL_VLLM_CONTAINERS`, and `ED4ALL_SEAT_LAUNCH_SPECS` plus a `seats:` annotation — never a new class.
   This mirrors the OpenAI-compatible provider-registry rule.
