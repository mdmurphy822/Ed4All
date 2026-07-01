# Pipeline invocation — per-stage operator guide

How to drive the `textbook_to_course` pipeline: run it end-to-end, **stop after any
stage to inspect its output**, re-run a single stage, and the environment you need
for a pure-local (no-cloud) run on constrained VRAM.

This is the **operational** companion to two other docs:

- `docs/operations/license-clean-run.md` — the **licensing** angle (which provider
  seats to pin so the trained-SLM corpus is ToS-clean). Read that for *why* to pin a
  seat; read this for *how* to invoke each stage.
- Root `CLAUDE.md` § Quick Start — the canonical `ed4all run` command surface and the
  full phase list.

Canonical entry point:

```bash
ed4all run textbook-to-course --corpus <PATH> --course-name <NAME> [--mode local|api]
```

---

## 0. Read this first — the "outline" vs "rewrite" naming trap

The two-pass content tiers are named in a way that misleads:

| Tier / phase | What it actually does |
|---|---|
| `content_generation_outline` ("outline") | Emits a **block PLAN** — dict blocks (block types, page layout, `target_co_ids`, `key_claims`). **No authored HTML prose.** |
| `content_generation_rewrite` ("rewrite") | The tier that **AUTHORS the prose** — the visible block HTML bodies. The deterministic fallback renderer (`_render_block_fallback_html`) runs *only inside this phase*. |

Consequences:

- **"Let the 7B write the content" = run *through* the rewrite tier.** That tier *is*
  the authoring pass; there is no separate "write" step before it.
- **"Plan-only, inspect the pedagogy structure" = `--stop-after content_generation_outline`.**
  You get the block plan (types, sequencing, floors, key-terms/FAQ pages, prereq
  order) but **no authored prose** — so prose/render-dependent gates
  (`rewrite_html_shape`, `mayer_ctml`, callout render, `rewrite_assessment_item_payload`)
  have nothing to score.
- The four two-pass sub-phases exist **only** under `COURSEFORGE_TWO_PASS=true`. Without
  it, a single `content_generation` phase authors in one shot (seat `COURSEFORGE_PROVIDER`).
  All the framework/pedagogy features (dynamic block planner, IB7 passes, key-terms/FAQ,
  new block types, floors) live on the **two-pass outline surface** and are no-ops in
  single-pass mode.

---

## 1. Phase order

**Single-pass** (`COURSEFORGE_TWO_PASS` unset):

```
dart_conversion → staging → chunking → objective_extraction → source_mapping
→ course_planning → concept_extraction → content_generation
→ assessment_synthesis → packaging → imscc_chunking → trainforge_assessment
→ training_synthesis → libv2_archival → vector_indexing → finalization
```

**Two-pass** (`COURSEFORGE_TWO_PASS=true`): identical, except the single
`content_generation` phase is disabled and replaced by the 4-phase slice:

```
… concept_extraction
→ content_generation_outline → inter_tier_validation
→ content_generation_rewrite → post_rewrite_validation
→ assessment_synthesis → …
```

---

## 2. Per-stage reference

Paths are relative to the Courseforge **export root**
`Courseforge/exports/PROJ-<course-name>-<YYYYMMDDHHMMSS>/` unless marked **LibV2**
(`LibV2/courses/<slug>/`, outside the export root) or **inputs/**.

| # | Phase | Writes (inspect this) | LLM seat | Stop-after / re-run / reuse |
|---|---|---|---|---|
| 1 | `dart_conversion` | staged `*_accessible.html` (out of tree) | SemantiK cascade (local) | `--stop-after dart_conversion`; skip via `--reuse-conversion` or `--skip-dart --dart-output-dir <dir>` |
| 2 | `staging` | `Courseforge/inputs/textbooks/<…>/` | deterministic | `--stop-after staging`; implied by `--skip-dart` |
| 3 | `chunking` | **LibV2** `dart_chunks/chunks.jsonl` + `manifest.json` | deterministic | `--stop-after chunking` |
| 4 | `objective_extraction` | `01_learning_objectives/textbook_structure.json` | `TEXTBOOK_SYNTHESIS_PROVIDER` | `--stop-after objective_extraction` |
| 5 | `source_mapping` | `source_module_map.json` (export root) | deterministic (TF-IDF) | `--stop-after source_mapping` |
| 6 | `course_planning` | `01_learning_objectives/synthesized_objectives.json` | `COURSEPLANNER_PROVIDER` (+ `TEXTBOOK_SYNTHESIS_PROVIDER` for Stage-2) | `--stop-after course_planning`; skip synthesis via `--reuse-objectives <path>` |
| 7 | `concept_extraction` | **LibV2** `graph/concept_graph_semantic.json`, `graph/domain_concept_vocabulary.json`; enriches objectives | `TEXTBOOK_SYNTHESIS_PROVIDER` | `--stop-after concept_extraction` |
| 8 (single) | `content_generation` | `03_content_development/` pages + `course.json` | `COURSEFORGE_PROVIDER` | `--stop-after content_generation` |
| 8a (two-pass) | `content_generation_outline` | `01_outline/blocks_outline.jsonl` | `COURSEFORGE_OUTLINE_PROVIDER` + block plan `ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER` | `--stop-after content_generation_outline`; subcommand `courseforge-outline` |
| 8b (two-pass) | `inter_tier_validation` | `01_outline/blocks_validated.jsonl`; `02_validation_report/report.json` | deterministic (validators) | `--stop-after inter_tier_validation`; subcommand `courseforge-validate` |
| 8c (two-pass) | `content_generation_rewrite` | `04_rewrite/blocks_final.jsonl` + `03_content_development/` pages | `COURSEFORGE_REWRITE_PROVIDER` | `--stop-after content_generation_rewrite`; subcommand `courseforge-rewrite` (+ `--blocks <types>`) |
| 8d (two-pass) | `post_rewrite_validation` | `04_rewrite/blocks_validated.jsonl`; `04_rewrite/02_validation_report/report.json` | deterministic (NLI/embedding scorers) | `--stop-after post_rewrite_validation`; subcommand `courseforge` runs the full 4-phase slice |
| 9 | `assessment_synthesis` | `06_assessments/` (QTI/imsdt/assignment XML + `manifest.json`) | `TRAINFORGE_ASSESSMENT_PROVIDER` | `--stop-after assessment_synthesis`; skip via `generate_assessments=false` |
| 10 | `packaging` | `05_final_package/<course>.imscc`; then `courseforge_validation_report.json` (export root) | deterministic | `--stop-after packaging` |
| 11 | `imscc_chunking` | **LibV2** `imscc_chunks/chunks.jsonl` + `manifest.json` | deterministic | `--stop-after imscc_chunking` |
| 12 | `trainforge_assessment` | **LibV2** `assessments.json` | `TRAINFORGE_ASSESSMENT_PROVIDER` | `--stop-after trainforge_assessment` |
| 13 | `training_synthesis` | `training_specs/instruction_pairs.jsonl` + `preference_pairs.jsonl` | `TRAINFORGE_SYNTHESIS_PROVIDER` | `--stop-after training_synthesis`; skip via `--skip-training` |
| 14 | `libv2_archival` | **LibV2** `courses/<slug>/` + `manifest` | deterministic | `--stop-after libv2_archival` |
| 15 | `vector_indexing` | **LibV2** vector index under `courses/<slug>/` | embedding model (no authoring seat) | `--stop-after vector_indexing` |
| 16 | `finalization` | run summary | deterministic | terminal phase |

---

## 3. Invocation patterns

**Full run:**

```bash
ed4all run textbook-to-course --corpus book.pdf --course-name PHYS_101
```

**Stop after a stage, to inspect it** — `--stop-after` halts *after* the named phase
completes and skips everything downstream. The name is the exact phase name from the
table (validated against the workflow; unknown → error):

```bash
# Build a retrieval-ready course but no training synthesis:
ed4all run textbook-to-course --corpus book.pdf --course-name PHYS_101 \
  --skip-training --stop-after imscc_chunking

# Plan-only: inspect the block plan / pedagogy structure, no prose authored:
COURSEFORGE_TWO_PASS=true ed4all run textbook-to-course --corpus book.pdf \
  --course-name PHYS_101 --stop-after content_generation_outline
```

**Re-run a single stage** without redoing everything upstream:

- **Reuse upstream artifacts:** `--reuse-conversion` (skip SemantiK), `--skip-dart
  --dart-output-dir <dir>` (feed pre-converted HTML), `--reuse-objectives <path>` (pin
  a prior `synthesized_objectives.json`, skip re-synthesis — also removes LLM
  nondeterminism across re-runs).
- **Courseforge stage subcommands** (two-pass only) re-drive just the content slice
  against an existing export, pre-populating upstream phases from disk:
  `courseforge-outline` (outline tier only), `courseforge-validate` (both validator
  seams, no LLM), `courseforge-rewrite` (rewrite tier; pair with `--blocks <types>`),
  `courseforge` (full 4-phase slice). Add `--force` to re-run past `_completed`
  checkpoints. Post-Courseforge phases (packaging…) are always skipped by these — run
  them via the canonical `ed4all run textbook-to-course`.

```bash
export COURSEFORGE_TWO_PASS=true
ed4all run courseforge-validate --course-name PHYS_101          # fire gates, no LLM
ed4all run courseforge-rewrite  --course-name PHYS_101 --blocks assessment_item
```

**Resume** a crashed/stopped run past completed phases:

```bash
ed4all run textbook-to-course --resume WF-20260420-abc12345
```

**Preflight** without dispatching anything: append `--dry-run`.

---

## 4. Timeouts — the knobs that actually fire

Four different timeouts can end a run. For a **slow local-7B synthesis** phase (the
common case), the one that fires is the **per-task in-process** timeout —
`ED4ALL_TASK_TIMEOUT_MINUTES`, *not* the batch or mailbox one.

| Env var | Scope | Default | Raise it when |
|---|---|---|---|
| `ED4ALL_TASK_TIMEOUT_MINUTES` | Per-task wall-clock on the **in-process** provider path (`asyncio.wait_for` around one tool call) | 60 min | **A single long phase-task is slow** (e.g. `course_planning` making hundreds of sequential 7B calls). **This is the knob for a slow local run.** |
| `ED4ALL_BATCH_TIMEOUT_MINUTES` | Whole-**phase** batch wall-clock (wraps the phase's entire parallel task set) | 30 min | A phase with *many parallel* tasks needs longer, and it has no per-phase `batch_timeout_minutes:` in `config/workflows.yaml` (the YAML value wins over this env). |
| `ED4ALL_AGENT_TIMEOUT_SECONDS` | Per-task **mailbox** subagent-dispatch wait only | 1800 s | Only when a task routes through the mailbox (i.e. its seat is *not* pinned to an in-process `local` provider). Note: the mailbox path **fails without retry** on timeout. |
| `ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS` | A single HTTP LLM call | 300 s (Courseforge tiers) / 60 s (Trainforge client) | Individual LLM requests stall (long passages, slow GPU). |

Companion single-call timeouts: `TEXTBOOK_SYNTHESIS_TIMEOUT_SECONDS` (300 s),
`ED4ALL_ANSWER_TIMEOUT_SECONDS` (120 s, retrieval-answer path).

**Why this matters (a real failure mode):** with `COURSEPLANNER_PROVIDER=local`,
`course_planning` runs **in-process** (it bypasses the mailbox), so it is bounded by
`ED4ALL_TASK_TIMEOUT_MINUTES`. If its single task exceeds that, it raises a timeout,
is **retried** (3 attempts total: 1 + `retry_attempts`=2, backoff `30s × 2^n`), and if
every attempt times out the phase is marked failed and produces **no artifact**.
Raising `ED4ALL_BATCH_TIMEOUT_MINUTES` does **not** help here — the *task* timeout
fires first. On a slow box, verify your effective `ED4ALL_TASK_TIMEOUT_MINUTES` and
raise it (e.g. `90`) before a full-book local-7B run.

---

## 5. Pure-local 7B on constrained VRAM (≈8 GB)

A fully local, no-cloud run needs several seats pinned to `local` — and **two of the
defaults point at cloud** (they will silently no-op / fail without a key):

| Env var | Default | Pure-local value | Seats |
|---|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `local` | global routing seat |
| `TEXTBOOK_SYNTHESIS_PROVIDER` | **`anthropic`** | `local` | structure/objective/concept synthesis (stages 4, 6, 7) |
| `COURSEPLANNER_PROVIDER` | subagent | `local` | course-outliner (stage 6) — forces in-process |
| `COURSEFORGE_PROVIDER` | — | `local` | single-pass content authoring |
| `COURSEFORGE_OUTLINE_PROVIDER` | — | `local` | two-pass outline tier |
| `COURSEFORGE_REWRITE_PROVIDER` | **`anthropic`** | `local` | two-pass rewrite (authoring) tier |
| `ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER` | **`nvidia`** | `local` | dynamic block planner — **the landmine**: defaults to the NVIDIA 70B and silently degrades to a fixed plan without the key |
| `CURRICULUM_ALIGNMENT_PROVIDER` | — | `local` | teaching-role classification |
| `TRAINFORGE_ASSESSMENT_PROVIDER` | subagent | `local` | assessment authoring |
| `TRAINFORGE_SYNTHESIS_PROVIDER` | — | `local` | training-pair synthesis |
| `COURSEFORGE_BLOCK_ROUTING_PATH` | shipped policy pins some blocks to Anthropic | `Courseforge/config/block_routing.license_clean.yaml` | per-block-type routing |

The local seat itself:

```bash
export LOCAL_SYNTHESIS_BASE_URL=http://localhost:11434/v1   # Ollama; vLLM :8000/v1
export LOCAL_SYNTHESIS_MODEL=qwen2.5-7b-16k:latest          # see VRAM note
export LOCAL_SYNTHESIS_API_KEY=local
unset ANTHROPIC_API_KEY                                     # belt-and-braces
```

**Model window vs VRAM (8 GB card):** the model + its KV cache must fit or it spills to
CPU and crawls. Measured on an 8 GB card:

- `qwen2.5-7b-16k` → ~6.7 GB resident, **fits** (fast). Good default for planning/synthesis.
- `qwen2.5-7b-32k` → ~8.7 GB, **exceeds** the card → partial CPU offload (slow). Use
  only if a stage truncates at 16k (a `PromptTruncatedError` / truncation tripwire tells
  you), and accept the slowdown.

**Timeouts for the local run** — raise the per-task one and the per-call one:

```bash
export ED4ALL_TASK_TIMEOUT_MINUTES=90        # slow in-process synthesis (§4)
export ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS=600
export TRAINFORGE_REQUIRE_EMBEDDINGS=true    # fail-closed if [embedding] extras missing
```

Then invoke with `--mode local` (the default). See `docs/operations/license-clean-run.md`
for the licensing rationale behind each seat and the Together fallback for larger VRAM.

---

## 6. Corpus-prep gotchas

- **Use the full text, not a chapterless slice.** A subset PDF that starts at a section
  heading (no `h1` chapter) makes the structure extractor promote section headings
  (`§1.1`, `§1.2`) to "chapters" and single-word vocab headings (`Variable`, `Constant`)
  to objective-bearing "sections" — an over-segmentation that explodes objective
  synthesis into hundreds of per-fragment calls and blows the task timeout. A full
  textbook with real `h1` chapter headings levels correctly. Inspect
  `01_learning_objectives/textbook_structure.json` early: an implausibly high
  chapter/section count (e.g. 11 "chapters" / 141 "sections" for 3 real chapters) is the
  tell.
- **Skip re-conversion when iterating.** `--reuse-conversion` (or `--skip-dart
  --dart-output-dir <dir>` pointing at a dir with `*_accessible.html`) reuses a prior
  SemantiK conversion so you can iterate on synthesis/authoring without re-running the
  (slow, model-nondeterministic) PDF cascade.

---

## See also

- `docs/operations/license-clean-run.md` — licensing / ToS-clean seat recipe.
- Root `CLAUDE.md` § Quick Start — canonical command surface; § Opt-In Behavior Flags — full env-var tables.
- `Courseforge/CLAUDE.md` § Operator stage subcommands — the two-pass stage subcommand internals.
- `config/workflows.yaml` — the authoritative phase list, per-phase `batch_timeout_minutes`, and validation gates.
