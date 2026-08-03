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
semantik_conversion → staging → chunking → objective_extraction → source_mapping
→ course_planning → concept_extraction → content_generation
→ assessment_synthesis → packaging → imscc_chunking → trainforge_assessment
→ training_synthesis → libv2_archival → vector_indexing
→ [training → post_training_validation → evaluation] → finalization
```

The bracketed tail is **opt-in** (`--with-training`); a default build skips all
three and still reaches `finalization`, because a skipped phase stamps
`_completed` and the dependency check reads only that.

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
`Courseforge/exports/<project-export>/` unless marked **LibV2**
(`LibV2/courses/<slug>/`, outside the export root) or **inputs/**.

| # | Phase | Writes (inspect this) | LLM seat | Stop-after / re-run / reuse |
|---|---|---|---|---|
| 1 | `semantik_conversion` | staged `*_accessible.html` (out of tree) | SemantiK cascade (local) | `--stop-after semantik_conversion`; skip via `--reuse-conversion` or `--skip-conversion --semantik-output-dir <dir>`. Old paused runs still `--resume` via the legacy conversion-phase read-alias. |
| 2 | `staging` | `Courseforge/inputs/textbooks/<…>/` | deterministic | `--stop-after staging`; implied by `--skip-conversion` |
| 3 | `chunking` | **LibV2** `semantik_chunks/chunks.jsonl` + `manifest.json` | deterministic | `--stop-after chunking` |
| 4 | `objective_extraction` | `01_learning_objectives/textbook_structure.json` | `TEXTBOOK_SYNTHESIS_PROVIDER` | `--stop-after objective_extraction` |
| 5 | `source_mapping` | `source_module_map.json` (export root) | deterministic (TF-IDF) | `--stop-after source_mapping` |
| 6 | `course_planning` | `01_learning_objectives/synthesized_objectives.json` | `COURSEPLANNER_PROVIDER` (+ `TEXTBOOK_SYNTHESIS_PROVIDER` for Stage-2) | `--stop-after course_planning`; skip synthesis via `--reuse-objectives <path>` |
| 7 | `concept_extraction` | **LibV2** `graph/concept_graph_semantic.json`, `graph/domain_concept_vocabulary.json`; enriches objectives | `TEXTBOOK_SYNTHESIS_PROVIDER` | `--stop-after concept_extraction` |
| 8 (single) | `content_generation` | `03_content_development/` pages + `course.json` | `COURSEFORGE_PROVIDER` | `--stop-after content_generation` |
| 8a (two-pass) | `content_generation_outline` | `01_outline/blocks_outline.jsonl` | `COURSEFORGE_OUTLINE_PROVIDER` + block plan `ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER` | `--stop-after content_generation_outline`; subcommand `courseforge-outline` |
| 8b (two-pass) | `inter_tier_validation` | `01_outline/blocks_validated.jsonl`; `02_validation_report/report.json` | deterministic (validators) | `--stop-after inter_tier_validation`; subcommand `courseforge-validate` |
| 8c (two-pass) | `content_generation_rewrite` | `04_rewrite/blocks_final.jsonl` + `03_content_development/` pages | `COURSEFORGE_REWRITE_PROVIDER` | `--stop-after content_generation_rewrite`; subcommand `courseforge-rewrite` (+ `--blocks <types>` / `--block-ids <ids>` / `--pages <ids>`) |
| 8d (two-pass) | `post_rewrite_validation` | `04_rewrite/blocks_validated.jsonl`; `04_rewrite/02_validation_report/report.json` | deterministic (NLI/embedding scorers) | `--stop-after post_rewrite_validation`; subcommand `courseforge` runs the full 4-phase slice |
| 9 | `assessment_synthesis` | `06_assessments/` (QTI/imsdt/assignment XML + `manifest.json`) | `TRAINFORGE_ASSESSMENT_PROVIDER` | `--stop-after assessment_synthesis`; skip via `generate_assessments=false` |
| 10 | `packaging` | `05_final_package/<course>.imscc`; then `courseforge_validation_report.json` (export root) | deterministic | `--stop-after packaging` |
| 11 | `imscc_chunking` | **LibV2** `imscc_chunks/chunks.jsonl` + `manifest.json` | deterministic | `--stop-after imscc_chunking` |
| 12 | `trainforge_assessment` | **LibV2** `assessments.json` | `TRAINFORGE_ASSESSMENT_PROVIDER` | `--stop-after trainforge_assessment` |
| 13 | `training_synthesis` | `training_specs/instruction_pairs.jsonl` + `preference_pairs.jsonl` | `TRAINFORGE_SYNTHESIS_PROVIDER` | `--stop-after training_synthesis`; skip via `--skip-training` |
| 14 | `libv2_archival` | **LibV2** `courses/<slug>/` + `manifest` | deterministic | `--stop-after libv2_archival` |
| 15 | `vector_indexing` | **LibV2** vector index under `courses/<slug>/` | embedding model (no authoring seat) | `--stop-after vector_indexing` |
| 16 | `training` | **LibV2** `models/<model_id>/` (adapter + `model_card`) | none (the trainer wants the whole card — `seats: []`) | opt-in via `--with-training`; `--stop-after training`; always skipped under `--skip-training` |
| 17 | `post_training_validation` | no artifact — gates only (`eval_gating`, `family_completeness`, both critical) | deterministic (validators) | opt-in via `--with-training`; `--stop-after post_training_validation` |
| 18 | `evaluation` | merged additively into `<model_dir>/eval/eval_report.json` | held-out harness + grounded-answer arms | opt-in via `--with-training`; `--stop-after evaluation` |
| 19 | `finalization` | run summary | deterministic | terminal phase |

Phases 16-18 are the **opt-in training tail**. They are skipped unless
`--with-training` is passed, and `--skip-training` wins when both are given. The
flag also works on `--resume` (it patches the persisted params before the
resumed phases run — without that patch it would be a silent no-op). Training an
already-archived course without rebuilding stays
`ed4all run trainforge_train --course-name <slug>`.

### 2.1 Pacing (`duration_weeks`) — how weeks are chosen when `--weeks` is unset

- `objective_extraction` first sets a provisional `duration_weeks = max(8, len(chapters))`
  (chapter-driven).
- `course_planning` (WS5 §3.2) then **re-scales** to the TERMINAL-objective-driven
  `max(8, num_tos)` — one week-block per WS1-clustered TO (the TO/cluster count is the
  authoritative pacing signal). COs distribute WITHIN each TO's week-block via the §2.2
  coverage-safe **ceil-stride slicer** (no CO dropped). It falls back to the legacy
  CO-count formula `max(8, ceil(len(chapter_objectives) / WAVE18_COS_PER_WEEK))` only when
  no TOs are available.
- The re-scale is **skipped** for `--weeks N` and `--reuse-objectives` runs (the operator's
  pacing decisions are preserved verbatim).
- The per-week CO placement cap is the separate, UNCONDITIONAL `ED4ALL_COS_PER_WEEK_CAP`
  (default auto) — see `docs/operations/behavior-flags.md`.

---

## 3. Invocation patterns

**Full run:**

```bash
ed4all run textbook-to-course --corpus book.pdf --course-name <course-name>
```

**Auto-named run (`--auto-name`)** — opt-in H1-derived, run-timestamped course
slugs (owner directive: slugs inherit the H1 title SemantiK creates, combined
with the run-init date/time). `--course-name` becomes the PROVISIONAL identity
(run_id / `runtime/state/runs/<run_id>` / log tagging; omit it and the provisional is
derived from the corpus filename). Immediately after `semantik_conversion`
(+ `heading_judge`) completes — and before `staging`, the first phase that
consumes identity into artifacts — the runner reads the accessible HTML's
`<h1>` and rebinds the workflow's `course_name` to:

```
canonical_slug(h1_title)   # lib/ontology/slugs.py, whole-token capped at 60 chars
  + "-" + YYYYMMDD-HHMM    # the run-INIT timestamp (workflow created_at)
# e.g. intro-to-linear-algebra-20260722-0704
```

Every post-conversion phase (staging, chunking, LibV2 archival, vector index…)
mints artifacts under the final slug; the raw `<h1>` is recorded on the params
as `display_title` for manifests/GUI, the old name survives as
`provisional_course_name`, and one `course_identity_rebind` decision-capture
event records the rebind. The resolution is persisted on the workflow state,
so a `--resume` keeps the same identity. **Honest fallbacks** (the provided
name is KEPT, reason logged + captured — never a fabricated title): multi-file
corpus (no single `<h1>` names it), missing `<h1>`, structural heading
(`Chapter 3`, `Part IV`…), numeric-only or >120-char junk, or a title with no
sluggable content. Default off → byte-identical current behavior.

```bash
ed4all run textbook-to-course --corpus book.pdf --auto-name          # provisional from filename
ed4all run textbook-to-course --corpus book.pdf --course-name tmp-book --auto-name
```

**Stop after a stage, to inspect it** — `--stop-after` halts *after* the named phase
completes and skips everything downstream. The name is the exact phase name from the
table (validated against the workflow; unknown → error):

```bash
# Build a retrieval-ready course but no training synthesis:
ed4all run textbook-to-course --corpus book.pdf --course-name <course-name> \
  --skip-training --stop-after imscc_chunking

# Plan-only: inspect the block plan / pedagogy structure, no prose authored:
COURSEFORGE_TWO_PASS=true ed4all run textbook-to-course --corpus book.pdf \
  --course-name <course-name> --stop-after content_generation_outline
```

**Re-run a single stage** without redoing everything upstream:

- **Reuse upstream artifacts:** `--reuse-conversion` (skip SemantiK; when set AND the
  prior `{stem}_accessible.html` + sidecars exist at the conversion output path, the
  cascade is skipped and the prior artifacts reused — the re-run model-nondeterminism
  guarantee; mirrors the `ED4ALL_REUSE_CONVERSION` env var, and the flag wins when both
  are set; see `SemantiK/CLAUDE.md §3.3a`), `--skip-conversion --semantik-output-dir <dir>` (feed
  pre-converted HTML), `--reuse-objectives <path>` (pin a prior objectives JSON, skip
  the course-outliner re-dispatch — removes LLM nondeterminism drift across re-runs that
  breaks chunk `learning_outcome_refs` continuity). `--reuse-objectives` accepts **both**
  the Courseforge synthesized form (`terminal_objectives` / `chapter_objectives`) and the
  LibV2 archive form (`terminal_outcomes` / `component_objectives`); the runner normalizes
  to the Courseforge form on disk before downstream phases consume it.
- **Courseforge stage subcommands** (two-pass only) re-drive just the content slice
  against an existing export, pre-populating upstream phases from disk:
  `courseforge-outline` (outline tier only), `courseforge-validate` (both validator
  seams, no LLM), `courseforge-rewrite` (rewrite tier; pair with `--blocks <types>`,
  `--block-ids <ids>`, and/or `--pages <page/module ids>`),
  `courseforge` (full 4-phase slice). Add `--force` to re-run past `_completed`
  checkpoints. Post-Courseforge phases (packaging…) are always skipped by these — run
  them via the canonical `ed4all run textbook-to-course`.
- **Rewrite re-roll scope (I4).** The rewrite tier normally reuses the
  `blocks_final.jsonl`-cached successful rewrites and re-rolls only failed/degraded
  blocks. Three ADDITIVE flags widen that eviction so specific blocks re-author even
  after a prior success (all three unset → byte-identical failure-driven reuse):
  `--blocks <types>` (stage 1 — every block of a TYPE),
  `--block-ids <id1,id2>` (stage 2 — exact block-instance IDs as they appear in the
  outline / `blocks_final.jsonl`, shape `{page_id}#{block_type}_{slug}_{idx}`), and
  `--pages <page1,page2>` (stage 2 — an exact `page_id` e.g. `week_01_content_02`, or a
  module prefix e.g. `week_01` for a whole week/module). The three compose. An unknown
  `--block-ids` id or `--pages` token (matching no outline block) fails the rewrite
  phase LOUDLY — never a silent no-op.

```bash
export COURSEFORGE_TWO_PASS=true
ed4all run courseforge-validate --course-name <course-name>          # fire gates, no LLM
ed4all run courseforge-rewrite  --course-name <course-name> --blocks assessment_item
ed4all run courseforge-rewrite  --course-name <course-name> \
  --block-ids 'week_03_content_01#objective_intro_0'            # one block instance
ed4all run courseforge-rewrite  --course-name <course-name> --pages week_03   # one module
```

**Resume** a crashed/stopped run past completed phases:

```bash
ed4all run textbook-to-course --resume <RUN_ID>
```

**Preflight** without dispatching anything: append `--dry-run`.

### 3.1 Hosted large-model build profile (`--provider nvidia`)

SETUP only — gated on a later RUN discussion; **nothing dispatches to the cloud
seat by default.** `--provider nvidia` (the vendor endpoint-registry key; also
via `LLM_PROVIDER=nvidia`) on a `COURSEFORGE_TWO_PASS=true` run:

- redirects the block-routing YAML to `Courseforge/config/block_routing.nvidia_large.yaml`
  — the **rewrite** tier runs on the hosted large model (`meta/llama-3.3-70b-instruct`
  via `NVIDIA_LARGE_MODEL`); the **outline** first draft stays local 7B;
- pins `NVIDIA_LARGE_MODEL` to the large model (closes the 30B-nano registry-default leak);
- routes the **textbook-synthesis** seat (`objective_extraction` / `course_planning` /
  `concept_extraction`) to `nvidia`;
- pins the **training** seat (`TRAINFORGE_SYNTHESIS_PROVIDER`) **LOCAL** by this branch
  (licensing — the SLM training corpus must never route through Llama-3.3).

All of the above is `setdefault` (explicit per-phase overrides win). The canonical
cloud-model knob is `NVIDIA_LARGE_MODEL` / the YAML, **never** `COURSEFORGE_REWRITE_MODEL`
(dead on the cloud tier). Run `--dry-run` first for the "wired but not firing" routing
preflight (resolve + assert, no dispatch):

```bash
export COURSEFORGE_TWO_PASS=true
ed4all run textbook-to-course --provider nvidia --course-name <course-name> \
  --corpus slice.pdf --skip-conversion --skip-training \
  --stop-after imscc_chunking --dry-run   # preflight: resolve+assert, NO dispatch
```

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
| `ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER` | **`nvidia`** | `local` | dynamic block planner — **the landmine**: defaults to the hosted large seat (registry key `nvidia`) and silently degrades to a fixed plan without the key |
| `CURRICULUM_ALIGNMENT_PROVIDER` | — | `local` | teaching-role classification |
| `TRAINFORGE_ASSESSMENT_PROVIDER` | subagent | `local` | assessment authoring |
| `TRAINFORGE_SYNTHESIS_PROVIDER` | — | `local` | training-pair synthesis |
| `COURSEFORGE_BLOCK_ROUTING_PATH` | shipped policy pins some blocks to Anthropic | `Courseforge/config/block_routing.license_clean.yaml` | per-block-type routing |

The local seat itself:

```bash
export LOCAL_SYNTHESIS_BASE_URL=http://localhost:11434/v1   # Ollama; vLLM :8000/v1
export LOCAL_SYNTHESIS_MODEL=qwen2.5-7b-16k:latest          # see VRAM note
export LOCAL_SYNTHESIS_API_KEY=local
export TEXTBOOK_SYNTHESIS_NUM_CTX=16384                     # the model's TRUE serving window
unset ANTHROPIC_API_KEY                                     # belt-and-braces
```

**`TEXTBOOK_SYNTHESIS_NUM_CTX` is load-bearing on the objective-synthesis path:** unset,
the Stage-2 window packer falls to the shared `resolve_num_ctx()` default of 4096, and the
fixed per-window costs (system prompt + the 4096-token response reserve + margins) alone
exceed that — the window budget goes negative and every chunk becomes its own degenerate
1-chunk window (one LLM call per chunk, objectives synthesized from isolated ~1500-char
fragments). Set it to the local model's actual serving window (verify with
`curl -s http://localhost:11434/api/show -d '{"name":"<model>"}'` → Modelfile `num_ctx`).

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
- **Skip re-conversion when iterating.** `--reuse-conversion` (or `--skip-conversion
  --semantik-output-dir <dir>` pointing at a dir with `*_accessible.html`) reuses a prior
  SemantiK conversion so you can iterate on synthesis/authoring without re-running the
  (slow, model-nondeterministic) PDF cascade.

### 6.1 The structure extractor's heading-level contract

`lib/semantic_structure_extractor` maps HTML headings by **level, not by wrapper**:

| Level | Role |
|---|---|
| `h1` | **book / document title** (there should be exactly one) |
| `h2` | **chapter** — an `h1`'s `h2` children become chapters; a bare `h2` is itself a chapter |
| `h3` | **section** |
| `h4`–`h6` | subsection |

Getting the level wrong is the single biggest source of over-segmentation when you hand-
prepare HTML:

- **Emit exactly one `h1`** (the book title). Using `h1` *per chapter* makes the extractor
  treat each chapter as a *document title* and promote its section `h2`s to chapters —
  e.g. 10 chapters × ~9 sections → **91 "chapters"**.
- **Chapter delimiters must be `h2`, section titles `h3`.** If your page bodies put
  section titles at `h2`, demote every body heading one level (`h2→h3`, `h3→h4`, …; caps
  at `h6`).
- **Duplicate heading *text* across chapters** (a generic "Introduction" / "Chapter
  Outline" on every chapter's intro page) gets bucketed into the *first* chapter (its
  section count balloons while the rest stay clean). Strip or uniquify generic nav
  headings.
- **Verify before you synthesize** (cheap, deterministic — no LLM):

  ```python
  from lib.semantic_structure_extractor.semantic_structure_extractor import extract_textbook_structure
  res = extract_textbook_structure("<staged>/<stem>_accessible.html")
  print(res["chapter_count"], [len(c["sections"]) for c in res["chapters"]])
  ```

  or `--stop-after objective_extraction` and read `textbook_structure.json`. Confirm the
  chapter count and per-chapter section counts match the real book before committing to a
  multi-hour synthesis.

### 6.2 Importing a documentation tree as a source corpus

For a Markdown or MDX documentation tree, use the deterministic, LLM-free
`ed4all import-docs` workflow. It emits one clean accessible-HTML page per source
document plus an `import_manifest.json`; when `mkdocs.yml` is present, its navigation
defines reading order, with unlisted documents appended in a stable order.

```bash
ed4all import-docs ./<docs-tree> --output ./<imported-corpus>
ed4all run textbook-to-course --corpus ./<imported-corpus> \
  --course-name <course-name>
```

Before starting the pipeline:

- Keep one meaningful document title per source page and use nested headings for its
  sections. The importer preserves that hierarchy for downstream structure extraction.
- Remove navigation-only pages, duplicated boilerplate, generated indexes, answer keys,
  and other non-instructional material that would contaminate `chapter_text`.
- Inspect `import_manifest.json` for the resolved document order and any escaped-markup
  leak markers. Correct the source and re-import when the manifest reports a leak.
- Use `--source-name`, `--license-note`, and `--provenance-tag` when the corresponding
  manifest metadata is appropriate. Use neutral, publishable values; never embed private
  corpus names, course slugs, machine paths, or source identities in tracked files.
- Keep source material and generated corpus artifacts in ignored operator-data locations.
  Only the generic importer and public operating instructions belong in version control.

### 6.3 Inspect what actually feeds the model

Before a long synthesis, confirm the two real Stage inputs are clean:

- **`chapter_text`** (Stage-1 draft-TO input) — from `extract_textbook_structure`, per
  chapter. Check it's real chapter prose, free of front-matter (author/funder/title
  pages) and answer-key text (`"Try It"`, `"Answer Key"`, publisher-site-URL vendor
  chrome). Front-matter bleed is what turns a real chapter into a garbage terminal
  objective.
- **Chunks** (Stage-2 grounding) — `--stop-after chunking`, then read **LibV2**
  `courses/<slug>/semantik_chunks/chunks.jsonl`. Confirm the chunk count is *book-scale* and
  that chunks are attributed across **all** chapters (a book-wide corpus siloed onto ch1
  is a structure-attribution bug). Stage-2 groups these chunks per chapter into
  `num_ctx`-sized windows and synthesizes candidate objectives against them, so a wrong
  chunk→chapter map directly corrupts the objectives.

---

## 7. Graceful stop (`ed4all stop`)

Every long-running stage polls a filesystem **stop sentinel** at its unit
boundaries (the same points where the fingerprinted resume sidecars append). On
a stop the running unit finishes, checkpoints, and the phase pauses — it is
never marked `failed`, and **worst-case loss is one in-flight LLM call**. One
documented exception: the pooled staged-HTML parse drains a poolful of files
rather than one unit — see the per-phase table below.

### Requesting a stop

```bash
ed4all stop <workflow_id|run_id>   # run-scoped: pause ONE run
ed4all stop --all                  # global: pause EVERY run
ed4all stop --clear-all            # remove the global STOP_ALL sentinel
```

- **Run-scoped** resolves the target against the `RUNNING` workflows in
  `runtime/state/workflows/*.json` (matching the workflow id or its `params.run_id`) and
  drops `<state_runs>/<run_id>/control/STOP_REQUESTED`. A target that matches no
  live run still writes a best-effort sentinel — a stray sentinel is a harmless
  no-op. `ed4all stop` never touches the running process; it only drops the
  sentinel and reports the currently-RUNNING workflows (a run whose state file
  is older than 24 h is annotated *possibly stale*).
- **`--all`** writes the operator-owned global `<state_runs>/STOP_ALL`. This is
  the master "halt everything" switch: **while it exists the runner refuses to
  start ANY run — fresh or `--resume`** — with an error naming the only command
  that clears it. It is *never* auto-cleared.
- **`--clear-all`** removes `STOP_ALL` (and any resolved run-scoped sentinel).
  Run this before you can launch a new run after a global stop.

A run-scoped sentinel is auto-cleared at the *start* of that run, so a paused
run resumes cleanly with a plain `--resume` (below); the operator-owned global
one is not.

### SIGTERM / SIGINT to a live `ed4all run`

Sending `SIGTERM` or `SIGINT` (Ctrl-C) to a running `ed4all run` is the **same
request** — the first signal writes the run-scoped sentinel and the run drains
to a checkpoint. A **second** signal restores the default disposition and
**hard-kills** the process (a hard kill re-stamps the still-`RUNNING` state file
`paused` so the resume path sees the truth).

### Exit code 3 = paused; resume with `--resume` (never `--force`)

A gracefully stopped run exits **code 3** (distinct from a completed run's 0 and
a gate/status failure's 2) and prints a resume hint. Resume it with a **plain**:

```bash
ed4all run <workflow> --resume <workflow_id>
```

The paused phase carries status `paused`; `--resume` replays from the resume
sidecars (each records exactly the units already completed, so no completed unit
re-runs). **Do NOT pass `--force` after a stop** — `--force` clears the resume
sidecars, discarding the checkpointed work you just paused to keep.

### Timeouts now become pauses (not hard failures)

The timeout knobs in §4 no longer hard-cancel first — they drain to a checkpoint:

- **Batch timeout** (`ED4ALL_BATCH_TIMEOUT_MINUTES` / per-phase
  `batch_timeout_minutes:`) writes the run-scoped stop sentinel at the deadline
  and grants a **grace window of `min(600s, 10% of the timeout)`** for in-flight
  workers to reach a unit boundary and return `paused`. A grace-drained batch
  surfaces `paused`, **not** `TIMEOUT`. Only if the grace *also* expires (an
  unresponsive worker that never consults the sentinel) does the executor clear
  the timeout-authored sentinel, hard-cancel, and mark the still-unfinished
  tasks `TIMEOUT` as before.
- **Task timeout** (`ED4ALL_TASK_TIMEOUT_MINUTES`) grace-drains the single slow
  task (via a per-task, non-cancelling channel — it never writes the run
  sentinel, since one slow task must not pause the whole run) and then keeps the
  **existing `TIMEOUT` classification + transient-retry ladder** unchanged. The
  resume sidecar makes a retry lossless — completed units are replayed, not
  re-run.
- **Mailbox waiter** (`ED4ALL_AGENT_TIMEOUT_SECONDS`) uses the same
  sentinel-plus-grace pattern: at the deadline it writes the run-scoped sentinel
  and extends by `min(600s, 10% of the timeout)`; a completion within grace (a
  normal result or the `GRACEFUL_STOP` envelope the servicer marshals) is handled
  normally / as `paused`, and only a grace-expiry surfaces the timeout.

### Worst-case loss by phase

Loss is bounded by the size of the *unit* each loop checkpoints:

| Phase / lane | Checkpoint unit | Worst-case loss on stop |
|---|---|---|
| Objective synthesis (Stage-2 windows / TO clusters) | one `num_ctx` window / cluster | one in-flight LLM call |
| Concept extraction (Stage-3 windows) | one window | one in-flight LLM call |
| Course outline tier | one unit | one in-flight LLM call |
| Content rewrite — batched lane | one router **round** (per-block winners land as they resolve) | one in-flight round |
| Assessment synthesis (W10) | one `(kind, terminal_id)` unit | one in-flight LLM call |
| Objective review / bloom-complement / sub-objectives | per-chunk / per-complement / per-CO | one in-flight LLM call |
| Training-pair synthesis | one pair | one in-flight pair |
| SLM training | trainer-native step/epoch checkpoint | since the last saved step |
| SemantiK conversion (`semantik_conversion`) | one **chapter** (paused resume auto-reuses finished `{stem}_accessible.html` + `.quality.json`) | one chapter — cascade seams land the mid-chapter stop at post-Stage-5e/pre-Stage-6, pre-Stage-13, and Stage-6 adapter-batch boundaries |
| Chunking / IMSCC chunking — staged-HTML parse (`ED4ALL_HTML_PARSE_WORKERS` > 1) | one **file**, but dispatched across a process pool | ~`3 × worker_count` in-flight files — the pool drains its outstanding work before the phase pauses. No LLM call is lost (the parse is deterministic stdlib work) and every drained file is simply re-parsed on resume. Set `ED4ALL_HTML_PARSE_WORKERS=1` for single-unit granularity on the byte-identical serial path. |
| Vector indexing | aborts pre-write | none — never a partial index |

### Known follow-up (out of scope)

GUI-enqueued runs share the orchestrator, so the sentinel works for them too —
but the GUI run-registry does not yet surface a `paused` status (the run shows
as still-running until refreshed). Adding `paused` to the run-registry vocabulary
and the Activity bridge is a tracked follow-up.

---

## 8. Scan-corpus build recipe — SemantiK vision extraction + reasoning-QC (vLLM-first)

A scanned / image-only PDF (no text layer) needs a different seat layout than the
born-digital pure-local recipe in §5: the textbook-tuned council BERTs collapse
off-domain, so structure + fidelity lean on a **vision-language model (VLM)** for
extraction and a **reasoning model** for a final structure / reading-order QC pass.
This section is the validated flag stack for that build. Genericize `<CORPUS>.pdf`
and `<COURSE_NAME>` to your own corpus — no corpus-specific paths belong in a run.

> **Superseded scripts (doc-only note):** the older per-corpus launcher scripts
> under your gitignored `inputs/<corpus>/` dir are **superseded by
> this recipe** and should not be used — they hard-code a stale
> `.venv` interpreter and an Ollama Qwen seat (both traps below). The scripts are
> left in the tree for reference only; drive new scan builds from the env stack
> here plus `ed4all run textbook-to-course` (or the SemantiK standalone convert
> path).

### 8.1 vLLM-first seat layout (single endpoint, continuous batching)

Serve ONE vLLM endpoint with a reasoning-capable multimodal model (validated:
`nemotron-3-nano-omni`) and route every SemantiK seat at it. vLLM's continuous
batcher (`--max-num-seqs N`) keeps the card saturated across the fan-out pools;
an Ollama seat serializes (batch = 1) and strands the batcher, so **do not** route
the reviewer / extraction / QC pools at Ollama even if one is running.

| Seat | Endpoint | Task | Thinking |
|---|---|---|---|
| TEXT reviewer (structure / block review) | vLLM `:8000` | structural-edit review, batched N-wide | OFF |
| Multimodal VLM extraction | vLLM `:8000` | per-page image → markdown transcription (a copy task) | OFF (~10× faster) |
| Stage-9b reasoning-QC | vLLM `:8000` | document-level structure + reading-order judgment | **ON** (reasoning is the point) |

One endpoint serves all three; the thinking-on/off split is per-request, so
extraction (thinking-off) and QC (thinking-on) coexist in the same process.

### 8.2 The env stack

```bash
# --- license-clean, pure-local (no cloud anywhere) ---
unset ANTHROPIC_API_KEY NVIDIA_API_KEY
export LLM_MODE=local
export LLM_PROVIDER=local

# --- Stage-6 / gap-fill specialist seat -> the vLLM endpoint (NOT a local GGUF) ---
# TRAP: SEMANTIK_SPECIALIST_PROVIDER unset = the local GGUF arm for the Stage-6
# Phase-1 draft AND the assembler pass-9b gap-fill -> a llama-cpp ImportError on a
# non-contiguous slice (contiguous chapters never hit the gap path, so it stays a
# silent no-op until a gap appears). Set the provider AND model explicitly — the
# endpoint arm's literal model default would 404 on a local vLLM.
export SEMANTIK_SPECIALIST_PROVIDER=endpoint
export SEMANTIK_SPECIALIST_PHASE1_PROVIDER=endpoint
export SEMANTIK_SPECIALIST_MODEL=<vllm-model-id>
# Second gate (the "dereliction fix"): provider=endpoint alone does NOT displace
# the local GGUF authoring tier — gap-fill / Stage-6 force_local unless this
# explicit opt-in is set. Safe when the endpoint is a loopback, license-clean
# on-device seat (displacement then changes no licensing).
export SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE=1

# --- text reviewer seat -> the vLLM endpoint, batched N-wide ---
export SEMANTIK_STRUCTURE_REVIEW=on
export SEMANTIK_BLOCK_REVIEW=on
export SEMANTIK_STRUCTURE_REVIEW_TEMPERATURE=0
export SEMANTIK_SPECIALIST_BASE_URL=http://localhost:8000/v1
export SEMANTIK_SPECIALIST_API_KEY=local
export SEMANTIK_STRUCTURE_REVIEW_MODEL=<vllm-model-id>
export SEMANTIK_SPECIALIST_CONCURRENCY=16        # leave headroom for the QC + extraction pools
export SEMANTIK_SPECIALIST_DISABLE_THINKING=1    # a reasoning model without this overruns
                                                 # max_tokens on <think> -> null content -> the
                                                 # whole review degrades unreviewed
export SEMANTIK_SPECIALIST_TIMEOUT_SECONDS=600   # default 120 is too low under a wide batch
                                                 # (Stage-6 ReadTimeout -> GGUF-fallback crash)

# --- VLM extraction seat (hybrid VLM + Tesseract fusion) ---
export SEMANTIK_VLM_EXTRACT=1
export SEMANTIK_VLM_FUSION=1
export SEMANTIK_VLM_STRUCT_HINTS=1
export SEMANTIK_VLM_PROVIDER=nvidia              # provider-agnostic registry key; base URL below
export SEMANTIK_VLM_BASE_URL=http://localhost:8000/v1
export SEMANTIK_VLM_API_KEY=local
export SEMANTIK_VLM_MODEL=<vllm-model-id>
export SEMANTIK_VLM_TIMEOUT_SECONDS=600
export SEMANTIK_VLM_DISABLE_THINKING=1           # extraction is a copy task -> thinking OFF
export SEMANTIK_VLM_CONCURRENCY=16               # per-page transcription-POST fan-out width

# --- Stage-9b reasoning-QC pass (reasoning model over the COMBINED HTML, thinking ON) ---
# A reasoning model QCs the assembled accessible-HTML block sequence (document-level
# TEXT windows + cross-page junction seams), NOT page images. Seat resolution: with
# SEMANTIK_REASONING_QC_BASE_URL unset the chain falls to the specialist text seat.
export SEMANTIK_REASONING_QC=on
export SEMANTIK_REASONING_QC_MODEL=<vllm-model-id>
# QC thinking stays ON: leave SEMANTIK_REASONING_QC_DISABLE_THINKING unset (that
# env is the ONLY path to a non-thinking QC request — the automatic thinking-off
# fallback was removed; a dense-window null/timeout rides the reasoning-preserving
# split ladder instead).
export SEMANTIK_REASONING_QC_CONCURRENCY=32      # the QC phase runs alone -> take all vLLM seqs
export SEMANTIK_REASONING_QC_WINDOW_BLOCKS=10    # small windows bound the deliberation (see trap)

# --- structure-fidelity fixes (default-off — enable for scans) ---
export ED4ALL_STRUCTURE_EXTRACT_GUARDS=1
export ED4ALL_STRUCTURE_OUTLINE_ANCHOR=1
export SEMANTIK_TITLE_SANITIZE=true

# --- cascade expansion stack (validated on a textbook scan) ---
export SEMANTIK_SEMANTIC_CLASS=1
export SEMANTIK_GOLD_SHELL=1
export SEMANTIK_UNIT_REGROUP=1
export SEMANTIK_UNIT_REGROUP_TABLE=1
export SEMANTIK_SECOND_PASS=1
export SEMANTIK_OCR_RENDER_SCALE=3.0
export SEMANTIK_PROMOTE_SECTION_HEADINGS=1
export SEMANTIK_SPLIT_FUSED_SECTION_TITLES=1
export SEMANTIK_OCR_CONFUSABLE_REPAIR=1
export SEMANTIK_STAGE6_PROSE_PASSTHROUGH=1
export SEMANTIK_ALLOW_THETA_STUB=1

# --- chunk-input fixes (apply at chunking; harmless for convert-only) ---
export TRAINFORGE_DROP_FRONTMATTER=true
export TRAINFORGE_DROP_APPARATUS_DUMPS=true
export TRAINFORGE_CHUNK_TYPE_CONTENT_AWARE=true
export ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR=20

# --- GPU-for-all-inference (every GPU-capable stage on the card) ---
export ED4ALL_NLI_DEVICE=cuda
export SEMANTIK_THETA_DEVICE=cuda
export ED4ALL_EMBEDDING_DEVICE=cuda   # now the CODE default too; kept explicit for provenance
# ED4ALL_GPU_LIFECYCLE default ON (load -> work -> unload at seams)

# --- observability + checkpoints ---
export ED4ALL_GENERATION_CHECKPOINT=on
export ED4ALL_VRAM_DOCTOR=1

# --- timeouts (multimodal reasoning-QC is slow; be generous) ---
export ED4ALL_TASK_TIMEOUT_MINUTES=600
export ED4ALL_BATCH_TIMEOUT_MINUTES=3000
export ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS=600
```

### 8.3 Traps this recipe documents

- **Stale `.venv` trap.** Invoke `ed4all` from the interpreter that carries the
  current pinned deps (on the reference box a `--user` system-python install), **not**
  the repo `.venv` — an old-box `.venv` can carry a `transformers` too old for the
  council backbone's `dtype=` kwarg, silently degrading structure detection
  (`[council] skipping semantic`). A silent capability drop, not a crash — verify
  the interpreter before a full-book run.
- **Specialist provider + displace double-gate.** `SEMANTIK_SPECIALIST_PROVIDER`
  unset routes the Stage-6 Phase-1 draft and pass-9b gap-fill at a local GGUF
  (llama-cpp `ImportError` on the first non-contiguous slice); and even with
  `provider=endpoint`, the authoring tier is NOT displaced off the local GGUF unless
  `SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE=1` is ALSO set. Pin the provider, the model
  id, AND the displace flag together (§8.2). Only safe to displace when the endpoint
  is a loopback, license-clean on-device seat.
- **Specialist timeout.** The default `SEMANTIK_SPECIALIST_TIMEOUT_SECONDS=120` is
  too low under a wide continuous batch — a Stage-6 ReadTimeout falls back to the
  local GGUF and crashes. Raise it (600) with the batch width.
- **QC sampling / budget (thinking-on over-deliberation).** A reasoning model in
  thinking mode at greedy `temperature 0` loops its `<think>` block on a content-
  dependent tail of windows, exhausting the completion window (`content=null` /
  `finish_reason=length`). Baked-in defaults handle this — thinking-on QC uses
  `SEMANTIK_REASONING_QC_TEMPERATURE=0.6` / `_TOP_P=0.95`, a hard
  `_MAX_TOKENS=16384` fail-fast backstop, and the native Nemotron
  `_REASONING_BUDGET=4096` trained thinking-budget — but the **window size is the
  bound that actually holds**: 30-block windows balloon past any soft budget, so
  drop `SEMANTIK_REASONING_QC_WINDOW_BLOCKS` to ~10 (a small window finishes in
  seconds with a clean stop). Shrink the problem, don't disable the intent — a
  null/timeout rides the reasoning-preserving split ladder, never a thinking-off
  retry.
- **Resume granularity.** `SEMANTIK_REASONING_QC_CHECKPOINT` (default ON) writes a
  per-UNIT content-addressed sidecar and the fan-out polls the stop sentinel before
  each submission, so `ed4all stop` mid-QC loses only the in-flight units (≤
  concurrency) and `--resume` serves every judged unit from cache. The Stage-6
  authoring twin is `SEMANTIK_STAGE6_CHECKPOINT`.

Full per-flag detail: `docs/operations/behavior-flags-semantik.md`. The container-
level vLLM seat lease (`ED4ALL_VLLM_CONTAINER_LIFECYCLE` / `ED4ALL_VLLM_CONTAINERS`)
and time-to-first-token metering (`ED4ALL_LLM_TTFT_METER`) are in the root
`CLAUDE.md` index + `docs/operations/behavior-flags.md`.

---

## 9. Declarative seat schedule (`ED4ALL_SEAT_SCHEDULE`)

On a single-GPU box the Super-120B seat cannot coexist with the conversion
(GLM-OCR / Qwen3-VL) seats or with the NLI/embedding validation chain — they
contend for the same VRAM. Historically the operator swapped the Super container
in and out by hand, and a **mis-scheduled swap starved validation 4-7x**
(observed live: a resident Super seat drops post-rewrite validation to ~3
blocks/min). `ED4ALL_SEAT_SCHEDULE` (default **OFF**) encodes the swap plan
declaratively so the orchestrator drives it.

**How it works.** Each phase in `config/workflows.yaml` carries an optional
`seats:` annotation (validated by `schemas/config/workflows_meta.schema.json`):

- `seats: [spark-super]` — the phase needs this logical seat served.
- `seats: []` — the phase explicitly needs **no** vLLM seat (the GPU belongs to
  NLI/embedding).
- *absent* — no opinion; the seat state carries over from the prior phase
  unchanged (used for the deterministic staging/chunking phases).

With the flag on, at each phase **start** boundary the runner reconciles the
resident seats to the phase's declared set — `docker stop` the no-longer-needed
seats, then `docker start` + **wait** + **coherence-probe** the newly-needed ones
(via `lib/vllm_container_lifecycle.py`). A newly-needed seat that fails to come up
in time, or that comes up but returns empty/degenerate content on the coherence
probe (**mode-collapse doctrine** — a restarted seat can pass `/v1/models` yet
emit soup), **fails the phase loudly** rather than running seat-starved. Every
other seat op is best-effort. At workflow start the full phase→seat plan is
logged once (a dry-run report of the "logical order").

**Seat ranges** for `textbook_to_course` (two-pass):

| Phase range | Seats | Why |
|---|---|---|
| `semantik_conversion` | `[spark-glm, spark-qwen]` | GLM-OCR extract + Qwe3-VL describe (the Super judge, ~5-10% of pages, is cascade-internal). |
| `staging` → `source_mapping` | *absent (no opinion)* | Deterministic phases; conversion seats stay resident until course_planning swaps them out in ONE transition. |
| `course_planning` → `assessment_synthesis` | `[spark-super]` | The whole synthesis range runs on ONE warm Super seat — planning, concept extraction, outline, inter-tier validation (kept warm), rewrite, and assessments back-to-back. |
| `post_rewrite_validation` → `trainforge_assessment` | `[]` | Super retires after assessment_synthesis; the NLI/embedding validation chain + packaging + chunking get the whole GPU seat-free. |
| `training_synthesis` | `[spark-super]` | In-build training-pair synthesis dispatches to the `local` registry row pointed at the Super seat (`LOCAL_SYNTHESIS_BASE_URL` → :8001) — the annotation re-seats Super for this phase so the reconcile doesn't evict the very seat the phase calls. |
| `libv2_archival` → `finalization` | `[]` | Archival + vector indexing + finalization run seat-free again. |

The `assessment_synthesis`-before-`post_rewrite_validation` ordering (assessment
consumes only objectives + chunks, never the rewritten blocks) is what lets Super
retire in a **single** transition instead of a stop-for-validation /
cold-start-for-assessment double swap — the two-pass `depends_on_when_env_value`
edges on both phases enforce it.

**Configuration.** Two registries bridge the logical names to docker:

```bash
export ED4ALL_SEAT_SCHEDULE=1
# logical seat name -> vLLM base URL
export ED4ALL_SEAT_BASE_URLS='spark-super=http://localhost:8001,spark-glm=http://localhost:8002,spark-qwen=http://localhost:8003'
# base URL -> docker container (the existing container registry, unchanged)
export ED4ALL_VLLM_CONTAINERS='http://localhost:8001=vllm-super,http://localhost:8002=vllm-glm,http://localhost:8003=vllm-qwen'
```

A seat name declared in YAML but missing from `ED4ALL_SEAT_BASE_URLS` is a config
gap → warn + skip (never a run-failing error), so a partial map soft-degrades to
no-op on the unmapped seats. Full flag detail: `docs/operations/behavior-flags.md`.

---

## See also

- `docs/operations/license-clean-run.md` — licensing / ToS-clean seat recipe.
- Root `CLAUDE.md` § Quick Start — canonical command surface; § Opt-In Behavior Flags — full env-var tables.
- `Courseforge/CLAUDE.md` § Operator stage subcommands — the two-pass stage subcommand internals.
- `config/workflows.yaml` — the authoritative phase list, per-phase `batch_timeout_minutes`, and validation gates.
