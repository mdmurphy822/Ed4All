# Worker coordination — A through G

This file is the operational manual for the multi-worker coordination phase running on branch family `worker-*`. It sits next to [`docs/architecture/ADR-001-pipeline-shape.md`](../architecture/ADR-001-pipeline-shape.md), which defines the contracts every worker below depends on.

If you are writing a new worker and your change touches any of ADR-001's Contracts 1–5, read ADR-001 first. If you are about to bump a shared constant, skip to the Coordination protocol section below.

## Workers A–G

| Letter | Branch | PR label | Status | Scope (one sentence) | Key files | Depends on |
|---|---|---|---|---|---|---|
| A | `claude/fix-package-quality-FyMue` (this worktree) | `worker-a` | in-flight | Ship ADR-001 plus this coordination doc; unblock B–G. | `docs/architecture/ADR-001-pipeline-shape.md`, `docs/contributing/workers.md`, `VERSIONING.md` | none |
| B | `worker-b/flow-metrics` | `worker-b` | blocked on A | Add five flow metrics to the base-pass quality report and bump `METRICS_SEMANTIC_VERSION` 3 → 4. | `Trainforge/process_course.py`, `Trainforge/tests/test_generator_defects.py` | A; `chunk-schema-v4` rebase point |
| C | `worker-c/training-pairs` | `worker-c` | blocked on A | Synthesize SFT/DPO instruction-pair training specs from aligned chunks. | `Trainforge/training_specs/*`, `lib/decision_capture.py`, `Trainforge/tests/fixtures/mini_course_training/` | A |
| D | `worker-d/chunk-summaries-and-recall` | `worker-d` | blocked on A | Add per-chunk summary and `retrieval_text` fields; extend recall metrics. | `Trainforge/process_course.py`, `Trainforge/tests/fixtures/mini_course_summaries/` | A; `chunk-schema-v4` rebase point |
| E | `worker-e/html-xpath-provenance` | `worker-e` | blocked on A | Carry HTML XPath provenance through the chunker. | `Trainforge/process_course.py` | A; `chunk-schema-v4` rebase point |
| F | `worker-f/typed-edge-graph` | `worker-f` | blocked on A | Typed-edge concept extractor producing `concept_graph_semantic.json`. | `Trainforge/graph/*`, `lib/decision_capture.py`, `Trainforge/tests/fixtures/mini_course_typed_graph/` | A |
| G | `worker-g/cross-package-index` | `worker-g` | done | Cross-package concept index + staleness check. | `LibV2/tools/*`, `LibV2/catalog/*` | A, F |
| H | `worker-h/courseforge-lo-specificity` | `worker-h` | done | Courseforge per-week learningObjectives specificity — fixes LO-fanout defect. | `Courseforge/scripts/generate_course.py`, `Courseforge/scripts/validate_page_objectives.py` | A |
| I | `worker-i/packager-validation-gate` | `worker-i` | in review (PR #5) | Wire `validate_page_objectives.py` into `package_multifile_imscc.py` as a pre-package gate. | `Courseforge/scripts/package_multifile_imscc.py` | H |
| J | `worker-j/libv2-reference-retrieval` | `worker-j` | in review | Reference retrieval: rationale payload, metadata-aware scoring, hand-curated gold queries, ADR-002 scope line. | `LibV2/tools/libv2/retriever.py`, `LibV2/tools/libv2/retrieval_scoring.py`, `LibV2/tools/libv2/cli.py`, `LibV2/tools/libv2/eval_harness.py`, `LibV2/courses/*/retrieval/`, `docs/architecture/ADR-002-retrieval-scope.md`, `docs/libv2/reference-retrieval.md` | A |

## Coordination protocol

### `chunk-schema-v4` rebase (applies to B, D, E)

The three workers that add chunk fields share one rebase point so the `CHUNK_SCHEMA_VERSION` bump is batched (ADR-001 Contract 1).

- Create branch `chunk-schema-v4` off `main` on the first of B/D/E to start. That worker declares the `CHUNK_SCHEMA_VERSION` constant in `Trainforge/process_course.py` and bumps it from `"v3"` to `"v4"`. The same PR threads the version string onto `manifest.json` (`chunk_schema_version`) and every chunk object (`schema_version`).
- The other two of B/D/E each branch from `chunk-schema-v4`, not from `main`.
- When a B/D/E PR is ready to merge, it merges into `chunk-schema-v4`.
- The **last** of the three to merge rebases `chunk-schema-v4` onto `main` and merges `chunk-schema-v4` into `main`.
- No worker bumps `CHUNK_SCHEMA_VERSION` outside this rebase point. One bump per release train.

### `lib/decision_capture.py` allowed-types protocol (applies to C and F)

ADR-001 Contract 3 spells this out. Operational summary:

- Today `log_decision` takes `decision_type` as a free string; there is no `ALLOWED_DECISION_TYPES` enum. C and F may add their decision types freely right now.
- When the enum lands (expected in Worker C's first PR that adds `instruction_pair_synthesis`), every subsequent new type goes into the enum in the same PR as the first production use site.
- Reviewers verify the new type is referenced from a production call site, not only a test.
- Type names: `snake_case`, tool-prefixed where ambiguous (e.g., `trainforge_typed_edge_inference`).
- If C and F open concurrently, they merge sequentially; the second merger appends its type alongside the first. No shared branch required.

### Fixture-subdir naming lock

`Trainforge/tests/fixtures/mini_course_*` follows `mini_course_<purpose-slug>` (all lowercase, underscore-separated). Current subdirs:

- `mini_course_clean/` — the synthetic-floor fixture used by the severity-flip trigger (`VERSIONING.md §3`).
- `mini_course_defective/` — exercises integrity gate failures.
- `mini_course_edge/` — edge-case chunks.

Planned additions (see ADR-001 Contract 4):

- Worker C: `mini_course_training/` (SFT/DPO pair generation).
- Worker D: `mini_course_summaries/` (per-chunk summary + `retrieval_text`).
- Worker F: `mini_course_typed_graph/` (typed-edge inference).

Every new fixture ships a `README.md` at its root that names what the fixture exercises and what CI assertions run against it. Use `mini_course_clean/README.md` as the template.

### Sequencing

```
A ─┬─► B ─┐
   ├─► C  │
   ├─► D ─┼─► chunk-schema-v4 rebase (B, D, E)
   ├─► E ─┘
   └─► F ─────► G
```

Critical path: A → F → G. Everything else parallels after A lands.

### Staging rule (all workers)

**Staging rule (all workers).** Every worker commit stages only the paths the worker plan lists. Use explicit `git add <path>` per file. Never `git add -A`, never `git add .`, never `git commit -a`. If the worktree has unexpected dirty state, note it in the PR description; do not include the state in your commit.

## Known follow-ups (not blocking any worker)

These are tracked ADR-001 follow-ups. Surface them in the relevant worker's PR description if the worker happens to touch the affected file, but none of them blocks any worker from starting.

- `FOLLOWUP-ADR001-1` — LibV2 importer OSCQR-flavored `quality_report.json` filename collision at `LibV2/tools/libv2/importer.py:323`. Proposed fix: rename to `quality/oscqr.json`.
- `FOLLOWUP-ADR001-2` — `cli/reporters/run_summarizer.py:238` reads dead key `quality_score`. Either fix the reader to consume `overall_quality_score`, or delete the reader.
- `FOLLOWUP-ADR001-3` — `Trainforge/align_chunks.py:687` docstring claims `METRICS_SEMANTIC_VERSION=2` semantics; the actual constant is v3. Docstring-only update.
- `FOLLOWUP-ADR001-4` — Enforce the additive-only contract in code: the unit test plus the `align_chunks.update_quality_report` refactor described in ADR-001's Migration sketch item 2.

## Spawn-prompt template for future workers

Copy-paste the block below when spinning up a new worker. Fill the bracketed slots. This is a template, not a command; the orchestrator adapts it to its own harness.

```
You are Worker <letter>, part of the Ed4All multi-worker coordination phase.

Your scope:
<one to three sentences describing the capability this worker ships>

Contracts you depend on:
- ADR-001 (docs/architecture/ADR-001-pipeline-shape.md): read before touching
  quality_report.json, METRICS_SEMANTIC_VERSION, CHUNK_SCHEMA_VERSION,
  lib/decision_capture.py, or Trainforge/tests/fixtures/mini_course_*.
- docs/contributing/workers.md: your row in the A–G table, plus the coordination
  protocol sections that apply to your letter.

Branch: worker-<letter>/<slug>
PR label: worker-<letter>

Coordination rules in force:
- Stage only the paths your plan lists. Use explicit `git add <path>` per file.
  Never `git add -A`, never `git add .`, never `git commit -a`. If the worktree
  has unexpected dirty state, note it in your PR description; do not include it
  in your commit.
- If your work touches the chunk schema (B, D, E-type changes), branch from
  `chunk-schema-v4`, not from main. See the rebase protocol in workers.md.
- If your work adds a decision-capture event type, follow Contract 3 in
  ADR-001 (type in the same PR as first production use site).
- If your work adds a fixture subdir under Trainforge/tests/fixtures/,
  follow the `mini_course_<purpose-slug>` naming lock and ship a README.md.

Done criteria:
<worker-specific checklist>

Report back with:
1. Commit SHA(s)
2. Files written (absolute paths)
3. PR URL once opened
4. Any contract changes you had to propose (these are ADR-worthy)
```
