# ADR-001: Pipeline shape — base pass and alignment pass

## Status

Proposed.

## Context

Trainforge's course-processing pipeline today has two passes that both write to `state/<run>/quality/quality_report.json`:

1. **Base pass** — `Trainforge/process_course.py::CourseProcessor._generate_quality_report`, called from `_write_metadata` at `process_course.py:2025`. Full-replacement write of the complete report.
2. **Alignment pass** — `Trainforge/align_chunks.py::update_quality_report` at `align_chunks.py:678`. Load-then-mutate write against the file the base pass produced.

These two writers overlap on the same JSON document without a declared contract. The overlap surfaces concretely when a downstream worker (Worker B in the active coordination phase) tries to add new metrics and bump `METRICS_SEMANTIC_VERSION`: the alignment pass's in-place mutation of base-pass-owned fields races the version bump in ways that are not detectable at read time.

### Field-level picture

| Key in `quality_report.json` | Base pass writes | Alignment pass writes | Hazard |
|---|---|---|---|
| `metrics_semantic_version` (int) | yes (currently `3`) | no | none |
| `overall_quality_score` (float) | yes | **overwrites** with a 0.6·base + 0.4·alignment blend (`align_chunks.py:751`) | Silent: consumer reading `v3` sees a v3 score that was blended by an alignment pass computed against v3 metrics, but the blend factor is invisible in the artifact. |
| `metrics.*` (base compliance metrics) | yes | no | none |
| `methodology.*` | yes | no | none |
| `integrity.broken_refs` | yes | **appends** its own broken-ref findings (`align_chunks.py:744`) | Silent: a second alignment run duplicates entries. Base pass and alignment pass compute "broken ref" against different valid-ID sets (flat vs week-scoped). |
| `integrity.orphan_week_scoped_refs` | no | yes (new key) | none (alignment owns it, alignment is the only writer) |
| `integrity.html_balance_violations` | yes | no | none |
| `integrity.follows_chunk_boundary_violations` | yes | no | none |
| `integrity.factual_inconsistency_flags` | yes | no | none |
| `integrity.uncovered_outcomes` | yes | no | none |
| `validation.*` | yes | no | none |
| `recommendations` | yes | no | none |
| `alignment.*` (prereq_concepts_coverage, teaching_role_coverage, learning_outcome_refs_coverage, teaching_role_consistency, teaching_role_distribution) | no | yes | none |

### Readers today

- `cli/reporters/run_summarizer.py:238` — reads for `quality_score` or `score` keys. **Neither is emitted by either pass.** Effectively dormant. Flagged as follow-up `FOLLOWUP-ADR001-2`.
- `tests/test_pipeline_integration.py:181-184` — asserts file exists and `overall_quality_score > 0.0`.
- `Trainforge/tests/test_generator_defects.py:272` — asserts `metrics_semantic_version == METRICS_SEMANTIC_VERSION`.
- LibV2 importer (`LibV2/tools/libv2/importer.py:323`) copies the file into `LibV2/courses/*/quality/quality_report.json`. No downstream LibV2 tool reads the copied file's metrics.
- `LibV2/tools/libv2/importer.py:323` *also* writes a **different, non-overlapping** `quality_report.json` (OSCQR-flavored stub: `oscqr_score`, `pattern_violations`, `corrections`, `last_evaluated`) when importing a source that has none. This is a separate schema sharing a filename. Flagged as follow-up `FOLLOWUP-ADR001-1`.

### `METRICS_SEMANTIC_VERSION` surface

- **Definition:** `Trainforge/process_course.py:58` → `METRICS_SEMANTIC_VERSION = 3`.
- **Writer:** `Trainforge/process_course.py:1649`.
- **Reader:** `Trainforge/tests/test_generator_defects.py:259,272`.
- **Docstring reference (stale):** `Trainforge/align_chunks.py:687` mentions v2 semantics; base pass is at v3. Flagged as follow-up `FOLLOWUP-ADR001-3`.
- **Narrative reference:** `VERSIONING.md §2.9`.

No LibV2 code and no CI gate read it. The bump radius for Worker B is strictly Trainforge-internal.

### `--align` CLI surface

- `Trainforge/process_course.py:2124` defines `--align`. When set, it invokes `Trainforge.align_chunks.main` inline with a hard-coded argparse namespace after base processing.
- `Trainforge/align_chunks.py:818-911` exposes a standalone CLI (`python -m Trainforge.align_chunks --corpus <dir> --objectives <file>`).
- The module docstring (`align_chunks.py:9-13`) cites a **load-bearing standalone use case**: re-run alignment against an already-processed corpus to iterate on alignment logic without paying the cost of HTML parse, boilerplate detection, and chunking again.
- No integration test exercises the `--align` flag; `tests/test_pipeline_integration.py` runs base only. Unit imports from `align_chunks` exist (`test_generator_defects.py:140,160`) but do not invoke the CLI.

## Decision

Keep base pass and alignment pass as separate stages. **Make alignment's write to `quality_report.json` additive-only** under a top-level `alignment` key, forbid alignment mutation of base-pass-owned keys, and require both passes to declare the `metrics_semantic_version` they target.

## Rationale

1. **The standalone alignment workflow is load-bearing.** The module docstring, the README combined example (`README.md:143`: `--align --import-to-libv2`), and the architectural intent all depend on re-running alignment without re-chunking. Merging destroys the cheap-iteration loop, and the replacement (a resume-from-chunks flag on the base pass) is real engineering that cannot ship as a doc-only PR.
2. **The current coupling is fixable without a merge.** The pain points (silent overall-score overwrite, silent `integrity.broken_refs` append) are bugs-in-contract, not bugs-in-architecture. Formalizing the contract — alignment writes only to the `alignment` top-level block; base owns `integrity` and `overall_quality_score` — removes the hazard Worker B is blocked on.
3. **The version-ownership story becomes clean.** Base pass owns `metrics_semantic_version` because base pass owns the base metrics block. Alignment declares which base version it was computed against (see Contract 2 below), so a stale re-run is detectable at read time. No shared ownership, no coordination round trip.
4. **Worker B can proceed immediately.** Adding five flow metrics under `metrics` and bumping to `v4` is a base-pass-only change; alignment is untouched.
5. **The merge option was tempting but solves the wrong problem.** Merging would fix "alignment silently mutates base fields" — fixable by convention. Merging would break "cheap re-run alignment without re-chunking" — not fixable by convention; requires a resume flag. Keep the split; formalize the convention.

## Rejected alternative: merge alignment into `CourseProcessor.process()`

| Axis | Keep split (chosen) | Merge |
|---|---|---|
| Cheap alignment re-run | Free (existing CLI) | Needs new `--resume-from-chunks` flag (real work, out of scope) |
| `quality_report.json` ownership clarity | Fixed by additive-only contract (this ADR) | Automatic (single writer) |
| Blast radius of this ADR | Zero code change today; contract is prose + test | Refactor `process_course.py::main`, delete `--align`, repoint README, add resume flag |
| Worker B unblock path | Immediate | Blocked until merge lands |
| Risk of regressing an existing test | Zero (no code change) | Non-zero (refactor) |
| Impact on Workers C/D/E/F | Neutral | Neutral |

The merge's only real win is "one writer." The additive-only contract gives equivalent safety with "two writers, disjoint keys" at far lower risk.

## Migration sketch

No code change is required to adopt this ADR's architecture. Contract enforcement lands in follow-up work:

1. **Unit test in the next Worker B PR** that loads a synthesized `quality_report.json`, runs `align_chunks.update_quality_report`, and asserts `overall_quality_score` is unchanged — or asserts alignment's blended score is written under `alignment.alignment_quality_score` once the contract is enforced in code.
2. **Code follow-up ticket `FOLLOWUP-ADR001-4`:** enforce the additive-only contract in `align_chunks.update_quality_report`. Do not overwrite `overall_quality_score`; write alignment's score to `alignment.alignment_quality_score`. Do not append to `integrity.broken_refs`; write alignment's broken-ref findings to `alignment.outcome_ref_broken_refs`.
3. **Open question for follow-up:** `integrity.broken_refs` dedup semantics. Today both passes write there; tomorrow, per this ADR, only the base pass writes there. The alignment pass computes its own broken-refs against a potentially different valid-ID set (hierarchical vs flat). Tracked within `FOLLOWUP-ADR001-4`.

## Contracts

These are the five contracts Workers B–G depend on. Changes to any of them require a new ADR.

### Contract 1 — Chunk-schema versioning

Workers B, D, and E each add fields to the chunk object. Policy:

- Single string constant `CHUNK_SCHEMA_VERSION` in `Trainforge/process_course.py`. Starts at `"v3"` (implied by the current chunk shape, not yet declared). The first worker to touch chunk schema declares the constant and bumps to `"v4"`.
- The version string lands on `manifest.json` as `chunk_schema_version` and on every chunk object as `schema_version`.
- Workers B, D, and E share a single rebase point: branch `chunk-schema-v4` off `main`. B, D, and E each branch from `chunk-schema-v4`, not from `main`. The last of the three to merge rebases `chunk-schema-v4` onto `main` and merges. This is documented in `docs/contributing/workers.md`.
- One bump per release train, batched. No worker bumps independently.

### Contract 2 — `METRICS_SEMANTIC_VERSION` ownership

- Lives at `Trainforge/process_course.py:58`. Owned by the **base pass**.
- Worker B owns the v3 → v4 bump (for five flow metrics).
- Any other worker later adding a base-pass metric coordinates the bump via the decision log at the bottom of this ADR (append-only; one line per bump; PR number + summary).
- Alignment pass does NOT bump this constant and does NOT write under `metrics`. Alignment writes under `alignment`.
- `align_chunks.update_quality_report` writes `alignment.base_metrics_semantic_version: <int>` alongside the alignment metrics. Downstream readers comparing `metrics_semantic_version` against `alignment.base_metrics_semantic_version` detect a stale-re-run skew without reading the alignment prose.

### Contract 3 — Decision-capture event-type ownership

`lib/decision_capture.py::log_decision` currently accepts `decision_type` as a free string. There is **no** `ALLOWED_DECISION_TYPES` constant anywhere in the tree today (`constants.py` has `OPERATION_MAP` but no type enum). Verified by grep.

This is an asset for Workers C and F: they can add their types (`instruction_pair_synthesis`, `preference_pair_generation`, `typed_edge_inference`) without touching a central enum, because there is no central enum to touch. Convention:

- Establish `lib/decision_capture.py::ALLOWED_DECISION_TYPES` as a new tuple constant. Creation is NOT Worker A's responsibility — Worker C creates it when it first adds its type.
- Until the enum exists, C and F can add types freely.
- Once the enum exists, PR review protocol: new types land in the same PR as the first use site; reviewer checks the `decision_type` string is referenced from at least one production call site, not only a test. Type names are `snake_case` and are scoped with a tool prefix when ambiguous (e.g., `trainforge_typed_edge_inference`).
- **Coordination for C and F:** if the two PRs open concurrently, they rebase through each other (sequential merge; second merger adds its type to the enum alongside C's). No shared branch required because both PRs touch a single tuple constant — merge conflicts are trivial.

### Contract 4 — Shared test fixtures

`Trainforge/tests/fixtures/mini_course_*` is canonical. Audit confirms three subdirs exist today: `mini_course_clean/`, `mini_course_defective/`, `mini_course_edge/`.

- **Naming pattern (locked):** `mini_course_<purpose-slug>`. All lowercase, underscore-separated. `<purpose-slug>` is a single noun or noun-phrase describing the capability the fixture exercises.
- Worker C adds: `mini_course_training/` (SFT/DPO pair generation fixture).
- Worker D adds: `mini_course_summaries/` (per-chunk summary + retrieval_text fixture).
- Worker F adds: `mini_course_typed_graph/` (typed-edge inference fixture).
- Every fixture MUST ship a `README.md` at its root documenting what the fixture exercises and what the CI assertions on it are. See the existing `mini_course_clean/README.md` for the template.

### Contract 5 — Worker branching

- Branch names: `worker-<letter>/<slug>`. Examples: `worker-b/flow-metrics`, `worker-c/training-pairs`, `worker-d/chunk-summaries-and-recall`, `worker-e/html-xpath-provenance`, `worker-f/typed-edge-graph`, `worker-g/cross-package-index`.
- PR label: `worker-<letter>` (lowercase letter).
- Workers never share branches except the `chunk-schema-v4` rebase point for B/D/E (Contract 1).

## Open questions / known issues deliberately not addressed by this ADR

- **`FOLLOWUP-ADR001-1`** — LibV2 importer OSCQR-flavored `quality_report.json` filename collision at `LibV2/tools/libv2/importer.py:323`. Different keys, different purpose, same filename as the Trainforge schema. Proposed fix: rename the importer's output to `quality/oscqr.json` (or an equivalent non-colliding path).
- **`FOLLOWUP-ADR001-2`** — `cli/reporters/run_summarizer.py:238` reads a dead key (`quality_score`) that neither writer emits. Either fix the reader to consume `overall_quality_score` or delete it.
- **`FOLLOWUP-ADR001-3`** — `Trainforge/align_chunks.py:687` docstring claims `METRICS_SEMANTIC_VERSION=2` semantics; the base pass is at v3. Docstring-only update.
- **`FOLLOWUP-ADR001-4`** — Enforce the additive-only contract in code. Covers the unit test and the `align_chunks.update_quality_report` refactor described in Migration sketch item 2.

The Courseforge-side template-chrome work tracked in `VERSIONING.md §4b` is not an ADR-001 follow-up — it is a separate v1.0 roadmap item. It is named here only because a reviewer tracing quality-report semantics may trip on it.

## Decision log (append-only)

| Date | PR | What | Owner |
|---|---|---|---|
| (pending) | (Worker B PR) | `METRICS_SEMANTIC_VERSION` 3 → 4 (adds five flow metrics) | Worker B |
