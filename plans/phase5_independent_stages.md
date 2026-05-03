# Phase 5 — Independent Stages CLI (Plan)

**Status:** plan only. **Depends on:** Phase 2 (Block dataclass + `block_id`), Phase 3 (router + outline/rewrite tiers), Phase 4 (statistical validators / classifiers). Does **not** ship without those.

## 1. Goal recap

Today, `cli/commands/run.py:43` accepts a fixed set of workflow names (`SUPPORTED_WORKFLOWS`) and runs them end-to-end via `PipelineOrchestrator.run(workflow_id)` (`cli/commands/run.py:826`). Phase 5 adds **stage-level subcommands** for the Courseforge two-pass pipeline so operators can run outline → validate → rewrite independently, and re-execute a single Block type after a failed pass.

> **Phase 5-prep refresh (against HEAD `e5603ac`):** §3 originally listed `courseforge-classify` as a separate subcommand. Investigation refresh found Phase 4 wired Bloom classification *inline* as a validator gate (`bloom_classifier_disagreement` at both `inter_tier_validation` and `post_rewrite_validation`, `config/workflows.yaml:1129, 1334`), **not** as a standalone classifier tier. There is no `_run_classification` handler in `MCP/tools/pipeline_tools.py` and no `classification` phase in `config/workflows.yaml`. Operators wanting to re-check Bloom-level alignment should use `courseforge-validate`, which already includes that gate. **Subcommand dropped from this plan.** §12 sequencing list updated accordingly.

## 2. CLI surface

Four new subcommands (was five; `courseforge-classify` dropped — see Phase 5-prep refresh above). The smallest add is per-stage entries, not a sub-group, so they slot into the existing `ed4all run <name>` dispatch in `cli/commands/run.py:43-55`. Add to `SUPPORTED_WORKFLOWS`:

```
courseforge-outline
courseforge-validate
courseforge-rewrite
courseforge          # full Courseforge slice (DART → IMSCC); thin alias over textbook_to_course
```

### Concrete invocations

```bash
# Outline-only (Phase 3 outline tier)
ed4all run courseforge-outline \
    --course-code PHYS_101 \
    --objectives Courseforge/exports/PROJ-PHYS_101-.../01_learning_objectives/synthesized_objectives.json \
    --staging Courseforge/inputs/PHYS_101 \
    --output Courseforge/exports/PROJ-PHYS_101-20260502/01_outline

# Deterministic + statistical gates (Phase 4)
# Includes Bloom's classifier disagreement (inline gate, not a separate tier).
ed4all run courseforge-validate \
    --outline Courseforge/exports/PROJ-PHYS_101-20260502/01_outline \
    --gates content_grounding,source_refs,bloom_classifier_disagreement

# Rewrite tier with per-block scope + per-stage model override
# (block-type vocabulary: 16 singular types per Courseforge/scripts/blocks.py:77)
ed4all run courseforge-rewrite \
    --outline Courseforge/exports/PROJ-PHYS_101-20260502/01_outline \
    --output  Courseforge/exports/PROJ-PHYS_101-20260502/04_rewrite \
    --blocks assessment_item,example \
    --model deepseek-v3 --api-provider deepseek

# Rewrite-tier re-run scoped to escalated blocks only
# (resume after a partial failure or A/B-test rewrite-tier model swaps)
ed4all run courseforge-rewrite \
    --outline Courseforge/exports/PROJ-PHYS_101-20260502/01_outline \
    --output  Courseforge/exports/PROJ-PHYS_101-20260502/04_rewrite \
    --escalated-only \
    --model claude-sonnet-4-6 --api-provider anthropic

# Validate stage that actively re-rolls failed blocks via the Phase 3 router
# (default: validate is read-only and only reports failures)
ed4all run courseforge-validate \
    --outline Courseforge/exports/PROJ-PHYS_101-20260502/01_outline \
    --regenerate-on-fail

# Full slice (equivalent to running all four in sequence)
ed4all run courseforge \
    --course-code PHYS_101 --corpus pdfs/ \
    --output Courseforge/exports/PROJ-PHYS_101-20260502
```

### Per-tier mapping

| CLI subcommand | Phase 3 tier / phase | Underlying handler |
|---|---|---|
| `courseforge-outline` | outline tier | `content_generation_outline` phase from `config/workflows.yaml` (Phase 3 deliverable) |
| `courseforge-validate` | gates only | runs Phase 4 validators against an OUTLINE_DIR (including the inline `bloom_classifier_disagreement` gate); no LLM call by default; with `--regenerate-on-fail` actively re-rolls failed blocks via the Phase 3 router |
| `courseforge-rewrite` | rewrite tier | `content_generation_rewrite` phase; `--escalated-only` filters to blocks with non-null `escalation_marker` |
| `courseforge` | all three + packaging | thin wrapper that fans out via `WorkflowRunner` |

### `courseforge-validate --regenerate-on-fail`

Default behaviour of `courseforge-validate` is read-only: it loads the OUTLINE_DIR, runs the Phase 4 validators, and writes `02_validation_report/report.json`. The new flag promotes it to active re-execution: any block whose validator returns `GateResult.action="regenerate"` (Phase 3 §6.5) is re-rolled in place via the Phase 3 router's self-consistency loop (Phase 3 §3.6), respecting the same `COURSEFORGE_OUTLINE_REGEN_BUDGET` ceiling. Blocks whose validator returns `action="escalate"` are flagged for the rewrite tier in the report but not re-routed (validate is a same-tier operation; tier-promotion belongs to `courseforge-rewrite`). Blocks whose validator returns `action="block"` are recorded as hard failures.

Default off because validate-as-side-effect changes the tool's contract; operators opt in deliberately. The flag is mutually exclusive with `--read-only` (which is the implicit default and need not be passed).

## 3. Per-block re-execution (`--blocks`)

`--blocks` is the headline Phase 5 feature. Accepted values are the 16 singular `Block.block_type` enum members from `Courseforge/scripts/blocks.py:77` plus the wildcard `all`:

```
objective, concept, example, assessment_item, explanation, prereq_set,
activity, misconception, callout, flip_card_grid, self_check_question,
summary_takeaway, reflection_prompt, discussion_prompt, chrome, recap,
all (default)
```

*(Block-type vocabulary: 16 singular types per `Courseforge/scripts/blocks.py:77`; plan §3 originally listed plural names that don't match the enum.)*

Multi-value: comma-separated, e.g. `--blocks assessment_item,example`.

### Two-pass env-var requirement

All four new subcommands MUST set `COURSEFORGE_TWO_PASS=true` in the workflow environment. The four target workflow phases — `content_generation_outline`, `inter_tier_validation`, `content_generation_rewrite`, and `post_rewrite_validation` — carry `enabled_when_env: "COURSEFORGE_TWO_PASS=true"` (`config/workflows.yaml:946, 984, 1148, 1206`); they SKIP when the flag is unset, and the legacy single-pass `content_generation` phase (with `enabled_when_env: "COURSEFORGE_TWO_PASS!=true"`) runs instead.

Choices:

- **(a) Auto-set inside each subcommand handler** so operators don't have to remember. Pair with a CLI-level `--no-two-pass` opt-out for operators who explicitly want the legacy single-pass code path.
- **(b) Fail loudly when unset** with a remediation hint (e.g., "courseforge-outline requires COURSEFORGE_TWO_PASS=true; export it or pass --two-pass").

**Recommendation: (a)** auto-set with a `--no-two-pass` opt-out. Phase 5's value proposition (per-block re-execution, escalated-only re-runs) only makes sense in the two-pass world; demanding an env-var dance from operators every time they invoke a Phase 5 subcommand is friction without benefit.

### Selection algorithm (rewrite tier only; validate ignores `--blocks`)

1. Load every `*.json` JSON-LD page under OUTLINE_DIR.
2. Filter to blocks whose `block_type` is in the requested set.
3. Group by `(week, page)` and dispatch to the rewrite worker with a `target_block_ids: [...]` param.
4. Worker emits the same JSON-LD page shape, replacing only the targeted blocks.
5. **Join key:** `Block.block_id` from Phase 2 — every existing block keeps its ID; the rewriter looks blocks up by ID and substitutes content. Untouched blocks are byte-identical to the input.

### Provenance

Each rewritten block appends one entry to `touched_by[]`:

```json
{
  "stage": "rewrite",
  "model": "deepseek-v3",
  "provider": "deepseek",
  "tier": "rewrite",
  "ts": "2026-05-02T13:54:00Z",
  "blocks_filter": ["assessment_item", "example"]
}
```

Old `touched_by[]` entries are preserved (append-only). Unchanged blocks get **no** new entry — that is the audit signal "this block was not re-run."

### Idempotency

Re-running with identical inputs and `--blocks` should yield byte-identical output for blocks **outside** the filter, and an additional `touched_by[]` entry plus possibly different content for blocks **inside** the filter (LLM nondeterminism). The system does **not** seed LLM calls today (no `--seed` flag in `run.py` / `BackendSpec` per `cli/commands/run.py:350`). Phase 5 inherits that nondeterminism; record an open question to seed later.

### `courseforge-rewrite --escalated-only`

A second selection mode (orthogonal to `--blocks`): re-run the rewrite tier on only blocks whose Phase 3 outline tier set a non-null `Block.escalation_marker` (Phase 2 deliverable; see `plans/phase2_intermediate_format_detailed.md` Subtasks 3, 10, 13 for the dataclass field, JSON Schema field, and SHACL property respectively). Selection algorithm:

1. Load every `*.json` JSON-LD page under OUTLINE_DIR.
2. Filter to blocks where `escalation_marker is not None`.
3. Group by `(week, page)` and dispatch to the rewrite worker.

Two operator workflows this enables:

- **Resume after a partial outline-tier failure.** The full outline run completes; some blocks exhausted the regen budget and carry `escalation_marker="outline_budget_exhausted"`. Operator runs `courseforge-rewrite --escalated-only` to push only those blocks through the rewrite tier without re-running the full `content_generation_rewrite` phase.
- **A/B-testing rewrite-tier model swaps.** Generate an OUTLINE_DIR once. Run `courseforge-rewrite --escalated-only --model claude-sonnet-4-6 --output run_A` and `courseforge-rewrite --escalated-only --model deepseek-v3 --output run_B`; compare resulting `04_rewrite/manifest.json.content_hash` distributions and human-review the divergent blocks.

`--escalated-only` is mutually exclusive with `--blocks` (you can't filter by both type and escalation status in the same run; nest the operator's intent into two sequential calls). When neither is set, all blocks pass through.

## 4. Stage I/O contracts

| Subcommand | Inputs (read-only) | Outputs (created) | Mutates |
|---|---|---|---|
| `courseforge-outline` | `--objectives PATH` (Courseforge form, validated by `_validate_reuse_objectives_file` at `cli/commands/run.py:209`); `--staging` DART HTML dir | OUTLINE_DIR with one JSON-LD page per `(week, page)`, conforming to `schemas/knowledge/courseforge_jsonld_v1.schema.json` | none |
| `courseforge-validate` | OUTLINE_DIR | `02_validation_report/report.json` | none |
| `courseforge-rewrite` | OUTLINE_DIR | FINAL_DIR (full copy with substitutions) | none — outline is read-only; final is new |
| `courseforge` | corpus + objectives | full pipeline output dirs | none |

(Phase 5-prep refresh: the previous `courseforge-classify` row was dropped — Phase 4 wired Bloom classification inline as a validator gate (`bloom_classifier_disagreement`), not a standalone tier. Operators re-check via `courseforge-validate`, which routes that gate through the same chain. The `03_classification/` output directory is also dropped from §6.)

## 5. Workflow-runner factoring

*(Citations refreshed Phase 5-prep against HEAD `e5603ac`; lines drifted since plan authoring.)*

The runner already supports the patterns Phase 5 needs:

- **Single-phase reuse via pre-population:** `WorkflowRunner.run_workflow` (`MCP/core/workflow_runner.py:802`) checks `phase_outputs[phase_name].get("_completed")` at line 860 and skips. Phase 5 leverages the same trick: when `ed4all run courseforge-rewrite` starts, it pre-populates `phase_outputs` for every phase **before** `content_generation_rewrite`, with absolute paths read from `--outline`.
- **Synthesised phase outputs:** `_synthesize_dart_skip_output` (`MCP/core/workflow_runner.py:1324`) and `_synthesize_course_planning_reuse_output` (line 1400) are the prototypes. Phase 5 adds one more: `_synthesize_outline_output(outline_dir)` returning the canonical `_completed: True` dicts. (Phase 5-prep refresh: `_synthesize_classification_output` dropped — Phase 4 wired classification inline as a validator gate, not a separate phase.)
- **Refusal on missing upstream:** existing `_dependencies_met` (`MCP/core/workflow_runner.py:1643`) already covers this; Phase 5's contribution is a clearer error message ("`--outline` directory does not contain `01_outline/manifest.json`; cannot run rewrite without prior outline").
- **`--force` re-run:** new flag. When set, the synthesised output sets `_completed: False` so the phase loop re-executes despite checkpoint presence.

### Phases that `_synthesize_outline_output` must pre-populate

*(Phase 5-prep refresh: list expanded for `chunking` + `concept_extraction` + `imscc_chunking` phases that landed post original plan authoring — Phase 6 added `concept_extraction`, Phase 7b added `chunking`, Phase 7c added `imscc_chunking`.)*

The full dependency chain for `textbook_to_course` post-Phase-7c is:

```
dart_conversion → staging → chunking → objective_extraction → source_mapping
  → concept_extraction → course_planning → content_generation_outline
  → inter_tier_validation → content_generation_rewrite → post_rewrite_validation
  → packaging → imscc_chunking → trainforge_assessment → training_synthesis
  → libv2_archival
```

Phases that `_synthesize_outline_output` must pre-populate before the rewrite tier runs (because the rewrite tier consumes their outputs via `inputs_from`):

- `staging` — `staging_dir` output
- `chunking` — `dart_chunks_path`, `dart_chunks_sha256` (Phase 7b)
- `objective_extraction` — `textbook_structure`
- `source_mapping` — `source_module_map`
- `concept_extraction` — `concept_graph_path`, `concept_graph_sha256` (Phase 6)
- `course_planning` — `synthesized_objectives`
- `content_generation_outline` — outline-tier blocks (the OUTLINE_DIR provided via `--outline`)
- `inter_tier_validation` — validation report (the JSONL emitted by `_run_inter_tier_validation`)

`imscc_chunking` runs **after** the rewrite tier, so it does NOT need pre-population for `courseforge-rewrite`; it would, however, need pre-population for any future post-rewrite-only re-execution subcommand.

What's reusable: `_topological_sort`, `_route_params`, `_create_phase_tasks`, `_extract_phase_outputs`, gate routing.
What's new: `_synthesize_outline_output`, `--blocks`/`--force` plumbing through `_build_workflow_params` (`cli/commands/run.py:65`), and the `target_block_ids` param routing entry in `config/workflows.yaml` for `content_generation_rewrite`.

## 6. Output directory layout

**Recommendation: keep the existing `Courseforge/exports/PROJ-{CODE}-{TIMESTAMP}/` envelope** (per `MCP/tools/courseforge_tools.py:32`, `EXPORTS_PATH = COURSEFORGE_PATH / "exports"`), and **add stage subdirectories inside it**:

```
Courseforge/exports/PROJ-PHYS_101-20260502/
├── 00_template_analysis/
├── 01_learning_objectives/      # existing (Wave 24)
├── 01_outline/                  # NEW Phase 5 — outline tier output
├── 02_validation_report/        # NEW — see "report.json is a NEW writer" below
├── 04_rewrite/                  # NEW — final JSON-LD pages
├── 05_imscc/                    # NEW alias for existing 06_packaging
├── content/                     # existing (post-rewrite materialised HTML)
└── project_config.json
```

*(Phase 5-prep refresh: `03_classification/` was dropped per §4 — classification runs inline as a validator gate, not as a standalone tier with its own output dir. The numbering gap (`02_` → `04_`) is intentional and preserves the originally proposed `04_rewrite/` slot.)*

Rationale: existing tooling (`pipeline_tools.py:2263`) already creates `00_..04_` subdirs. Phase 5 reuses the convention and **only** adds 3 new top-level dirs (was 4 before `03_classification/` was dropped). Avoids breaking `--reuse-objectives` (`cli/commands/run.py:209`) and IMSCC packaging (`MCP/tools/courseforge_tools.py:140`) which key off `01_learning_objectives/`.

Each stage subdir contains:
- `manifest.json` — list of input artifacts + content hash + provenance + LLM model/provider used
- `pages/` — one JSON-LD file per `(week, page)`
- `_logs/` — per-task stdout/stderr

`manifest.json.content_hash` is a SHA-256 over canonicalised JSON-LD pages, used by tests to assert "blocks outside `--blocks` are unchanged."

### `02_validation_report/report.json` is a NEW writer (Phase 5 deliverable)

`02_validation_report/report.json` is a NEW writer that Phase 5 must add. The shipped `_run_inter_tier_validation` (Phase 4 N0 + 3.5 in `MCP/tools/pipeline_tools.py`) emits **JSONL only** — `blocks_validated_path` and `blocks_failed_path` — NOT a per-stage `report.json`. The operator-facing `report.json` is a Phase 5 deliverable that aggregates the JSONL into a structured per-block summary.

Schema (Phase 5 deliverable; lands as part of subcommand handlers in §12 ST 4):

```json
{
  "run_id": "WF-20260502-...",
  "phase": "inter_tier_validation",      // or "post_rewrite_validation"
  "schema_version": "v1",
  "total_blocks": 247,
  "passed": 210,
  "failed": 30,
  "escalated": 7,
  "per_block": [
    {
      "block_id": "...",
      "block_type": "assessment_item",
      "page": "...",
      "week": 4,
      "status": "passed|failed|escalated",
      "gate_results": [
        {
          "gate_id": "bloom_classifier_disagreement",
          "action": "regenerate|escalate|block|null",
          "passed": false,
          "issues": [...]
        }
      ],
      "escalation_marker": "outline_budget_exhausted | null"
    }
  ]
}
```

Both `courseforge-validate` (default read-only mode) and `courseforge-validate --regenerate-on-fail` write this artifact; `courseforge-rewrite` writes its own equivalent under `04_rewrite/02_validation_report/report.json` covering the post-rewrite gate run.

## 7. Dry-run / plan-only per stage

Existing `--dry-run` (`cli/commands/run.py:586` → `_dry_run_plan`) prints the planned phase sequence. Phase 5 inherits it with two changes:

1. When the subcommand is `courseforge-rewrite`, the planner prunes phases before `content_generation_rewrite` and marks them `<REUSED>` (mirroring the existing `<REUSED>` annotation for `--reuse-objectives` at `cli/commands/run.py:704`).
2. Adds `--blocks` info to the plan dict when present:
   ```
   blocks_filter: ["assessment_item", "example"]
   estimated_block_count: 247  (read from outline manifest)
   ```

No new code — just extra fields in `_dry_run_plan`'s return.

## 8. Help text + documentation

- Per-subcommand `--help` in the click decorator, with the "Examples" block pattern from `cli/commands/run.py:503-516`.
- Update `CLAUDE.md` § "Quick Start" (currently `CLAUDE.md:5-30`) to add a "Phase 5: Stage-by-stage Courseforge" subsection beneath the canonical entry-point examples.
- Update `CLAUDE.md:55-63` "Available Workflows" table with the four new entries (Phase 5-prep refresh: was "five new entries" before `courseforge-classify` was dropped).
- New file (Phase 5 deliverable, **not** this plan): `Courseforge/docs/stages.md` — one-page operator reference linking each subcommand to its tier, inputs, outputs, and `--blocks` semantics.

## 9. Decision-capture wiring

`PipelineOrchestrator._captures_dir` at `MCP/orchestrator/pipeline_orchestrator.py:127` already produces `training-captures/{tool}/{course_code}/phase_{phase}`. Phase 5 stages emit into `training-captures/courseforge/{COURSE_CODE}/phase_content_generation_outline/`, `..._rewrite/`, etc. — already aligned, no code change. The `phase_` prefix is part of the existing naming, so per-stage captures are addressable by phase name out of the box.

## 10. Backward compatibility

- `ed4all run textbook-to-course` keeps working unchanged (no entry in `SUPPORTED_WORKFLOWS` is removed).
- `ed4all run courseforge` (the new full-slice command) is a strict alias path — it forwards to `textbook_to_course` with `--corpus` populated.
- `--reuse-objectives` keeps its semantics; the new `courseforge-rewrite --blocks` flag is **disjoint** from it.
- Existing dry-run output gains optional fields; consumers reading the JSON output get backwards-compatible additions.

## 11. Testing plan

Unit tests:
- `tests/cli/test_run_courseforge_subcommands.py` — argparse/click dispatch, flag validation, `--blocks` parsing (rejects unknown values).
- `tests/workflow/test_synthesize_outline_output.py` — given a fixture OUTLINE_DIR, the synthesiser returns the right `_completed` dict and the runner skips upstream phases.

Integration tests (existing fixture pattern under `MCP/tests/test_workflow_runner_reuse_objectives.py`):
- Run `courseforge-outline` standalone on a small fixture; assert OUTLINE_DIR has the right page count and JSON-LD validates against `schemas/knowledge/courseforge_jsonld_v1.schema.json`.
- Run `courseforge-rewrite` on that OUTLINE_DIR; assert the FINAL_DIR has the same page count and `manifest.json.content_hash` differs.

Re-execution test (the headline behavioural test):
- Full pipeline run → snapshot every block's `content_hash`.
- `ed4all run courseforge-rewrite --blocks assessment_item --model X` over the same OUTLINE_DIR.
- Assert: every `assessment_item` block's `content_hash` may have changed; every other block's hash is **byte-identical**; every `assessment_item` block has one new `touched_by[]` entry; non-`assessment_item` blocks have no new entry.

## 12. Sequencing (subtasks in order)

1. Add `target_block_ids` routing param + `--blocks` plumbing into `_build_workflow_params` and `config/workflows.yaml` `content_generation_rewrite.inputs_from`.
2. Add `_synthesize_outline_output` to `MCP/core/workflow_runner.py` (Phase 5-prep refresh: `_synthesize_classification_output` dropped — no separate classifier tier exists; classification runs inline via `bloom_classifier_disagreement` gate).
3. Extend `SUPPORTED_WORKFLOWS` with the four new names (Phase 5-prep refresh: was five; `courseforge-classify` dropped per §1); add per-name parameter validation (course-code required, mutually-exclusive flags).
4. Implement subcommand handlers. Outline / rewrite reuse `_create_and_run`; validate uses a new `_validate_only` path (no orchestrator). Each handler MUST set `COURSEFORGE_TWO_PASS=true` in the workflow environment (see §3 sub-section "Two-pass env-var requirement"). Each handler also writes the operator-facing `02_validation_report/report.json` aggregation when running validation (see §6).
5. Add `--force` for re-run-despite-checkpoint.
6. Plumb `--blocks` through dry-run plan output.
7. Tests (unit, integration, re-execution).
8. Docs update (`CLAUDE.md` + new `Courseforge/docs/stages.md`).
9. Smoke run on a fixture corpus end-to-end and per-stage.

## 13. Risks & rollback

1. **Subcommand sprawl.** Four new entries grow `SUPPORTED_WORKFLOWS` from 6 to 10 (Phase 5-prep refresh: was "five new entries → 11" before `courseforge-classify` was dropped). Mitigation: prefix all four with `courseforge-`; add a "Courseforge stages" section in help output. Rollback: remove the new entries — main `textbook_to_course` is untouched.
2. **Schema drift between stages.** OUTLINE_DIR written by Phase 3 v1 may be read by Phase 5 rewrite v2 after a JSON-LD schema change. Mitigation: write `manifest.json.schema_version`; rewrite refuses if mismatch. Rollback: pin schema version in CI.
3. **Partial re-execution corruption.** `--blocks rewrite` could fail mid-page, leaving FINAL_DIR with some blocks rewritten and others stale. Mitigation: write to `FINAL_DIR.tmp/` then rename atomically (mirrors how IMSCC packaging already finalises). Rollback: delete `FINAL_DIR`, re-run.

## 14. Open questions

1. **Should `courseforge-validate` run Phase 4 statistical gates only, or include Phase 4 + the existing critical gates from `config/workflows.yaml:639-672` (`content_grounding`, `source_refs`)?** Defaulting to "all gates available for the outline phase" feels right but couples Phase 5 to the gate inventory.
2. **Should we add `--seed` for reproducible LLM calls in rewrite tier?** Not in scope per the prompt, but the idempotency contract is weaker without it. Recommend filing as a separate follow-up.
3. ~~**Is `courseforge-classify` mutating OUTLINE_DIR acceptable…?**~~ *(Resolved Phase 5-prep refresh: subcommand dropped — §1.)*
4. ~~**What's the correct Block-type vocabulary?**~~ *(Resolved Phase 5-prep refresh: §3 now cites the 16 singular `BLOCK_TYPES` from `Courseforge/scripts/blocks.py:77`.)*
5. ~~**Should `--blocks` be valid on `courseforge-classify` too?**~~ *(Resolved Phase 5-prep refresh: subcommand dropped — §1.)*

### Critical Files for Implementation
- `/home/user/Ed4All/cli/commands/run.py`
- `/home/user/Ed4All/MCP/core/workflow_runner.py`
- `/home/user/Ed4All/config/workflows.yaml`
- `/home/user/Ed4All/MCP/orchestrator/pipeline_orchestrator.py`
- `/home/user/Ed4All/CLAUDE.md`
