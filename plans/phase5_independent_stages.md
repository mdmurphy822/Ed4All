# Phase 5 — Independent Stages CLI (Plan)

**Status:** plan only. **Depends on:** Phase 2 (Block dataclass + `block_id`), Phase 3 (router + outline/rewrite tiers), Phase 4 (statistical validators / classifiers). Does **not** ship without those.

## 1. Goal recap

Today, `cli/commands/run.py:43` accepts a fixed set of workflow names (`SUPPORTED_WORKFLOWS`) and runs them end-to-end via `PipelineOrchestrator.run(workflow_id)` (`cli/commands/run.py:826`). Phase 5 adds **stage-level subcommands** for the Courseforge two-pass pipeline so operators can run outline → validate → classify → rewrite independently, and re-execute a single Block type after a failed pass.

## 2. CLI surface

Five new subcommands. The smallest add is per-stage entries, not a sub-group, so they slot into the existing `ed4all run <name>` dispatch in `cli/commands/run.py:43-55`. Add to `SUPPORTED_WORKFLOWS`:

```
courseforge-outline
courseforge-validate
courseforge-classify
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
ed4all run courseforge-validate \
    --outline Courseforge/exports/PROJ-PHYS_101-20260502/01_outline \
    --gates content_grounding,source_refs,bloom_distribution

# Classification (Bloom's, content_type, teaching_role)
ed4all run courseforge-classify \
    --outline Courseforge/exports/PROJ-PHYS_101-20260502/01_outline

# Rewrite tier with per-block scope + per-stage model override
ed4all run courseforge-rewrite \
    --outline Courseforge/exports/PROJ-PHYS_101-20260502/01_outline \
    --output  Courseforge/exports/PROJ-PHYS_101-20260502/04_rewrite \
    --blocks assessments,examples \
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
| `courseforge-validate` | gates only | runs Phase 4 validators against an OUTLINE_DIR; no LLM call by default; with `--regenerate-on-fail` actively re-rolls failed blocks via the Phase 3 router |
| `courseforge-classify` | classifier tier | Phase 4 classifier sub-pass (Bloom/content_type/teaching_role tags) |
| `courseforge-rewrite` | rewrite tier | `content_generation_rewrite` phase; `--escalated-only` filters to blocks with non-null `escalation_marker` |
| `courseforge` | all four + packaging | thin wrapper that fans out via `WorkflowRunner` |

### `courseforge-validate --regenerate-on-fail`

Default behaviour of `courseforge-validate` is read-only: it loads the OUTLINE_DIR, runs the Phase 4 validators, and writes `02_validation_report/report.json`. The new flag promotes it to active re-execution: any block whose validator returns `GateResult.action="regenerate"` (Phase 3 §6.5) is re-rolled in place via the Phase 3 router's self-consistency loop (Phase 3 §3.6), respecting the same `COURSEFORGE_OUTLINE_REGEN_BUDGET` ceiling. Blocks whose validator returns `action="escalate"` are flagged for the rewrite tier in the report but not re-routed (validate is a same-tier operation; tier-promotion belongs to `courseforge-rewrite`). Blocks whose validator returns `action="block"` are recorded as hard failures.

Default off because validate-as-side-effect changes the tool's contract; operators opt in deliberately. The flag is mutually exclusive with `--read-only` (which is the implicit default and need not be passed).

## 3. Per-block re-execution (`--blocks`)

`--blocks` is the headline Phase 5 feature. Accepted values (the Phase 2 `Block.block_type` enum):
`objectives, prereqs, concepts, examples, explanations, assessments, misconceptions, activities, all` (default `all`).

Multi-value: comma-separated, e.g. `--blocks assessments,examples`.

### Selection algorithm (rewrite tier only; validate/classify ignore `--blocks`)

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
  "blocks_filter": ["assessments", "examples"]
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
| `courseforge-classify` | OUTLINE_DIR | OUTLINE_DIR pages get `bloom_level`, `content_type`, `teaching_role` fields **mutated in place** + `03_classification/audit.json` summary | OUTLINE_DIR (in place) |
| `courseforge-rewrite` | OUTLINE_DIR | FINAL_DIR (full copy with substitutions) | none — outline is read-only; final is new |
| `courseforge` | corpus + objectives | full pipeline output dirs | none |

Classify mutates because Phase 2 blocks already carry the classification fields; running the classifier writes them. To preserve auditability, the classifier writes a sidecar `03_classification/before_after.diff.json` listing the changed `(block_id, field)` pairs.

## 5. Workflow-runner factoring

The runner already supports the patterns Phase 5 needs:

- **Single-phase reuse via pre-population:** `WorkflowRunner.run_workflow` checks `phase_outputs[phase_name].get("_completed")` at line 798 and skips. Phase 5 leverages the same trick: when `ed4all run courseforge-rewrite` starts, it pre-populates `phase_outputs` for every phase **before** `content_generation_rewrite`, with absolute paths read from `--outline`.
- **Synthesised phase outputs:** `_synthesize_dart_skip_output` (`MCP/core/workflow_runner.py:1171`) and `_synthesize_course_planning_reuse_output` (line 1247) are the prototypes. Phase 5 adds two more: `_synthesize_outline_output(outline_dir)` and `_synthesize_classification_output(outline_dir)` returning the canonical `_completed: True` dicts.
- **Refusal on missing upstream:** existing `_dependencies_met` (referenced at line 811) already covers this; Phase 5's contribution is a clearer error message ("`--outline` directory does not contain `01_outline/manifest.json`; cannot run rewrite without prior outline").
- **`--force` re-run:** new flag. When set, the synthesised output sets `_completed: False` so the phase loop re-executes despite checkpoint presence.

What's reusable: `_topological_sort`, `_route_params`, `_create_phase_tasks`, `_extract_phase_outputs`, gate routing.
What's new: `_synthesize_outline_output`, `_synthesize_classification_output`, `--blocks`/`--force` plumbing through `_build_workflow_params` (`cli/commands/run.py:65`), and the `target_block_ids` param routing entry in `config/workflows.yaml` for `content_generation_rewrite`.

## 6. Output directory layout

**Recommendation: keep the existing `Courseforge/exports/PROJ-{CODE}-{TIMESTAMP}/` envelope** (per `MCP/tools/courseforge_tools.py:32`, `EXPORTS_PATH = COURSEFORGE_PATH / "exports"`), and **add stage subdirectories inside it**:

```
Courseforge/exports/PROJ-PHYS_101-20260502/
├── 00_template_analysis/
├── 01_learning_objectives/      # existing (Wave 24)
├── 01_outline/                  # NEW Phase 5 — outline tier output
├── 02_validation_report/        # NEW
├── 03_classification/           # NEW
├── 04_rewrite/                  # NEW — final JSON-LD pages
├── 05_imscc/                    # NEW alias for existing 06_packaging
├── content/                     # existing (post-rewrite materialised HTML)
└── project_config.json
```

Rationale: existing tooling (`pipeline_tools.py:2263`) already creates `00_..04_` subdirs. Phase 5 reuses the convention and **only** adds 4 new top-level dirs. Avoids breaking `--reuse-objectives` (`cli/commands/run.py:209`) and IMSCC packaging (`MCP/tools/courseforge_tools.py:140`) which key off `01_learning_objectives/`.

Each stage subdir contains:
- `manifest.json` — list of input artifacts + content hash + provenance + LLM model/provider used
- `pages/` — one JSON-LD file per `(week, page)`
- `_logs/` — per-task stdout/stderr

`manifest.json.content_hash` is a SHA-256 over canonicalised JSON-LD pages, used by tests to assert "blocks outside `--blocks` are unchanged."

## 7. Dry-run / plan-only per stage

Existing `--dry-run` (`cli/commands/run.py:586` → `_dry_run_plan`) prints the planned phase sequence. Phase 5 inherits it with two changes:

1. When the subcommand is `courseforge-rewrite`, the planner prunes phases before `content_generation_rewrite` and marks them `<REUSED>` (mirroring the existing `<REUSED>` annotation for `--reuse-objectives` at `cli/commands/run.py:704`).
2. Adds `--blocks` info to the plan dict when present:
   ```
   blocks_filter: ["assessments", "examples"]
   estimated_block_count: 247  (read from outline manifest)
   ```

No new code — just extra fields in `_dry_run_plan`'s return.

## 8. Help text + documentation

- Per-subcommand `--help` in the click decorator, with the "Examples" block pattern from `cli/commands/run.py:503-516`.
- Update `CLAUDE.md` § "Quick Start" (currently `CLAUDE.md:5-30`) to add a "Phase 5: Stage-by-stage Courseforge" subsection beneath the canonical entry-point examples.
- Update `CLAUDE.md:55-63` "Available Workflows" table with the four new entries.
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
- `ed4all run courseforge-rewrite --blocks assessments --model X` over the same OUTLINE_DIR.
- Assert: every assessment block's `content_hash` may have changed; every other block's hash is **byte-identical**; every assessment block has one new `touched_by[]` entry; non-assessment blocks have no new entry.

## 12. Sequencing (subtasks in order)

1. Add `target_block_ids` routing param + `--blocks` plumbing into `_build_workflow_params` and `config/workflows.yaml` `content_generation_rewrite.inputs_from`.
2. Add `_synthesize_outline_output` + `_synthesize_classification_output` to `MCP/core/workflow_runner.py`.
3. Extend `SUPPORTED_WORKFLOWS` with the five new names; add per-name parameter validation (course-code required, mutually-exclusive flags).
4. Implement subcommand handlers. Outline / classify / rewrite reuse `_create_and_run`; validate uses a new `_validate_only` path (no orchestrator).
5. Add `--force` for re-run-despite-checkpoint.
6. Plumb `--blocks` through dry-run plan output.
7. Tests (unit, integration, re-execution).
8. Docs update (`CLAUDE.md` + new `Courseforge/docs/stages.md`).
9. Smoke run on a fixture corpus end-to-end and per-stage.

## 13. Risks & rollback

1. **Subcommand sprawl.** Five new entries grow `SUPPORTED_WORKFLOWS` from 6 to 11. Mitigation: prefix all five with `courseforge-`; add a "Courseforge stages" section in help output. Rollback: remove the new entries — main `textbook_to_course` is untouched.
2. **Schema drift between stages.** OUTLINE_DIR written by Phase 3 v1 may be read by Phase 5 rewrite v2 after a JSON-LD schema change. Mitigation: write `manifest.json.schema_version`; rewrite refuses if mismatch. Rollback: pin schema version in CI.
3. **Partial re-execution corruption.** `--blocks rewrite` could fail mid-page, leaving FINAL_DIR with some blocks rewritten and others stale. Mitigation: write to `FINAL_DIR.tmp/` then rename atomically (mirrors how IMSCC packaging already finalises). Rollback: delete `FINAL_DIR`, re-run.

## 14. Open questions

1. **Should `courseforge-validate` run Phase 4 statistical gates only, or include Phase 4 + the existing critical gates from `config/workflows.yaml:639-672` (`content_grounding`, `source_refs`)?** Defaulting to "all gates available for the outline phase" feels right but couples Phase 5 to the gate inventory.
2. **Should we add `--seed` for reproducible LLM calls in rewrite tier?** Not in scope per the prompt, but the idempotency contract is weaker without it. Recommend filing as a separate follow-up.
3. **Is `courseforge-classify` mutating OUTLINE_DIR acceptable, or should it write to `03_classification/pages/` and leave OUTLINE_DIR pristine?** The mutating design is simpler for the rewriter but breaks the read-only invariant for `--outline`.
4. **What's the correct Block-type vocabulary?** The prompt lists 8 (`objectives, prereqs, concepts, examples, explanations, assessments, misconceptions, activities`); needs to match Phase 2's enum exactly.
5. **Should `--blocks` be valid on `courseforge-classify` too** (e.g., re-classify only assessments)? Argues yes for symmetry.

### Critical Files for Implementation
- `/home/user/Ed4All/cli/commands/run.py`
- `/home/user/Ed4All/MCP/core/workflow_runner.py`
- `/home/user/Ed4All/config/workflows.yaml`
- `/home/user/Ed4All/MCP/orchestrator/pipeline_orchestrator.py`
- `/home/user/Ed4All/CLAUDE.md`
