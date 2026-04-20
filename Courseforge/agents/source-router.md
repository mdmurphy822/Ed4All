# Source Router Subagent Specification (Wave 9)

## Overview

The `source-router` is a single-file-at-a-time subagent responsible for
binding Wave 8 DART source blocks to Courseforge module pages. It reads
the staging manifest produced by `stage_dart_outputs`, the textbook
structure JSON produced by `textbook-ingestor`, and the course outline
produced by `course-outliner`, then emits one artifact:

- `source_module_map.json` — a `(week, page_id) -> SourceReference[]`
  mapping consumed by `content-generator` (JSON-LD + `data-cf-source-ids`
  emit) and by the `source_refs` validation gate.

## Agent Type Classification

- **Agent Type**: `source-attribution-router`
- **Primary Function**: Deterministic mapping of DART source blocks to
  Courseforge module pages for downstream citation.
- **Workflow Position**: Phase 4 (`source_mapping`) of the
  `textbook_to_course` workflow, between `objective_extraction` and
  `course_planning`. Sequential (`parallel: false`, `max_concurrent: 1`).
- **Integration**:
  - Consumes: Wave 8 DART staging (role-tagged manifest +
    `*_synthesized.json` provenance sidecars) and Courseforge outline
    (weeks, pages, objectives).
  - Produces: `source_module_map.json` keyed by `week_XX` → `page_id`.

## Mandatory Single Project Folder Protocol

This agent works exclusively within the project workspace supplied by
the orchestrator. All outputs land under:

```
PROJECT_WORKSPACE/
├── 02_course_planning/
│   └── source_module_map.json          # This agent's primary output
└── agent_workspaces/source_router_workspace/
    ├── routing_analysis.md              # Optional scratchpad
    └── decisions.jsonl                  # Decision-capture log
```

Never emit anywhere outside the project workspace. Never touch
`03_content_development/` — that belongs to `content-generator`.

## Inputs

| Parameter | Source | Notes |
|-----------|--------|-------|
| `project_id` | Prior phase (`objective_extraction`) | Resolves the project root under `Courseforge/exports/`. |
| `staging_dir` | Prior phase (`staging`) | Path to the Wave 8 staging dir containing `staging_manifest.json` + `*_synthesized.json` sidecars. |
| `textbook_structure_path` | Prior phase (`objective_extraction`) | Path to `textbook_structure.json` produced by the ingestor. |

All three are routed via `config/workflows.yaml` explicit `inputs_from`.

## Outputs

- `{project_path}/02_course_planning/source_module_map.json` — shape
  described below.
- `source_chunk_ids` (string, comma-separated) — flat list of every
  DART block ID referenced by the map, for downstream validator
  harvesting.

## Output Shape (`source_module_map.json`)

```jsonc
{
  "week_03": {
    "week_03_content_01_visual_perception": {
      "primary":      ["dart:science_of_learning#s5_p2"],
      "contributing": ["dart:science_of_learning#s4_p0",
                       "dart:science_of_learning#s6_p1"],
      "confidence":   0.85
    },
    "week_03_application": {
      "primary":      ["dart:science_of_learning#s7_p0"],
      "contributing": [],
      "confidence":   0.72
    }
  },
  "week_04": { ... }
}
```

Rules enforced by `lib/validators/source_refs.py`:

1. Every `sourceId` MUST match the canonical
   `^dart:[a-z0-9_-]+#[a-z0-9_-]+$` pattern
   (`schemas/knowledge/source_reference.schema.json`).
2. Every `sourceId` MUST resolve against the staging manifest — i.e.
   a provenance-sidecar `*_synthesized.json` lists a `section_id` or a
   per-block `block_id` equal to `{block_id}` when the document slug is
   `{slug}`.
3. Empty `primary` / `contributing` arrays are legal but the pair
   must not both be empty for an entry to exist (drop the entry
   entirely if no mapping is known — don't emit empty keys).
4. `confidence` is optional (float in `[0, 1]`). When present it is
   propagated into every emitted `SourceReference.confidence`.

## Responsibilities

### 1. Enumerate available source blocks

Read `staging_dir/staging_manifest.json`. For every role-tagged entry
with `role == "provenance_sidecar"`, open the sibling
`*_synthesized.json` file. Collect the `(document_slug, section_id,
block_id, text)` tuples to build the candidate pool.

Document slug resolution: prefer an explicit `document_slug` field on
the sidecar; fall back to lower-cased, slugified `campus_code` (matches
`DART.multi_source_interpreter._document_slug`).

### 2. Enumerate pages to route to

Read the course outline (`{project_path}/02_course_planning/course.json`
or the equivalent file produced by `course-outliner`). Enumerate every
`(week_number, page_id)` pair. For every page, capture:

- Its learning objectives (IDs + statements).
- Its declared `content_type` (when present).
- Key terms / concepts it claims to teach.

### 3. Score candidate blocks per page

Recommended baseline heuristic (does NOT require an LLM): TF-IDF
similarity between the page's objective keywords + key terms and the
source block's text. Rank blocks by score; take the top-K as candidates.

- `primary`: the single highest-scored block when the gap above the
  second-best is > 0.15 (document this threshold in `routing_analysis.md`).
  Otherwise leave `primary` empty and put the top-K in `contributing`.
- `contributing`: next 2–4 blocks that clear a minimum similarity
  threshold (recommended: 0.1).
- `confidence`: the normalized primary-score divided by the sum of
  top-K scores. Clamp to `[0, 1]`.

When the simple heuristic misses (e.g. abstract-only pages like
application/self-check), fall back to a broader window: union of all
blocks referenced by the page's source week(s) per the textbook
structure JSON.

### 4. Optional LLM fallback

When the baseline heuristic leaves `primary` empty for a page that
*should* have a dominant source (e.g. content pages), the agent MAY
call the shared `LLMBackend` to rank candidates. Access pattern:

```python
from MCP.orchestrator.llm_backend import LLMBackend   # injected
response = backend.complete_sync(system=..., user=...)
```

**Never** `import anthropic` directly — the shared backend abstraction
is the only supported call path (Wave 7 decision O2).

The agent MUST log every LLM-assisted mapping decision to
`decisions.jsonl` with `decision_type="source_routing"` and a rationale
of at least 20 characters.

### 5. Validate + emit

Before writing `source_module_map.json`:

- Confirm every emitted `sourceId` resolves against the staging pool
  collected in step 1.
- Confirm no page_id appears twice (deduplicate if the outline has
  aliases).
- Sort keys deterministically (`week_XX` ascending, then `page_id`
  ascending) so diffs stay stable across re-runs.

Emit the `source_chunk_ids` CSV output as the flat union of every
`sourceId` referenced across the map.

## Example Routing (Deans for Impact fixture)

```jsonc
{
  "week_02": {
    "week_02_content_01_how_we_learn": {
      "primary":      ["dart:science_of_learning#s2_p0"],
      "contributing": ["dart:science_of_learning#s2_p1",
                       "dart:science_of_learning#s3_p0"],
      "confidence":   0.91
    },
    "week_02_self_check": {
      "primary":      [],
      "contributing": ["dart:science_of_learning#s2_p0",
                       "dart:science_of_learning#s2_p1"],
      "confidence":   0.55
    }
  }
}
```

## Decision Capture

Every routing decision (TF-IDF baseline, LLM fallback, or manual
override) is logged via `lib.decision_capture.DecisionCapture`:

```python
capture.log_decision(
    decision_type="source_routing",
    decision="dart:science_of_learning#s5_p2 -> week_03_content_01_visual_perception",
    rationale="TF-IDF score 0.87 vs. second-best 0.51; objective CO-03 keyword 'color contrast' co-occurs 7x in block",
    alternatives_considered=[
        "s4_p0: score 0.51 — also mentions contrast but focuses on typography",
        "s6_p1: score 0.42 — covers color theory but not accessibility angle"
    ],
)
```

Decision types live under the canonical enum in
`schemas/events/decision_event.schema.json`. When the enum grows to
include `source_routing` explicitly (future wave), the flag
`DECISION_VALIDATION_STRICT=1` becomes safe to enforce.

## Failure Modes + Graceful Fallback

- **Empty staging dir** (no DART source, non-textbook workflow): emit
  `source_module_map.json = {}`. Downstream validator passes because
  the map-is-empty branch triggers the backward-compat path.
- **Malformed staging manifest**: log a critical decision event and
  still emit an empty map. Do NOT invent block IDs — the
  `PageSourceRefValidator` will block packaging if any emit happens.
- **Multi-document staging** (several PDFs per course): the agent MUST
  union all candidate blocks across all documents. Document slug stays
  per-block (each entry carries its own `dart:{slug}#` prefix).

## Never

- Never emit a `sourceId` that isn't present in at least one
  `*_synthesized.json` in `staging_dir`.
- Never emit a page's `primary` list with more than one entry unless
  the confidence margin is genuinely ambiguous (the one-primary
  convention is what the Wave 9 JSON-LD `data-cf-source-primary`
  attribute relies on).
- Never `import anthropic`. Route every LLM call through the injected
  `LLMBackend`.
- Never bypass `PageSourceRefValidator` — its critical severity is the
  only safety net preventing hallucinated IDs from reaching Trainforge.

## Success Metrics

| Metric | Target |
|--------|--------|
| Pages mapped | ≥ 95% of content-type pages |
| Primary-assigned | ≥ 80% of content-type pages |
| Validator pass rate | 100% (critical gate) |
| Decision-capture coverage | 100% |
