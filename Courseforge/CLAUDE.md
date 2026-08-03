# Courseforge

AI-powered instructional design system that creates and remediates accessible, LMS-ready IMSCC course packages.

> **Universal Protocols**: See root `/CLAUDE.md` for orchestrator protocol, execution rules (ONE agent = ONE file, max 10 parallel), decision capture requirements, and error handling. This file contains Courseforge-specific guidance only.

---

## Quick Start

### Course Creation Mode
**Input**: Exam objectives (PDF/text) + optional SemantiK-converted textbooks (accessible HTML)
**Output**: Single IMSCC file ready for Brightspace import

### Course Intake/Remediation Mode
**Input**: Any IMSCC package (Canvas, Blackboard, Moodle, Brightspace, etc.)
**Output**: Fully accessible, enhanced IMSCC with 100% WCAG 2.2 AA compliance

### Provider selection

Set `COURSEFORGE_PROVIDER=local` to route content authoring through a license-clean local OSS provider. See § Opt-In Behavior Flags below for the env-var contract and `docs/LICENSING.md` for the ToS posture.

---

## Workflow Pipelines

### Pipeline 1: Course Creation
```
INPUT                         PROCESSING                              OUTPUT
─────                         ──────────                              ──────
Exam Objectives ──┐
(PDF/text)        │
                  ├──► exam-research ──► course-outliner ──► content-generator ──► brightspace-packager ──► IMSCC
Textbooks ────────┘         │                │                    │
(SemantiK HTML)       requirements-      oscqr-            quality-assurance
                      collector          evaluator              (per batch)
```

### Pipeline 2: Intake & Remediation
```
INPUT                         PROCESSING                                    OUTPUT
─────                         ──────────                                    ──────
Any IMSCC Package ──► imscc-intake-parser ──► content-analyzer ──┬──► semantik-automation-coordinator
(Canvas, Blackboard,          │                   │               │           (PDF/Office → HTML)
 Moodle, Brightspace)         │                   │               │
                              │                   │               ├──► accessibility-remediation
                              ▼                   │               │           (WCAG fixes)
                     LMS Detection                │               │
                     Version Detection            │               ├──► content-quality-remediation
                     Content Inventory            │               │           (Educational depth)
                                                  │               │
                                                  │               ├──► intelligent-design-mapper
                                                  │               │           (Component styling)
                                                  │               │
                                                  │               └──► remediation-validator ──► brightspace-packager ──► Improved IMSCC
                                                  │                           (Final QA)
                                                  ▼
                                         Remediation Queue
```

---

## Orchestrator Protocol

**The orchestrator is a lightweight task manager. Specialized agents determine frameworks and content structure.**

### Orchestrator Responsibilities
1. Create timestamped project folder in `exports/`
2. Invoke planning agent → receive todo list (NO EXECUTION)
3. Load todo list into TodoWrite (single source of truth)
4. Execute todos via specialized agents in parallel batches
5. Coordinate quality validation
6. Invoke final packaging

### Workflow Steps
```
USER REQUEST →
  STEP 1: Planning agent analyzes request, returns todo list (NO execution) →
  STEP 2: Orchestrator loads todo list into TodoWrite →
  STEP 3: Orchestrator executes todos via agents (agents do NOT modify TodoWrite) →
  STEP 4: Quality validation (oscqr-course-evaluator + quality-assurance) →
  STEP 5: Package generation (brightspace-packager) →
  OUTPUT: Single IMSCC file
```

---

## Available Agents

### Course Creation Agents
| Agent | Purpose | When to Use |
|-------|---------|-------------|
| `requirements-collector` | Course specification gathering | New course projects |
| `course-outliner` | Synthesize canonical `TO-NN` / `CO-NN` objectives from textbook structure; persist `synthesized_objectives.json`. Routes to `plan_course_structure`. | Creating course framework |
| `content-generator` | Educational content creation | Content development (1 file per agent) |
| `quality-assurance` | Pattern prevention and validation | Quality gates |
| `oscqr-course-evaluator` | Educational quality assessment | OSCQR evaluation |
| `brightspace-packager` | IMSCC package generation | Final deployment |
| `textbook-ingestor` | Textbook content processing | Entry point for textbook materials |
| `source-router` | Bind SemantiK source blocks to Courseforge module pages | Source attribution for pipeline runs |

### Intake & Remediation Agents
| Agent | Purpose | When to Use |
|-------|---------|-------------|
| `imscc-intake-parser` | Universal IMSCC package parsing | Importing existing courses |
| `content-analyzer` | Accessibility/quality gap detection | Analyzing imported content |
| `semantik-automation-coordinator` | Orchestrates the SemantiK v2 conversion cascade (routes to `extract_and_convert_pdf`). A read-compat dispatch alias in `MCP/core/executor.py::AGENT_TOOL_MAPPING` covers legacy pre-SemantiK resume states. | Converting PDFs/Office docs to accessible HTML |
| `accessibility-remediation` | Automatic WCAG fixes | Fixing accessibility issues |
| `content-quality-remediation` | Educational depth enhancement | Improving thin content |
| `intelligent-design-mapper` | AI-driven component selection | Applying interactive styling |
| `remediation-validator` | Final remediation QA (routes to `get_courseforge_status`; WCAG verification itself runs as the `wcag_compliance` gate, not as an agent tool) | Post-remediation validation |

---

## Critical Execution Protocols

### Individual File Protocol (MANDATORY)
- ONE agent = ONE file (never multiple files per agent)
- Maximum 10 simultaneous Task calls per batch
- Wait for batch completion before next batch

**Correct:**
```python
Task(content-generator, "Create week_01_module_01_introduction.html")
Task(content-generator, "Create week_01_module_02_concepts.html")
# ... up to 10 per batch
```

**Wrong:**
```python
Task(content-generator, "Create all Week 1 content")  # NEVER DO THIS
```

---

## Project Structure

```
/Courseforge/
├── CLAUDE.md                    # This file
├── README.md                    # Project overview
├── docs/                        # Documentation
│   ├── troubleshooting.md       # Error patterns and solutions
│   ├── workflow-reference.md    # Detailed workflow protocols
│   ├── getting-started.md       # Quick start guide
│   ├── per-week-learning-objectives.md
│   └── template-chrome-roles.md
├── agents/                      # Agent specifications
├── config/                      # block_routing.yaml, block_catalog.yaml
├── generators/                  # Outline / rewrite / synthesis LLM providers
├── router/                      # Two-pass router, policy, inter-tier gates
├── inputs/                      # Input files
│   ├── exam-objectives/         # Certification exam PDFs/docs
│   ├── textbooks/               # SemantiK-converted accessible HTML textbooks
│   ├── existing-packages/       # IMSCC packages for intake
│   └── course-data/             # Per-course input data
├── templates/                   # HTML templates and components
├── schemas/                     # IMSCC and content schemas
├── imscc-standards/             # Brightspace/IMSCC technical specs
├── scripts/                     # Automation scripts
│   ├── imscc-extractor/         # Universal IMSCC extraction
│   ├── component-applier/       # Interactive component application
│   ├── accessibility-validator/ # Accessibility checks
│   └── remediation-validator/   # Final quality validation
├── exports/                     # Generated course packages
│   └── YYYYMMDD_HHMMSS_name/    # Timestamped project folders
└── runtime/                     # Agent workspaces (auto-created)
```

### Export Project Structure
```
exports/YYYYMMDD_HHMMSS_coursename/
├── 00_template_analysis/
├── 01_learning_objectives/      # synthesized_objectives.json
├── 01_outline/                  # two-pass outline tier: blocks*.jsonl
├── 02_course_planning/
├── 02_validation_report/        # inter_tier_validation report.json
├── 03_content_development/
│   ├── week_01/
│   └── week_XX/
├── 04_quality_validation/
├── 04_rewrite/                  # two-pass rewrite tier + its report.json
├── 05_final_package/
├── 06_assessments/              # W10 QTI / imsdt / assignment XML + manifest
├── agent_workspaces/
├── project_log.md
└── coursename.imscc              # Final deliverable
```

---

## Textbook Integration

Textbooks must be pre-processed to accessible HTML before use. Conversion runs
in-process via the `[semantik]` extra through the `semantik_conversion` workflow
phase; the `semantik-converter` agent drives the SemantiK v2 cascade (a
read-compat dispatch alias in `MCP/core/executor.py::AGENT_TOOL_MAPPING` covers
legacy pre-SemantiK resume states):

1. Run the `semantik_conversion` phase (e.g. via `ed4all run textbook-to-course`),
   which converts the textbook PDF to accessible HTML and stages it for
   Courseforge.
2. The SemantiK v2 cascade produces WCAG 2.2 AA accessible HTML.
3. Staged output lands under `Courseforge/inputs/textbooks/{run_id}/`.
4. Reference in course generation.

For a conversion-only slice with no course scaffolding, `ed4all convert`
(`cli/commands/convert.py`) runs the same cascade standalone and emits
`{stem}_accessible.html` plus sidecars. See `docs/operations/convert-verb.md`.

---

## Quality Standards

### Pattern Prevention
See `docs/guides/troubleshooting.md` for complete pattern list. Critical patterns:
- Schema/namespace consistency (IMS CC 1.1)
- Assessment XML format (QTI 1.2)
- Content completeness (all weeks substantive)
- Organization hierarchy (no empty structures)

### OSCQR Evaluation

Shipped gate: `oscqr_score` in `config/workflows.yaml`, validator
`lib.validators.oscqr.OSCQRValidator`, `threshold.min_score: 0.7`, severity
**warning** (`on_fail: warn` / `on_error: warn`) — it reports below-threshold
courses, it does not block them. The threshold is applied by the gate manager,
not inside the validator.

A higher pre-production bar and a blocking severity flip are aspirations, not
shipped behavior; do not document them as enforced until the gate config says so.

---

## Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Troubleshooting | `docs/guides/troubleshooting.md` | Error patterns and solutions |
| Workflow Reference | `docs/reference/workflow-reference.md` | Detailed execution protocols |
| Getting Started | `docs/guides/getting-started.md` | Quick start guide |
| Pattern Prevention | `docs/guides/troubleshooting.md` | Error patterns and prevention |
| Agent Specs | `agents/*.md` | Individual agent protocols |

---

## CSS Color Palette (for content generation)

```css
Primary Blue: #2c5aa0
Success Green: #28a745
Warning Yellow: #ffc107
Danger Red: #dc3545
Light Gray: #f8f9fa
Border Gray: #e0e0e0
```

---

## Opt-In Behavior Flags

Courseforge owns the `COURSEFORGE_*` / `COURSEPLANNER_*` / `TEXTBOOK_SYNTHESIS_*` env-var prefixes. The full per-flag table
(name, default, behavior, guardrails) lives in
[`docs/operations/behavior-flags-courseforge.md`](../docs/operations/behavior-flags-courseforge.md) —
read or grep that file before adding, removing, or changing a flag.

---

## MCP Tools

Courseforge is exposed via the Ed4All MCP server with these tools:

| Tool | Description |
|------|-------------|
| `create_course_project` **[DEPRECATED]** | Initialize a standalone (non-pipeline) course project. Still functional for external MCP clients, but new integrations should route through the pipeline-internal `extract_textbook_structure` + `plan_course_structure`. |
| `generate_course_content` | Generate content for weeks |
| `package_imscc` | Package course as IMSCC. Runtime delegates to `Courseforge/scripts/packaging/package_multifile_imscc.py` (IMS CC v1.3 namespaces, per-week LO validation, `course_metadata.json` bundling). |
| `intake_imscc_package` | Import existing IMSCC |
| `remediate_course_content` | Fix content issues |
| `get_courseforge_status` | Get project status |

---

## Metadata Output

Courseforge HTML pages embed machine-readable instructional design metadata for downstream consumption by Trainforge.

### Summary (downstream contract)

Courseforge HTML pages include machine-readable metadata for downstream Trainforge consumption:

- **`data-cf-*` attributes**: Inline metadata on HTML elements (role, objective IDs, Bloom's levels/verbs, cognitive domain, content types, teaching role, key terms, component, purpose). Canonical attribute table below.
- **JSON-LD blocks**: Structured `<script type="application/ld+json">` per page with learning objectives, section metadata, misconceptions, and assessment suggestions. Canonical shape: `schemas/knowledge/courseforge_jsonld_v1.schema.json`.

Priority extraction in Trainforge: JSON-LD > `data-cf-*` attributes > regex heuristics.

### HTML Data Attributes (`data-cf-*`)

| Attribute | Element | Purpose |
|-----------|---------|---------|
| `data-cf-role` | `<body>` (template chrome) | Page role classification (e.g. `template-chrome`) |
| `data-cf-objective-id` | `<li>` (objectives) | Learning objective identifier (canonical `TO-NN` / `CO-NN` pattern) |
| `data-cf-bloom-level` | `<li>`, `.self-check`, `.activity-card` | Bloom's taxonomy level |
| `data-cf-bloom-verb` | `<li>` (objectives) | Detected Bloom's verb |
| `data-cf-bloom-range` | `<section>`, `<h2>` | Section-level Bloom level span (emit-only) |
| `data-cf-cognitive-domain` | `<li>` (objectives) | Knowledge domain (factual/conceptual/procedural/metacognitive) |
| `data-cf-content-type` | `<h2>`, `<h3>`, `.callout` | Section content classification |
| `data-cf-teaching-role` | `<section>`, component wrappers | Pedagogical teaching role |
| `data-cf-key-terms` | `<h2>`, `<h3>` | Comma-separated term slugs |
| `data-cf-term` | key-term `<span>` | Individual term slug (emit-only) |
| `data-cf-component` | `.flip-card`, `.self-check`, `.activity-card` | Interactive component type |
| `data-cf-purpose` | `.flip-card`, `.self-check`, `.activity-card` | Pedagogical purpose |
| `data-cf-objective-ref` | `.self-check`, `.activity-card` | Associated learning objective |
| `data-cf-source-ids` | `<section>`, headings, component wrappers | SemantiK `sourceId`(s) that ground this block. Shape: `semantik:{slug}#{block_id}`. Carried through from SemantiK's `data-semantik-block-id` when source material is present; elided when no source grounding exists. |
| `data-cf-source-primary` | `<section>`, headings, component wrappers | The primary `sourceId` for the block (subset of `data-cf-source-ids`) when one source dominates. |
| `data-cf-block-id` | every block-bearing wrapper (`<section>`, headings, component wrappers) | Stable Block ID for cross-referencing JSON-LD `blocks[]` (gated behind `COURSEFORGE_EMIT_BLOCKS`). Shape: `{page_id}#{block_type}_{slug}_{idx}` per `Courseforge/scripts/blocks.py::Block.stable_id`. |
| `data-cf-curie` | appended hidden `<span>` | Space-separated minted CURIE(s) force-injected by `RewriteProvider` when the rewrite LLM dropped a source block's CURIEs after the remediation budget. Injected as an appended `<span hidden data-cf-curie="…">…</span>` whose **text content** carries the CURIE tokens — the str-path validator strips HTML tags (and their attributes) before scraping, so only text content survives; the `hidden` attribute keeps the span out of the render + accessibility tree. Absent on RDF corpora whose rewrite output retains CURIEs in prose, and on any block the LLM authored cleanly. See § "Three-stage textbook synthesis" → dynamic CURIE minting. |

**Flag-gated emit-only attributes** (absent by default; emitted only when the named opt-in flag is truthy — byte-stable snapshots when off):

| Attribute | Element | Purpose | Gate |
|-----------|---------|---------|------|
| `data-cf-bloom-triple` | every pedagogical block wrapper | Combined `(verb · level · knowledge-type)` chip (IB6.5); additive alongside the three separate objective attrs (`generate_course.py` / `blocks.py`). | `ED4ALL_BLOCK_QUALITY_RUBRIC` |
| `data-cf-udl-representations` | block wrapper | UDL multiple-means-of-representation count (`Block.n_representations`). | `ED4ALL_BLOCK_A11Y` |
| `data-cf-udl-response-formats` | block wrapper | UDL allowed learner response formats (`Block.response_formats`). | `ED4ALL_BLOCK_A11Y` |
| `data-cf-udl-engagement` | block wrapper | UDL engagement affordance (`Block.engagement_affordance`). | `ED4ALL_BLOCK_A11Y` |
| `data-cf-fade-state` | `worked_example` `<section>` (B05) | Worked→completion→independent fade stage (`Block.fade_state`). | `ED4ALL_NEW_BLOCK_TYPES` |
| `data-cf-transcript` | `multimedia` transcript `<details>` (B04) | Marks the time-based-media transcript disclosure for the B04 a11y stack. | `ED4ALL_NEW_BLOCK_TYPES` |
| `data-cf-scenario-mode` | `scenario-card` `<section>` (B09) | Case / scenario / branching mode (`Block.scenario_mode`, default `scenario`). | `ED4ALL_NEW_BLOCK_TYPES` |

Attributes stop at the **section / component wrapper level** — never on every `<p>` / `<li>` / `<tr>` in prose.

### Ancestor-walkable grounding

`ContentGroundingValidator` walks each non-trivial `<p>` / `<li>` / `<figcaption>` / `<blockquote>`'s ancestor chain to find the first `data-cf-source-ids` attribute. Three emit-side contracts keep that walk passing:

1. **Content sections are wrapped in `<section data-cf-source-ids="…">`.** `Courseforge/scripts/rendering/generate_course.py::_render_content_sections` wraps each h2/h3 + paragraph group in a `<section>` carrying the section's resolved source-ids.
2. **`content_NN` pages inherit `content_01` grounding.** `_page_refs_for` falls back from `content_NN` → `content_01` in the `source_module_map`. The source-router emits a single per-week `content_01` entry; every generated content page in that week shares the same SemantiK source region.
3. **Objectives `<section>` mirrors page-level source-ids.** `_render_objectives(..., source_ids=…)` stamps the page's resolved source-ids onto the `.objectives` wrapper so the injected objectives section carries the same grounding.

Converter-side slug contract (see `SemantiK/CLAUDE.md`): the `semantik:{slug}#{block_id}` slug uses `lowercase + space-to-hyphen` normalization (not `canonical_slug`'s underscore collapse), matching the validator's `_resolve_valid_block_ids` rule.

### JSON-LD Structured Metadata

Each page includes a `<script type="application/ld+json">` block in `<head>` with:
- `learningObjectives`: ID (canonical `TO-NN` / `CO-NN`), statement, Bloom's level/verb, cognitive domain, assessment suggestions
- `sections`: Heading, content type, Bloom's range, key terms with definitions, optional per-section `sourceReferences`
- `misconceptions`: Common misconceptions with corrections
- `suggestedAssessmentTypes`: Recommended question formats
- `prerequisitePages`: Cross-page prerequisite refs
- `sourceReferences`: Optional page-level SemantiK source references (canonical `{sourceId, role, weight?, confidence?, pages?, extractor?}` shape). Page-level JSON-LD `role` is authoritative (`primary` / `contributing` / `corroborating`) and takes precedence over attribute-level roles.

Canonical shape: `schemas/knowledge/courseforge_jsonld_v1.schema.json`. Context namespace: `https://ed4all.dev/ns/courseforge/v1`.

### Learning Objective IDs

Emitted LO IDs follow the pattern `^[A-Z]{2,}-\d{2,}$` from the canonical helper `lib/ontology/learning_objectives.py::mint_lo_id`:

- `TO-NN` — terminal (course-wide) objective.
- `CO-NN` — chapter-level objective.

Synthesized objectives are persisted to `{project}/01_learning_objectives/synthesized_objectives.json` by the `plan_course_structure` phase in the `textbook_to_course` pipeline. Downstream Trainforge consumers match case-insensitively; the `TRAINFORGE_PRESERVE_LO_CASE` flag preserves the emit case.

---

## Block format

Every page-level pedagogical unit (objective, concept, example, callout, flip card, self-check question, activity, …) is constructed as a frozen `Block` dataclass first, then projected to HTML via `Block.to_html_attrs()` and to a JSON-LD entry via `Block.to_jsonld_entry()`. Mutations return a new instance via `dataclasses.replace`; the `with_touch` helper appends to the immutable `touched_by` audit chain.

- **Dataclass + 30-value `BLOCK_TYPES` enum**: `Courseforge/scripts/blocks.py` (`Block` frozen dataclass + `BLOCK_TYPES` frozenset). `BLOCK_TYPES` is the authoritative palette (30 members: the 16 original + 5 Wave-2 + 3 Issue-I6 + 4 IB5 framework-aligned additions — `hook` B02 / `multimedia` B04 / `worked_example` B05 / `diagram` B06 — + 1 B15 `resources` + 1 FR-INT-02 `guided_practice` B08, the framework-aligned types emitted only via the dynamic planner under `ED4ALL_NEW_BLOCK_TYPES`); do not re-list the tokens here (they drift) — read the frozenset. Each token's canonical framework B-code parent is declared per-entry in `Courseforge/config/block_catalog.yaml` (`framework_block`).
- **Canonical JSON-LD shape**: `schemas/knowledge/courseforge_jsonld_v1.schema.json` (`$defs.Block`, `$defs.Touch`, `$defs.QualityScore`, top-level optional `blocks[]` / `provenance` / `contentHash`).

### Framework B-code reconciliation

The 30 Ed4All `BLOCK_TYPES` are finer-grained instances of the instruction-blocks framework's **canonical 15-block catalog (B01–B15)**: Objectives (B01), Hook/Activation (B02), Exposition (B03), Multimedia (B04), Worked Example (B05), Visual Model/Diagram (B06), Knowledge-Check (B07), Guided Practice (B08), Case/Scenario (B09), Discussion (B10), Reflection (B11), Callout (B12), Summary (B13), Graded Assessment (B14), Resources (B15). Each Ed4All type declares its canonical parent per-entry as `framework_block` in `Courseforge/config/block_catalog.yaml` (`framework_block_secondary` for a type that straddles two parents); `chrome` is non-pedagogical scaffolding and maps to `null`. The single-source-of-truth loader is `lib/ontology/framework_blocks.py` (`FRAMEWORK_BLOCKS` label table + `framework_block_for(block_type)`). The 30→15 map (IB2 + IB5 + B15 + FR-INT-02):

| Ed4All `block_type` | `framework_block` (primary) | secondary |
|---|---|---|
| `objective` | B01 | — |
| `recap` | B02 | — |
| `prereq_set` | B02 | — |
| `concept` | B03 | — |
| `explanation` | B03 | — |
| `example` | B05 | B03 |
| `formula` | B03 | B06 |
| `misconception` | B03 | B12 |
| `table` | B06 | B03 |
| `acronym` | B06 | B12 |
| `self_check_question` | B07 | — |
| `assessment_item` | B14 | — |
| `activity` | B08 | — |
| `problem` | B08 | — |
| `checklist` | B08 | B13 |
| `scenario` | B09 | — |
| `discussion_prompt` | B10 | — |
| `reflection_prompt` | B11 | — |
| `callout` | B12 | — |
| `key_idea` | B12 | — |
| `vocab_card` | B12 | B03 |
| `flip_card_grid` | B12 | B03 |
| `summary_takeaway` | B13 | — |
| `chrome` | `null` | — |
| `hook` | B02 | — |
| `multimedia` | B04 | — |
| `worked_example` | B05 | B08 |
| `diagram` | B06 | B03 |
| `resources` | B15 | — |
| `guided_practice` | B08 | — |

IB5 landed the four dedicated framework-aligned first-class types — `hook` (B02), `multimedia` (B04, the mandatory time-based-media a11y stack), `worked_example` (B05, distinct from the single-instance `example`), and `diagram` (B06) — so B04 gained an Ed4All primary. The B15 wave then added `resources` (B15 Resources / Further Reading — an accessible list of curated external links each with descriptive 2.4.4 link text), which closed the **last catalog gap**: EVERY canonical B-code now has an Ed4All primary. The five framework-aligned types (`hook`/`multimedia`/`worked_example`/`diagram`/`resources`) are emitted ONLY via the dynamic planner path under `ED4ALL_NEW_BLOCK_TYPES` (default OFF) so legacy snapshots stay byte-stable; `resources` additionally carries the `resource_link_purpose` WCAG-2.4.4 gate (warning-day-1, no-op when the flag is off).

When `COURSEFORGE_EMIT_BLOCKS=true`, `Courseforge/scripts/rendering/generate_course.py::_build_page_metadata` emits three additional top-level JSON-LD fields per page:

- `blocks[]` — ordered array of per-block JSON-LD entries built by `Block.to_jsonld_entry()`. Trainforge's `process_course._extract_section_metadata` prefers this projection over the `data-cf-*` HTML-attribute fallback when present.
- `provenance` — `{runId, pipelineVersion: "phase2", tiers[]}`. `runId` reads `COURSEFORGE_RUN_ID` from the environment.
- `contentHash` — SHA-256 hex of the meta dict canonicalised with `json.dumps(..., sort_keys=True, ensure_ascii=False)` BEFORE the `contentHash` field itself is added.

When the flag is off (default), the new fields are elided.

---

## Two-pass router

The content-generator surface splits into an **outline tier** (small local 7B model — terse, structurally-shaped first draft) and a **rewrite tier** (configurable cloud or large-local model — pedagogically rich final author). The two tiers are separated by an **inter-tier validation** seam that runs deterministic gates over the outline output before authorising the rewrite call. A symmetric **post-rewrite validation** seam runs the same gate chain against the rewrite-tier HTML emit before packaging.

Master gate: `COURSEFORGE_TWO_PASS=true` (opt-in, default off). When unset, the legacy single-pass `content_generation` workflow phase runs unchanged. When set, the workflow splits into:

```
content_generation_outline → inter_tier_validation → content_generation_rewrite → post_rewrite_validation
```

### Cross-links

- `Courseforge/router/router.py::CourseforgeRouter` — orchestrator. Public dispatch methods: `route_all(blocks)` (single outline candidate per block) and `route_with_self_consistency(block, ...)` (per-block multi-sample outline draft with validator-driven regen budget). When `n_candidates > 1`, `route_all` dispatches each block through `route_with_self_consistency`. `route_rewrite_with_remediation` is the rewrite-tier analogue.
- `Courseforge/generators/_outline_provider.py::OutlineProvider` — defaults: `local` provider, `qwen2.5:7b-instruct-q4_K_M` model, JSON-mode + lenient parse + grammar-aware backend payload (GBNF / JSON-Schema / vLLM guided / `format: json`). Both tiers now accept the full registry-superset provider set (the base's `_default_supported_providers()` — every `kind: openai_compatible` row in `config/endpoints.yaml` plus each tier's non-registry tags), so a `block_routing.yaml` row pinning e.g. `provider: groq` constructs cleanly instead of raising `ValueError` at the tier constructor.
- `Courseforge/generators/_rewrite_provider.py::RewriteProvider` — defaults: `anthropic` provider, `claude-sonnet-4-6` model. Per-block-type pinning via `block_routing.yaml` (e.g. `assessment_item` always rewrite-tier Anthropic, `flip_card_grid` local); the registry-superset provider set above applies here too.
- `Courseforge/router/policy.py::load_block_routing_policy` — loader + resolver. Resolution priority (high → low): per-call kwargs > `block_routing.yaml` > tier-default env vars (`COURSEFORGE_OUTLINE_*` / `COURSEFORGE_REWRITE_*`) > hardcoded defaults table (`DEFAULT_BLOCK_ROUTING`).
- `Courseforge/router/inter_tier_gates.py` — four shape-discriminating Block validators wired at both the inter-tier and post-rewrite seams: `BlockCurieAnchoringValidator`, `BlockContentTypeValidator`, `BlockPageObjectivesValidator`, `BlockSourceRefValidator`. Each `Block*Validator` dispatches on `isinstance(block.content, dict | str)`: dict path validates outline-tier dicts; str path strips HTML and validates rewrite-tier output through the same chain. Each emits `GateResult` with an `action` field (`regenerate` | `block` | `escalate` | `None`).
- `Courseforge/router/remediation.py` — builds a structured suffix from a failed `GateResult` and the block's prior validation history. The suffix is appended to the outline / rewrite prompt on the next regen iteration so the model sees concrete failure signals instead of a blind retry.
- Workflow definition: `config/workflows.yaml::textbook_to_course` (and `course_generation`). The `post_rewrite_validation` phase mirrors `inter_tier_validation` against the rewrite-tier `blocks_final_path`. Gates: `rewrite_curie_anchoring`, `rewrite_content_type`, `rewrite_page_objectives`, `rewrite_source_refs`, plus the statistical-tier gates listed below.

### Block-routing config

Optional per-block-type override file (`Courseforge/config/block_routing.yaml`); missing or empty file is the supported "env-vars + defaults only" mode. Schema: `schemas/courseforge/block_routing.schema.json` (Draft 2020-12, `additionalProperties: false`). Override the path via `COURSEFORGE_BLOCK_ROUTING_PATH`.

```yaml
version: 1
defaults:
  outline:
    provider: local
    model: qwen2.5:7b-instruct-q4_K_M
  rewrite:
    provider: anthropic
    model: claude-sonnet-4-6
blocks:
  assessment_item:
    rewrite:
      provider: anthropic
      model: claude-sonnet-4-6
  prereq_set:
    escalate_immediately: true   # skip outline; rewrite authors from scratch
```

### Block fields driving the router

- `validation_attempts: int` — incremented every time the validator chain fires `action="regenerate"`. The router caps the loop at `COURSEFORGE_OUTLINE_REGEN_BUDGET` / `COURSEFORGE_REWRITE_REGEN_BUDGET` (defaults `10` / `10`); per-block-type override via `regen_budget_rewrite` in `block_routing.yaml`.
- `escalation_marker: Optional[str]` — one of the canonical values in the `_ESCALATION_MARKERS` frozenset:
  - `outline_budget_exhausted` — regen budget hit OR `escalate_immediately: true` policy short-circuit fired (provenance carried via `Touch.purpose="escalate_immediately"`).
  - `structural_unfixable` — a validator returned `action="block"`.
  - `validator_consensus_fail` — every self-consistency candidate failed validation; surviving best-effort candidate carries this marker. Reused at the rewrite seam when the rewrite-tier regen budget runs out.
  - `outline_dispatch_error` / `rewrite_dispatch_error` — outline / rewrite tier dispatch raised an exception (network / provider raise / unhandled exception). Block is preserved at its original index so the IMSCC W5 filter catches the marker; postmortems can tell dispatch failures apart from semantic exhaustion.
  - `per_claim_attribution_unfixable` — Wave 1.5 W1.5.C: outline-tier regen budget exhausted purely on per-claim source-attribution misses (`OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS`) with no block-level structural miss across the regen chain. The rewrite-tier prompt-builder reads this marker via `_ESCALATION_MARKER_CONTEXT` and treats the per-claim citation map as advisory rather than authoritative; preserve block-level `source_refs[]` grounding instead.
  - `input_prompt_truncated` — rewrite-overflow-fix-2026-06: the rewrite-tier input-truncation tripwire (`ED4ALL_REWRITE_TRUNCATION_TRIPWIRE`, default ON) detected the served context window silently truncated the prompt HEAD (the server-reported `usage.prompt_tokens` fell far below the local estimate), so the system-prompt authoring CONTRACT was dropped and the model authored ungrounded. HARD, NON-RETRYABLE — the block is stamped + short-circuited (re-dispatching the same over-window prompt re-truncates) rather than retried. Surfaces as `escalated`; routes through the W5 packager-side escalation filter. Operator fix: raise the served window / `ED4ALL_REWRITE_NUM_CTX`, or enable `ED4ALL_REWRITE_FIT_WINDOW` to shrink the prompt to fit.
  - `rewrite_scaffold_overflow` — rewrite-overflow-fix-2026-07: the whole-prompt fit-window budget (`ED4ALL_REWRITE_FIT_WINDOW` on) found the NON-CHUNK scaffold alone (trimmed system prompt + outline dict + per-claim + objectives + contract) already exceeds the served window (`sys + scaffold + reserve ≥ ED4ALL_REWRITE_NUM_CTX`), so no grounding chunk could fit. HARD, NON-RETRYABLE — the block is stamped + NEVER dispatched (authoring an over-window prompt would silently head-truncate the CONTRACT). Surfaces as `escalated`; routes through the W5 packager-side escalation filter. Operator fix: raise the served window / `ED4ALL_REWRITE_NUM_CTX`, or shrink the upstream outline payload.
  - `block_objective_undelivered` — the regen budget exhausted PURELY on block-objective delivery misses (the most recent `BlockObjectiveDeliveryValidator` chain carried only `BLOCK_OBJECTIVE_*` codes, with no upstream structural miss), so the router stamps this instead of the generic `validator_consensus_fail`. The surviving best-effort candidate is additionally stamped `objective_alignment[*].status="unverifiable"` so the JSON-LD audit trail records the unverifiable delivery state for every declared `objective_id`; the rewrite-tier prompt-builder reads the marker via `_ESCALATION_MARKER_CONTEXT`.
  - `best_of_n_no_clean_candidate` — best-of-N selection: no candidate cleared BOTH objective-coverage AND zero-contradiction among the validator-passing samples, so the entailment-argmax selector took the highest-entailment passing candidate and stamped this marker rather than fabricating a clean winner. The block still ships (it passed the validator chain); the marker tells a postmortem the NLI selection had no clean pick.

Legacy validators returning `action=None, passed=False` retain regenerate-loop semantics; only EXPLICIT `action="block"` / `action="escalate"` triggers a short-circuit.

### Touch chain

`_TOUCH_TIERS` is `{outline, validation, rewrite, outline_val, rewrite_val}` (`validation` is the legacy validator-driven retouch value). The canonical post-validation Touch chain on a clean two-pass run is `outline → outline_val → rewrite → rewrite_val`. JSON-LD + SHACL Touch.tier enums carry the same five values so downstream consumers (Trainforge chunk extraction, training-data export) can filter on tier without string matching.

### Statistical-tier validators

Layered on top of the structural seam (CURIE / content_type / page_objectives / source_refs), the statistical tier catches semantic drift — output that parses cleanly but says the wrong thing. Wired symmetrically at both `inter_tier_validation` and `post_rewrite_validation`. Default thresholds in `docs/validation/validators.md`.

- `lib/validators/objective_assessment_similarity.py::ObjectiveAssessmentSimilarityValidator` — cosine-similarity floor between assessment-item stem and referenced LO text.
- `lib/validators/concept_example_similarity.py::ConceptExampleSimilarityValidator` — cosine-similarity floor between concept definition and illustrating example.
- `lib/validators/objective_roundtrip_similarity.py::ObjectiveRoundtripSimilarityValidator` — cosine-similarity floor between rewrite-tier LO paraphrase and source objective.
- `lib/validators/courseforge_outline_shacl.py::CourseforgeOutlineShaclValidator` — wrapper around `schemas/context/courseforge_v1.shacl-rules.ttl` shape constraints, applied to outline-tier Block emit.
- `lib/validators/bloom/classifier_disagreement.py::BloomClassifierDisagreementValidator` (the old `lib/validators/bloom_classifier_disagreement.py` path is a back-compat re-export shim that emits a `PendingDeprecationWarning`) — wraps `lib/classifiers/bloom_bert_ensemble.py::BloomBertEnsemble`, three SHA-pinned HuggingFace classifiers voting on Bloom level. Fires `action="regenerate"` on (a) ensemble majority disagrees with declared `bloomLevel` (`bert_ensemble_disagreement` event) or (b) ensemble dispersion above `_DISPERSION_THRESHOLD = 0.7` (`bert_ensemble_dispersion_high` event). Under `ED4ALL_BLOOM_TRIVOTE` (default off) the gate is re-founded on three interpretable voters instead — the generator's own asserted `bloom_level`, zero-shot DeBERTa entailment, and the deterministic verb level from `lib/ontology/bloom.py` — and the ensemble's first two members are never loaded. See `docs/validation/validators.md` for the ensemble member list and `docs/operations/behavior-flags.md` for the trivote flag.

The embedding wrapper at `lib/embedding/` degrades gracefully when the optional `[embedding]` extras are absent (warning-severity `EMBEDDING_DEPS_MISSING` GateIssue, `passed=True, action=None`); set `TRAINFORGE_REQUIRE_EMBEDDINGS=true` to fail-closed in production.

### Decision-capture events

Per-tier and per-decision; every router-side choice and every LLM call lands as a typed event:

- `block_outline_call` (per outline-tier LLM call, emitted by `OutlineProvider`).
- `block_rewrite_call` (per rewrite-tier LLM call, emitted by `RewriteProvider`).
- `block_validation_action` (per validator-chain run, emitted by `CourseforgeRouter`). Carries a `tier` field (`"outline"` | `"rewrite"`) disambiguating the two seams.
- `block_escalation` (per terminal escalation: budget exhausted, structural-unfixable, consensus failure).
- `block_best_of_n_selection` (W5 — per best-of-N winner selection under the `entailment_argmax` selector, emitted by `CourseforgeRouter._emit_best_of_n_decision`; rationale interpolates the per-candidate entailment spread + chosen index + tier/temp/model).
- `statistical_validation_pass` / `statistical_validation_fail` (per statistical-tier gate run).
- `bert_ensemble_disagreement` / `bert_ensemble_dispersion_high` / `bert_ensemble_member_loaded`.

Phase enum values used in capture paths: `courseforge-content-generator-outline`, `courseforge-content-generator-rewrite`, `courseforge-post-rewrite-validation`.

---

## ABCD authorship + concept extraction

ABCD-framework authorship attaches discrete `audience` / `behavior` / `condition` / `degree` fields to every synthesized learning objective, with verb-Bloom alignment gated at the `course_planning` phase. A standalone `concept_extraction` phase runs AFTER `course_planning` (phase-ordering fix, Option A1): `course_planning` mints `synthesized_objectives.json` and back-fills chunk `learning_outcome_refs[]` first, then `concept_extraction` builds the concept graph with the LO-driven typed-edge rules (`prerequisite_from_lo_order`, `targets_concept_from_lo`) available on fresh runs, runs the deterministic concept-objective linker to populate `LearningObjective.keyConcepts[]`, and writes the enriched key_concepts back into `synthesized_objectives.json`. (Pre-fix, `concept_extraction` ran between `source_mapping` and `course_planning`, so fresh runs without `--reuse-objectives` had no objectives at graph-build time and produced a degraded `related_to`-dominated graph.)

- `schemas/knowledge/courseforge_jsonld_v1.schema.json::$defs.AbcdObjective` — canonical ABCD shape: `{audience: str, behavior: {verb: str, action_object: str}, condition: str, degree: str}`. All four required when `abcd` is present. Referenced from `$defs.LearningObjective.properties.abcd` as an optional pointer.
- `lib/ontology/learning_objectives.py::BLOOMS_VERBS` — `Dict[str, FrozenSet[str]]` keyed on the canonical six Bloom levels. Single source of truth for the verb-Bloom alignment check.
- `lib/ontology/learning_objectives.py::compose_abcd_prose` — deterministic prose composer. Format: `"{Audience} will {verb} {action_object} {condition}, {degree}."`
- `lib/validators/abcd_objective.py::AbcdObjectiveValidator` — `abcd_verb_alignment` gate at `course_planning`. For each LO with `abcd` present, asserts `abcd.behavior.verb.lower() in BLOOMS_VERBS[lo.bloom_level]`. Emits `decision_type="abcd_authored"` on pass; `code="ABCD_VERB_BLOOM_MISMATCH"` + `action="regenerate"` on miss. Legacy LOs without `abcd` skip the check (warning-severity `ABCD_MISSING`).
- `lib/validators/concept_graph.py::ConceptGraphValidator` — `concept_graph` gate at `concept_extraction`. Gates on (a) ≥10 concept nodes, (b) ≥5 edge types present, (c) every node carries a `class` field, (d) every edge carries a `relation_type` field. Optional per-edge provenance when `TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE=true`.
- `MCP/tools/pipeline_tools.py::_run_concept_extraction` — phase handler. Reads staged SemantiK chunks via `Trainforge.chunker.chunk_content`, invokes `Trainforge.pedagogy_graph_builder.build_pedagogy_graph`, persists the graph to `LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json`, computes `concept_graph_sha256`, routes the hash through `phase_outputs.concept_extraction.concept_graph_sha256`.
- `lib/ontology/concept_objective_linker.py::link_concepts_to_objectives` — invoked from `_plan_course_structure` after objective synthesis. Two-stage match: (1) substring match between concept-graph node slugs and the LO's existing `keyConcepts[]`; (2) for unmatched nodes, scan the LO statement for verbatim concept-slug occurrence.

---

## Three-stage textbook synthesis

An operator-locked, large-LLM, three-stage synthesis architecture mapped onto the three existing post-conversion phases of `config/workflows.yaml::textbook_to_course`. It lifts course outline, domain-concept vocabulary, and per-chapter objective authoring off the deterministic small-model paths and onto a single configurable large LLM. The deterministic paths remain the canonical no-LLM fallback. Full design: `plans/textbook-llm-synthesis-3stage-2026-05.md`.

The three stages (pipeline order is Stage 1 → Stage 3 → Stage 2 + reconcile — the passes are independent):

- **Stage 1 — outline synthesis** (`objective_extraction`, handler `_extract_textbook_structure`): one call over the whole-textbook chapter/section skeleton → a course-level `semantic_outline` + a set of DRAFT `TO-NN` terminal objectives. Folds three optional top-level keys into `textbook_structure.json` (`semantic_outline`, `draft_terminal_objectives`, `structure_enrichment`).
- **Stage 3 — domain-concept vocabulary** (`concept_extraction`, handler `_run_concept_extraction`): N per-chapter calls (batched ≤10) fed each chapter's full `chapter_text` → a course-level domain-concept vocabulary. The vocabulary is compiled into `domain_concept_seeds` and used to re-tag the loaded SemantiK chunks in-memory before the co-occurrence graph build, so the semantic graph finally carries real `DomainConcept` nodes. Persists `domain_concept_vocabulary.json` as a sibling of `concept_graph_semantic.json`.
- **Stage 2 — per-chapter objective synthesis + reconciliation** (`course_planning`, handler `_plan_course_structure`): N per-chapter calls → that chapter's `CO-NN` objectives, then one reconciliation call that re-authors the Stage-1 draft `TO-NN` against the synthesized `CO-NN` set. Replaces the deterministic synthesizer when the flag is set.

Provider: `Courseforge/generators/_textbook_synthesis_provider.py::TextbookSynthesisProvider` — a single `_BaseLLMProvider` subclass exposing `synthesize_outline` / `synthesize_concepts` / `synthesize_chapter_objectives` / `reconcile_terminal_objectives`. Selected by the `TEXTBOOK_SYNTHESIS_PROVIDER` / `TEXTBOOK_SYNTHESIS_MODEL` env vars (see § Opt-In Behavior Flags). **Default-off:** when `TEXTBOOK_SYNTHESIS_PROVIDER` is unset, no provider is constructed and all three phases run their deterministic fallbacks — Stage 1 emits `textbook_structure.json` byte-identical, Stage 3 runs with empty seeds, Stage 2 runs `synthesize_objectives_from_topics` with no reconciliation. Stage 1 is a single call → fail-loud on parse-retry exhaustion; Stages 2/3 are N per-chapter calls → fail-soft per-chapter (a single bad chapter degrades, others are kept; all-N-fail falls back to the deterministic floor). Three warning-severity gates audit the per-stage output: `textbook_outline_enrichment`, `domain_concept_vocabulary`, `chapter_objective_coverage` (see `docs/validation/gates.md`).

### Dynamic CURIE minting

`BlockCurieAnchoringValidator` requires every outline/rewrite Block to carry a non-empty, text-anchored `content["curies"]`. RDF/SHACL corpora satisfy this for free (prose literally contains `sh:path` etc.); a prose corpus (math, history, K-12) has zero RDF CURIEs, so the gate would 100%-fail. Dynamic minting closes that gap by deriving a per-course CURIE namespace from the Stage-3 `domain_concept_vocabulary.json`:

- **Helpers** — `lib/ontology/curie_discovery.py::mint_curie_prefix` / `curie_for_concept` / `build_minted_curie_map`. A minted CURIE is `{prefix}:{localname}` where `prefix` is a grammar-valid lowercase course abbreviation and `localname` is the concept slug with hyphens→underscores. The local name satisfies the *intersection* of the two CURIE grammars in the codebase — the outline schema `_CURIE_PATTERN` (allows hyphens) and the canonical `CURIE_REGEX` in `lib/ontology/curie_extraction.py` (does not) — so a minted CURIE round-trips through `extract_curies` intact.
- **Outline tier** — `_run_content_generation_outline` mints CURIEs onto every outline block whose `content["curies"]` is empty, matching the block's `key_claims` text against the vocabulary via `extract_concept_tags`. Per-block `curie_minting` decision event.
- **Validator** — `BlockCurieAnchoringValidator` accepts an optional `minted_curie_map` (threaded by `MCP/hardening/gate_input_routing.py` + the validation handlers). A minted CURIE anchors when any of its vocabulary *surface forms* appears in the block text — the concept is genuinely discussed — instead of requiring the synthetic CURIE token literally. RDF CURIEs keep the literal-token check.
- **Rewrite tier** — `RewriteProvider` force-injects any still-missing CURIE as an appended `<span hidden>` carrying the CURIE tokens as text content (not an attribute — the str-path validator strips tags before scraping, so attribute-borne CURIEs would be invisible), instead of raising on remediation-budget exhaustion, so minted CURIEs survive into the published HTML.

**Backward-compat:** the absence of `domain_concept_vocabulary.json` is the gate — RDF/legacy corpora and every existing fixture run byte-identical; no behavior flag. When `minted_curie_map` is not supplied the validator runs legacy literal-token anchoring unchanged.

---

## Chunkset architecture

Two provenance-anchored chunk surfaces emit per course: a **SemantiK chunkset** (rooted in the textbook PDF) before objective extraction, and a **IMSCC chunkset** (rooted in the packaged IMSCC) post-packaging. The canonical chunker at `Trainforge/chunker/` is shared by SemantiK, Courseforge, and Trainforge.

- **SemantiK chunkset**: `chunking` workflow phase (between `staging` and `objective_extraction`). Agent: `semantik-chunker` (utility-style, no LLM dispatch; emits via `Trainforge.chunker.chunk_content`). Persists `LibV2/courses/<slug>/semantik_chunks/chunks.jsonl` + sibling `manifest.json`. Emits `semantik_chunks_path` + `semantik_chunks_sha256` through `phase_outputs.chunking`.
- **IMSCC chunkset**: `imscc_chunking` workflow phase (between `packaging` and `training_synthesis`). Helper: `MCP/tools/pipeline_tools.py::_run_imscc_chunking`. Reads HTML entries in-memory from the packaged `.imscc` zip via `zipfile.ZipFile`. Emits `chunkset_kind="imscc"` plus `source_imscc_sha256` (SHA-256 of the archive bytes). Persists at `LibV2/courses/<slug>/imscc_chunks/`.
- **Sidecar manifest schema**: `schemas/library/chunkset_manifest.schema.json`. Symmetric across the conversion and IMSCC chunksets: `chunkset_kind` enum (`"semantik"` — the live emit — plus a legacy pre-SemantiK value accepted dual-read for old archives, and `"imscc"`) discriminator + conditional source-SHA requirement (`source_semantik_html_sha256` for `semantik`; `source_imscc_sha256` for IMSCC). Required: `chunks_sha256`, `chunker_version` (resolved from `Trainforge.chunker.CHUNKER_SCHEMA_VERSION`), `chunkset_kind`. Optional: `chunks_count`, `generated_at`.
- **Chunkset-manifest gate**: `lib/validators/chunkset_manifest.py::ChunksetManifestValidator` fires at both chunking phases. Verifies manifest existence + schema + `chunks_sha256` round-trip + `chunker_version` match + conditional source-SHA. GateIssue codes: `MANIFEST_MISSING`, `MANIFEST_PARSE_ERROR`, `MANIFEST_SCHEMA_INVALID`, `CHUNKS_SHA256_MISMATCH`, `CHUNKER_VERSION_MISMATCH`, `SOURCE_SHA256_MISSING`.
- **Course-manifest hash triangle**: `lib/validators/libv2_manifest.py::LibV2ManifestValidator` fail-closes at `libv2_archival` on any of three required hashes missing, malformed, or divergent: `semantik_chunks_sha256`, `imscc_chunks_sha256`, `concept_graph_sha256`. Each fires a `MISSING_*` / `INVALID_*` / `*_HASH_MISMATCH` GateIssue triplet.
- **Backfill for legacy archives**: a backfill script under `LibV2/tools/libv2/scripts/` migrates pre-chunkset archives (no `semantik_chunks/` directory). Idempotent by default; `--force` for re-emit, `--dry-run` for plan-only.

---

## Operator stage subcommands

Four operator-facing subcommands re-drive the Courseforge two-pass pipeline one tier at a time without re-executing the upstream `semantik_conversion → staging → chunking → objective_extraction → source_mapping → course_planning → concept_extraction` chain. Use case: a previous full run produced an OUTLINE_DIR; the operator wants to re-run only the rewrite tier under a different teacher model, re-run validation after tweaking a gate threshold, or A/B-test outline-tier model swaps.

The four subcommands route through the canonical `textbook_to_course` workflow with the `courseforge_stage` workflow param set; the workflow runner pre-populates upstream phase outputs via `_synthesize_outline_output` and skips non-whitelisted phases via `_should_skip_phase`:

| Subcommand | Active phases (executed) | Skipped via whitelist |
|---|---|---|
| `courseforge-outline` | `content_generation_outline` | inter_tier_validation, content_generation_rewrite, post_rewrite_validation |
| `courseforge-validate` | `inter_tier_validation`, `post_rewrite_validation` | content_generation_outline, content_generation_rewrite |
| `courseforge-rewrite` | `content_generation_rewrite`, `post_rewrite_validation` | content_generation_outline, inter_tier_validation |
| `courseforge` | all four | (none — full two-pass slice) |

Pre-Courseforge phases pre-populate from the project export root via `_synthesize_outline_output`; their `_completed=True` markers fire the runner's already-completed skip path. Post-Courseforge phases (packaging, imscc_chunking, trainforge_assessment, training_synthesis, libv2_archival, finalization) skip via the `courseforge_stage` whitelist regardless of which subcommand fired — Phase 5 is intentionally scoped to the Courseforge two-pass surface only. Operators who want to re-run a post-Courseforge phase use the canonical `ed4all run textbook-to-course` entry point.

### CLI flags (at `cli/commands/run.py`)

- `--blocks <comma-separated>` — per-block-TYPE re-execution scope, wired end-to-end (CLI `_parse_blocks_filter` → `target_block_ids` workflow param → `content_generation_rewrite` routing → `_run_content_generation_rewrite`). Tokens must come from the canonical `BLOCK_TYPES` enum (`Courseforge/scripts/blocks.py`); unknown tokens fail fast at parse time. Semantics are an ADDITIVE eviction over the rewrite tier's default failure-driven reuse: normally the tier reuses `blocks_final.jsonl`-cached successful rewrites and re-rolls only failed/degraded blocks. When `target_block_ids` is set, every cached block whose `block_type` is in the set is also evicted (`_evict_rewrite_cache_by_block_type`) so it re-rolls this pass even after a prior success; blocks of every other type keep byte-identical cache reuse. Unset → byte-identical failure-driven reuse. The GUI failure panel's `blocks` CSV option is normalized into the same param (`run_service._normalize_blocks_param`). Validate-tier subcommands ignore `--blocks`. Dry-run plan annotates the rewrite phase with `<FILTERED:assessment_item,...>`.
- `--block-ids <comma-separated>` (I4 stage 2) — instance-scoped re-execution scope, wired end-to-end (CLI `_parse_csv_tokens` → `target_block_instance_ids` workflow param → `content_generation_rewrite` routing → `_run_content_generation_rewrite`). Tokens are EXACT block-instance IDs as they appear in the outline / `blocks_final.jsonl` (shape `{page_id}#{block_type}_{slug}_{idx}`). ADDITIVE over `--blocks`: only the named instance(s) are evicted from the rewrite failure-driven cache (`_evict_rewrite_cache_by_block_id`) so they re-author this pass even after a prior success; every out-of-scope block keeps byte-identical cache reuse. Consumed by the rewrite tier only. An id carried by no outline block fails the rewrite phase LOUDLY (validated against the outline block universe, so a typo is caught even on a first pass with no cache on disk) — never a silent no-op.
- `--pages <comma-separated>` (I4 stage 2) — page/module-scoped re-execution scope, wired end-to-end (CLI `_parse_csv_tokens` → `target_page_ids` workflow param → `content_generation_rewrite` routing → `_run_content_generation_rewrite`). Tokens are an exact `page_id` (one page, e.g. `week_01_content_02`) OR a module prefix (a whole week/module, e.g. `week_01`); matching is `_page_membership_match`. ADDITIVE over `--blocks` and `--block-ids`: every block on a matched page/module is evicted (`_evict_rewrite_cache_by_page`) and re-authored; all other blocks stay byte-identical. Consumed by the rewrite tier only. A page token that matches no outline block fails the rewrite phase LOUDLY — never a silent no-op.
- GUI failure panel: the studio failure panel's `block_ids` and `pages` options normalize into these same `target_block_instance_ids` / `target_page_ids` params (alongside `blocks` → `target_block_ids`) via `run_service._normalize_blocks_param`. The panel now renders a **per-page / per-block picker** (`create.js::renderRewritePicker`) that posts these as raw JSON **arrays** (CSV strings are still accepted for API callers). Server-side, `_normalize_blocks_param` normalizes either shape and applies a subsumption dedup — a `block_ids` entry whose page a selected `pages` token already covers (same `_page_membership_match` rule, mirrored as `run_service._page_covers` with a parity test) is dropped, since the page eviction already re-authors it.
- `--force` — re-run phases despite a pre-existing `_completed` checkpoint. The synthesizer pre-populates upstream phases with `_completed=True`; `--force` flips that to `False` so the phase loop re-executes them.

### `02_validation_report/report.json` writer

The `_run_inter_tier_validation` and `_run_post_rewrite_validation` phase helpers emit JSONL only — `blocks_validated.jsonl` + `blocks_failed.jsonl` next to the consumed Block file. The operator-facing structured per-block summary lives at:

- `<project_root>/02_validation_report/report.json` for the outline tier's `inter_tier_validation` phase emit.
- `<project_root>/04_rewrite/02_validation_report/report.json` for the rewrite tier's `post_rewrite_validation` phase emit.

The writer fires automatically after each validation phase completes inside `WorkflowRunner.run_workflow` (best-effort — filesystem failures are warning-logged and don't abort the run). Schema (`MCP/core/workflow_runner.py::WorkflowRunner._VALIDATION_REPORT_SCHEMA_VERSION = "v2"`, emitted by `_write_validation_report`):

```json
{
  "run_id": "<workflow_id>",
  "phase": "inter_tier_validation",
  "schema_version": "v2",
  "total_blocks": 0,
  "passed": 0,
  "failed": 0,
  "escalated": 0,
  "curie_force_injected": 0,
  "per_block": [
    {
      "block_id": "<id>",
      "block_type": "assessment_item",
      "page": "<page_id|null>",
      "week": 4,
      "status": "passed|failed|escalated",
      "gate_results": [
        {"gate_id": "<id>", "action": "<action|null>", "passed": false, "issue_count": 2}
      ],
      "escalation_marker": "<marker|null>",
      "curie_force_injected": true
    }
  ],
  "phase_level_gate_results": [
    {
      "gate_id": "<id>",
      "action": "<action|null>",
      "passed": false,
      "issue_count": 0,
      "unattributed_issue_count": 0
    }
  ]
}
```

Field semantics that are load-bearing for downstream readers (the calibration harness in particular):

- **Per-block `issue_count` is per-block** (the v2 fix). Each `GateResult.issues[]` carries a `location`; the writer builds a `gate_id -> Counter(location)` map once, then a block's `gate_results[].issue_count` counts ONLY the issues whose `location` equals THAT `block_id`. Schema v1 attached the same phase-level count to every block, smearing course-wide totals and inflating per-gate fire-rates toward 100%. The per-gate `passed` stays PHASE-level — a gate either passed or failed for the whole phase.
- **`phase_level_gate_results` (v2 addition)** preserves the structural issues that belong to no single block. `location` for these is an objective id (`triangle_completeness` → `CO-01`), a module/page id (`retrieval_presence` → a `page_id`), or `None`. Per gate it carries the TOTAL `issue_count` plus `unattributed_issue_count` (issues whose `location` matched no `block_id`).
- **`curie_force_injected`** — the top-level count plus a per-block boolean marks blocks that passed `rewrite_curie_anchoring` only because `RewriteProvider` force-injected their CURIE anchoring after the rewrite LLM exhausted its remediation budget. `status` stays `passed` (the appended hidden span legitimately anchors them); the flag exists so operators can quantify that class instead of reading those blocks as clean rewrites. The per-block flag is emitted only when `true`, so clean entries stay byte-stable.

Blocks with non-null `escalation_marker` count as `escalated` rather than `failed`.

The stage-subcommand contract is **read-only against upstream outputs** — the synthesizer never re-writes pre-Courseforge artifacts; it only reads them off disk to populate the in-memory `phase_outputs` dict. A failed stage subcommand can be retried as many times as needed without contaminating the upstream chain.

---

## Template Components

Content generators should incorporate components from the expanded template library.

### Layout Components (`templates/component/`)
| Component | Template | Use Case |
|-----------|----------|----------|
| Accordion | `accordion_template.html` | FAQ, expandable definitions, progressive disclosure |
| Tabs | `tabs_template.html` | Section organization, resource grouping |
| Card Layout | `card_layout_template.html` | Content grids, feature highlights |
| Flip Card | `flip_card_template.html` | Term/definition, before/after reveals |
| Timeline | `timeline_template.html` | Sequential processes, chronological content |
| Progress Indicator | `progress_indicator_template.html` | Module progress bars, step indicators |
| Callout | `callout_template.html` | Info/warning/success/danger alerts |

### Interactive Components (`templates/interactive/`)
| Component | Template | Use Case |
|-----------|----------|----------|
| Self-Check | `self_check_template.html` | Quick formative assessment with feedback |
| Reveal Content | `reveal_content_template.html` | Click-to-reveal answers, spoilers |
| Inline Quiz | `inline_quiz_template.html` | Multi-question embedded assessments |

### Accessibility Themes (`templates/theme/`)
| Theme | File | Description |
|-------|------|-------------|
| High Contrast | `color_schemes/high_contrast.css` | WCAG AAA (7:1+) override |
| Dyslexia-Friendly | `typography/dyslexia_friendly.css` | Optimized reading typography |

### CSS Foundation
- Base variables: `templates/_base/variables.css`
- Official color palette integrated across all templates
- Bootstrap 4.3.1 compatible

---

## Intake & Remediation Workflow

### Supported IMSCC Sources
Courseforge can import and remediate IMSCC packages from:
- **Brightspace/D2L** - Detected via `d2l_2p0` namespace
- **Canvas** - Detected via `canvas.instructure` namespace
- **Blackboard** - Detected via `blackboard.com` namespace
- **Moodle** - Detected via `moodle.org` namespace
- **Sakai** - Detected via `sakaiproject.org` namespace
- **Generic IMSCC** - Standard IMS CC 1.1/1.2/1.3

### Intake Workflow Steps
```
1. Place IMSCC package in: inputs/existing-packages/
2. Invoke imscc-intake-parser agent
3. Agent extracts, detects source LMS, inventories content
4. content-analyzer identifies remediation needs
5. Parallel remediation:
   - semantik-automation-coordinator: PDFs/Office → accessible HTML
   - accessibility-remediation: WCAG 2.2 AA fixes
   - content-quality-remediation: Educational enhancements
   - intelligent-design-mapper: Interactive component styling
6. remediation-validator: Final quality validation
7. brightspace-packager: Generate improved IMSCC
```

### Remediation Capabilities
| Capability | Target |
|------------|--------|
| PDF Conversion | 100% to accessible HTML via the SemantiK v2 cascade (`semantik_conversion` phase) |
| Office Documents | 100% to accessible HTML via the SemantiK v2 cascade (`semantik_conversion` phase) |
| Alt Text | AI-generated for all images |
| Heading Structure | Automatic hierarchy correction |
| Color Contrast | WCAG AA (4.5:1 minimum) |
| Keyboard Navigation | Full accessibility |
| Component Styling | AI-selected interactive elements |
| Quality Enhancement | Learning objectives, summaries, checks |

### Scripts for Course Generation
| Script | Location | Purpose |
|--------|----------|---------|
| `generate_course.py` | `scripts/rendering/` | Multi-file weekly course generation. Emits page-level JSON-LD, `course_metadata.json`, prerequisite-page refs, `data-cf-teaching-role`, and `data-cf-source-ids` / page-level `sourceReferences` when SemantiK source material is staged. Accepts `--emit-mode {full,outline}` (default `full`); outline mode strips content/example/assessment HTML bodies but preserves their JSON-LD `blocks[]` projections, and stamps `course_metadata.blocks_summary.outline_only=true` so downstream consumers can detect the tier. |
| `package_multifile_imscc.py` | `scripts/packaging/` | Packages multi-file output into IMSCC. Structural validation is on by default (per-week `learningObjectives` must resolve to the week's LO manifest). Auto-discovers `course.json` and bundles `course_metadata.json` at the zip root. Manifest uses IMS Common Cartridge v1.3 namespaces; resources are nested under per-week `<item>` wrappers in the organization tree. **This is the runtime target of the MCP `package_imscc` tool** — `MCP/tools/pipeline_tools.py::_package_imscc` imports and delegates here instead of hand-rolling a ZIP. Accepts `--outline-only` to package an outline-tier deliverable; reads `course_metadata.blocks_summary.outline_only` written by `generate_course.py --emit-mode outline`. |

`--emit-mode outline` (`generate_course.py`) and `--outline-only` (`package_multifile_imscc.py`) produce a stripped-down deliverable carrying only objectives + summaries; content/example/assessment HTML bodies are dropped while their JSON-LD `blocks[]` entries persist for downstream consumers (Trainforge `process_course.py` skips `instruction_pair` extraction when `course_metadata.blocks_summary.outline_only=true`). Outline mode is the input shape the two-pass pipeline expects from the outline tier.

### Scripts for Intake
| Script | Location | Purpose |
|--------|----------|---------|
| `imscc_extractor.py` | `scripts/imscc-extractor/` | Universal IMSCC parsing + source-LMS detection |
| `component_applier.py` | `scripts/component-applier/` | Interactive component application |
| `accessibility_validator.py` | `scripts/accessibility-validator/` | Accessibility checks over remediated HTML |
| `remediation_validator.py` | `scripts/remediation-validator/` | Final quality validation (`RemediationValidator` / `ValidationReport` / `ValidationSeverity`) |

These are library modules invoked by the remediation agents, not the gate suite.
Blocking quality enforcement lives in `config/workflows.yaml::validation_gates`
(`wcag_compliance`, `cartridge_conformance`, and the rest); the
`remediation-validator` *agent* itself routes to `get_courseforge_status`.

### Success Metrics

Design targets for the intake/remediation surface. These are goals used to scope
the work — they are **not** measurements, and no harness in this repo currently
reports against them. Treat any figure below as unvalidated until a named eval
produces it.

| Metric | Target |
|--------|--------|
| IMSCC import success | 95%+ (any source LMS) |
| WCAG compliance | 100% Level AA |
| SemantiK conversion | 98%+ for PDFs |
| Component accuracy | 90%+ appropriate selections |
