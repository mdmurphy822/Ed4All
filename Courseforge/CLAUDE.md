# Courseforge

AI-powered instructional design system that creates and remediates accessible, LMS-ready IMSCC course packages.

> **Universal Protocols**: See root `/CLAUDE.md` for orchestrator protocol, execution rules (ONE agent = ONE file, max 10 parallel), decision capture requirements, and error handling. This file contains Courseforge-specific guidance only.

---

## Quick Start

### Course Creation Mode
**Input**: Exam objectives (PDF/text) + optional DART-processed textbooks (HTML)
**Output**: Single IMSCC file ready for Brightspace import

### Course Intake/Remediation Mode (NEW)
**Input**: Any IMSCC package (Canvas, Blackboard, Moodle, Brightspace, etc.)
**Output**: Fully accessible, enhanced IMSCC with 100% WCAG 2.2 AA compliance

### Provider selection (Phase 1 ToS unblock)

Set `COURSEFORGE_PROVIDER=local` to route content authoring through a license-clean local OSS provider; see root `CLAUDE.md` § Opt-In Behavior Flags for the env-var contract and `docs/LICENSING.md` for the ToS posture.

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
(DART HTML)           requirements-      oscqr-            quality-assurance
                      collector          evaluator              (per batch)
```

### Pipeline 2: Intake & Remediation (NEW)
```
INPUT                         PROCESSING                                    OUTPUT
─────                         ──────────                                    ──────
Any IMSCC Package ──► imscc-intake-parser ──► content-analyzer ──┬──► dart-automation-coordinator
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
| `course-outliner` | Synthesize canonical `TO-NN` / `CO-NN` objectives from textbook structure; persist `synthesized_objectives.json`. Wave 24+ routes this agent to `plan_course_structure` (no longer just creates project dirs). | Creating course framework |
| `content-generator` | Educational content creation | Content development (1 file per agent) |
| `quality-assurance` | Pattern prevention and validation | Quality gates |
| `oscqr-course-evaluator` | Educational quality assessment | OSCQR evaluation |
| `brightspace-packager` | IMSCC package generation | Final deployment |
| `textbook-ingestor` | Textbook content processing | Entry point for textbook materials |
| `source-router` | Bind DART source blocks to Courseforge module pages | Source attribution for pipeline runs |

### Intake & Remediation Agents (NEW)
| Agent | Purpose | When to Use |
|-------|---------|-------------|
| `imscc-intake-parser` | Universal IMSCC package parsing | Importing existing courses |
| `content-analyzer` | Accessibility/quality gap detection | Analyzing imported content |
| `dart-automation-coordinator` | Automated DART conversion orchestration | Converting PDFs/Office docs to accessible HTML |
| `accessibility-remediation` | Automatic WCAG fixes | Fixing accessibility issues |
| `content-quality-remediation` | Educational depth enhancement | Improving thin content |
| `intelligent-design-mapper` | AI-driven component selection | Applying interactive styling |

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
│   └── getting-started.md       # Quick start guide
├── agents/                      # Agent specifications
├── inputs/                      # Input files
│   ├── exam-objectives/         # Certification exam PDFs/docs
│   ├── textbooks/               # DART-processed HTML textbooks
│   ├── existing-packages/       # IMSCC packages for intake (NEW)
│   └── existing-packages/       # IMSCC packages for intake
├── templates/                   # HTML templates and components
├── schemas/                     # IMSCC and content schemas
├── imscc-standards/             # Brightspace/IMSCC technical specs
├── scripts/                     # Automation scripts
│   ├── imscc-extractor/         # Universal IMSCC extraction
│   ├── component-applier/       # Interactive component application
│   └── remediation-validator/   # Final quality validation
├── exports/                     # Generated course packages
│   └── YYYYMMDD_HHMMSS_name/    # Timestamped project folders
└── runtime/                     # Agent workspaces (auto-created)
```

### Export Project Structure
```
exports/YYYYMMDD_HHMMSS_coursename/
├── 00_template_analysis/
├── 01_learning_objectives/
├── 02_course_planning/
├── 03_content_development/
│   ├── week_01/
│   └── week_XX/
├── 04_quality_validation/
├── 05_final_package/
├── agent_workspaces/
├── project_log.md
└── coursename.imscc              # Final deliverable
```

---

## Textbook Integration

Textbooks must be pre-processed through DART before use:

1. Run textbook PDF through DART (set `DART_PATH` environment variable):
   ```bash
   cd $DART_PATH
   python convert.py textbook.pdf -o /path/to/courseforge/inputs/textbooks/
   ```
2. DART produces WCAG 2.2 AA accessible HTML
3. Place output in `inputs/textbooks/`
4. Reference in course generation

---

## Quality Standards

### Pattern Prevention
See `docs/troubleshooting.md` for complete pattern list. Critical patterns:
- Schema/namespace consistency (IMS CC 1.1)
- Assessment XML format (QTI 1.2)
- Content completeness (all weeks substantive)
- Organization hierarchy (no empty structures)

### OSCQR Evaluation
Automatic quality assessment after course outline completion:
- 70% threshold for pre-development
- 90% threshold for pre-production
- 100% accessibility compliance required

---

## Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Troubleshooting | `docs/troubleshooting.md` | Error patterns and solutions |
| Workflow Reference | `docs/workflow-reference.md` | Detailed execution protocols |
| Getting Started | `docs/getting-started.md` | Quick start guide |
| Pattern Prevention | `docs/troubleshooting.md` | Error patterns and prevention |
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

## Metadata Output

Courseforge HTML pages embed machine-readable instructional design metadata for downstream consumption by Trainforge.

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
| `data-cf-source-ids` | `<section>`, headings, component wrappers | DART `sourceId`(s) that ground this block. Shape: `dart:{slug}#{block_id}`. Carried through from DART's `data-dart-block-id` when source material is present; elided when no source grounding exists. |
| `data-cf-source-primary` | `<section>`, headings, component wrappers | The primary `sourceId` for the block (subset of `data-cf-source-ids`) when one source dominates. |
| `data-cf-block-id` | every block-bearing wrapper (`<section>`, headings, component wrappers) | Stable Block ID for cross-referencing JSON-LD `blocks[]` (Phase 2; gated behind `COURSEFORGE_EMIT_BLOCKS`). Shape: `{page_id}#{block_type}_{slug}_{idx}` per `Courseforge/scripts/blocks.py::Block.stable_id`. Elided when the flag is off. |

Attributes stop at the **section / component wrapper level** — never on every `<p>` / `<li>` / `<tr>` in prose.

### Wave 35: ancestor-walkable grounding

`ContentGroundingValidator` walks each non-trivial `<p>` / `<li>` /
`<figcaption>` / `<blockquote>`'s ancestor chain to find the first
`data-cf-source-ids` attribute. Three emit-side contracts keep that
walk passing:

1. **Content sections are wrapped in `<section data-cf-source-ids="…">`.**
   `Courseforge/scripts/generate_course.py::_render_content_sections`
   now wraps each h2/h3 + paragraph group in a `<section>` wrapper
   carrying the section's resolved source-ids. Pre-Wave-35 the
   attribute lived only on the `<h2>` (a DOM sibling of the `<p>`),
   which the validator's ancestor walk couldn't reach.
2. **`content_NN` pages inherit `content_01` grounding.**
   `_page_refs_for` falls back from `content_NN` → `content_01` in
   the `source_module_map`. The source-router only emits a single
   per-week `content_01` entry; every generated content page in that
   week shares the same DART source region, so the fallback is the
   correct grounding (not a workaround).
3. **Objectives `<section>` mirrors page-level source-ids.**
   `ensure_objectives_on_page` scans the page body for the first
   `<section data-cf-source-ids="…">` wrapper and stamps the same
   ids onto the injected objectives section. Long synthesized LO
   statements that exceed the 30-word non-trivial floor otherwise
   flagged as ungrounded under the ancestor walk.

DART-side slug contract (see `DART/CLAUDE.md`): the `dart:{slug}#{block_id}`
slug uses `lowercase + space-to-hyphen` normalization (not
`canonical_slug`'s underscore collapse), matching the validator's
`_resolve_valid_block_ids` rule.

### JSON-LD Structured Metadata

Each page includes a `<script type="application/ld+json">` block in `<head>` with:
- `learningObjectives`: ID (canonical `TO-NN` / `CO-NN`), statement, Bloom's level/verb, cognitive domain, assessment suggestions
- `sections`: Heading, content type, Bloom's range, key terms with definitions, optional per-section `sourceReferences`
- `misconceptions`: Common misconceptions with corrections
- `suggestedAssessmentTypes`: Recommended question formats
- `prerequisitePages`: Cross-page prerequisite refs
- `sourceReferences`: Optional page-level DART source references (canonical `{sourceId, role, weight?, confidence?, pages?, extractor?}` shape). Page-level JSON-LD `role` is authoritative (`primary` / `contributing` / `corroborating`) and takes precedence over attribute-level roles.

Canonical shape: `schemas/knowledge/courseforge_jsonld_v1.schema.json`. Context namespace: `https://ed4all.dev/ns/courseforge/v1`.

### Learning Objective IDs

Emitted LO IDs follow the pattern `^[A-Z]{2,}-\d{2,}$` from the canonical helper `lib/ontology/learning_objectives.py::mint_lo_id`:

- `TO-NN` — terminal (course-wide) objective.
- `CO-NN` — chapter-level objective.

Synthesized objectives are persisted to `{project}/01_learning_objectives/synthesized_objectives.json` by the `plan_course_structure` phase in the `textbook_to_course` pipeline. Downstream Trainforge consumers match case-insensitively; the `TRAINFORGE_PRESERVE_LO_CASE` flag preserves the emit case.

### Phase 2: intermediate Block format

`Courseforge/scripts/blocks.py` (`:223-265`) defines the canonical
intermediate `Block` dataclass that the renderer + JSON-LD builder
both consume. Every page-level pedagogical unit (objective, concept,
example, callout, flip card, self-check question, activity, …) is
constructed as a `Block` first, then projected to HTML via
`Block.to_html_attrs()` and to a JSON-LD entry via
`Block.to_jsonld_entry()`. The dataclass is frozen — mutations return
a new instance via `dataclasses.replace`, and the `with_touch`
helper appends to the immutable `touched_by` audit chain.

Field surface (mirrors the docstring at `blocks.py:223-265`):

```python
@dataclass(frozen=True)
class Block:
    block_id: str                    # stable: {page_id}#{block_type}_{slug}_{idx}
    block_type: str                  # one of BLOCK_TYPES (16 values)
    page_id: str
    sequence: int
    content: Union[str, Dict[str, Any]]
    template_type: Optional[str]     # e.g. flip_card, self_check
    key_terms: Tuple[str, ...]
    objective_ids: Tuple[str, ...]   # canonical TO-NN / CO-NN refs
    bloom_level: Optional[str]
    bloom_verb: Optional[str]
    bloom_range: Optional[str]       # section-level span (emit-only)
    bloom_levels: Tuple[str, ...]
    bloom_verbs: Tuple[str, ...]
    cognitive_domain: Optional[str]
    teaching_role: Optional[str]
    content_type_label: Optional[str]
    purpose: Optional[str]
    component: Optional[str]
    source_ids: Tuple[str, ...]      # DART dart:{slug}#{block_id} grounding
    source_primary: Optional[str]
    source_references: Tuple[Dict[str, Any], ...]
    touched_by: Tuple[Touch, ...]    # cumulative outline / validation / rewrite chain
    content_hash: Optional[str]
    validation_attempts: int         # Phase-3 feedback-driven; default 0
    escalation_marker: Optional[str] # one of {outline_budget_exhausted,
                                     # structural_unfixable, validator_consensus_fail}
```

The `BLOCK_TYPES` enum (`blocks.py:77-96`) is a 16-value frozenset:
`objective`, `concept`, `example`, `assessment_item`, `explanation`,
`prereq_set`, `activity`, `misconception`, `callout`,
`flip_card_grid`, `self_check_question`, `summary_takeaway`,
`reflection_prompt`, `discussion_prompt`, `chrome`, `recap`. The
`block_type` constructor argument is validated against this set;
unknown values raise `ValueError`.

When `COURSEFORGE_EMIT_BLOCKS=true`,
`Courseforge/scripts/generate_course.py::_build_page_metadata`
(`:2085-2098`) emits three additional top-level JSON-LD fields per
page beyond the legacy `learningObjectives` / `sections` /
`misconceptions` / `sourceReferences` shape:

- `blocks[]`: ordered array of per-block JSON-LD entries built by
  `Block.to_jsonld_entry()` — the canonical projection of the Block
  list for the page. Trainforge's `process_course._extract_section_metadata`
  (`Trainforge/process_course.py:2333-2341`) prefers this projection
  over the `data-cf-*` HTML-attribute fallback when present.
- `provenance`: `{runId, pipelineVersion, tiers[]}` — `runId` reads
  `COURSEFORGE_RUN_ID` from the environment, `pipelineVersion` is
  pinned to `phase2`, `tiers[]` is reserved for Phase 3's outline /
  validation / rewrite tier provenance.
- `contentHash`: SHA-256 hex of the meta dict canonicalised
  (`json.dumps(..., sort_keys=True, ensure_ascii=False)`) BEFORE the
  `contentHash` field itself is added. The hash excludes itself from
  its own payload, which keeps it deterministic across re-runs that
  produce identical content.

When the flag is off (default), the new fields are elided and emit
stays byte-identical to the pre-Phase-2 snapshot, which is the
contract the legacy regression suite (`Courseforge/scripts/tests/`)
pins. Canonical schema shape: `schemas/knowledge/courseforge_jsonld_v1.schema.json`
(`$defs.Block`, `$defs.Touch`, top-level optional `blocks[]` /
`provenance` / `contentHash`).

### Phase 3: outline-rewrite two-pass router

Phase 3 layers a two-tier router over the Phase 2 `Block` format. The
content-generator surface splits into an **outline tier** (small local
7B model — terse, structurally-shaped first draft) and a **rewrite
tier** (configurable cloud or large-local model — pedagogically rich
final author). The two tiers are separated by an **inter-tier
validation** seam that runs four deterministic gates over the outline
output before authorising the rewrite call. Master gate: feature flag
`COURSEFORGE_TWO_PASS=true` (opt-in, default off). When the flag is
unset the legacy single-pass `content_generation` workflow phase runs
unchanged; when it is set the workflow splits into
`content_generation_outline` → `inter_tier_validation` →
`content_generation_rewrite`.

Cross-links:

- `Courseforge/router/router.py::CourseforgeRouter` — orchestrator.
  Two public dispatch methods: `route_all(blocks)` (Wave N — single
  outline candidate per block, every successful outline proceeds to
  rewrite) and `route_with_self_consistency(block, ...)` (Wave N+1 —
  per-block multi-sample outline draft + validator-driven regen budget,
  with policy-driven candidate count and budget). `route_all` calls
  `route()` directly today; widening it to dispatch via
  `route_with_self_consistency` is a Phase 3 followup.
- `Courseforge/generators/_outline_provider.py::OutlineProvider` —
  outline-tier subclass of `_BaseLLMProvider`. Defaults: `local`
  provider, `qwen2.5:7b-instruct-q4_K_M` model, JSON-mode + lenient
  parse + grammar-aware backend payload (GBNF / JSON-Schema / vLLM
  guided / `format: json`).
- `Courseforge/generators/_rewrite_provider.py::RewriteProvider` —
  rewrite-tier subclass of `_BaseLLMProvider`. Defaults: `anthropic`
  provider, `claude-sonnet-4-6` model. Per-block-type pinning via
  `block_routing.yaml` (e.g. `assessment_item` always rewrite-tier
  Anthropic, `flip_card_grid` local).
- `Courseforge/router/policy.py::load_block_routing_policy` — loader
  + resolver. Resolution priority (high → low): per-call kwargs >
  `block_routing.yaml` > tier-default env vars
  (`COURSEFORGE_OUTLINE_*` / `COURSEFORGE_REWRITE_*`) > hardcoded
  defaults table (`DEFAULT_BLOCK_ROUTING`).
- `Courseforge/router/inter_tier_gates.py` — four Block-input
  validators wired into `config/workflows.yaml::inter_tier_validation`
  per Subtask 52: `BlockCurieAnchoringValidator`,
  `BlockContentTypeValidator`, `BlockPageObjectivesValidator`,
  `BlockSourceRefValidator`. Each emits `GateResult` with an
  `action` field (`regenerate` | `block` | `escalate` | `None`) that
  the router consumes to decide whether to retry, skip rewrite, or
  short-circuit to the failure pile.

Block-routing config (`Courseforge/config/block_routing.yaml`) is an
optional per-block-type override file; missing or empty file is the
supported "env-vars + defaults only" mode. Schema:
`schemas/courseforge/block_routing.schema.json` (Draft 2020-12,
`additionalProperties: false`). Override the path via
`COURSEFORGE_BLOCK_ROUTING_PATH`. Example:

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

Per-block Phase-3 fields on the `Block` dataclass
(`Courseforge/scripts/blocks.py:223-265`):

- `validation_attempts: int` — incremented every time the inter-tier
  validator chain fires `action="regenerate"` against the block. The
  router caps the loop at `COURSEFORGE_OUTLINE_REGEN_BUDGET` (default
  3, will bump to 10 in Phase 3.5). Once the cap is hit and the chain
  still rejects the block, it is stamped with
  `escalation_marker="outline_budget_exhausted"` and skips the
  rewrite tier.
- `escalation_marker: Optional[str]` — one of three values from the
  `_ESCALATION_MARKERS` frozenset:
  - `outline_budget_exhausted` — regen budget hit OR the
    `escalate_immediately: true` policy short-circuit fired (Worker
    3F path; provenance carried via `Touch.purpose="escalate_immediately"`).
  - `structural_unfixable` — a validator returned `action="block"`
    (Worker J semantics; the `Block` has no `status` field, so the
    marker is the canonical signal).
  - `validator_consensus_fail` — every self-consistency candidate
    failed validation with no candidate dominating; the router gives
    up rather than retry (Worker H path).

`route_with_self_consistency` interacts with these fields by sampling
N candidates (`COURSEFORGE_OUTLINE_N_CANDIDATES`, default 3) per
block, running the validator chain against each, and selecting the
highest-scoring passing candidate. When no candidate passes after
the regen budget is exhausted, the surviving best-effort candidate is
stamped with the appropriate `escalation_marker` and returned with
`validation_attempts` reflecting the total loop iterations. Legacy
validators that return `action=None, passed=False` retain
retry-loop semantics (regenerate); only EXPLICIT `action="block"`
or `action="escalate"` triggers a short-circuit.

Decision-capture is per-tier and per-decision; every router-side
choice and every LLM call lands as a typed event:

- `_emit_block_outline_call` (in `OutlineProvider`) — one
  `block_outline_call` decision per outline-tier LLM call. Rationale
  interpolates block_id, model, candidate index, success / parse
  failure status.
- `_emit_block_rewrite_call` (in `RewriteProvider`) — one
  `block_rewrite_call` decision per rewrite-tier LLM call.
- `_emit_block_validation_action` (in `CourseforgeRouter`) — one
  `block_validation_action` decision per validator chain run with
  `action`, `passed`, and the decisive validator name.
- `_emit_block_escalation` (in `CourseforgeRouter`) — one
  `block_escalation` decision per terminal escalation (budget
  exhausted, structural-unfixable block, consensus failure). Phase
  enum values: `courseforge-content-generator-outline` and
  `courseforge-content-generator-rewrite` (added per Subtask 7).

Known cross-validator inconsistencies (Worker M flagged for Phase
3.5 cleanup, intentionally documented here so consumers don't trip
on them):

- `BlockPageObjectivesValidator` requires a `valid_objective_ids`
  input key alongside `blocks` — undocumented in the validator
  registry and asymmetric with the other three Block validators
  which take only `blocks`. The router populates the key from the
  course's synthesised objectives JSON before dispatch.
- `BlockContentTypeValidator` enforces the chunk-type taxonomy
  (Trainforge-side enum) rather than the section-level
  content-type taxonomy. Symmetric naming hides the divergence;
  Phase 3.5 will rename or split.

When `COURSEFORGE_TWO_PASS=false` (default), none of the above
fires. The legacy `content_generation` phase runs the Phase 1
single-pass `ContentGeneratorProvider`, no router instantiation
happens, and the new `Block` fields stay at their defaults
(`validation_attempts=0`, `escalation_marker=None`).

### Phase 3.5: symmetric validation + remediation

Phase 3.5 closes three gaps left open by Phase 3: (1) the rewrite tier
emitted HTML with no validator gate downstream of it, so a rewrite that
silently dropped a CURIE / `content_type` / `objective_ref` / `sourceId`
sailed straight into packaging; (2) regen budgets defaulted to `3`,
which proved too tight once self-consistency calibration ran on a real
corpus; (3) `COURSEFORGE_TWO_PASS=true` runs would dispatch the
legacy single-pass content-generator tool against the new phase names
because the executor only resolved tools by agent name. Phase 3.5
adds **symmetric post-rewrite validation**, **remediation-injection
loops**, **bumped regen budgets**, and a **phase-name-aware tool
dispatch shim**. None of these changes alter behavior when
`COURSEFORGE_TWO_PASS` is unset.

Cross-links:

- `Courseforge/router/remediation.py` — canonical remediation module.
  Builds a structured suffix from a failed `GateResult` and the
  block's prior validation history; the suffix is appended to the
  outline / rewrite prompt on the next regen iteration so the model
  sees concrete failure signals instead of a blind retry. Wired into
  both `route_with_self_consistency` (outline tier, Subtask 18) and
  `route_rewrite_with_remediation` (rewrite tier, Subtask 19).
- `Courseforge/router/inter_tier_gates.py` — shape-discriminating
  Block adapters. Each of the four `Block*Validator` classes
  (`BlockCurieAnchoringValidator`, `BlockContentTypeValidator`,
  `BlockPageObjectivesValidator`, `BlockSourceRefValidator`)
  dispatches on `isinstance(block.content, dict | str)`: the dict
  path preserves the legacy outline-tier contract byte-stable, while
  the str path strips HTML and validates rewrite-tier output via the
  same gate chain. Single chain, two input shapes — symmetry is
  carried by the validator, not the workflow definition.
- `post_rewrite_validation` workflow phase (`config/workflows.yaml`,
  both `course_generation` and `textbook_to_course` workflows).
  Mirrors `inter_tier_validation` against the rewrite-tier
  `blocks_final_path` so HTML-emit drift is caught before packaging.
  Gates: `rewrite_curie_anchoring`, `rewrite_content_type`,
  `rewrite_page_objectives`, `rewrite_source_refs`. Severity matches
  the inter-tier seam (warning at the gate level; the router's
  `route_rewrite_with_remediation` loop is what fails closed when
  the rewrite-tier regen budget runs out).
- `_TOUCH_TIERS` extended with `outline_val` + `rewrite_val`
  (Subtask 14). Validator-tier Touches now stamp the `Block.touched_by`
  chain at the validation seam, so the canonical post-Phase-3.5
  Touch chain on a clean two-pass run is `outline → outline_val →
  rewrite → rewrite_val`. JSON-LD + SHACL Touch.tier enums (Subtasks
  15-16) carry the same enum values so downstream consumers (Trainforge
  chunk extraction, training data export) can filter on tier without
  string-matching against magic values.
- Bumped regen budgets (Subtask 20):
  `_DEFAULT_OUTLINE_REGEN_BUDGET = 10` (was `3`) and a new
  `_DEFAULT_REWRITE_REGEN_BUDGET = 10`. The bump from `3` was driven
  by self-consistency calibration on a real corpus; budget `3` was
  the placeholder before any data existed. Override per-block-type
  via `block_routing.yaml`'s new `regen_budget_rewrite` field
  (Subtask 21) or globally via the new `COURSEFORGE_REWRITE_REGEN_BUDGET`
  env var.
- `route_rewrite_with_remediation` (Subtask 19) — rewrite-tier
  analogue of `route_with_self_consistency`. Runs the post-rewrite
  validator chain, and when a gate returns `action="regenerate"`,
  injects the remediation suffix and re-rolls up to the rewrite-tier
  budget. Once the budget is exhausted with no candidate passing,
  the surviving best-effort candidate is stamped with
  `escalation_marker="validator_consensus_fail"` (semantic mirror of
  the outline-tier `validator_consensus_fail` from Worker H —
  consensus failure at the rewrite seam reuses the same marker
  rather than minting a new one, since the consumer-side handling
  is identical).
- `_PHASE_TOOL_MAPPING` dispatch shim in `MCP/core/executor.py`
  (Subtask 31). The executor's `_dispatch_agent_task` now checks
  `_PHASE_TOOL_MAPPING.get(phase_name)` BEFORE falling back to
  `AGENT_TOOL_MAPPING.get(agent_type)`. Mapping:
  `content_generation_outline → run_content_generation_outline`,
  `inter_tier_validation → run_inter_tier_validation`,
  `content_generation_rewrite → run_content_generation_rewrite`,
  `post_rewrite_validation → run_post_rewrite_validation`. Closes
  the Phase 3 review's HIGH-severity gap where
  `COURSEFORGE_TWO_PASS=true` would otherwise run the legacy
  single-pass `generate_course_content` tool against three
  separately-named phases.
- Three new tool handlers in `MCP/tools/pipeline_tools.py`:
  `_run_content_generation_outline` (Subtask 28),
  `_run_inter_tier_validation` (Subtask 29),
  `_run_content_generation_rewrite` (Subtask 30). Each emits a
  snake_case Block JSONL projection via `_block_to_snake_case_entry`
  (the round-trip helper from `MCP/tools/pipeline_tools.py` —
  Worker N1's silent-bug fix on Wave B; the JSONL must round-trip
  through `Block(**snake_case_kwargs)` cleanly so the next phase's
  handler can reload the block scaffolds without an aliased-attr
  parse step). Decision-capture parity with the Wave-B
  `_run_post_rewrite_validation` helper.
- `route_all` self-consistency dispatch (Subtask 33). When
  `n_candidates > 1` (resolved via the same priority chain as
  every other policy field), `route_all` now dispatches each block
  through `route_with_self_consistency` instead of calling
  `route()` directly. Closes Worker H's flagged followup; backward
  compatible because the fall-through `n_candidates == 1` path
  still calls `route()`.
- `block_validation_action` event extended with a `tier` field
  (Subtask 17 + 26). Disambiguates the inter-tier validation seam
  (`tier="outline"`) from the post-rewrite validation seam
  (`tier="rewrite"`) for downstream training-capture consumers.

Phase 3.5 follow-ups intentionally not closed in this batch:

- `inter_tier_validation` and `post_rewrite_validation` declare
  `agents: []` in `config/workflows.yaml` (Worker N1 flagged). The
  `_PHASE_TOOL_MAPPING` shim makes this work — the executor synthesises
  a single phase-scoped task and routes it through the phase-name
  handler — but a future cleanup should mint a synthetic
  `validator-only` agent type so the workflow YAML stays semantically
  symmetric with the rest of the Phase 3 pipeline.

When `COURSEFORGE_TWO_PASS=false` (default), none of the Phase 3.5
additions fire either: the legacy `content_generation` phase runs the
Phase 1 single-pass surface, the new tool handlers stay unloaded, the
`_PHASE_TOOL_MAPPING` shim's keys never match any phase the workflow
emits, and the bumped regen-budget defaults are never read.

### Operator smoke runbook (post-Phase-3.5)

The following sequence verifies the Phase 3.5 wiring end-to-end on a
clean checkout. Run after `git pull`, before merging Phase 3.5 work
to a downstream branch.

```bash
# 1. Set the master gate.
export COURSEFORGE_TWO_PASS=true

# 2. Run the textbook_to_course workflow with the new phase split.
ed4all run textbook-to-course \
  --corpus tests/fixtures/textbooks/demo_303.pdf \
  --course-name DEMO_303

# 3. Verify the canonical post-Phase-3.5 Touch chain on a sample block:
#    outline → outline_val → rewrite → rewrite_val.
jq -r '.[0].touched_by[].tier' \
  Courseforge/exports/PROJ-DEMO_303-*/03_content_development/blocks_validated.json
# Expected (sorted unique across all blocks): local, outline,
# outline_val, rewrite, rewrite_val.

# 4. Verify remediation injection fired at both seams. The
#    block_validation_action events carry a tier field that
#    disambiguates the inter-tier seam ("outline") from the
#    post-rewrite seam ("rewrite").
jq -r 'select(.decision_type=="block_validation_action") | .ml_features.tier' \
  training-captures/courseforge/DEMO_303/phase_courseforge-content-generator-outline/*.jsonl
# Expected: at least one "outline" tier event.
jq -r 'select(.decision_type=="block_validation_action") | .ml_features.tier' \
  training-captures/courseforge/DEMO_303/phase_courseforge-post-rewrite-validation/*.jsonl
# Expected: at least one "rewrite" tier event.

# 5. Verify validator-consensus failure stamps the rewrite tier when
#    the rewrite-tier regen budget runs out (the symmetric counterpart
#    of the outline-tier validator_consensus_fail marker).
jq -r '.[] | select(.escalation_marker=="validator_consensus_fail")
        | {block_id: .block_id, attempts: .validation_attempts}' \
  Courseforge/exports/PROJ-DEMO_303-*/03_content_development/blocks_validated.json
# Expected: any blocks listed here exhausted their tier budget.

# 6. Verify the executor's _PHASE_TOOL_MAPPING dispatched the three
#    new handlers instead of the legacy generate_course_content tool.
grep -E "phase=(content_generation_outline|inter_tier_validation|content_generation_rewrite|post_rewrite_validation)" \
  state/runs/*/workflow.log
# Expected: four log entries per workflow run, one per phase, in
# dependency order.

# 7. Run the regression suite to confirm no drift.
pytest Courseforge/router/tests/test_remediation.py \
       Courseforge/router/tests/test_inter_tier_gates_shape_dispatch.py \
       Courseforge/router/tests/test_remediation_injection.py \
       Courseforge/router/tests/test_route_all_self_consistency.py \
       MCP/tools/tests/test_pipeline_tools_phase3_handlers.py \
       lib/tests/test_phase3_5_decision_event_tier_field.py -v
```

Cross-link: the canonical Phase 3.5 plan (`plans/phase3_5_post_rewrite_validation.md`)
includes a parallel "Final smoke test" section keyed to subtask
verification; the runbook above is its operator-facing companion.

### Phase 4: statistical-tier validators + BERT ensemble

Phase 4 layers a statistical-tier validation seam on top of the Phase 3
two-pass router and the Phase 3.5 symmetric validation surface. Where
Phase 3.5 caught structural / shape drift (CURIE / `content_type` /
`page_objectives` / `sourceId`), Phase 4 catches **semantic** drift —
the rewrite-tier output that parses cleanly but says the wrong thing.
Five new gates fire at the same two seams (`inter_tier_validation` and
`post_rewrite_validation`), keeping the symmetric-validation contract
that Phase 3.5 established. None of these changes alter behavior when
`COURSEFORGE_TWO_PASS` is unset.

Cross-links:

- `lib/embedding/` — sentence-embedding wrapper around
  `sentence-transformers` (default model `all-MiniLM-L6-v2`, 384-dim,
  ~80 MB on disk). Behind a thin abstraction so callers degrade
  gracefully when the optional `[embedding]` extras are absent: missing
  deps surface as a warning-severity `EMBEDDING_DEPS_MISSING` GateIssue
  with `passed=True, action=None`. Strict-mode opt-in via
  `TRAINFORGE_REQUIRE_EMBEDDINGS=true` flips that fallback to critical
  (raises `EmbedderDepsMissing`) so production runs that *expect*
  semantic gating fail loudly when the deps are missing.
- `lib/validators/objective_assessment_similarity.py` —
  `ObjectiveAssessmentSimilarityValidator`. Cosine-similarity floor
  between every assessment-item block's stem and its referenced
  learning-objective text. Default threshold `0.55` (calibrated against
  `all-MiniLM-L6-v2`'s intrinsic similarity floor; topically-related
  but not semantically-aligned pairs cluster below ~0.40, so the 0.55
  default leaves a buffer above the noise floor). Below threshold emits
  `action="regenerate"` so the rewrite-tier router re-rolls with the
  remediation suffix.
- `lib/validators/concept_example_similarity.py` —
  `ConceptExampleSimilarityValidator`. Cosine-similarity floor between
  every concept-block definition and its illustrating example. Default
  threshold `0.50` — strictly looser than the objective↔assessment
  gate's 0.55 because examples are intentionally more concrete than
  the abstract concept they illustrate, and the embedding model treats
  the surface-form gap as moderate semantic distance.
- `lib/validators/objective_roundtrip_similarity.py` —
  `ObjectiveRoundtripSimilarityValidator`. Cosine-similarity floor
  between the rewrite-tier learning-objective paraphrase and the
  source objective text. Default threshold `0.70` — strictly tighter
  than the previous two gates because a paraphrase MUST preserve
  meaning; below 0.70 indicates semantic drift, not just surface-form
  variation.
- `lib/validators/courseforge_outline_shacl.py` —
  `CourseforgeOutlineShaclValidator`. Statistical-tier wrapper around
  `schemas/context/courseforge_v1.shacl-rules.ttl` shape constraints,
  applied to the outline-tier Block emit before the rewrite tier sees
  it. Catches structural drift the per-Block shape adapters miss (e.g.
  cross-block constraints).
- `lib/classifiers/bloom_bert_ensemble.py` — `BloomBertEnsemble`. Three
  SHA-pinned HuggingFace classifiers vote on the Bloom's-taxonomy
  level of every assessment-item block: (1)
  `kabir5297/bloom_taxonomy_classifier` (purpose-built 6-class Bloom
  classifier — natively aligned with the canonical `BLOOM_LEVELS`
  enum), (2) `distilbert-base-uncased-finetuned-sst-2-english`
  (sentiment model, contributes to dispersion via the low-resolution
  `_SST2_TO_BLOOM` mapping), (3)
  `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (zero-shot NLI
  against the six Bloom-level labels as hypotheses). Aggregates per-
  member votes via temperature-scaled majority + entropy-based
  dispersion. Member SHAs default to `"main"`; replacing those with
  concrete `huggingface_hub.HfApi().model_info(repo_id).sha` pins is a
  Phase 4 followup so the audit trail records exactly which revision
  produced each classification (captured in the
  `bert_ensemble_member_loaded` decision event).
- `lib/validators/bloom_classifier_disagreement.py` —
  `BloomClassifierDisagreementValidator`. Wraps the BERT ensemble and
  fires `action="regenerate"` on two signals: (a) the ensemble's
  majority-vote Bloom level disagrees with the block's declared
  `bloomLevel` attribute (mid-tier signal — emits
  `bert_ensemble_disagreement`); (b) ensemble dispersion exceeds
  `_DISPERSION_THRESHOLD = 0.7` (independent signal — emits
  `bert_ensemble_dispersion_high`; high entropy across members signals
  an unstable consensus that's worth re-rolling regardless of whether
  the majority happens to agree with the block-declared level).
  **Symmetric wiring** per the Phase 3.5 contract: gate ID
  `outline_bloom_classifier_disagreement` fires at
  `inter_tier_validation` and `rewrite_bloom_classifier_disagreement`
  fires at `post_rewrite_validation`, both routed through the same
  validator class with the same threshold.
- Phase 4 Wave N0 phase-handler dispatch fix (Subtasks 1-4): the
  Phase 3.5 `_PHASE_TOOL_MAPPING` shim was extended so the new
  Phase 4 phases route through dedicated handlers end-to-end. Closes
  the executor-side gap that would otherwise have run the legacy
  single-pass content-generator tool against the Phase 4 phase names
  and silently bypassed every statistical-tier gate.
- `scripts/calibrate_phase4_thresholds.py` — per-course threshold
  calibration with temperature scaling (BERT) + dispersion-threshold
  sweep (Subtasks 32-34). Emits
  `LibV2/courses/<slug>/eval/calibrated_thresholds.yaml` so the per-
  course thresholds can be loaded back into the gate `config.thresholds`
  dict at workflow time without rebuilding the gate registry.

Phase 4 follow-ups intentionally not closed in this batch:

- Concrete HuggingFace model SHAs for the three BERT ensemble members
  (placeholder `"main"` revision today). Resolution path:
  `huggingface_hub.HfApi().model_info(repo_id).sha` against a trusted
  pin set, captured in the `bert_ensemble_member_loaded` decision
  event so the audit trail records exactly which revision produced
  each classification.
- The calibrated thresholds emitted to `eval/calibrated_thresholds.yaml`
  are not yet auto-loaded back into the gate registry at workflow
  time — operators apply them manually by editing
  `config/workflows.yaml::validation_gates[].config.thresholds`. A
  future cleanup should make `WorkflowRunner` resolve a per-course
  YAML overlay before instantiating each validator.

When `COURSEFORGE_TWO_PASS=false` (default), none of the Phase 4 gates
fire either: the legacy `content_generation` phase doesn't carry the
new `inter_tier_validation` / `post_rewrite_validation` phase names,
the `_PHASE_TOOL_MAPPING` shim's Phase 4 entries never match, and the
embedding / BERT extras stay unloaded.

### Operator smoke runbook (Phase 4 statistical tier)

Extends the Phase 3.5 runbook above. The sequence below verifies the
Phase 4 statistical-tier wiring end-to-end on a clean checkout with
the optional extras installed. Run after the Phase 3.5 smoke
verifies clean.

```bash
# 1. Install the optional embedding extras (sentence-transformers +
#    transformers + torch). Without these, Phase 4 gates degrade to
#    EMBEDDING_DEPS_MISSING / BERT_ENSEMBLE_DEPS_MISSING warning
#    GateIssues (passed=True, action=None) — the workflow still runs,
#    just without the statistical-tier signal.
pip install -e '.[embedding]'

# 2. Set the Phase 3 master gate + Phase 4 strict mode (optional —
#    omit TRAINFORGE_REQUIRE_EMBEDDINGS to keep the graceful-degrade
#    fallback for CPU-only dev boxes).
export COURSEFORGE_TWO_PASS=true
export TRAINFORGE_REQUIRE_EMBEDDINGS=true

# 3. Run the textbook_to_course workflow with the Phase 4 gates wired.
ed4all run textbook-to-course \
  --corpus tests/fixtures/textbooks/demo_303.pdf \
  --course-name DEMO_303

# 4. Verify the Phase 4 BERT ensemble gates fired at both seams.
#    The four new decision_event types are emitted by the BERT
#    ensemble validator (bert_ensemble_disagreement /
#    bert_ensemble_dispersion_high) and by every Phase 4 validator
#    that runs to completion (statistical_validation_pass /
#    statistical_validation_fail).
grep -lE '"decision_type":\s*"(bert_ensemble_disagreement|bert_ensemble_dispersion_high|statistical_validation_pass|statistical_validation_fail)"' \
  training-captures/courseforge/DEMO_303/phase_courseforge-content-generator-outline/*.jsonl \
  training-captures/courseforge/DEMO_303/phase_courseforge-post-rewrite-validation/*.jsonl
# Expected: at least one match per phase directory; statistical_validation_pass
# / _fail emit on every block, the bert_* events emit only when the
# ensemble actually disagrees / disperses.

# 5. Run the calibration script against the holdout corpus to refresh
#    per-course thresholds. Writes
#    LibV2/courses/<slug>/eval/calibrated_thresholds.yaml.
python scripts/calibrate_phase4_thresholds.py \
  --course-slug demo-303 \
  --gate objective_assessment \
  --sweep-from 0.30 --sweep-to 0.80 --steps 11
ls LibV2/courses/demo-303/eval/calibrated_thresholds.yaml
# Expected: file exists; YAML carries per-gate calibrated thresholds.

# 6. Verify the graceful-degrade fallback when the [embedding] extras
#    are absent. Uninstall, re-run, confirm the warning-severity
#    GateIssue surfaces but the workflow does NOT fail closed
#    (because TRAINFORGE_REQUIRE_EMBEDDINGS is now unset).
pip uninstall -y sentence-transformers
unset TRAINFORGE_REQUIRE_EMBEDDINGS
ed4all run textbook-to-course \
  --corpus tests/fixtures/textbooks/demo_303.pdf \
  --course-name DEMO_303_DEGRADED
grep -lE '"code":\s*"EMBEDDING_DEPS_MISSING"' \
  training-captures/courseforge/DEMO_303_DEGRADED/phase_courseforge-content-generator-outline/*.jsonl
# Expected: at least one match; workflow exits 0 because the gate
# emitted passed=True, action=None.
```

Cross-link: the canonical Phase 4 plan
(`plans/phase4_statistical_tier_detailed.md`) includes a parallel
verification matrix keyed to subtask numbers; the runbook above is
its operator-facing companion.

#### Phase 4 end-to-end verification matrix

The matrix below pins the per-gate evidence operators should grep for
on a clean Phase 4 run. Each row maps a Phase 4 gate to (a) the gate
ID emitted at both seams, (b) the decision-event types it produces,
and (c) the artifact path the gate writes its issues to. Use this
matrix to confirm a full Phase 4 run exercised every statistical-tier
surface (and not just a subset).

| Gate | Outline-tier gate ID | Rewrite-tier gate ID | Decision events | Default threshold |
|------|----------------------|----------------------|-----------------|-------------------|
| Objective↔Assessment cosine | `outline_objective_assessment_similarity` | `rewrite_objective_assessment_similarity` | `statistical_validation_pass` / `statistical_validation_fail` | `min_cosine = 0.55` |
| Concept↔Example cosine | `outline_concept_example_similarity` | `rewrite_concept_example_similarity` | `statistical_validation_pass` / `statistical_validation_fail` | `min_cosine = 0.50` |
| Objective roundtrip cosine | `outline_objective_roundtrip_similarity` | `rewrite_objective_roundtrip_similarity` | `statistical_validation_pass` / `statistical_validation_fail` | `min_cosine = 0.70` |
| SHACL outline shapes | `outline_courseforge_shacl` | `rewrite_courseforge_shacl` | `statistical_validation_pass` / `statistical_validation_fail` | n/a (binary shape conformance) |
| BERT ensemble disagreement | `outline_bloom_classifier_disagreement` | `rewrite_bloom_classifier_disagreement` | `bert_ensemble_disagreement` / `bert_ensemble_dispersion_high` / `bert_ensemble_member_loaded` / `statistical_validation_pass` / `statistical_validation_fail` | `dispersion_threshold = 0.7` |

End-to-end smoke after a clean run on a holdout corpus:

```bash
# 1. Seed env (extends the Phase 4 runbook above).
export COURSEFORGE_TWO_PASS=true
export TRAINFORGE_REQUIRE_EMBEDDINGS=true
export COURSEFORGE_OUTLINE_PROVIDER=local      # ToS-clean outline tier
export COURSEFORGE_REWRITE_PROVIDER=anthropic  # Pedagogy-rich rewrite

# 2. Holdout corpus (small, fast — keeps the smoke tractable).
ed4all run textbook-to-course \
  --corpus tests/fixtures/textbooks/holdout_smoke.pdf \
  --course-name HOLDOUT_SMOKE

# 3. Confirm every Phase 4 gate fired at BOTH the inter-tier and
#    post-rewrite seams. The five outline-tier gate IDs and five
#    rewrite-tier gate IDs should each have at least one
#    statistical_validation_pass OR _fail event in the matching
#    phase JSONL.
for gate in \
  outline_objective_assessment_similarity \
  outline_concept_example_similarity \
  outline_objective_roundtrip_similarity \
  outline_courseforge_shacl \
  outline_bloom_classifier_disagreement; do
  count=$(grep -lE "\"gate_id\":\\s*\"$gate\"" \
    training-captures/courseforge/HOLDOUT_SMOKE/phase_courseforge-content-generator-outline/*.jsonl 2>/dev/null | wc -l)
  echo "$gate: $count files"
done
# Expected: each gate ≥ 1. Repeat for the rewrite-tier IDs against
# the phase_courseforge-post-rewrite-validation directory.

# 4. Calibrate per-course thresholds against this holdout corpus.
#    Sweep the BERT dispersion threshold first (independent of the
#    embedding gates), then the three cosine gates.
for gate in bert_ensemble objective_assessment concept_example objective_roundtrip; do
  python scripts/calibrate_phase4_thresholds.py \
    --course-slug holdout-smoke \
    --gate "$gate" \
    --sweep-from 0.30 --sweep-to 0.90 --steps 13
done
# Expected: LibV2/courses/holdout-smoke/eval/calibrated_thresholds.yaml
# carries one top-level key per gate, each with a calibrated_threshold
# float and a sweep[] table. Apply by hand to
# config/workflows.yaml::validation_gates[].config.thresholds — the
# auto-overlay loader is a Phase 4 followup.
test -s LibV2/courses/holdout-smoke/eval/calibrated_thresholds.yaml \
  && echo "OK: calibrated_thresholds.yaml emitted"

# 5. Graceful-degrade smoke. Uninstall the [embedding] extras and
#    rerun WITHOUT TRAINFORGE_REQUIRE_EMBEDDINGS so the warning-
#    severity fallback path runs end-to-end without failing closed.
pip uninstall -y sentence-transformers transformers torch
unset TRAINFORGE_REQUIRE_EMBEDDINGS
ed4all run textbook-to-course \
  --corpus tests/fixtures/textbooks/holdout_smoke.pdf \
  --course-name HOLDOUT_SMOKE_DEGRADED
# Expected: workflow exits 0; every embedding gate emits an
# EMBEDDING_DEPS_MISSING GateIssue with passed=True, action=None;
# the BERT ensemble emits BERT_ENSEMBLE_DEPS_MISSING with the same
# warning posture; no statistical_validation_fail events fire because
# the gates short-circuit before evaluation.
grep -clE "\"code\":\\s*\"(EMBEDDING_DEPS_MISSING|BERT_ENSEMBLE_DEPS_MISSING)\"" \
  training-captures/courseforge/HOLDOUT_SMOKE_DEGRADED/phase_courseforge-*/*.jsonl
# Expected: at least one match per phase directory.
```

The graceful-degrade path is the production-safety contract: a CPU-only
dev box without GPU/torch wheels still completes a `textbook_to_course`
run, just without the statistical-tier signal. Production deployments
that depend on the signal MUST set `TRAINFORGE_REQUIRE_EMBEDDINGS=true`
so missing extras fail loudly instead of silently bypassing the gates.

### Phase 6: ABCD authorship + concept extraction

Phase 6 layers two orthogonal upgrades on top of the Phase 4
statistical-tier surface: (1) **ABCD-framework authorship** for every
synthesized learning objective — discrete `audience` / `behavior` /
`condition` / `degree` fields per LO, with verb-Bloom alignment gated
by a new validator at `course_planning`; and (2) a **standalone
`concept_extraction` workflow phase** between `source_mapping` and
`course_planning`, decoupling the pedagogy / concept-graph build from
its legacy `libv2_archival`-bundled call site so the synthesizer can
read the graph and the linker can populate `LearningObjective.keyConcepts[]`
deterministically before content generation. Neither change requires
`COURSEFORGE_TWO_PASS`; the new gates fire on every `textbook_to_course`
run regardless of the two-pass master gate. Legacy LOs without the
`abcd` field skip the verb-alignment check (warning-severity
`ABCD_MISSING` GateIssue, `passed=True, action=None`), preserving
backward compatibility for pre-Phase-6 corpora.

Cross-links:

- `schemas/knowledge/courseforge_jsonld_v1.schema.json::$defs.AbcdObjective`
  (commit `98e27c0`, Phase 6 Subtask 1) — the canonical ABCD shape.
  `{audience: str, behavior: {verb: str, action_object: str}, condition: str, degree: str}`,
  all four required when `abcd` is present, `additionalProperties: false`.
  Referenced from `$defs.LearningObjective.properties.abcd` as an
  optional pointer so legacy LOs without the field validate unchanged.
- `lib/ontology/learning_objectives.py::BLOOMS_VERBS`
  (commit `b46e433`, Phase 6 Subtask 2) — `Dict[str, FrozenSet[str]]`
  keyed on the canonical six Bloom levels (`remember`, `understand`,
  `apply`, `analyze`, `evaluate`, `create`); values are frozensets of
  verbs lifted from `schemas/taxonomies/bloom_verbs.json` at import
  time and `lru_cache`'d. Single source of truth for the verb-Bloom
  alignment check.
- `lib/ontology/learning_objectives.py::compose_abcd_prose`
  (commit `b46e433`, Phase 6 Subtask 3) — deterministic prose
  composer. Format: `"{Audience} will {verb} {action_object} {condition}, {degree}."`
  Strips trailing periods on inputs to avoid double-periods,
  capitalises the first letter, and emits a terminal period.
  Round-trip tested across 6 Bloom levels × 3 verbs each in
  `lib/ontology/tests/test_compose_abcd_prose.py` (commit `986de36`,
  47 tests).
- `lib/validators/abcd_objective.py::AbcdObjectiveValidator`
  (commit `99e880b`, Phase 6 Subtasks 4-5) — wired as the
  `abcd_verb_alignment` gate at the `course_planning` phase
  (`config/workflows.yaml::textbook_to_course::course_planning::validation_gates`,
  warning severity in Phase 6 — promotion to critical follows once
  corpus calibration confirms safe). For each LO with `abcd` present:
  asserts `abcd.behavior.verb.lower() in BLOOMS_VERBS[lo.bloom_level]`.
  On miss, emits a `GateIssue` with `code="ABCD_VERB_BLOOM_MISMATCH"`
  + `action="regenerate"` and a `decision_event` of type
  `abcd_verb_bloom_mismatch`. On pass, emits `decision_type="abcd_authored"`.
  Both decision-type values were added to
  `schemas/events/decision_event.schema.json::decision_type.enum` in
  alphabetical position so `DECISION_VALIDATION_STRICT=true` runs
  don't fail closed on the first emit.
- `Courseforge/agents/course-outliner.md` (commit `e8e59fe`, Phase 6
  Subtask 6) — agent prompt amendment. The course-outliner now emits
  ABCD as a discrete sub-object per LO + 2 worked examples covering
  `remember` and `apply` Bloom levels. The downstream
  `_plan_course_structure` widening (commits `3d662ca` Subtasks 7-8)
  threads the ABCD payload through `_cgh.synthesize_objectives_from_topics`
  and persists per LO in `synthesized_objectives.json`.
- `lib/validators/concept_graph.py::ConceptGraphValidator`
  (commit `c10f19d`, Phase 6 Subtask 14) — wired as the
  `concept_graph` gate at the new `concept_extraction` phase, warning
  severity initially. Gates the emitted graph on (a) ≥10 concept
  nodes, (b) ≥5 edge types present (taxonomic + pedagogical mix),
  (c) every node carries a `class` field, (d) every edge carries a
  `relation_type` field, plus optional per-edge provenance when
  `TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE=true`.
  18 tests in `lib/validators/tests/test_concept_graph.py`.
- `MCP/tools/pipeline_tools.py::_run_concept_extraction`
  (commit `e0ea640`, Phase 6 Subtasks 11-12) — the new
  `concept_extraction` phase handler. Reads staged DART chunks via
  `ed4all_chunker.chunk_content(...)` (Phase 7a chunker package),
  invokes `Trainforge.pedagogy_graph_builder.build_pedagogy_graph`,
  persists the graph to
  `LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json`,
  computes `concept_graph_sha256` (SHA-256 of canonicalised graph
  JSON), and routes the hash through
  `phase_outputs.concept_extraction.concept_graph_sha256` so
  downstream phases (`course_planning`, `libv2_archival`) can consume
  it. 6 smoke tests pin the emit shape.
- `lib/ontology/concept_objective_linker.py::link_concepts_to_objectives`
  (commit `763aa7c`, Phase 6 Subtasks 15-16) — deterministic linker
  invoked from `_plan_course_structure` after objective synthesis but
  before `synthesized_objectives.json` is persisted. Two-stage match
  per roadmap §6.6: (1) substring match between concept-graph node
  slugs and the LO's existing `keyConcepts[]`; (2) for unmatched
  nodes, scan the LO statement for verbatim concept-slug occurrence.
  Returns the enriched objectives list with populated `keyConcepts[]`.
  12 tests.
- `pedagogy-graph-builder` agent spec
  (`Trainforge/agents/pedagogy-graph-builder.md`, commit `95e9dda`,
  Phase 6 Subtask 10) — the standalone graph-builder agent dispatched
  by the new `concept_extraction` phase. Inputs: `dart_chunks_path`.
  Output: `concept_graph_path` + `concept_graph_sha256`. Wraps
  `Trainforge.pedagogy_graph_builder.build_pedagogy_graph` directly;
  the agent registry routes `concept_extraction` here via the
  `_PHASE_TOOL_MAPPING` entry that points
  `concept_extraction → run_concept_extraction`.
- `concept_graph_sha256` field in `course_manifest.json`
  (commit `4c0ce9d` schema add + commit `c3a9f72` validator,
  Phase 6 Subtasks 17-20) — top-level optional 64-char lowercase
  hex hash. Emitted by `archive_to_libv2` from the
  `phase_outputs.concept_extraction.concept_graph_sha256` route.
  Read back by `LibV2ManifestValidator._check_concept_graph_sha256`
  with warning severity in Phase 6: missing field emits
  `MANIFEST_CONCEPT_GRAPH_SHA256_MISSING`, malformed shape emits
  `MANIFEST_CONCEPT_GRAPH_SHA256_MALFORMED`, hash-vs-on-disk-graph
  disagreement emits `MANIFEST_CONCEPT_GRAPH_SHA256_MISMATCH`.
  Phase 7c promotes all three to critical severity. 7 tests in
  `lib/validators/tests/test_libv2_manifest_concept_graph.py`.

When the new `concept_extraction` phase runs but the upstream DART
output yields a degenerate graph (zero edges, single node-class), the
`ConceptGraphValidator` warning surfaces in the workflow log and the
graph is still persisted — the warning posture is deliberate so
small-corpus textbooks don't block the course build, but the
manifest's `concept_graph_sha256` field captures what was actually
emitted so downstream Trainforge consumption is honest about the
input it saw. The Phase 6 followup is to promote both the
`concept_graph` gate and the `abcd_verb_alignment` gate to critical
once a real-corpus calibration pass confirms the thresholds don't
trip on healthy textbooks.

### Operator smoke runbook (Phase 6 ABCD + concept extraction)

End-to-end walkthrough for verifying a clean Phase 6 run on a real
corpus. Independent of `COURSEFORGE_TWO_PASS` — Phase 6 surfaces fire
at the `course_planning` and `concept_extraction` phases, both of
which run in single-pass mode, so this runbook is valid regardless of
the Phase 4 master gate. Setting `COURSEFORGE_TWO_PASS=true` here is
optional and just exercises the full Phase 4 statistical-tier surface
alongside the Phase 6 gates.

```bash
# 1. (Optional) Set Phase 4 master gate so the full statistical-tier
#    surface fires alongside the Phase 6 gates. Phase 6 gates fire
#    independent of this — they're wired at course_planning and
#    concept_extraction, both of which run in single-pass mode.
export COURSEFORGE_TWO_PASS=true

# 2. Run textbook-to-course end-to-end. The new concept_extraction
#    phase fires between source_mapping and course_planning; the
#    course_planning phase reads the emitted concept_graph_path and
#    invokes link_concepts_to_objectives before persisting
#    synthesized_objectives.json.
ed4all run textbook-to-course \
  --corpus tests/fixtures/textbooks/demo_303.pdf \
  --course-name DEMO_303

# 3. Verify ABCD-shaped LOs in the synthesized objectives. Each LO
#    should carry the four required ABCD fields when the outliner
#    populated them.
jq -r '.learning_outcomes[] | select(.abcd != null) |
  "\(.id): \(.abcd.behavior.verb) (\(.bloom_level)) — audience=\(.abcd.audience)"' \
  Courseforge/exports/PROJ-DEMO_303-*/01_learning_objectives/synthesized_objectives.json
# Expected: one line per ABCD-tagged LO; the verb shown should match
# BLOOMS_VERBS[bloom_level] (mismatches surface as ABCD_VERB_BLOOM_MISMATCH
# GateIssues — see step 5).

# 4. Verify the concept_extraction phase emitted a graph + the LibV2
#    archive carries the SHA-256.
ls LibV2/courses/demo-303/concept_graph/concept_graph_semantic.json
jq -r '.concept_graph_sha256' LibV2/courses/demo-303/manifest.json
# Expected: graph file exists; manifest field is a 64-char lowercase
# hex string. Verify the hash agrees with the on-disk graph:
sha256sum LibV2/courses/demo-303/concept_graph/concept_graph_semantic.json
# Expected: hex prefix matches the manifest's concept_graph_sha256.

# 5. Confirm the ABCD verb-alignment gate fired at course_planning.
#    Phase 6 emits two decision_event types: abcd_authored on every
#    LO that passes the verb check, and abcd_verb_bloom_mismatch on
#    every LO that fails (paired with an action="regenerate"
#    GateIssue carrying code="ABCD_VERB_BLOOM_MISMATCH").
grep -lE '"decision_type":\s*"(abcd_authored|abcd_verb_bloom_mismatch)"' \
  training-captures/courseforge/DEMO_303/phase_courseforge-course-outliner/*.jsonl
# Expected: at least one match; abcd_authored should fire for every
# LO that passes (so on a healthy run, this is the default). Mismatches
# are rare on a calibrated outliner.

# 6. Confirm the concept_graph gate ran at concept_extraction. The
#    validator emits warnings (severity warning in Phase 6) on
#    missing edge types or thin graphs; on a healthy corpus, those
#    warnings should be absent.
grep -lE '"gate_id":\s*"concept_graph"' \
  training-captures/courseforge/DEMO_303/phase_*concept_extraction*/*.jsonl 2>/dev/null
# Expected: ≥1 match. Inspect with `jq` to confirm
# passed=true, action=null.

# 7. Run the Phase 6 test surface end-to-end (no corpus required).
python -m pytest \
  lib/validators/tests/test_abcd_objective.py \
  lib/validators/tests/test_concept_graph.py \
  lib/ontology/tests/test_compose_abcd_prose.py \
  lib/ontology/tests/test_concept_objective_linker.py \
  lib/validators/tests/test_libv2_manifest_concept_graph.py \
  -v
# Expected: all PASSED. The five test files cover the validator
# contracts (≥6 tests each on AbcdObjectiveValidator and
# ConceptGraphValidator), the prose round-trip (47 tests across 6
# Bloom levels × 3 fixture verbs), the linker (12 tests covering
# both substring + verbatim-statement match paths), and the LibV2
# manifest hash verification (7 tests).
```

The Phase 6 contract is **fail-soft**: every new gate is warning-
severity in Phase 6 so partial-coverage corpora (legacy LOs without
ABCD, thin concept graphs from sparse textbooks) don't block the
course build. The decision-event JSONL captures the full audit trail
regardless of severity, so operators can grep for `ABCD_VERB_BLOOM_MISMATCH`
or `MANIFEST_CONCEPT_GRAPH_SHA256_*` codes after the run to
prioritise outliner / extractor calibration work without disrupting
the build itself. Phase 7+ promotes both gates to critical once a
real-corpus calibration pass confirms the thresholds.

### Phase 7b: DART chunkset

Phase 7b lifts the chunker invocation out of every downstream consumer
and into a standalone `chunking` workflow phase that emits the DART
chunkset once, deterministically, before any objective extraction or
concept-graph work runs. Two motivations: (1) the Phase 7a chunker
package (`ed4all-chunker`) is now the single canonical chunker shared
by DART, Courseforge, and Trainforge — Phase 7b promotes that
delegation surface to a first-class phase boundary so the chunkset is
addressable, hashable, and validatable as its own artifact rather than
an in-process side-effect of the synthesizer; (2) the dual-chunkset
architecture (DART chunkset emitted in Phase 7b, IMSCC chunkset emitted
post-packaging in Phase 7c) gives downstream consumers two
provenance-anchored chunk surfaces — one rooted in the textbook PDF,
one rooted in the packaged IMSCC — instead of conflating both into a
single `corpus/` blob whose source provenance was ambiguous.

Cross-links:

- `config/workflows.yaml::textbook_to_course::chunking`
  (commit `8459ff8`, Phase 7b Subtask 10) — new workflow phase wired
  between `staging` and `objective_extraction`. `agents: [dart-chunker]`,
  `depends_on: [staging]`, outputs `dart_chunks_path` +
  `dart_chunks_sha256`. The downstream `objective_extraction.depends_on`
  was widened from `[staging]` to `[chunking]` so the phase ordering
  becomes `staging → chunking → objective_extraction → source_mapping →
  concept_extraction → course_planning`. Phase 6's
  `concept_extraction.depends_on` is widened in parallel to
  `[source_mapping, chunking]` (Worker W14.5) so the consumer-side
  refactor (ST 14.5) can route the upstream `dart_chunks_path` directly
  into `_run_concept_extraction` and retire the inline v4-chunk
  projection it currently carries.
- `dart-chunker` agent registration
  (`config/agents.yaml::dart-chunker`, commit `458a5b0`, Phase 7b
  Subtask 9) — utility-style agent registered with `type: utility`. No
  standalone `.md` spec because the agent is a deterministic chunker
  transformation with no LLM dispatch (mirrors the `textbook-stager`
  precedent). The agent is dispatched in-code via the
  `MCP/core/executor.py::AGENT_TOOL_MAPPING` entry
  `"dart-chunker": "run_dart_chunking"` which routes to the
  `_run_dart_chunking` helper registered below.
- `MCP/tools/pipeline_tools.py::_run_dart_chunking`
  (`:6361-6627`, commit `5ccbf0c`, Phase 7b Subtask 11) — async helper
  registered in `_build_tool_registry` as `registry["run_dart_chunking"]`
  (mirrors the Phase 6 `_run_concept_extraction` template). Inputs
  resolved via the workflow YAML's `inputs_from`: `course_name`
  (workflow params) + `staging_dir` (upstream `staging` phase output).
  The helper invokes `ed4all_chunker.chunk_content` against the staged
  DART HTML files (parsed via `Trainforge/parsers/html_content_parser.py::HTMLContentParser`
  into `ContentSection` objects, then threaded into the chunker with a
  thin `ChunkerContext` whose `create_chunk` callback emits canonical
  v4-shape chunk dicts), persists the chunks to
  `LibV2/courses/<slug>/dart_chunks/chunks.jsonl`, writes the sibling
  `manifest.json` (next bullet), computes the SHA-256 of the chunks
  file, and surfaces both `dart_chunks_path` + `dart_chunks_sha256`
  through `phase_outputs.chunking` for downstream consumption.
- `schemas/library/chunkset_manifest.schema.json`
  (commit `626a53b`, Phase 7b Subtask 12) — the canonical sidecar
  manifest schema for both chunkset kinds. Symmetric across DART
  (Phase 7b) and IMSCC (Phase 7c): a single `chunkset_kind` enum
  discriminator (`"dart"` | `"imscc"`) plus a conditional source-SHA
  requirement keeps both sidecars on a single contract. Required
  fields: `chunks_sha256` (64-char lowercase hex), `chunker_version`
  (resolved from `ed4all_chunker.__version__`), `chunkset_kind`. The
  conditional branch requires `source_dart_html_sha256` when
  `chunkset_kind == "dart"` and `source_imscc_sha256` when
  `chunkset_kind == "imscc"`, anchoring the chunkset to its upstream
  source artifact. Optional: `chunks_count` (non-negative integer),
  `generated_at` (ISO-8601 timestamp). The manifest forward-references
  the Triangle-invariant chain validation (PDF → DART HTML → chunks)
  scheduled for Phase 7c followup.
- `lib/validators/chunkset_manifest.py::ChunksetManifestValidator`
  (commit `1f68ffa`, Phase 7b Subtask 13) — wired as the
  `chunkset_manifest` gate at the new `chunking` phase, warning
  severity in Phase 7b. Verifies: (a) the on-disk
  `dart_chunks/manifest.json` exists and parses as JSON; (b) it
  conforms to `schemas/library/chunkset_manifest.schema.json`; (c) the
  manifest's `chunks_sha256` matches the SHA-256 of the actual
  `chunks.jsonl` file (catches stale-manifest drift); (d) the
  manifest's `chunker_version` matches the installed
  `ed4all_chunker.__version__` (catches version-skew across rebuilds);
  (e) the conditional source-SHA field is present and well-formed for
  the declared `chunkset_kind`. GateIssue codes: `MANIFEST_MISSING`,
  `MANIFEST_PARSE_ERROR`, `MANIFEST_SCHEMA_INVALID`,
  `CHUNKS_SHA256_MISMATCH`, `CHUNKER_VERSION_MISMATCH`,
  `SOURCE_SHA256_MISSING`. Phase 7c promotes the gate from warning to
  critical once Worker W13 confirms the thresholds don't trip on a
  clean corpus rebuild.

When the new `chunking` phase runs but the upstream DART staging
yielded a thin or degenerate HTML output (single section, missing
headings, etc.), the `ChunksetManifestValidator` warning surfaces in
the workflow log and the chunks + manifest are still persisted — the
warning posture mirrors the Phase 6 `concept_graph` gate's fail-soft
contract so small-corpus textbooks don't block the course build, but
the manifest's `chunks_sha256` field captures what was actually
emitted so downstream concept-extraction / synthesis is honest about
the input it saw. The Phase 7c followup is to (a) promote the
`chunkset_manifest` gate to critical severity, (b) rename
`LibV2/courses/<slug>/corpus/` → `imscc_chunks/` and add a
post-packaging `imscc_chunking` workflow phase that emits the IMSCC
chunkset using the same `dart-chunker` agent + `_run_dart_chunking`
helper (the chunker is symmetric across both source kinds), and (c)
extend `LibV2ManifestValidator` to require both `dart_chunks_sha256`
and `imscc_chunks_sha256` in the top-level course manifest alongside
the Phase 6 `concept_graph_sha256` (also promoted to critical at the
same boundary).

### Phase 7c: IMSCC chunkset + manifest gate promotion

Phase 7c lands the symmetric IMSCC chunkset alongside the Phase 7b
DART chunkset, renames the legacy `corpus/` directory, and promotes
the three-hash triangle (DART chunks ↔ IMSCC chunks ↔ concept graph)
to critical-severity manifest enforcement at the `libv2_archival`
phase. Together with Phase 7b, the result is a per-course set of two
provenance-anchored chunk surfaces (one rooted in the textbook PDF
via `source_dart_html_sha256`, one rooted in the packaged IMSCC via
`source_imscc_sha256`) plus a course-manifest hash triangle that
fail-closes when any of the three required artifacts goes missing,
malforms, or drifts from disk.

Cross-links:

- Phase 7c ST 15 (commit `090d286`) renamed `LibV2/courses/<slug>/corpus/`
  → `imscc_chunks/`. Back-compat read shim:
  `lib/libv2_storage.py::resolve_imscc_chunks_path` (aliased
  `resolve_imscc_chunks_dir`). Legacy archives keep resolving with a
  deprecation warning until the Phase 8 drop.
- Phase 7c ST 16 (commit `0e4f2fb`) added the `imscc_chunking`
  workflow phase between `packaging` and `training_synthesis` plus
  the `_run_imscc_chunking` helper at
  `MCP/tools/pipeline_tools.py:6771`. Mirrors `_run_dart_chunking`'s
  template; reads HTML entries in-memory from the packaged `.imscc`
  zip via `zipfile.ZipFile`, emits `chunkset_kind="imscc"` plus
  `source_imscc_sha256` (SHA-256 of the archive bytes).
- Phase 7c ST 17 (commit `c61e608`) extended
  `lib/validators/libv2_manifest.py::LibV2ManifestValidator` with
  three new check methods (`_check_dart_chunks_sha256`,
  `_check_imscc_chunks_sha256`) plus promoted the existing
  `_check_concept_graph_sha256` from warning to critical. Each fires
  a `MISSING_*` / `INVALID_*` / `*_HASH_MISMATCH` GateIssue triplet
  against the matching course-manifest field. Fail-closed: zero
  critical issues required at the `libv2_archival` gate.
- Phase 7c ST 18 (commit `d3d23f0`) added
  `LibV2/tools/libv2/scripts/backfill_dart_chunks.py` for migrating
  legacy archives that lack `dart_chunks/`. Operator-driven; idempotent
  by default with `--force` for re-emit and `--dry-run` for plan-only.

### Operator smoke runbook (Phase 7b + 7c chunkset chain)

End-to-end walkthrough for verifying a clean Phase 7b/c run on a real
corpus and migrating a legacy archive that pre-dates the chunkset
work. Phase 7b/c introduce no new env vars beyond the Phase 7a chunker
prereq (`pip install -e ./ed4all-chunker` so `ed4all_chunker.chunk_content`
is importable); the new workflow phases fire automatically on every
`textbook_to_course` run.

```bash
# 1. Phase 7a prereq — make the canonical chunker package importable.
#    Phase 7b/c assume ``ed4all_chunker.chunk_content`` resolves; the
#    helpers fail-soft with a warning + empty chunks shell otherwise.
pip install -e ./ed4all-chunker

# 2. Run textbook-to-course end-to-end. The chunking phase fires
#    between staging and objective_extraction (Phase 7b ST 10); the
#    imscc_chunking phase fires between packaging and
#    training_synthesis (Phase 7c ST 16). Both run automatically
#    regardless of COURSEFORGE_TWO_PASS or any other env flag.
ed4all run textbook-to-course \
  --corpus tests/fixtures/textbooks/demo_303.pdf \
  --course-name DEMO_303

# 3. Verify the DART chunkset landed under the new
#    LibV2/courses/<slug>/dart_chunks/ directory.
ls LibV2/courses/demo-303/dart_chunks/{chunks.jsonl,manifest.json}
# Expected: both files exist. The manifest is the canonical
# chunkset_manifest schema shape (chunkset_kind="dart" +
# source_dart_html_sha256 anchoring the chunkset to the staged DART
# HTML inputs).
jq -r '.chunkset_kind, .chunks_sha256, .chunker_version' \
  LibV2/courses/demo-303/dart_chunks/manifest.json
# Expected: "dart" + 64-char lowercase hex + the installed
# ed4all_chunker.__version__.

# 4. Verify the IMSCC chunkset landed at imscc_chunks/ (Phase 7c
#    rename of corpus/).
ls LibV2/courses/demo-303/imscc_chunks/{chunks.jsonl,manifest.json}
jq -r '.chunkset_kind, .chunks_sha256, .source_imscc_sha256' \
  LibV2/courses/demo-303/imscc_chunks/manifest.json
# Expected: "imscc" + two 64-char lowercase hex digests; the
# source_imscc_sha256 hashes the packaged .imscc archive bytes.

# 5. Verify the LibV2 course manifest carries the three Phase 7c
#    required hashes (the triangle: DART ↔ IMSCC ↔ concept graph).
#    Phase 7c ST 17 promoted all three fields to required + critical;
#    a missing, malformed, or divergent value fails the
#    libv2_archival gate.
jq -r '.dart_chunks_sha256, .imscc_chunks_sha256, .concept_graph_sha256' \
  LibV2/courses/demo-303/manifest.json
# Expected: three 64-char lowercase hex strings, each agreeing with
# the on-disk artifact bytes (chunks.jsonl files + concept_graph_semantic.json).

# 6. Round-trip the on-disk hash to confirm no drift between the
#    manifest record and the persisted artifact.
sha256sum LibV2/courses/demo-303/dart_chunks/chunks.jsonl
sha256sum LibV2/courses/demo-303/imscc_chunks/chunks.jsonl
sha256sum LibV2/courses/demo-303/concept_graph/concept_graph_semantic.json
# Expected: each hex prefix agrees with the matching manifest field.
# Any divergence fires the corresponding *_HASH_MISMATCH critical at
# the next libv2_archival gate run.

# 7. Confirm the chunkset_manifest gate fired at both chunking phases
#    (warning severity in Phase 7b/c — promotion to critical follows
#    in a future calibration wave).
grep -lE '"gate_id":\s*"chunkset_manifest"' \
  training-captures/*/DEMO_303/phase_*chunking*/*.jsonl 2>/dev/null
# Expected: ≥2 matches (one per chunking phase). Inspect with `jq`
# to confirm passed=true on a healthy corpus.

# 8. Run the Phase 7b/c integration test surface end-to-end (no
#    real corpus required — the test ships its own synthetic DART
#    HTML + IMSCC zip fixtures).
python -m pytest \
  lib/validators/tests/test_libv2_manifest_dual_chunkset.py \
  lib/validators/tests/test_libv2_manifest_concept_graph.py \
  lib/validators/tests/test_chunkset_manifest.py \
  LibV2/tests/test_backfill_dart_chunks.py \
  -v
# Expected: all PASSED. Coverage spans the integration chain (10
# tests), the concept-graph leg of the triangle (Phase 6 ST 19 +
# Phase 7c ST 17 promotion), the chunkset sidecar schema gate
# (Phase 7b ST 13), and the operator backfill script (Phase 7c ST 18).
```

#### Backfilling a legacy archive (operator workflow)

Pre-Phase-7b archives have no `dart_chunks/` directory and no
`dart_chunks_sha256` in the course manifest. Running the
`libv2_archival` gate against such an archive fires
`MISSING_DART_CHUNKS_SHA256` at critical severity, blocking every
subsequent run. Operator migration:

```bash
# Plan-only: enumerate every course under LibV2/courses/ and report
# which need backfill (no writes).
python -m LibV2.tools.libv2.scripts.backfill_dart_chunks \
  --libv2-root LibV2/courses --dry-run

# Backfill a single course by slug. Idempotent — skips when
# LibV2/courses/<slug>/dart_chunks/manifest.json already exists.
python -m LibV2.tools.libv2.scripts.backfill_dart_chunks \
  --libv2-root LibV2/courses --course-slug rdf-shacl-551-2

# Force re-emit when the existing dart_chunks/ is stale (e.g. the
# chunker package was updated and chunker_version no longer matches).
python -m LibV2.tools.libv2.scripts.backfill_dart_chunks \
  --libv2-root LibV2/courses --course-slug rdf-shacl-551-2 --force

# Re-run the libv2_archival gate against the backfilled archive to
# confirm the MISSING_DART_CHUNKS_SHA256 critical no longer fires.
python -c "
from lib.validators.libv2_manifest import LibV2ManifestValidator
result = LibV2ManifestValidator().validate({
    'manifest_path': 'LibV2/courses/rdf-shacl-551-2/manifest.json',
})
critical = [i for i in result.issues if i.severity == 'critical']
print(f'critical issues: {[i.code for i in critical]}')
print(f'passed: {result.passed}')
"
# Expected: no MISSING_DART_CHUNKS_SHA256 in the critical list. The
# script also updates the course-level manifest.json::dart_chunks_sha256
# field so the round-trip hash check passes on the next gate run.
```

The Phase 7c contract is **fail-closed** at the `libv2_archival`
phase: any of the three required hashes missing, malformed, or
divergent from the on-disk artifact blocks archival. The
`chunkset_manifest` gate at the two chunking phases stays
warning-severity in Phase 7c so a thin-corpus run still emits a
sidecar manifest the operator can inspect; promotion to critical is
scheduled for a future calibration wave once the thresholds are
confirmed against a clean rebuild on a real corpus.

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

## Intake & Remediation Workflow (NEW)

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
   - dart-automation-coordinator: PDFs/Office → accessible HTML
   - accessibility-remediation: WCAG 2.2 AA fixes
   - content-quality-remediation: Educational enhancements
   - intelligent-design-mapper: Interactive component styling
6. remediation-validator: Final quality validation
7. brightspace-packager: Generate improved IMSCC
```

### Remediation Capabilities
| Capability | Target |
|------------|--------|
| PDF Conversion | 100% to accessible HTML via DART |
| Office Documents | 100% to accessible HTML via DART |
| Alt Text | AI-generated for all images |
| Heading Structure | Automatic hierarchy correction |
| Color Contrast | WCAG AA (4.5:1 minimum) |
| Keyboard Navigation | Full accessibility |
| Component Styling | AI-selected interactive elements |
| Quality Enhancement | Learning objectives, summaries, checks |

### Scripts for Course Generation
| Script | Location | Purpose |
|--------|----------|---------|
| `generate_course.py` | `scripts/` | Multi-file weekly course generation. Emits page-level JSON-LD, `course_metadata.json`, prerequisite-page refs, `data-cf-teaching-role`, and `data-cf-source-ids` / page-level `sourceReferences` when DART source material is staged. Phase 2: accepts `--emit-mode {full,outline}` (default `full`); outline mode strips content/example/assessment HTML bodies but preserves their JSON-LD `blocks[]` projections, and stamps `course_metadata.blocks_summary.outline_only=true` so downstream consumers can detect the tier. |
| `package_multifile_imscc.py` | `scripts/` | Packages multi-file output into IMSCC. Structural validation is on by default (per-week `learningObjectives` must resolve to the week's LO manifest). Auto-discovers `course.json` and bundles `course_metadata.json` at the zip root. Manifest uses IMS Common Cartridge v1.3 namespaces; resources are nested under per-week `<item>` wrappers in the organization tree. **This is the runtime target of the MCP `package_imscc` tool** — `MCP/tools/pipeline_tools.py::_package_imscc` imports and delegates here instead of hand-rolling a ZIP. Phase 2: accepts `--outline-only` to package an outline-tier deliverable; reads `course_metadata.blocks_summary.outline_only` written by `generate_course.py --emit-mode outline`. |

`--emit-mode outline` (`generate_course.py`) and `--outline-only` (`package_multifile_imscc.py`) produce a stripped-down deliverable carrying only objectives + summaries; content/example/assessment HTML bodies are dropped while their JSON-LD `blocks[]` entries persist for downstream consumers (Trainforge `process_course.py` skips `instruction_pair` extraction when `course_metadata.blocks_summary.outline_only=true`). Outline mode is the input shape Phase 3's two-pass pipeline expects from the outline tier.

### Scripts for Intake
| Script | Location | Purpose |
|--------|----------|---------|
| `imscc_extractor.py` | `scripts/imscc-extractor/` | Universal IMSCC parsing |
| `component_applier.py` | `scripts/component-applier/` | Interactive component application |
| `remediation_validator.py` | `scripts/remediation-validator/` | Final quality validation |

### Success Metrics
| Metric | Target |
|--------|--------|
| IMSCC import success | 95%+ (any source LMS) |
| WCAG compliance | 100% Level AA |
| DART conversion | 98%+ for PDFs |
| Component accuracy | 90%+ appropriate selections |
