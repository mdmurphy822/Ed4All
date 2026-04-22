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
| `generate_course.py` | `scripts/` | Multi-file weekly course generation. Emits page-level JSON-LD, `course_metadata.json`, prerequisite-page refs, `data-cf-teaching-role`, and `data-cf-source-ids` / page-level `sourceReferences` when DART source material is staged. |
| `package_multifile_imscc.py` | `scripts/` | Packages multi-file output into IMSCC. Structural validation is on by default (per-week `learningObjectives` must resolve to the week's LO manifest). Auto-discovers `course.json` and bundles `course_metadata.json` at the zip root. Manifest uses IMS Common Cartridge v1.3 namespaces; resources are nested under per-week `<item>` wrappers in the organization tree. **This is the runtime target of the MCP `package_imscc` tool** — `MCP/tools/pipeline_tools.py::_package_imscc` imports and delegates here instead of hand-rolling a ZIP. |

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
