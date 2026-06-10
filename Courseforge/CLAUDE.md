# Courseforge

AI-powered instructional design system that creates and remediates accessible, LMS-ready IMSCC course packages.

> **Universal Protocols**: See root `/CLAUDE.md` for orchestrator protocol, execution rules (ONE agent = ONE file, max 10 parallel), decision capture requirements, and error handling. This file contains Courseforge-specific guidance only.

---

## Quick Start

### Course Creation Mode
**Input**: Exam objectives (PDF/text) + optional DART-processed textbooks (HTML)
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
(DART HTML)           requirements-      oscqr-            quality-assurance
                      collector          evaluator              (per batch)
```

### Pipeline 2: Intake & Remediation
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
| `course-outliner` | Synthesize canonical `TO-NN` / `CO-NN` objectives from textbook structure; persist `synthesized_objectives.json`. Routes to `plan_course_structure`. | Creating course framework |
| `content-generator` | Educational content creation | Content development (1 file per agent) |
| `quality-assurance` | Pattern prevention and validation | Quality gates |
| `oscqr-course-evaluator` | Educational quality assessment | OSCQR evaluation |
| `brightspace-packager` | IMSCC package generation | Final deployment |
| `textbook-ingestor` | Textbook content processing | Entry point for textbook materials |
| `source-router` | Bind DART source blocks to Courseforge module pages | Source attribution for pipeline runs |

### Intake & Remediation Agents
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

1. Run textbook PDF through DART:
   ```bash
   python -m DART.pdf_converter textbook.pdf -o /path/to/courseforge/inputs/textbooks/
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

## Opt-In Behavior Flags

Courseforge-owned env-var toggles. Every `COURSEFORGE_*_*` flag is no-op when `COURSEFORGE_TWO_PASS` is unset, **except** `COURSEFORGE_ALLOW_TEMPLATE_EMITTER` (the anti-silent-template guard escape hatch, which applies to every content_generation run) and `COURSEFORGE_RUN_ID` (a provenance label, read on every emit). Long-form rationale also lives in `schemas/ONTOLOGY.md` § 12.

| Flag | When on |
|------|---------|
| `COURSEFORGE_ALLOW_TEMPLATE_EMITTER` | Escape hatch for the anti-silent-template guard (`lib/validators/content_authorship.py::ContentAuthorshipValidator`). When unset (default), the `content_authorship` gate **blocks** any `content_generation` run where the deterministic `generate_week` template emitter produced the pages AND LLM authoring was intended (`COURSEFORGE_PROVIDER` / `COURSEFORGE_TWO_PASS` / `ED4ALL_AGENT_DISPATCH` set, or a constructed provider/router that degraded to the template floor on every page at runtime). Set to `1`/`true` to permit the template emitter anyway — for offline / no-LLM smoke runs where templated pages are acceptable. `_generate_course_content` records the decision (plus `generator_mode`, `template_fallback_fired`, `llm_authored_pages`/`template_fallback_pages`) in `<export>/content_generation_provenance.json`; the gate reads it. NOT gated by `COURSEFORGE_TWO_PASS`. |
| `COURSEFORGE_RUN_ID` | Provenance label stamped into the JSON-LD `blocks[]` `provenance.runId` field by `Courseforge/scripts/generate_course.py::_build_page_metadata` (`:2193`, read via `os.environ.get("COURSEFORGE_RUN_ID", "")`). Default empty string — no run identifier in the emitted metadata. Purely a provenance tag (lets a `blocks[]`-emitting run be traced back to its orchestrator run); selects no provider/model and gates no behavior. NOT gated by `COURSEFORGE_TWO_PASS` — read on every `blocks[]` emit (which is itself gated by `COURSEFORGE_EMIT_BLOCKS`). |
| `COURSEFORGE_BLOCK_ROUTING_PATH` | Absolute or repo-relative path to the `block_routing.yaml` policy file consumed by `Courseforge/router/policy.py::load_block_routing_policy` and `Courseforge/router/router.py::CourseforgeRouter._resolve_spec`. Default `Courseforge/config/block_routing.yaml`; setting this var overrides the default location for ops topologies that ship the policy out-of-tree (e.g. mounted ConfigMap in containerised runs). The file itself is optional — when missing, the loader returns an empty policy and the router falls through to tier-default env vars (`COURSEFORGE_OUTLINE_*` / `COURSEFORGE_REWRITE_*`) plus the hardcoded defaults table at `policy.py::DEFAULT_BLOCK_ROUTING`. Schema (Draft 2020-12, `additionalProperties: false`): `schemas/courseforge/block_routing.schema.json`. Resolution priority (high → low): per-call kwargs > YAML > tier env vars > hardcoded defaults. **GPT Feedback v2 Wave 3 W3.C** (active): the YAML now carries a per-block-type `validators` matrix (`{required[], optional[], fail_action}`) for every entry in `Courseforge/scripts/blocks.py::BLOCK_TYPES`. `CourseforgeRouter._dispatch_validation_chain` filters the global `validation_gates` list down to the per-block_type allowed set; required-gate failures stamp `fail_action` (`regenerate`/`escalate`/`block`) onto the resulting `GateResult`, optional-gate failures emit warning-only (`GateResult.action` coerced to `pass`, issues preserved), and gates not in either array are skipped silently. Resolution priority for the validator matrix mirrors the provider/model chain: per-block-id override (YAML `overrides[]`) > per-block-type `validators` block (YAML `blocks.<type>.validators`) > tier env defaults > hardcoded `DEFAULT_BLOCK_ROUTING` table. Backward-compat: a block_type with no validators block falls through to the legacy "all gates run" behavior. No-op when `COURSEFORGE_TWO_PASS` is unset (default off). |
| `COURSEFORGE_EMIT_BLOCKS` | Intermediate Block format: `true` / `false`, default `false`. When truthy, `Courseforge/scripts/generate_course.py::_build_page_metadata` emits the new top-level JSON-LD `blocks[]` (canonical projection of the `Block` dataclass via `Block.to_jsonld_entry()`) plus `provenance` (`{runId, pipelineVersion: "phase2", tiers[]}`) and `contentHash` (SHA-256 of the canonicalised meta payload before the hash field is added) — and stamps `data-cf-block-id="{page_id}#{block_type}_{slug}_{idx}"` on every block-bearing wrapper (`<section>` / heading / component wrapper). Default off keeps emit byte-stable for the legacy snapshot regression suite under `Courseforge/scripts/tests/`, and Trainforge's `process_course._extract_section_metadata` falls back to the existing `data-cf-*` attribute chain so legacy corpora (and corpora rebuilt with the flag off) extract identically. Cross-link: canonical schema shape at `schemas/knowledge/courseforge_jsonld_v1.schema.json` (`$defs.Block`, `$defs.Touch`, top-level optional `blocks[]` / `provenance` / `contentHash`); dataclass + emitters at `Courseforge/scripts/blocks.py` (`Block` dataclass + the 16-value `BLOCK_TYPES` enum). Rolls forward to `true` after byte-stable confirmation across a clean run; the flag drops once the two-pass pipeline depends on the new fields unconditionally. |
| `COURSEFORGE_OUTLINE_GRAMMAR_MODE` | Selects the constrained-decoding payload `Courseforge/generators/_outline_provider.py::OutlineProvider._build_grammar_payload` injects into the wire body via `OpenAICompatibleClient.chat_completion(extra_payload=...)`. Values: `gbnf` (llama.cpp `grammar: <gbnf-string>`), `json_schema` (Ollama 0.5+ full-schema `format: <json-schema-dict>` and Together AI `response_format: {type: "json_schema", json_schema: {...}}`), `json_object` (default — Ollama legacy `format: "json"` + OpenAI `response_format: {"type": "json_object"}`), `none` (no grammar payload; falls back to JSON-mode-only + lenient parse). Unset → autodetect from the resolved outline provider (local → `json_object`; vLLM → `gbnf`; Together → `json_schema`; Anthropic → `none`). The Anthropic outline tier (rare; not the default) does not support sample-time grammar and always falls back to JSON-mode + lenient parse + remediation retry. No-op when `COURSEFORGE_TWO_PASS` is unset. On Qwen-2.5-7B-Q4 + Ollama 0.5+, attempt-2 convergence is expected; `COURSEFORGE_OUTLINE_REGEN_BUDGET` default of 10 leaves ample headroom. |
| `COURSEFORGE_OUTLINE_MODEL` | Model identifier the outline-tier provider expects (e.g. `qwen2.5:7b-instruct-q4_K_M` for Ollama, `Qwen/Qwen2.5-7B-Instruct` for vLLM, `meta-llama/Llama-3.1-8B-Instruct-Turbo` for Together). Default `qwen2.5:7b-instruct-q4_K_M` — matches the smaller `LOCAL_SYNTHESIS_MODEL` default so an out-of-box Ollama install on an 8 GB GPU runs the outline tier with no further tuning. Captured per call in the `block_outline_call` decision event so the audit trail records which 7B-class model produced each block draft. Reuses the same `LOCAL_SYNTHESIS_BASE_URL` / `LOCAL_SYNTHESIS_API_KEY` connection plumbing when the outline provider resolves to `local`. (License: model-specific — Qwen2.5-7B/14B/32B are Apache 2.0 and training-permitted; see `docs/LICENSING.md`.) No-op when `COURSEFORGE_TWO_PASS` is unset. |
| `COURSEFORGE_OUTLINE_N_CANDIDATES` | Self-consistency budget per block consumed by `Courseforge/router/router.py::CourseforgeRouter.route_with_self_consistency` via `_resolve_n_candidates`. Default `3`. Higher values trade outline-tier latency / cost for distractor-quality stability; sequential dispatch (not parallel asyncio) for the first cut so per-candidate decision capture stays straightforward — `route()` is called N times, each emit is one `block_outline_call` decision event tagged with its candidate index, and the router selects the highest-scoring passing candidate after running the validator chain against each. When every candidate fails validation across the regen budget, the surviving best-effort candidate is stamped with `escalation_marker="validator_consensus_fail"`. `route_all` (the multi-block dispatch entry point) consults the same `_resolve_n_candidates` chain, so the layer-5 default of `3` fires for every block by default; per-block-type opt-out via `block_routing.yaml::blocks.{type}.n_candidates: 1` (resolves at layer 2) short-circuits a single block-type to direct single-candidate dispatch. No-op when `COURSEFORGE_TWO_PASS` is unset. |
| `COURSEFORGE_OUTLINE_PROVIDER` | Selects the LLM backend for the outline-tier authoring surface (`Courseforge/generators/_outline_provider.py::OutlineProvider`, instantiated by `Courseforge/router/router.py::CourseforgeRouter._get_outline_provider`). Values: `local` (default — ToS-clean local OSS via the shared `OpenAICompatibleClient`; recommended for the outline-tier first draft since the output is later ingested as Trainforge training chunks), `together` (ToS-clean cloud OSS), `anthropic` (ToS-restricted; rarely the right choice for outline tier — the rewrite tier is the natural Anthropic seat). Reuses the same `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` env vars as the synthesis pipelines so one local server serves all three task surfaces. Captured per call in the `block_outline_call` decision event. Per-block-type pinning via `block_routing.yaml` overrides this tier-default (e.g. a block-type-level `outline.provider: local` mapping in YAML wins over this env var). No-op when `COURSEFORGE_TWO_PASS` is unset. (License: see `docs/LICENSING.md`.) |
| `COURSEFORGE_OUTLINE_REGEN_BUDGET` | Per-block outline-tier regeneration cap consumed by `Courseforge/router/router.py::CourseforgeRouter` via `_resolve_regen_budget`. Default `10`. Each time the inter-tier validator chain returns `action="regenerate"` against a block, `Block.validation_attempts` increments and the outline tier re-rolls; once the counter hits the budget and the chain still rejects, the block is stamped with `escalation_marker="outline_budget_exhausted"` and skips the rewrite tier. The `block_routing.yaml` per-type `escalate_immediately: true` short-circuit is a separate path that ALSO sets `escalation_marker="outline_budget_exhausted"` — provenance is carried via `Touch.purpose="escalate_immediately"` to disambiguate from the budget-exhausted path. Legacy validators (returning `action=None, passed=False`) trigger regenerate-loop semantics; only EXPLICIT `action="block"` / `action="escalate"` triggers short-circuits. No-op when `COURSEFORGE_TWO_PASS` is unset. |
| `COURSEFORGE_PROVIDER` | Selects the LLM backend for the Courseforge content-generator surface (`Courseforge/generators/_provider.py::ContentGeneratorProvider`, instantiated by `MCP/tools/pipeline_tools.py::_generate_course_content`). Values: `anthropic` (class default — ToS-restricted for training-data; mirrors `CURRICULUM_ALIGNMENT_PROVIDER`), `together` (ToS-clean cloud OSS via the shared `OpenAICompatibleClient`), `local` (ToS-clean local OSS via the shared `OpenAICompatibleClient`). **Recommended setting for ToS-clean course content is `local`** since the authored HTML is later ingested as Trainforge training chunks. Reuses the same `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` env vars as the synthesis pipeline so one local server serves both task surfaces. Default unset → the legacy deterministic-template path runs (`build_week_data` falls back to DART-paragraph synthesis with no LLM call), preserving backward compatibility on every existing run. **Short-circuit semantics:** setting `COURSEFORGE_PROVIDER` to any non-empty value overrides `ED4ALL_AGENT_DISPATCH=true` for the `content-generator` agent only — the executor falls through to the in-process provider call instead of dispatching the Claude Code subagent, while every other agent (course-outliner, oscqr-course-evaluator, etc.) keeps dispatching unchanged. Captured per call in the `content_generator_call` decision event. (License: see `docs/LICENSING.md` for the per-provider ToS + per-model layer.) |
| `COURSEPLANNER_PROVIDER` | Selects the LLM backend for the Courseforge course-outliner surface (`Courseforge/generators/_outliner_provider.py::OutlinerProvider`, instantiated by `MCP/tools/pipeline_tools.py::_plan_course_structure`). Values: `anthropic` (class default — ToS-restricted for training-data), `together`, `local`, **plus any provider registered in the W-D12 `_OPENAI_COMPATIBLE_PROVIDERS` registry at `MCP/orchestrator/llm_backend.py`** (`groq`, `fireworks`, `deepseek`, …). Adding a new OpenAI-compatible provider is a registry-entry change — **NOT** a new subclass — per the W-D12 dynamic-references contract. The class consults the registry at construction time, so a new entry flows through without a code edit here. **Recommended setting for ToS-clean course planning is `local`** (Apache 2.0 Qwen) since the synthesised objective text lands in `synthesized_objectives.json` and propagates to every downstream chunk's `learning_outcome_refs[]` — i.e. it touches training data. Reuses the same `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` env vars as `COURSEFORGE_PROVIDER` so one local server serves both task surfaces. Override the model via `COURSEPLANNER_MODEL`. Default unset → the legacy Claude Code subagent dispatch fires (when `ED4ALL_AGENT_DISPATCH=true`) or the deterministic `synthesize_objectives_from_topics` path runs, preserving backward compatibility on every existing run. **Short-circuit semantics:** setting `COURSEPLANNER_PROVIDER` to any non-empty value overrides `ED4ALL_AGENT_DISPATCH=true` for the `course-outliner` agent only — the executor falls through to `_plan_course_structure`, which constructs an `OutlinerProvider` and uses it in lieu of the deterministic synthesizer. Other Wave-74 agents (content-generator, oscqr-course-evaluator, etc.) keep dispatching unchanged. Captured per call in the `course_outline_call` decision event whose rationale interpolates the runtime provider name (NOT a static label) per the W-D12 dynamic-rationale contract. (License: see `docs/LICENSING.md` for the per-provider ToS + per-model layer.) |
| `TEXTBOOK_SYNTHESIS_PROVIDER` | Selects the LLM backend for the three-stage large-LLM textbook synthesis surface (`Courseforge/generators/_textbook_synthesis_provider.py::TextbookSynthesisProvider`, instantiated by the three post-DART phase handlers in `MCP/tools/pipeline_tools.py` — `_extract_textbook_structure` for Stage 1, `_run_concept_extraction` for Stage 3, `_plan_course_structure` for Stage 2 + reconciliation). Values: `anthropic` (class default — ToS-restricted for training-data), `together`, `local`, **plus any provider registered in the W-D12 `_OPENAI_COMPATIBLE_PROVIDERS` registry at `MCP/orchestrator/llm_backend.py`** (`groq`, `fireworks`, `deepseek`, …). Adding a new OpenAI-compatible provider is a registry-entry change — **NOT** a new subclass — per the W-D12 dynamic-references contract. **Recommended setting for a ToS-clean corpus is `local`** (Apache 2.0 Qwen): the Stage-3 domain-concept vocabulary propagates into chunk `concept_tags[]` and the Stage-1/2 synthesized objectives propagate into every downstream chunk's `learning_outcome_refs[]` — i.e. this surface touches training data. Reuses the same `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` env vars as `COURSEPLANNER_PROVIDER` so one local server serves every task surface. Override the model via `TEXTBOOK_SYNTHESIS_MODEL`. **Default unset → no in-process textbook-synthesis provider runs:** Stage 1 emits `textbook_structure.json` byte-identical (no new keys), Stage 3 runs with empty `domain_concept_seeds` (status quo), Stage 2 runs the deterministic `synthesize_objectives_from_topics` path (or the `COURSEPLANNER_PROVIDER`/`OutlinerProvider` path if THAT env var is set) with no reconciliation step — every existing run is unaffected. **Short-circuit semantics:** each of the three phase handlers checks `os.environ.get("TEXTBOOK_SYNTHESIS_PROVIDER","").strip()` for a non-empty value before constructing the provider; an empty/unset value short-circuits to the deterministic fallback. Captured per call in the `textbook_outline_call` (Stage 1), `textbook_concept_call` (Stage 3), `chapter_objective_call` (Stage 2), and `terminal_objective_reconciliation` (reconcile) decision events whose rationales interpolate the runtime-resolved provider name (NOT a static label) per the W-D12 dynamic-rationale contract. (License: see `docs/LICENSING.md` for the per-provider ToS + per-model layer.) See § "Three-stage textbook synthesis" below for the architecture. |
| `TEXTBOOK_SYNTHESIS_MODEL` | Model-ID override for the `TextbookSynthesisProvider` backend selected by `TEXTBOOK_SYNTHESIS_PROVIDER`. Resolution chain (high → low): per-call kwarg > `TEXTBOOK_SYNTHESIS_MODEL` > the per-backend baseline (`claude-sonnet-4-6` for Anthropic, `qwen2.5:14b-instruct-q4_K_M` for local, the Together OSS default for `together`). Captured per call in the four `textbook_*` decision events alongside the resolved provider. No-op when `TEXTBOOK_SYNTHESIS_PROVIDER` is unset. (License: model-specific — Qwen2.5-7B/14B/32B are Apache 2.0 and training-permitted; see `docs/LICENSING.md`.) |
| `COURSEFORGE_REWRITE_MODEL` | Model identifier the rewrite-tier provider expects (e.g. `claude-sonnet-4-6` for Anthropic, `qwen2.5:14b-instruct-q4_K_M` for a 14B local model, `meta-llama/Llama-3.3-70B-Instruct-Turbo` for Together). Default `claude-sonnet-4-6` — pedagogy-rich rewrite of a structurally-shaped outline draft is the model class where the cloud provider's instruction-following margin pays for itself, especially on `assessment_item` / `prereq_set` / `misconception` blocks where misconception-targeted distractor authoring drives the entire downstream eval signal. Captured per call in the `block_rewrite_call` decision event. Per-block-type pinning via `block_routing.yaml` overrides this default (e.g. `flip_card_grid` is structurally simple enough that the local 14B model handles rewrite at parity, and `assessment_item` MUST stay on Anthropic to avoid silent regression — the shipped policy file pins both). (License: see `docs/LICENSING.md`.) No-op when `COURSEFORGE_TWO_PASS` is unset. |
| `COURSEFORGE_REWRITE_PROVIDER` | Selects the LLM backend for the rewrite-tier authoring surface (`Courseforge/generators/_rewrite_provider.py::RewriteProvider`, instantiated by `Courseforge/router/router.py::CourseforgeRouter._get_rewrite_provider`). Values: `anthropic` (default — ToS-restricted for training-data; the rewrite tier is the natural Anthropic seat because the rewrite-tier output is the canonical published HTML, which Trainforge re-ingests as chunks but with a clear training-data licensing posture documented in `docs/LICENSING.md`), `together` (ToS-clean cloud OSS), `local` (ToS-clean — pair with a 14B+ model via `COURSEFORGE_REWRITE_MODEL` for acceptable rewrite-tier quality on multi-step reasoning blocks), `claude_session` (Wave6 — dispatches the rewrite tier to the `content-generator` Claude Code subagent via `MCP/orchestrator/local_dispatcher.py::LocalDispatcher`; subagent inherits the Wave4b `model: sonnet` frontmatter pin + Wave4-W27 MANDATORY directives from `Courseforge/agents/content-generator.md`, so a Claude Max session can drive the rewrite tier with no `ANTHROPIC_API_KEY`). The `claude_session` backend requires a `LocalDispatcher` injected by the workflow runner / MCP tool layer; standalone CLI runs raise `RuntimeError` at provider construction. Reuses the same `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` env vars as the synthesis pipelines. Captured per call in the `block_rewrite_call` decision event. Per-block-type pinning via `block_routing.yaml` overrides this tier-default. No-op when `COURSEFORGE_TWO_PASS` is unset. (License: see `docs/LICENSING.md`.) |
| `COURSEFORGE_REWRITE_REGEN_BUDGET` | Per-block rewrite-tier regeneration cap consumed by `Courseforge/router/router.py::CourseforgeRouter._resolve_rewrite_regen_budget` and the `route_rewrite_with_remediation` loop. Default `10` — matches `COURSEFORGE_OUTLINE_REGEN_BUDGET` because the post-rewrite validator chain runs the same four shape-discriminating `Block*Validator` adapters as the inter-tier seam (CURIE / content_type / page_objectives / source_refs), so the same budget makes sense across both seams. Each time the post-rewrite validator chain returns `action="regenerate"` against a block, the rewrite tier re-rolls with the remediation suffix appended to the prompt (`Courseforge/router/remediation.py`); once the budget is exhausted and the chain still rejects, the surviving best-effort candidate is stamped with `escalation_marker="validator_consensus_fail"` (semantic mirror of the outline-tier consensus-failure marker — the consumer-side handling is identical so the marker is reused rather than minting a new value). Per-block-type override via `block_routing.yaml`'s `regen_budget_rewrite` field wins over this env var. No-op when `COURSEFORGE_TWO_PASS` is unset. |
| `COURSEFORGE_TWO_PASS` | Two-pass router master gate: `true` / `false`, default `false`. When unset (or any value other than `true` / `1` / `yes` / `on` — case-insensitive truthy match per `Courseforge/scripts/blocks.py::_EMIT_BLOCKS_TRUTHY` at `:40`), the legacy single-pass `content_generation` workflow phase runs unchanged via the `ContentGeneratorProvider` — every existing run is unaffected. When truthy, `MCP/core/workflow_runner.py::_should_skip_phase` honours the `enabled_when_env: "COURSEFORGE_TWO_PASS=true"` predicate on the two-pass phases (`content_generation_outline` → `inter_tier_validation` → `content_generation_rewrite`) and skips the legacy phase via its `enabled_when_env: "COURSEFORGE_TWO_PASS!=true"` predicate. The two-pass pipeline instantiates `Courseforge/router/router.py::CourseforgeRouter` in-process inside `MCP/tools/pipeline_tools.py::_generate_course_content` and dispatches each `Block` through outline → inter-tier validators → rewrite. Setting this var implicitly opts into the in-process content-generator path (same short-circuit semantics as `COURSEFORGE_PROVIDER`). Captured at workflow scope by the new `phase` enum values `courseforge-content-generator-outline` and `courseforge-content-generator-rewrite`. Cross-link: canonical workflow definition at `config/workflows.yaml::textbook_to_course` (and `course_generation`); architecture overview at `Courseforge/CLAUDE.md` § "Two-pass router". |

---

## MCP Tools

Courseforge is exposed via the Ed4All MCP server with these tools:

| Tool | Description |
|------|-------------|
| `create_course_project` **[DEPRECATED]** | Initialize a standalone (non-pipeline) course project. Still functional for external MCP clients, but new integrations should route through the pipeline-internal `extract_textbook_structure` + `plan_course_structure`. |
| `generate_course_content` | Generate content for weeks |
| `package_imscc` | Package course as IMSCC. Runtime delegates to `Courseforge/scripts/package_multifile_imscc.py` (IMS CC v1.3 namespaces, per-week LO validation, `course_metadata.json` bundling). |
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
| `data-cf-source-ids` | `<section>`, headings, component wrappers | DART `sourceId`(s) that ground this block. Shape: `dart:{slug}#{block_id}`. Carried through from DART's `data-dart-block-id` when source material is present; elided when no source grounding exists. |
| `data-cf-source-primary` | `<section>`, headings, component wrappers | The primary `sourceId` for the block (subset of `data-cf-source-ids`) when one source dominates. |
| `data-cf-block-id` | every block-bearing wrapper (`<section>`, headings, component wrappers) | Stable Block ID for cross-referencing JSON-LD `blocks[]` (gated behind `COURSEFORGE_EMIT_BLOCKS`). Shape: `{page_id}#{block_type}_{slug}_{idx}` per `Courseforge/scripts/blocks.py::Block.stable_id`. |
| `data-cf-curie` | appended hidden `<span>` | Space-separated minted CURIE(s) force-injected by `RewriteProvider` when the rewrite LLM dropped a source block's CURIEs after the remediation budget. Injected as an appended `<span hidden data-cf-curie="…">…</span>` whose **text content** carries the CURIE tokens — the str-path validator strips HTML tags (and their attributes) before scraping, so only text content survives; the `hidden` attribute keeps the span out of the render + accessibility tree. Absent on RDF corpora whose rewrite output retains CURIEs in prose, and on any block the LLM authored cleanly. See § "Three-stage textbook synthesis" → dynamic CURIE minting. |

Attributes stop at the **section / component wrapper level** — never on every `<p>` / `<li>` / `<tr>` in prose.

### Ancestor-walkable grounding

`ContentGroundingValidator` walks each non-trivial `<p>` / `<li>` / `<figcaption>` / `<blockquote>`'s ancestor chain to find the first `data-cf-source-ids` attribute. Three emit-side contracts keep that walk passing:

1. **Content sections are wrapped in `<section data-cf-source-ids="…">`.** `Courseforge/scripts/generate_course.py::_render_content_sections` wraps each h2/h3 + paragraph group in a `<section>` carrying the section's resolved source-ids.
2. **`content_NN` pages inherit `content_01` grounding.** `_page_refs_for` falls back from `content_NN` → `content_01` in the `source_module_map`. The source-router emits a single per-week `content_01` entry; every generated content page in that week shares the same DART source region.
3. **Objectives `<section>` mirrors page-level source-ids.** `_render_objectives(..., source_ids=…)` stamps the page's resolved source-ids onto the `.objectives` wrapper so the injected objectives section carries the same grounding.

DART-side slug contract (see `DART/CLAUDE.md`): the `dart:{slug}#{block_id}` slug uses `lowercase + space-to-hyphen` normalization (not `canonical_slug`'s underscore collapse), matching the validator's `_resolve_valid_block_ids` rule.

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

## Block format

Every page-level pedagogical unit (objective, concept, example, callout, flip card, self-check question, activity, …) is constructed as a frozen `Block` dataclass first, then projected to HTML via `Block.to_html_attrs()` and to a JSON-LD entry via `Block.to_jsonld_entry()`. Mutations return a new instance via `dataclasses.replace`; the `with_touch` helper appends to the immutable `touched_by` audit chain.

- **Dataclass + 16-value `BLOCK_TYPES` enum**: `Courseforge/scripts/blocks.py` (`Block` frozen dataclass + `BLOCK_TYPES` frozenset — `objective`, `concept`, `example`, `assessment_item`, `explanation`, `prereq_set`, `activity`, `misconception`, `callout`, `flip_card_grid`, `self_check_question`, `summary_takeaway`, `reflection_prompt`, `discussion_prompt`, `chrome`, `recap`).
- **Canonical JSON-LD shape**: `schemas/knowledge/courseforge_jsonld_v1.schema.json` (`$defs.Block`, `$defs.Touch`, top-level optional `blocks[]` / `provenance` / `contentHash`).

When `COURSEFORGE_EMIT_BLOCKS=true`, `Courseforge/scripts/generate_course.py::_build_page_metadata` emits three additional top-level JSON-LD fields per page:

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
- `Courseforge/generators/_outline_provider.py::OutlineProvider` — defaults: `local` provider, `qwen2.5:7b-instruct-q4_K_M` model, JSON-mode + lenient parse + grammar-aware backend payload (GBNF / JSON-Schema / vLLM guided / `format: json`).
- `Courseforge/generators/_rewrite_provider.py::RewriteProvider` — defaults: `anthropic` provider, `claude-sonnet-4-6` model. Per-block-type pinning via `block_routing.yaml` (e.g. `assessment_item` always rewrite-tier Anthropic, `flip_card_grid` local).
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

Legacy validators returning `action=None, passed=False` retain regenerate-loop semantics; only EXPLICIT `action="block"` / `action="escalate"` triggers a short-circuit.

### Touch chain

`_TOUCH_TIERS` includes `outline`, `outline_val`, `rewrite`, `rewrite_val`. The canonical post-validation Touch chain on a clean two-pass run is `outline → outline_val → rewrite → rewrite_val`. JSON-LD + SHACL Touch.tier enums carry the same values so downstream consumers (Trainforge chunk extraction, training-data export) can filter on tier without string matching.

### Statistical-tier validators

Layered on top of the structural seam (CURIE / content_type / page_objectives / source_refs), the statistical tier catches semantic drift — output that parses cleanly but says the wrong thing. Wired symmetrically at both `inter_tier_validation` and `post_rewrite_validation`. Default thresholds in `docs/validation/validators.md`.

- `lib/validators/objective_assessment_similarity.py::ObjectiveAssessmentSimilarityValidator` — cosine-similarity floor between assessment-item stem and referenced LO text.
- `lib/validators/concept_example_similarity.py::ConceptExampleSimilarityValidator` — cosine-similarity floor between concept definition and illustrating example.
- `lib/validators/objective_roundtrip_similarity.py::ObjectiveRoundtripSimilarityValidator` — cosine-similarity floor between rewrite-tier LO paraphrase and source objective.
- `lib/validators/courseforge_outline_shacl.py::CourseforgeOutlineShaclValidator` — wrapper around `schemas/context/courseforge_v1.shacl-rules.ttl` shape constraints, applied to outline-tier Block emit.
- `lib/validators/bloom_classifier_disagreement.py::BloomClassifierDisagreementValidator` — wraps `lib/classifiers/bloom_bert_ensemble.py::BloomBertEnsemble` (three SHA-pinned HuggingFace classifiers vote on Bloom level). Fires `action="regenerate"` on (a) ensemble majority disagrees with declared `bloomLevel` (`bert_ensemble_disagreement` event) or (b) ensemble dispersion above `_DISPERSION_THRESHOLD = 0.7` (`bert_ensemble_dispersion_high` event). See `docs/validation/validators.md` for the BERT ensemble member list.

The embedding wrapper at `lib/embedding/` degrades gracefully when the optional `[embedding]` extras are absent (warning-severity `EMBEDDING_DEPS_MISSING` GateIssue, `passed=True, action=None`); set `TRAINFORGE_REQUIRE_EMBEDDINGS=true` to fail-closed in production.

### Decision-capture events

Per-tier and per-decision; every router-side choice and every LLM call lands as a typed event:

- `block_outline_call` (per outline-tier LLM call, emitted by `OutlineProvider`).
- `block_rewrite_call` (per rewrite-tier LLM call, emitted by `RewriteProvider`).
- `block_validation_action` (per validator-chain run, emitted by `CourseforgeRouter`). Carries a `tier` field (`"outline"` | `"rewrite"`) disambiguating the two seams.
- `block_escalation` (per terminal escalation: budget exhausted, structural-unfixable, consensus failure).
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
- `MCP/tools/pipeline_tools.py::_run_concept_extraction` — phase handler. Reads staged DART chunks via `Trainforge.chunker.chunk_content`, invokes `Trainforge.pedagogy_graph_builder.build_pedagogy_graph`, persists the graph to `LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json`, computes `concept_graph_sha256`, routes the hash through `phase_outputs.concept_extraction.concept_graph_sha256`.
- `lib/ontology/concept_objective_linker.py::link_concepts_to_objectives` — invoked from `_plan_course_structure` after objective synthesis. Two-stage match: (1) substring match between concept-graph node slugs and the LO's existing `keyConcepts[]`; (2) for unmatched nodes, scan the LO statement for verbatim concept-slug occurrence.

---

## Three-stage textbook synthesis

An operator-locked, large-LLM, three-stage synthesis architecture mapped onto the three existing post-DART phases of `config/workflows.yaml::textbook_to_course`. It lifts course outline, domain-concept vocabulary, and per-chapter objective authoring off the deterministic small-model paths and onto a single configurable large LLM. The deterministic paths remain the canonical no-LLM fallback. Full design: `plans/textbook-llm-synthesis-3stage-2026-05.md`.

The three stages (pipeline order is Stage 1 → Stage 3 → Stage 2 + reconcile — the passes are independent):

- **Stage 1 — outline synthesis** (`objective_extraction`, handler `_extract_textbook_structure`): one call over the whole-textbook chapter/section skeleton → a course-level `semantic_outline` + a set of DRAFT `TO-NN` terminal objectives. Folds three optional top-level keys into `textbook_structure.json` (`semantic_outline`, `draft_terminal_objectives`, `structure_enrichment`).
- **Stage 3 — domain-concept vocabulary** (`concept_extraction`, handler `_run_concept_extraction`): N per-chapter calls (batched ≤10) fed each chapter's full `chapter_text` → a course-level domain-concept vocabulary. The vocabulary is compiled into `domain_concept_seeds` and used to re-tag the loaded DART chunks in-memory before the co-occurrence graph build, so the semantic graph finally carries real `DomainConcept` nodes. Persists `domain_concept_vocabulary.json` as a sibling of `concept_graph_semantic.json`.
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

Two provenance-anchored chunk surfaces emit per course: a **DART chunkset** (rooted in the textbook PDF) before objective extraction, and a **IMSCC chunkset** (rooted in the packaged IMSCC) post-packaging. The canonical chunker at `Trainforge/chunker/` is shared by DART, Courseforge, and Trainforge.

- **DART chunkset**: `chunking` workflow phase (between `staging` and `objective_extraction`). Helper: `MCP/tools/pipeline_tools.py::_run_dart_chunking`. Agent: `dart-chunker` (utility-style, no LLM dispatch; routes via `AGENT_TOOL_MAPPING["dart-chunker"] = "run_dart_chunking"`). Persists `LibV2/courses/<slug>/dart_chunks/chunks.jsonl` + sibling `manifest.json`. Emits `dart_chunks_path` + `dart_chunks_sha256` through `phase_outputs.chunking`.
- **IMSCC chunkset**: `imscc_chunking` workflow phase (between `packaging` and `training_synthesis`). Helper: `MCP/tools/pipeline_tools.py::_run_imscc_chunking`. Reads HTML entries in-memory from the packaged `.imscc` zip via `zipfile.ZipFile`. Emits `chunkset_kind="imscc"` plus `source_imscc_sha256` (SHA-256 of the archive bytes). Persists at `LibV2/courses/<slug>/imscc_chunks/`.
- **Sidecar manifest schema**: `schemas/library/chunkset_manifest.schema.json`. Symmetric across DART and IMSCC: `chunkset_kind` enum (`"dart"` | `"imscc"`) discriminator + conditional source-SHA requirement (`source_dart_html_sha256` for DART, `source_imscc_sha256` for IMSCC). Required: `chunks_sha256`, `chunker_version` (resolved from `Trainforge.chunker.CHUNKER_SCHEMA_VERSION`), `chunkset_kind`. Optional: `chunks_count`, `generated_at`.
- **Chunkset-manifest gate**: `lib/validators/chunkset_manifest.py::ChunksetManifestValidator` fires at both chunking phases. Verifies manifest existence + schema + `chunks_sha256` round-trip + `chunker_version` match + conditional source-SHA. GateIssue codes: `MANIFEST_MISSING`, `MANIFEST_PARSE_ERROR`, `MANIFEST_SCHEMA_INVALID`, `CHUNKS_SHA256_MISMATCH`, `CHUNKER_VERSION_MISMATCH`, `SOURCE_SHA256_MISSING`.
- **Course-manifest hash triangle**: `lib/validators/libv2_manifest.py::LibV2ManifestValidator` fail-closes at `libv2_archival` on any of three required hashes missing, malformed, or divergent: `dart_chunks_sha256`, `imscc_chunks_sha256`, `concept_graph_sha256`. Each fires a `MISSING_*` / `INVALID_*` / `*_HASH_MISMATCH` GateIssue triplet.
- **Backfill for legacy archives**: `LibV2/tools/libv2/scripts/backfill_dart_chunks.py` migrates pre-chunkset archives (no `dart_chunks/` directory). Idempotent by default; `--force` for re-emit, `--dry-run` for plan-only.

---

## Operator stage subcommands

Four operator-facing subcommands re-drive the Courseforge two-pass pipeline one tier at a time without re-executing the upstream `dart_conversion → staging → chunking → objective_extraction → source_mapping → course_planning → concept_extraction` chain. Use case: a previous full run produced an OUTLINE_DIR; the operator wants to re-run only the rewrite tier under a different teacher model, re-run validation after tweaking a gate threshold, or A/B-test outline-tier model swaps.

The four subcommands route through the canonical `textbook_to_course` workflow with the `courseforge_stage` workflow param set; the workflow runner pre-populates upstream phase outputs via `_synthesize_outline_output` and skips non-whitelisted phases via `_should_skip_phase`:

| Subcommand | Active phases (executed) | Skipped via whitelist |
|---|---|---|
| `courseforge-outline` | `content_generation_outline` | inter_tier_validation, content_generation_rewrite, post_rewrite_validation |
| `courseforge-validate` | `inter_tier_validation`, `post_rewrite_validation` | content_generation_outline, content_generation_rewrite |
| `courseforge-rewrite` | `content_generation_rewrite`, `post_rewrite_validation` | content_generation_outline, inter_tier_validation |
| `courseforge` | all four | (none — full two-pass slice) |

Pre-Courseforge phases pre-populate from the project export root via `_synthesize_outline_output`; their `_completed=True` markers fire the runner's already-completed skip path. Post-Courseforge phases (packaging, imscc_chunking, trainforge_assessment, training_synthesis, libv2_archival, finalization) skip via the `courseforge_stage` whitelist regardless of which subcommand fired — Phase 5 is intentionally scoped to the Courseforge two-pass surface only. Operators who want to re-run a post-Courseforge phase use the canonical `ed4all run textbook-to-course` entry point.

### CLI flags (at `cli/commands/run.py`)

- `--blocks <comma-separated>` — per-block re-execution scope. Tokens must come from the canonical 16-singular `BLOCK_TYPES` enum (`Courseforge/scripts/blocks.py:77`); unknown tokens fail fast at parse time. The rewrite tier reads the list via `target_block_ids` workflow param and re-rolls only blocks whose `block_type` matches; every other block is byte-identical to the input. Validate-tier subcommands ignore `--blocks`. Dry-run plan annotates the rewrite phase with `<FILTERED:assessment_item,...>`.
- `--force` — re-run phases despite a pre-existing `_completed` checkpoint. The synthesizer pre-populates upstream phases with `_completed=True`; `--force` flips that to `False` so the phase loop re-executes them.

### `02_validation_report/report.json` writer

The `_run_inter_tier_validation` and `_run_post_rewrite_validation` phase helpers emit JSONL only — `blocks_validated.jsonl` + `blocks_failed.jsonl` next to the consumed Block file. The operator-facing structured per-block summary lives at:

- `<project_root>/02_validation_report/report.json` for the outline tier's `inter_tier_validation` phase emit.
- `<project_root>/04_rewrite/02_validation_report/report.json` for the rewrite tier's `post_rewrite_validation` phase emit.

The writer fires automatically after each validation phase completes inside `WorkflowRunner.run_workflow` (best-effort — filesystem failures are warning-logged and don't abort the run). Schema (`_VALIDATION_REPORT_SCHEMA_VERSION = "v1"`):

```json
{
  "run_id": "WF-...",
  "phase": "inter_tier_validation",
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
        {"gate_id": "...", "action": "...", "passed": false, "issue_count": 2}
      ],
      "escalation_marker": "outline_budget_exhausted | null"
    }
  ]
}
```

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
| `generate_course.py` | `scripts/` | Multi-file weekly course generation. Emits page-level JSON-LD, `course_metadata.json`, prerequisite-page refs, `data-cf-teaching-role`, and `data-cf-source-ids` / page-level `sourceReferences` when DART source material is staged. Accepts `--emit-mode {full,outline}` (default `full`); outline mode strips content/example/assessment HTML bodies but preserves their JSON-LD `blocks[]` projections, and stamps `course_metadata.blocks_summary.outline_only=true` so downstream consumers can detect the tier. |
| `package_multifile_imscc.py` | `scripts/` | Packages multi-file output into IMSCC. Structural validation is on by default (per-week `learningObjectives` must resolve to the week's LO manifest). Auto-discovers `course.json` and bundles `course_metadata.json` at the zip root. Manifest uses IMS Common Cartridge v1.3 namespaces; resources are nested under per-week `<item>` wrappers in the organization tree. **This is the runtime target of the MCP `package_imscc` tool** — `MCP/tools/pipeline_tools.py::_package_imscc` imports and delegates here instead of hand-rolling a ZIP. Accepts `--outline-only` to package an outline-tier deliverable; reads `course_metadata.blocks_summary.outline_only` written by `generate_course.py --emit-mode outline`. |

`--emit-mode outline` (`generate_course.py`) and `--outline-only` (`package_multifile_imscc.py`) produce a stripped-down deliverable carrying only objectives + summaries; content/example/assessment HTML bodies are dropped while their JSON-LD `blocks[]` entries persist for downstream consumers (Trainforge `process_course.py` skips `instruction_pair` extraction when `course_metadata.blocks_summary.outline_only=true`). Outline mode is the input shape the two-pass pipeline expects from the outline tier.

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
