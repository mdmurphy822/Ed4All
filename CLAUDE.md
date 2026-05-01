# Ed4All Hybrid Orchestrator

Unified orchestration system for DART, Courseforge, Trainforge, and LibV2.

## Quick Start

### Canonical entry point

```bash
# Primary: run any workflow end-to-end via the unified CLI
ed4all run <workflow_name> --corpus <PATH> --course-name <NAME> [--mode local|api]

# Examples
ed4all run textbook-to-course --corpus textbook.pdf --course-name PHYS_101
ed4all run textbook-to-course --corpus ./pdfs/ --course-name BIO_201 --weeks 16
ed4all run rag_training --corpus course.imscc --course-name CHEM_101 --mode api
ed4all run textbook-to-course --corpus x.pdf --course-name T --dry-run   # plan only
ed4all run textbook-to-course --resume WF-20260420-abc12345               # resume

# Wave 80: pin the course_planning phase to a previously-synthesized
# objectives JSON instead of re-dispatching the course-outliner
# subagent. Eliminates LLM-nondeterminism drift across re-runs that
# breaks chunk learning_outcome_refs continuity. Accepts both
# Courseforge synthesized form (terminal_objectives/chapter_objectives)
# and Wave 75 LibV2 archive form (terminal_outcomes/component_objectives);
# the runner normalizes to the Courseforge form on disk before
# downstream phases consume it.
ed4all run textbook-to-course --corpus pdfs/ --course-name PHYS_101 \
  --reuse-objectives Courseforge/exports/PROJ-PHYS_101-.../01_learning_objectives/synthesized_objectives.json
```

Modes:

- `--mode local` (default): uses the current Claude Code session as the LLM;
  no API key required. Phase workers are dispatched as subagents.
- `--mode api`: uses the Anthropic SDK directly (requires `ANTHROPIC_API_KEY`).
  Workers run as Python coroutines and call the SDK directly.

Environment toggles (override or supplement CLI flags):

| Env Var | Default | Purpose |
|---------|---------|---------|
| `LLM_MODE` | `local` | Chooses `local` or `api` if `--mode` isn't passed. |
| `LLM_PROVIDER` | `anthropic` | Provider in api mode (`anthropic` or `openai`; `openai` is stubbed, reserved for a later wave). |
| `LLM_MODEL` | per-provider | Model ID override (e.g., a specific Claude release). |
| `ANTHROPIC_API_KEY` | — | Required for api mode with Anthropic. |
| `OPENAI_API_KEY` | — | Reserved; OpenAI backend not yet implemented. |

### MCP Server
```bash
cd MCP
python server.py
```

### Available Workflows
| Workflow | Description | Max Concurrent |
|----------|-------------|----------------|
| `textbook_to_course` | Full PDF → Course → Assessments pipeline | 10 |
| `course_generation` | Generate new course from objectives | 10 |
| `intake_remediation` | Import and remediate IMSCC | 4 |
| `batch_dart` | Batch PDF to HTML conversion | 4 |
| `rag_training` | Trainforge assessment generation | 5 |
| `trainforge_train` | Train a course-pinned SLM adapter (Wave 90; post-import LibV2 stage) | 1 |

---

## Project Structure

```
Ed4All/
├── DART/                    # PDF to accessible HTML conversion
├── Courseforge/             # Course content generation & packaging
├── Trainforge/              # Assessment-based RAG training
├── LibV2/                   # Course content repository
│   ├── courses/             # Educational content storage
│   ├── catalog/             # Derived indexes
│   └── tools/               # CLI & retrieval engine
├── MCP/
│   ├── server.py            # FastMCP server (core file tools)
│   ├── tools/               # Domain tool modules
│   ├── core/                # Orchestrator config, executor, workflow runner
│   ├── hardening/           # Error classifier, validation gates, checkpointing
│   ├── ipc/                 # Inter-process status tracking
│   └── tests/               # MCP tool & orchestrator tests
├── cli/                     # CLI commands (ed4all entry point)
├── lib/                     # Shared libraries & validators
├── config/                  # Workflow & agent configs
├── schemas/                 # JSON schemas for validation
├── state/                   # Shared state & progress tracking
├── training-captures/       # Decision capture output
├── ci/                      # CI integrity checks
└── .github/                 # CI/CD workflows
```

---

## Orchestrator Protocol

### Phase 1: Planning (NO EXECUTION)

Planning agent creates comprehensive todo list:
- Analyze requirements
- Break into discrete tasks
- Assign to appropriate agents
- **NO file creation, NO code execution**

### Phase 2: Load TodoWrite

TodoWrite is the **single source of truth**:
- All agents read from TodoWrite
- All agents update TodoWrite
- Status tracking: `pending` -> `in_progress` -> `completed`

### Phase 3: Batch Execution

Execute via parallel agent dispatch:
- **Maximum 10 simultaneous Task calls per batch**
- Wait for ALL batch completions before next batch
- Use `poll_task_completions()` to check status

### Phase 4: Quality Validation

Every artifact validated before finalization:
- DART: WCAG compliance check
- Courseforge: IMSCC validation
- Trainforge: Assessment quality scoring

### Phase 5: Packaging

Final packaging and export:
- Update GENERATION_PROGRESS.md
- Export training captures
- Archive logs

---

## Decision Capture

### CRITICAL REQUIREMENT

**ALL Claude decisions MUST be logged** to `training-captures/` in JSONL format.

### Required Fields

Every decision event MUST include:
- `decision_type`: Category of decision (e.g., `content_selection`, `question_generation`)
- `decision`: The actual choice made
- `rationale`: Why this decision was made (**minimum 20 characters**)

### Using Decision Capture

```python
from lib.decision_capture import DecisionCapture

capture = DecisionCapture(
    course_code="INT_101",
    phase="content-generator",
    tool="courseforge",
    streaming=True
)

capture.log_decision(
    decision_type="content_structure",
    decision="Use 6-week modular structure",
    rationale="Aligns with competency-based approach and allows flexible pacing for diverse learners",
    alternatives_considered=[
        "8-week linear: Too rigid for self-paced learning",
        "4-week intensive: Insufficient depth for foundational content"
    ]
)
```

### Output Locations

```
training-captures/
├── dart/{COURSE_CODE}/
│   └── decisions_{PDF_NAME}_{TIMESTAMP}.jsonl
├── courseforge/{COURSE_CODE}/
│   ├── phase_input-research/
│   ├── phase_content-generator/
│   └── phase_brightspace-packager/
└── trainforge/{COURSE_CODE}/
    ├── phase_content-analysis/
    ├── phase_question-generation/
    └── phase_validation/
```

---

## Individual File Protocol (MANDATORY)

### ONE Agent = ONE File

Each agent works on exactly ONE file at a time:
- No shared file editing
- No concurrent writes to same file
- Use file locking for state files

### Maximum Parallelism

```
Maximum 10 simultaneous Task calls per batch
```

### Batch Completion

Wait for ALL tasks in batch to complete:
```python
# CORRECT: Wait for batch
tasks = [dispatch_agent_task(...) for i in range(10)]
await poll_task_completions(workflow_id)  # Wait for all

# WRONG: Fire and forget
for i in range(50):
    dispatch_agent_task(...)  # No waiting!
```

---

## MCP Tool Reference

### Core File Tools

| Tool | Description |
|------|-------------|
| `list_directory` | List directory contents (READ_ONLY sandbox) |
| `read_file` | Read file contents (READ_ONLY sandbox) |
| `write_file` | Write to files (RESTRICTED sandbox: runtime/, state/) |
| `file_info` | Get file/directory metadata (READ_ONLY sandbox) |

### DART Tools

| Tool | Description |
|------|-------------|
| `convert_pdf_multi_source` | Convert PDF via multi-source synthesis |
| `batch_convert_multi_source` | Batch convert multiple PDFs |
| `validate_wcag_compliance` | Validate HTML accessibility |
| `get_dart_status` | Get DART processing status |
| `list_available_campuses` | List configured campus sources |
| `extract_and_convert_pdf` | Extract and convert a single PDF |

### Courseforge Tools

| Tool | Description |
|------|-------------|
| `create_course_project` **[DEPRECATED Wave 28e]** | Initialize a standalone (non-pipeline) course project. Still functional for external MCP clients, but new integrations should route through the pipeline-internal `extract_textbook_structure` + `plan_course_structure` (Wave 24). |
| `generate_course_content` | Generate content for weeks |
| `package_imscc` | Package course as IMSCC. Runtime delegates to `Courseforge/scripts/package_multifile_imscc.py` (IMS CC v1.3 namespaces, per-week LO validation, `course_metadata.json` bundling). |
| `intake_imscc_package` | Import existing IMSCC |
| `remediate_course_content` | Fix content issues |
| `get_courseforge_status` | Get project status |

### Courseforge Metadata Output

Courseforge HTML pages include machine-readable metadata for downstream Trainforge consumption:
- **`data-cf-*` attributes**: Inline metadata on HTML elements (role, objective IDs, Bloom's levels/verbs, cognitive domain, content types, teaching role, key terms, component, purpose). See `Courseforge/CLAUDE.md` for the canonical attribute table.
- **JSON-LD blocks**: Structured `<script type="application/ld+json">` per page with learning objectives, section metadata, misconceptions, and assessment suggestions. Canonical shape: `schemas/knowledge/courseforge_jsonld_v1.schema.json`.

This metadata follows priority extraction in Trainforge: JSON-LD > data-cf-* attributes > regex heuristics.

### DART Source-Provenance Output

DART-produced HTML + synthesized JSON carry per-block source attribution so downstream consumers can trace every claim back to its PDF origin:
- **`data-dart-*` attributes**: `data-dart-block-id`, `data-dart-source`, `data-dart-sources`, `data-dart-pages`, `data-dart-confidence`, `data-dart-strategy` on `<section>` + component wrappers. See `DART/CLAUDE.md` § "Source provenance" for the canonical attribute table + confidence scale.
- **Per-block envelopes** in `*_synthesized.json` `data.contacts[]`, `data.rows[]`, `data.pair_provenance[]`: `{value, source, pages, confidence, method}` shape.
- **Canonical shape**: `schemas/knowledge/source_reference.schema.json` (shared by Courseforge JSON-LD and Trainforge chunks + evidence arms).

Priority extraction chain (extends the Courseforge chain above): JSON-LD > `data-cf-*` > `data-dart-*` > regex heuristics.

### Orchestrator Tools

| Tool | Description |
|------|-------------|
| `create_workflow` | Create new workflow instance |
| `get_workflow_status` | Check workflow progress |
| `dispatch_agent_task` | Dispatch task to agent |
| `poll_task_completions` | Wait for task completions |
| `execute_workflow_task` | Execute a single workflow task |
| `complete_workflow_task` | Mark workflow task complete |
| `update_generation_progress` | Update progress file |
| `acquire_batch_lock` | Lock resource for batch |
| `release_batch_lock` | Release batch lock |

### Trainforge Tools

| Tool | Description |
|------|-------------|
| `analyze_imscc_content` | Analyze IMSCC for assessment |
| `generate_assessments` | Generate questions |
| `validate_assessment` | Validate assessment quality |
| `export_training_data` | Export training captures |
| `get_trainforge_status` | Get processing status |
| `synthesize_training` | Synthesize SFT + DPO training pairs from a Trainforge corpus. `provider` accepts `"mock"`, `"anthropic"`, `"claude_session"`, `"together"` (Together AI's OSS-teacher path; ToS-clean for training-data generation, default model `meta-llama/Llama-3.3-70B-Instruct-Turbo`, override via `TOGETHER_SYNTHESIS_MODEL`; requires `TOGETHER_API_KEY`), or `"local"` (a local OpenAI-compatible model server such as Ollama / vLLM / llama.cpp / LM Studio; default base URL `http://localhost:11434/v1`, override via `LOCAL_SYNTHESIS_BASE_URL`; default model `qwen2.5:14b-instruct-q4_K_M`, override via `LOCAL_SYNTHESIS_MODEL`; API key optional). |

### Pipeline Tools

| Tool | Description |
|------|-------------|
| `stage_dart_outputs` | Stage DART outputs for Courseforge |
| `get_pipeline_status` | Check pipeline progress |
| `validate_dart_markers` | Validate DART output markers |
| `archive_to_libv2` | Archive course artifacts to LibV2 |

**Pipeline-internal registry-only tools** (wired into `MCP/tools/pipeline_tools.py::_build_tool_registry` for workflow-phase dispatch; intentionally **not** decorated with `@mcp.tool()` — not reachable from external MCP clients):

| Tool | Phase | Purpose |
|------|-------|---------|
| `build_source_module_map` | `source_mapping` | TF-IDF-driven router that maps DART source blocks to Courseforge module pages. Output: `source_module_map.json`. |
| `extract_textbook_structure` | `objective_extraction` | Runs `SemanticStructureExtractor` over every staged DART HTML file and merges per-file chapter/section hierarchies into a single `textbook_structure.json`. |
| `plan_course_structure` | `course_planning` | Synthesizes canonical `TO-NN` / `CO-NN` learning objectives from the textbook structure and publishes `synthesized_objectives.json`. |

**Removed in Wave 28f**: the deprecated `create_textbook_pipeline_tool`
/ `run_textbook_pipeline_tool` MCP tools and the `ed4all
textbook-to-course` top-level CLI subcommand have been deleted. Use
`create_workflow(workflow_type="textbook_to_course", ...)` via
`cli/commands/run.py`, or the canonical `ed4all run
textbook-to-course --corpus ... --course-name ...`.

### Analysis Tools

| Tool | Description |
|------|-------------|
| `analyze_training_data` | Analyze training capture data |
| `get_quality_distribution` | Get quality score distribution |
| `preview_export_filter` | Preview export filter results |

---

## Shared State

### GENERATION_PROGRESS.md

Location: `state/GENERATION_PROGRESS.md`

Central progress tracking file:
- Active workflows table
- Component status tables
- Batch locks table
- Error log

### File-Based IPC

Use `StatusTracker` for multi-terminal coordination:
```python
from MCP.ipc.status_tracker import StatusTracker

tracker = StatusTracker("state/status")
tracker.set_status("W001", "content_generator", "Module_3.html", "IN_PROGRESS")
```

---

## Workflow Execution

### Course Generation Workflow

```
1. planning
   └── Course outline, week structure, objectives mapping

2. content_generation
   └── Generate all modules (parallel batches of 10)

3. packaging
   └── Create IMSCC package

4. validation
   └── QA checks, accessibility, structure

5. finalization
   └── Export captures, archive logs
```

### Intake Remediation Workflow

```
1. parsing
   └── Extract IMSCC contents

2. analysis
   └── Identify issues, plan remediation

3. remediation
   └── Fix identified issues

4. validation
   └── Verify fixes

5. packaging
   └── Repackage IMSCC
```

### RAG Training Workflow (Trainforge)

```
1. extraction
   └── Parse IMSCC, extract content & learning objectives

2. indexing
   └── Build vector index for RAG retrieval

3. assessment_generation
   └── Generate questions with full decision capture

4. validation
   └── Validate assessment quality and Bloom's alignment
```

### Textbook-to-Course Workflow

```
1. dart_conversion
   └── Convert PDF textbooks to accessible HTML (multi-source synthesis)

2. staging
   └── Stage DART outputs to Courseforge inputs

3. objective_extraction
   └── Parse staged DART HTML into textbook_structure.json (chapters,
       sections, content blocks); auto-scales duration_weeks to max(8,
       chapters) when --weeks is unset.

4. source_mapping
   └── Map DART source blocks to Courseforge module pages; emits
       source_module_map.json consumed by content_generation.

5. course_planning
   └── Synthesize canonical TO-NN / CO-NN learning objectives from
       textbook_structure; emits synthesized_objectives.json.

6. content_generation
   └── Generate course content modules (parallel batches of 10). Every
       emitted sourceId must resolve against the DART staging manifest
       (source_refs gate).

7. packaging
   └── Package course as IMSCC via the mature multi-file packager.

8. trainforge_assessment (optional)
   └── Generate assessments from the IMSCC package. Fails closed if any
       assessment objective_id isn't covered by a chunk's
       learning_outcome_refs.

9. training_synthesis (optional)
   └── Synthesize instruction + preference training pairs from the
       generated chunks + assessments. Routes via the
       `training-synthesizer` agent (tool: `synthesize_training`).
       Optional phase: skipped when no `ANTHROPIC_API_KEY` or when
       `--skip-training` is passed on the CLI.

10. libv2_archival
   └── Archive course artifacts to LibV2 (raw PDFs, DART HTML, IMSCC,
       RAG corpus). Gated by libv2_manifest integrity checks.

11. finalization
   └── Final validation and training data export.
```

---

## Agent Registry

### Courseforge Agents

| Agent | Purpose |
|-------|---------|
| `course-outliner` | Create course structure |
| `requirements-collector` | Gather specifications & prerequisites |
| `content-generator` | Generate module content |
| `brightspace-packager` | Package for Brightspace LMS |
| `oscqr-course-evaluator` | OSCQR quality evaluation |
| `quality-assurance` | Pattern prevention & validation |

### DART / Remediation Agents

| Agent | Purpose |
|-------|---------|
| `dart-automation-coordinator` | Orchestrate PDF conversion |
| `dart-converter` | Multi-source synthesis conversion |
| `imscc-intake-parser` | Extract & inventory IMSCC packages |
| `content-analyzer` | Detect accessibility & quality gaps |
| `accessibility-remediation` | WCAG fixes, alt text, headings |
| `content-quality-remediation` | Educational depth & enhancement |
| `intelligent-design-mapper` | Component selection & styling |
| `remediation-validator` | Final QA & WCAG verification |

### Textbook Pipeline Agents

| Agent | Purpose |
|-------|---------|
| `textbook-stager` | Stage DART outputs for Courseforge |
| `textbook-ingestor` | Parse DART HTML & extract objectives |
| `source-router` | Bind DART source blocks to Courseforge module pages (TF-IDF + confidence scoring) |
| `libv2-archivist` | Archive course artifacts to LibV2 |

### Trainforge Agents

| Agent | Purpose |
|-------|---------|
| `assessment-extractor` | Parse IMSCC & extract content |
| `rag-indexer` | Build vector embeddings & index |
| `assessment-generator` | Generate questions & distractors |
| `assessment-validator` | Validate quality & Bloom's alignment |
| `training-synthesizer` | Synthesize instruction + preference training pairs from chunks + assessments (routes to `synthesize_training`). |

---

## Quality Standards

### Decision Rationale

Every decision rationale MUST:
- Be at least 20 characters
- Explain the "why" not just the "what"
- Reference alternatives when applicable

### LLM call-site instrumentation

Every Claude / LLM call site MUST wire up a `DecisionCapture`
instance and emit at least one decision per call (per-batch when the
call is batched). Static boilerplate rationales are forbidden —
rationale must interpolate dynamic signals specific to the call
(block IDs, image hashes, page numbers, model + max_tokens, confidence
distributions, etc.) so captures are replayable post-hoc. A regression
test MUST assert that the capture fires on the call path. Precedents:

- DART LLM classifier: `DART/converter/llm_classifier.py` → one
  `structure_detection` capture per batch (see
  `DART/tests/test_llm_classifier_capture_wiring.py`).
- DART alt-text generator: `DART/pdf_converter/alt_text_generator.py`
  → one `alt_text_generation` capture per figure (see
  `DART/tests/test_alt_text_generator_capture_wiring.py`).
- DART pipeline entry point: `MCP/tools/pipeline_tools.py::_raw_text_to_accessible_html`
  → one `pipeline_run_attribution` capture per run (see
  `DART/tests/test_pipeline_run_attribution.py`).
- Trainforge synthesis provider: `Trainforge/generators/_anthropic_provider.py`
  → one `synthesis_provider_call` capture per call (see
  `Trainforge/tests/test_anthropic_synthesis_provider.py`).
- Trainforge curriculum-alignment provider: `Trainforge/generators/_curriculum_provider.py`
  (consumed by `Trainforge/align_chunks.py::classify_teaching_roles`)
  → one `curriculum_alignment_call` capture per teaching-role classification
  (see `Trainforge/tests/test_curriculum_alignment_provider.py`).
- Trainforge OpenAI-compatible HTTP client: `Trainforge/generators/_openai_compatible_client.py`
  → one `llm_chat_call` capture per call when wired with a capture; surface
  used by future task providers that compose the client directly (see
  `Trainforge/tests/test_openai_compatible_client.py`).

### Assessment Quality (Trainforge)

- Bloom's taxonomy alignment required
- Learning objective mapping required
- Distractor misconception targeting required

### Content Quality (Courseforge)

- WCAG 2.2 AA compliance
- Clear learning objectives per module
- Consistent formatting

### Conversion Quality (DART)

- Semantic HTML structure
- Alt text for all images
- Proper heading hierarchy

---

## Error Handling

### Error Classification (Phase 0 Hardening)

Errors are classified to determine retry behavior:
- **Transient**: `timeout`, `rate_limit`, `connection_error`, `service_unavailable` → retryable
- **Permanent**: `validation_error`, `missing_input`, `permission_denied`, `schema_error` → no retry

### Retry Protocol

Failed tasks retry up to 3 times with exponential backoff:
1. First retry: After 5 seconds
2. Second retry: After 30 seconds
3. Third retry: After 120 seconds
4. After 3 failures: Log to error table, require manual intervention

### Poison Pill Detection

Stops a batch when the same error pattern repeats:
- Default threshold: 3 same-pattern failures within 5 minutes
- Prevents runaway batch failures from consuming resources

### Phase Checkpointing

Each phase completion creates a checkpoint in `state/runs/{run_id}/checkpoints/`:
- Enables crash recovery without re-running completed phases
- Checkpoints include phase outputs and state snapshots

### Error Logging

All errors logged to:
- GENERATION_PROGRESS.md error table
- Individual JSONL capture files

---

## Validation Gates

Validation gates run after workflow phases to enforce quality:

### Gate Configuration

```yaml
validation_gates:
  - gate_id: content_structure
    validator: lib.validators.content.ContentStructureValidator
    severity: critical     # critical | warning
    threshold:
      max_critical_issues: 0
    behavior:
      on_fail: block       # block | warn
      on_error: fail_closed # fail_closed | warn
```

### Severity Levels

| Severity | Behavior |
|----------|----------|
| `critical` | Blocks workflow progression on failure |
| `warning` | Logs warning, allows workflow to continue |

### Active Gates

Source of truth: `config/workflows.yaml::validation_gates`. Phase column below shows the phase at which each gate fires; severity in parentheses (`critical` when unmarked).

| Workflow | Phase | Gate | Validator |
|----------|-------|------|-----------|
| `course_generation` | `content_generation` | `content_structure` | ContentStructureValidator |
| `course_generation` | `packaging` | `imscc_structure` | IMSCCValidator |
| `course_generation` | `packaging` | `page_objectives` | PageObjectivesValidator |
| `course_generation` | `validation` | `wcag_compliance` | WCAGValidator |
| `course_generation` | `validation` | `oscqr_score` | OSCQRValidator (warning) |
| `intake_remediation` | `parsing` | `imscc_parse` | IMSCCParseValidator |
| `intake_remediation` | `validation` | `wcag_compliance` | WCAGValidator |
| `batch_dart` | `multi_source_synthesis` | `dart_markers` | DartMarkersValidator |
| `batch_dart` | `validation` | `wcag_aa_compliance` | WCAGValidator |
| `textbook_to_course` | `dart_conversion` | `dart_markers` | DartMarkersValidator |
| `textbook_to_course` | `content_generation` | `content_structure` | ContentStructureValidator (warning) |
| `textbook_to_course` | `content_generation` | `source_refs` | PageSourceRefValidator |
| `textbook_to_course` | `content_generation` | `content_grounding` | ContentGroundingValidator (Wave 31) |
| `textbook_to_course` | `packaging` | `imscc_structure` | IMSCCValidator (warning) |
| `textbook_to_course` | `packaging` | `page_objectives` | PageObjectivesValidator |
| `textbook_to_course` | `trainforge_assessment` | `imscc_input_valid` | IMSCCValidator (pre-assessment) |
| `textbook_to_course` | `trainforge_assessment` | `assessment_quality` | AssessmentQualityValidator |
| `textbook_to_course` | `trainforge_assessment` | `assessment_objective_alignment` | AssessmentObjectiveAlignmentValidator |
| `textbook_to_course` | `training_synthesis` | `synthesis_quota` | SynthesisQuotaValidator (Wave 110, warning) |
| `textbook_to_course` | `training_synthesis` | `min_edge_count` | MinEdgeCountValidator (Wave 91) |
| `textbook_to_course` | `training_synthesis` | `synthesis_diversity` | SynthesisDiversityValidator (Wave 91) |
| `textbook_to_course` | `training_synthesis` | `property_coverage` | PropertyCoverageValidator (Wave 109 — no-ops on courses without a property manifest) |
| `textbook_to_course` | `training_synthesis` | `synthesis_leakage` | SynthesisLeakageValidator (Wave 121 — fails closed at >5% verbatim chunk leakage) |
| `textbook_to_course` | `training_synthesis` | `curie_anchoring` | CurieAnchoringValidator (Wave 135c — binary per-pair anchoring sentinel, default min_pair_anchoring_rate=0.95; replaces Wave 130b curie_preservation) |
| `textbook_to_course` | `libv2_archival` | `libv2_manifest` | LibV2ManifestValidator |
| `textbook_to_course` | `libv2_archival` | `kg_quality_report` | KGQualityValidator (Wave 91 promotion: critical, thresholds 0.95/0.95/0.95/0.5) |
| `rag_training` | `assessment_generation` | `assessment_quality` | AssessmentQualityValidator |
| `rag_training` | `assessment_generation` | `bloom_alignment` | BloomAlignmentValidator (warning) |
| `rag_training` | `assessment_generation` | `leak_check` | LeakCheckValidator |
| `rag_training` | `assessment_generation` | `outcome_ref_integrity` | LeakCheckValidator (warning) |
| `rag_training` | `assessment_generation` | `content_fact_check` | ContentFactValidator (warning) |
| `rag_training` | `assessment_generation` | `question_quality` | QuestionQualityValidator |
| `rag_training` | `validation` | `final_quality` | FinalQualityValidator |
| `trainforge_train` | `post_training_validation` | `eval_gating` | EvalGatingValidator (Wave 108 — fails closed on regression / yes-bias / no-bias / source-match drop) |

---

## Configuration Files

### workflows.yaml

Defines workflow phases and concurrency limits.
Location: `config/workflows.yaml`

### agents.yaml

Defines agent capabilities and project paths.
Location: `config/agents.yaml`

### workflows_meta.schema.json

Meta-schema that validates `config/workflows.yaml` at load time (phase routing, gate shape, `inputs_from` references).
Location: `schemas/config/workflows_meta.schema.json`

---

## Opt-In Behavior Flags

Environment-variable toggles gate opt-in strict / stable-ID / provenance / experimental-rule-graph behavior. All default off to preserve backward compatibility with legacy corpora. See `schemas/ONTOLOGY.md` § 12 for full rationale per flag.

| Flag | When on |
|------|---------|
| `TRAINFORGE_CONTENT_HASH_IDS` | Chunk IDs become re-chunk-stable content hashes. |
| `TRAINFORGE_SCOPE_CONCEPT_IDS` | Concept node IDs become `{course_id}:{slug}` for cross-course disambiguation. |
| `TRAINFORGE_PRESERVE_LO_CASE` | LO refs retain emit case (`TO-01` vs `to-01`). |
| `TRAINFORGE_VALIDATE_CHUNKS` | Enforces `schemas/knowledge/chunk_v4.schema.json` on every chunk write. |
| `TRAINFORGE_ENFORCE_CONTENT_TYPE` | Constrains `content_type_label` to the canonical 8-value enum. |
| `TRAINFORGE_STRICT_EVIDENCE` | Strips the FallbackProvenance arm from the evidence discriminator. |
| `TRAINFORGE_SOURCE_PROVENANCE` | Evidence arms emit `source_references[]` sourced from chunks' `source.source_references[]`. Off: arms emit the pre-provenance shape. |
| `DECISION_VALIDATION_STRICT` | Fails closed on unknown `decision_type` values in decision captures. |
| `DART_LLM_CLASSIFICATION` | DART's block classifier routes through Claude via `LLMClassifier` instead of heuristic regex. Requires an injected `LLMBackend`. |
| `LOCAL_DISPATCHER_ALLOW_STUB` | Permits `LocalDispatcher` to emit a stubbed `PhaseOutput` (`status="ok"`) when no `agent_tool` callable was wired in. Tests / dry-run only. Default off so production `--mode local` runs fail loudly instead of silently succeeding with empty phase outputs. |
| `ED4ALL_AGENT_DISPATCH` | Wave 74: route tasks for subagent-classified agents (see `AGENT_SUBAGENT_SET` in `MCP/core/executor.py`) through `dispatcher.dispatch_task` instead of the in-process `tool_registry` entry. Closes the Wave 38 gap that caused content_generation / assessment phases to run as in-process templates regardless of `--mode`. Default off so Wave 74 Session 1 lands the infrastructure without altering any existing run. Agents that stay in-process (DART extraction, TF-IDF routing, packaging, WCAG validation, archival) are unaffected. |
| `ED4ALL_AGENT_TIMEOUT_SECONDS` | Wave 74: override the default 1800 s mailbox timeout for per-task subagent dispatches. Longer than Wave 73's 120 s default because content-generator / remediation subagents can legitimately take 10+ min to produce a full week's module output. |
| `ED4ALL_STAGE_MODE` | Wave 74 cleanup: selects how `stage_dart_outputs` materialises DART HTML / `_synthesized.json` / `.quality.json` / `{stem}_figures/` into `Courseforge/inputs/textbooks/{run_id}/`. Values: `copy` (legacy deep-copy), `symlink` (default — single-inode references back to DART output), `hardlink` (Windows fallback when symlinks are blocked). Saves ~70MB per textbook-to-course run. Manifest format and downstream readers are unchanged regardless of mode. |
| `TRAINFORGE_PROVENANCE_CORPUS` | Worker L: absolute path to a locally regenerated `chunks.jsonl` that the `Trainforge/tests/test_provenance.py` suite loads to assert 100% `source.html_xpath` + `source.char_span` coverage on a real corpus. Default unset → those tests `pytest.skip()` so CI on a clean checkout passes without needing a corpus checkout. Pre-Worker-E corpora (no provenance fields) trip the same skip. Test-only — no production code path reads this flag. |
| `TRAINFORGE_SEED_TECH_CONCEPTS` | Wave 82 Phase C: when on, `lib/ontology/tech_anchors.py::detect_anchors` scans chunk text for canonical W3C surface forms (`RDF`, `RDFS`, `OWL`, `SHACL`, `SPARQL`, `Turtle`, `JSON-LD`, `owl:sameAs`, …) and appends matching anchor slugs to `concept_tags`, so the existing 2-chunk co-occurrence gate admits foundational-tech standalone concept nodes (`rdf-shacl-551-2` audit fix). Default off so legacy corpora don't shift their tag distributions on rebuild. |
| `TRAINFORGE_VALIDATE_RULE_OUTPUTS` | Wave 82: enables `lib.validators.semantic_graph_rule_output.SemanticGraphRuleOutputValidator` (warning-severity gate on `textbook_to_course::libv2_archival`). Compares per-rule edge counts in the just-emitted `concept_graph_semantic.json` against a baseline; flags any rule that had ≥10 edges in baseline but zero in current with an unchanged `rule_version` (Wave 82 silent-zero regression class). Default off so corpora without a baseline path keep passing the gate. |
| `TRAINFORGE_EMIT_TRIG` | Wave 84-85 Phase 3 (rdf-shacl-enrichment plan): when on, `Trainforge/process_course.py` additionally writes a sibling `concept_graph_semantic.trig` whose per-rule named graphs are scoped by `(run_id, rule_name)` IRI and carry `ed4all:edgeCount` / `ed4all:inputChunkCount` provenance metadata that SPARQL consumers can diff across runs. JSON output is byte-identical whether the flag is on or off — TriG emit is purely additive. Default off so legacy consumers and `rdflib`-less environments stay clean. |
| `TRAINFORGE_USE_SHACL_RULES` | Wave 84-85 Phase 5 (rdf-shacl-enrichment plan): when on, `Trainforge/rag/shacl_rule_runner.py` runs `schemas/context/courseforge_v1.shacl-rules.ttl` via pyshacl `advanced=True, inplace=True` and projects inferred `ed4all:isDefinedBy` triples back into the same edge-dict shape `defined_by_from_first_mention.py` emits. Equivalence pinned by `Trainforge/tests/test_shacl_rules_defined_by.py`. Default off → the canonical Python rule stays authoritative until SHACL parity proves out across the project test suite. |
| `TRAINFORGE_SHACL_CLOSED_WORLD` | Wave 88: merges `schemas/context/courseforge_v1.shacl-closed.ttl` into the SHACL shapes graph at validation time, declaring `sh:closed true ; sh:ignoredProperties (rdf:type)` on `cfshapes:ChunkShape` and `cfshapes:TypedEdgeShape`. Unminted predicates on chunk / typed-edge nodes fire `sh:ClosedConstraintComponent` violations. Default off; Wave 87 minted the 33 chunk-structural predicates + 2 anchor classes that this closure asserts against, so flipping the flag on a clean Wave 87+ corpus is mass-violation-free. Closed-world overhead measured at ~0.5s on a 1000-node fixture. |
| `ANTHROPIC_SYNTHESIS_MODEL` | Wave 91: overrides the default `claude-sonnet-4-6` Anthropic model used by `Trainforge/generators/_anthropic_provider.py` for the chunk → instruction-pair paraphrase pass. `ANTHROPIC_API_KEY` is the hard prerequisite; this flag is purely the model-ID dial. Captured per call in the `synthesis_provider_call` decision event. (License: Anthropic Commercial Terms — outputs restricted from training-data use; see `docs/LICENSING.md`.) |
| `TOGETHER_API_KEY` | Wave 113 prep: required when `--provider together`. Routes synthesis through Together AI's OpenAI-compatible chat-completions endpoint. Together's ToS permits using the output as training data for another model — unlike Anthropic's ToS, which is the motivation for offering this provider for SLM training-data generation. Missing key with `--provider together` raises `RuntimeError` (no silent mock fallback). (License: see `docs/LICENSING.md` for the ToS + per-model layer.) |
| `TOGETHER_SYNTHESIS_MODEL` | Wave 113 prep: overrides the default `meta-llama/Llama-3.3-70B-Instruct-Turbo` used by `Trainforge/generators/_together_provider.py`. Common alternatives: `Qwen/Qwen2.5-72B-Instruct-Turbo`, `deepseek-ai/DeepSeek-V3`. Captured per call in the `synthesis_provider_call` decision event so the audit trail records which OSS teacher produced each pair. (License: model-specific — Llama 3.3 Community / Qwen / DeepSeek; see `docs/LICENSING.md`.) |
| `LOCAL_SYNTHESIS_BASE_URL` | Wave 113: base URL of a local OpenAI-compatible model server used by `Trainforge/generators/_local_provider.py` when `--provider local` is selected. Defaults to the Ollama default `http://localhost:11434/v1`; common alternatives are vLLM `http://localhost:8000/v1`, llama.cpp server `http://localhost:8080/v1`, and LM Studio `http://localhost:1234/v1`. Captured per call in the `synthesis_provider_call` decision event so the audit trail can tell which local server produced each pair. |
| `LOCAL_SYNTHESIS_MODEL` | Wave 113: model identifier the local server expects (e.g. `qwen2.5:14b-instruct-q4_K_M` for Ollama, `Qwen/Qwen2.5-32B-Instruct` for vLLM). Defaults to the smaller `qwen2.5:14b-instruct-q4_K_M` so an out-of-box Ollama install on an 8 GB GPU works without further tuning. Default `qwen2.5:7b-instruct-q4_K_M` is reliable for paraphrase tasks WHEN `json_mode=True` (default in `_local_provider.py` since the Wave 113 JSON-hardening commit — sends both Ollama's `format: "json"` and OpenAI's `response_format: {"type": "json_object"}` plus a strict-JSON prompt directive plus lenient response extraction). For free-form structured-output tasks beyond paraphrase, prefer 14B+ models. (License: model-specific — Qwen2.5-7B/14B/32B are Apache 2.0 and training-permitted; see `docs/LICENSING.md` for the full per-model table.) |
| `LOCAL_SYNTHESIS_API_KEY` | Wave 113: optional auth key for the local server. Most local servers ignore the auth header, so the provider does NOT raise when this is unset (unlike `TOGETHER_API_KEY` / `ANTHROPIC_API_KEY`); the constructor sends a placeholder `"local"` string in the Authorization header so reverse-proxy servers that DO check auth see a stable value. Set this only when the local server proxies to a remote provider that requires auth. |
| `CURRICULUM_ALIGNMENT_PROVIDER` | Selects the LLM backend for `Trainforge/align_chunks.py` teaching-role classification (`Trainforge/generators/_curriculum_provider.py::CurriculumAlignmentProvider`). Values: `anthropic` (default — ToS-restricted for training-data), `together` (ToS-clean cloud OSS via the shared `OpenAICompatibleClient`), `local` (8GB-VRAM-friendly with 14B 4-bit via the shared `OpenAICompatibleClient`). Reuses the same `TOGETHER_*` / `LOCAL_*` env vars as the synthesis pipeline so one local server serves both task surfaces. Captured per call in the `curriculum_alignment_call` decision event. (License: see `docs/LICENSING.md`.) |

---

## Licensing & ToS Posture

Canonical reference: **`docs/LICENSING.md`**. Read it before running any training-data synthesis pass.

The project distinguishes two surfaces with different licensing exposure:

- **Development tools** (Claude Code, OpenAI Codex) generate code, prose, and shell invocations. Their ToS restricts training-data routing, but on this project that restriction is moot — these tools never produce training data, so the dev tool you use has zero effect on the trained SLM's licensing.
- **LLM providers** (`--provider anthropic` / `claude_session` / `together` / `local`) generate the paraphrased instruction / preference pairs that become training data. The trained model is a derivative work of those outputs, so the provider's ToS + the underlying model license decide whether the corpus is shippable.

Default posture: training-data synthesis routes to license-clean providers — `--provider local` with an Apache 2.0 model (Qwen2.5-7B/14B/32B) for an air-gapped clean corpus, or `--provider together` with a hosted OSS model as the cloud fallback. Anthropic providers stay wired for backward compatibility but are not the recommended default for training data.

**Maintenance contract:** any new behavior flag in the table above that selects an LLM provider, model ID, or synthesis backend MUST land with a corresponding row in `docs/LICENSING.md`'s "Synthesis providers" table. Drift between this file's per-flag rows and `docs/LICENSING.md` is a documentation bug.

---

## Canonical Helpers

Single-source-of-truth loaders under `lib/ontology/`:

- `lib/ontology/bloom.py` — Bloom verb / level / cognitive-domain detection.
- `lib/ontology/slugs.py::canonical_slug` — unified slug helper.
- `lib/ontology/teaching_roles.py` — `(component, purpose) → role` mapper.
- `lib/ontology/taxonomy.py::load_taxonomy(name)` — generic JSON-taxonomy loader, reads from `schemas/taxonomies/`.

Validators under `lib/validators/` (see Active Gates above for wiring):

- `lib/validators/page_objectives.py` — objective coverage per page.
- `lib/validators/content_type.py` — content_type enum enforcement (gated).
- `lib/validators/evidence.py` — per-rule evidence discriminator loader; strict mode drops FallbackProvenance.
- `lib/validators/assessment_objective_alignment.py` — fail-loud gate keeping every assessment question's `objective_id` covered by at least one chunk's `learning_outcome_refs`.
- `lib/validators/source_refs.py` — verifies every emitted Courseforge `sourceId` resolves against the DART staging manifest.
- `lib/validators/libv2_manifest.py` — validates LibV2 manifest JSON, scaffold completeness, and on-disk artifact hash/size agreement.
- `lib/validators/libv2_model.py` — validates emitted `model_card.json` against `schemas/models/model_card.schema.json`. Critical: schema match, weights file presence + size + sha256 agreement, `pedagogy_graph_hash` resolves to extant graph in same course. Warning: missing eval scores, missing license, malformed HF repo regex. Wired as the `libv2_model` gate (Wave 89).
- `lib/validators/kg_quality.py` — KG-quality report (completeness / consistency / accuracy / coverage); thin wrapper over `Trainforge/rag/kg_quality_report.py::KGQualityReporter`. Wave 91 promotion: thresholds 0.95 / 0.95 / 0.95 / 0.5 (was advisory 0.0 at roll-out).
- `lib/validators/min_edge_count.py` — Wave 91. Pre-synthesis gate: critical-fails on pedagogy graph with <100 edges, <4 distinct edge types, or concept graph with <50 nodes. Closes the silent zero-edge regression class for the synthesis surface.
- `lib/validators/synthesis_diversity.py` — Wave 91. Post-synthesis gate: critical-fails when top-3 templates >60% of pairs, single template >35%, or distinct templates <8. Warns when total pairs <100.
- `lib/validators/synthesis_leakage.py` — Wave 121 + 122. Post-synthesis gate covering two contamination vectors: (a) verbatim-span leakage from `chunk.text` (Wave 121, default 5% rate / 50-char span); (b) assessment-outline scaffolding patterns like `Question N (XX-NN, Bloom: ...)` (Wave 122, default 0% — zero tolerance, structural contamination). Tunable via gate `config.thresholds.leak_rate_threshold`, `leak_span_chars`, `assessment_scaffold_rate_threshold`.

**Canonical LO helper**: `lib/ontology/learning_objectives.py` owns the single source of truth for LO identity (`mint_lo_id`, `validate_lo_id`, `hierarchy_from_id`, `split_terminal_chapter`). Pattern `^[A-Z]{2,}-\\d{2,}$` mirrors `schemas/knowledge/courseforge_jsonld_v1.schema.json`. `schemas/knowledge/course.schema.json` is the canonical shape for Trainforge-emitted `course.json` consumed by LibV2.

---

## Individual Project Guides

- **DART**: `DART/CLAUDE.md`
- **Courseforge**: `Courseforge/CLAUDE.md`
- **Trainforge**: `Trainforge/CLAUDE.md`
- **LibV2**: `LibV2/CLAUDE.md`
- **Ontology map + v0.2.0 changes**: `schemas/ONTOLOGY.md`
- **KG-quality review (source of v0.2.0 work)**: `plans/kg-quality-review-2026-04/review.md`

---

## Training Pipeline (Waves 89–93)

SLM training is a **post-import LibV2 stage**, not a step in `Trainforge/process_course.py`. The trainer reads `training_specs/*.jsonl` from an already-imported LibV2 course and writes `LibV2/courses/<slug>/models/<model_id>/`. End-state: courses carry trained QLoRA adapters with model card + eval report + decision log; Hugging Face is the upload target.

- **Top-level command**: `ed4all run trainforge_train --course-code <slug> --base-model qwen2.5-1.5b` (Wave 90 registered the workflow in `config/workflows.yaml::trainforge_train`).
- **Direct entry point**: `python -m Trainforge.train_course --course-code <slug> --base-model <name> [--dry-run] [--backend local|runpod]`. Requires `pip install ed4all[training]` for non-dry-run mode.
- **Schemas**: `schemas/models/model_card.schema.json` (Wave 89 + Wave 92's `holdout_graph_hash` extension) and `schemas/models/model_pointers.schema.json` (Wave 93 promotion ledger).
- **Deep dive**: `Trainforge/CLAUDE.md` § "Training Pipeline" — base model registry, provider configuration, 5×3 eval matrix, 7-hash provenance, promotion workflow, decision-capture contract.
- **Stage diagram**: see `LibV2/CLAUDE.md` for the post-import sub-stage ASCII.

---

## Training Data Export

### Formats Supported

| Format | Use Case |
|--------|----------|
| `alpaca` | Fine-tuning with instruction format |
| `openai` | OpenAI-compatible training |
| `dpo` | Direct Preference Optimization |
| `raw` | Raw JSONL for custom processing |

### Export Command

```bash
# Via CLI
ed4all export-training <run_id> --format dpo

# Via MCP tool
export_training_data(format="dpo", date_range={"start": "2025-01-01", "end": "2025-01-31"}, min_quality="proficient")
```

---

## Summary Checklist

Before starting any workflow:

- [ ] MCP server running
- [ ] TodoWrite initialized
- [ ] Decision capture configured
- [ ] GENERATION_PROGRESS.md cleared/ready
- [ ] Appropriate config loaded

During execution:

- [ ] Maximum 10 parallel tasks per batch
- [ ] All decisions logged with rationale (20+ chars)
- [ ] One agent per file
- [ ] Wait for batch completion before next batch

After completion:

- [ ] All tasks marked completed
- [ ] Training captures exported
- [ ] GENERATION_PROGRESS.md updated
- [ ] Errors reviewed and addressed
