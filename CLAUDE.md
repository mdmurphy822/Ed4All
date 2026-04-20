# Ed4All Hybrid Orchestrator

Unified orchestration system for DART, Courseforge, Trainforge, and LibV2.

## Quick Start

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
| `create_course_project` | Initialize new course project |
| `generate_course_content` | Generate content for weeks |
| `package_imscc` | Package course as IMSCC |
| `intake_imscc_package` | Import existing IMSCC |
| `remediate_course_content` | Fix content issues |
| `get_courseforge_status` | Get project status |

### Courseforge Metadata Output

Courseforge HTML pages include machine-readable metadata for downstream Trainforge consumption:
- **`data-cf-*` attributes**: Inline metadata on HTML elements (role, objective IDs, Bloom's levels/verbs, cognitive domain, content types, teaching role, key terms, component, purpose). See `Courseforge/CLAUDE.md` for the canonical attribute table.
- **JSON-LD blocks**: Structured `<script type="application/ld+json">` per page with learning objectives, section metadata, misconceptions, and assessment suggestions. Canonical shape: `schemas/knowledge/courseforge_jsonld_v1.schema.json`.

This metadata follows priority extraction in Trainforge: JSON-LD > data-cf-* attributes > regex heuristics.

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

### Pipeline Tools

| Tool | Description |
|------|-------------|
| `create_textbook_pipeline_tool` | Initialize textbook-to-course pipeline |
| `stage_dart_outputs` | Stage DART outputs for Courseforge |
| `get_pipeline_status` | Check pipeline progress |
| `validate_dart_markers` | Validate DART output markers |
| `run_textbook_pipeline_tool` | Execute full textbook pipeline |
| `archive_to_libv2` | Archive course artifacts to LibV2 |

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
   └── Extract structure and synthesize learning objectives

4. course_planning
   └── Create course structure from objectives

5. content_generation
   └── Generate course content modules (parallel batches of 10)

6. packaging
   └── Package course as IMSCC

7. trainforge_assessment (optional)
   └── Generate assessments from IMSCC package

8. libv2_archival
   └── Archive course artifacts to LibV2

9. finalization
   └── Final validation and training data export
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
| `libv2-archivist` | Archive course artifacts to LibV2 |

### Trainforge Agents

| Agent | Purpose |
|-------|---------|
| `assessment-extractor` | Parse IMSCC & extract content |
| `rag-indexer` | Build vector embeddings & index |
| `assessment-generator` | Generate questions & distractors |
| `assessment-validator` | Validate quality & Bloom's alignment |

---

## Quality Standards

### Decision Rationale

Every decision rationale MUST:
- Be at least 20 characters
- Explain the "why" not just the "what"
- Reference alternatives when applicable

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

| Workflow | Gate | Validator |
|----------|------|-----------|
| `course_generation` | `content_structure` | ContentStructureValidator |
| `course_generation` | `imscc_structure` | IMSCCValidator |
| `course_generation` | `wcag_compliance` | WCAGValidator |
| `course_generation` | `oscqr_score` | OSCQRValidator |
| `course_generation` | `page_objectives` | PageObjectivesValidator (Wave 2) |
| `batch_dart` | `wcag_aa_compliance` | WCAGValidator |
| `batch_dart` | `dart_markers` | DartMarkersValidator (Wave 6) |
| `textbook_to_course` | `dart_markers` | DartMarkersValidator (Wave 6) |
| `rag_training` | `assessment_quality` | AssessmentQualityValidator |
| `rag_training` | `bloom_alignment` | BloomAlignmentValidator |
| `rag_training` | `leak_check` | LeakChecker |

---

## Configuration Files

### workflows.yaml

Defines workflow phases and concurrency limits.
Location: `config/workflows.yaml`

### agents.yaml

Defines agent capabilities and project paths.
Location: `config/agents.yaml`

### workflows_meta.schema.json

Meta-schema that validates `config/workflows.yaml` at load time (phase routing, gate shape, `inputs_from` references). Wave 6 Worker V.
Location: `schemas/config/workflows_meta.schema.json`

---

## Opt-In Behavior Flags

Seven environment-variable toggles gate opt-in strict / stable-ID behavior. All default off to preserve backward compatibility with legacy corpora. See `schemas/ONTOLOGY.md` § 12 for landing wave and full rationale.

| Flag | When on |
|------|---------|
| `TRAINFORGE_CONTENT_HASH_IDS` | Chunk IDs become re-chunk-stable content hashes. |
| `TRAINFORGE_SCOPE_CONCEPT_IDS` | Concept node IDs become `{course_id}:{slug}` for cross-course disambiguation. |
| `TRAINFORGE_PRESERVE_LO_CASE` | LO refs retain emit case (`TO-01` vs `to-01`). |
| `TRAINFORGE_VALIDATE_CHUNKS` | Enforces `schemas/knowledge/chunk_v4.schema.json` on every chunk write. |
| `TRAINFORGE_ENFORCE_CONTENT_TYPE` | Constrains `content_type_label` to the canonical 8-value enum. |
| `TRAINFORGE_STRICT_EVIDENCE` | Strips the FallbackProvenance arm from the evidence discriminator. |
| `DECISION_VALIDATION_STRICT` | Fails closed on unknown `decision_type` values in decision captures. |

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

---

## Individual Project Guides

- **DART**: `DART/CLAUDE.md`
- **Courseforge**: `Courseforge/CLAUDE.md`
- **Trainforge**: `Trainforge/CLAUDE.md`
- **LibV2**: `LibV2/CLAUDE.md`
- **Ontology map + v0.2.0 changes**: `schemas/ONTOLOGY.md`
- **KG-quality review (source of v0.2.0 work)**: `plans/kg-quality-review-2026-04/review.md`

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
