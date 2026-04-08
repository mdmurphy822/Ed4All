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
| `course_generation` | Generate new course from objectives | 10 |
| `intake_remediation` | Import and remediate IMSCC | 4 |
| `batch_dart` | Batch PDF to HTML conversion | 4 |
| `rag_training` | Trainforge assessment generation | 4 |

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
│   ├── server.py            # FastMCP server
│   └── tools/               # Tool modules
├── orchestrator/            # Multi-terminal coordination
├── state/                   # Shared state & progress tracking
├── schemas/                 # JSON schemas for validation
├── training-captures/       # Decision capture output
├── lib/                     # Shared libraries
└── config/                  # Workflow & agent configs
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
    run_id="RUN_20250101_143022",
    course_id="INT_101",
    operation="course_generation"
)

capture.log_decision(
    decision_type="content_outline",
    decision="Use 6-week modular structure",
    rationale="Aligns with competency-based approach and allows flexible pacing for diverse learners",
    alternatives_considered=[
        {"option": "8-week linear", "rejected_because": "Too rigid for self-paced learning"},
        {"option": "4-week intensive", "rejected_because": "Insufficient depth for foundational content"}
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

### DART Tools

| Tool | Description |
|------|-------------|
| `convert_pdf_to_html` | Convert single PDF to accessible HTML |
| `batch_convert_documents` | Batch convert multiple PDFs |
| `validate_wcag_compliance` | Validate HTML accessibility |
| `get_dart_status` | Get DART processing status |

### Courseforge Tools

| Tool | Description |
|------|-------------|
| `create_course_project` | Initialize new course project |
| `generate_course_content` | Generate content for weeks |
| `package_imscc` | Package course as IMSCC |
| `intake_imscc_package` | Import existing IMSCC |
| `remediate_course_content` | Fix content issues |
| `get_courseforge_status` | Get project status |

### Orchestrator Tools

| Tool | Description |
|------|-------------|
| `create_workflow` | Create new workflow instance |
| `get_workflow_status` | Check workflow progress |
| `dispatch_agent_task` | Dispatch task to agent |
| `poll_task_completions` | Wait for task completions |
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
from orchestrator.ipc.status_tracker import StatusTracker

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
   └── Parse IMSCC, extract content

2. embedding
   └── Create vector embeddings

3. indexing
   └── Build RAG index

4. assessment_generation
   └── Generate questions with full capture
```

---

## Agent Registry

### Courseforge Agents

| Agent | Purpose |
|-------|---------|
| `course-outliner` | Create course structure |
| `content-generator` | Generate module content |
| `brightspace-packager` | Package for Brightspace LMS |

### DART Agents

| Agent | Purpose |
|-------|---------|
| `dart-automation-coordinator` | Orchestrate PDF conversion |

### Trainforge Agents

| Agent | Purpose |
|-------|---------|
| `content-analyzer` | Analyze IMSCC content |
| `assessment-generator` | Generate assessments |
| `validator` | Validate assessment quality |

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

- WCAG 2.1 AA compliance
- Clear learning objectives per module
- Consistent formatting

### Conversion Quality (DART)

- Semantic HTML structure
- Alt text for all images
- Proper heading hierarchy

---

## Error Handling

### Retry Protocol

Failed tasks retry up to 3 times:
1. First retry: Immediate
2. Second retry: After 5 seconds
3. Third retry: After 30 seconds
4. After 3 failures: Log to error table, require manual intervention

### Error Logging

All errors logged to:
- GENERATION_PROGRESS.md error table
- Individual JSONL capture files

---

## Configuration Files

### workflows.yaml

Defines workflow phases and concurrency limits.
Location: `config/workflows.yaml`

### agents.yaml

Defines agent capabilities and project paths.
Location: `config/agents.yaml`

---

## Individual Project Guides

- **DART**: `DART/CLAUDE.md`
- **Courseforge**: `Courseforge/CLAUDE.md`
- **Trainforge**: `Trainforge/CLAUDE.md`
- **LibV2**: `LibV2/CLAUDE.md`

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

```python
from MCP.tools.trainforge_tools import export_training_data

result = export_training_data(
    format="dpo",
    date_range={"start": "2025-01-01", "end": "2025-01-31"},
    min_quality="proficient"
)
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
