# Courseforge Workflow Reference

Complete workflow phases, execution protocols, and pattern prevention.

---

## Simplified Courseforge Pipeline

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

---

## Phase 1: Input Analysis & Planning

**Orchestrator Actions:**
1. Create timestamped project folder: `exports/YYYYMMDD_HHMMSS_coursename/`
2. Invoke planning agent based on input type:
   - Exam objectives → `exam-research` agent
   - New course → `requirements-collector` agent
3. Planning agent analyzes input and returns todo list (NO EXECUTION)
4. Orchestrator loads todo list into TodoWrite

**Key Principle:** Planning agents provide structured todo lists. They do NOT execute tasks.

---

## Phase 2: Course Framework Development

**Orchestrator Actions:**
1. Invoke `course-outliner` agent
2. Agent determines optimal learning progression based on:
   - Knowledge prerequisites
   - Cognitive relationships
   - Assessment alignment
3. Agent creates course structure files
4. OSCQR evaluation triggers automatically after outline completion

**Timeline-Free Design:** Structure based on learning progression, not arbitrary week counts.

---

## Phase 3: Content Generation

**Execution Protocol:**
1. Review content generation tasks from todo list
2. Execute content-generator agents in parallel batches

**Critical Batch Constraints:**
- Maximum: 10 agents per batch (proven optimal limit)
- Each agent creates exactly ONE file
- Wait for batch completion before next batch
- Update TodoWrite after each batch

**Anti-Patterns (NEVER DO):**
- ❌ Assign multiple files to one agent
- ❌ Use prompts like "create Week X content"
- ❌ Exceed 10 simultaneous Task calls

**Correct Pattern:**
```python
# BATCH 1 (10 agents max)
Task(content-generator, "week_01_module_01_introduction.html")
Task(content-generator, "week_01_module_02_concepts.html")
# ... up to 10 total

# WAIT for completion, update todos, then BATCH 2
```

---

## Phase 4: Quality Validation & Packaging

**Orchestrator Actions:**
1. Invoke validation agents in parallel:
   - `quality-assurance` agent (pattern prevention)
   - `oscqr-course-evaluator` agent (educational quality)
2. If issues found: reinvoke agents to fix
3. When validation complete: invoke `brightspace-packager`
4. Output: Single IMSCC file

---

## Orchestrator Protocol Summary

| Step | Actor | Action |
|------|-------|--------|
| 1 | Planning Agent | Analyzes input, returns todo list (NO execution) |
| 2 | Orchestrator | Loads todo list into TodoWrite |
| 3 | Orchestrator | Executes todos via appropriate agents |
| 4 | Execution Agents | Work from todo specs (NO todo modifications) |
| 5 | Orchestrator | Manages all todo state changes |

**Critical Rule:** Only orchestrator modifies TodoWrite. No agent-to-agent todo feedback loops.

---

## Pattern Prevention Reference

### Orchestration Patterns

| Pattern | Issue | Prevention |
|---------|-------|------------|
| O1 | Orchestrator making pedagogical decisions | Delegate to specialized agents |
| O5 | Multi-file agent batching | ONE AGENT = ONE FILE always |
| O6 | Incorrect parallel execution | Exactly 10 Task calls per batch |
| O7 | Queue saturation | Never exceed 10 simultaneous calls |

### Content Patterns

| Pattern | Issue | Prevention |
|---------|-------|------------|
| 16 | Post-import quality issues | Pre-packaging validation |
| 19 | Single-page consolidation | Maintain module structure |
| 21 | Incomplete content | Validate all weeks before packaging |
| 22 | Superficial content | Ensure educational depth |

---

## File Naming Convention

Content files must follow this pattern:
```
week_XX_module_YY_description.html
```

Examples:
- `week_01_module_01_introduction.html`
- `week_01_module_02_core_concepts.html`
- `week_01_module_03_applications.html`

---

## Project Folder Structure

```
exports/YYYYMMDD_HHMMSS_coursename/
├── 00_template_analysis/
├── 01_learning_objectives/
├── 02_course_planning/
├── 03_content_development/
│   ├── week_01/
│   ├── week_02/
│   └── ...
├── 04_quality_validation/
├── 05_final_package/
├── agent_workspaces/
├── project_log.md
└── coursename.imscc
```

---

## Validation Checklist

**Before Content Generation:**
- [ ] Planning agent has provided todo list
- [ ] Orchestrator has loaded todos into TodoWrite
- [ ] Individual file assignments prepared (not multi-file)
- [ ] Batch size ≤10 confirmed

**Before Packaging:**
- [ ] All content files created
- [ ] Quality validation passed
- [ ] OSCQR evaluation completed
- [ ] No placeholder content detected

**After Packaging:**
- [ ] Single IMSCC file generated
- [ ] Package size >100KB
- [ ] Test import in Brightspace
