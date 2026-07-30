# Ed4All Pipeline Architecture

> Living architecture document describing the end-to-end pipeline from raw input materials through accessible conversion, course generation, assessment training, and unified storage.

---

## Pipeline Overview

```
                         Ed4All Unified Pipeline
 ============================================================================

  RAW INPUTS                                                    UNIFIED STORE
  (PDF, DOCX,     SemantiK        Courseforge       Trainforge      LibV2
   textbooks)    (Conversion)    (Course Gen)     (RAG Training)  (Repository)
                                                                  
  +----------+   +----------+   +------------+   +------------+  +-----------+
  |          |   |          |   |            |   |            |  |           |
  | PDF docs |-->| v2       |-->| Staging &  |-->| IMSCC      |->| raw/      |
  | Office   |   | Semantic |   | Objective  |   | Content    |  |   pdf/    |
  | HTML     |   | Cascade  |   | Extraction |   | Analysis   |  |   html/   |
  |          |   |          |   |            |   |            |  |   imscc/  |
  +----------+   +-----+----+   +------+-----+   +------+-----+  |           |
                       |               |                |         | corpus/   |
                 WCAG 2.2 AA     Digital Pedagogy   RAG Corpus   | graph/    |
                 Semantic HTML   IMSCC Package      Assessments  | quality/  |
                 Quality Report  OSCQR Validated    Decisions    | manifest  |
                                                                  +-----------+
```

---

## Component Architecture

### 1. SemantiK (Semantic PDF → Accessible-HTML Conversion)

**Purpose**: Ingest raw PDF/document inputs and convert them to semantically rich, WCAG 2.2 AA compliant HTML. SemantiK is the **license-clean conversion engine** — its extraction stack carries no PyMuPDF/MuPDF (AGPL-3) or Poppler (GPL-2), so the subsystem ships Apache-2.0. It runs **in-process** within Ed4All (via the `[semantik]` extra) and needs no cloud LLM at runtime; a hosted large-model endpoint is an opt-in quality seat, not a dependency.

**Pipeline Position**: First stage - raw material ingestion and conversion.

```
Input Sources           13-Stage v2 Semantic Cascade            Output
============           ===========================            ======

                     extract  -> features -> council BERTs ---+
PDF / Office Doc --->|-> structure-graph -> Qwen specialists  +--> Accessible HTML
                     |-> hard/soft gates -> assemble -> theta -+    + Quality Report
                     +-> exit decider (ship / flag / lane) ----+
```

**Core principle**: learned models are narrow candidate generators; deterministic code orchestrates, gates, and assembles. BERT specialists classify, Qwen adapters generate, and deterministic code owns composition / hierarchy / ARIA / validation — which is what lets SemantiK make an auditable WCAG conformance claim. The full 13-stage cascade (extraction → feature engineering → BERT council → structure graph → Qwen specialist generation → hard/soft gates → assembly → theta semantic-preservation → exit decider) is documented in **[`SemantiK/architecture.md`](SemantiK/architecture.md)**; this section covers only its pipeline position and wire contract.

**Key Capabilities**:
- License-clean local extraction (pikepdf + pypdfium2 + pdfplumber + Tesseract; no AGPL/GPL dependencies)
- BERT council classification (structure, semantic role, merge/split, table, math) + cross-BERT reranking
- Qwen specialist HTML generation (prose, table, math) — local GGUF or an opt-in hosted large-model seat
- Semantic HTML with content-type-specific subclassing (not just generic h1-h6)
- WCAG 2.2 AA compliance enforced by eliminating per-region + document-scope hard gates (axe, html5, heading-tree, landmark, lang, title)
- Dark mode, reduced motion, and responsive design support
- Quality report generation with confidence scoring and a per-document exit decision

**Source-provenance contract**:

Conversion output goes beyond generic heading tags: each region is emitted as content-type-specific semantic HTML and stamped with `data-semantik-*` provenance markers plus a `semantik:{slug}#{block_id}` sourceId. That vocabulary is the **stable source-provenance contract** every downstream Ed4All consumer (Courseforge staging, source-mapping, the chunker, the Ask path) binds to, so generated content can always be traced back to its exact source block. Downstream readers also accept the legacy pre-SemantiK provenance markers so un-migrated local corpora and paused runs still resolve.

The core provenance attributes carried on each block:

| Attribute | Purpose |
|---|---|
| `data-semantik-block-id` | Stable per-block identity — the target of the `semantik:{slug}#{block_id}` sourceId |
| `data-semantik-block-role` | Semantic role of the block (heading, prose, table, figure, ...) |
| `data-semantik-source` | Source reference binding the block to its origin page/region |
| `data-semantik-page-kind` | Classified page kind for the source page |
| `data-semantik-pages` | Source page number(s) the block was extracted from |
| `data-semantik-wcag` | Per-block WCAG conformance annotation |
| `data-semantik-cell-roles` | Table cell-role map (header/data) for accessible tables |
| `data-semantik-demoted-role` | Records a role-demotion decision for auditability |
| `data-semantik-fabricated` | Marks a deterministically fabricated region (e.g. a synthesized title) |

**Output Artifacts**:
- `{stem}_accessible.html` - Semantic, accessible HTML (carrying the `data-semantik-*` provenance markers)
- `{name}.quality.json` - Quality report with confidence score, WCAG results, content profile

**Entry Points**:
- `SemantiK/semantik_structure/cascade.py::run_full_cascade()` - Full v2 cascade over a single PDF
- `MCP/tools/pipeline_tools.py::_run_semantik_v2_conversion` - In-process Ed4All bridge seam (the `semantik_conversion` phase)

---

### 2. Courseforge (Course Generation & Packaging)

**Purpose**: Take the accessible HTML produced by SemantiK and apply digital pedagogy and instructional design to create a complete digital course package (IMSCC).

**Pipeline Position**: Second stage - transforms accessible content into structured courseware.

```
SemantiK HTML Output               Courseforge Pipeline                    IMSCC Package
================                   ====================                   =============

                    +--- exam-research --------+
Textbook HTML ----->|                          |
                    +--- requirements ----+    |
Exam Objectives --->|    collector        |    v
                    |                     +--> course-outliner
                    |                              |
                    |                    +---------+----------+
                    |                    |                    |
                    |              objective-           content-generator
                    |              synthesizer          (10 parallel max)
                    |                    |                    |
                    |                    v                    v
                    |              Bloom's Aligned     HTML Modules
                    |              Objectives          (600+ words each)
                    |                    |                    |
                    |                    +--------+-----------+
                    |                             |
                    |                    quality-assurance
                    |                    oscqr-evaluator
                    |                             |
                    |                             v
                    +----------> brightspace-packager ---> IMSCC 1.3
```

**Key Capabilities**:
- 19 specialized agents coordinated via orchestrator protocol
- Learning Science RAG corpus (1,144 chunks, 16 pedagogical domains)
- Frameworks: UDL, ADDIE, SAM, Bloom's Taxonomy, Cognitive Load Theory
- Pattern prevention (22 identified anti-patterns)
- OSCQR quality evaluation (70% pre-dev, 90% pre-prod, 100% accessibility)
- Interactive component library (flip cards, accordions, tabs, self-checks)
- Multi-LMS IMSCC support (Brightspace, Canvas, Blackboard, Moodle, Sakai)

**Input Sources**:
- `Courseforge/inputs/textbooks/` - SemantiK accessible-HTML output (staged)
- `Courseforge/inputs/exam-objectives/` - Certification exam objectives
- `Courseforge/inputs/existing-packages/` - IMSCC for intake/remediation

**Output Artifacts**:
- `Courseforge/exports/{project_id}/{course_name}.imscc` - Final course package
- `Courseforge/exports/{project_id}/03_content_development/` - Generated HTML modules
- `runtime/training-captures/courseforge/{COURSE_CODE}/` - Decision capture JSONL

---

### 3. Trainforge (Assessment-Based RAG Training)

**Purpose**: Process the IMSCC course package to create a RAG corpus with pedagogically sound assessments, validated against Bloom's taxonomy.

**Pipeline Position**: Third stage - transforms course content into searchable RAG corpus and assessment data.

```
IMSCC Package                Trainforge Pipeline                    RAG Corpus
=============                ====================                   ==========

                  +---> IMSCC Parser --------+
Course Package -->|     (multi-LMS detect)   |
                  |                          v
                  |     HTML Content Parser ---> Learning Objectives
                  |     (sections, concepts,     Bloom's Levels
                  |      Bloom's detection)      Key Concepts
                  |                          |
                  |              +-----------+
                  |              v
                  |     RAG Indexer (TF-IDF)
                  |     Multi-Query Retrieval
                  |     Reciprocal Rank Fusion
                  |              |
                  |              v
                  |     Assessment Generator
                  |     (MCQ, T/F, Essay, etc.)
                  |     Distractor Targeting
                  |     Leak Detection
                  |              |
                  |              v
                  +---> Assessment Validator
                        (quality >= 0.9)
                        (Bloom alignment 100%)
                        (max 3 revision loops)
                              |
                              v
                        Assessment JSON
                        + Decision Captures
                        + RAG Corpus Chunks
```

**Key Capabilities**:
- IMSCC parsing with LMS auto-detection (Brightspace, Canvas, Blackboard, Moodle, Sakai)
- Bloom's taxonomy verb analysis for cognitive level detection
- TF-IDF retrieval with multi-query decomposition and Reciprocal Rank Fusion
- 6 question types: MCQ, True/False, Fill-in-Blank, Short Answer, Essay
- Distractor misconception targeting with quality validation
- Answer leak detection and prevention
- Decision capture for every generation decision (JSONL)

**Quality Thresholds**:
| Metric | Minimum |
|---|---|
| Objective Coverage | 90% |
| Bloom Alignment | 100% |
| Question Quality | 75% |
| Overall Score | 90% |

**Output Artifacts**:
- `Trainforge/output/{assessment_id}.json` - Assessment with questions, rationale, RAG metrics
- `runtime/training-captures/trainforge/{COURSE_CODE}/` - Decision capture JSONL
- `LibV2/courses/{slug}/imscc_chunks/chunks.jsonl` - IMSCC chunkset (Phase 7c rename of `corpus/chunks.jsonl`)

---

### 4. LibV2 (Unified Content Repository)

**Purpose**: Store ALL pipeline artifacts together - raw inputs, SemantiK accessible HTML, course packages, and RAG corpus - in a single, indexed, retrievable repository.

**Pipeline Position**: Final stage - unified archival and retrieval.

```
Pipeline Outputs              LibV2 Storage                     Retrieval
================              =============                     =========

Raw PDFs ---------> source/pdf/                    
SemantiK HTML ----> source/html/           libv2 retrieve "query"
Course IMSCC -----> source/imscc/               --domain physics
IMSCC Chunks -----> imscc_chunks/chunks.jsonl   --chunk-type example
Knowledge Graph --> graph/concept_graph.json    --limit 10
Pedagogy Model ---> pedagogy/pedagogy_model.json
Quality Report ---> quality/quality_report.json
Training Config --> training_specs/dataset_config.json

                    manifest.json (unified metadata)
                         |
                         v
                    catalog/ (derived indexes)
                    by_division/, by_domain/, by_subdomain/
```

**Per-Course Storage Structure**:
```
LibV2/courses/{slug}/
  manifest.json              # Unified metadata (classification, provenance, quality)
  source/
    pdf/                     # Original input PDFs (with SHA-256 checksums)
    html/                    # SemantiK accessible HTML output
    imscc/                   # Courseforge IMSCC package
  corpus/
    chunks.jsonl             # RAG corpus (streaming format)
    corpus_stats.json        # Chunk type/difficulty distribution
  graph/
    concept_graph.json       # Knowledge graph
  pedagogy/
    pedagogy_model.json      # Teaching patterns & learning sequences
  training_specs/
    dataset_config.json      # SLM training configuration
  quality/
    quality_report.json      # OSCQR scores, validation status
```

**Key Capabilities**:
- Streaming-first retrieval (line-by-line JSONL, no full corpus load)
- TF-IDF ranking with multi-query decomposition and RRF fusion
- Classification hierarchy: Division > Domain > Subdomain > Topic > Subtopic
- Source artifact tracking with SHA-256 checksums
- Cross-course retrieval for comparative analysis
- Schema validation (strict mode)
- Training data export (Alpaca, OpenAI, DPO, Raw formats)

**Retrieval Methods**:
- `libv2 retrieve "query"` - Single-query TF-IDF retrieval
- `libv2 multi-retrieve "query"` - Multi-query with decomposition and fusion
- `libv2 catalog search` - Metadata-only catalog search (zero token cost)

---

## End-to-End Pipeline Flow

### Textbook-to-Course Pipeline (`textbook_to_course`)

The primary unified workflow that chains all four components:

```
Phase                    Component    Agent(s)                  Output
=====                    =========    ========                  ======

1. semantik_conversion   SemantiK     semantik-converter        Accessible HTML + Quality JSON
       |
       v
2. staging               Pipeline     textbook-stager           Staged files in Courseforge/inputs/
       |
       v
3. objective_extraction  Courseforge   textbook-ingestor         Project + Learning Objectives
       |
       v
4. course_planning       Courseforge   course-outliner           Course structure + Week plan
       |
       v
5. content_generation    Courseforge   content-generator (x10)   HTML modules (batched by week)
       |
       v
6. packaging             Courseforge   brightspace-packager      IMSCC 1.3 package
       |
       v
7. trainforge_assessment Trainforge   assessment-generator (x5) Assessment JSON + RAG corpus
       |    (optional)
       v
8. libv2_archival        LibV2        libv2-archivist           Unified storage (all artifacts)
       |
       v
9. finalization          Pipeline     brightspace-packager      Progress update + export
```

> The `semantik-converter` agent drives the `semantik_conversion` phase and
> emits the `data-semantik-*` / `semantik:{slug}#{block_id}` source-provenance
> contract that every downstream consumer binds to. Legacy pre-SemantiK
> provenance markers and phase names are still accepted on READ (checkpoint +
> phase_outputs aliases) so old paused runs still `--resume`.

### Inter-Phase Data Routing

Each phase's outputs are automatically routed to the next phase's inputs:

| Source Phase | Output Key | Target Phase | Input Parameter |
|---|---|---|---|
| `semantik_conversion` | `output_paths` | `staging` | (staged HTML paths) |
| `staging` | `staging_dir` | `objective_extraction` | (implicit) |
| `objective_extraction` | `project_id`, `objective_ids` | `course_planning`, `content_generation` | `project_id` |
| `content_generation` | `content_paths` | `packaging` | (implicit via project_id) |
| `packaging` | `package_path` | `trainforge_assessment` | `imscc_path` |
| `packaging` | `package_path` | `libv2_archival` | `imscc_path` |
| `semantik_conversion` | `output_paths` | `libv2_archival` | `html_paths` |
| `trainforge_assessment` | `output_path` | `libv2_archival` | `assessment_path` |
| `libv2_archival` | `course_slug` | `finalization` | `course_slug` |

### Validation Gates

Quality is enforced at critical boundaries between components:

```
                      SemantiK             Courseforge              Trainforge
                     ===========           ==============           ============

                     WCAG 2.2 AA           Content Structure        Assessment Quality
                     (min score 0.95)      (0 critical issues)      (min score 0.8)
                          |                      |                        |
                          v                      v                        v
                     [GATE: block]          [GATE: block]           [GATE: block]
                          |                      |                        |
                          |                IMSCC Structure           Bloom Alignment
                          |                (0 critical issues)      (min score 0.7)
                          |                      |                        |
                          |                [GATE: block]            [GATE: warn]
                          |                      |                        |
                          |                OSCQR Score              Leak Check
                          |                (min 0.7)               (0 leaks)
                          |                      |                        |
                          |                [GATE: warn]            [GATE: block]
```

---

## MCP Orchestrator

The unified MCP server (`MCP/server.py`) exposes all component tools through a single interface:

### Tool Categories

| Category | Tools | Component |
|---|---|---|
| **Core File** | `list_directory`, `read_file`, `write_file`, `file_info` | MCP Server |
| **Conversion** | `extract_and_convert_pdf` (registry-only seam → `_run_semantik_v2_conversion`) | SemantiK |
| **Courseforge** | `create_course_project`, `generate_course_content`, `package_imscc`, `intake_imscc_package`, `remediate_course_content` | Courseforge |
| **Trainforge** | `analyze_imscc_content`, `generate_assessments`, `validate_assessment`, `export_training_data` | Trainforge |
| **Pipeline** | `create_textbook_pipeline`, `stage_semantik_outputs`, `run_textbook_pipeline`, `get_pipeline_status`, `validate_semantik_markers` | Pipeline |
| **LibV2** | `archive_to_libv2` | LibV2 |
| **Orchestrator** | `create_workflow`, `get_workflow_status`, `dispatch_agent_task`, `poll_task_completions`, `execute_workflow_task` | Orchestrator |
| **Analysis** | `analyze_training_data`, `get_quality_distribution`, `preview_export_filter` | Analysis |

### Orchestration Policies

1. **Maximum 10 simultaneous tasks per batch** - Enforced by executor
2. **One agent = one file** - No shared file editing, no concurrent writes
3. **Batch completion required** - All tasks in batch must complete before next batch
4. **Dependency resolution** - Strict mode, no phase skipping without completion
5. **Retry with backoff** - 3 retries at 5s, 30s, 120s for transient errors
6. **Poison pill detection** - 3 same-pattern failures in 5 minutes stops the batch
7. **Phase checkpointing** - Crash recovery without re-running completed phases
8. **Validation gates** - Critical gates block progression, warning gates log and continue

---

## Decision Capture

Every pipeline decision is logged to `runtime/training-captures/` in JSONL format for model training:

```
runtime/training-captures/
  semantik/{COURSE_CODE}/             # SemantiK conversion captures
    decisions_{PDF_NAME}_{TIMESTAMP}.jsonl
  courseforge/{COURSE_CODE}/
    phase_input-research/
    phase_content-generator/
    phase_brightspace-packager/
  trainforge/{COURSE_CODE}/
    phase_content-analysis/
    phase_question-generation/
    phase_validation/
  textbook-pipeline/{COURSE_CODE}/
    pipeline_decisions_{TIMESTAMP}.jsonl
```

**Required Fields**: `decision_type`, `decision`, `rationale` (min 20 chars)

**Export Formats**: Alpaca, OpenAI, DPO, Raw JSONL

---

## Configuration

| File | Purpose |
|---|---|
| `config/workflows.yaml` | Workflow phase definitions, retry policies, validation gates |
| `config/agents.yaml` | Agent capabilities, tool mappings, project paths |
| `CLAUDE.md` | Root orchestrator protocol (this overrides all) |
| `SemantiK/CLAUDE.md` | SemantiK conversion-cascade guidance (see also `SemantiK/architecture.md`) |
| `Courseforge/CLAUDE.md` | Courseforge agent and content guidance |
| `Trainforge/CLAUDE.md` | Trainforge RAG and assessment guidance |
| `LibV2/CLAUDE.md` | LibV2 retrieval and storage guidance |

---

## Available Workflows

| Workflow | Pipeline | Max Concurrent |
|---|---|---|
| `textbook_to_course` | SemantiK -> Courseforge -> Trainforge -> LibV2 | 10 |
| `course_generation` | Courseforge (from objectives) | 10 |
| `rag_training` | Trainforge assessment generation | 5 |
| `trainforge_train` | Train a course-pinned SLM adapter (post-import LibV2 stage) | 1 |

---

## Extending the Architecture

This document serves as the canonical reference for Ed4All's pipeline architecture. When adding new functionality:

1. **New component**: Add a section to this document describing its pipeline position, inputs/outputs, and capabilities
2. **New workflow phase**: Update the pipeline flow diagram and inter-phase routing table
3. **New MCP tool**: Add to the tool categories table and register in `MCP/server.py`
4. **New validation gate**: Add to the validation gates diagram and `config/workflows.yaml`
5. **New agent**: Add to `config/agents.yaml` and the relevant component section
6. **New storage artifact**: Update the LibV2 per-course storage structure

Keep this document in sync with the implementation. It is the single source of truth for how the pipeline fits together.
