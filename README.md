# Ed4All

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Automate the creation of high-quality knowledge domain packages — accessible content, structured courses, and concept graphs — from any source material.**

Building a knowledge corpus for AI tutoring, RAG retrieval, or LLM fine-tuning currently requires weeks of manual curation: extracting content, structuring it pedagogically, tagging it with learning science metadata, and validating quality. Ed4All reduces that to a single pipeline run.

Give it source materials and a knowledge domain. It produces three outputs:

1. **Accessible HTML** -- WCAG 2.2 AA compliant versions of the original materials, with semantic structure, proper heading hierarchy, and full assistive technology support
2. **Digital Course Packages** -- LMS-ready IMSCC packages with weekly modules, Bloom's-aligned learning objectives, interactive assessments, and machine-readable instructional design metadata
3. **Knowledge-Domain Language Graphs** -- RAG-optimized corpus with concept co-occurrence graphs, pedagogical metadata on every chunk, and structured training data ready for retrieval or fine-tuning

> **Current state (v0.2.0).** KG-quality hardening complete across 6 waves: unified schema directory (`schemas/`), canonical ontology helpers under `lib/ontology/`, 8 concept-graph edge types, opt-in flags for re-chunk-stable IDs and strict validation, and typed evidence per inference rule. See [`schemas/ONTOLOGY.md`](schemas/ONTOLOGY.md) § 12 for the full v0.2.0 change summary, and [`plans/kg-quality-review-2026-04/review.md`](plans/kg-quality-review-2026-04/review.md) for the review this work was based on.

### Why this matters

Every chunk in the output carries Bloom's taxonomy level, content type classification, key terms with definitions, misconceptions, and learning outcome references. This isn't a text dump — it's a pedagogically structured knowledge representation that LLMs can use for grounded generation, tutoring, and domain-specific reasoning.

The concept graph captures tag co-occurrence with pedagogical metadata (Bloom's levels, content types, key terms) and 8 typed semantic relationships — 3 taxonomic (`is-a`, `prerequisite`, `related-to`) plus 5 pedagogical (`assesses`, `exemplifies`, `misconception-of`, `derived-from-objective`, `defined-by`) — in a parallel `concept_graph_semantic.json` artifact. A physics corpus produces physics concepts; an accessibility corpus produces accessibility concepts — no manual ontology work required.

### Who this is for

- **EdTech developers** building AI tutors that need domain-specific, pedagogically structured training data
- **Universities and instructional designers** creating accessible online courses at scale
- **AI researchers** working on educational applications, RAG systems, or domain-adapted language models
- **Accessibility teams** remediating document libraries to meet WCAG 2.2 AA compliance

---

## Architecture

```
                         Source Materials
                    (PDFs, textbooks, web content)
                               |
                               v
                    +---------------------+
                    |        DART         |
                    |  Document Accessibility  |
                    |   Remediation Tool  |
                    +---------------------+
                               |
                     Accessible HTML (WCAG 2.2 AA)
                               |
                               v
                    +---------------------+
                    |     Courseforge      |
                    |  Course Generation  |
                    |   & IMSCC Packaging |
                    +---------------------+
                               |
              IMSCC Package + JSON-LD Metadata
                               |
                               v
                    +---------------------+
                    |     Trainforge      |
                    |  Content Extraction |
                    |  & RAG Processing   |
                    +---------------------+
                               |
              Chunked Corpus + Concept Graph
                               |
                               v
                    +---------------------+
                    |       LibV2         |
                    | Knowledge Repository|
                    |  & Language Graphs  |
                    +---------------------+
```

### What Each Stage Produces

**DART** converts source PDFs into semantic, accessible HTML:
- Multi-source synthesis (pdftotext + pdfplumber + OCR) for maximum fidelity
- WCAG 2.2 AA compliance: skip links, ARIA landmarks, heading hierarchy, alt text, table scopes
- Dark mode and reduced-motion support
- Quality reports with confidence scores

**Courseforge** generates structured course content:
- Multi-file weekly modules (overview, content pages, activities, self-check quizzes, summaries, discussions)
- Learning objectives with Bloom's taxonomy alignment (remember through create)
- Machine-readable metadata: `data-cf-*` HTML attributes and JSON-LD blocks per page
- IMSCC packaging compatible with Brightspace, Canvas, Blackboard, and Moodle

**Trainforge** processes course content into a RAG-optimized corpus:
- Pedagogical chunking (500-word target units preserving section boundaries)
- Metadata extraction: Bloom's levels, content types, key terms with definitions, misconceptions
- Chunk alignment: prerequisite concepts, teaching roles, learning outcome references
- Assessment generation grounded in source content with decision capture

**LibV2** stores and indexes the final knowledge artifacts:
- Flat-storage repository with semantic classification (division, domain, subdomain, topic)
- **Reference retrieval**: hand-rolled Okapi BM25 with character n-gram boosting, metadata filters (concept tags, LOs, Bloom's, teaching role, content type, week), and optional metadata-aware scoring (concept-graph overlap, LO match, prereq coverage). Intentionally bounded — a *reference implementation*, not a production retrieval system. See [ADR-002](docs/architecture/ADR-002-retrieval-scope.md) for the scope line and [`docs/libv2/reference-retrieval.md`](docs/libv2/reference-retrieval.md) for usage.
- **Retrieval rationale**: every result can carry a structured `rationale` payload (bm25/ngram/boost breakdown, matched concept tags, matched LOs, applied filters) so downstream consumers can reason about *why* a chunk was retrieved.
- **Gold-standard queries** per course at `LibV2/courses/<slug>/retrieval/gold_queries.jsonl`, hand-curated (not LO-derived), driving `libv2 retrieval-eval` recall@k + MRR.
- Concept co-occurrence graphs + typed-edge `concept_graph_semantic.json`
- Cross-package concept index at `LibV2/catalog/cross_package_concepts.json`
- Source artifact archival with SHA-256 checksums
- Quality metrics and validation reports

---

## Quick Start

### Prerequisites

- Python 3.9+
- (Optional) Tesseract OCR for PDF processing
- (Optional) poppler-utils for PDF extraction

### Installation

```bash
git clone https://github.com/mdmurphy822/Ed4All.git
cd Ed4All
python -m venv venv
source venv/bin/activate
pip install -e ".[full]"
```

### Full Pipeline (PDF to LibV2)

```bash
# One command: convert PDF, generate course, process corpus, import to LibV2
ed4all run textbook-to-course --corpus textbook.pdf --course-name COURSE_101 --weeks 12
```

### Stage by Stage

```bash
# 1. Convert PDF to accessible HTML
python DART/convert.py textbook.pdf -o DART/output/

# 2. Generate course from structured data
python Courseforge/scripts/generate_course.py course_data.json output_dir/

# 3. Package as IMSCC
python Courseforge/scripts/package_multifile_imscc.py output_dir/ course.imscc

# 4. Process through Trainforge
python -m Trainforge.process_course \
  --imscc course.imscc --course-code COURSE_101 \
  --division STEM --domain physics \
  --output Trainforge/output/course_101 \
  --align --import-to-libv2

# 5. Query the knowledge graph
python -m LibV2.tools.libv2.cli retrieve "your query" --limit 10
```

### MCP Server

```bash
cd MCP && python server.py
```

---

## Components

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **DART** | Document accessibility remediation | PDFs, combined JSON | WCAG 2.2 AA HTML |
| **Courseforge** | Course generation & packaging | Objectives, content data | IMSCC packages with metadata |
| **Trainforge** | Content extraction & RAG processing | IMSCC packages | Chunked corpus, concept graphs |
| **LibV2** | Knowledge repository & retrieval | Trainforge output | Indexed, searchable corpus |
| **MCP Server** | Unified tool orchestration | Tool calls | Coordinated pipeline execution |
| **CLI** | Pipeline management | Commands | Run reports, exports |

## Workflows

| Workflow | Description |
|----------|-------------|
| `textbook_to_course` | Full pipeline: PDF -> Accessible HTML -> Course -> Corpus -> LibV2 |
| `course_generation` | Generate course from objectives and content data |
| `intake_remediation` | Import and remediate existing IMSCC packages |
| `batch_dart` | Batch PDF to accessible HTML conversion |
| `rag_training` | Assessment-based training data generation |

## CLI

```bash
ed4all run textbook-to-course --corpus textbook.pdf --course-name COURSE_101   # Full pipeline (primary)
ed4all run <workflow> --dry-run --corpus <PATH> --course-name <NAME>           # Plan only, no execution
ed4all run <workflow> --resume <run_id>                                        # Resume a prior run
ed4all validate-run <run_id>                           # Validate run integrity
ed4all summarize-run <run_id>                          # Generate run report
ed4all diff-runs <run_a> <run_b>                       # Compare two runs
ed4all export-training <run_id> --format dpo           # Export training data
ed4all fsck                                            # LibV2 storage integrity check
ed4all list-runs                                       # List recent runs
```

---

## Metadata Flow

A key design principle is that instructional design metadata flows through the entire pipeline without loss:

```
Courseforge                    Trainforge                   LibV2
-----------                    ----------                   -----
Bloom's level on objectives -> JSON-LD extraction ->        bloom_level on chunks
Content type on sections    -> data-cf-* parsing ->         content_type_label
Key terms with definitions  -> Structured extraction ->     key_terms array
Misconceptions per topic    -> Page-level propagation ->    misconceptions array
Learning objective IDs      -> Priority chain matching ->   learning_outcome_refs
```

Trainforge uses a three-tier extraction priority: **JSON-LD** (authoritative, from Courseforge) > **data-cf-* attributes** (inline HTML) > **regex heuristics** (fallback for non-Courseforge content).

---

## Project Structure

```
Ed4All/
├── DART/                    # PDF to accessible HTML conversion
├── Courseforge/             # Course content generation & packaging
│   └── scripts/            # generate_course.py, package_multifile_imscc.py
├── Trainforge/              # Content extraction & RAG processing
│   ├── process_course.py   # IMSCC -> corpus pipeline
│   ├── align_chunks.py     # Pedagogical metadata alignment
│   ├── parsers/            # IMSCC, HTML, QTI parsers
│   └── generators/         # Assessment & content extraction
├── LibV2/                   # Knowledge repository
│   ├── courses/            # Flat-storage course data
│   ├── catalog/            # Derived indexes
│   └── tools/              # CLI & retrieval engine
├── MCP/                     # FastMCP server, orchestrator, and tools
│   ├── core/               # Orchestrator config, executor, workflow runner
│   ├── hardening/          # Error classifier, validation gates, checkpointing
│   └── ipc/                # Inter-process status tracking
├── cli/                     # CLI commands and run management
├── lib/                     # Shared libraries & validators
├── config/                  # Workflow & agent configs
├── schemas/                 # JSON schemas for validation
├── state/                   # Shared state & progress tracking
├── training-captures/       # Decision capture output
├── ci/                      # CI integrity checks
└── .github/                 # CI/CD workflows
```

## Running Tests

```bash
pytest                              # Run all tests
pytest --cov --cov-report=html      # With coverage
pytest Trainforge/tests/ -v         # Trainforge tests (75 tests)
pytest Courseforge/scripts/tests/   # Courseforge script tests
```

## Documentation

Each component has its own guide:

- [Orchestrator Protocol](CLAUDE.md) -- Main orchestration, workflows, and decision capture
- [DART](DART/CLAUDE.md) -- PDF conversion and multi-source synthesis
- [Courseforge](Courseforge/CLAUDE.md) -- Course generation, metadata output, templates
- [Trainforge](Trainforge/CLAUDE.md) -- Assessment generation, metadata extraction, RAG processing
- [LibV2](LibV2/CLAUDE.md) -- Repository structure, retrieval API, import/export

## License

MIT License - see [LICENSE](LICENSE)
