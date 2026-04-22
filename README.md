# Ed4All

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Turn a textbook PDF into an accessible, course-ready package — semantic HTML, weekly modules, learning objectives, assessments, and a knowledge graph — in a single command.**

Building a usable knowledge package from raw source material is weeks of manual work: extracting content, tagging it with learning science metadata, structuring it into pedagogically sound modules, writing aligned assessments, and validating accessibility. Ed4All runs that pipeline end-to-end, and everything it produces is WCAG 2.2 AA compliant by default.

## What you get

Point Ed4All at a textbook PDF (or a directory of PDFs) and a course name, and it produces:

- **Accessible HTML** — semantic structure, proper heading hierarchy, alt text for images, ARIA landmarks, keyboard navigation, dark mode, and full WCAG 2.2 AA coverage.
- **An LMS-ready IMSCC package** — weekly modules with pages, activities, self-checks, summaries, and discussions, importable into Brightspace, Canvas, Blackboard, or Moodle.
- **Bloom's-aligned learning objectives** — per module and per page, each tagged with a cognitive domain and linked back to the source content.
- **A knowledge graph** — chunked content with key terms, misconceptions, learning-outcome references, and an 8-relation concept graph covering taxonomic and pedagogical structure.
- **A reusable archive** — the course is indexed into a local knowledge repository you can query with BM25 retrieval, filter by concept or objective, and reuse across courses.

Every chunk carries its Bloom's level, content type, key terms, misconceptions, and the original PDF region it came from, so downstream LLMs can ground their answers in cited source material.

## Who it's for

- **Instructors and instructional designers** producing online courses from textbook source material at scale.
- **Accessibility teams** remediating document libraries to WCAG 2.2 AA compliance.
- **EdTech and ML teams** building AI tutors, RAG assistants, or domain-adapted language models that need pedagogically structured training data.
- **Researchers** studying retrieval quality, assessment generation, or learning-science-aligned content representations.

## Quick start

Requires Python 3.9+. Optional system tools (`tesseract-ocr`, `poppler-utils`) improve extraction on scanned or image-heavy PDFs.

```bash
git clone https://github.com/mdmurphy822/Ed4All.git
cd Ed4All
pip install -e ".[full]"

# Convert a textbook PDF into a full course package
ed4all run textbook-to-course --corpus my_textbook.pdf --course-name MY_COURSE_101
```

By default Ed4All runs in **local mode** — no API key required. To route through the Anthropic API instead, set `ANTHROPIC_API_KEY` and add `--mode api`.

That single command runs the full pipeline — accessibility conversion, objective synthesis, course planning, module generation, IMSCC packaging, knowledge-graph building, and archival. The IMSCC file lands in `Courseforge/exports/`, and the searchable archive lands in `LibV2/courses/`.

Other useful commands:

```bash
ed4all run --help                                     # List workflows and flags
ed4all run textbook-to-course --dry-run ...           # Plan only, no execution
ed4all run textbook-to-course --resume <run_id>       # Resume an interrupted run
ed4all list-runs                                      # Show recent runs
```

## What's inside

Ed4All is organised around four components that each do one job well, plus the glue that orchestrates them:

- **DART** turns PDFs into accessible, semantic HTML using multi-source extraction (text layer, layout analysis, OCR, and optional LLM classification) with per-block source provenance.
- **Courseforge** generates structured weekly course modules with learning objectives, assessments, interactive components, and rich machine-readable metadata, and packages them as IMSCC.
- **Trainforge** extracts content from the course package into pedagogically tagged chunks, builds a typed concept graph, and generates Bloom's-aligned assessments.
- **LibV2** is the archive and retrieval layer: a flat-storage course repository with BM25 retrieval, metadata filters, and cross-course concept indexes.

Supporting directories: **MCP** hosts the orchestrator and tool server, **cli** is the `ed4all` command line entry point, and **lib** holds shared validators and ontology helpers. Output artefacts land under `Courseforge/exports/`, `LibV2/courses/`, and `training-captures/`.

## Going deeper

- Developer guide and orchestration protocol: [`CLAUDE.md`](CLAUDE.md)
- Component guides: [`DART/CLAUDE.md`](DART/CLAUDE.md), [`Courseforge/CLAUDE.md`](Courseforge/CLAUDE.md), [`Trainforge/CLAUDE.md`](Trainforge/CLAUDE.md), [`LibV2/CLAUDE.md`](LibV2/CLAUDE.md)
- Ontology and schemas: [`schemas/ONTOLOGY.md`](schemas/ONTOLOGY.md)

## License

MIT — see [LICENSE](LICENSE).
