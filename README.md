# Ed4All

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Turn a textbook PDF into an accessible, course-ready package — semantic HTML, weekly modules, learning objectives, assessments, and a knowledge graph — in a single command.**

Building a usable knowledge package from raw source material is weeks of manual work: extracting content, tagging it with learning science metadata, structuring it into pedagogically sound modules, writing aligned assessments, and validating accessibility. Ed4All runs that pipeline end-to-end, and everything it produces is WCAG 2.2 AA compliant by default.

## What you get

Point Ed4All at a textbook PDF (or a directory of PDFs) and a course name, and it produces:

- **Accessible HTML** — semantic structure, proper heading hierarchy, alt text for images, ARIA landmarks, keyboard navigation, dark mode, and full WCAG 2.2 AA coverage.
- **An LMS-ready IMSCC package** — weekly modules with pages, activities, self-checks, summaries, and discussions, importable into Brightspace, Canvas, Blackboard, or Moodle.
- **Bloom's-aligned learning objectives** — per module and per page, each tagged with a cognitive domain and linked back to the source content.
- **A knowledge graph** — chunked content with key terms, misconceptions, learning-outcome references, and a typed concept graph covering taxonomic and pedagogical structure.
- **A reusable archive** — the course is indexed into a local knowledge repository you can query with BM25 retrieval, filter by concept or objective, and reuse across courses.

Every chunk carries its Bloom's level, content type, key terms, misconceptions, and the original PDF region it came from, so downstream LLMs can ground their answers in cited source material.

## Who it's for

- **Instructors and instructional designers** producing online courses from textbook source material at scale.
- **Accessibility teams** remediating document libraries to WCAG 2.2 AA compliance.
- **EdTech and ML teams** building AI tutors, RAG assistants, or domain-adapted language models that need pedagogically structured training data.
- **Researchers** studying retrieval quality, assessment generation, or learning-science-aligned content representations.

## How it works

`textbook-to-course` runs up to 21 phases end-to-end. Each phase checkpoints on
completion (a failed or stopped run resumes where it left off), quality gates
validate the artifacts between phases, and GPU model seats are started and
stopped automatically so only the models a phase needs are ever resident.

| # | Phase | What it does | GPU workload |
|---|-------|--------------|--------------|
| 1 | `semantik_conversion` | PDF → accessible semantic HTML (layout extraction + image alt text) | OCR + vision models |
| 2 | `heading_judge` | Re-levels ambiguous heading levels via a large-model judge *(optional)* | Large model |
| 3 | `staging` | Stages converted HTML for course generation | — |
| 4 | `chunking` | Emits the deterministic source chunkset | — |
| 5 | `objective_extraction` | Parses staged HTML into the textbook structure (chapters, sections, blocks) | — |
| 6 | `source_mapping` | Maps source blocks to course module pages | — |
| 7 | `course_planning` | Synthesizes terminal + component learning objectives from the structure | Large model |
| 8 | `concept_extraction` | Builds the typed concept / knowledge graph | Large model |
| 9 | `content_generation` | Single-pass content authoring *(alternative to 10–12)* | Large model |
| 10 | `content_generation_outline` | Two-pass tier 1: terse per-block outlines | Large model |
| 11 | `inter_tier_validation` | Structural validators over the outline tier (no LLM) | — |
| 12 | `content_generation_rewrite` | Two-pass tier 2: full HTML block bodies | Large model |
| 13 | `assessment_synthesis` | Grounded quizzes, assignments, and discussions as QTI/IMS XML *(optional)* | Large model |
| 14 | `post_rewrite_validation` | The largest gate set: prose entailment, claim support, block quality | Local validators (NLI + embeddings) |
| 15 | `packaging` | Packages the course as IMSCC | — |
| 16 | `imscc_chunking` | Emits the post-packaging retrieval chunkset | — |
| 17 | `trainforge_assessment` | Generates assessments from the packaged course *(optional)* | — |
| 18 | `training_synthesis` | Synthesizes instruction + preference training pairs *(optional)* | Large model |
| 19 | `libv2_archival` | Archives all artifacts to the local course library | — |
| 20 | `vector_indexing` | Builds the per-course vector index so the course is immediately askable | Local embeddings |
| 21 | `finalization` | Final validation and training-data export | — |

Once a course is archived, an optional follow-on workflow trains a
course-pinned adapter from the synthesized pairs (`ed4all run trainforge_train
--course-code <slug> --base-model <name>`):

| # | Phase | What it does | GPU workload |
|---|-------|--------------|--------------|
| 22 | `trainforge_train` | LoRA fine-tune of a small language model on the course's instruction + preference pairs (bf16 PEFT, licensing preflight, full provenance card) | Training (exclusive — all serving seats stopped) |
| 23 | post-training evaluation | 5-layer × 3-tier eval matrix vs the base model, adapter audit, promote / hold / reject decision | Local eval models |

**GPU workload** legend: *OCR + vision models* — the lightweight extraction and
alt-text seats; *Large model* — the main authoring/judging model seat; *Local
validators / embeddings* — in-process models (no serving seat); *—* —
deterministic CPU work. Seats in different groups are swapped at phase
boundaries automatically, with health and coherence checks at every start.

## Quick start

Requires Python 3.10+. Optional system tools (`tesseract-ocr`, `poppler-utils`) improve extraction on scanned or image-heavy PDFs.

```bash
git clone https://github.com/mdmurphy822/Ed4All.git
cd Ed4All
pip install -e ".[full]"

# Convert a textbook PDF into a full course package
ed4all run textbook-to-course --corpus my_textbook.pdf --course-name MY_COURSE_101
```

`[full]` installs the CPU-light surface: PDF-to-HTML conversion, the MCP server,
the dev/test toolchain, and the GUI. It deliberately leaves out the two heavy
ML extras so a default install never pulls multi-GB GPU wheels:

| Extra | Adds | When you need it |
|-------|------|------------------|
| `embedding` | `sentence-transformers` + `torch` | Semantic retrieval and the statistical-tier content validators. Without it those validators degrade to warnings and retrieval falls back to BM25. |
| `training` | `torch` + `transformers` + `trl` + `peft` + `bitsandbytes` | Fine-tuning a course-pinned SLM adapter (`ed4all run trainforge_train`). Requires a GPU. |

Install them alongside `[full]` only when needed:

```bash
pip install -e '.[full,embedding]'   # + semantic retrieval / statistical validators
pip install -e '.[full,training]'    # + SLM fine-tuning (GPU)
```

By default Ed4All runs in **local mode** — no API key required. To route orchestration through the Anthropic API instead, set `ANTHROPIC_API_KEY` and add `--mode api`.

Content generation is **model-agnostic**: every authoring, synthesis, and answer provider speaks the OpenAI-compatible API, so any local model server (Ollama, vLLM, llama.cpp) or hosted endpoint plugs in with just a base URL, an API key, and a model name — configuration, not code. Swap models or providers per task without touching the pipeline.

That single command runs the full pipeline — accessibility conversion, objective synthesis, course planning, module generation, IMSCC packaging, knowledge-graph building, and archival. The IMSCC file lands in `Courseforge/exports/`, and the searchable archive lands in `LibV2/courses/`.

### Prefer a GUI?

Ed4All ships a browser-based control panel for the whole pipeline — upload PDFs, manage API keys and environment, choose which model runs each task (including local **Ollama** and **vision/VLM** models), edit course topics and learning objectives, launch a full run or a single stage with live logs, and query the knowledge base — no command line required.

```bash
# One click: builds a virtualenv, installs, starts the server, opens your browser
./run-gui.sh           # macOS / Linux
run-gui.bat            # Windows (double-click)
```

If you already ran `pip install -e ".[full]"` above (it includes the GUI), just launch it directly:

```bash
ed4all gui             # serves http://127.0.0.1:8077
```

### Prefer containers?

A two-service Docker Compose stack serves the GUI on `http://localhost:8077`
alongside a local model backend — no API key, no command line:

```bash
docker compose up -d                                          # build + start the stack
docker compose exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M   # pull the answer model (one-shot)
# then open http://localhost:8077, upload a PDF, and run the pipeline
```

CPU-only by default; optional NVIDIA GPU support, volume layout, and remote-deploy auth are covered in [`docs/operations/docker.md`](docs/operations/docker.md).

Full GUI guide — the six panels, settings and secret handling, model routing, retrieval, and how Claude Code sessions can drive it: [`gui/README.md`](gui/README.md).

Other useful commands:

```bash
ed4all run --help                                     # List workflows and flags
ed4all run textbook-to-course --dry-run ...           # Plan only, no execution
ed4all run textbook-to-course --resume <run_id>       # Resume an interrupted run
ed4all stop <run_id>                                  # Checkpoint and pause at the next unit boundary
ed4all list-runs                                      # Show recent runs
```

### Running a full production build

The quick-start command above is the happy path and needs no configuration. A
full multi-hour build against local model seats — seat topology, the phase
sequence and what each phase produces, how to read gate outcomes, and the
resume/stop procedure — is covered in one place:
[`docs/operations/full-run-playbook.md`](docs/operations/full-run-playbook.md).

Environment for such a run starts from [`run-env.example.sh`](run-env.example.sh).
Read its hardware-profile section before copying any concurrency, batch-size, or
GPU-lifecycle setting: those values are tuned for a single-GPU large-memory host
and will exhaust VRAM on a small card unedited.

## The built-in assistant

Ed4All ships a local AI assistant for operating the pipeline — ask it what a
run is doing, why a gate failed, or have it start the next build:

```bash
ed4all assistant                          # interactive chat
ed4all assistant --once "why did my last run fail?"
ed4all assistant --debug                  # open a session pre-loaded with the
                                          # most recent failure's diagnostics
```

It speaks through a fixed set of typed tools — run status and gate reports,
bounded log tails, `ed4all doctor` diagnostics, build-cost and quality reports,
course library inspection, grounded Q&A against any built course's index, and a
small set of guarded actions (start / resume / stop runs, seat start/stop,
support bundles). It can also walk you through **model-seat setup**: it audits
your seat-swap environment variables, discovers running model containers, and
generates a ready-to-source configuration block. It never gets shell access or
free-form file access — everything flows through validated tools, so it can
help without being able to hurt.

The assistant is **model-agnostic**: it talks to any OpenAI-compatible local
endpoint (`ED4ALL_ASSISTANT_BASE_URL` / `ED4ALL_ASSISTANT_MODEL`), restricted
to localhost by design. The reference deployment — what it is tuned and tested
against — is NVIDIA's **Nemotron Nano** (the NeMo model family) served with
vLLM; set `ED4ALL_ASSISTANT_AUTOSTART=1` to let the CLI bring that seat up on
demand, and `ED4ALL_ASSISTANT_DEBUG_ON_FAILURE=1` to have failed pipeline runs
print the exact debug command to investigate them. The same assistant is
available as a chat panel in the GUI.

## What's inside

Ed4All is organised around four components that each do one job well, plus the glue that orchestrates them:

- **SemantiK** turns PDFs into accessible, semantic HTML using a license-clean extraction cascade (text layer, layout analysis, OCR, and learned structure/semantic classification) with per-block source provenance.
- **Courseforge** generates structured weekly course modules with learning objectives, assessments, interactive components, and rich machine-readable metadata, and packages them as IMSCC.
- **Trainforge** extracts content from the course package into pedagogically tagged chunks, builds a typed concept graph, and generates Bloom's-aligned assessments.
- **LibV2** is the archive and retrieval layer: a flat-storage course repository with BM25 retrieval, metadata filters, and cross-course concept indexes.

Supporting directories: **MCP** hosts the orchestrator and tool server, **cli** is the `ed4all` command line entry point, and **lib** holds shared validators and ontology helpers. Output artefacts land under `Courseforge/exports/`, `LibV2/courses/`, and `training-captures/`.

## Going deeper

- Developer guide and orchestration protocol: [`CLAUDE.md`](CLAUDE.md)
- Component guides: [`SemantiK/CLAUDE.md`](SemantiK/CLAUDE.md), [`Courseforge/CLAUDE.md`](Courseforge/CLAUDE.md), [`Trainforge/CLAUDE.md`](Trainforge/CLAUDE.md), [`LibV2/CLAUDE.md`](LibV2/CLAUDE.md)
- Ontology and schemas: [`schemas/ONTOLOGY.md`](schemas/ONTOLOGY.md)

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
