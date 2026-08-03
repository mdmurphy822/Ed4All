<div align="center">

<pre align="center">
╭────────────────────────────────────────────────────────────╮
│      ███████╗██████╗ ██╗  ██╗ █████╗ ██╗     ██╗           │
│      ██╔════╝██╔══██╗██║  ██║██╔══██╗██║     ██║           │
│      █████╗  ██║  ██║███████║███████║██║     ██║           │
│      ██╔══╝  ██║  ██║╚════██║██╔══██║██║     ██║           │
│      ███████╗██████╔╝     ██║██║  ██║███████╗███████╗      │
│      ╚══════╝╚═════╝      ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝      │
╰────────────────────────────────────────────────────────────╯
</pre>

# Ed4All

### Turn learning materials into accessible courses—and course-grounded AI

Ed4All transforms books, PDFs, HTML, and documentation into structured,
accessible HTML, modular digital course content, and LMS-ready IMS Common
Cartridge packages.

From the same source-grounded content, Ed4All can build a searchable course
library, generate supervised fine-tuning (SFT) and preference (DPO) pairs, and
optionally train a course-specific LoRA adapter.

Its custom hybrid retrieval layer combines lexical BM25 and dense vector search
with reciprocal rank fusion (RRF), helping applications answer questions from
the course's indexed content.

**One source. Four useful outcomes.**

Accessible HTML · Digital course + IMSCC · Grounded training data · Hybrid retrieval

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache--2.0-22C55E)](LICENSE)
[![CI](https://github.com/mdmurphy822/Ed4All/actions/workflows/ci.yml/badge.svg)](https://github.com/mdmurphy822/Ed4All/actions/workflows/ci.yml)

[Get started](#quick-start) · [See the pipeline](#from-source-to-course-grounded-ai) · [Explore the components](#components) · [Read the documentation](#documentation)

</div>

---

## What Ed4All does

- **Converts source material** into semantic, accessibility-oriented HTML with source provenance and automated validation.
- **Builds digital courses** with modules, learning objectives, activities,
  assessments, and machine-readable educational metadata.
- **Packages courses for an LMS** using IMS Common Cartridge (IMSCC).
- **Indexes course content** in a reusable local library with lexical, semantic, and hybrid-RRF retrieval.
- **Generates grounded training data** as SFT instruction pairs and DPO
  preference pairs derived from course content.
- **Optionally trains a LoRA adapter** and evaluates it alongside the course's
  retrieval system.

## From source to course-grounded AI

```text
Books and learning materials
            |
            v
Structured, accessible HTML
            |
            v
Modular digital course
      |             |
      |             +--> IMS Common Cartridge package
      |
      +--> Searchable course archive
      |          |
      |          +--> BM25 + dense vectors --> RRF --> grounded answers
      |
      +--> Source-grounded SFT + DPO pairs
                 |
                 +--> optional LoRA adapter
```

In plain language: Ed4All converts source material, organizes it as a course,
packages it for LMS delivery, and indexes the result for search. The same
grounded course content can also supply training pairs and, when explicitly
enabled, a LoRA training workflow. Training is optional; the course package and
retrieval library remain useful on their own.

## Quick start

Ed4All requires Python 3.10 or newer. Tesseract OCR and Poppler improve
extraction from scanned or image-heavy PDFs.

```bash
git clone https://github.com/mdmurphy822/Ed4All.git
cd Ed4All
pip install -e ".[full]"

ed4all run textbook-to-course \
  --corpus <path-to-source> \
  --course-name <course-name>
```

Authoring and synthesis phases require a configured model provider. See the
[pipeline invocation guide](docs/operations/pipeline-invocation.md) for local
and hosted OpenAI-compatible endpoint setup.

The default install leaves out the largest machine-learning dependencies. Add
only the capabilities you need:

| Extra | Adds | Use it for |
|---|---|---|
| `embedding` | Sentence Transformers and PyTorch | Dense retrieval, hybrid RRF, and embedding-backed validators |
| `training` | Transformers, TRL, PEFT, and training dependencies | Optional SFT/DPO LoRA training on a supported GPU |

```bash
pip install -e '.[full,embedding]'
pip install -e '.[full,training]'
```

Before a production build, review the [full-run playbook](docs/operations/full-run-playbook.md)
and [licensing posture](docs/LICENSING.md).

## Choose your outcome

### Convert documents to accessible HTML

Create remediated HTML without generating a course:

```bash
ed4all convert <source> --output <output-directory>
```

See the [conversion guide](docs/operations/convert-verb.md).

### Build and package a digital course

```bash
ed4all run textbook-to-course \
  --corpus <path-to-source> \
  --course-name <course-name>
```

This orchestrates conversion, course planning and generation, validation,
IMSCC packaging, archival, and indexing according to the selected workflow and
configuration.

### Query existing course content

Query an archived and indexed course through the retrieval layer:

```bash
libv2 retrieve "<question>" --course <course-name> --engine hybrid-rrf
```

Hybrid RRF combines BM25 term matching with dense vector similarity. Results
retain course and chunk provenance so downstream answer systems can cite the
retrieved material. See [retrieval and serving](docs/architecture/retrieval-and-serving.md).

### Generate training data or train an adapter

Training-pair synthesis produces SFT instructions and DPO preferences from the
course's chunks and assessments. Adapter training is a separate, opt-in,
GPU-bound stage:

```bash
# Build a course and explicitly include its training stages.
ed4all run textbook-to-course \
  --corpus <path-to-source> \
  --course-name <course-name> \
  --with-training

# Or train from an already archived course.
ed4all run trainforge_train \
  --course-name <course-name> \
  --base-model <supported-model>
```

Model licenses and provider terms determine whether generated pairs and trained
derivatives are distributable. Read [Licensing and ToS posture](docs/LICENSING.md)
first. Full training runs and promotion decisions remain operator-driven.

## Why Ed4All

- **One grounded content lineage.** Every stage works from the same source material.
- **Accessibility is part of the pipeline.** Semantic structure and automated
  checks are designed to support WCAG 2.2 AA targets; final conformance still
  depends on the source, configuration, generated content, and human review.
- **Standards-based packaging.** Courseforge emits IMS Common Cartridge.
- **Retrieval is a first-class deliverable.** A course can be searched and
  queried without training an adapter.
- **Training is explicit.** LoRA stages do not attach to a default build unless
  the operator opts in.
- **Providers are configurable.** Authoring can use configured local or hosted
  OpenAI-compatible endpoints.

## Components

- **SemantiK** converts documents into accessibility-oriented HTML with source provenance.
- **Courseforge** creates modular course content, learning activities, and IMS
  Common Cartridge packages.
- **Trainforge** creates tagged chunks, assessments, knowledge structures,
  SFT/DPO pairs, and optional LoRA training inputs.
- **LibV2** archives and queries course content through lexical, semantic, and
  hybrid reciprocal-rank-fusion retrieval.

**MCP** orchestrates workflows; **cli**, **gui**, and **lib** provide the user
and shared service surfaces.

## Ways to run it

- **Command line:** `ed4all run --help` and the [invocation guide](docs/operations/pipeline-invocation.md).
- **Browser:** `ed4all gui` and the [GUI guide](gui/README.md).
- **Containers:** the provided Compose deployment and [Docker guide](docs/operations/docker.md).
- **Long-running workflows:** checkpoints, resume, and graceful stop are covered
  by the [operations runbook](docs/operations/full-run-playbook.md).

## Documentation

- [Architecture overview](ARCHITECTURE.md)
- [Pipeline flow](docs/architecture/pipeline-flow.md)
- [Validation gates](docs/validation/gates.md)
- [Licensing and ToS posture](docs/LICENSING.md)
- Component guides: [SemantiK](SemantiK/README.semantic.md),
  [Courseforge](Courseforge/README.md), [Trainforge](Trainforge/README.md), and
  [LibV2](LibV2/README.md)

## License

Ed4All is available under the Apache License 2.0. See [LICENSE](LICENSE).
