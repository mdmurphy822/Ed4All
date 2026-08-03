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

<!-- development-token-stats:start -->
<div align="center">
<table>
<thead>
<tr bgcolor="#1F6FEB"><th align="center" colspan="4"><font color="#FFFFFF">🎓 Development Token Tracking</font></th></tr>
<tr>
<td align="center" width="25%" bgcolor="#EDE9FE"><font color="#111827"><strong>12,886,474,014</strong><br><sub>🧠 DEVELOPMENT TOKENS</sub></font></td>
<td align="center" width="25%" bgcolor="#DBEAFE"><font color="#111827"><strong>53</strong><br><sub>🧭 SESSIONS</sub></font></td>
<td align="center" width="25%" bgcolor="#D1FAE5"><font color="#111827"><strong>6,441</strong><br><sub>💬 USER TURNS OBSERVED</sub></font></td>
<td align="center" width="25%" bgcolor="#FFEDD5"><font color="#111827"><strong>1,211,236</strong><br><sub>🧱 TRACKED TEXT LOC</sub></font></td>
</tr>
</thead>
<tbody>
<tr bgcolor="#334155"><th align="center"><font color="#FFFFFF">🤝 COLLABORATOR</font></th><th align="center"><font color="#FFFFFF">TOKENS</font></th><th align="center"><font color="#FFFFFF">SESSIONS</font></th><th align="center"><font color="#FFFFFF">USER TURNS</font></th></tr>
<tr><td align="center">Claude</td><td align="center">12,016,436,779</td><td align="center">38</td><td align="center">5,961</td></tr>
<tr><td align="center">Codex</td><td align="center">870,037,235</td><td align="center">15</td><td align="center">480</td></tr>
<tr bgcolor="#0E7490"><th align="center"><font color="#FFFFFF">↔️ TOKEN FLOW</font></th><th align="center"><font color="#FFFFFF">READ</font></th><th align="center"><font color="#FFFFFF">WRITTEN</font></th><th align="center"><font color="#FFFFFF">AVG / SESSION</font></th></tr>
<tr><td align="center">All sessions</td><td align="center">12,840,026,230</td><td align="center">46,447,784</td><td align="center">243,141,019</td></tr>
<tr bgcolor="#6D28D9"><th align="center"><font color="#FFFFFF">🔎 TOKEN DETAIL</font></th><th align="center"><font color="#FFFFFF">COUNT</font></th><th align="center"><font color="#FFFFFF">TOKEN DETAIL</font></th><th align="center"><font color="#FFFFFF">COUNT</font></th></tr>
<tr><td align="center">Fresh input</td><td align="center">24,636,112</td><td align="center">Cache writes</td><td align="center">285,091,038</td></tr>
<tr><td align="center">Cache reads</td><td align="center">12,530,299,080</td><td align="center">Model output</td><td align="center">46,447,784</td></tr>
<tr><td align="center">Reasoning output subset</td><td align="center">367,349</td><td align="center">Counted again in total</td><td align="center">No</td></tr>
<tr bgcolor="#0369A1"><th align="center"><font color="#FFFFFF">⏱️ SESSION DURATION</font></th><th align="center"><font color="#FFFFFF">CLAUDE AVG</font></th><th align="center"><font color="#FFFFFF">CODEX AVG</font></th><th align="center"><font color="#FFFFFF">COMBINED AVG</font></th></tr>
<tr><td align="center">First-to-last observed event</td><td align="center">18h 20m</td><td align="center">13h 51m</td><td align="center">17h 4m</td></tr>
<tr bgcolor="#C2410C"><th align="center"><font color="#FFFFFF">📚 TRACKED TEXT</font></th><th align="center"><font color="#FFFFFF">LINES</font></th><th align="center"><font color="#FFFFFF">TRACKED TEXT</font></th><th align="center"><font color="#FFFFFF">LINES</font></th></tr>
<tr><td align="center">Application source</td><td align="center">553,319</td><td align="center">Tests</td><td align="center">504,588</td></tr>
<tr><td align="center">Documentation</td><td align="center">37,933</td><td align="center">Tooling / configuration</td><td align="center">114,220</td></tr>
<tr><td align="center">Other text</td><td align="center">1,176</td><td align="center">Total physical lines</td><td align="center">1,211,236</td></tr>
</tbody>
</table>
<sub>[How these privacy-safe project metrics are counted →](docs/operations/development-token-stats.md)</sub>
</div>
<!-- development-token-stats:end -->

## From source to course-grounded AI

```mermaid
flowchart LR
    source["Books · PDFs · HTML<br/>Learning materials"]

    subgraph build["Build an accessible digital course"]
        direction LR
        semantik["SemantiK<br/>Structure + accessibility"]
        html["Accessible HTML<br/>with source provenance"]
        courseforge["Courseforge<br/>Modules + activities + assessments"]
        course["Modular digital course"]

        semantik --> html --> courseforge --> course
    end

    subgraph deliver["Deliver and reuse"]
        direction TB
        imscc["LMS-ready<br/>IMS Common Cartridge"]
        library["LibV2 course archive<br/>Searchable local library"]
    end

    subgraph intelligence["Grounded course intelligence"]
        direction TB
        retrieval["BM25 + dense retrieval"]
        rrf["Reciprocal rank fusion"]
        answers["Course-grounded results"]
        pairs["Trainforge<br/>SFT instructions + DPO preferences"]
        lora["Optional LoRA adapter"]

        retrieval --> rrf --> answers
        pairs -. operator opt-in .-> lora
    end

    source --> semantik
    course --> imscc
    course --> library
    library --> retrieval
    course --> pairs

    classDef sourceNode fill:#eef6ff,stroke:#2563eb,color:#172554,stroke-width:2px;
    classDef buildNode fill:#f0fdf4,stroke:#16a34a,color:#14532d;
    classDef deliveryNode fill:#fff7ed,stroke:#ea580c,color:#7c2d12;
    classDef intelligenceNode fill:#faf5ff,stroke:#9333ea,color:#581c87;

    class source sourceNode;
    class semantik,html,courseforge,course buildNode;
    class imscc,library deliveryNode;
    class retrieval,rrf,answers,pairs,lora intelligenceNode;
```

The flow has three layers: Ed4All first converts source material into accessible
HTML and a modular course; it then packages and archives that course; finally,
the archive supports hybrid retrieval while the grounded course content can
supply training pairs. LoRA training is a separate operator opt-in. The LMS
package, course archive, and retrieval system remain useful without training an
adapter.

## Quick start

Ed4All requires Python 3.10 or newer. Tesseract OCR and Poppler improve
extraction from scanned or image-heavy PDFs.

See the [installation guide](docs/operations/installation.md) for platform
dependencies, capability extras, Playwright, and the required third-party IMS
Common Cartridge schemas.

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

- [Installation and local dependencies](docs/operations/installation.md)
- [Architecture overview](ARCHITECTURE.md)
- [Pipeline flow](docs/architecture/pipeline-flow.md)
- [Validation gates](docs/validation/gates.md)
- [Licensing and ToS posture](docs/LICENSING.md)
- Component guides: [SemantiK](SemantiK/README.md),
  [Courseforge](Courseforge/README.md), [Trainforge](Trainforge/README.md), and
  [LibV2](LibV2/README.md)

## License

Ed4All is available under the Apache License 2.0. See [LICENSE](LICENSE).
