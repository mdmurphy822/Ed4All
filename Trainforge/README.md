<div align="center">

<pre align="center">
╭─────────────────────────────────────────────────────────────────────────────────────────────╮
│  ████████╗ ██████╗   █████╗  ██╗ ███╗   ██╗ ███████╗  ██████╗  ██████╗   ██████╗  ███████╗  │
│  ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║ ████╗  ██║ ██╔════╝ ██╔═══██╗ ██╔══██╗ ██╔════╝  ██╔════╝  │
│     ██║    ██████╔╝ ███████║ ██║ ██╔██╗ ██║ █████╗   ██║   ██║ ██████╔╝ ██║  ███╗ █████╗    │
│     ██║    ██╔══██╗ ██╔══██║ ██║ ██║╚██╗██║ ██╔══╝   ██║   ██║ ██╔══██╗ ██║   ██║ ██╔══╝    │
│     ██║    ██║  ██║ ██║  ██║ ██║ ██║ ╚████║ ██║      ╚██████╔╝ ██║  ██║ ╚██████╔╝ ███████╗  │
│     ╚═╝    ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝ ╚═╝  ╚═══╝ ╚═╝       ╚═════╝  ╚═╝  ╚═╝  ╚═════╝  ╚══════╝  │
╰─────────────────────────────────────────────────────────────────────────────────────────────╯
</pre>

# Trainforge

### Turn packaged courses into grounded learning data—and optional adapters

Trainforge transforms IMS Common Cartridge packages into structured chunks,
assessments, knowledge artifacts, and source-grounded SFT and DPO pairs.
Adapter training is an explicit, operator-controlled stage.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Training](https://img.shields.io/badge/Training-Optional-7C3AED)](../docs/operations/nemotron-lora-canary.md)
[![License](https://img.shields.io/badge/License-Apache--2.0-22C55E)](../LICENSE)

[Quick start](#quick-start) · [See the flow](#from-course-package-to-training-assets) · [Understand licensing](../docs/LICENSING.md) · [Read the architecture](architecture.md)

</div>

---

## What Trainforge delivers

- **A reusable course corpus.** The canonical chunker preserves instructional
  structure, metadata, and source references for downstream retrieval.
- **Learning-aware artifacts.** Assessments, objectives, concept relationships,
  and pedagogy metadata stay connected to the course they came from.
- **Grounded training pairs.** SFT instructions and DPO preferences are
  synthesized from course artifacts and validated before publication.
- **Optional adapter training.** A separate post-import stage can fit and
  evaluate a course-pinned LoRA adapter, with provenance recorded in LibV2.

## From course package to training assets

```mermaid
flowchart LR
    package["IMS Common Cartridge<br/>course package"]

    subgraph prepare["Prepare the course corpus"]
        direction LR
        parse["Parse content,<br/>objectives + assessments"]
        chunks["Canonical v4 chunks<br/>metadata + provenance"]
        graph["Concept + pedagogy<br/>artifacts"]
        parse --> chunks --> graph
    end

    subgraph synthesize["Create grounded learning data"]
        direction TB
        sft["SFT instruction pairs"]
        dpo["DPO preference pairs"]
        gates["Validation + licensing<br/>preflight"]
        sft --> gates
        dpo --> gates
    end

    subgraph optional["Operator-controlled training"]
        direction TB
        lora["Optional LoRA fit"]
        evaluation["Evaluation + model card"]
        lora --> evaluation
    end

    archive["LibV2 course archive"]

    package --> parse
    chunks --> sft
    chunks --> dpo
    graph --> archive
    chunks --> archive
    gates --> archive
    gates -. explicit opt-in .-> lora
    evaluation --> archive

    classDef input fill:#eef6ff,stroke:#2563eb,color:#172554,stroke-width:2px;
    classDef corpus fill:#f0fdf4,stroke:#16a34a,color:#14532d;
    classDef data fill:#fff7ed,stroke:#ea580c,color:#7c2d12;
    classDef train fill:#faf5ff,stroke:#9333ea,color:#581c87;

    class package input;
    class parse,chunks,graph,archive corpus;
    class sft,dpo,gates data;
    class lora,evaluation train;
```

In plain language: Trainforge parses a packaged course, produces a structured
corpus and knowledge artifacts, and uses those grounded inputs to create SFT
and DPO data. Validated artifacts can be archived without training anything.
LoRA fitting and evaluation happen only when an operator explicitly starts the
post-import training workflow.

## Quick start

Install Ed4All from the repository root. Add the large training dependencies
only when you intend to evaluate or fit an adapter:

```bash
pip install -e '.[full]'
pip install -e '.[full,training]'  # optional adapter work
```

Build Trainforge artifacts from an existing IMSCC package:

```bash
ed4all run rag_training \
  --corpus <IMSCC_PATH> \
  --course-name <COURSE_NAME>
```

Train from an already imported LibV2 course only after reviewing the training
environment and licensing requirements:

```bash
ed4all run trainforge_train \
  --course-name <COURSE_SLUG> \
  --base-model <SUPPORTED_MODEL>
```

The full `textbook_to_course` workflow also reaches Trainforge; its training
stages run only when explicitly enabled. Generated course data belongs under
the ignored LibV2 course archive, not in Git.

## Training and licensing are deliberate

Training-pair synthesis and adapter fitting are different surfaces. The model
provider that authors SFT/DPO content, the underlying teacher license, the base
model license, and the source corpus rights all affect whether an artifact can
be distributed. The deterministic `mock` provider is for plumbing tests, not a
shippable training corpus. Read [Licensing and ToS posture](../docs/LICENSING.md)
before synthesis or training.

Production fitting is GPU-bound and operator-driven. Follow the
[Nemotron LoRA canary runbook](../docs/operations/nemotron-lora-canary.md) and
the repository-managed training environment; do not treat a successful dry run
as approval to train or promote an adapter.

## Key surfaces

| Surface | Role |
|---|---|
| `chunker/` | Shared canonical chunking contract |
| `parsers/` | IMSCC, QTI, HTML, and provenance extraction |
| `generators/` | Assessments and grounded pair generation |
| `synthesis/` | Canonical SFT/DPO synthesis implementation |
| `rag/` | Concept, pedagogy, and retrieval-support artifacts |
| `training/` | Optional LoRA fitting, evaluation, and model cards |

## Documentation

- [Trainforge architecture](architecture.md)
- [Trainforge operating contract](CLAUDE.md)
- [Behavior flags](../docs/operations/behavior-flags-trainforge.md)
- [Pipeline invocation](../docs/operations/pipeline-invocation.md)
- [Installation and local dependencies](../docs/operations/installation.md)
- [Licensing and ToS posture](../docs/LICENSING.md)

## License

Trainforge is distributed with Ed4All under the [Apache License 2.0](../LICENSE).
