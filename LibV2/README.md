<div align="center">

<pre align="center">
╭──────────────────────────────────────────────────────╮
│  ██╗     ██╗██████╗ ██╗   ██╗██████╗                │
│  ██║     ██║██╔══██╗██║   ██║╚════██╗               │
│  ██║     ██║██████╔╝██║   ██║ █████╔╝               │
│  ██║     ██║██╔══██╗╚██╗ ██╔╝██╔═══╝                │
│  ███████╗██║██████╔╝ ╚████╔╝ ███████╗               │
│  ╚══════╝╚═╝╚═════╝   ╚═══╝  ╚══════╝               │
╰──────────────────────────────────────────────────────╯
</pre>

# LibV2

### Archive a course once. Retrieve its best evidence when you need it.

LibV2 is Ed4All's local course library: an auditable archive for structured
course artifacts with lexical BM25, dense semantic search, and hybrid
reciprocal rank fusion (RRF).

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Retrieval](https://img.shields.io/badge/Retrieval-BM25%20%2B%20Dense-2563EB)](#retrieval-flow)
[![License](https://img.shields.io/badge/License-Apache--2.0-22C55E)](../LICENSE)

[Quick start](#quick-start) · [See retrieval](#retrieval-flow) · [Browse commands](#everyday-commands) · [Read the architecture](../docs/architecture/retrieval-and-serving.md)

</div>

---

## What LibV2 delivers

- **A durable course archive.** Each course keeps its manifests, structured
  chunks, learning outcomes, graphs, quality artifacts, and source lineage
  together under one immutable slug.
- **Three explicit retrieval engines.** Use lexical BM25, dense semantic
  search, or hybrid RRF—without an external vector database.
- **Auditable results.** Retrieval preserves chunk identity and course metadata
  so downstream systems can inspect why evidence was selected.
- **Local lifecycle tools.** Import, validate, index, migrate, back up, evaluate,
  and query course archives through the LibV2 CLI.

## Retrieval flow

```mermaid
flowchart LR
    imported["Trainforge course artifacts"]

    subgraph archive["LibV2 course archive"]
        direction TB
        manifest["Manifest + hash-linked<br/>course metadata"]
        chunks["Structured course chunks"]
        catalog["Catalog + classification"]
        vectors["Local dense vector index"]
        manifest --> chunks
        manifest --> catalog
        chunks --> vectors
    end

    query["Course-scoped query"]

    subgraph retrieve["Choose a retrieval engine"]
        direction TB
        bm25["Lexical BM25"]
        dense["Dense semantic search"]
        fusion["Hybrid reciprocal<br/>rank fusion"]
        bm25 --> fusion
        dense --> fusion
    end

    results["Ranked chunks<br/>metadata + rationale"]

    imported --> manifest
    query --> bm25
    query --> dense
    chunks --> bm25
    vectors --> dense
    bm25 --> results
    dense --> results
    fusion --> results

    classDef input fill:#eef6ff,stroke:#2563eb,color:#172554,stroke-width:2px;
    classDef storage fill:#f0fdf4,stroke:#16a34a,color:#14532d;
    classDef search fill:#faf5ff,stroke:#9333ea,color:#581c87;
    classDef output fill:#fff7ed,stroke:#ea580c,color:#7c2d12;

    class imported,query input;
    class manifest,chunks,catalog,vectors storage;
    class bm25,dense,fusion search;
    class results output;
```

In plain language: LibV2 imports a course into a manifest-backed local archive.
A query can use BM25 for lexical matching, a local vector index for dense
similarity, or hybrid RRF to combine both rankings. Each engine returns bounded,
structured results; semantic and hybrid retrieval fail loudly when their
required vector index is missing or stale rather than pretending BM25 was used.

## Quick start

Install from the Ed4All repository root, then use the module CLI directly or
create the optional `libv2` shell alias documented in the operating contract.

```bash
pip install -e '.[full,embedding]'

# List archived courses without loading chunk content.
python -m LibV2.tools.libv2.cli catalog list

# Query one course with lexical retrieval.
python -m LibV2.tools.libv2.cli retrieve "<QUERY>" \
  --course <COURSE_SLUG> \
  --engine lexical \
  --limit 10

# Build the local dense index, then run hybrid RRF.
python -m LibV2.tools.libv2.cli vector-index build \
  --course <COURSE_SLUG>
python -m LibV2.tools.libv2.cli retrieve "<QUERY>" \
  --course <COURSE_SLUG> \
  --engine hybrid-rrf \
  --limit 10
```

Course content is runtime data and remains ignored by Git. Use query-based
retrieval instead of opening an entire `chunks.jsonl`; keep result sets bounded
so content stays useful and reviewable.

## Everyday commands

```bash
# Inspect and validate.
python -m LibV2.tools.libv2.cli info <COURSE_SLUG>
python -m LibV2.tools.libv2.cli validate course <COURSE_SLUG>
python -m LibV2.tools.libv2.cli vector-index status --course <COURSE_SLUG>
python -m LibV2.tools.libv2.cli vector-index verify --course <COURSE_SLUG>

# Compare retrieval engines over an authored gold set.
python -m LibV2.tools.libv2.cli retrieval-benchmark \
  --course <COURSE_SLUG>
```

The lexical engine works without the dense index. Semantic and hybrid-RRF modes
require a valid index whose manifest matches the current chunkset. Rebuild the
index when the source chunk hash changes.

## Archive model

Each ignored `courses/<COURSE_SLUG>/` directory can carry canonical chunksets,
course metadata, a concept graph, training specifications, quality reports,
queries, retrieval evaluations, vector-index artifacts, and optional trained
adapters. Catalog files are derived navigation surfaces; course manifests and
their artifact hashes are the integrity boundary.

Do not edit archived course data directly. Use LibV2 commands so validation,
backup, migration, and index consistency checks remain in the loop.

## Documentation

- [LibV2 operating contract and command reference](CLAUDE.md)
- [Retrieval and serving architecture](../docs/architecture/retrieval-and-serving.md)
- [Library versioning and migration](../docs/operations/library-versioning.md)
- [Installation and local dependencies](../docs/operations/installation.md)
- [Validation gates](../docs/validation/gates.md)
- [Ed4All overview](../README.md)

## License

LibV2 is distributed with Ed4All under the [Apache License 2.0](../LICENSE).
