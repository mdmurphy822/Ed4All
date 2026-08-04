# Ed4All architecture

Ed4All turns source learning material into accessible web content, a portable
digital course, training-ready examples, and a searchable course library. Four
engines share a provenance contract, so a learner-facing answer or generated
artifact can be traced back through the course to its source.

This page is the public system map. Implementation details live in the linked
architecture and operations guides.

## The system at a glance

```mermaid
flowchart LR
    SOURCE["Books and learning material"] --> SEM["SemantiK<br/>Accessible HTML + provenance"]
    SEM --> COURSE["Courseforge<br/>Digital course + IMSCC"]
    COURSE --> TRAIN["Trainforge<br/>RAG corpus + SFT/DPO pairs"]
    TRAIN --> LIB["LibV2<br/>Archive + hybrid RRF retrieval"]
    LIB --> ANSWER["Grounded answers<br/>or optional LoRA training"]

    classDef source fill:#FFF7ED,stroke:#C2410C,color:#431407,stroke-width:2px;
    classDef transform fill:#EFF6FF,stroke:#1D4ED8,color:#172554,stroke-width:2px;
    classDef library fill:#ECFDF5,stroke:#047857,color:#052E2B,stroke-width:2px;
    classDef outcome fill:#F5F3FF,stroke:#6D28D9,color:#2E1065,stroke-width:2px;
    class SOURCE source;
    class SEM,COURSE,TRAIN transform;
    class LIB library;
    class ANSWER outcome;
```

The main workflow follows this write path. LibV2 also provides the read path:
lexical BM25 and dense retrieval produce ranked candidates, reciprocal rank
fusion (RRF) combines them, and the answer layer verifies grounding and citation
anchors before presenting a response.

## Four engines, one learning pipeline

### SemantiK: source material to accessible HTML

SemantiK's preferred path renders source pages, extracts their regions through
the GLM-OCR SDK, normalizes and enriches those regions with deterministic code,
and applies the Super heading judge before producing accessibility-oriented
HTML and provenance. Compatibility conversion paths remain isolated behind the
same downstream contract and are not the production-qualified default.

Its most important output is not just HTML. Every emitted content block carries
a stable source identity and page/region provenance. Downstream stages preserve
those references instead of treating generated course content as detached prose.

**Inputs:** PDF and supported document-derived content.

**Outputs:** semantic HTML, block-level provenance, and quality reports.

**Deep dive:** [SemantiK architecture](SemantiK/architecture.md) and
[SemantiK overview](SemantiK/README.md).

### Courseforge: accessible HTML to a digital course

Courseforge transforms source-grounded HTML and learning objectives into modular
course content. It plans instruction, authors learning activities and assessment
material, applies the project's content and accessibility checks, and packages
the result as IMS Common Cartridge (IMSCC) for LMS import workflows.

Course pages retain machine-readable learning objectives, concepts, teaching
roles, and source references. That metadata gives later assessment, retrieval,
and training stages a common instructional vocabulary.

**Inputs:** SemantiK HTML, objectives, or an existing supported IMSCC package.

**Outputs:** structured course pages, learning metadata, validation reports, and
an IMSCC package.

**Deep dive:** [Courseforge overview](Courseforge/README.md),
[Courseforge architecture](Courseforge/architecture.md), and
[getting started](Courseforge/docs/guides/getting-started.md).

### Trainforge: a course package to retrieval and training data

Trainforge parses course content into pedagogically meaningful chunks, builds a
typed concept graph, and prepares the data used by retrieval and evaluation. It
can also synthesize supervised fine-tuning (SFT) examples and direct preference
optimization (DPO) pairs through the project's license-controlled provider
routes.

Adapter training is optional. When requested, the post-import training stage can
produce and evaluate a course-pinned LoRA adapter with provenance hashes and a
model card. The course and retrieval corpus remain useful without training an
adapter.

**Inputs:** IMSCC packages and their source-grounding metadata.

**Outputs:** chunks, assessments, knowledge graphs, SFT/DPO pairs, quality
reports, and optional evaluated adapter artifacts.

**Deep dive:** [Trainforge overview](Trainforge/README.md),
[Trainforge architecture](Trainforge/architecture.md), and
[training-data licensing](docs/LICENSING.md).

### LibV2: the searchable course memory

LibV2 is the durable archive and retrieval layer. It keeps source artifacts,
course packages, chunks, graphs, training specifications, quality reports, and
optional model artifacts together under a course-level manifest.

Its custom hybrid retrieval path combines two complementary signals:

1. **Lexical retrieval** uses BM25 to reward precise term matches.
2. **Dense retrieval** finds semantically related passages through a local
   vector index.
3. **Reciprocal rank fusion** combines both rankings without requiring their raw
   scores to share a scale.

Retrieval results retain source metadata and a ranking rationale. The grounded
answer path can refuse unsupported questions and validates citation anchors
before returning an answer.

**Inputs:** the artifacts produced across the pipeline.

**Outputs:** a catalogued course archive, filterable retrieval results, and
grounded answers over existing content.

**Deep dive:** [LibV2 overview](LibV2/README.md),
[LibV2 architecture](LibV2/architecture.md), and
[retrieval and serving architecture](docs/architecture/retrieval-and-serving.md).

## Artifact contract

Each stage adds value without discarding the evidence created before it:

| Boundary | Contract passed forward |
|---|---|
| SemantiK -> Courseforge | Accessible HTML, stable block identities, source pages/regions, quality results |
| Courseforge -> Trainforge | IMSCC, learning objectives, authored components, concepts, and source references |
| Trainforge -> LibV2 | Canonical chunks, assessments, graphs, training pairs, and validation reports |
| LibV2 -> retrieval/training | Versioned manifests, content hashes, indexes, model cards, and promotion state |

Canonical shapes live under [`schemas/`](schemas/README.md). The current
ontology and identity rules are documented in
[`schemas/ONTOLOGY.md`](schemas/ONTOLOGY.md).

## Trust contract

Ed4All is designed so that a successful tool call is not, by itself, proof of a
good artifact.

- **Validation gates guard boundaries.** Workflow configuration attaches
  validators to the artifacts they govern. Blocking failures stop progression;
  warning results remain visible for review.
- **Provenance survives transformation.** Source block identities flow into
  course pages, chunks, graphs, retrieval results, and citations.
- **Model decisions are inspectable.** LLM call sites emit structured decision
  events with dynamic rationale and run context.
- **Training has a licensing boundary.** Development tooling and training-pair
  synthesis are separate surfaces. Provider and model eligibility is checked
  before synthesis artifacts are treated as shippable.
- **Retrieval prefers refusal to invention.** Grounding and citation checks can
  withhold an answer when the archived content does not support it.
- **Long stages are resumable.** Checkpoints, resume sidecars, and stop
  sentinels let operators interrupt work at unit boundaries.

The sources of truth are
[`config/workflows.yaml`](config/workflows.yaml) for phase and gate wiring,
[`docs/validation/gates.md`](docs/validation/gates.md) for gate behavior, and
[`docs/architecture/decision-capture.md`](docs/architecture/decision-capture.md)
for decision events.

## Orchestration

The `ed4all` CLI drives the workflow engine in `MCP/`. The engine resolves phase
dependencies, routes tools and agents, checkpoints progress, and runs validation
gates at declared boundaries. Pipeline tools may be internal workflow surfaces
rather than public MCP endpoints; the workflow configuration and executor are
the authoritative routing layer.

```bash
ed4all run textbook-to-course \
  --corpus <CORPUS_PATH> \
  --course-name <COURSE_NAME>
```

Training is a distinct, optional operation with additional licensing, hardware,
and evaluation requirements. See the
[full-run playbook](docs/operations/full-run-playbook.md) and
[pipeline invocation guide](docs/operations/pipeline-invocation.md).

## Extension points

Ed4All grows through explicit registries and contracts rather than implicit
directory discovery:

| Extension | Primary contract |
|---|---|
| Workflow or phase | `config/workflows.yaml` plus its tool route and validation gates |
| Pipeline agent | `config/agents.yaml`, an agent specification or documented in-code implementation, and phase wiring |
| Validation rule | A validator under `lib/validators/`, workflow gate wiring, tests, and gate documentation |
| Schema or taxonomy | The appropriate `schemas/` family, compatibility tests, and ontology documentation |
| LLM-backed feature | Decision capture, regression coverage, provider configuration, and licensing documentation where applicable |
| Operator or user surface | A thin CLI, GUI, or MCP adapter over shared package logic |

Repository placement rules apply at every directory level. See the
[repository organization schema](docs/architecture/repo-organization.md) before
adding a new package or top-level surface.

## Where to go next

| Goal | Start here |
|---|---|
| Understand the exact workflow graph | [Pipeline flow](docs/architecture/pipeline-flow.md) |
| Understand validation and failure semantics | [Validation architecture](docs/architecture/validation-architecture.md) |
| Query course content | [LibV2 overview](LibV2/README.md) |
| Run the full pipeline | [Full-run playbook](docs/operations/full-run-playbook.md) |
| Operate the web interface | [GUI guide](gui/README.md) |
| Contribute safely | [Agent instructions](AGENTS.md) |

This overview explains how the system fits together. The linked subsystem guides,
schemas, workflow configuration, and validation documentation own the details.
