# Full-run playbook

Use this runbook to turn private source books and learning materials into an
accessible course, a packaged course archive, retrieval artifacts, and—only
when explicitly requested—training pairs and a LoRA adapter.

This is the operator's end-to-end path. Detailed installation, provider,
licensing, flags, and per-phase controls live in their owning documents:

- [Installation and local dependencies](installation.md)
- [Pipeline invocation](pipeline-invocation.md)
- [License-clean pipeline run](license-clean-run.md)
- [Licensing and terms posture](../LICENSING.md)
- [Behavior flags](behavior-flags.md)
- [Validation gates](../validation/gates.md)
- [Backup and restore](backup-restore.md)

## 1. What the run produces

```mermaid
flowchart LR
    classDef source fill:#E8F1FF,stroke:#2457A6,color:#102A43,stroke-width:2px
    classDef semantik fill:#E5F7ED,stroke:#247A45,color:#123622,stroke-width:2px
    classDef course fill:#FFF4CC,stroke:#946200,color:#3D2A00,stroke-width:2px
    classDef training fill:#F2E8FF,stroke:#6941A5,color:#2E174D,stroke-width:2px
    classDef archive fill:#FFE8F1,stroke:#9B285E,color:#48142D,stroke-width:2px

    S["Private books and learning materials"]:::source
    O["SemantiK<br/>GLM-OCR SDK conversion"]:::semantik
    E["Structure and accessibility enrichment"]:::semantik
    H["Super heading judge"]:::semantik
    C["Courseforge<br/>objectives, course content, assessments, IMSCC"]:::course
    T["Trainforge<br/>chunks, graphs, retrieval, SFT and DPO pairs"]:::training
    L["LibV2<br/>private archive, index, adapters, model cards"]:::archive

    S --> O --> E --> H --> C --> T --> L
```

Color is supplementary; every node has a text label. Training is an optional,
operator-controlled extension. A complete course and LibV2 archive do not
require fitting an adapter.

## 2. Privacy and prerequisites

Source material, converted HTML, course identifiers, generated course content,
training pairs, adapters, run state, and logs are private. Keep all of them in
ignored runtime or LibV2 locations. Never add a corpus path, course name, run
identifier, endpoint, hostname, or generated artifact to tracked code, docs,
comments, fixtures, or examples.

Before starting:

1. Confirm you have permission to process the source and use any generated
   output for the intended purpose.
2. Read [the licensing policy](../LICENSING.md) before generating SFT/DPO data
   or fitting an adapter.
3. Install only the capabilities needed for this run by following
   [installation.md](installation.md). Dependencies and model weights are
   installed or staged locally; they are not stored in this repository.
4. Put source files outside the repository and choose an ignored data root.
5. Configure model providers and endpoints in an ignored operator environment
   file. Follow [license-clean-run.md](license-clean-run.md); do not rely on
   inherited provider defaults.

Load private values from an ignored operator environment. The placeholders
below describe required values; never replace them in tracked documentation:

```bash
export ED4ALL_SOURCE="<PRIVATE_SOURCE_PATH>"
export ED4ALL_COURSE_NAME="<PRIVATE_COURSE_NAME>"
export ED4ALL_DATA_ROOT="<PRIVATE_DATA_ROOT>"
export ED4ALL_HOME="$ED4ALL_DATA_ROOT"
export ED4ALL_LIBV2_ROOT="$ED4ALL_DATA_ROOT/LibV2"
```

Do not commit the environment file or these values.

## 3. Preflight

Run preflight from the repository environment before every campaign:

```bash
ed4all --version
ed4all doctor
ed4all doctor --run textbook_to_course --mode local
```

If your configuration uses reachable model services, add the bounded endpoint
probe documented by `ed4all doctor --help`:

```bash
ed4all doctor --run textbook_to_course --mode local --ping
```

Resolve failures before dispatch. In particular, verify:

- the source path is readable and remains outside Git;
- required system packages, Python extras, browser assets, schemas, and local
  model files are present;
- configured endpoints advertise the intended models and return coherent text;
- enough storage exists for converted HTML, packages, indexes, checkpoints, and
  optional adapter artifacts;
- the selected synthesis provider and base model are license-compatible with
  the intended output.

## 4. Plan, inspect, then run

### 4.1 Dry run

A dry run resolves workflow configuration and phase ordering without executing
the pipeline:

```bash
ed4all run textbook-to-course \
  --corpus "$ED4ALL_SOURCE" \
  --course-name "$ED4ALL_COURSE_NAME" \
  --skip-training \
  --dry-run
```

Use the same environment for the dry run and real run. Conditional workflow
settings can change the phase graph. Review the resolved providers, models,
output roots, optional phases, and stopping point before continuing.

### 4.2 Inspection pilot

For a new source, stop after structural extraction before committing to course
authoring:

```bash
ed4all run textbook-to-course \
  --corpus "$ED4ALL_SOURCE" \
  --course-name "$ED4ALL_COURSE_NAME" \
  --skip-training \
  --stop-after objective_extraction
```

Inspect the accessible HTML, heading hierarchy, source coverage, chapter and
section structure, reading order, and extraction diagnostics. Resume only when
the source structure is correct. A poor conversion should be fixed in
SemantiK, not compensated for downstream.

To inspect the planned pedagogy before prose generation, use the two-pass
outline stop described in [pipeline-invocation.md](pipeline-invocation.md).

### 4.3 Full course build

The recommended first full run builds and archives the course without training:

```bash
ed4all run textbook-to-course \
  --corpus "$ED4ALL_SOURCE" \
  --course-name "$ED4ALL_COURSE_NAME" \
  --skip-training
```

This path covers conversion, enrichment, heading judgment, course design and
authoring, assessment and package generation, Trainforge extraction, LibV2
archival, and retrieval indexing according to the live workflow configuration.

Training-pair synthesis and adapter fitting require separate review. Do not add
`--with-training` merely to make the run “complete.”

## 5. Stage-by-stage review

Use phase outputs as approval boundaries:

| Boundary | Review before continuing |
|---|---|
| **SemantiK conversion** | OCR fidelity, reading order, tables, figures, alt text, language, and WCAG-oriented HTML structure |
| **Enrichment and heading judge** | Heading levels, document outline, semantic roles, provenance, and correction diagnostics |
| **Courseforge planning** | Course identity, duration, objectives, module mapping, concept coverage, and pedagogy plan |
| **Courseforge authoring** | Source grounding, accessible HTML, activities, assessment alignment, and package validation |
| **Trainforge** | IMSCC chunks, objective links, concept and pedagogy graphs, assessment artifacts, and retrieval behavior |
| **LibV2** | Manifest integrity, private archive layout, quality reports, vector index, and provenance |
| **Optional training** | Pair quality, licensing, holdouts, training telemetry, evaluation, and model card |

The exact phase names, selective rerun controls, reuse flags, and output paths
are maintained in [pipeline-invocation.md](pipeline-invocation.md).

## 6. Validation gates

Phase gates run after their owning phase. Stop at the first failed gate and fix
the artifact or environment that caused it. Do not reduce a threshold, downgrade
severity, suppress an issue, or enable a fallback simply to obtain a green run.

Interpret results carefully:

- a passed gate validated its configured contract;
- a warning may permit continuation but still requires review;
- a structured skip means the validator did not receive its required inputs;
- a waiver is an explicit audited exception, not a pass;
- a critical failure blocks the phase;
- a validator exception follows the gate's error policy and may fail closed.

Use [validation gates](../validation/gates.md) for the live phase mapping and
[validation architecture](../architecture/validation-architecture.md) for exact
failure semantics.

## 7. Training-pair pilot

After the course is archived, review licensing and run a bounded synthesis pilot
before generating a full pair set. The pilot writes private output into the
course archive and does not train a model:

```bash
: "${ED4ALL_COURSE_SLUG:?load the private slug from ignored operator configuration}"

python3 -m Trainforge.scripts.ops.pilot_synthesis \
  --corpus "$ED4ALL_LIBV2_ROOT/courses/$ED4ALL_COURSE_SLUG" \
  --course-code "$ED4ALL_COURSE_SLUG" \
  --provider local \
  --max-pairs 50
```

Review the generated pilot report, source grounding, property coverage,
instruction diversity, preference quality, and every gate result. If a gate
fails, improve the source, routing, or approved model; do not weaken the gate.
The provider rules and environment variables are owned by
[license-clean-run.md](license-clean-run.md).

## 8. Stop and resume safely

Request a checkpointed stop at the next supported unit boundary:

```bash
ed4all stop "$WORKFLOW_ID"
```

A graceful stop exits with code `3`. Resume the same workflow with:

```bash
ed4all run textbook-to-course --resume "$WORKFLOW_ID"
```

Do not add `--force` after a graceful stop; it clears resume sidecars and
discards the checkpoint granularity you stopped to preserve. A global stop and
its explicit clear operation are documented in
[pipeline-invocation.md](pipeline-invocation.md#7-graceful-stop-ed4all-stop).

## 9. Outputs to inspect

Paths vary with `ED4ALL_HOME`, export settings, and workflow configuration. Use
the paths printed by the run rather than assuming a machine-specific layout.
The important artifact families are:

- accessible SemantiK HTML and conversion quality sidecars;
- Courseforge objectives, source mapping, outlines, authored pages, validation
  reports, assessments, and IMSCC package;
- Trainforge chunk sets, assessments, concept and pedagogy graphs, and optional
  SFT/DPO pair files;
- LibV2 manifest, private course archive, quality reports, vector index, and
  optional model directory;
- run checkpoints, decision captures, usage records, and final summary under
  the configured runtime state root.

Back up private artifacts according to
[backup-restore.md](backup-restore.md), and sanitize any support bundle before
sharing it.

## 10. Troubleshooting

| Symptom | First action |
|---|---|
| Conversion output is incomplete or structurally wrong | Stop after SemantiK, inspect conversion diagnostics, and correct the source/conversion path before resuming |
| Heading structure is implausible | Review heading-judge outputs and enrichment inputs; do not hand-author around a broken document outline |
| Provider is unreachable or incoherent | Run `ed4all doctor --run textbook_to_course --mode local --ping`; verify the configured endpoint and model |
| A gate reports missing inputs | Treat it as a wiring or upstream-artifact problem; inspect the phase checkpoint and gate input routing |
| CUDA or memory failure | Stop cleanly, release competing workloads, verify the intended device profile, then resume from checkpoints |
| Run was interrupted | Use `--resume`; do not start a second run against the same mutable course archive |
| Package fails import validation | Fix Courseforge output or packaging metadata and rerun the owning phase |
| Retrieval is empty or poor | Verify archival completed, inspect chunks and provenance, then verify the vector index and embedding configuration |
| Training-pair pilot fails | Review the pilot report and provider/model quality; keep the full synthesis blocked |

For deeper diagnostics, use the troubleshooting and post-mortem sections in
[pipeline-invocation.md](pipeline-invocation.md) and create a sanitized support
bundle only when necessary.

## 11. Post-training promotion is manual

Adapter training is GPU-bound and must be explicitly requested. For an existing
private LibV2 course, the operator-controlled workflow is:

```bash
ed4all run trainforge_train \
  --course-name "$ED4ALL_COURSE_SLUG" \
  --base-model "$ED4ALL_BASE_MODEL"
```

Follow the current training qualification runbook linked from
[Trainforge](../../Trainforge/README.md). A successful process exit, completed
training loop, or passing automated evaluation does **not** authorize promotion.
The operator must review the evaluation matrix, licensing, holdout integrity,
regressions, model card, and promotion ledger, then explicitly choose promote,
hold, or reject.

## 12. DGX Spark example profile

This optional profile keeps machine-specific choices outside the playbook. Use
operator-configured endpoints and local model IDs; do not copy endpoint values
or model names into tracked files.

```bash
# Source an ignored file containing provider endpoints and model IDs.
source "$ED4ALL_OPERATOR_ENV"

# Use the GPU for explicitly configured embedding and NLI workloads.
export ED4ALL_EMBEDDING_DEVICE=cuda
export ED4ALL_NLI_DEVICE=cuda

ed4all doctor --run textbook_to_course --mode local --ping
```

Tune concurrency, free-memory floors, and service lifecycle using the
[DGX Spark profile](spark-profile.md) and [seat scripts](seat-scripts.md). Keep
all endpoint URLs configurable. Do not infer readiness from GPU size alone;
preflight the exact services and run configuration that will execute.
