# Build your first LMS-ready course

Courseforge turns accessible source material into a structured digital course:
an outcome-aligned outline, modular learning pages, assessments, validation
reports, and an IMS Common Cartridge for LMS review and import.

```text
SOURCE MATERIAL
      │
      ▼
COURSE PLAN + OBJECTIVES
      │
      ▼
LESSONS + ACTIVITIES + ASSESSMENTS
      │
      ▼
VALIDATION REPORTS + IMSCC PACKAGE
```

This guide uses the supported `ed4all` CLI. You do not need to invoke
individual agents or create export directories by hand.

## Before you begin

You need:

- an Ed4All development environment with the project dependencies installed;
- a PDF, a directory of source files, or SemantiK-produced accessible HTML;
- a short, non-sensitive course name;
- access to an IMSCC-compatible LMS if you want to test the final import.

Run commands from the Ed4All repository root. Use placeholders in scripts and
documentation; do not commit source filenames, course names, run IDs, or
generated course data.

## Run the complete pipeline

```bash
ed4all run textbook-to-course \
  --corpus <CORPUS_PATH> \
  --course-name <COURSE_NAME>
```

For a PDF, the workflow sends the source through SemantiK first. Accessible
HTML then flows into Courseforge for planning, authoring, validation, and
packaging. The final project is written beneath `Courseforge/exports/`; course
artifacts in that directory are working data and are not for source control.

The command runs in local mode by default. Use `--mode api` only when the
required provider credentials and licensing posture are configured. Provider
and model choices can affect whether generated material is suitable for
downstream training; read [Licensing](../../../docs/LICENSING.md) before
changing synthesis providers.

## Review the result

The export contains the important review seams below. Exact filenames and
optional artifacts vary by workflow configuration.

```text
Courseforge/exports/<PROJECT_ID>/
├── 01_learning_objectives/    # canonical objective set
├── 01_outline/                # source-grounded block outline
├── 02_validation_report/      # inter-tier findings
├── 03_content_development/    # authored course pages
├── 04_rewrite/                # rewritten blocks and post-rewrite report
├── 05_final_package/          # IMSCC deliverable
└── 06_assessments/            # generated assessment resources
```

Before publishing:

1. Read the validation reports and resolve failed gates at their source.
2. Review lessons and assessments for instructional accuracy and tone.
3. Navigate the course with a keyboard and assistive technology appropriate
   to your learners.
4. Import the IMSCC into a test course in the target LMS.
5. Confirm links, navigation, assessment behavior, and learner-visible labels.

Courseforge validates known contracts; it does not replace editorial,
accessibility, or LMS acceptance review.

## Resume or inspect a run

Long workflows can be stopped and resumed at phase boundaries. Use the run ID
reported by the CLI:

```bash
ed4all run textbook-to-course --resume <RUN_ID>
```

For phase-by-phase operation, selective rewrites, stop behavior, and provider
flags, use the canonical [pipeline invocation
runbook](../../../docs/operations/pipeline-invocation.md).

Useful Courseforge slices include:

```bash
# Run deterministic validation without an authoring call.
ed4all run courseforge-validate --course-name <COURSE_NAME>

# Re-run the rewrite tier for selected block types.
ed4all run courseforge-rewrite \
  --course-name <COURSE_NAME> \
  --blocks <BLOCK_TYPES> \
  --force
```

These commands operate on an existing project. The complete
`textbook-to-course` command remains the recommended first run.

## Bring an existing cartridge

Courseforge also supports IMSCC intake and remediation. The workflow inspects
the package, inventories content, routes repair work, reruns validation, and
packages a revised cartridge. Start with the [workflow
reference](../reference/workflow-reference.md) to select the correct intake
path and required inputs.

## Understand the quality gates

Courseforge checks package structure, objective alignment, content quality,
accessibility signals, and IMSCC/QTI contracts at defined workflow seams. A
failed gate means the artifact needs attention. Do not lower thresholds or
downgrade severity to make a run pass.

For the complete gate inventory and issue codes, see:

- [Validation gates](../../../docs/validation/gates.md)
- [Troubleshooting](troubleshooting.md)
- [Learning-objective contract](../reference/per-week-learning-objectives.md)
- [Template-chrome contract](../reference/template-chrome-roles.md)

## Where to go next

- Read the [Courseforge overview](../../README.md) for the product surface.
- Use the [workflow reference](../reference/workflow-reference.md) for phase
  details and remediation flows.
- Browse the [local schema index](../../schemas/README.md) when integrating
  Courseforge artifacts.
- Return to the [Ed4All overview](../../../README.md) to see how Courseforge
  connects accessible conversion, retrieval, training-data preparation, and
  adapter training.
