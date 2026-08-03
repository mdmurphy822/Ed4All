# Courseforge

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)

**Turn learning materials into structured, portable digital courses.**

Courseforge is Ed4All's course-authoring and packaging engine. It combines
learning objectives with accessible textbook HTML, maps source material into a
teachable sequence, builds modular course pages and assessments, and packages
the result as an IMS Common Cartridge for review and LMS import.

```text
OBJECTIVES + ACCESSIBLE SOURCE HTML
                 │
                 ▼
        OUTLINE AND COURSE PLAN
                 │
                 ▼
     MODULAR PAGES + ASSESSMENTS
                 │
                 ▼
       VALIDATION + IMS CC 1.3
```

## What Courseforge delivers

- A source-grounded course outline with canonical learning objectives.
- Modular HTML lessons, activities, discussions, and self-checks.
- Assessment resources and the metadata downstream Ed4All stages use for
  retrieval and training-data preparation.
- An IMS CC 1.3 package designed for standards-based LMS import.
- Validation reports that make accessibility, structure, and package issues
  visible before release.

Courseforge's validators are quality gates, not a blanket guarantee. Review
their reports and test the finished cartridge in the target LMS before
publishing it to learners.

## Run it

From the Ed4All repository root:

```bash
ed4all run textbook-to-course \
  --corpus <CORPUS_PATH> \
  --course-name <course-name>
```

The pipeline converts source material through SemantiK when needed, stages the
accessible HTML for Courseforge, authors the course, runs the configured
validation gates, and writes the finished project beneath
`Courseforge/exports/`.

For an existing cartridge, use the intake and remediation workflow described
in the [workflow reference](docs/workflow-reference.md).

## Explore the system

- [Getting started](docs/getting-started.md) — prerequisites and first-run
  guidance.
- [Workflow reference](docs/workflow-reference.md) — creation, intake, and
  remediation phases.
- [Troubleshooting](docs/troubleshooting.md) — common packaging and validation
  failures.
- [Learning-objective contract](docs/per-week-learning-objectives.md) — how
  page objectives stay aligned with course outcomes.
- [Template-chrome contract](docs/template-chrome-roles.md) — how repeated page
  furniture stays out of retrieval and training text.
- [Local schema index](schemas/README.md) — Courseforge-specific rendering and
  IMSCC contracts.

Courseforge is one stage of Ed4All. See the [project overview](../README.md) for
the complete accessible-content, course, retrieval, and training pipeline.

## License

Apache License 2.0. See [LICENSE](../LICENSE).
