# Courseforge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WCAG 2.2 AA](https://img.shields.io/badge/WCAG-2.2%20AA-green.svg)](https://www.w3.org/WAI/WCAG22/quickref/)

**Generate LMS-ready course packages from learning objectives or accessible textbook content.**

Courseforge turns a set of objectives (and optional DART-processed textbook HTML) into a complete, accessible, IMSCC-packaged online course. Output includes weekly modules with content pages, activities, self-checks, summaries, and discussions — all WCAG 2.2 AA compliant and ready to import into Brightspace, Canvas, Blackboard, or Moodle. Every page carries rich machine-readable metadata (Bloom's-aligned learning objectives, content types, key terms, misconceptions) so downstream tools can ground assessments and retrieval in cited source material. Courseforge can also ingest an existing IMSCC package from any supported LMS and remediate it to 100% WCAG 2.2 AA compliance.

## Quick example

```bash
# From the repo root, as part of the full Ed4All pipeline:
ed4all run textbook-to-course --corpus my_textbook.pdf --course-name MY_COURSE_101
```

Finished packages land under `Courseforge/exports/YYYYMMDD_HHMMSS_coursename/`. For intake/remediation of an existing package, drop the IMSCC into `Courseforge/inputs/existing-packages/` and run the `intake_remediation` workflow.

## LMS compatibility

Brightspace / D2L, Canvas, Blackboard, Moodle, Sakai, and any LMS that supports IMSCC 1.1 or later.

## More

See [`Courseforge/CLAUDE.md`](CLAUDE.md) for the full agent pipeline, metadata contract (`data-cf-*` + JSON-LD), template components, and quality standards.

## License

MIT
