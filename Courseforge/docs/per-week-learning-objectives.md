# Per-Week `learningObjectives` Specificity

*Short note for anyone regenerating a Courseforge course or reviewing the
`learningObjectives` block in a Courseforge HTML page.*

## TL;DR

Every generated HTML page has a `<script type="application/ld+json">` block
with a `learningObjectives` array. Each entry must reference a canonical
objective ID (a `CO-##` or `TO-##` declared in the course's exam-objectives
JSON, e.g. `inputs/exam-objectives/SAMPLE_101_objectives.json`) and the list
must be the subset of canonical objectives that apply to that page's week.

If a page lists objectives that belong to a different week (or invents
week-local IDs like `W07-CO-01`), Trainforge's chunker will either fail to
resolve them against `course.json` (breaking `outcome_reverse_coverage`) or
collapse them onto the same four IDs (breaking training-pair diversity,
recall@k, and prerequisite-edge inference).

## The selection rule

For a page generated under `week_<N>/`:

1. Start with the 4 terminal objectives (`TO-01..TO-04`). These apply across
   the whole course.
2. Add the chapter objectives whose `chapter` field names a week range that
   includes week N. The range is parsed with the regex
   `r"[Ww]eek\s+(\d+)(?:\s*-\s*(\d+))?"` — the same regex used in
   `Trainforge.process_course.load_objectives`, so emit and resolve stay in
   lockstep.

Concretely for a sample 12-week course (`SAMPLE_101`):

| Week(s) | Chapter CO IDs emitted on each page |
|---------|-------------------------------------|
| 1 – 2   | `CO-01, CO-02, CO-03, CO-04`         |
| 3 – 4   | `CO-05, CO-06, CO-07, CO-08`         |
| 5 – 6   | `CO-09, CO-10, CO-11, CO-12`         |
| 7 – 8   | `CO-13, CO-14, CO-15, CO-16`         |
| 9 – 10  | `CO-17, CO-18, CO-19, CO-20`         |
| 11 – 12 | `CO-21, CO-22, CO-23, CO-24`         |

Every week's pages additionally carry `TO-01..TO-04`.

Overview / non-week pages that resolve to "week 0" emit only the
terminal objectives.

## How generation consumes this

`scripts/generate_course.py` accepts an optional `--objectives` flag:

```
python Courseforge/scripts/generate_course.py \
    inputs/course-data/SAMPLE_101_course_data.json \
    exports/SAMPLE_101_COURSE/03_content_development \
    --objectives inputs/exam-objectives/SAMPLE_101_objectives.json
```

When the flag is present, each week's `objectives` list in the course data
JSON is **overridden** with the canonical subset for that week before any
page is rendered. The emitted `<li data-cf-objective-id="...">` markup and
the JSON-LD `learningObjectives` array both carry the canonical IDs.

Without the flag the old behaviour (pass through whatever IDs the course
data JSON declares) is preserved so older workflows that already use
canonical IDs keep working. New course generation should always pass
`--objectives`.

## How to check a generated course

`scripts/validate_page_objectives.py` walks a tree of generated HTML and
asserts every JSON-LD `learningObjectives` block references only IDs that
are declared for that page's week:

```
python Courseforge/scripts/validate_page_objectives.py \
    --objectives inputs/exam-objectives/SAMPLE_101_objectives.json \
    --pages exports/SAMPLE_101_COURSE/03_content_development
```

Exit code `0` on success, `1` if any page leaks another week's IDs. The
same check runs as unit tests in
`scripts/tests/test_generate_course_lo_specificity.py`:

- A fabricated "correct" page (canonical IDs from the right week) passes.
- A fabricated "buggy" page (all canonical IDs emitted on every week) is
  rejected with a message naming the extraneous IDs.
- `resolve_week_objectives(week=3, ...)` returns the expected TO+CO set.
- `resolve_week_objectives(week=0, ...)` returns the terminal objectives
  only.

## Why this mattered

Before the fix, a typical `<COURSE>_course_data.json` declared week-local
IDs like `W01-CO-01..W12-CO-04` on each week's `objectives`. Each week
independently numbered its COs `01..04`. Trainforge normalizes by
stripping the `W0N-` prefix (see
`Trainforge.process_course._extract_objective_refs`), so every week's
chunks reduced to `co-01..co-04`. On a twelve-week course whose
`course.json` declares `CO-01..CO-24` plus `TO-01..TO-04` (28 outcomes
total), that meant:

- `outcome_reverse_coverage = 0.143` (4 of 28 outcomes had any chunk).
- 24 outcomes uncovered.
- Training-pair synthesis, retrieval benchmarks, and prerequisite-edge
  inference all capped by this convergent ceiling.

After the fix, regenerating the course with `--objectives` and re-running
Trainforge yields `outcome_reverse_coverage = 1.0`, zero uncovered
outcomes, and 28 distinct LO refs distributed across the corpus.
