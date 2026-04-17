# mini_course — synthetic defect fixture

Used by `Trainforge/tests/test_generator_defects.py`.

Three lessons across two modules. Each source HTML embeds a specific defect pattern so every regression test can exercise its target behaviour against a small, deterministic corpus without depending on a real IMSCC package.

| File | Module | Lesson | Defects exercised |
|---|---|---|---|
| `source_html/week_01_overview.html` | m1 | w01 | footer contamination; JSON-LD with `w01-co-02` ref; Bloom-verb-heavy text without explicit JSON-LD bloomLevel |
| `source_html/week_01_selfcheck.html` | m1 | w01q | unbalanced `<div>` tag; atomic `<div data-cf-role="activity-card">` that must not be split |
| `source_html/week_02_concepts.html` | m2 | w02 | SC name variants (Contrast Minimum / Contrast Minimum, Level AA); factual claim "87 success criteria"; arithmetic contradiction 29+29+17+4 |
| `source_html/week_03_review.html` | m2 | w03 | broken outcome ref `w99-co-99`; dual-ID reference `w02-to-01` |
| `course_objectives.json` | — | — | flat `co-*` + week-scoped `w0X-co-*` outcome hierarchy for dual-emission tests |

No PII, no copyrighted material. The `© 2026 ACME_FIX_ME` footer is intentional bait for the boilerplate detector.
