# mini_course_clean — baseline fixture

Three lessons across two modules. JSON-LD complete, no footer contamination,
balanced HTML, every outcome ref resolves to `course_objectives.json`.

A strict-mode run against this fixture must:
- produce `integrity.broken_refs == []`
- produce `integrity.html_balance_violations == []`
- produce `integrity.follows_chunk_boundary_violations == []`
- produce `integrity.factual_inconsistency_flags == []`
- produce `metrics.footer_contamination_rate == 0`

If any of those fail, the pipeline self-trust layer is wrong.
