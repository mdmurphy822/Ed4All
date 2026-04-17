# mini_course_summaries — per-chunk summary + retrieval benchmark fixture

Six synthesized chunks across two terminal outcomes. Used by
`Trainforge/tests/test_summary_factory.py` and
`Trainforge/tests/test_retrieval_benchmark.py` to exercise:

- Deterministic extractive summary generation
- 40 ≤ len(summary) ≤ 400 bound
- LO-tag-bearing sentence selection
- `schema_version` stamping on chunks
- BM25 recall@k computation over the `text`, `summary`, and
  `retrieval_text` variants (see `ADR-001` Contract 1 / v4 chunk schema)

CI assertions running against this fixture:

- Every chunk in `chunks.jsonl` has a `summary` field with length ∈ [40, 400]
- Every chunk has `schema_version == "v4"`
- `build_question_set` yields one question per LO in `course.json`
- `run_benchmark` returns `variants["text"]["recall@5"]` and
  `variants["summary"]["recall@5"]` both ∈ [0.0, 1.0]

This fixture is intentionally small (six chunks) so tests run fast. It
is NOT a replacement for end-to-end regeneration against a real IMSCC —
see `Trainforge/output/wcag_201/` for the full-course regression surface.
