# Trainforge

**Turn a packaged course into a pedagogically tagged knowledge graph and aligned assessments.**

Trainforge consumes IMSCC course packages (from Courseforge or any supported LMS), chunks the content into pedagogical units, and enriches each chunk with Bloom's taxonomy levels, content types, key terms with definitions, misconceptions, and references back to the source PDF region. It builds a typed concept graph over the corpus — three taxonomic relations (is-a, prerequisite, related-to) plus five pedagogical ones (assesses, exemplifies, misconception-of, derived-from-objective, defined-by) — and generates assessment questions grounded in the retrieved content with full decision capture for downstream training.

## Quick example

```bash
# As part of the full pipeline:
ed4all run textbook-to-course --corpus my_textbook.pdf --course-name MY_COURSE_101

# Or standalone RAG training on an existing IMSCC:
ed4all run rag_training --corpus path/to/course.imscc --course-name MY_COURSE_101
```

Output lands under `Trainforge/output/` (chunks + concept graph) and `training-captures/trainforge/<COURSE_CODE>/` (decision JSONL).

## More

See [`Trainforge/CLAUDE.md`](CLAUDE.md) for the chunk shape, metadata extraction priority chain, Bloom's targeting rubric, concept-graph edge taxonomy, and decision-capture contract.

## License

MIT
