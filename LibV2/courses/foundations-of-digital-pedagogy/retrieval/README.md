# Gold-standard retrieval queries — DIGPED_101

Fifteen hand-curated queries against `corpus/chunks.jsonl` (86 chunks, 12 weeks).

Same contract as WCAG_201's `retrieval/` README: every `relevant_chunk_ids` entry is a chunk whose text the curator read; LO-derived labeling was explicitly avoided.

## Query mix (15 total)

| Category | Example |
|---|---|
| Single-concept | "what is behaviorism and how does it view learning" |
| Core-theory (Cognitive Load, CTML) | "cognitive load theory working memory limits instructional design" |
| Multi-concept | "formative assessment vs summative assessment purpose" |
| Process / model (ADDIE, UbD, CoI) | "ADDIE analysis phase needs learner task context analysis" |
| Bridging (appears in adjacent domain too) | "POUR principles of web accessibility in course design" |
| Procedural / how-to | "how do I design a synchronous online teaching session" |
| Synonym recall | "scaffolding Vygotsky zone of proximal development" |

## Running the evaluation

```
libv2 retrieval-eval --course foundations-of-digital-pedagogy
```

Report lands at `retrieval/evaluation_results.json`.

## Cross-course observation

Query `digped_q015` ("POUR principles ...") matches `digped_101_chunk_00074` in this course; the same query against `best-practices-in-digital-web-design-for-accessibi` matches WCAG_201's own POUR chunks. This is by design — it demonstrates that LibV2's `cross_package_concepts.json` index (Worker G) can surface shared concepts across courses, and the reference retriever works the same way inside each one.
