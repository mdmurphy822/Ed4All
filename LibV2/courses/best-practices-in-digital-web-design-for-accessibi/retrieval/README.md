# Gold-standard retrieval queries — WCAG_201

Twenty hand-curated queries against `corpus/chunks.jsonl` to sanity-check LibV2's reference retriever. Not a benchmark; a documentation artifact with teeth — regression-detectable recall@k + MRR.

## Curation method

Every `relevant_chunk_ids` entry is a chunk whose text the curator read end-to-end. LO-derived labeling (query = LO statement, relevant = chunks carrying that LO ref) was rejected because it conflates retrieval quality with LO-tagging quality. Auto-expansion was explicitly avoided.

## Query mix (20 total)

| Category | Count | Example |
|---|---|---|
| Single-concept semantic | 5 | "how do I set accessible color contrast for body text" |
| Multi-concept | 3 | "ARIA live region announcing form validation errors" |
| Structured SC references | 3 | "WCAG SC 1.4.3 contrast minimum requirements" |
| Synonym recall | 3 | "screen reader testing accessibility assistive technology" |
| Procedural / how-to | 3 | "how do I test my site with NVDA screen reader" |
| Bloom-level targeted (Apply/Create) | 2 | "apply POUR principles to an ecommerce checkout page" |
| Misconception-driven | 1 | "do all images need alt text or do some use empty alt" |

## Record shape

```json
{
  "id": "wcag_qNNN",
  "query": "human-readable question a consumer might ask",
  "relevant_chunk_ids": ["wcag_201_chunk_00042", ...],
  "kind": "hand-curated",
  "notes": "rationale — why each chunk is relevant, any caveats"
}
```

`notes` is mandatory for audit — reviewers should be able to spot-check any entry by reading the chunks and the note together.

## How to add more queries

1. Pick a query pattern not well-covered by the existing set (check the table above).
2. Find candidate chunks: `libv2 retrieve "your query" -c best-practices-in-digital-web-design-for-accessibi --limit 10`.
3. **Read every chunk you plan to label** — confirm its text actually answers the query. Don't trust concept_tags alone.
4. Append a `hand-curated` record. Set `kind: "lo-derived"` only if the only label you can justify is "the chunk references the LO", not "the chunk answers the query".
5. Re-run `libv2 retrieval-eval --course best-practices-in-digital-web-design-for-accessibi` to confirm the new query's rank numbers are sensible.

## Running the evaluation

```
libv2 retrieval-eval --course best-practices-in-digital-web-design-for-accessibi
```

Writes `evaluation_results.json` next to this file. Report aggregate MRR + recall@1/5/10 + per-query rank-of-first-relevant + top-result rationale.
