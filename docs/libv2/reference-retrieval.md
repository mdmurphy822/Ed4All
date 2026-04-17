# LibV2 reference retrieval

**This is a reference implementation, not a production retrieval system.**
See [ADR-002](../architecture/ADR-002-retrieval-scope.md) for the scope line and why it sits where it does.

## What you get

| Capability | Implementation | Status |
|---|---|---|
| Metadata filtering | `ChunkFilter` (11 fields) | Ships |
| BM25 ranking | Hand-rolled Okapi, k1=1.5, b=0.75 | Ships |
| Character n-gram boosting | Jaccard on trigrams, weight 0.15 | Ships |
| Structured tokenization | `aria-labelledby`, `sc-1.4.3` preserved | Ships |
| `retrieval_text`-aware indexing | v4 summaries used when present | Ships |
| Rationale payload | BM25/ngram/boost breakdown + matched metadata | Opt-in |
| Metadata-aware scoring | concept-graph overlap, LO match, prereq coverage | Opt-in (default on) |
| Multi-query decomposition + RRF | `multi_retriever.py` | Ships |
| Hand-curated gold queries + recall@k eval | `libv2 retrieval-eval` | Ships |
| Dense embeddings, cross-encoder reranker, online API | — | **Out of scope** (build your own) |

## CLI

### Basic retrieval

```
libv2 retrieve "color contrast body text" \
    --course best-practices-in-digital-web-design-for-accessibi \
    --limit 5
```

### With rationale

```
libv2 retrieve "color contrast body text" \
    --course best-practices-in-digital-web-design-for-accessibi \
    --limit 3 --include-rationale
```

Output adds a per-result line:
```
bm25=7.008 ngram=0.049 boost=+0.023
concept-tags: color-contrast
```

### Metadata filters (v4)

```
libv2 retrieve "modal dialogs" \
    --course best-practices-... \
    --week 10 \
    --teaching-role transfer \
    --content-type example
```

### Scoring controls

All of these are independent:

```
--no-metadata-scoring          # pure BM25
--no-concept-graph-boost       # keep LO + (optional) prereq, drop concept overlap
--no-lo-boost                  # keep concept + prereq, drop LO match
--prefer-self-contained        # enable prereq-coverage boost (off by default, niche)
--lo-filter co-03 --lo-filter co-05   # always-boost chunks tagged with these LOs
```

### JSON output

`--output json` returns the full result list including the rationale payload when enabled.

### Evaluation

```
libv2 retrieval-eval --course best-practices-in-digital-web-design-for-accessibi
```

Reads `LibV2/courses/<slug>/retrieval/gold_queries.jsonl`, writes `evaluation_results.json` alongside, prints aggregate MRR + recall@1/5/10.

## Python API

```python
from pathlib import Path
from LibV2.tools.libv2.retriever import retrieve_chunks

results = retrieve_chunks(
    repo_root=Path("."),
    query="color contrast body text",
    course_slug="best-practices-in-digital-web-design-for-accessibi",
    limit=5,
    include_rationale=True,
)

for r in results:
    print(r.chunk_id, r.score)
    if r.rationale:
        print(" ", r.rationale["matched_concept_tags"])
        print(" ", r.rationale["boost_contributions"])
```

`RetrievalResult` fields: `chunk_id`, `text`, `score`, `course_slug`, `domain`, `chunk_type`, `difficulty`, `concept_tags`, `source`, `tokens_estimate`, `learning_outcome_refs`, `bloom_level`, and (opt-in) `rationale`.

### Lower-level index

```python
from LibV2.tools.libv2.retriever import LazyBM25

index = LazyBM25(chunks, use_retrieval_text=True, structured_tokens=True)
for chunk, score in index.search("skip link", limit=10, min_relevance=0.5):
    ...
```

## The rationale payload

When `include_rationale=True`:

```json
{
  "bm25_score": 7.008,
  "ngram_score": 0.049,
  "metadata_boost": 0.023,
  "final_score": 7.17,
  "matched_concept_tags": ["color-contrast"],
  "matched_lo_refs": [],
  "matched_key_terms": [{"term": "contrast ratio", "definition": "..."}],
  "applied_filters": {"course_slug": "best-practices-..."},
  "boost_contributions": {
    "concept_graph_overlap": 0.25,
    "lo_match": 0.0,
    "prereq_coverage": 0.0
  }
}
```

Use cases:
- **Debugging recall failures.** Low `bm25_score` but expected → your query missed the chunk's indexed text; a `summary`/`retrieval_text` mismatch is common.
- **Debugging ranking order.** Two chunks with similar BM25; check `metadata_boost` — the one with concept-graph or LO matches will rank higher.
- **Downstream reasoning.** A decision/rule layer reading `rationale.matched_lo_refs` can apply per-LO policy without re-running retrieval. This is the differentiator vs generic RAG that doesn't carry metadata.

## When to build your own retrieval

Build your own if any of these are true:

- **You need dense embeddings** for semantic recall on paraphrased queries. BM25 alone won't get there; fine-tune an embedding model on your chunk set.
- **You need a reranker.** Even a small cross-encoder reranking top-50 candidates measurably improves quality; adding one triples the latency, so it belongs in your retrieval layer, not ours.
- **You need custom ranking signals.** User profile, recency, author authority, per-tenant boosts, paid-content priority — all domain-specific, all yours.
- **You need an online API.** HTTP, auth, rate-limiting, sharding, multi-tenancy — all outside LibV2's scope. Embed `retrieve_chunks()` in your server.
- **You need a full eval framework.** Hit@k + MRR is the baseline; ablation sweeps, per-query-type breakdowns, retrieval-vs-generation attribution, all belong in your evaluation tooling.

The reference implementation makes building your own easier, not redundant: the rationale payload tells you what the baseline found, the gold queries are ready-made starting benchmarks, and the v4 chunk metadata (concept tags, LOs, prereqs, content types, summaries) is the contract you build on.

## Adding gold queries to your own corpus

1. Build an IMSCC through Courseforge → Trainforge → LibV2, or import an existing package.
2. Create `LibV2/courses/<your-slug>/retrieval/gold_queries.jsonl`. One JSON record per line; `{id, query, relevant_chunk_ids, kind, notes}`.
3. Hand-read each chunk you label — confirm the text actually answers the query. LO-derived shortcuts inflate recall@k against LO-tagging quality, not retrieval quality.
4. `libv2 retrieval-eval --course <your-slug>` produces `evaluation_results.json`.
5. Track the numbers alongside your pipeline-version bumps. If recall@5 drops after a pipeline change, open the per-query entries and diff the rationales.

## Pre-existing artifacts

- `LibV2/courses/best-practices-in-digital-web-design-for-accessibi/retrieval/gold_queries.jsonl` — 20 hand-curated WCAG queries.
- `LibV2/courses/foundations-of-digital-pedagogy/retrieval/gold_queries.jsonl` — 15 hand-curated DIGPED queries.
- `LibV2/tools/libv2/retriever.py` — BM25 + metadata filters + rationale.
- `LibV2/tools/libv2/retrieval_scoring.py` — three metadata-aware boost functions.
- `LibV2/tools/libv2/eval_harness.py` — `evaluate_retrieval()` + the pre-existing `RetrievalEvaluator`.
- `LibV2/tools/libv2/cli.py` — `retrieve` and `retrieval-eval` subcommands.
