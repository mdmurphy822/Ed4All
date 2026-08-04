# LibV2 reference retrieval

LibV2 provides Ed4All's production retrieval contract: lexical BM25, dense
semantic search, rank-domain reciprocal rank fusion (RRF), metadata-aware
filtering, refusal thresholds, and citation validation. See the
[retrieval and serving architecture](../architecture/retrieval-and-serving.md)
for the complete request path.

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
| Dense semantic retrieval | Local vector-index providers | Ships |
| Multi-query decomposition + rank-domain RRF | `multi_retriever.py` | Ships |
| Grounded answers | refusal + citation validation | Ships |
| Hand-curated gold queries + recall@k eval | `libv2 retrieval-eval` | Ships |

## CLI

### Basic retrieval

```bash
libv2 retrieve "color contrast body text" \
    --course "$COURSE_SLUG" \
    --limit 5
```

### With rationale

```bash
libv2 retrieve "color contrast body text" \
    --course "$COURSE_SLUG" \
    --limit 3 --include-rationale
```

Output adds a per-result line:
```text
bm25=7.008 ngram=0.049 boost=+0.023
concept-tags: color-contrast
```

### Metadata filters (v4)

```bash
libv2 retrieve "modal dialogs" \
    --course "$COURSE_SLUG" \
    --week 10 \
    --teaching-role transfer \
    --content-type example
```

### Scoring controls

All of these are independent:

```text
--no-metadata-scoring          # pure BM25
--no-concept-graph-boost       # keep LO + (optional) prereq, drop concept overlap
--no-lo-boost                  # keep concept + prereq, drop LO match
--prefer-self-contained        # enable prereq-coverage boost (off by default, niche)
--lo-filter co-03 --lo-filter co-05   # always-boost chunks tagged with these LOs
```

### JSON output

`--output json` returns the full result list including the rationale payload when enabled.

### Evaluation

```bash
libv2 retrieval-eval --course "$COURSE_SLUG"
```

Reads `LibV2/courses/<course-slug>/retrieval/gold_queries.jsonl`, writes
`evaluation_results.json` alongside, and prints aggregate MRR and recall@1/5/10.

## Python API

```python
from pathlib import Path
from LibV2.tools.libv2.retriever import retrieve_chunks

results = retrieve_chunks(
    repo_root=Path("."),
    query="color contrast body text",
    course_slug="<course-slug>",
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
  "applied_filters": {"course_slug": "<course-slug>"},
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

## When to extend the retrieval layer

Build your own if any of these are true:

- **You need a domain-specific reranker.** Add it after fused candidate
  generation and benchmark the latency/quality tradeoff against the existing
  hybrid baseline.
- **You need custom ranking signals.** User profile, recency, author authority, per-tenant boosts, paid-content priority — all domain-specific, all yours.
- **You need an online API.** HTTP, auth, rate-limiting, sharding, multi-tenancy — all outside LibV2's scope. Embed `retrieve_chunks()` in your server.
- **You need a different evaluation program.** Keep the shipped recall, MRR,
  ablation, and grounded-answer evidence as a baseline, then add domain-specific
  measures without weakening the existing gates.

The rationale payload explains what the baseline found, while the gold-query
contract and v4 chunk metadata provide stable extension points.

## Adding gold queries to your own corpus

1. Build an IMSCC through Courseforge → Trainforge → LibV2, or import an existing package.
2. Create `LibV2/courses/<course-slug>/retrieval/gold_queries.jsonl`. One JSON record per line; `{id, query, relevant_chunk_ids, kind, notes}`.
3. Hand-read each chunk you label — confirm the text actually answers the query. LO-derived shortcuts inflate recall@k against LO-tagging quality, not retrieval quality.
4. `libv2 retrieval-eval --course "$COURSE_SLUG"` produces `evaluation_results.json`.
5. Track the numbers alongside your pipeline-version bumps. If recall@5 drops after a pipeline change, open the per-query entries and diff the rationales.

## Pre-existing artifacts

- `LibV2/tools/libv2/retriever.py` — BM25 + metadata filters + rationale.
- `LibV2/tools/libv2/retrieval_scoring.py` — three metadata-aware boost functions.
- `LibV2/tools/libv2/evaluation/harness.py` — `evaluate_retrieval()` + the pre-existing `RetrievalEvaluator`.
- `LibV2/tools/libv2/cli.py` — `retrieve` and `retrieval-eval` subcommands.
- `LibV2/tools/libv2/tests/test_eval_harness_retrieval.py` — a three-chunk synthetic fixture (see `_write_fixture`) shows the expected `gold_queries.jsonl` shape end-to-end. Users curate their own per-course queries locally; no course-specific query file ships in this repo.
