# ADR-002 — Retrieval scope for LibV2 (reference-implementation line)

## Status

Proposed (Worker J, 2026-04-17).

## Context

Downstream consumers of Ed4All packages (decision engines, orchestration layers, rule-based execution, tutor systems, council workflows) all need some form of retrieval against the chunks LibV2 stores. Before this ADR the line between "what LibV2 ships" and "what downstream consumers build" was informal: LibV2 shipped a BM25 retriever (`LibV2/tools/libv2/retriever.py`) that production callers in Trainforge used directly, but there was no written contract about how rich that retriever should become, where it stopped, or what consumers could assume.

Two failure modes resulted:
1. Scope creep pressure on LibV2 retrieval (reranker? dense embeddings? online query API?). Each pull expands the LibV2 surface area and couples LibV2's lifecycle to anyone who depends on a specific retrieval feature.
2. Ambient confusion about where retrieval quality issues belong — "Ed4All, I tried it, the retrieval was slow" vs. "my retrieval implementation using Ed4All's chunks as input was slow" become indistinguishable without a clear scope line.

Worker J's work closes the gap between what the retriever does today (BM25 + n-gram) and what a *reference implementation* of retrieval should do (rationale payload, metadata-aware scoring, gold-standard evaluation, architectural boundary documentation). This ADR names the line.

## Decision

**Ed4All produces structured, validated knowledge packages. LibV2 stores them and exposes reference retrieval. What you do with retrieved knowledge is your problem.**

LibV2's reference retrieval is intentionally bounded. Anything more sophisticated belongs downstream.

## Rationale

1. **Reference implementations are documentation.** A working `libv2 retrieve` + `libv2 retrieval-eval` that demonstrates the intended query patterns against the package format means no downstream consumer has to reverse-engineer how to use chunks. That is a documentation deliverable with teeth — gold-standard queries, rationale payloads, tests.
2. **Coupling boundaries matter.** A full retrieval system (vector index, reranker, online query API, eval infrastructure with ablation) is a separate product. Merging it into LibV2 would couple two different lifecycles — an Ed4All package-format bump would force every retrieval consumer to rev too, and vice versa.
3. **Quality signals stay honest.** By owning only a reference implementation, LibV2 can state numbers diagnostically (gold-standard MRR, recall@k) without committing to production SLA. Consumers can measure their own retrieval quality against the same gold set and compare.
4. **Metadata-aware scoring is the natural differentiator.** Generic RAG can't weight chunks by concept-graph overlap, LO match, or prereq coverage because generic RAG doesn't have those metadata fields. LibV2's reference retriever does. The three boost functions in `retrieval_scoring.py` demonstrate that metadata lift without closing the door on consumers doing more.

## Rejected alternatives

- **"LibV2 ships a production retrieval API."** Rejected — couples LibV2's lifecycle to every consumer's retrieval SLA. Retrieval engines evolve fast (new embedding models, new rerankers); the chunk schema should not.
- **"LibV2 ships only BM25, no rationale, no metadata boosts."** Rejected — this is what the repo had before Worker J, and the gap it leaves (no diagnostic output for debugging; no explanation of how metadata fields matter) is exactly what forces downstream consumers to reinvent the wheel.
- **"Ship no retrieval at all; consumers write their own."** Rejected — guarantees every consumer's retrieval implementation is slightly different and the project's reputation absorbs their quality issues. The essay framing ("Oh Ed4All, I tried it, the retrieval was slow") anticipates this.

## What's in scope for LibV2 reference retrieval

- **BM25 index over chunks.** Hand-rolled Okapi BM25 with k1=1.5, b=0.75, character-trigram n-gram boosting (`LibV2/tools/libv2/retriever.py::LazyBM25`).
- **Metadata filters.** `ChunkFilter` supports `chunk_type`, `difficulty`, `concept_tags`, `min_tokens`, `max_tokens`, `learning_outcome_refs`, `bloom_level`, `teaching_role`, `content_type_label`, `module_id`, `week_num`. Filter-first, rank-second.
- **Structured tokenization** that preserves hyphenated slugs (`aria-labelledby`, `skip-link`) and WCAG SC references (`sc-1.4.3`, `wcag-2.2`) as single tokens.
- **`retrieval_text`-aware indexing.** When a chunk carries v4's `retrieval_text` (summary + key terms), the index uses it instead of the full chunk body.
- **Rationale payload.** With `include_rationale=True`, every result carries `{bm25_score, ngram_score, metadata_boost, final_score, matched_concept_tags, matched_lo_refs, matched_key_terms, applied_filters, boost_contributions}`.
- **Metadata-aware score boosts.** Three pure functions in `retrieval_scoring.py`: concept-graph overlap, LO match (explicit + implicit), prereq coverage. Multiplicative blend capped at `MAX_TOTAL_BOOST = 0.5`.
- **Gold-standard query sets.** Hand-curated per-course queries at `LibV2/courses/<slug>/retrieval/gold_queries.jsonl` (20 for WCAG_201, 15 for DIGPED_101 at time of writing).
- **Evaluation harness.** `evaluate_retrieval()` computes MRR + recall@1/5/10 + per-query rationale. `libv2 retrieval-eval` CLI.
- **Multi-query decomposition** (pre-existing, `multi_retriever.py`) — kept, documented as advanced API.

## What's explicitly out of scope

- **Dense embeddings.** No embedding model, no vector index, no hybrid (dense+sparse) fusion. A downstream consumer adding these picks the model, handles the cache, owns the upgrade cadence.
- **Cross-encoder rerankers.** The inference cost, model-version churn, and hyperparameter surface (how many candidates to rerank) belong downstream.
- **Full eval infrastructure.** No ablation testing, no index-version regression harness, no MRR/NDCG beyond the recall@k + MRR shipped. The 35 gold queries are reference data for consumers' own eval, not a benchmark LibV2 optimizes against.
- **Online query APIs.** No HTTP server, no auth, no rate-limiting, no multi-tenant concerns. LibV2 is a library + CLI; online-retrieval-as-a-service belongs downstream.
- **Domain-specific scoring beyond the three metadata boosts.** Custom reranking by user profile, recency, author authority, etc., are all out of scope.

## Contracts

1. **Back-compat for `retrieve_chunks` callers.** When `include_rationale=False` (the default), `RetrievalResult.to_dict()` output is byte-identical to the pre-Worker-J schema. Production callers in `Trainforge/rag/libv2_bridge.py` are unaffected. Pinned by `Trainforge/tests/test_retrieval_improvements.py::TestWorkerJBackCompat`.
2. **Metadata-aware scoring default on, escape hatch per boost.** Pure BM25 is one flag away: `--no-metadata-scoring`, or any of `--no-concept-graph-boost` / `--no-lo-boost`. Callers who need determinism against changing per-course graphs can switch it off.
3. **Gold queries are hand-curated, not LO-derived.** Each `relevant_chunk_ids` entry must be a chunk whose text a human read. `kind: "lo-derived"` is reserved for explicit, tagged LO-expansion — retrieved numbers against LO-derived queries are NOT comparable to hand-curated numbers. This is written into both courses' `retrieval/README.md`.
4. **Evaluation numbers are diagnostic, not gates.** MRR 0.67 on WCAG_201 is today's baseline; it may move with corpus changes. Downstream consumers build their own gates on their own retrieval; ours are for sanity-checking.

## Decision log (append-only)

| Date | PR | What | Owner |
|---|---|---|---|
| 2026-04-17 | Worker J PR | Reference retrieval scope established. Rationale payload, metadata-aware scoring, 35 hand-curated gold queries across 2 courses, `retrieval-eval` CLI. | Worker J |

## Open questions / known issues not addressed

- `FOLLOWUP-ADR002-1` — The WCAG SC ref tokenization (`sc-1.4.3`) currently relies on a normalization pre-pass in `_canonicalize_query`; longer-term, `Trainforge/rag/wcag_canonical_names.canonicalize_sc_references` should emit the hyphenated form directly so the pre-pass is redundant.
- `FOLLOWUP-ADR002-2` — Metadata-aware scoring weight tuning. The default 0.3/0.3/0.2 split is reasonable but unvalidated against a large gold set. When more courses carry gold queries, run a small sweep and commit the resulting weights.
- `FOLLOWUP-ADR002-3` — Cross-course rationale. When `retrieve_chunks` is called without `course_slug`, the rationale's per-course metadata (graph, pedagogy) is loaded per candidate, which is fine but inefficient for very large catalogs. A `MultiCourseScorer` cache would help at scale; not needed today.
- `FOLLOWUP-ADR002-4` — Dense-embedding optional module. Deliberately out of scope for this ADR; if the project ever decides to ship one it should go in `LibV2/tools/libv2/retriever_dense.py` as a *separate* module, with its own opt-in flag — not folded into `retriever.py`.
