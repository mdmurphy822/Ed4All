# RAG Indexer Agent

## Purpose

Builds the per-course on-device vector index that backs real semantic
retrieval. The agent runs a **deterministic** index build — real embeddings +
a numpy exact-search index — with **no LLM dispatch** and no decision-capture
obligation (precedent: the `dart-chunker` agent).

This agent is backed by the `run_vector_indexing` registry tool
(`MCP/tools/pipeline_tools.py`). It supersedes the prior behavior where the
`rag-indexer` agent was mis-mapped to `analyze_imscc_content` — an HTML/word
count scan that never produced an index and let the `indexing` phase "succeed"
without building anything.

## Responsibilities

1. **Embedding generation**: encode each chunk's `section_heading + text`
   (policy `text+heading`) with the env-configured embedding provider
   (`ED4ALL_EMBEDDING_PROVIDER`, default `st`).
2. **Vector indexing**: persist a pure-numpy exact-search index to
   `LibV2/courses/<slug>/vector_index/`:
   - `embeddings.npy` — float32 `[N, dim]`, L2-normalized rows.
   - `id_map.json` — load-bearing chunk-id row order.
   - `manifest.json` — provenance manifest (provider/model/dim/SHAs/chunkset
     pin) per `schemas/library/vector_index_manifest.schema.json`.
3. **Provenance**: record the model fingerprint, chunkset kind, and the
   source-chunks SHA so the query path can fail closed on staleness.

## Inputs

- The course (resolved to `LibV2/courses/<slug>/`) whose chunkset
  (`imscc_chunks/` → `dart_chunks/` → legacy `corpus/`) is embedded.
- Embedding provider / model selection via the `ED4ALL_EMBEDDING_*` env family
  (or per-call `provider` / `model` kwargs).

## Outputs

- `LibV2/courses/<slug>/vector_index/{embeddings.npy,id_map.json,manifest.json}`.
- A success envelope surfacing the manifest path, model fingerprint, embedding
  dim, chunk count, and source-chunks SHA.

## Failure modes (fail-closed — no silent degradation)

The build **fails the phase** (returns `{"success": false, "error": ...}`) —
it NEVER falls back to a lexical/BM25 or file-counting result — when:

- the embedding backend is unavailable (`EmbeddingBackendUnavailable`: weights
  not cached offline, local embeddings server down, `[embedding]` extra
  missing) → `error_type: embedding_backend_unavailable`;
- the course directory or its chunkset is missing → `error_type:
  course_missing` / `chunkset_missing`;
- a fresh index already exists and `force` was not set → `error_type:
  fresh_index_exists`.

## Offline posture

The query path loads the embedding client with `offline=True` so a query can
never trigger a model download. The build path (this agent) runs with
`offline=False` so provisioning may download weights once — downloads happen at
build time only.

## Integration

Works with:
- `assessment-extractor` agent (the `indexing` phase depends on `extraction`).
- LibV2 storage (the index lands under the course tree alongside the chunkset).
- The semantic retrieval path, which reads `vector_index/` and fails closed
  (`SemanticIndexMissing` / `SemanticIndexStale` / `FakeIndexRefused`) rather
  than returning lexical results.
