# Chunk schema v4

Owner of this doc: the first of Workers B/D/E to declare `CHUNK_SCHEMA_VERSION = "v4"`
in `Trainforge/process_course.py`. That worker landed the initial version; the
other two amend their added fields in-place as they merge into the
`chunk-schema-v4` rebase branch (see
[`ADR-001` Contract 1](../architecture/ADR-001-pipeline-shape.md#contract-1--chunk-schema-versioning)
and [`docs/contributing/workers.md`](../contributing/workers.md)).

## Why v4 exists

Three independent workers (B, D, E) each add fields to the chunk object.
Bumping `CHUNK_SCHEMA_VERSION` once, collectively, avoids three silent
coordinated-breakage releases. Every chunk at v4 carries:

- `schema_version: "v4"` (string, stamped on every chunk by
  `CourseProcessor._create_chunk`)

and every `manifest.json` at v4 carries:

- `chunk_schema_version: "v4"` (string, stamped by
  `CourseProcessor._generate_manifest`).

Readers checking schema compatibility MUST read `chunk_schema_version` from
`manifest.json` and/or `schema_version` from individual chunks. v3 consumers
MUST be updated to handle v4's new fields as optional — they are additive;
none of v1–v3's fields have been removed or renamed.

## Fields added at v4

### Worker D — per-chunk summary and retrieval_text

| Field | Type | Required? | Semantics |
|---|---|---|---|
| `summary` | string | yes (Worker D) | 2–3 sentences, 40–400 characters, never exceeds `len(text)`. Deterministic extractive generation; see `Trainforge/generators/summary_factory.py`. Used by retrieval to boost recall; measured by `Trainforge/rag/retrieval_benchmark.py`. |
| `retrieval_text` | string | no | Optional. When present, composed as `summary + " " + key_terms_joined`. Emitted only when it demonstrably lifts recall@k on the held-out LO-statement question set. Absent in the initial Worker D PR unless the benchmark proves a positive delta; see the PR body for the measured lift. |

Worker D's writer: `Trainforge/generators/summary_factory.py::generate`.
Benchmark: `Trainforge/rag/retrieval_benchmark.py::run_benchmark`.
Benchmark artifact location: `<output>/quality/retrieval_benchmark.json`.
Activated via the `--benchmark-retrieval` CLI flag on `Trainforge/process_course.py`.

### Worker B — (to be filled by Worker B)

Worker B amends this section with the five flow-metrics field names it
adds to chunks (if any land on the chunk object; several of B's metrics
live on the quality report, not the chunk).

### Worker E — (to be filled by Worker E)

Worker E amends this section with the HTML XPath provenance field(s).
Reserved name: `xpath_provenance` (scalar or list, TBD by Worker E).

## Field-level invariants

The following invariants are enforced by `Trainforge/tests/`:

- `summary` length ∈ [40, 400]. Asserted by
  `test_summary_factory.test_extractive_length_bounded`.
- `summary` is deterministic under identical inputs. Asserted by
  `test_summary_factory.test_extractive_deterministic`.
- `len(summary) <= len(text)` on real chunks (the pure-function guard in
  `summary_factory._clamp_length` handles near-empty edge cases
  defensively by padding). Asserted by
  `test_summary_factory.test_summary_not_longer_than_text`.
- `schema_version` equals `CHUNK_SCHEMA_VERSION` on every chunk after
  regeneration. Asserted by
  `test_summary_factory.test_schema_version_stamped`.
- `manifest.json::chunk_schema_version` equals `CHUNK_SCHEMA_VERSION`.
  Asserted by `test_summary_factory.test_manifest_schema_version`.

## Migration path

v3 → v4 is additive-only. A v3 corpus can be regenerated into v4 by
re-running `python -m Trainforge.process_course ...` against the same
`--imscc`. LibV2 importers reading chunk metadata must treat
`schema_version`, `summary`, and `retrieval_text` as optional; see
`LibV2/tools/libv2/retriever.py::RetrievalResult` for the reader contract.

## Versioning policy

One bump per release train. No worker bumps `CHUNK_SCHEMA_VERSION`
independently. See `ADR-001` Contract 1.
