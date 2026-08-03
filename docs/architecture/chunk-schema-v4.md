# Chunk schema v4

This document records the durable chunk-v4 contract declared by
`CHUNK_SCHEMA_VERSION = "v4"` in `Trainforge/process_course.py`. See
[`ADR-001` chunk-schema contract](ADR-001-pipeline-shape.md#chunk-schema)
for the versioning boundary.

## Compatibility contract

Chunk schema v4 adds retrieval enrichment and source provenance without
removing or renaming earlier fields. Every v4 chunk carries:

- `schema_version: "v4"` (string, stamped on every chunk by
  `CourseProcessor._create_chunk`)

and every `manifest.json` at v4 carries:

- `chunk_schema_version: "v4"` (string, stamped by
  `CourseProcessor._generate_manifest`).

Readers checking schema compatibility MUST read `chunk_schema_version` from
`manifest.json` and/or `schema_version` from individual chunks. v3 consumers
MUST be updated to handle v4's new fields as optional — they are additive;
none of v1–v3's fields have been removed or renamed.

## Retrieval enrichment

| Field | Type | Required? | Semantics |
|---|---|---|---|
| `summary` | string | yes | 2–3 sentences, 40–400 characters, never exceeding `len(text)`. Deterministic extractive generation lives in `Trainforge/generators/postprocessing/summary_factory.py`; retrieval benchmarks measure its recall effect. |
| `retrieval_text` | string | no | When present, composed as `summary + " " + key_terms_joined`. Enable it only after the held-out retrieval benchmark demonstrates a positive recall delta. |

Summary writer: `Trainforge/generators/postprocessing/summary_factory.py::generate`.
Benchmark: `Trainforge/rag/retrieval_benchmark.py::run_benchmark`.
Benchmark artifact location: `<output>/quality/retrieval_benchmark.json`.
Activated via the `--benchmark-retrieval` CLI flag on `Trainforge/process_course.py`.

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

One coordinated bump per release train. See `ADR-001` Contract 1.
