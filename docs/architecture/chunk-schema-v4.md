# Chunk schema v4

Chunk v4 is Ed4All’s shared retrieval record. It preserves readable course
content, its source trail, accessibility state, and its links to the learning
model in one JSON object. Trainforge writes these objects to `chunks.jsonl` and
records the same contract version in the adjacent manifest.

The machine-readable source of truth is
[`schemas/knowledge/chunk_v4.schema.json`](../../schemas/knowledge/chunk_v4.schema.json).
This guide explains how its major surfaces fit together; it does not duplicate
the schema’s complete property catalog.

## Data flow

```mermaid
flowchart LR
    source["Private source material"] --> semantik["SemantiK accessible HTML"]
    semantik --> chunker["Trainforge chunking"]
    course["Course structure and objectives"] --> chunker
    chunker --> records["Chunk v4 JSONL"]
    records --> retrieval["Retrieval and grounded generation"]
    records --> validation["Quality and provenance validation"]
    records --> graph["Concept and objective graph"]
```

Text equivalent: private source material is converted to accessible HTML by
SemantiK. Trainforge combines that HTML with course structure and objectives to
emit chunk-v4 records. Retrieval, validation, and graph-building consumers read
the same records.

SemantiK supplies accessible structure and block metadata; Trainforge owns the
chunk boundary and materializes the final record. The live emission path is
[`Trainforge/pipeline/process_course.py`](../../Trainforge/pipeline/process_course.py), with
boundary logic under [`Trainforge/chunker`](../../Trainforge/chunker).

## Identity and sequence

Every record has an `id`, `schema_version`, `chunk_type`, and `follows_chunk`.
The identifier is opaque to consumers: use it for joins and citations, but do
not parse it to recover course metadata. `follows_chunk` preserves local reading
order without requiring array position to remain stable.

`schema_version` is the literal `v4`. The chunkset manifest carries a matching
`chunk_schema_version`, allowing a reader to reject an unsupported collection
before streaming every row. Run provenance may also be present, but it is not a
substitute for content identity or source provenance.

## Text and content

The required content pair serves two different needs:

- `text` is the canonical plain-text retrieval and synthesis surface.
- `html` preserves the source fragment for rendering and structural review.

Consumers should not silently substitute HTML, summaries, or legacy aliases
when `text` is missing. Classification and sizing fields describe the record’s
content type, difficulty, Bloom level, word count, and estimated token count.
Optional enrichment can add summaries, retrieval text, key terms, claims,
misconceptions, or cognitive-task metadata without changing the canonical prose.

## Source provenance

The required `source` object locates the chunk within the course structure. It
can carry document hashes, structural locations, and typed source references.
Each source reference uses the shared
[`source_reference.schema.json`](../../schemas/knowledge/source_reference.schema.json)
contract so a consumer can trace a chunk back to the evidence that supports it.

Top-level `source_pages` may preserve a normalized page span when page evidence
is available. A missing optional locator means the producer could not provide
that locator; it must not be replaced with a guessed path or page number.
Provenance-sensitive consumers should use the structured source object rather
than extracting identifiers from HTML or display text.

## Accessibility state

Chunks may carry SemantiK’s block role and confidence, WCAG block status,
figure alternative text, semantic-preservation score, and document
certification state. These fields preserve upstream evidence for later gates;
they do not independently certify the final course.

A flagged status remains a flagged status downstream. Consumers must not treat
an absent accessibility field as a pass. The authoritative gate behavior is
described in [Validation architecture](validation-architecture.md), and the
SemantiK stage is described in
[`SemantiK/architecture.md`](../../SemantiK/architecture.md).

## Objective and concept links

`learning_outcome_refs` links a chunk to the outcomes it supports, while
`concept_tags` provides normalized concept labels for retrieval and graph
construction. Optional targeted-concept and objective-alignment records add the
Bloom demand and delivery evidence needed for more precise audits.

These links are references, not embedded copies of the full objective or graph.
Writers should preserve canonical identifiers, deduplicate repeated links, and
keep source-backed claims attached to their evidence. Consumers should tolerate
an empty required reference array where the schema permits one, then apply the
appropriate completeness or grounding gate for their use case.

Ontology helpers and graph-building code live under
[`lib/ontology`](../../lib/ontology). Canonical validation behavior is indexed
in [Validators](../validation/validators.md).

## JSON-LD view

[`schemas/context/chunk_v4_v1.jsonld`](../../schemas/context/chunk_v4_v1.jsonld)
maps the JSON record to RDF terms for graph export and semantic queries. It is a
consumer-side context: the normal JSONL emitter does not inject `@context` into
each row.

Use the context when an RDF representation is needed, and retain the original
JSON record for schema validation and byte-oriented artifact checks. Round-trip
tests protect the fields that have defined RDF mappings; an extension without a
context term remains JSON metadata until its semantic meaning is standardized.

## Versioning and validation

The version boundary and compatibility policy are defined in
[ADR-001](ADR-001-pipeline-shape.md#chunk-schema). Producers stamp both each
record and its manifest. Readers should branch on those explicit versions, not
on the presence of a recently added optional field.

Chunk validation uses JSON Schema Draft 2020-12 with the repository’s offline
schema registry so shared taxonomy references resolve without network access.
The production validation switch is documented in
[Trainforge behavior flags](../operations/behavior-flags-trainforge.md). When
strict validation is enabled, malformed emitted records fail the chunking stage;
validation should not coerce or discard fields to make a record pass.

The root record intentionally admits additive enrichment fields, while the
structural `source` object and several nested shapes are closed. This supports
experimentation without weakening the provenance core.

## Extending the contract

For a new interoperable field:

1. Define its purpose, ownership, type, absence semantics, and privacy impact.
2. Add it to the canonical schema and update the producer that owns the value.
3. Update the JSON-LD context when the field needs a stable RDF meaning.
4. Add focused tests for schema validity, emitter propagation, consumer
   behavior, and JSON-LD round trips where applicable.
5. Update the chunkset version only when the compatibility rules in ADR-001
   require it; do not infer a version bump from field count.

Extensions must not include source text, local paths, course names, hostnames,
or other private run details in public fixtures or documentation. Use synthetic
identifiers and generated examples. Actual chunksets and manifests remain
private runtime artifacts and must stay out of source control.
