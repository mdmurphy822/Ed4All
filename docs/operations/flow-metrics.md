# Flow metrics

Flow metrics show whether enrichment metadata survives the path from parsed
course content to Trainforge chunks. They are observability signals: they help
operators find missing or incomplete metadata without changing the content,
repairing records, or relaxing validation.

Trainforge writes these values to `quality_report.json`. The current contract is
`metrics_semantic_version: 5`; the implementation and emitted methodology live
in [`Trainforge/pipeline/process_course.py`](../../Trainforge/pipeline/process_course.py).

## Report shape

The report separates summary, measurement, methodology, and evidence:

- `package_completeness` is a top-level summary of enrichment coverage.
- `metrics` contains the individual flow metrics alongside the other quality
  metrics.
- `methodology` explains each calculation, including denominator behavior.
- `integrity` lists affected chunk IDs for findings that can be traced to
  individual records.
- `validation` reports the base quality decision and its issues.

Flow metrics do not alter `overall_quality_score`, and
`package_completeness` does not independently determine `validation.passed`.
The authoritative validation behavior remains in
[`docs/validation/gates.md`](../validation/gates.md) and
`config/workflows.yaml`.

## Metric reference

### `content_type_label_coverage`

The fraction of chunks with a non-empty `content_type_label`, such as an
explanation, example, procedure, or definition.

A lower value means fewer chunks retain content-type metadata. Review the
parser-to-chunk metadata path before assuming the source omitted the labels.

### `key_terms_coverage`

The fraction of chunks with at least one `key_terms` entry.

This metric reports presence, not definition quality. A chunk can count as
covered even when one of its terms has no definition.

### `key_terms_with_definitions_rate`

The fraction of structured key-term entries whose `definition` field is
non-empty. Its denominator is the total number of structured key-term entries,
not the number of chunks.

When no structured key terms exist, the value is `0.0`. Review
`methodology.key_terms_with_definitions_rate` before interpreting that value as
a metadata-transfer defect.

### `misconceptions_present_rate`

The fraction of eligible chunks with at least one `misconceptions` entry.
Eligibility depends on parser evidence:

- When the parser found JSON-LD misconceptions, the denominator is chunks from
  pages that declared them.
- When it found none, the denominator falls back to all chunks and the metric is
  `0.0`.

The report records the active denominator in
`methodology.misconceptions_present_rate`. This distinction prevents an absent
upstream signal from being mistaken for a failed metadata join.

### `interactive_components_rate`

The fraction of chunks whose HTML matches a canonical interactive-component
pattern, including flip cards, accordions, tabs, callouts, knowledge checks,
and activity cards.

Interactive components are detected from chunk HTML rather than a first-class
chunk field. Treat this value as pattern-based observability, not a complete
inventory of every interactive behavior.

## Integrity evidence

Two flow checks provide record-level evidence:

| Field | Meaning |
|---|---|
| `integrity.chunks_with_empty_definitions` | Chunk IDs containing at least one structured key term without a definition. |
| `integrity.chunks_missing_misconceptions` | Eligible chunk IDs whose source page declared misconceptions but whose chunk has none. |

The other flow metrics are aggregate coverage signals and do not emit a list of
every uncovered chunk.

## Package completeness

`package_completeness` is the rounded, equally weighted mean of:

1. `metrics.bloom_level_coverage`
2. `metrics.content_type_label_coverage`
3. `metrics.key_terms_coverage`
4. `metrics.misconceptions_present_rate`
5. `metrics.interactive_components_rate`

It is a compact answer to “how much of the expected enrichment reached the
package?” It is not a weighted quality score. In particular,
`key_terms_with_definitions_rate` is diagnostic and is not one of the aggregate
components.

Use the aggregate to orient an investigation, then inspect the component
metrics, their methodology strings, and the integrity lists. Avoid applying a
universal interpretation to metadata that may be legitimately absent from a
particular source.

## Investigation workflow

1. Confirm `metrics_semantic_version` before comparing reports produced by
   different versions.
2. Read the metric's `methodology` entry to identify its denominator and empty-
   input behavior.
3. Inspect the corresponding `integrity` list when one is available.
4. Compare the affected chunks with their parsed source metadata and HTML.
5. Fix the parser, metadata join, or chunk-emission path responsible for the
   loss; do not change validation thresholds to conceal it.
6. Regenerate the report and confirm both the metric and its integrity evidence.

The chunk contract is documented in
[`docs/architecture/chunk-schema-v4.md`](../architecture/chunk-schema-v4.md).
Base-pass ownership of `metrics_semantic_version` and the separation between
base and alignment metrics are defined by
[`ADR-001`](../architecture/ADR-001-pipeline-shape.md#quality-metrics).

## Versioning contract

- The base quality-report pass owns `metrics_semantic_version` and `metrics`.
- The alignment pass does not bump the base semantic version or write into the
  base `metrics` object.
- Alignment records the base version it observed so consumers can detect stale
  alignment output.
- Any metric-shape or semantic change requires a coordinated version decision;
  a prose clarification alone does not.
