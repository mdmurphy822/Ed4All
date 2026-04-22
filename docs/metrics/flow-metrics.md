# Flow metrics — `quality_report.json` (METRICS_SEMANTIC_VERSION 5)

Worker B added five flow metrics to the base-pass quality report. Worker P added a single top-level aggregate (`package_completeness`) that rolls those five into one honest number so consumers can read package metadata health at a glance. Each individual metric still surfaces a **silent metadata drop** between the HTML parser and the chunk writer that the previous `metrics` block couldn't see, because the previous metrics all looked at a single property in isolation (bloom coverage, LO coverage, etc.) and not at the flow from parser output to chunk output.

The theme: these metrics don't raise quality; they raise **visibility**. When one drops below expectation, the bug is upstream (usually in `_extract_section_metadata` or in `_create_chunk`), and the right fix is Worker C's backfill, not a weighting tweak here.

All five live under `metrics` in `quality_report.json`. Two of them also attach an `integrity.*` failure list so a reviewer can jump straight to the offending chunk IDs instead of scanning the full corpus.

See ADR-001 Contract 2 for the ownership story (base pass owns `metrics_semantic_version`; alignment does not bump it). See `Trainforge/process_course.py::_compute_flow_metrics` for the implementation.

## `content_type_label_coverage`

- **What it measures.** Fraction of chunks carrying a non-empty `content_type_label` (`explanation`, `example`, `procedure`, `definition`, etc.).
- **Why it matters.** Courseforge JSON-LD declares `contentType` per section. `_extract_section_metadata` threads that onto chunks. When this metric dips below ~0.8 on a Courseforge-sourced IMSCC, the upstream fell back to `data-cf-*` parsing (lossier) or the heading-match failed — either way, a downstream consumer that filters by content type is now reasoning about a biased subset.
- **Threshold reading.** 1.0 is the expected target for Courseforge output. Anything below 0.7 on a Courseforge-sourced package means the section-metadata-to-chunk join is broken on many pages; investigate the heading normalizer in `_extract_section_metadata`.

## `key_terms_coverage`

- **What it measures.** Fraction of chunks with at least one `key_terms` entry.
- **Why it matters.** Key terms come from JSON-LD `keyTerms` or from `data-cf-key-terms` attributes. They're a major signal for retrieval and for Worker C's training-pair synthesis — a dropped `key_terms` field yields a chunk that looks content-dense but has no surface terminology hooks.
- **Threshold reading.** This one is genuinely variable. Narrative pages often have zero key terms and that's fine; procedural or definitional pages should have some. A corpus-level reading below 0.3 on a Courseforge course is the signal for "upstream is silently dropping these."

## `key_terms_with_definitions_rate`

- **What it measures.** Across every key-term entry on every chunk, the fraction whose `definition` field is non-empty.
- **Denominator note.** Denominator is the **total key-term count**, not the chunk count. A chunk with 3 terms and 2 definitions contributes `2/3`, not `1.0`.
- **Why it matters.** There is a known fallback in `_extract_section_metadata` (the `data-cf-key-terms` path, around `process_course.py:955`) that yields terms with empty definitions: it parses the comma-separated term list but has no way to recover the definitions because `data-cf-key-terms` is term-strings-only. The JSON-LD path carries definitions. When this metric dips, the corpus is silently using the lossy fallback.
- **Integrity list.** `integrity.chunks_with_empty_definitions` names the chunk IDs that have at least one term with an empty definition, so a reviewer can jump straight to the page.
- **Threshold reading.** 1.0 on a JSON-LD-fidelity Courseforge course; below 0.5 means the fallback path is dominating.

## `misconceptions_present_rate`

- **What it measures.** Fraction of chunks carrying at least one `misconceptions` entry, computed **over the eligible denominator only** — the set of chunks whose parent page had at least one misconception in its JSON-LD.
- **Denominator note.** Threading is populated in `_chunk_content` (`self._pages_with_misconceptions`, a set of `lesson_id`s). When the parser found misconceptions somewhere in the corpus, the denominator is the chunks from those pages. When **no** page had misconceptions anywhere, the denominator falls back to all chunks — in which case the metric is 0.0 and the methodology string announces the fallback.
- **Why it matters.** Misconceptions are the one metadata field whose absence is actually informative. If a page declared misconceptions in its JSON-LD but they didn't land on any chunk from that page, the pedagogy signal was silently dropped between parse and chunk. Without this metric, there was no way to tell from `quality_report.json` alone that half a corpus's misconceptions never reached retrieval.
- **Integrity list.** `integrity.chunks_missing_misconceptions` names the chunk IDs whose parent page had misconceptions but whose own chunk dict does not, so a reviewer can jump straight to the broken join.
- **Threshold reading.** 1.0 is the target when `pages_with_json_ld_misconceptions` is the denominator. 0.0 with `all_chunks_fallback` is not a failure — it just means the corpus never had JSON-LD misconceptions to begin with.

## `interactive_components_rate`

- **What it measures.** Fraction of chunks whose HTML matches one of the parser's `COMPONENT_PATTERNS` (flip-card, accordion, tabs, callout, knowledge-check, activity-card).
- **Threading caveat.** Interactive components are not yet threaded onto chunks as a first-class field. The parser produces `parsed_items[i]["interactive_components"]`, but `_create_chunk` does not copy that list onto the chunk. This metric therefore uses a regex fallback against each chunk's own HTML.
- **Why it matters.** Interactive components are a distinct content type for downstream training-pair synthesis (they often signal `apply`-level Bloom). A corpus with zero detected interactive components in `quality_report.json` on a Courseforge course is a signal that the parser-to-chunk join never carried them through.
- **Follow-up.** Promoting interactive components to a first-class chunk field belongs to Worker E's HTML-provenance track, not Worker B. Tracked as `FOLLOWUP-WORKER-B-1`.
- **Threshold reading.** Corpus-dependent. A heavily interactive Courseforge course should land above 0.5; a text-dense course may legitimately sit below 0.2. What the metric catches is "expected pattern matches are simply absent" — the silent drop — not "this course doesn't use interactive components."

## Integrity fields summary

| Integrity field | Populated by | Purpose |
|---|---|---|
| `chunks_with_empty_definitions` | `key_terms_with_definitions_rate` | chunk IDs with ≥1 term lacking a definition |
| `chunks_missing_misconceptions` | `misconceptions_present_rate` | chunk IDs whose parent page had misconceptions but whose own chunk did not |

`content_type_label_coverage`, `key_terms_coverage`, and `interactive_components_rate` do not attach integrity lists — their dip signals a corpus-wide upstream issue rather than per-chunk join failures, and dumping every affected chunk ID would obscure the signal.

## `package_completeness` aggregate (v5, Worker P)

- **What it measures.** A single flat mean of the five enrichment coverage fractions:
  - `bloom_level_coverage`
  - `content_type_label_coverage`
  - `key_terms_coverage`
  - `misconceptions_present_rate`
  - `interactive_components_rate`
- **Where it lives.** Top level of `quality_report.json`, sibling of `overall_quality_score` — **not inside `metrics`**.
- **What it answers.** "Of the metadata this package claims to provide, how much actually landed."
- **What it is NOT.**
  - Not a weighted quality score. Equal weight per component, rounded to 3 decimals.
  - Not feeding `overall_quality_score`. That formula is unchanged (25% size + 20% tags + 20% html + 20% bloom + 15% LO).
  - Not gating `validation.passed`. A package can ship with low completeness; what matters to the validation gate is referential integrity and the existing thresholds.
- **Threshold reading.**
  - `≥ 0.9` — package enrichment is fully populated; downstream filters get an unbiased sample.
  - `0.5 – 0.9` — partial enrichment; consumers who filter by flow-metric fields (content type, key terms, misconceptions) get a biased subset. Open the individual `metrics.*` values to see which field is dropping.
  - `< 0.5` — enrichment pipeline is broken or source data lacks most metadata. Investigation territory (see VERSIONING.md §4.4a).

### Example (excerpt)

```json
{
  "metrics_semantic_version": 5,
  "overall_quality_score": 0.834,
  "package_completeness": 0.741,
  "metrics": {
    "bloom_level_coverage": 1.0,
    "content_type_label_coverage": 0.527,
    "key_terms_coverage": 0.527,
    "misconceptions_present_rate": 0.542,
    "interactive_components_rate": 0.618
  },
  "methodology": {
    "package_completeness": "Flat mean of bloom_level_coverage, content_type_label_coverage, key_terms_coverage, misconceptions_present_rate, and interactive_components_rate. …"
  }
}
```

A consumer reading `package_completeness: 0.741` knows about 26% of enrichment is missing at a glance; opening the individual metrics tells them which fields.

## Versioning

- Base pass only. Alignment pass must not bump `METRICS_SEMANTIC_VERSION` and must not write under `metrics` (ADR-001 Contract 2).
- Bumps are logged in the ADR-001 decision log.
- v4 (Worker B): five flow metrics added.
- v5 (Worker P): `package_completeness` top-level aggregate added. Scoring impact: **none**. The aggregate is observability-only; it does not feed `overall_quality_score` or gate `validation.passed`.
