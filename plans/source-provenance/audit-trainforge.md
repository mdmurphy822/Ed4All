# Trainforge Source-Provenance Audit

Scope: Audit Trainforge's current source-provenance plumbing (chunks, concept
graph, typed-edge evidence, archival) and the hooks where a proposed
Courseforge `sourceReferences` field would land. Read-only; no code changes.

## Current state — chunk-level source fields

`schemas/knowledge/chunk_v4.schema.json:131-184` defines `$defs/Source`, the
only provenance block on a chunk. Strictly required: `course_id`, `module_id`,
`lesson_id`. Optional but emitted today:

- `module_title`, `lesson_title` — human labels.
- `resource_type` — IMSCC resource type (e.g. `webcontent`, `imsqti_xmlv1p2`).
- `section_heading` — H-tag text, with `(part N)` suffix on multi-part splits.
- `position_in_module` — integer ordinal.
- `html_xpath` — absolute xpath into the IMSCC HTML element (audit-trail).
- `char_span: [start, end]` — char offsets into the xpath-resolved container.
- `item_path` — IMSCC-relative HTML file path.

`additionalProperties:false` on `Source` (schema line 134): **structural
core is strict** — new source fields require a schema change. Emit site:
`Trainforge/process_course.py::_create_chunk` at L1290-1310.

Gaps vs. DART-anchored provenance:

- No `source_page` / `source_document` — chunks point at the IMSCC HTML
  file (`item_path`), not at the DART-original PDF page or DART chunk ID.
- No `dart_chunk_id` field — the link back from Trainforge chunk to the
  DART multi-source-synthesis chunk that fed Courseforge is absent.
- No `source_block_id` — block-level granularity within a Courseforge
  page is not carried; granularity today stops at `section_heading` +
  `char_span`.

Root `additionalProperties:true` (schema line 23) allows *additive*
forward-compat on chunks, but any new `source.*` key needs the Source
sub-schema opened up or declared in the `$defs/Source.properties` block.

## Current state — concept graph source fields

`schemas/knowledge/concept_graph_semantic.schema.json:22-51` — node props:

- `id`, `label`, `frequency`.
- `run_id` + `created_at` (Wave 4, Worker P).
- `course_id` (Wave 4, Worker O) — only under
  `TRAINFORGE_SCOPE_CONCEPT_IDS=true`; enables `{course_id}:{slug}`
  disambiguation.
- `occurrences: List[chunk_id]` (Wave 5, Worker S) — sorted ASC, populated
  from the inverted `concept_to_chunks` index at
  `Trainforge/process_course.py:2250, 2295-2297`.

No direct node-level source pointer (no PDF page, DART chunk, or
IMSCC page id). Source is reconstructable **only** by dereferencing each
`occurrences[]` chunk_id → `chunks.jsonl` → `chunk.source.*`. That's a
two-hop for every concept lookup.

Edge-level provenance (schema lines 84-106):

- `rule` + `rule_version` (every edge).
- `evidence` (rule-discriminated oneOf — see next section).
- `run_id` + `created_at` (Wave 4, Worker P; optional).
- `confidence`, `weight`.

No generic `source_chunks: []` array at the edge level; source anchoring is
entirely encoded inside the `evidence` arm, and only some arms carry it.

`occurrences[]` quality caveat: stable across re-chunks only when
`TRAINFORGE_CONTENT_HASH_IDS=true` (Worker N Wave 4). Default
position-based chunk IDs invalidate occurrence entries on every re-chunk
(documented at `defined_by_from_first_mention.py:22-25`).

## Current state — evidence discriminator source-awareness

Per the 8 modeled arms in `concept_graph_semantic.schema.json:$defs`:

| Evidence arm | Source anchor? | Fields |
|---|---|---|
| `IsAEvidence` (L112-123) | **chunk** | `chunk_id`, `term`, `definition_excerpt`, `pattern` |
| `PrerequisiteEvidence` (L124-135) | **abstract** | `target_first_lo`, `target_first_lo_position`, `source_first_lo`, `source_first_lo_position` — LO-position scalars only, no chunk_id |
| `RelatedEvidence` (L136-145) | **abstract** | `cooccurrence_weight`, `threshold` — pure numeric signal |
| `AssessesEvidence` (L146-156) | **partial chunk** | `question_id`, `objective_id`, optional `source_chunk_id` |
| `ExemplifiesEvidence` (L157-167) | **chunk** | `chunk_id`, `concept_slug`, `content_type` |
| `MisconceptionOfEvidence` (L168-177) | **abstract** | `misconception_id`, `concept_id` — no chunk |
| `DerivedFromObjectiveEvidence` (L178-187) | **chunk** | `chunk_id`, `objective_id` |
| `DefinedByEvidence` (L188-198) | **chunk** | `chunk_id`, `concept_slug`, `first_mention_position` |
| `FallbackProvenance` (L287-310) | **varies** | `evidence: object` — lenient passthrough |

All modeled arms set `additionalProperties:false` (strict mode, via
`TRAINFORGE_STRICT_EVIDENCE=true`), so **augmenting any arm with new
source-ref fields is a schema change**, not a free-form addition. Arms
that currently carry no chunk pointer (`PrerequisiteEvidence`,
`RelatedEvidence`, `MisconceptionOfEvidence`) must be extended if we want
per-edge PDF-page traceability.

## HTML parser consumption

`Trainforge/parsers/html_content_parser.py` reads from Courseforge output:

JSON-LD (priority 1, `_extract_json_ld` L276-287):

- `pageId`, `learningObjectives[]` (id/statement/bloomLevel/bloomVerb/
  cognitiveDomain/keyConcepts/assessmentSuggestions), `sections[]`
  (heading/contentType/bloomRange/keyTerms), `misconceptions[]`,
  `suggestedAssessmentTypes`, `prerequisitePages` (consume-only; not
  emitted — see `plans/kg-quality-review-2026-04/discovery/c-jsonld-contract.md:82`).

`data-cf-*` attributes (priority 2):

- `data-cf-content-type`, `data-cf-key-terms`, `data-cf-teaching-role`
  (Wave 2, Worker K), `data-cf-objective-ref` + `data-cf-objective-id`
  (Wave 3, Worker M), `data-cf-bloom-level`, `data-cf-bloom-verb`,
  `data-cf-cognitive-domain`, `data-cf-role="template-chrome"` (Worker Q —
  boilerplate strip).

**No existing slot for `sourceReferences` or `data-cf-source-*`** on
either `ContentSection` or `ParsedHTMLModule` dataclasses
(L28-90). Adding them requires new dataclass fields **plus** new
extractor branches in `_extract_sections` (L303-373) and/or
`_extract_json_ld`.

## Proposed extensions

**Chunk schema (`chunk_v4.schema.json`):**

- Open `$defs/Source` with opt-in fields:
  - `source_references: array` — list of `{dart_chunk_id, source_page,
    block_span?, source_document?}` objects, one per block merged into
    this chunk. Provenance granularity is per-chunk because a merged
    chunk (`_merge_small_sections` at `process_course.py:1111`) may
    aggregate multiple source blocks.
  - Alternative: flatten `source_page: integer`, `source_document:
    string`, `dart_chunk_refs: [string]` if multi-source aggregation is
    rare.
- Keep `Source.additionalProperties:false` intact but declare the new
  properties explicitly so the strict path stays strict.

**Concept graph nodes (`concept_graph_semantic.schema.json`):**

- Add optional `source_refs: array` of
  `{chunk_id, dart_chunk_id?, source_page?}` on nodes so consumers can
  answer "what PDF page first mentioned this concept" without the
  two-hop dereference. Populate from the first chunk in
  `occurrences[]` at `_build_tag_graph` emit time.

**Evidence arms (per-rule):**

- `PrerequisiteEvidence`: add optional `target_chunk_id`,
  `source_chunk_id` — the chunks where the first-mention lookup
  resolved. Cheap: `_first_positions_by_concept`
  (`prerequisite_from_lo_order.py:53`) already tracks `chunk_id` — it
  just isn't threaded into `evidence`.
- `RelatedEvidence`: add optional `source_chunks: [chunk_id]` —
  co-occurring chunks (or a representative sample). Requires re-linking
  `related_from_cooccurrence.py` to chunks (currently consumes only
  `concept_graph.edges`, so this is a non-trivial wiring change).
- `MisconceptionOfEvidence`: add optional `source_chunk_id` — the chunk
  in which the misconception was declared (JSON-LD `misconceptions[]` is
  already per-page).
- All arms with `chunk_id` should optionally carry `dart_chunk_id` and
  `source_page` (propagated from the chunk's new `source.source_references`).
- Bump `RULE_VERSION` on every rule that gets an evidence-shape change
  (each rule file's `RULE_VERSION` constant). `rule_versions` is tracked
  at `typed_edge_inference.py:356`.

## Consumption path for Courseforge `sourceReferences`

If Courseforge emits `sourceReferences: [{dart_chunk_id, block_span,
source_page, source_document}]` **per page (JSON-LD top-level)** and
**per section** (under `sections[].sourceReferences`), plus inline
`data-cf-dart-chunk-id` / `data-cf-source-page` on block-level elements:

1. **Schema contract (`schemas/knowledge/courseforge_jsonld_v1.schema.json`)**
   — add `sourceReferences` as an allowed top-level key and inside
   `$defs/Section.properties`. `additionalProperties:false` on root
   (L9) and Section (L117) **blocks forward-compat today**; schema
   must be updated first.

2. **Parser (`Trainforge/parsers/html_content_parser.py`)**:
   - `_extract_json_ld` (L276): already captures the whole JSON-LD dict
     into `parsed.metadata["courseforge"]` — available as-is to
     downstream.
   - `_extract_sections` (L303-373): add regex for
     `data-cf-dart-chunk-id="..."` and `data-cf-source-page="..."`,
     store on a new `ContentSection.source_references: List[Dict]` field.
   - `ParsedHTMLModule` (L70-90): add `source_references: List[Dict]` +
     the parsed section.sourceReferences union.

3. **Chunker (`Trainforge/process_course.py`)**:
   - Thread `parsed.metadata["courseforge"].sourceReferences` through
     `_chunk_content` (L1008) into the per-item dict at L965-992.
   - In `_create_chunk` (L1271), fold section/page `sourceReferences`
     into `source["source_references"]` alongside the existing
     `html_xpath` / `char_span` / `item_path`.
   - For merged sections (`_merge_small_sections` L1111), aggregate
     sourceReferences from all merged sections — a chunk can span
     multiple DART source chunks.

4. **Concept-graph `_build_tag_graph`
   (`process_course.py:2215`)**: when emitting a node, pull the first
   `chunk_id` from `occurrences[]` → look up its
   `source.source_references[0]` → attach as `node.source_refs`.

5. **Inference rules**: each rule that constructs `evidence` with a
   `chunk_id` should also copy the chunk's `source_references[0]` into
   the evidence (or add a minimal `dart_chunk_id` + `source_page` pair).
   Bumps `RULE_VERSION`. Rules with no chunk anchor today
   (`prerequisite_from_lo_order`, `related_from_cooccurrence`) need
   plumbing to find a representative chunk per concept (concept →
   `occurrences[0]` is a reasonable proxy).

## LibV2 archival preservation

`MCP/tools/pipeline_tools.py::archive_to_libv2` (L419-570) copies files
**verbatim** (`shutil.copy2`): PDFs, DART HTML, IMSCC, assessment. The
chunks.jsonl / concept_graph.json content is not transformed — if
`assessment_path` points at the Trainforge corpus dir, the files land
in `courses/<slug>/corpus/` as-is (L501-502). Manifest (L534-548)
carries only SHA-256 checksums + classification, no per-chunk/per-node
metadata.

→ **Source-ref metadata survives archival unchanged** because archival
doesn't read chunk/graph internals. The only risk is the manifest not
advertising the presence of source refs; recommend adding a capability
flag (e.g. `manifest.features.source_provenance: true`) so LibV2
retrieval callers can fast-skip source-grounded queries on legacy
corpora.

## Opt-in flag integration

Current flags (root `CLAUDE.md` § Opt-In): `TRAINFORGE_CONTENT_HASH_IDS`,
`TRAINFORGE_SCOPE_CONCEPT_IDS`, `TRAINFORGE_PRESERVE_LO_CASE`,
`TRAINFORGE_VALIDATE_CHUNKS`, `TRAINFORGE_ENFORCE_CONTENT_TYPE`,
`TRAINFORGE_STRICT_EVIDENCE`, `DECISION_VALIDATION_STRICT`.

Assessment:

- Chunk-level source refs are **additive** — `chunk_v4.schema.json`
  root is `additionalProperties:true`, `Source.additionalProperties:false`
  is the one pinch point. If we explicitly add
  `Source.source_references` as an **optional** property (not
  required), legacy chunks without it validate and new chunks emit it
  → no flag needed. Consumers that require it can gate themselves.
- Evidence-arm extensions are **NOT free**. Every modeled arm has
  `additionalProperties:false` (strict mode). Adding `source_chunk_id`
  etc. to e.g. `PrerequisiteEvidence` is a schema-breaking change *under
  strict mode* — rule-version bump and a coordinated `TRAINFORGE_STRICT_EVIDENCE`
  re-validation cycle is needed. A new flag **`TRAINFORGE_SOURCE_PROVENANCE`**
  could gate **emit** (not schema), letting the rollout happen in waves:
  off → no new fields emitted, on → fields emitted + rule_version bumped.
  This lets existing LibV2 corpora keep replaying under the old schema.
- Node-level `source_refs[]` on concept graph is additive (same pattern
  as Wave 5 `occurrences[]`) → no flag required.

Recommendation: **introduce `TRAINFORGE_SOURCE_PROVENANCE`** gating the
**evidence-arm extension emit path only**; chunk-level `source_references`
and node-level `source_refs` are additive-optional and need no flag.

## Key code paths

- Chunk emit (source block construction): `Trainforge/process_course.py:1290-1310`.
- Chunk ID + source_locator derivation: `Trainforge/process_course.py:1220-1267`.
- Item-level JSON-LD + data-cf capture: `Trainforge/process_course.py:965-992`.
- HTML parser JSON-LD entry: `Trainforge/parsers/html_content_parser.py:276-287`.
- HTML parser section attribute scans: `Trainforge/parsers/html_content_parser.py:303-373`.
- Concept graph builder (occurrences, node emit): `Trainforge/process_course.py:2215-2315`.
- Typed-edge orchestrator + per-edge stamping: `Trainforge/rag/typed_edge_inference.py:120-138, 363-385`.
- Rule-module evidence construction: `Trainforge/rag/inference_rules/*.py` (each `infer()` builds `provenance.evidence`).
- Chunk schema Source block: `schemas/knowledge/chunk_v4.schema.json:131-184`.
- Concept-graph schema evidence arms: `schemas/knowledge/concept_graph_semantic.schema.json:112-310`.
- Courseforge JSON-LD contract: `schemas/knowledge/courseforge_jsonld_v1.schema.json:9-160` (root + Section are `additionalProperties:false`).
- LibV2 archival: `MCP/tools/pipeline_tools.py:419-570`.

## Risks / things to watch

- **Source sub-schema strictness**: `chunk_v4.schema.json` `Source`
  uses `additionalProperties:false`. Any new source field must be
  declared in-schema, or the strict path
  (`TRAINFORGE_VALIDATE_CHUNKS=true`) fails closed on new chunks.
- **Evidence arm strictness**: all 8 modeled arms use
  `additionalProperties:false`. Adding `dart_chunk_id` to e.g.
  `PrerequisiteEvidence` breaks strict-mode validation of **old**
  graphs (they don't carry it — still ok because the field is
  optional) and of **new** graphs emitting it (ok only if we add the
  field to the arm). Bump `RULE_VERSION` on every touched rule so
  downstream `rule_versions` map exposes the schema generation.
- **Chunk ID stability**: source-ref back-edges only stay stable across
  re-runs under `TRAINFORGE_CONTENT_HASH_IDS=true` (see
  `process_course.py:101` + `defined_by_from_first_mention.py:22-25`).
  Without it, every re-chunk invalidates node `occurrences[]` and any
  `source_chunk_id` inside evidence.
- **Merged-section aggregation**: `_merge_small_sections` collapses
  multiple sections into one chunk; a single `source_reference` on the
  chunk is insufficient. Must model `source_references` as an array.
- **`related_from_cooccurrence` doesn't see chunks today**
  (`related_from_cooccurrence.py:50`: `del chunks, course`). Threading
  source refs into `RelatedEvidence` requires re-plumbing the rule to
  scan chunks per co-occurring pair — non-trivial latency impact at
  scale.
- **Backward compat on existing LibV2 corpora**: 131-chunk live
  conformance (schema comment on chunk_v4 L6) presumes today's Source
  shape; any chunk-side emit change needs a re-run to populate source
  refs. Legacy corpora will have empty `source_references` → consumers
  must treat absence as "unknown", not error.

## Open questions for synthesis phase

1. **DART chunk ID format**: what does a DART chunk ID look like today? Is
   it PDF-page-scoped, or a multi-source-synthesis composite? Answer
   shapes whether `dart_chunk_id` is sufficient or we also need
   `{source_document, source_page}` triple.
2. **Courseforge emit granularity**: per-page (JSON-LD top-level),
   per-section (JSON-LD `sections[]`), or per-block (inline
   `data-cf-dart-chunk-id`)? Parser work scales very differently for
   each.
3. **Flag the evidence-arm extension**: do we bite the schema bump now
   (non-flag) and accept strict-mode churn, or ship behind
   `TRAINFORGE_SOURCE_PROVENANCE` and leave strict-mode callers on the
   legacy shape?
4. **Multi-source chunks**: DART does multi-source synthesis. A single
   Courseforge block may pull from 2+ PDF pages. Do we model
   `sourceReferences` as a sorted primary + secondaries, or flat list?
