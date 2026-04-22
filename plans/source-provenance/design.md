# Design: End-to-end source provenance (Waves 7–10 on dev-v0.2.0)

## Summary

This work threads per-block source attribution from DART through Courseforge into Trainforge's chunks, concept graph, and typed-edge evidence. It lands as Waves 7–10 on `dev-v0.2.0` (same pattern as Waves 1–6: one worker per wave, worker branch merged via PR into dev-v0.2.0, main untouched, no new dev-branch). Today:

- **DART** emits a document-level `confidence_score` and a static hand-coded `sources_used` label per section type. No per-block source, no page refs on rendered blocks, zero `data-dart-*` attributes. The only structured provenance artifact (`*.quality.json`) is never staged to Courseforge.
- **Courseforge** content-generator agents are completely source-blind (receive only `project_id`); nothing in JSON-LD or `data-cf-*` carries source refs; source→module mapping dies at the objective-synthesizer boundary.
- **Trainforge** chunks have `html_xpath`/`char_span` but no link back to DART chunks or PDF pages; 4 of 8 evidence arms anchor in `chunk_id`, 3 are purely abstract (no source anchor at all).

One win the audits surface: **IMSCC packaging is byte-level** — Courseforge → Trainforge carries anything we emit in HTML intact. So the main work is at the emit sites, not the transport.

This design threads four artifacts through the pipeline:

1. Canonical `SourceReference` shape shared by all three subsystems' schemas.
2. DART per-block provenance (JSON sidecar + `data-dart-*` HTML attributes + staged `*.quality.json`).
3. Courseforge `source_mapping` phase + grounded content-gen agent + `sourceReferences` in JSON-LD + `data-cf-source-ids` HTML attributes.
4. Trainforge propagation: chunk `source.source_references[]`, concept node `source_refs[]`, and evidence-arm extensions (gated by `TRAINFORGE_SOURCE_PROVENANCE`).

## Cross-boundary contract

### Canonical `SourceReference` schema

New file: `schemas/knowledge/source_reference.schema.json`. Referenced by `$ref` from the three consumer schemas. Single source of truth prevents drift.

```jsonc
{
  "$id": "https://ed4all.dev/schemas/knowledge/source_reference.schema.json",
  "title": "SourceReference",
  "type": "object",
  "required": ["sourceId", "role"],
  "additionalProperties": false,
  "properties": {
    "sourceId": {
      "type": "string",
      "pattern": "^dart:[a-z0-9_-]+#[a-z0-9_]+$",
      "description": "Canonical DART block identifier: dart:{document_slug}#{block_id}. block_id is content-hash (16-hex) when TRAINFORGE_CONTENT_HASH_IDS=true, positional (s3_c0 style) otherwise."
    },
    "role": {
      "type": "string",
      "enum": ["primary", "contributing", "corroborating"],
      "description": "primary=the block IS this source; contributing=the block draws from this source; corroborating=this source supports the claim but isn't synthesized from."
    },
    "weight": {
      "type": "number", "minimum": 0, "maximum": 1,
      "description": "Optional. Relative contribution weight when multi-source."
    },
    "confidence": {
      "type": "number", "minimum": 0, "maximum": 1,
      "description": "Optional. Ingestor's confidence that this mapping is correct."
    },
    "pages": {
      "type": "array", "items": {"type": "integer", "minimum": 1},
      "description": "Optional. PDF page numbers. Populated when DART carries real page tracking."
    },
    "extractor": {
      "type": "string",
      "enum": ["pdftotext", "pdfplumber", "ocr", "claude", "synthesized"],
      "description": "Optional. Which extraction stage produced this block (propagated from DART's per-block envelope)."
    }
  }
}
```

### ID format

`dart:{document_slug}#{block_id}`. Examples:

- `dart:science_of_learning#s3_c0` (positional fallback)
- `dart:science_of_learning#a3f9d812ac04bbc1` (content-hash, 16-hex)

Prefer content-hash under `TRAINFORGE_CONTENT_HASH_IDS=true` for re-run stability. Positional fallback for legacy emit.

### Granularity tiers

Three tiers, emitted at different levels:

- **Page-level** — JSON-LD `sourceReferences[]` on top-level `CourseforgePage`. Required; always emitted when source-grounded generation runs.
- **Section-level** — JSON-LD `sections[].sourceReferences[]`. Optional; emit only when a section draws from different sources than the page overall.
- **Block-level** — `data-cf-source-ids="id1,id2"` attribute on `<section>`, headings, and component wrappers. Optional; emitted when section-level specificity is insufficient.

Never emit on every `<p>` / `<li>` / `<tr>` — HTML bloat concern from Courseforge audit is real, especially for textbook-scale runs.

## DART changes (Wave 8)

### Per-section JSON enrichment

`multi_source_interpreter.py::auto_synthesize_section` currently emits:

```json
{"section_type": "...", "section_title": "...", "data": {...}, "sources_used": "..."}
```

Rewrite to:

```json
{
  "section_id": "s3",
  "section_type": "contacts",
  "section_title": "...",
  "page_range": [3, 4],
  "provenance": {
    "sources": ["pdftotext", "pdfplumber"],
    "strategy": "pdfplumber_headers+pdftotext_entities",
    "confidence": 0.87
  },
  "data": { ... }
}
```

- `sources_used` → `provenance.strategy` (free-form string retained for back-compat).
- `provenance.sources` — typed enum array (`pdftotext|pdfplumber|ocr|synthesized`).
- `provenance.confidence` — per-section; seeds the Courseforge→Trainforge confidence field.
- `page_range` — propagated from `build_section_context` (currently computed and dropped). **Depends on form-feed preservation in `clean_text`** — see Risks.

### Per-block envelope

Wrap every leaf value in matcher functions (`synthesize_contacts`, `synthesize_systems_table`, `synthesize_roster`). Example for contacts:

```json
{
  "block_id": "s3_c0",
  "name":  {"value": "Jane Doe", "source": "pdfplumber", "pages": [3], "confidence": 1.0},
  "email": {"value": "jdoe@campus.edu", "source": "pdftotext", "pages": [3], "confidence": 0.8, "method": "name_pattern"}
}
```

Canonical confidence scale (must document):

- `1.0` — direct table extraction (pdfplumber structured)
- `0.8` — name-pattern match
- `0.6` — proximity match
- `0.4` — local-part / derivation synthesis
- `0.2` — OCR-only fallback

### `data-dart-*` HTML emission

On every renderable block in `generate_html_from_synthesized`:

- `data-dart-block-id="s3_c0"` — matches JSON sidecar block_id
- `data-dart-source="pdfplumber"` — primary source
- `data-dart-sources="pdfplumber,pdftotext"` — comma-joined when multi-source
- `data-dart-pages="3-4"` — page span (only when real page tracking available)
- `data-dart-confidence="0.87"` — 2-decimal precision
- `data-dart-strategy="..."` — optional free-form

Placement: `<section>`, `<div class="contact-card">`, `<tr>`, `<p>` (when block-level). Scoped to section-level by default; block-level opt-in via env flag to control HTML size.

### Staging handoff fix

`MCP/tools/pipeline_tools.py::stage_dart_outputs` currently copies HTML + `_synthesized.json`; extend to also copy `*.quality.json`. Manifest (L291) gains role-tagged entries:

```json
{"files": [
  {"path": "science_of_learning.html", "role": "content"},
  {"path": "science_of_learning_synthesized.json", "role": "provenance_sidecar"},
  {"path": "science_of_learning.quality.json", "role": "quality_sidecar"}
]}
```

### Validator extension

`lib/validators/dart_markers.py` — add warning-level checks for `data-dart-source` + `data-dart-block-id` presence. Promote to critical in Wave 9 once emission is stable.

### DART files touched (Wave 8)

- `DART/multi_source_interpreter.py` — matcher return shape, synthesis record shape, renderer
- `DART/pdf_converter/claude_processor.py` — add minimal `data-dart-source="claude_llm"` on legacy path (don't let it drift further)
- `MCP/tools/pipeline_tools.py::stage_dart_outputs` — copy quality sidecar, role-tag manifest
- `lib/validators/dart_markers.py` — warning-level provenance checks
- `schemas/knowledge/source_reference.schema.json` — **NEW** canonical shape

### DART risks

- **Form-feed stripping in `clean_text` (L116)** prevents real page tracking. Wave 8 ships with section-level `page_range` from context (fixture-based, rough); real per-block `pages` requires a separate refactor. Model the schema for pages, emit empty when we can't determine.
- **OCR-only docs** score low confidence even when content is correct. Add `ocr_quality` sub-signal (Tesseract per-word confidence) in a follow-up.
- **Legacy `claude_processor` divergence** is accepted — it gets a minimal `source="claude_llm"` stamp; full parity is non-goal.

## Courseforge changes (Wave 9)

### New `source_mapping` phase

Insert in `config/workflows.yaml` between `objective_extraction` and `course_planning`:

```yaml
- name: source_mapping
  agents: [source-router]
  parallel: false
  depends_on: [objective_extraction]
  timeout_minutes: 30
  inputs_from:
    - {param: project_id,              source: phase_outputs, phase: objective_extraction, output: project_id}
    - {param: staging_dir,             source: phase_outputs, phase: staging,               output: staging_dir}
    - {param: textbook_structure_path, source: phase_outputs, phase: objective_extraction, output: textbook_structure_path}
  outputs: [source_module_map_path, source_chunk_ids]
```

Then `course_planning.inputs_from` gains `source_module_map_path`, and critically, **`content_generation.inputs_from`** (today just `project_id`) gains `source_module_map_path` + `staging_dir`.

### `source-router` agent spec (new)

New file: `Courseforge/agents/source-router.md`. Responsibilities:

- Read the staging dir's manifest → know what DART sources are available.
- Read the textbook structure JSON → know what chapters/sections exist.
- Read the course outline (weeks, module types) — received from objective-synthesizer's output.
- For each (week, module_type, page_id): identify primary + contributing source block IDs. Output `source_module_map.json`:

```jsonc
{
  "week_03": {
    "content_01": {
      "primary":      ["dart:science_of_learning#s5_p2"],
      "contributing": ["dart:science_of_learning#s4_p0", "dart:science_of_learning#s6_p1"],
      "confidence":   0.85
    },
    "application_01": { ... }
  }
}
```

Heuristic or LLM-assisted; tf-idf between objective keywords and source block text is a reasonable start. Confidence surfaces to JSON-LD downstream.

### Schema additions

`schemas/knowledge/courseforge_jsonld_v1.schema.json`:

- Open `additionalProperties: false` on root and `$defs/Section` (both currently strict).
- Add optional `sourceReferences` (array of `SourceReference` via `$ref`) at page level (after `prerequisitePages`, L59).
- Add optional `sourceReferences` inside `Section.properties`.

### Emitter changes

`Courseforge/scripts/generate_course.py`:

- Load `source_module_map.json` alongside `prerequisite_map` (mirror pattern at L911-913).
- `_build_page_metadata` (L622-661) — add `source_references` kwarg, emit into JSON-LD when non-empty.
- `_build_sections_metadata` (L595-619) — same for section-level.
- Six call-sites in `generate_week` (L724-845) — thread the kwarg through.
- Emit `data-cf-source-ids="..."` on `<section>` + heading + component-wrapper elements (same sites that carry `data-cf-content-type` / `data-cf-teaching-role`).
- **Remove** Worker W's leftover `data-cf-objectives-count` row from `schemas/ONTOLOGY.md:640` while we're touching these files.

### Agent prompt update (critical coupling)

`Courseforge/agents/content-generator.md`:

- Add new "Source Material" section to the prompt template (around L415-448).
- Agent receives a curated slice of DART source chunks (only those mapped to its page).
- Agent cites specific source block IDs in output `data-cf-source-ids` and JSON-LD.
- **This MUST ship in the same PR as the schema change.** Courseforge audit flagged: emit-side schema alone is *worse than today* — model will invent plausible chapter IDs.

### Validator

New file: `lib/validators/source_refs.py`. Check that every `sourceId` emitted on a page resolves against the staging manifest. Critical severity (fail packaging on bad refs). Gate: `source_refs` on `content_generation` phase in workflows.yaml.

### Courseforge files touched (Wave 9)

- `config/workflows.yaml` — new `source_mapping` phase; `content_generation.inputs_from` expansion; `source_refs` gate
- `schemas/knowledge/courseforge_jsonld_v1.schema.json` — open root + Section; add `sourceReferences`
- `Courseforge/agents/source-router.md` — **NEW** agent spec
- `Courseforge/agents/content-generator.md` — prompt update
- `Courseforge/scripts/generate_course.py` — emit sites
- `lib/validators/source_refs.py` — **NEW** validator
- `schemas/ONTOLOGY.md` — remove stale `data-cf-objectives-count` row; add `data-cf-source-ids` + `data-cf-source-primary` rows
- Tests: Courseforge run with empty source map (graceful fallback) + populated source map (round-trip)

### Courseforge risks

- **Bad routing poisons downstream** — mitigated by confidence scores + validator + (optional) `min_confidence` soft gate.
- **JSON-LD size inflation** — dedup via page-level emit + override at section-level only when different.
- **`course_generation` workflow** (non-textbook, pure objectives) has no DART — `sourceReferences` must stay optional; empty array on pure-LO courses.
- **Agent hallucination without prompt change** — schema change + prompt change are coupled; ship together or not at all.

## Trainforge changes (Waves 7C + 7D)

### Wave 10 — chunk + node propagation (unflagged, additive)

`schemas/knowledge/chunk_v4.schema.json`:

- Open `$defs/Source` to add optional `source_references: array of SourceReference` (via `$ref` to shared schema).
- Keep `Source.additionalProperties: false` — declared explicitly.

`schemas/knowledge/concept_graph_semantic.schema.json`:

- Add optional `source_refs` (array of `SourceReference`) on node schema.
- Populate from `occurrences[0]` → that chunk's `source.source_references[0]` at `_build_tag_graph` emit time.

`Trainforge/parsers/html_content_parser.py`:

- `_extract_json_ld` (L276) — already captures full JSON-LD dict; no change needed to capture `sourceReferences`.
- `_extract_sections` (L303-373) — add regex for `data-cf-source-ids="..."`; store on new `ContentSection.source_references: List[str]` field.
- `ParsedHTMLModule` — add `source_references: List[Dict]` aggregated from sections + page-level JSON-LD.

`Trainforge/process_course.py`:

- Thread `parsed.metadata["courseforge"].sourceReferences` through `_chunk_content` (L1008) into per-item dict (L965-992).
- `_create_chunk` (L1271) — fold into `source.source_references`.
- `_merge_small_sections` (L1111) — aggregate from all merged sections (array, not single value).
- `_build_tag_graph` (L2215) — populate node `source_refs` from first occurrence's chunk.

### Wave 11 — evidence arm enrichment (flag-gated)

New opt-in flag: `TRAINFORGE_SOURCE_PROVENANCE` (documented in root CLAUDE.md flag table).

When ON, every inference rule with a `chunk_id` in its evidence arm **also** copies `source_references` from that chunk. Schema changes to `concept_graph_semantic.schema.json` $defs:

| Arm | Change |
|---|---|
| `IsAEvidence` | +optional `source_references[]` |
| `ExemplifiesEvidence` | +optional `source_references[]` |
| `DerivedFromObjectiveEvidence` | +optional `source_references[]` |
| `DefinedByEvidence` | +optional `source_references[]` |
| `AssessesEvidence` | +optional `source_references[]` (complements existing optional `source_chunk_id`) |
| `PrerequisiteEvidence` | +optional `target_chunk_id`, `source_chunk_id`, `source_references[]` (chunk IDs already tracked in `_first_positions_by_concept`; just thread through) |
| `RelatedEvidence` | +optional `source_chunks[]`, `source_references[]` — **requires re-plumbing `related_from_cooccurrence.py` to see chunks** (today discards them at L50). Larger refactor; could defer to 7E if 7D gets tight. |
| `MisconceptionOfEvidence` | +optional `source_chunk_id`, `source_references[]` |

**Bump `RULE_VERSION`** on every touched rule file. `rule_versions` map in `typed_edge_inference.py:356` auto-exposes the shift.

### Archival preservation

`MCP/tools/pipeline_tools.py::archive_to_libv2` (L419-570) uses `shutil.copy2` — byte-level, no transformation. Source refs survive unchanged.

Add advisory flag to archive manifest:

```json
{"features": {"source_provenance": true}}
```

So LibV2 retrieval callers can fast-skip source-grounded queries on legacy corpora.

### Trainforge files touched

Wave 10:
- `schemas/knowledge/chunk_v4.schema.json` — Source block extension
- `schemas/knowledge/concept_graph_semantic.schema.json` — node `source_refs[]`
- `Trainforge/parsers/html_content_parser.py` — `data-cf-source-ids` regex, ContentSection field
- `Trainforge/process_course.py` — thread through chunker + graph builder
- Tests: round-trip IMSCC with source refs → chunks + graph carry them

Wave 11:
- `schemas/knowledge/concept_graph_semantic.schema.json` — 8 evidence arm extensions
- `Trainforge/rag/inference_rules/*.py` — 8 rule files, RULE_VERSION bumps
- `Trainforge/rag/typed_edge_inference.py` — confirm version tracking still works
- Root `CLAUDE.md` — new flag row in opt-in table
- `MCP/tools/pipeline_tools.py::archive_to_libv2` — feature flag in manifest
- Tests: strict + lenient mode with flag on/off

### Trainforge risks

- **Source sub-schema strictness** — additions must be declared in-schema; fine as long as we keep the additive discipline.
- **Merged-section aggregation** requires array shape from the start — baked in.
- **`related_from_cooccurrence` re-plumbing** is the largest single piece of work in 7D; consider splitting to 7E if 7D estimates blow.
- **Chunk ID stability** — source refs only stay valid across re-runs under `TRAINFORGE_CONTENT_HASH_IDS=true`. Document this dependency; don't flip content-hash default yet.
- **Legacy corpora** have no source refs — consumers treat absence as "unknown", not error.

## Opt-in flag strategy

| Change | Flag | Default |
|---|---|---|
| DART per-block emit | none (always on) | emit |
| DART `data-dart-*` HTML attributes | none | emit |
| `*.quality.json` staging | none | copy |
| Courseforge `source_mapping` phase | none (but empty map = fallback) | run |
| Courseforge `sourceReferences` JSON-LD | none (optional field) | emit when map is populated |
| Courseforge `data-cf-source-ids` | none | emit |
| Trainforge chunk `source.source_references` | none (optional field) | carry-through |
| Trainforge node `source_refs[]` | none (optional field) | populate |
| **Trainforge evidence-arm enrichment** | **`TRAINFORGE_SOURCE_PROVENANCE`** | **off** |

Rationale: only the evidence-arm enrichment needs gating because every arm has `additionalProperties: false` under `TRAINFORGE_STRICT_EVIDENCE=true`. Everything else is additive-optional and degrades gracefully on absence.

## Wave structure

All waves land on `dev-v0.2.0` via PRs (matching Waves 1–6 pattern). One worker per wave, worktree-isolated (Step 0 guardrail from Wave 5+).

```
Wave 8   DART + shared schema        (1 worker; foundation; depends on Wave 7 orchestration merge)
   │     schemas/knowledge/source_reference.schema.json
   │     DART emission + staging + data-dart-* attributes
   │     lib/validators/dart_markers.py (warning-level provenance)
   │     Legacy claude_processor: minimal data-dart-source="claude_llm" stamp only (P5 decision)
   │
Wave 9   Courseforge (1 worker; depends on Wave 8 merge)
   │     source_mapping phase + source-router agent
   │     content-generator agent prompt update (CRITICAL — ships with schema change)
   │     schemas/knowledge/courseforge_jsonld_v1.schema.json extension
   │     generate_course.py emit sites
   │     lib/validators/source_refs.py
   │     data-cf-source-ids HTML attributes — section/heading/component only (P2 decision)
   │
Wave 10  Trainforge chunk + node (1 worker; depends on Wave 9 merge)
   │     chunk_v4 Source.source_references[]
   │     concept_graph_semantic node.source_refs[]
   │     html_content_parser + process_course.py propagation
   │
Wave 11  Trainforge evidence-arm enrichment (1 worker; depends on Wave 10; flag-gated)
         TRAINFORGE_SOURCE_PROVENANCE
         5 evidence arm extensions (IsA, Exemplifies, DerivedFromObjective, DefinedBy, Assesses) + RULE_VERSION bumps (P4 decision)
         archive_to_libv2 manifest feature flag
         Prerequisite/Related/MisconceptionOf deferred to Wave 12 (related_from_cooccurrence re-plumbing cost)
```

Four sequential waves, one worker each. Why sequential not parallel: each wave's schema extension is a downstream dependency for the next. Parallelizing would create merge conflicts on shared schema files.

## Migration

- Zero-migration for existing LibV2 corpora. All new fields are optional; absence = "unknown".
- Archive manifest `features.source_provenance: true` flag lets LibV2 retrieval callers distinguish enriched vs. legacy corpora.
- Re-running any course through the full pipeline after Wave 11 populates source refs end-to-end.
- Opt-in flag defaults stay OFF through this wave series; flip-to-default-on is a separate future decision once legacy corpora regenerate.

## Decisions (confirmed)

All locked in. See `plans/pipeline-orchestration/design.md` for orchestration decisions O1–O6.

| # | Decision | Confirmed |
|---|---|---|
| P1 | DART confidence scale | 5-value float (1.0/0.8/0.6/0.4/0.2) with documented semantics |
| P2 | Block-level `data-cf-source-ids` scope | `<section>` + headings + component wrappers only; not `<p>`/`<li>`/`<tr>` |
| P3 | Multi-source shape | `SourceReference[]` with required `role` enum + optional weight/confidence |
| P4 | Abstract evidence arm enrichment | Enrich 4 chunk-anchored + Assesses in Wave 11; defer Prerequisite/Related/MisconceptionOf to Wave 12 |
| P5 | Legacy `claude_processor` parity | Minimal `data-dart-source="claude_llm"` stamp only; no deeper parity |

## Explicit non-goals

- No changes to the existing 8 edge types or the Wave 5 occurrences[] design.
- No rewrite of existing LibV2 corpora.
- No flag default flips during this wave series.
- No cross-course source-ref merging (cross-course concept aliases remains deferred).
- No real-time provenance display in LMS (separate UI work).
- No pipeline run until all four waves land.

## Ready-to-go checklist before implementation

- [ ] User answers the 7 decision questions above.
- [ ] Branch target confirmed.
- [ ] Worker prompts drafted per wave (with worktree isolation Step 0 guardrail).
- [ ] Test corpus identified for validation (Deans for Impact 12pp is still the safest first test once all waves land).

## Wave landing log

- [x] **Wave 8 — DART + shared schema** (worker-a7c7f9d4): `schemas/knowledge/source_reference.schema.json` (new), DART per-section `provenance` block + per-block envelopes, `data-dart-*` HTML attributes on `<section>` + `.contact-card`, legacy `claude_processor` minimal stamp, `stage_dart_outputs` copies `*.quality.json` + role-tagged manifest, `DartMarkersValidator` warning-level provenance checks, DART/CLAUDE.md + root CLAUDE.md doc updates.
- [x] **Wave 9 — Courseforge source attribution** (worker-aa): `source_mapping` phase + `source-router` agent, `courseforge_jsonld_v1.schema.json` extended with page + section `sourceReferences`, `generate_course.py` emits refs into JSON-LD + `data-cf-source-ids`, `lib.validators.source_refs.PageSourceRefValidator` gate on `content_generation`, `content-generator` prompt updated, `DartMarkersValidator` promoted to critical on malformed attrs.
- [x] **Wave 10 — Trainforge chunk + node** (worker-bb): `chunk_v4.schema.json` `Source` gains optional `source_references[]`; `concept_graph_semantic.schema.json` node gains `source_refs[]`; parser + chunker + graph-builder threaded through; `archive_to_libv2` advisory `features.source_provenance` flag.
- [x] **Wave 11 — Trainforge evidence-arm enrichment** (worker-cc): `concept_graph_semantic.schema.json` — five chunk-anchored evidence arms (`IsAEvidence`, `ExemplifiesEvidence`, `DerivedFromObjectiveEvidence`, `DefinedByEvidence`, `AssessesEvidence`) gain optional `source_references[]`. Three abstract arms (`PrerequisiteEvidence`, `RelatedEvidence`, `MisconceptionOfEvidence`) deferred per P4. Rule modules bump `RULE_VERSION` 1 -> 2 unconditionally; emit is flag-gated behind new `TRAINFORGE_SOURCE_PROVENANCE` env var. `archive_to_libv2` gains companion `features.evidence_source_provenance` advisory flag. Root + Trainforge CLAUDE.md + ONTOLOGY.md § 12 updated.
