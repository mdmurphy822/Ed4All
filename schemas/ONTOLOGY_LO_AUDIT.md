# Learning Objective Storage Audit (Wave 80 diagnostic)

> **Scope.** Diagnostic catalog of every place a learning objective
> (LO) lives across Ed4All. Field shapes below were read directly
> from the live `rdf-shacl-550` archive. **No code changes.**

## TL;DR

- **10 storage surfaces** carry LO identity / metadata.
- **6 independent on-disk schemas** + 2 reference-only surfaces +
  1 evidence-only surface.
- **8 drift points cataloged** (case, field-name, key-name).
- **Wave 81 actions:** (1) freeze a canonical schema at
  `schemas/knowledge/learning_objective_v1.schema.json`; (2) route
  all emitters through `lib/ontology/learning_objectives.py::
  to_canonical_lo()`; (3) lowercase IDs on-disk everywhere.

## Storage surfaces

Live-archive paths for the audit subject:

```
Courseforge: Courseforge/exports/PROJ-RDF_SHACL_550-20260424135037/
LibV2:       LibV2/courses/rdf-shacl-550-rdf-shacl-550/
Wave 79:     /tmp/wave79_smoke/instruction_pairs.jsonl
```

### 1. `synthesized_objectives.json` (Courseforge)

- **Path glob:** `Courseforge/exports/PROJ-*/01_learning_objectives/synthesized_objectives.json`
- **Emitter:** `course-outliner` subagent → `plan_course_structure`
  phase (`MCP/tools/pipeline_tools.py`).
- **Reader(s):** Trainforge `process_course.py::load_objectives`;
  LibV2 archivist (mirrors → `objectives.json`).
- **Top keys:** `course_name`, `mint_method`, `duration_weeks`,
  `synthesized_at`, `source_corpus[]`, `terminal_objectives[]`,
  `chapter_objectives[]`, `learning_outcomes[]` (denormalized
  terminal subset), `bloom_distribution`.
- **ID format:** UPPERCASE `TO-NN` / `CO-NN`.
- **Parent link:** `parent_to` → terminal id (CO only).
- **Bloom shape:** flat `bloom_level`, `bloom_verb`, `cognitive_domain`.
- **Version field:** none; only `mint_method:
  "subagent_course_outliner_v1"`.
- **Per-LO fields:** `{id, text, statement, bloom_level, bloom_verb,
  cognitive_domain, weeks[]|week, source_refs[]?, parent_to?}`
  (`text` and `statement` duplicate the same string).

### 2. `textbook_structure.json` (Courseforge)

- **Path glob:** `Courseforge/exports/PROJ-*/01_learning_objectives/textbook_structure.json`
- **Emitter:** `extract_textbook_structure` phase
  (`SemanticStructureExtractor`).
- **Reader(s):** `plan_course_structure`.
- **Schema:** chapter hierarchy carries pre-LO **explicit** objectives
  mined from textbook prose, **not** canonical `TO/CO` IDs. Each
  chapter: `{id, headingLevel, headingText, headingId,
  explicitObjectives[], contentBlocks[], sections[], source_file}`.
  `explicitObjectives[]` items are `{text, source: "inline"}` (no ID).
- **Live counts:** 322 chapters, 14 with non-empty `explicitObjectives`.
- **Version field:** none.

### 3. `learning_outcomes_synthesis.md` (Courseforge)

- **Path glob:** `Courseforge/exports/PROJ-*/01_learning_objectives/learning_outcomes_synthesis.md`
- **Status:** **NOT EMITTED** by the live Wave-78 pipeline. The
  audited `01_learning_objectives/` directory contains only the two
  JSON files above. **Phantom surface — referenced by docs/task
  briefs, no emitter produces it.**

### 4. Per-page Courseforge HTML JSON-LD

- **Path glob:** `Courseforge/exports/PROJ-*/03_content_development/week_*/*.html`
- **Emitter:** `Courseforge/scripts/generate_course.py`.
- **Reader(s):** Trainforge `assessment-extractor` (JSON-LD >
  `data-cf-*` > `data-dart-*` > regex).
- **Schema:** `<script type="application/ld+json">` with `@context:
  https://ed4all.dev/ns/courseforge/v1`, `@type: CourseModule`.
  `learningObjectives[]` entries are **camelCase**: `{id, statement,
  bloomLevel, bloomVerb, cognitiveDomain, hierarchyLevel,
  assessmentSuggestions[]}`. `hierarchyLevel` enum: `terminal | chapter`.
- **ID format:** UPPERCASE `TO-NN` / `CO-NN`.
- **Spot-check:** `week_01_overview.html` carries `TO-01` +
  `CO-01..CO-06`; `week_01_content_01.html` carries `CO-01`.
- **Parent link:** none in JSON-LD — flat list.

### 5. `LibV2/courses/*/objectives.json` (Wave 75 archive)

- **Emitter:** `libv2-archivist` subagent during `libv2_archival`.
- **Reader(s):** Trainforge `process_course.py::load_objectives`;
  LibV2 retrieval scoring; `libv2_manifest` gate.
- **Top keys:** `schema_version`, `course_code`,
  `terminal_outcomes[]`, `component_objectives[]`, `objective_count`.
- **ID format:** **lowercase** `to-nn` / `co-nn` — drift vs. (1).
- **Parent link:** `parent_terminal` (CO → TO) — drift vs. (1)'s `parent_to`.
- **Per-LO fields:** `{id, statement, bloom_level, bloom_verb,
  cognitive_domain, weeks[]|week, parent_terminal?, source_refs[]?}`.
- **Version field:** `schema_version: "v1"`. Schema:
  `schemas/knowledge/objectives_v1.schema.json`.

### 6. `LibV2/courses/*/course.json`

- **Emitter:** Trainforge `process_course.py::_build_course_json`.
- **Reader(s):** `LibV2/tools/libv2/retrieval_scoring.py::load_course_outcomes`;
  `LibV2/tools/libv2/validator.py::validate_learning_outcomes`.
- **Top keys:** `course_code`, `title`, `learning_outcomes[]`.
- **Per-LO shape:** flat, terminal-first. `{id, statement,
  hierarchy_level, bloom_level}`. Component entries additionally
  carry `type: "component"` (legacy discriminator preserved alongside
  newer `hierarchy_level` enum — drift point D).
- **ID format:** **lowercase** `to-nn` / `co-nn`.
- **Parent link:** none on this surface (lives only on `objectives.json`).
- **Version field:** none on file (pinned by
  `schemas/knowledge/course.schema.json`).

### 7. `LibV2/courses/*/corpus/chunks.jsonl`

- **Emitter:** Trainforge `process_course.py` chunker.
- **Reader(s):** LibV2 retrieval; `assessment_objective_alignment`;
  Wave 79 instruction-pair extractor.
- **LO storage:** `learning_outcome_refs[]` per chunk (flat ID list,
  no metadata).
- **ID format:** **lowercase**. Live sample (line 1): `["co-01",
  "co-02", "co-03", "to-01"]`.
- **Cross-validation:** matches `objectives.json` ID case → OK.
- **Version field:** chunk-level `schema_version` (currently `chunk_v4`).

### 8. `LibV2/courses/*/graph/pedagogy_graph.json`

- **Emitter:** Trainforge pedagogy_graph builder.
- **Reader(s):** LibV2 retrieval; pedagogy validators; Wave 71 typed
  edge boost.
- **Top keys:** `kind`, `schema_version: "v2"`, `course_id`, `nodes[]`,
  `edges[]`, `stats`, `generated_at`.
- **Node classes (live counts):** `BloomLevel`(6),
  `DifficultyLevel`(3), `Outcome`(7), `ComponentObjective`(29),
  `Module`(12), `Chunk`(219), `Concept`(599), `Misconception`(67).
- **`Outcome`:** `{id, class, label, statement, bloom_level}`.
- **`ComponentObjective`:** `{id, class, label, statement,
  bloom_level, parent_terminal, week}`.
- **Edge `relation_type` enum (live counts):** `supports_outcome`(29),
  `at_bloom_level`(36), `follows`(11), `belongs_to_module`(219),
  `practices`(40), `derived_from_objective`(734) — **snake_case** not
  `derived-from-objective`, `chunk_at_difficulty`(219), `teaches`(623),
  `exemplifies`(159), `assesses`(71),
  `assessment_validates_outcome`(17), `interferes_with`(437),
  `concept_supports_outcome`(951), `prerequisite_of`(703).
- **ID format:** **UPPERCASE** `TO-NN` / `CO-NN` (drift vs. (5)/(6)/(7)).
- **Version field:** `schema_version: "v2"`.

### 9. `LibV2/courses/*/graph/concept_graph_semantic.json`

- **Emitter:** Trainforge concept-graph-semantic builder.
- **Reader(s):** LibV2 retrieval; Wave 71 boost.
- **Top keys:** `kind: "concept_semantic"`, `generated_at`,
  `rule_versions{}`, `nodes[]`, `edges[]`.
- **LO presence:** **not in node IDs**. Concept nodes are slugified
  domain terms (`rdfxml`, `named-graph`). LO IDs appear **only inside
  `edge.provenance.evidence`** for rules `prerequisite_from_lo_order`,
  `targets_concept_from_lo`, `derived_from_lo_ref` — all **lowercase**.
- **Edge types (live counts):** `prerequisite`(43), `related-to`(110).
  The expected `derived-from-objective` (per `Trainforge/CLAUDE.md`
  §195) does **not** appear here — drift point E.
- **Version field:** per-rule `rule_versions{}` ints (no top-level).

### 10. Trainforge Wave 79 instruction pairs

- **Path glob:** `/tmp/wave79_*/instruction_pairs.jsonl`.
- **Emitter:** Trainforge instruction-pair extractor (Wave 79).
- **Reader(s):** Wave 79 export; downstream SFT trainers.
- **LO storage:** `lo_refs[]` per pair (NOT `objective_ids[]` as
  speculated in the task brief — drift point F).
- **Sample:** `{bloom_level: "remember", chunk_id:
  "rdf_shacl_550_chunk_00012", completion, content_type:
  "assessment_item", decision_capture_id, lo_refs:
  ["co-01","co-02","co-03","to-01"], prompt, provider, schema_version:
  "v1", seed, template_id}`.
- **ID format:** **lowercase** (matches chunks).
- **Version field:** `schema_version: "v1"`.

## Schema drift points

| # | Surface A | Surface B | Drift |
|---|-----------|-----------|-------|
| A | `synthesized_objectives.json` | `objectives.json` (Wave 75) | `terminal_objectives` ↔ `terminal_outcomes`; `chapter_objectives` ↔ `component_objectives` |
| B | `synthesized_objectives.json` | `objectives.json` | `parent_to` ↔ `parent_terminal` (CO → TO back-pointer) |
| C | Courseforge (synth, JSON-LD, pedagogy_graph) | LibV2 (`objectives.json`, `course.json`, `chunks.jsonl`, `lo_refs`) | UPPERCASE ↔ lowercase. Mitigated by case-insensitive readers + `TRAINFORGE_PRESERVE_LO_CASE`, but on-disk drift persists. |
| D | `course.json` carries `hierarchy_level: "chapter"` AND legacy `type: "component"` | `objectives.json` segments by section, no flat list | Redundant CO discriminators (post-Wave-75 transitional). |
| E | `Trainforge/CLAUDE.md` §195 documents `derived-from-objective` (kebab) | `pedagogy_graph.json` emits `derived_from_objective` (snake); `concept_graph_semantic.json` does not emit it | Kebab vs. snake; docs vs. emit attribution drift. |
| F | Task brief speculated `objective_ids[]` on Wave 79 pairs | Live emit uses `lo_refs[]` | Field is `lo_refs` (lowercase). |
| G | JSON-LD camelCase (`bloomLevel`, `bloomVerb`, etc.) | All other surfaces snake_case | camelCase ↔ snake_case at HTML/JSON boundary. |
| H | `synthesized_objectives.json` duplicates `text` + `statement` | Other surfaces use `statement` only | Synth-side legacy duplication. |

## Canonical schema proposal (Wave 81)

Recommended canonical shape, lifted from
`schemas/knowledge/objectives_v1.schema.json` and extended:

```jsonc
{
  "id": "to-01",                     // ^[a-z]{2,}-\d{2,}$
  "statement": "Analyze RDF data...", // single field, no `text` alias
  "hierarchy_level": "terminal",      // terminal | chapter
  "parent_terminal": null,            // null | string (chapter only)
  "bloom_level": "analyze",
  "bloom_verb": "analyze",
  "cognitive_domain": "conceptual",
  "weeks": [1, 2],                    // terminal only
  "week": null,                       // chapter only
  "source_refs": ["dart:owl2_primer_accessible#s1"],  // optional
  "version_hash": "sha256:...",       // content-addressable LO id
  "created_at": "2026-04-24T13:55:00Z"
}
```

**Single emitter shim.** All 6 emitters route through
`lib/ontology/learning_objectives.py::to_canonical_lo()` which (1)
lowercases ID, (2) picks `statement` over `text`, (3) coerces
`parent_to`/`parentTerminal` → `parent_terminal`, (4) emits
snake_case + a frozen camelCase view for JSON-LD, (5) computes
`version_hash`.

**Migration.** Emit canonical alongside legacy for one release;
flip readers; deprecate legacy keys. One-shot `libv2 migrate-los`
backfills `version_hash` on archived courses.

**ID-case policy.** Lowercase becomes the on-disk canonical
(already true for 4 of 6 surfaces). UPPERCASE is acceptable only at
the rendering boundary (HTML labels, JSON-LD pretty output gated by
`TRAINFORGE_PRESERVE_LO_CASE`). Pedagogy_graph node IDs flip
lowercase under Wave 81.

## Who blocks what

- **`chunks.jsonl` ↔ `objectives.json`** cross-validation requires
  case-insensitive normalization (Wave 76 A's fix in
  `lib/validators/assessment_objective_alignment.py`).
- **Wave 79 pairs** use `lo_refs[]`; consumers expecting
  `objective_ids[]` (per task brief) need a normalization shim.
- **`process_course.load_objectives`** accepts both
  `terminal_objectives` and `terminal_outcomes` — dual-shape
  acceptance baked into a critical reader.
- **Pedagogy_graph node IDs are UPPERCASE**, every other Trainforge
  reference is lowercase → traversers must `.upper()` going chunks →
  graph or `.lower()` going back. Easy to mis-wire.
- **`derived-from-objective` docs** (kebab in `Trainforge/CLAUDE.md`)
  do not match on-disk `derived_from_objective` (snake) in
  `pedagogy_graph.json`. Grep for the kebab form silently misses the
  edges.

## Recommended Wave 81 work

1. **Freeze a canonical LO schema** at
   `schemas/knowledge/learning_objective_v1.schema.json`; have
   `objectives_v1.schema.json` `$ref` it; validate at every emit.
2. **Single emitter helper:** add
   `lib/ontology/learning_objectives.py::to_canonical_lo()` and
   migrate all 6 emitters (synth, JSON-LD, archivist, pedagogy_graph,
   chunks normalizer, instruction-pair) to call it. Regression tests
   assert no emitter writes a non-canonical key.
3. **Lowercase IDs on disk everywhere** — flip pedagogy_graph node
   IDs to lowercase; drop `parent_to` for `parent_terminal` on synth.
4. **Kill the phantom `learning_outcomes_synthesis.md`** in docs OR
   add an emitter — pick one.
5. **Reconcile `derived-from-objective` naming** — pick kebab or
   snake and migrate `Trainforge/CLAUDE.md` + tests + emit to match.
6. **JSON-LD camelCase view:** generate camelCase mechanically from
   the canonical snake_case via `to_jsonld_lo()`. No hand-typed
   camelCase in any emitter.
7. **Add `version_hash`** (sha256 of statement + `bloom_level`) per
   LO so downstream artifacts pin exact versions and detect edits.
8. **Promote `objective_count`** to a canonical manifest field so
   `course.json` and `objectives.json` agree without a join.

---

*Audit subject: `rdf-shacl-550-rdf-shacl-550` (built 2026-04-24).
Every field example was verified against an actual file on disk.*
