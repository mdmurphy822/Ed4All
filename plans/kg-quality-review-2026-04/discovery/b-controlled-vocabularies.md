# Worker B — Controlled-vocabulary audit

## Summary

- **8 vocabularies audited** across Courseforge → JSON-LD / data-cf-* → Trainforge → LibV2 → schemas.
- **Schema coverage: 2 of 8 have a formal enum.** `assessment_type` (split across two schemas) and `question_type` (split across two schemas) are enumerated — both with drift. The remaining 6 (`cognitive_domain`, `content_type`, `teaching_role`, `module_type`, `BLOOM_CHUNK_STRATEGY`, the Trainforge-extension `decision_type` values) are code-only.
- **Top-3 drift risks ranked by KG-impact** (join reliability first):
  1. `content_type` — free-string across 4 surfaces (`data-cf-content-type`, JSON-LD `sections[].contentType`, chunk `content_type_label`, `instruction_pair.content_type`), no enum, 13+ distinct values observed. Degrades *every* chunk-type filter (LibV2 `ChunkFilter.content_type_label`, Trainforge template selection `(bloom, content_type)` lookups, BLOOM_CHUNK_STRATEGY retrieval). This is the highest-leverage KG-impact item in this audit.
  2. `assessment_type` / `question_type` — four divergent enums (LO schema 9, course_metadata 7, `question_factory.VALID_TYPES` 7, `trainforge_decision.question_type` 8; plus CF JSON-LD `assessmentSuggestions` free-list picking from a 6-value implicit set). A `Question` node typed `multiple_response` by the factory fails validation against the `trainforge_decision` event enum (D-review Claim 5 corrected direction). Breaks type-based question joins and assessment-method KG edges.
  3. `cognitive_domain` — emitted on every Courseforge objective (`data-cf-cognitive-domain`, JSON-LD `cognitiveDomain`), consumed by Trainforge (`html_content_parser.py:327,357`), **but no schema enumerates the 4 values**; mapping to Bloom lives only in `BLOOM_TO_DOMAIN` at `generate_course.py:146`. Courseforge decides per emit with `.get(bloom_level, "conceptual")`, so any Bloom-level drift (Worker A) silently reroutes the cognitive-domain KG edge.

No wholly new drift classes were surfaced beyond the prior-review catalogue; this audit tightens the file:line evidence and makes explicit that `teaching_role`, `module_type`, and `BLOOM_CHUNK_STRATEGY` have **zero** schema presence (not just "absent enum" — the terms themselves never appear in `schemas/`).

## Per-vocabulary

### 1. cognitive_domain

- **Source of truth**: `Courseforge/scripts/generate_course.py:146` (`BLOOM_TO_DOMAIN` dict, 6 Bloom keys → 4 domain values).
- **Producers**:
  - `Courseforge/scripts/generate_course.py:313` — `data-cf-cognitive-domain="{domain}"` on `<li>` objective items.
  - `Courseforge/scripts/generate_course.py:520,527` — JSON-LD `cognitiveDomain` field inside `_build_objectives_metadata`.
- **Consumers**:
  - `Trainforge/parsers/html_content_parser.py:37` — `LearningObjective.cognitive_domain` dataclass field.
  - `Trainforge/parsers/html_content_parser.py:327` — reads `lo.get("cognitiveDomain")` from JSON-LD.
  - `Trainforge/parsers/html_content_parser.py:357` — reads `data-cf-cognitive-domain` attribute.
  - `Trainforge/tests/test_metadata_extraction.py:42,51,189` — only test assertions; no downstream runtime joins.
- **Schematized**: **no**. The four values `{factual, conceptual, procedural, metacognitive}` do not appear in any `schemas/**/*.schema.json`. Only `schemas/taxonomies/pedagogy_framework.yaml:773,784` mentions "cognitive domain" as a search-term string for RAG, not as an enum.
- **Drift evidence**: Fallback default `"conceptual"` at `generate_course.py:313,520` silently absorbs any unmapped Bloom level. `BLOOM_TO_DOMAIN["create"]="procedural"` and `BLOOM_TO_DOMAIN["analyze"]="conceptual"` are pedagogically contestable and single-sourced; no validator checks.
- **KG-impact**: Objective → Domain edges are single-sourced from one code-level dict; a Bloom-classification disagreement (Worker A drift) silently reroutes the domain edge. Dedupe of objectives by "domain" is unreliable whenever Bloom inference diverges.

### 2. content_type (data-cf-content-type / JSON-LD sections[].contentType / chunk content_type_label / instruction_pair.content_type)

- **Source of truth**: **none canonical.** Three independent producer paths, one consumer filter surface, one schema field that declines to enumerate.
- **Producers**:
  - `Courseforge/scripts/generate_course.py:388–405` — `_infer_content_type()` returns one of 8 values: `definition, example, procedure, comparison, exercise, overview, summary, explanation`.
  - `Courseforge/scripts/generate_course.py:417,423` — emits `data-cf-content-type="{content_type}"` on h2/h3.
  - `Courseforge/scripts/generate_course.py:448,452` — callout emits `application-note` or `note` (two extra values not in `_infer_content_type`).
  - `Courseforge/scripts/generate_course.py:553,556` — JSON-LD `sections[].contentType` mirrors `_infer_content_type`.
  - `Courseforge/scripts/textbook-objective-generator/bloom_taxonomy_mapper.py:141–173` — `CONTENT_TYPE_DEFAULTS` with 20 keys (including `term, glossary, description, concept, how_to, relationship, structure, evaluation, criteria, judgment, design, solution, synthesis`) — a *disjoint* second producer vocabulary.
- **Consumers**:
  - `Trainforge/parsers/html_content_parser.py:26` — `ContentSection.content_type` field from `data-cf-content-type`.
  - `Trainforge/parsers/html_content_parser.py:292` — regex fallback reading `data-cf-content-type`.
  - `Trainforge/process_course.py:1099,1162,1220,1279,1300–1336` — sets `content_type_label` on chunk, with trace source `jsonld_section_match` / `data_cf_fallback` / `none_*`.
  - `Trainforge/generators/instruction_factory.py:48,183–201` — `(bloom, content_type)` keys `TEMPLATE_CATALOG`; falls back to `(bloom, "_default")` when content_type is unrecognised.
  - `LibV2/tools/libv2/retriever.py:62,439–440,565,623` — `ChunkFilter.content_type_label` equality filter (no enum guard).
  - `LibV2/tools/libv2/cli.py:378,388,417` — `--content-type` CLI flag passes through to retriever.
- **Schematized**: **partial + incoherent.**
  - `schemas/knowledge/instruction_pair.schema.json:13,46–48` requires `content_type` but as **free-string** (no enum).
  - `schemas/academic/course_metadata.schema.json:348–355` has a **different** `contentTypes` enum: `{lecture, reading, video, interactive, assignment, discussion, quiz, project}` — orthogonal pedagogical categories, not the same vocabulary.
  - No schema enumerates the Courseforge/Trainforge chunk-level content_type.
- **Drift evidence**: Minimum 13 distinct values in circulation across the two Courseforge producers (`definition, example, procedure, comparison, exercise, overview, summary, explanation, application-note, note, term, glossary, description, concept, how_to, relationship, structure, evaluation, criteria, judgment, design, solution, synthesis`); `instruction_factory._select_template` silently downgrades unknown values to `_default`; LibV2 retrieval filter is plain string-equality so `"application-note"` vs `"note"` vs `"example"` partitions the corpus unpredictably. Cites prior-review D1 / Claim 1 on Codex "free-string" finding.
- **KG-impact**: *Highest-leverage drift in this audit.* Every Chunk→ContentType KG edge is unconstrained; BLOOM_CHUNK_STRATEGY retrieval (§7) joins on a string that is produced by two disjoint mappers. Query expressivity ("give me all example chunks") cannot be guaranteed to recall; dedupe by content-type fails.

### 3. teaching_role

- **Source of truth**: `Trainforge/align_chunks.py:33` — `VALID_ROLES = {"introduce", "elaborate", "reinforce", "assess", "transfer", "synthesize"}`.
- **Producers**:
  - `Trainforge/align_chunks.py:487,502,522,572,578,582` — `classify_teaching_roles()` writes `chunk["teaching_role"]` (LLM or mock classifier).
  - `Trainforge/process_course.py:2737,2809` — invoked post-processing; field name "teaching_role" hardcoded.
- **Consumers**:
  - `LibV2/tools/libv2/retriever.py:61,435–436,564,622` — `ChunkFilter.teaching_role` equality filter.
  - `LibV2/tools/libv2/cli.py:377,387,416` — `--teaching-role` CLI flag.
  - `LibV2/tools/libv2/tests/test_retriever_v4.py:45–48` — filter tests.
  - Metrics only: `Trainforge/align_chunks.py:702,723,730,734–737,774–777,905` emit `teaching_role_coverage`, `teaching_role_consistency`, `teaching_role_distribution` for reporting.
- **Schematized**: **no**. Zero hits for `teaching_role|teachingRole|introduce|elaborate|reinforce|synthesize` as enum values in `schemas/**/*.schema.json`. `schemas/taxonomies/pedagogy_framework.yaml` and `taxonomy.json` contain the *words* but not as a formal chunk-field enum.
- **Drift evidence**: Courseforge emits neither `data-cf-teaching-role` nor JSON-LD `teachingRole`. Yet Courseforge *does* emit `data-cf-component ∈ {flip-card, self-check, activity}` (`generate_course.py:345,374,487`) and `data-cf-purpose ∈ {term-definition, formative-assessment, practice}` (same lines) — deterministic candidates for `teaching_role` classification that are currently ignored. Classifier at `align_chunks.py:572–582` uses `_mock_role(chunk, concept_first_seen)` or LLM, introducing nondeterminism where a rule would suffice.
- **KG-impact**: `teaching_role` is the LibV2 retrieval lever (filter-first, rank-second per ADR-002) but the producer and consumer live in different subsystems with no contract. A Chunk→TeachingRole KG edge is LLM-classified each run, so the same course's KG differs between runs; Courseforge's deterministic pedagogical signal (component, purpose) is dropped on the floor.

### 4. assessment_type

- **Source of truth**: **split, 4 divergent enums.**
- **Producers / schema definitions**:
  - `schemas/academic/learning_objectives.schema.json:219–225` — `assessmentSuggestions[]` enum (9 values): `{exam, quiz, assignment, project, discussion, presentation, portfolio, demonstration, case_study}`.
  - `schemas/academic/course_metadata.schema.json:365–368` — `assessments[].format` enum (5 values actually present: `{quiz, assignment, discussion, project, exam}`; prior review cited 7 but line range shows 5). Also `:363` `type ∈ {formative, summative}`.
  - `Trainforge/generators/question_factory.py:81–89` — `VALID_TYPES` list (7 values): `{multiple_choice, multiple_response, true_false, fill_in_blank, short_answer, essay, matching}`.
  - `schemas/events/trainforge_decision.schema.json:62–65` — `question_type` enum (8 values): `{multiple_choice, true_false, short_answer, essay, matching, fill_in_blank, ordering, hotspot}`.
  - `Courseforge/scripts/generate_course.py:534–541,593` — JSON-LD `assessmentSuggestions`/`suggestedAssessmentTypes` picks from implicit 5-value set `{multiple_choice, true_false, fill_in_blank, short_answer, essay}`.
- **Consumers**:
  - `Trainforge/parsers/html_content_parser.py:56` — `suggested_assessment_types: List[str]` (free-string field).
  - `Trainforge/generators/question_factory.py:91–98` — `BLOOM_QUESTION_MAP` hardcoded `(bloom → list[question_type])`.
- **Schematized**: **yes but four-way divergent.** No union defined. Cites prior-review Claim 5 (direction corrected).
- **Drift evidence**:
  - CF JSON-LD `assessmentSuggestions` values do not all appear in LO schema's 9-value enum (`multiple_choice, true_false, fill_in_blank` are not in LO schema's `{exam, quiz, assignment, project, discussion, presentation, portfolio, demonstration, case_study}`). CF emits question-*format* names into a LO-schema field that expects assessment-*method* names — category mismatch.
  - Factory emits `multiple_response`, which is not in `trainforge_decision.question_type` (8-value enum).
  - Decision-event enum has `ordering` and `hotspot`, which the factory never emits.
- **KG-impact**: A `Question` node produced by `question_factory.create_multiple_response` (`question_factory.py:175,199,222`) fails validation when logged as a `trainforge_decision` event; the event is either rejected or the value silently coerced. `Objective→SuggestedAssessment` edges emitted from CF JSON-LD are semantically incompatible with LO-schema consumers that expect method-level categories. Assessment-method KG queries (e.g. "show me all projects covering LO-03") cannot be answered reliably.
- **Proposed union (enumeration only, no recommendation)**: the minimal `question_type` superset appears to be `{multiple_choice, multiple_response, true_false, short_answer, essay, matching, fill_in_blank, ordering, hotspot}` (9 values). `assessment_type` (course-level method) vs `question_type` (item-level format) need separation — they are semantically different vocabularies conflated by the CF-emit path.

### 5. module_type

- **Source of truth**: `Courseforge/scripts/generate_course.py:572,584` — parameter `module_type: str` passed into `_build_page_metadata`, emitted as JSON-LD `moduleType`.
- **Producers (emit-call sites)**:
  - `generate_course.py:647–656` — `"overview"`.
  - `generate_course.py:667–677` — `"content"`.
  - `generate_course.py:685–695` — `"application"`.
  - `generate_course.py:703–713` — `"assessment"` (for self_check page; note self-check maps to the string `"assessment"`, not `"self-check"`).
  - `generate_course.py:728–738` — `"summary"`.
  - `generate_course.py:754–759` — `"discussion"` (sixth value, not in the 5-value enum the scope note cites).
- **Actual emitted vocabulary**: `{overview, content, application, assessment, summary, discussion}` — 6 values, not the 5 in `schemas/ONTOLOGY.md:108,625`. The ontology map has not caught `discussion`.
- **Consumers**:
  - `Trainforge/tests/test_metadata_extraction.py:34,150` — only. Asserts `moduleType="content"` round-trips through the parser.
  - `Trainforge/parsers/html_content_parser.py` — **no read**. Grep for `moduleType` in Trainforge returns only the test file above.
- **Schematized**: **no**. `schemas/ONTOLOGY.md` descriptively documents 5 values; no `.schema.json` enumerates it. `Courseforge/schemas/template-integration/educational_template_schema.json:294` mentions `moduleTypes` but is an unrelated template-library schema.
- **Drift evidence**: Cites prior-review D2. Newly surfaced here: the `discussion` sixth value is emitted but undocumented; the ontology-map enum is already stale relative to the emit path.
- **KG-impact**: Pages carry `moduleType` in JSON-LD but nothing downstream joins on it — a Module→Page KG edge cannot be partitioned by pedagogical role, so queries of the form "list all assessment modules for course X" have no graph-level answer. Prior-review D2 exact language.

### 6. question_type (Claim 5 drift — refuted direction)

Already covered under §4 "assessment_type"; separated here only because the master plan lists it as its own vocabulary.

- **Drift summary**: factory-only value `multiple_response` (`Trainforge/generators/question_factory.py:83,175,199,222`); schema-only values `ordering`+`hotspot` (`schemas/events/trainforge_decision.schema.json:62–65`). Core overlap: 6 values. Total disagreement: 3 values (1 factory-only, 2 schema-only).
- **Schematized**: yes, twice, divergently.
- **KG-impact**: As §4. A Question node's `type` is a KG primary-key attribute; divergent enums mean the same logical question materializes under two distinct type labels depending on which surface captured it.

### 7. BLOOM_CHUNK_STRATEGY (bloom level → preferred chunk type)

- **Source of truth**: `Trainforge/rag/libv2_bridge.py:310–317` — `{remember:(explanation,example), understand:(explanation,example), apply:(example,explanation), analyze:(example,explanation), evaluate:(None,example), create:(None,explanation)}`.
- **Producers**: code-only; the dict is the entire contract.
- **Consumers**:
  - `Trainforge/rag/libv2_bridge.py:341` — `retrieve_for_objective()` uses it to parameterize `multi_query_retrieve(chunk_type=...)`.
  - `Trainforge/tests/test_retrieval_improvements.py:104` — asserts the dict shape.
- **Schematized**: **no**. Zero schema references. Not in `schemas/ONTOLOGY.md` at all (despite ONTOLOGY.md's exhaustive tables).
- **Drift evidence**: The strategy references `"explanation"` and `"example"` as chunk-type values, but `chunk_type` itself is free-string (see §2 on content_type confusion: chunks carry both `chunk_type` and `content_type_label` with overlapping-but-different semantics — `process_course.py:874,1082` sets `chunk_type`, while `content_type_label` is set at `:1162`). Silent miss: if a chunk is labeled `chunk_type="overview"` or `content_type_label="explanation"` but `chunk_type="example"`, the strategy filter partitions the corpus differently from how the pedagogical vocabulary suggests.
- **KG-impact**: Retrieval-to-objective KG edges (for `apply`/`analyze` Bloom levels) depend on a code-only dict pairing two free-string vocabularies. A rename of either value silently degrades the `evaluate`/`create` fallback paths (which already use `None` for primary — an undocumented escape hatch). No versioning, no provenance — retrieval behavior cannot be reconstructed from the KG alone.

### 8. decision_type enum (Trainforge custom types)

- **Source of truth**: `schemas/events/decision_event.schema.json:7,61–103` — 40-value enum (`approach_selection, strategy_decision, source_selection, source_interpretation, textbook_integration, existing_content_usage, content_structure, content_depth, content_adaptation, example_selection, pedagogical_strategy, assessment_design, bloom_level_assignment, learning_objective_mapping, accessibility_measures, format_decision, component_selection, quality_judgment, validation_result, error_handling, prompt_response, file_creation, outcome_signal, chunk_selection, question_generation, distractor_generation, revision_decision, source_usage, alignment_check, structure_detection, heading_assignment, alt_text_generation, math_conversion, research_approach, query_decomposition, retrieval_ranking, result_fusion, chunk_deduplication, index_strategy`).
- **Producers (drift)**:
  - `Trainforge/synthesize_training.py:226,261,287,312` — logs `decision_type="instruction_pair_synthesis"` (4 sites) and `"preference_pair_generation"` (1 site at `:288`). Neither value is in the 40-value enum.
  - `lib/decision_capture.py:78,89–90` — `DecisionCapture` internally allowlists `"instruction_pair_synthesis"` and `"preference_pair_generation"` (comment at `:78` explicitly notes "Worker C's first landing decision type"), so validation passes at the Python layer but not against the JSON-Schema.
  - `docs/contributing/workers.md:39` — policy note that the enum is expected to grow as each type lands; acknowledges the drift as intentional-but-unlanded.
- **Consumers**:
  - Any validator enforcing `schemas/events/decision_event.schema.json` rejects the Trainforge types.
  - Analysis tools (`analyze_training_data`, export filters) iterate by decision_type; missing-from-enum types may be excluded by filter helpers that constrain to the enum.
- **Schematized**: **yes (canonical) but producer bypasses it** via in-code allowlist.
- **Drift evidence**: Cites prior-review C3. `ADR-001-pipeline-shape.md:117` proposes "no central enum to touch", formalizing the drift pattern. This is not a bug but a contract gap: the schema claims authority, the code claims none.
- **KG-impact**: Decision-event provenance KG edges split into two populations: schema-validated (40-value) and code-allowlisted (42-value). Cross-worker analytics (e.g. "count all distractor_generation events") may or may not include synthesis-stage pairs depending on which surface filters the stream. Provenance auditability is degraded: a KG "Decision" node's type field has two legitimate authorities.

## Cross-vocab observations

- **Emit-side single-sourcing is the dominant pattern.** Five of eight vocabularies (`cognitive_domain`, `content_type`, `teaching_role`, `module_type`, `BLOOM_CHUNK_STRATEGY`) have their entire vocabulary defined inside one Python file; none reference a `schemas/` enum. This is a contract shape, not a schema gap.
- **Content_type is the most fragmented vocabulary in the repo.** Four producer surfaces (two in CF, one in the textbook-objective-generator, one in Trainforge `process_course._extract_section_metadata`) and three consumer surfaces (parser, instruction_factory, LibV2 retriever) with no canonical enum. Compare to Bloom which has 5+ copies (Worker A) but agrees on the 6-value vocabulary — content_type does not even agree on vocabulary size.
- **Question_type / assessment_type conflation.** CF JSON-LD `suggestedAssessmentTypes` carries values from the *question_type* enum (`{multiple_choice, true_false, short_answer, ...}`) into a field that LO schema treats as an *assessment_method* enum (`{exam, quiz, assignment, ...}`). These are different vocabularies describing different KG relations (Objective→Question-format vs Objective→Assessment-method); neither schema catches the category error.
- **Teaching_role could be derived from existing emit surface.** `data-cf-component` + `data-cf-purpose` (emitted deterministically at `generate_course.py:345,374,487`) partition every interactive element into three classes that map cleanly onto `VALID_ROLES` (e.g. `term-definition → introduce`, `practice → transfer`, `formative-assessment → assess`). Current implementation uses an LLM classifier over chunk text and discards the deterministic signal — a provenance inversion.
- **Proposed enum unions (enumeration only, not a recommendation)**:
  - `cognitive_domain`: `{factual, conceptual, procedural, metacognitive}` — 4 values, already stable.
  - `content_type` (chunk-level): minimum 13 values (from CF `_infer_content_type` ∪ callout variants ∪ `BloomTaxonomyMapper.CONTENT_TYPE_DEFAULTS`); no consensus without explicit decision.
  - `teaching_role`: `{introduce, elaborate, reinforce, assess, transfer, synthesize}` — 6 values, already stable in one place.
  - `module_type`: `{overview, content, application, assessment, summary, discussion}` — 6 values from actual emit sites.
  - `question_type`: `{multiple_choice, multiple_response, true_false, short_answer, essay, matching, fill_in_blank, ordering, hotspot}` — 9-value union.
  - `assessment_method` (distinct): `{exam, quiz, assignment, project, discussion, presentation, portfolio, demonstration, case_study}` — the LO-schema enum, semantically distinct from `question_type`.

## KG-impact summary table

| Vocabulary | Severity | Producers | Consumers | Schema | KG-impact (1-line) |
|---|---|---|---|---|---|
| `cognitive_domain` | High | CF 2 sites | TF parser + tests | No | Objective→Domain edge single-sourced; Bloom drift reroutes silently. |
| `content_type` | **Critical** | CF 4 sites + TOG mapper 20-val | TF parser + factory + LibV2 retriever | No (free-string in instruction_pair) | Chunk→ContentType edge unconstrained; retrieval and template joins mis-partition. |
| `teaching_role` | High | TF align_chunks only | LibV2 retriever | No | Chunk→TeachingRole edge LLM-classified, non-reproducible; CF deterministic signal dropped. |
| `assessment_type` | High | 4 divergent schemas + CF JSON-LD | TF factory + parser | Yes (4-way divergent) | Objective→AssessmentMethod and Question→Type edges mismatch across surfaces. |
| `module_type` | Medium | CF 6 emit sites | TF tests only | No | Module→Page partitioning absent from KG; 6th value `discussion` undocumented. |
| `question_type` | High | TF factory + CF JSON-LD | TF decision events | Yes (2-way divergent) | Question `type` primary-key attribute has two incompatible authorities. |
| `BLOOM_CHUNK_STRATEGY` | Medium | TF code-only dict | TF retrieval | No | Bloom→ChunkType retrieval table unversioned; free-string pairing, no provenance. |
| `decision_type` | Medium | TF 4 sites + lib allowlist | schema validators | Yes (canonical) but bypassed | Decision-event KG edges split into schema-validated vs code-allowlisted populations. |
