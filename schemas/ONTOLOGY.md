# Ed4All Ontology Map — Current State

## § 0 Header & scope

This document is a **descriptive snapshot** of the Ed4All ontology as it exists in branch `dev-v0.2.0` at or near commit `fea48f8` (post Workers R/S/T merges). It catalogs what classes, relations, taxonomies, provenance mechanisms, serialization surfaces, identifiers, constraints, and version counters exist in the code and schemas **today** — nothing more.

There is no gap analysis, no target ontology, no improvement recommendation, and no implementation plan in this document. Those belong elsewhere.

**Last generated:** 2026-04-19.
**Regeneration:** re-run Worker U's sub-plan against the current tree (`~/.claude/plans/worker-u-schemas-ontology-map.md`); every `Definition:` path can be grep-verified against the filesystem.

---

## § 1 At-a-glance diagram

Five layers, producer → consumer arrows (downward). Every arrow is grounded in code referenced in § 11.

```
   Academic layer               Objective layer        Knowledge layer
   ┌──────────────────┐        ┌──────────────────┐   ┌──────────────────┐
   │ Course           │        │ LearningObjective│   │ Chunk            │
   │  └ Module        │  LOs   │  (TO-NN course / │   │  └ concept_tags  │
   │     └ Page       │──────▶│   CO-NN chapter /│──▶│ Concept          │
   │        └ Section │        │   WNN-CO-NN wk)  │   │ TypedEdge        │
   │           └ CB   │        └──────────────────┘   │ KeyTerm          │
   └──────────────────┘                               │ Misconception    │
          │                                           └──────────────────┘
          │ emit JSON-LD + data-cf-*                           │
          ▼                                                    │
   [ Courseforge HTML page ]  ──chunked by Trainforge──────────┘
                                              │
                                              ▼
                                  Assessment layer
                                  ┌──────────────────┐
                                  │ Assessment       │
                                  │  └ Question      │  misconception-backed
                                  │     ├ Choice     │  distractors
                                  │     └ Distractor │
                                  │ InstructionPair  │
                                  │ PreferencePair   │
                                  └──────────────────┘

   Library layer (terminal sink)
   ┌──────────────────┐
   │ CourseManifest   │ ← classifications from schemas/taxonomies/
   │ CatalogEntry     │
   └──────────────────┘

   Provenance/event spine (cross-cutting, orthogonal to all layers)
     DecisionEvent ─▶ TrainforgeDecisionEvent
     AuditEvent  ·  HashChainedEvent  ·  SessionAnnotation  ·  RunManifest
     InputRef    ·  OutputArtifact
```

---

## § 2 Class catalog

Per-class subsections follow a fixed template (definition path, production site, consumption site, required fields, optional fields, discriminators, example). Enum cardinalities and field lists come directly from the cited schema or dataclass.

### Course

**Definition:** `schemas/academic/course_metadata.schema.json`
**Instance production:** Authored by course-outliner agent / Courseforge planning phase.
**Instance consumption:** `Courseforge/scripts/generate_course.py` (page emit), brightspace-packager agent (IMSCC), LibV2 importer (manifest derivation).

**Required fields (top-level):** `courseIdentification`, `courseDescription`, `instructionalTeam`, `courseStructure`, `assessmentFramework`, `accessibility`, `metadata`.

**Key sub-field tables** (not a full nested listing — see schema for full tree):

| Sub-object | Required members |
| ---------- | ---------------- |
| `courseIdentification` | `courseNumber` (pattern `^[A-Z]{2,4}[0-9]{3,4}[A-Z]?$`), `courseTitle` (5–100 chars), `department`, `courseLevel` |
| `courseDescription` | `shortDescription` (≤300 chars), `learningOutcomes[]` (3–12 items) |
| `instructionalTeam` | `primaryInstructor` {`name`, `title`} |
| `courseStructure` | `duration`, `schedule`, `modules[]` |
| `metadata` | `version` (semver), `lastUpdated` (date-time), `updatedBy` |

**Optional fields:** `courseSubtitle`, `school`, `institution`, `credits`, `prerequisites`, `resources`, `accessibility.universalDesign`, etc. See schema for full list.

**Discriminators:** `courseIdentification.courseNumber` pattern; `courseLevel` enum.

### Module

**Definition:** `schemas/academic/course_metadata.schema.json` (items of `courseStructure.modules`).
**Instance production:** course-outliner / requirements-collector agents.
**Instance consumption:** `Courseforge/scripts/generate_course.py::generate_week()` (lines 598-738).

**Required fields:** `moduleNumber` (int ≥ 1), `title`, `learningObjectives[]` (strings).

**Optional fields:** `description` (≤500 chars), `estimatedDuration` {`value`, `unit ∈ {hours,days,weeks}`}, `contentTypes[]` (enum), `assessments[]`.

**Discriminators:** `moduleNumber` integer position; module pages use a separate `moduleType` discriminator (see Page).

### Page

**Definition:** Courseforge HTML output; structure encoded in JSON-LD emitted at `Courseforge/scripts/generate_course.py:571-595` (`_build_page_metadata`). No standalone schema file.
**Instance production:** `Courseforge/scripts/generate_course.py::generate_week()` emits five pages per week: overview, content_XX, application, self_check, summary.
**Instance consumption:** `Trainforge/process_course.py::_chunk_content()` (line 787) parses these pages during IMSCC ingestion.

**Required fields** (JSON-LD emit):
| field | type | notes |
| ----- | ---- | ----- |
| `@context` | const | `https://ed4all.dev/ns/courseforge/v1` |
| `@type` | const | `CourseModule` |
| `courseCode` | string | e.g. `WCAG_201` |
| `weekNumber` | int | week position |
| `moduleType` | enum | `{overview, content, application, assessment, summary}` |
| `pageId` | string | slug, e.g. `week_01_content_02_accessibility_basics` |

**Optional fields:** `learningObjectives[]` (see LearningObjective), `sections[]` (see Section), `misconceptions[]`, `suggestedAssessmentTypes[]`.

**Discriminators:** `moduleType` enum value.

**Instance example:**
```json
{"@context":"https://ed4all.dev/ns/courseforge/v1","@type":"CourseModule",
 "courseCode":"WCAG_201","weekNumber":1,"moduleType":"content",
 "pageId":"week_01_content_01_introduction",
 "learningObjectives":[{"id":"TO-01","statement":"...","bloomLevel":"understand"}]}
```

### Section (textbook / page)

**Definition:** `schemas/academic/textbook_structure.schema.json#/definitions/section` (also Page JSON-LD `sections[]`).
**Instance production:** textbook-ingestor agent (textbook_structure); `Courseforge/scripts/generate_course.py:549-568` (`_build_sections_metadata`) for page emit.
**Instance consumption:** `Trainforge/parsers/html_content_parser.py` for chunk alignment.

**Required fields (textbook_structure):** `id`, `headingText`, `contentBlocks[]`.

**Optional fields:** `headingLevel` (2–6), `headingId` (HTML id), `subsections[]` (recursive).

**Page JSON-LD section shape** (emitted by Courseforge):

| field | type | notes |
| ----- | ---- | ----- |
| `heading` | string | rendered h2/h3 text |
| `contentType` | enum | see § 4 content-type list |
| `keyTerms[]` | object | `{term, definition}` pairs from flip cards |
| `bloomRange[]` | array | bloom-level spread in the section |

### ContentBlock

**Definition:** `schemas/academic/textbook_structure.schema.json#/definitions/contentBlock`.
**Instance production:** textbook-ingestor parsing DART-produced HTML.
**Instance consumption:** chapter-level section extractors in Courseforge objective synthesis.

**Required fields:** `id`, `blockType`.

**Optional fields:** `content` (str, may include inline HTML), `listItems[]`, `tableData` {`caption`, `headers[]`, `rows[][]`}, `figureData` {`src`, `alt`, `caption`}, `containsDefinitions` (bool), `containsKeyTerms` (bool), `wordCount` (int ≥ 0).

**Discriminators:** `blockType` enum (see § 4, 14 values).

### UI components: Accordion · ContentDisplay · EnhancedContentDisplay · CourseCard · EducationalTemplate

These remain Courseforge-local (see `/schemas/README.md`).

**Definition paths:**
- `Courseforge/schemas/content-display/accordion-schema.json`
- `Courseforge/schemas/content-display/content-display-schema.json`
- `Courseforge/schemas/content-display/enhanced-content-display-schema.json`
- `Courseforge/schemas/layouts/course_card_schema.json`
- `Courseforge/schemas/template-integration/educational_template_schema.json`

**Instance production:** Courseforge content-generator / intelligent-design-mapper agents.
**Instance consumption:** brightspace-packager (HTML → IMSCC).

Each describes an inline HTML component produced by Courseforge (accordions for FAQ/definitions, content-display wrappers for hero sections, course-card layouts for grids, educational template shells). They are UI-only — none leave the Courseforge HTML output.

### FlipCard · SelfCheck · ActivityCard (code-only)

**Definition:** No schema; emitted inline as HTML with `data-cf-component=*` by `Courseforge/scripts/generate_course.py`.

- FlipCard: `_render_flip_cards()` (lines 336-352), `data-cf-component="flip-card"`, `data-cf-purpose="term-definition"`, `data-cf-term=<slug>`.
- SelfCheck: `_render_self_check()` (lines 355-385), `data-cf-component="self-check"`, `data-cf-purpose="formative-assessment"`, carries `data-cf-bloom-level` and optional `data-cf-objective-ref`.
- ActivityCard: `_render_activities()` (lines 480-497), `data-cf-component="activity"`, `data-cf-purpose="practice"`, carries `data-cf-bloom-level` and optional `data-cf-objective-ref`.

**Instance consumption:** `Trainforge/parsers/html_content_parser.py` (component detection); `Trainforge/process_course.py` (interactive-components coverage metric).

### LearningObjective

**Definition:** `schemas/academic/learning_objectives.schema.json#/definitions/learningObjective`.
**Instance production:** textbook-ingestor + objective-synthesizer agents (extracted from textbooks); course-outliner agent (new courses).
**Instance consumption:** `Courseforge/scripts/generate_course.py:512-546` (`_build_objectives_metadata` → JSON-LD); Trainforge chunk alignment (`learning_outcome_refs` on chunk).

**Required fields:**
| field | type | notes |
| ----- | ---- | ----- |
| `objectiveId` | string | unique ID; prefix discriminates scope (see below) |
| `statement` | string | ≥10 chars |
| `bloomLevel` | enum | `{remember, understand, apply, analyze, evaluate, create}` |

**Optional fields:** `bloomVerb`, `keyConcepts[]`, `sourceReference` (`{headingId, headingText, pageNumber, elementPath}`), `assessmentSuggestions[]` (9-value enum), `prerequisiteObjectives[]` (string IDs), `extractionSource` (7-value enum).

**Discriminators / subtype signaling:** ID prefix:
- `TO-NN` — Terminal Objective (course-level).
- `CO-NN` — Chapter Objective.
- `WNN-CO-NN` — Week-scoped Chapter Objective (legacy; week-prefix normalization described at `Courseforge/scripts/generate_course.py:605-613`).

### Chunk

**Definition:** No standalone JSON Schema; produced as a Python dict by `Trainforge/process_course.py::_create_chunk()` (line 1038). Schema-versioned via `CHUNK_SCHEMA_VERSION = "v4"` (line 86).
**Instance production:** `_chunk_content` + `_chunk_text_block` + `_create_chunk` in `process_course.py`.
**Instance consumption:** `Trainforge/rag/typed_edge_inference.py`, `Trainforge/generators/*.py`, instruction/preference pair synthesis, LibV2 corpus.

**Required fields:**
| field | type | notes |
| ----- | ---- | ----- |
| `id` | string | pattern `^<course>_chunk_\d{5}$` |
| `schema_version` | const | `"v4"` |
| `chunk_type` | string | e.g. `explanation`, `example`, `overview` |
| `text` | string | stripped plain text |
| `html` | string | raw HTML fragment |
| `source` | object | see below |
| `concept_tags` | string[] | normalized tags |
| `learning_outcome_refs` | string[] | LO IDs (may be empty) |
| `difficulty` | string | see § 4 |
| `tokens_estimate` | int | `word_count * 1.3` |
| `word_count` | int | |
| `follows_chunk` | string \| null | prev chunk_id for boundary continuity |

**`source` sub-object:** `course_id`, `module_id`, `module_title`, `lesson_id`, `lesson_title`, `resource_type`, `section_heading`, `position_in_module`, optional `html_xpath` + `char_span` (Section 508 audit trail, lines 1067-1077), optional `item_path` (IMSCC-relative path).

**Enrichment fields (post-base chunk):** `bloom_level`, `bloom_level_source`, `content_type_label`, `key_terms`, `misconceptions`, `_metadata_trace` (Worker M1 diagnostic, `process_course.py:1219`).

**Discriminators:** `chunk_type` string; `schema_version` version stamp.

### Concept

**Definition:** `schemas/knowledge/concept_graph_semantic.schema.json#/properties/nodes/items`.
**Instance production:** Trainforge concept-graph builder (co-occurrence pass + key-term promotion).
**Instance consumption:** `Trainforge/rag/typed_edge_inference.py` (node lookup); LibV2 concept indexes.

**Required fields:** `id` (string).
**Optional fields:** `label` (string), `frequency` (int ≥ 0), plus arbitrary `additionalProperties`.

**Discriminators:** `id` is a normalized kebab-case slug (e.g. `cognitive-load-theory`).

### TypedEdge

**Definition:** `schemas/knowledge/concept_graph_semantic.schema.json#/properties/edges/items`.
**Instance production:** `Trainforge/rag/typed_edge_inference.py` orchestrator, applying three inference rules (see § 4 taxonomy list).
**Instance consumption:** LibV2 retrieval (graph-aware ranking); Trainforge question generation (prerequisite traversal).

**Required fields:**
| field | type | notes |
| ----- | ---- | ----- |
| `source` | string | concept id |
| `target` | string | concept id |
| `type` | enum | `{prerequisite, is-a, related-to}` |
| `provenance` | object | `{rule, rule_version, evidence?}` (rule_version ≥ 1) |

**Optional fields:** `confidence` (0–1), `weight` (number), plus `additionalProperties`.

**Discriminators:** `type` enum; `related-to` is treated undirected, others directed (`typed_edge_inference.py:57`).

**Instance example:**
```json
{"source":"scaffolding","target":"constructivism","type":"is-a","confidence":0.82,
 "provenance":{"rule":"is_a_from_key_terms","rule_version":1,
               "evidence":{"container":"constructivism","member":"scaffolding"}}}
```

See § JSON-LD round-trip for the RDF projection of this shape (edges reify as `ed4all:TypedEdge` blank nodes that carry the `provenance` block).

### KeyTerm

**Definition:** Appears in two surfaces:
1. `schemas/academic/textbook_structure.schema.json#/properties/extractedConcepts/properties/keyTerms/items`
2. Courseforge Page JSON-LD `sections[].keyTerms[]` (`generate_course.py:559-563`)

**Instance production:** textbook-ingestor (textbook), content-generator agent (course pages via flip-card authoring).
**Instance consumption:** Trainforge `_extract_section_metadata` merges `keyTerms[].term` into `concept_tags` (`process_course.py:1107-1116`).

**Required fields (textbook_structure):** `term`.
**Optional fields (textbook_structure):** `context`, `emphasisType ∈ {strong, em, heading, callout}`, `chapterId`, `sectionId`.
**Page JSON-LD shape:** `{term, definition}` pairs (both strings).

### Misconception

**Definition:** No standalone schema; emitted in Courseforge JSON-LD (`page_metadata.misconceptions[]`) and stored on Chunk (enrichment).
**Instance production:** content-generator authors them per week; merged into content-page JSON-LD at `generate_course.py:671`.
**Instance consumption:** `Trainforge/generators/preference_factory.py` — explicit misconceptions back DPO `rejected` answers with stable IDs (`_misconception_id`, lines 140-143: `{chunk_id}_mc_{index:02d}_{hash}`).

**Structural shape (as emitted):** a free-form dict — most commonly `{misconception: string, correction: string}`; passed through without enum constraint.

### Assessment

**Definition:** Python dataclass `AssessmentData` at `Trainforge/generators/assessment_generator.py:111`.
**Instance production:** `AssessmentGenerator.generate()` (assessment-generator agent).
**Instance consumption:** brightspace-packager (QTI emit); assessment-validator agent; `trainforge_decision.schema.json`.

**Fields:** `assessment_id` (str), `title` (str), `course_code` (str), `questions` (list[QuestionData]), `objectives_targeted` (list[str]), `bloom_levels` (list[str]), `created_at` (ISO datetime), `status` (default `"generated"`).

`to_dict()` also emits derived `question_count` and `total_points`.

### Question

Two closely-related representations exist:

**Definition (factory-side):** dataclass `Question` at `Trainforge/generators/question_factory.py:36`.
**Definition (generator-side):** dataclass `QuestionData` at `Trainforge/generators/assessment_generator.py:81`.

**Instance production:** `QuestionFactory.create_*` methods (with Bloom-alignment enforcement, line 103); `AssessmentGenerator.generate()`.
**Instance consumption:** brightspace-packager QTI emit; validators (bloom, question_quality, leak_check).

**Required fields (Question):** `question_id`, `question_type` (enum, 7 values — see § 4), `stem`, `bloom_level`, `objective_id`.
**Optional fields:** `points` (default 1.0), `feedback`, `choices[]` (QuestionChoice), `correct_answers[]` (for fill-in/matching), `case_sensitive` (bool).

**QuestionData additional fields:** `source_chunks[]`, `generation_rationale`.

**Discriminators:** `question_type` enum selects which field subset is populated (choices for MCQ, correct_answers for FIB/matching, stem-only for essay).

### QuestionChoice

**Definition:** dataclass at `Trainforge/generators/question_factory.py:28`.
**Fields:** `text` (str), `is_correct` (bool, default False), `feedback` (Optional[str]).

### Distractor

**Definition:** `schemas/events/trainforge_decision.schema.json#/allOf/1/properties/question_data/properties/distractors/items` (schema-side); `QuestionChoice` with `is_correct=False` (code-side).

**Fields (schema):**
| field | type | notes |
| ----- | ---- | ----- |
| `text` | string | distractor option |
| `misconception_targeted` | string | stable misconception id / description |
| `plausibility_score` | number | |

**Production:** `QuestionFactory.create_multiple_choice`; distractor rationale captured via `trainforge_capture.log_distractor_rationale`.

### InstructionPair

**Definition:** `schemas/knowledge/instruction_pair.schema.json`.
**Instance production:** `Trainforge/synthesize_training.py` → `Trainforge/generators/instruction_factory.py`.
**Instance consumption:** Downstream SFT trainers (Alpaca / OpenAI format).

**Required fields:**
| field | type | notes |
| ----- | ---- | ----- |
| `prompt` | string | 40–400 chars, no ≥50-char verbatim span from source |
| `completion` | string | 50–600 chars |
| `chunk_id` | string | source chunk |
| `lo_refs` | string[] | ≥1 (never empty) |
| `bloom_level` | enum | 6 Bloom levels |
| `content_type` | string | free-string from chunk |
| `seed` | int | deterministic template-selection seed |
| `decision_capture_id` | string | event_id in decisions JSONL |

**Optional fields:** `template_id` (Bloom × content-type template id), `provider ∈ {mock, anthropic}`, `schema_version` (const `"v1"`).

**Discriminators:** `schema_version` const.

### PreferencePair

**Definition:** `schemas/knowledge/preference_pair.schema.json`.
**Instance production:** `Trainforge/generators/preference_factory.py`.
**Instance consumption:** DPO training pipelines.

**Required fields:** `prompt` (40–400 chars), `chosen` (50–600), `rejected` (50–600; must differ from chosen with token-Jaccard δ ≥ 0.3), `chunk_id`, `lo_refs[]` (≥1), `seed`, `decision_capture_id`.
**Optional fields:** `misconception_id` (null when rule-synthesized), `rejected_source ∈ {misconception, rule_synthesized}`, `provider ∈ {mock, anthropic}`, `schema_version` const `"v1"`.

**Discriminators:** `rejected_source`; `misconception_id` nullability.

### WCAGCompliance

**Definition:** `schemas/compliance/wcag22_compliance.schema.json`.
**Instance production:** accessibility-remediation agent; DART conversion validation.
**Instance consumption:** `lib/validators/content.py::ContentStructureValidator`; WCAGValidator validation gate.

**Top-level required:** `complianceLevel.standard ∈ {WCAG_2.0_AA, WCAG_2.1_AA, WCAG_2.2_AA}` (default `WCAG_2.2_AA`).

**Structure:** 4 principle blocks — `perceivableRequirements`, `operableRequirements`, `understandableRequirements`, `robustRequirements`. Each carries nested booleans + numeric thresholds for the WCAG 2.2 success criteria listed in § 4.

**Discriminators:** `complianceLevel.standard` enum.

### Bootstrap5Migration

**Definition:** `Courseforge/schemas/framework-migration/bootstrap5_migration_schema.json`.
**Instance production:** Courseforge accessibility-remediation / framework-migration tooling.
**Instance consumption:** brightspace-packager (for LMSes expecting Bootstrap 5 markup).

Tool-local; describes the transformation contract used when upgrading older Bootstrap-4 content to Bootstrap-5 in place. Not consumed outside Courseforge.

### CourseManifest

**Definition:** `schemas/library/course_manifest.schema.json`.
**Instance production:** `LibV2/tools/libv2/importer.py` emits this per course import.
**Instance consumption:** LibV2 catalog generator, retrieval engine, fsck.

**Required fields:** `libv2_version` (semver), `slug` (regex `^[a-z0-9][a-z0-9-]*[a-z0-9]$`, 3–100 chars), `import_timestamp`, `sourceforge_manifest` (with `sourceforge_version`, `export_timestamp`, `course_id`, `course_title`), `classification`, `content_profile`.

**`classification` required:** `division ∈ {STEM, ARTS}`, `primary_domain`.
**`classification` optional:** `secondary_domains[]`, `subdomains[]`, `topics[]`, `subtopics[]`.

**Other optionals:** `ontology_mappings` (ACM CCS + LCSH; see § 4), `relationships` (`prerequisites[]`, `related_courses[]`, `successor_courses[]`), `quality_metadata` (`validation_status`, `completeness_score`, `annotation_coverage`), `provenance` (`source_type`, `source_path`, `original_provider`, `import_pipeline_version`).

### CatalogEntry

**Definition:** `schemas/library/catalog_entry.schema.json`.
**Instance production:** `LibV2/tools/libv2/catalog.py` (from manifest).
**Instance consumption:** LibV2 CLI (`catalog list`, `catalog stats`), retrieval engine indexes.

**Required fields:** `slug`, `title`, `division ∈ {STEM, ARTS}`, `primary_domain`.
**Optional fields:** `secondary_domains[]`, `subdomains[]`, `chunk_count` (int ≥ 0), `concept_count`, `token_count`, `difficulty_primary ∈ {foundational, intermediate, advanced, mixed}`, `language` (default `"en"`), `validation_status ∈ {pending, validated, failed}`.

### DecisionEvent

**Definition:** `schemas/events/decision_event.schema.json`.
**Instance production:** `lib/decision_capture.py::DecisionCapture.log_decision` (line 441).
**Instance consumption:** validators, training-data export (alpaca/openai/dpo/raw), MCP analysis tools.

**Required fields (6):** `run_id`, `timestamp`, `operation`, `decision_type`, `decision`, `rationale` (≥20 chars — enforced by schema minLength).

**Phase-0-hardening optional fields:** `event_id` (`^EVT_[a-f0-9]{16}$`), `seq` (monotonic int per run), `task_id` (`^T-[a-f0-9]{8}$`), `is_default` (bool), `outputs[]` (artifact pointers).

**Other optionals:** `course_id` (`^[A-Z]{2,8}_[0-9]{3}$`), `module_id` (`^[A-Z]{2,8}_[0-9]{3}_W[0-9]{2}_M[0-9]{2}$`), `artifact_id` (SHA-256 short hash), `phase` (23-value enum), `tool ∈ {dart, courseforge, trainforge, orchestrator}`, `alternatives_considered[]`, `context`, `confidence ∈ [0.0, 1.0]`, `ml_features`, `inputs_ref[]` (see InputRef), `prompt_ref`, `outcome`, `metadata`.

**Discriminators:** `tool` (which tool produced); `decision_type` (40-value enum — see § 4); `phase`.

### TrainforgeDecisionEvent

**Definition:** `schemas/events/trainforge_decision.schema.json` — `allOf: [{$ref: decision_event.schema.json}, {additional properties}]`.
**Instance production:** `lib/trainforge_capture.py::TrainforgeCapture` (wraps DecisionCapture).
**Instance consumption:** same as DecisionEvent + Trainforge-specific quality analyzers.

**Extra properties over DecisionEvent:**
- `assessment_context` — required `imscc_source`, optional `learning_objective_id`, `bloom_target` (6-level enum), `source_chunks[]` {`chunk_id`, `content_hash`, `relevance_score`, `token_count`, `source_file`}, `domain`, `domain_weight` (0–100).
- `question_data` — `question_id`, `question_type` (8-value enum — see § 4), `question_stem`, `correct_answer`, `distractors[]` (see Distractor), `explanation`, `difficulty ∈ {easy, medium, hard}`, `points` (≥1), `time_estimate_seconds`, `rubric` (criteria with weight/levels).
- `rag_metrics` — `chunks_retrieved`, `chunks_used`, `retrieval_latency_ms`, `generation_latency_ms`, `context_token_count`, `embedding_model`, `similarity_threshold`.
- `alignment_check` — `lo_coverage_score`, `bloom_alignment_score`, `content_alignment_score` (all 0–1), `passed` (bool), `issues[]`.
- `revision_chain[]` — ordered `{revision_number, timestamp, reason, changes_made[], validator_feedback}`.

### AuditEvent

**Definition:** `schemas/events/audit_event.schema.json`.
**Instance production:** cross-cutting audit logger (file access, tool invocation, state changes).
**Instance consumption:** compliance / audit reports; MCP hardening checks.

**Required fields:** `run_id` (alphanumeric+underscore), `event_id` (`^EVT_[a-f0-9]{16}$`), `seq` (int ≥0), `timestamp`, `event_type` (8-value enum — see § 4).

**Optional fields:** `component`, `worker_id`, `task_id` (`^T-[a-f0-9]{8}$`), `details` (oneOf over 7 sub-schemas keyed by `event_type`), `redacted_fields[]`, `metadata`.

**Details sub-schemas:** `file_access_details` (operation, path, content_hash, size_bytes, success), `tool_invocation_details` (tool_name, version, args/result hashes, duration, exit_status), `state_change_details` (state_file, change_type ∈ {create,update,delete,lock,unlock}, prev/new hash), `workflow_event_details` (workflow_id, event_subtype ∈ 8 values), `validation_event_details` (validator_name, passed, score, waived, waiver_reason), `error_details` (error_type, message, recoverable, stack_trace), `security_event_details` (security_event_type ∈ 4 values).

### HashChainedEvent

**Definition:** `schemas/events/hash_chained_event.schema.json`.
**Instance production:** `lib/hash_chain.py` (tamper-evident append).
**Instance consumption:** audit verification, replay engine (`lib/replay_engine.py`).

**Required fields:** `seq` (≥0), `prev_hash` (`genesis` or 64-char hex), `event_hash` (SHA-256 of `prev_hash + JSON(event)`), `timestamp`, `event` (arbitrary object).

**Discriminators:** `prev_hash == "genesis"` marks the first link.

### SessionAnnotation

**Definition:** `schemas/events/session_annotation.schema.json`.
**Instance production:** `lib/run_finalizer.py` (aggregates JSONL decision files at session end).
**Instance consumption:** session-level quality reporting; training-data export filtering.

**Required fields:** `session_id`, `run_id`, `tool ∈ {dart, courseforge, trainforge, orchestrator}`, `started_at`, `status ∈ {running, complete, error, partial, cancelled}`.

**Optional fields:** `course_id` (`^[A-Z]{2,3}_[0-9]{3}$`), `phase`, `completed_at`, `duration_seconds`, `inputs` (bundle of input-type arrays), `outputs` (`files_created[]`, `imscc_package`, `total_modules`, `total_assessments`, `total_questions`, `training_examples_produced`), `decision_summary` (`total_decisions`, `by_type` map, `by_quality_level` with 4-level enum, `avg_rationale_length`, `avg_confidence`), `quality_metrics` (`validation_pass_rate`, `revision_rate`, `error_rate`, `bloom_distribution`), `acceptance` (`accepted`, `rejection_reason`, `revision_count`, `superseded_by`, `reviewer_notes`), `errors[]`, `decision_files[]`.

### RunManifest

**Definition:** `schemas/events/run_manifest.schema.json`.
**Instance production:** `lib/run_manager.py` (at run initialization; written once, never mutated).
**Instance consumption:** run discovery, fsck, replay engine.

**Required fields:** `run_id` (`^RUN_[0-9]{8}_[0-9]{6}_[a-f0-9]{8}$`), `created_at`, `workflow_type ∈ {course_generation, intake_remediation, batch_dart, rag_training, textbook_to_course}`, `config_hashes` (`workflows_yaml`, `agents_yaml`, `schemas` — all `^sha256:[a-f0-9]{64}$`), `immutable` (const `true`).

**Optional fields:** `git_commit` (40-hex or null), `git_dirty` (bool), `operator`, `goals[]`, `workflow_params`, `environment` (python_version, platform, hostname, etc.), `inputs[]` (`path`, `content_hash`, `hash_algorithm`, `size_bytes`), `schema_version` (default `"1.0.0"`).

**Discriminators:** `workflow_type` enum; `immutable` const.

### InputRef

**Definition:** Python dataclass `InputRef` at `lib/provenance.py:37`. Mirrors `schemas/events/decision_event.schema.json#/properties/inputs_ref/items`.
**Instance production:** `lib.provenance.create_input_ref()`.
**Instance consumption:** decision events `inputs_ref[]`, run manifest `inputs[]`.

**Fields:** `source_type` (string: textbook/pdf/imscc/web_search/prompt_template/assessment_bank/html/agent_output), `path_or_id` (string), `content_hash` (string, default `""`), `hash_algorithm ∈ {sha256, sha512, blake3}` (default sha256), `size_bytes` (int), `byte_range` (`{start, end}`, optional), `excerpt_range` (human string), `metadata` (dict).

### OutputArtifact (OutputRef)

**Definition:** Python dataclass `OutputRef` at `lib/provenance.py:91`. Mirrors `schemas/events/decision_event.schema.json#/properties/outputs/items`.
**Instance production:** `lib.provenance.create_output_ref()`.
**Instance consumption:** decision events `outputs[]`.

**Fields:** `artifact_type` (string: html/imscc/assessment/chunk/etc.), `path` (string), `content_hash`, `hash_algorithm` (default sha256), `size_bytes`, `byte_range`, `metadata`.

---

## § 3 Relation inventory

Single table of directed/undirected relations that cross class boundaries. Cardinality notation: 1 (exactly one), 0..1 (optional), 0..* (any), 1..* (at least one).

| Relation | Domain → Range | Cardinality | Directed | Surface | Defined in |
|---|---|---|---|---|---|
| `hasModule` | Course → Module | 1..* | yes | `courseStructure.modules[]` | `schemas/academic/course_metadata.schema.json` |
| `hasPage` | Module → Page | 1..* | yes | file tree / weekly emit | `Courseforge/scripts/generate_course.py:598` |
| `hasSection` | Page → Section | 0..* | yes | JSON-LD `sections[]` | `generate_course.py:549-568` |
| `hasContentBlock` | Section → ContentBlock | 1..* | yes | `sections[].contentBlocks[]` | `schemas/academic/textbook_structure.schema.json#/definitions/section` |
| `hasSubsection` | Section → Section | 0..* | yes | recursive `subsections[]` | same |
| `hasChild` (TOC) | TOC item → TOC item | 0..* | yes | recursive `children[]` | `textbook_structure.schema.json` |
| `hasCourseObjective` | Course → LearningObjective | 1..* | yes | `courseObjectives[]` | `schemas/academic/learning_objectives.schema.json` |
| `hasChapterObjective` | Chapter → LearningObjective | 0..* | yes | `chapters[].chapterObjectives[]` | same |
| `hasSectionObjective` | Section → LearningObjective | 0..* | yes | `sections[].sectionObjectives[]` | same |
| `prerequisiteOf` (LO) | LearningObjective → LearningObjective | 0..* | yes | LO `prerequisiteObjectives[]` | same |
| `targetsObjective` | Page / Question → LearningObjective | 0..* | yes | `data-cf-objective-ref`, `objective_id` | `generate_course.py:378,491`; `question_factory.py:42` |
| `follows` (chunk order) | Chunk → Chunk | 0..1 | yes | `follows_chunk` field | `Trainforge/process_course.py:1085` |
| `hasConceptTag` | Chunk → Concept | 0..* | yes | `concept_tags[]` | `process_course.py:1087` |
| `hasLORef` | Chunk → LearningObjective | 1..* | yes | `learning_outcome_refs[]` (required for pair generation) | `process_course.py:1088` |
| `typedEdge:is-a` | Concept → Concept | 0..* | yes | TypedEdge type=`is-a` | `Trainforge/rag/inference_rules/is_a_from_key_terms.py` |
| `typedEdge:prerequisite` | Concept → Concept | 0..* | yes | TypedEdge type=`prerequisite` | `Trainforge/rag/inference_rules/prerequisite_from_lo_order.py` |
| `typedEdge:related-to` | Concept ↔ Concept | 0..* | no | TypedEdge type=`related-to` | `Trainforge/rag/inference_rules/related_from_cooccurrence.py` |
| `hasQuestion` | Assessment → Question | 1..* | yes | `AssessmentData.questions` | `Trainforge/generators/assessment_generator.py:117` |
| `hasChoice` | Question → QuestionChoice | 0..* | yes | `Question.choices` | `question_factory.py:45` |
| `hasDistractor` | Question → Distractor | 0..* | yes | `question_data.distractors[]` | `schemas/events/trainforge_decision.schema.json` |
| `targetsMisconception` | Distractor → Misconception | 0..1 | yes | `misconception_targeted` | same |
| `derivedFromChunk` | InstructionPair / PreferencePair → Chunk | 1 | yes | `chunk_id` | `schemas/knowledge/instruction_pair.schema.json`, `preference_pair.schema.json` |
| `encodesMisconception` | PreferencePair → Misconception | 0..1 | yes | `misconception_id` (nullable) | `schemas/knowledge/preference_pair.schema.json` |
| `coursePrerequisite` | Course → Course | 0..* | yes | `relationships.prerequisites[]` | `schemas/library/course_manifest.schema.json` |
| `relatedCourse` | Course ↔ Course | 0..* | no | `relationships.related_courses[]` | same |
| `classifiedAs` | Course → Domain/Subdomain | 1..* | yes | `classification.*` | same |
| `ontologyMap:acm_ccs` | Course → ACM CCS code | 0..* | yes | `ontology_mappings.acm_ccs[]` | same |
| `ontologyMap:lcsh` | Course → LCSH URI | 0..* | yes | `ontology_mappings.lcsh[]` (with relation type) | same |
| `producesOutput` | Decision → OutputArtifact | 0..* | yes | `outputs[]` | `schemas/events/decision_event.schema.json` |
| `usedInput` | Decision → InputRef | 0..* | yes | `inputs_ref[]` | same |
| `linksToTask` | Decision / AuditEvent → Task | 0..1 | yes | `task_id` | `decision_event.schema.json`, `audit_event.schema.json` |
| `chainsFrom` | HashChainedEvent → HashChainedEvent | 0..1 | yes | `prev_hash` | `hash_chained_event.schema.json` |
| `belongsToSession` | DecisionEvent → SessionAnnotation | 1 | yes | `run_id`/`session_id` linkage | `session_annotation.schema.json` |

---

## § 4 Taxonomies

All enums quoted verbatim from the cited source.

### Bloom's Taxonomy (6 levels)

Source: `schemas/academic/learning_objectives.schema.json#/definitions/learningObjective/properties/bloomLevel`, `schemas/events/decision_event.schema.json#/properties/ml_features/properties/bloom_levels/items`.

```
remember, understand, apply, analyze, evaluate, create
```

### BLOOM_VERBS

Source: `Courseforge/scripts/generate_course.py:136-143`.

```python
BLOOM_VERBS = {
    "remember":   ["define", "list", "recall", "identify", "recognize", "name", "state"],
    "understand": ["explain", "describe", "summarize", "interpret", "classify", "compare"],
    "apply":      ["apply", "demonstrate", "implement", "solve", "use", "execute"],
    "analyze":    ["analyze", "differentiate", "examine", "compare", "contrast", "organize"],
    "evaluate":   ["evaluate", "assess", "critique", "judge", "justify", "argue"],
    "create":     ["create", "design", "develop", "construct", "produce", "formulate"],
}
```

### BLOOM_TO_DOMAIN

Source: `Courseforge/scripts/generate_course.py:146-153`.

```python
BLOOM_TO_DOMAIN = {
    "remember":   "factual",
    "understand": "conceptual",
    "apply":      "procedural",
    "analyze":    "conceptual",
    "evaluate":   "metacognitive",
    "create":     "procedural",
}
```

### BLOOM_QUESTION_MAP

Source: `Trainforge/generators/question_factory.py:91-98`.

```python
BLOOM_QUESTION_MAP = {
    "remember":   ["multiple_choice", "true_false", "fill_in_blank", "matching"],
    "understand": ["multiple_choice", "short_answer", "fill_in_blank", "matching"],
    "apply":      ["multiple_choice", "short_answer", "essay"],
    "analyze":    ["multiple_choice", "short_answer", "essay", "matching"],
    "evaluate":   ["essay", "short_answer", "multiple_choice"],
    "create":     ["essay", "short_answer"],
}
```

### Question types

Two enums exist:

**Trainforge internal (factory):** `Trainforge/generators/question_factory.py:81-89`
```
multiple_choice, multiple_response, true_false, fill_in_blank, short_answer, essay, matching
```

**Trainforge decision-capture schema:** `schemas/events/trainforge_decision.schema.json:64`
```
multiple_choice, true_false, short_answer, essay, matching, fill_in_blank, ordering, hotspot
```

### Content types

**Page JSON-LD `sections[].contentType`** (inferred in `Courseforge/scripts/generate_course.py:388-405`):
```
definition, example, procedure, comparison, exercise, overview, summary, explanation
```
(`"application-note"` is also emitted for callout-warning type in `_render_content_sections` line 447.)

**Textbook `contentBlock.blockType`** (14 values): source `schemas/academic/textbook_structure.schema.json#/definitions/contentBlock/properties/blockType`
```
paragraph, heading, list_ordered, list_unordered, definition_list, table, figure,
callout_info, callout_warning, callout_note, code_block, blockquote, example, summary
```

### Module types (Page `moduleType`)

Source: emit sites in `Courseforge/scripts/generate_course.py::generate_week()` (lines 647, 667, 686, 704, 728):
```
overview, content, application, assessment, summary
```
(`self_check` pages are labeled `moduleType="assessment"`.)

### `data-cf-*` attribute vocabulary

Source: `Courseforge/scripts/generate_course.py` (multiple render sites).

| Attribute | Element | Allowed values |
| --- | --- | --- |
| `data-cf-role` | header, footer, a.skip-link | `template-chrome` (only) — lines 289, 290, 297 |
| `data-cf-component` | div.flip-card, div.self-check, div.activity-card | `flip-card`, `self-check`, `activity` — lines 345, 374, 487 |
| `data-cf-purpose` | same | `term-definition`, `formative-assessment`, `practice` — lines 345, 374, 487 |
| `data-cf-objective-id` | li (in `.objectives`) | LO ID string (`TO-NN`, `CO-NN`, `WNN-CO-NN`) — line 314 |
| `data-cf-objective-ref` | .self-check, .activity-card | LO ID string — lines 378, 491 |
| `data-cf-bloom-level` | li, .self-check, .activity-card | 6-Bloom-level enum — lines 316, 375, 488 |
| `data-cf-bloom-verb` | li | detected verb — line 318 |
| `data-cf-bloom-range` | h2/h3 | e.g. `understand,apply` — line 427 |
| `data-cf-cognitive-domain` | li | 4 values `{factual, conceptual, procedural, metacognitive}` — line 320 |
| `data-cf-content-type` | h2, h3, .callout | content-type enum (above) — lines 423, 452 |
| `data-cf-key-terms` | h2, h3 | comma-separated kebab-slug terms — line 425 |
| `data-cf-term` | .flip-card | kebab-slug term — line 346 |
| `data-cf-source-ids` | `<section>`, h2/h3, .flip-card, .self-check, .activity-card, .discussion-prompt, .objectives | Comma-joined DART `sourceId` list (`dart:{slug}#{block_id}`) — Wave 9. Never emitted on `<p>`/`<li>`/`<tr>` (P2 decision). |
| `data-cf-source-primary` | same elements as `data-cf-source-ids` | Single dominant `sourceId`. Emitted only when the enclosing block has one unambiguous primary source. Wave 9. |

### WCAG 2.2 AA structure

Source: `schemas/compliance/wcag22_compliance.schema.json` (4 principle blocks).

Principles (lines 46, 384, 659, 859):
1. **Perceivable** — `perceivableRequirements`
2. **Operable** — `operableRequirements`
3. **Understandable** — `understandableRequirements`
4. **Robust** — `robustRequirements`

Explicitly-referenced WCAG 2.2 success criteria (from `description` strings):
- **2.4.11** Focus Not Obscured (Minimum) — Level A
- **2.4.12** Focus indicator fully visible — Level AA
- **2.4.13** Focus Appearance — Level AA (min outline 2 CSS px, contrast ratio ≥3.0)
- **2.5.7** Dragging Movements — Level AA
- **2.5.8** Target Size (Minimum) — Level AA (min 24 CSS px)
- **3.2.6** Consistent Help — Level A
- **3.3.7** Redundant Entry — Level A
- **3.3.8** Accessible Authentication (Minimum) — Level AA
- **3.3.9** Accessible Authentication (Enhanced) — Level AAA

**Testing tools:**
- Automated: `axe-core, WAVE, Lighthouse, Pa11y, aXe DevTools`
- Manual: `NVDA, JAWS, VoiceOver, TalkBack, keyboard_testing, color_contrast_analyzer`

### Difficulty

Two different enums exist:

- **Trainforge question-level:** `easy, medium, hard` — `schemas/events/trainforge_decision.schema.json:93`.
- **LibV2 course-level:** `foundational, intermediate, advanced, mixed` — `schemas/library/catalog_entry.schema.json:48`.

### Course level

Source: `schemas/academic/course_metadata.schema.json:43`.
```
undergraduate, graduate, professional, continuing_education
```

### LibV2 Division / Domain / Subdomain / Topic

Source: `schemas/taxonomies/taxonomy.json`.

**Divisions:** `STEM`, `ARTS`.

**STEM domains (11):**
`physics, chemistry, biology, mathematics, computer-science, engineering, medicine, environmental-science, data-science, educational-technology`. (Each has 4–10 subdomains; each subdomain has ~4 topics. Full tree in `schemas/taxonomies/taxonomy.json`.)

**ARTS domains (8):**
`education, design, visual-arts, music, literature, philosophy, history, linguistics`.

Sample leaf path: `STEM → computer-science → algorithms → {sorting, searching, graph-algorithms, dynamic-programming, complexity-theory}`.

### Pedagogy framework (12 tier domains)

Source: `schemas/taxonomies/pedagogy_framework.yaml`.
```
 1. foundational_theories       (Constructivism, Connectivism, Behaviorism, Cognitivism, Cognitive Load Theory)
 2. instructional_design        (UbD, ADDIE, SAM, Dick & Carey, Gagne, Merrill)
 3. multimedia_learning         (Mayer's principles, Dual coding)
 4. assessment_analytics        (Formative, Summative, Competency-based, Authentic, Learning analytics, Psychometrics)
 5. engagement_motivation       (Self-Determination, Gamification, Flow, Expectancy-Value, Attention/Engagement)
 6. emerging_technologies       (AI in education, Immersive learning, Microlearning, LXPs)
 7. social_learning             (Community of Inquiry, Peer learning, Online discussion, Social learning theory)
 8. hybrid_blended              (Blended models, Online course design)
 9. accessibility_udl           (UDL, Accessibility standards, Inclusive design)
10. quality_standards           (OSCQR, Quality Matters, Course evaluation)
11. learning_objectives         (Bloom's taxonomy, Writing objectives, Constructive alignment)
12. professional_development    (Digital literacy, Evidence-based practice, Faculty development)
```
Coverage thresholds (`thresholds:`): `excellent=0.8, good=0.6, partial=0.4, minimal=0.2`.

### LO `extractionSource` enum

Source: `schemas/academic/learning_objectives.schema.json:234`.
```
explicit, definition, concept, procedure, example, summary, inferred
```

### LO `assessmentSuggestions` enum

Source: `schemas/academic/learning_objectives.schema.json:223` (9 values).
```
exam, quiz, assignment, project, discussion, presentation, portfolio, demonstration, case_study
```

Courseforge's code-side per-Bloom map (`generate_course.py:533-540`) emits a narrower subset into page JSON-LD: `multiple_choice, true_false, fill_in_blank, short_answer, essay`.

### Edge-type enum

Source: `schemas/knowledge/concept_graph_semantic.schema.json:45`.
```
prerequisite, is-a, related-to
```
**Precedence on (source, target) collisions** (`Trainforge/rag/typed_edge_inference.py:48-52`): `is-a (3) > prerequisite (2) > related-to (1)`.

### Inference rule names

Source: `Trainforge/rag/inference_rules/*.py`:
- `is_a_from_key_terms` (version 1) — `is_a_from_key_terms.py:26-27`
- `prerequisite_from_lo_order` (version 1) — `prerequisite_from_lo_order.py:19-20`
- `related_from_cooccurrence` (version 1) — `related_from_cooccurrence.py:21-22`
- LLM escalation path (OFF by default) for "uncertain" pairs — `typed_edge_inference.py:15-24`.

### Validation status enum

Source: `schemas/library/catalog_entry.schema.json:55`, `schemas/library/course_manifest.schema.json:178`.
```
pending, validated, failed
```

### LibV2 ontology mapping types

Source: `schemas/library/course_manifest.schema.json:86-116`.
- **ACM CCS entry**: `{code, term, relevance ∈ [0, 1000]}`.
- **LCSH entry**: `{uri, term, type ∈ {primary, related, broader, narrower}}`.

### `audit_event.schema.json` enums

Source: lines 30-42, plus details oneOf variants.

**`event_type`:** `file_access, tool_invocation, state_change, decision_event, workflow_event, validation_event, error, security_event`.
**`file_access_details.operation`:** `read, write, delete, create, list`.
**`state_change_details.change_type`:** `create, update, delete, lock, unlock`.
**`workflow_event_details.event_subtype`:** `start, phase_start, phase_complete, task_dispatch, task_complete, complete, failed, aborted`.
**`security_event_details.security_event_type`:** `sandbox_violation, permission_denied, secret_detected, path_traversal_attempt`.

### `decision_event.schema.json` — decision_type enum (40)

Source: `schemas/events/decision_event.schema.json:63-103`.
```
approach_selection, strategy_decision, source_selection, source_interpretation,
textbook_integration, existing_content_usage, content_structure, content_depth,
content_adaptation, example_selection, pedagogical_strategy, assessment_design,
bloom_level_assignment, learning_objective_mapping, accessibility_measures,
format_decision, component_selection, quality_judgment, validation_result,
error_handling, prompt_response, file_creation, outcome_signal, chunk_selection,
question_generation, distractor_generation, revision_decision, source_usage,
alignment_check, structure_detection, heading_assignment, alt_text_generation,
math_conversion, research_approach, query_decomposition, retrieval_ranking,
result_fusion, chunk_deduplication, index_strategy
```

### `decision_event.schema.json` — phase enum (23)

Source: `schemas/events/decision_event.schema.json:53`.
```
input-research, exam-research, course-outliner, content-generator, brightspace-packager,
dart-conversion, dart-validation, trainforge-assessment, validation, content-analysis,
question-generation, assessment-assembly, courseforge-input-research, courseforge-exam-research,
courseforge-course-outliner, courseforge-content-generator, courseforge-brightspace-packager,
trainforge-content-analysis, trainforge-question-generation, trainforge-assessment-assembly,
trainforge-validation, libv2-retrieval, libv2-indexing, libv2-fusion
```
(plus `null` allowed.)

### Workflow type enum

Source: `schemas/events/run_manifest.schema.json:40`.
```
course_generation, intake_remediation, batch_dart, rag_training, textbook_to_course
```

### Session status / tool / quality-level enums

- Tool (`session_annotation.schema.json:18`, `decision_event.schema.json:56`): `dart, courseforge, trainforge, orchestrator`.
- Session `status` (`session_annotation.schema.json:46`): `running, complete, error, partial, cancelled`.
- Quality level (`decision_event.schema.json:281`): `exemplary, proficient, developing, inadequate`.
- Edit distance (`decision_event.schema.json:265`): `none, low, medium, high`.

### Hash algorithms

Source: `lib/provenance.py:19`; `schemas/events/run_manifest.schema.json:99`; `schemas/events/decision_event.schema.json:195,243`.
```
sha256 (default), sha512, blake3
```

---

## § 5 Provenance mechanisms

Five independent surfaces carry provenance. Each surface answers "where did this come from?" for a different artifact class.

### 5.1 Decision ledger (JSONL)

**Surface:** append-only JSONL files under `training-captures/<tool>/<course>/phase_<phase>/decisions_*.jsonl`.
**Producer:** `lib/decision_capture.py::DecisionCapture` (line 139; `log_decision` at line 441).
**Contract:** `schemas/events/decision_event.schema.json` (+ `trainforge_decision.schema.json`).
**Key fields for provenance:** `event_id`, `seq`, `run_id`, `task_id`, `inputs_ref[]` (InputRef), `outputs[]` (OutputArtifact), `prompt_ref`.

Every captured decision carries a `rationale ≥ 20 chars` (schema-enforced `minLength`) plus optional `alternatives_considered[]`.

### 5.2 Edge provenance

**Surface:** on every `TypedEdge` emitted by the concept-graph orchestrator.
**Producer:** the three inference rules (see § 4).
**Contract:** `schemas/knowledge/concept_graph_semantic.schema.json#/properties/edges/items/properties/provenance` — required `{rule, rule_version}`, optional `evidence` (arbitrary dict; e.g. for `is_a_from_key_terms`, `{container, member}`).

Document-level `rule_versions` object maps rule name → integer version (`concept_graph_semantic.schema.json:18-21`).

### 5.3 Chunk source provenance

**Surface:** `chunk.source` sub-object (see § 2 Chunk).
**Producer:** `Trainforge/process_course.py::_create_chunk()` lines 1057-1077.
**Key fields:** `course_id`, `module_id`, `lesson_id`, `section_heading`, `position_in_module`, optional `html_xpath` + `char_span` (Section 508 / ADA Title II round-trip: Worker E's audit trail), `item_path` (IMSCC-relative).

A downstream diagnostic field `chunk._metadata_trace` (added by Worker M1 at `process_course.py:1219`) records where each enrichment field came from (JSON-LD / data-cf-* / heuristic). This is flagged as a temporary diagnostic in the referenced ADR.

### 5.4 LibV2 course manifest provenance

**Surface:** `CourseManifest.provenance` (`schemas/library/course_manifest.schema.json:196-215`).
**Producer:** `LibV2/tools/libv2/importer.py`.
**Fields:** `source_type`, `source_path`, `original_provider`, `import_pipeline_version`.
In addition, `sourceforge_manifest` at the top level carries `{sourceforge_version, export_timestamp, course_id, course_title}` to link the manifest back to the producing Courseforge run.

### 5.5 Content-hash provenance

**Surface:** cryptographic hashes on files and excerpts.
**Producer:** `lib/provenance.py` — `hash_file()`, `create_input_ref()`, `create_output_ref()` (InputRef at line 37, OutputRef at line 91).
**Consumers:** `RunManifest.config_hashes` + `RunManifest.inputs[]` (immutable snapshot at run init); `DecisionEvent.inputs_ref[]` / `.outputs[]`; `AuditEvent.details.content_hash`.

Every content-hash entry carries `hash_algorithm ∈ {sha256, sha512, blake3}` (default sha256).

### Overlap table — which surface carries which fields

| Field | Decision ledger | Edge provenance | Chunk source | Manifest provenance | Content-hash |
|---|---|---|---|---|---|
| `run_id`          | yes | — | — | (via import run) | yes (RunManifest) |
| `rule` + version  | — | yes | — | — | — |
| `rule_versions` map | — | (document-level) | — | — | — |
| `html_xpath` + `char_span` | — | — | yes | — | — |
| `content_hash`    | yes (inputs_ref/outputs) | — | — | — | yes |
| `source_type`     | yes (inputs_ref) | — | (resource_type) | yes (provenance) | — |
| `import_pipeline_version` | — | — | — | yes | — |
| `prev_hash` / chain | — | — | — | — | yes (HashChainedEvent) |
| `alternatives_considered` | yes | — | — | — | — |
| `operator` / `git_commit` | — | — | — | — | yes (RunManifest) |

---

## § 6 Serialization surfaces

Five distinct serializations carry this ontology into bytes on disk. They are not interchangeable — each has a different audience.

### 6.1 JSON-LD emit (Courseforge Page metadata)

Emit site: `Courseforge/scripts/generate_course.py:263-279` (injects `<script type="application/ld+json">` into `<head>`); dict builder at `:571-595`.

Shape (trimmed):
```json
{
  "@context": "https://ed4all.dev/ns/courseforge/v1",
  "@type": "CourseModule",
  "courseCode": "WCAG_201",
  "weekNumber": 1,
  "moduleType": "content",
  "pageId": "week_01_content_01_intro",
  "learningObjectives": [
    {"id": "TO-01", "statement": "...", "bloomLevel": "understand",
     "bloomVerb": "explain", "cognitiveDomain": "conceptual",
     "keyConcepts": ["wcag", "aa"],
     "assessmentSuggestions": ["multiple_choice", "short_answer", "fill_in_blank"],
     "prerequisiteObjectives": ["TO-00"]}
  ],
  "sections": [
    {"heading": "WCAG 2.2 AA basics", "contentType": "definition",
     "keyTerms": [{"term": "perceivable", "definition": "..."}],
     "bloomRange": ["understand", "apply"]}
  ],
  "misconceptions": [{"misconception": "...", "correction": "..."}],
  "suggestedAssessmentTypes": ["multiple_choice", "true_false"]
}
```

Consumer: `Trainforge/process_course.py::_extract_section_metadata` (priority chain documented in `Trainforge/CLAUDE.md` metadata-extraction section: JSON-LD > `data-cf-*` > regex heuristic).

See § JSON-LD round-trip for the four `@context` files that wrap this and the other JSON artifacts as RDF (Phase 1.1–1.2 of the RDF/SHACL plan).

### 6.2 `data-cf-*` attribute vocabulary

See the full table in § 4. Emit sites: `Courseforge/scripts/generate_course.py` (`_render_objectives`, `_render_flip_cards`, `_render_self_check`, `_render_activities`, `_render_content_sections`). Consumer: `Trainforge/parsers/html_content_parser.py` extracts these attributes during chunk alignment; `_CHROME_TAGS` skip filter (`html_content_parser.py:140`) drops `data-cf-role="template-chrome"` subtrees so navigation headers/footers don't pollute chunks.

### 6.3 JSON Schema (draft-07) files

16 files across `schemas/`, loaded via `lib/validation.py:104` (`SCHEMAS_DIR.rglob("*.json")`). Full index in § 10.

Sizes (lines):

| Subfolder | Files | Total lines |
|---|---|---|
| academic/ | 3 | 1,395 |
| compliance/ | 1 | 1,131 |
| events/ | 6 | 1,108 |
| knowledge/ | 3 | 214 |
| library/ | 2 | 276 |
| taxonomies/ | 2 | 1,339 (json+yaml) |

### 6.4 Python dataclasses

| Dataclass | Location | Purpose |
|---|---|---|
| `Chunk` (dict shape; no dataclass yet) | `Trainforge/process_course.py:1079-1092` | unit of retrieval |
| `Question` | `Trainforge/generators/question_factory.py:36` | factory-side question |
| `QuestionChoice` | `question_factory.py:28` | choice option |
| `QuestionData` | `Trainforge/generators/assessment_generator.py:81` | generator-side question |
| `AssessmentData` | `assessment_generator.py:112` | assessment bundle |
| `InputRef` | `lib/provenance.py:37` | provenance input pointer |
| `OutputRef` | `lib/provenance.py:91` | provenance output pointer |
| `ByteRange` | `lib/provenance.py:27` | byte range within a file |
| `FsckIssue` / `FsckResult` | `lib/libv2_fsck.py:23,46` | fsck reporting |

### 6.5 JSONL ledger format

Append-only newline-delimited JSON files — one record per decision event. Example paths:
```
training-captures/courseforge/INT_101/phase_content-generator/decisions_20260419_101530.jsonl
training-captures/trainforge/WCAG_201/phase_question-generation/decisions_20260419_101530.jsonl
training-captures/dart/MTH_101/decisions_textbook.pdf_20260419_101530.jsonl
```

Hash-chained variant (`HashChainedEvent`) wraps each record with `{seq, prev_hash, event_hash, timestamp, event}` to make the ledger tamper-evident. `lib/hash_chain.py` is the writer; `lib/replay_engine.py` verifies chains on read.

---

## § 7 Identity & IDs

Every identifier scheme currently in use.

| ID type | Regex / pattern | Example | Generator |
|---|---|---|---|
| LibV2 course slug | `^[a-z0-9][a-z0-9-]*[a-z0-9]$`, 3–100 chars | `wcag-22-aa-compliance` | `LibV2/tools/libv2/importer.py:28` (`slugify`) |
| Slug-uniqueness | suffix `-<N>` where N ≥ 2 | `wcag-22-aa-compliance-2` | `importer.py:49` (`ensure_unique_slug`) |
| Course code | `^[A-Z]{2,8}_[0-9]{3}$` (decision event), `^[A-Z]{2,3}_[0-9]{3}$` (session annotation — narrower) | `WCAG_201`, `INT_101` | Hand-assigned |
| LO ID — terminal | `TO-NN` | `TO-05` | Course-outliner agent |
| LO ID — chapter | `CO-NN` | `CO-03` | Objective-synthesizer |
| LO ID — week-scoped (legacy) | `WNN-CO-NN` | `W03-CO-01` | Deprecated in favor of canonical CO-NN; week-prefix normalization at `generate_course.py:605-613` |
| Module ID | `^[A-Z]{2,8}_[0-9]{3}_W[0-9]{2}_M[0-9]{2}$` | `WCAG_201_W01_M02` | Course-outliner |
| Chunk ID | `^<course>_chunk_\d{5}$` | `wcag_201_chunk_00042` | `Trainforge/process_course.py:790` (`prefix = f"{self.course_code.lower()}_chunk_"`; `f"{prefix}{i:05d}"` at 1003, 1027) |
| Concept tag | kebab-case normalized slug | `cognitive-load-theory` | `process_course.py::normalize_tag` |
| Misconception ID | `<chunk_id>_mc_<NN>_<hash>` | `wcag_201_chunk_00042_mc_01_a3f8` | `preference_factory.py:140-143` |
| Event ID | `^EVT_[a-f0-9]{16}$` | `EVT_a3f8c1d2e4b5f6a7` | `lib/decision_capture.py:46-59` (fallback); `lib/sequence_manager.py` (primary) |
| Task ID | `^T-[a-f0-9]{8}$` | `T-a3f8c1d2` | Orchestrator executor |
| Run ID (tool-scoped) | `{TOOL}_{COURSE}_{YYYYMMDD_HHMMSS}` (free text; `^[A-Za-z0-9_]+$` on audit event) | `trainforge_wcag_201_20260419_101530` | `lib/run_manager.py` |
| Run ID (hardened) | `^RUN_[0-9]{8}_[0-9]{6}_[a-f0-9]{8}$` | `RUN_20260419_101530_a3f8c1d2` | `lib/run_manager.py` (hardened mode) |
| Content hash | 64-hex for SHA-256; also `sha256:<hex>` prefix form in run manifest | `sha256:a3f8…` | `lib/provenance.py::hash_file` |
| Git commit | 40-hex | (40 hex chars) | run-manifest capture |
| Session ID | free-string | (tool-specific) | `lib/run_finalizer.py` |

No IRIs are minted yet. JSON-LD `@context` points at `https://ed4all.dev/ns/courseforge/v1` (non-resolvable at time of writing). LibV2 JSON Schemas use `$id: https://libv2.local/schema/<name>.schema.json`; project-root schemas use `urn:ed4all:schemas:<name>` or `https://ed4all.dev/schemas/…` depending on file age.

---

## § 8 Constraint inventory

Where the ontology is actually enforced.

### 8.1 JSON Schema constraints

Per-file counts of constraint keywords (approximate, grep-based):

| File | `required` blocks | enums | patterns / formats | min/max | Notable |
|---|---|---|---|---|---|
| `academic/course_metadata.schema.json` | many | `courseLevel` 4, `grade` 10, `unit` 4, `format` 4, `pacing` 3, many more | `courseNumber` pattern, ISBN pattern, `version` semver pattern | credit 0–20, duration 1–52, `learningOutcomes` 3–12, rationale counts | MIT OCW shape |
| `academic/learning_objectives.schema.json` | LO required {objectiveId, statement, bloomLevel} | 6-Bloom, 9-assessment, 7-extractionSource | chapterId `^ch[0-9]+$`, sectionId `^ch[0-9]+_s[0-9]+$` | courseObjectives 3–15, statement ≥10 | |
| `academic/textbook_structure.schema.json` | `documentInfo` {title, sourcePath, sourceFormat} | 3-sourceFormat, 14-blockType, 4-sourceType (explicit objectives), 4-emphasisType | level 1–6 | | |
| `compliance/wcag22_compliance.schema.json` | `complianceLevel.standard` | 3-standard, 5-automated, 6-manual, 4-labelAssociation, many booleans | focusAppearance contrast ≥3.0, minOutlineWidth ≥2, targetSize ≥24 | | 1131 lines total |
| `events/decision_event.schema.json` | run_id, timestamp, operation, decision_type, decision, rationale | 40-decision_type, 23-phase, 8-source_type, 4-hash_algorithm, 3-edit_distance, 4-quality_level, 4-tool | event_id, module_id, course_id, task_id, content_hash patterns | rationale ≥20, confidence 0–1 | rationale minLength is the headline gate |
| `events/trainforge_decision.schema.json` | imscc_source | 6-bloom, 8-question_type, 3-difficulty | | `lo_coverage_score` 0–1 etc. | allOf ref to decision_event |
| `events/audit_event.schema.json` | run_id, event_id, seq, timestamp, event_type | 8-event_type, 5-file_op, 5-state_change, 8-workflow_subtype, 4-security_type | event_id `^EVT_…`, content_hash, run_id alphanumeric | seq ≥0 | `additionalProperties: false` at root |
| `events/hash_chained_event.schema.json` | seq, prev_hash, event_hash, timestamp, event | — | `prev_hash = "genesis"` OR 64-hex; `event_hash` 64-hex | seq ≥0 | tamper-evident |
| `events/run_manifest.schema.json` | run_id, created_at, workflow_type, config_hashes, immutable | 5-workflow_type, 3-hash_algorithm | run_id `^RUN_…`, git_commit 40-hex, content_hash 64-hex | `immutable const: true` | |
| `events/session_annotation.schema.json` | session_id, run_id, tool, started_at, status | 4-tool, 5-status, 4-quality_level | course_id `^[A-Z]{2,3}_[0-9]{3}$` | validation_pass_rate 0–1 etc. | |
| `knowledge/concept_graph_semantic.schema.json` | kind, generated_at, nodes, edges; edge {source,target,type,provenance}; provenance {rule, rule_version} | 3-edge-type | kind const `concept_semantic`; rule_version ≥1 | confidence 0–1 | |
| `knowledge/instruction_pair.schema.json` | prompt, completion, chunk_id, lo_refs, bloom_level, content_type, seed, decision_capture_id | 6-bloom, 2-provider | schema_version const `"v1"` | prompt 40–400, completion 50–600, lo_refs ≥1 item, content_type ≥1 char | |
| `knowledge/preference_pair.schema.json` | prompt, chosen, rejected, chunk_id, lo_refs, seed, decision_capture_id | 2-rejected_source, 2-provider | `misconception_id` nullable | same length bounds as above | token-Jaccard δ ≥ 0.3 asserted in test-time, not schema |
| `library/catalog_entry.schema.json` | slug, title, division, primary_domain | 2-division, 4-difficulty_primary, 3-validation_status | | chunk_count ≥0 etc. | |
| `library/course_manifest.schema.json` | libv2_version, slug, import_timestamp, sourceforge_manifest {…}, classification {division, primary_domain}, content_profile {total_chunks, total_tokens} | 2-division, 4-LCSH relation type | slug pattern, libv2_version semver, language `^[a-z]{2}$` | slug 3–100 chars, ACM relevance 0–1000, 0–1 quality scores | |

### 8.2 Runtime validators (9 files in `lib/validators/`)

| File | Class | Gate behavior |
|---|---|---|
| `assessment.py` | `AssessmentQualityValidator` | assessment-level quality gate (`gate_id: assessment_quality`) |
| `bloom.py` | `BloomAlignmentValidator` + `detect_bloom_level` helper | detects Bloom level from question stem; validates alignment (`gate_id: bloom_alignment`) |
| `content.py` | `ContentStructureValidator` | HTML structure for course modules (`gate_id: content_structure`) |
| `content_facts.py` | `ContentFactValidator` (+ `FactFlag`) | text factual-accuracy scanning — **warning-only** (§4.6) |
| `imscc.py` | `IMSCCValidator` | IMSCC package structure (`gate_id: imscc_structure`) |
| `leak_check.py` | `LeakCheckValidator` (wraps `lib.leak_checker.LeakChecker`) | answer-key leak detection (`gate_id: leak_check`) |
| `oscqr.py` | `OSCQRValidator` | OSCQR rubric score (`gate_id: oscqr_score`) |
| `question_quality.py` | `QuestionQualityValidator` | per-question quality (`gate_id: question_quality`) plus stop-word / Jaccard helpers |
| `__init__.py` | package init | exposes validators |

### 8.3 Validation gates (workflow → gate → validator)

From root `/CLAUDE.md` § Active Gates:

| Workflow | Gate | Validator |
|---|---|---|
| `course_generation` | `content_structure` | `ContentStructureValidator` |
| `course_generation` | `imscc_structure` | `IMSCCValidator` |
| `course_generation` | `wcag_compliance` | `WCAGValidator` (Wave 31 — semantic upgrade) |
| `course_generation` | `oscqr_score` | `OSCQRValidator` (Wave 31 — real impl) |
| `batch_dart` | `wcag_aa_compliance` | `WCAGValidator` (Wave 31 — semantic upgrade) |
| `textbook_to_course` | `content_grounding` | `ContentGroundingValidator` (Wave 31) |
| `rag_training` | `assessment_quality` | `AssessmentQualityValidator` |
| `rag_training` | `bloom_alignment` | `BloomAlignmentValidator` |
| `rag_training` | `leak_check` | `LeakChecker` |

Severity behavior (`critical | warning`), block/warn on fail, fail-closed vs warn on error. Defined in workflow config (`config/workflows.yaml`).

### 8.4 Test-time assertions

- `Trainforge/tests/test_generator_defects.py:272` — asserts `metrics_semantic_version == METRICS_SEMANTIC_VERSION` (current value 5).
- Preference-pair token-Jaccard δ ≥ 0.3 between `chosen` and `rejected` (not schema-enforced; asserted in generator + tests).
- Instruction-pair verbatim-span check: prompt/completion must not contain ≥50-char substring of source chunk (schema description at `schemas/knowledge/instruction_pair.schema.json:23`; enforced in `Trainforge/generators/instruction_factory.py`).

### 8.5 LibV2 fsck checks

Source: `lib/libv2_fsck.py:103` (`check_all`).

- `_check_blobs` (line 146) — every blob's SHA-256 hash matches its filename.
- `_check_catalog` (line 188) — catalog `course_index.json` consistency.
- `_check_runs` (line 240) — every `runs/*/` has a valid `run_manifest.json`.
- `_check_symlinks` (line 127) — symlinks resolve.
- `_check_orphans` (line 130) — detects files not referenced by any index.
- Cross-package concept index staleness (Worker G) (line 132).

---

## § 9 Versioning

Every version counter currently in use:

| Counter | Current value | Location |
|---|---|---|
| `CHUNK_SCHEMA_VERSION` | `"v4"` | `Trainforge/process_course.py:86`; written to every chunk `schema_version` field (line 1081) and quality report (lines 1749, 1779) |
| `METRICS_SEMANTIC_VERSION` | `5` | `Trainforge/process_course.py:74` (Worker P bumped 4→5 for `package_completeness`) |
| Inference `RULE_VERSION` | `1` (each) | `Trainforge/rag/inference_rules/*.py` — per-rule integer; on each edge as `provenance.rule_version` |
| `rule_versions` object | `{rule_name: int ≥1}` | `schemas/knowledge/concept_graph_semantic.schema.json:17-21` — document-level map |
| `schema_version` (decision event) | free string; defaults vary | `lib/decision_capture.py` sets `decision_event` default |
| `schema_version` (RunManifest) | `"1.0.0"` default | `schemas/events/run_manifest.schema.json:114` |
| `schema_version` (InstructionPair / PreferencePair) | const `"v1"` | `schemas/knowledge/{instruction,preference}_pair.schema.json` |
| `metadata.version` (Course) | semver `^\d+\.\d+\.\d+$` | `schemas/academic/course_metadata.schema.json:714` |
| `libv2_version` (CourseManifest) | semver `^\d+\.\d+\.\d+$` | `schemas/library/course_manifest.schema.json:11` |
| `toolVersion` (LearningObjectives document) | semver | `schemas/academic/learning_objectives.schema.json:37` |
| Schema title `version` fields | `"1.0.0"` / `"2.0.0"` (WCAG) | top-of-schema JSON comments |

---

## § 10 Schema file index

`/schemas/` tree (canonical shapes, loadable from any project):

| Path | Classes defined | Description |
|---|---|---|
| `schemas/academic/course_metadata.schema.json` | Course, Module | Full academic course metadata (MIT OCW shape) |
| `schemas/academic/learning_objectives.schema.json` | LearningObjective | Extracted LO hierarchy (course/chapter/section/subsection) |
| `schemas/academic/textbook_structure.schema.json` | Section, ContentBlock, TOC entry, key-term / definition / procedure / example records | DART-processed textbook semantic structure |
| `schemas/compliance/wcag22_compliance.schema.json` | WCAGCompliance | WCAG 2.2 AA requirements (4 principles + SC codes) |
| `schemas/config/workflows_meta.schema.json` | WorkflowMeta | Meta-schema for `config/workflows.yaml` (phase routing, gate shape, inputs_from references) |
| `schemas/events/decision_event.schema.json` | DecisionEvent | Base Claude-decision ledger record |
| `schemas/events/trainforge_decision.schema.json` | TrainforgeDecisionEvent | Decision ledger + assessment/rag/alignment context (allOf) |
| `schemas/events/audit_event.schema.json` | AuditEvent | Unified audit event w/ event-type-keyed details |
| `schemas/events/hash_chained_event.schema.json` | HashChainedEvent | Tamper-evident hash chain wrapper |
| `schemas/events/session_annotation.schema.json` | SessionAnnotation | Aggregated session summary across decisions |
| `schemas/events/run_manifest.schema.json` | RunManifest | Immutable run initialization snapshot |
| `schemas/knowledge/chunk_v4.schema.json` | Chunk | Trainforge chunk contract (gated by `TRAINFORGE_VALIDATE_CHUNKS`) |
| `schemas/knowledge/concept_graph_semantic.schema.json` | Concept, TypedEdge, rule_versions | Typed-edge concept graph (8 edge types, per-rule evidence discriminator) |
| `schemas/knowledge/course.schema.json` | Course (course.json) | Canonical shape for Trainforge-emitted `course.json`, consumed by LibV2 retrieval |
| `schemas/knowledge/courseforge_jsonld_v1.schema.json` | CourseforgePage | JSON-LD block emitted per Courseforge HTML page |
| `schemas/knowledge/instruction_pair.schema.json` | InstructionPair | SFT training pair (prompt/completion) |
| `schemas/knowledge/instruction_pair.strict.schema.json` | InstructionPair (strict) | Opt-in strict variant of the SFT pair schema |
| `schemas/knowledge/misconception.schema.json` | Misconception | First-class misconception entity (content-hash IDs) |
| `schemas/knowledge/preference_pair.schema.json` | PreferencePair | DPO training pair (chosen/rejected) |
| `schemas/knowledge/source_reference.schema.json` | SourceReference | Canonical `{sourceId, role, weight?, confidence?, pages?, extractor?}` shape shared across DART / Courseforge / Trainforge |
| `schemas/library/catalog_entry.schema.json` | CatalogEntry | LibV2 master-catalog row |
| `schemas/library/course_manifest.schema.json` | CourseManifest | LibV2 extended course metadata (validated by `LibV2ManifestValidator`) |
| `schemas/taxonomies/taxonomy.json` | (STEM/ARTS data; not a schema) | Division/domain/subdomain/topic hierarchy |
| `schemas/taxonomies/pedagogy_framework.yaml` | (pedagogy framework data) | 12-tier pedagogy gap-analysis framework |
| `schemas/taxonomies/bloom_verbs.json` | (taxonomy) | 60-verb / 6-level authoritative Bloom's list |
| `schemas/taxonomies/question_type.json` | (taxonomy) | 7-value question-factory enum |
| `schemas/taxonomies/assessment_method.json` | (taxonomy) | formative / summative / diagnostic |
| `schemas/taxonomies/content_type.json` | (taxonomy) | 8-value section classification enum |
| `schemas/taxonomies/cognitive_domain.json` | (taxonomy) | factual / conceptual / procedural / metacognitive |
| `schemas/taxonomies/teaching_role.json` | (taxonomy) | (component, purpose) → role mapping |
| `schemas/taxonomies/module_type.json` | (taxonomy) | 6-value moduleType enum |

Tool-local schemas (NOT under `/schemas/`):

| Path | Classes defined |
|---|---|
| `Courseforge/schemas/content-display/accordion-schema.json` | Accordion |
| `Courseforge/schemas/content-display/content-display-schema.json` | ContentDisplay |
| `Courseforge/schemas/content-display/enhanced-content-display-schema.json` | EnhancedContentDisplay |
| `Courseforge/schemas/content-display/page-title-standards.json` | PageTitleStandards |
| `Courseforge/schemas/layouts/course_card_schema.json` | CourseCard |
| `Courseforge/schemas/template-integration/educational_template_schema.json` | EducationalTemplate |
| `Courseforge/schemas/framework-migration/bootstrap5_migration_schema.json` | Bootstrap5Migration |

---

## § 11 Code pointers

Exact file:line anchors to key emit/consume sites. Grep-verified against the tree at snapshot time.

### Emit — Courseforge

- **Page JSON-LD injection:** `Courseforge/scripts/generate_course.py:273-279` (wraps dict from `_build_page_metadata`).
- **`_build_page_metadata`:** `Courseforge/scripts/generate_course.py:571-595`.
- **`_build_objectives_metadata`:** `Courseforge/scripts/generate_course.py:512-546`.
- **`_build_sections_metadata`:** `Courseforge/scripts/generate_course.py:549-568`.
- **`_render_objectives`** (data-cf-objective-id + data-cf-bloom-level/verb/cognitive-domain): `generate_course.py:305-333`.
- **`_render_flip_cards`** (data-cf-component=flip-card): `generate_course.py:336-352`.
- **`_render_self_check`** (data-cf-component=self-check + objective-ref): `generate_course.py:355-385`.
- **`_render_content_sections`** (data-cf-content-type + key-terms + bloom-range + application-note callout): `generate_course.py:408-477`.
- **`_render_activities`** (data-cf-component=activity): `generate_course.py:480-497`.
- **Template-chrome tags:** `generate_course.py:289, 290, 297` (`data-cf-role="template-chrome"` on skip-link, header, footer).
- **`BLOOM_VERBS` / `BLOOM_TO_DOMAIN` / `detect_bloom_level`:** `generate_course.py:136-166`.
- **`generate_week` (5-page emit):** `generate_course.py:598-738`.

### Consume — Trainforge

- **`_chunk_content`:** `Trainforge/process_course.py:787` — builds prefix, iterates parsed items.
- **`_chunk_text_block`:** `Trainforge/process_course.py:932`.
- **`_create_chunk`:** `Trainforge/process_course.py:1038` — full chunk shape (lines 1079-1092) + audit-trail source (1067-1077).
- **`CHUNK_SCHEMA_VERSION`:** `Trainforge/process_course.py:86`.
- **`METRICS_SEMANTIC_VERSION`:** `Trainforge/process_course.py:74`.
- **`_metadata_trace` (Worker M1 diagnostic):** `Trainforge/process_course.py:1219`.
- **Metadata extraction priority chain (JSON-LD → data-cf-* → heuristic):** `Trainforge/process_course.py:1099-1130` (`_extract_section_metadata`).
- **Typed-edge orchestrator:** `Trainforge/rag/typed_edge_inference.py:1-24` (module docstring), precedence at lines 48-52, collision dedupe at 71-100.
- **Inference rule: is-a from key terms:** `Trainforge/rag/inference_rules/is_a_from_key_terms.py:26-27` (name + version), emit at lines 155-165.
- **Inference rule: prerequisite from LO order:** `Trainforge/rag/inference_rules/prerequisite_from_lo_order.py:19-20`, emit at 140-150.
- **Inference rule: related from co-occurrence:** `Trainforge/rag/inference_rules/related_from_cooccurrence.py:21-22`, emit at 70-80.
- **Template-chrome skip filter:** `Trainforge/parsers/html_content_parser.py:64, 78, 84, 102, 109, 138, 140` (`_CHROME_TAGS = {"header","footer","a","div","nav","aside"}`).

### Question generation

- **`BLOOM_QUESTION_MAP`:** `Trainforge/generators/question_factory.py:91-98`.
- **`VALID_TYPES` (7-value factory enum):** `question_factory.py:81-89`.
- **`QuestionChoice` / `Question` dataclasses:** `question_factory.py:28, 36`.
- **`QuestionData` / `AssessmentData` dataclasses:** `Trainforge/generators/assessment_generator.py:81, 112`.
- **Misconception ID generator:** `Trainforge/generators/preference_factory.py:140-143`.

### Provenance + ledger

- **`DecisionCapture`:** `lib/decision_capture.py:139`; `log_decision` at line 441; legacy output dir at line 187.
- **`InputRef` / `OutputRef` / `ByteRange`:** `lib/provenance.py:27, 37, 91`; hash-algorithm constants at line 19.
- **Hash chain:** `lib/hash_chain.py` (writer); `lib/replay_engine.py` (verifier).
- **Schema resolver constants:** `lib/validation.py:24-26` (`DECISION_SCHEMA_PATH`, `TRAINFORGE_SCHEMA_PATH`, `SESSION_SCHEMA_PATH`); recursive discovery at `:104`.
- **`SCHEMAS_DIR`:** `lib/path_constants.py:87`.

### LibV2

- **Slugify:** `LibV2/tools/libv2/importer.py:28` (`slugify`) and `:49` (`ensure_unique_slug`).
- **File checksum (sha256 w/ prefix):** `LibV2/tools/libv2/importer.py:71-77`.
- **fsck entry:** `lib/libv2_fsck.py:103` (`check_all`); sub-checks at lines 146, 188, 240, 127, 130, 132.

---

## § 12 v0.2.0 changes (Waves 1–6 summary)

Additive section. Descriptive snapshot of what Waves 1–6 (commits `fea48f8` → post-Worker-V/W Wave 6 merges on `dev-v0.2.0`) added to the Ed4All ontology. No goal-setting, no gap analysis — just what's in the tree now that wasn't there at v0.2.0-alpha. For full rationale per recommendation, see `plans/kg-quality-review-2026-04/review.md` and the per-worker sub-plans under `plans/kg-quality-review-2026-04/`.

### Wave-by-wave headline

| Wave | Scope | Representative workers / PRs | KG impact |
| ---- | ----- | ---------------------------- | --------- |
| 1 | Decision-capture hardening, schema-directory unification, slug / Bloom / teaching-role canonicalization, 44-value `decision_type` enum | Workers A–G (PRs #14–22) | Canonical ID pipeline; fewer emit-side forks. |
| 2 | Content-type enum; page-objectives validator; DART marker validation tool | Workers H–L (PRs #23–26) | Section metadata constrained; objective coverage enforced per page. |
| 3 | Typed-edge precedence + dedupe; fixture-driven golden tuples; preservation of LO-ref case | Worker M (PR #24) | 3-tier typed-edge artifact stabilised. |
| 4 | Opt-in content-hash chunk IDs; opt-in course-scoped concept IDs; run_id + created_at provenance fields on chunks/nodes/edges; courseforge_jsonld_v1 schema; strict instruction-pair variant; first-class Misconception schema | Workers N–R (PRs #27–31) | Provenance + stability surface complete; Misconception promoted from inline field to standalone entity. |
| 5 | `occurrences[]` back-reference on concept nodes; opt-in `content_type` enforcement; 5 pedagogical edge types (assesses, exemplifies, misconception-of, derived-from-objective, defined-by) | Workers S–U (PRs #33–35, commit `5bf2c9a`) | Graph expanded to 8 edge types; concept→chunk inverted index published. |
| 6 | Workflow meta-schema (`config/workflows_meta.schema.json`) + YAML phase routing; `validate_dart_markers` wired as validation gate; per-rule evidence discriminator on `concept_graph_semantic`; removal of emit-only `data-cf-objectives-count`; agent-doc sweep; this ONTOLOGY refresh | Workers V–W (Wave 6) | Governance surface tightened; evidence becomes typed per rule; docs reflect actual ontology. |

### Taxonomies added (`schemas/taxonomies/`)

Eight canonical taxonomy files ship under `schemas/taxonomies/` — all loadable via `lib/ontology/taxonomy.py::load_taxonomy(name)`:

- `bloom_verbs.json` — 60-verb / 6-level authoritative list (loader: `lib/ontology/bloom.py`).
- `question_type.json` — 7-value factory enum for assessments (MCQ, TF, short-answer, essay, fill-in-blank, matching, numeric).
- `assessment_method.json` — formative / summative / diagnostic split.
- `content_type.json` — 8-value section classification (explanation, example, procedure, definition, callout, assessment, summary, introduction).
- `cognitive_domain.json` — factual / conceptual / procedural / metacognitive (Anderson-Krathwohl).
- `teaching_role.json` — (component, purpose) → role mapping; loader: `lib/ontology/teaching_roles.py`.
- `module_type.json` — 6-value enum (including `discussion`, surfaced by Worker B and schematized by Worker F).
- `taxonomy.json` (pre-existing) — STEM/ARTS division hierarchy.
- `pedagogy_framework.yaml` (pre-existing) — 12-tier gap framework.

### Knowledge schemas added (`schemas/knowledge/`)

- `courseforge_jsonld_v1.schema.json` — canonical shape of the JSON-LD block emitted by `generate_course.py::_build_page_metadata`.
- `chunk_v4.schema.json` — Trainforge chunk contract (opt-in enforcement via `TRAINFORGE_VALIDATE_CHUNKS=true`).
- `misconception.schema.json` — first-class Misconception entity. IDs follow `mc_[0-9a-f]{16}` pattern (content hash). Optional `concept_id` / `lo_id` links enable explicit misconception-of edges.
- `instruction_pair.strict.schema.json` — opt-in strict variant of the SFT pair schema.
- `concept_graph_semantic.schema.json` (modified) — now carries a `oneOf` discriminator on `edges[].provenance` keyed by rule name (see § 12 — per-rule evidence discriminator below).
- `preference_pair.schema.json` (pre-existing) — DPO pairs.

### Config meta-schema (`schemas/config/`)

- `workflows_meta.schema.json` — validates `config/workflows.yaml` at load time. Validates top-level keys, per-workflow `phases[]`, gate shape (`gate_id`, `validator`, `severity`, `threshold`, `behavior`), `inputs_from` referencing prior-phase `outputs`, and no-duplicate-phase-names. Landed in Wave 6 (Worker V). Implemented via `_load_phase_routing_from_yaml()` in `MCP/core/workflow_runner.py`.

### 6-value `moduleType`

`moduleType` enum, cited on the Page class in § 2, is now `{overview, content, application, self_check, summary, discussion}`. The `discussion` value was surfaced by Worker B (decision capture), schematized by Worker F, and persists in `schemas/taxonomies/module_type.json`.

### 10 edge types in the concept graph

Three taxonomic edges (unchanged):
- `is-a` — from `is_a_from_key_terms` rule. **Reserved for class-level subsumption.** When both endpoints are `cf:Concept` instances (the canonical case), the rule emits `broader-than` instead — see SKOS edges below.
- `prerequisite` — from `prerequisite_from_lo_order` rule. The underlying `cf:hasPrerequisite` predicate is declared `owl:TransitiveProperty` and paired with `cf:isPrerequisiteOf` via `owl:inverseOf` (Wave 86), so OWL-RL reasoning materializes indirect prereq chains for free.
- `related-to` — from `related_from_cooccurrence` rule (co-occurrence reuse).

Five pedagogical edges (Wave 5, Worker U):
- `assesses` — question → LO (`assesses_from_question_lo`).
- `exemplifies` — chunk → concept, when chunk is flagged as example (`exemplifies_from_example_chunks`).
- `misconception-of` — misconception → concept, from Misconception entity's `concept_id` (`misconception_of_from_misconception_ref`).
- `derived-from-objective` — chunk → LO (`derived_from_lo_ref`, materializes existing `learning_outcome_refs`).
- `defined-by` — concept → chunk, using `occurrences[0]` as canonical first mention (`defined_by_from_first_mention`).

Two SKOS edges (Wave 86):
- `broader-than` → `skos:broader`. Concept-layer hierarchy. Emitted by `is_a_from_key_terms` when both endpoints are `cf:Concept` instances (the W3C-canonical concept-hierarchy pattern). The previous behavior collapsed concept-layer subsumption onto class-level `is-a`; the split lets SKOS-aware consumers walk the hierarchy via standard SPARQL property paths.
- `narrower-than` → `skos:narrower`. Reserved for the inverse direction; not currently emitted by any rule but registered in `lib/ontology/edge_predicates.py::SLUG_TO_IRI` for future taxonomy-expansion work.

All ten are listed in `concept_graph_semantic.schema.json::properties.edges.items.properties.type.enum`. Heterogeneous endpoints (chunk IDs, LO IDs, misconception IDs, question IDs, concept IDs) are federated-by-convention: consumers resolve by ID-namespace prefix; no new node types are added to the schema.

### Per-rule evidence discriminator (REC-PRV-02, Wave 6)

`edges[].provenance` carries a `oneOf` discriminator with nine arms. Each of the 8 modeled rules has a `{Rule}Provenance` arm binding `rule = {name}` (via `const`) to a specific evidence `$def`:

| Rule | Evidence `$def` | Required fields |
| ---- | --------------- | --------------- |
| `is_a_from_key_terms` | `IsAEvidence` | chunk_id, term, definition_excerpt, pattern |
| `prerequisite_from_lo_order` | `PrerequisiteEvidence` | target_first_lo, target_first_lo_position, source_first_lo, source_first_lo_position |
| `related_from_cooccurrence` | `RelatedEvidence` | cooccurrence_weight, threshold |
| `assesses_from_question_lo` | `AssessesEvidence` | question_id, objective_id (+ optional source_chunk_id) |
| `exemplifies_from_example_chunks` | `ExemplifiesEvidence` | chunk_id, concept_slug, content_type ∈ {content_type_label, chunk_type} |
| `misconception_of_from_misconception_ref` | `MisconceptionOfEvidence` | misconception_id, concept_id |
| `derived_from_lo_ref` | `DerivedFromObjectiveEvidence` | chunk_id, objective_id |
| `defined_by_from_first_mention` | `DefinedByEvidence` | chunk_id, concept_slug, first_mention_position |

The 9th arm (`FallbackProvenance`) matches any rule NOT in that list via `not: enum`, accepting any evidence shape. That keeps the default **lenient** — preserves backward-compat for legacy graphs and future rules. Strict mode is opt-in via `TRAINFORGE_STRICT_EVIDENCE=true` or `lib/validators/evidence.py::get_schema(strict=True)`; strict strips FallbackProvenance so unknown rules + shape-drifting known rules fail validation.

### First-class `Misconception` entity

Misconceptions moved from a Section field to a standalone entity (`schemas/knowledge/misconception.schema.json`, Worker R). Required fields: `id` (pattern `^mc_[0-9a-f]{16}$`, derived from a content hash of `{concept_id, misconception_text, correction}`), `concept_id` (optional link to concept node), `lo_id` (optional link), `misconception_text`, `correction`, provenance block. Materialized as `misconception-of` edges when `concept_id` is populated (see above).

### `occurrences[]` back-reference on concept nodes

Concept nodes optionally carry `occurrences: List[str]` — the sorted-ASC list of chunk IDs that reference the concept. Populated from a chunk→concept inverted index at `_build_tag_graph` time (Wave 5, Worker S). Consumed by `defined_by_from_first_mention` for its canonical first-mention edge. Stability depends on chunk-ID scheme: position-based IDs (default) invalidate occurrences on re-chunk; content-hash IDs (`TRAINFORGE_CONTENT_HASH_IDS=true`, Wave 4 Worker N) keep occurrences stable.

### Opt-in flags (behavior toggles)

Eleven environment-variable flags gate opt-in behavior to preserve backward-compat with legacy corpora. All default off; each flag represents a toggle a regeneration run can flip to enforce the newer contract.

| Flag | When on |
| ---- | ------- |
| `TRAINFORGE_CONTENT_HASH_IDS` | Chunk IDs are content hashes instead of positional indices — re-chunk-stable. |
| `TRAINFORGE_SCOPE_CONCEPT_IDS` | Concept node IDs become `{course_id}:{slug}` — allows cross-course disambiguation. |
| `TRAINFORGE_PRESERVE_LO_CASE` | LO references retain emit case (e.g. `TO-01`) instead of lower-casing (`to-01`). |
| `TRAINFORGE_VALIDATE_CHUNKS` | Enforces `chunk_v4.schema.json` on every chunk write; fails closed on shape drift. |
| `TRAINFORGE_ENFORCE_CONTENT_TYPE` | Constrains `content_type_label` values to the `content_type.json` enum; fails closed on unknowns. |
| `TRAINFORGE_STRICT_EVIDENCE` | Strips FallbackProvenance from the evidence discriminator; unknown rules + shape-drifting known rules fail validation. |
| `TRAINFORGE_SOURCE_PROVENANCE` | Evidence arms emit `source_references[]` sourced from chunks' `source.source_references[]`. Off: arms emit the pre-provenance shape. |
| `DECISION_VALIDATION_STRICT` | Fails closed on unknown `decision_type` values in decision captures. |
| `DART_LLM_CLASSIFICATION` | DART's block classifier routes through Claude via `LLMClassifier` instead of heuristic regex. Requires an injected `LLMBackend`. |
| `LOCAL_DISPATCHER_ALLOW_STUB` | Permits `LocalDispatcher` to emit a stubbed `PhaseOutput` when no `agent_tool` callable is wired in. Tests / dry-run only; production `--mode local` runs fail loudly without it set. |

### Always-emit provenance fields

Three artifacts unconditionally carry `run_id` + `created_at` (REC-PRV-01, Wave 4.1 Worker P):

- Chunks (`Trainforge/process_course.py::_create_chunk`).
- Concept nodes (`Trainforge/rag/typed_edge_inference.py`).
- Concept edges (same).

Both sourced from the active `DecisionCapture` instance. Legacy artifacts without these fields still validate under the schemas (backward-compat retained).

### Canonical helpers (`lib/ontology/`)

All loaders read from `schemas/taxonomies/`; single source of truth per domain:

- `lib/ontology/slugs.py::canonical_slug` — unified slug helper (REC-ID-03, Wave 4 Worker Q).
- `lib/ontology/bloom.py` — `get_verbs()`, `get_verbs_list()`, `get_verb_objects()`, `get_all_verbs()`, `detect_bloom_level()`.
- `lib/ontology/teaching_roles.py` — `(component, purpose) → role` mapper.
- `lib/ontology/taxonomy.py::load_taxonomy(name)` — generic JSON-taxonomy loader.

### Validators consolidated (`lib/validators/`)

- `lib/validators/page_objectives.py` (Worker L) — wrapped as the `page_objectives` validation gate on packaging.
- `lib/validators/content_type.py` (Worker T) — gated by `TRAINFORGE_ENFORCE_CONTENT_TYPE=true`.
- `lib/validators/evidence.py` (Worker W) — thin loader over `concept_graph_semantic.schema.json`; strict-mode opt-in removes the fallback arm.

### New validation gates

Two new gates wire into `config/workflows.yaml` (Worker V, Wave 6):

- `page_objectives` on the `packaging` phase of `course_generation` (Wave 2, Worker L).
- `dart_markers` on the `dart_conversion` phase of `batch_dart` + `textbook_to_course` (REC-CTR-06, Wave 6 Worker V).

### Decision type enum expansion

`decision_type` enum in `schemas/events/decision_event.schema.json` grew from 39 → 44 values. Five additions (Wave 1, Worker G): `concept_graph_publish`, `chunk_validation_failure`, `opt_in_flag_override`, `typed_edge_dedup`, `evidence_discriminator_fallback`. The `DECISION_VALIDATION_STRICT=true` flag (above) enforces the enum at write time; default is lenient (unknown values pass with a warning).

### Wave 8 changes — DART source provenance

Wave 8 lands the shared `SourceReference` shape and threads per-block provenance through DART's emit path:

- **New canonical schema**: `schemas/knowledge/source_reference.schema.json` — `{sourceId, role, weight?, confidence?, pages?, extractor?}`. `sourceId` matches `^dart:[a-z0-9_-]+#[a-z0-9_-]+$`. Shared by Courseforge JSON-LD (Wave 9) and Trainforge chunks + evidence (Waves 10–11).
- **DART per-section record** gains a `provenance: {sources, strategy, confidence}` block + `section_id` + `page_range`. Legacy `sources_used` retained for back-compat.
- **DART per-block envelope** — every leaf in matcher output (`synthesize_contacts`, `synthesize_systems_table`, `synthesize_roster`, campus-info/credentials matchers) carries a `{value, source, pages, confidence, method}` envelope + a `block_id` (positional `s3_c0` or content-hash under `TRAINFORGE_CONTENT_HASH_IDS=1`).
- **`data-dart-*` HTML attributes**: `data-dart-block-id`, `data-dart-source`, `data-dart-sources`, `data-dart-pages`, `data-dart-confidence`, `data-dart-strategy` on every `<section>` + `.contact-card`. Scoped per P2 decision — never on per-`<p>`/`<li>`/`<tr>` children.
- **Staging handoff fix**: `stage_dart_outputs` now role-tags manifest entries as `content` / `provenance_sidecar` / `quality_sidecar`, and the `.quality.json` sidecar is copied to the staging dir (previously it was written but never staged).
- **DartMarkersValidator** adds warning-level `data-dart-source` / `data-dart-block-id` presence checks (promoted to critical-on-malformed in Wave 9).
- **Confidence scale**: canonical 5-value float set (1.0 direct-table, 0.8 name-pattern, 0.6 proximity, 0.4 derivation, 0.2 OCR-fallback) documented in `DART/multi_source_interpreter.py` module constants.

### Wave 9 changes — Courseforge source attribution

Wave 9 is the emit-side Courseforge counterpart to Wave 8's DART provenance:

- **New workflow phase** `source_mapping` in `textbook_to_course` (`config/workflows.yaml`) runs between `objective_extraction` and `course_planning`. Driven by the new `source-router` agent. Output: `source_module_map.json` keyed by `week_XX` → `page_id` → `{primary, contributing, confidence}`.
- **`content_generation.inputs_from`** expanded from the opaque `project_id` to also receive `source_module_map_path` + `staging_dir`, so per-page agents can cite their DART sources.
- **`courseforge_jsonld_v1.schema.json`** gains optional `sourceReferences` at page-level AND inside `$defs/Section.properties`. Both `$ref` the canonical `source_reference.schema.json` shape. Elided when empty → backward-compat for courses without DART input.
- **`generate_course.py`** emits `sourceReferences[]` in JSON-LD and `data-cf-source-ids` + optional `data-cf-source-primary` attributes on `<section>` / headings / component wrappers (`.flip-card`, `.self-check`, `.activity-card`, `.discussion-prompt`, `.objectives`). Never on `<p>`/`<li>`/`<tr>` (P2 decision).
- **New validator**: `lib.validators.source_refs.PageSourceRefValidator`, wired as a critical-severity gate on the `content_generation` phase. Verifies every emitted `sourceId` resolves against the staging manifest's provenance sidecars. Graceful fallback: empty `source_module_map.json` → no refs expected, gate passes clean.
- **content-generator prompt** gains a "Source Material" section that declares the three emission surfaces (JSON-LD page, JSON-LD section, HTML attrs) and the inviolable "zero invention" rule. Schema change + prompt change ship together per the Courseforge audit.
- **`DartMarkersValidator`** promoted: `data-dart-source` / `data-dart-block-id` attributes that are present-but-empty now raise critical severity (`EMPTY_DATA_DART_SOURCE`, `EMPTY_DATA_DART_BLOCK_ID`). Fully-absent attributes stay at warning severity so pre-Wave-8 legacy HTML keeps passing — only the "emitted-but-malformed" case blocks.

### Wave 10 changes — Trainforge chunk + node source-provenance propagation

Wave 10 threads Courseforge's `sourceReferences` + `data-cf-source-ids` through Trainforge's parser → chunker → concept-graph builder so chunks carry `source.source_references[]` and concept nodes carry `source_refs[]`. Fully additive + unflagged; absence = "unknown" for pre-Wave-9 corpora.

- **`chunk_v4.schema.json`** `$defs/Source` gains optional `source_references[]` (items `$ref` the canonical `source_reference.schema.json`). Strict `additionalProperties: false` preserved — the field is declared explicitly. Legacy chunks without the field continue to validate under `TRAINFORGE_VALIDATE_CHUNKS=true`.
- **`concept_graph_semantic.schema.json`** node gains optional `source_refs[]` (same `$ref`). Populated from the chunk at `occurrences[0]` (Wave 5 sorted-ID ordering) at `_build_tag_graph` emit time. Evidence arm shapes (IsA, Prerequisite, Related, Assesses, Exemplifies, MisconceptionOf, DerivedFromObjective, DefinedBy) are **NOT** touched in Wave 10 — that work lives in Wave 11 behind `TRAINFORGE_SOURCE_PROVENANCE`.
- **`Trainforge/parsers/html_content_parser.py`** — `ContentSection` gains `source_references: List[str]` (raw `data-cf-source-ids` strings); `ParsedHTMLModule` gains `source_references: List[Dict]` (full SourceReference dicts aggregated via precedence: page-level JSON-LD > section-level JSON-LD > HTML `data-cf-source-ids` auto-roled as `contributing`). First-seen wins on sourceId collision so JSON-LD's authoritative role survives.
- **`Trainforge/process_course.py`** — `_chunk_content` threads parser aggregations into the per-item dict; `_merge_small_sections` now returns 4-tuples `(heading, text, chunk_type, merged_source_ids)` and unions sourceIds across every collapsed section (dedupe + insertion-order preserved); `_create_chunk` folds the resolved refs into `source["source_references"]` via new `_resolve_chunk_source_references` helper (full precedence chain); `_build_tag_graph` copies `source.source_references[]` from each concept's first-occurrence chunk onto `node["source_refs"]`.
- **`MCP/tools/pipeline_tools.py::archive_to_libv2`** — LibV2 archive manifest gains `features.source_provenance` advisory flag. Scans the archived `chunks.jsonl` once at manifest-build time; true when at least one chunk carries `source.source_references[]` with at least one entry; false otherwise (missing file, read errors, legacy corpus). Lets LibV2 retrieval callers fast-skip source-grounded queries on pre-Wave-10 corpora.
- **No new env flag.** All fields are optional; absence = "unknown". Evidence arm enrichment (the only mandatory-via-flag work) waits for Wave 11.

### Wave 11 changes — Trainforge evidence-arm source provenance

Wave 11 completes the end-to-end provenance chain by threading Wave 10's chunk-level `source.source_references[]` into the **five chunk-anchored** edge evidence arms. Gated behind a new opt-in flag so strict-mode consumers on legacy corpora keep passing.

- **`concept_graph_semantic.schema.json`** `$defs` — five evidence arms gain optional `source_references[]` (items `$ref` the canonical `source_reference.schema.json`):
  - `IsAEvidence`, `ExemplifiesEvidence`, `DerivedFromObjectiveEvidence`, `DefinedByEvidence` — straightforward chunk-anchored rules; evidence already carries `chunk_id`.
  - `AssessesEvidence` — complements the pre-existing optional `source_chunk_id`; refs are copied from the chunk that `source_chunk_id` resolves to.
  - `additionalProperties: false` preserved on every arm — field declared explicitly. **Schema admits the field unconditionally**; only the rule-level emit is flag-gated so `TRAINFORGE_STRICT_EVIDENCE=true` behaviour is identical with / without the new flag.
- **Three abstract arms deliberately NOT extended (P4 deferral to a future Wave 12)**: `PrerequisiteEvidence`, `RelatedEvidence`, `MisconceptionOfEvidence`. `related_from_cooccurrence.py` discards chunks before it sees co-occurrence signals (L50 `del chunks, course`); threading refs into `RelatedEvidence` is a non-trivial re-plumbing we don't attempt in this wave. Prerequisite + MisconceptionOf are cheap wins but bundled with the Related refactor so the abstract-arm work lands as one coherent wave.
- **Rule modules** `is_a_from_key_terms.py`, `exemplifies_from_example_chunks.py`, `derived_from_lo_ref.py`, `defined_by_from_first_mention.py`, `assesses_from_question_lo.py` each bump `RULE_VERSION` from 1 -> 2 (unconditional — version reflects schema-generation shift, not runtime emit state). When `TRAINFORGE_SOURCE_PROVENANCE=true` each rule copies the originating chunk's `source.source_references[]` into the evidence arm (or the `source_chunk_id`-resolved chunk's refs for `AssessesEvidence`). Flag off or chunk carries no refs → field omitted (pre-Wave-11 shape retained for back-compat).
- **`defined_by_from_first_mention.py`** + **`assesses_from_question_lo.py`** gain a `_build_chunk_index` helper (flag-gated) so the rule can resolve `chunk_id` -> `source.source_references[]`. Pre-Wave-11 both rules discarded `chunks`; the retention is wrapped in the flag check so flag-off behaviour is byte-identical.
- **`typed_edge_inference.py::rule_versions`** map auto-surfaces the version bumps; no orchestrator changes required.
- **`MCP/tools/pipeline_tools.py::archive_to_libv2`** — second advisory flag `features.evidence_source_provenance` joins `features.source_provenance`. A new `_detect_evidence_source_provenance` helper scans the archived `concept_graph_semantic.json` (checking `graph/` then `corpus/` subdirs); true when at least one edge carries `provenance.evidence.source_references[]`; false otherwise. Lets LibV2 consumers distinguish chunk-level (Wave 10) from evidence-level (Wave 11) provenance.
- **New env flag** `TRAINFORGE_SOURCE_PROVENANCE` joins the root CLAUDE.md table (now eight `TRAINFORGE_*` / decision-capture flags). Default off. See § "Opt-In Behavior Flags" for rationale.

### Deferred / out-of-scope

Not landed in v0.2.0 (tracked for future waves):

- Concept aliases / cross-course equivalence edges (would add a new edge sub-type and touch the Worker O scoped-ID path).
- Flipping any opt-in flag default to "on" (waits on a regeneration cycle of legacy LibV2 corpora).
- `SectionContentType` enforcement (companion to `TRAINFORGE_ENFORCE_CONTENT_TYPE` — Worker T addressed ChunkType only).

---

## JSON-LD round-trip

Wave 1+2 of the RDF/SHACL enrichment plan (`plans/rdf-shacl-enrichment-2026-04-26.md`) added a JSON-LD `@context` to four of the canonical artifacts above so each round-trips losslessly to RDF without a JSON rewrite. This section documents the resulting surface for maintainers — the JSON shapes themselves are unchanged.

### Context files and the artifacts they wrap

- `schemas/context/concept_graph_semantic_v1.jsonld` — wraps `concept_graph_semantic.json` (nodes + reified `TypedEdge` blank nodes carrying per-rule provenance; see § 2 TypedEdge, § 5.2).
- `schemas/context/chunk_v4_v1.jsonld` — wraps the `chunk_v4.schema.json` shape; reuses 14 concept-graph predicates verbatim. Wave 87 minted 33 predicates + 2 anchor classes (`cf:KeyTerm`, `cf:SourceReference`) into `courseforge_v1.vocabulary.ttl`, closing the bulk of the `_phase2_followup` queue. The single residual `_phase3_followup` entry is `ed4all:metadataTrace` (an open-shape diagnostic dict whose canonical class is unresolved; deferred until the dict shape is either codified or removed from chunks).
- `schemas/context/course_v1.jsonld` — wraps `course.json`; mints `ed4all:hasLearningObjective`, `ed4all:courseCode`, `ed4all:loSubtype`. LO IRIs minted at `https://ed4all.io/lo/<id>` via a course-scoped `@base` override.
- `schemas/context/pedagogy_graph_v1.jsonld` — wraps `pedagogy_graph.json`; concept references resolve into the same IRI universe as the concept-graph context (see "cross-artifact join" below).

### The two namespaces (Phase 2.5)

The contexts split predicates across two namespaces by deliberate policy:

- `ed4all:` → `https://ed4all.io/vocab/`. Document-shape predicates (`hasNode`, `hasEdge`, `edgeType`, `edgeSource`, `edgeTarget`) and most domain predicates (`hasPrerequisite`, `isMisconceptionOf`, `assessesObjective`, …).
- `cf:` → `https://ed4all.dev/ns/courseforge/v1#`. Typed classes (`cf:Concept`, `cf:LearningObjective`, `cf:Misconception`, `cf:Chunk`) and the Wave 57+ vocabulary that already lived in `schemas/context/courseforge_v1.vocabulary.ttl`.

Cross-namespace bridges are declared in `schemas/context/aliases.ttl` via `owl:equivalentProperty` / `owl:equivalentClass` rather than `owl:sameAs` (Q11 corpus guidance: never collapse vocabulary identity onto individual identity). `lib/ontology/aliases.py` walks the equivalence closure at load time.

### Reified-edge pattern (canonical for per-edge provenance)

Edges in `concept_graph_semantic_v1.jsonld` materialize as typed `ed4all:TypedEdge` blank nodes carrying `(rule, rule_version, evidence, run_id, created_at)` rather than collapsing to bare `<source> <type> <target>` triples. This preserves per-edge metadata as a reachable subgraph that SPARQL can join against. **Convention:** any new artifact that needs per-edge metadata should follow the same reified-blank-node pattern (Q46 corpus guidance). The pedagogy_graph context applies it consistently.

**RDF-star is superseded here, not deferred.** The reified-blank-node pattern above plus the named-graph provenance shipped in Phase 3 of `plans/rdf-shacl-enrichment-2026-04-26.md` cover the per-edge-metadata use case that originally motivated RDF-star evaluation. Q49's tooling-maturity caveat still holds, but the design has moved past needing RDF-star — do not add it back without a use case that named graphs and reified TypedEdges genuinely cannot model.

### Cross-artifact join

All four contexts share `@base: https://ed4all.io/concept/`, so `concept:foo` (pedagogy_graph) and the bare `foo` slug (concept_graph) both expand to the same IRI `https://ed4all.io/concept/foo`. Wave 2 evidence: 672/672 pedagogy concept IRIs join cleanly to the 1,100-IRI concept_graph universe on the `rdf-shacl-551-2` fixture. This is what makes pedagogy ↔ concept_graph SPARQL queries trivial — no ID translation layer is needed.

### Round-trip tests

Four regression suites under `Trainforge/tests/` enforce triple-count parity through pyld expand → rdflib parse → Turtle serialize → re-parse:

- `test_concept_graph_jsonld_roundtrip.py` (4 tests) — node + edge + reified-provenance triple-count delta = 0.
- `test_chunk_v4_jsonld_roundtrip.py` (4 tests) — chunk shape including the typed `ed4all:SourceReference` sub-shape.
- `test_course_jsonld_roundtrip.py` (6 tests) — course + LO IRI minting under the scoped `@base`.
- `test_pedagogy_graph_jsonld_roundtrip.py` (5 tests, incl. cross-artifact concept IRI join).

### Authoring a new context

- Consult `lib/ontology/edge_predicates.py::SLUG_TO_IRI` for canonical slug ↔ IRI mappings before minting anything.
- Reuse predicates from prior contexts where the semantics match; the chunk context reuses 14 predicates from concept_graph verbatim.
- Inline-flag genuinely new predicates with a `_phase3_followup` marker so the next vocabulary-extension wave knows what to mint formally. (Phase-numbered marker convention: `_phaseN_followup` where N is the next planned mint wave; updated when a predicate's mint is deferred for a known reason.)
- Keep document-shape predicates in `ed4all:`; mint typed classes and edge predicates with `rdfs:domain` / `rdfs:range` in `cf:` (the namespace policy above).
- Land a round-trip test alongside the context: load a real fixture, parse via pyld + rdflib, serialize to Turtle, re-parse, and assert triple-count delta = 0.

### rdflib runtime dependency

`rdflib>=7.0.0,<8.0.0` is declared in `[project.dependencies]` in `pyproject.toml` (not just under the `pyshacl` dev-extra) because `lib/ontology/aliases.py` imports it at runtime to walk the equivalence closure during `concept_classifier.canonicalize_alias` calls.
