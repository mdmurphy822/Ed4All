# `/schemas/` — Unified Schema Home

Single project-root home for all domain-shared JSON Schemas, taxonomies, and event contracts used across SemantiK, Courseforge, Trainforge, and LibV2.

For the full current-state ontology map (classes, relations, taxonomies, provenance, identity, constraints, versioning), see **[`ONTOLOGY.md`](./ONTOLOGY.md)**.

## Folder tree

```
schemas/
├── README.md                                this file
├── ONTOLOGY.md                              current-state ontology map
├── academic/                                course-level academic metadata
│   ├── course_metadata.schema.json          full course (MIT OCW shape)
│   ├── courseforge_page_types.schema.json   page-level type enum
│   ├── learning_objectives.schema.json      extracted LOs + hierarchy
│   └── textbook_structure.schema.json       SemantiK-processed HTML structure
│
├── aggregators/                             post-loop aggregator output shapes
│   ├── coverage_map.schema.json             objective→chunk→question→pair map (Wave 3 G1)
│   └── trainforge_assessment_quality_report.schema.json   assessment-quality rollup
│
├── compliance/                              accessibility standards
│   └── wcag22_compliance.schema.json        WCAG 2.2 AA requirement matrix
│
├── config/                                  orchestrator config meta-schemas
│   └── workflows_meta.schema.json           validates config/workflows.yaml
│
├── context/                                 RDF / SHACL contexts + vocabulary
│   ├── aliases.ttl                          owl:equivalentProperty bridges
│   ├── courseforge_v1.shacl-closed.ttl      sh:closed overlay (gated)
│   ├── courseforge_v1.shacl-rules.ttl       SHACL inference rules (gated)
│   ├── courseforge_v1.shacl.ttl             canonical SHACL shapes
│   ├── courseforge_v1.vocabulary.ttl        cf: vocabulary
│   └── training_pair.shacl.ttl              SHACL shapes for training pairs
│
├── courseforge/                             Courseforge-specific config schemas
│   └── block_routing.schema.json            per-block-type routing policy
│
├── eval/                                    Trainforge eval defaults
│   ├── default_eval_config.yaml             5×3 stage matrix config
│   └── default_rubric.md                    LLM-as-judge rubric
│
├── events/                                  decision + audit log contracts
│   ├── audit_event.schema.json              unified audit event
│   ├── decision_event.schema.json           Claude decision ledger (base)
│   ├── hash_chained_event.schema.json       tamper-evident chain wrapper
│   ├── run_manifest.schema.json             immutable run-init snapshot
│   ├── session_annotation.schema.json       aggregated session summary
│   └── trainforge_decision.schema.json      decision + Q&A context (allOf)
│
├── governance/                              promotion-chain + course-status shapes
│   ├── course_status.schema.json            5-value status enum
│   ├── phase_output.schema.json             canonical phase output envelope
│   └── promotion_chain.schema.json          9-arrow promotion chain
│
├── knowledge/                               knowledge-graph + training pairs
│   ├── chunk_v4.schema.json                 Trainforge chunk contract
│   ├── concept_graph_semantic.schema.json   typed-edge concept graph
│   ├── course.schema.json                   Trainforge-emitted course.json
│   ├── course_metadata.schema.json          Trainforge-emitted course metadata
│   ├── courseforge_jsonld_v1.schema.json    Courseforge emit JSON-LD shape
│   ├── domain_concept_vocabulary.schema.json  per-corpus domain-concept vocabulary
│   ├── instruction_pair.schema.json         SFT pairs (prompt/completion)
│   ├── instruction_pair.strict.schema.json  opt-in strict SFT variant
│   ├── misconception.schema.json            first-class misconception entity
│   ├── objectives_v1.schema.json            synthesized objectives shape
│   ├── pair_audit_fields.schema.json        shared training-pair audit fields
│   ├── preference_pair.schema.json          DPO pairs (chosen/rejected)
│   ├── source_reference.schema.json         {sourceId, role, …} canonical shape
│   └── textbook_structure_enrichment.schema.json  LLM enrichment overlay on textbook_structure
│
├── library/                                 LibV2 course repository
│   ├── catalog_entry.schema.json            course entry in master catalog
│   ├── chunkset_drift_report.schema.json    SemantiK vs IMSCC chunkset drift sidecar
│   ├── chunkset_manifest.schema.json        per-*_chunks/ manifest
│   ├── course_manifest.schema.json          extended course metadata
│   └── packaging_report.schema.json         IMSCC packaging report sidecar
│
├── models/                                  SLM training artifacts (Wave 89-93)
│   ├── model_card.schema.json               adapter model_card.json shape
│   └── model_pointers.schema.json           promotion-ledger pointer file
│
├── quality/                                 OSCQR quality-review items
│   ├── oscqr_items.json                      OSCQR rubric items (data)
│   └── oscqr_items.schema.json               schema for OSCQR items
│
├── taxonomies/                              controlled vocabularies
│   ├── assessment_method.json               formative / summative / diagnostic
│   ├── bloom_verbs.json                     6-level / 60-verb canonical list
│   ├── cognitive_domain.json                factual / conceptual / procedural / metacognitive
│   ├── cognitive_task_type.json             cognitive task-type enum
│   ├── content_type.json                    section content-type enum
│   ├── lo_hierarchy.json                    TO/CO/SubCO hierarchy enum
│   ├── module_type.json                     6-value module-type enum
│   ├── pedagogy_framework.yaml              12-tier pedagogy framework
│   ├── question_type.json                   canonical 9-value enum (W-D1 P0.2 reconciliation target)
│   ├── taxonomy.json                        STEM/ARTS division hierarchy
│   └── teaching_role.json                   (component, purpose) → role mapping
│
├── tests/                                   per-schema fixture suites
│   └── fixtures/
│       ├── wave1/                           Wave 1 W1.A fixtures (observed_bloom + promotion_decision + trainable pair)
│       └── wave_d11/                        Wave D11 fixtures (evidence-quote / per-claim-support)
│
└── training/                                Trainforge training-pair fixtures + schemas
    ├── family_map.rdf_shacl.yaml            per-family CURIE clusters fixture
    ├── family_map.schema.json               schema for family_map.<family>.yaml
    ├── property_manifest.rdf_shacl.yaml     per-corpus property manifest fixture
    ├── property_manifest.schema.json        schema for property_manifest.<family>.yaml
    ├── schema_translation_catalog.rdf_shacl.yaml   schema-translation catalog fixture
    ├── schema_translation_catalog.schema.json     schema for translation catalog
    ├── semantic_profiles.schema.json        per-domain semantic profile shape
    ├── semantic_profiles.yaml               default semantic profiles fixture
    └── synthesis_summary.schema.json        per-corpus synthesis summary
```

## Naming convention

- **`<name>.schema.json`** — JSON Schema (draft-07). All validators live under this suffix.
- **`<name>.json`** — plain data file (e.g. `taxonomy.json` — the STEM/ARTS hierarchy itself, not a schema that describes one).
- **`<name>.yaml`** — YAML data file (e.g. `pedagogy_framework.yaml`).

Every `<name>.schema.json` file declares `"$schema": "http://json-schema.org/draft-07/schema#"` as its first key.

## How loaders find schemas

Schema discovery is centralized and recursive, so new files in any subdirectory are picked up automatically.

- **Root constant:** `lib/path_constants.py:87` — `SCHEMAS_DIR = "schemas"`.
- **Recursive discovery:** `lib/validation.py:104` — `SCHEMAS_DIR.rglob("*.json")` loads every schema file from every subfolder into the resolver registry.
- **Named fast-paths:** `lib/validation.py:24-26` resolves these specific paths by name:
  - `DECISION_SCHEMA_PATH = SCHEMAS_DIR / "events" / "decision_event.schema.json"`
  - `TRAINFORGE_SCHEMA_PATH = SCHEMAS_DIR / "events" / "trainforge_decision.schema.json"`
  - `SESSION_SCHEMA_PATH = SCHEMAS_DIR / "events" / "session_annotation.schema.json"`
- **CI integrity:** `ci/integrity_check.py` walks the same tree on every PR.

Adding a new schema is a one-step operation: drop the file under the appropriate subfolder; no loader update required.

## What is NOT here

Seven tool-local schemas remain under `Courseforge/schemas/` because they describe Courseforge-internal HTML component structures or tool-specific migrations — not artifacts that cross tool boundaries:

| Path | Scope |
|---|---|
| `Courseforge/schemas/content-display/accordion-schema.json` | Courseforge UI: accordion component |
| `Courseforge/schemas/content-display/content-display-schema.json` | Courseforge UI: generic content-display |
| `Courseforge/schemas/content-display/enhanced-content-display-schema.json` | Courseforge UI: enhanced content-display |
| `Courseforge/schemas/content-display/page-title-standards.json` | Courseforge UI: page-title rules |
| `Courseforge/schemas/layouts/course_card_schema.json` | Courseforge UI: course card layout |
| `Courseforge/schemas/template-integration/educational_template_schema.json` | Courseforge template system |
| `Courseforge/schemas/framework-migration/bootstrap5_migration_schema.json` | Courseforge Bootstrap migration |

IMS CC / QTI XSDs under `Courseforge/schemas/imscc/` are upstream IMS Global specs, also unchanged.

## Subfolder purpose at a glance

- **`academic/`** — what a course, its chapters/sections, and its learning objectives look like before they become HTML.
- **`aggregators/`** — output shapes for the post-loop aggregators that roll up per-phase signals into a single operator-facing JSON (coverage map, validation report, quality report, promotion chain).
- **`compliance/`** — what WCAG 2.2 AA compliance looks like as a checkable manifest.
- **`config/`** — meta-schemas that describe `config/*.yaml` orchestrator files themselves.
- **`context/`** — RDF/SHACL artifacts: the `cf:` vocabulary, the canonical SHACL shapes, opt-in closed-world overlay (`TRAINFORGE_SHACL_CLOSED_WORLD`), opt-in inference rules (`TRAINFORGE_USE_SHACL_RULES`), and cross-namespace alias bridges.
- **`courseforge/`** — Courseforge-specific config schemas (per-block-type routing policy).
- **`eval/`** — Trainforge eval-harness defaults (`default_eval_config.yaml` for the 5×3 stage matrix; `default_rubric.md` for the LLM-as-judge rubric).
- **`events/`** — append-only contracts (decisions, audits, hash chains, run manifests, session summaries) that record everything that happened.
- **`governance/`** — promotion-chain + 5-value course-status enum + canonical PhaseOutput envelope.
- **`knowledge/`** — concept-graph edges, chunk contract, JSON-LD emit shape, misconception entity, training pair shapes (instruction + preference + strict variant), source-reference canonical shape.
- **`library/`** — how a course surfaces in LibV2 (catalog entry, course manifest, chunkset manifest + drift report, packaging report).
- **`models/`** — SLM-training artifacts: adapter `model_card.json` shape (Wave 89 + Wave 92) + promotion-ledger pointer file (Wave 93).
- **`quality/`** — OSCQR quality-review rubric items (data file + its schema).
- **`taxonomies/`** — controlled vocabularies referenced by everything above (loaded via `lib/ontology/`).
- **`tests/`** — per-schema fixture suites (fixtures live under `tests/fixtures/wave1/` and `tests/fixtures/wave_d11/`).
- **`training/`** — Trainforge per-corpus fixtures + their schemas (family map, property manifest, schema translation catalog, semantic profiles, synthesis summary).
