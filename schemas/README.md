# `/schemas/` — Unified Schema Home

Single project-root home for all domain-shared JSON Schemas, taxonomies, and event contracts used across DART, Courseforge, Trainforge, and LibV2.

For the full current-state ontology map (classes, relations, taxonomies, provenance, identity, constraints, versioning), see **[`ONTOLOGY.md`](./ONTOLOGY.md)**.

## Folder tree

```
schemas/
├── README.md                              this file
├── ONTOLOGY.md                            current-state ontology map
│
├── academic/                              course-level academic metadata
│   ├── course_metadata.schema.json         full course (MIT OCW-shape)
│   ├── courseforge_page_types.schema.json  page-level type enum
│   ├── learning_objectives.schema.json     extracted LOs + hierarchy
│   └── textbook_structure.schema.json      DART-processed HTML structure
│
├── compliance/                            accessibility standards
│   └── wcag22_compliance.schema.json       WCAG 2.2 AA requirement matrix
│
├── config/                                orchestrator config meta-schemas
│   └── workflows_meta.schema.json          validates config/workflows.yaml
│
├── events/                                decision + audit log contracts
│   ├── decision_event.schema.json          Claude decision ledger (base)
│   ├── trainforge_decision.schema.json     extends decision w/ Q&A context
│   ├── audit_event.schema.json             unified audit event
│   ├── hash_chained_event.schema.json      tamper-evident chain wrapper
│   ├── session_annotation.schema.json      aggregated session summary
│   └── run_manifest.schema.json            immutable run-init snapshot
│
├── knowledge/                             knowledge-graph + training pairs
│   ├── chunk_v4.schema.json                Trainforge chunk contract
│   ├── concept_graph_semantic.schema.json  typed-edge concept graph
│   ├── courseforge_jsonld_v1.schema.json   Courseforge emit JSON-LD contract
│   ├── instruction_pair.schema.json        SFT pairs (prompt/completion)
│   ├── instruction_pair.strict.schema.json opt-in strict SFT variant
│   ├── misconception.schema.json           first-class misconception entity
│   └── preference_pair.schema.json         DPO pairs (chosen/rejected)
│
├── library/                               LibV2 course repository
│   ├── catalog_entry.schema.json           course entry in master catalog
│   └── course_manifest.schema.json         extended course metadata
│
└── taxonomies/                            controlled vocabularies
    ├── assessment_method.json              formative / summative / diagnostic
    ├── bloom_verbs.json                    6-level / 60-verb canonical list
    ├── cognitive_domain.json               factual / conceptual / procedural / metacognitive
    ├── content_type.json                   section content-type enum
    ├── module_type.json                    6-value module-type enum
    ├── pedagogy_framework.yaml             12-tier pedagogy gap framework
    ├── question_type.json                  7-value factory enum
    ├── taxonomy.json                       STEM/ARTS division hierarchy
    └── teaching_role.json                  (component, purpose) -> role mapping
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

Six tool-local schemas remain under `Courseforge/schemas/` because they describe Courseforge-internal HTML component structures or tool-specific migrations — not artifacts that cross tool boundaries:

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
- **`compliance/`** — what WCAG 2.2 AA compliance looks like as a checkable manifest.
- **`config/`** — meta-schemas that describe `config/*.yaml` orchestrator files themselves.
- **`events/`** — the append-only contracts (decisions, audits, hash chains, run manifests, session summaries) that record everything that happened.
- **`knowledge/`** — concept-graph edges, chunk contracts, JSON-LD emit shape, misconception entities, and the instruction/preference training pairs derived from chunks.
- **`library/`** — how a course surfaces in LibV2 (catalog entry + full manifest).
- **`taxonomies/`** — the controlled vocabularies referenced by everything above (loaded via `lib/ontology/`).
