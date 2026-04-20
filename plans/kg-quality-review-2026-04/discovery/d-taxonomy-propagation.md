# Worker D — LibV2 taxonomy propagation audit

## Summary
- `taxonomy.json` consumers: **2 production code paths** (both LibV2), plus 1 CLI shim — zero references from Courseforge, Trainforge, or shared `lib/`.
- `pedagogy_framework.yaml` consumers: **0** — file exists on disk but is never loaded by any code in the repo. Referenced only in prose docs (`schemas/README.md`, `schemas/ONTOLOGY.md`, `LibV2/README.md`, `PLAN_NOTES.md`).
- Courseforge emit propagation: **none** — `generate_course.py::_build_page_metadata:571` emits JSON-LD with `courseCode`, `weekNumber`, `moduleType`, `learningObjectives`, `sections`, `misconceptions`, `suggestedAssessmentTypes` but **no division/primary_domain/subdomain/topics**. No `data-cf-*` attribute carries taxonomy coordinates either.
- Trainforge consume propagation: **none** — subject classification is fabricated from CLI flags at manifest-emit time (`Trainforge/process_course.py:2731–2735`, `:1757–1764`), not consumed from any upstream Courseforge signal.
- **Top KG-impact:** Cross-course dedupe and taxonomy-aware retrieval are blocked at the source — a Courseforge-generated course enters Trainforge with no subject-taxonomy fingerprint, so the only way a `division`/`primary_domain` ever reaches LibV2 is via a human CLI flag that was never validated against `taxonomy.json`. Free-text `domain="pedagogy"` call sites slip through silently because `search_catalog` does equality matching, returning an empty set without error.

## taxonomy.json consumer map

| File:line | Usage | KG-relevance |
|---|---|---|
| `LibV2/tools/libv2/concept_vocabulary.py:112` | `ConceptVocabulary._load_taxonomy()` — walks `divisions → domains → subdomains → topics` into a flat `canonical_terms` set | Anchors concept-tag normalization to the authoritative taxonomy; without this, any chunk-level `concept_tags` KG edge is free-text |
| `LibV2/tools/libv2/concept_vocabulary.py:383` | `analyze_course_concepts()` resolves `schemas/taxonomies/taxonomy.json` and builds a `ConceptVocabulary` | Gate for `libv2 concepts analyze` CLI — KG hygiene step that flags tags unknown to taxonomy |
| `LibV2/tools/libv2/concept_vocabulary.py:411` | `clean_course_concepts()` — same resolution, used to prune invalid tags from `chunks.json` + `guardrails.json` | Only path in the repo that removes off-taxonomy concept labels before KG emission |
| `LibV2/tools/libv2/validator.py:197` | `validate_taxonomy_compliance()` — loads taxonomy, checks manifest `classification.division` + `classification.primary_domain` against `taxonomy.divisions[division].domains` | Only enforcement point verifying a course's declared taxonomy coords exist; called from `validate_course` (`validator.py:461`) |
| `LibV2/tools/libv2/cli.py:696,786` | CLI shims invoking `analyze_course_concepts` / `clean_course_concepts` | Exposure surface only |

Indirect/transitive consumer: `LibV2/tools/libv2/retriever.py:609–611` calls `search_catalog(domain=…, division=…, subdomain=…)` — the catalog entries it filters were themselves populated from manifest values that were taxonomy-validated by `validator.py:197`. The retriever never reads `taxonomy.json` directly.

**Gap:** No consumer outside `LibV2/tools/libv2/` references the taxonomy. Courseforge content generation and Trainforge chunking both emit course manifests whose `classification` block is sourced from CLI flags and is only validated **after** the course reaches LibV2 import.

## pedagogy_framework.yaml consumer map

| File:line | Usage | KG-relevance |
|---|---|---|
| — | (no code consumers) | — |
| `schemas/README.md:41,48` | Documented as YAML data file | Prose only |
| `schemas/ONTOLOGY.md:704,1117` | Listed as "12-tier pedagogy gap-analysis framework" | Prose only |
| `LibV2/README.md:38` | Directory listing | Prose only |
| `LibV2/CLAUDE.md:139` | Referenced as "pedagogy framework" resource | Prose only |

Related but **unrelated** artifact: `LibV2/tools/libv2/retrieval_scoring.py:240 load_pedagogy_model()` loads each course's `pedagogy/pedagogy_model.json` (a per-course model emitted by `Trainforge/process_course.py:2634`). This is **not** `pedagogy_framework.yaml`; the two share a name root but are different things. The 12-tier framework at `schemas/taxonomies/pedagogy_framework.yaml` is orphan.

**KG-impact:** The framework's 12 tiers (foundational_theories, learning_sciences, instructional_design, …) never flow into any KG node. Queries like "find chunks covering Tier 3 instructional-design topics" are impossible — the dimension exists on paper only. Any "pedagogy coverage" report the framework was designed to drive must be recomputed manually.

## Downstream classification fields

Reference: `schemas/library/course_manifest.schema.json:48–117` defines the full `classification` block (`division`, `primary_domain`, `secondary_domains`, `subdomains`, `topics`, `subtopics`) plus `ontology_mappings` (ACM CCS + LCSH) at the course-manifest level. No equivalent exists at the page or chunk level.

| Field | Present in | Absent from | Should live in |
|---|---|---|---|
| `division` | `course_manifest.schema.json:52`; Trainforge emit `process_course.py:1758` (from CLI `--division`) | Courseforge JSON-LD; `data-cf-*`; chunk metadata | Course manifest (exists) + inherit into page JSON-LD + chunk record |
| `primary_domain` | same as above | Courseforge JSON-LD (`_build_page_metadata:571` omits it); chunks | Page JSON-LD `subject` + chunk `primary_domain` for retrieval filters |
| `secondary_domains` | `course_manifest.schema.json:61`; Trainforge `process_course.py:510,1760` | Courseforge entirely | Course manifest only (course-wide attribute) |
| `subdomains` | `course_manifest.schema.json:67`; Trainforge `process_course.py:1761` | Courseforge; chunks | Manifest + optionally per-page (pages often scope to one subdomain) |
| `topics` | `course_manifest.schema.json:73`; Trainforge `process_course.py:1762` (auto-extracted from objectives) | Courseforge JSON-LD | Both — page-level `topics[]` would unlock topic-grained retrieval |
| `subtopics` | `course_manifest.schema.json:79`; Trainforge `process_course.py:1763` (auto from concept graph) | Courseforge; derived post-hoc | Post-hoc derivation is the right pattern; document it |
| `ontology_mappings.acm_ccs` | `course_manifest.schema.json:90` | Everywhere else — never populated in observed manifests | Manifest; emit-side wiring required |
| `ontology_mappings.lcsh` | `course_manifest.schema.json:103` | Same | Manifest; emit-side wiring required |

**KG-impact:** The existing manifest schema already defines the propagation target for all six taxonomy fields plus two external-ontology cross-walks. The gap is entirely on the **producer** side: Courseforge never emits these, and Trainforge fabricates them from CLI flags. ACM CCS / LCSH cross-walks are schema-slots with no producer — severely limits federated KG joins across external library catalogs.

## CrossCourseRAG domain shape trace

`CrossCourseRAG.__init__(self, domain: Optional[str] = None)` at `Trainforge/rag/libv2_bridge.py:532` stores `self.domain = domain` (`:539`) and forwards it to `retrieve_chunks(..., domain=self.domain, ...)` at `Trainforge/rag/libv2_bridge.py:572`.

Downstream: `LibV2/tools/libv2/retriever.py:551 retrieve_chunks` forwards to `search_catalog(..., domain=domain, ...)` at `:610`, which does case-insensitive equality match against `CatalogEntry.primary_domain` or membership in `secondary_domains` (`LibV2/tools/libv2/catalog.py:111–116`).

| Call site | Value shape | Source | KG-impact |
|---|---|---|---|
| `Trainforge/rag/__init__.py:23` (docstring example) | Free-text string `"pedagogy"` | Hard-coded literal in the module docstring | `"pedagogy"` is **not** a domain in `schemas/taxonomies/taxonomy.json` (the closest canonical node is `educational-technology` at `taxonomy.json:279`). `search_catalog` returns `[]` silently → `retrieve_chunks` returns `[]` with no error, misleading downstream code into thinking no content exists. |
| `Trainforge/rag/libv2_bridge.py:428` | `CrossCourseRAG()` — domain defaults to `None` | Programmatic fallback when the primary per-course retrieval returns too few chunks | With `domain=None` the filter is disabled — effectively an "any-domain" cross-course fetch. This is fine for fallback semantics but masks the fact that no canonical-domain fallback is ever attempted. |
| `Trainforge/rag/libv2_bridge.py:610 get_cross_course_rag(domain=None)` | Passes through `Optional[str]` | Public API | Accepts any string; no validation against taxonomy. A typo (`"physcs"` vs `"physics"`) returns empty with no warning. |

**No caller validates `domain` against `taxonomy.json` before dispatch.** The `ConceptVocabulary` loader would be the natural guard but is never invoked on this path.

## Propagation-path proposal

Minimum surface to make subject taxonomy queryable across the DART→Courseforge→Trainforge→LibV2 chain:

- **Course-level (manifest):** Emit the `classification` + `ontology_mappings` block from Courseforge's course-planning phase (not CLI flags at Trainforge ingestion). Target file: a `course.json` or `manifest-stub.json` written alongside the IMSCC package so `process_course.py` can load it instead of requiring `--division/--domain/--subdomain` args. Existing schema target: `schemas/library/course_manifest.schema.json:48`. Validation gate: `validate_taxonomy_compliance` already exists at `LibV2/tools/libv2/validator.py:197` — promote it to run on the Courseforge-emitted stub, not just after LibV2 import.
- **Page-level (JSON-LD):** Extend `_build_page_metadata` (`Courseforge/scripts/generate_course.py:571`) to inherit `division`, `primary_domain`, `subdomains` from the course-level classification and optionally add page-scoped `topics[]`. Namespace: the existing `https://ed4all.dev/ns/courseforge/v1` context (Worker C catalogs the contract). This lets Trainforge's HTML parser (`Trainforge/parsers/html_content_parser.py`) extract taxonomy coords at chunk-emission time rather than receiving them via CLI.
- **Chunk-level (chunk record):** Have Trainforge's chunk emitter copy `primary_domain`, `subdomains`, and page-scoped `topics` onto each chunk. `LibV2/tools/libv2/retriever.py:691` already reads `_domain` off chunks but the field is not reliably populated on the producer side. Formalizing it unlocks domain-filtered retrieval inside a course (today only possible across courses via catalog filter).

Taxonomy validation should fail-closed at the Courseforge emit step, not at LibV2 import — once a course reaches LibV2 with a bad `domain`, downstream chunks have already been misclassified. `ConceptVocabulary._load_taxonomy` provides the reusable loader; expose it via `lib/ontology/` (new) so Courseforge can consume it without reaching into LibV2.

## KG-impact summary

| Gap | Severity | Query/join that breaks |
|---|---|---|
| Courseforge emits no taxonomy classification in manifest or JSON-LD | **critical** | Cross-course dedupe by taxonomy node is impossible — two courses covering "kinematics" have no shared `primary_domain=physics` + `subdomain=mechanics` node to merge on. Federated queries like "all STEM/physics/mechanics chunks across courses" only work after a human assigns domain via CLI. |
| Trainforge derives `classification` from CLI flags, not upstream signal | **critical** | Any automation (MCP `create_textbook_pipeline_tool`, orchestrator workflows) must pass correct flags manually. Silent misclassification when defaults kick in (`--division STEM` is the default even for ARTS content per `process_course.py:2731`). KG acquires wrong edges with no audit trail. |
| `pedagogy_framework.yaml` has zero consumers | **high** | Queries "which tiers of the 12-tier pedagogy framework does this course cover" and "find gaps in pedagogy coverage across the LibV2 corpus" cannot be answered programmatically. The framework dimension is absent from the KG despite being designed as a first-class axis. |
| `CrossCourseRAG(domain=...)` accepts unvalidated strings | **high** | Typo or free-text domain (`"pedagogy"`, `"phys"`, `"math"`) returns empty result set with no error — query appears to succeed with zero hits. Downstream assessment generation silently loses cross-course grounding. Any KG query layered on top inherits this silent-failure mode. |
| `ontology_mappings.acm_ccs` and `.lcsh` defined in schema but never populated | **high** | Cross-walk joins to external ontologies (ACM digital library, Library of Congress) are impossible. The schema slot signals an intent the pipeline never fulfills — federated KG scenarios blocked. |
| Chunk records inconsistently carry `_domain` | **medium** | `retriever.py:691` reads `chunk.get("_domain", "")` but there is no producer contract ensuring the field is set. Chunk-level domain filters silently downgrade to catalog-level filters; retrieval scoring loses a dimension. |
| No validation gate at Courseforge emit for taxonomy coords | **medium** | `validate_taxonomy_compliance` runs only at LibV2 import — by then the IMSCC and chunks already exist with whatever `division`/`domain` the CLI flag supplied. Errors are caught late, requiring re-run of the full pipeline. Provenance auditability suffers (can't tell whether misclassification originated in Courseforge planning, Trainforge ingestion, or LibV2 import). |
| `subtopics` auto-extracted from concept-graph frequency (`process_course.py:1705–1721`) with no taxonomy grounding | **medium** | Subtopic KG edges are frequency artifacts, not taxonomy nodes. Two courses may produce `subtopics=["bloom", "udl"]` without ever touching canonical taxonomy topics — blocks subtopic-grained cross-course joins. |
