# Pipeline Execution Fixes — Output Contracts

**Status**: BLOCK worker deliverable — consumed by workers α, β, γ in parallel.
**Scope**: The three stub tool functions in `MCP/tools/pipeline_tools.py` that
the textbook-to-course pipeline calls during phases 5, 7, and 8.

This document pins down what each of those functions is expected to produce.
Every field shape and path here is sourced from an existing schema or an
existing emit site in the codebase — no new schemas, no new vocabulary. Gaps
(schema-missing corners) are flagged inline rather than patched.

---

## Stub inventory

| # | Function | File/lines | Worker | Phase |
|---|----------|------------|--------|-------|
| 1 | `_generate_course_content` | `MCP/tools/pipeline_tools.py:1207-1427` | α | content_generation |
| 2 | `_generate_assessments` | `MCP/tools/pipeline_tools.py:1547-1675` | β | trainforge_assessment |
| 3 | `_archive_to_libv2` | `MCP/tools/pipeline_tools.py:1676-1754` | γ | libv2_archival |

Each contract below is structured as:
1. **Inputs** (kwargs the orchestrator already passes in today)
2. **Output paths** (absolute within the current run's project workspace)
3. **Per-file field shape** (JSON / HTML examples)
4. **Gates that MUST pass** (validators already wired in `config/workflows.yaml`)
5. **Schema gaps** (if any) — items worth opening follow-up PRs for

---

## 1. Courseforge content-generator contract (Worker α)

**Function**: `_generate_course_content(**kwargs)`
**Canonical reference implementation**: `Courseforge/scripts/generate_course.py`
(especially `_wrap_page` at :268-307, `_build_page_metadata` at :728-778,
`_build_objectives_metadata` at :622-656, `_build_sections_metadata` at
:690-725). Tests assert shape against
`schemas/knowledge/courseforge_jsonld_v1.schema.json`.

### Inputs

```
project_id           str  — timestamped export dir under Courseforge/exports/
course_name          str  — optional; config is the authority (course_code)
duration_weeks       int  — sourced via config; default 12
objectives_path      str  — read from project_config.json
staging_dir          Path — COURSEFORGE_INPUTS / run_id / *.html (DART output)
source_module_map    dict — project_path/source_module_map.json (Wave 9 output)
```

All of these are already visible to the current stub — no signature change.

### Output paths

```
Courseforge/exports/{project_id}/
  03_content_development/
    week_01/
      week_01_overview.html
      week_01_content_01_<slug>.html           # one per content_module
      week_01_application.html
      week_01_self_check.html
      week_01_summary.html
    week_02/...
    ...
    week_{N}/...                                # N = duration_weeks
  source_module_map.json                        # pre-existing input
```

The existing stub emits `week_{NN}/module.html` only. Worker α replaces this
with the 5-page-per-week structure above. The existing IMSCC packager
(`_package_imscc` at :1433-1532) already walks `**/*.html`, so any 5-page shape
lands in the cartridge.

### HTML shape (every page)

Every emitted HTML file is a complete accessibility-compliant document and
MUST include, in order:

1. `<!DOCTYPE html>` and `<html lang="en">`
2. `<head>` with `<meta charset>`, `<meta viewport>`, `<title>`, optional `<style>`
3. **Exactly one** `<script type="application/ld+json">` block inside `<head>`
   — body is the JSON-LD document described below.
4. `<body>` with first element `<a href="#main-content" class="skip-link"
   data-cf-role="template-chrome">Skip to main content</a>`
5. `<header role="banner" data-cf-role="template-chrome">` with breadcrumbs
6. `<main id="main-content" role="main">` containing:
   - `<h1>` page title
   - `<section id="objectives" aria-labelledby="objectives-heading">` — each
     `<li>` carries `data-cf-objective-id`, `data-cf-bloom-level`,
     `data-cf-bloom-verb`, `data-cf-cognitive-domain`. Reference emit at
     `generate_course.py:305-333`.
   - One or more `<section>` blocks. Every `<h2>` inside has
     `data-cf-content-type` (enum per `schemas/taxonomies/content_type.schema.json`:
     `definition` | `example` | `procedure` | `comparison` | `exercise` |
     `overview` | `summary` | `explanation`), optional `data-cf-key-terms`,
     optional `data-cf-bloom-range`.
   - Interactive components where applicable:
     `<div class="flip-card" data-cf-component="flip-card"
     data-cf-purpose="term-definition">`;
     `<div class="self-check" data-cf-component="self-check"
     data-cf-purpose="formative-assessment" data-cf-bloom-level="..."
     data-cf-objective-ref="...">`;
     `<div class="activity-card" data-cf-component="activity"
     data-cf-purpose="practice" data-cf-bloom-level="..."
     data-cf-objective-ref="...">`.
7. `<footer role="contentinfo" data-cf-role="template-chrome">`

Canonical `data-cf-*` table: `schemas/ONTOLOGY.md` § 4.12 (lines 629-648).
**Every attribute listed there is already reachable from the HTML text the
stub produces; the job is to emit them.**

### JSON-LD body shape

One `<script type="application/ld+json">` block per page. The JSON object
MUST validate against `schemas/knowledge/courseforge_jsonld_v1.schema.json`:

```json
{
  "@context": "https://ed4all.dev/ns/courseforge/v1",
  "@type": "CourseModule",
  "courseCode": "TESTPIPE_101",
  "weekNumber": 1,
  "moduleType": "overview",
  "pageId": "week_01_overview",
  "learningObjectives": [
    {
      "id": "CO-01",
      "statement": "Describe photosynthesis as a light-driven chemical reaction.",
      "bloomLevel": "understand",
      "bloomVerb": "describe",
      "cognitiveDomain": "conceptual",
      "keyConcepts": ["photosynthesis", "chloroplast"],
      "assessmentSuggestions": ["multiple_choice", "short_answer", "fill_in_blank"]
    }
  ],
  "sections": [
    {
      "heading": "What is Photosynthesis?",
      "contentType": "definition",
      "keyTerms": [
        {"term": "chloroplast", "definition": "The organelle where photosynthesis occurs."}
      ],
      "bloomRange": ["remember", "understand"]
    }
  ],
  "misconceptions": [
    {
      "misconception": "Plants only need sunlight to live.",
      "correction": "Plants need light, water, and CO2; photosynthesis consumes all three."
    }
  ],
  "suggestedAssessmentTypes": ["multiple_choice", "short_answer"],
  "prerequisitePages": [],
  "sourceReferences": []
}
```

Requireds: `@context`, `@type`, `courseCode`, `weekNumber`, `moduleType`,
`pageId`. Every other array key is omitted when empty (per the schema's
elision convention at `courseforge_jsonld_v1.schema.json:40-67`).

`moduleType` is one of: `overview` | `content` | `application` | `assessment`
| `summary` | `discussion` (from `schemas/taxonomies/module_type.schema.json`).

### Source grounding (Wave 9)

1. Open `{project_path}/source_module_map.json` (written by the
   `source_mapping` phase via `_build_source_module_map` at
   `pipeline_tools.py:1756-1780`).
2. If the map is non-empty: for each `(week, module_type, page_id)` key, the
   value is a list of `SourceReference` dicts (per
   `schemas/knowledge/source_reference.schema.json`). Emit each page's
   JSON-LD with `sourceReferences: [...]` and each `<section>`'s wrapper
   with `data-cf-source-ids="dart:doc#b1,dart:doc#b2"` +
   `data-cf-source-primary="dart:doc#b1"`. See emit helper at
   `generate_course.py:310-315` (`_source_attr_string`).
3. If the map is empty (the common case for the current stub source-router):
   **fall back to keyword-matching** objectives against staged DART HTML.
   The existing stub already does this at `pipeline_tools.py:1266-1301`
   (`_find_relevant_sections`) — preserve that behavior but paraphrase the
   text into the new 5-page structure. With empty map, emit pages without
   `sourceReferences` and without `data-cf-source-*`. Both page and
   attribute keys are elided-when-empty per schema — not zero-length.

### Decision capture (CRITICAL)

Log a decision for each week and each generated page. Use
`lib.decision_capture.DecisionCapture(course_code, phase="content-generator",
tool="courseforge", streaming=True)`:

- `decision_type`: `"content_structure"` on first week — records the 5-page
  split choice. Rationale ≥ 20 chars.
- `decision_type`: `"content_selection"` per page — records which DART
  sections / objectives were paraphrased. Rationale ≥ 20 chars.

### Gates this output must pass

| Gate | Validator | Schema / check |
|------|-----------|----------------|
| `content_structure` (course_generation) | `lib.validators.content.ContentStructureValidator` | Week dirs exist, each has ≥ 4 HTML pages. |
| `page_objectives` (course_generation, Wave 2) | `lib.validators.page_objectives.PageObjectivesValidator` | Every page HTML has ≥ 1 `data-cf-objective-id` attr. |
| `wcag_compliance` (course_generation) | `lib.validators.wcag.WCAGValidator` | skip-link, role=main, heading hierarchy, alt text. |
| `dart_markers` (textbook_to_course) | `lib.validators.dart_markers.DartMarkersValidator` | DART section/class markers preserved when carrying forward DART content. |
| `content_type` (opt-in) | `lib.validators.content_type.ContentTypeValidator` | `data-cf-content-type` values ∈ canonical 8-value enum. |
| JSON-LD validator | `jsonschema.validate(page_jsonld, courseforge_jsonld_v1.schema.json)` | Schema validation on every emitted `<script type="application/ld+json">` body. |

### Schema gaps (do NOT patch in this block)

- `module_type` enum doesn't include `self_check` — the existing reference
  implementation uses `assessment` (or reuses `content`) for self-check
  pages. Worker α should emit `moduleType: "assessment"` for the
  self_check page to stay schema-clean.
- `schemas/knowledge/courseforge_jsonld_v1.schema.json` references a v1
  `$id` that enforces `courseCode` pattern `^[A-Z]{2,}_?\\d{3,}$`. The
  test fixture `TESTPIPE_101` matches; confirm any course code chosen
  for real runs also matches.

---

## 2. Trainforge-execution contract (Worker β)

**Function**: `_generate_assessments(**kwargs)`
**Canonical CLI**: `python -m Trainforge.process_course` (see
`Trainforge/process_course.py::build_parser` at :3292-3362 and `main` at
:3365-3425). The stub imports `AssessmentGenerator` directly but never runs
the full course processor — so chunks, graph, and misconceptions never land
on disk.

### Inputs

```
course_id          str  — e.g. "TESTPIPE_101"
imscc_path         str  — path to the .imscc written by _package_imscc
question_count     int  — honor `assessment_count` workflow param
bloom_levels       str  — comma-separated; honor workflow param
objective_ids      str  — comma-separated; upstream passes a list
project_workspace  Path — derive from course_id: state/runs/{run_id}/trainforge/
                          OR Courseforge/exports/{project_id}/trainforge/
                          (β chooses — document the choice in the
                          decision capture).
```

### Output paths

```
{project_workspace}/trainforge/
  chunks.jsonl                         # one JSON object per line
  concept_graph_semantic.json          # the typed-edge graph
  misconceptions.json                  # first-class misconception entities
  assessments.json                     # AssessmentGenerator output (well-formed)
  manifest.json                        # Trainforge's own manifest (optional)
  quality/quality_report.json          # optional; produced by CourseProcessor
training-captures/trainforge/{course_id}/
  phase_question-generation/
    decisions_{timestamp}.jsonl
```

### `chunks.jsonl` field shape

Each line is a JSON object validating against
`schemas/knowledge/chunk_v4.schema.json` (strict under
`TRAINFORGE_VALIDATE_CHUNKS=true`):

```json
{
  "id": "testpipe_101_chunk_00001",
  "schema_version": "v4",
  "chunk_type": "explanation",
  "text": "Photosynthesis is the process by which ...",
  "html": "<section data-cf-content-type=\"explanation\">...</section>",
  "follows_chunk": null,
  "source": {
    "course_id": "TESTPIPE_101",
    "module_id": "week_01",
    "lesson_id": "week_01_content_01_intro",
    "section_heading": "What is Photosynthesis?",
    "position_in_module": 0
  },
  "concept_tags": ["photosynthesis", "chloroplast"],
  "learning_outcome_refs": ["co-01"],
  "difficulty": "foundational",
  "tokens_estimate": 120,
  "word_count": 92,
  "bloom_level": "understand",
  "content_type_label": "explanation",
  "key_terms": [
    {"term": "chloroplast", "definition": "..."}
  ],
  "misconceptions": [
    {"misconception": "...", "correction": "..."}
  ],
  "run_id": "WF-...-abc12345",
  "created_at": "2026-04-20T00:00:00+00:00"
}
```

Required fields per schema (`chunk_v4.schema.json:8-22`): `id`,
`schema_version`, `chunk_type`, `text`, `html`, `follows_chunk`, `source`
(with `course_id`/`module_id`/`lesson_id`), `concept_tags`,
`learning_outcome_refs`, `difficulty`, `tokens_estimate`, `word_count`,
`bloom_level`.

`id` pattern (with `TRAINFORGE_CONTENT_HASH_IDS=true`):
`^[a-z][a-z0-9_]*_chunk_[0-9a-f]{16}$`.

Chunk count: ≥ 5 per page for a 2-week course with nontrivial content; the
fixture corpus is sized for ≥ 10 chunks total.

### `concept_graph_semantic.json` shape

Validates against `schemas/knowledge/concept_graph_semantic.schema.json`:

```json
{
  "kind": "concept_semantic",
  "generated_at": "2026-04-20T00:00:00+00:00",
  "rule_versions": {"is_a_from_key_terms": 1, "related_from_cooccurrence": 1},
  "nodes": [
    {"id": "photosynthesis", "label": "Photosynthesis", "frequency": 5,
     "occurrences": ["testpipe_101_chunk_00001", "..."]}
  ],
  "edges": [
    {
      "source": "chloroplast",
      "target": "photosynthesis",
      "type": "is-a",
      "confidence": 0.8,
      "provenance": {
        "rule": "is_a_from_key_terms",
        "rule_version": 1,
        "evidence": {
          "chunk_id": "testpipe_101_chunk_00001",
          "term": "chloroplast",
          "definition_excerpt": "The organelle where photosynthesis occurs.",
          "pattern": "is a(n)?"
        }
      }
    }
  ]
}
```

Edge `type` enum: `prerequisite | is-a | related-to | assesses | exemplifies
| misconception-of | derived-from-objective | defined-by` (8 values).
Provenance shape under strict mode validates via
`lib/validators/evidence.py::get_schema` — the discriminator is on
`provenance.rule`; the `FallbackProvenance` arm is stripped when
`TRAINFORGE_STRICT_EVIDENCE=true`.

Acceptance targets: ≥ 3 edges; ≥ 2 distinct edge `type` values in the set
(e.g. at least one `is-a` and one `related-to`).

### `misconceptions.json` shape

Validates against `schemas/knowledge/misconception.schema.json`:

```json
{
  "misconceptions": [
    {
      "id": "mc_0123456789abcdef",
      "misconception": "Plants only need sunlight to live.",
      "correction": "Plants need light, water, and CO2.",
      "concept_id": "photosynthesis",
      "lo_id": "CO-01"
    }
  ]
}
```

`id` pattern: `^mc_[0-9a-f]{16}$` (sha256 of
`misconception + "|" + correction`, first 16 hex chars).

### `assessments.json` shape

Whatever `AssessmentGenerator.to_dict()` produces today (the existing stub
already produces this via `assessment.to_dict()` at
`pipeline_tools.py:1653`) — but written as a single well-formed JSON
document, not the `json.dump()`-then-append-metadata pattern that currently
produces "Extra data" parse errors. Wave 11 question shape lives in
`schemas/knowledge/preference_pair.schema.json` and
`schemas/knowledge/instruction_pair.schema.json`; those are NOT required
outputs of this phase — they're downstream synthesis products.

Minimum:
- `len(questions) == question_count` when generator doesn't reject any
- Each question has `bloom_level ∈ bloom_levels` param
- Each question references at least one `objective_id` from `objective_ids` param

### Decision capture (CRITICAL)

Via `create_trainforge_capture(course_id, imscc_path)`:

- `decision_type`: `"content_selection"` — which chunks retrieved per question
- `decision_type`: `"question_generation"` — per question, rationale ≥ 20 chars
- `decision_type`: `"distractor_generation"` — per distractor, misconception targeted

### Gates this output must pass

| Gate | Validator | Schema |
|------|-----------|--------|
| `assessment_quality` (rag_training) | `lib.validators.assessment.AssessmentQualityValidator` | question + distractor quality scoring |
| `bloom_alignment` (rag_training) | `lib.validators.bloom.BloomAlignmentValidator` | question bloom_level matches param |
| `leak_check` (rag_training) | `lib.validators.leak.LeakChecker` | no answer-leaking stems |
| Chunk validator (opt-in) | `TRAINFORGE_VALIDATE_CHUNKS=true` | `chunk_v4.schema.json` per line |
| Evidence validator (opt-in) | `TRAINFORGE_STRICT_EVIDENCE=true` | per-rule evidence discriminator |

### Schema gaps

- `manifest.json` shape for Trainforge's own output dir is documented only
  in `Trainforge/process_course.py` (the CourseProcessor writes it to
  `output/manifest.json`) — there's no schema. Worker β should follow the
  shape produced by `CourseProcessor.process()` which is the authoritative
  emit site.

---

## 3. LibV2-archival contract (Worker γ)

**Function**: `_archive_to_libv2(**kwargs)` (registry wrapper) — the `@mcp.tool()`
variant at `pipeline_tools.py:556-726` already has 90% of what's needed. The
registry wrapper at `:1680-1754` needs to be brought to parity, PLUS it needs
to copy the Trainforge phase's outputs from `{project_workspace}/trainforge/`
into `LibV2/courses/{slug}/{corpus,graph,pedagogy,training_specs}/`.

### Inputs

```
course_name       str  — canonical course identifier (e.g. TESTPIPE_101)
domain            str  — optional; default "general"
division          str  — default "STEM"
pdf_paths         str  — comma-separated original PDFs
html_paths        str  — comma-separated DART outputs
imscc_path        str  — Courseforge IMSCC package
assessment_path   str  — Trainforge output DIR or chunks.jsonl
subdomains        str  — comma-separated
```

### Output paths

```
LibV2/courses/{slug}/
  source/
    pdf/               # shutil.copy2 from pdf_paths
    html/              # shutil.copy2 from html_paths (Worker γ: plus .quality.json if adjacent)
    imscc/             # shutil.copy2 from imscc_path
  corpus/
    chunks.jsonl       # copied from trainforge/ output
  graph/
    concept_graph_semantic.json   # copied from trainforge/ output
    misconceptions.json            # copied from trainforge/ output
  pedagogy/            # populated if Courseforge emits pedagogy JSON-LD sidecars
                       # (today: empty; leave dir present)
  training_specs/
    assessments.json   # copied from trainforge/ output
  quality/
    quality_report.json  # if Trainforge emitted one
  manifest.json        # see shape below
```

Slug: `course_name.lower().replace("_", "-").replace(" ", "-")`.

### manifest.json shape

Use the shape already emitted by the MCP tool variant at
`pipeline_tools.py:686-704`:

```json
{
  "libv2_version": "1.2.0",
  "slug": "testpipe-101",
  "import_timestamp": "2026-04-20T00:00:00",
  "classification": {
    "division": "STEM",
    "primary_domain": "general",
    "subdomains": []
  },
  "source_artifacts": {
    "pdf": [{"path": "...", "checksum": "sha256-hex", "size": 1234}],
    "html": [...],
    "imscc": {"path": "...", "checksum": "...", "size": 5678}
  },
  "provenance": {
    "source_type": "textbook_to_course_pipeline",
    "import_pipeline_version": "1.0.0"
  },
  "features": {
    "source_provenance": false,
    "evidence_source_provenance": false
  }
}
```

### Feature flag population

Use the existing helpers (DO NOT reimplement):
- `_detect_source_provenance(course_dir)` at `pipeline_tools.py:42-75`
  — scans `{course_dir}/corpus/chunks.jsonl` for chunks carrying
  `source.source_references[]`. Returns `True` iff ≥ 1 chunk has it
  populated.
- `_detect_evidence_source_provenance(course_dir)` at `pipeline_tools.py:78-110`
  — scans `{course_dir}/graph/concept_graph_semantic.json` for
  `edges[].provenance.evidence.source_references[]`. Returns `True`
  iff ≥ 1 edge evidence has it populated.

Call each after copying the respective file in and use the booleans
literally as the `features.*` values.

### Copy strategy

Byte-level `shutil.copy2` — never transform. The schemas at
`corpus/chunks.jsonl` and `graph/concept_graph_semantic.json` already
carry source_references[] threaded by Worker β; Worker γ MUST NOT
re-emit or filter those fields. Tests in
`MCP/tests/test_archive_libv2_provenance_flag.py` enforce this.

### Decision capture

Via `DecisionCapture(course_code=course_name, phase="archivist",
tool="libv2", streaming=True)`:

- `decision_type`: `"content_selection"` — records the set of artifacts
  archived, rationale ≥ 20 chars.

### Gates this output must pass

| Gate / check | Validator | Schema |
|--------------|-----------|--------|
| manifest shape | implicit / ad-hoc | no JSON schema today (gap noted below) |
| source_provenance flag | `MCP/tests/test_archive_libv2_provenance_flag.py` | passes when features flag matches chunks.jsonl content |
| evidence_source_provenance flag | same test file | passes when flag matches graph edges |
| all expected dirs exist | integration test | `corpus/`, `graph/`, `pedagogy/`, `training_specs/`, `quality/`, `source/{pdf,html,imscc}/` |

### Schema gaps

- There is no `libv2_manifest.schema.json` today. The shape above is
  authoritative by convention (it's what the MCP tool variant emits and
  what LibV2 tooling at `LibV2/tools/libv2/` reads). Formalizing this
  into a schema under `schemas/library/` would be a small PR — NOT done
  in this block.
- `pedagogy/` is reserved for Courseforge pedagogy JSON-LD sidecars.
  Today Courseforge emits the JSON-LD inline inside each page — no
  sidecar — so this directory is always empty. Future Wave work would
  extract and replicate the JSON-LD bodies under `pedagogy/`.

---

## Cross-cutting requirements

### Pipeline must remain backward compatible

- All eight `TRAINFORGE_*` / `DECISION_VALIDATION_STRICT` opt-in flags
  default OFF. Workers α/β/γ emit the richer shapes **unconditionally**
  when the data is there; they do NOT predicate emit on flag state. The
  flags only gate validation / ID-stability / evidence-arm shape.

### Full test suite must still pass

The `tests/integration/test_pipeline_end_to_end.py` integration test is
marked `@pytest.mark.slow` so the default `pytest` run excludes it. Workers
α/β/γ run it locally with `pytest -m slow tests/integration/`.

### Decision capture rationale length

Every decision event's `rationale` ≥ 20 characters. Enforced by
`lib/decision_capture.py`; `DECISION_VALIDATION_STRICT=true` also validates
`decision_type` against the 44-value enum in
`schemas/events/decision_event.schema.json`.

---

## Shared fixtures (BLOCK-produced)

These land alongside this doc in `tests/fixtures/pipeline/`:

| Fixture | Purpose | Consumer |
|---------|---------|----------|
| `fixture_corpus.pdf` | Tiny PDF — 2–3 pages, 2 sections, 1 misconceptions section | α (DART input), integration test |
| `reference_week_01/*.html` | 5 hand-crafted Courseforge pages with full `data-cf-*` + JSON-LD | α (target shape), integration test |
| `reference_libv2/corpus/chunks.jsonl` | 3-chunk reference validating under chunk_v4 strict | γ (archival target shape) |
| `reference_libv2/graph/concept_graph_semantic.json` | 5-node reference validating under concept_graph strict | γ + β |
| `reference_libv2/graph/misconceptions.json` | 1-entity reference validating under misconception schema | β + γ |

### Contents topic

Fixtures use **photosynthesis basics** as the topic — generic, non-Ed4All,
not overlapping with any research corpus. This keeps the test isolated from
repo-specific content.

---

## End-to-end integration test

`tests/integration/test_pipeline_end_to_end.py` — marked `@pytest.mark.slow`
and `@pytest.mark.integration`. Gates the parallel-worker completion by
asserting the three contracts above. **The test is committed failing** — it
is expected to fail until all three workers land. Running `pytest -m slow`
after each worker merge checks progress.

Assertions enforced (per stub):
- α: HTML pages exist at the 5-file-per-week paths; first page carries
  JSON-LD + `data-cf-role="template-chrome"` + ≥1 `data-cf-objective-id`;
  JSON-LD validates against `courseforge_jsonld_v1.schema.json`; content
  is NOT the old `DIGPED 101` hardcoded template.
- β: `chunks.jsonl` exists; ≥ 5 chunks; each validates under
  `chunk_v4.schema.json` strict; `concept_graph_semantic.json` has
  ≥ 3 edges spanning ≥ 2 distinct `type` values; `misconceptions.json`
  has ≥ 1 entity with `id` matching `^mc_[0-9a-f]{16}$`.
- γ: `LibV2/courses/testpipe-101/corpus/chunks.jsonl` exists (copied
  byte-for-byte from Trainforge output); `graph/` files landed;
  `manifest.features.source_provenance` key exists.
