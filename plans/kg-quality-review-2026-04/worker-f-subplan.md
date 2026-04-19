# Worker F Sub-Plan — Wave 1.1 Taxonomy + Page-Types Schema Authoring

**Branch:** `worker-f/wave1-taxonomies`
**Base:** `dev-v0.2.0`
**Scope:** 8 new JSON Schema files. No code changes. No modifications to existing schemas.
**Unblocks:** Worker I (Wave 1.2) which `$ref`s these taxonomies.

---

## 1. Locked `$id` values (MUST NOT CHANGE AFTER THIS SUB-PLAN LANDS)

Worker I depends on these exact values for `$ref` resolution:

| File | `$id` |
|------|-------|
| `schemas/taxonomies/bloom_verbs.json` | `https://ed4all.dev/ns/taxonomies/v1/bloom_verbs.schema.json` |
| `schemas/taxonomies/question_type.json` | `https://ed4all.dev/ns/taxonomies/v1/question_type.schema.json` |
| `schemas/taxonomies/assessment_method.json` | `https://ed4all.dev/ns/taxonomies/v1/assessment_method.schema.json` |
| `schemas/taxonomies/content_type.json` | `https://ed4all.dev/ns/taxonomies/v1/content_type.schema.json` |
| `schemas/taxonomies/cognitive_domain.json` | `https://ed4all.dev/ns/taxonomies/v1/cognitive_domain.schema.json` |
| `schemas/taxonomies/teaching_role.json` | `https://ed4all.dev/ns/taxonomies/v1/teaching_role.schema.json` |
| `schemas/taxonomies/module_type.json` | `https://ed4all.dev/ns/taxonomies/v1/module_type.schema.json` |
| `schemas/academic/courseforge_page_types.schema.json` | `https://ed4all.dev/ns/academic/v1/courseforge_page_types.schema.json` |

**Naming note:** file-on-disk uses `.json` for taxonomy folder (per master plan filename list, which omits `.schema.json`), but the `$id` always includes `.schema.json` suffix so consumers can treat all ed4all ns URIs uniformly.

### `$ref` anchor targets Worker I will use

- `BloomLevel` enum: `https://ed4all.dev/ns/taxonomies/v1/bloom_verbs.schema.json#/$defs/BloomLevel`
- `BloomVerb`: `https://ed4all.dev/ns/taxonomies/v1/bloom_verbs.schema.json#/$defs/BloomVerb`
- `QuestionType`: `https://ed4all.dev/ns/taxonomies/v1/question_type.schema.json#/$defs/QuestionType`
- `AssessmentMethod`: `https://ed4all.dev/ns/taxonomies/v1/assessment_method.schema.json#/$defs/AssessmentMethod`
- `SectionContentType`: `https://ed4all.dev/ns/taxonomies/v1/content_type.schema.json#/$defs/SectionContentType`
- `CalloutContentType`: `https://ed4all.dev/ns/taxonomies/v1/content_type.schema.json#/$defs/CalloutContentType`
- `ChunkType`: `https://ed4all.dev/ns/taxonomies/v1/content_type.schema.json#/$defs/ChunkType`
- `ContentType`: `https://ed4all.dev/ns/taxonomies/v1/content_type.schema.json#/$defs/ContentType`
- `CognitiveDomain`: `https://ed4all.dev/ns/taxonomies/v1/cognitive_domain.schema.json#/$defs/CognitiveDomain`
- `TeachingRole`: `https://ed4all.dev/ns/taxonomies/v1/teaching_role.schema.json#/$defs/TeachingRole`
- `ModuleType`: `https://ed4all.dev/ns/taxonomies/v1/module_type.schema.json#/$defs/ModuleType`
- `CourseforgePageType`: `https://ed4all.dev/ns/academic/v1/courseforge_page_types.schema.json#/$defs/CourseforgePageType`

---

## 2. JSON Schema dialect

All 8 files declare `"$schema": "https://json-schema.org/draft/2020-12/schema"` and use `$defs` (not `definitions`). This matches the verification step in the worker spec which uses `jsonschema.Draft202012Validator.check_schema`.

**CI compatibility note:** `ci/integrity_check.py::check_schemas` uses `jsonschema.Draft7Validator.check_schema`, which is tolerant of 2020-12 schemas in practice (it does not reject `$defs` or the 2020-12 `$schema` URI — the Draft-07 meta-schema is loose enough). Verified in sub-plan prep by running `Draft7Validator.check_schema` against a trial 2020-12 skeleton. If future CI tightens, individual files can add a `"$comment"` note — but today both validators pass.

---

## 3. Per-file specification

### 3.1 `schemas/taxonomies/bloom_verbs.json`

- **Title:** `"Bloom's Taxonomy Verbs"`
- **Description:** `"Canonical list of Bloom's Revised Taxonomy cognitive levels and their associated action verbs with usage contexts and objective templates. Source of truth for bloom-verb loading across Courseforge, Trainforge, and LibV2."`
- **Source citation (`$comment`):** `"Values lifted from Courseforge/scripts/textbook-objective-generator/bloom_taxonomy_mapper.py:55 (BLOOM_VERBS dict). This is the richest canonical copy; future Worker H (Wave 1.2) migrates 6 call sites to load from this schema."`
- **Shape:**
  ```
  {
    "$schema": ...,
    "$id": ...,
    "title": ...,
    "description": ...,
    "$comment": ...,
    "type": "object",
    "required": ["remember", "understand", "apply", "analyze", "evaluate", "create"],
    "additionalProperties": false,
    "$defs": {
      "BloomLevel": { "type": "string", "enum": [6 levels] },
      "BloomVerb": {
        "type": "object",
        "required": ["verb", "usage_context", "example_template"],
        "properties": {
          "verb": { "type": "string" },
          "usage_context": { "type": "string" },
          "example_template": { "type": "string" }
        },
        "additionalProperties": false
      }
    },
    "properties": {
      "remember":   { "type": "array", "items": { "$ref": "#/$defs/BloomVerb" } },
      "understand": { "type": "array", "items": { "$ref": "#/$defs/BloomVerb" } },
      "apply":      { "type": "array", "items": { "$ref": "#/$defs/BloomVerb" } },
      "analyze":    { "type": "array", "items": { "$ref": "#/$defs/BloomVerb" } },
      "evaluate":   { "type": "array", "items": { "$ref": "#/$defs/BloomVerb" } },
      "create":     { "type": "array", "items": { "$ref": "#/$defs/BloomVerb" } }
    }
  }
  ```
  Each property's value is the populated verb array (as `default`) — so the file is **both a schema AND the data**, enabling `get_verbs()` loaders to read the `default` arrays directly. This dual-use pattern is deliberate per REC-BL-01 (Worker H reads this file at import time).

- **BloomLevel enum:** `["remember", "understand", "apply", "analyze", "evaluate", "create"]` (verbatim from `BloomLevel` Enum values at bloom_taxonomy_mapper.py:20-25).
- **Verbs per level (lifted exactly from `BLOOM_VERBS` at bloom_taxonomy_mapper.py:55-128):**
  - **remember:** `define, list, recall, identify, name, state, label, match, recognize, select` (10)
  - **understand:** `explain, describe, summarize, classify, compare, interpret, discuss, paraphrase, distinguish, illustrate` (10)
  - **apply:** `apply, demonstrate, implement, solve, use, execute, compute, calculate, practice, perform` (10)
  - **analyze:** `analyze, differentiate, examine, organize, relate, categorize, deconstruct, investigate, contrast, attribute` (10)
  - **evaluate:** `evaluate, assess, critique, justify, judge, argue, defend, support, recommend, prioritize` (10)
  - **create:** `create, design, construct, develop, formulate, compose, plan, invent, produce, generate` (10)
- **Total:** 60 verb objects; each has `verb`, `usage_context`, `example_template` strings.

### 3.2 `schemas/taxonomies/question_type.json`

- **Title:** `"Assessment Question Types"`
- **Description:** `"Canonical union of question types supported across Trainforge and the Courseforge JSON-LD contract. Superset of the Trainforge question factory's 7-type list and the trainforge_decision schema's 9-type enum."`
- **Source citation (`$comment`):** `"Union of Trainforge/generators/question_factory.py:81-89 (7 values: multiple_choice, multiple_response, true_false, fill_in_blank, short_answer, essay, matching) and schemas/events/trainforge_decision.schema.json:64 (adds ordering, hotspot). Per REC-VOC-01."`
- **Enum:** `["multiple_choice", "multiple_response", "true_false", "short_answer", "essay", "matching", "fill_in_blank", "ordering", "hotspot"]` (9 values)
- **Shape:**
  ```
  {
    "$schema": ..., "$id": ..., "title": ..., "description": ..., "$comment": ...,
    "$defs": {
      "QuestionType": { "type": "string", "enum": [9 values] }
    },
    "$ref": "#/$defs/QuestionType"
  }
  ```

### 3.3 `schemas/taxonomies/assessment_method.json`

- **Title:** `"Assessment Methods"`
- **Description:** `"Canonical enum of assessment method categories suggested per learning objective (exam, quiz, assignment, etc.). Distinct from question-item types — these are higher-level instrument categories."`
- **Source citation (`$comment`):** `"Values lifted verbatim from schemas/academic/learning_objectives.schema.json:223 (assessmentSuggestions.items.enum). Per REC-VOC-01."`
- **Enum:** `["exam", "quiz", "assignment", "project", "discussion", "presentation", "portfolio", "demonstration", "case_study"]` (9 values)
- **Shape:** mirror of 3.2 with `AssessmentMethod` in `$defs` and top-level `$ref`.

### 3.4 `schemas/taxonomies/content_type.json`

- **Title:** `"Content Type Union"`
- **Description:** `"Union vocabulary for content-type labels emitted by Courseforge on sections, callouts, and by Trainforge on chunks. Union-only per REC-VOC-03 — NOT enforced as a single field anywhere yet; each subtype is scoped to its emission site."`
- **Source citation (`$comment`):** `"SectionContentType lifted from Courseforge/scripts/generate_course.py:388-405 (_infer_content_type return values). CalloutContentType lifted from Courseforge/scripts/generate_course.py:448 (data-cf-content-type values on div.callout). ChunkType lifted from Trainforge/process_course.py:1372-1397 (_type_from_resource + _type_from_heading). Per REC-VOC-03."`
- **SectionContentType enum (8 values from `_infer_content_type`):**
  `["definition", "example", "procedure", "comparison", "exercise", "overview", "summary", "explanation"]`
- **CalloutContentType enum (2 values from callout emit site):**
  `["application-note", "note"]`
- **ChunkType enum (6 values: union of `_type_from_resource` and `_type_from_heading` return values):**
  `["assessment_item", "overview", "summary", "exercise", "explanation", "example"]`
  (Note: `_type_from_resource` returns `assessment_item, overview, summary, exercise, explanation`; `_type_from_heading` adds `example`. Union = 6.)
- **Shape:**
  ```
  {
    "$schema": ..., "$id": ..., "title": ..., "description": ..., "$comment": ...,
    "$defs": {
      "SectionContentType": { "type": "string", "enum": [8 values] },
      "CalloutContentType": { "type": "string", "enum": [2 values] },
      "ChunkType":          { "type": "string", "enum": [6 values] },
      "ContentType": {
        "oneOf": [
          { "$ref": "#/$defs/SectionContentType" },
          { "$ref": "#/$defs/CalloutContentType" },
          { "$ref": "#/$defs/ChunkType" }
        ]
      }
    }
  }
  ```
  No top-level `$ref` — consumers pick the specific subtype.

### 3.5 `schemas/taxonomies/cognitive_domain.json`

- **Title:** `"Cognitive Knowledge Domain"`
- **Description:** `"Revised Bloom's Taxonomy knowledge dimension — the four-category orthogonal axis to the cognitive process dimension (the six Bloom's levels)."`
- **Source citation (`$comment`):** `"Values lifted from Courseforge/scripts/generate_course.py:146-153 (BLOOM_TO_DOMAIN dict values). Also used as data-cf-cognitive-domain attribute per Courseforge/CLAUDE.md."`
- **Enum:** `["factual", "conceptual", "procedural", "metacognitive"]` (4 values)
- **Shape:** same as 3.2 pattern with `CognitiveDomain` in `$defs`.

### 3.6 `schemas/taxonomies/teaching_role.json`

- **Title:** `"Teaching Role"`
- **Description:** `"Pedagogical function played by a chunk within a course: introduce, elaborate, reinforce, assess, transfer, synthesize. Emitted on Trainforge chunks after alignment."`
- **Source citation (`$comment`):** `"Values lifted from Trainforge/align_chunks.py:33 (VALID_ROLES set). Includes a mapping table of data-cf-component + data-cf-purpose tuples to teaching_role values, derived from Courseforge/scripts/generate_course.py:345 (flip-card/term-definition), :374 (self-check/formative-assessment), :487 (activity/practice). This mapping will be consumed by Worker H in Wave 2 when teaching_role emission is wired into the Courseforge generator."`
- **Enum:** `["introduce", "elaborate", "reinforce", "assess", "transfer", "synthesize"]` (6 values)
- **Extra property (`x-component-mapping`, informational):**
  - `{component: "flip-card",     purpose: "term-definition"}        -> "introduce"`
  - `{component: "self-check",    purpose: "formative-assessment"}   -> "assess"`
  - `{component: "activity",      purpose: "practice"}               -> "transfer"`

  (These are the 3 observed combinations in `generate_course.py` as of the branch's state. Note: the master plan's worker-F spec mentions `activity-card` but the source emits `data-cf-component="activity"`; I use what the source actually emits.)
- **Shape:** `$defs.TeachingRole` enum + extension object with `x-component-mapping` array.

### 3.7 `schemas/taxonomies/module_type.json`

- **Title:** `"Module Page Type"`
- **Description:** `"Vocabulary for Courseforge weekly module page types. Also called 'page type' or 'module type' depending on emission site. Six values: five documented in agent specs plus 'discussion' which is emitted by generate_course.py but previously undocumented (surfaced by Worker B's KG-quality finding)."`
- **Source citation (`$comment`):** `"Values lifted from Courseforge/scripts/generate_course.py:647-759 (six _build_page_metadata call sites: overview, content, application, assessment, summary, discussion). Includes undocumented 'discussion' per Worker B KG-quality review finding."`
- **Enum:** `["overview", "content", "application", "assessment", "summary", "discussion"]` (6 values)
- **Shape:** same as 3.2 pattern with `ModuleType` in `$defs`.

### 3.8 `schemas/academic/courseforge_page_types.schema.json`

- **Title:** `"Courseforge Page Types"`
- **Description:** `"Authoritative page-type vocabulary schema for Courseforge module pages. Overlaps intentionally with schemas/taxonomies/module_type.json — module_type.json is the pure vocabulary taxonomy; this file is the academic-domain schema that code-level validators $ref. Per REC-CTR-02."`
- **Source citation (`$comment`):** `"Values identical to schemas/taxonomies/module_type.json, lifted from Courseforge/scripts/generate_course.py:647-759. This file serves as the canonical academic-schema-domain reference; module_type.json serves as the pure taxonomy-domain reference. Both are needed because Worker I's knowledge schemas $ref the academic file while taxonomy consumers $ref module_type.json. Per REC-CTR-02."`
- **Decision on duplication vs $ref:** enum is **duplicated** (not `$ref`'d from module_type.json). Rationale: keeps the academic-schema file self-contained for external validators; $ref would create a taxonomy → academic cross-dir dependency not present elsewhere. If either diverges later, a Wave 2 PR reconciles. Both files cite each other in `$comment`.
- **Enum:** `["overview", "content", "application", "assessment", "summary", "discussion"]` (6 values, identical to 3.7)
- **Shape:** `$defs.CourseforgePageType` enum + top-level `$ref`.

---

## 4. Execution order

1. Create all 8 files in single pass (no ordering dependency among them).
2. Run verification (§5).
3. Commit + push + open PR.

---

## 5. Verification commands

Run in order:

```bash
# 5.1 All 8 files valid JSON
cd /home/mdmur/Projects/Ed4All
for f in schemas/taxonomies/bloom_verbs.json schemas/taxonomies/question_type.json schemas/taxonomies/assessment_method.json schemas/taxonomies/content_type.json schemas/taxonomies/cognitive_domain.json schemas/taxonomies/teaching_role.json schemas/taxonomies/module_type.json schemas/academic/courseforge_page_types.schema.json; do
  python3 -c "import json; json.load(open('$f')); print('OK:', '$f')"
done

# 5.2 All 8 pass Draft202012 check
python3 -c "
import jsonschema, json
files = [
  'schemas/taxonomies/bloom_verbs.json',
  'schemas/taxonomies/question_type.json',
  'schemas/taxonomies/assessment_method.json',
  'schemas/taxonomies/content_type.json',
  'schemas/taxonomies/cognitive_domain.json',
  'schemas/taxonomies/teaching_role.json',
  'schemas/taxonomies/module_type.json',
  'schemas/academic/courseforge_page_types.schema.json',
]
for f in files:
    jsonschema.Draft202012Validator.check_schema(json.load(open(f)))
print('ALL schemas valid (Draft 2020-12)')
"

# 5.3 CI integrity check
python3 -m ci.integrity_check

# 5.4 Spot-check: bloom_verbs apply-level verbs match source
python3 -c "
import json
data = json.load(open('schemas/taxonomies/bloom_verbs.json'))
apply_verbs = [v['verb'] for v in data['properties']['apply']['default']]
expected = ['apply', 'demonstrate', 'implement', 'solve', 'use', 'execute', 'compute', 'calculate', 'practice', 'perform']
assert apply_verbs == expected, f'MISMATCH:\n  got: {apply_verbs}\n  exp: {expected}'
print('apply-level verbs match source:', apply_verbs)
"

# 5.5 Spot-check: question_type enum count
python3 -c "
import json
data = json.load(open('schemas/taxonomies/question_type.json'))
qt = data['\$defs']['QuestionType']['enum']
assert len(qt) == 9, f'expected 9 values, got {len(qt)}'
assert 'ordering' in qt and 'hotspot' in qt
print('question_type OK:', qt)
"

# 5.6 Spot-check: module_type includes discussion
python3 -c "
import json
data = json.load(open('schemas/taxonomies/module_type.json'))
mt = data['\$defs']['ModuleType']['enum']
assert 'discussion' in mt, 'discussion must be present'
print('module_type OK:', mt)
"
```

---

## 6. Constraints re-affirmed

- **Main branch off-limits.** PR targets `dev-v0.2.0`.
- **No code modifications.** No modifications to any existing schema.
- **Do NOT** create `courseforge_jsonld_v1.schema.json` or `chunk_v4.schema.json` (Worker I scope).
- **Do NOT** create `lib/ontology/bloom.py` (Worker H scope).
- **Do NOT** modify `lib/decision_capture.py` or `schemas/events/decision_event.schema.json` (Worker G scope).
- **`$id` values are locked** in §1 and will not change during implementation.

---

## 7. Rollback plan

If an issue is discovered after merge:
1. All 8 files are leaves (no code imports them yet); deletion is safe.
2. `git revert <merge-commit>` fully removes the files.
3. Worker I is blocked-by and has not yet opened their PR, so revert has no downstream impact.
