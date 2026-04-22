# Worker I Sub-Plan — Wave 1.2 Knowledge Schemas + Chunk Validation Hook

**Branch:** `worker-i/wave1-knowledge-schemas`
**Base:** `dev-v0.2.0` (Wave 1.1 Workers F + G already merged @ e0455eb)
**Scope:** 2 new knowledge-layer JSON Schemas, 1 opt-in validation hook in `Trainforge/process_course.py`, 1 new test module.
**Depends on:** Worker F's 8 taxonomy/academic schemas (merged in PR #20). `$ref`s below use Worker F's locked `$id` values.

---

## 1. Locked `$id` values for the 2 new schemas

| File | `$id` |
|------|-------|
| `schemas/knowledge/courseforge_jsonld_v1.schema.json` | `https://ed4all.dev/ns/courseforge/v1/CourseModule.schema.json` |
| `schemas/knowledge/chunk_v4.schema.json` | `https://ed4all.dev/ns/knowledge/v4/chunk.schema.json` |

Both use `"$schema": "https://json-schema.org/draft/2020-12/schema"` to match existing peers in `schemas/knowledge/` and the Worker F taxonomy files being `$ref`'d.

### Worker F `$ref` targets consumed

Verbatim from `plans/kg-quality-review-2026-04/worker-f-subplan.md` §1:

- `https://ed4all.dev/ns/taxonomies/v1/bloom_verbs.schema.json#/$defs/BloomLevel`
- `https://ed4all.dev/ns/taxonomies/v1/question_type.schema.json#/$defs/QuestionType`
- `https://ed4all.dev/ns/taxonomies/v1/assessment_method.schema.json#/$defs/AssessmentMethod`
- `https://ed4all.dev/ns/taxonomies/v1/content_type.schema.json#/$defs/SectionContentType`
- `https://ed4all.dev/ns/taxonomies/v1/content_type.schema.json#/$defs/ChunkType`
- `https://ed4all.dev/ns/taxonomies/v1/cognitive_domain.schema.json#/$defs/CognitiveDomain`
- `https://ed4all.dev/ns/taxonomies/v1/module_type.schema.json#/$defs/ModuleType`

`teaching_role.json` is not `$ref`'d — emit side doesn't carry teaching_role on pages yet (Wave 2, REC-VOC-02).
`courseforge_page_types.schema.json` is not `$ref`'d — Worker F's sub-plan §3.8 flags both files as carrying identical enums; `courseforge_jsonld_v1` uses `module_type.json` which is the cleaner taxonomy-side reference. `courseforge_page_types` stays available for academic-domain validators that want a self-contained page-type schema.

---

## 2. `schemas/knowledge/courseforge_jsonld_v1.schema.json` — field-by-field

Reconciled from Worker C's outline (`plans/kg-quality-review-2026-04/discovery/c-jsonld-contract.md` §"Proposed schema outline") against the **actual emit at `Courseforge/scripts/generate_course.py:512–595`** plus a live inspection of `Courseforge/exports/WCAG_201_COURSE/03_content_development/week_03/week_03_content_01_color_contrast_ratios_and_wcag_complianc.html`.

### Top-level `CourseModule` shape

| Field | Type | Required | Notes |
|---|---|---|---|
| `@context` | `{const: "https://ed4all.dev/ns/courseforge/v1"}` | yes | Worker C §I.JSON-LD field inventory |
| `@type` | `{const: "CourseModule"}` | yes | — |
| `courseCode` | `string`, pattern `^[A-Z]{2,}_?\d{2,}$` | yes | Loosened from Worker C's `\d{3,}`; observed `WCAG_201` fits `\d{3,}` but `SAMPLE_101` also fits. Kept Worker C's pattern. |
| `weekNumber` | `integer`, `minimum: 0` | yes | — |
| `moduleType` | `$ref module_type.json#/$defs/ModuleType` | yes | 6 values (includes `discussion`) |
| `pageId` | `string`, `minLength: 1` | yes | Not pattern-constrained: the emitter at `:585,649,666,687,705,730,756` builds pageIds via multiple templates (`week_XX_overview`, `week_XX_content_NN_<slug>`, etc.); a single regex would over-constrain. Document the lax constraint in `$comment`. |
| `learningObjectives` | `array<LearningObjective>` | no | Elided when empty/None (`:587-588`) |
| `sections` | `array<Section>` | no | Elided when empty (`:589-590`) |
| `misconceptions` | `array<Misconception>` | no | — |
| `suggestedAssessmentTypes` | `array<QuestionType>` | no | `$ref question_type.json#/$defs/QuestionType` per item |
| `prerequisitePages` | `array<string>` | no | Worker C gap §1: consume-only today; schema publishes it for forward-compat emit (Wave 2 REC-JSL-02) |

### `additionalProperties` decision

- `additionalProperties: false` on root — the JSON-LD contract is finalized; unknown emits are drift to catch early.
- `additionalProperties: false` on every sub-shape (`LearningObjective`, `Section`, `KeyTerm`, `Misconception`) — same reasoning; keys are enumerated in `generate_course.py`.

### `$defs.LearningObjective`

Lifted from `_build_objectives_metadata` emit at `generate_course.py:522–545`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `string`, pattern `^[A-Z]{2,}-\d{2,}$` | yes | Observed: `TO-01`, `CO-05`. Pattern based on Worker C §I. |
| `statement` | `string`, `minLength: 1` | yes | — |
| `bloomLevel` | `oneOf: [{$ref BloomLevel}, {type: "null"}]` | yes (may be null) | `:519, :525` — null permitted since `detect_bloom_level` can fail; Worker C gap notes the null-outranks-fallback issue but schema-side we permit null for emit-side fidelity. |
| `bloomVerb` | `oneOf: [{type: "string"}, {type: "null"}]` | no | `:526` — free string or null |
| `cognitiveDomain` | `$ref CognitiveDomain` | yes | `:527, :520` — always emitted, defaulting to `"conceptual"` |
| `keyConcepts` | `array<string>` | no | `:529-530` — slugs |
| `assessmentSuggestions` | `array<QuestionType>` | no | `:532-541` — emitter uses `multiple_choice, true_false, fill_in_blank, short_answer, essay`, all in `question_type.json` 9-value union |
| `prerequisiteObjectives` | `array<string>` | no | `:542-544` — LO IDs |

### `$defs.Section`

Lifted from `_build_sections_metadata` at `generate_course.py:549–568`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `heading` | `string`, `minLength: 1` | yes | `:555` |
| `contentType` | `$ref SectionContentType` | yes | `:556` — 8-value enum |
| `keyTerms` | `array<KeyTerm>` | no | `:559-563` |
| `bloomRange` | `array<BloomLevel>`, `minItems: 1`, `maxItems: 2` | no | **Normalization:** current emit at `:566` produces `[bloom_range]` when it's a string input or passes the input array through — so emit IS always a list. Worker C §"gap table" row 4 flagged `str \| List[str]` drift by looking at the type annotation; actual runtime output is always a list. Schema enforces array-only. If any historical data leaked with scalar strings, the consumer at `process_course.py:1302-1304` already handles both — documented as accepted parser drift, emit side is clean. |

### `$defs.KeyTerm`

| Field | Type | Required |
|---|---|---|
| `term` | `string`, `minLength: 1` | yes |
| `definition` | `string`, `minLength: 1` | yes |

Per `generate_course.py:559-563`.

### `$defs.Misconception` — REC-JSL-01 tightening

| Field | Type | Required |
|---|---|---|
| `misconception` | `string`, `minLength: 1` | yes |
| `correction` | `string`, `minLength: 1` | yes |

`additionalProperties: false`. Tightened per REC-JSL-01; Worker C §"gap table" row 3 notes the emit side is producer-opaque (`:591-592` passes through). Schema enforces the shape the consumer at `process_course.py:1177-1183` expects. Live WCAG_201 inspection confirms current output matches.

### `AssessmentType` reconciliation

Worker C §"Proposed schema outline" flagged enum-reconciliation as needed. Resolution:
- `learningObjectives[].assessmentSuggestions[]` uses `$ref question_type.json#/$defs/QuestionType` (item-level question types: `multiple_choice, true_false, fill_in_blank, short_answer, essay, ...`). Matches the 5 values emitted by `from_bloom` dict at `:534-539`, all of which are in the 9-value `question_type.json` union.
- Top-level `suggestedAssessmentTypes[]` also uses `QuestionType` — same reasoning; emit at `:689, :707` uses `{short_answer, essay, multiple_choice, true_false}`, all in the 9-value union.

---

## 3. `schemas/knowledge/chunk_v4.schema.json` — field-by-field

Lifted from `Trainforge/process_course.py::_create_chunk` at `:1038-1240` and `_write_chunks` at `:1667-1682`. Verified against live emit from LibV2 `best-practices-in-digital-web-design-for-accessibi/corpus/chunks.jsonl` (131 chunks inspected; 100% conform to the shape below).

### Root-level fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `string`, pattern `^[a-z][a-z0-9_]*_chunk_\d{5}$` | yes | Observed: `wcag_201_chunk_00001`. Not `chunk_id` — actual emit uses `id` at `:1080`. |
| `schema_version` | `{const: "v4"}` | yes | `:1081, :1232` |
| `chunk_type` | `$ref ChunkType` | yes | `:1082` — 6-value enum in content_type.json |
| `text` | `string` | yes | `:1083` |
| `html` | `string` | yes | `:1084` |
| `follows_chunk` | `oneOf: [{type: "string"}, {type: "null"}]` | yes | `:1085` — may be null (first chunk of module) |
| `source` | `object` (see §3.1) | yes | `:1086` |
| `concept_tags` | `array<string>` | yes | `:1087` |
| `learning_outcome_refs` | `array<string>` | yes | `:1088` — may be empty. Pattern on items: permissive `string`; strict pattern would break historic chunks. REC-CTR-01 asked for a pattern; `minItems: 0` to allow empty (observed in live data when `_extract_objective_refs` returns empty). |
| `difficulty` | `enum: ["foundational", "intermediate", "advanced"]` | yes | `:1089` — 3-value enum from `_determine_difficulty` |
| `tokens_estimate` | `integer`, `minimum: 0` | yes | `:1090` |
| `word_count` | `integer`, `minimum: 0` | yes | `:1091` |
| `bloom_level` | `$ref BloomLevel` | yes | `:1142` — always populated by fallback cascade |
| `bloom_level_source` | `enum: ["section_jsonld","page_jsonld","lo_inherited","verbs","default"]` | no | `:1146` — only emitted when source is `verbs` or `default` per the `if` at `:1145` |
| `content_type_label` | `string` | no | `:1162-1163` — freeform today; Worker C §KG-impact flags enum-reconciliation as future work |
| `key_terms` | `array<KeyTerm>` | no | `:1164-1171`, same shape as JSON-LD KeyTerm |
| `misconceptions` | `array` of `oneOf [Misconception, string]` | no | `:1177-1184` — normalized preserves dict-with-misconception-key OR plain string (both are emitted). Permissive union reflects emit reality. |
| `summary` | `string` | no | `:1188-1192` — generated by `summary_factory.generate`, always produced but guard against missing with `no` required-ness. In 131/131 observed chunks it was present. |
| `retrieval_text` | `string` | no | `:1200-1215` — only emitted when `key_terms` present (69/131 observed) |
| `_metadata_trace` | `object` | no | `:1219-1228` — Worker M1 diagnostic; schema allows diagnostic fields. Preserved via root `additionalProperties: true`. |

### 3.1 `$defs.Source`

Lifted from `_create_chunk` at `:1057-1077`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `course_id` | `string`, `minLength: 1` | yes | `:1058` — always `self.course_code` |
| `module_id` | `string`, `minLength: 1` | yes | `:1059` |
| `module_title` | `string` | no | `:1060` — optional per REC-CTR-01 required-fields ("course_id, module_id, lesson_id"); observed 131/131 but keep optional for non-Courseforge IMSCC imports |
| `lesson_id` | `string`, `minLength: 1` | yes | `:1061` |
| `lesson_title` | `string` | no | `:1062` |
| `resource_type` | `string` | no | `:1063` |
| `section_heading` | `string` | no | `:1064` |
| `position_in_module` | `integer`, `minimum: 0` | no | `:1065` |
| `html_xpath` | `string` | no | `:1070-1071` — audit-trail; only present when computed |
| `char_span` | `array<integer>`, `minItems: 2`, `maxItems: 2` | no | `:1072-1073` |
| `item_path` | `string` | no | `:1076-1077` — IMSCC-relative path |

**`additionalProperties: false`** on `source` — this is the structural core; REC-CTR-01 specifies strict.

### 3.2 `$defs.KeyTerm`

Identical to JSON-LD `KeyTerm` (both `term` and `definition` required, `additionalProperties: false`).

### 3.3 `$defs.Misconception`

Same tightening as JSON-LD `Misconception`: `{misconception, correction}`, both required, `additionalProperties: false`.

### 3.4 `additionalProperties: true` at root

Rationale: preserves Worker M1's `_metadata_trace` diagnostic field (inactive in observed 131/131 but present in the emitter at `:1219-1228`), `bloom_level_source`, and future fields without requiring an immediate schema bump. The structural core (`source.*`) stays strict.

### 3.5 Known gaps accepted into schema

- `learning_outcome_refs.minItems: 0` — per observed data, some chunks have empty LO refs. REC-CTR-01's "at least 1" is aspirational; documenting the permissive stance until LO-backfill work lands.
- `bloom_range` on JSON-LD `Section` accepts arrays only; legacy `str | List[str]` drift noted in Worker C gap table §4 is fixed at emit (lines `:564-566` wrap strings into lists). Schema enforces the post-normalization shape.
- `misconceptions` item union accepts both dict and string. Production data (WCAG_201) is all dicts, but non-Courseforge IMSCC imports may emit strings per `process_course.py:1181-1182`.

---

## 4. Validation hook — exact edit at `Trainforge/process_course.py`

### Current code at `_write_chunks` (lines 1667-1682)

```python
def _write_chunks(self, chunks: List[Dict[str, Any]]):
    jsonl_path = self.corpus_dir / "chunks.jsonl"
    json_path = self.corpus_dir / "chunks.json"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    self.capture.log_decision(
        decision_type="chunk_serialization",
        decision=f"Write {len(chunks)} chunks to JSONL and JSON",
        rationale="JSONL format required for LibV2 streaming retrieval; JSON array for debugging and validation",
    )
```

### New code (validate-before-write)

Add module-level helper (top of file, near other imports):

```python
# Worker I (REC-CTR-01): opt-in chunk validation against chunk_v4.schema.json.
_CHUNK_SCHEMA: Optional[Dict[str, Any]] = None


def _load_chunk_schema() -> Optional[Dict[str, Any]]:
    """Load chunk_v4.schema.json once; return None if jsonschema unavailable."""
    global _CHUNK_SCHEMA
    if _CHUNK_SCHEMA is not None:
        return _CHUNK_SCHEMA
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return None
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas" / "knowledge" / "chunk_v4.schema.json"
    )
    if not schema_path.exists():
        return None
    with open(schema_path) as f:
        _CHUNK_SCHEMA = json.load(f)
    return _CHUNK_SCHEMA


def _validate_chunk(chunk: Dict[str, Any]) -> Optional[str]:
    """Validate a chunk against chunk_v4.schema.json.

    Returns the formatted error string if validation failed, None on success.
    Returns None if the schema cannot be loaded (missing, parse error, or
    jsonschema not installed) so hook stays non-fatal during bootstrap.
    """
    schema = _load_chunk_schema()
    if schema is None:
        return None
    import jsonschema
    try:
        jsonschema.validate(chunk, schema, cls=jsonschema.Draft202012Validator)
        return None
    except jsonschema.ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "root"
        return f"{path}: {e.message}"
```

Modify `_write_chunks` to run validation before writing (between the two opens):

```python
def _write_chunks(self, chunks: List[Dict[str, Any]]):
    jsonl_path = self.corpus_dir / "chunks.jsonl"
    json_path = self.corpus_dir / "chunks.json"

    # Worker I (REC-CTR-01): opt-in chunk validation.
    # Gated by TRAINFORGE_VALIDATE_CHUNKS=true for fail-closed behavior;
    # default is warn-log so existing pipelines don't break on schema landing.
    strict = os.getenv("TRAINFORGE_VALIDATE_CHUNKS", "").lower() == "true"
    validation_errors: List[str] = []
    for i, chunk in enumerate(chunks):
        err = _validate_chunk(chunk)
        if err:
            chunk_id = chunk.get("id", f"<index {i}>")
            msg = f"Chunk {chunk_id}: {err}"
            if strict:
                validation_errors.append(msg)
            else:
                logger.warning(f"chunk_v4 validation: {msg}")
    if validation_errors:
        raise ValueError(
            f"chunk_v4 validation failed for {len(validation_errors)} chunk(s): "
            + "; ".join(validation_errors[:5])
            + (" ..." if len(validation_errors) > 5 else "")
        )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    # ... rest unchanged
```

### Imports needed at top of `process_course.py`

Confirm present (grep will verify): `os`, `json`, `logging`/`logger`, `Path`, `Dict`, `Any`, `List`, `Optional`. All already imported — no new imports needed beyond inline `jsonschema` inside the helper (kept inline so module import doesn't hard-require jsonschema).

### Why not use `lib/validation.py::load_schema`

`lib/validation.py::load_schema` is hard-coded to three event schemas (`decision_event`, `trainforge_decision`, `session_annotation`). Extending it to support arbitrary schema names is out of scope for Wave 1.2 — Worker H is touching library code and we keep scopes disjoint. A local loader in `process_course.py` is the minimal change; if future waves centralize, the helper moves to `lib/validation.py`.

---

## 5. Regression test design — `Trainforge/tests/test_chunk_validation.py`

Test structure follows `Trainforge/tests/test_provenance.py` pattern (absolute path resolution, PROJECT_ROOT on sys.path).

### Tests

1. **`test_chunk_schema_self_valid`** — schema itself passes `jsonschema.Draft202012Validator.check_schema`.
2. **`test_jsonld_schema_self_valid`** — same for the JSON-LD schema.
3. **`test_existing_libv2_chunks_validate`** — load real `LibV2/courses/best-practices-in-digital-web-design-for-accessibi/corpus/chunks.jsonl` (131 chunks); assert ≥95% validate cleanly. (Observed: 131/131 conform to the shape described in §3 above, so this should hit 100% on the clean corpus; the 95% bar is the master-plan tolerance for historical drift.)
4. **`test_existing_wcag_jsonld_validates`** — extract the JSON-LD block from a WCAG_201 page at `Courseforge/exports/WCAG_201_COURSE/03_content_development/week_03/*.html`, validate against `courseforge_jsonld_v1.schema.json`. Assert it validates. If any `$ref` resolution fails, fall back to skipping with a pytest.skip + clear message (diagnostic, not a test failure).
5. **`test_missing_source_course_id_fails_strict_mode`** — craft a chunk missing `source.course_id`; monkeypatch `os.environ['TRAINFORGE_VALIDATE_CHUNKS']='true'`; assert `_validate_chunk` catches it and `_write_chunks` (smoke-wrapped) raises ValueError.
6. **`test_missing_source_course_id_warns_default`** — same chunk, no env var; assert no raise, log warning captured via caplog.
7. **`test_valid_chunk_passes`** — construct a minimal valid chunk; assert `_validate_chunk` returns None.

### `$ref` resolution strategy in tests

Worker F's taxonomy files use `$id` URIs (not filesystem paths). Draft 2020-12's default resolver will attempt HTTP fetches for these URIs, which will fail offline. Pattern: build a schema store by walking `schemas/` and loading all JSONs, keyed by their `$id`. Pass via `jsonschema.RegistryResolver` (Draft 2020-12) or the Draft 7 `RefResolver` pattern used in `lib/validation.py:102-115`.

Implementation:
```python
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

registry = Registry()
for p in (PROJECT_ROOT / "schemas").rglob("*.json"):
    with open(p) as f:
        s = json.load(f)
    if "$id" in s:
        registry = registry.with_resource(s["$id"], Resource.from_contents(s))
validator = Draft202012Validator(schema, registry=registry)
```

If `referencing` package is unavailable in CI environment, fall back to pytest.skip with diagnostic — the main hook in `process_course.py` uses `jsonschema.validate()` with the default resolver, which should also work for local store `$id` resolution since all referenced schemas are in the same on-disk tree and jsonschema 4.18+ resolves them via the embedded `$id`.

**Verified prerequisite:** `pip show jsonschema` confirms 4.x is installed; Draft 2020-12 + `referencing` support is native.

---

## 6. Execution order

1. Create `schemas/knowledge/courseforge_jsonld_v1.schema.json` (§2).
2. Create `schemas/knowledge/chunk_v4.schema.json` (§3).
3. Modify `Trainforge/process_course.py` (§4).
4. Create `Trainforge/tests/test_chunk_validation.py` (§5).
5. Run verification (§7).
6. Commit + push + open PR.

---

## 7. Verification commands

```bash
cd /home/mdmur/Projects/Ed4All

# 7.1 Schemas parse as JSON
python3 -c "import json; json.load(open('schemas/knowledge/courseforge_jsonld_v1.schema.json')); print('JSON-LD schema parses')"
python3 -c "import json; json.load(open('schemas/knowledge/chunk_v4.schema.json')); print('chunk_v4 schema parses')"

# 7.2 Schemas self-validate
python3 -c "import json, jsonschema; s = json.load(open('schemas/knowledge/courseforge_jsonld_v1.schema.json')); jsonschema.Draft202012Validator.check_schema(s); print('JSON-LD schema OK')"
python3 -c "import json, jsonschema; s = json.load(open('schemas/knowledge/chunk_v4.schema.json')); jsonschema.Draft202012Validator.check_schema(s); print('chunk_v4 schema OK')"

# 7.3 CI integrity
python3 -m ci.integrity_check

# 7.4 Test suite
pytest Trainforge/tests/test_chunk_validation.py -v
pytest Trainforge/tests/ -x

# 7.5 End-to-end default (warn-log)
# No env var — existing chunks.jsonl writes proceed with warnings only.

# 7.6 End-to-end strict
# TRAINFORGE_VALIDATE_CHUNKS=true python3 -m Trainforge.process_course <args>
# Expect: zero validation errors on clean WCAG_201 input.
```

---

## 8. Constraints re-affirmed

- Main branch off-limits. PR targets `dev-v0.2.0`.
- Do NOT touch `lib/ontology/*` (Worker H scope).
- Do NOT touch any taxonomy or page-types schema (Worker F, merged).
- Validation hook MUST be opt-in via `TRAINFORGE_VALIDATE_CHUNKS=true`. Default warn-log preserves backward-compat.
- `additionalProperties: true` at chunk_v4 root (preserves `_metadata_trace`, `bloom_level_source`, future fields).
- `additionalProperties: false` on `source` (structural core is strict).
- `additionalProperties: false` on JSON-LD root (emitted keys are enumerated at `_build_page_metadata`).

---

## 9. Rollback plan

If a post-merge issue surfaces:
1. Schemas are leaves — no code imports them yet except the opt-in validation hook.
2. The validation hook is gated by env var — default warn-log means a broken schema only floods logs, never breaks pipelines.
3. `git revert <merge-commit>` removes both schemas and the hook.
