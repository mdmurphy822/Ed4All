# Worker T Sub-Plan — REC-VOC-03 Phase 2: Opt-in content_type enforcement

**Branch:** `worker-t/wave5-content-type-enforcement`
**Base:** `dev-v0.2.0` @ `dec8e3f` (Wave 4 merged)
**Master plan:** `/home/mdmur/.claude/plans/we-have-several-branches-gentle-melody.md` — § Wave 5.1, Worker T

## Goal

REC-VOC-03 Phase 2 wires Worker F's Wave 1 `schemas/taxonomies/content_type.json` (`$defs/ChunkType` / `$defs/SectionContentType` enums) into the two free-string `content_type` fields that today accept arbitrary strings:

1. **`instruction_pair.schema.json` `content_type` field** (line 46–50, free-string today).
2. **`LibV2/tools/libv2/retriever.py::ChunkFilter.content_type_label`** (line 62, `Optional[str]`, no validation).

Gated behind `TRAINFORGE_ENFORCE_CONTENT_TYPE=true` — default behavior unchanged (backward-compat). Flag-off is the existing free-string shape; flag-on adds a single membership check against the union schema's `ChunkType` enum.

This closes the "most fragmented vocabulary in the repo" per Worker B's Wave-0 ontology review and aligns the `instruction_factory` template catalog's silent `_default` fallback with a controlled vocabulary.

---

## 1. Current-state anchors (verified 2026-04-19)

### 1a. `schemas/taxonomies/content_type.json` (Worker F, Wave 1)

Already landed; single schema file with three enum `$defs`:

- `SectionContentType`: `["definition", "example", "procedure", "comparison", "exercise", "overview", "summary", "explanation"]` — Courseforge section labels.
- `CalloutContentType`: `["application-note", "note"]` — Courseforge callout labels.
- `ChunkType`: `["assessment_item", "overview", "summary", "exercise", "explanation", "example"]` — Trainforge chunk labels (from `_type_from_resource` + `_type_from_heading` return values).

Plus a `ContentType` discriminated `oneOf` union over all three.

Worker T enforces `ChunkType` specifically (the Trainforge chunk-side union) because that's what populates instruction_pair.content_type and ChunkFilter.content_type_label. `SectionContentType` stays exposed via a separate helper for completeness / future Courseforge-side callers.

### 1b. `schemas/knowledge/instruction_pair.schema.json:46–50`

```json
"content_type": {
  "type": "string",
  "description": "Content-type label from the source chunk. Free-string (derived from content_type_label or chunk_type when the label is missing).",
  "minLength": 1
}
```

Free-string; `minLength: 1`. No enum.

### 1c. Emission site (where `content_type` is populated on instruction_pair records)

`Trainforge/generators/instruction_factory.py:183–190`:

```python
def _normalize_content_type(chunk: Dict[str, Any]) -> str:
    label = chunk.get("content_type_label")
    if label:
        return str(label).strip().lower()
    ct = chunk.get("chunk_type")
    if ct:
        return str(ct).strip().lower()
    return "explanation"
```

Flows into pair dict at line ~374 (`"content_type": content_type`), then surfaces in `Trainforge/synthesize_training.py:270` (log) and the written JSONL (`instruction_pairs.jsonl`).

Important finding: the pipeline writing instruction pairs lives in `Trainforge/synthesize_training.py::run_synthesis` (not `process_course.py` as the master plan hinted — the prompt used the general "process_course.py" as shorthand for "Trainforge pair-emission"). Hook point is `synthesize_training.py:274–275` (just before `instruction_records.append`). Wiring there matches Worker I's existing `TRAINFORGE_VALIDATE_CHUNKS` pattern (fail-closed when flag on, no-op when off).

### 1d. LibV2 ChunkFilter site

`LibV2/tools/libv2/retriever.py:62`:

```python
content_type_label: Optional[str] = None
```

Used at line 439–440 to filter chunks whose `content_type_label` field matches. Entry-point is `retrieve_chunks(...)` at line 565 (CLI `--content-type` flag wires through).

Nothing in the ChunkFilter dataclass validates the value today. A user passing `--content-type totally_bogus` silently filters everything out with no warning.

### 1e. Worker I's pattern (reference for the wiring style)

`Trainforge/process_course.py:1987–2009`:

```python
strict = os.getenv("TRAINFORGE_VALIDATE_CHUNKS", "").lower() == "true"
validation_errors: List[str] = []
for i, chunk in enumerate(chunks):
    err = _validate_chunk(chunk)
    if err is None:
        continue
    chunk_id = chunk.get("id", f"<index {i}>")
    msg = f"Chunk {chunk_id}: {err}"
    if strict:
        validation_errors.append(msg)
    else:
        logger.warning("chunk_v4 validation: %s", msg)
if validation_errors:
    preview = "; ".join(validation_errors[:5])
    suffix = " ..." if len(validation_errors) > 5 else ""
    raise ValueError(
        f"chunk_v4 validation failed for {len(validation_errors)} chunk(s): "
        f"{preview}{suffix}"
    )
```

- Env var `TRAINFORGE_VALIDATE_CHUNKS=true` toggles fail-closed.
- Default (unset / other values) → warn-log, pipeline continues.

Worker T mirrors this exactly: `TRAINFORGE_ENFORCE_CONTENT_TYPE=true` flag, default is **pass-through** (not warn — since the "warning" would fire on literally every pair a pipeline is already emitting today, which is noise). Warn-log tier would wait for Phase 3 if/when we decide to surface migration-pressure signals.

### 1f. `lib.paths` exposure

`lib/paths.py:49` defines `SCHEMAS_PATH = PROJECT_ROOT / "schemas"` (plural, not `SCHEMAS_DIR` as the prompt wrote). Sub-plan uses the actual attribute name.

### 1g. Existing tests that touch content_type

- `Trainforge/tests/test_training_synthesis.py` — asserts `content_type` in emitted pair dict. Uses whatever the mock chunks carry; must stay green with flag off.
- `LibV2/tools/libv2/tests/test_retriever_v4.py:50–53` — filters by `content_type_label="explanation"`, which is a valid ChunkType enum value. Stays green flag-on and flag-off.

No existing test uses an invalid content_type, so flag-on defaults don't break anything. 

---

## 2. Design decisions (committed)

### 2.1 Strict-schema variant vs. conditional allOf

**Chosen: strict-schema variant** (new `schemas/knowledge/instruction_pair.strict.schema.json` sibling file).

Rationale:
- JSON Schema doesn't support env-var-conditional `$ref`. `allOf` / `if` / `then` can't branch on external state.
- Worker I's Wave 3 pattern (chunk_v4 validation) chose the same approach: a Python-side validator reads a single schema file and toggles via env var. Sub-plan unifies on that pattern.
- The sibling-strict schema is a **new file**, zero touch on the existing `instruction_pair.schema.json`. Existing consumers (legacy JSONL validators, downstream SFT loaders) are byte-identical.
- The strict schema is the source of truth for flag-on behavior; Python helper reads `content_type.json#/$defs/ChunkType` enum and does a set-membership check. Schema file exists primarily for **external** validators (e.g., CI integrity check, third-party consumers) to reuse the stricter shape when they opt in.

### 2.2 ChunkType-only vs. full ContentType union

**Chosen: ChunkType only for Trainforge enforcement.** `SectionContentType` accessible via a separate helper (`get_valid_section_content_types()`) for completeness but not wired into any enforcement path in this PR.

Rationale:
- Trainforge's `content_type` field is populated from `_normalize_content_type` (instruction_factory.py:183) which chains `content_type_label` → `chunk_type` → fallback. Both upstream sources are ChunkType-domain (Courseforge tags `data-cf-content-type` with SectionContentType values which map 1:1 to ChunkType member names — `explanation`, `example`, `summary`, `overview`, `exercise` — plus `assessment_item` which is Trainforge-only). The overlap is substantial but not total.
- Full ContentType union would accept Callout values (`application-note`, `note`) which are NOT valid Trainforge chunk types. Accepting them would defeat the enforcement.
- `SectionContentType` is a superset for Courseforge side (includes `definition`, `procedure`, `comparison`) that Trainforge doesn't emit today. If a future chunk carries a SectionContentType-only value (say `definition`), flag-on enforcement will reject it — which is the correct behavior because the current emitter can't produce it. If emitters are extended later to produce them, the ChunkType enum should be extended in Wave F then, not silently accepted.

Both helpers are exposed so callers can pick their domain.

### 2.3 LibV2 ChunkFilter scope

**Keep in scope.** Diff estimate: ~30 lines (dataclass `__post_init__` validator + one new import + two new test cases). Well under the 400-line threshold.

Enforcement path for LibV2: `ChunkFilter` gets a `__post_init__` that calls `validate_chunk_type(self.content_type_label)` if the field is non-`None` and the flag is on. Flag-off behavior unchanged.

### 2.4 Flag-off behavior: pass-through, NOT warn-log

The master plan's "validator layer" phrasing could read as "warn when flag off, error when flag on". Sub-plan explicitly decides: **flag off = silent passthrough, flag on = fail-closed (raise)**. Matches Worker I's shape (Worker I is actually a warn-default because chunk_v4 is a richer validator where the default warning gives migration pressure; content_type enforcement only has one bit of state, so the "warn for every single pair" alternative is pure noise). No migration-pressure middle tier needed yet.

### 2.5 Module-level env-var read (with a test escape hatch)

```python
ENFORCE_CONTENT_TYPE = os.getenv("TRAINFORGE_ENFORCE_CONTENT_TYPE", "").lower() == "true"
```

Evaluated once at import. Tests that need to toggle the flag use `monkeypatch.setenv` + `importlib.reload` (matches `Trainforge/tests/test_chunk_validation.py:309–336` pattern). Alternative (re-reading on every call) would be more flexible but slower and mismatched with the established pattern.

To keep tests clean and avoid module-reload gymnastics, I'll expose an `_is_enforcement_enabled()` function in `content_type.py` that re-reads on each call, and `validate_chunk_type` / `validate_section_content_type` call through it. Slightly less efficient, zero reload boilerplate in tests. Precedent: `lib.secure_paths` uses the same "function over module constant" pattern for run-time-toggleable gates.

### 2.6 LRU cache on schema read

`functools.lru_cache(maxsize=1)` on `_load_content_type_schema()` — one file read per process, mirrors Worker I's `_chunk_v4_schema` caching. Prevents O(n_pairs) file reads in hot loops.

---

## 3. Files touched

### NEW

1. **`lib/validators/content_type.py`** — shared helper. ~60 lines.
   - `_load_content_type_schema()` — cached JSON load.
   - `get_valid_chunk_types() -> frozenset[str]`
   - `get_valid_section_content_types() -> frozenset[str]`
   - `_is_enforcement_enabled() -> bool` — reads `TRAINFORGE_ENFORCE_CONTENT_TYPE` each call.
   - `validate_chunk_type(value: str) -> bool` — flag-off returns True; flag-on returns `value in get_valid_chunk_types()`.
   - `validate_section_content_type(value: str) -> bool` — same shape for SectionContentType.
   - `assert_chunk_type(value, context="") -> None` — raises `ValueError` when flag on + invalid; no-op otherwise. Convenience for call-sites that want fail-closed semantics.

2. **`schemas/knowledge/instruction_pair.strict.schema.json`** — byte-for-byte copy of `instruction_pair.schema.json` EXCEPT:
   - `"$id"` renamed to `"urn:ed4all:schemas:instruction_pair_strict"`.
   - `"title"` becomes `"Instruction Pair (SFT, strict)"`.
   - `content_type` property swapped from free-string to:
     ```json
     "content_type": {
       "$ref": "../taxonomies/content_type.json#/$defs/ChunkType",
       "description": "Content-type label. Must be a member of the ChunkType enum (Trainforge union). Gated by TRAINFORGE_ENFORCE_CONTENT_TYPE=true in the Python validator layer."
     }
     ```
   - Existing file untouched.

3. **`lib/tests/test_content_type_enforcement.py`** — regression tests (6 tests listed in § 5).

### MODIFIED

4. **`Trainforge/synthesize_training.py`** — wire the validator call. Add ~15 lines:
   - Import `assert_chunk_type` (or `validate_chunk_type` + explicit raise) from `lib.validators.content_type`.
   - Just before `instruction_records.append(inst_result.pair)` (line 275), invoke validation on `inst_result.pair["content_type"]`.
   - On flag-off (or valid): no-op. On flag-on + invalid: raise `ValueError` with chunk_id + offending value. Fail-closed matches Worker I.
   - Log a decision via the existing `capture` if validation rejects (for trainability — Wave 4 Worker P added `decision_capture_id`-style fields; aligns with the decision-capture protocol).

5. **`LibV2/tools/libv2/retriever.py`** — add `__post_init__` to `ChunkFilter`:
   ```python
   def __post_init__(self):
       from lib.validators.content_type import assert_chunk_type
       if self.content_type_label is not None:
           assert_chunk_type(self.content_type_label, context="ChunkFilter.content_type_label")
   ```
   ~8 lines including the import guard.

### UNCHANGED

- `schemas/knowledge/instruction_pair.schema.json` — NOT modified (strict-variant approach).
- `schemas/taxonomies/content_type.json` — source of truth; untouched.
- `Trainforge/generators/instruction_factory.py` — `_normalize_content_type` stays as-is; validation happens at emission, not normalization.

---

## 4. LibV2 scope calculation (final)

Diff estimate:

| File | Lines added | Lines modified |
|------|-------------|----------------|
| `lib/validators/content_type.py` | ~60 | 0 (new) |
| `lib/validators/__init__.py` | ~2 (export) | 0 |
| `schemas/knowledge/instruction_pair.strict.schema.json` | ~75 (copy) | 0 (new) |
| `lib/tests/test_content_type_enforcement.py` | ~120 (6 tests) | 0 (new) |
| `Trainforge/synthesize_training.py` | ~15 | ~2 (imports) |
| `LibV2/tools/libv2/retriever.py` | ~10 | 0 |
| **Total** | **~280 lines** | **~4 lines** |

Comfortably under 400. LibV2 stays in scope.

---

## 5. Regression tests

`lib/tests/test_content_type_enforcement.py`:

1. **`test_flag_off_accepts_any_string`**
   - `monkeypatch.delenv("TRAINFORGE_ENFORCE_CONTENT_TYPE", raising=False)`
   - Assert `validate_chunk_type("foobar")` returns True.
   - Assert `validate_chunk_type("")` returns True (no minLength check at this layer).
   - Assert `assert_chunk_type("foobar")` does NOT raise.

2. **`test_flag_on_accepts_valid_chunk_type`**
   - `monkeypatch.setenv("TRAINFORGE_ENFORCE_CONTENT_TYPE", "true")`
   - For each ChunkType enum value (`assessment_item`, `overview`, `summary`, `exercise`, `explanation`, `example`): assert `validate_chunk_type(v)` returns True and `assert_chunk_type(v)` does not raise.

3. **`test_flag_on_rejects_invalid_chunk_type`**
   - Flag on; assert `validate_chunk_type("foobar")` returns False.
   - Assert `assert_chunk_type("foobar")` raises `ValueError` with "foobar" in the message.

4. **`test_legacy_instruction_pairs_still_validate_default`**
   - Load an existing fixture instruction_pair record (construct a dict mimicking the schema) with `content_type="random_free_string"`.
   - Validate against the default `instruction_pair.schema.json` (using `jsonschema.validate`) — passes.
   - Validate against the STRICT `instruction_pair.strict.schema.json` with same record — fails (invalid enum).
   - Substitute `content_type="explanation"` — passes both.

5. **`test_get_valid_chunk_types_returns_expected_set`**
   - `assert get_valid_chunk_types() == frozenset({"assessment_item", "overview", "summary", "exercise", "explanation", "example"})`.
   - Guards against upstream taxonomy drift (if Worker F extends the enum, this test updates intentionally).

6. **`test_get_valid_section_content_types_returns_expected_set`**
   - `assert get_valid_section_content_types() == frozenset({"definition", "example", "procedure", "comparison", "exercise", "overview", "summary", "explanation"})`.

Test-file imports `importlib` + reload only if needed. Prefer the `_is_enforcement_enabled()` call-per-call pattern so `monkeypatch.setenv` works without reloading.

### Additional (inline) integration test

In `Trainforge/tests/test_training_synthesis.py` — **do NOT modify**. Existing tests already cover flag-off path (default behavior unchanged). Flag-on path for synthesize_training is exercised transitively via the unit tests + schema validation tests. An integration regression test belongs to Wave 6 governance; YAGNI here.

### LibV2 test

Add a test to `LibV2/tools/libv2/tests/test_retriever_v4.py` OR a new adjacent file — sub-plan chooses **new file** to avoid collision with Worker J's test module:

`LibV2/tools/libv2/tests/test_chunk_filter_content_type_enforcement.py`:

1. **`test_flag_off_accepts_arbitrary_content_type_label`** — `ChunkFilter(content_type_label="bogus")` constructs silently.
2. **`test_flag_on_accepts_valid_chunk_type`** — `ChunkFilter(content_type_label="explanation")` constructs silently.
3. **`test_flag_on_rejects_invalid_content_type_label`** — `ChunkFilter(content_type_label="bogus")` raises `ValueError`.
4. **`test_flag_on_none_content_type_label_passes`** — `ChunkFilter(content_type_label=None)` constructs silently (no enforcement when unset).

---

## 6. Verification

```bash
python3 -m ci.integrity_check
source /home/mdmur/Projects/Ed4All/venv/bin/activate && pytest lib/tests/test_content_type_enforcement.py -x
pytest lib/tests/ Trainforge/tests/ Courseforge/scripts/tests/ MCP/tests/ LibV2/tools/libv2/tests/ -q
```

Expected: all green. CI integrity 8/8.

---

## 7. Non-goals (explicit)

- No change to `instruction_pair.schema.json` free-string shape (strict-variant pattern).
- No `SectionContentType` enforcement wiring (exposed helper only).
- No `CalloutContentType` or `ContentType` union enforcement.
- No migration of existing instruction_pair JSONL files.
- No Courseforge-side enforcement (Courseforge emission is the source of truth for SectionContentType; future wave can gate it separately if needed).
- No warn-log middle tier.
- No auto-correction / coercion (reject, don't normalize).

---

## 8. Commit + PR

Commit: `Worker T: REC-VOC-03 Phase 2 — opt-in content_type enforcement`

PR target: `dev-v0.2.0`.

KG-impact call-out in PR body: chunk→ContentType KG edge becomes controlled-vocabulary when flag enabled; instruction_factory silent `_default` fallback gets a detectable hard edge.
