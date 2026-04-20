# Worker N Sub-Plan — REC-ID-01: Content-hash chunk IDs (opt-in)

**Branch:** `worker-n/wave4-content-hash-ids`
**Base:** `dev-v0.2.0` @ `e3dde95` (Wave 3 merged: Workers J/K/L/M)
**Master plan:** `/home/mdmur/.claude/plans/we-have-several-branches-gentle-melody.md` — § Wave 4.1, Worker N

## Goal

Make chunk IDs content-addressable (opt-in via env var) so that re-chunking the same source produces the same IDs — keeping edge-evidence references that quote chunk IDs stable across re-runs. Legacy position-based IDs remain the default to preserve backward compatibility with already-ingested LibV2 courses.

---

## 1. Current-state anchors (re-verified 2026-04-19)

The master plan's "L1003, 1027" anchor has drifted. Current sites in `Trainforge/process_course.py`:

| Line | Call-site | Context |
|------|-----------|---------|
| 1193 | `chunk_id=f"{prefix}{start_id:05d}",` | Single-chunk branch (fits in one chunk) inside `_chunk_item_sections` / similar |
| 1209 | `prev_id = follows_chunk_id if i == 0 else f"{prefix}{start_id + i - 1:05d}"` | `follows_chunk` back-reference when splitting into sub-chunks |
| 1217 | `chunk_id=f"{prefix}{start_id + i:05d}",` | Multi-chunk branch (sub_texts loop) |

**Note on L1209 (`prev_id`):** this computes the *previous* chunk's ID to set `follows_chunk`. Under content-hash mode, re-computing that ID by formula no longer works — the previous chunk's hash is only known once it's been generated. Fix: keep a running reference to the last emitted chunk's ID in the loop, rather than re-computing from position. This is a small refactor inside the split branch.

**Existing schema anchor (verified):**
`schemas/knowledge/chunk_v4.schema.json:27` currently:
```
"pattern": "^[a-z][a-z0-9_]*_chunk_\\d{5}$"
```

---

## 2. Canonicalization input — decision

The master plan proposes `text + source_path + schema_version` as hash input. Availability check at the ID-generation site:

- **text**: available (`text` or `sub_text` local var).
- **source_path equivalent**: `item["item_path"]` carries the IMSCC-relative path to the source HTML file (see `process_course.py:936` and `:1266`). This is stable across re-runs as long as the IMSCC hasn't been restructured. Confirmed present for all non-quiz and quiz items in `_flatten_items`. Optional in the Source schema (line 171–174 of chunk_v4.schema.json) but in practice always populated by the parser for webcontent items.
- **schema_version**: the literal string `"v4"` matches the current chunk schema version.

**Chosen canonical input (per master plan):**
```
text + "|" + source_path + "|" + "v4"
```

**Fallback:** if `item.get("item_path")` is ever empty (defensive), use `item["module_id"] + "/" + item["item_id"]` as the stable locator. The helper signature accepts a pre-resolved `source_locator` string, so the caller picks which composition to pass.

**Rationale for `text + source_locator + schema_version`:**
- `text` alone would collide across two chunks with identical prose (rare but possible — boilerplate section).
- Adding `source_locator` disambiguates per-file.
- Adding `"v4"` prevents cross-schema-version collisions if we ever re-hash the same content under a different schema revision.

**Multi-chunk disambiguation:** in the split-text branch (sub_texts loop), each `sub_text` is a different substring of the full text, so the hashes differ naturally. No positional salt needed.

**What the hash does NOT include (intentional):**
- `start_id` / position — that's the whole point of content addressing.
- `course_code` — already encoded in the `prefix` which stays concatenated outside the hash.
- `section_heading` — not strictly part of the content contract; including it would make hashes churn when headings are stylistically edited. Section heading changes *should* not invalidate IDs as long as text is identical.

---

## 3. Exact schema pattern diff

`schemas/knowledge/chunk_v4.schema.json` line 27:

```diff
-      "pattern": "^[a-z][a-z0-9_]*_chunk_\\d{5}$",
+      "pattern": "^[a-z][a-z0-9_]*_chunk_(\\d{5}|[0-9a-f]{16})$",
```

And line 28 description updated:
```diff
-      "description": "Globally unique chunk identifier: '<course_slug>_chunk_<5-digit sequence>'. Example: wcag_201_chunk_00001."
+      "description": "Globally unique chunk identifier. Default form: '<course_slug>_chunk_<5-digit position>'. When TRAINFORGE_CONTENT_HASH_IDS=true, form is '<course_slug>_chunk_<16-hex-char sha256>'. Example: wcag_201_chunk_00001 or wcag_201_chunk_a3f2b9c8d1e4f567."
```

Both legacy (5-digit) and new (16-hex) IDs validate under the relaxed pattern, so existing LibV2 chunks continue to pass.

---

## 4. Implementation

### 4.1 Module-level helper (top of `process_course.py`)

Add after the existing `import` block (os + hashlib not currently imported at module level — the file has `import re, json, sys` etc.). Add:

```python
import hashlib
import os

USE_CONTENT_HASH_IDS = os.getenv("TRAINFORGE_CONTENT_HASH_IDS", "").lower() == "true"


def _generate_chunk_id(prefix: str, start_id: int, text: str, source_locator: str) -> str:
    """Generate a chunk ID.

    Default (legacy): position-based `{prefix}{start_id:05d}`.
    When `TRAINFORGE_CONTENT_HASH_IDS=true`: content-addressed
    `{prefix}{sha256(text|source_locator|v4)[:16]}`, stable across re-chunks.
    """
    if USE_CONTENT_HASH_IDS:
        payload = f"{text}|{source_locator}|v4"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}{digest}"
    return f"{prefix}{start_id:05d}"
```

### 4.2 Call-site migrations

**Site A — L1193 (single-chunk branch).** The full `text` and `item["item_path"]` are both in scope.

```diff
+            locator = item.get("item_path") or f"{item['module_id']}/{item['item_id']}"
             chunks.append(self._create_chunk(
-                chunk_id=f"{prefix}{start_id:05d}",
+                chunk_id=_generate_chunk_id(prefix, start_id, text, locator),
                 text=text, html=html, item=item,
                 ...
             ))
```

**Site B — L1217 (multi-chunk branch).** Each `sub_text` is the content for its own chunk.

```diff
+            locator = item.get("item_path") or f"{item['module_id']}/{item['item_id']}"
+            last_chunk_id = follows_chunk_id
             for i, sub_text in enumerate(sub_texts):
                 part_heading = ...
-                prev_id = follows_chunk_id if i == 0 else f"{prefix}{start_id + i - 1:05d}"
+                prev_id = last_chunk_id
+                this_chunk_id = _generate_chunk_id(prefix, start_id + i, sub_text, locator)
                 char_span = ...
                 ...
                 chunks.append(self._create_chunk(
-                    chunk_id=f"{prefix}{start_id + i:05d}",
+                    chunk_id=this_chunk_id,
                     text=sub_text, html="" if i > 0 else html, item=item,
                     ...
                     follows_chunk_id=prev_id,
                     ...
                 ))
+                last_chunk_id = this_chunk_id
```

Under legacy mode, `_generate_chunk_id(prefix, start_id + i, ...)` returns `f"{prefix}{start_id + i:05d}"` — identical to the legacy formula, so `follows_chunk` continues to reference the prior chunk correctly.

Under hash mode, `last_chunk_id` tracks the actual hash of the just-emitted chunk, so `follows_chunk` always references the correct ID (can no longer be re-derived positionally).

---

## 5. Regression test design

Location: `Trainforge/tests/test_content_hash_ids.py` (new file).

### 5.1 Approach for env-var reads

`USE_CONTENT_HASH_IDS` is read once at module import. Straight `monkeypatch.setenv` AFTER the module has been imported won't flip the flag. Two options considered:

1. **Re-import via `importlib.reload`** — brittle if other tests hold references.
2. **Refactor to read env per-call inside `_generate_chunk_id`** — cleaner, slightly higher overhead (1 `os.getenv` per chunk generated).

**Decision: refactor to read env per-call.** This is the approach that matches existing patterns in the codebase (e.g., Worker M's `TRAINFORGE_PRESERVE_LO_CASE` reads at the call site). Also avoids import-order landmines for downstream test modules.

Revised helper:

```python
def _generate_chunk_id(prefix: str, start_id: int, text: str, source_locator: str) -> str:
    use_hash = os.getenv("TRAINFORGE_CONTENT_HASH_IDS", "").lower() == "true"
    if use_hash:
        payload = f"{text}|{source_locator}|v4"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}{digest}"
    return f"{prefix}{start_id:05d}"
```

Keep the module-level `USE_CONTENT_HASH_IDS` constant for any documentation/log tools that want to introspect it, but the helper no longer uses it.

### 5.2 Tests (five)

1. **`test_flag_off_uses_position`**
   - No env override.
   - `_generate_chunk_id("wcag_201_chunk_", 42, "some text", "path/a.html")` → `"wcag_201_chunk_00042"`.
   - Assert match `^wcag_201_chunk_\d{5}$`.

2. **`test_flag_on_uses_content_hash`**
   - `monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "true")`.
   - Same call → assert match `^wcag_201_chunk_[0-9a-f]{16}$`.

3. **`test_content_hash_stable_across_runs`**
   - Flag on; call twice with identical inputs; assert equal IDs.

4. **`test_content_hash_differs_on_text_change`**
   - Flag on; call with `"some text"` then with `"some text!"`; assert different IDs.

5. **`test_schema_accepts_both_formats`**
   - Load `schemas/knowledge/chunk_v4.schema.json` via `jsonschema`.
   - Construct a minimal-but-valid chunk dict with `id = "wcag_201_chunk_00001"` (legacy form); validate passes.
   - Construct same dict with `id = "wcag_201_chunk_a3f2b9c8d1e4f567"` (hash form); validate passes.
   - Construct same dict with `id = "wcag_201_chunk_invalid"`; validate fails (negative control, optional but good hygiene).

The minimal chunk dict for the schema test needs all required fields per the schema: `id`, `schema_version`, `chunk_type`, `text`, `html`, `follows_chunk`, `source` (with `course_id`, `module_id`, `lesson_id`), `concept_tags`, `learning_outcome_refs`, `difficulty`, `tokens_estimate`, `word_count`, `bloom_level`. Taxonomy `$ref` fields (`chunk_type`, `bloom_level`) need the resolver pointed at the `schemas/` directory OR use values that obviously conform and rely on `Draft202012Validator` not fetching refs (use `jsonschema.validators.RefResolver` with local file loader, same pattern Worker I tests use).

If `jsonschema` ref resolution is awkward, fall back to validating the pattern directly via `re.match` against the updated regex — the core claim of the test is "the pattern accepts both forms," and that's exactly what the regex check verifies without needing taxonomy resolution.

**Chosen fallback approach for test 5**: use `re.match` on the pattern string loaded from the schema file. Simpler, no resolver fuss, still proves the schema accepts both forms.

---

## 6. Verification plan

```bash
# 1. CI integrity
python3 -m ci.integrity_check

# 2. New regression tests
pytest Trainforge/tests/test_content_hash_ids.py -x

# 3. Full relevant suites — no regressions
pytest lib/tests/ Trainforge/tests/ Courseforge/scripts/tests/ MCP/tests/ -q

# 4. Manual flag-on spot-check (optional but recommended)
TRAINFORGE_CONTENT_HASH_IDS=true python3 -c "
import os
os.environ['TRAINFORGE_CONTENT_HASH_IDS']='true'
from Trainforge.process_course import _generate_chunk_id
print(_generate_chunk_id('wcag_201_chunk_', 0, 'hello', 'a.html'))
"
```

Expected test baseline: Wave 3 = 688 passing. Wave 4.1 Worker N adds 5 tests → expect ≥693.

---

## 7. Constraints observed

- **No touch** `schemas/knowledge/concept_graph_semantic.schema.json` (Worker O + P).
- **No touch** `Trainforge/generators/preference_factory.py` (Worker R).
- **No migration** of existing LibV2 chunks — opt-in only.
- **Default unchanged** — env var defaults to off, existing behavior preserved.
- **Target branch** `dev-v0.2.0`; main untouched.

---

## 8. Files touched summary

**Modified (2):**
- `Trainforge/process_course.py` — add `_generate_chunk_id` helper + 3 call-site migrations (L1193, L1209 refactor, L1217).
- `schemas/knowledge/chunk_v4.schema.json:27,28` — pattern relaxed + description updated.

**New (2):**
- `Trainforge/tests/test_content_hash_ids.py` — 5 regression tests.
- `plans/kg-quality-review-2026-04/worker-n-subplan.md` — this file.

---

## 9. KG impact (carried to PR body)

Edge-rule evidence that references chunk IDs by value (e.g., `Trainforge/rag/inference_rules/is_a_from_key_terms.py:163`) survives across re-chunks when the flag is on. Makes the graph rerun-stable for the subset of runs using the flag. Sets the stage for LNK-02 (misconception entity) in Wave 4.2, whose IDs follow the same content-hash pattern.
