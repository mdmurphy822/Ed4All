# Worker R Sub-Plan — REC-LNK-02: First-class Misconception entity

**Branch:** `worker-r/wave4-misconception-entity`
**Base:** `dev-v0.2.0` @ `3edf730` (Wave 4.1 merged: Workers N, O, P)
**Master plan:** `/home/mdmur/.claude/plans/we-have-several-branches-gentle-melody.md` — § Wave 4.2, Worker R

## Goal

Promote misconceptions to a first-class Knowledge-Graph entity with a **stable content-hash ID**. Replace the current unstable position-based ID format (`{chunk_id}_mc_{index:02d}_{hash}`) at `Trainforge/generators/preference_factory.py:140–143` — that format drifts whenever chunk IDs drift (which is the very churn Worker N is starting to contain) and it embeds a positional index that's meaningless once misconceptions become citable objects on their own. The new ID is `sha256(misconception_text + "|" + correction_text)[:16]` prefixed with `mc_`, giving stable addressability independent of the chunk that surfaced the misconception.

The schema adds **optional** `concept_id` and `lo_id` slots so a future wave (REC-LNK-04) can wire typed links from a misconception to the concept it targets and the LO it threatens. Those fields are **not** populated in this PR — they're forward-compatibility placeholders only.

---

## 1. Current-state anchors (verified 2026-04-19)

### 1a. The ID helper in `preference_factory.py`

```python
# Trainforge/generators/preference_factory.py, lines 140–143
def _misconception_id(chunk_id: str, index: int, text: str) -> str:
    """Stable id for a misconception: chunk_id + index + short hash of text."""
    short = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{chunk_id}_mc_{index:02d}_{short}"
```

### 1b. Call-sites that need updating

`rg "_misconception_id|misconception_id" Trainforge/` returns:

| File | Line | Context |
|------|------|---------|
| `Trainforge/generators/preference_factory.py` | 47 | `@dataclass` field (result holder — `Optional[str]`, shape unchanged) |
| `Trainforge/generators/preference_factory.py` | 140 | Helper definition — **rewrite** |
| `Trainforge/generators/preference_factory.py` | 337 | `mc_id = _misconception_id(chunk_id, idx, str(mc.get("misconception", "")))` — **update call** |
| `Trainforge/generators/preference_factory.py` | 389 | Result construction (passes `mc_id` through — unchanged) |
| `Trainforge/generators/preference_factory.py` | 396 | `"misconception_id": mc_id` on the emitted pair dict — unchanged |
| `Trainforge/generators/preference_factory.py` | 411 | Result construction — unchanged |
| `Trainforge/synthesize_training.py` | 292 | Logging only — unchanged |

Only the helper signature + single in-module call-site (L337) need structural change. The rest pass the ID around by string, so new-format IDs flow through without further edits.

### 1c. What does today's format guarantee and what's unstable?

- **Today:** ID = `{chunk_id}_mc_{index:02d}_{sha256(text)[:8]}`
  - Drifts whenever `chunk_id` drifts (e.g., re-chunking under legacy position-based IDs).
  - Drifts whenever the ordering of misconceptions within a chunk changes (since `index` is a list-position).
  - Incorporates 8 hex chars of `misconception` text ONLY — ignores `correction`, so two misconceptions with the same "wrong belief" but different corrections collide.
- **New:** ID = `mc_{sha256(misconception_text + "|" + correction_text)[:16]}`
  - Stable across chunk IDs (does not reference chunk).
  - Stable across list position.
  - Distinguishes misconceptions that share a wrong belief but have different corrections.
  - 16 hex chars (64-bit address space) vs. 8 (32-bit) — substantially lower accidental-collision risk.

### 1d. Downstream consumers of the old ID

None. Review notes (§ "REC-LNK-02" in `plans/kg-quality-review-2026-04/review.md`) explicitly flag the old format as unstable; nothing stable references it. The ID flows into:
- `preference_pair.misconception_id` (string field; new form fits)
- `synthesize_training.py:292` (log line; string interpolation; new form fits)

No LibV2 artifact schema, no graph edge, no cross-run join depends on the legacy form. Safe to replace.

---

## 2. Canonicalization — decision

**Chosen canonical hash input:** `misconception_text.strip() + "|" + correction_text.strip()`

Rationale:
1. **Whitespace normalization (outer `.strip()` only):** Upstream Courseforge JSON-LD emits misconception prose with trailing whitespace / newline variation; stripping outer whitespace prevents cosmetic churn from invalidating IDs. Inner whitespace (e.g., double spaces, tabs) is preserved — normalizing it would require a judgment call on what's "cosmetic" vs. "semantic" and could mask genuine text edits. Outer-only matches the behavior Worker N chose for chunk text.
2. **`"|"` separator:** Prevents boundary ambiguity — otherwise `misconception="Ab"` + `correction="cd"` collides with `misconception="Abcd"` + `correction=""`.
3. **Both fields mandatory in the hash:** A misconception without a correction isn't a complete pedagogical object. Empty correction is tolerated (`.strip()` just yields `""`), but the hash reflects it — two misconceptions with the same wrong belief and different corrections get different IDs.
4. **No version tag in the hash:** Unlike Worker N's chunk IDs (which include `"v4"` to guard against cross-schema-version collisions), misconceptions have a single current form. If/when the schema adds required fields (say, in Wave 5 when `concept_id` becomes required), we'd introduce versioning then. YAGNI now.

**Hash algorithm:** `sha256`, first 16 hex chars → 64-bit namespace. Follows Worker N's precedent.

**ID prefix:** `mc_` — two-character tag to make the ID visually distinguishable from chunk IDs (which look like `wcag_201_chunk_a3f2b9c8d1e4f567`). Future edge-type enums can pattern-match on `mc_` for quick type discrimination.

**Full pattern:** `^mc_[0-9a-f]{16}$`

---

## 3. Schema outline

File: `schemas/knowledge/misconception.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://ed4all.dev/ns/knowledge/v1/misconception.schema.json",
  "title": "Misconception",
  "description": "First-class Misconception entity with content-hash ID (REC-LNK-02). Replaces the earlier unstable chunk-position-based misconception_id format.",
  "type": "object",
  "required": ["id", "misconception", "correction"],
  "additionalProperties": false,
  "properties": {
    "id": {
      "type": "string",
      "pattern": "^mc_[0-9a-f]{16}$",
      "description": "Content-hash misconception ID: sha256(misconception_text + '|' + correction_text)[:16]. Stable across runs when text is unchanged."
    },
    "misconception": {
      "type": "string",
      "minLength": 1,
      "description": "The incorrect belief or reasoning."
    },
    "correction": {
      "type": "string",
      "minLength": 1,
      "description": "The correct explanation."
    },
    "concept_id": {
      "type": "string",
      "description": "Optional typed link to the concept this misconception targets. Populated when upstream emit provides it (future wave may wire from JSON-LD)."
    },
    "lo_id": {
      "type": "string",
      "description": "Optional typed link to the LO this misconception threatens."
    }
  }
}
```

**Why `additionalProperties: false`:** Misconceptions are small, content-hash-addressable objects. Their schema should be strict so that future additions (e.g., Bloom's level, bibliographic citation, `chunk_id` back-ref) are a deliberate schema change, not silently admitted. Contrast with `chunk_v4.schema.json` which keeps `additionalProperties: true` to preserve diagnostic side-channel fields (`_metadata_trace`). No such diagnostic channel exists for misconceptions.

**Why `minLength: 1` on `misconception` / `correction`:** An empty-string "misconception" or "correction" is never pedagogically meaningful. The factory already filters these out (`preference_factory.py:306–309` normalizes misconceptions by requiring non-empty `misconception` text). Schema mirrors the code.

---

## 4. Implementation

### 4.1 New helper in `preference_factory.py`

Replace the L140–143 helper. Signature changes (drops `chunk_id` and `index`; adds `correction_text`):

```python
def _misconception_id(misconception_text: str, correction_text: str) -> str:
    """Content-hash misconception ID (REC-LNK-02).

    Stable across runs and across chunk re-chunking. The hash input is
    ``misconception_text.strip() + "|" + correction_text.strip()`` — outer
    whitespace is normalized but inner whitespace is preserved, so cosmetic
    edits don't churn IDs but real text edits do.
    """
    mt = (misconception_text or "").strip()
    ct = (correction_text or "").strip()
    content = f"{mt}|{ct}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"mc_{digest}"
```

### 4.2 Call-site update (L337)

Current:
```python
mc_id = _misconception_id(chunk_id, idx, str(mc.get("misconception", "")))
```

New:
```python
mc_id = _misconception_id(
    str(mc.get("misconception", "")),
    str(mc.get("correction", "")),
)
```

### 4.3 No other code changes

- `PreferenceSynthesisResult.misconception_id` stays `Optional[str]` — format is opaque to the dataclass.
- `pair["misconception_id"] = mc_id` at L396 stays — string value, new format fits.
- `synthesize_training.py:292` log line stays — prints whatever string it gets.
- `preference_pair.schema.json` — **check**: does it constrain `misconception_id` format? If so, update it; otherwise leave alone.

### 4.4 Schema back-compat check

`schemas/knowledge/preference_pair.schema.json` is the schema for the emitted preference-pair record. Its `misconception_id` field (if present) would need to accept the new format. Quick read: `grep "misconception_id" schemas/knowledge/preference_pair.schema.json` — if the field is there with a pattern, relax it; if it's `type: string` only, no change needed.

*(Pre-verified: field is `type: string` with no pattern → no change needed. If that turns out wrong at implementation time, update the pattern to `^mc_[0-9a-f]{16}$` as part of this PR.)*

---

## 5. Test design

Two new test files, following Worker N's / Worker P's test-organization pattern:

### 5.1 `lib/tests/test_misconception_schema.py` — schema conformance (5 tests)

1. **`test_schema_valid_json_schema`** — `Draft202012Validator.check_schema(schema)` runs clean (i.e., the schema itself is a valid draft-2020-12 schema).
2. **`test_valid_misconception_validates`** — minimal valid dict `{id, misconception, correction}` passes validation. Also test with optional `concept_id`, `lo_id` present.
3. **`test_missing_required_fails`** — drop `misconception` or `correction` → validation error. Loop over required fields.
4. **`test_invalid_id_pattern_fails`** — non-conforming IDs fail: `"mc_xyz"` (too short, non-hex), `"misconception_abc"` (wrong prefix), `"mc_0123456789abcdef0"` (17 chars), `"mc_0123456789ABCDEF"` (uppercase), `"MC_0123456789abcdef"` (uppercase prefix).
5. **`test_additional_properties_rejected`** — dict with extra key `{..., "extra_field": "x"}` → validation error (strict schema).

### 5.2 `Trainforge/tests/test_misconception_id.py` — helper behavior (5 tests)

1. **`test_misconception_id_format_matches_schema`** — the helper's output always matches `^mc_[0-9a-f]{16}$`.
2. **`test_misconception_id_stable_across_runs`** — `_misconception_id("wrong belief", "correct answer")` called twice → equal IDs.
3. **`test_misconception_id_differs_on_text_change`** — change `misconception` text (one char) → different ID.
4. **`test_misconception_id_differs_on_correction_change`** — change `correction` text (one char) → different ID.
5. **`test_misconception_id_whitespace_normalized`** — `_misconception_id(" wrong ", "right ")` == `_misconception_id("wrong", "right")` — outer whitespace stripped, but inner whitespace still matters (sanity: `_misconception_id("wr ong", "right")` != `_misconception_id("wrong", "right")`).

### 5.3 Optional integration sanity (fold into `test_misconception_id.py` if trivial)

Not adding as a separate test — the existing `Trainforge/tests/test_training_synthesis.py` already exercises `synthesize_preference_pair` end-to-end. Once the helper is updated, that test still passes iff the emitted `misconception_id` in the pair dict is just a string the downstream plumbing doesn't re-parse. If it does regress, the existing suite flags it and we won't need a new integration test.

---

## 6. Verification plan

```bash
# 1. CI integrity gate
python3 -m ci.integrity_check

# 2. New tests pass in isolation
source venv/bin/activate
pytest lib/tests/test_misconception_schema.py Trainforge/tests/test_misconception_id.py -x

# 3. Full suite — no regressions
pytest lib/tests/ Trainforge/tests/ Courseforge/scripts/tests/ MCP/tests/ -q
```

Baseline (post-Wave-4.1): tests pass count is what the dev-v0.2.0 head produces. Expect +≥10 new tests from this PR.

---

## 7. Constraints observed

- **No touch** `lib/ontology/slugs.py` (Worker Q's Wave 4.2 scope).
- **No touch** `schemas/knowledge/chunk_v4.schema.json` or `concept_graph_semantic.schema.json` (Wave 4.1 scope).
- **No wiring** of `concept_id` / `lo_id` at emit side — schema-only placeholders for a future wave.
- **No env var gating** — the prior format was explicitly flagged unstable; replacing is safe without a flag.
- **Target branch** `dev-v0.2.0`; main untouched.

---

## 8. Files touched summary

**New (3):**
- `plans/kg-quality-review-2026-04/worker-r-subplan.md` — this file.
- `schemas/knowledge/misconception.schema.json` — new entity schema.
- `lib/tests/test_misconception_schema.py` — schema conformance tests (5).
- `Trainforge/tests/test_misconception_id.py` — helper regression tests (5).

**Modified (1):**
- `Trainforge/generators/preference_factory.py` — rewrite `_misconception_id` helper + update single call-site at L337.

---

## 9. KG impact (carried to PR body)

Misconceptions gain **stable identity**. Queries like "all misconceptions targeting concept Z" become stable across re-emits — the ID no longer churns when chunks churn. The `concept_id` / `lo_id` optional fields land as placeholders so REC-LNK-04 (pedagogical edge-type expansion — Wave 5) can wire `misconception-of` edges without requiring another schema change. Sets up Worker Q's slug unification to interoperate with misconception-to-concept links later.
