# Worker O Sub-plan — REC-ID-02: Course-scoped concept IDs

**Branch:** `worker-o/wave4-concept-scoping`
**Base:** `dev-v0.2.0`
**Master plan:** `/home/mdmur/.claude/plans/we-have-several-branches-gentle-melody.md` → Wave 4.1 / Worker O

---

## Goal

Add opt-in course-scoped concept IDs behind `TRAINFORGE_SCOPE_CONCEPT_IDS=true`.

- Flag OFF (default): concept node IDs stay flat slugs (e.g. `"accessibility"`). `course_id` field absent. Byte-identical to today's output.
- Flag ON: concept node IDs become composite `"{course_id}:{slug}"` (e.g. `"wcag_201:accessibility"`); node dicts also carry a `course_id` field.

Schema gains optional `course_id` on nodes — legacy artifacts continue to validate.

---

## Current-state anchors (verified on this branch)

1. **`schemas/knowledge/concept_graph_semantic.schema.json` L22–33** — current node schema:
   ```json
   "nodes": {
     "type": "array",
     "items": {
       "type": "object",
       "required": ["id"],
       "additionalProperties": true,
       "properties": {
         "id": {"type": "string"},
         "label": {"type": "string"},
         "frequency": {"type": "integer", "minimum": 0}
       }
     }
   }
   ```
   No `course_id` field; `additionalProperties: true` already permits the flag-off path (absent). Need to declare the field explicitly so flag-on path validates with `additionalProperties: true` coverage, plus documentation.

2. **Concept-node emit sites.** Only ONE effective construction site exists today:
   - `Trainforge/process_course.py::_build_tag_graph` (L2151–2201) builds nodes verbatim: `{"id": tag, "label": tag.replace("-", " ").title(), "frequency": freq}`. The `tag` string IS the concept slug. This is the primary scoping point.
   - `Trainforge/rag/typed_edge_inference.py::_build_nodes` (L108–122) copies nodes from the already-built co-occurrence graph. Since nodes were already scoped upstream, the semantic graph inherits scoping for free.
   - Edge rule modules (`inference_rules/*.py`) reference nodes by ID from `node_ids = {n["id"] for n in concept_graph["nodes"]}`. Because they key off what's already in the graph, they naturally produce scoped source/target strings IF nodes are already scoped.
   - **Critical wrinkle:** Rules also internally slugify text (e.g. `is_a_from_key_terms._slugify` on "term" and parent phrases). Those slugs must be scoped before lookup against `node_ids` — else flag-on rules silently stop matching anything.
   - `prerequisite_from_lo_order._first_positions_by_concept` iterates `chunk.get("concept_tags", [])` — also an unscoped value.

3. **How course_id reaches emit sites.**
   - `Trainforge/process_course.py:1248` — every chunk carries `source.course_id = self.course_code`. Worker J's Wave 2 work confirmed this is reliably populated.
   - `CourseProcessor.__init__` stores `self.course_code`. `_build_tag_graph` is a CourseProcessor method — direct `self.course_code` access available with zero plumbing.
   - For rule modules: need to thread via chunks' `source.course_id`. Chunks already carry it; rules can read per-chunk. Alternative: have build_semantic_graph accept an explicit `course_id` parameter derived upstream.

4. **Concept-tag storage on chunks.** `concept_tags` in chunks (e.g. `["wcag", "accessibility-attribute"]`) are flat slugs. When flag is on, the scoping happens at GRAPH construction time; chunks themselves remain unscoped. This is deliberate — chunks are the raw substrate and concept_tags are slugs, not graph-node-ids. The scoping is applied when tags become nodes.

5. **Edge references in evidence.** `is_a_from_key_terms` writes `evidence.chunk_id`. Chunks use their own `id` namespace (`wcag_201_chunk_00001`), which already encodes course_code. No schema-level scoping needed there.

---

## Design

### When flag is OFF (backward-compat path)
- `_build_tag_graph` produces nodes as today: `id = tag` (flat slug).
- Rules look up `node_ids = {flat_slug, ...}`. They produce edges referencing flat slugs.
- Node dicts do NOT include `course_id` key. This matches every pre-Wave-4 artifact in LibV2.

### When flag is ON (new behavior)
- `_build_tag_graph` produces nodes as `{"id": f"{course_id}:{slug}", "label": ..., "frequency": ..., "course_id": course_id}`.
- Co-occurrence edges now reference scoped IDs.
- Rules must match scoped IDs. Two options considered:
  - **Option A (chosen):** Helper `_make_concept_id(slug, course_id)` applied at every rule-local lookup site. Rules walk chunks' `source.course_id` per-chunk — since rules operate over chunks from one course at a time via the current pipeline, this is stable.
  - Option B: Re-slug everything at rule entry. More invasive.

**Chosen path:** `_make_concept_id` helper, with both process_course.py and rule modules using it. The helper is module-level and reads the env var on import, matching Worker N's pattern for `USE_CONTENT_HASH_IDS`.

### Cross-course merge is explicit only
With flag on, two courses with `concept_tags: ["accessibility"]` produce two separate nodes: `"course_a:accessibility"` and `"course_b:accessibility"`. This is the defect fix — previously the two silently merged into one node.

Wave 5 will add `aliases[]` / equivalence edges for explicit cross-course merge. Out of scope here.

---

## Code changes

### 1. `schemas/knowledge/concept_graph_semantic.schema.json`
Add `course_id` to node `properties` block:
```json
"course_id": {
  "type": "string",
  "description": "When TRAINFORGE_SCOPE_CONCEPT_IDS=true, concept ID format becomes '{course_id}:{slug}' and this field carries the scoping course_id. Default behavior: concept IDs are flat slugs, course_id absent."
}
```
No `required` change — stays optional. No edge-schema change (edges reference node IDs by value; scoping is transparent to edge shape).

### 2. `Trainforge/rag/typed_edge_inference.py`
Module-level:
```python
import os

SCOPE_CONCEPT_IDS = os.getenv("TRAINFORGE_SCOPE_CONCEPT_IDS", "").lower() == "true"

def _make_concept_id(slug: str, course_id: str | None) -> str:
    """Return the scoped concept ID when the flag is on, else the flat slug.

    When SCOPE_CONCEPT_IDS is True and course_id is provided, returns
    ``f"{course_id}:{slug}"``. Otherwise returns ``slug`` unchanged.
    Exposed as a module-level helper so rule modules can import it and
    produce node IDs that match the graph's scoped namespace.
    """
    if SCOPE_CONCEPT_IDS and course_id:
        return f"{course_id}:{slug}"
    return slug
```
Export from module so rule modules can import.

### 3. `Trainforge/process_course.py::_build_tag_graph`
Thread `course_id` through. Since the graph is built per-course, populate nodes with scoped IDs when flag on:
- Read `course_id = self.course_code`.
- When constructing node dicts, use `_make_concept_id(tag, course_id)` for `id`.
- When flag on, also include `"course_id": course_id` on the node dict.
- For co-occurrence edges: map both source/target through `_make_concept_id` so edge endpoints match scoped node IDs.

### 4. Rule modules — lookup scoping
Three rule modules consult `node_ids`; two also slugify from text:
- **`is_a_from_key_terms.py`**: `_slugify(term)` → child candidate. `_candidate_parent_ids(phrase, node_ids)` → parent candidates.
  - When scope flag on: map each produced slug through `_make_concept_id(slug, course_id)` before comparing against `node_ids`.
  - `course_id` is read per-chunk via `chunk["source"]["course_id"]` (chunks always carry this).
- **`prerequisite_from_lo_order.py`**: iterates `chunk.get("concept_tags", [])` and compares to `node_ids`. Same per-chunk course_id scoping.
- **`related_from_cooccurrence.py`**: consumes concept_graph.edges which already use scoped IDs (from #3). No changes needed — the edges come in pre-scoped.

### 5. Node label handling
When flag on, node labels keep the human-readable slug form (without course_id prefix): `label = slug.replace("-", " ").title()`. Only `id` is composite. Matches downstream display expectations.

---

## Test design (`Trainforge/tests/test_concept_scoping.py`)

Five tests. Each test sets/resets env var via `monkeypatch.setenv` / `monkeypatch.delenv` and reimports the module (or uses explicit helper calls) to ensure the flag is read fresh.

### Test 1 — `test_flag_off_flat_slug_ids`
- Default env (no var set).
- Build concept graph from chunks tagged with `["accessibility"]` for course `wcag_201`.
- Assert: node `id == "accessibility"`; `"course_id"` NOT in node dict.

### Test 2 — `test_flag_on_composite_ids`
- `TRAINFORGE_SCOPE_CONCEPT_IDS=true`.
- Same inputs as Test 1 but flag on.
- Assert: node `id == "wcag_201:accessibility"`; `node["course_id"] == "wcag_201"`.

### Test 3 — `test_schema_accepts_both_formats`
- Validate a minimal flag-off artifact against `concept_graph_semantic.schema.json`: passes.
- Validate a minimal flag-on artifact (nodes with `course_id` + composite IDs) against same schema: passes.
- Uses `jsonschema.validate` (same pattern as `test_typed_edge_inference.py:258`).

### Test 4 — `test_cross_course_no_silent_merge_when_scoped`
- Flag on.
- Ingest concept graphs for two courses: `course_a` with `concept_tags=["accessibility"]` and `course_b` with `concept_tags=["accessibility"]`.
- Merge node lists (simulating multi-course LibV2 aggregation).
- Assert: two distinct nodes present — `"course_a:accessibility"` and `"course_b:accessibility"`.

### Test 5 — `test_course_id_field_populated_when_flag_on`
- Flag on: node dicts carry `course_id` key equal to the scoping course.
- Flag off: node dicts do NOT contain `course_id` key (via `"course_id" not in node`).

### Env-var handling strategy
Because `SCOPE_CONCEPT_IDS` is captured at module import time, tests use `importlib.reload(typed_edge_inference)` inside each test or use `monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", True/False)` to override directly. The `_make_concept_id` helper references the module global, so direct attribute override is clean and avoids reload fragility — this is the pattern chosen.

---

## Verification commands

```bash
python3 -m ci.integrity_check
source venv/bin/activate && pytest Trainforge/tests/test_concept_scoping.py -x
pytest lib/tests/ Trainforge/tests/ Courseforge/scripts/tests/ MCP/tests/ -q
```

Baseline: Wave 3 at 688 tests passing. Target: 693+ (688 + 5 new).

---

## Risk / blast-radius notes

- **Cross-worker coordination.** Worker P also modifies `concept_graph_semantic.schema.json` (adds `run_id` + `created_at`). Both workers touch the node `properties` block. Merge order (per master plan): P first → O rebases if needed. Addition sites are distinct keys; conflict resolution is trivial (accept both).
- **Flag-off path stays byte-identical.** Default artifacts in LibV2 (flag off) continue to have flat slug IDs and no `course_id` field. Existing `test_typed_edge_inference.py` keeps passing without modification.
- **Re-slugifying inside rules.** The `_slugify` in `is_a_from_key_terms.py` converts key-term text to a graph-node-id — when flag on this output must be scoped before lookup. If we forget this, flag-on produces no is-a edges (silent degradation). Test 2 catches this.
- **`related_from_cooccurrence.py` is passive.** It consumes pre-scoped edges from the base concept graph. Zero rule-side changes. The scoping is applied once at the `_build_tag_graph` site.
- **Out of scope.** `chunk_v4.schema.json` (Worker N + P); `process_course.py` chunk-ID generation (Worker N); Courseforge emit (this is Trainforge consume + schema); LibV2 data migration (opt-in only).

---

## Commit message

```
Worker O: REC-ID-02 — opt-in course-scoped concept IDs + schema course_id field
```

## PR

Base: `dev-v0.2.0`. Title: `Worker O: Wave 4 course-scoped concept IDs (opt-in)`. Body per master plan.
