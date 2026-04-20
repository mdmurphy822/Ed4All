# Worker S Sub-plan — REC-LNK-01: `occurrences[]` back-reference on concept nodes

**Branch:** `worker-s/wave5-concept-occurrences`
**Base:** `dev-v0.2.0`
**Master plan:** `/home/mdmur/.claude/plans/we-have-several-branches-gentle-melody.md` → Wave 5.1 / Worker S

---

## Goal

Add an optional `occurrences[]` back-reference on concept nodes — a list of chunk IDs that reference each concept. Populated from the existing chunk→tag inverted index at graph-build time. Schema additive; always-on additive behavior (no env var per master plan design decision #1).

**KG-impact.** "Which chunks define concept X?" becomes O(1) graph traversal instead of O(N) chunk scan. Needed by Worker U's Wave 5.2 `defined-by` rule (concept→first-chunk) and generally makes concept navigation graph-first.

**Stability note.** Back-references are accurate at graph-build time. Under position-based chunk IDs (default), re-chunking invalidates entries. Under `TRAINFORGE_CONTENT_HASH_IDS=true` (Worker N's Wave 4 flag), back-references survive re-chunks. Schema description documents this caveat.

---

## Current-state anchors (verified on this branch)

1. **`schemas/knowledge/concept_graph_semantic.schema.json` L22–46** — current node schema carries optional `id` (required), `label`, `frequency`, `run_id`, `created_at`, `course_id`. `additionalProperties: true` already permits unknown keys — we add `occurrences` as a documented optional property alongside the Wave 4 additions.

2. **`Trainforge/process_course.py::_build_tag_graph`** (L2215–2291) — single effective concept-node emit site for `concept_graph.json` (kind=`concept`). Iterates chunks, builds per-tag frequency + pairwise co-occurrence, emits nodes with `{id, label, frequency}` (+ optional `course_id` when Worker O's flag is on).

3. **`Trainforge/rag/typed_edge_inference.py::_build_nodes`** (L156–181) — copies the base concept-graph nodes verbatim into the semantic-graph artifact. Currently preserves `id`, `label`, `frequency`; stamps `run_id` + `created_at`. `course_id` is NOT currently carried forward (only reached `concept_graph.json`), but that is Worker O's scope, not mine — I leave it.

4. **`_make_concept_id(slug, course_id)`** helper at `Trainforge/rag/typed_edge_inference.py:55–70` already produces the composite `{course_id}:{slug}` or flat-slug IDs depending on `SCOPE_CONCEPT_IDS`. `_build_tag_graph` already uses it for node IDs. The inverted-index keys must use the SAME helper so that node-ID and occurrences-key align under either flag state.

5. **Wave 4 Worker P** stamps `run_id` + `created_at` on base-graph nodes via `_stamp_node_provenance` call-site in `_build_tag_graph` (and typed-edge `_build_nodes`). I do NOT touch this logic — `occurrences` is emitted ADDITIONALLY.

---

## Architecture decision — where to populate

Two places emit concept-graph nodes:

| Site | Artifact | Kind |
|---|---|---|
| `process_course._build_tag_graph` | `concept_graph.json` | `concept` (also builds pedagogy mirror via same fn) |
| `typed_edge_inference._build_nodes` | `concept_graph_semantic.json` | `concept_semantic` (copies from base) |

Master plan says modify `_build_tag_graph` — and that's where `chunks` live. **Decision: populate ONLY in `_build_tag_graph`** (this worker's scope). `typed_edge_inference.py` is explicitly off-limits for this worker per task constraint (Worker U's Wave 5.2 scope).

Per-artifact state after this worker merges:

- `concept_graph.json` — nodes carry `occurrences[]` (Worker S adds this).
- `concept_graph_semantic.json` — nodes DO NOT yet carry `occurrences[]` because `_build_nodes` only copies a fixed field list. **Worker U handoff**: during Wave 5.2, Worker U extends `_build_nodes` to carry `occurrences` forward (a 2-line diff within their scope) so the `defined-by` rule and any semantic-graph consumer can read it. The schema I publish in `concept_graph_semantic.schema.json` declares the field as optional, making the eventual Worker U carry-forward schema-compliant.

Rationale for splitting the work this way:
1. Data production happens where `chunks` live — `_build_tag_graph`.
2. Schema documentation publishes the contract up-front so Worker U's carry-forward is a trivial mechanical change, not a schema decision.
3. Schema has `additionalProperties: true` so in the interim (between S merging and U merging) the semantic-graph artifact remains valid with or without `occurrences`.

**Downstream note for Worker U.** The `defined-by` rule (concept→first-chunk) can read `occurrences[0]` directly from a semantic-graph node after Worker U's carry-forward lands. Alternatively, the rule can read the base `concept_graph.json` (which S writes fully). Either path works — U's sub-plan picks.

---

## Ordering decision

**Sorted ASCII-ASC by chunk ID string.** Native Python `sorted()` on the chunk-ID list. This is deterministic across runs for any given input chunk set, cross-platform stable, and — critically — matches the pattern Worker O used for edge endpoints (`key = tuple(sorted([a, b]))`).

Under content-hash chunk IDs (Worker N's flag on), sort is by the hex suffix of each ID (still stable ASCII order). Under position-based IDs (default), sort is by the 5-digit zero-padded position → equivalent to natural-order-by-position. Either way the output is byte-identical across re-runs of the same input.

---

## Changes

### 1. `schemas/knowledge/concept_graph_semantic.schema.json`

Add `occurrences` property to the node `properties` block (alongside the existing `run_id`/`created_at`/`course_id` Wave 4 additions):

```json
"occurrences": {
  "type": "array",
  "items": { "type": "string" },
  "description": "Chunk IDs that reference this concept (REC-LNK-01, Worker S Wave 5.1). Populated from chunk→concept inverted index at graph-build time. Sorted for deterministic output. Stable across re-chunks only when TRAINFORGE_CONTENT_HASH_IDS=true (Worker N's flag); position-based chunk IDs invalidate entries on re-chunk."
}
```

Node-schema `required` stays `["id"]`. Legacy nodes without `occurrences` still validate because `additionalProperties: true` and the new property is optional.

### 2. `Trainforge/process_course.py::_build_tag_graph`

Extend to build the inverted index and attach it to nodes. Minimal diff: one new `defaultdict(list)` accumulator populated alongside the existing `tag_frequency` / `co_occurrence` loop; one new `if` block after the node dict is constructed.

```python
# Before the node emit loop, build inverted index:
concept_to_chunks: Dict[str, List[str]] = defaultdict(list)
for chunk in chunks:
    chunk_id = chunk.get("id")
    if not chunk_id:
        continue
    for tag in chunk.get("concept_tags", []):
        if not _accept(tag):
            continue
        node_id = _make_concept_id(tag, course_id)
        concept_to_chunks[node_id].append(chunk_id)

# Then in the node construction loop:
# ... existing node = {id, label, frequency} ...
if node_id in concept_to_chunks:
    node["occurrences"] = sorted(concept_to_chunks[node_id])
```

Deduplication: if a chunk lists the same tag twice in its `concept_tags` array, the chunk ID ends up twice in `occurrences`. Guard with a per-chunk `set()` to keep each `(chunk_id, node_id)` pair unique. Then `sorted(list(set(...)))`.

**Frequency relationship.** After this change, `len(node["occurrences"])` equals the number of DISTINCT chunks mentioning the concept — which is NOT necessarily equal to `node["frequency"]`. `frequency` counts total tag occurrences (a single chunk that lists a tag twice counts twice); `occurrences` counts distinct chunks. Document this in a comment so future maintainers don't "fix" the difference.

### 3. Pedagogy mirror (`_generate_pedagogy_graph`)

`_build_tag_graph` is also called for the pedagogy mirror (kind=`pedagogy`). The same inverted-index logic runs there too — pedagogy-tag nodes get their own `occurrences[]` listing chunks that carry pedagogy tags. This is a free byproduct (no extra code), and Wave 5 schema additions apply to any node that matches the schema. Tests assert this side-effect doesn't break the pedagogy graph.

---

## Test design — `Trainforge/tests/test_concept_occurrences.py`

Fixture: minimal `_build_concept_graph` helper mirroring `test_concept_scoping.py`'s pattern — avoids spinning up the full `CourseProcessor` pipeline. Chunks constructed inline.

### Test 1 — `test_node_carries_occurrences_list`
Build a graph from 3 chunks (IDs: `c_00001`, `c_00002`, `c_00003`), concept tags:
- `c_00001`: `["a", "b"]`
- `c_00002`: `["a", "c"]`
- `c_00003`: `["b"]`

With min-freq=2 filter: expected nodes are `a` (freq=2), `b` (freq=2), `c` filtered out. Assert:
- node `a` has `occurrences == ["c_00001", "c_00002"]`
- node `b` has `occurrences == ["c_00001", "c_00003"]`

### Test 2 — `test_occurrences_are_sorted`
Build a graph where same tag appears in chunks added in reverse-ID order. Assert `occurrences[]` is sorted ASC regardless of chunk-iteration order.

### Test 3 — `test_occurrences_match_inverted_index`
Build a graph; then from the SAME chunks compute a manual `concept_to_chunks` dict and assert per-node `occurrences[]` equals the sorted values of that manual dict for every node. Catches any drift between inverted-index and node emission.

### Test 4 — `test_legacy_nodes_without_occurrences_validate`
Build a minimal `concept_graph_semantic.json` dict with nodes that DO NOT carry `occurrences` (legacy shape). Load the schema. Assert `jsonschema.validate` passes.

### Test 5 — `test_occurrences_survive_rechunk_under_content_hash`
Monkeypatch `TRAINFORGE_CONTENT_HASH_IDS=true`. Build graph twice from semantically identical chunks (same text, same source locator, same concept_tags) — content-hash IDs will be identical across runs. Assert `occurrences[]` identical across the two graphs. This closes the "stability claim in schema description" loop.

Additional micro-assertion embedded across tests: when a chunk has a duplicate tag, the chunk ID appears only ONCE in `occurrences[]` (dedup sanity).

---

## Verification

1. `python3 -m ci.integrity_check` → 8/8 (no schema-validation regressions).
2. `pytest Trainforge/tests/test_concept_occurrences.py -x -v` → 5 tests pass.
3. `pytest lib/tests/ Trainforge/tests/ Courseforge/scripts/tests/ MCP/tests/ -q` → full suite green (baseline ~857 + 5 new = ~862).
4. Spot-check: construct graph, assert every emitted node carries `occurrences[]`, and the sum of `len(occurrences)` across nodes equals the count of distinct (chunk_id, concept_id) pairs in the source.

---

## Scope boundaries

**Out of scope (owned by other workers, do not touch):**
- `Trainforge/rag/typed_edge_inference.py` precedence / edge enum / rule orchestration — Worker U Wave 5.2.
- `schemas/knowledge/instruction_pair.schema.json` + strict variant + content_type enforcement — Worker T Wave 5.1.
- Any LibV2 file (Worker T's scope boundary).
- Wave 6 provenance/governance items.

**Env vars.** None. Per master plan design decision #1, `occurrences[]` is always-on additive. The only env var that indirectly affects this worker is Worker N's `TRAINFORGE_CONTENT_HASH_IDS` — and only because it determines whether `occurrences[]` survives re-chunks. That flag is OUT of this worker's scope to modify; test 5 simply exercises it.

---

## Integrity-check / test inventory

- New file: `Trainforge/tests/test_concept_occurrences.py` (5 tests).
- Modified: `schemas/knowledge/concept_graph_semantic.schema.json` (+1 property on node schema).
- Modified: `Trainforge/process_course.py` (+~15 lines in `_build_tag_graph`).
- New sub-plan file: `plans/kg-quality-review-2026-04/worker-s-subplan.md` (this file).

(Intentionally NOT modified: `Trainforge/rag/typed_edge_inference.py` — Worker U scope.)
