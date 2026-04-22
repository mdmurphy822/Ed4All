# Worker U Sub-plan — REC-LNK-04: 5 new pedagogical edge types

**Branch:** `worker-u/wave5-pedagogical-edges`
**Base:** `dev-v0.2.0` (@ 10a13c8, Wave 5.1 merged)
**Master plan:** `/home/mdmur/.claude/plans/we-have-several-branches-gentle-melody.md` → Wave 5.2 / Worker U

---

## Goal

Add 5 new pedagogical edge types to the typed concept graph. Each gets its own rule module (one file per rule) following the existing interface pattern. Extend the schema enum, precedence map, and rule registry. Apply Worker S's handoff — carry `occurrences[]` from the co-occurrence graph into the semantic graph `_build_nodes` output.

**Federation-by-convention.** Edges reference external-namespace IDs directly — concept IDs (flat slug or `{course_id}:{slug}`), LO IDs (`TO-NN`, `CO-NN`), chunk IDs, misconception IDs (`mc_[0-9a-f]{16}`). No new node types are added to the concept-graph schema — consumers resolve endpoints by ID namespace prefix.

**Precedence.** All 5 new types at tier 2 (same as `prerequisite`). They don't collide with the existing taxonomic edges because their endpoint domains are different (is-a/prerequisite/related-to connect concept↔concept; new types cross concept↔chunk, concept↔LO, chunk↔LO, misconception↔concept, question↔LO).

---

## Current-state anchors (verified on this branch)

1. **`Trainforge/rag/inference_rules/__init__.py`** — thin registry exposing three `infer_*` functions (`is_a`, `prerequisite`, `related`). Extend by adding 5 new imports + `__all__` entries.

2. **`Trainforge/rag/inference_rules/related_from_cooccurrence.py`** — canonical minimal rule. Interface: module-level `RULE_NAME`, `RULE_VERSION=1`, `EDGE_TYPE`; function `infer(chunks, course, concept_graph, **kwargs) -> List[Dict]`; deterministic output sorted by `(source, target)`; returns a list of edge dicts with `{source, target, type, confidence, provenance: {rule, rule_version, evidence}}`.

3. **`Trainforge/rag/typed_edge_inference.py::_PRECEDENCE`** (L75–79) — 3 entries (`is-a`:3, `prerequisite`:2, `related-to`:1). Extend with 5 new keys at value 2.

4. **`Trainforge/rag/typed_edge_inference.py::build_semantic_graph`** (L314–330) — rule invocation loop over a fixed `(fn, rule_mod, kwargs)` tuple sequence. Add 5 new entries; keep invocation order deterministic.

5. **`Trainforge/rag/typed_edge_inference.py::_build_nodes`** (L156–181) — currently copies `{id, label, frequency}` + stamps `run_id` + `created_at`. Worker S's handoff: also carry `occurrences[]` when the source node has it. This preserves Worker S's invariant that `occurrences[]` survives from `concept_graph.json` through to `concept_graph_semantic.json`.

6. **`Trainforge/rag/typed_edge_inference.py::_UNDIRECTED_TYPES`** (L84) — only `related-to`. All 5 new types are directed (concept→chunk has a clear direction; LO→question too). No change needed.

7. **`_make_concept_id(slug, course_id)`** (L55–70) — scoping helper honoured throughout existing rules. New rules must use it when resolving concept slugs to node IDs (specifically the `defined-by` and `exemplifies` rules which derive concept IDs from chunk `concept_tags`).

8. **Chunk shape** (from `Trainforge/process_course.py` L1273–1400) — every emitted chunk has `id`, `concept_tags: List[str]`, `learning_outcome_refs: List[str]` (lowercased), `chunk_type: str` (e.g. `"document_text"`, `"example"`), optional `content_type_label: str` (e.g. `"example"`, `"explanation"`, `"none"`). `source.course_id` present under scoped-ID mode.

9. **Misconception schema** (Worker R, Wave 4) at `schemas/knowledge/misconception.schema.json` — `id` (`mc_[0-9a-f]{16}`), `misconception`, `correction`, optional `concept_id`, optional `lo_id`. Misconceptions are NOT currently threaded into `build_semantic_graph`'s call signature — the rule accepts them via `**kwargs` as `misconceptions=[...]` when available, emits empty gracefully otherwise.

10. **Assessment question shape** — questions are NOT currently in `build_semantic_graph`'s call signature. Rule accepts via `**kwargs` as `questions=[...]`; emits empty when absent. Questions expected to carry `id`, `objective_id`, optionally `source_chunk_id`.

---

## Design decisions

### D1 — Evidence shapes (finalized per master plan)

| Rule | Evidence |
|------|----------|
| `assesses` | `{question_id, objective_id, source_chunk_id}` (source_chunk_id optional; omitted if not present on question) |
| `exemplifies` | `{chunk_id, concept_slug, content_type}` |
| `misconception-of` | `{misconception_id, concept_id}` |
| `derived-from-objective` | `{chunk_id, objective_id}` |
| `defined-by` | `{chunk_id, concept_slug, first_mention_position: 0}` |

### D2 — Confidence formulas

| Rule | Confidence | Rationale |
|------|-----------|-----------|
| `derived-from-objective` | `1.0` | Explicit reference from `chunk.learning_outcome_refs[]` — not inferred. |
| `assesses` | `1.0` | Explicit reference from `question.objective_id`. |
| `misconception-of` | `1.0` | Explicit reference from `misconception.concept_id`. |
| `exemplifies` | `0.8` | Chunk flagged as example + has concept tags — strong but not explicit "exemplifies concept X" assertion. |
| `defined-by` | `0.7` | First-mention by chunk-ID-sort-order is a proxy for "canonical definition". Not necessarily pedagogical first-definition but structurally reasonable. |

### D3 — Signal source for each rule

| Rule | Signal | Current availability |
|------|--------|---------------------|
| `derived-from-objective` | `chunk.learning_outcome_refs[]` | **Available now** — every chunk emits this. |
| `defined-by` | Worker S's `occurrences[]` on each concept node, first entry (sorted ASC by chunk_id) | **Available now** — Wave 5.1 merged. |
| `exemplifies` | Chunks where `chunk_type == "example"` OR `content_type_label == "example"`, using their `concept_tags` | **Available in principle** — chunks carry both fields; fires when a course has example-type content. |
| `misconception-of` | Misconceptions passed via `**kwargs` (`misconceptions=[...]`) with `concept_id` populated | **Awaiting upstream wiring** — misconceptions not currently threaded into `build_semantic_graph`. Rule emits empty until future wave wires them. |
| `assesses` | Questions passed via `**kwargs` (`questions=[...]`) with `objective_id` populated | **Awaiting upstream wiring** — not currently threaded. Emits empty until future wave. |

### D4 — Why `**kwargs` for misconceptions/questions instead of extending `build_semantic_graph` signature

Keeping the core `build_semantic_graph(chunks, course, concept_graph)` signature stable avoids breaking existing call sites. The rule modules accept `misconceptions` / `questions` via `**kwargs`, and the orchestrator's invocation loop threads them through when provided. Current call sites pass neither → rules emit empty → **no behavioral regression**.

A future wave can add explicit `misconceptions=...` / `questions=...` parameters to `build_semantic_graph` once upstream pipelines emit them. The rules are correct today; only the signal plumbing is deferred.

### D5 — Precedence tier 2 for all 5 new types

Master plan design decision #3: all new types at precedence 2 (same as `prerequisite`). Since they use disjoint endpoint namespaces from the taxonomic edges (concept↔concept), they should never collide with `is-a`/`prerequisite`/`related-to` on the same `(source, target)` pair in practice. The tier 2 assignment is defensive — if two new rules ever produce edges on the same endpoint pair, they tie-break by rule-invocation order (deterministic) just like the existing three rules.

### D6 — Scope-ID helper honoured in rules that touch concepts

`defined-by` and `exemplifies` derive concept node IDs. Both use `_make_concept_id(slug, course_id)` via the chunk's `source.course_id` — matching the pattern in `is_a_from_key_terms.py` and `prerequisite_from_lo_order.py`. The `derived-from-objective`, `assesses`, and `misconception-of` rules don't resolve concept slugs — they emit endpoint IDs that come directly from upstream data (chunk IDs, LO IDs, question IDs, misconception IDs, concept IDs), unchanged.

### D7 — Worker S handoff in `_build_nodes`

Current (L171–181):
```python
node = {
    "id": n["id"],
    "label": n.get("label", n["id"]),
    "frequency": n.get("frequency", 0),
}
```

Extension: when source node carries `occurrences`, copy it verbatim. 2-line addition — Worker S flagged this in sub-plan as a Wave 5.2 handoff (her scope stopped at `concept_graph.json`; the semantic-graph node emit site belongs to typed_edge_inference).

---

## Files created (5 rule modules)

### 1. `Trainforge/rag/inference_rules/derived_from_lo_ref.py`
- `RULE_NAME = "derived_from_lo_ref"`, `RULE_VERSION = 1`, `EDGE_TYPE = "derived-from-objective"`.
- Iterate chunks. For each chunk with `id` and non-empty `learning_outcome_refs`, emit one edge per `(chunk_id, lo_id)` pair.
- Edge: `{source: chunk_id, target: lo_id, type: "derived-from-objective", confidence: 1.0, provenance: {..., evidence: {chunk_id, objective_id}}}`.
- Dedup on `(source, target)` — a chunk listing the same LO twice emits one edge.
- Sort by `(source, target)`.

### 2. `Trainforge/rag/inference_rules/defined_by_from_first_mention.py`
- `RULE_NAME = "defined_by_from_first_mention"`, `RULE_VERSION = 1`, `EDGE_TYPE = "defined-by"`.
- Iterate concept_graph nodes. For each node with non-empty `occurrences`, take the first entry (already sorted ASC by `_build_tag_graph` → `sorted(occurrences)`).
- Edge: `{source: concept_id, target: chunk_id, type: "defined-by", confidence: 0.7, provenance: {..., evidence: {chunk_id, concept_slug, first_mention_position: 0}}}`.
- `concept_slug` = node["id"] stripped of the `{course_id}:` prefix if present (for backward-compat readability in evidence).
- Sort by `(source, target)`.

### 3. `Trainforge/rag/inference_rules/exemplifies_from_example_chunks.py`
- `RULE_NAME = "exemplifies_from_example_chunks"`, `RULE_VERSION = 1`, `EDGE_TYPE = "exemplifies"`.
- Iterate chunks. For each chunk where `chunk_type == "example"` OR `content_type_label == "example"`, emit one edge per `concept_tag` (after `_make_concept_id` scoping) that resolves to a node in `concept_graph`.
- Edge: `{source: chunk_id, target: concept_id, type: "exemplifies", confidence: 0.8, provenance: {..., evidence: {chunk_id, concept_slug, content_type}}}`.
- `content_type` captures whichever of the two fields triggered the rule — preferring `content_type_label` when both are `"example"`.
- Sort by `(source, target)`.

### 4. `Trainforge/rag/inference_rules/misconception_of_from_misconception_ref.py`
- `RULE_NAME = "misconception_of_from_misconception_ref"`, `RULE_VERSION = 1`, `EDGE_TYPE = "misconception-of"`.
- Signature: `infer(chunks, course, concept_graph, *, misconceptions=None, **kwargs)`.
- If `misconceptions is None` or empty → return `[]` (graceful no-op until upstream wires signal).
- For each misconception with populated `concept_id` → emit edge `{source: misconception_id, target: concept_id, type: "misconception-of", confidence: 1.0, provenance: {..., evidence: {misconception_id, concept_id}}}`.
- Sort by `(source, target)`.

### 5. `Trainforge/rag/inference_rules/assesses_from_question_lo.py`
- `RULE_NAME = "assesses_from_question_lo"`, `RULE_VERSION = 1`, `EDGE_TYPE = "assesses"`.
- Signature: `infer(chunks, course, concept_graph, *, questions=None, **kwargs)`.
- If `questions is None` or empty → `[]`.
- For each question with `id` + `objective_id` → emit edge `{source: question_id, target: objective_id, type: "assesses", confidence: 1.0, provenance: {..., evidence: {question_id, objective_id, source_chunk_id}}}` (source_chunk_id only when present on question).
- Sort by `(source, target)`.

---

## Files modified

### 6. `schemas/knowledge/concept_graph_semantic.schema.json`
- Extend edge `type` enum from `["prerequisite", "is-a", "related-to"]` to include `"assesses"`, `"exemplifies"`, `"misconception-of"`, `"derived-from-objective"`, `"defined-by"`.
- Add `$comment` inside the edge schema documenting the federation-by-convention ID-namespace approach: endpoints may be concept IDs (flat slug or `{course_id}:{slug}`), LO IDs (`TO-NN`/`CO-NN`), chunk IDs, misconception IDs (`mc_*`), or question IDs. Consumers resolve by namespace prefix — no new node types added.

### 7. `Trainforge/rag/typed_edge_inference.py`
- Extend `_PRECEDENCE` with 5 new entries at value 2 (alphabetized for readability):
  ```python
  _PRECEDENCE: Dict[str, int] = {
      "is-a": 3,
      "assesses": 2,
      "defined-by": 2,
      "derived-from-objective": 2,
      "exemplifies": 2,
      "misconception-of": 2,
      "prerequisite": 2,
      "related-to": 1,
  }
  ```
- Import 5 new rule modules.
- Extend rule-invocation loop: add tuples for each new rule in a fixed order (alphabetical by EDGE_TYPE after the existing three for predictability).
- Thread `misconceptions` + `questions` kwargs from `build_semantic_graph` through to the rules (new optional parameters on `build_semantic_graph`, defaulting to `None`). Backward-compat: existing calls pass neither → rules emit empty.
- Apply Worker S handoff in `_build_nodes`: when source node has `occurrences`, copy onto the output node.

### 8. `Trainforge/rag/inference_rules/__init__.py`
- Import 5 new `infer` functions; export via `__all__`.
- Update the docstring note on EDGE_TYPE to list the expanded enum.

---

## Regression tests — `Trainforge/tests/test_pedagogical_edges.py`

One test per rule (happy-path + empty-signal + evidence shape):

1. `test_derived_from_lo_ref_happy_path` — chunk with `learning_outcome_refs: ["to-01", "co-05"]` emits 2 edges, each with evidence `{chunk_id, objective_id}`, confidence 1.0.
2. `test_derived_from_lo_ref_empty` — no chunks with refs → no edges.
3. `test_defined_by_from_first_mention_happy_path` — concept node with `occurrences: ["chunk_b", "chunk_a", "chunk_c"]` (pre-sort) produces edge to `chunk_a` after ASC-sort. Evidence: `{chunk_id: "chunk_a", concept_slug, first_mention_position: 0}`.
4. `test_defined_by_empty_when_no_occurrences` — nodes without `occurrences` → no edges.
5. `test_exemplifies_chunk_type_example` — chunk with `chunk_type: "example"` + `concept_tags: ["widget", "stage"]` + both concepts in graph → 2 edges.
6. `test_exemplifies_content_type_label_example` — `chunk_type: "document_text"`, `content_type_label: "example"` → also fires.
7. `test_exemplifies_skips_non_example` — `chunk_type: "document_text"`, `content_type_label: "explanation"` → no edges.
8. `test_misconception_of_empty_when_no_misconceptions_kwarg` — no `misconceptions` kwarg → empty output.
9. `test_misconception_of_empty_when_no_concept_id` — misconception missing `concept_id` → empty output.
10. `test_misconception_of_happy_path` — misconception with `concept_id` populated → edge emitted.
11. `test_assesses_empty_when_no_questions_kwarg` — no `questions` kwarg → empty output.
12. `test_assesses_happy_path` — question with `id` + `objective_id` → edge emitted. `source_chunk_id` propagates when present.
13. `test_edge_enum_includes_new_types` — synthetic artifact with one edge of each new type validates against schema.
14. `test_precedence_new_types_tier_2_noninterference` — synthetic graph with `is-a` + `exemplifies` on same `(source, target)` keeps the `is-a` (tier 3 beats tier 2). With `exemplifies` + `related-to` on same pair, `exemplifies` wins (tier 2 beats tier 1).
15. `test_deterministic_output` — run `build_semantic_graph` twice with same inputs + fixed `now` → byte-identical JSON.
16. `test_integration_all_five_types` — synthetic fixture with chunks (incl. example), misconceptions, questions, LO refs, and populated `occurrences` on concept-graph nodes → all 5 new edge types emit ≥1 edge each.
17. `test_occurrences_carry_forward_in_build_nodes` — concept_graph node with `occurrences: [...]` flows through to semantic-graph node output. Guards Worker S's handoff.

---

## Verification

1. `python3 -m ci.integrity_check` → all checks pass (no new schemas added; schema-count checksum unaffected).
2. `pytest Trainforge/tests/test_pedagogical_edges.py -x` → green.
3. `pytest Trainforge/tests/test_typed_edge_inference.py lib/tests/ Trainforge/tests/ Courseforge/scripts/tests/ MCP/tests/ LibV2/tools/libv2/tests/ -q` → no regressions.

---

## Risks and mitigations

- **Risk:** Adding `misconceptions`/`questions` kwargs breaks deterministic hash of existing fixture output. **Mitigation:** default both to `None` → existing test fixture output unchanged (rules short-circuit on empty kwargs).
- **Risk:** `defined-by` rule re-emits an edge per `occurrences` entry. **Mitigation:** only the first-sorted entry is used; iteration is once-per-concept-node.
- **Risk:** `exemplifies` concept-slug-to-node-ID lookup mismatches the scoped-ID flag state. **Mitigation:** use `_make_concept_id(tag, chunk.source.course_id)` (pattern proven in `is_a_from_key_terms.py`).
- **Risk:** Precedence tier-2 ties between new rules produce non-deterministic output. **Mitigation:** rule-invocation loop runs in fixed order; `_apply_precedence` is stable on equal-precedence keys.

---

## Scope boundaries

- Do NOT wire misconceptions/questions into `build_semantic_graph` call sites in production code paths. That's a future wave's job (upstream pipelines need to emit + thread through).
- Do NOT add new node types to the concept-graph schema. Federation-by-convention only.
- Do NOT change precedence for existing types.
- Do NOT touch the base `concept_graph.json` builder (`_build_tag_graph` in `process_course.py`).
- Do NOT modify Worker S's `occurrences[]` emission logic — only carry it forward in `_build_nodes`.
