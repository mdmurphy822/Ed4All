# Worker P Sub-Plan — Wave 4.1 REC-PRV-01: `run_id` + `created_at` provenance fields

**Branch:** `worker-p/wave4-provenance-fields`
**Base:** `dev-v0.2.0` @ `e3dde95`
**Master plan:** `/home/mdmur/.claude/plans/we-have-several-branches-gentle-melody.md` § Worker P

## Scope

Thread `run_id` + `created_at` onto every newly-emitted:
- chunk dict (`Trainforge/process_course.py::_create_chunk`)
- concept graph node (typed-edge artifact emitted by
  `Trainforge/rag/typed_edge_inference.py::build_semantic_graph`)
- concept graph edge (all three rule emitters + LLM escalation path)

Schemas get OPTIONAL additions. Legacy artifacts without the fields continue
to validate. Always-emit on new runs. No env-var gate — purely additive.

---

## 1. Source of `run_id`

### 1.1 Decision-capture context is the source of truth

`lib/decision_capture.py:158–191` — `DecisionCapture.__init__` assigns
`self.run_id` in priority order:

1. If a hardened `RunContext` is active (`get_current_run()` returns non-None),
   use `self._run_context.run_id`.
2. Else, fall back to `os.environ.get('RUN_ID', f"{tool}_{course_code}_{session_id}")`.

Both paths populate `self.run_id` before any decision is logged and before
any chunk is emitted. The same value is already written on every
`decision_event.schema.json` record via `_build_record` at L372.

### 1.2 How `process_course.py` inherits the run_id

`CourseProcessor.__init__` at L683-689 creates
`self.capture = DecisionCapture(course_code=..., phase="content_extraction",
tool="trainforge", streaming=True)` before any chunk is emitted. So
`self.capture.run_id` is always available at `_create_chunk` call time and
is the same `run_id` that decision-event records carry for this run. This is
the canonical source.

**Decision for P:** stamp `chunk["run_id"] = self.capture.run_id`. No new
env var, no new context manager. Reuses the already-wired ledger context.

### 1.3 How concept-graph emitters inherit the run_id

`_generate_semantic_concept_graph` at L2096-2136 calls `build_semantic_graph(
..., decision_capture=self.capture)`. The `decision_capture` argument is
already the canonical `DecisionCapture` instance for this run. We thread
`run_id` + `created_at` through the call by:

- Adding `run_id: Optional[str]` and `now: Optional[datetime]` parameters to
  `build_semantic_graph` (note: `now` already exists for `generated_at`; we
  extend it to also stamp per-node/per-edge `created_at`).
- Extracting `run_id` from `decision_capture.run_id` when not provided
  explicitly, so callers who pass only `decision_capture` get correct
  provenance for free.
- Threading both into `_build_nodes`, each rule's emit site (via a small
  `_stamp_provenance` helper on the orchestrator), and `_llm_escalate`.

Rule modules (`is_a_from_key_terms.py`, `prerequisite_from_lo_order.py`,
`related_from_cooccurrence.py`) stay pure — the orchestrator is the
stamping point. Rule modules don't know about `run_id`; the orchestrator
decorates their output. This preserves the current "rule modules are
behavior-pure fixtures" contract.

---

## 2. `created_at` format

ISO 8601 UTC string: `datetime.now(timezone.utc).isoformat()`.

- For chunks: each chunk gets its own `created_at` at `_create_chunk` call
  time — different chunks in the same run may differ by sub-second.
- For concept-graph nodes and edges: all use the same `created_at` — the
  top-level `build_semantic_graph` already accepts a `now` override (used
  by tests for byte-identical output across runs); we reuse it so
  per-node/per-edge `created_at` equals the artifact's top-level
  `generated_at`. This is deliberate — the graph is an atomic snapshot;
  nothing is emitted sequentially during build, so there's no timestamp
  skew to preserve.

Deterministic tests can pin `now=FIXED_NOW` and get byte-identical output
(existing fixture test at `test_deterministic_fallback_produces_identical_artifacts`
is preserved because all new fields inherit the `now` override).

---

## 3. Schema additions

### 3.1 `schemas/knowledge/chunk_v4.schema.json`

Add two optional properties at the top-level `properties` object (not in
`required[]`):

```json
"run_id": {
  "type": "string",
  "description": "Pipeline run that produced this chunk. Matches decision_event.schema.json run_id format. Always populated on new runs; legacy chunks without this field still validate."
},
"created_at": {
  "type": "string",
  "format": "date-time",
  "description": "ISO 8601 UTC timestamp when chunk was emitted. Legacy chunks without this field still validate."
}
```

No `required[]` change — legacy chunks keep validating. The root
`additionalProperties: true` already permits the fields, so adding them to
`properties` is documentation + type/format assertion, not a contract
change.

### 3.2 `schemas/knowledge/concept_graph_semantic.schema.json`

Add `run_id` + `created_at` to BOTH the node schema (`properties.nodes.items.properties`)
AND the edge schema (`properties.edges.items.properties`). Both optional
(not appended to `required[]`). Both are string types; `created_at` has
`format: date-time`.

Schema conflict note: Worker O (not yet dispatched when P writes this
plan) is adding `course_id` to the node schema. Worker P's node-schema
diff touches a different sub-object region of the same node block — the
additions to `properties.nodes.items.properties` are textually adjacent
but not conflicting. Per master plan § "Schema conflict note", P merges
first; O rebases if needed.

---

## 4. Emit-side changes

### 4.1 `process_course.py::_create_chunk`

Add two lines to the chunk dict build at L1269:

```python
chunk: Dict[str, Any] = {
    "id": chunk_id,
    "schema_version": CHUNK_SCHEMA_VERSION,
    ...
    "run_id": self.capture.run_id,
    "created_at": datetime.now(timezone.utc).isoformat(),
}
```

Import `timezone` from `datetime` if not already present (currently only
`datetime` is imported at L35). Add `from datetime import datetime, timezone`
or similar. Verify no shadowing of the already-imported `datetime` class.

`self.capture` is always present in real runs but unit tests at
`Trainforge/tests/test_provenance.py` bypass `__init__` and may not have
`self.capture`. Mirror the existing defensive pattern at L1346:
`capture = getattr(self, "capture", None)`. If `capture is None`, skip
`run_id` entirely (`created_at` still stamped — it's from `datetime.now`,
not capture).

### 4.2 `typed_edge_inference.py`

1. Extend `build_semantic_graph` signature:
   - Existing `now: Optional[datetime] = None` stays as artifact-level
     override.
   - Add `run_id: Optional[str] = None` — when None, extracted from
     `decision_capture.run_id` if `decision_capture` is not None; else None.

2. Centralize stamping in a helper:
   ```python
   def _stamp_provenance(obj: Dict[str, Any], run_id: Optional[str],
                         created_at: str) -> Dict[str, Any]:
       if run_id:
           obj["run_id"] = run_id
       obj["created_at"] = created_at
       return obj
   ```

3. Apply to nodes in `_build_nodes` — each node dict gets stamped.

4. Apply to edges after rule execution but before precedence resolution
   (so dropped-by-precedence edges are cheap; stamping a dict that gets
   dropped is harmless). Easiest: stamp in the loop that accumulates
   `rule_edges` and in `_llm_escalate`'s normalization loop.

### 4.3 No changes to rule modules

`is_a_from_key_terms.py`, `prerequisite_from_lo_order.py`,
`related_from_cooccurrence.py` stay untouched — they return rule-pure edge
dicts. The orchestrator decorates. This preserves determinism tests and
keeps rule modules free of time-dependent state.

---

## 5. Regression tests

New file `Trainforge/tests/test_run_provenance.py`. Uses the existing
fixture dir `Trainforge/tests/fixtures/mini_course_typed_graph/` for
semantic-graph tests. For chunk tests, builds a minimal in-memory
`CourseProcessor` stub (mirrors pattern in `test_provenance.py`).

### 5.1 `test_chunks_carry_run_id_and_created_at`

- Build a minimal `CourseProcessor` (bypass `__init__` OK for unit test);
  attach a stub `capture` object with `.run_id = "test_run_42"`; call
  `_create_chunk(...)`; assert `chunk["run_id"] == "test_run_42"`; assert
  `chunk["created_at"]` parses as ISO 8601 and is timezone-aware
  (contains `+00:00` or ends in `Z`).

### 5.2 `test_concept_nodes_carry_run_id_and_created_at`

- Load fixture; invoke `build_semantic_graph(..., run_id="test_run_42",
  now=FIXED_NOW)`; for every node in `graph["nodes"]` assert
  `n["run_id"] == "test_run_42"` and `n["created_at"] == FIXED_NOW.isoformat()`.

### 5.3 `test_concept_edges_carry_run_id_and_created_at`

- Same fixture; for every edge assert `e["run_id"] == "test_run_42"` and
  `e["created_at"] == FIXED_NOW.isoformat()`.

### 5.4 `test_legacy_chunks_without_fields_validate`

- Load `schemas/knowledge/chunk_v4.schema.json`; synthesize a minimal
  chunk dict WITHOUT `run_id` / `created_at` that satisfies all existing
  `required[]` fields; use `_load_chunk_validator` from `process_course`
  (or a direct `jsonschema.Draft202012Validator`); assert validator
  produces no errors.

### 5.5 `test_run_id_source_from_decision_capture`

- Create a `DecisionCapture(course_code="TEST", phase="content_extraction",
  tool="trainforge")`; assert `build_semantic_graph(chunks, course,
  concept_graph, decision_capture=capture, now=FIXED_NOW)` stamps
  `run_id` equal to `capture.run_id` on every node and edge (i.e. when
  no explicit `run_id` kwarg is passed, it falls back to the capture).

---

## 6. File-level change summary

| File | Change |
|------|--------|
| `schemas/knowledge/chunk_v4.schema.json` | +2 optional properties: `run_id`, `created_at` |
| `schemas/knowledge/concept_graph_semantic.schema.json` | +2 optional properties on BOTH node and edge schemas |
| `Trainforge/process_course.py` | Import `timezone`; stamp `run_id` + `created_at` in `_create_chunk` |
| `Trainforge/rag/typed_edge_inference.py` | Add `run_id` kwarg to `build_semantic_graph`; add `_stamp_provenance` helper; stamp all nodes + edges + LLM edges |
| `Trainforge/tests/test_run_provenance.py` | NEW — 5 regression tests |
| `plans/kg-quality-review-2026-04/worker-p-subplan.md` | THIS FILE |

No changes to:
- Rule modules (`is_a_from_key_terms.py`, `prerequisite_from_lo_order.py`,
  `related_from_cooccurrence.py`) — they remain rule-pure.
- Chunk ID generation (Worker N's scope).
- Concept ID scoping (Worker O's scope).
- `preference_factory.py` / misconception handling (Worker R's Wave 4.2 scope).
- Main branch.

---

## 7. Verification

```bash
cd /home/mdmur/Projects/Ed4All/.claude/worktrees/worker-p
python3 -m ci.integrity_check
source venv/bin/activate && pytest Trainforge/tests/test_run_provenance.py -x
pytest lib/tests/ Trainforge/tests/ Courseforge/scripts/tests/ MCP/tests/ -q
```

Expected: integrity check stays 8/8; 5 new regression tests pass; no
regressions in the existing ≥688 test baseline.
