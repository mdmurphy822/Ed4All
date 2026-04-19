# mini_course_typed_graph — typed-edge inference fixture

Small, deterministic fixture for Worker F's typed-edge concept-graph
inference. Exercises all three rule modules (`is_a_from_key_terms`,
`prerequisite_from_lo_order`, `related_from_cooccurrence`) plus precedence
resolution.

The fixture is stored as the three JSON artifacts the inference layer
consumes directly (chunks, course, co-occurrence concept graph). We do NOT
re-run the full pipeline through IMSCC parsing — that is the job of
`mini_course_clean` and related fixtures. Keeping this fixture narrow to
the typed-edge layer lets CI fail with a rule-specific message when the
inference rules drift.

## Files

| File | Purpose |
|---|---|
| `chunks.jsonl` | Four chunks tagged with concept tags, learning-outcome refs, and `key_terms[].definition` strings that match the `is-a` patterns. |
| `course.json` | `learning_outcomes` ordered so the prerequisite rule has a signal. |
| `concept_graph.json` | Co-occurrence base graph (input to the `related-to` rule). |
| `expected_semantic_graph.json` | Golden output — edges the orchestrator must produce, sorted by `(type, source, target)`. |

## CI assertions

1. `build_semantic_graph(chunks, course, concept_graph)` produces an `edges`
   list whose `(source, target, type)` tuples exactly match
   `expected_semantic_graph.json`'s tuples.
2. The output validates against `schemas/knowledge/concept_graph_semantic.schema.json`.
3. Two back-to-back invocations produce byte-identical artifacts once
   `generated_at` is held fixed.

## What the fixture intentionally exercises

- **`is-a`**: chunk `c2` carries a key_term whose definition contains
  "is a type of accessibility-attribute" and both `aria-role` and
  `accessibility-attribute` are nodes in the co-occurrence graph.
- **`prerequisite`**: concept `wcag` first appears in a chunk tagged to
  `co-01` (position 0); concept `pour` first appears in a chunk tagged to
  `co-02` (position 1). Edge: `pour --prerequisite--> wcag`.
- **`related-to`**: concepts `wcag` and `accessibility-attribute` co-occur
  in 3 chunks (threshold met); `pour` and `aria-role` co-occur only once
  (threshold not met, no edge).
- **Precedence**: `aria-role` and `accessibility-attribute` have both an
  `is-a` claim and a `related-to` claim; the orchestrator keeps only the
  `is-a` edge.
