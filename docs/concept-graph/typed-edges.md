# Typed-edge concept graph

This document explains `graph/concept_graph_semantic.json` — the typed-edge
companion to `graph/concept_graph.json` emitted by every Trainforge course
processing run.

`concept_graph.json` captures which concept tags co-occur inside chunks.
It is useful for dense retrieval and for surfacing clusters, but it says
nothing about *why* two concepts are connected. The semantic graph layers
relation types on top: `is-a`, `prerequisite`, `related-to`.

Files you care about:

- `Trainforge/rag/typed_edge_inference.py` — orchestrator.
- `Trainforge/rag/inference_rules/` — one module per rule.
- `schemas/concept_graph_semantic.schema.json` — the wire format.
- `Trainforge/tests/fixtures/mini_course_typed_graph/` — smoke fixture.

## The three rules

| Rule | Edge type | Signal | Default confidence |
|---|---|---|---|
| `is_a_from_key_terms` | `is-a` | `key_terms[].definition` contains "is a type of X" / "is a form of X" / "refers to an X" | 0.8 |
| `prerequisite_from_lo_order` | `prerequisite` | Concept A first appears in a chunk tagged to an earlier learning-outcome position than concept B's first chunk. Both concepts must share at least one chunk for the pair to be considered. | 0.6 |
| `related_from_cooccurrence` | `related-to` | Concept pair co-occurs in `>= threshold` chunks (default 3). Reuses the `weight` field from `concept_graph.json` rather than recomputing. | 0.4 + 0.05·weight |

Every rule module exposes a pure `infer(chunks, course, concept_graph,
**kwargs)` function and three constants (`RULE_NAME`, `RULE_VERSION`,
`EDGE_TYPE`). Adding a fourth rule is a drop-in module plus an entry in
the orchestrator's rule list.

## Per-edge provenance

Every edge carries:

```json
{
  "source": "aria-role",
  "target": "accessibility-attribute",
  "type": "is-a",
  "confidence": 0.8,
  "provenance": {
    "rule": "is_a_from_key_terms",
    "rule_version": 1,
    "evidence": {
      "chunk_id": "mini_chunk_00042",
      "term": "aria-role",
      "definition_excerpt": "An ARIA role is a type of accessibility-attribute...",
      "pattern": "..."
    }
  }
}
```

`provenance.rule` + `provenance.rule_version` let a consumer cheaply filter
by generating rule or re-derive an older edge set from the chunks. The
`evidence` block is free-form per rule but must be JSON-serializable.

## Precedence

Two rules can fire on the same `(source, target)` pair. The orchestrator
resolves collisions deterministically:

```
is-a   >   prerequisite   >   related-to
```

The lower-precedence edge is dropped; the kept edge's provenance is
unchanged. `related-to` is treated as undirected for collision purposes,
so a directed `prerequisite` edge between `X` and `Y` suppresses an
undirected `related-to` between the same nodes.

Rationale: `is-a` is the strongest claim — it declares a taxonomic
subclass relationship. `prerequisite` is a dependency claim grounded in
curricular ordering. `related-to` is the weakest — mere co-occurrence.
When we have evidence for the stronger claim, shadowing the weaker one
keeps downstream consumers (Worker G's cross-package index, the RAG
retrieval layer) from double-counting the same pair.

## Optional LLM escalation

The orchestrator accepts an `llm_enabled=True` flag plus an `llm_callable`.
When both are supplied, the callable is invoked with the rule-based edge
list and may propose additional edges. Every proposed edge:

1. Must reference two nodes that are already in the co-occurrence graph
   (so the LLM cannot invent concepts).
2. Has its `provenance.rule` forced to `"llm_typed_edge"` regardless of
   what the callable returned.
3. Triggers a `typed_edge_inference` decision-capture log entry, matching
   the Ed4All decision-capture contract (`CLAUDE.md`: "ALL Claude
   decisions MUST be logged").

The LLM path is **off by default**. The default runtime is byte-identical
across repeated invocations for the same `(chunks, course, concept_graph,
now)` tuple. Turn it on only when:

- You have a high-stakes package where the rule-based output is
  materially thin (typical symptom: most `key_terms[].definition` strings
  are short noun phrases, so `is_a_from_key_terms` fires rarely).
- You are actively grading the LLM's output for downstream training.

The flag is exposed on the CLI as `--typed-edges-llm`; the in-process API
surface is
`CourseProcessor(... typed_edges_llm=True)`.

## Known limits

- Prerequisite inference relies on the `learning_outcomes` ordering in
  `course.json`. A course whose outcomes are listed in thematic rather
  than instructional order will produce misleading prerequisite edges.
- `is_a_from_key_terms` only fires when the parent term is *already* a
  node. If a definition names a parent that never appeared as a concept
  tag elsewhere, no edge is produced — we would rather have silence than
  a dangling edge.
- The default `related_to` threshold of 3 is deliberately conservative.
  Small packages (<10 chunks) may produce zero `related-to` edges. Drop
  the threshold to 2 by wiring a `related_threshold` kwarg through if
  that matches your ingestion scale.

## Roadmap

- Worker G (`worker-g/cross-package-index`) consumes this artifact to
  build LibV2's cross-package concept index.
- A future rule will use `misconceptions[]` to emit `confuses-with`
  edges; the precedence policy has room for another directed type above
  `related-to`.
