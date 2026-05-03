# Pedagogy Graph Builder Agent

## Purpose

Build a typed pedagogical / concept graph from chunked DART HTML output
during the Phase 6 `concept_extraction` workflow phase, BEFORE objective
synthesis runs. Wraps the deterministic
`Trainforge/pedagogy_graph_builder.py::build_pedagogy_graph` helper and
persists the result to the LibV2 course tree as
`concept_graph_semantic.json` so downstream phases (objective synthesis,
content generation, training synthesis) consume one canonical graph
instead of re-deriving it.

Per roadmap §6.6's two-stage recommendation, the concept graph is built
ONCE here and then read by `plan_course_structure` (objective synthesizer)
via `concept_graph_path`. The deterministic
`concept_objective_linker.py` pass populates
`LearningObjective.keyConcepts[]` from concept-graph slugs after objective
synthesis lands.

## Inputs

| Field | Source | Required |
|-------|--------|----------|
| `dart_chunks_path` | `chunking` phase output (Phase 7a `ed4all-chunker` package) — JSONL of v4 chunks emitted from staged DART HTML | yes |
| `course_id` | Workflow context (`course_code`, uppercased to match `manifest.json` convention) | yes |
| `concept_classes` | Optional mapping of concept slug -> class label sourced from `concept_graph.json` (Worker B's classifier, Wave 76) | no |

The phase deliberately reads ONLY `dart_chunks_path` — NOT
`objectives_path` — because objective synthesis runs AFTER this phase.
That ordering is what makes the two-stage decoupling possible per
roadmap §6.6.

## Outputs

| Field | Destination | Shape |
|-------|-------------|-------|
| `concept_graph_path` | `LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json` | Typed graph: 8 edge types (3 taxonomic + 5 pedagogical), Bloom level + difficulty level nodes, concept nodes with `class` field, edges with `relation_type` field |
| `concept_graph_sha256` | `LibV2/courses/<slug>/manifest.json::concept_graph_sha256` | SHA-256 over canonicalised graph JSON. Optional in Phase 6; promoted to required in Phase 7c |

The graph shape is the canonical `build_pedagogy_graph` output documented
in `Trainforge/CLAUDE.md` § "Schemas and concept graph": 8 edge types
(`is-a`, `prerequisite`, `related-to`, `assesses`, `exemplifies`,
`misconception-of`, `derived-from-objective`, `defined-by`); each edge
carries `relation_type`; each concept node carries `class` field
(`DomainConcept` / `Misconception` / `PedagogicalMarker` /
`AssessmentOption` / `LowSignal` / `InstructionalArtifact`).

When `TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE=true`, edges additionally
carry `provenance: {chunk_ids: List[str], rule_version: str}` per the
roadmap §6.1 recommendation.

## Workflow

1. Receive `dart_chunks_path` from the `chunking` phase output. Load the
   JSONL into `List[Dict]` chunks.
2. Resolve `course_id` from workflow context. The builder's Wave 82
   fallback derives it from chunk-ID prefix when caller passes empty,
   but we pass it explicitly so the value is auditable.
3. Call
   `Trainforge.pedagogy_graph_builder.build_pedagogy_graph(chunks, course_id=course_id, concept_classes=...)`.
   No `objectives` argument — that surface stays empty in this phase
   (objectives don't exist yet at this point in the chain).
4. Persist the returned graph dict to
   `LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json` via
   atomic tmp + rename.
5. Compute SHA-256 over the canonicalised JSON. Write the hex digest
   into `manifest.json::concept_graph_sha256` (optional field added by
   Subtask 17) and route it through the phase output bus as
   `concept_graph_sha256` so downstream phases can pin against it.

## Decision Capture

One `decision_type="concept_graph_built"` event per phase invocation,
emitted under `phase=trainforge-concept-extraction`. The rationale
interpolates dynamic signals so captures are replayable post-hoc:

- Number of input chunks consumed
- Number of concept nodes emitted (per class)
- Per-edge-type counts (the 8 canonical edge types)
- Whether `concept_classes` filter was supplied
- `concept_graph_sha256` digest

Static boilerplate rationales are forbidden — every replay should be
distinguishable from every other.

## Validation

The phase is gated by the new `concept_graph` validation gate
(severity `warning` in Phase 6, promoted later) wrapping
`lib/validators/concept_graph.py::ConceptGraphValidator` (lands in
Subtask 14). Floors:

- ≥10 concept nodes
- ≥5 edge types present (taxonomic + pedagogical mix)
- Every node carries `class` field
- Every edge carries `relation_type` field
- When `TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE=true`, every edge
  carries `provenance.chunk_ids` + `provenance.rule_version`

## Integration

Works upstream with:

- `chunking` phase (Phase 7a `ed4all-chunker` package) — receives
  chunks JSONL via `dart_chunks_path`.

Works downstream with:

- `course_planning` (objective synthesizer) — reads
  `concept_graph_path` and produces objectives whose
  `keyConcepts[]` get populated by the deterministic
  `concept_objective_linker.py` pass between objective synthesis and
  content generation.
- `Trainforge/process_course.py` — Subtask 13 refactors the
  `build_pedagogy_graph` call site there to load
  `concept_graph_semantic.json` from this phase's output instead of
  re-building. Legacy corpora without a Phase 6 concept-extraction
  output keep working via the existing fallback path.
- `lib/validators/libv2_manifest.py::LibV2ManifestValidator` —
  Subtask 19 extends the manifest validator to read
  `concept_graph_sha256` (advisory in Phase 6, critical in Phase 7c)
  per `schemas/library/course_manifest.schema.json`.

## Why a separate agent (and not just inline in `process_course.py`)?

Two reasons:

1. **Pipeline-stage decoupling.** The concept graph is consumed by
   THREE downstream surfaces (objective synthesis, content generation,
   training synthesis). Building it once at a known phase boundary,
   with a stable on-disk artifact + sha256 hash, is what lets each
   downstream consumer reproduce its run by pinning against a single
   artifact.
2. **Deterministic + cheap.** `build_pedagogy_graph` is pure Python —
   no LLM call. The agent is a thin wrapper around a deterministic
   helper, dispatched through the same phase machinery as the
   LLM-driven phases so the workflow runner has uniform observability
   (one phase output bus, one decision capture surface, one validation
   gate).
