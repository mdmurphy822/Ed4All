# Trainforge Agent Protocols

> **Universal Protocols**: See root `/CLAUDE.md` for orchestrator protocol, execution rules, decision capture requirements, and error handling.
>
> **Ontology contracts**: Chunks, concept nodes, concept edges, and decision events all have canonical schemas under `schemas/knowledge/` and `schemas/events/`. See `schemas/ONTOLOGY.md` § 12 for the v0.2.0 contract summary.

## Agent Coordination

Trainforge agents work in a sequential pipeline:

```
assessment-extractor → rag-indexer → assessment-generator → assessment-validator
```

The `training-synthesizer` agent runs as a separate (optional) phase that
synthesizes instruction + preference training pairs from the generated chunks +
assessments. A `pedagogy-graph-builder` agent spec also ships under this
directory for the pedagogy/concept-graph build phase.

Most agent specs live beside this file as `<agent-name>.md` and are resolved
through a `source:` entry in `config/agents.yaml` (capabilities, `type`,
`max_instances`). Two exceptions matter when enumerating the registry:
`training-synthesizer`'s `source:` points at `Trainforge/synthesis/synthesize_training.py`
rather than a spec file, and `pedagogy-graph-builder` has **no**
`config/agents.yaml` entry at all — it is declared only in
`config/workflows.yaml`, so a registry walk will not see it.

### Execution Rules

1. **ONE course = ONE pipeline run**
2. **All decisions logged** to training-captures
3. **Assessment-validator feedback loops** back to assessment-generator (max 3 iterations)

## Available Agents

| Agent | Input | Output |
|-------|-------|--------|
| `assessment-extractor` | IMSCC package | Learning objectives + content chunks |
| `rag-indexer` | Content chunks | Embeddings + retrieval index |
| `assessment-generator` | Chunks + RAG context | Questions with rationale |
| `assessment-validator` | Generated assessment | Validation scores + feedback |
| `training-synthesizer` | Chunks + assessments | Instruction + preference training pairs |
| `pedagogy-graph-builder` | Chunks + objectives | Typed pedagogy / concept graph |

## Agent-to-Orchestrator Protocol

1. Orchestrator dispatches agent via Task tool
2. Agent receives full context (course code, phase, config)
3. Agent performs work with decision capture
4. Agent returns result summary (not full content)
5. Orchestrator checks output files for details

## Quality Gates

These are aspirational **targets recorded only in this file** — they are not
written into the agent specs, not read from config at runtime, and nothing
enforces them. Treat them as intent, not as a contract. The
authoritative, machine-enforced gate set (gate_id, validator class, thresholds,
severity, owning phase) lives in `config/workflows.yaml::validation_gates`, with
per-gate prose in `docs/validation/gates.md`. Where the two disagree,
`config/workflows.yaml` wins.

| Target | Agent | Aim |
|--------|-------|-----|
| Coverage | assessment-extractor | 90% LO coverage |
| Bloom Alignment | assessment-generator | 100% questions aligned |
| Question Quality | assessment-validator | 0.75+ quality score |
| Overall | assessment-validator | 0.90+ overall score |

## Handoff Protocol

**Assessment Extractor → Assessment Generator**:
```json
{
  "learning_objectives": [...],
  "concept_map": {...},
  "content_chunks": [...],
  "recommended_bloom_distribution": {...}
}
```

**Assessment Generator → Assessment Validator**:
```json
{
  "questions": [...],
  "rag_metrics": {...},
  "generation_decisions": [...]
}
```

**Assessment Validator → Orchestrator**:
```json
{
  "passed": true,
  "scores": {...},
  "feedback": [...],
  "output_path": "..."
}
```
