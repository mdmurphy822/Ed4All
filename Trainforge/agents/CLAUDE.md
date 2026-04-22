# Trainforge Agent Protocols

> **Universal Protocols**: See root `/CLAUDE.md` for orchestrator protocol, execution rules, decision capture requirements, and error handling.
>
> **Ontology contracts**: Chunks, concept nodes, concept edges, and decision events all have canonical schemas under `schemas/knowledge/` and `schemas/events/`. See `schemas/ONTOLOGY.md` § 12 for the v0.2.0 contract summary.

## Agent Coordination

Trainforge agents work in a sequential pipeline:

```
content-analyzer → assessment-generator → validator
```

### Execution Rules

1. **ONE course = ONE pipeline run**
2. **All decisions logged** to training-captures
3. **Validator feedback loops** back to generator (max 3 iterations)

## Available Agents

| Agent | Input | Output |
|-------|-------|--------|
| `content-analyzer` | IMSCC manifest, LibV2 corpus | Content analysis JSON |
| `assessment-generator` | Analysis + RAG chunks | Questions with rationale |
| `validator` | Generated assessment | Validation scores + feedback |

## Agent-to-Orchestrator Protocol

1. Orchestrator dispatches agent via Task tool
2. Agent receives full context (course code, phase, config)
3. Agent performs work with decision capture
4. Agent returns result summary (not full content)
5. Orchestrator checks output files for details

## Quality Gates

| Gate | Agent | Threshold |
|------|-------|-----------|
| Coverage | content-analyzer | 90% LO coverage required |
| Bloom Alignment | assessment-generator | 100% questions aligned |
| Question Quality | validator | 0.75+ quality score |
| Overall | validator | 0.90+ overall score |

## Handoff Protocol

**Content Analyzer → Assessment Generator**:
```json
{
  "learning_objectives": [...],
  "concept_map": {...},
  "content_chunks": [...],
  "recommended_bloom_distribution": {...}
}
```

**Assessment Generator → Validator**:
```json
{
  "questions": [...],
  "rag_metrics": {...},
  "generation_decisions": [...]
}
```

**Validator → Orchestrator**:
```json
{
  "passed": true,
  "scores": {...},
  "feedback": [...],
  "output_path": "..."
}
```
