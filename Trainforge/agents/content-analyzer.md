# Content Analyzer Agent

## Purpose

Analyze IMSCC course content to identify assessment opportunities and prepare context for question generation.

## Input

- IMSCC package path or course code
- LibV2 corpus reference
- Course learning objectives

## Output

```json
{
  "course_code": "PYTHON_101",
  "analysis_timestamp": "2026-01-10T14:00:00Z",
  "learning_objectives": [
    {
      "id": "LO-1.1",
      "text": "Explain the purpose of variables",
      "bloom_level": "understand",
      "concepts": ["variables", "data types", "assignment"]
    }
  ],
  "concept_map": {
    "variables": {
      "related_concepts": ["data types", "scope", "naming conventions"],
      "importance": 0.95,
      "corpus_coverage": 142
    }
  },
  "content_summary": {
    "total_modules": 6,
    "total_chunks": 283,
    "content_types": ["lecture", "example", "exercise"]
  },
  "recommended_bloom_distribution": {
    "remember": 0.15,
    "understand": 0.25,
    "apply": 0.35,
    "analyze": 0.15,
    "evaluate": 0.05,
    "create": 0.05
  }
}
```

## Workflow

1. **Parse IMSCC** - Extract manifest and content structure
2. **Extract Learning Objectives** - Identify explicit LOs from content
3. **Build Concept Map** - Identify key concepts and relationships
4. **Query LibV2** - Get chunk counts and coverage metrics
5. **Recommend Bloom Distribution** - Based on content depth and LO verbs

## Decision Capture

Log all analysis decisions:

```python
capture.log_decision(
    decision_type="concept_identification",
    decision="Identified 'exception handling' as key concept",
    rationale="Appears in 3 modules, 47 chunks, multiple code examples"
)
```

## Quality Criteria

- 90%+ learning objective identification
- Concept map covers all major topics
- LibV2 coverage confirmed for each concept
