# Assessment Generator Agent

## Purpose

Generate Bloom's taxonomy-aligned assessment questions from analyzed course content using RAG-retrieved context.

## Input

- Content analysis from content-analyzer
- LibV2 RAG retrieval results
- Course learning objectives with Bloom levels
- Recommended question distribution

## Output

```json
{
  "assessment_id": "ASM-PYTHON_101-20260110",
  "course_code": "PYTHON_101",
  "generated_at": "2026-01-10T15:30:00Z",
  "questions": [
    {
      "id": "Q001",
      "type": "multiple_choice",
      "bloom_level": "apply",
      "learning_objective": "LO-3.2",
      "stem": "Given the following code snippet, which exception handler correctly catches a file not found error?",
      "options": [
        {"id": "A", "text": "except FileNotFoundError:", "is_correct": true},
        {"id": "B", "text": "except FileError:", "is_correct": false},
        {"id": "C", "text": "except IOError:", "is_correct": false},
        {"id": "D", "text": "except OpenError:", "is_correct": false}
      ],
      "correct_answer": "A",
      "distractor_rationale": {
        "B": "Common misconception - students invent plausible-sounding exception names",
        "C": "Partial knowledge - IOError is parent class but less specific",
        "D": "Common misconception - confuses operation name with exception"
      },
      "source_chunks": ["chunk_142", "chunk_143", "chunk_156"],
      "generation_rationale": "Tests practical application of exception handling syntax, grounded in module 3 content"
    }
  ],
  "rag_metrics": {
    "total_chunks_retrieved": 47,
    "chunks_used": 23,
    "average_relevance_score": 0.82
  },
  "bloom_distribution": {
    "remember": 3,
    "understand": 5,
    "apply": 7,
    "analyze": 3,
    "evaluate": 1,
    "create": 1
  }
}
```

## Workflow

1. **Receive Analysis** - Accept content analysis and LO mapping
2. **Query LibV2** - Retrieve relevant chunks for each LO
3. **Select Question Type** - Based on Bloom level and content type
4. **Generate Stem** - Create clear, unambiguous question
5. **Generate Distractors** - Target specific misconceptions
6. **Capture Decisions** - Log all generation choices

## Question Type Selection

| Bloom Level | Primary Type | Alternative |
|-------------|--------------|-------------|
| Remember | Multiple Choice | True/False |
| Understand | Multiple Choice | Short Answer |
| Apply | Multiple Choice | Code Completion |
| Analyze | Multiple Choice | Matching |
| Evaluate | Short Answer | Essay |
| Create | Essay | Project-based |

## Bloom's Verb Patterns

### Remember (Knowledge)
- Define, List, Recall, Identify, Name, State
- Pattern: "What is...?", "Which of the following...?", "List the..."

### Understand (Comprehension)
- Explain, Describe, Summarize, Interpret, Classify
- Pattern: "Explain why...", "What does X mean?", "Describe how..."

### Apply (Application)
- Apply, Demonstrate, Use, Implement, Solve
- Pattern: "How would you use...?", "Apply X to...", "Given this scenario..."

### Analyze (Analysis)
- Analyze, Compare, Contrast, Differentiate, Examine
- Pattern: "Compare and contrast...", "What are the differences...?", "Analyze the relationship..."

### Evaluate (Evaluation)
- Evaluate, Judge, Justify, Critique, Assess
- Pattern: "Evaluate the effectiveness...", "Which approach is best...?", "Justify your choice..."

### Create (Synthesis)
- Create, Design, Develop, Construct, Formulate
- Pattern: "Design a...", "Develop a plan for...", "Create a solution..."

## Distractor Generation Guidelines

### Quality Requirements

1. **Plausibility** - Must seem correct to someone with the misconception
2. **Distinctiveness** - Clearly different from correct answer
3. **Misconception Targeting** - Each distractor targets specific error
4. **No Tricks** - Avoid "all of the above" or "none of the above"

### Common Misconception Categories

| Category | Example |
|----------|---------|
| Syntax confusion | Mixing similar operators |
| Terminology mix-up | Confusing related terms |
| Partial application | Correct concept, wrong context |
| Overgeneralization | Extending rule beyond scope |
| Order errors | Wrong sequence of operations |

## Decision Capture

```python
capture.log_question_generation(
    question_id="Q001",
    bloom_level="apply",
    learning_objective="LO-3.2",
    question_type="multiple_choice",
    source_chunks=["chunk_142", "chunk_143"],
    generation_rationale="Selected MCQ to assess practical exception handling"
)

capture.log_distractor_rationale(
    question_id="Q001",
    distractors={
        "B": {"text": "except FileError:", "misconception": "invented_exception_name"},
        "C": {"text": "except IOError:", "misconception": "parent_class_confusion"},
        "D": {"text": "except OpenError:", "misconception": "operation_name_confusion"}
    }
)
```

## RAG Integration

### Chunk Retrieval

```python
from libv2 import MultiRetriever

retriever = MultiRetriever(course_slug="python-101")
chunks = retriever.retrieve(
    query=learning_objective.text,
    top_k=15,
    min_score=0.7
)
```

### Relevance Filtering

- Minimum relevance score: 0.7
- Maximum chunks per question: 5
- Prefer chunks with code examples for Apply/Analyze levels

## Quality Thresholds

| Metric | Minimum |
|--------|---------|
| Stem word count | 10 |
| Options per question | 4 |
| Distractor rationale length | 20 chars |
| Source chunks per question | 1 |

## Error Handling

| Error | Action |
|-------|--------|
| Insufficient chunks | Lower retrieval threshold, log warning |
| No matching LO | Skip question, report gap |
| Duplicate stem | Regenerate with different focus |
| Weak distractors | Request regeneration from validator feedback |
