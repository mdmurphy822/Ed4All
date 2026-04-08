# Validator Agent

## Purpose

Validate assessment quality and provide feedback for revision if needed.

## Input

- Generated assessment from assessment-generator
- Course learning objectives
- Quality thresholds

## Output

```json
{
  "validation_id": "VAL-PYTHON_101-20260110",
  "passed": true,
  "scores": {
    "objective_coverage": 1.0,
    "bloom_alignment": 1.0,
    "question_quality": 0.75,
    "distractor_quality": 0.80,
    "overall": 0.92
  },
  "feedback": [
    {
      "question_id": "Q005",
      "issue": "stem_clarity",
      "severity": "warning",
      "message": "Question stem could be more specific",
      "suggestion": "Add context about the specific scenario"
    }
  ],
  "revision_required": false,
  "output_path": "/training-captures/trainforge/PYTHON_101/..."
}
```

## Validation Criteria

### Objective Coverage (weight: 0.30)
- Each LO has at least 1 question
- Score = LOs_covered / total_LOs

### Bloom Alignment (weight: 0.25)
- Each question targets stated Bloom level
- Verb patterns match level
- Score = aligned_questions / total_questions

### Question Quality (weight: 0.25)
- Stem clarity (no ambiguity)
- Answer definitiveness (one clearly correct)
- Content grounding (supported by corpus)
- Score = quality_checks_passed / total_checks

### Distractor Quality (weight: 0.20)
- Each distractor targets misconception
- Plausible but incorrect
- No "trick" answers
- Score = quality_distractors / total_distractors

## Feedback Categories

| Category | Severity | Action |
|----------|----------|--------|
| `stem_clarity` | warning | Suggest revision |
| `bloom_mismatch` | error | Require revision |
| `missing_rationale` | error | Require revision |
| `weak_distractor` | warning | Suggest improvement |
| `coverage_gap` | error | Add questions |

## Revision Loop

```
Generator → Validator → [PASS] → Output
                     → [FAIL] → Generator (with feedback)
                              → max 3 iterations
                              → [STILL FAIL] → Manual review
```

## Decision Capture

```python
capture.log_decision(
    decision_type="validation_result",
    decision="Assessment passed with 0.92 overall score",
    rationale="All critical criteria met, 2 minor warnings logged"
)
```

## Thresholds

| Metric | Pass | Warn | Fail |
|--------|------|------|------|
| Objective Coverage | >= 0.90 | >= 0.80 | < 0.80 |
| Bloom Alignment | = 1.0 | >= 0.95 | < 0.95 |
| Question Quality | >= 0.75 | >= 0.60 | < 0.60 |
| Overall | >= 0.90 | >= 0.80 | < 0.80 |
