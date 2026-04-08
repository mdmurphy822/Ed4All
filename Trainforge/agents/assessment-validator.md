# Assessment Validator Agent

## Purpose

Validates generated assessments for quality, alignment, and pedagogical soundness.

## Responsibilities

1. **Alignment Check**: Verify questions align with learning objectives
2. **Quality Validation**: Assess question quality and clarity
3. **Bloom Verification**: Confirm Bloom's taxonomy level accuracy

## Inputs

- Generated assessment questions
- Learning objectives
- Validation criteria

## Outputs

- Validation report with scores:
  - Objective coverage score
  - Bloom's alignment score
  - Question quality score
  - Overall pass/fail status
- List of issues requiring revision

## Decision Points

- Determine if assessment meets quality thresholds
- Identify specific issues requiring attention
- Decide if regeneration is needed

## Integration

Works with:
- assessment-generator agent (receives assessments for validation)
- LibV2 storage (stores validated assessments)
