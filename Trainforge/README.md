# Trainforge

Assessment-Based RAG Training for IMSCC Packages

## Overview

Trainforge generates comprehensive training data for Claude by:
1. Analyzing IMSCC course packages from Courseforge
2. Querying LibV2 RAG corpus for relevant content
3. Generating Bloom's taxonomy-aligned assessments
4. Capturing all decisions for model fine-tuning

## Quick Start

```python
from Trainforge.parsers.imscc_parser import IMSCCParser
from Trainforge.generators.assessment_generator import AssessmentGenerator

# Parse IMSCC package
parser = IMSCCParser()
package = parser.parse("/path/to/course.imscc")

# Generate assessments via MCP tools or direct API
```

## Directory Structure

```
Trainforge/
├── CLAUDE.md                    # Agent instructions
├── README.md                    # This file
├── parsers/                     # Content extraction
│   ├── imscc_parser.py          # IMSCC package parsing
│   ├── qti_parser.py           # QTI assessment parsing
│   └── html_content_parser.py  # HTML content extraction
├── rag/                         # RAG integration
│   └── libv2_bridge.py         # LibV2 retrieval interface
├── generators/                  # Assessment generation
│   ├── assessment_generator.py  # Main orchestrator
│   └── question_factory.py     # Question type factory
├── decision_capture/            # Decision capture integration
│   └── decision_logger.py      # Central capture logger
├── validation/                  # Quality validation
├── agents/                      # Agent specifications
│   ├── content-analyzer.md     # Content analysis agent
│   ├── assessment-generator.md # Question generation agent
│   └── validator.md            # Quality validation agent
├── examples/                    # Sample outputs
│   └── sample_assessment.json  # Example assessment
├── output/                      # Generated output
└── tests/                       # Test suite
    └── test_parsers.py         # Parser smoke tests
```

## Pipeline Workflow

```
IMSCC Package (from Courseforge)
        │
        ▼
┌───────────────────┐
│ Content Analyzer  │ ──► Learning objectives, concepts, structure
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ LibV2 RAG Query   │ ──► Relevant chunks from corpus
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Assessment Gen    │ ──► Questions with Bloom's alignment
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Validator         │ ──► Quality scores, feedback
└───────────────────┘
        │
        ▼
Training Capture (JSONL) + Assessment (JSON)
```

## LibV2 Integration

Trainforge uses LibV2 for RAG retrieval:

```python
from libv2 import MultiRetriever

retriever = MultiRetriever(course_slug="python-101")
chunks = retriever.retrieve(
    query="exception handling best practices",
    top_k=10
)
```

See `Trainforge/rag/libv2_bridge.py` for the retrieval interface.

## Assessment Quality Standards

Every generated question must meet:

| Criterion | Requirement |
|-----------|-------------|
| Bloom's Level | Explicitly aligned to 1 of 6 levels |
| Learning Objective | Mapped to specific LO |
| Distractor Quality | Each incorrect answer targets a misconception |
| Stem Clarity | Unambiguous, complete question |
| Content Grounding | Supported by RAG-retrieved content |

See `Trainforge/CLAUDE.md` for Bloom's level targeting details.

## Decision Capture

All generation decisions are logged:

```python
from lib.trainforge_capture import TrainforgeDecisionCapture

with TrainforgeDecisionCapture(
    course_code="PYTHON_101",
    phase="question-generation"
) as capture:
    capture.log_question_generation(
        question_id="Q001",
        bloom_level="apply",
        learning_objective="LO-3.2",
        rationale="Tests practical application of try-except blocks"
    )
```

## Output Format

```json
{
  "assessment_id": "ASM-PYTHON_101-20260110",
  "course_code": "PYTHON_101",
  "questions": [
    {
      "id": "Q001",
      "stem": "Which statement correctly handles...",
      "bloom_level": "apply",
      "learning_objective": "LO-3.2",
      "options": [...],
      "correct_answer": "B",
      "distractor_rationale": {...}
    }
  ],
  "validation": {
    "passed": true,
    "scores": {
      "bloom_alignment": 1.0,
      "objective_coverage": 0.95
    }
  }
}
```

## Dependencies

```
# LibV2 for RAG
libv2 (internal)

# Decision capture
lib.trainforge_capture
lib.streaming_capture

# Core libraries
beautifulsoup4>=4.9.0
lxml>=4.6.0
```
