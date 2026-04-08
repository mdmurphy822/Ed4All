# Course Content Parser

**Version**: v1.0.0  
**Created**: 2025-08-05  
**Purpose**: Extract and structure content from markdown course materials for IMSCC generation

## Overview

The Course Content Parser extracts academic content from structured markdown course files and creates a standardized JSON output containing all necessary components for IMSCC package generation. This script parses sub-modules per week with dynamic counts based on content complexity while maintaining content accuracy and quality.

## Features

- **Atomic Operations**: Single execution with complete success or failure
- **Content Validation**: Ensures substantial academic content in each sub-module
- **Template Resolution**: Removes hardcoded references and placeholder text
- **Quality Assurance**: Validates minimum word counts and content requirements
- **Error Handling**: Comprehensive validation with detailed error reporting

## Input Requirements

### Expected Directory Structure
```
[course_folder]/
├── course_info.md          # Course metadata and description
├── syllabus.md            # Complete course syllabus
├── assessment_guide.md    # Assignment details and rubrics
├── modules/               # Weekly course content
│   ├── week_01.md
│   ├── week_02.md
│   ├── week_03.md
│   └── week_04.md
└── settings.json          # Course configuration
```

### Required Content Structure per Week
Each week_XX.md file must contain 7 distinct sub-modules:
1. **Module Overview** - Week introduction and learning objectives
2. **Concept Summary Pages** (2-3) - Text-based concept explanations (300-800 words each)
3. **Key Concepts Accordion** - Interactive key terms with definitions (5-10 concepts)
4. **Visual/Graphical/Math Display** - Charts, diagrams, equations
5. **Examples of Learning Concepts in Application** - Theory-to-practice demonstrations
6. **Real World Application Examples** - Industry connections and current examples
7. **Study Questions for Learning Reflection** - Knowledge testing and critical thinking

## Output Format

### Structured JSON Schema
```json
{
  "course_info": {
    "title": "Course Title",
    "description": "Course description",
    "learning_objectives": ["objective1", "objective2"],
    "credits": 3,
    "duration_weeks": 4
  },
  "syllabus": {
    "policies": "Course policies text",
    "schedule": "Course schedule",
    "requirements": "Technical requirements"
  },
  "weeks": [
    {
      "week_number": 1,
      "title": "Week Title",
      "sub_modules": [
        {
          "type": "overview",
          "title": "Module Overview",
          "content": "Substantial content text",
          "learning_objectives": ["obj1", "obj2"],
          "word_count": 250
        },
        {
          "type": "concept_summary",
          "title": "Concept Summary 1",
          "content": "Academic content 300-800 words",
          "key_concepts": ["concept1", "concept2"],
          "word_count": 450
        }
      ]
    }
  ],
  "assessments": [
    {
      "week": 1,
      "type": "assignment",
      "title": "Assignment Title",
      "description": "Full assignment instructions",
      "word_limit": "700-1000 words",
      "points": 100,
      "rubric": "Detailed rubric criteria"
    }
  ]
}
```

## Usage

### Command Line Usage
```bash
python course_content_parser.py --input /path/to/course/folder --output structured_course.json
```

### Python API Usage
```python
from course_content_parser import CourseContentParser

parser = CourseContentParser()
result = parser.parse_course_folder('/path/to/course/folder')
parser.save_structured_content(result, 'output.json')
```

## Validation Requirements

### Content Quality Standards
- **Minimum Word Counts**: 
  - Module Overview: 200+ words
  - Concept Summaries: 300-800 words each
  - Key Concept Definitions: 50-200 words each
  - Application Examples: 250+ words
  - Study Questions: 150+ words
- **Academic Rigor**: Substantial educational content, not placeholder text
- **Template Resolution**: Zero unresolved `{variable}` patterns
- **Reference Validation**: No hardcoded textbook or resource references

### Structural Requirements
- **Exactly 7 Sub-modules per Week**: Parser must identify and extract all required types
- **Content Accuracy**: Actual markdown content transferred, not just structure
- **Learning Objectives**: Clear, measurable objectives for each module
- **Assessment Integration**: One writing assignment per week (700-1000 words)

## Error Handling

### Critical Errors (System Exit)
- Missing required course files (course_info.md, week files)
- Insufficient content in any sub-module (below minimum word counts)
- Unresolved template variables in content
- Structural parsing failures

### Validation Warnings
- Content quality concerns
- Missing optional elements
- Formatting inconsistencies

## Dependencies

### Required Packages
```
python >= 3.8
markdown >= 3.3.0
json >= 2.0.9
re >= 2.2.1
pathlib >= 1.0.1
```

### Installation
```bash
pip install markdown pathlib
```

## Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Test Coverage
- Content extraction accuracy
- Error handling scenarios
- Edge cases and boundary conditions
- Performance with large course files

## Configuration

### Config File: `config/parser_config.json`
```json
{
  "min_word_counts": {
    "overview": 200,
    "concept_summary": 300,
    "key_concept": 50,
    "application_example": 250,
    "study_questions": 150
  },
  "required_sub_modules": 7,
  "max_weeks": 16,
  "validation_strict": true
}
```

## Changelog

### v1.0.0 (2025-08-05)
- Initial implementation
- Core parsing functionality
- Content validation system
- Error handling framework
- Documentation and examples

## Contributing

- Follow atomic operation principles
- Maintain comprehensive error handling
- Add unit tests for new functionality
- Update documentation for all changes
- Validate against real course materials

## License

MIT License - See LICENSE file in project root.