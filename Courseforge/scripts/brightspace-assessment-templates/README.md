# Brightspace Assessment Templates

IMSCC-compliant assessment XML generators and validators for Brightspace/D2L integration.

## Overview

This package provides strict XML templates, generators, and validators for creating Brightspace-compatible IMSCC packages. All components use the **correct namespaces verified from actual Brightspace exports**.

## Correct Namespaces (CRITICAL)

| Component | Correct Namespace | Resource Type |
|-----------|-------------------|---------------|
| Assignment | `http://www.imsglobal.org/xsd/imscc_extensions/assignment` | `assignment_xmlv1p0` |
| Discussion | `http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3` | `imsdt_xmlv1p3` |
| Quiz (QTI) | `http://www.imsglobal.org/xsd/ims_qtiasiv1p2` | `imsqti_xmlv1p2/imscc_xmlv1p3/assessment` |
| Manifest | `http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1` | N/A |

### Common Mistakes to Avoid

- **Assignment**: Do NOT use `d2l_2p0` or `desire2learn` namespaces
- **Discussion**: Root element is `<topic>`, NOT `<discussion>`
- **Quiz**: Use IMSCC 1.3 resource type, not 1.1

## Directory Structure

```
brightspace-assessment-templates/
├── generators/                     # Python XML generators
│   ├── __init__.py
│   ├── base_generator.py          # Base class, ID generation
│   ├── assignment_generator.py    # Assignment XML
│   ├── discussion_generator.py    # Discussion topic XML
│   ├── quiz_generator.py          # QTI assessment XML
│   └── manifest_generator.py      # Manifest XML
├── validators/                     # Python XML validators
│   ├── __init__.py
│   ├── xml_validator.py           # Base XSD validation
│   ├── assignment_validator.py    # Assignment validation
│   ├── discussion_validator.py    # Discussion validation
│   ├── qti_validator.py           # QTI assessment validation
│   └── manifest_validator.py      # Manifest validation
├── templates/                      # XML template files
│   ├── assignment_template.xml
│   ├── discussion_template.xml
│   ├── quiz_container_template.xml
│   └── question_types/            # QTI question templates
│       ├── multiple_choice.xml
│       ├── multiple_response.xml
│       ├── true_false.xml
│       ├── fill_in_blank.xml
│       └── essay.xml
├── tests/                          # Unit tests
│   ├── test_generators.py
│   └── test_validators.py
├── brightspace_assessment_generator.py  # Main generator class
└── README.md
```

## Usage

### Generate Assessments

```python
from brightspace_assessment_generator import BrightspaceAssessmentGenerator
from generators.quiz_generator import (
    create_multiple_choice_question,
    create_true_false_question,
    create_essay_question,
)

# Initialize generator with validation enabled
generator = BrightspaceAssessmentGenerator(validate_output=True)

# Generate assignment
assignment_xml = generator.generate_assignment(
    title="Week 1 Assignment",
    instructions="<p>Complete the following tasks...</p>",
    points=100.0,
    submission_types=['file', 'text']
)

# Generate discussion
discussion_xml = generator.generate_discussion(
    title="Week 1 Discussion",
    prompt="<p>Discuss the key concepts from this week...</p>"
)

# Generate quiz with multiple question types
questions = [
    create_multiple_choice_question(
        "<p>Which answer is correct?</p>",
        [
            {"text": "<p>Option A</p>", "is_correct": True},
            {"text": "<p>Option B</p>", "is_correct": False},
        ],
        points=2.0
    ),
    create_true_false_question(
        "<p>The sky is blue.</p>",
        correct_answer=True,
        points=1.0
    ),
    create_essay_question(
        "<p>Explain the concept in your own words.</p>",
        points=10.0
    ),
]

quiz_xml = generator.generate_quiz(
    title="Week 1 Quiz",
    questions=questions,
    max_attempts=2,
    time_limit=1800  # 30 minutes in seconds
)
```

### Validate XML

```python
from validators import (
    AssignmentValidator,
    DiscussionValidator,
    QTIValidator,
)

# Validate assignment
validator = AssignmentValidator()
result = validator.validate(assignment_xml)

if result.is_valid:
    print("Assignment XML is valid")
else:
    for error in result.errors:
        print(f"Error: {error}")
```

## QTI Question Types

The quiz generator supports all 5 standard QTI question types:

| Type | Profile | Description |
|------|---------|-------------|
| Multiple Choice | `cc.multiple_choice.v0p1` | Single answer selection |
| Multiple Response | `cc.multiple_response.v0p1` | Multiple answer selection |
| True/False | `cc.true_false.v0p1` | Binary choice |
| Fill in Blank | `cc.fib.v0p1` | Text entry response |
| Essay | `cc.essay.v0p1` | Free-form text (manual grading) |

## Brightspace ID Format

IDs follow the Brightspace convention: `i` prefix + 32-character lowercase hex:

```python
from generators import generate_brightspace_id

id = generate_brightspace_id()
# Example: i9c92b88bf2b64efa9cc8e6943b6028fb
```

## Running Tests

```bash
cd scripts/brightspace-assessment-templates
python3 -m unittest discover -s tests -v
```

## XML Output Examples

### Assignment (Correct Format)

```xml
<?xml version="1.0" encoding="utf-8"?>
<assignment xmlns="http://www.imsglobal.org/xsd/imscc_extensions/assignment"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="...">
  <title>Week 1 Assignment</title>
  <instructor_text texttype="text/html">&lt;p&gt;Instructions&lt;/p&gt;</instructor_text>
  <submission_formats>
    <format type="file" />
  </submission_formats>
  <gradable points_possible="100.000000000">true</gradable>
</assignment>
```

### Discussion (Correct Format)

```xml
<?xml version="1.0" encoding="utf-8"?>
<topic xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <title>Week 1 Discussion</title>
  <text texttype="text/html">&lt;p&gt;Discussion prompt&lt;/p&gt;</text>
</topic>
```

## Manifest Resource Types

When adding resources to `imsmanifest.xml`, use these exact types:

```xml
<!-- Assignment -->
<resource identifier="..." type="assignment_xmlv1p0" href="assignment.xml">
  <file href="assignment.xml" />
</resource>

<!-- Discussion -->
<resource identifier="..." type="imsdt_xmlv1p3" href="discussion.xml">
  <file href="discussion.xml" />
</resource>

<!-- Quiz -->
<resource identifier="..." type="imsqti_xmlv1p2/imscc_xmlv1p3/assessment" href="quiz.xml">
  <file href="quiz.xml" />
</resource>
```

## Schema Validation

XSD schemas are located in `/schemas/imscc/`:

- `cc_extresource_assignmentv1p0.xsd` - Assignment schema
- `ccv1p3_imsdt_v1p3.xsd` - Discussion topic schema
- `ccv1p3_qtiasiv1p2p1.xsd` - QTI assessment schema

## Related Documentation

- `/imscc-standards/brightspace-specific/` - Brightspace-specific IMSCC details
- `/schemas/imscc/README.md` - Schema documentation
