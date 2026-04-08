# Assessment XML Generator

**Version**: v1.0.0  
**Created**: 2025-08-05  
**Purpose**: Generate native Brightspace assessment XML files (QTI, D2L assignments, discussions)

## Overview

The Assessment XML Generator creates properly formatted XML files for native Brightspace assessment tools including assignments, quizzes, and discussions. This script ensures full compatibility with Brightspace import processes and generates functional assessment objects with proper gradebook integration.

## Features

- **Native Brightspace Integration**: Creates XML that imports as functional Brightspace tools
- **QTI 1.2 Compliance**: Standards-compliant quiz XML for seamless integration
- **D2L Assignment XML**: Proper dropbox configuration with rubrics and grading
- **Discussion Forum XML**: Threaded discussions with grading parameters
- **Gradebook Integration**: Automatic point values, weighting, and scoring configuration
- **Atomic Operations**: Single execution with complete validation

## Input Requirements

### Structured Assessment Data
Expected input format from Course Content Parser:
```json
{
  "assessments": [
    {
      "week": 1,
      "type": "assignment",
      "title": "Linear Systems Analysis",
      "description": "Complete assignment description with instructions",
      "word_limit": "700-1000 words",
      "points": 100,
      "rubric": "Detailed rubric criteria"
    },
    {
      "week": 1,
      "type": "quiz",
      "title": "Week 1 Knowledge Check",
      "questions": [
        {
          "type": "multiple_choice",
          "question": "What is a vector space?",
          "options": ["Option A", "Option B", "Option C", "Option D"],
          "correct_answer": 0,
          "points": 5
        }
      ],
      "total_points": 50
    }
  ]
}
```

## Output Structure

### Generated XML Files
For each assessment:
- `assignment_week_01.xml` - D2L assignment with dropbox configuration
- `quiz_week_01.xml` - QTI 1.2 compliant quiz XML
- `discussion_week_01.xml` - D2L discussion forum with grading

### Assessment Types Supported

#### 1. Assignments (D2L XML Format)
- **Resource Type**: `assignment_xmlv1p0`
- **Features**: Dropbox configuration, file upload settings, rubric integration
- **Grading**: Point values, due dates, late penalties
- **Submission**: File upload with format restrictions

#### 2. Quizzes (QTI 1.2 Format)
- **Resource Type**: `imsqti_xmlv1p2/imscc_xmlv1p1/assessment`
- **Question Types**: Multiple choice, true/false, short answer, essay
- **Features**: Time limits, attempt restrictions, feedback
- **Grading**: Automatic scoring with manual review options

#### 3. Discussions (D2L XML Format)
- **Resource Type**: `discussion_xmlv1p0`
- **Features**: Threaded discussions, grading rubrics
- **Participation**: Initial posts and response requirements
- **Grading**: Point-based assessment with detailed rubrics

## XML Structure Examples

### D2L Assignment XML
```xml
<?xml version="1.0" encoding="UTF-8"?>
<assignment xmlns="http://www.d2l.org/xsd/d2lcp_v1p0" 
           xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <header>
        <title>Linear Systems Analysis</title>
        <description>
            <text>Complete assignment description with detailed instructions...</text>
        </description>
    </header>
    <submission>
        <dropbox>
            <name>Week 1 Assignment Dropbox</name>
            <instructions>Submit your completed assignment here.</instructions>
            <due_date>2025-08-12T23:59:59</due_date>
            <points_possible>100</points_possible>
            <file_submission enabled="true">
                <allowed_extensions>.pdf,.doc,.docx</allowed_extensions>
                <max_file_size>10485760</max_file_size>
            </file_submission>
        </dropbox>
    </submission>
    <grading>
        <rubric>
            <criterion id="content" points="40">
                <name>Content Quality</name>
                <description>Demonstrates understanding of key concepts</description>
            </criterion>
            <criterion id="analysis" points="35">
                <name>Analysis and Application</name>
                <description>Applies concepts to solve problems effectively</description>
            </criterion>
            <criterion id="communication" points="25">
                <name>Written Communication</name>
                <description>Clear, organized, and professional writing</description>
            </criterion>
        </rubric>
    </grading>
</assignment>
```

### QTI Quiz XML Structure
```xml
<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2">
    <assessment ident="QUIZ_WEEK_01" title="Week 1 Knowledge Check">
        <qtimetadata>
            <qtimetadatafield>
                <fieldlabel>cc_maxattempts</fieldlabel>
                <fieldentry>3</fieldentry>
            </qtimetadatafield>
            <qtimetadatafield>
                <fieldlabel>cc_timelimit</fieldlabel>
                <fieldentry>1800</fieldentry>
            </qtimetadatafield>
        </qtimetadata>
        <section ident="root_section">
            <item ident="QUESTION_01" title="Vector Space Definition">
                <itemmetadata>
                    <qtimetadata>
                        <qtimetadatafield>
                            <fieldlabel>question_type</fieldlabel>
                            <fieldentry>multiple_choice_question</fieldentry>
                        </qtimetadatafield>
                        <qtimetadatafield>
                            <fieldlabel>points_possible</fieldlabel>
                            <fieldentry>5</fieldentry>
                        </qtimetadatafield>
                    </qtimetadata>
                </itemmetadata>
                <presentation>
                    <material>
                        <mattext texttype="text/html">What is a vector space?</mattext>
                    </material>
                    <response_lid ident="response1" rcardinality="Single">
                        <render_choice>
                            <response_label ident="choice_a">
                                <material><mattext>A set of vectors with defined operations</mattext></material>
                            </response_label>
                            <response_label ident="choice_b">
                                <material><mattext>A geometric representation of data</mattext></material>
                            </response_label>
                            <response_label ident="choice_c">
                                <material><mattext>A mathematical function</mattext></material>
                            </response_label>
                            <response_label ident="choice_d">
                                <material><mattext>A coordinate system</mattext></material>
                            </response_label>
                        </render_choice>
                    </response_lid>
                </presentation>
                <resprocessing>
                    <outcomes>
                        <decvar maxvalue="5" minvalue="0" varname="SCORE" vartype="Decimal"/>
                    </outcomes>
                    <respcondition continue="No">
                        <conditionvar>
                            <varequal respident="response1">choice_a</varequal>
                        </conditionvar>
                        <setvar action="Set" varname="SCORE">5</setvar>
                    </respcondition>
                </resprocessing>
            </item>
        </section>
    </assessment>
</questestinterop>
```

## Usage

### Command Line Interface
```bash
python assessment_xml_generator.py --input structured_course.json --output assessments_xml/
```

### Python API
```python
from assessment_xml_generator import AssessmentXMLGenerator

generator = AssessmentXMLGenerator()
result = generator.generate_assessment_xml('structured_course.json', 'output_directory/')
```

## Configuration

### Config File: `config/assessment_config.json`
```json
{
  "default_settings": {
    "assignment_points": 100,
    "quiz_points": 50,
    "discussion_points": 25,
    "max_attempts": 3,
    "time_limit_minutes": 30
  },
  "file_submission": {
    "allowed_extensions": [".pdf", ".doc", ".docx", ".txt"],
    "max_file_size_mb": 10,
    "multiple_files": false
  },
  "grading_schema": {
    "assignment_rubric": {
      "content": 40,
      "analysis": 35,
      "communication": 25
    },
    "quiz_settings": {
      "show_correct_answers": true,
      "randomize_questions": false,
      "allow_partial_credit": true
    }
  },
  "brightspace_integration": {
    "dropbox_creation": true,
    "gradebook_integration": true,
    "notification_settings": true
  }
}
```

## Assessment Content Guidelines

### Assignment Structure
- **Clear Instructions**: Detailed description of requirements
- **Word Limits**: Specific word count requirements (700-1000 words)
- **Rubric Criteria**: Detailed evaluation standards
- **Submission Guidelines**: File format and submission process
- **Due Dates**: Clear deadlines with time zones

### Quiz Design
- **Question Variety**: Multiple choice, true/false, short answer
- **Point Distribution**: Balanced point values across questions
- **Feedback**: Immediate or delayed feedback options
- **Attempt Limits**: Reasonable number of attempts
- **Time Management**: Appropriate time limits for content complexity

### Discussion Forums
- **Initial Post Requirements**: Substantial original contributions
- **Response Requirements**: Meaningful peer interactions
- **Grading Rubrics**: Clear participation expectations
- **Moderation Guidelines**: Instructor facilitation standards

## Validation Requirements

### XML Compliance
- **Schema Validation**: All XML validates against appropriate schemas
- **Namespace Accuracy**: Correct namespace declarations for each format
- **Resource Type Verification**: Proper resource type assignments
- **Metadata Completeness**: All required metadata fields populated

### Brightspace Integration
- **Import Testing**: XML imports successfully without errors
- **Functional Verification**: All assessment tools work as intended
- **Gradebook Integration**: Points and weighting appear correctly
- **Student Experience**: Assessments display and function properly

## Error Handling

### Critical Errors (System Exit)
- Invalid assessment data structure
- XML generation failures
- Schema validation errors
- File system write issues

### Validation Warnings
- Missing optional metadata
- Suboptimal question distribution
- Accessibility concerns in content

## Dependencies

### Required Packages
```
python >= 3.8
lxml >= 4.6.0
xml.etree.ElementTree (built-in)
uuid >= 1.30 (built-in)
datetime (built-in)
```

### External Validation
- XML Schema validation tools
- QTI validation services
- D2L XML format checkers

## Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Testing
- Brightspace import validation
- Assessment functionality verification
- Gradebook integration testing
- Student workflow testing

### Quality Assurance
- [ ] XML validates against schemas
- [ ] Brightspace import succeeds
- [ ] Assessment tools function correctly
- [ ] Gradebook integration works
- [ ] Student interface displays properly

## Security Considerations

- **Input Sanitization**: Clean all user-provided content
- **XML Injection Prevention**: Escape special characters properly
- **File Upload Security**: Validate file types and sizes
- **Content Filtering**: Remove potentially harmful content

## Performance Optimization

- **Batch Processing**: Generate multiple assessments efficiently
- **Memory Management**: Handle large assessment datasets
- **XML Optimization**: Minimize file sizes while maintaining functionality
- **Caching**: Cache generated XML templates for reuse

## Changelog

### v1.0.0 (2025-08-05)
- Initial implementation with QTI 1.2 and D2L XML support
- Native Brightspace integration with functional assessment tools
- Comprehensive error handling and validation
- Automated gradebook integration and point configuration

## License

MIT License - See LICENSE file in project root.