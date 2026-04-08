# IMSCC Extractor

Universal IMSCC package import foundation for Courseforge's intake system.

## Overview

The IMSCC Extractor is the core infrastructure component that enables Courseforge to import and process IMSCC packages from **any LMS** (Brightspace, Canvas, Blackboard, Moodle, Sakai, or generic). It provides:

- **Universal LMS Detection**: Automatically identifies source LMS from manifest namespaces and file patterns
- **Multi-Version Support**: Handles IMS Common Cartridge 1.0, 1.1, 1.2, and 1.3
- **Content Classification**: Categorizes all resources by type (HTML, PDF, Office docs, images, assessments)
- **Remediation Analysis**: Identifies content requiring DART conversion or accessibility fixes
- **Structured Output**: Generates comprehensive JSON manifest for downstream processing

## Usage

### Basic Extraction
```bash
python imscc_extractor.py --input package.imscc --output /path/to/extracted/
```

### Analyze Without Extracting
```bash
python imscc_extractor.py --input package.imscc --analyze-only
```

### JSON Output
```bash
python imscc_extractor.py --input package.imscc --json
```

### Human-Readable Summary
```bash
python imscc_extractor.py --input package.imscc --summary
```

## Programmatic Usage

```python
from imscc_extractor import IMSCCExtractor, ExtractedCourse

# Extract package
extractor = IMSCCExtractor(
    imscc_path=Path('/path/to/package.imscc'),
    output_path=Path('/path/to/output/')
)
result: ExtractedCourse = extractor.extract()

# Access extraction results
print(f"Source LMS: {result.source_lms.value}")
print(f"IMSCC Version: {result.imscc_version.value}")
print(f"Total Resources: {result.total_resources}")
print(f"Resources needing remediation: {result.resources_needing_remediation}")

# Get JSON export
json_output = extractor.to_json()

# Get summary
summary = extractor.get_extraction_summary()
```

## LMS Detection

The extractor automatically detects the source LMS using:

| LMS | Namespace Patterns | File Patterns |
|-----|-------------------|---------------|
| **Brightspace** | `d2l_2p0`, `desire2learn` | `d2l_`, `D2L` |
| **Canvas** | `canvas.instructure` | `canvas_`, `assignment_groups` |
| **Blackboard** | `blackboard.com`, `bb_` | `bb_`, `res00` |
| **Moodle** | `moodle.org` | `moodle_`, `backup_` |
| **Sakai** | `sakaiproject.org` | `sakai_` |

## Output Structure

### ExtractedCourse Object

```python
@dataclass
class ExtractedCourse:
    # Identification
    package_path: str
    extraction_path: str
    extraction_timestamp: str

    # LMS Detection
    source_lms: LMSType
    imscc_version: IMSCCVersion
    lms_detection_confidence: float
    detection_evidence: List[str]

    # Course Metadata
    title: str
    description: str
    identifier: str
    language: str

    # Content Structure
    organization: List[OrganizationItem]
    resources: Dict[str, Resource]

    # Remediation Analysis
    total_resources: int
    resources_needing_remediation: int
    remediation_summary: Dict[str, int]

    # File Inventory
    html_files: List[str]
    pdf_files: List[str]
    office_files: List[str]
    image_files: List[str]
    media_files: List[str]
    assessment_files: List[str]
    other_files: List[str]
```

### Remediation Summary

```json
{
  "pdf_conversion": 5,
  "office_conversion": 3,
  "image_alt_text": 12,
  "html_accessibility": 8,
  "total_needing_remediation": 28
}
```

## Integration with Remediation Pipeline

The extractor output feeds directly into the remediation workflow:

```
IMSCC Package → IMSCCExtractor → ExtractedCourse →
  → dart-batch-processor (PDFs)
  → accessibility-remediation (HTML)
  → content-quality-remediation (educational depth)
  → intelligent-design-mapper (component selection)
  → brightspace-packager (repackaging)
```

## Resource Types

| Type | Description | Remediation |
|------|-------------|-------------|
| `html` | HTML web content | Accessibility fixes |
| `pdf` | PDF documents | DART conversion |
| `office_doc` | Word, PowerPoint, Excel | DART conversion |
| `image` | Images | Alt text generation |
| `video` | Video files | Captions/transcripts |
| `audio` | Audio files | Transcripts |
| `quiz_qti` | QTI assessments | Format validation |
| `assignment` | Assignment definitions | D2L XML check |
| `discussion` | Discussion topics | Format validation |
| `link` | Web links | Link validation |
| `lti` | LTI integrations | Configuration check |

## Error Handling

The extractor provides detailed error information:

```python
extracted_course.errors    # Critical errors
extracted_course.warnings  # Non-critical warnings
```

## Logging

Logs are written to:
- Console (INFO level)
- `imscc_extractor.log` file (DEBUG level)

## Dependencies

- Python 3.8+
- No external dependencies (uses standard library only)

## Version History

- **1.0.0** (2025-12-08): Initial release with universal LMS support
