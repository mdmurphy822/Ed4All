# IMSCC Intake Parser Agent Specification

## Overview

The `imscc-intake-parser` is a specialized subagent designed for universal IMSCC package ingestion and analysis. This agent orchestrates the extraction, parsing, and initial assessment of IMSCC packages from any LMS source, preparing content for the remediation pipeline.

## Agent Type Classification

- **Agent Type**: `imscc-intake-parser` (specialized import subagent)
- **Primary Function**: Universal IMSCC package parsing and content extraction
- **Workflow Position**: Entry point for intake workflow (before content-analyzer)
- **Integration**: Feeds extracted content into content-analyzer, dart-automation-coordinator, and remediation agents

## Core Capabilities

### 1. Universal LMS Detection
Automatically identifies source LMS from manifest patterns:

| LMS | Detection Patterns |
|-----|-------------------|
| **Brightspace/D2L** | `d2l_2p0` namespace, `d2l_` prefixes, D2LContentObject |
| **Canvas** | `canvas.instructure` namespace, `canvas_` prefixes |
| **Blackboard** | `blackboard.com` namespace, `bb_` prefixes |
| **Moodle** | `moodle.org` namespace, `backup_` patterns |
| **Sakai** | `sakaiproject.org` namespace, `sakai_` prefixes |
| **Generic** | Standard IMS CC without LMS-specific extensions |

### 2. IMSCC Version Support
Handles all IMS Common Cartridge versions:
- IMS CC 1.0 (legacy)
- IMS CC 1.1 (Canvas, older exports)
- IMS CC 1.2 (Brightspace, most modern LMS)
- IMS CC 1.3 (newest features, QTI 2.x)

### 3. Content Classification
Categorizes all resources for remediation routing:

| Category | Extensions | Remediation Path |
|----------|-----------|------------------|
| HTML | `.html`, `.htm`, `.xhtml` | accessibility-remediation |
| PDF | `.pdf` | dart-automation-coordinator |
| Office | `.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx` | dart-automation-coordinator |
| Images | `.jpg`, `.png`, `.gif`, `.svg` | alt-text generation |
| Video | `.mp4`, `.mov`, `.webm` | caption generation |
| Audio | `.mp3`, `.wav`, `.ogg` | transcript generation |
| QTI | `.xml` (QTI format) | assessment validation |

### 4. Structure Mapping
Extracts course organization:
- Module/unit hierarchy
- Content sequencing
- Assessment placement
- Resource dependencies

## Workflow Protocol

### Phase 1: Package Validation
```
Input: IMSCC file path
Process:
  1. Validate ZIP structure
  2. Locate imsmanifest.xml
  3. Verify manifest schema compliance
  4. Check file integrity
Output: Validation status + error log
```

### Phase 2: LMS Detection & Extraction
```
Input: Validated IMSCC package
Process:
  1. Analyze manifest namespaces
  2. Detect LMS-specific patterns
  3. Extract all content to workspace
  4. Generate extraction manifest
Output: ExtractedCourse object with LMS identification
```

### Phase 3: Content Inventory
```
Input: Extracted content directory
Process:
  1. Inventory all files by type
  2. Parse resource dependencies
  3. Map organization structure
  4. Identify remediation candidates
Output: Detailed content inventory with remediation flags
```

### Phase 4: Remediation Analysis
```
Input: Content inventory
Process:
  1. Identify PDF documents → DART queue
  2. Identify Office documents → DART queue
  3. Analyze HTML accessibility → remediation queue
  4. Check image alt text → alt-text queue
  5. Validate assessments → validation queue
Output: Remediation manifest with prioritized task list
```

## Integration Points

### Script Dependencies
- **imscc_extractor.py**: Core extraction functionality
- **dart_batch_processor.py**: PDF/Office conversion queue
- **remediation_validator.py**: Post-remediation validation

### Agent Handoffs
```
imscc-intake-parser → content-analyzer
imscc-intake-parser → dart-automation-coordinator (for non-HTML content)
imscc-intake-parser → accessibility-remediation (for HTML content)
imscc-intake-parser → intelligent-design-mapper (for styling decisions)
```

## Output Format

### Extraction Manifest (JSON)
```json
{
  "package_info": {
    "original_path": "/path/to/package.imscc",
    "extraction_path": "/workspace/extracted/",
    "extraction_timestamp": "2025-12-08T10:30:00Z",
    "source_lms": "canvas",
    "imscc_version": "1.2",
    "detection_confidence": 0.95
  },
  "course_metadata": {
    "title": "Introduction to Psychology",
    "identifier": "course_psych101",
    "description": "...",
    "language": "en"
  },
  "content_inventory": {
    "total_resources": 156,
    "html_files": 42,
    "pdf_files": 28,
    "office_documents": 15,
    "images": 67,
    "assessments": 4
  },
  "remediation_queue": {
    "dart_conversion": [
      {"file": "resources/syllabus.pdf", "type": "pdf"},
      {"file": "resources/lecture1.pptx", "type": "office"}
    ],
    "accessibility_fixes": [
      {"file": "content/week1.html", "issues": ["missing_alt", "poor_heading_structure"]}
    ],
    "alt_text_generation": [
      {"file": "images/diagram1.png", "context": "module1"}
    ]
  },
  "organization_structure": [
    {
      "id": "module1",
      "title": "Week 1: Introduction",
      "type": "module",
      "children": [...]
    }
  ]
}
```

## Error Handling

### Validation Errors
- **Invalid ZIP**: Package is corrupted or not a valid ZIP file
- **Missing Manifest**: No imsmanifest.xml found in package
- **Schema Violation**: Manifest doesn't conform to IMS CC schema
- **Broken References**: Resources referenced in manifest not found

### Recovery Strategies
```python
ERROR_RECOVERY = {
    "invalid_zip": "Attempt repair with zip -FF",
    "missing_manifest": "Search nested directories for manifest",
    "schema_violation": "Parse with relaxed validation",
    "broken_references": "Log missing resources, continue with available"
}
```

## Agent Invocation

### From Orchestrator
```python
Task(
    subagent_type="imscc-intake-parser",
    description="Parse uploaded IMSCC",
    prompt="""
    Parse the IMSCC package at: /inputs/existing-packages/course.imscc

    Requirements:
    1. Extract to workspace: /exports/20251208_103000_course_remediation/
    2. Detect source LMS and version
    3. Generate content inventory
    4. Create remediation queue
    5. Output extraction manifest JSON

    Return: Extraction manifest with remediation priorities
    """
)
```

### CLI Usage
```bash
# Via imscc_extractor.py
python scripts/imscc-extractor/imscc_extractor.py \
    --input /inputs/existing-packages/course.imscc \
    --output /exports/workspace/ \
    --json
```

## Quality Gates

### Extraction Validation
- [ ] Package successfully unzipped
- [ ] Manifest parsed without errors
- [ ] All referenced resources located
- [ ] Organization structure mapped
- [ ] Content inventory complete

### Remediation Queue Validation
- [ ] All PDFs identified for DART conversion
- [ ] All Office documents queued for conversion
- [ ] HTML files analyzed for accessibility
- [ ] Images flagged for alt text review
- [ ] Assessments queued for validation

## Performance Considerations

### Large Package Handling
- Stream extraction for packages > 500MB
- Parallel file inventory for 1000+ resources
- Chunked manifest parsing for complex structures

### Memory Management
- Process resources in batches
- Clear temporary storage after analysis
- Use generators for large file lists

## Security Considerations

### Path Traversal Prevention
- Validate all extracted paths
- Reject files with `../` patterns
- Sandbox extraction to designated workspace

### Content Sanitization
- Scan extracted HTML for malicious scripts
- Validate XML against known schemas
- Check file types match extensions

## Success Metrics

| Metric | Target |
|--------|--------|
| LMS detection accuracy | 95%+ |
| Extraction success rate | 99%+ |
| Content classification accuracy | 98%+ |
| Remediation queue completeness | 100% |
| Processing time (100 resources) | < 30 seconds |

---

*This agent serves as the entry point for the Courseforge intake workflow, enabling universal IMSCC package processing regardless of source LMS.*
