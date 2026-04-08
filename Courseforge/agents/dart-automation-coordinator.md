# DART Automation Coordinator Agent Specification

## Overview

The `dart-automation-coordinator` is a specialized subagent that orchestrates automatic DART conversion of all non-accessible content. It manages the conversion pipeline for PDFs, Office documents, and other non-HTML content, ensuring 100% accessible course materials.

## Agent Type Classification

- **Agent Type**: `dart-automation-coordinator` (specialized conversion orchestrator)
- **Primary Function**: Coordinate batch document conversion to accessible HTML
- **Workflow Position**: Post-analysis phase (after content-analyzer, before accessibility-remediation)
- **Integration**: Uses dart_batch_processor.py script, feeds to accessibility-remediation agent

## Core Capabilities

### 1. Conversion Queue Management
Manages prioritized queue of documents requiring DART conversion:

| Priority | Document Type | Reason |
|----------|--------------|--------|
| **Critical** | Syllabus, Assignments | Required for course access |
| **High** | Lecture materials | Core learning content |
| **Medium** | Supplementary readings | Supporting materials |
| **Low** | Optional resources | Enhancement content |

### 2. Parallel Conversion Orchestration
Coordinates parallel processing with optimal resource utilization:
- Default: 4 parallel workers
- Adjustable based on system resources
- Progress tracking with estimated completion times
- Automatic retry for failed conversions

### 3. Conversion Validation
Validates all conversions meet accessibility standards:
- WCAG 2.2 AA compliance verification
- Semantic HTML structure check
- Image alt text presence
- Heading hierarchy validation
- Reading order verification

### 4. Resource Replacement
Manages automatic replacement of original documents:
- Updates manifest references to point to HTML versions
- Preserves original files as fallback
- Generates accessibility metadata
- Creates conversion audit trail

## Workflow Protocol

### Phase 1: Queue Building
```
Input: Remediation manifest from content-analyzer
Process:
  1. Parse remediation queue for DART candidates
  2. Prioritize by document type and course position
  3. Validate file accessibility
  4. Create ordered conversion queue
Output: Prioritized conversion task list
```

### Phase 2: Conversion Execution
```
Input: Conversion task queue
Process:
  1. Initialize dart_batch_processor
  2. Execute parallel conversions
  3. Monitor progress and handle failures
  4. Retry failed conversions (max 2 attempts)
  5. Generate conversion report
Output: Converted HTML files + conversion report
```

### Phase 3: Validation
```
Input: Converted HTML files
Process:
  1. WCAG 2.2 AA compliance check
  2. Semantic structure validation
  3. Content integrity verification
  4. Navigation/linking check
Output: Validation report with any issues
```

### Phase 4: Integration
```
Input: Validated HTML files + original manifest
Process:
  1. Update course manifest with new HTML resources
  2. Create resource mappings (original → converted)
  3. Generate fallback links for partial conversions
  4. Update organization structure
Output: Updated course package with accessible content
```

## Conversion Targets

### Documents Requiring Conversion
| File Type | Extension | Conversion Method |
|-----------|-----------|-------------------|
| PDF | `.pdf` | Direct DART conversion |
| Word | `.doc`, `.docx` | LibreOffice → PDF → DART |
| PowerPoint | `.ppt`, `.pptx` | LibreOffice → PDF → DART |
| Excel | `.xls`, `.xlsx` | LibreOffice → PDF → DART |
| OpenDocument | `.odt`, `.odp`, `.ods` | LibreOffice → PDF → DART |

### Conversion Output Standards
All converted content must meet:
- **WCAG 2.2 AA** - Full accessibility compliance
- **Semantic HTML5** - Proper structure and landmarks
- **Responsive Design** - Mobile-friendly layout
- **Brightspace Compatible** - D2L import ready

## Script Integration

### Primary Script
```bash
python scripts/dart-batch-processor/dart_batch_processor.py \
    --input-manifest remediation_queue.json \
    --output-dir /path/to/converted/ \
    --max-workers 4
```

### DART Configuration
```python
import os
DART_CONFIG = {
    "dart_path": os.environ.get("DART_PATH", "/path/to/DART"),  # Set DART_PATH env var
    "timeout_seconds": 300,
    "max_retries": 2,
    "retry_delay": 5,
    "output_format": "html",
    "enable_ocr": True,
    "ocr_dpi": 300,
    "ocr_language": "eng"
}
```

## Agent Invocation

### From Orchestrator
```python
Task(
    subagent_type="dart-automation-coordinator",
    description="Convert all non-HTML content",
    prompt="""
    Process all documents requiring DART conversion.

    Input: Remediation queue at /workspace/remediation_queue.json
    Output: /workspace/converted_content/

    Requirements:
    1. Convert all PDFs to accessible HTML
    2. Convert all Office documents to accessible HTML
    3. Validate WCAG 2.2 AA compliance
    4. Update course manifest with new resources
    5. Generate conversion report

    Return: Conversion summary with success/failure counts
    """
)
```

## Output Format

### Conversion Report (JSON)
```json
{
  "conversion_summary": {
    "total_documents": 28,
    "successful": 27,
    "failed": 1,
    "duration_seconds": 156.3
  },
  "converted_files": [
    {
      "original": "resources/syllabus.pdf",
      "converted": "converted/syllabus/syllabus.html",
      "status": "completed",
      "wcag_compliance": true
    }
  ],
  "failed_conversions": [
    {
      "file": "resources/complex_diagram.pdf",
      "error": "OCR failed - image too complex",
      "fallback": "placeholder HTML created"
    }
  ],
  "manifest_updates": {
    "resources_replaced": 27,
    "new_resources_added": 27
  }
}
```

## Error Handling

### Conversion Failures
| Error Type | Recovery Action |
|------------|-----------------|
| DART timeout | Retry with extended timeout |
| OCR failure | Create placeholder with download link |
| LibreOffice error | Skip Office conversion, flag for manual |
| Validation failure | Log issues, pass to accessibility-remediation |

### Fallback Strategies
1. **Partial Conversion**: Extract readable content, note gaps
2. **Placeholder HTML**: Create accessible wrapper with download link
3. **Manual Queue**: Flag complex documents for human review

## Quality Gates

### Pre-Conversion
- [ ] All source files accessible and readable
- [ ] Sufficient disk space for conversion output
- [ ] DART installation validated
- [ ] LibreOffice available for Office documents

### Post-Conversion
- [ ] All HTML files pass WCAG 2.2 AA validation
- [ ] Semantic structure present (headings, landmarks)
- [ ] No broken links or missing resources
- [ ] Content matches original (text comparison)
- [ ] Images have alt text (generated by DART OCR)

## Performance Targets

| Metric | Target |
|--------|--------|
| PDF conversion success | 98%+ |
| Office conversion success | 95%+ |
| WCAG compliance rate | 100% |
| Average conversion time | <10s per page |
| Total batch throughput | 50+ docs/hour |

## Integration Points

### Upstream (Input)
- `content-analyzer` → Remediation queue manifest

### Downstream (Output)
- `accessibility-remediation` → Converted HTML for additional fixes
- `brightspace-packager` → Updated manifest with accessible content

---

*This agent ensures 100% accessible course content through automated DART conversion, supporting Courseforge's goal of fully accessible educational materials.*
