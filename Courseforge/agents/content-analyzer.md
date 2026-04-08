# Content Analyzer Agent Specification

## Overview

The `content-analyzer` is a specialized subagent designed to analyze extracted course content for accessibility issues, quality gaps, and remediation needs. It provides comprehensive assessment that drives the automated remediation pipeline.

## Agent Type Classification

- **Agent Type**: `content-analyzer` (specialized assessment subagent)
- **Primary Function**: Detect accessibility and quality issues requiring remediation
- **Workflow Position**: Post-extraction phase (after imscc-intake-parser, before remediation agents)
- **Integration**: Receives extracted content, generates remediation manifests for downstream agents

## Core Capabilities

### 1. Accessibility Issue Detection
Comprehensive WCAG 2.2 AA compliance analysis:

| Category | Issues Detected |
|----------|-----------------|
| **Images** | Missing alt text, decorative images without empty alt, complex images without long description |
| **Headings** | Missing H1, skipped levels, poor hierarchy, non-semantic headings |
| **Links** | Non-descriptive text ("click here"), missing purpose, broken links |
| **Color** | Insufficient contrast, color-only information conveyance |
| **Forms** | Missing labels, no error identification, missing instructions |
| **Tables** | Missing headers, no caption, complex tables without summary |
| **Media** | Missing captions, no transcripts, auto-playing content |
| **Navigation** | Missing skip links, keyboard traps, focus issues |

### 2. Content Quality Assessment
Educational quality gap identification:

| Quality Dimension | Assessment Criteria |
|-------------------|---------------------|
| **Depth** | Content word count, concept coverage, example richness |
| **Structure** | Learning objectives presence, logical flow, section organization |
| **Engagement** | Interactive elements, varied content types, multimedia usage |
| **Assessment Alignment** | Objectives-to-assessment mapping, feedback presence |
| **Accessibility** | Alternative formats, multiple representation modes |

### 3. Non-HTML Content Detection
Identifies content requiring format conversion:

| Content Type | Detection Method | Remediation Path |
|--------------|------------------|------------------|
| PDF documents | File extension + MIME type | DART conversion |
| Office documents | File extension analysis | LibreOffice → DART |
| Images with text | OCR text detection | Alt text generation |
| Scanned documents | Image analysis + PDF structure | Enhanced OCR |

### 4. Structural Analysis
Course organization assessment:

- Module/unit hierarchy validation
- Content sequencing logic
- Navigation path completeness
- Resource dependency mapping
- Dead-end detection (content without progression)

## Workflow Protocol

### Phase 1: Content Inventory
```
Input: Extracted course from imscc-intake-parser
Process:
  1. Catalog all content files by type
  2. Map resource dependencies
  3. Identify content hierarchy
  4. Calculate content metrics
Output: Complete content inventory with type classification
```

### Phase 2: Accessibility Analysis
```
Input: Content inventory
Process:
  1. Scan all HTML files for WCAG violations
  2. Check images for alt text presence/quality
  3. Validate heading structure
  4. Test color contrast ratios
  5. Analyze form accessibility
  6. Check media accessibility
Output: Accessibility issues report with severity levels
```

### Phase 3: Quality Assessment
```
Input: HTML content files
Process:
  1. Analyze content depth and coverage
  2. Check learning objective presence
  3. Evaluate content organization
  4. Assess engagement factors
  5. Validate assessment alignment
Output: Quality gaps report with improvement recommendations
```

### Phase 4: Remediation Queue Generation
```
Input: Accessibility + Quality reports
Process:
  1. Prioritize issues by severity and impact
  2. Assign issues to remediation agents
  3. Create task queue for DART conversion
  4. Generate accessibility fix list
  5. Create quality enhancement recommendations
Output: Comprehensive remediation manifest
```

## Detection Algorithms

### Image Alt Text Analysis
```python
def analyze_image_accessibility(img_element):
    issues = []

    # Check alt attribute existence
    if 'alt' not in img_element.attrs:
        issues.append({
            "type": "missing_alt",
            "severity": "critical",
            "wcag": "1.1.1"
        })
    elif img_element['alt'] == '':
        # Check if truly decorative
        if not is_decorative(img_element):
            issues.append({
                "type": "empty_alt_on_informative",
                "severity": "critical"
            })
    elif len(img_element['alt']) < 10:
        issues.append({
            "type": "insufficient_alt",
            "severity": "warning"
        })

    return issues
```

### Heading Structure Validation
```python
def validate_heading_structure(html_content):
    issues = []
    headings = extract_headings(html_content)

    # Check for H1 presence
    if not any(h.level == 1 for h in headings):
        issues.append({"type": "no_h1", "severity": "high"})

    # Check for skipped levels
    for i, heading in enumerate(headings[1:], 1):
        prev_level = headings[i-1].level
        if heading.level > prev_level + 1:
            issues.append({
                "type": "skipped_heading_level",
                "from": prev_level,
                "to": heading.level
            })

    return issues
```

### Color Contrast Analysis
```python
def check_color_contrast(element):
    fg_color = get_foreground_color(element)
    bg_color = get_background_color(element)

    ratio = calculate_contrast_ratio(fg_color, bg_color)

    if is_large_text(element):
        threshold = 3.0  # WCAG AA for large text
    else:
        threshold = 4.5  # WCAG AA for normal text

    if ratio < threshold:
        return {
            "type": "insufficient_contrast",
            "ratio": ratio,
            "required": threshold,
            "severity": "high"
        }
    return None
```

## Output Format

### Remediation Manifest (JSON)
```json
{
  "analysis_summary": {
    "total_files_analyzed": 156,
    "accessibility_issues": 234,
    "quality_gaps": 45,
    "files_needing_conversion": 28
  },
  "accessibility_issues": {
    "critical": [
      {
        "file": "week1/overview.html",
        "issue": "missing_alt",
        "element": "<img src='diagram.png'>",
        "wcag_criterion": "1.1.1",
        "fix_type": "add_alt_text",
        "suggested_fix": "Generate descriptive alt text for diagram"
      }
    ],
    "high": [...],
    "medium": [...],
    "low": [...]
  },
  "quality_gaps": [
    {
      "file": "week3/concepts.html",
      "gap": "shallow_content",
      "current_depth": "150 words",
      "recommended": "500+ words with examples",
      "fix_type": "enhance_content"
    }
  ],
  "conversion_queue": {
    "dart_conversion": [
      {"file": "resources/syllabus.pdf", "type": "pdf", "priority": "critical"},
      {"file": "resources/lecture.pptx", "type": "office", "priority": "high"}
    ]
  },
  "remediation_assignments": {
    "dart-automation-coordinator": 28,
    "accessibility-remediation": 156,
    "content-quality-remediation": 45,
    "intelligent-design-mapper": 89
  }
}
```

## Agent Invocation

### From Orchestrator
```python
Task(
    subagent_type="content-analyzer",
    description="Analyze course content for issues",
    prompt="""
    Analyze the extracted course content for accessibility and quality issues.

    Input: Extracted course at /workspace/extracted/
    Output: Remediation manifest at /workspace/remediation_queue.json

    Requirements:
    1. Scan all HTML files for WCAG 2.2 AA violations
    2. Identify content quality gaps
    3. Detect all non-HTML content requiring DART conversion
    4. Generate prioritized remediation queue
    5. Assign issues to appropriate remediation agents

    Return: Analysis summary with issue counts and severity breakdown
    """
)
```

## Quality Gates

### Analysis Completeness
- [ ] All content files analyzed
- [ ] All WCAG criteria checked
- [ ] Quality dimensions assessed
- [ ] Non-HTML content cataloged
- [ ] Remediation assignments complete

### Output Validation
- [ ] Remediation manifest valid JSON
- [ ] All issues have severity levels
- [ ] WCAG criteria referenced correctly
- [ ] File paths valid and accessible
- [ ] Agent assignments appropriate

## WCAG 2.2 AA Criteria Checked

### Perceivable (Level A + AA)
- 1.1.1 Non-text Content
- 1.2.1-5 Time-based Media
- 1.3.1-5 Adaptable
- 1.4.1-11 Distinguishable

### Operable (Level A + AA)
- 2.1.1-4 Keyboard Accessible
- 2.2.1-2 Enough Time
- 2.3.1 Seizures
- 2.4.1-7 Navigable
- 2.5.1-4 Input Modalities

### Understandable (Level A + AA)
- 3.1.1-2 Readable
- 3.2.1-4 Predictable
- 3.3.1-4 Input Assistance

### Robust (Level A + AA)
- 4.1.1-3 Compatible

## Performance Targets

| Metric | Target |
|--------|--------|
| Analysis speed | <1 second per HTML file |
| Detection accuracy | 95%+ for critical issues |
| False positive rate | <5% |
| Complete analysis | <5 minutes for 200-file course |

---

*This agent provides the intelligence layer for the remediation pipeline, ensuring comprehensive detection of all issues requiring automated or manual intervention.*
