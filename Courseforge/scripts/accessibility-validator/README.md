# Accessibility Validator

WCAG 2.2 AA compliance checker for HTML educational content.

## Overview

The Accessibility Validator performs comprehensive accessibility checks on HTML content to ensure compliance with WCAG 2.2 AA guidelines. It identifies issues with images, headings, forms, tables, links, and more.

## Features

- **Alt text validation** - Checks all images for meaningful alt text
- **Heading hierarchy** - Validates proper heading structure (no skipped levels)
- **Form accessibility** - Verifies label associations and fieldset usage
- **Table accessibility** - Checks for headers, scope, and captions
- **Link text quality** - Flags generic link text like "click here"
- **ARIA landmarks** - Validates main, navigation, and other landmarks
- **Color contrast** - Identifies potential contrast issues
- **Language declarations** - Ensures lang attribute is present
- **Skip links** - Recommends skip navigation where appropriate

## Installation

```bash
pip install beautifulsoup4 lxml
```

## Usage

### Single File Validation
```bash
python accessibility_validator.py --input page.html
```

### Directory Validation
```bash
python accessibility_validator.py --input-dir /content/ --output report.json --format json
```

### Strict Mode
```bash
python accessibility_validator.py --input page.html --strict
```

## Issue Severity Levels

| Level | Description | Impact |
|-------|-------------|--------|
| CRITICAL | WCAG A failures | Must fix - content inaccessible |
| HIGH | WCAG AA failures | Should fix - accessibility barriers |
| MEDIUM | Best practice violations | Recommended improvements |
| LOW | Enhancement opportunities | Optional improvements |

## WCAG Criteria Checked

### Level A (Critical)
- 1.1.1 Non-text Content (images, alt text)
- 1.3.1 Info and Relationships (headings, forms)
- 2.1.1 Keyboard (focus, navigation)
- 3.1.1 Language of Page (lang attribute)
- 4.1.2 Name, Role, Value (form controls)

### Level AA (High)
- 1.4.3 Contrast (Minimum)
- 2.4.6 Headings and Labels
- 2.4.7 Focus Visible

## Output Formats

### Text Report
```
======================================================================
WCAG 2.2 AA ACCESSIBILITY VALIDATION REPORT
======================================================================
File: content/week_01_overview.html
Timestamp: 2025-12-08T10:30:00
----------------------------------------------------------------------
Total Issues: 5
  Critical: 1
  High: 2
  Medium: 2
  Low: 0
----------------------------------------------------------------------
WCAG 2.2 AA Compliant: NO
======================================================================

ISSUES FOUND:

1. [CRITICAL] WCAG 1.1.1
   Element: <img src="diagram.png">
   Issue: Image missing alt attribute
   Fix: Add alt attribute with descriptive text
```

### JSON Report
```json
{
  "file_path": "content/week_01_overview.html",
  "timestamp": "2025-12-08T10:30:00",
  "total_issues": 5,
  "critical_count": 1,
  "high_count": 2,
  "medium_count": 2,
  "low_count": 0,
  "wcag_aa_compliant": false,
  "issues": [...]
}
```

## Integration

### With Remediation Validator
```python
from accessibility_validator import AccessibilityValidator

validator = AccessibilityValidator()
report = validator.validate_file(Path("content.html"))

if not report.wcag_aa_compliant:
    print(f"File has {report.critical_count} critical issues")
```

### In Quality Assurance Pipeline
The accessibility validator should be run:
1. After content generation
2. Before IMSCC packaging
3. As part of final quality gate

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Validation passed (WCAG AA compliant) |
| 1 | Validation failed (has critical/high issues) |

## Dependencies

- beautifulsoup4>=4.9.0
- lxml>=4.6.0
- Python 3.8+

## License

Part of the Courseforge project for educational content development.
