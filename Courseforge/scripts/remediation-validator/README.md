# Remediation Validator

Comprehensive quality assurance validation for remediated course content, ensuring WCAG 2.2 AA compliance, OSCQR standards adherence, and Brightspace compatibility.

## Overview

The Remediation Validator performs final quality assurance checks on course content that has been through the remediation pipeline. It validates accessibility compliance, educational quality standards, and technical compatibility before IMSCC packaging.

## Features

- **WCAG 2.2 AA compliance validation** - Automated accessibility checking
- **OSCQR standards adherence** - Educational quality metrics
- **Brightspace compatibility testing** - LMS-specific validation
- **Content integrity verification** - Broken link detection
- **WCAG 2.2-specific checks** - Focus appearance/obscured, target size, dragging movements, consistent help, accessible authentication

## Installation

```bash
pip install beautifulsoup4 lxml
```

## Usage

### Validate Course Directory
```bash
python remediation_validator.py --course-dir /path/to/course/
```

### Generate JSON Report
```bash
python remediation_validator.py --course-dir /path/to/course/ --json --output report.json
```

### Before/After Comparison
```bash
python remediation_validator.py --course-dir /path/to/course/ --before original/ --after remediated/ --compare
```

## Validation Checks

### WCAG 2.2 AA Accessibility
- **Images**: Alt text presence and quality
- **Headings**: Proper hierarchy (no skipped levels)
- **Links**: Descriptive link text (not "click here")
- **Forms**: Label associations
- **Color**: Sufficient contrast ratios
- **Tables**: Header cells and scope attributes
- **Language**: Document language declaration
- **Keyboard**: Focus indicators and navigation

### OSCQR Standards
- Course overview and navigation
- Technology requirements documentation
- Design consistency and layout
- Content accessibility and engagement
- Interaction opportunities
- Assessment clarity and feedback

### Brightspace Compatibility
- HTML rendering validation
- CSS compatibility
- File path integrity
- Resource reference validation
- Manifest completeness

### WCAG 2.2-Specific Checks
- Focus not obscured (2.4.11)
- Focus appearance (2.4.13)
- Target size minimum (2.5.8)
- Dragging movements alternative (2.5.7)
- Consistent help (3.2.6)
- Accessible authentication (3.3.8)

## Severity Levels

| Level | Description | Action Required |
|-------|-------------|-----------------|
| CRITICAL | Import will fail | Must fix before packaging |
| HIGH | Major functionality issues | Should fix before packaging |
| MEDIUM | Quality concerns | Recommended fixes |
| LOW | Minor improvements | Optional enhancements |

## Output

### Text Report
```
╔══════════════════════════════════════════════════════════════╗
║           REMEDIATION VALIDATION REPORT                       ║
╠══════════════════════════════════════════════════════════════╣
║ Files Validated: 42                                           ║
║ WCAG Issues: 3 (2 High, 1 Medium)                            ║
║ OSCQR Compliance: 94%                                         ║
║ Pattern Violations: 0                                         ║
║ Overall Status: PASS (with warnings)                          ║
╚══════════════════════════════════════════════════════════════╝
```

### JSON Report
```json
{
  "summary": {
    "files_validated": 42,
    "wcag_issues": 3,
    "oscqr_compliance": 94,
    "pattern_violations": 0,
    "status": "PASS"
  },
  "issues": [...],
  "recommendations": [...]
}
```

## Integration

Part of the remediation pipeline:

1. **IMSCC Extractor** - Parse incoming course
2. **Content Analyzer** - Identify remediation needs
3. **DART Batch Processor** - Convert documents
4. **Component Applier** - Enhance content
5. **Remediation Validator** - Quality assurance ← This script
6. **Brightspace Packager** - Final IMSCC creation

## Exit Codes

Exit status is keyed off the report's overall score:

| Code | Meaning |
|------|---------|
| 0 | Passed (overall score >= 95) |
| 1 | Passed with warnings (overall score >= 80) |
| 2 | Failed (overall score < 80) |
| 3 | Error (validation raised an exception) |

## Dependencies

- beautifulsoup4>=4.9.0
- lxml>=4.6.0
- Python 3.8+

## License

Part of the Courseforge project for educational content development.
