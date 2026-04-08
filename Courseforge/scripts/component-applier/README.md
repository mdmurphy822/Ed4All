# Component Applier

Transform plain HTML content with Bootstrap 4.3.1 interactive components for enhanced educational engagement.

## Overview

The Component Applier script analyzes HTML content and automatically applies appropriate interactive Bootstrap components based on content patterns. It detects sections that would benefit from accordions, timelines, callouts, flip cards, and other interactive elements.

## Features

- **Pattern-based content detection** - Automatically identifies content types
- **Bootstrap 4.3.1 compatible output** - Consistent styling framework
- **WCAG 2.2 AA accessibility compliance** - All components include ARIA attributes
- **Brightspace D2L compatibility** - Tested for LMS import
- **AI-assisted recommendations** (optional) - Claude API for enhanced analysis

## Installation

```bash
pip install beautifulsoup4 lxml
```

## Usage

### Single File Processing
```bash
python component_applier.py --input content.html --output styled.html
```

### Directory Processing
```bash
python component_applier.py --input-dir /content/ --output-dir /styled/
```

### With Component Mapping
```bash
python component_applier.py --mapping mapping.json --input-dir /content/
```

### JSON Output Report
```bash
python component_applier.py --input-dir /content/ --output-dir /styled/ --json
```

## Component Types

| Component | Detected Patterns | Use Case |
|-----------|------------------|----------|
| Accordion | Definitions, FAQ, glossary | Expandable content sections |
| Timeline | Steps, sequences, processes | Sequential procedures |
| Callout Info | Tips, hints, best practices | Helpful notes |
| Callout Warning | Warnings, cautions, alerts | Important notices |
| Callout Success | Achievements, completed | Positive feedback |
| Callout Danger | Critical, danger | Error warnings |
| Flip Card | Compare, versus, before/after | Two-sided reveals |
| Knowledge Check | Quiz, self-assessment | Review questions |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COURSEFORGE_PATH` | Base Courseforge directory | Auto-detected |

## Output

The script generates:
- Enhanced HTML files with Bootstrap components
- Processing report (text or JSON format)
- Log file (`component_applier.log`)

## Example Output Report

```
Component Application Report
============================
Files processed: 28
Successful: 28
Components applied: 45

Component Distribution:
  accordion: 12
  callout_info: 15
  timeline: 8
  knowledge_check: 10
```

## Integration

Works with other Courseforge scripts:
- **DART Batch Processor** - Apply components after accessibility conversion
- **Remediation Validator** - Validate component accessibility
- **Brightspace Packager** - Include enhanced content in IMSCC

## Dependencies

- beautifulsoup4>=4.9.0
- lxml>=4.6.0
- Python 3.8+

## License

Part of the Courseforge project for educational content development.
