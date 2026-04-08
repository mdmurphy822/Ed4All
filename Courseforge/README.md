# Courseforge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![WCAG 2.2 AA](https://img.shields.io/badge/WCAG-2.2%20AA-green.svg)](https://www.w3.org/WAI/WCAG22/quickref/)

**Education Reimagined. Access for All.**

AI-powered course generation and remediation system that creates and improves accessible, LMS-ready IMSCC packages.

## Overview

Courseforge uses a multi-agent orchestration system to create high-quality online courses compatible with Brightspace/D2L and other learning management systems. It also provides comprehensive intake and remediation capabilities for existing courses.

### Key Features

**Course Creation**
- **Simplified Pipeline**: Exam objectives + textbooks → IMSCC package
- **DART Integration**: Accessibility-first textbook processing
- **Multi-Agent System**: Specialized agents for each phase of development
- **WCAG 2.2 AA**: Built-in accessibility compliance
- **OSCQR Standards**: Educational quality validation
- **Pattern Prevention**: 22+ error patterns identified and prevented

**Course Intake & Remediation (NEW)**
- **Universal IMSCC Import**: Canvas, Blackboard, Moodle, Brightspace, Sakai
- **Automated DART Conversion**: PDFs and Office docs → accessible HTML
- **AI-Powered Accessibility Fixes**: Alt text, heading structure, contrast
- **Intelligent Component Styling**: AI-selected interactive elements
- **100% WCAG 2.2 AA Compliance**: Guaranteed accessibility output

## Example Package

See [`examples/intro_python.imscc`](examples/) for a complete 12-week Introduction to Python course demonstrating:
- Proper IMSCC 1.3 structure
- QTI 1.2 quizzes with Brightspace compatibility
- Native assignment integration (`assignment_xmlv1p0`)
- Discussion topics
- WCAG 2.2 AA compliant content

## Quick Start

### Mode 1: Course Creation
1. Place exam objectives in `inputs/exam-objectives/`
2. (Optional) Process textbooks through DART and place in `inputs/textbooks/`
3. Invoke: `exam-research → course-outliner → content-generator → brightspace-packager`
4. Output: IMSCC file in `exports/YYYYMMDD_HHMMSS_coursename/`

### Mode 2: Course Remediation (NEW)
1. Place existing IMSCC in `inputs/existing-packages/`
2. Invoke: `imscc-intake-parser → content-analyzer → remediation agents → brightspace-packager`
3. Output: Improved IMSCC with 100% accessibility compliance

## Project Structure

```
Courseforge/
├── CLAUDE.md              # Orchestration instructions
├── README.md              # This file
├── docs/                  # Documentation
│   ├── troubleshooting.md
│   ├── workflow-reference.md
│   └── getting-started.md
├── agents/                # Agent specifications
├── inputs/                # Input files
│   ├── exam-objectives/
│   ├── textbooks/
│   └── existing-packages/ # IMSCC for intake (NEW)
├── scripts/               # Automation scripts (NEW)
│   ├── imscc-extractor/
│   ├── dart-batch-processor/
│   ├── component-applier/
│   └── remediation-validator/
├── templates/             # HTML templates
├── schemas/               # IMSCC schemas
├── imscc-standards/       # Technical specs
├── exports/               # Generated packages
└── runtime/               # Agent workspaces
```

## Available Agents

### Course Creation
| Agent | Purpose |
|-------|---------|
| `exam-research` | Certification objective analysis |
| `requirements-collector` | Course specification gathering |
| `course-outliner` | Structure and learning objectives |
| `content-generator` | Educational content creation |
| `educational-standards` | Pedagogical compliance |
| `quality-assurance` | Pattern prevention |
| `oscqr-course-evaluator` | Quality assessment |
| `brightspace-packager` | IMSCC packaging |

### Intake & Remediation (NEW)
| Agent | Purpose |
|-------|---------|
| `imscc-intake-parser` | Universal IMSCC import |
| `content-analyzer` | Accessibility/quality gap detection |
| `dart-automation-coordinator` | Automated document conversion |
| `accessibility-remediation` | WCAG 2.2 AA fixes |
| `content-quality-remediation` | Educational depth enhancement |
| `intelligent-design-mapper` | AI component selection |

## Workflow

```
USER REQUEST →
  exam-research (analyze objectives) →
  course-outliner (create structure) →
  content-generator (create content, 10 agents/batch) →
  quality-assurance + oscqr-course-evaluator (validate) →
  brightspace-packager (package) →
  IMSCC OUTPUT
```

## Textbook Processing

Textbooks must be processed through DART before use:

```bash
# Set DART_PATH to your DART installation directory
cd $DART_PATH
python convert.py textbook.pdf -o /path/to/courseforge/inputs/textbooks/
```

DART produces WCAG 2.2 AA accessible HTML with:
- Semantic structure
- Alt text for images
- MathML for equations
- Proper heading hierarchy

## Quality Standards

### OSCQR Thresholds
- Pre-development: 70%
- Pre-production: 90%
- Accessibility: 100%

### Pattern Prevention
Critical patterns addressed:
- Schema/namespace consistency
- Assessment XML format (QTI 1.2)
- Content completeness
- Organization hierarchy

See `docs/troubleshooting.md` for complete pattern list.

## Documentation

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Main orchestration instructions |
| `docs/troubleshooting.md` | Error patterns and solutions |
| `docs/workflow-reference.md` | Execution protocols |
| `docs/getting-started.md` | Quick start guide |
| `agents/*.md` | Individual agent specs |

## Technical Requirements

- Python 3.8+
- Claude Code access
- Brightspace/D2L for import testing

## DART Setup (Optional)

DART (Digital Accessibility Remediation Tool) is a separate tool used for converting PDFs and Office documents to accessible HTML. It's **optional** but recommended for textbook processing.

### Installation

DART is available as a separate project. Once installed, configure the environment variable:

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export DART_PATH=/path/to/your/DART/installation
```

### Usage with Courseforge

Once `DART_PATH` is set, Courseforge scripts will automatically detect and use DART for:
- PDF textbook conversion
- Office document (Word, PowerPoint) conversion
- Batch document processing via `scripts/dart-batch-processor/`

### Without DART

Courseforge works without DART for:
- Course creation from exam objectives
- IMSCC package generation
- Course remediation (accessibility fixes to existing HTML)

Only textbook PDF/Office conversion requires DART.

## Compatibility

- Brightspace/D2L
- Canvas
- Blackboard
- Moodle
- Any LMS supporting IMSCC 1.1+

---

**Version**: 1.0.0
**License**: MIT
**Last Updated**: December 2025

---

*Love and be loved, live and be free.*
