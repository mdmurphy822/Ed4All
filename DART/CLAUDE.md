# DART

> **Universal Protocols**: See root `/CLAUDE.md` for orchestrator protocol, execution rules, decision capture requirements, and error handling. This file contains DART-specific guidance only.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DART (Document Accessibility Remediation Tool) converts PDFs to WCAG 2.2 AA compliant HTML using **multi-source synthesis**. This approach combines multiple extraction sources to produce optimal output:

| Source | Strength | Use |
|--------|----------|-----|
| pdftotext | Text accuracy (99%+), URLs, phone/email | Content extraction |
| pdfplumber | Table structure (headers, rows, cols) | Structure detection |
| OCR | Layout/position validation | Verification |

## Primary Entry Point

```python
# Multi-source synthesis (PREFERRED)
from multi_source_interpreter import convert_single_pdf, batch_synthesize_all

# Single file conversion
result = convert_single_pdf("batch_output/combined/ADI_combined.json", "output.html")

# Batch conversion with zip output
html_files = batch_synthesize_all()
create_zip(html_files, "/path/to/output.zip")
```

## CLI Usage

```bash
# Single file conversion
python multi_source_interpreter.py --input batch_output/combined/ADI_combined.json --output output.html

# Batch conversion
python multi_source_interpreter.py --batch --zip /path/to/output.zip
```

## MCP Tools

DART is exposed via the Ed4All MCP server with these tools:

| Tool | Description |
|------|-------------|
| `convert_pdf_multi_source` | Convert single PDF using multi-source synthesis |
| `batch_convert_multi_source` | Batch convert all PDFs |
| `validate_wcag_compliance` | Validate HTML for WCAG 2.2 AA |
| `get_dart_status` | Get DART capabilities |
| `list_available_campuses` | List available combined JSONs |

## Architecture

```
Combined JSON (pdftotext + tables + OCR)
              │
              ▼
    export_section_contexts()
              │
              ▼
    auto_synthesize_section()
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
contacts   systems   roster
    │         │         │
    └─────────┼─────────┘
              ▼
    generate_html_from_synthesized()
              │
              ▼
        WCAG 2.2 AA HTML
```

## Key Functions

### multi_source_interpreter.py

| Function | Purpose |
|----------|---------|
| `export_section_contexts()` | Build multi-source context for each section |
| `auto_synthesize_section()` | Synthesize optimal output from all sources |
| `synthesize_contacts()` | Match phones/emails to contact names |
| `synthesize_systems_table()` | Build 3-column systems table |
| `synthesize_roster()` | Build course/roster key-value pairs |
| `generate_html_from_synthesized()` | Render WCAG HTML from synthesized data |
| `batch_synthesize_all()` | Process all campuses |
| `create_zip()` | Package HTML files |

## Section Types

| Type | Rendering | Source Strategy |
|------|-----------|----------------|
| campus-info | Key-value table | pdftotext parsing |
| credentials | Key-value table | pdftotext parsing |
| no-account | Paragraphs | pdftotext prose |
| guest | Paragraphs | pdftotext prose |
| contacts | Contact cards | pdfplumber headers + pdftotext entities |
| roster | Key-value table | pdfplumber labels + pdftotext content |
| systems | 3-column table | pdfplumber structure + pdftotext fills |

## Directory Structure

```
DART/
├── multi_source_interpreter.py  # PRIMARY - Multi-source synthesis engine
├── pdf_converter/               # PDF extraction utilities
├── batch_output/
│   ├── combined/               # *_combined.json input files
│   ├── synthesized/            # *_synthesized.json intermediate
│   └── html/                   # *_synthesized.html output files
└── templates/                  # Reference templates
```

## WCAG 2.2 AA Features

- Skip navigation links
- ARIA landmarks (main, nav, contentinfo)
- Semantic heading hierarchy (h1 → h2)
- Table scope attributes
- Contact cards with microdata
- Focus management (scroll-margin-top)
- Reduced motion support
- Dark mode support

## System Dependencies

- `poppler-utils` (pdftotext/pdfinfo)
- `pdfplumber` (table extraction)
- `tesseract-ocr` (optional, for OCR validation)
