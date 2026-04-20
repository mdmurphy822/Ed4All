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

## Entry Points

DART has two entry points serving different purposes:

- **`convert.py`** — Convenience wrapper for PDF to WCAG HTML conversion. Calls `pdf_converter` directly on a raw PDF file.
- **`multi_source_interpreter.py`** — Multi-source synthesis engine for combined JSON inputs (pdftotext + pdfplumber + OCR). This is the preferred path when pre-extracted source data is available.

```python
# Multi-source synthesis (PREFERRED when combined JSON exists)
from multi_source_interpreter import convert_single_pdf, batch_synthesize_all

# Single file conversion from combined JSON
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
| `validate_dart_markers` | Validate DART output markers. Wired as the `dart_markers` gate on `batch_dart` and `textbook_to_course` (Wave 6). |
| `get_dart_status` | Get DART capabilities |
| `list_available_campuses` | List available combined JSONs |
| `extract_and_convert_pdf` | Extract and convert a single PDF to accessible HTML |

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
| `parse_sections_from_text()` | Parse sections from clean pdftotext output |
| `validate_wcag()` | Run WCAG 2.2 AA validation on generated HTML |
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
├── batch_output/               # Created at runtime during batch processing
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

## Source provenance (Wave 8)

DART emits per-block source attribution through three linked artifacts so
downstream Courseforge / Trainforge can cite the PDF region every claim
derives from. Canonical shape: `schemas/knowledge/source_reference.schema.json`.
Design spec: `plans/source-provenance/design.md`.

### Per-section record shape (`*_synthesized.json`)

```jsonc
{
  "section_id": "s3",
  "section_type": "contacts",
  "section_title": "Campus Contacts",
  "page_range": [3, 4],
  "provenance": {
    "sources": ["pdfplumber", "pdftotext"],
    "strategy": "pdfplumber_headers+pdftotext_entities",
    "confidence": 0.87
  },
  "data": { "contacts": [ ... ] },
  "sources_used": { "structure": "...", "content": "..." }  // legacy back-compat
}
```

`sources_used` is retained so legacy consumers keep working; `provenance.strategy`
is the new canonical location.

### Per-block envelope

Every leaf value that came from multi-source matching is wrapped in a
`{value, source, pages, confidence, method}` envelope. Example contact:

```jsonc
{
  "block_id": "s3_c0",
  "name":  "Jane Doe",
  "email": "jdoe@campus.edu",
  "name_provenance":  {"value": "Jane Doe",        "source": "pdfplumber", "pages": [3], "confidence": 1.0, "method": "table_header"},
  "email_provenance": {"value": "jdoe@campus.edu", "source": "pdftotext",  "pages": [3], "confidence": 0.8, "method": "name_pattern"}
}
```

`block_id` is positional (`s3_c0`) by default and becomes a content-hash
(16-hex) when `TRAINFORGE_CONTENT_HASH_IDS=1` — both shapes validate
against the canonical `sourceId` pattern `^dart:{slug}#{block_id}$`.

### Confidence scale (canonical)

| Value | Meaning |
|-------|---------|
| `1.0` | Direct table extraction (pdfplumber structured row/cell) |
| `0.8` | Name-pattern match (e.g. `jdoe@` matching Jane Doe) |
| `0.6` | Proximity match (nearest email/phone to a name in the text stream) |
| `0.4` | Derivation / synthesis (contact reconstructed from email local-part) |
| `0.2` | OCR-only fallback (no pdftotext/pdfplumber corroboration) |

Documented in `DART/multi_source_interpreter.py` as module-level constants
(`CONFIDENCE_DIRECT_TABLE`, `CONFIDENCE_NAME_PATTERN`, etc.). Downstream
validators (Courseforge source-router, Trainforge inference rules) read
these values; do not invent new scale points.

### `data-dart-*` HTML attributes

Emitted on every `<section>` + `.contact-card` + `<tr>` in multi-source
output. Per the design doc's P2 decision, attributes stop at the section /
component wrapper level — never on every `<p>` / `<li>` / `<tr>` in prose,
to keep HTML size bounded at textbook scale.

| Attribute | Shape | Notes |
|-----------|-------|-------|
| `data-dart-block-id` | `"s3"` or `"s3_c0"` or 16-hex | Matches `block_id` in synthesized JSON |
| `data-dart-source` | `pdftotext \| pdfplumber \| ocr \| synthesized \| claude_llm` | Primary source enum |
| `data-dart-sources` | Comma-joined list | Only emitted when multi-source |
| `data-dart-pages` | `"3"` or `"3-5"` or `"3,5,7"` | Omitted when unknown |
| `data-dart-confidence` | 2-decimal float | Omitted when `1.0` (the implicit default) |
| `data-dart-strategy` | Free-form | Mirrors `provenance.strategy` in JSON |

The legacy `claude_processor` / `_generate_html_from_structure` path stamps
only a minimal `data-dart-source="claude_llm"` on the section wrapper (P5
decision — full parity is non-goal).

### Staging handoff

`MCP/tools/pipeline_tools.py::stage_dart_outputs` copies three artifacts
to the Courseforge staging dir and role-tags them in
`staging_manifest.json`:

```jsonc
{
  "files": [
    {"path": "science_of_learning.html",                "role": "content"},
    {"path": "science_of_learning_synthesized.json",    "role": "provenance_sidecar"},
    {"path": "science_of_learning.quality.json",        "role": "quality_sidecar"}
  ]
}
```

### Validator

`lib/validators/dart_markers.py` checks for `data-dart-source` /
`data-dart-block-id` on every `<section>` at **warning** severity only.
Promotion to critical is deferred to Wave 9 to let new emission paths
shake out edge cases first.

### Known gaps (deferred)

- **Real per-block page tracking** — `clean_text` strips form feeds at L116;
  per the design doc, keeping form feeds is a separate refactor. Wave 8
  ships section-level `page_range` from fixture estimates; per-block
  `pages` stays empty when genuinely unknown.
- **OCR-quality sub-signal** — OCR-only blocks score `0.2` regardless of
  Tesseract per-word confidence. A separate `ocr_quality` field is
  follow-up work.
