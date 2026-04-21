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

## Ontology-aware pipeline (Waves 12–16)

Raw-text pdftotext conversion is now a 4-phase pipeline under
`DART/converter/`, replacing the ~900-LOC regex monolith that used to
live in `MCP/tools/pipeline_tools.py::_raw_text_to_accessible_html`.

Wave 16 adds a dual-extraction pre-phase so pdfplumber tables,
PyMuPDF figures, and Tesseract OCR text all survive into the HTML
output — pdftotext is still the only hard dependency, every other
extractor is optional and degrades gracefully.

### Phases

1. **Segment** (`block_segmenter.py`) — split raw pdftotext output into
   `RawBlock` instances on blank-line / form-feed boundaries; compute
   stable `block_id` hashes + neighbor context.
2. **Classify** (`heuristic_classifier.py` or `llm_classifier.py`) —
   assign exactly one `BlockRole` from the 35-value enum in
   `block_roles.py`. Heuristic classifier is the offline default; LLM
   classifier routes through Claude via `MCP/orchestrator/llm_backend.py`
   for ambiguous blocks.
3. **Template** (`block_templates.py`) — render each classified block
   with DPUB-ARIA + schema.org + microdata. Every role has exactly one
   registered template. `data-dart-block-role` / `data-dart-block-id` /
   `data-dart-confidence` provenance attributes survive unchanged.
4. **Assemble** (`document_assembler.py`) — wrap rendered blocks in the
   full HTML document shell, inject Dublin Core / schema.org JSON-LD /
   accessibility summary in `<head>`, run post-assembly cross-reference
   resolution.

### Wave 15 `<head>` enrichment

The assembler emits in this order:

1. `<meta charset>`, `<meta viewport>`, `<title>`
2. Dublin Core `<meta name="DC.*">` tags derived from the caller's
   `metadata` dict: `DC.title`, `DC.creator`, `DC.date`, `DC.language`
   (defaults to `en`), `DC.rights`, `DC.subject`. Missing values are
   silently omitted — no empty `content=""` tags.
3. Document-level schema.org JSON-LD. `@type` switches on
   `metadata["document_type"]`: `"arxiv"` → `ScholarlyArticle`,
   `"textbook"` → `Book`, else `CreativeWork`. `hasPart` lists every
   `CHAPTER_OPENER` block; URLs point at the article's `id="chap-N"`
   anchor (same template emits that id).
4. Accessibility summary JSON-LD with `accessMode`,
   `accessibilityFeature`, `accessibilitySummary` advertising the
   WCAG 2.2 AA feature set the bundled templates + CSS provide.
5. WCAG 2.2 AA `<style>` bundle from `DART/templates/wcag22_css.py`.

### Cross-reference resolution

Runs as the last step in `assemble_html` (`DART/converter/cross_refs.py`).
Rewrites in-text references into real anchors **only when the target
exists in the classified block list**:

| Phrase | Rewrite | Target source |
|--------|---------|---------------|
| `Chapter N` / `See Chapter N` | `<a href="#chap-N">` | `CHAPTER_OPENER` attribute or scraped from raw text |
| `Figure N.M` | `<a href="#fig-N-M">` | `FIGURE.number` attribute |
| `Section N.M` | `<a href="#sec-N-M">` | `SECTION_HEADING.number` or heading text scrape |
| `[N]` citation marker | `<a href="#ref-N">` | `BIBLIOGRAPHY_ENTRY.number` or scraped `[N]` prefix |

Orphan references (no matching target) are silently left as plain text —
no broken links ever emitted. Already-linked spans
(`<a>See Chapter 1</a>`) are not double-wrapped. The `<head>` block is
passed through untouched so `<title>` / Dublin Core text never receives
accidental anchors.

### Toggles

| Env var | Effect |
|---------|--------|
| `DART_LLM_CLASSIFICATION=true` | Route classification through Claude via `LLMClassifier` instead of the heuristic regex path. Requires an injected `LLMBackend`. |
| `DART_LEGACY_CONVERTER=true` | Force the pre-Wave-15 regex-driven `_raw_text_to_accessible_html_legacy` path in `MCP/tools/pipeline_tools.py`. One-release safety fallback; do not extend. |

### Entry point

```python
from DART.converter import convert_pdftotext_to_html

html = convert_pdftotext_to_html(
    raw_text,
    title="My Book",
    metadata={
        "authors": "Jane Doe, John Smith",
        "date": "2026-04-20",
        "language": "en",
        "rights": "CC BY 4.0",
        "subject": "accessibility, WCAG",
        "document_type": "textbook",
    },
)
```

`MCP/tools/pipeline_tools.py::_raw_text_to_accessible_html` is now a
thin orchestrator that delegates to this entry point unless the legacy
flag is set.

### Wave 16 dual-extraction flow

Raw pdftotext extraction is the baseline; `DART.converter.extractor`
adds a structured-extraction layer that preserves tables, figures, and
OCR content when upstream dependencies are available.

```python
from DART.converter import (
    aconvert_pdftotext_to_html,
    default_classifier,
    extract_document,
    segment_extracted_document,
)
from DART.converter.document_assembler import assemble_html

# 1. Extract — pdftotext always, pdfplumber / PyMuPDF / Tesseract
#    contribute additively when available.
doc = extract_document("/path/to/book.pdf", llm=my_backend)  # llm optional

# 2. Segment — combine prose blocks with hinted structured blocks.
blocks = segment_extracted_document(doc)

# 3. Classify — extractor hints short-circuit the classifier at
#    confidence 1.0 so pdfplumber tables never get prose-classified.
classifier = default_classifier(llm=my_backend)
classified = classifier.classify_sync(blocks)  # or ``await classifier.classify``

# 4. Assemble — same document assembler as the raw-text-only path.
html = assemble_html(classified, title="My Book", metadata={})
```

`ExtractedDocument` shape:

| Field | Type | Notes |
|-------|------|-------|
| `raw_text` | `str` | pdftotext output (required — the only hard dep). |
| `source_pdf` | `str` | Source path. |
| `pages_count` | `int` | Derived from form-feed markers when present. |
| `tables` | `list[ExtractedTable]` | pdfplumber extractions; empty on failure. |
| `figures` | `list[ExtractedFigure]` | PyMuPDF extractions; empty on failure. |
| `ocr_text` | `Optional[str]` | Populated only when Tesseract + PyMuPDF both available. |

`ExtractedTable.{page, bbox, header_rows, body_rows, caption}` —
header_rows / body_rows are lists of stringified cells.

`ExtractedFigure.{page, bbox, image_path, alt_text, caption}` — alt_text
is populated only when an `LLMBackend` is injected into
`extract_document(..., llm=...)`.

### Wave 16 structured block integration

`RawBlock` gained two optional fields: `extractor_hint: BlockRole`
(the segmenter stamps this when the block was produced by a
structured extractor) and `extra: dict` (the structured payload —
header rows / body rows / image path / alt / caption). Both default
to empty so pre-Wave-16 callers stay compatible.

The classifier layer honours the hint:

* `HeuristicClassifier` — emits the hinted role at confidence 1.0 and
  forwards `extra` into `ClassifiedBlock.attributes`.
* `LLMClassifier` — skips hinted blocks entirely (they never appear in
  the prompt, so the backend is never asked to classify e.g. a table's
  row text as prose). Hinted blocks carry
  `classifier_source="extractor_hint"` in the output.

### Wave 16 TABLE / FIGURE / FORMULA_MATH templates

* **TABLE** — accepts both legacy (`headers` + `rows`) and structured
  (`header_rows` + `body_rows`) attribute shapes. In the structured
  path, `<thead>` cells carry `scope="col"` and the first cell of
  every `<tbody>` row carries `scope="row"`.
* **FIGURE** — accepts both `src` (legacy) and `image_path` (Wave 16)
  for the image source. Emits `<figure>` + `<img alt>` +
  `<figcaption>` with schema.org `ImageObject` microdata.
* **FORMULA_MATH** — delegates to `DART.converter.mathml` so LaTeX
  delimiters (`$...$`, `\(...\)`, `\[...\]`) and plain
  equation-on-a-line patterns (`E = mc^2`) all render as:

  ```html
  <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" ...>
    <semantics>
      <mtext>{raw_formula}</mtext>
      <annotation encoding="text/plain">{fallback}</annotation>
    </semantics>
  </math>
  ```

  No LaTeX-to-MathML compilation — the `<annotation>` arm preserves
  the raw source for assistive tech. Full LaTeX fidelity is out of
  scope for this wave.

### Wave 16 pipeline_tools plumbing

`MCP/tools/pipeline_tools.py::_raw_text_to_accessible_html` takes an
optional `source_pdf` kwarg:

* `source_pdf` omitted → raw-text-only path (Wave 15 behaviour).
* `source_pdf` provided → routes through
  `extract_document(source_pdf, llm=...)` so pdfplumber tables,
  PyMuPDF figures, and OCR text contribute structured blocks. Any
  extractor failure degrades back to the raw-text path — the dispatch
  never blocks on an optional dep.

`extract_and_convert_pdf` (the MCP pipeline tool) already passes
`source_pdf` so end-to-end textbook runs pick up the enrichment
automatically.

### Wave 17 figure persistence

Wave 16 detected figures but left `ExtractedFigure.image_path` empty,
so the rendered HTML had `<figure>` wrappers with empty `<img src>`
and a literal `(figure)` placeholder caption. Wave 17 closes that gap:

* `extract_document(pdf_path, *, llm=None, figures_dir=None)` —
  optional `figures_dir` kwarg. When set, image bytes returned by the
  PyMuPDF extractor are written to
  `figures_dir / {page:04d}-{hash8}.{ext}` where `hash8` is the first
  eight hex chars of `sha256(bytes)` and `ext` is derived from the
  detected format (`png`/`jpeg`/…). `ExtractedFigure.image_path` is
  set to the **relative** filename (no directory prefix) so the
  caller / assembler layer decides the path written into `<img src>`.
  Re-running on the same bytes is idempotent — same filename, no
  double-write.
* `figures_dir=None` (the default) preserves Wave 16 behaviour: no
  disk I/O, `image_path` stays empty.
* `MCP/tools/pipeline_tools.py::_raw_text_to_accessible_html`
  auto-derives a sibling `{stem}_figures/` directory next to the
  output HTML when called with `output_path=<html path>`, so a PDF at
  `/foo/bates.pdf` converted to `/foo/out/bates.html` persists
  figures under `/foo/out/bates_figures/` and the HTML carries
  relative `<img src="bates_figures/...png">` entries (the bundle is
  portable). Explicit `figures_dir=<path>` wins. Neither set → a
  tempdir fallback (not portable, but keeps the end-to-end round trip
  working for tests / ad-hoc runs).
* Caption detection is best-effort — the extractor scans the matching
  pdftotext page for lines matching
  `^(Figure|Fig\.?|Image) N[.M]?\s*[:\-–—]` (case-insensitive) and
  binds the first unclaimed match per page. When no pattern match is
  found it falls back to PyMuPDF's `nearby_caption`. When both are
  empty, `caption` stays `None` and the template degrades to a
  caption-less `<figure>` — it **never** emits the literal
  placeholder string `"(figure)"`.
* Alt-text still requires an injected `LLMBackend` (Wave 16 behaviour
  unchanged). When no backend is provided, the FIGURE template emits
  `alt="" role="presentation"` — a WCAG 2.2 AA decorative fallback —
  rather than an empty `alt=""` (which screen readers read as the
  filename) or a missing attribute (invalid HTML).

Public MCP `extract_and_convert_pdf` (`MCP/tools/dart_tools.py`)
accepts an optional `figures_dir: Optional[str]` kwarg for callers
that need to override the sibling-dir derivation; currently only the
Wave-16 dual-extraction path honours it (the legacy
`PDFToAccessibleHTML` strategy ignores it).
