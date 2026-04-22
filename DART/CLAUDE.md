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

## Source provenance

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

### `dart:{slug}` normalization (canonical)

The `{slug}` half of the sourceId is derived from the staged HTML's
`path.stem` with a **gentle** transform:

```
slug = path.stem.lower().replace(" ", "-")
```

Underscores, hyphens, and digits pass through unchanged. This matches
three coordinated consumers that ALL derive their "valid block ID
universe" from the same `path.stem`:

- `lib/validators/content_grounding.py::_resolve_valid_block_ids` —
  `slug = html_path.stem.lower().replace(" ", "-")`.
- `lib/validators/source_refs.py` — same rule.
- `MCP/tools/pipeline_tools.py::_build_source_module_map` (Wave 9
  source-router) — `slug = sidecar.stem.replace("_synthesized", "").lower().replace(" ", "-")`.
- `MCP/tools/_content_gen_helpers.py::_topic_source_references`
  (Wave 35 content-generator) — `slug = stem.lower().replace(" ", "-")`.

Do **not** use `lib.ontology.slugs.canonical_slug` for sourceId slugs
— it collapses underscores into one token, producing slugs like
`batesteachingdigitalageaccessible` instead of
`bates_teaching_digital_age_accessible`, and the validator's valid
block-id set won't contain the collapsed form. `canonical_slug` is
still the right helper for key-concept slugs (see Courseforge
`_content_gen_helpers.synthesize_objectives_from_topics`), just not
for the `dart:{slug}` portion of a sourceId.

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
| `data-dart-source` | `pdftotext \| pdfplumber \| pymupdf \| ocr \| synthesized \| claude_llm \| dart_converter` | Primary source enum. `dart_converter` is the default emitted value for heuristic-classifier blocks. |
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
`data-dart-block-id` on every `<section>`. Absent-attribute is a
warning (so legacy HTML keeps passing); present-but-empty
(`data-dart-source=""`, `data-dart-block-id=""`) is a critical
failure (`EMPTY_DATA_DART_SOURCE`, `EMPTY_DATA_DART_BLOCK_ID`).

### Known gaps

- **Real per-block page tracking** — `clean_text` strips form feeds at
  L116; keeping form feeds is a separate refactor. Section-level
  `page_range` ships from fixture estimates; per-block `pages` stays
  empty when genuinely unknown.
- **OCR-quality sub-signal** — OCR-only blocks score `0.2` regardless of
  Tesseract per-word confidence. A separate `ocr_quality` field is
  follow-up work.

## Multi-extractor pipeline

Raw-text pdftotext conversion is a 4-phase pipeline under
`DART/converter/`. pdftotext provides the text baseline; a
dual-extraction layer adds pdfplumber tables, PyMuPDF figures /
TOC / metadata / text spans / links, and Tesseract OCR when
available. A reconciliation layer picks the best source per data
type. pdftotext is the only hard dependency; every other extractor
is optional and degrades gracefully.

### Extractor peers

| Extractor | Contributes | Status |
|-----------|-------------|--------|
| **pdftotext** | `raw_text` (line/column prose) | Hard dep. |
| **pdfplumber** | `tables` (structure / bordered tables) | Optional. Primary for tables. |
| **PyMuPDF (fitz)** | `figures` (raster bytes + caption) | Optional. |
| **PyMuPDF (fitz)** | `toc` (native outline / bookmarks) | Optional. |
| **PyMuPDF (fitz)** | `pdf_metadata` (title / author / dates) | Optional. |
| **PyMuPDF (fitz)** | `text_spans` (bbox + font size + bold/italic) | Optional. |
| **PyMuPDF (fitz)** | `links` (URI + internal goto) | Optional. |
| **PyMuPDF (fitz)** | `tables` (find_tables fallback) | Optional. |
| **Tesseract** | `ocr_text` (scanned / image-only pages) | Optional. |

Reconciliation rules:

* **Tables** — pdfplumber wins when it returns non-empty. Only when
  pdfplumber yields zero tables does PyMuPDF's `find_tables()` fill
  in (textbook-style text-heavy PDFs rarely work for pdfplumber's
  border-based detection). Each `ExtractedTable` carries a `source`
  attribute (`"pdfplumber"` | `"pymupdf"`), threaded through the
  `<table>` as `data-dart-table-extractor="..."` for debuggability.
* **Headings (font-size promoter)** — when `text_spans` is
  populated, the heuristic classifier promotes fallback `PARAGRAPH`
  blocks whose dominant span renders at ≥ 1.5× the document's median
  body font size to `SUBSECTION_HEADING`, and ≥ 1.9× to
  `SECTION_HEADING`. Bold is a secondary tiebreaker that fires
  between 1.15× and 1.5×. Promotion never overrides an explicit
  regex-classified role — it only lifts fallback paragraphs.
* **Metadata merge** — PyMuPDF's `doc.metadata` fills blanks in the
  caller-supplied `metadata` dict (`title`, `authors`, `subject`,
  `date`) but never overrides caller values. `creationDate` is
  normalised from PDF-spec format (`D:YYYYMMDDHHmmSS±OFS`) to ISO
  8601 (`YYYY-MM-DD`).
* **TOC** — when `doc.toc` is non-empty, the segmenter prepends a
  synthetic `TOC_NAV` block carrying the structured entries list.
  The `TOC_NAV` template renders `<nav role="doc-toc">` with nested
  `<ol>`/`<li>` keyed by level; entries link to `#chap-N` /
  `#sec-N-M` when the title matches those patterns, else `#page-N`.
* **Links** — external `uri` links stay out of scope (prose wraps
  them client-side). Internal `goto` links are surfaced to the
  cross-reference resolver as `targets["page"]` so page-scoped
  anchors can resolve in later waves; existing rewriters
  (`Chapter N`, `Figure N.M`, `Section N.M`, `[N]`) are untouched.

### `ExtractedDocument` fields

```
toc:          list[ExtractedTOCEntry]  # {level, title, page}
pdf_metadata: dict                     # normalised; ISO dates
text_spans:   list[ExtractedTextSpan]  # {page, bbox, text, font_size, font_name, is_bold, is_italic}
links:        list[ExtractedLink]      # {page, bbox, uri, dest_page}
```

Every PyMuPDF-sourced field defaults empty and degrades to `[]` / `{}`
when PyMuPDF is unavailable (import fails or the document won't open).

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

### `<head>` enrichment

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

### Dual-extraction flow

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
| `tables` | `list[ExtractedTable]` | pdfplumber primary; PyMuPDF fallback when pdfplumber=[]. Each table carries `source` ∈ {`pdfplumber`, `pymupdf`}. |
| `figures` | `list[ExtractedFigure]` | PyMuPDF extractions; empty on failure. |
| `ocr_text` | `Optional[str]` | Populated only when Tesseract + PyMuPDF both available. |
| `toc` | `list[ExtractedTOCEntry]` | PyMuPDF native outline. Empty when no outline. |
| `pdf_metadata` | `dict` | Normalised from `doc.metadata`. Dates in ISO 8601. |
| `text_spans` | `list[ExtractedTextSpan]` | PyMuPDF spans: font size + bbox + bold/italic flags. Used by the font-size heading promoter. |
| `links` | `list[ExtractedLink]` | PyMuPDF hyperlinks. `uri` for external, `dest_page` for internal. |

`ExtractedTable.{page, bbox, header_rows, body_rows, caption, source}` —
header_rows / body_rows are lists of stringified cells; `source` defaults
to `"pdfplumber"` for backward compat.

`ExtractedFigure.{page, bbox, image_path, alt_text, caption}` — alt_text
is populated only when an `LLMBackend` is injected into
`extract_document(..., llm=...)`.

`ExtractedTOCEntry.{level, title, page}` — native PDF bookmarks. Levels
are 1-indexed.

`ExtractedTextSpan.{page, bbox, text, font_size, font_name, is_bold, is_italic}`
— used by `HeuristicClassifier(text_spans=..., median_body_font_size=...)`
for font-size-based heading promotion.

`ExtractedLink.{page, bbox, uri, dest_page}` — external URIs or internal
1-indexed page targets.

### Structured block integration

`RawBlock` carries two optional fields: `extractor_hint: BlockRole`
(the segmenter stamps this when the block was produced by a
structured extractor) and `extra: dict` (the structured payload —
header rows / body rows / image path / alt / caption). Both default
empty so callers that feed raw-text-only input keep working.

The classifier layer honours the hint:

* `HeuristicClassifier` — emits the hinted role at confidence 1.0 and
  forwards `extra` into `ClassifiedBlock.attributes`.
* `LLMClassifier` — skips hinted blocks entirely (they never appear in
  the prompt, so the backend is never asked to classify e.g. a table's
  row text as prose). Hinted blocks carry
  `classifier_source="extractor_hint"` in the output.

### TABLE / FIGURE / FORMULA_MATH templates

* **TABLE** — accepts both legacy (`headers` + `rows`) and structured
  (`header_rows` + `body_rows`) attribute shapes. In the structured
  path, `<thead>` cells carry `scope="col"` and the first cell of
  every `<tbody>` row carries `scope="row"`.
* **FIGURE** — accepts both `src` and `image_path` for the image
  source. Emits `<figure>` + `<img alt>` + `<figcaption>` with
  schema.org `ImageObject` microdata.
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

### `pipeline_tools` plumbing

`MCP/tools/pipeline_tools.py::_raw_text_to_accessible_html` takes an
optional `source_pdf` kwarg:

* `source_pdf` omitted → raw-text-only path.
* `source_pdf` provided → routes through
  `extract_document(source_pdf, llm=...)` so pdfplumber tables,
  PyMuPDF figures, and OCR text contribute structured blocks. Any
  extractor failure degrades back to the raw-text path — the dispatch
  never blocks on an optional dep.

`extract_and_convert_pdf` (the MCP pipeline tool) already passes
`source_pdf` so end-to-end textbook runs pick up the enrichment
automatically.

### Figure persistence

Figure bytes detected by the PyMuPDF extractor are written to disk so
the rendered HTML carries real `<img src>` pointers (never empty
`<img>` or literal `(figure)` placeholder captions).

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
* `figures_dir=None` (the default) skips disk I/O; `image_path` stays
  empty.
* `MCP/tools/pipeline_tools.py::_raw_text_to_accessible_html`
  auto-derives a sibling `{stem}_figures/` directory next to the
  output HTML when called with `output_path=<html path>`, so a PDF at
  `/foo/textbook.pdf` converted to `/foo/out/textbook.html` persists
  figures under `/foo/out/textbook_figures/` and the HTML carries
  relative `<img src="textbook_figures/...png">` entries (the bundle is
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
* Alt-text requires an injected `LLMBackend`. When no backend is
  provided, the FIGURE template emits `alt="" role="presentation"` —
  a WCAG 2.2 AA decorative fallback — rather than an empty `alt=""`
  (which screen readers read as the filename) or a missing attribute
  (invalid HTML).

Public MCP `extract_and_convert_pdf` (`MCP/tools/dart_tools.py`)
accepts an optional `figures_dir: Optional[str]` kwarg for callers
that need to override the sibling-dir derivation; only the
dual-extraction path honours it (the legacy `PDFToAccessibleHTML`
strategy ignores it).

## Output contract for downstream consumers

Courseforge source-router, the `dart_markers` gate,
`semantic_structure_extractor`, `stage_dart_outputs`, and
`archive_to_libv2` rely on this output contract:

### `class="dart-document"` + `class="dart-section"` emit

- The assembler stamps `class="dart-document"` on the `<main>` wrapper
  (one per document).
- Every top-level `<section>` / `<article>` / `<aside>` wrapper template
  carries `class="dart-section"` — existing classes (`pullquote`,
  `callout callout-info`, etc.) are preserved and prepended.
- When a document classifies into only leaf blocks (no structural
  wrappers fire), the assembler wraps the body in a fallback
  `<section class="dart-section" aria-labelledby="main-content-heading">`
  so the `dart_markers` gate's `aria_sections` + `dart_semantic_classes`
  critical checks always pass.

### `data-dart-source` source enum

The `data-dart-source` attribute is stamped on every wrapper.
Routing (see `DART/converter/block_templates.py::_data_dart_source_value`):

| classifier_source | upstream extractor | emitted value |
|-------------------|--------------------|--------------|
| `extractor_hint`  | `pdfplumber`       | `pdfplumber` |
| `extractor_hint`  | `pymupdf`          | `pymupdf`    |
| `extractor_hint`  | `pdftotext`        | `pdftotext`  |
| `llm`             | *                  | `claude_llm` |
| `heuristic` / default | *              | `dart_converter` |

### Provenance attribute placement rule

Attributes stop at the **section / component wrapper level**. Never on
every `<p>` / `<span>` / `<li>` / `<h3>` / `<cite>` / `<a>` /
`<figcaption>` in prose — those are leaf nodes, and the enclosing
wrapper carries the provenance. The canonical leaf-role set is
`DART/converter/block_templates._WAVE19_LEAF_ROLES`. This keeps HTML
size bounded at textbook scale (measured ~20% reduction vs. full-leaf
stamping on a several-hundred-page textbook).

### Sidecar emit

`MCP/tools/pipeline_tools.py::_raw_text_to_accessible_html` emits two
sidecars next to the HTML when `output_path` is provided (mirrors the
figure-persistence tempdir guard):

- `{stem}_synthesized.json` — per-section provenance sidecar in the
  canonical shape (`sections[].section_id`, `section_title`,
  `section_type`, `page_range`, `provenance{sources, strategy,
  confidence}`, `data{text, block_roles, attributes, head_block_id}`).
  Consumed by `_build_source_module_map` (the Courseforge
  source-router). Built by `DART.converter.sidecars.build_synthesized_sidecar`.
- `{stem}.quality.json` — WCAG + confidence aggregate sidecar.
  Consumed by `archive_to_libv2` when populating
  `{course}/quality/*.quality.json`. Built by
  `DART.converter.sidecars.build_quality_sidecar`.

### `{stem}_figures/` propagation

Both `stage_dart_outputs` (MCP tool + registry variant) and
`archive_to_libv2` copy a sibling `{stem}_figures/` directory
alongside the HTML when present; a missing dir is silently skipped.

### `data-dart-pages` — scope

Both emission paths use the plural `data-dart-pages` form:

- **Converter path** (`DART/converter/block_templates.py::_provenance_attrs`):
  emits `data-dart-pages="N"` on every section / component wrapper
  when the block has a known page. The value is drawn from
  `raw.extra["page_label"]` when the page-chrome detector supplied a
  printed-page label for that page; otherwise falls back to the
  physical form-feed-derived `raw.page`.
- **Multi-source synthesis path** (`DART/multi_source_interpreter.py`):
  emits `data-dart-pages` as a range (`"3-5"`) or comma-joined list
  (`"3,5,7"`) covering a section `page_range`.

`lib/validators/dart_markers.py` enforces only
`data-dart-source` + `data-dart-block-id` presence; the page
attribute remains optional (omitted when no page is known).

## List detection

pdftotext output faithfully reproduces list markers (`•`, `·`, `▪`,
`●`, `◦`, `○`, `▸`, `►`, `-`, `*`, `1.`, `1)`, `a.`, `a)`, `i.`,
`(1)`) as literal leading characters on each item's block. The
heuristic classifier promotes marker-led blocks to `LIST_ITEM` and
the assembler groups consecutive runs into synthesized
`LIST_UNORDERED` / `LIST_ORDERED` blocks.

### Marker charset

| Family | Characters |
|--------|------------|
| Unicode bullets | `•`, `·`, `▪`, `●`, `◦`, `○`, `▸`, `►`, `■`, `□`, `◼`, `◾` |
| ASCII markers | `-`, `*` (only when followed by whitespace AND body starts with capital / digit) |
| Numbered | `1.`, `1)`, `(1)`, `12.` |
| Alphabetic | `a.`, `a)`, `b.`, `b)` |
| Roman numeral | `i.`, `iv.`, `vii)` (lowercase / uppercase) |

Detection lives in
`DART/converter/heuristic_classifier.py::_match_list_marker` and fires
after bibliography + chapter classification so existing
higher-priority rules win. A guard drops long prose-like blocks that
happen to start with a marker but carry no embedded sibling markers
and look like multi-sentence prose.

### Multi-item expansion

pdftotext often fuses sibling list items onto one logical block
(`"1. Foo. 2. Bar. 3. Baz."` or `"• Foo • Bar • Baz"`). The classifier
runs a post-promotion expander (`_maybe_expand_numbered_run`) that
splits such fused blocks into one `LIST_ITEM` per detected item.
Ordered expansion requires ascending numbers within +1–+2 of each
other; unordered expansion triggers whenever embedded unicode
bullets appear after the first item body. Each expanded item
receives a deterministic `block_id` suffix (`{original}#2`, `#3`)
so downstream anchors stay unique.

### Grouping rule

`DART/converter/document_assembler.py::_group_consecutive_lists`
folds a run of consecutive `LIST_ITEM` blocks into a single
synthesized `LIST_UNORDERED` / `LIST_ORDERED` block with
`attributes.items = [{text, marker, marker_type, sub_items?}, ...]`.
A run breaks on any non-list block OR a `marker_type` change
(unordered → ordered starts a new list). Single-item runs still
emit a one-item `<ul>` / `<ol>` wrapper — we never leave a stray
`<li>` without a parent.

### Nesting heuristic (best-effort)

When a single `RawBlock` carries multiple lines AND the trailing
lines are indented (≥ 4 leading spaces) AND start with a marker
themselves, those become `sub_items` on the parent item. In the
current pipeline the segmenter whitespace-collapses blocks before
the classifier sees them, so this path rarely fires end-to-end —
it's retained so callers that feed raw multi-line blocks
(tests, future waves that preserve layout) still get nested
output.

### Template output + attribute placement

The `<ul>` / `<ol>` **is** the dart-section component wrapper.
It carries `class="dart-section"`, `data-dart-block-role`,
`data-dart-block-id`, `data-dart-source`, `data-dart-pages`
(propagated from the first item's page), and
`data-dart-confidence`. `<li>` children are leaves and **never**
carry `data-dart-*` attributes. When every item in an unordered
list shares the same bullet glyph, a style-hint class
(`list-dot` / `list-square` / `list-circle` / `list-triangle` /
`list-dash` / `list-asterisk` / ...) is appended after
`dart-section` so CSS can reflect the authored marker variant
while the markup stays semantic. Ordered lists emit a `start="N"`
attribute only when the first authored marker is not the
default starting value (1 / a / i) — avoids clutter on the
common 1-indexed case.

### Stray LIST_ITEM fallback

If a `LIST_ITEM` escapes grouping (shouldn't happen normally),
`_tpl_list_item` emits a single-item `<ul>` / `<ol>` wrapper
rather than a naked `<li>` — keeps the HTML valid.

## Page chrome detection

pdftotext faithfully reproduces **running headers / running footers /
page numbers** as text lines in every page of its output. For a
long textbook (hundreds of pages), the result is hundreds of
spurious content-polluting `<p>` blocks in the emitted HTML.

`DART/converter/page_chrome.py` runs between pdftotext extraction and
block segmentation to detect + strip that chrome.

### Algorithm

Primary signal: **frequency**. Split pdftotext output on form-feed,
collect the top-3 and bottom-3 non-blank lines of every page, and
count how often each normalised line (after stripping trailing
digits) appears across pages. Any line above the configured
`min_repeat_fraction` (default 0.3 = 30% of pages) is chrome.

Secondary signal (when PyMuPDF `text_spans` are available): **bbox
layout confirmation**. A frequency candidate whose bbox lives in the
top 10% or bottom 10% of the page is upgraded to confirmed chrome.
Spans are only used to upgrade — never to filter out a frequency
hit — so the detector works end-to-end when PyMuPDF is missing.

### Page-number extraction

When a chrome line ends in digits (`"<Book Title> 164"`,
`"Chapter 3 — 47"`, or just `"164"`), the detector splits the fixed
prefix from the variable page-number tail and remembers
`{page_number_1_indexed: original_chrome_line}` on
`PageChrome.page_number_lines`. The segmenter then stamps the
extracted numeric label into every block on that page as
`RawBlock.extra["page_label"]`, so `data-dart-pages` surfaces the
book's printed page number (which is what downstream Courseforge +
Trainforge citations need) rather than the PDF's physical page.

### False-positive guards

Applied after frequency thresholding:

- **Long lines** (≥ 80 chars): never chrome — running headers are
  short by convention.
- **Heading markers** (`Chapter N`, `Section N.M`, `Part N`, etc.):
  excluded even when they repeat — they're structural content.
- **Short fixed-prefix with variable tail** (< 3 chars): excluded as
  ambiguous (catches numbered-list bleed).
- **Bare page numbers** (a lone `"164"` appearing on most pages):
  legitimate chrome — detected via a page-number-only sentinel key.

### Document-level signal

`PageChrome` is returned on `ExtractedDocument.page_chrome` and
surfaced into `{stem}_synthesized.json` under
`document_provenance.page_chrome_detected` as
`{headers: [...], footers: [...], pages_numbered: N}` for
debuggability. Short documents (< 4 pages) return an empty
`PageChrome` and no stripping occurs — there's not enough signal.

### Downstream consumers

- `data-dart-pages` on every section / component wrapper (see above).
- Per-section `page_range: [first_page, last_page]` emitted by the
  synthesized sidecar; populated from per-block `raw.page` via
  form-feed tracking.

### `doc-chapter` extractor path

`lib/semantic_structure_extractor/semantic_structure_extractor.py`
recognises DART's `<article role="doc-chapter">` wrappers as the
primary chapter grouping signal. The legacy `<h2>`-hierarchy
heuristic is retained as a graceful fallback for generic third-party
HTML that lacks the explicit doc-chapter role.

## Decision capture

A `DecisionCapture` instance is threaded through every Claude call
site in the pipeline. The table below is the source of truth for
what fires where.

| Call site | Decision type | Trigger | Rationale signals |
|-----------|---------------|---------|-------------------|
| `MCP/tools/pipeline_tools.py::_raw_text_to_accessible_html` | `pipeline_run_attribution` | Once per pipeline run at function entry | backend, classifier_mode, raw_text length, title, output_path state, figures_dir state, llm injection state |
| `DART/converter/llm_classifier.py::LLMClassifier._classify_batch` | `structure_detection` | One per batch (typical batch_size=20) | block-ID range, LLM vs heuristic-fallback counts, fallback fraction, avg confidence, low-confidence fraction, char prompt payload, model + max_tokens |
| `DART/pdf_converter/alt_text_generator.py::AltTextGenerator.generate` | `alt_text_generation` (via `DARTDecisionCapture.log_alt_text_decision` + `log_decision`) | One per figure | page, bbox, image hash (first 12 chars of sha256), width×height, chosen source (claude / ocr / caption / generic), caption presence, alt-text length, long-description length, context length |
| `MCP/tools/dart_tools.py::convert_pdf_multi_source` | `approach_selection` + `validation_result` | Once per call | multi-source synthesis details (legacy-path telemetry) |

### Plumbing contract

* `_raw_text_to_accessible_html(capture=...)` — optional kwarg. When
  `None` (default) and `source_pdf` is provided, the function builds
  a short-lived `DARTDecisionCapture` keyed on the normalised PDF
  stem and finalises it on exit. When the caller supplies a capture,
  that capture is used for all emits (including the per-batch LLM +
  per-figure alt-text records).
* `default_classifier(llm=..., capture=...)` — forwards `capture`
  into `LLMClassifier` when routing goes to the LLM path. The
  heuristic classifier ignores `capture` (no Claude calls = nothing
  to log).
* `extract_document(pdf_path, *, llm=..., figures_dir=..., capture=...)` —
  forwards `capture` into the figure-extraction loop, which hands
  it to `AltTextGenerator(..., capture=capture)`.

### Course-code normalisation

`MCP/tools/dart_tools.py::normalize_course_code` coerces any PDF
filename into the canonical `^[A-Z]{2,8}_[0-9]{3}$` pattern so
every DART capture's `course_id` field passes schema validation.
Strategy: uppercase + underscore-normalise, pick the first
≥2-char alphabetic chunk as prefix (truncated to 8), append a
deterministic 3-digit SHA-256-based suffix. Same input always
produces the same output.

### Off-switch parity

All capture emits are best-effort — a capture-emit exception is
logged at DEBUG and swallowed so a capture regression never blocks
the HTML return path. Tests that don't care about captures keep
passing byte-for-byte (the `capture=None` default silently skips).
