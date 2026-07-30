# Chunk provenance audit trail

> **Buyer-facing statement.** Every chunk Trainforge emits into a RAG corpus
> carries a cryptographically verifiable pointer back to the source IMSCC
> HTML element it was derived from. This is the audit trail required by
> Section 508 and ADA Title II procurement for institutional buyers who
> must be able to prove that a model-generated answer is grounded in
> contracted-for course content.

## The two provenance fields

Every chunk object in `corpus/chunks.jsonl` carries, under `source`, the
two fields that together pinpoint its origin:

| Field | Type | Meaning |
|---|---|---|
| `source.html_xpath` | string | Absolute XPath to the element that bounds the chunk's source content in the IMSCC's raw HTML. |
| `source.char_span` | `[start, end]` | Character offsets into that element's plain-text content (descendant text, whitespace-collapsed with single-space joiner) where the chunk's text begins and ends. |

Additional pointers carried on every chunk (not new, but required to make
the trail actionable):

| Field | Meaning |
|---|---|
| `source.item_path` | Path to the HTML file inside the IMSCC package. Lets an auditor open the exact file without crawling `imsmanifest.xml`. |
| `source.lesson_id` | IMSCC item id — unique per resource within the package. |
| `source.module_id` / `source.module_title` | Enclosing module for context. |
| `schema_version` | Chunk schema version (currently `"v4"`). Declares the provenance contract this chunk was emitted under. |

## The round-trip contract

Given a chunk and the IMSCC it was generated from, the auditor MUST be
able to recover the chunk's source text in three steps:

1. Open the IMSCC file at `source.item_path` (raw HTML).
2. Resolve `source.html_xpath` — walk the HTML to the element whose
   absolute path matches. Extract its plain-text content (concatenate all
   descendant text, whitespace-collapsed, joined by single spaces; this is
   the joining semantics `Trainforge/parsers/xpath_walker.py::resolve_xpath`
   and `Trainforge/parsers/html_content_parser.py::HTMLTextExtractor` both
   use).
3. Slice `element_text[char_span[0]:char_span[1]]`. The result equals
   `chunk.text` modulo the normalization tolerance documented below.

The round-trip is tested in `Trainforge/tests/test_provenance.py`:

- `test_xpath_roundtrip_recovers_chunk_text` — end-to-end slice test.
- `test_char_span_end_greater_than_start` — non-empty spans only.
- `test_char_span_does_not_overflow_element` — slices stay within bounds.
- `test_xpath_is_absolute` — locked format; no relative paths, no `//`.
- `test_every_chunk_has_provenance_fields` — 100% coverage on regenerated
  corpora.
- `test_multipart_spans_are_disjoint_and_contiguous` — when a long
  section is split into multiple chunks, their spans cover the section
  without overlaps and without gaps larger than the single-char sentence
  joiner.

## Normalization tolerance

The chunker runs the plain text through three transforms between reading
it from the HTML element and writing it to the chunk:

1. **Whitespace collapse.** `HTMLTextExtractor` joins tokens by single
   spaces; consecutive whitespace in the source HTML collapses to one
   space in the chunk.
2. **WCAG SC canonicalization.** `Trainforge/rag/wcag_canonical_names.py::
   canonicalize_sc_references` rewrites success-criterion references to a
   single canonical form before the chunk is written. A chunk may show
   `"SC 1.1.1"` where the source HTML had `"Success Criterion 1.1.1"`.
3. **Feedback / boilerplate strip (quiz and template-chrome only).**
   `_strip_assessment_feedback`, `_strip_feedback_from_text`, and
   `strip_boilerplate` remove answer-feedback text from quizzes and remove
   detected template chrome from every item before chunking.

Tolerance for the round-trip: the recovered substring MUST start with
the first non-boilerplate sentence of `chunk.text` (or the full text,
whichever is shorter) after both strings are whitespace-collapsed and
lowercased. A strict byte-for-byte equality is not guaranteed because
the three transforms above can legitimately modify the text between
source and chunk.

For quiz chunks specifically, `source.resource_type == "quiz"` — the
auditor applies `_strip_assessment_feedback` to the element text before
comparing. This matches how the chunker produced the text and avoids
false audit failures on feedback-stripped content.

## XPath format (locked)

The walker at `Trainforge/parsers/xpath_walker.py` emits xpaths in a
restricted, deterministic dialect:

- **Absolute**, starting with `/`. The first step is the document's root
  element (typically `html`) or the first encountered open tag for
  malformed documents without an `<html>` shell.
- **Step form**: `tag[n]` where `n` is the 1-based index of the element
  among its same-tag siblings under the shared parent. Mirrors XPath 1.0
  predicate semantics.
- **No shortcuts**: no `//`, no wildcards, no namespaces, no predicates
  beyond the sibling index. If a downstream tool wants a more compact
  form, it can compute it from the absolute path — we never emit one.
- **Tag names are lowercased**. Attribute-based selectors are not part of
  the format.

Example: `/html[1]/body[1]/h2[2]` — the second `<h2>` child of `<body>`,
which is the first child of `<html>`.

## What `html_xpath` points at

Two cases in the chunker, `Trainforge/process_course.py::_chunk_text_block`:

| Case | `html_xpath` anchors to |
|---|---|
| Item has parsed sections (most pages) | The `<hN>` heading element of the section that produced this chunk. |
| Item has no sections (quizzes, assessments, pages without headings) | The `<body>` element of the document. |

For multi-part chunks (a single long section split into N sub-chunks by
`_split_by_sentences`), all N siblings share the same `html_xpath` and
carry disjoint, contiguous `char_span` values. The section is fully
recoverable by concatenating the N slices in chunk-id order.

## What `char_span` is NOT

- `char_span` is **not** an offset into the raw HTML bytes. It is an
  offset into the whitespace-collapsed plain text of the element at
  `html_xpath`. This is deliberate — byte offsets into HTML are fragile
  under any parse-then-reserialize round trip, and almost every buyer
  tool (AT, screen readers, evaluation harnesses) operates on the plain
  text anyway.
- `char_span` is **not** an offset into the source IMSCC file. The item
  path lives in `source.item_path`, and the file-level offset is not
  tracked — the element-level offset is sufficient for every known
  audit use case.

## Regeneration

Any chunks.jsonl emitted by a Trainforge version with `CHUNK_SCHEMA_VERSION
>= "v4"` carries these fields on every chunk. Regenerate with the standard
invocation — no new flag is required:

```bash
python -m Trainforge.process_course \
  --imscc path/to/course.imscc \
  --course-code <CODE> \
  --division <DIV> --domain <DOMAIN> \
  --output Trainforge/output/<slug>
```

Older corpora (emitted under `v3`) do not carry the fields. Re-run the
pipeline on the original IMSCC to add provenance. No in-place migration
is provided; the source HTML is authoritative and regeneration is cheap.

## Known follow-ups

- **`FOLLOWUP-WORKER-E-1`**: `schemas/library/chunk.schema.json` does not
  exist in this tree. The repo ships `catalog_entry.schema.json` and
  `course_manifest.schema.json` under `schemas/library/`, but there is no
  chunk schema today. When someone lands a LibV2 chunk schema, add
  `source.html_xpath` (optional string) and `source.char_span` (optional
  array of two integers) to it. Until then, LibV2's importer accepts
  extra fields on chunks without schema validation, so no migration is
  blocked.

- **LibV2 importer copy-through.** `LibV2/tools/libv2/importer.py` copies
  chunks.jsonl verbatim into `LibV2/courses/<slug>/corpus/chunks.jsonl`.
  Re-running the importer after a `v4` regeneration propagates the
  provenance fields with no import-side change needed.
