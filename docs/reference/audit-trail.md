# Trace a chunk to source HTML

Trainforge v4 chunks can carry an element locator and character span that let
an auditor trace retrieved text back to the private IMS Common Cartridge HTML
from which it was derived. This supports provenance review; it is not a legal
certification or a substitute for reviewing the source package.

## Provenance fields

The chunk's `source` object provides the locator:

| Field | Meaning |
|---|---|
| `item_path` | Path to the source HTML member inside the private package. |
| `html_xpath` | Absolute, deterministic XPath for the source text container. |
| `char_span` | Half-open `[start, end]` offsets in that container's normalized plain text. |
| `lesson_id` | Package resource identifier. |
| `module_id` | Enclosing module identifier. |
| `source_document_sha256` | Optional digest joining the chunk to an upstream source artifact. |

`html_xpath`, `char_span`, `item_path`, and `source_document_sha256` are
optional in the v4 schema for compatibility with older or non-IMSCC inputs.
Their absence means this locator form is unavailable; it must never be replaced
with an invented path or span.

The authoritative shape is
[`schemas/knowledge/chunk_v4.schema.json`](../../schemas/knowledge/chunk_v4.schema.json).

## Round-trip procedure

Given a private chunk and its matching source package:

1. Open the package member named by `source.item_path`.
2. Resolve `source.html_xpath` with
   `Trainforge.parsers.xpath_walker.resolve_xpath`.
3. Read `start, end = source.char_span`.
4. Slice the resolved text with `element_text[start:end]`.
5. Compare the slice with the chunk using the normalization rules below.

```python
from Trainforge.parsers.xpath_walker import resolve_xpath

element_text = resolve_xpath(private_html, chunk["source"]["html_xpath"])
if element_text is None:
    raise ValueError("source XPath does not resolve")

start, end = chunk["source"]["char_span"]
if not 0 <= start < end <= len(element_text):
    raise ValueError("source span is outside the resolved element")

source_slice = element_text[start:end]
```

Keep source HTML, chunk records, and comparison output private. They can contain
licensed text and course identifiers.

## XPath dialect

`Trainforge.parsers.xpath_walker` emits a restricted XPath form:

- the path is absolute and begins with `/`;
- each step is a lowercase `tag[index]` pair;
- sibling indexes are one-based within the shared parent;
- `//`, wildcards, namespace prefixes, and attribute predicates are not used;
  and
- malformed HTML without a body falls back to its first indexed root element.

For heading-led content, the locator targets the heading's parent container so
the resolved text includes the section body. Content without a matching
heading uses the body or document-root fallback.

## Text normalization

Offsets address normalized descendant text, not raw HTML bytes and not file
offsets. The XPath walker and HTML extractor share their whitespace assembler:
source whitespace collapses, inline-element boundaries do not add characters,
and block-element boundaries add a separator.

Chunk text can differ from the direct slice because the processing path may:

- canonicalize supported reference notation;
- remove template chrome, scripts, styles, or assessment feedback;
- merge adjacent sections; or
- split a long text block into multiple chunks.

Compare using the owning parser's documented normalization. Do not claim
byte-for-byte recovery when one of these transforms applies.

For a split block, sibling spans share the same source container, do not
overlap, and remain contiguous apart from the permitted joiner boundary.

## Verification checklist

For each audited chunk, verify that:

- the package member exists and belongs to the same private source artifact;
- the XPath is absolute and resolves exactly once under the deterministic
  walker;
- the span contains two non-negative integers with `start < end`;
- `end` does not exceed the resolved text length;
- the normalized slice supports the emitted chunk text; and
- multipart sibling spans do not overlap or leave an unexplained gap.

The regression coverage is concentrated in:

- `Trainforge/tests/test_provenance.py`;
- `Trainforge/tests/test_chunk_script_leak.py`;
- `Trainforge/parsers/tests/test_html_extractor_inline_boundaries.py`; and
- `Trainforge/tests/test_chunker_smoke.py`.

Some provenance tests accept `TRAINFORGE_PROVENANCE_CORPUS` to audit an
operator-local regenerated corpus. Never point that variable at tracked or
public course data.

## Regeneration and compatibility

Older chunksets may not include this locator. Regenerate from the authoritative
private source package with the current Trainforge pipeline when provenance is
required. There is no supported in-place process that can reconstruct an exact
XPath and span without the source HTML.

LibV2 preserves chunk fields when importing a compatible chunkset, so the
locator travels with the private archive. Consumers must still validate the
chunk schema and source-artifact identity before trusting it.

See [Chunk schema v4](../architecture/chunk-schema-v4.md) for the wider chunk
contract and [Pipeline invocation](../operations/pipeline-invocation.md) for
private artifact handling.
