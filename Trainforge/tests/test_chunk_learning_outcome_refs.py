"""LO-anchoring regression — chunks carry ``learning_outcome_refs``.

Root cause closed by this suite: the shared chunk-emit callbacks in
``MCP/tools/pipeline_tools.py`` (the SemantiK-chunking and IMSCC-chunking
phase handlers) hardcoded ``learning_outcome_refs: []`` and never
harvested the learning-outcome (LO) ids the Courseforge content page
carried. Empty LO refs failed the ``packet_integrity_strict`` gate
(``UNANCHORED_ASSESSMENT``) and cascaded into an empty / unparseable
``assessments.json``.

The fix adds ``Trainforge.chunker.extract_learning_outcome_refs(item,
section_heading)``, wired into both chunk-emit callbacks. It harvests LO
ids the page validator reads:

  1. JSON-LD ``learningObjectives[].id`` (surfaced onto
     ``item["learning_objectives"]``).
  2. ``data-cf-objective-id`` / ``data-cf-objective-ref`` section
     attributes (surfaced onto ``ContentSection.objective_refs``).

Section-scoped refs win so a multi-LO page anchors each chunk to its own
section's LO. Every ref is validated against the canonical LO pattern
``^[A-Z]{2,}-\\d{2,}$``. When the source HTML carries no LO metadata
(legacy corpora) the helper returns ``[]`` — byte-identical to the
prior hardcoded behaviour.

Tests:
  1. A page with JSON-LD + ``data-cf-objective-id`` yields non-empty,
     section-anchored, pattern-valid LO refs.
  2. A page with NO LO metadata yields ``[]`` (no regression).
  3. Case policy: default lowercases; ``TRAINFORGE_PRESERVE_LO_CASE``
     preserves emit case.
  4. End-to-end through the canonical chunker: every emitted chunk on a
     real LO-bearing page carries non-empty ``learning_outcome_refs``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.chunker import (  # noqa: E402
    ChunkerContext,
    chunk_content,
    extract_learning_outcome_refs,
)
from Trainforge.parsers.html_content_parser import HTMLContentParser  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures — minimal Courseforge content pages
# ---------------------------------------------------------------------------

# Page carrying LO ids in BOTH forms the page validator reads:
# JSON-LD learningObjectives[].id AND data-cf-objective-id on sections.
# Two distinct LO ids (TO-01, TO-02) so section-scoped anchoring is
# observable.
_PAGE_WITH_LOS = """<!DOCTYPE html>
<html lang="en"><head><title>Week 1 Core Concepts</title>
<script type="application/ld+json">
{
  "@context": "https://ed4all.dev/ns/courseforge/v1",
  "@type": "CoursePage",
  "learningObjectives": [
    {"id": "TO-01", "bloomLevel": "understand"},
    {"id": "TO-02", "bloomLevel": "analyze"}
  ]
}
</script></head><body>
<main>
  <section data-cf-objective-id="TO-01" data-cf-content-type="concept">
    <h2>What Is Digital Accessibility?</h2>
    <p>Digital accessibility is the practice of designing software and
    digital tools so people with disabilities can perceive and operate
    them. This section introduces the core idea and its scope across
    graphical user interfaces of every kind. The Web Accessibility
    Initiative of the World Wide Web Consortium defines accessibility as
    ensuring that websites, tools, and technologies are designed and
    developed so that people with disabilities can use them. That
    definition extends naturally to any graphical user interface,
    including desktop applications, mobile apps, web dashboards, and
    embedded controls. A well designed accessible interface accounts for
    visual, auditory, physical, cognitive, and neurological variation
    rather than only the most obvious cases. Barriers are not inherent
    properties of a person's disability; they arise from a mismatch
    between the design of an interface and the needs of its user, and a
    change to the design can make the barrier disappear entirely.</p>
  </section>
  <section data-cf-objective-id="TO-02" data-cf-content-type="concept">
    <h2>The Business and Legal Case</h2>
    <p>Accessibility delivers measurable business value and reduces legal
    exposure under the relevant statutes. This section analyses the
    drivers that make accessible design a strategic priority rather than
    an afterthought for product teams. Organizations that invest in
    accessibility reach a larger audience, improve usability for every
    user, and reduce the risk of costly litigation under disability
    discrimination law. The business case rests on market reach, brand
    reputation, and search engine optimization, while the legal case
    rests on statutory obligations that apply across many jurisdictions.
    Teams that treat accessibility as a quality attribute embedded in
    their design process, rather than a remediation task performed at the
    end, ship better products and incur lower long term costs across the
    entire lifecycle of the software they build and maintain.</p>
  </section>
</main>
</body></html>"""

# Page with NO LO metadata at all (legacy-style). No JSON-LD
# learningObjectives, no data-cf-objective-* attributes.
_PAGE_WITHOUT_LOS = """<!DOCTYPE html>
<html lang="en"><head><title>Plain Page</title></head><body>
<main>
  <section>
    <h2>Some Topic</h2>
    <p>This is plain prose content with no learning-objective metadata
    whatsoever. A legacy corpus or an accessibility-converted textbook page
    looks exactly like this — no JSON-LD block and no Courseforge data-cf
    objective attributes anywhere in the markup at all.</p>
  </section>
</main>
</body></html>"""

_LO_PATTERN = __import__(
    "lib.ontology.learning_objectives", fromlist=["LO_ID_PATTERN"]
).LO_ID_PATTERN


def _item_from_html(html: str) -> dict:
    """Parse HTML into the chunker ``item`` dict shape."""
    parsed = HTMLContentParser().parse(html)
    return {
        "item_id": "page-1",
        "title": parsed.title,
        "resource_type": "page",
        "module_id": "m1",
        "module_title": parsed.title,
        "word_count": parsed.word_count,
        "sections": parsed.sections,
        "learning_objectives": parsed.learning_objectives,
        "key_concepts": parsed.key_concepts,
        "interactive_components": parsed.interactive_components,
        "raw_html": html,
        "objective_refs": parsed.objective_refs,
        "source_references": parsed.source_references,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_lo_refs_from_page_with_metadata():
    """A page with LO JSON-LD + data-cf-objective-id yields section-anchored
    pattern-valid refs."""
    item = _item_from_html(_PAGE_WITH_LOS)

    s1 = extract_learning_outcome_refs(
        item, "What Is Digital Accessibility?"
    )
    s2 = extract_learning_outcome_refs(item, "The Business and Legal Case")

    assert s1 == ["to-01"], s1
    assert s2 == ["to-02"], s2  # section-scoped: anchors to its own LO
    # Pattern validity (upper-cased candidate matches canonical pattern).
    for refs in (s1, s2):
        for r in refs:
            assert _LO_PATTERN.match(r.upper()), r


def test_extract_lo_refs_page_level_fallback():
    """When a chunk's section heading doesn't match any section, the helper
    falls back to the structured page-level LO ids."""
    item = _item_from_html(_PAGE_WITH_LOS)
    refs = extract_learning_outcome_refs(item, "No Such Heading")
    # Falls back to page-level learning_objectives ids (TO-01, TO-02).
    assert set(refs) == {"to-01", "to-02"}, refs


def test_no_lo_metadata_yields_empty():
    """Legacy page without LO metadata yields [] — no regression."""
    item = _item_from_html(_PAGE_WITHOUT_LOS)
    assert extract_learning_outcome_refs(item, "Some Topic") == []
    assert extract_learning_outcome_refs(item, None) == []


def test_case_policy(monkeypatch):
    """Default lowercases; TRAINFORGE_PRESERVE_LO_CASE preserves emit case."""
    item = _item_from_html(_PAGE_WITH_LOS)

    monkeypatch.delenv("TRAINFORGE_PRESERVE_LO_CASE", raising=False)
    assert extract_learning_outcome_refs(
        item, "What Is Digital Accessibility?"
    ) == ["to-01"]

    monkeypatch.setenv("TRAINFORGE_PRESERVE_LO_CASE", "true")
    assert extract_learning_outcome_refs(
        item, "What Is Digital Accessibility?"
    ) == ["TO-01"]


def test_non_lo_attribute_values_are_filtered():
    """Free-text / non-pattern objective values never reach the refs."""
    html = _PAGE_WITH_LOS.replace(
        'data-cf-objective-id="TO-01"',
        'data-cf-objective-id="introduce the topic"',
    )
    item = _item_from_html(html)
    refs = extract_learning_outcome_refs(
        item, "What Is Digital Accessibility?"
    )
    # The free-text value is filtered out; falls back to page-level ids.
    assert "introduce the topic" not in refs
    for r in refs:
        assert _LO_PATTERN.match(r.upper()), r


def test_end_to_end_chunker_anchors_every_chunk():
    """Every chunk the canonical chunker emits on an LO-bearing page carries
    non-empty learning_outcome_refs (mirrors the pipeline_tools callback)."""
    item = _item_from_html(_PAGE_WITH_LOS)

    def _cc(*, chunk_id, text, html, item, section_heading, chunk_type,
            follows_chunk_id=None, position_in_module=0, html_xpath=None,
            char_span=None, section_source_ids=None, merged_headings=None):
        return {
            "id": chunk_id,
            "learning_outcome_refs": extract_learning_outcome_refs(
                item, section_heading
            ),
            "section_heading": section_heading,
        }

    result = chunk_content(
        [item], "course_1", boilerplate_spans=None,
        ctx=ChunkerContext(create_chunk=_cc),
    )
    chunks = list(result.chunks)
    assert chunks, "expected at least one chunk"
    # The core invariant the fix restores: every chunk emitted from an
    # LO-bearing page carries non-empty learning_outcome_refs (vs. the
    # prior hardcoded []). Distinct per-section anchoring (TO-01 vs TO-02)
    # is proven directly in test_extract_lo_refs_from_page_with_metadata;
    # the canonical chunker may merge short adjacent sections into one
    # chunk (merge_small_sections), which legitimately collapses the two
    # sections here onto the first section's LO.
    for c in chunks:
        assert c["learning_outcome_refs"], (
            f"chunk {c['id']} ({c['section_heading']!r}) has no "
            f"learning_outcome_refs"
        )
    # Every emitted ref is a canonical LO id.
    for c in chunks:
        for r in c["learning_outcome_refs"]:
            assert _LO_PATTERN.match(r.upper()), r
