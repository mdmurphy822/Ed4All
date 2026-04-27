"""Wave 84 regression test for the JSON-LD section heading-mismatch bug.

The audit on rdf-shacl-551-2 (2026-04-26) found 109/295 chunks (37%)
silently lost ``content_type_label`` and ``key_terms`` because the
chunk's anchor heading (the FIRST section in a merged buffer) did not
match the heading any JSON-LD section was keyed under. The merger
collapsed multiple sections into one chunk; the metadata lookup only
tried the anchor heading, never the sub-headings.

This test pins the fix: ``_extract_section_metadata`` accepts
``merged_headings`` and walks every heading in priority order until it
finds a matching JSON-LD section. The anchor heading is checked first
so single-section chunks behave identically to pre-Wave-84.
"""

from __future__ import annotations

from Trainforge.process_course import CourseProcessor


def _bare_processor() -> CourseProcessor:
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.MIN_CHUNK_SIZE = 100
    proc.MAX_CHUNK_SIZE = 800
    return proc


def _item_with_jsonld_sections(*sections):
    """Build a fake ``item`` whose courseforge_metadata has the given
    JSON-LD sections shape."""
    return {
        "title": "Page Title",
        "courseforge_metadata": {"sections": list(sections)},
        "sections": [],  # data-cf-* path empty
    }


class TestMergedHeadingsResolveJsonLdMetadata:
    def test_anchor_heading_match_wins(self):
        # Single-section case: anchor matches → metadata lifted.
        proc = _bare_processor()
        item = _item_with_jsonld_sections(
            {
                "heading": "First H2",
                "contentType": "explanation",
                "keyTerms": [{"term": "alpha", "definition": "first letter"}],
            }
        )
        bloom, ctype, kt, trace = proc._extract_section_metadata(
            item, "First H2", merged_headings=["First H2"]
        )
        assert ctype == "explanation"
        assert kt[0]["term"] == "alpha"
        assert trace["content_type_label"] == "jsonld_section_match"

    def test_subheading_match_when_anchor_drifts(self):
        # The merger anchored on the page-title h1, but the JSON-LD
        # section is keyed off "Sub Two". Pre-Wave-84 lookup fell
        # through to none_heading_mismatch; post-Wave-84 the second
        # candidate heading matches.
        proc = _bare_processor()
        item = _item_with_jsonld_sections(
            {
                "heading": "Sub Two",
                "contentType": "example",
                "keyTerms": [{"term": "beta", "definition": "second letter"}],
            }
        )
        bloom, ctype, kt, trace = proc._extract_section_metadata(
            item, "Page Title", merged_headings=["Page Title", "Sub One", "Sub Two"]
        )
        assert ctype == "example"
        assert kt[0]["term"] == "beta"
        assert trace["content_type_label"] == "jsonld_section_match"

    def test_first_match_wins_when_multiple_merged_headings_match(self):
        # Two merged headings each have their own JSON-LD section.
        # Anchor (first heading) takes priority → its section's metadata wins.
        proc = _bare_processor()
        item = _item_with_jsonld_sections(
            {
                "heading": "Anchor",
                "contentType": "explanation",
                "keyTerms": [{"term": "anchor-term"}],
            },
            {
                "heading": "Sub",
                "contentType": "example",
                "keyTerms": [{"term": "sub-term"}],
            },
        )
        _, ctype, kt, _ = proc._extract_section_metadata(
            item, "Anchor", merged_headings=["Anchor", "Sub"]
        )
        # Anchor's metadata wins because it's checked first.
        assert ctype == "explanation"
        assert kt[0]["term"] == "anchor-term"

    def test_no_match_in_any_heading_returns_none(self):
        # When nothing matches across all merged headings, fall through
        # to the documented none_heading_mismatch trace value.
        proc = _bare_processor()
        item = _item_with_jsonld_sections(
            {"heading": "Different", "contentType": "explanation"}
        )
        _, ctype, kt, trace = proc._extract_section_metadata(
            item, "Anchor", merged_headings=["Anchor", "Sub"]
        )
        assert ctype is None
        assert kt == []
        assert trace["content_type_label"] == "none_heading_mismatch"

    def test_back_compat_when_merged_headings_omitted(self):
        # Pre-Wave-84 callers that don't pass merged_headings must keep
        # working — the behavior reduces to "match by anchor heading".
        proc = _bare_processor()
        item = _item_with_jsonld_sections(
            {
                "heading": "Anchor",
                "contentType": "procedure",
                "keyTerms": [],
            }
        )
        _, ctype, _, trace = proc._extract_section_metadata(item, "Anchor")
        assert ctype == "procedure"
        assert trace["content_type_label"] == "jsonld_section_match"

    def test_part_suffix_stripped_on_anchor_and_merged(self):
        # When a chunk is split into parts (Wave 3 _chunk_text_block adds
        # " (part N)"), every heading in the candidate list strips that
        # suffix before the equality test.
        proc = _bare_processor()
        item = _item_with_jsonld_sections(
            {
                "heading": "Real Heading",
                "contentType": "explanation",
            }
        )
        _, ctype, _, _ = proc._extract_section_metadata(
            item,
            "Real Heading (part 2)",
            merged_headings=["Real Heading (part 2)"],
        )
        assert ctype == "explanation"

    def test_data_cf_fallback_walks_merged_headings_too(self):
        # When JSON-LD sections are absent but data-cf-* attributes are
        # present, the same "walk every merged heading" logic applies on
        # the fallback path. This pins the symmetry.
        from Trainforge.parsers.html_content_parser import ContentSection

        proc = _bare_processor()
        item = {
            "title": "Page Title",
            "courseforge_metadata": None,
            "sections": [
                ContentSection(
                    heading="Sub Two",
                    level=2,
                    content="x",
                    word_count=1,
                    template_type=None,
                    content_type="example",
                    key_terms=["gamma"],
                ),
            ],
        }
        _, ctype, kt, trace = proc._extract_section_metadata(
            item, "Anchor", merged_headings=["Anchor", "Sub Two"]
        )
        assert ctype == "example"
        assert kt[0]["term"] == "gamma"
        assert trace["content_type_label"] == "data_cf_fallback"
