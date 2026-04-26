"""Wave 83 Phase B regression tests for section-boundary-respecting HTML extraction.

The pre-Wave-83 ``_extract_section_html`` used a flat regex slice from
``<hN>{heading}`` to the next ``<h[1-6]`` — clipping ``<section>`` open
or close tags whenever adjacent headings lived in different sections.
That was the load-bearing cause of the rdf-shacl-551 audit's 203/295
unbalanced-section chunks.

Wave 83 Phase B walks ``<section>``/``</section>`` events relative to
the heading's position to determine enclosure, then slices the input
string accordingly. This test file pins the four required behaviors
listed in plans/wave-83-html-balance-2026-04/plan.md:

  1. Heading inside a ``<section>`` → returned HTML carries open + close.
  2. Heading outside any ``<section>`` (page-title h1) → heading element
     only, no spurious section tags.
  3. Two headings inside the same ``<section>`` → both yield the same
     outer section HTML.
  4. ``_BalanceChecker.check`` returns True for every output.
"""

from __future__ import annotations

from Trainforge.process_course import CourseProcessor, _BalanceChecker


# ---------------------------------------------------------------------------
# Behavior 1: heading inside a <section>
# ---------------------------------------------------------------------------


class TestHeadingInsideSection:
    def test_returns_full_section_with_open_and_close_tags(self):
        html = (
            '<section data-cf-template-type="explanation">'
            '<h2>Core Idea</h2>'
            '<p>RDF triples are statements.</p>'
            '</section>'
        )
        result = CourseProcessor._extract_section_html(html, "Core Idea")
        assert result.startswith('<section data-cf-template-type="explanation">')
        assert result.endswith('</section>')
        assert "<p>RDF triples are statements.</p>" in result

    def test_audit_layout_two_adjacent_sections(self):
        # The exact rdf-shacl-551 layout: two `<section>` blocks back-to-back.
        # Pre-Wave-83 regex slice would clip the closing </section> from
        # section A and the opening <section> from section B.
        html = (
            '<section data-cf-template-type="explanation">'
            '<h2>First Heading</h2>'
            '<p>First content.</p>'
            '</section>'
            '<section data-cf-template-type="example">'
            '<h2>Second Heading</h2>'
            '<p>Second content.</p>'
            '</section>'
        )
        first = CourseProcessor._extract_section_html(html, "First Heading")
        second = CourseProcessor._extract_section_html(html, "Second Heading")
        # Each must be its own balanced section; no leakage across the boundary.
        assert _BalanceChecker.check(first), f"first unbalanced: {first}"
        assert _BalanceChecker.check(second), f"second unbalanced: {second}"
        # And the contents are isolated.
        assert "Second" not in first
        assert "First" not in second

    def test_section_with_nested_elements_preserved(self):
        # Nested tags inside the section must come through intact.
        html = (
            '<section>'
            '<h2>Hello</h2>'
            '<div class="callout"><p>important <em>note</em></p></div>'
            '<ul><li>a</li><li>b</li></ul>'
            '</section>'
        )
        result = CourseProcessor._extract_section_html(html, "Hello")
        assert '<div class="callout"><p>important <em>note</em></p></div>' in result
        assert "<ul><li>a</li><li>b</li></ul>" in result
        assert _BalanceChecker.check(result)


# ---------------------------------------------------------------------------
# Behavior 2: heading outside any <section> (page-title h1)
# ---------------------------------------------------------------------------


class TestHeadingOutsideAnySection:
    def test_h1_page_title_returns_only_heading_element(self):
        html = (
            '<h1>Page Title</h1>'
            '<section><h2>First Section</h2><p>body</p></section>'
        )
        result = CourseProcessor._extract_section_html(html, "Page Title")
        assert result == "<h1>Page Title</h1>"
        # No spurious section tags clipped in.
        assert "<section" not in result
        assert "</section" not in result

    def test_h1_outside_section_balance_check_passes(self):
        html = (
            '<h1>Page Title</h1>'
            '<section><h2>X</h2><p>body</p></section>'
        )
        result = CourseProcessor._extract_section_html(html, "Page Title")
        assert _BalanceChecker.check(result)

    def test_heading_between_two_sections_no_section_clipping(self):
        # A heading sitting BETWEEN sections (not enclosed by either) must
        # not clip from either side.
        html = (
            '<section><h2>A</h2></section>'
            '<h2>Bare Heading</h2>'
            '<section><h2>B</h2></section>'
        )
        result = CourseProcessor._extract_section_html(html, "Bare Heading")
        assert result == "<h2>Bare Heading</h2>"
        assert _BalanceChecker.check(result)


# ---------------------------------------------------------------------------
# Behavior 3: two headings inside the same <section>
# ---------------------------------------------------------------------------


class TestTwoHeadingsSameSection:
    def test_both_yield_same_outer_section_html(self):
        html = (
            '<section>'
            '<h2>Top</h2><p>top body</p>'
            '<h3>Sub</h3><p>sub body</p>'
            '</section>'
        )
        top = CourseProcessor._extract_section_html(html, "Top")
        sub = CourseProcessor._extract_section_html(html, "Sub")
        assert top == sub
        assert _BalanceChecker.check(top)
        assert "<h2>Top</h2>" in top
        assert "<h3>Sub</h3>" in top


# ---------------------------------------------------------------------------
# Behavior 4: balance check — every output passes _BalanceChecker
# ---------------------------------------------------------------------------


class TestAllOutputsBalanced:
    def test_audit_failure_repro_now_balanced(self):
        # The exact rdf-shacl-551 layout per the investigator's findings:
        # h1 with no body, then 5 sections. Pre-Wave-83 every chunk's
        # HTML had unbalanced section tags. Post-Wave-83 every output
        # must pass the balance check.
        html = (
            '<h1>RDF Triples and the Graph Model</h1>'
            '<section role="region" aria-label="Learning Objectives">'
            '<h2>This Page Supports</h2><ul><li>obj 1</li></ul>'
            '</section>'
            '<section data-cf-template-type="explanation">'
            '<h2>The Core Idea</h2><p>core body</p>'
            '</section>'
            '<section data-cf-template-type="explanation">'
            '<h2>Anatomy of a Triple</h2><p>anatomy body</p>'
            '</section>'
        )
        for heading in [
            "RDF Triples and the Graph Model",
            "This Page Supports",
            "The Core Idea",
            "Anatomy of a Triple",
        ]:
            result = CourseProcessor._extract_section_html(html, heading)
            assert result, f"Empty result for {heading!r}"
            assert _BalanceChecker.check(result), (
                f"Unbalanced HTML for heading {heading!r}: {result[:200]}..."
            )


# ---------------------------------------------------------------------------
# Edge cases — defensive behavior
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_inputs_return_empty(self):
        assert CourseProcessor._extract_section_html("", "Hello") == ""
        assert CourseProcessor._extract_section_html("<h2>Hello</h2>", "") == ""

    def test_heading_not_found_returns_empty(self):
        html = "<section><h2>Different</h2></section>"
        assert CourseProcessor._extract_section_html(html, "Missing") == ""

    def test_unclosed_section_falls_back_to_eof(self):
        # Defensive: source HTML missing the closing </section>. We can't
        # recover a balanced fragment but should not raise; return
        # everything from the section start to EOF as best-effort.
        html = '<section><h2>Hello</h2><p>body</p>'  # no </section>
        result = CourseProcessor._extract_section_html(html, "Hello")
        assert result.startswith("<section>")
        # The result is unbalanced — that's the source's fault, flagged
        # by the balance check downstream. Wave 83 surfaces this honestly
        # rather than masking it with the legacy regex's accidental clip.

    def test_nested_sections_picks_innermost(self):
        # If the heading is inside a section that's inside another section,
        # we return the INNERMOST enclosing section.
        html = (
            '<section class="outer">'
            '<section class="inner">'
            '<h2>Target</h2><p>body</p>'
            '</section>'
            '</section>'
        )
        result = CourseProcessor._extract_section_html(html, "Target")
        assert result.startswith('<section class="inner">')
        assert result.endswith('</section>')
        assert _BalanceChecker.check(result)

    def test_case_insensitive_heading_tag(self):
        # Tag name is case-insensitive per HTML5; heading text comparison
        # uses the literal string. Match SHOULD succeed regardless of
        # tag-name casing in source.
        html = '<SECTION><H2>Hello</H2><p>x</p></SECTION>'
        result = CourseProcessor._extract_section_html(html, "Hello")
        assert _BalanceChecker.check(result)
        assert "Hello" in result

    def test_special_chars_in_heading_text_escaped(self):
        # Headings with regex-special chars (parens, dots) must not break
        # the locator — re.escape handles this.
        html = '<section><h2>SPARQL (1.1)</h2><p>body</p></section>'
        result = CourseProcessor._extract_section_html(html, "SPARQL (1.1)")
        assert "SPARQL (1.1)" in result
        assert _BalanceChecker.check(result)
