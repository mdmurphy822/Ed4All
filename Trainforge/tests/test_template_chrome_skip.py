"""Worker Q: Trainforge HTMLTextExtractor skips `data-cf-role=template-chrome`
subtrees. Courseforge now emits the role on header/footer/skip-link so
downstream consumers don't ingest repeated page boilerplate.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.chunker.helpers import (
    extract_plain_text,
    extract_plain_text_with_curies,
)
from Trainforge.parsers.html_content_parser import HTMLTextExtractor


def _extract(html: str) -> str:
    x = HTMLTextExtractor()
    x.feed(html)
    return x.get_text()


class TestTemplateChromeSkip:
    def test_footer_chrome_skipped(self):
        html = """<html><body>
          <main><p>Real body content.</p></main>
          <footer role="contentinfo" data-cf-role="template-chrome">
            <p>&copy; 2026 SYNTHETIC_COURSE. All rights reserved.</p>
          </footer>
        </body></html>"""
        text = _extract(html)
        assert "Real body content." in text
        assert "rights reserved" not in text.lower()
        assert "2026" not in text

    def test_header_chrome_skipped(self):
        html = """<html><body>
          <header role="banner" data-cf-role="template-chrome">
            <p>SYNTHETIC_COURSE &mdash; Week 3</p>
          </header>
          <main><h1>Topic</h1><p>Body.</p></main>
        </body></html>"""
        text = _extract(html)
        assert "Topic" in text
        assert "Body." in text
        assert "Week 3" not in text
        assert "SYNTHETIC_COURSE" not in text

    def test_skip_link_chrome_skipped(self):
        """Skip-to-main links are chrome too; Courseforge now marks them."""
        html = """<html><body>
          <a href="#main-content" class="skip-link" data-cf-role="template-chrome">Skip to main content</a>
          <main><p>Body.</p></main>
        </body></html>"""
        text = _extract(html)
        assert "Body." in text
        assert "Skip to main content" not in text

    def test_unmarked_element_not_skipped(self):
        """A `<footer>` without the data-cf-role attribute is content-bearing
        and must NOT be skipped — the role is the opt-in signal."""
        html = """<html><body>
          <main><p>Body.</p></main>
          <footer><p>Per-chunk footer that's actually content.</p></footer>
        </body></html>"""
        text = _extract(html)
        assert "Per-chunk footer" in text

    def test_nested_content_inside_chrome_still_skipped(self):
        html = """<html><body>
          <footer data-cf-role="template-chrome">
            <div><p>Nested <strong>chrome</strong> text.</p></div>
          </footer>
          <main><p>Keep me.</p></main>
        </body></html>"""
        text = _extract(html)
        assert "Keep me." in text
        assert "Nested" not in text
        assert "chrome" not in text

    def test_content_before_and_after_chrome_kept(self):
        html = """<html><body>
          <p>Before.</p>
          <footer data-cf-role="template-chrome"><p>Chrome.</p></footer>
          <p>After.</p>
        </body></html>"""
        text = _extract(html)
        assert "Before." in text
        assert "After." in text
        assert "Chrome." not in text

    def test_script_and_style_still_skipped(self):
        """Preserve the pre-existing script/style skip behavior."""
        html = """<html><head>
          <style>body { color: red; }</style>
          <script>alert('x');</script>
        </head><body><p>Content.</p></body></html>"""
        text = _extract(html)
        assert "Content." in text
        assert "red" not in text
        assert "alert" not in text


class TestScreenReaderOnlySkip:
    """SemantiK's gold-shell emits screen-reader-only structural labels —
    ``<p class="sr-only" hidden>Paragraph block</p>`` — as an accessibility
    surface. Those labels are NOT document content; left in, thousands of
    "Paragraph block" / "List block" label tokens leak into chunk ``text`` and
    pollute the downstream objectives + retrieval pipeline. Keyed on the
    ``sr-only`` / ``visually-hidden`` class (and ``aria-hidden="true"``), NOT a
    bare ``hidden`` attribute (progressive-disclosure reveal content stays).
    """

    def test_sr_only_label_skipped(self):
        """The exact SemantiK gold-shell label shape
        (``<p class="sr-only" hidden>...</p>``) must be dropped; real prose
        survives."""
        html = (
            "<section>"
            '<p id="s0" class="sr-only" hidden>Definition list block</p>'
            "<p>Whole numbers are the counting numbers plus zero.</p>"
            '<p id="s1" class="sr-only" hidden>Paragraph block</p>'
            "<p>A variable stands for a number we do not yet know.</p>"
            "</section>"
        )
        text = _extract(html)
        assert "Whole numbers are the counting numbers plus zero." in text
        assert "A variable stands for a number we do not yet know." in text
        assert "Definition list block" not in text
        assert "Paragraph block" not in text

    def test_visually_hidden_and_aria_hidden_skipped(self):
        """``visually-hidden`` class and ``aria-hidden="true"`` are also a11y
        surfaces that must not reach chunk text."""
        html = (
            "<section>"
            "<p>Real body content.</p>"
            '<span class="visually-hidden">Metadata drop block</span>'
            '<span aria-hidden="true">decorative marker</span>'
            "</section>"
        )
        text = _extract(html)
        assert "Real body content." in text
        assert "Metadata drop block" not in text
        assert "decorative marker" not in text

    def test_inline_sr_only_span_skipped_prose_survives(self):
        """An inline sr-only ``<span>`` inside a real ``<p>`` is dropped while
        the surrounding prose in the same paragraph is kept."""
        html = (
            "<p>Before label "
            '<span class="sr-only">List block</span>'
            " after label.</p>"
        )
        text = _extract(html)
        assert "Before label" in text
        assert "after label." in text
        assert "List block" not in text

    def test_bare_hidden_reveal_content_still_kept(self):
        """Control (byte-compat with the curie-skip contract): a BARE
        ``hidden`` attribute is progressive-disclosure reveal content and
        stays in text — only the sr-only class / aria-hidden are the opt-in
        skip signals."""
        html = (
            "<section>"
            "<p>First paragraph.</p>"
            "<div hidden>genuinely hidden reveal content</div>"
            "<p>Second paragraph.</p>"
            "</section>"
        )
        text = _extract(html)
        assert "First paragraph." in text
        assert "Second paragraph." in text
        assert "genuinely hidden reveal content" in text

    def test_chunker_plain_text_helper_drops_sr_only_label(self):
        """Chunk-level surface: the chunker's ``extract_plain_text`` helper
        (which produces chunk ``text``) must not carry the sr-only label."""
        html = (
            "<section><h2>Whole Numbers</h2>"
            '<p class="sr-only" hidden>Paragraph block</p>'
            "<p>Whole numbers add zero to the counting numbers.</p>"
            "</section>"
        )
        text = extract_plain_text(html)
        assert "Whole numbers add zero to the counting numbers." in text
        assert "Paragraph block" not in text


class TestCurieAnchorSkip:
    """The Courseforge rewrite tier's RewriteProvider._force_inject_curies
    appends a hidden ``<span data-cf-curie="...">`` carrying synthetic
    minted CURIE tokens as TEXT CONTENT, so the post-rewrite
    rewrite_curie_anchoring gate can regex-scrape them. That span is a
    rewrite-tier validator anchor only — its synthetic tokens must NOT
    reach chunk text, where the training-synthesis paraphrase pass would
    learn to emit identifiers that exist in no real textbook.
    """

    def test_force_injected_curie_span_skipped(self):
        """The exact force-injected span shape (hidden + data-cf-curie)
        must be skipped; surrounding prose survives."""
        html = (
            "<html><body>"
            "<p>Cardinality constraints bound how many values a property "
            "may carry.</p>"
            '<span hidden data-cf-curie="ns:concept">ns:concept</span>'
            "</body></html>"
        )
        text = _extract(html)
        assert "Cardinality constraints bound how many values" in text
        assert "ns:concept" not in text

    def test_multi_curie_span_skipped(self):
        """Force-injection space-joins multiple missing CURIEs into one
        span's text content — all of them must be dropped."""
        html = (
            "<html><body>"
            "<p>Real prose body.</p>"
            '<span hidden data-cf-curie="sh:minCount sh:maxCount">'
            "sh:minCount sh:maxCount</span>"
            "</body></html>"
        )
        text = _extract(html)
        assert "Real prose body." in text
        assert "sh:minCount" not in text
        assert "sh:maxCount" not in text

    def test_nested_content_inside_curie_anchor_skipped(self):
        """Any subtree rooted at a data-cf-curie element is skipped, not
        just the element's direct text."""
        html = (
            "<html><body>"
            "<p>Keep me.</p>"
            '<span data-cf-curie="ns:concept">'
            "<em>nested</em> ns:concept text</span>"
            "</body></html>"
        )
        text = _extract(html)
        assert "Keep me." in text
        assert "nested" not in text
        assert "ns:concept" not in text

    def test_no_curie_attr_identical_to_before(self):
        """Control: HTML without any data-cf-curie attribute extracts
        byte-identically — the fix is scoped to the attribute, so legacy /
        RDF corpora and existing fixtures are unaffected."""
        html = (
            "<html><body>"
            "<p>First paragraph.</p>"
            "<span hidden>genuinely hidden reveal content</span>"
            "<p>Second paragraph.</p>"
            "</body></html>"
        )
        text = _extract(html)
        # A bare ``hidden`` span (progressive-disclosure / reveal content)
        # is NOT skipped — only data-cf-curie is the opt-in signal.
        assert "First paragraph." in text
        assert "Second paragraph." in text
        assert "genuinely hidden reveal content" in text

    def test_content_before_and_after_curie_span_kept(self):
        """Force-injection appends the span at end-of-fragment; prose
        before and after it survives."""
        html = (
            "<html><body>"
            "<p>Before.</p>"
            '<span hidden data-cf-curie="ns:concept">ns:concept</span>'
            "<p>After.</p>"
            "</body></html>"
        )
        text = _extract(html)
        assert "Before." in text
        assert "After." in text
        assert "ns:concept" not in text

    def test_chunker_plain_text_helper_drops_synthetic_token(self):
        """Chunk-level surface: the chunker's ``extract_plain_text`` helper
        (which produces chunk ``text``, the surface training synthesis
        paraphrases) must not carry the force-injected synthetic token."""
        html = (
            "<section><h2>Property Shapes</h2>"
            "<p>A property shape constrains the values a node may carry "
            "for a given predicate.</p>"
            '<span hidden data-cf-curie="ns:concept">ns:concept</span>'
            "</section>"
        )
        text = extract_plain_text(html)
        assert "property shape constrains the values" in text
        assert "ns:concept" not in text


class TestCurieAnchorHarvest:
    """U2: the HTMLTextExtractor still SKIPS the force-injected
    ``data-cf-curie`` subtree (376b64f invariant — synthetic tokens stay
    out of chunk text), but it now HARVESTS the CURIE attribute values so
    the downstream ``curie_anchoring`` gate can still see the anchors.
    The ``data-cf-curie-forced="true"`` marker partitions the harvest
    into the forced subset.
    """

    def test_force_injected_curie_harvested_not_in_text(self):
        """A force-injected span (hidden + data-cf-curie +
        data-cf-curie-forced): the token is harvested into get_curies()
        AND get_forced_curies(), but the 376b64f skip invariant holds —
        it does not appear in extracted text."""
        html = (
            "<html><body>"
            "<p>Cardinality constraints bound how many values a property "
            "may carry.</p>"
            '<span hidden data-cf-curie="ns:concept" '
            'data-cf-curie-forced="true">ns:concept</span>'
            "</body></html>"
        )
        x = HTMLTextExtractor()
        x.feed(html)
        # 376b64f invariant: synthetic token stays out of text.
        assert "Cardinality constraints bound how many values" in x.get_text()
        assert "ns:concept" not in x.get_text()
        assert "ns:concept" not in extract_plain_text(html)
        # U2: the token is harvested and flagged as force-injected.
        assert "ns:concept" in x.get_curies()
        assert "ns:concept" in x.get_forced_curies()

    def test_multi_token_curie_harvested_without_forced_marker(self):
        """A data-cf-curie span WITHOUT the forced marker: both
        space-split tokens land in get_curies(); get_forced_curies()
        stays empty."""
        html = (
            "<html><body>"
            "<p>Real prose body.</p>"
            '<span hidden data-cf-curie="a:b c:d">a:b c:d</span>'
            "</body></html>"
        )
        x = HTMLTextExtractor()
        x.feed(html)
        assert "Real prose body." in x.get_text()
        assert "a:b" not in x.get_text()
        assert "c:d" not in x.get_text()
        assert "a:b" in x.get_curies()
        assert "c:d" in x.get_curies()
        # No forced marker → forced harvest stays empty.
        assert x.get_forced_curies() == []

    def test_no_curie_attr_empty_harvest(self):
        """Control: HTML with no data-cf-curie span yields empty curie
        + forced-curie harvests and unchanged text."""
        html = (
            "<html><body>"
            "<p>First paragraph.</p>"
            "<span hidden>genuinely hidden reveal content</span>"
            "<p>Second paragraph.</p>"
            "</body></html>"
        )
        x = HTMLTextExtractor()
        x.feed(html)
        assert "First paragraph." in x.get_text()
        assert "Second paragraph." in x.get_text()
        assert "genuinely hidden reveal content" in x.get_text()
        assert x.get_curies() == []
        assert x.get_forced_curies() == []

    def test_extract_plain_text_with_curies_tuple_shape(self):
        """The chunker helper ``extract_plain_text_with_curies`` returns
        ``(text, curies, forced_curies)`` on the force-injected case."""
        html = (
            "<section><h2>Property Shapes</h2>"
            "<p>A property shape constrains the values a node may carry "
            "for a given predicate.</p>"
            '<span hidden data-cf-curie="ns:concept" '
            'data-cf-curie-forced="true">ns:concept</span>'
            "</section>"
        )
        result = extract_plain_text_with_curies(html)
        assert isinstance(result, tuple)
        assert len(result) == 3
        text, curies, forced_curies = result
        assert isinstance(text, str)
        assert isinstance(curies, list)
        assert isinstance(forced_curies, list)
        # Text contract: byte-identical to extract_plain_text.
        assert text == extract_plain_text(html)
        assert "property shape constrains the values" in text
        assert "ns:concept" not in text
        # Harvest contract.
        assert "ns:concept" in curies
        assert "ns:concept" in forced_curies
