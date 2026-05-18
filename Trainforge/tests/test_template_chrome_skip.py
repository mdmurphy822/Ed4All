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

from Trainforge.chunker.helpers import extract_plain_text
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
            <p>&copy; 2026 SAMPLE_101. All rights reserved.</p>
          </footer>
        </body></html>"""
        text = _extract(html)
        assert "Real body content." in text
        assert "rights reserved" not in text.lower()
        assert "2026" not in text

    def test_header_chrome_skipped(self):
        html = """<html><body>
          <header role="banner" data-cf-role="template-chrome">
            <p>SAMPLE_101 &mdash; Week 3</p>
          </header>
          <main><h1>Topic</h1><p>Body.</p></main>
        </body></html>"""
        text = _extract(html)
        assert "Topic" in text
        assert "Body." in text
        assert "Week 3" not in text
        assert "SAMPLE_101" not in text

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
