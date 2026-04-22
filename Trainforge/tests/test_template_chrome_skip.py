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
