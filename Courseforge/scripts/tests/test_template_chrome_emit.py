"""Worker Q: Courseforge generate_course.py emits `data-cf-role="template-chrome"`
on repeated page-chrome elements (header, footer, skip-link). Trainforge's
HTMLTextExtractor uses that role to skip the subtree when building chunk
text — so boilerplate doesn't end up in every derivative artifact.

This test confirms the EMIT side. The SKIP side is tested in
Trainforge/tests/test_template_chrome_skip.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _render_minimal_page() -> str:
    """Render a tiny page through generate_course's shell builder so we
    can inspect the HTML it emits for the chrome role."""
    # generate_course's page-shell builder is a private helper; invoke
    # generate_week against a minimal week_data fixture to exercise the
    # same shell in-situ. We don't need the full generator — a quick
    # end-to-end emission suffices.
    import tempfile

    from generate_course import generate_week  # noqa: E402

    week_data = {
        "week_number": 3,
        "title": "Minimal Week Three",
        "topics": [],
        "objectives": [
            {"id": "CO-05", "statement": "Explain X", "bloom_level": "understand"},
        ],
        "readings": [],
        "estimated_time": "30 minutes",
        "activities": [],
        "self_check": [],
        "summary": "One-line summary.",
        "discussion": {"prompt": "Discuss.", "instructions": "Reply by Friday."},
    }
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        generate_week(week_data, out, "SAMPLE_101")
        overview = (out / "week_03" / "week_03_overview.html").read_text()
        return overview


class TestTemplateChromeEmit:
    def test_footer_has_template_chrome_role(self):
        html = _render_minimal_page()
        # The footer Courseforge emits is the per-page copyright / chrome.
        assert 'data-cf-role="template-chrome"' in html
        # And it appears specifically on the footer element (not just somewhere).
        # Crude but definitive: look for the footer opening tag with the role.
        assert '<footer role="contentinfo" data-cf-role="template-chrome">' in html

    def test_header_has_template_chrome_role(self):
        html = _render_minimal_page()
        assert '<header role="banner" data-cf-role="template-chrome">' in html

    def test_skip_link_has_template_chrome_role(self):
        """Skip-to-main-content links are chrome by definition — repeated
        on every page, assistive-tech metadata only."""
        html = _render_minimal_page()
        assert 'class="skip-link" data-cf-role="template-chrome"' in html

    def test_main_does_not_have_template_chrome_role(self):
        """Content is content — <main> must NOT carry the chrome role."""
        html = _render_minimal_page()
        # `<main id="main-content" role="main">` is the content container;
        # it must not be chrome-flagged.
        assert '<main id="main-content" role="main">' in html
        # Defensive: the string `data-cf-role="template-chrome"` must not
        # appear on the <main> opening tag.
        import re
        main_match = re.search(r"<main[^>]*>", html)
        assert main_match is not None
        assert "template-chrome" not in main_match.group(0)
