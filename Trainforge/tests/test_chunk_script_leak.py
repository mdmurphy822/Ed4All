"""Regression: inline ``<script>`` / ``<style>`` content must never leak into
chunk ``text``, and char_spans must map to real content.

Root cause (pre-fix): ``HTMLContentParser._extract_sections`` ran its
heading regex (``<h([1-6])...>``) over the *raw* page HTML. A self-check /
application page's inline grading ``<script>`` builds result markup as a JS
string literal — e.g. ``resultEl.innerHTML = '<h3>Your Results</h3>' + msg`` —
so the ``<h3>`` *inside the JavaScript string* matched the heading regex and
created a PHANTOM section boundary. The body slice for that phantom heading
began mid-script (no opening ``<script>`` tag in the slice), so
``HTMLTextExtractor`` never entered script-skip mode and emitted the raw JS as
the section ``content``. That JS then leaked into the chunk ``text``, and
because the JS does not appear in the script-stripped ``container_text`` the
chunker locates spans against, the chunk's ``char_span`` was fabricated
(``_locate``'s no-match fallback) — citation anchoring flagged it
``SPAN_FABRICATED`` and the fail-closed citation gate blocked any answer
citing it.

Fix: ``_extract_sections`` strips ``<script>`` / ``<style>`` subtrees BEFORE
the heading regex (``_SCRIPT_STYLE_RE``), reusing the canonical
``<script ...>...</script>`` strip pattern (``re.S | re.I``) already used by
``lib/retrieval/citation_anchor.py``. Pages without inline script extract
byte-identically (the regex no-ops).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from Trainforge.chunker import ChunkerContext, chunk_content  # noqa: E402
from Trainforge.parsers.html_content_parser import HTMLContentParser  # noqa: E402
from Trainforge.parsers.xpath_walker import resolve_xpath  # noqa: E402


# A self-check page whose inline grading script builds an ``<h3>`` heading as a
# JavaScript string literal — the exact shape that produced the phantom-heading
# leak on the live corpus.
SCRIPT_BEARING_HTML = """<!DOCTYPE html>
<html>
<head><title>Self-Check Page</title></head>
<body>
<section data-cf-template-type="self-check" data-cf-objective-id="TO-03">
  <h2>Self-Check: Core Concepts</h2>
  <p>Answer these questions to test your understanding of the material.</p>
  <div class="self-check" id="sc1">
    <p>Question one prose that is genuine page content.</p>
  </div>
  <script>
    function gradeSelfCheck() {
      var count = countCorrect();
      var feedback =
        '\\n + ' + ' ' + (count === 6 ? 'Excellent — full marks!' : 'Review the material');
      resultEl.innerHTML = '<h3>Your Results</h3>' + feedback;
    }
    document.querySelector('.btn').addEventListener('click', gradeSelfCheck);
  </script>
  <h2>Application Exercise</h2>
  <p>Now apply the concepts you reviewed to a realistic scenario.</p>
</section>
</body>
</html>
"""

# A few characteristic JavaScript fragments that must never appear in any
# emitted chunk text. Kept distinct from prose so a substring check is sharp.
_JS_FRAGMENTS = (
    "count === 6",
    "resultEl.innerHTML",
    "Excellent — full marks!",
    "function gradeSelfCheck",
    "addEventListener",
    "' + feedback;",
)

# The phantom heading the JS string literal carried. It must NOT be promoted to
# a real section heading.
_PHANTOM_HEADING = "Your Results"


class _RecordingContext:
    """Minimal ChunkerContext callback that mirrors the production source dict.

    Captures the chunker-computed ``text`` / ``html_xpath`` / ``char_span`` so
    the test can assert (a) no JS leaks into ``text`` and (b) the span maps to
    real content via xpath round-trip.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "id": kwargs["chunk_id"],
            "text": kwargs["text"],
            "source": {
                "html_xpath": kwargs.get("html_xpath"),
                "char_span": kwargs.get("char_span"),
            },
        }


def _parse_to_item(html: str) -> Dict[str, Any]:
    """Run the real HTML parser + shape the parsed_item the IMSCC path builds.

    Mirrors ``MCP/tools/pipeline_tools.py::_run_imscc_chunking``'s parsed_item
    construction (the fields ``chunk_content`` reads).
    """
    parsed = HTMLContentParser().parse(html)
    return {
        "module_id": "self-check-page",
        "item_id": "self-check-page",
        "module_title": parsed.title or "Self-Check Page",
        "title": parsed.title or "Self-Check Page",
        "resource_type": "page",
        "raw_html": html,
        "sections": parsed.sections,
        "misconceptions": [],
    }


def test_extract_sections_drops_phantom_heading_inside_script() -> None:
    """The ``<h3>`` inside the JS string literal must not become a section."""
    sections = HTMLContentParser()._extract_sections(SCRIPT_BEARING_HTML)
    headings = [s.heading for s in sections]

    assert _PHANTOM_HEADING not in headings, (
        f"phantom heading promoted to a real section: {headings!r}"
    )
    # The two genuine headings survive.
    assert "Self-Check: Core Concepts" in headings
    assert "Application Exercise" in headings

    # No section body carries any JavaScript fragment.
    for section in sections:
        for frag in _JS_FRAGMENTS:
            assert frag not in section.content, (
                f"script content leaked into section {section.heading!r}: "
                f"{frag!r} found in {section.content!r}"
            )


def test_imscc_chunks_carry_no_script_text() -> None:
    """End-to-end: no emitted chunk's ``text`` contains script fragments."""
    item = _parse_to_item(SCRIPT_BEARING_HTML)
    recorder = _RecordingContext()
    ctx = ChunkerContext(create_chunk=recorder)

    result = chunk_content([item], course_code="TEST_101", ctx=ctx)

    assert result.chunks, "expected at least one chunk from the self-check page"
    for chunk in result.chunks:
        text = chunk["text"]
        for frag in _JS_FRAGMENTS:
            assert frag not in text, (
                f"script content leaked into chunk {chunk['id']}: "
                f"{frag!r} found in chunk text {text!r}"
            )


def test_imscc_chunk_spans_map_to_real_content() -> None:
    """Every chunk's char_span resolves to its own text in the container.

    A fabricated span (the pre-fix failure mode) would NOT round-trip: the
    leaked JS text does not appear in the script-stripped ``container_text``
    the xpath walker resolves, so the slice would not contain the chunk prose.
    """
    item = _parse_to_item(SCRIPT_BEARING_HTML)
    recorder = _RecordingContext()
    ctx = ChunkerContext(create_chunk=recorder)

    result = chunk_content([item], course_code="TEST_101", ctx=ctx)
    assert result.chunks

    for call in recorder.calls:
        xpath = call.get("html_xpath")
        span = call.get("char_span")
        text = call["text"]
        assert xpath, f"chunk {call['chunk_id']} has no html_xpath"
        assert span and span[1] > span[0], (
            f"chunk {call['chunk_id']} has a non-positive span: {span}"
        )

        container_text = resolve_xpath(SCRIPT_BEARING_HTML, xpath) or ""
        # The resolved container is script-stripped by the xpath walker. A
        # genuine (non-fabricated) span means the chunk's leading prose is
        # locatable in that container text.
        prose_head = " ".join(text.split()[:6])
        collapsed_container = " ".join(container_text.split())
        assert prose_head and prose_head in collapsed_container, (
            f"chunk {call['chunk_id']} span does not map to real container "
            f"content: prose_head={prose_head!r} not in container"
        )


def test_script_free_page_is_unaffected() -> None:
    """A page with no inline script extracts identically (regex no-ops)."""
    html = """<!DOCTYPE html><html><head><title>Plain</title></head><body>
    <section><h2>Topic One</h2><p>First section prose.</p>
    <h2>Topic Two</h2><p>Second section prose.</p></section>
    </body></html>"""

    sections = HTMLContentParser()._extract_sections(html)
    headings = [s.heading for s in sections]
    assert headings == ["Topic One", "Topic Two"]
    assert sections[0].content == "First section prose."
    assert sections[1].content == "Second section prose."
