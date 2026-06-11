"""Per-status fragment structure + copy + escaping tests for ``answer_render``.

Pure ``dict -> str`` renderer tests: no FastAPI, no LibV2, no model. They pin
the rendered-fragment contract that the WS-D a11y gate validates — heading
structure, citation link hrefs, adversarial-input escaping, and the frozen
display copy per status.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from gui.services import answer_render as ar


# --------------------------------------------------------------------- helpers


class _Collector(HTMLParser):
    """Minimal well-formedness + structure collector over a fragment string."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list = []
        self.hrefs: list = []
        self.data: list = []

    def handle_starttag(self, tag, attrs):  # noqa: D401
        self.tags.append(tag)
        for k, v in attrs:
            if k == "href" and v is not None:
                self.hrefs.append(v)

    def handle_data(self, data):  # noqa: D401
        self.data.append(data)


def _parse(html: str) -> _Collector:
    c = _Collector()
    c.feed(html)
    return c


def _answered_payload(**overrides):
    base = {
        "status": ar.STATUS_ANSWERED,
        "course_slug": "phys-101",
        "answer_text": "Velocity is a vector.\n\nIt has magnitude and direction.",
        "citations": [
            {
                "chunk_id": "chunk_0001",
                "item_path": "ch01/velocity.html",
                "section_heading": "Velocity",
                "module_id": "M1",
                "page_label": "Velocity",
                "anchor_status": "resolved_exact",
                "source_path": "/x/velocity.html",
                "text_quote": "Velocity is the rate of change of position.",
                "link_target": {
                    "kind": "course_page",
                    "item_path": "ch01/velocity.html",
                    "fragment": {"kind": "heading", "value": "velocity"},
                    "char_span": [0, 10],
                },
            }
        ],
        "refusal": None,
        "confidence": {},
        "groundedness": None,
        "warnings": [],
        "model_id": "qwen2.5:7b",
        "prompt_version": "v1",
        "generated_at": "2026-06-09T00:00:00Z",
        "latency_ms": 1234.0,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------- skeleton


def test_answered_skeleton_and_heading():
    html = ar.render_answer_fragment(_answered_payload())
    c = _parse(html)
    assert html.startswith('<section class="answer" data-status="answered"')
    assert 'aria-labelledby="answer-h"' in html
    # exactly one h2 with the tabindex + id
    assert html.count("<h2 ") == 1
    assert 'id="answer-h"' in html
    assert 'tabindex="-1"' in html
    assert ">Answer</h2>" in html
    # heading hierarchy: h2 then h3 (Sources), never an h1/h4 in the fragment
    assert "<h1" not in html and "<h4" not in html
    assert "<h3>Sources</h3>" in html
    assert c.tags  # parsed without error


def test_answered_citation_link_text_and_href():
    html = ar.render_answer_fragment(_answered_payload())
    c = _parse(html)
    # link text is the full "Source: {page_label}"
    assert ">Source: Velocity</a>" in html
    assert len(c.hrefs) == 1
    href = c.hrefs[0]
    assert href.startswith("/api/learn/source/phys-101?item_path=")
    # heading fragment → URL #slug
    assert href.endswith("#velocity")
    assert "ch01" in href  # item_path urlencoded but recognisable
    # module text tag + ol.sources
    assert '<span class="src-module">Module M1</span>' in html
    assert '<ol class="sources">' in html
    # resolved_exact → no approximate marker
    assert "approximate location" not in html
    assert "<blockquote" in html


def test_non_heading_fragment_has_no_url_fragment():
    payload = _answered_payload()
    payload["citations"][0]["link_target"]["fragment"] = {"kind": "xpath", "value": "/html/body"}
    html = ar.render_answer_fragment(payload)
    href = _parse(html).hrefs[0]
    assert "#" not in href  # xpath / None → top-of-page


def test_no_approximate_marker_rendered():
    """The hedge text is gone by user directive: the Source link is the
    direct hop to the chunk's holder, regardless of anchor exactness."""
    payload = _answered_payload()
    payload["citations"][0]["anchor_status"] = "resolved_heading"
    html = ar.render_answer_fragment(payload)
    assert "approximate location" not in html
    assert "src-approx" not in html


def test_answered_with_warnings_advisory_first():
    html = ar.render_answer_fragment(
        _answered_payload(status=ar.STATUS_ANSWERED_WITH_WARNINGS)
    )
    assert 'data-status="answered_with_warnings"' in html
    assert ">Answer</h2>" in html
    # advisory note is the FIRST child after the heading
    note = '<p class="advisory" role="note">'
    assert note in html
    h2_close = html.index("</h2>")
    assert html.index(note) < html.index("<p>Velocity")
    assert html.index(note) > h2_close


# --------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    "status,heading,first_body",
    [
        (ar.STATUS_REFUSED_LOW_CONFIDENCE, "No answer found", "find course material"),
        (ar.STATUS_REFUSED_NOT_IN_COURSE, "Not covered in this course", "appear to be covered"),
        (ar.STATUS_BLOCKED_INVALID_CITATION, "Answer withheld", "verify it against"),
        (ar.STATUS_BLOCKED_CITATION_GATE, "Answer withheld", "verify it against"),
    ],
)
def test_refused_blocked_copy_and_no_citations(status, heading, first_body):
    # Even if a (contract-violating) payload smuggles answer_text/citations,
    # the renderer must NOT leak them for these statuses.
    payload = _answered_payload(
        status=status,
        answer_text="LEAKED ANSWER TEXT",
        citations=_answered_payload()["citations"],
    )
    html = ar.render_answer_fragment(payload)
    assert f'data-status="{status}"' in html
    assert f">{heading}</h2>" in html
    assert first_body in html
    assert "LEAKED ANSWER TEXT" not in html
    assert "Source:" not in html
    assert "<ol" not in html
    assert "<h3" not in html


# ----------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    "key,heading_substr",
    [
        ("error_backend_down", "The answer engine"),
        ("error_misconfigured", "Something went wrong"),
        ("error_index", "Something went wrong"),
        ("error_compose", "Something went wrong"),
        ("error_generic", "Something went wrong"),
    ],
)
def test_error_fragment_copy(key, heading_substr):
    html = ar.render_error_fragment(key)
    assert f'data-status="{key}"' in html
    assert heading_substr in html
    assert "</h2>" in html
    assert 'tabindex="-1"' in html
    assert "facilitator" in html


def test_unknown_error_key_falls_back_generic():
    html = ar.render_error_fragment("does_not_exist")
    assert ">Something went wrong</h2>" in html


def test_unknown_status_renders_generic_error():
    html = ar.render_answer_fragment(_answered_payload(status="bogus_status"))
    assert ">Something went wrong</h2>" in html


# -------------------------------------------------------------------- escaping


def test_adversarial_answer_text_is_escaped():
    payload = _answered_payload(
        answer_text="<script>alert('xss')</script> & <b>bold</b>",
        citations=[],
    )
    html = ar.render_answer_fragment(payload)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&amp;" in html


def test_adversarial_citation_fields_are_escaped():
    payload = _answered_payload()
    payload["citations"][0]["page_label"] = '<img src=x onerror=alert(1)>'
    payload["citations"][0]["text_quote"] = "</blockquote><script>x</script>"
    payload["citations"][0]["module_id"] = '"><script>'
    html = ar.render_answer_fragment(payload)
    assert "<img src=x" not in html
    assert "<script>" not in html
    assert "&lt;img" in html


def test_adversarial_item_path_in_href_is_url_encoded():
    payload = _answered_payload()
    payload["citations"][0]["item_path"] = 'a"b<c>'
    payload["citations"][0]["link_target"]["item_path"] = 'a"b<c>'
    payload["citations"][0]["link_target"]["fragment"] = {"kind": None, "value": None}
    html = ar.render_answer_fragment(payload)
    href = _parse(html).hrefs[0]
    # raw quote / angle brackets must not appear unencoded in the attribute
    assert '"' not in href.split("item_path=")[1]
    assert "<" not in href and ">" not in href


def test_source_url_for_builds_expected_shape():
    cit = {
        "item_path": "ch 1/p.html",
        "link_target": {
            "item_path": "ch 1/p.html",
            "fragment": {"kind": "heading", "value": "intro-section"},
        },
    }
    url = ar.source_url_for(cit, "bio-201")
    assert url.startswith("/api/learn/source/bio-201?item_path=")
    assert "ch%201" in url or "ch+1" in url  # space encoded
    assert url.endswith("#intro-section")


def test_no_metadata_leak_in_fragment():
    html = ar.render_answer_fragment(_answered_payload())
    # latency / model_id / confidence are JSON-only, never in learner HTML
    assert "1234" not in html
    assert "qwen2.5" not in html
    assert "prompt_version" not in html


# ------------------------------------------ claim-attribution surfacing (2026-06)


def test_supporting_excerpt_preferred_over_text_quote():
    """The src-quote blockquote prefers supporting_excerpt over text_quote."""
    payload = _answered_payload()
    payload["citations"][0]["text_quote"] = "Legacy whole-answer overlap sentence."
    payload["citations"][0]["supporting_excerpt"] = "The claim-backing sentence."
    html = ar.render_answer_fragment(payload)
    assert "<blockquote" in html
    assert "The claim-backing sentence." in html
    assert "Legacy whole-answer overlap sentence." not in html


def test_text_quote_fallback_when_no_excerpt():
    payload = _answered_payload()
    payload["citations"][0]["text_quote"] = "Fallback quote here."
    payload["citations"][0]["supporting_excerpt"] = None
    html = ar.render_answer_fragment(payload)
    assert "Fallback quote here." in html


def test_supports_n_span_rendered():
    payload = _answered_payload()
    payload["citations"][0]["supported_claim_count"] = 3
    html = ar.render_answer_fragment(payload)
    assert '<span class="src-support">Supports 3 statements</span>' in html


def test_supports_one_statement_singular():
    payload = _answered_payload()
    payload["citations"][0]["supported_claim_count"] = 1
    html = ar.render_answer_fragment(payload)
    assert '<span class="src-support">Supports 1 statement</span>' in html


def test_supports_span_omitted_when_zero_or_absent():
    # Absent (legacy payload) → no span.
    html = ar.render_answer_fragment(_answered_payload())
    assert "src-support" not in html
    # Explicit zero → no span.
    payload = _answered_payload()
    payload["citations"][0]["supported_claim_count"] = 0
    assert "src-support" not in ar.render_answer_fragment(payload)


def test_supporting_excerpt_is_escaped():
    payload = _answered_payload()
    payload["citations"][0]["text_quote"] = None
    payload["citations"][0]["supporting_excerpt"] = (
        "</blockquote><script>alert(1)</script>"
    )
    html = ar.render_answer_fragment(payload)
    assert "<script>alert(1)" not in html
    assert "&lt;script&gt;" in html


# --------------------------------------------------- B4 provenance disclosure


def test_legacy_citation_omits_provenance_disclosure():
    # No source_block / pdf_pages → no disclosure toggle at all.
    html = ar.render_answer_fragment(_answered_payload())
    assert "src-detail-toggle" not in html
    assert "PDF page" not in html


def test_provenance_disclosure_renders_block_and_pdf_links():
    payload = _answered_payload()
    payload["citations"][0]["source_block"] = "dart:mini_alpha#s3_c0"
    payload["citations"][0]["pdf_pages"] = [7, 12]
    html = ar.render_answer_fragment(payload)
    # Disclosure pattern: a toggle button controlling a hidden detail panel.
    assert '<button type="button" class="src-detail-toggle" aria-expanded="false"' in html
    assert 'aria-controls="src-detail-chunk_0001"' in html
    assert 'id="src-detail-chunk_0001" class="src-detail" hidden' in html
    # Informational source block id.
    assert "<code>dart:mini_alpha#s3_c0</code>" in html
    # One labelled, new-tab PDF link per page.
    assert html.count('class="src-pdf-link"') == 2
    assert 'target="_blank" rel="noopener"' in html
    assert 'aria-label="Open PDF page 7, opens in new tab"' in html
    assert "source-pdf?file=mini_alpha&amp;page=12" in html
    assert ">PDF page 12</a>" in html


def test_provenance_block_only_when_no_pages():
    payload = _answered_payload()
    payload["citations"][0]["source_block"] = "dart:mini_alpha#s3_c0"
    payload["citations"][0]["pdf_pages"] = []
    html = ar.render_answer_fragment(payload)
    assert "src-detail-toggle" in html
    assert "<code>dart:mini_alpha#s3_c0</code>" in html
    # No PDF links when no pages are known.
    assert "src-pdf-link" not in html


def test_provenance_pages_only_when_no_block():
    payload = _answered_payload()
    payload["citations"][0]["source_block"] = None
    payload["citations"][0]["pdf_pages"] = [3]
    html = ar.render_answer_fragment(payload)
    assert "src-detail-toggle" in html
    assert "src-pdf-link" in html
    assert ">PDF page 3</a>" in html
    assert "src-block" not in html


def test_provenance_source_block_is_escaped():
    payload = _answered_payload()
    payload["citations"][0]["source_block"] = 'dart:x#"><script>alert(1)</script>'
    payload["citations"][0]["pdf_pages"] = []
    html = ar.render_answer_fragment(payload)
    assert "<script>alert(1)" not in html
    assert "&lt;script&gt;" in html


# ------------------------------------------ original-source deep link (req. 2)


def test_original_source_link_rendered_when_source_block_present():
    payload = _answered_payload()
    payload["citations"][0]["source_block"] = "dart:sample-algebra-2e-ch1-3#edd36456647a9cd8"
    payload["citations"][0]["pdf_pages"] = []
    html = ar.render_answer_fragment(payload)
    # The source-block row is a "View original source" deep link (new tab).
    assert 'class="src-original-link"' in html
    assert ">View original source (accessible HTML)</a>" in html
    assert 'target="_blank" rel="noopener"' in html
    assert "opens in new tab" in html
    # href: /api/courses/{slug}/source-doc?doc=<doc>&ref=<block>#dart-<block>
    assert (
        "/api/courses/phys-101/source-doc?doc=sample-algebra-2e-ch1-3"
        "&amp;ref=edd36456647a9cd8#dart-edd36456647a9cd8" in html
    )
    # The sourceId rides along as secondary <code> text.
    assert "<code>dart:sample-algebra-2e-ch1-3#edd36456647a9cd8</code>" in html


def test_original_source_url_helper_shape():
    cit = {"source_block": "dart:foo-doc#blk_9"}
    url = ar.original_source_url(cit, "my-course")
    assert url == "/api/courses/my-course/source-doc?doc=foo-doc&ref=blk_9#dart-blk_9"
    # Non-dart / malformed source blocks → empty (no link emitted).
    assert ar.original_source_url({"source_block": "imscc:x#y"}, "s") == ""
    assert ar.original_source_url({"source_block": "dart:onlydoc"}, "s") == ""
    assert ar.original_source_url({}, "s") == ""


def test_original_source_link_suppressed_when_no_source_block():
    # Legacy citation (no source_block) → no original-source link at all.
    html = ar.render_answer_fragment(_answered_payload())
    assert "src-original-link" not in html


def test_original_source_link_suppressed_when_flag_off():
    payload = _answered_payload()
    payload["citations"][0]["source_block"] = "dart:foo#blk_1"
    payload["citations"][0]["pdf_pages"] = [3]
    html = ar.render_answer_fragment(payload, include_original_source_links=False)
    # No original-source link, no PDF-page link (both gated by the toggle).
    assert "src-original-link" not in html
    assert "src-pdf-link" not in html
    # The informational <code> sourceId still shows (it is not a redistribution
    # link, just an identifier).
    assert "<code>dart:foo#blk_1</code>" in html


def test_original_source_link_url_encoded():
    payload = _answered_payload()
    payload["citations"][0]["source_block"] = "dart:doc with space#blk a"
    payload["citations"][0]["pdf_pages"] = []
    html = ar.render_answer_fragment(payload)
    # The doc / ref halves are percent-encoded in the href.
    assert "doc=doc%20with%20space" in html
    assert "ref=blk%20a" in html
