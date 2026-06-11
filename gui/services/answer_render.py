"""Server-side accessible HTML fragment rendering for the learner answer UI.

This module is THE single rendering path (D2): the answer region's HTML is
produced *once*, here, in Python. ``POST /api/learn/ask`` returns both the raw
``GroundedAnswer.to_dict()`` JSON and the rendered ``html`` string; the learner
JS swaps the string into the answer region verbatim and never re-derives markup
from the JSON (no drift between a JS renderer and the gated renderer). The same
strings are what the automated WCAG 2.2 AA gate validates.

Display copy per status / error key is **WS4-owned** (the grounded-answer stack
deliberately leaves learner-facing copy out; refusal copy keys off
``refusal.reason_code`` and never off model internals). The frozen copy lives in
``STATUS_COPY`` / ``ERROR_COPY`` below.

No template engine, no new deps: stdlib ``html.escape`` on every dynamic string.
Heading structure is fixed (one ``<h2>`` per fragment, ``<h3>`` for the sources
list) so the fragment slots under the page's single ``<h1>`` without skipping a
level. Citations render as focusable text links ("Source: {page_label}") whose
``href`` is built server-side from the citation's ``link_target`` — the JS never
constructs URLs.
"""

from __future__ import annotations

import re
from html import escape
from typing import Any, Dict, List, Optional
from urllib.parse import quote

# --------------------------------------------------------------------------- #
# Status / reason-code constants (mirrors of the grounded-answer contract).
#
# Re-declared as literals (not imported) so this module stays import-light: the
# renderer is a pure ``dict -> str`` transform with zero LibV2 / Trainforge
# dependency, importable in any test without the retrieval stack.
# --------------------------------------------------------------------------- #

STATUS_ANSWERED = "answered"
STATUS_ANSWERED_WITH_WARNINGS = "answered_with_warnings"
STATUS_REFUSED_LOW_CONFIDENCE = "refused_low_confidence"
STATUS_REFUSED_NOT_IN_COURSE = "refused_not_in_course"
STATUS_BLOCKED_INVALID_CITATION = "blocked_invalid_citation"
STATUS_BLOCKED_CITATION_GATE = "blocked_citation_gate"

REASON_LOW_CONFIDENCE = "low_confidence"
REASON_NOT_IN_COURSE_MODEL = "not_in_course_model"

_ANSWERED_STATUSES = frozenset(
    {STATUS_ANSWERED, STATUS_ANSWERED_WITH_WARNINGS}
)
_REFUSED_BLOCKED_STATUSES = frozenset(
    {
        STATUS_REFUSED_LOW_CONFIDENCE,
        STATUS_REFUSED_NOT_IN_COURSE,
        STATUS_BLOCKED_INVALID_CITATION,
        STATUS_BLOCKED_CITATION_GATE,
    }
)


# --------------------------------------------------------------------------- #
# Frozen display copy (D3).
#
# ``heading`` is the <h2> text; ``body`` is a list of paragraph strings rendered
# as <p> elements for the refusal/blocked/error variants. The ``answered`` /
# ``answered_with_warnings`` bodies are assembled programmatically (answer text +
# sources), so they carry no static body paragraphs here.
# --------------------------------------------------------------------------- #

STATUS_COPY: Dict[str, Dict[str, Any]] = {
    STATUS_ANSWERED: {"heading": "Answer", "body": []},
    STATUS_ANSWERED_WITH_WARNINGS: {"heading": "Answer", "body": []},
    STATUS_REFUSED_LOW_CONFIDENCE: {
        "heading": "No answer found",
        "body": [
            "We couldn't find course material that answers this question "
            "confidently.",
            "Try rephrasing your question, or use terms from the course "
            "materials.",
        ],
    },
    STATUS_REFUSED_NOT_IN_COURSE: {
        "heading": "Not covered in this course",
        "body": [
            "This doesn't appear to be covered in this course's materials.",
            "Try asking about a topic from the course.",
        ],
    },
    STATUS_BLOCKED_INVALID_CITATION: {
        "heading": "Answer withheld",
        "body": [
            "We found a possible answer but couldn't verify it against the "
            "course materials, so it wasn't shown.",
            "Try rephrasing your question.",
        ],
    },
    STATUS_BLOCKED_CITATION_GATE: {
        "heading": "Answer withheld",
        "body": [
            "We found a possible answer but couldn't verify it against the "
            "course materials, so it wasn't shown.",
            "Try rephrasing your question.",
        ],
    },
}

# The advisory note prepended to ``answered_with_warnings`` (first child).
_ADVISORY_COPY = (
    "Parts of this answer may not be fully supported by the course "
    "materials. Check the sources below."
)

# Error-key copy (the typed-error map's learner-facing keys, § 4.4). The
# operator-actionable detail rides the JSON ``detail`` field and is NEVER
# rendered here.
ERROR_COPY: Dict[str, Dict[str, Any]] = {
    "error_backend_down": {
        "heading": "The answer engine isn't available",
        "body": [
            "The local answer engine isn't running. Please tell the session "
            "facilitator.",
        ],
    },
    "error_misconfigured": {
        "heading": "Something went wrong",
        "body": [
            "The course assistant hit a problem and couldn't answer. Please "
            "tell the session facilitator.",
        ],
    },
    "error_index": {
        "heading": "Something went wrong",
        "body": [
            "The course assistant hit a problem and couldn't answer. Please "
            "tell the session facilitator.",
        ],
    },
    "error_compose": {
        "heading": "Something went wrong",
        "body": [
            "The course assistant hit a problem and couldn't answer. Please "
            "tell the session facilitator.",
        ],
    },
    "error_generic": {
        "heading": "Something went wrong",
        "body": [
            "The course assistant hit a problem and couldn't answer. Please "
            "tell the session facilitator.",
        ],
    },
}

_ANSWER_HEADING_ID = "answer-h"


# --------------------------------------------------------------------------- #
# Citation link URL (server-side; the JS never constructs URLs).
# --------------------------------------------------------------------------- #


def source_url_for(citation: Dict[str, Any], slug: str) -> str:
    """Build the citation-back URL for one citation.

    ``/api/learn/source/{slug}?item_path={urlencoded}`` plus a ``#{slug}`` URL
    fragment **only** when ``link_target.fragment.kind == "heading"`` (xpath /
    None fragments land at top-of-page; the viewer adds a banner note). The
    fragment value is the heading slug already minted by the grounded-answer
    stack — re-used verbatim so navigation lands on the cited section.
    """
    item_path = citation.get("item_path") or ""
    link_target = citation.get("link_target") or {}
    # Prefer the link_target's item_path (the anchor-resolved one) when present.
    target_item = link_target.get("item_path") or item_path
    url = "/api/learn/source/{slug}?item_path={item}".format(
        slug=quote(str(slug), safe=""),
        item=quote(str(target_item), safe=""),
    )
    fragment = link_target.get("fragment") or {}
    if fragment.get("kind") == "heading" and fragment.get("value"):
        # The slug is URL-fragment-safe by construction (lowercased a-z0-9 +
        # hyphen), but quote defensively in case of contract drift.
        url = "{url}#{frag}".format(
            url=url, frag=quote(str(fragment["value"]), safe="-")
        )
    return url


# --------------------------------------------------------------------------- #
# Fragment assembly helpers.
# --------------------------------------------------------------------------- #


def _section(status: str, heading: str, inner: str) -> str:
    """Wrap the inner body in the fixed answer-region skeleton.

    The ``<h2>`` carries ``tabindex="-1"`` so the JS can move focus to it after
    a swap (D6), and the section is ``aria-labelledby`` the heading id.
    """
    return (
        '<section class="answer" data-status="{status}" '
        'aria-labelledby="{hid}">'
        '<h2 id="{hid}" tabindex="-1">{heading}</h2>'
        "{inner}"
        "</section>"
    ).format(
        status=escape(status, quote=True),
        hid=_ANSWER_HEADING_ID,
        heading=escape(heading),
        inner=inner,
    )


def _paragraphs(blocks: List[str]) -> str:
    """Render a list of body strings as escaped ``<p>`` elements."""
    return "".join("<p>{}</p>".format(escape(b)) for b in blocks)


def _answer_paragraphs(answer_text: Optional[str]) -> str:
    """Split the answer text on blank lines into escaped ``<p>`` paragraphs."""
    if not answer_text:
        return ""
    # Split on one-or-more blank lines; normalise CRLF first.
    normalized = str(answer_text).replace("\r\n", "\n").replace("\r", "\n")
    raw_paras = [p.strip() for p in normalized.split("\n\n")]
    paras = [p for p in raw_paras if p]
    if not paras:
        # No blank-line separators: treat the whole thing as one paragraph.
        stripped = normalized.strip()
        paras = [stripped] if stripped else []
    return "".join("<p>{}</p>".format(escape(p)) for p in paras)


def _pdf_pages_label(pages: List[int]) -> str:
    """Compact human label for a sorted PDF-page list: "page 12" / "pages 3, 7"."""
    if len(pages) == 1:
        return "page {}".format(pages[0])
    return "pages {}".format(", ".join(str(p) for p in pages))


def source_pdf_page_url(citation: Dict[str, Any], slug: str, page: int) -> str:
    """Build the deep link to one archived-PDF page (B4 final hop).

    ``/api/courses/{slug}/source-pdf?file=<basename>&page=N``. The file name is
    the basename of the citation's ``source_path`` (the archived course-page
    HTML path) when present, else derived from the sourceId slug; the endpoint
    re-resolves the actual PDF via the archived sidecars and whitelists the
    filename against the source dir, so an unresolvable name 404s rather than
    serving the wrong bytes.
    """
    source_id = str(citation.get("source_block") or "")
    # sourceId shape: dart:{slug}#{block_id}. The slug half names the document.
    file_hint = ""
    if source_id.startswith("dart:") and "#" in source_id:
        file_hint = source_id[len("dart:") :].split("#", 1)[0]
    return "/api/courses/{slug}/source-pdf?file={file}&page={page}".format(
        slug=quote(str(slug), safe=""),
        file=quote(file_hint, safe=""),
        page=int(page),
    )


def original_source_url(citation: Dict[str, Any], slug: str) -> str:
    """Build the deep link to the original archived DART document for one citation.

    ``/api/courses/{slug}/source-doc?doc=<doc>&ref=<block>#dart-<block>`` from the
    citation's ``source_block`` (``dart:{doc}#{block}``). The endpoint serves the
    sanitized accessible HTML with a serve-time-injected ``id="dart-{block}"``, so
    the ``#dart-<block>`` fragment lands on the exact cited block. Emit-then-
    resolve: the link is built whenever ``source_block`` parses; the endpoint
    404s with an explanation when the DART doc isn't archived (provenance-without-
    artifacts archives) — no per-render existence probe (the renderer is a pure
    dict→str transform that must stay import-light, mirroring the source-pdf hop).

    Returns ``""`` when ``source_block`` doesn't parse as ``dart:{doc}#{block}``.
    """
    source_id = str(citation.get("source_block") or "")
    if not source_id.startswith("dart:") or "#" not in source_id:
        return ""
    doc, block = source_id[len("dart:"):].split("#", 1)
    doc = doc.strip()
    block = block.strip()
    if not doc or not block:
        return ""
    return "/api/courses/{slug}/source-doc?doc={doc}&ref={ref}#dart-{frag}".format(
        slug=quote(str(slug), safe=""),
        doc=quote(doc, safe=""),
        ref=quote(block, safe=""),
        frag=quote(block, safe="-"),
    )


def _citation_provenance(
    citation: Dict[str, Any], slug: str, *, include_original_source_links: bool = True
) -> str:
    """Render the expandable provenance detail row for one citation (B4).

    A disclosure: a ``<button class="src-detail-toggle" aria-expanded="false"
    aria-controls=...>`` over a ``<div class="src-detail" hidden>`` carrying the
    full chain — the informational source block id and one labelled link per PDF
    page. Absent fields simply omit their entries (legacy corpora show only what
    they have, so the toggle is suppressed entirely when there is nothing to
    expand). The JS (drawer.js) re-wires the page links to ``target=_blank``;
    the markup is fully server-built so the JS never constructs URLs.

    When ``include_original_source_links`` (the operator toggle, §2.5) and the
    ``source_block`` parses as ``dart:{doc}#{block}``, the source-block row renders
    as a "View original source (accessible HTML)" deep link (new tab) instead of
    plain ``<code>`` text — closing the requirement-2 gap (a citation that hops to
    the exact original-document position). The ``<code>`` sourceId rides along as
    secondary operator/debug text.
    """
    source_block = citation.get("source_block")
    pdf_pages = [p for p in (citation.get("pdf_pages") or []) if isinstance(p, int)]
    if not source_block and not pdf_pages:
        return ""

    chunk_id = str(citation.get("chunk_id") or "")
    panel_id = "src-detail-{}".format(
        re.sub(r"[^a-zA-Z0-9_-]+", "-", chunk_id) or "x"
    )

    rows: List[str] = []
    if source_block:
        original_url = (
            original_source_url(citation, slug)
            if include_original_source_links
            else ""
        )
        if original_url:
            # Link the original source + keep the sourceId as secondary text.
            rows.append(
                '<li class="src-block">'
                '<a href="{href}" class="src-original-link" '
                'target="_blank" rel="noopener" '
                'aria-label="View original source (accessible HTML), opens in new tab">'
                'View original source (accessible HTML)</a> '
                '<code>{sid}</code></li>'.format(
                    href=escape(original_url, quote=True),
                    sid=escape(str(source_block)),
                )
            )
        else:
            rows.append(
                '<li class="src-block">Source block '
                '<code>{}</code></li>'.format(escape(str(source_block)))
            )
    if include_original_source_links:
        for page in pdf_pages:
            href = source_pdf_page_url(citation, slug, page)
            rows.append(
                '<li class="src-pdf"><a href="{href}" class="src-pdf-link" '
                'target="_blank" rel="noopener" '
                'aria-label="Open PDF page {page}, opens in new tab">'
                'PDF page {page}</a></li>'.format(
                    href=escape(href, quote=True), page=int(page)
                )
            )
    if not rows:
        # The toggle is off and there were only links to show → nothing to expand.
        return ""
    detail = (
        '<button type="button" class="src-detail-toggle" '
        'aria-expanded="false" aria-controls="{pid}">Provenance</button>'
        '<ul id="{pid}" class="src-detail" hidden>{rows}</ul>'
    ).format(pid=escape(panel_id, quote=True), rows="".join(rows))
    return detail


def _citation_li(
    citation: Dict[str, Any], slug: str, *, include_original_source_links: bool = True
) -> str:
    """Render one citation as a focusable ``<li>`` with a "Source:" link.

    Order: link → (module tag) → (approximate-location tag) → (text quote) →
    (B4 provenance disclosure). Every dynamic field is escaped. ``module_id``
    and the approximate marker are text spans (never color-only state).
    """
    page_label = citation.get("page_label") or "Source"
    href = source_url_for(citation, slug)
    parts = [
        '<a href="{href}">Source: {label}</a>'.format(
            href=escape(href, quote=True), label=escape(str(page_label))
        )
    ]
    # Prefer the human-readable module_title: module_id is a filename stem
    # ("content_01") that repeats across weeks and reads like a mis-citation.
    module_label = citation.get("module_title") or citation.get("module_id")
    if module_label:
        parts.append(
            '<span class="src-module">Module {}</span>'.format(
                escape(str(module_label))
            )
        )
    if citation.get("anchor_status") != "resolved_exact":
        parts.append(
            '<span class="src-approx">(approximate location)</span>'
        )
    text_quote = citation.get("text_quote")
    if text_quote:
        parts.append(
            '<blockquote class="src-quote">{}</blockquote>'.format(
                escape(str(text_quote))
            )
        )
    parts.append(
        _citation_provenance(
            citation, slug, include_original_source_links=include_original_source_links
        )
    )
    return "<li>{}</li>".format("".join(parts))


def _sources_block(
    citations: List[Dict[str, Any]], slug: str, *, include_original_source_links: bool = True
) -> str:
    """Render the ``<h3>Sources</h3><ol>…</ol>`` block, or "" when empty."""
    if not citations:
        return ""
    items = "".join(
        _citation_li(c, slug, include_original_source_links=include_original_source_links)
        for c in citations
    )
    return '<h3>Sources</h3><ol class="sources">{}</ol>'.format(items)


# --------------------------------------------------------------------------- #
# Public render entry points.
# --------------------------------------------------------------------------- #


def render_answer_fragment(
    payload: Dict[str, Any], *, include_original_source_links: bool = True
) -> str:
    """Render one ``GroundedAnswer.to_dict()`` payload to an HTML fragment.

    Dispatches on ``status``. For refused/blocked statuses the contract
    guarantees ``answer_text=None`` and ``citations=[]``; this renderer asserts
    that defensively and renders NO answer text and NO citations regardless of
    what the payload carries, so a contract regression can never leak an
    unverified answer to a learner.

    ``include_original_source_links`` (the operator ``ED4ALL_SOURCE_MATERIALS``
    toggle, §2.5) gates the citation-side original-source + PDF-page deep links;
    the ask path threads it (read once per request). Default ``True`` preserves
    purity for existing tests / the default-on posture.

    An unknown status falls back to the generic-error fragment (fail-safe: a
    learner sees a benign "something went wrong" rather than raw internals).
    """
    status = str(payload.get("status") or "")
    slug = str(payload.get("course_slug") or "")

    if status in _ANSWERED_STATUSES:
        body_parts: List[str] = []
        if status == STATUS_ANSWERED_WITH_WARNINGS:
            body_parts.append(
                '<p class="advisory" role="note">{}</p>'.format(
                    escape(_ADVISORY_COPY)
                )
            )
        body_parts.append(_answer_paragraphs(payload.get("answer_text")))
        citations = payload.get("citations") or []
        body_parts.append(
            _sources_block(
                list(citations),
                slug,
                include_original_source_links=include_original_source_links,
            )
        )
        heading = STATUS_COPY[status]["heading"]
        return _section(status, heading, "".join(body_parts))

    if status in _REFUSED_BLOCKED_STATUSES:
        # Defensive: never render answer text / citations for these statuses.
        copy = STATUS_COPY[status]
        return _section(status, copy["heading"], _paragraphs(copy["body"]))

    # Unknown status → generic error fragment (fail-safe).
    return render_error_fragment("error_generic")


def render_error_fragment(error_key: str) -> str:
    """Render a learner-safe error fragment for a typed-error copy key.

    ``error_key`` is one of the ``ERROR_COPY`` keys; an unknown key falls back
    to ``error_generic``. The operator-actionable detail is NEVER rendered here
    (it stays in the JSON ``detail`` field).
    """
    copy = ERROR_COPY.get(error_key) or ERROR_COPY["error_generic"]
    # Use the error_key as the data-status so the gate / tests can distinguish.
    return _section(error_key, copy["heading"], _paragraphs(copy["body"]))


__all__ = [
    "STATUS_COPY",
    "ERROR_COPY",
    "render_answer_fragment",
    "render_error_fragment",
    "source_url_for",
    "source_pdf_page_url",
    "original_source_url",
]
