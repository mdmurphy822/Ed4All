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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

# A sourceId is ``{prefix}:{doc}#{block}`` where the prefix is the canonical
# ``semantik`` OR the legacy ``dart`` (dual-READ compat: emitters mint
# ``semantik:``; the reader still accepts ``dart:`` for pre-migration corpora).
_SOURCE_ID_PREFIXES: Tuple[str, ...] = ("semantik:", "dart:")


def _strip_source_prefix(source_id: str) -> Optional[str]:
    """Return the ``{doc}#{block}`` remainder after a canonical ``semantik:`` or
    legacy ``dart:`` prefix, or ``None`` when neither prefix is present."""
    for prefix in _SOURCE_ID_PREFIXES:
        if source_id.startswith(prefix):
            return source_id[len(prefix):]
    return None

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
# Two distinct no-answer FAMILIES (refused = low-confidence / not-in-course;
# blocked = grounding / citation gate withheld a candidate answer). The
# "Why no answer?" disclosure (§8) reads OFF this split so a learner sees the
# failure mode in human terms instead of a single flat "no answer".
_REFUSED_STATUSES = frozenset(
    {STATUS_REFUSED_LOW_CONFIDENCE, STATUS_REFUSED_NOT_IN_COURSE}
)
_BLOCKED_STATUSES = frozenset(
    {STATUS_BLOCKED_INVALID_CITATION, STATUS_BLOCKED_CITATION_GATE}
)
_REFUSED_BLOCKED_STATUSES = _REFUSED_STATUSES | _BLOCKED_STATUSES


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

# --------------------------------------------------------------------------- #
# Status pill (§8 — non-color-only, glyph + plain-language label).
#
# Server-rendered MIRROR of the JS ``statusPill`` ``_ANSWER`` map
# (``gui/static/shared/components/pill.js``): the learner answer status label
# is part of the gated single-render fragment, so it is built HERE (Python),
# never re-templated client-side. Each entry is (glyph, plain-language label,
# pill css class). The glyph is ``aria-hidden`` decoration; the text label is
# the a11y truth, so the pill is never color-only (WCAG 1.4.1).
# --------------------------------------------------------------------------- #

_STATUS_PILL: Dict[str, Dict[str, str]] = {
    STATUS_ANSWERED: {"glyph": "●", "label": "Answered", "cls": "done"},
    STATUS_ANSWERED_WITH_WARNINGS: {
        "glyph": "△",
        "label": "Answered with warnings",
        "cls": "warn",
    },
    STATUS_REFUSED_LOW_CONFIDENCE: {
        "glyph": "–",
        "label": "No clear answer",
        "cls": "skipped",
    },
    STATUS_REFUSED_NOT_IN_COURSE: {
        "glyph": "–",
        "label": "Not in this course",
        "cls": "skipped",
    },
    STATUS_BLOCKED_INVALID_CITATION: {
        "glyph": "△",
        "label": "Couldn't verify",
        "cls": "warn",
    },
    STATUS_BLOCKED_CITATION_GATE: {
        "glyph": "△",
        "label": "Couldn't verify",
        "cls": "warn",
    },
}

# "Why no answer?" plain-language disclosure (§8). DISTINGUISHES the two
# failure modes in human terms off the REAL engine status — refused (low
# confidence: the materials don't contain a clear answer) vs blocked (a
# candidate answer existed but didn't pass our grounding check, so it was
# withheld). Presentation wording over data already in the payload — no model
# internals, no reason_code, no secrets. Rendered as a plain ``<p>`` so it
# slots under the answer ``<h2>`` without adding a heading level.
_WHY_NO_ANSWER_REFUSED = (
    "Why no answer? The course materials don't contain a clear answer to "
    "this question."
)
_WHY_NO_ANSWER_BLOCKED = (
    "Why no answer? There's related information, but it didn't pass our "
    "grounding check, so we're not showing an answer."
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


def _status_pill(status: str) -> str:
    """Render the answer-status pill (glyph + plain-language text label).

    Mirrors the JS ``statusPill`` markup (``span.pill`` > ``span.pill-glyph``
    [aria-hidden] + ``span.pill-label``) so the redesign's status vocabulary
    reads consistently across surfaces. Non-color-only (glyph + text). Returns
    ``""`` for a status with no pill mapping (error fragments carry their own
    heading copy and need no pill)."""
    spec = _STATUS_PILL.get(status)
    if not spec:
        return ""
    return (
        '<span class="pill pill-{cls}" data-kind="answer:{status}">'
        '<span class="pill-glyph" aria-hidden="true">{glyph}</span>'
        '<span class="pill-label">{label}</span>'
        "</span>"
    ).format(
        cls=escape(spec["cls"], quote=True),
        status=escape(status, quote=True),
        glyph=escape(spec["glyph"]),
        label=escape(spec["label"]),
    )


def _why_no_answer(status: str) -> str:
    """Render the "Why no answer?" plain-language disclosure for a no-answer
    status, distinguishing refused (low confidence) from blocked (grounding
    gate). Returns ``""`` for answered / unknown statuses."""
    if status in _REFUSED_STATUSES:
        copy = _WHY_NO_ANSWER_REFUSED
    elif status in _BLOCKED_STATUSES:
        copy = _WHY_NO_ANSWER_BLOCKED
    else:
        return ""
    return '<p class="why-no-answer">{}</p>'.format(escape(copy))


def _section(status: str, heading: str, inner: str, *, pill: str = "") -> str:
    """Wrap the inner body in the fixed answer-region skeleton.

    The ``<h2>`` carries ``tabindex="-1"`` so the JS can move focus to it after
    a swap (D6), and the section is ``aria-labelledby`` the heading id. The
    optional ``pill`` (a pre-rendered ``_status_pill`` string) renders inside
    the heading row, before the body — never as a separate heading level.
    """
    return (
        '<section class="answer" data-status="{status}" '
        'aria-labelledby="{hid}">'
        '<div class="answer-head">'
        '<h2 id="{hid}" tabindex="-1">{heading}</h2>'
        "{pill}"
        "</div>"
        "{inner}"
        "</section>"
    ).format(
        status=escape(status, quote=True),
        hid=_ANSWER_HEADING_ID,
        heading=escape(heading),
        pill=pill,
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
    """Compact human label for a sorted PDF-page list: "page 12" / "pages 3, 7".

    Thin delegate to the shared ``lib/page_label.pdf_pages_label`` helper (OQ-2:
    single source of truth across the answer / LO-map / "View in textbook"
    surfaces). Output is byte-identical to the historical inline body.
    """
    from lib.page_label import pdf_pages_label  # noqa: PLC0415

    return pdf_pages_label(pages)


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
    # sourceId shape: {dart|semantik}:{slug}#{block_id}. The slug half names the
    # document.
    file_hint = ""
    _remainder = _strip_source_prefix(source_id)
    if _remainder is not None and "#" in _remainder:
        file_hint = _remainder.split("#", 1)[0]
    return "/api/courses/{slug}/source-pdf?file={file}&page={page}".format(
        slug=quote(str(slug), safe=""),
        file=quote(file_hint, safe=""),
        page=int(page),
    )


def original_source_url(citation: Dict[str, Any], slug: str) -> str:
    """Build the deep link to the original archived source document for a citation.

    ``/api/courses/{slug}/source-doc?doc=<doc>&ref=<block>#semantik-<block>`` from
    the citation's ``source_block`` (``semantik:{doc}#{block}``, legacy
    ``dart:{doc}#{block}`` dual-read). The endpoint serves the sanitized
    accessible HTML with a serve-time-injected ``id="semantik-{block}"``, so the
    ``#semantik-<block>`` fragment lands on the exact cited block. Emit-then-
    resolve: the link is built whenever ``source_block`` parses; the endpoint
    404s with an explanation when the source doc isn't archived
    (provenance-without-artifacts archives) — no per-render existence probe (the
    renderer is a pure dict→str transform that must stay import-light, mirroring
    the source-pdf hop).

    Returns ``""`` when ``source_block`` doesn't parse as
    ``{semantik|dart}:{doc}#{block}``.
    """
    source_id = str(citation.get("source_block") or "")
    _remainder = _strip_source_prefix(source_id)
    if _remainder is None or "#" not in _remainder:
        return ""
    doc, block = _remainder.split("#", 1)
    doc = doc.strip()
    block = block.strip()
    if not doc or not block:
        return ""
    # Thread the first cited PDF page into the source-doc route so the
    # serve-time ``id="page-N"`` anchor resolves (Phase 3). Page-less citations
    # stay byte-identical to today (no ``&page=`` param emitted). Anti-
    # fabrication: only a page that's actually on ``citation.pdf_pages`` is
    # used — never invented.
    pages = [p for p in (citation.get("pdf_pages") or []) if isinstance(p, int)]
    page_param = "&page={}".format(min(pages)) if pages else ""
    return "/api/courses/{slug}/source-doc?doc={doc}&ref={ref}{page}#semantik-{frag}".format(
        slug=quote(str(slug), safe=""),
        doc=quote(doc, safe=""),
        ref=quote(block, safe=""),
        page=page_param,
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

    Order: link → (module tag) → (supports-N span) → (supporting excerpt /
    text quote) → (B4 provenance disclosure). Every dynamic field is escaped.
    ``module_id`` and the supports-N marker are text spans (never color-only
    state).
    """
    page_label = citation.get("page_label") or "Source"
    href = source_url_for(citation, slug)
    # Source-document citations: the learner source endpoint serves the
    # whole original document with no block anchor, so the main link landed
    # at the top — the attribution front matter (user-reported). When the
    # citation carries a source_block and the deep-linkable source-doc URL
    # resolves, make THAT the main link so the click lands on the cited block.
    if include_original_source_links and citation.get("source_block"):
        anchored = original_source_url(citation, slug)
        if anchored:
            href = anchored
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
    # "Supports N statements" — the claim-attribution count (additive). A text
    # span (never color-only); omitted when zero / absent so legacy payloads and
    # attribution-disabled answers render unchanged.
    try:
        support_count = int(citation.get("supported_claim_count") or 0)
    except (TypeError, ValueError):
        support_count = 0
    if support_count > 0:
        noun = "statement" if support_count == 1 else "statements"
        parts.append(
            '<span class="src-support">Supports {n} {noun}</span>'.format(
                n=support_count, noun=noun
            )
        )
    # No "(approximate location)" hedge: the Source link IS the direct hop to
    # the chunk's holder (course page for imscc chunks; the provenance block
    # below adds the original-source deep link for source-document chunks). A non-exact
    # anchor only means the fragment may land at the section rather than the
    # sentence — the page itself is authoritative from the chunk's item_path.
    #
    # Prefer the attribution ``supporting_excerpt`` (the chunk sentence that
    # actually backs an answer claim) over the legacy whole-answer
    # ``text_quote``; fall back to ``text_quote`` when attribution found none.
    quote = citation.get("supporting_excerpt") or citation.get("text_quote")
    if quote:
        parts.append(
            '<blockquote class="src-quote">{}</blockquote>'.format(
                escape(str(quote))
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
        return _section(
            status, heading, "".join(body_parts), pill=_status_pill(status)
        )

    if status in _REFUSED_BLOCKED_STATUSES:
        # Defensive: never render answer text / citations for these statuses.
        # The "Why no answer?" disclosure (§8) leads the body, distinguishing
        # refused (low confidence) from blocked (grounding gate) in human terms.
        copy = STATUS_COPY[status]
        inner = _why_no_answer(status) + _paragraphs(copy["body"])
        return _section(
            status, copy["heading"], inner, pill=_status_pill(status)
        )

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

# Internal-but-test-visible copy constants (the a11y gate + render tests assert
# the "Why no answer?" wording per failure mode).
__all__ += ["_WHY_NO_ANSWER_REFUSED", "_WHY_NO_ANSWER_BLOCKED"]
