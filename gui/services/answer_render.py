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


def _citation_li(citation: Dict[str, Any], slug: str) -> str:
    """Render one citation as a focusable ``<li>`` with a "Source:" link.

    Order: link → (module tag) → (approximate-location tag) → (text quote).
    Every dynamic field is escaped. ``module_id`` and the approximate marker
    are text spans (never color-only state).
    """
    page_label = citation.get("page_label") or "Source"
    href = source_url_for(citation, slug)
    parts = [
        '<a href="{href}">Source: {label}</a>'.format(
            href=escape(href, quote=True), label=escape(str(page_label))
        )
    ]
    module_id = citation.get("module_id")
    if module_id:
        parts.append(
            '<span class="src-module">Module {}</span>'.format(
                escape(str(module_id))
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
    return "<li>{}</li>".format("".join(parts))


def _sources_block(citations: List[Dict[str, Any]], slug: str) -> str:
    """Render the ``<h3>Sources</h3><ol>…</ol>`` block, or "" when empty."""
    if not citations:
        return ""
    items = "".join(_citation_li(c, slug) for c in citations)
    return '<h3>Sources</h3><ol class="sources">{}</ol>'.format(items)


# --------------------------------------------------------------------------- #
# Public render entry points.
# --------------------------------------------------------------------------- #


def render_answer_fragment(payload: Dict[str, Any]) -> str:
    """Render one ``GroundedAnswer.to_dict()`` payload to an HTML fragment.

    Dispatches on ``status``. For refused/blocked statuses the contract
    guarantees ``answer_text=None`` and ``citations=[]``; this renderer asserts
    that defensively and renders NO answer text and NO citations regardless of
    what the payload carries, so a contract regression can never leak an
    unverified answer to a learner.

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
        body_parts.append(_sources_block(list(citations), slug))
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
]
