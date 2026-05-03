"""HTML + resource helpers used by the Ed4All chunker.

Phase 7a Subtask 3 lifts five small helpers out of
``Trainforge/process_course.py``::CourseProcessor as standalone module
functions so Subtask 4 (the chunker proper) can call them without a
``CourseProcessor`` instance.

Lifted helpers:
    - :func:`extract_plain_text` (was ``CourseProcessor._extract_plain_text``)
    - :func:`strip_assessment_feedback` (was ``CourseProcessor._strip_assessment_feedback``)
    - :func:`strip_feedback_from_text` (was ``CourseProcessor._strip_feedback_from_text``)
    - :func:`extract_section_html` (was ``CourseProcessor._extract_section_html``)
    - :func:`type_from_resource` (was ``CourseProcessor._type_from_resource``)

All five were ``@staticmethod`` on the class — none touched ``self`` —
so the lift is a straight rename + indent shift. The ``CourseProcessor``
``_``-prefixed methods continue to exist as thin wrappers that delegate
here, keeping every existing call site (``self._extract_plain_text(...)``
or ``CourseProcessor._extract_section_html(...)`` from the regression
suite) working without modification.

``HTMLTextExtractor`` is imported lazily inside :func:`extract_plain_text`
to avoid a module-load circular import: ``Trainforge.parsers.html_content_parser``
itself transitively depends on Trainforge code paths that pull in
``Trainforge.process_course`` in the wider import graph; lazy import
defers the resolution to call time.
"""

from __future__ import annotations

import re
from typing import List, Tuple

__all__ = [
    "extract_plain_text",
    "extract_section_html",
    "strip_assessment_feedback",
    "strip_feedback_from_text",
    "type_from_resource",
]


# ---------------------------------------------------------------------------
# Resource → chunk-type mapping
# ---------------------------------------------------------------------------

_RESOURCE_TYPE_TO_CHUNK_TYPE = {
    "quiz": "assessment_item",
    "overview": "overview",
    "summary": "summary",
    "discussion": "exercise",
    "application": "exercise",
}


def type_from_resource(resource_type: str) -> str:
    """Map an IMSCC ``resource_type`` to a canonical chunk type.

    Default for unmapped types is ``"explanation"`` (legacy parity
    with ``CourseProcessor._type_from_resource``).
    """

    return _RESOURCE_TYPE_TO_CHUNK_TYPE.get(resource_type, "explanation")


# ---------------------------------------------------------------------------
# Plain-text extraction
# ---------------------------------------------------------------------------


def extract_plain_text(html: str) -> str:
    """Return the plain-text projection of ``html``.

    Wraps :class:`Trainforge.parsers.html_content_parser.HTMLTextExtractor`,
    which already implements the Worker-Q template-chrome skip and the
    canonical ``<script>`` / ``<style>`` subtree drop. The lazy import
    avoids a module-load circular-import risk (the parser module pulls
    in ``lib.ontology.bloom`` and other Trainforge-adjacent modules).
    """

    # Local import keeps ``ed4all_chunker.helpers`` import-cycle-free.
    from Trainforge.parsers.html_content_parser import HTMLTextExtractor

    extractor = HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


# ---------------------------------------------------------------------------
# Assessment-feedback stripping (HTML + plain text)
# ---------------------------------------------------------------------------


def strip_assessment_feedback(html: str) -> str:
    """Remove answer feedback from quiz/self-check HTML before text extraction.

    Courseforge quizzes embed correct/incorrect feedback in
    ``<div class="sc-feedback">`` blocks and ``data-correct`` attributes
    on labels. This strips that content so assessment chunks contain
    only question stems and answer options without revealing
    correctness.
    """

    # Remove feedback divs (Courseforge self-check pattern)
    cleaned = re.sub(
        r'<div\s+class="sc-feedback"[^>]*>.*?</div>',
        '', html, flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove data-correct attributes from labels
    cleaned = re.sub(
        r'\s+data-correct="[^"]*"',
        '', cleaned,
    )
    return cleaned


def strip_feedback_from_text(text: str) -> str:
    """Remove residual feedback markers from plain text extraction.

    Handles both line-level and inline feedback patterns since the text
    extractor often concatenates feedback inline with answer options.
    """

    # Remove inline feedback: "Correct. <explanation>" or
    # "Incorrect. <explanation>". These appear after answer option text,
    # running to the next answer option or end.
    text = re.sub(
        r'\s*(?:Correct|Incorrect)\.\s+[^.]*(?:\.[^A-Z])*\.?',
        '', text,
    )
    # Also remove standalone lines
    lines = text.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Correct.") or stripped.startswith("Incorrect."):
            continue
        filtered.append(line)
    return '\n'.join(filtered)


# ---------------------------------------------------------------------------
# Section-aware HTML slicing
# ---------------------------------------------------------------------------


def extract_section_html(html: str, heading: str) -> str:
    """Return the HTML fragment for ``heading``, respecting section boundaries.

    Wave 83 (Phase B) replaces the legacy heading-to-heading regex slice
    with a section-boundary-aware extractor. The legacy slice ran from
    ``<hN>{heading}`` to the next ``<h[1-6]`` and ignored
    ``<section>...</section>`` wrappers entirely. When two adjacent
    headings lived in different ``<section>`` blocks (the canonical
    Courseforge layout), the slice would clip a closing ``</section>``
    from one fragment and an opening ``<section>`` from the next —
    leaving every chunk's HTML unbalanced. This was the load-bearing
    cause of the rdf-shacl-551 audit's 203/295 unbalanced-section
    chunks.

    New behavior:
      - If the heading lives **inside** a ``<section>``, return the
        full enclosing section (open tag → matching close tag).
        Balanced HTML guaranteed.
      - If the heading lives **outside** any ``<section>`` (e.g. a
        page-title ``<h1>``), return just the heading element itself.
        No spurious section tags.
      - If the heading isn't found, return ``""`` (legacy behavior).

    Implementation: regex-locate the heading, regex-locate every
    ``<section>``/``</section>`` event, walk events to determine
    enclosure depth at the heading's position, slice the input string
    accordingly. No new dependencies. The output is always a
    substring of the input HTML, so any well-formedness in the source
    is preserved verbatim.
    """

    if not heading or not html:
        return ""

    # Step 1: locate the heading element. Match tolerantly across
    # whitespace and inline content within the heading body.
    heading_re = re.compile(
        r"<h([1-6])\b[^>]*>\s*" + re.escape(heading) + r"\s*</h\1>",
        re.DOTALL | re.IGNORECASE,
    )
    h_match = heading_re.search(html)
    if not h_match:
        return ""
    h_start, h_end = h_match.span()

    # Step 2: collect every <section> open/close event with its byte span.
    # ``</section\s*>`` allows whitespace before the closing bracket but
    # nothing else (per HTML5).
    section_event_re = re.compile(
        r"<section\b[^>]*>|</section\s*>", re.IGNORECASE
    )

    # Step 3: walk events to determine section enclosure at h_start.
    # Events strictly before the heading update an open-section stack;
    # events strictly after are saved for the close-search step.
    open_stack: List[int] = []  # stack of section-open START offsets
    events_after_heading: List[Tuple[int, int, bool]] = []  # (start, end, is_open)
    for m in section_event_re.finditer(html):
        is_open = m.group(0).lower().startswith("<section")
        s, e = m.span()
        if e <= h_start:
            if is_open:
                open_stack.append(s)
            elif open_stack:
                open_stack.pop()
            # Mismatched </section> with empty stack: ignore (defensive).
        elif s >= h_end:
            events_after_heading.append((s, e, is_open))
        # Events overlapping the heading (e.g. heading nested inside a
        # section's start tag) are exotic and ignored.

    if not open_stack:
        # Heading is outside any section → return just the heading element.
        return html[h_start:h_end]

    # Step 4: the innermost enclosing section starts at open_stack[-1].
    # Walk forward through events_after_heading tracking depth (we're
    # already 1-deep inside the enclosing section). Stop when we hit
    # the matching close.
    section_start = open_stack[-1]
    depth = 1
    for s, e, is_open in events_after_heading:
        if is_open:
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return html[section_start:e]

    # Source HTML is unbalanced (section never closes). Best-effort:
    # return from section_start to end. Caller's HTML balance check
    # will flag this as a real source defect, distinct from the
    # legacy clipping bug.
    return html[section_start:]
