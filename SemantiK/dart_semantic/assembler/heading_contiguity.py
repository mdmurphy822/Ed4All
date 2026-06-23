"""Whole-document heading-contiguity normalization (post-assembly).

Stage 9's Sub-task 2 (``pass_9a``) normalizes the heading tree only over
regions whose ``Region.kind == "heading"``. That is correct for the
STRUCTURAL headings the council/structure-graph identified — but it does
NOT see ``<hN>`` tags that a Stage-6 specialist (e.g. the hosted 70B
prose seat) emitted *inside* a non-heading region's generated HTML body.
Those embedded headings are concatenated into the document verbatim and
bypass normalization, so a 70B fragment like
``<h2>…</h2><h6>EXAMPLE 1.6</h6>`` produces an ``h2 → h6`` level skip
that the Stage-10 ``heading_tree`` document gate rejects
(``wcag_status = "failed"``) — even though every per-region fragment was
individually well-formed.

This module closes that gap with a DETERMINISTIC, document-order pass
over the FINAL assembled HTML string. It re-levels EVERY ``<hN>`` tag in
the body (structural + embedded + gap-filled) so the emitted hierarchy is
contiguous (promote-the-first / demote-forward / never-skip — the same
rules as :func:`heading_tree.normalize_heading_levels`), judging in
DOCUMENT CONTEXT rather than per-fragment. It is purely an attribute /
tag-level rewrite: heading TEXT, ids, and all other attributes are
preserved byte-for-byte; only the ``hN`` level digit (open + matching
close) changes when a skip would otherwise occur.

Idempotent: a document whose headings are already contiguous is returned
byte-identical (the normalized level sequence equals the input, so no
tag is rewritten).
"""
from __future__ import annotations

import re

from .heading_tree import normalize_heading_levels

# Tag-name only — we keep the original attributes verbatim and only swap
# the level digit. Group 1 = level digit, group 2 = attrs blob.
_OPEN_RE = re.compile(r"<h([1-6])\b([^>]*)>", re.IGNORECASE)
_CLOSE_RE = re.compile(r"</h([1-6])\s*>", re.IGNORECASE)


def normalize_document_heading_levels(html: str) -> str:
    """Re-level every ``<hN>`` in ``html`` to a contiguous hierarchy.

    Walks the document in source order, collecting the open/close tag
    spans of every heading, computes the never-skip normalized level
    sequence with :func:`heading_tree.normalize_heading_levels`, and
    rewrites only the level digit of each open tag and its matching
    close tag. All other bytes (attributes, text, ids,
    ``aria-level``) are preserved.

    Returns the input unchanged when there are no headings or when the
    sequence is already contiguous (idempotent).
    """
    if not html:
        return html

    # Collect (open_match) in document order. We pair each open tag with
    # the FIRST following close tag of the SAME old level — the same
    # pairing rule ``pass_9a._rewrite_heading_level`` uses per-fragment.
    opens = list(_OPEN_RE.finditer(html))
    if not opens:
        return html

    raw_levels = [int(m.group(1)) for m in opens]
    new_levels = normalize_heading_levels(raw_levels)
    if new_levels == raw_levels:
        return html  # already contiguous — byte-identical

    # Rebuild the string left-to-right. For each heading we rewrite its
    # open tag, then find + rewrite the matching close tag that occurs
    # before the NEXT heading's open tag (or EOF for the last heading).
    out: list[str] = []
    cursor = 0
    for i, open_m in enumerate(opens):
        old_lvl = open_m.group(1)
        attrs = open_m.group(2) or ""
        new_lvl = new_levels[i]

        # Emit everything up to this open tag, then the rewritten open.
        out.append(html[cursor : open_m.start()])
        out.append(f"<h{new_lvl}{attrs}>")
        cursor = open_m.end()

        if new_lvl == int(old_lvl):
            # Level unchanged — leave the close tag untouched (the next
            # iteration / final flush emits it verbatim).
            continue

        # Search for the matching close tag, bounded by the next heading's
        # open tag so we never rewrite a close that belongs to a later
        # heading (defensive against malformed / unclosed fragments).
        next_open = opens[i + 1].start() if i + 1 < len(opens) else len(html)
        close_pat = re.compile(rf"</h{old_lvl}\s*>", re.IGNORECASE)
        close_m = close_pat.search(html, cursor, next_open)
        if close_m is None:
            # No matching close before the next heading — leave the body
            # as-is (the open-tag level still improves contiguity; an
            # unclosed-tag defect is the html5 gate's job, not ours).
            continue
        out.append(html[cursor : close_m.start()])
        out.append(f"</h{new_lvl}>")
        cursor = close_m.end()

    out.append(html[cursor:])
    return "".join(out)


__all__ = ["normalize_document_heading_levels"]
