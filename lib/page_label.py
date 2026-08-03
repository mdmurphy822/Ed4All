#!/usr/bin/env python3
"""Shared page-label formatters for source-page citations (single source of truth).

Three surfaces cite a source page now: the grounded-answer provenance disclosure
(``gui/services/answer_render.py``), the Learning-Objectives-Map deep links
(``Courseforge/scripts/rendering/render_learning_objectives_page.py``), and the "View in
textbook" block links (``gui/services/imscc_service.py``). They live in three
packages (a GUI service, a GUI service, and a Courseforge script), so the label
formatter is hoisted HERE — a dependency-free ``lib/`` module all three can
import — rather than letting each surface grow its own drifting formatter (the
OQ-2 decision in ``plans/finegrain/page-number-deeplinks-2026-06.md``).

RISK-A (physical page != printed page). ``data-semantik-pages`` carries the
fallback chain printed-label -> physical-PDF-page; on a book with front-matter
offset the physical page is wrong-by-offset relative to the book's printed
"p. 47". The attribute does NOT record which signal produced it, so the
consuming surface cannot tell a printed label from a physical fallback. We
therefore NEVER render a bare "p. N" that implies a *printed* page on the new
surfaces. ``pdf_page_citation`` labels honestly as **"PDF p. N"** — matching the
existing answer-render "PDF page N" convention (answer_render.py:457) — so a
reader knows it is the PDF/physical page, not a claimed printed page. Once the
printed-label hit-rate is high enough, a ``method`` discriminator can switch this
to a confident "p. N (printed)".
"""

from __future__ import annotations

from typing import List, Optional

#: The asserted page-kind values (sibling ``data-semantik-page-kind`` attribute,
#: emitted on the SAME element as ``data-semantik-pages``). ``printed`` /
#: ``interpolated`` are genuine BOOK page labels; ``physical`` is the PDF/page
#: index (front-matter offset → wrong-by-offset relative to the printed page).
#: An ABSENT / unknown kind is treated as ``physical`` (back-compat: a corpus
#: that carries no kind attribute).
_PRINTED_KINDS = frozenset({"printed", "interpolated"})


def pdf_pages_label(pages: List[int]) -> str:
    """Compact human label for a sorted PDF-page list: "page 12" / "pages 3, 7".

    This is the EXACT pre-existing ``answer_render._pdf_pages_label`` body,
    hoisted verbatim so the grounded-answer provenance disclosure stays
    byte-identical. New surfaces use ``pdf_page_citation`` instead.
    """
    if len(pages) == 1:
        return "page {}".format(pages[0])
    return "pages {}".format(", ".join(str(p) for p in pages))


def pdf_page_citation(pages: List[int]) -> str:
    """Honest short citation for the LO-map + "View in textbook" link surfaces.

    Format: ``"PDF p. 12"`` (single) / ``"PDF pp. 3, 7"`` (multi). The "PDF"
    qualifier is the RISK-A mitigation: ``data-semantik-pages`` may be the
    physical PDF page (front-matter offset) rather than the book's printed page, and the
    attribute doesn't say which, so we never imply a printed "p. N". Returns
    ``""`` for an empty list so the caller omits the label entirely (page-less
    surfaces stay byte-identical to today).
    """
    clean = [int(p) for p in pages if isinstance(p, int)]
    if not clean:
        return ""
    if len(clean) == 1:
        return "PDF p. {}".format(clean[0])
    return "PDF pp. {}".format(", ".join(str(p) for p in clean))


def page_citation(pages: List[int], kind: Optional[str] = None) -> str:
    """Kind-aware short citation for the LO-map + "View in textbook" surfaces.

    SemantiK emits a sibling ``data-semantik-page-kind`` attribute on the SAME
    element as ``data-semantik-pages``, with values
    ``printed`` | ``interpolated`` | ``physical``
    (PINNED CONTRACT). When ``kind`` is one of the PRINTED kinds, the number IS
    the book's printed page, so the honest label is the bare **"p. 47"** /
    **"pp. 3, 4, 5"**. For ``physical`` — or an ABSENT / unknown / empty kind,
    which back-compat treats as ``physical`` (the whole existing corpus carries
    no kind attribute) — the label stays the honest PDF-page fallback
    ``pdf_page_citation`` already mints (**"PDF p. 47"**).

    Anti-fabrication: this NEVER upgrades a ``physical`` page to a bare "p. N" —
    it only relabels based on the kind SemantiK itself asserts. Returns ``""`` for an
    empty list so the caller omits the label entirely (page-less surfaces stay
    byte-identical to today).

    The absent/``physical`` path is BYTE-IDENTICAL to ``pdf_page_citation`` so
    every existing (kind-less) corpus renders exactly as it does today.
    """
    clean = [int(p) for p in pages if isinstance(p, int)]
    if not clean:
        return ""
    normalized_kind = (kind or "").strip().lower()
    if normalized_kind not in _PRINTED_KINDS:
        # physical / absent / unknown → honest PDF-page fallback (unchanged).
        return pdf_page_citation(clean)
    # printed / interpolated → the genuine book printed page: bare "p. N".
    if len(clean) == 1:
        return "p. {}".format(clean[0])
    return "pp. {}".format(", ".join(str(p) for p in clean))


__all__ = ["pdf_pages_label", "pdf_page_citation", "page_citation"]
