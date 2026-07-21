#!/usr/bin/env python3
"""Tests for the kind-aware page-label formatters in ``lib/page_label.py``.

The crux contract (page-number printed-label accuracy, SURFACING side):

  - kind ``printed`` / ``interpolated``  → bare "p. N" / "pp. N, M" (book page).
  - kind ``physical`` / absent / unknown → "PDF p. N" (honest PDF-page fallback).

BACK-COMPAT / NO-REGRESSION: an absent kind MUST render BYTE-IDENTICAL to the
legacy ``pdf_page_citation`` (the whole existing corpus carries no kind attr).
"""

from __future__ import annotations

from lib.page_label import page_citation, pdf_page_citation, pdf_pages_label


# --------------------------------------------------------------------------- #
# Legacy formatters stay byte-identical (answer_render + existing callers).
# --------------------------------------------------------------------------- #


def test_pdf_pages_label_unchanged():
    """answer_render delegate — must stay byte-identical."""
    assert pdf_pages_label([12]) == "page 12"
    assert pdf_pages_label([3, 7]) == "pages 3, 7"


def test_pdf_page_citation_unchanged():
    """The existing honest PDF-page citation — must stay byte-identical."""
    assert pdf_page_citation([47]) == "PDF p. 47"
    assert pdf_page_citation([3, 4, 5]) == "PDF pp. 3, 4, 5"
    assert pdf_page_citation([]) == ""


# --------------------------------------------------------------------------- #
# Kind-aware citation — printed / interpolated → bare "p. N".
# --------------------------------------------------------------------------- #


def test_page_citation_printed_single():
    assert page_citation([47], "printed") == "p. 47"


def test_page_citation_printed_multi():
    assert page_citation([3, 4, 5], "printed") == "pp. 3, 4, 5"


def test_page_citation_interpolated_is_printed_label():
    """``interpolated`` is a derived BOOK page → bare "p. N", not "PDF p. N"."""
    assert page_citation([12], "interpolated") == "p. 12"
    assert page_citation([12, 13], "interpolated") == "pp. 12, 13"


# --------------------------------------------------------------------------- #
# Kind-aware citation — physical / absent / unknown → "PDF p. N" (back-compat).
# --------------------------------------------------------------------------- #


def test_page_citation_physical_is_pdf_label():
    assert page_citation([47], "physical") == "PDF p. 47"
    assert page_citation([3, 4, 5], "physical") == "PDF pp. 3, 4, 5"


def test_page_citation_absent_kind_byte_identical_to_legacy():
    """An ABSENT kind (None / "" ) → BYTE-IDENTICAL to ``pdf_page_citation``.

    The whole existing corpus carries no kind attr, so this is the no-regression
    contract: every kind-less surface renders exactly as it does today.
    """
    for pages in ([47], [3, 4, 5], [1], [10, 20, 30], []):
        assert page_citation(pages, None) == pdf_page_citation(pages)
        assert page_citation(pages, "") == pdf_page_citation(pages)
        assert page_citation(pages) == pdf_page_citation(pages)


def test_page_citation_unknown_kind_falls_back_to_pdf():
    """An unknown / garbage kind is treated as physical (anti-fabrication)."""
    assert page_citation([47], "weird") == "PDF p. 47"
    assert page_citation([47], "PRINTED-ish") == "PDF p. 47"


def test_page_citation_kind_is_case_insensitive():
    assert page_citation([47], "Printed") == "p. 47"
    assert page_citation([47], "  INTERPOLATED  ") == "p. 47"
    assert page_citation([47], "PHYSICAL") == "PDF p. 47"


def test_page_citation_empty_pages_returns_empty_regardless_of_kind():
    """Page-less surfaces omit the label entirely (no suffix), any kind."""
    assert page_citation([], "printed") == ""
    assert page_citation([], "physical") == ""
    assert page_citation([], None) == ""


def test_page_citation_anti_fabrication_never_upgrades_physical():
    """A physical page is NEVER relabeled as a bare printed "p. N"."""
    # Same numbers, different SemantiK-asserted kinds → different honest labels.
    assert page_citation([47], "physical") == "PDF p. 47"
    assert page_citation([47], "printed") == "p. 47"
