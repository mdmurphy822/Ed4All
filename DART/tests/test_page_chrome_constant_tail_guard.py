"""Regression: a CONSTANT version-string footer must not be mistaken for a
printed page number (the OpenStax page-stamping defect).

Reproduction context
--------------------

The OpenStax "sample-algebra-2e" PDF carries a constant footer URL on
most pages::

    This OpenStax book is available for free at http://cnx.org/content/col31130/1.4

Two coupled bugs in ``DART/converter/page_chrome.py`` conspired to stamp
EVERY block on those 77 physical pages with ``data-dart-pages="4"``:

1. **Tokenization** — the old ``_TRAILING_DIGITS_RE`` allowed the trailing
   page-number digits to be glued to a non-space char, so the ``4`` in the
   version tail ``col31130/1.4`` was read as page 4.
2. **Missing invariant** — a real running page number VARIES across the
   pages it appears on. The constant footer mapped 77 pages -> the single
   label "4", which the segmenter then stamped onto 1124 blocks.

Both guards live in ``detect_page_chrome``. This test exercises them at the
unit level (``detect_page_chrome`` directly) and end-to-end through
``extract_document`` + ``segment_extracted_document`` on a programmatically
built multi-page PDF (the leaf-fallback / pdftotext path).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DART.converter.page_chrome import detect_page_chrome  # noqa: E402

_FORM_FEED = "\x0c"

# The exact constant footer that triggered the production defect — a fixed
# URL whose tail ``col31130/1.4`` ends in a version digit glued to ``1.``.
_OPENSTAX_FOOTER = (
    "This OpenStax book is available for free at "
    "http://cnx.org/content/col31130/1.4"
)


# ---------------------------------------------------------------------------
# (a) UNIT — detect_page_chrome
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestConstantTailNotPagination:
    def test_constant_version_footer_yields_no_page_numbers(self) -> None:
        """A constant version-string footer ending in a version digit must
        NOT populate ``page_number_lines`` — the digit never varies, so it
        is fixed chrome text, not pagination."""
        pages = [
            f"Distinct body paragraph for page {i} with words alpha{i} "
            f"beta{i} gamma{i} delta{i} epsilon{i}.\n\n{_OPENSTAX_FOOTER}"
            for i in range(1, 9)
        ]
        raw = _FORM_FEED.join(pages)

        chrome = detect_page_chrome(raw)

        # The footer IS still detected + stripped as chrome (it repeats)...
        assert any(
            "col31130/1.4" in f for f in chrome.footers
        ), "constant footer should still be recognised as chrome"
        # ...but it must NOT hijack page_number_lines to a single label.
        assert chrome.page_number_lines == {}, (
            "constant version-string tail must not be mistaken for "
            f"pagination, got {chrome.page_number_lines!r}"
        )
        # Belt-and-braces: no entry resolves to the spurious "4".
        from DART.converter.page_chrome import (
            _normalise,
            _strip_trailing_digits,
        )

        resolved = {
            _strip_trailing_digits(_normalise(line))[1]
            for line in chrome.page_number_lines.values()
        }
        assert 4 not in resolved

    def test_varying_footer_still_populates_page_numbers(self) -> None:
        """A footer whose trailing number genuinely VARIES per page is real
        pagination and must still populate page_number_lines with the
        varying numbers."""
        pages = [
            f"Distinct body paragraph for page {i} with unique words "
            f"alpha{i} beta{i} gamma{i}.\n\nSome Book {10 + i}"
            for i in range(1, 9)
        ]
        raw = _FORM_FEED.join(pages)

        chrome = detect_page_chrome(raw)

        assert chrome.page_number_lines, "varying footer should be pagination"
        # The recorded raw lines carry the varying page numbers 11..18.
        from DART.converter.page_chrome import (
            _normalise,
            _strip_trailing_digits,
        )

        numbers = sorted(
            _strip_trailing_digits(_normalise(line))[1]
            for line in chrome.page_number_lines.values()
        )
        assert numbers == list(range(11, 19))


# ---------------------------------------------------------------------------
# (b) END-TO-END — extract_document + segment_extracted_document
# ---------------------------------------------------------------------------


def _build_multipage_pdf(page_bodies: list[list[str]]) -> bytes:
    """Build a multi-page PDF using the repo's pipeline fixture builder.

    Reuses ``tests/fixtures/pipeline/build_fixture_pdf.build_pdf`` (the
    pure-bytes PDF writer used everywhere in this repo — no reportlab /
    PyMuPDF dependency). Importorskip so the end-to-end test self-skips
    rather than failing if the builder ever moves.
    """
    fixtures_dir = PROJECT_ROOT / "tests" / "fixtures" / "pipeline"
    if str(fixtures_dir) not in sys.path:
        sys.path.insert(0, str(fixtures_dir))
    build_fixture_pdf = pytest.importorskip("build_fixture_pdf")
    return build_fixture_pdf.build_pdf(page_bodies)


@pytest.mark.integration
@pytest.mark.dart
def test_end_to_end_constant_footer_does_not_collapse_pages() -> None:
    """Full leaf-fallback path: distinct blocks on distinct physical pages
    get DISTINCT ``data-dart-pages`` stamps (via ``raw.page``) because the
    constant version-string footer no longer hijacks ``page_label`` to a
    single value."""
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext binary not available")

    from DART.converter.block_segmenter import segment_extracted_document
    from DART.converter.extractor import extract_document

    # 6 pages: each carries DISTINCT body text (digit-free so the body is
    # not itself mistaken for pagination) AND the SAME constant footer.
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    page_bodies = []
    for w in words:
        page_bodies.append(
            [
                f"Section about {w} topics",
                "",
                f"Distinct body paragraph discussing {w} concepts with "
                f"words {w}one {w}two {w}three covered here.",
                f"Second distinct sentence on the {w} section covering "
                f"{w}four {w}five {w}six in detail.",
                "",
                _OPENSTAX_FOOTER,
            ]
        )

    pdf_bytes = _build_multipage_pdf(page_bodies)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
        fh.write(pdf_bytes)
        pdf_path = fh.name
    try:
        doc = extract_document(pdf_path)
        assert _FORM_FEED in doc.raw_text, "fixture should be multi-page"

        # The constant footer was stripped as chrome but contributed no
        # page label.
        assert doc.page_chrome.page_number_lines == {}, (
            "constant footer must not produce a page-number mapping, got "
            f"{doc.page_chrome.page_number_lines!r}"
        )

        blocks = segment_extracted_document(doc)
        assert blocks, "expected segmented blocks"

        # No block carries a (spurious) page_label — the constant footer no
        # longer hijacks it.
        labels = {(b.extra or {}).get("page_label") for b in blocks}
        assert labels == {None}, f"unexpected page labels: {labels!r}"

        # Page stamps come from physical raw.page and are NOT all identical.
        stamped = [b.page for b in blocks if b.page is not None]
        assert stamped, "blocks should carry physical page numbers"
        distinct = set(stamped)
        assert len(distinct) > 1, (
            "blocks collapsed to a single page — the page-stamping defect "
            f"regressed: {distinct!r}"
        )
        # Distinct stamped pages track the physical page count (6 pages).
        assert len(distinct) == len(page_bodies)
    finally:
        Path(pdf_path).unlink(missing_ok=True)
