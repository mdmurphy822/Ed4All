"""Wave 25 Fix 2: FOOTNOTE classifier guard against page-chrome superstrings.

Leading-digit running footers ("{N} <author-name>") trip the
footnote-marker regex because the leading integer looks like a footnote
reference. The chrome detector (Wave 25 Fix 1) normally strips these
upstream, but the classifier carries a belt-and-braces guard that
refuses FOOTNOTE promotion for blocks whose text is a substring of the
page's detected chrome line.

These tests exercise the guard in isolation (no extractor / pdftotext
required) — page_chrome is a synthetic stand-in.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from DART.converter.block_roles import BlockRole, RawBlock
from DART.converter.heuristic_classifier import HeuristicClassifier


def _make_block(text: str, *, page: int | None = 4) -> RawBlock:
    return RawBlock(
        text=text,
        block_id="blk-test",
        page=page,
        extractor="pdftotext",
    )


@pytest.mark.unit
@pytest.mark.dart
class TestFootnoteGuardUsingPageNumberLines:
    def test_block_matching_page_chrome_superstring_not_footnote(self):
        # Synthetic page-chrome record: page 4's running footer is
        # "4 J. Smith" (the leading-digit author-name pattern).
        chrome = SimpleNamespace(
            headers=set(),
            footers={"j. smith"},
            page_number_lines={4: "4 J. Smith"},
        )
        clf = HeuristicClassifier(page_chrome=chrome)
        block = _make_block("4 J. Smith", page=4)
        result = clf.classify_sync([block])[0]
        # Guard fires: NOT FOOTNOTE.
        assert result.role != BlockRole.FOOTNOTE

    def test_real_footnote_still_classified_footnote(self):
        # Chrome record exists but the block text does not match.
        chrome = SimpleNamespace(
            headers=set(),
            footers={"j. smith"},
            page_number_lines={4: "4 J. Smith"},
        )
        clf = HeuristicClassifier(page_chrome=chrome)
        block = _make_block(
            "1 See the discussion in Jones et al. 2020 for further reading.",
            page=4,
        )
        result = clf.classify_sync([block])[0]
        assert result.role == BlockRole.FOOTNOTE

    def test_no_page_chrome_legacy_behavior_preserved(self):
        # Absent page_chrome → the guard is a no-op. The phantom-footnote
        # would still emerge on a leading-digit author-name residue,
        # proving the guard only activates with chrome context.
        clf = HeuristicClassifier()
        block = _make_block("4 J. Smith", page=4)
        result = clf.classify_sync([block])[0]
        # Legacy (pre-Wave-25) path: the leading-digit-looking FOOTNOTE
        # match succeeds.
        assert result.role == BlockRole.FOOTNOTE

    def test_block_with_no_page_skips_guard(self):
        # block.page=None → guard is inapplicable (can't look up the
        # page's chrome line). Classifier falls through to its normal
        # classification rules, so a footnote-looking text still
        # classifies as FOOTNOTE.
        chrome = SimpleNamespace(
            headers=set(),
            footers={"j. smith"},
            page_number_lines={4: "4 J. Smith"},
        )
        clf = HeuristicClassifier(page_chrome=chrome)
        block = _make_block("4 J. Smith", page=None)
        result = clf.classify_sync([block])[0]
        # Without a page, the guard can't compare against a chrome line
        # for that page — legacy FOOTNOTE path wins.
        assert result.role == BlockRole.FOOTNOTE


@pytest.mark.unit
@pytest.mark.dart
class TestFootnoteGuardUsingHeaderFooterSets:
    def test_block_matching_footer_set_without_page_mapping_guarded(self):
        # Page has no page_number_lines entry (chrome detector didn't
        # record a page for this instance), but the footer set carries
        # the normalised chrome form. The guard still fires via the
        # headers/footers fallback path.
        chrome = SimpleNamespace(
            headers=set(),
            footers={"j. smith"},
            page_number_lines={},
        )
        clf = HeuristicClassifier(page_chrome=chrome)
        block = _make_block("7 J. Smith", page=7)
        result = clf.classify_sync([block])[0]
        assert result.role != BlockRole.FOOTNOTE
