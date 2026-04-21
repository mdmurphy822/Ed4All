"""Pipeline-flip sanity tests for ``MCP/tools/pipeline_tools``.

Originally landed in Wave 15 to pin both the new and legacy paths so
the flip to :mod:`DART.converter` couldn't regress silently. Wave 28f
removed the legacy ``_raw_text_to_accessible_html_legacy`` function
and the ``DART_LEGACY_CONVERTER`` safety fallback after one release
of grace, so only the new-path sanity assertions remain here.

New path signals (Waves 12-15):

* ``data-dart-block-role="..."`` on every rendered block
* Dublin Core ``<meta name="DC.*">`` tags in ``<head>``
* Document-level schema.org ``<script type="application/ld+json">``
"""

from __future__ import annotations

import pytest

from MCP.tools.pipeline_tools import _raw_text_to_accessible_html


RAW_SAMPLE = (
    "Chapter 1: Foundations\n\n"
    "This is the introductory paragraph with enough words to carry real "
    "content that survives the segmenter.\n\n"
    "This is a second paragraph that follows naturally.\n\n"
    "[1] Doe, J. (2024). Accessibility by design. ACM Press."
)


# ---------------------------------------------------------------------------
# New-path behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestNewConverterPath:
    def test_default_path_uses_new_converter(self):
        html_out = _raw_text_to_accessible_html(RAW_SAMPLE, "Test Doc")

        # Wave 12+ per-block role attribute.
        assert "data-dart-block-role=" in html_out
        # Wave 15 Dublin Core in <head>.
        assert '<meta name="DC.title"' in html_out
        # Wave 15 schema.org JSON-LD in <head>.
        assert '<script type="application/ld+json">' in html_out

    def test_legacy_flag_is_ignored(self, monkeypatch):
        """Wave 28f: the ``DART_LEGACY_CONVERTER`` flag is no longer
        honoured. Setting it MUST have no effect — the new path still
        runs."""
        monkeypatch.setenv("DART_LEGACY_CONVERTER", "true")
        html_out = _raw_text_to_accessible_html(RAW_SAMPLE, "Test Doc")

        # All new-path signals must still be present.
        assert "data-dart-block-role=" in html_out
        assert '<meta name="DC.title"' in html_out
        assert '<script type="application/ld+json">' in html_out


# ---------------------------------------------------------------------------
# End-to-end sanity
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestPipelineEndToEnd:
    def test_new_path_end_to_end_produces_expected_wrappers(self):
        """Sanity: a raw pdftotext-shaped input with chapter + paragraphs
        + bibliography entry produces valid HTML with all expected
        ontology-aware wrappers."""
        html_out = _raw_text_to_accessible_html(RAW_SAMPLE, "Test Doc")

        # Valid document shell.
        assert html_out.startswith("<!DOCTYPE html>")
        assert "<main id=\"main-content\"" in html_out
        # Wave 19: <h1> carries an id attribute — match ``<h1 `` (space).
        assert html_out.count("<h1 ") == 1

        # Wave 13 wrappers survive the flip. Wave 19: the class attribute
        # now precedes the role attribute on the <article> open tag.
        assert "<article " in html_out
        assert 'role="doc-chapter"' in html_out
        # Bibliography entry wrapped in Wave 13 <ol role="doc-bibliography">.
        assert '<ol role="doc-bibliography">' in html_out
        assert 'role="doc-endnote"' in html_out

    def test_new_path_accepts_metadata_arg(self):
        """The refactored signature accepts a metadata dict that flows
        into Dublin Core + schema.org output."""
        html_out = _raw_text_to_accessible_html(
            RAW_SAMPLE,
            "Test Doc",
            {"authors": "Jane Doe", "document_type": "textbook"},
        )
        assert '<meta name="DC.creator" content="Jane Doe">' in html_out
        assert '"@type": "Book"' in html_out

    def test_new_path_without_metadata_still_works(self):
        """Calling with the legacy 2-arg shape must keep working."""
        html_out = _raw_text_to_accessible_html(RAW_SAMPLE, "Test Doc")
        # Default @type kicks in when no document_type supplied.
        assert '"@type": "CreativeWork"' in html_out

    def test_valid_html_skeleton(self):
        """The new path emits a valid HTML5 skeleton."""
        html_out = _raw_text_to_accessible_html(RAW_SAMPLE, "Test Doc")
        assert html_out.startswith("<!DOCTYPE html>")
        assert "</html>" in html_out
        assert "<main" in html_out
        assert "</main>" in html_out
