"""Wave 15 tests: pipeline-flip behaviour in ``MCP/tools/pipeline_tools``.

The ``_raw_text_to_accessible_html`` helper in
:mod:`MCP.tools.pipeline_tools` used to be a 600+ LOC regex pipeline.
Wave 15 flipped it to delegate to :mod:`DART.converter` by default, with
a ``DART_LEGACY_CONVERTER=true`` safety fallback. These tests pin both
paths so the flip can't regress silently.

New path signals (emit by Waves 12–14, NOT by legacy):

* ``data-dart-block-role="..."`` on every rendered block
* Dublin Core ``<meta name="DC.*">`` tags (Wave 15 head enrichment)
* Document-level schema.org ``<script type="application/ld+json">``

Legacy path signals (kept for backward compat):

* Pre-Wave-15 ``<section id="..." aria-labelledby="...">`` shape on
  every heading section
* No ``data-dart-block-role`` (legacy never emitted it)
"""

from __future__ import annotations

import os

import pytest

from MCP.tools.pipeline_tools import (
    _raw_text_to_accessible_html,
    _raw_text_to_accessible_html_legacy,
)


RAW_SAMPLE = (
    "Chapter 1: Foundations\n\n"
    "This is the introductory paragraph with enough words to carry real "
    "content that survives the segmenter.\n\n"
    "This is a second paragraph that follows naturally.\n\n"
    "[1] Doe, J. (2024). Accessibility by design. ACM Press."
)


# ---------------------------------------------------------------------------
# Flag routing
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestPipelineFlipRouting:
    def test_default_path_uses_new_converter(self, monkeypatch):
        # DART_LEGACY_CONVERTER unset — new path expected.
        monkeypatch.delenv("DART_LEGACY_CONVERTER", raising=False)
        html_out = _raw_text_to_accessible_html(RAW_SAMPLE, "Test Doc")

        # New path signature: per-block role attribute from Wave 12+.
        assert "data-dart-block-role=" in html_out
        # Wave 15 Dublin Core in <head>.
        assert '<meta name="DC.title"' in html_out
        # Wave 15 schema.org JSON-LD in <head>.
        assert '<script type="application/ld+json">' in html_out

    def test_legacy_flag_uses_legacy_path(self, monkeypatch):
        monkeypatch.setenv("DART_LEGACY_CONVERTER", "true")
        # Use an input that produces a real heading in the legacy path,
        # so we can detect its canonical ``aria-labelledby`` shape.
        raw_with_heading = (
            "Book Title\n\n"
            "Chapter 1: Foundations\n\n"
            "This is the introductory paragraph with enough words to carry "
            "real content that survives the segmenter.\n\n"
            "This is a second paragraph that follows naturally."
        )
        html_out = _raw_text_to_accessible_html(raw_with_heading, "Test Doc")

        # Legacy never emits block-role attributes.
        assert "data-dart-block-role=" not in html_out
        # Legacy never emits Dublin Core nor JSON-LD.
        assert '<meta name="DC.title"' not in html_out
        assert 'application/ld+json' not in html_out
        # Legacy *does* emit its canonical <section id=... aria-labelledby=>
        # shape for every heading.
        assert 'aria-labelledby=' in html_out

    def test_legacy_path_exposed_as_explicit_function(self):
        """The legacy function must stay importable for the safety
        fallback to be addressable from anywhere."""
        html_out = _raw_text_to_accessible_html_legacy(RAW_SAMPLE, "Test Doc")
        assert html_out.startswith("<!DOCTYPE html>")
        assert "data-dart-block-role=" not in html_out

    def test_case_insensitive_legacy_flag(self, monkeypatch):
        """The legacy flag should match canonically: only ``true``
        (any case) flips; anything else keeps the new path."""
        monkeypatch.setenv("DART_LEGACY_CONVERTER", "TRUE")
        html_out = _raw_text_to_accessible_html(RAW_SAMPLE, "Test Doc")
        assert "data-dart-block-role=" not in html_out

        monkeypatch.setenv("DART_LEGACY_CONVERTER", "yes")  # not "true"
        html_out = _raw_text_to_accessible_html(RAW_SAMPLE, "Test Doc")
        assert "data-dart-block-role=" in html_out


# ---------------------------------------------------------------------------
# End-to-end sanity
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestPipelineFlipEndToEnd:
    def test_new_path_end_to_end_produces_expected_wrappers(self, monkeypatch):
        """Sanity: a raw pdftotext-shaped input with chapter + paragraphs
        + bibliography entry produces valid HTML with all expected
        ontology-aware wrappers."""
        monkeypatch.delenv("DART_LEGACY_CONVERTER", raising=False)
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

    def test_new_path_accepts_metadata_arg(self, monkeypatch):
        """The refactored signature accepts a metadata dict that flows
        into Dublin Core + schema.org output."""
        monkeypatch.delenv("DART_LEGACY_CONVERTER", raising=False)
        html_out = _raw_text_to_accessible_html(
            RAW_SAMPLE,
            "Test Doc",
            {"authors": "Jane Doe", "document_type": "textbook"},
        )
        assert '<meta name="DC.creator" content="Jane Doe">' in html_out
        assert '"@type": "Book"' in html_out

    def test_new_path_without_metadata_still_works(self, monkeypatch):
        """Calling with the legacy 2-arg shape must keep working."""
        monkeypatch.delenv("DART_LEGACY_CONVERTER", raising=False)
        html_out = _raw_text_to_accessible_html(RAW_SAMPLE, "Test Doc")
        # Default @type kicks in when no document_type supplied.
        assert '"@type": "CreativeWork"' in html_out

    def test_both_paths_produce_valid_html_skeleton(self, monkeypatch):
        """Regardless of flag, both paths emit a valid HTML5 skeleton."""
        for flag_value in ("", "true"):
            if flag_value:
                monkeypatch.setenv("DART_LEGACY_CONVERTER", flag_value)
            else:
                monkeypatch.delenv("DART_LEGACY_CONVERTER", raising=False)
            html_out = _raw_text_to_accessible_html(RAW_SAMPLE, "Test Doc")
            assert html_out.startswith("<!DOCTYPE html>")
            assert "</html>" in html_out
            assert "<main" in html_out
            assert "</main>" in html_out
