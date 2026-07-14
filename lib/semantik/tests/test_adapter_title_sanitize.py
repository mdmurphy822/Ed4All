"""SEMANTIK_TITLE_SANITIZE must reach the accessible-HTML <title>/<h1>/<h2>.

Live regression (2026-07-14, ch01 vendor-gold comparison): the run recipe set
``SEMANTIK_TITLE_SANITIZE=true`` and the sanitizer itself worked
(``'Chapter 1 Foundations 55 ✓ Solution'`` -> ``'Chapter 1 Foundations'``), yet the
emitted HTML still shipped::

    <title>Chapter 1 Foundations 55 ✓ Solution</title>

because ``sanitize_running_header_title`` was only wired into the extractor
(``textbook_structure.json``) and NEVER into ``lib/semantik/adapter.py``'s title
path — which called ``_sanitize_heading_text`` (LaTeX-strip only). The flag was
ON and the fused running-header title shipped anyway, and that string becomes the
COURSE NAME downstream.
"""

from __future__ import annotations

import pytest

from lib.semantik import adapter

# The exact string the live ch01 scan produced: running header + page number +
# the first content glyph, all fused into the chapter title by OCR.
FUSED = "Chapter 1 Foundations 55 ✓ Solution"
CLEAN = "Chapter 1 Foundations"


@pytest.fixture
def _sanitize_on(monkeypatch):
    monkeypatch.setenv("SEMANTIK_TITLE_SANITIZE", "1")


@pytest.fixture
def _sanitize_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_TITLE_SANITIZE", raising=False)


def test_document_title_strips_running_header_when_flag_on(_sanitize_on):
    assert adapter._sanitize_document_title(FUSED) == CLEAN


def test_document_title_byte_identical_when_flag_off(_sanitize_off):
    """Flag off must be a strict no-op beyond the pre-existing LaTeX strip."""
    assert adapter._sanitize_document_title(FUSED) == FUSED


def test_clean_title_untouched_either_way(_sanitize_on):
    assert adapter._sanitize_document_title(CLEAN) == CLEAN
    assert adapter._sanitize_document_title("1.4 Multiply and Divide Integers") == (
        "1.4 Multiply and Divide Integers"
    )


def test_latex_strip_still_composes(_sanitize_on):
    """The LaTeX strip must still run BEFORE the running-header strip."""
    assert adapter._sanitize_document_title(r"\textbf{Chapter 1 Foundations} 55") == CLEAN


def test_empty_and_none_are_safe(_sanitize_on):
    assert adapter._sanitize_document_title(None) == ""
    assert adapter._sanitize_document_title("") == ""


def test_sanitizer_failure_never_breaks_render(monkeypatch, _sanitize_on):
    """A raising sanitizer degrades to the unsanitized title, never a crash."""

    def _boom(_text):
        raise RuntimeError("sanitizer exploded")

    monkeypatch.setattr(adapter, "_sanitize_running_header_title", _boom)
    assert adapter._sanitize_document_title(FUSED) == FUSED
