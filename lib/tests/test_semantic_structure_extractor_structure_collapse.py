"""WS4 §4 — STRUCTURE_COLLAPSE_SUSPECTED early-warning + structureDiagnostics.

Pins the structure-collapse signature detection in
``SemanticStructureExtractor`` (single chapter carrying an anomalously high
section count — the 141:1 collapse seen on answer-key-contaminated corpora):

- A WARNING fires (and the ``structureDiagnostics.structureCollapseSuspected``
  output key is populated) when the extractor collapses the whole document into
  ONE chapter with > ``_STRUCTURE_COLLAPSE_SECTION_THRESHOLD`` sections.
- It stays silent (and the output key is absent) for a normally-structured
  multi-chapter document.
- The threshold boundary is strict ``>`` (exactly 40 → silent; 41 → warning).

Mirrors the ``caplog`` precedent in
``test_semantic_structure_extractor_heading_fallback.py``.
"""

from __future__ import annotations

import logging

import pytest

from lib.semantic_structure_extractor import SemanticStructureExtractor
from lib.semantic_structure_extractor.semantic_structure_extractor import (
    _STRUCTURE_COLLAPSE_SECTION_THRESHOLD,
)

_LOGGER_NAME = "lib.semantic_structure_extractor.semantic_structure_extractor"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _one_chapter_many_sections_html(n_sections: int) -> str:
    """A single <h2> chapter carrying ``n_sections`` <h3> sections.

    This is the heading-only synthesis shape: one h2 (one chapter) with N h3s
    (N sections), reproducing the single-chapter / many-section collapse
    signature.
    """
    h3s = "\n".join(
        f'    <h3 id="s{i}">Section {i}</h3>\n'
        f"    <p>Section {i} substantive body content.</p>"
        for i in range(n_sections)
    )
    return f"""
<!DOCTYPE html>
<html lang="en">
<head><title>Collapse Doc</title></head>
<body>
  <main>
    <h1>Collapse Doc</h1>
    <h2 id="ch1">The One Big Chapter</h2>
    <p>Chapter intro paragraph.</p>
{h3s}
  </main>
</body>
</html>
"""


_NORMAL_MULTI_CHAPTER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><title>Normal Doc</title></head>
<body>
  <main>
    <h1>Normal Doc</h1>
    <h2 id="ch1">Chapter One</h2>
    <p>Chapter one intro.</p>
    <h3 id="c1s1">Section 1.1</h3>
    <p>Section 1.1 body.</p>
    <h3 id="c1s2">Section 1.2</h3>
    <p>Section 1.2 body.</p>
    <h2 id="ch2">Chapter Two</h2>
    <p>Chapter two intro.</p>
    <h3 id="c2s1">Section 2.1</h3>
    <p>Section 2.1 body.</p>
    <h3 id="c2s2">Section 2.2</h3>
    <p>Section 2.2 body.</p>
    <h2 id="ch3">Chapter Three</h2>
    <p>Chapter three intro.</p>
    <h3 id="c3s1">Section 3.1</h3>
    <p>Section 3.1 body.</p>
  </main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Warning fires / silent
# ---------------------------------------------------------------------------


def test_collapse_warning_fires_on_one_chapter_many_sections(caplog):
    """One chapter with > threshold sections → STRUCTURE_COLLAPSE_SUSPECTED."""
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    extractor = SemanticStructureExtractor()
    result = extractor.extract(
        _one_chapter_many_sections_html(60), source_path="collapse.html"
    )
    # Sanity: the fixture really does collapse into one chapter.
    assert len(result["chapters"]) == 1
    collapse_warnings = [
        rec for rec in caplog.records
        if "STRUCTURE_COLLAPSE_SUSPECTED" in rec.message
    ]
    assert collapse_warnings, (
        "expected STRUCTURE_COLLAPSE_SUSPECTED warning, got: "
        f"{[r.message for r in caplog.records]}"
    )


def test_collapse_warning_silent_for_normal_structure(caplog):
    """Multi-chapter input → no collapse warning."""
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    extractor = SemanticStructureExtractor()
    result = extractor.extract(
        _NORMAL_MULTI_CHAPTER_HTML, source_path="normal.html"
    )
    assert len(result["chapters"]) > 1
    collapse_warnings = [
        rec for rec in caplog.records
        if "STRUCTURE_COLLAPSE_SUSPECTED" in rec.message
    ]
    assert not collapse_warnings, (
        f"collapse warning fired unexpectedly: "
        f"{[r.message for r in collapse_warnings]}"
    )


# ---------------------------------------------------------------------------
# structureDiagnostics output key
# ---------------------------------------------------------------------------


def test_structure_diagnostics_field_on_extract_output():
    """A collapsed input carries structureDiagnostics; a normal one does not."""
    extractor = SemanticStructureExtractor()

    collapsed = extractor.extract(_one_chapter_many_sections_html(60))
    assert "structureDiagnostics" in collapsed
    diag = collapsed["structureDiagnostics"]["structureCollapseSuspected"]
    assert diag["suspected"] is True
    assert diag["chapter_count"] == 1
    assert diag["section_count"] == 60
    assert diag["threshold"] == _STRUCTURE_COLLAPSE_SECTION_THRESHOLD

    normal = extractor.extract(_NORMAL_MULTI_CHAPTER_HTML)
    assert "structureDiagnostics" not in normal


# ---------------------------------------------------------------------------
# Threshold boundary (strict >)
# ---------------------------------------------------------------------------


def test_collapse_threshold_boundary(caplog):
    """Exactly threshold sections → silent; threshold+1 → warning."""
    extractor = SemanticStructureExtractor()
    thr = _STRUCTURE_COLLAPSE_SECTION_THRESHOLD

    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)

    caplog.clear()
    result_at = extractor.extract(_one_chapter_many_sections_html(thr))
    # Boundary fixture must really produce one chapter with exactly `thr`
    # sections, else the boundary assertion is vacuous.
    assert len(result_at["chapters"]) == 1
    assert len(result_at["chapters"][0]["sections"]) == thr
    assert "structureDiagnostics" not in result_at
    assert not [
        r for r in caplog.records if "STRUCTURE_COLLAPSE_SUSPECTED" in r.message
    ]

    caplog.clear()
    result_over = extractor.extract(_one_chapter_many_sections_html(thr + 1))
    assert len(result_over["chapters"]) == 1
    assert len(result_over["chapters"][0]["sections"]) == thr + 1
    assert "structureDiagnostics" in result_over
    assert [
        r for r in caplog.records if "STRUCTURE_COLLAPSE_SUSPECTED" in r.message
    ]
