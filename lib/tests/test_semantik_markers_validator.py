"""SemantiKMarkersValidator provenance checks.

SemantiKMarkersValidator enforces the accessibility markers (skip link,
main role, aria-labelledby, semantik-section class) plus warning-level
checks for ``data-semantik-source`` and ``data-semantik-block-id`` on every
``<section>`` element.

The contract for these tests:

* When every ``<section>`` carries both provenance attributes, the
  validator raises no warnings related to provenance.
* When attributes are missing on some ``<section>`` tags, the validator
  emits ``warning``-severity GateIssues — never ``critical``. The gate
  still passes (``passed=True``) as long as the critical markers are present.
* ``score`` is computed from the critical markers only; warning-level
  issues do not reduce the score.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.validators.semantik_markers import SemantiKMarkersValidator  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: minimal HTML that passes the critical markers but varies on
# data-semantik-* attribute coverage.
# ---------------------------------------------------------------------------


_SHELL_TOP = '''<!DOCTYPE html>
<html lang="en">
<head><title>t</title></head>
<body>
<a href="#main" class="skip">Skip</a>
<main id="main" role="main">
<article class="semantik-document">
'''

_SHELL_BOTTOM = '''
</article>
</main>
</body>
</html>
'''


def _fully_attributed_html() -> str:
    section = (
        '<section id="s0" class="semantik-section" aria-labelledby="s0-h" '
        'data-semantik-block-id="s0" data-semantik-source="pdfplumber" '
        'data-semantik-pages="3">'
        '<h2 id="s0-h">Contacts</h2>'
        '</section>'
    )
    return _SHELL_TOP + section + _SHELL_BOTTOM


def _partially_attributed_html() -> str:
    section_a = (
        '<section id="s0" class="semantik-section" aria-labelledby="s0-h" '
        'data-semantik-block-id="s0" data-semantik-source="pdfplumber">'
        '<h2 id="s0-h">Contacts</h2>'
        '</section>'
    )
    # Second section is completely plain — missing both attrs.
    section_b = (
        '<section id="s1" class="semantik-section" aria-labelledby="s1-h">'
        '<h2 id="s1-h">Roster</h2>'
        '</section>'
    )
    return _SHELL_TOP + section_a + section_b + _SHELL_BOTTOM


def _unattributed_html() -> str:
    section = (
        '<section id="s0" class="semantik-section" aria-labelledby="s0-h">'
        '<h2 id="s0-h">Contacts</h2>'
        '</section>'
    )
    return _SHELL_TOP + section + _SHELL_BOTTOM


def _missing_block_id_only_html() -> str:
    section = (
        '<section id="s0" class="semantik-section" aria-labelledby="s0-h" '
        'data-semantik-source="pdfplumber">'
        '<h2 id="s0-h">Contacts</h2>'
        '</section>'
    )
    return _SHELL_TOP + section + _SHELL_BOTTOM


def _no_sections_html() -> str:
    return _SHELL_TOP + '<p>No sections at all.</p>' + _SHELL_BOTTOM


# ---------------------------------------------------------------------------
# Positive: fully-attributed HTML has no provenance warnings
# ---------------------------------------------------------------------------


class TestFullyAttributedHtml:
    def test_passes_with_no_warnings(self):
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": _fully_attributed_html()})
        assert result.passed is True
        prov_issues = [
            i for i in result.issues
            if i.code in ("MISSING_DATA_SEMANTIK_SOURCE", "MISSING_DATA_SEMANTIK_BLOCK_ID")
        ]
        assert prov_issues == []

    def test_score_is_perfect(self):
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": _fully_attributed_html()})
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# Negative: warnings emitted when attributes are missing
# ---------------------------------------------------------------------------


class TestPartiallyAttributedHtml:
    def test_emits_warnings_for_missing_attrs(self):
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": _partially_attributed_html()})
        codes = [i.code for i in result.issues]
        assert "MISSING_DATA_SEMANTIK_SOURCE" in codes
        assert "MISSING_DATA_SEMANTIK_BLOCK_ID" in codes

    def test_provenance_issues_are_warning_severity(self):
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": _partially_attributed_html()})
        for issue in result.issues:
            if issue.code in (
                "MISSING_DATA_SEMANTIK_SOURCE",
                "MISSING_DATA_SEMANTIK_BLOCK_ID",
            ):
                assert issue.severity == "warning"

    def test_warnings_do_not_block_gate(self):
        """passed=True despite warnings — critical markers still present."""
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": _partially_attributed_html()})
        assert result.passed is True

    def test_warning_message_reports_counts(self):
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": _partially_attributed_html()})
        source_issue = next(
            i for i in result.issues if i.code == "MISSING_DATA_SEMANTIK_SOURCE"
        )
        # "1/2 <section> elements missing data-semantik-source"
        assert "1/2" in source_issue.message


class TestFullyUnattributedHtml:
    def test_emits_both_warnings(self):
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": _unattributed_html()})
        codes = {i.code for i in result.issues}
        assert "MISSING_DATA_SEMANTIK_SOURCE" in codes
        assert "MISSING_DATA_SEMANTIK_BLOCK_ID" in codes

    def test_still_passes_gate(self):
        """Still passes — critical markers present; these are warnings."""
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": _unattributed_html()})
        assert result.passed is True


class TestPartialAttributeCoverage:
    def test_block_id_missing_triggers_only_block_id_warning(self):
        validator = SemantiKMarkersValidator()
        result = validator.validate(
            {"html_content": _missing_block_id_only_html()}
        )
        codes = {i.code for i in result.issues}
        assert "MISSING_DATA_SEMANTIK_BLOCK_ID" in codes
        assert "MISSING_DATA_SEMANTIK_SOURCE" not in codes


# ---------------------------------------------------------------------------
# Edge: no sections at all
# ---------------------------------------------------------------------------


class TestNoSections:
    def test_no_provenance_warnings_when_no_sections(self):
        """No <section> elements -> no per-section provenance checks fire."""
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": _no_sections_html()})
        prov_codes = {
            i.code for i in result.issues
            if i.code in ("MISSING_DATA_SEMANTIK_SOURCE", "MISSING_DATA_SEMANTIK_BLOCK_ID")
        }
        assert prov_codes == set()


# ---------------------------------------------------------------------------
# Score contract: warnings do not reduce the score
# ---------------------------------------------------------------------------


class TestScoreContract:
    def test_score_ignores_warning_issues(self):
        """Score is 1.0 when every critical marker is present even with warnings."""
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": _partially_attributed_html()})
        # Warnings present, but score must still reflect only critical markers.
        assert result.score == 1.0
        assert any(i.severity == "warning" for i in result.issues)

    def test_missing_critical_marker_still_blocks_gate(self):
        """Critical severity contract unchanged: missing skip link blocks."""
        html = _fully_attributed_html().replace('class="skip"', 'class="navlink"')
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": html})
        assert result.passed is False
        critical_codes = {
            i.code for i in result.issues if i.severity == "critical"
        }
        assert "MISSING_SKIP_LINK" in critical_codes


# ---------------------------------------------------------------------------
# Attribute detection is tolerant of quote style + order
# ---------------------------------------------------------------------------


class TestAttributeDetection:
    def test_single_quoted_attrs_recognized(self):
        """Single-quoted data-semantik-* attrs are detected."""
        section = (
            "<section id='s0' class='semantik-section' aria-labelledby='s0-h' "
            "data-semantik-block-id='s0' data-semantik-source='pdfplumber'>"
            "<h2 id='s0-h'>T</h2></section>"
        )
        html = _SHELL_TOP + section + _SHELL_BOTTOM
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": html})
        codes = {i.code for i in result.issues}
        assert "MISSING_DATA_SEMANTIK_SOURCE" not in codes
        assert "MISSING_DATA_SEMANTIK_BLOCK_ID" not in codes

    def test_attribute_order_does_not_matter(self):
        """data-semantik-source after other attributes still counts."""
        section = (
            '<section data-semantik-source="pdftotext" id="s0" class="semantik-section" '
            'aria-labelledby="s0-h" data-semantik-block-id="s0">'
            '<h2 id="s0-h">T</h2></section>'
        )
        html = _SHELL_TOP + section + _SHELL_BOTTOM
        validator = SemantiKMarkersValidator()
        result = validator.validate({"html_content": html})
        codes = {i.code for i in result.issues}
        assert "MISSING_DATA_SEMANTIK_SOURCE" not in codes
