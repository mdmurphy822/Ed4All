"""Wave 31 — WCAGValidator semantic-check tests.

Pre-Wave-31 WCAGValidator was tag-presence only: it would pass a document
that had 279 figures with empty alt + 12 empty objectives lists + 90 dead
TOC anchors, while simultaneously firing 3 false-positive HIGH findings
on legitimate modern patterns (`:focus:not(:focus-visible)`, dedup
`role="main"`, 1px visually-hidden skip-link).

Wave 31 adds real semantic checks per the 5-persona audit:

* SC 1.1.1 — figure + figcaption + alt="" = critical
* SC 1.3.1 — empty <ul>/<ol> + empty doc-chapter = critical
* SC 2.4.1 / 2.4.5 — dead TOC anchors = critical
* SC 2.4.6 — PDF-extraction artifacts in headings = warning

And fixes 3 false positives:

* Focus indicator: `:focus:not(:focus-visible){outline:none}` paired
  with `:focus-visible{outline:...}` is a valid modern pattern.
* Landmarks: single `<main role="main">` = 1 landmark, not 2.
* Target size: 1px visually-hidden skip-link exempt per SC 2.5.8.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DART.pdf_converter.wcag_validator import (  # noqa: E402
    IssueSeverity,
    WCAGValidator,
)


def _wrap(body: str, extra_head: str = "") -> str:
    return (
        f'<!DOCTYPE html><html lang="en"><head>'
        f'<title>Test Page</title>{extra_head}</head>'
        f'<body><main><h1>Test</h1>{body}</main></body></html>'
    )


# ---------------------------------------------------------------------- #
# SC 1.1.1 — figure + figcaption + alt=""
# ---------------------------------------------------------------------- #


class TestImageAltTextSemantic:
    def test_10_informational_figures_empty_alt_critical(self):
        body = ""
        for i in range(10):
            body += (
                f'<figure><img src="img{i}.png" alt="">'
                f'<figcaption>Caption for figure {i}</figcaption></figure>'
            )
        result = WCAGValidator().validate(_wrap(body))
        critical = [i for i in result.issues if i.severity == IssueSeverity.CRITICAL and i.criterion == "1.1.1"]
        assert len(critical) > 0, "Informational figures with empty alt must fail critical"
        assert not result.wcag_aa_compliant

    def test_decorative_figures_pass(self):
        # No figcaption + empty alt = decorative = pass.
        body = ''.join(
            f'<figure role="presentation"><img src="img{i}.png" alt=""></figure>'
            for i in range(5)
        )
        result = WCAGValidator().validate(_wrap(body))
        crit_111 = [i for i in result.issues if i.severity == IssueSeverity.CRITICAL and i.criterion == "1.1.1"]
        assert len(crit_111) == 0

    def test_alt_populated_passes(self):
        body = '<figure><img src="chart.png" alt="Bar chart showing growth">'
        body += '<figcaption>Quarterly growth chart</figcaption></figure>'
        result = WCAGValidator().validate(_wrap(body))
        crit_111 = [i for i in result.issues if i.severity == IssueSeverity.CRITICAL and i.criterion == "1.1.1"]
        assert len(crit_111) == 0


# ---------------------------------------------------------------------- #
# SC 1.3.1 — empty lists / doc-chapters
# ---------------------------------------------------------------------- #


class TestEmptyLists:
    def test_empty_ul_critical(self):
        body = '<h2>Objectives</h2><ul></ul>'
        result = WCAGValidator().validate(_wrap(body))
        crit = [i for i in result.issues
                if i.severity == IssueSeverity.CRITICAL and i.criterion == "1.3.1"]
        assert len(crit) >= 1
        assert not result.wcag_aa_compliant

    def test_populated_list_passes(self):
        body = '<ul><li>A</li><li>B</li><li>C</li></ul>'
        result = WCAGValidator().validate(_wrap(body))
        crit = [i for i in result.issues
                if i.severity == IssueSeverity.CRITICAL and i.criterion == "1.3.1"
                and 'empty' in i.message.lower()]
        assert len(crit) == 0


class TestEmptyDocChapters:
    def test_empty_doc_chapter_critical(self):
        body = '<article role="doc-chapter"><h2>Chapter 1</h2></article>'
        result = WCAGValidator().validate(_wrap(body))
        crit = [i for i in result.issues
                if i.severity == IssueSeverity.CRITICAL and i.criterion == "1.3.1"
                and 'doc-chapter' in (i.element or '')]
        assert len(crit) >= 1

    def test_populated_doc_chapter_passes(self):
        body = (
            '<article role="doc-chapter"><h2>Chapter 1</h2>'
            '<p>This is a full paragraph of body content. It goes into '
            'substantial detail about the topic, well over the twenty-'
            'word threshold that marks a real chapter body.</p></article>'
        )
        result = WCAGValidator().validate(_wrap(body))
        crit = [i for i in result.issues
                if i.severity == IssueSeverity.CRITICAL and i.criterion == "1.3.1"
                and 'doc-chapter' in (i.element or '')]
        assert len(crit) == 0


# ---------------------------------------------------------------------- #
# SC 2.4.1 / 2.4.5 — TOC anchor resolution
# ---------------------------------------------------------------------- #


class TestTOCAnchors:
    def test_half_dead_anchors_critical(self):
        # 10 TOC links, only 5 resolve.
        toc_items = "".join(f'<li><a href="#sec-{i}">Section {i}</a></li>' for i in range(10))
        targets = "".join(f'<section id="sec-{i}"><h2>S{i}</h2><p>content</p></section>' for i in range(5))
        body = f'<nav role="doc-toc"><ol>{toc_items}</ol></nav>{targets}'
        result = WCAGValidator().validate(_wrap(body))
        crit = [i for i in result.issues
                if i.severity == IssueSeverity.CRITICAL
                and i.criterion in ("2.4.1", "2.4.5")]
        assert len(crit) >= 1, "5/10 dead TOC links must fire CRITICAL"
        assert not result.wcag_aa_compliant

    def test_all_anchors_resolve_passes(self):
        toc_items = "".join(f'<li><a href="#sec-{i}">Section {i}</a></li>' for i in range(5))
        targets = "".join(f'<section id="sec-{i}"><h2>S{i}</h2></section>' for i in range(5))
        body = f'<nav role="doc-toc"><ol>{toc_items}</ol></nav>{targets}'
        result = WCAGValidator().validate(_wrap(body))
        # Filter out the legacy "missing skip-link" MEDIUM (separate check
        # from TOC anchor resolution).
        dead_anchor_issues = [
            i for i in result.issues
            if i.criterion in ("2.4.1", "2.4.5")
            and ('dead' in i.message.lower() or 'do not resolve' in i.message.lower())
        ]
        assert len(dead_anchor_issues) == 0


# ---------------------------------------------------------------------- #
# SC 2.4.6 — PDF-extraction artifacts in headings
# ---------------------------------------------------------------------- #


class TestPDFArtifactHeadings:
    def test_chapter_with_trailing_page_number_warn(self):
        body = '<h2>Chapter 3 Photosynthesis 47</h2>'
        result = WCAGValidator().validate(_wrap(body))
        med = [i for i in result.issues
               if i.severity == IssueSeverity.MEDIUM and i.criterion == "2.4.6"]
        assert len(med) >= 1

    def test_clean_heading_passes(self):
        body = '<h2>Introduction to Algorithms</h2>'
        result = WCAGValidator().validate(_wrap(body))
        med = [i for i in result.issues
               if i.criterion == "2.4.6" and 'artifact' in i.message.lower()]
        assert len(med) == 0


# ---------------------------------------------------------------------- #
# False-positive fixes
# ---------------------------------------------------------------------- #


class TestFalsePositivesFixed:
    def test_modern_focus_visible_pattern_passes(self):
        """`:focus:not(:focus-visible){outline:none}` + `:focus-visible{outline:2px}`
        is the canonical modern pattern — no warning."""
        css = """
        :focus:not(:focus-visible) { outline: none; }
        :focus-visible { outline: 2px solid #0a84ff; }
        """
        body = f'<p>Hello world</p>'
        html = _wrap(body, extra_head=f'<style>{css}</style>')
        result = WCAGValidator().validate(html)
        focus_high = [i for i in result.issues
                      if i.criterion == "2.4.7" and i.severity == IssueSeverity.HIGH]
        assert len(focus_high) == 0, (
            "Modern focus-visible pattern must not trigger a warning: "
            + str([(i.severity.value, i.message) for i in result.issues])
        )

    def test_main_with_role_main_single_landmark(self):
        """A single <main role="main"> is ONE landmark, not two."""
        html = (
            '<!DOCTYPE html><html lang="en"><head><title>T</title></head>'
            '<body><main role="main"><h1>Main</h1><p>Content</p></main></body></html>'
        )
        result = WCAGValidator().validate(html)
        multi = [i for i in result.issues
                 if 'multiple main' in i.message.lower()]
        assert len(multi) == 0

    def test_multiple_main_still_flagged(self):
        """Distinct <main> elements (actually two) are still flagged."""
        html = (
            '<!DOCTYPE html><html lang="en"><head><title>T</title></head>'
            '<body><main><h1>One</h1></main><div role="main"><h2>Two</h2></div></body></html>'
        )
        result = WCAGValidator().validate(html)
        multi = [i for i in result.issues
                 if 'multiple main' in i.message.lower()]
        assert len(multi) >= 1

    def test_visually_hidden_skip_link_exempt(self):
        """1px visually-hidden skip-link is exempt from 24px target-size."""
        css = """
        .skip-link {
            position: absolute;
            left: -9999px;
            width: 1px;
            height: 1px;
            overflow: hidden;
        }
        .skip-link:focus {
            position: static;
            width: auto;
            height: auto;
        }
        .btn.skip-link { width: 1px; height: 1px; }
        """
        body = '<a href="#main" class="skip-link">Skip to main content</a><p>text</p>'
        html = _wrap(body, extra_head=f'<style>{css}</style>')
        result = WCAGValidator().validate(html)
        target_issues = [i for i in result.issues
                         if i.criterion == "2.5.8" and i.severity == IssueSeverity.HIGH]
        assert len(target_issues) == 0, (
            f"Visually-hidden skip-link must not trigger target-size: "
            f"{[(i.severity.value, i.message) for i in target_issues]}"
        )


# ---------------------------------------------------------------------- #
# Score formula
# ---------------------------------------------------------------------- #


class TestScoreFormula:
    def test_clean_html_gets_full_score_via_gate_adapter(self, tmp_path):
        html_path = tmp_path / "clean.html"
        html_path.write_text(_wrap(
            '<p>Real content</p>'
            '<ul><li>A</li><li>B</li></ul>'
        ), encoding="utf-8")
        result = WCAGValidator().validate({"html_path": str(html_path)})
        assert result.passed is True
        assert result.score is not None
        assert result.score >= 0.9

    def test_critical_failures_drop_score(self, tmp_path):
        # Lots of critical issues
        body = ""
        for i in range(20):
            body += f'<figure><img src="x{i}.png" alt=""><figcaption>Cap {i}</figcaption></figure>'
        body += '<ul></ul>' * 5
        html_path = tmp_path / "bad.html"
        html_path.write_text(_wrap(body), encoding="utf-8")
        result = WCAGValidator().validate({"html_path": str(html_path)})
        assert result.passed is False
        assert result.score is not None
        assert result.score < 0.9
