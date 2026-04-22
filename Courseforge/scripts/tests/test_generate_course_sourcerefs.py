"""Wave 9 — ``generate_course.py`` source-provenance emit tests.

Covers:

* Page-level JSON-LD ``sourceReferences[]`` emitted when
  ``source_module_map`` is populated.
* HTML ``data-cf-source-ids`` + optional ``data-cf-source-primary``
  attributes on ``<section>`` / headings / component wrappers.
* Section-level override shape via ``section["source_references"]``.
* Backward-compat path: empty / None source_module_map -> no refs
  emitted, no attributes on wrappers, no errors raised.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_course import (  # noqa: E402
    _build_page_metadata,
    _build_sections_metadata,
    _page_refs_for,
    _refs_primary,
    _refs_to_id_list,
    _source_attr_string,
    _summary_recap_paragraphs,
    generate_course,
    generate_week,
)


# ---------------------------------------------------------------------- #
# Helpers for extracting JSON-LD + attributes from rendered HTML
# ---------------------------------------------------------------------- #


_JSON_LD_RE = re.compile(
    r'<script\s+type="application/ld\+json">(.*?)</script>', re.DOTALL,
)
_SOURCE_IDS_RE = re.compile(r'data-cf-source-ids="([^"]*)"')
_SOURCE_PRIMARY_RE = re.compile(r'data-cf-source-primary="([^"]*)"')


def _extract_json_ld(html: str) -> dict:
    match = _JSON_LD_RE.search(html)
    assert match, "Page HTML missing JSON-LD block"
    return json.loads(match.group(1))


def _all_source_id_attrs(html: str):
    return [m.group(1) for m in _SOURCE_IDS_RE.finditer(html)]


def _all_source_primary_attrs(html: str):
    return [m.group(1) for m in _SOURCE_PRIMARY_RE.finditer(html)]


# ---------------------------------------------------------------------- #
# Fixtures: minimal week data + populated + empty source maps
# ---------------------------------------------------------------------- #


@pytest.fixture
def week_data():
    return {
        "week_number": 3,
        "title": "Visual Perception",
        "objectives": [
            {"id": "CO-03", "statement": "Apply color contrast rules",
             "bloom_level": "apply"},
        ],
        "overview_text": ["Intro paragraph."],
        "readings": ["Ch. 5 pp. 80-92"],
        "content_modules": [
            {
                "title": "POUR Principles",
                "sections": [
                    {
                        "heading": "Definition",
                        "content_type": "definition",
                        "paragraphs": ["POUR stands for ..."],
                        "flip_cards": [
                            {"term": "Perceivable",
                             "definition": "Info available to the senses"}
                        ],
                    },
                    {
                        "heading": "Example",
                        "content_type": "example",
                        "paragraphs": ["Consider a form without labels ..."],
                    },
                ],
            }
        ],
        "activities": [
            {"title": "Color Audit",
             "description": "Evaluate contrast on a real page.",
             "bloom_level": "apply"},
        ],
        "self_check_questions": [
            {
                "question": "Which principle covers alt text?",
                "bloom_level": "remember",
                "options": [
                    {"text": "Perceivable", "correct": True, "feedback": "Yes"},
                    {"text": "Operable", "correct": False, "feedback": "No"},
                ],
            }
        ],
        "key_takeaways": ["POUR is the accessibility foundation."],
        "reflection_questions": ["Which principle feels most challenging?"],
        "discussion": {"prompt": "Share an accessibility barrier you have seen."},
    }


@pytest.fixture
def populated_source_map():
    """A populated map covering every page type generate_week can emit."""
    return {
        "week_03": {
            "week_03_overview": {
                "primary": ["dart:science_of_learning#s5_p0"],
                "contributing": ["dart:science_of_learning#s4_p0"],
                "confidence": 0.82,
            },
            "week_03_content_01_pour_principles": {
                "primary": ["dart:science_of_learning#s5_p2"],
                "contributing": [
                    "dart:science_of_learning#s4_p0",
                    "dart:science_of_learning#s6_p1",
                ],
                "confidence": 0.9,
            },
            "week_03_application": {
                "primary": ["dart:science_of_learning#s7_p0"],
                "contributing": [],
                "confidence": 0.75,
            },
            "week_03_self_check": {
                "primary": [],
                "contributing": ["dart:science_of_learning#s5_p2"],
                "confidence": 0.5,
            },
            "week_03_summary": {
                "primary": ["dart:science_of_learning#s5_p0"],
                "contributing": [],
                "confidence": 0.7,
            },
            "week_03_discussion": {
                "primary": ["dart:science_of_learning#s5_p0"],
                "contributing": [],
                "confidence": 0.6,
            },
        }
    }


# ---------------------------------------------------------------------- #
# Unit tests on the helper functions
# ---------------------------------------------------------------------- #


class TestHelpers:
    def test_refs_to_id_list_skips_malformed_entries(self):
        refs = [
            {"sourceId": "dart:slug#s0", "role": "primary"},
            {"role": "contributing"},
            "not-a-dict",
            {"sourceId": "", "role": "primary"},
        ]
        assert _refs_to_id_list(refs) == ["dart:slug#s0"]

    def test_refs_to_id_list_empty_inputs(self):
        assert _refs_to_id_list(None) == []
        assert _refs_to_id_list([]) == []

    def test_refs_primary_picks_single_primary(self):
        refs = [
            {"sourceId": "dart:slug#s0", "role": "primary"},
            {"sourceId": "dart:slug#s1", "role": "contributing"},
        ]
        assert _refs_primary(refs) == "dart:slug#s0"

    def test_refs_primary_returns_none_when_multiple_primaries(self):
        refs = [
            {"sourceId": "dart:slug#s0", "role": "primary"},
            {"sourceId": "dart:slug#s1", "role": "primary"},
        ]
        assert _refs_primary(refs) is None

    def test_refs_primary_returns_none_when_no_primary(self):
        refs = [{"sourceId": "dart:slug#s0", "role": "contributing"}]
        assert _refs_primary(refs) is None

    def test_page_refs_for_populated(self, populated_source_map):
        refs = _page_refs_for(populated_source_map, 3, "week_03_content_01_pour_principles")
        assert refs is not None
        assert refs[0]["role"] == "primary"
        assert refs[0]["sourceId"] == "dart:science_of_learning#s5_p2"
        # confidence propagates from the map entry to every ref.
        assert all(r["confidence"] == 0.9 for r in refs)

    def test_page_refs_for_empty_map(self):
        assert _page_refs_for(None, 3, "x") is None
        assert _page_refs_for({}, 3, "x") is None

    def test_page_refs_for_short_key_fallback(self, populated_source_map):
        """If the map stores short keys (post-prefix), lookup still works."""
        short_map = {
            "week_03": {
                "content_01_pour_principles": {
                    "primary": ["dart:x#y"],
                    "contributing": [],
                    "confidence": 0.5,
                }
            }
        }
        refs = _page_refs_for(short_map, 3, "week_03_content_01_pour_principles")
        assert refs is not None
        assert refs[0]["sourceId"] == "dart:x#y"

    def test_source_attr_string_empty(self):
        assert _source_attr_string(None) == ""
        assert _source_attr_string([]) == ""

    def test_source_attr_string_joined_with_primary(self):
        out = _source_attr_string(["dart:slug#a", "dart:slug#b"], "dart:slug#a")
        assert 'data-cf-source-ids="dart:slug#a,dart:slug#b"' in out
        assert 'data-cf-source-primary="dart:slug#a"' in out

    def test_source_attr_string_no_primary(self):
        out = _source_attr_string(["dart:slug#a"])
        assert 'data-cf-source-ids="dart:slug#a"' in out
        assert "data-cf-source-primary" not in out


class TestBuildPageMetadata:
    def test_source_references_elided_when_absent(self):
        meta = _build_page_metadata("SAMPLE_101", 3, "content", "p")
        assert "sourceReferences" not in meta

    def test_source_references_emitted_when_populated(self):
        refs = [{"sourceId": "dart:x#y", "role": "primary"}]
        meta = _build_page_metadata(
            "SAMPLE_101", 3, "content", "p", source_references=refs,
        )
        assert meta["sourceReferences"] == refs


class TestBuildSectionsMetadata:
    def test_section_source_refs_elided_by_default(self):
        sections = _build_sections_metadata(
            [{"heading": "h", "content_type": "explanation"}]
        )
        assert "sourceReferences" not in sections[0]

    def test_section_source_refs_emitted_when_declared(self):
        refs = [{"sourceId": "dart:x#y", "role": "primary"}]
        sections = _build_sections_metadata([
            {
                "heading": "h",
                "content_type": "definition",
                "source_references": refs,
            }
        ])
        assert sections[0]["sourceReferences"] == refs


# ---------------------------------------------------------------------- #
# Integration: full generate_week round-trip
# ---------------------------------------------------------------------- #


class TestGenerateWeekWithSourceMap:
    def test_all_pages_carry_source_references(
        self, tmp_path, week_data, populated_source_map
    ):
        out = tmp_path / "out"
        generate_week(
            week_data, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        week_dir = out / "week_03"
        expected_pages = [
            "week_03_overview.html",
            "week_03_content_01_pour_principles.html",
            "week_03_application.html",
            "week_03_self_check.html",
            "week_03_summary.html",
            "week_03_discussion.html",
        ]
        for name in expected_pages:
            page_path = week_dir / name
            assert page_path.exists(), f"Missing emitted page {name}"
            meta = _extract_json_ld(page_path.read_text())
            assert "sourceReferences" in meta, (
                f"{name} JSON-LD should carry sourceReferences when the "
                "source_module_map populates that page."
            )
            assert meta["sourceReferences"], (
                f"{name} sourceReferences must be non-empty"
            )
            for ref in meta["sourceReferences"]:
                assert ref["sourceId"].startswith("dart:")
                assert ref["role"] in ("primary", "contributing", "corroborating")

    def test_html_wrappers_carry_data_cf_source_ids(
        self, tmp_path, week_data, populated_source_map
    ):
        out = tmp_path / "out"
        generate_week(
            week_data, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        content_html = (out / "week_03" / "week_03_content_01_pour_principles.html").read_text()
        attrs = _all_source_id_attrs(content_html)
        assert attrs, "Content page must carry data-cf-source-ids attributes"
        assert any("dart:science_of_learning#s5_p2" in a for a in attrs)

    def test_data_cf_source_primary_emitted_when_unambiguous(
        self, tmp_path, week_data, populated_source_map
    ):
        out = tmp_path / "out"
        generate_week(
            week_data, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        content_html = (out / "week_03" / "week_03_content_01_pour_principles.html").read_text()
        primaries = _all_source_primary_attrs(content_html)
        assert primaries
        assert all(p == "dart:science_of_learning#s5_p2" for p in primaries)

    def test_self_check_wrapper_carries_source_ids(
        self, tmp_path, week_data, populated_source_map
    ):
        out = tmp_path / "out"
        generate_week(
            week_data, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        sc_html = (out / "week_03" / "week_03_self_check.html").read_text()
        # self-check wrapper must carry data-cf-source-ids
        assert 'class="self-check"' in sc_html
        assert 'data-cf-source-ids="dart:science_of_learning#s5_p2"' in sc_html

    def test_activity_card_carries_source_ids(
        self, tmp_path, week_data, populated_source_map
    ):
        out = tmp_path / "out"
        generate_week(
            week_data, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        app_html = (out / "week_03" / "week_03_application.html").read_text()
        assert 'class="activity-card"' in app_html
        assert 'data-cf-source-ids="dart:science_of_learning#s7_p0"' in app_html

    def test_no_source_attrs_on_p_or_li_elements(
        self, tmp_path, week_data, populated_source_map
    ):
        """P2 decision: never on per-paragraph / list-item / table-row."""
        out = tmp_path / "out"
        generate_week(
            week_data, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        for page in (out / "week_03").glob("*.html"):
            html = page.read_text()
            # Simple scan: no <p ...data-cf-source-ids> / <li ...source-ids> /
            # <tr ...source-ids> anywhere in the rendered pages.
            assert not re.search(r"<p\b[^>]*data-cf-source-ids", html), page.name
            assert not re.search(r"<li\b[^>]*data-cf-source-ids", html), page.name
            assert not re.search(r"<tr\b[^>]*data-cf-source-ids", html), page.name


# ---------------------------------------------------------------------- #
# Backward compat: no source map -> no emit, no errors
# ---------------------------------------------------------------------- #


class TestBackwardCompat:
    def test_generate_week_with_none_map_emits_no_source_refs(
        self, tmp_path, week_data
    ):
        out = tmp_path / "out"
        generate_week(week_data, out, "SAMPLE_101", source_module_map=None)
        for page in (out / "week_03").glob("*.html"):
            html = page.read_text()
            meta = _extract_json_ld(html)
            assert "sourceReferences" not in meta, (
                f"{page.name} must not emit sourceReferences when map is None"
            )
            assert "data-cf-source-ids" not in html, (
                f"{page.name} must not emit data-cf-source-ids without a map"
            )

    def test_generate_week_with_empty_map_emits_no_source_refs(
        self, tmp_path, week_data
    ):
        out = tmp_path / "out"
        generate_week(week_data, out, "SAMPLE_101", source_module_map={})
        for page in (out / "week_03").glob("*.html"):
            html = page.read_text()
            meta = _extract_json_ld(html)
            assert "sourceReferences" not in meta

    def test_generate_week_with_map_missing_this_week_emits_nothing(
        self, tmp_path, week_data
    ):
        """A map that covers other weeks but not this one -> no emit here."""
        out = tmp_path / "out"
        other_week_map = {
            "week_05": {
                "week_05_overview": {
                    "primary": ["dart:x#y"], "contributing": [], "confidence": 0.5
                }
            }
        }
        generate_week(
            week_data, out, "SAMPLE_101", source_module_map=other_week_map,
        )
        for page in (out / "week_03").glob("*.html"):
            html = page.read_text()
            assert "data-cf-source-ids" not in html


# ---------------------------------------------------------------------- #
# Full course round-trip via generate_course
# ---------------------------------------------------------------------- #


class TestGenerateCourseRoundTrip:
    def test_generate_course_loads_source_module_map_from_path(
        self, tmp_path, week_data, populated_source_map
    ):
        course_data = {
            "course_code": "SAMPLE_101",
            "course_title": "Sample",
            "weeks": [week_data],
        }
        data_path = tmp_path / "course_data.json"
        data_path.write_text(json.dumps(course_data))
        map_path = tmp_path / "source_module_map.json"
        map_path.write_text(json.dumps(populated_source_map))
        out = tmp_path / "out"
        generate_course(
            str(data_path), str(out),
            source_module_map_path=str(map_path),
        )
        content_html = (out / "week_03" / "week_03_content_01_pour_principles.html").read_text()
        assert 'data-cf-source-ids="dart:science_of_learning#s5_p2' in content_html

    def test_generate_course_with_no_map_preserves_legacy_shape(
        self, tmp_path, week_data
    ):
        course_data = {
            "course_code": "SAMPLE_101",
            "course_title": "Sample",
            "weeks": [week_data],
        }
        data_path = tmp_path / "course_data.json"
        data_path.write_text(json.dumps(course_data))
        out = tmp_path / "out"
        generate_course(str(data_path), str(out))
        for page in (out / "week_03").glob("*.html"):
            html = page.read_text()
            assert "data-cf-source-ids" not in html
            meta = _extract_json_ld(html)
            assert "sourceReferences" not in meta


# ---------------------------------------------------------------------- #
# Wave 41 — ancestor-walkable grounding on overview / application /
# self_check / summary page bodies. Mirrors the Wave 35 fix in
# ``_render_content_sections`` so the ContentGroundingValidator's
# ancestor walk finds ``data-cf-source-ids`` on every non-trivial body
# <p>/<li>/<figcaption>/<blockquote>. Before Wave 41 these page bodies
# emitted raw <h2> + <p> siblings of <main> with no grounding ancestor
# — smoke test on hifi_rag.pdf flagged 10 ungrounded paragraphs across
# these page types and scored 0.63 (< 1.0 threshold → gate FAILED).
# ---------------------------------------------------------------------- #


# Word count mirrors the validator's NON_TRIVIAL_WORD_FLOOR = 30: each
# non-trivial <p>/<li> needs ≥ 30 words for the ancestor walk to even
# consider it. We embed generous filler so each candidate element
# clears the floor without depending on the helpers's shape.
_NON_TRIVIAL_PARAGRAPH = (
    "This sample paragraph deliberately contains enough substantive "
    "educational prose to clear the thirty-word non-trivial threshold "
    "that the content grounding validator enforces when walking "
    "ancestors for the grounding attribute across every emitted page "
    "of the generated course."
)


def _ancestor_source_ids(element):
    """Mirror of :meth:`ContentGroundingValidator._find_source_ids`.

    Walks the element + every parent in the BeautifulSoup tree, returning
    the first ``data-cf-source-ids`` encountered (or ``None``).
    """
    cur = element
    while cur is not None and hasattr(cur, "get"):
        val = cur.get("data-cf-source-ids")
        if val:
            return val
        cur = cur.parent
    return None


def _non_trivial_candidates(soup):
    """Yield every non-trivial <p>/<li>/<figcaption>/<blockquote>.

    Matches the candidate-set walk in ContentGroundingValidator.validate.
    """
    # Strip the nav/header/footer chrome first, same as the validator.
    for tag in soup.find_all(["nav", "header", "footer"]):
        tag.decompose()
    for el in soup.find_all(["p", "li", "figcaption", "blockquote"]):
        text = el.get_text(separator=" ", strip=True)
        if len(text.split()) >= 30:
            yield el


class TestWave41OverviewBodyWrap:
    @pytest.fixture
    def week_with_long_overview(self, week_data):
        """Week fixture with a long overview paragraph that clears the
        non-trivial word floor so the ancestor walk evaluates it."""
        week_data = dict(week_data)
        week_data["overview_text"] = [_NON_TRIVIAL_PARAGRAPH]
        return week_data

    def test_overview_body_wrapped_with_source_ids(
        self, tmp_path, week_with_long_overview, populated_source_map
    ):
        bs4 = pytest.importorskip("bs4")
        out = tmp_path / "out"
        generate_week(
            week_with_long_overview, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        html = (out / "week_03" / "week_03_overview.html").read_text()
        soup = bs4.BeautifulSoup(html, "html.parser")
        candidates = list(_non_trivial_candidates(soup))
        assert candidates, (
            "Overview page must emit at least one non-trivial <p>/<li> "
            "for this test to be meaningful."
        )
        for el in candidates:
            ids = _ancestor_source_ids(el)
            assert ids, (
                f"Overview <{el.name}> {el.get_text()[:60]!r} must have "
                "a data-cf-source-ids ancestor (Wave 41 grounding)."
            )
            assert "dart:science_of_learning#s5_p0" in ids

    def test_overview_no_wrap_when_map_empty(
        self, tmp_path, week_with_long_overview
    ):
        """Back-compat: empty source map → no <section data-cf-source-ids>
        wrapper should be emitted. Mirrors the Wave 35 / Wave 9 invariant
        enforced by :class:`TestBackwardCompat`.

        The fixture's overview_text contains the literal phrase
        "data-cf-source-ids" as prose, so we check for the HTML
        attribute pattern (``data-cf-source-ids="…"``) rather than the
        bare token to avoid a false positive.
        """
        out = tmp_path / "out"
        generate_week(
            week_with_long_overview, out, "SAMPLE_101",
            source_module_map=None,
        )
        html = (out / "week_03" / "week_03_overview.html").read_text()
        assert not re.search(r'data-cf-source-ids="', html), (
            "Overview page must not emit any data-cf-source-ids "
            "attribute when the source_module_map is None (Wave 9 "
            "back-compat contract)."
        )
        # And no <section> wrapper emitted by Wave 41 at all.
        assert not re.search(
            r'<section\s+data-cf-source-ids', html
        ), "Wave 41 wrapper must not emit when source_module_map is None."


class TestWave41ApplicationBodyWrap:
    @pytest.fixture
    def week_with_long_activity(self, week_data):
        """Week fixture with an activity whose description is long enough
        to produce at least one non-trivial candidate on the application
        page body.
        """
        week_data = dict(week_data)
        week_data["activities"] = [
            {
                "title": "Color Audit",
                "description": _NON_TRIVIAL_PARAGRAPH,
                "bloom_level": "apply",
            },
        ]
        return week_data

    def test_application_body_wrapped_with_source_ids(
        self, tmp_path, week_with_long_activity, populated_source_map
    ):
        bs4 = pytest.importorskip("bs4")
        out = tmp_path / "out"
        generate_week(
            week_with_long_activity, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        html = (out / "week_03" / "week_03_application.html").read_text()
        soup = bs4.BeautifulSoup(html, "html.parser")
        candidates = list(_non_trivial_candidates(soup))
        assert candidates, "Application page needs at least one non-trivial <p>/<li>."
        for el in candidates:
            ids = _ancestor_source_ids(el)
            assert ids, (
                f"Application <{el.name}> {el.get_text()[:60]!r} must "
                "have a data-cf-source-ids ancestor (Wave 41 grounding)."
            )
            assert "dart:science_of_learning#s7_p0" in ids


class TestWave41SelfCheckBodyWrap:
    @pytest.fixture
    def week_with_long_self_check(self, week_data):
        """Self-check fixture with a question long enough that the emitted
        <p>/<li> markup passes the 30-word threshold.
        """
        week_data = dict(week_data)
        long_q = (
            "Which POUR principle most directly covers alt text for "
            "images, captions for audio, and transcripts for videos in "
            "WCAG 2.2 AA content that must be accessible to users with "
            "sensory disabilities?"
        )
        week_data["self_check_questions"] = [
            {
                "question": long_q,
                "bloom_level": "remember",
                "options": [
                    {"text": "Perceivable", "correct": True, "feedback": "Yes"},
                    {"text": "Operable", "correct": False, "feedback": "No"},
                ],
            }
        ]
        return week_data

    def test_self_check_body_wrapped_with_source_ids(
        self, tmp_path, week_with_long_self_check, populated_source_map
    ):
        bs4 = pytest.importorskip("bs4")
        out = tmp_path / "out"
        # Inject a populated self_check map entry with primary IDs so the
        # body wrapper has something to emit (the default fixture has an
        # empty primary list which would suppress the wrapper).
        sm = json.loads(json.dumps(populated_source_map))
        sm["week_03"]["week_03_self_check"] = {
            "primary": ["dart:science_of_learning#s5_p2"],
            "contributing": [],
            "confidence": 0.8,
        }
        generate_week(
            week_with_long_self_check, out, "SAMPLE_101",
            source_module_map=sm,
        )
        html = (out / "week_03" / "week_03_self_check.html").read_text()
        soup = bs4.BeautifulSoup(html, "html.parser")
        candidates = list(_non_trivial_candidates(soup))
        assert candidates, "Self-check page needs at least one non-trivial <p>/<li>."
        for el in candidates:
            ids = _ancestor_source_ids(el)
            assert ids, (
                f"Self-check <{el.name}> {el.get_text()[:60]!r} must "
                "have a data-cf-source-ids ancestor (Wave 41 grounding)."
            )
            assert "dart:science_of_learning#s5_p2" in ids


class TestWave41SummaryBodyWrap:
    @pytest.fixture
    def week_with_long_summary(self, week_data):
        """Summary fixture with a long key-takeaway list item + preview so
        the summary body carries non-trivial <p>/<li> children.
        """
        week_data = dict(week_data)
        week_data["key_takeaways"] = [_NON_TRIVIAL_PARAGRAPH]
        week_data["next_week_preview"] = _NON_TRIVIAL_PARAGRAPH
        return week_data

    def test_summary_body_wrapped_with_source_ids(
        self, tmp_path, week_with_long_summary, populated_source_map
    ):
        bs4 = pytest.importorskip("bs4")
        out = tmp_path / "out"
        generate_week(
            week_with_long_summary, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        html = (out / "week_03" / "week_03_summary.html").read_text()
        soup = bs4.BeautifulSoup(html, "html.parser")
        candidates = list(_non_trivial_candidates(soup))
        assert candidates, "Summary page needs at least one non-trivial <p>/<li>."
        for el in candidates:
            ids = _ancestor_source_ids(el)
            assert ids, (
                f"Summary <{el.name}> {el.get_text()[:60]!r} must "
                "have a data-cf-source-ids ancestor (Wave 41 grounding)."
            )
            assert "dart:science_of_learning#s5_p0" in ids


# ---------------------------------------------------------------------- #
# Wave 43 — summary pages emit a Chapter Recap <section> carrying 1-3
# substantive <p>s from the week's content_modules so the summary page
# contributes non-trivial paragraphs to ContentGroundingValidator's
# AGGREGATE_EMPTY_PAGES count. Pre-Wave-43 summary pages emitted only
# the Key Takeaways list (5-15-word <li>s) + reflection prompts -> 0
# non-trivial paragraphs -> when summary was ≥~15% of total pages
# (e.g. 8/44 on hifi_rag), AGGREGATE_EMPTY_PAGES tripped critical.
# ---------------------------------------------------------------------- #


# Long topic paragraph mirroring what DART produces — ≥30 words so the
# validator's NON_TRIVIAL_WORD_FLOOR considers the rendered <p>, and ≥
# a few distinctive phrases so the test can assert the recap is the
# DART-sourced text rather than boilerplate.
_RECAP_PARAGRAPH = (
    "Retrieval-Augmented Generation blends parametric language models "
    "with a non-parametric retrieval pass so the generator can ground "
    "its answer in the freshest documents instead of relying only on "
    "training-time weights, which materially reduces hallucination on "
    "long-tail factual prompts."
)


class TestWave43SummaryRecapHelper:
    """Unit tests on :func:`_summary_recap_paragraphs`."""

    def test_picks_first_non_trivial_paragraph_per_module(self):
        modules = [
            {
                "title": "POUR Principles",
                "sections": [
                    {"heading": "Definition",
                     "paragraphs": [_RECAP_PARAGRAPH]},
                ],
            },
            {
                "title": "Contrast Ratios",
                "sections": [
                    {"heading": "Calculation",
                     "paragraphs": [_RECAP_PARAGRAPH + " (module 2)"]},
                ],
            },
        ]
        out = _summary_recap_paragraphs(modules)
        assert 1 <= len(out) <= 3
        assert all(len(p.split()) >= 30 for p in out)

    def test_skips_short_paragraphs_below_non_trivial_floor(self):
        modules = [
            {
                "title": "X",
                "sections": [
                    {"heading": "h", "paragraphs": ["Short."]},
                ],
            },
        ]
        assert _summary_recap_paragraphs(modules) == []

    def test_empty_content_modules_returns_empty(self):
        assert _summary_recap_paragraphs([]) == []
        assert _summary_recap_paragraphs(None) == []

    def test_caps_total_words(self):
        modules = [
            {
                "title": f"m{i}",
                "sections": [
                    {"heading": "h", "paragraphs": [_RECAP_PARAGRAPH]},
                ],
            }
            for i in range(10)
        ]
        out = _summary_recap_paragraphs(modules, max_total_words=60)
        total_words = sum(len(p.split()) for p in out)
        # First paragraph (~35 words) lands whole; next would push over
        # the 60-word cap, so the loop should break after 1-2 items.
        assert total_words <= 100
        assert 1 <= len(out) <= 2

    def test_caps_paragraph_length(self):
        long_para = " ".join(["word"] * 200)  # ~200 words, ~1000 chars
        modules = [
            {
                "title": "X",
                "sections": [
                    {"heading": "h", "paragraphs": [long_para]},
                ],
            },
        ]
        out = _summary_recap_paragraphs(
            modules, max_chars_per_paragraph=200, max_total_words=1000,
        )
        assert len(out) == 1
        # Truncated + ellipsis; length cap ≤ 200 chars + a few trailing.
        assert len(out[0]) <= 210
        assert out[0].endswith("...")

    def test_dedupes_identical_paragraphs(self):
        modules = [
            {
                "title": f"m{i}",
                "sections": [
                    {"heading": "h", "paragraphs": [_RECAP_PARAGRAPH]},
                ],
            }
            for i in range(3)
        ]
        out = _summary_recap_paragraphs(modules)
        assert len(out) == 1

    def test_short_paragraph_does_not_block_later_substantive_dupe_prefix(self):
        """Wave 44 regression: pre-Wave-44 a short ineligible paragraph
        added its 80-char prefix to the ``seen`` set BEFORE the 30-word
        eligibility check. A later substantive paragraph sharing the
        same opening text was then silently dropped, leaving the recap
        empty on corpora where successive sections use a common lead-in
        phrase (e.g. "In this chapter we examine...").
        """
        # Shared 80-char prefix on two paragraphs — first is short
        # (ineligible), second is substantive (eligible).
        prefix = "In this chapter we examine the key ideas and methods that"
        assert len(prefix) >= 50  # shares 80-char key with the substantive one
        short = prefix + " briefly."
        long_body = prefix + " " + " ".join(["word"] * 40)
        assert len(short.split()) < 30
        assert len(long_body.split()) >= 30

        modules = [
            {
                "title": "brief",
                "sections": [
                    {"heading": "h", "paragraphs": [short]},
                ],
            },
            {
                "title": "substantive",
                "sections": [
                    {"heading": "h", "paragraphs": [long_body]},
                ],
            },
        ]
        out = _summary_recap_paragraphs(modules)
        # Must keep the substantive paragraph; pre-Wave-44 returned [].
        assert len(out) == 1
        assert len(out[0].split()) >= 30


class TestWave43SummaryRecapEmit:
    """Integration tests: generate_week emits Chapter Recap on summary."""

    @pytest.fixture
    def week_with_topic_paragraphs(self, week_data):
        """Week fixture whose content_modules carry a non-trivial
        paragraph the recap helper can lift."""
        week_data = dict(week_data)
        week_data["content_modules"] = [
            {
                "title": "POUR Principles",
                "sections": [
                    {
                        "heading": "Definition",
                        "content_type": "definition",
                        "paragraphs": [_RECAP_PARAGRAPH],
                    },
                ],
            },
        ]
        return week_data

    def test_summary_page_emits_recap_paragraphs_when_topics_present(
        self, tmp_path, week_with_topic_paragraphs, populated_source_map,
    ):
        bs4 = pytest.importorskip("bs4")
        out = tmp_path / "out"
        generate_week(
            week_with_topic_paragraphs, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        html = (out / "week_03" / "week_03_summary.html").read_text()
        # Chapter Recap heading must appear (distinct from Key Takeaways).
        assert "<h2>Chapter Recap</h2>" in html
        soup = bs4.BeautifulSoup(html, "html.parser")
        # At least one <p> with ≥30 words, wrapped in a
        # <section data-cf-source-ids="…"> ancestor.
        non_trivial_ps = [
            p for p in soup.find_all("p")
            if len(p.get_text(separator=" ", strip=True).split()) >= 30
        ]
        assert non_trivial_ps, (
            "Wave 43 recap must emit at least one non-trivial <p> "
            "(≥30 words) on the summary page."
        )
        # Every non-trivial <p> must carry a source-ids ancestor — the
        # whole point of this wave.
        for p in non_trivial_ps:
            ids = _ancestor_source_ids(p)
            assert ids, (
                "Wave 43 recap <p> must have a data-cf-source-ids "
                "ancestor (AGGREGATE_EMPTY_PAGES fix)."
            )
            assert "dart:science_of_learning#s5_p0" in ids

    def test_summary_page_no_recap_when_topics_empty(
        self, tmp_path, week_data, populated_source_map,
    ):
        """Back-compat: empty / paragraph-less content_modules -> no
        Chapter Recap heading (never emit <h2> with no body prose)."""
        week_data = dict(week_data)
        week_data["content_modules"] = []
        out = tmp_path / "out"
        generate_week(
            week_data, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        html = (out / "week_03" / "week_03_summary.html").read_text()
        assert "<h2>Chapter Recap</h2>" not in html, (
            "Recap heading must not appear when content_modules is empty "
            "(legacy shell back-compat)."
        )
        # Legacy shell Key Takeaways + Reflection still render.
        assert "Key Takeaways" in html

    def test_summary_page_no_recap_when_paragraphs_too_short(
        self, tmp_path, week_data, populated_source_map,
    ):
        """Topic paragraphs below the non-trivial floor -> no recap
        emitted (prevents heading-only ungrounded section)."""
        week_data = dict(week_data)
        week_data["content_modules"] = [
            {
                "title": "Short",
                "sections": [
                    {"heading": "h", "paragraphs": ["Too short."]},
                ],
            },
        ]
        out = tmp_path / "out"
        generate_week(
            week_data, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        html = (out / "week_03" / "week_03_summary.html").read_text()
        assert "<h2>Chapter Recap</h2>" not in html

    def test_summary_recap_grounded_ancestor_walk(
        self, tmp_path, week_with_topic_paragraphs, populated_source_map,
    ):
        """Every non-trivial <p> or <li> on the summary page has an
        ancestor carrying data-cf-source-ids (mirrors the validator's
        grounding walk)."""
        bs4 = pytest.importorskip("bs4")
        out = tmp_path / "out"
        generate_week(
            week_with_topic_paragraphs, out, "SAMPLE_101",
            source_module_map=populated_source_map,
        )
        html = (out / "week_03" / "week_03_summary.html").read_text()
        soup = bs4.BeautifulSoup(html, "html.parser")
        candidates = list(_non_trivial_candidates(soup))
        assert candidates, (
            "Summary page must contribute at least one non-trivial "
            "<p>/<li> after Wave 43."
        )
        for el in candidates:
            ids = _ancestor_source_ids(el)
            assert ids, (
                f"Summary <{el.name}> {el.get_text()[:60]!r} lacks a "
                "data-cf-source-ids ancestor."
            )

    def test_summary_recap_no_emit_when_map_missing(
        self, tmp_path, week_with_topic_paragraphs,
    ):
        """Back-compat: source_module_map=None -> no Chapter Recap (no
        grounding ancestor available, so we must not emit ungrounded
        prose on a page that otherwise had none)."""
        out = tmp_path / "out"
        generate_week(
            week_with_topic_paragraphs, out, "SAMPLE_101",
            source_module_map=None,
        )
        html = (out / "week_03" / "week_03_summary.html").read_text()
        assert "<h2>Chapter Recap</h2>" not in html
        assert "data-cf-source-ids" not in html
