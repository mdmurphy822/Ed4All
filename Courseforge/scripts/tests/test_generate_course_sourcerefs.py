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
