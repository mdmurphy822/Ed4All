"""Unit tests for the deterministic Learning Objectives Map renderer.

Covers both synthesized-objectives shapes (Courseforge dict-keyed-by-chapter
+ flat list, and LibV2 archive ``terminal_outcomes`` / ``component_objectives``),
missing-parent honesty (unmapped grouping), HTML escaping, heading hierarchy,
well-formed JSON-LD, and the deterministic (idempotent) render contract.

Synthetic PHYS_101-style fixtures only — no real course identifiers.
"""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from render_learning_objectives_page import (  # noqa: E402
    render_learning_objectives_html,
    write_learning_objectives_page,
)


# ---------------------------------------------------------------------------
# Fixtures (synthetic)
# ---------------------------------------------------------------------------

COURSEFORGE_DICT_SHAPE = {
    "course_name": "PHYS_101",
    "course_title": "Introductory Mechanics",
    "terminal_objectives": [
        {
            "id": "TO-01",
            "statement": "Apply Newton's laws to predict motion.",
            "bloom_level": "apply",
            "cognitive_domain": "procedural",
            "chapter": "1",
        },
        {
            "id": "TO-02",
            "statement": "Analyze energy transfer in mechanical systems.",
            "bloom_level": "analyze",
            "cognitive_domain": "conceptual",
            "chapter": "2",
        },
    ],
    "chapter_objectives": {
        "1": [
            {
                "id": "CO-01",
                "terminal_id": "TO-01",
                "statement": "Identify forces acting on a body.",
                "bloom_level": "remember",
                "cognitive_domain": "factual",
                "section": "1.1",
            },
            {
                "id": "CO-02",
                "terminal_id": "TO-01",
                "statement": "Compute net force from a free-body diagram.",
                "bloom_level": "apply",
                "cognitive_domain": "procedural",
                "section": "1.2",
            },
        ],
        "2": [
            {
                "id": "CO-03",
                "terminal_id": "TO-02",
                "statement": "Distinguish kinetic from potential energy.",
                "bloom_level": "understand",
                "cognitive_domain": "conceptual",
                "section": "2.1",
                "key_concepts": ["kinetic energy", "potential energy"],
            },
        ],
    },
}

ARCHIVE_SHAPE = {
    "course_name": "PHYS_101",
    "course_title": "Introductory Mechanics",
    "terminal_outcomes": [
        {"id": "TO-01", "statement": "Apply Newton's laws.", "bloom_level": "apply"},
    ],
    "component_objectives": [
        {
            "id": "CO-01",
            "parent_terminal": "TO-01",
            "statement": "Identify forces.",
            "bloom_level": "remember",
        },
    ],
}

FLAT_SHAPE = {
    "course_name": "PHYS_101",
    "course_title": "Introductory Mechanics",
    "terminal_objectives": [
        {"id": "TO-01", "statement": "Apply Newton's laws.", "bloom_level": "apply"},
    ],
    "chapter_objectives": [
        {
            "id": "CO-01",
            "terminal_id": "TO-01",
            "statement": "Identify forces.",
            "bloom_level": "remember",
            "chapter": "1",
        },
    ],
}

UNMAPPED_SHAPE = {
    "course_name": "PHYS_101",
    "course_title": "Introductory Mechanics",
    "terminal_objectives": [
        {"id": "TO-01", "statement": "Apply Newton's laws.", "bloom_level": "apply"},
    ],
    "chapter_objectives": {
        "1": [
            {
                "id": "CO-01",
                "terminal_id": "TO-01",
                "statement": "Identify forces.",
                "bloom_level": "remember",
            },
            {
                # No terminal_id — must render under "Unmapped objectives".
                "id": "CO-99",
                "statement": "An orphan objective with no parent terminal.",
                "bloom_level": "understand",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_jsonld(html_text: str) -> dict:
    m = re.search(
        r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>',
        html_text,
        re.DOTALL,
    )
    assert m, "JSON-LD block not found"
    return json.loads(m.group(1))


class _HeadingCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.headings: list[int] = []
        self._cur: int | None = None

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._cur = int(tag[1])

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.headings.append(self._cur)  # type: ignore[arg-type]
            self._cur = None


def _heading_levels(html_text: str) -> list[int]:
    c = _HeadingCollector()
    c.feed(html_text)
    return c.headings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_renders_terminal_and_chapter_objectives_dict_shape():
    out = render_learning_objectives_html(
        COURSEFORGE_DICT_SHAPE, course_code="PHYS_101"
    )
    for oid in ("TO-01", "TO-02", "CO-01", "CO-02", "CO-03"):
        assert oid in out, f"{oid} missing from rendered page"
    assert "Apply Newton&#x27;s laws to predict motion." in out
    # Nested grouping: CO-01/CO-02 under TO-01; CO-03 under TO-02.
    assert "Supporting chapter objectives" in out


def test_renders_archive_shape():
    out = render_learning_objectives_html(ARCHIVE_SHAPE, course_code="PHYS_101")
    assert "TO-01" in out
    assert "CO-01" in out
    # parent_terminal -> partOf in JSON-LD.
    data = _extract_jsonld(out)
    co = next(o for o in data["learningObjectives"] if o["id"] == "CO-01")
    assert co.get("partOf") == "TO-01"


def test_renders_flat_chapter_objectives():
    out = render_learning_objectives_html(FLAT_SHAPE, course_code="PHYS_101")
    assert "CO-01" in out and "TO-01" in out


def test_unmapped_objective_is_honest_not_fabricated():
    out = render_learning_objectives_html(UNMAPPED_SHAPE, course_code="PHYS_101")
    assert "Unmapped objectives" in out
    assert "CO-99" in out
    # The orphan must NOT be silently nested under TO-01.
    data = _extract_jsonld(out)
    co99 = next(o for o in data["learningObjectives"] if o["id"] == "CO-99")
    assert "partOf" not in co99, "orphan CO must not fabricate a parent"


def test_html_escaping_prevents_injection():
    evil = {
        "course_name": "PHYS_101",
        "course_title": "Mechanics <script>alert(1)</script>",
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Use < and > and & and \"quotes\".",
                "bloom_level": "apply",
            },
        ],
        "chapter_objectives": {},
    }
    out = render_learning_objectives_html(evil, course_code="PHYS_101")
    # The raw <script> in the title must be escaped in the rendered HTML
    # body (it is legitimately present inside the JSON-LD string value,
    # which json.dumps emits — but never as a live <script>alert).
    assert "<script>alert(1)</script>" not in out.replace(
        '"@type": "CoursePage"', ""
    )
    assert "&lt;script&gt;" in out
    assert "&amp;" in out


def test_heading_hierarchy_single_h1_no_skips():
    out = render_learning_objectives_html(
        COURSEFORGE_DICT_SHAPE, course_code="PHYS_101"
    )
    levels = _heading_levels(out)
    assert levels.count(1) == 1, "exactly one <h1> required for WCAG"
    # No skipped levels (e.g. h1 -> h3 with no h2).
    prev = 0
    for lvl in levels:
        assert lvl - prev <= 1 or lvl <= prev, (
            f"heading level jump from h{prev} to h{lvl} skips a level"
        )
        prev = lvl


def test_jsonld_well_formed_and_canonical_context():
    out = render_learning_objectives_html(
        COURSEFORGE_DICT_SHAPE, course_code="PHYS_101"
    )
    data = _extract_jsonld(out)
    assert data["@context"] == "https://ed4all.dev/ns/courseforge/v1"
    assert data["@type"] == "CoursePage"
    assert data["courseCode"] == "PHYS_101"
    assert data["weekNumber"] == 0
    ids = {o["id"] for o in data["learningObjectives"]}
    assert {"TO-01", "TO-02", "CO-01", "CO-02", "CO-03"} <= ids


def test_data_cf_metadata_present():
    out = render_learning_objectives_html(
        COURSEFORGE_DICT_SHAPE, course_code="PHYS_101"
    )
    assert 'data-cf-content-type="objectives"' in out
    assert 'data-cf-objective-id="TO-01"' in out
    assert 'data-cf-objective-id="CO-01"' in out
    assert 'data-cf-bloom-level="apply"' in out
    assert 'data-cf-cognitive-domain="factual"' in out


def test_summary_table_maps_co_chapter_to():
    out = render_learning_objectives_html(
        COURSEFORGE_DICT_SHAPE, course_code="PHYS_101"
    )
    assert "Objective Map Summary" in out
    assert "<table" in out and "<caption>" in out
    assert "<th scope=\"row\">CO-01</th>" in out


def test_key_concepts_render_as_tags():
    out = render_learning_objectives_html(
        COURSEFORGE_DICT_SHAPE, course_code="PHYS_101"
    )
    assert "kinetic energy" in out and "potential energy" in out


def test_render_is_deterministic():
    a = render_learning_objectives_html(
        COURSEFORGE_DICT_SHAPE, course_code="PHYS_101"
    )
    b = render_learning_objectives_html(
        COURSEFORGE_DICT_SHAPE, course_code="PHYS_101"
    )
    assert a == b, "renderer must be deterministic (byte-identical)"


def test_empty_objectives_does_not_crash():
    out = render_learning_objectives_html(
        {"course_name": "PHYS_101", "course_title": "Empty"},
        course_code="PHYS_101",
    )
    assert "No learning objectives are defined" in out
    # JSON-LD still well-formed with empty list.
    data = _extract_jsonld(out)
    assert data["learningObjectives"] == []


def test_write_to_file_roundtrip(tmp_path: Path):
    obj_path = tmp_path / "synthesized_objectives.json"
    obj_path.write_text(json.dumps(COURSEFORGE_DICT_SHAPE), encoding="utf-8")
    out_path = tmp_path / "course_overview" / "learning_objectives.html"
    written = write_learning_objectives_page(
        obj_path, out_path, course_code="PHYS_101"
    )
    assert written == out_path
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "TO-01" in text
