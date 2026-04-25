"""Tests for ``ed4all libv2 generate-study-pack`` (Wave 77 Worker δ).

Exercises both the renderer engine and the Click CLI shim. Most tests
build a synthetic archive in ``tmp_path`` so they remain hermetic; one
integration test reaches into the real
``LibV2/courses/rdf-shacl-550-rdf-shacl-550/`` archive to verify the
canonical week-7 ordering against production data.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from click.testing import CliRunner

from cli.commands.libv2_generate_study_pack import (
    _parse_difficulties,
    _parse_weeks,
    generate_study_pack_command,
)
from cli.commands.libv2_generate_study_pack import (
    register_generate_study_pack_command,
)
from cli.commands.libv2_validate_packet import libv2_group


# Attach generate-study-pack to the shared ``libv2`` click group so the
# CLI tests below can invoke it via ``runner.invoke(libv2_group, ...)``.
# This mirrors what ``cli.commands.__init__::register_libv2_command``
# does for the production CLI; idempotent in case the registration ran
# elsewhere first.
if "generate-study-pack" not in libv2_group.commands:
    register_generate_study_pack_command(libv2_group)
from LibV2.tools.study_pack_renderer import (
    StudyPackError,
    render_html,
    render_json,
    render_markdown,
    render_study_pack,
)


# ---------------------------------------------------------------------- #
# Real archive fixture                                                   #
# ---------------------------------------------------------------------- #

REAL_ARCHIVE = (
    Path(__file__).resolve().parents[2]
    / "LibV2"
    / "courses"
    / "rdf-shacl-550-rdf-shacl-550"
)


def _real_archive_available() -> bool:
    return (REAL_ARCHIVE / "corpus" / "chunks.json").exists()


# ---------------------------------------------------------------------- #
# Synthetic archive helpers                                              #
# ---------------------------------------------------------------------- #


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_chunk(
    chunk_id: str,
    *,
    chunk_type: str,
    module_id: str,
    resource_type: str,
    title: str,
    text: str,
    word_count: int,
    los: Optional[List[str]] = None,
    difficulty: Optional[str] = "foundational",
    bloom: str = "understand",
    position: int = 0,
    source_refs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "chunk_type": chunk_type,
        "text": text,
        "word_count": word_count,
        "difficulty": difficulty,
        "bloom_level": bloom,
        "learning_outcome_refs": list(los or []),
        "source": {
            "course_id": "TEST_101",
            "module_id": module_id,
            "module_title": title,
            "lesson_title": title,
            "resource_type": resource_type,
            "section_heading": title,
            "position_in_module": position,
            "source_references": list(source_refs or []),
        },
    }


def _build_synthetic_archive(root: Path, slug: str) -> Path:
    """Assemble a tiny week-1 archive with all chunk-bucket types."""
    archive = root / slug
    archive.mkdir(parents=True)

    chunks = [
        _build_chunk(
            "c01",
            chunk_type="overview",
            module_id="week_01_overview",
            resource_type="overview",
            title="Week 1 Overview",
            text="Overview of week 1.",
            word_count=120,
            los=["to-01"],
        ),
        _build_chunk(
            "c02",
            chunk_type="explanation",
            module_id="week_01_content_01",
            resource_type="page",
            title="Content 1",
            text="Body of content 1.",
            word_count=400,
            los=["co-01", "to-01"],
        ),
        _build_chunk(
            "c03",
            chunk_type="explanation",
            module_id="week_01_content_02",
            resource_type="page",
            title="Content 2",
            text="Body of content 2.",
            word_count=600,
            los=["co-02"],
        ),
        _build_chunk(
            "c04",
            chunk_type="example",
            module_id="week_01_application",
            resource_type="application",
            title="Week 1 Application",
            text="Apply the ideas.",
            word_count=300,
            los=["co-01"],
        ),
        _build_chunk(
            "c05",
            chunk_type="exercise",
            module_id="week_01_content_06",
            resource_type="exercise",
            title="Hands-on Exercise",
            text="Exercise body.",
            word_count=250,
            difficulty="intermediate",
        ),
        _build_chunk(
            "c06",
            chunk_type="assessment_item",
            module_id="week_01_self_check",
            resource_type="quiz",
            title="Week 1 Self-Check",
            text="Question. Answer.",
            word_count=200,
            los=["co-01"],
        ),
        _build_chunk(
            "c07",
            chunk_type="summary",
            module_id="week_01_summary",
            resource_type="summary",
            title="Week 1 Summary",
            text="Summary of week 1.",
            word_count=100,
        ),
    ]

    _write_json(archive / "corpus" / "chunks.json", chunks)
    _write_json(
        archive / "objectives.json",
        {
            "course_code": "TEST_101",
            "terminal_outcomes": [
                {
                    "id": "to-01",
                    "statement": "Master the foundations.",
                    "bloom_level": "analyze",
                    "weeks": [1, 2],
                },
            ],
            "component_objectives": [
                {
                    "id": "co-01",
                    "statement": "Identify the foundational ideas.",
                    "bloom_level": "remember",
                    "parent_terminal": "to-01",
                    "week": 1,
                },
                {
                    "id": "co-02",
                    "statement": "Explain the foundational ideas.",
                    "bloom_level": "understand",
                    "parent_terminal": "to-01",
                    "week": 1,
                },
            ],
        },
    )
    _write_json(
        archive / "course.json",
        {
            "course_code": "TEST_101",
            "title": "Test 101 Course",
            "learning_outcomes": [
                {"id": "to-01", "statement": "x", "hierarchy_level": "terminal"},
            ],
        },
    )
    return archive


# ---------------------------------------------------------------------- #
# Helper-function unit tests                                             #
# ---------------------------------------------------------------------- #


def test_parse_weeks_single():
    assert _parse_weeks("7") == [7]


def test_parse_weeks_csv():
    assert _parse_weeks("1,2,3") == [1, 2, 3]


def test_parse_weeks_range():
    assert _parse_weeks("3-6") == [3, 4, 5, 6]


def test_parse_weeks_dedupes_and_sorts():
    assert _parse_weeks("3,1,3,2") == [1, 2, 3]


def test_parse_weeks_invalid():
    with pytest.raises(Exception):
        _parse_weeks("abc")


def test_parse_difficulties_csv():
    assert _parse_difficulties("foundational,intermediate") == [
        "foundational",
        "intermediate",
    ]


def test_parse_difficulties_invalid():
    with pytest.raises(Exception):
        _parse_difficulties("kindergarten")


def test_parse_difficulties_none():
    assert _parse_difficulties(None) is None
    assert _parse_difficulties("") is None


# ---------------------------------------------------------------------- #
# Synthetic-archive renderer tests                                       #
# ---------------------------------------------------------------------- #


def test_renderer_default_excludes_exercises_and_self_check(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    pack = render_study_pack(archive, weeks=[1])
    types = [c.chunk_type for c in pack.chunks]
    # Exercises and assessment_item filtered out by default.
    assert "exercise" not in types
    assert "assessment_item" not in types
    # Overview, both explanations, application, summary remain.
    assert "overview" in types
    assert "summary" in types


def test_renderer_include_exercises_adds_exercise_chunks(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    without = render_study_pack(archive, weeks=[1])
    with_ex = render_study_pack(archive, weeks=[1], include_exercises=True)
    assert "exercise" not in {c.chunk_type for c in without.chunks}
    assert "exercise" in {c.chunk_type for c in with_ex.chunks}
    assert len(with_ex.chunks) == len(without.chunks) + 1


def test_renderer_orders_buckets_correctly(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    pack = render_study_pack(
        archive,
        weeks=[1],
        include_exercises=True,
        include_self_check=True,
    )
    # Canonical order: overview -> content_01 -> content_02 -> application
    # -> exercises -> self_check -> summary.
    assert [c.chunk_id for c in pack.chunks] == [
        "c01",
        "c02",
        "c03",
        "c04",
        "c05",
        "c06",
        "c07",
    ]


def test_renderer_difficulty_filter(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    pack = render_study_pack(
        archive,
        weeks=[1],
        include_exercises=True,
        difficulties=["intermediate"],
    )
    assert all(c.difficulty == "intermediate" for c in pack.chunks)


def test_renderer_no_chunks_for_unknown_week(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    with pytest.raises(StudyPackError) as excinfo:
        render_study_pack(archive, weeks=[99])
    assert "no chunks found" in str(excinfo.value).lower()


def test_renderer_lesson_plan_collects_objectives_and_assessments(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    pack = render_study_pack(
        archive,
        weeks=[1],
        include_self_check=True,
        lesson_plan=True,
    )
    # Objectives table contains all distinct LOs referenced.
    ids = {str(o.get("id")).lower() for o in pack.objectives_referenced}
    assert {"to-01", "co-01", "co-02"} <= ids
    # Statements present (not just ID-only stubs).
    for obj in pack.objectives_referenced:
        assert obj.get("statement"), f"missing statement for {obj.get('id')}"
    # Assessment chunks captured separately.
    assert pack.assessment_chunks
    assert pack.assessment_chunks[0].chunk_id == "c06"


def test_renderer_timing_estimate_formula(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    pack = render_study_pack(archive, weeks=[1], include_exercises=True)
    # 400 words -> 8 min raw -> rounds to 10 min.
    c2 = next(c for c in pack.chunks if c.chunk_id == "c02")
    assert c2.estimated_minutes == 10
    # 600 words -> 12 min raw -> rounds to 10 min.
    c3 = next(c for c in pack.chunks if c.chunk_id == "c03")
    assert c3.estimated_minutes == 10
    # 250 words -> 5 min raw -> 5 min.
    c5 = next(c for c in pack.chunks if c.chunk_id == "c05")
    assert c5.estimated_minutes == 5


def test_renderer_timing_capped_at_30_minutes(tmp_path):
    archive = tmp_path / "big"
    archive.mkdir()
    _write_json(
        archive / "corpus" / "chunks.json",
        [
            _build_chunk(
                "huge",
                chunk_type="explanation",
                module_id="week_01_content_01",
                resource_type="page",
                title="Long",
                text="x",
                word_count=10_000,
            )
        ],
    )
    pack = render_study_pack(archive, weeks=[1])
    assert pack.chunks[0].estimated_minutes == 30


def test_renderer_aggregates_source_references(tmp_path):
    archive = tmp_path / "src"
    archive.mkdir()
    _write_json(
        archive / "corpus" / "chunks.json",
        [
            _build_chunk(
                "ov",
                chunk_type="overview",
                module_id="week_01_overview",
                resource_type="overview",
                title="OV",
                text="ov",
                word_count=50,
                source_refs=[
                    {"sourceId": "dart:foo#s1", "role": "primary"},
                    {"sourceId": "dart:bar#s9", "role": "contributing"},
                ],
            ),
            _build_chunk(
                "p1",
                chunk_type="explanation",
                module_id="week_01_content_01",
                resource_type="page",
                title="C1",
                text="p1",
                word_count=50,
                source_refs=[
                    # Duplicate of dart:foo#s1 — should dedupe.
                    {"sourceId": "dart:foo#s1", "role": "secondary"},
                    {"sourceId": "dart:baz#s2", "role": "contributing"},
                ],
            ),
        ],
    )
    pack = render_study_pack(archive, weeks=[1], lesson_plan=True)
    ids = [r["sourceId"] for r in pack.aggregated_source_references]
    assert ids == ["dart:bar#s9", "dart:baz#s2", "dart:foo#s1"]
    assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------- #
# Format renderer tests                                                  #
# ---------------------------------------------------------------------- #


def test_render_markdown_has_h1_h2_and_chunks(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    pack = render_study_pack(archive, weeks=[1])
    md = render_markdown(pack)
    # H1 = course code + week.
    assert md.startswith("# TEST_101: Week 1 Study Pack")
    # H2 buckets present (at least Overview).
    assert "## Overview" in md
    assert "## Core Content" in md
    assert "## Summary" in md
    # First chunk title is rendered as H3.
    assert "### Week 1 Overview" in md


def test_render_markdown_lesson_plan_adds_objective_table(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    pack = render_study_pack(
        archive,
        weeks=[1],
        include_self_check=True,
        lesson_plan=True,
    )
    md = render_markdown(pack)
    assert "Lesson Plan" in md
    assert "## Learning Objectives" in md
    # Markdown table header.
    assert "| ID | Bloom | Statement |" in md
    assert "TO-01" in md
    # Per-chunk timing meta line.
    assert "min" in md
    # Resources / assessment items section appear in lesson-plan mode.
    assert "## Assessment Items" in md


def test_render_markdown_exercise_uses_code_fence(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    pack = render_study_pack(archive, weeks=[1], include_exercises=True)
    md = render_markdown(pack)
    # Code fence around the exercise body.
    assert "```\nExercise body." in md


def test_render_markdown_self_check_uses_blockquote_callout(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    pack = render_study_pack(archive, weeks=[1], include_self_check=True)
    md = render_markdown(pack)
    # Self-check items rendered as markdown blockquote callouts.
    assert "> Question. Answer." in md


def test_render_html_parses_cleanly(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    pack = render_study_pack(
        archive,
        weeks=[1],
        include_exercises=True,
        include_self_check=True,
        lesson_plan=True,
    )
    html = render_html(pack)
    # html5 doctype + tags present.
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>TEST_101" in html
    assert "<h1>" in html
    assert "<h2>Overview</h2>" in html
    assert "<h2>Core Content</h2>" in html
    assert 'class="exercise-block"' in html
    assert 'class="callout-self-check"' in html
    # Validate parses cleanly with html.parser.
    parser = _CountingHTMLParser()
    parser.feed(html)
    parser.close()
    assert not parser.errors
    assert parser.tag_counts.get("h1", 0) == 1


class _CountingHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tag_counts: Dict[str, int] = {}
        self.errors: List[str] = []

    def handle_starttag(self, tag: str, attrs):  # noqa: ANN001
        self.tag_counts[tag] = self.tag_counts.get(tag, 0) + 1

    def error(self, message: str) -> None:  # type: ignore[override]
        self.errors.append(message)


def test_render_json_has_expected_keys(tmp_path):
    archive = _build_synthetic_archive(tmp_path, "demo")
    pack = render_study_pack(
        archive,
        weeks=[1],
        include_self_check=True,
        lesson_plan=True,
    )
    payload = json.loads(render_json(pack))
    assert payload["course_code"] == "TEST_101"
    assert payload["weeks"] == [1]
    assert payload["lesson_plan_mode"] is True
    assert isinstance(payload["chunks"], list)
    assert payload["chunks"][0]["chunk_id"] == "c01"
    assert "objectives" in payload
    assert "resources" in payload


# ---------------------------------------------------------------------- #
# CLI tests                                                              #
# ---------------------------------------------------------------------- #


def test_cli_help_lists_generate_study_pack():
    runner = CliRunner()
    result = runner.invoke(libv2_group, ["generate-study-pack", "--help"])
    assert result.exit_code == 0, result.output
    assert "--slug" in result.output
    assert "--week" in result.output
    assert "--include-exercises" in result.output
    assert "--include-self-check" in result.output
    assert "--lesson-plan" in result.output
    assert "--format" in result.output


def test_cli_default_md_output(tmp_path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_synthetic_archive(courses_root, "demo")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-study-pack",
            "--slug",
            "demo",
            "--week",
            "1",
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "# TEST_101: Week 1 Study Pack" in result.output


def test_cli_html_output_parses(tmp_path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_synthetic_archive(courses_root, "demo")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-study-pack",
            "--slug",
            "demo",
            "--week",
            "1",
            "--format",
            "html",
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith("<!DOCTYPE html>")
    parser = _CountingHTMLParser()
    parser.feed(result.output)
    parser.close()
    assert not parser.errors


def test_cli_json_output_is_valid_json(tmp_path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_synthetic_archive(courses_root, "demo")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-study-pack",
            "--slug",
            "demo",
            "--week",
            "1",
            "--format",
            "json",
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["course_code"] == "TEST_101"


def test_cli_no_chunks_found_exits_with_clear_error(tmp_path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_synthetic_archive(courses_root, "demo")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-study-pack",
            "--slug",
            "demo",
            "--week",
            "99",
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code == 1
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "no chunks found" in combined.lower()


def test_cli_writes_to_output_file(tmp_path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_synthetic_archive(courses_root, "demo")
    target = tmp_path / "out" / "pack.md"

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-study-pack",
            "--slug",
            "demo",
            "--week",
            "1",
            "--output",
            str(target),
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert target.exists()
    body = target.read_text(encoding="utf-8")
    assert "# TEST_101: Week 1 Study Pack" in body


def test_cli_include_exercises_adds_chunks(tmp_path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_synthetic_archive(courses_root, "demo")
    runner = CliRunner()

    base = runner.invoke(
        libv2_group,
        [
            "generate-study-pack",
            "--slug",
            "demo",
            "--week",
            "1",
            "--format",
            "json",
            "--courses-root",
            str(courses_root),
        ],
    )
    assert base.exit_code == 0
    base_n = json.loads(base.output)["total_chunks"]

    plus = runner.invoke(
        libv2_group,
        [
            "generate-study-pack",
            "--slug",
            "demo",
            "--week",
            "1",
            "--include-exercises",
            "--format",
            "json",
            "--courses-root",
            str(courses_root),
        ],
    )
    assert plus.exit_code == 0
    plus_n = json.loads(plus.output)["total_chunks"]
    assert plus_n == base_n + 1
    types = [c["chunk_type"] for c in json.loads(plus.output)["chunks"]]
    assert "exercise" in types


# ---------------------------------------------------------------------- #
# Integration test against real archive                                  #
# ---------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _real_archive_available(),
    reason="rdf-shacl-550 archive not available",
)
def test_real_archive_week_7_emits_18_chunks_in_canonical_order():
    pack = render_study_pack(
        REAL_ARCHIVE,
        weeks=[7],
        include_exercises=True,
        include_self_check=True,
    )
    assert len(pack.chunks) == 18, [c.chunk_id for c in pack.chunks]

    # Bucket sequence: overview -> content -> application -> self_check -> summary.
    # Week 7 in the canonical archive has no exercise chunks, so EXERCISES
    # is skipped between APPLICATION and SELF_CHECK.
    bucket_order = []
    for c in pack.chunks:
        if not bucket_order or bucket_order[-1] != c.bucket:
            bucket_order.append(c.bucket)
    # 0=overview, 1=content, 2=application, 4=self_check, 5=summary.
    assert bucket_order == [0, 1, 2, 4, 5], bucket_order

    # Content_NN ordinals are monotonically nondecreasing in the content bucket.
    content_chunks = [c for c in pack.chunks if c.bucket == 1]
    ordinals = [c.content_ordinal for c in content_chunks]
    assert ordinals == sorted(ordinals)
    assert ordinals[0] == 1
    assert ordinals[-1] == 6


@pytest.mark.skipif(
    not _real_archive_available(),
    reason="rdf-shacl-550 archive not available",
)
def test_real_archive_week_7_lesson_plan_has_timings_and_objective_table():
    pack = render_study_pack(
        REAL_ARCHIVE,
        weeks=[7],
        include_exercises=True,
        include_self_check=True,
        lesson_plan=True,
    )
    # Every chunk has a non-negative timing estimate, capped at 30 min.
    for c in pack.chunks:
        assert 0 <= c.estimated_minutes <= 30
    # At least the overview chunk has a non-zero estimate.
    assert any(c.estimated_minutes > 0 for c in pack.chunks)
    # Objectives table includes TO-04 (the canonical week-7 terminal LO).
    ids = {str(o.get("id")).lower() for o in pack.objectives_referenced}
    assert "to-04" in ids
    # All collected objectives (other than unresolved stubs) carry statements.
    for obj in pack.objectives_referenced:
        if obj.get("_unresolved"):
            continue
        assert obj.get("statement"), f"empty statement: {obj}"


@pytest.mark.skipif(
    not _real_archive_available(),
    reason="rdf-shacl-550 archive not available",
)
def test_real_archive_week_7_md_format_contains_expected_landmarks():
    pack = render_study_pack(
        REAL_ARCHIVE,
        weeks=[7],
        include_exercises=True,
        include_self_check=True,
    )
    md = render_markdown(pack)
    assert md.startswith("# RDF_SHACL_550: Week 7 Study Pack")
    assert "## Overview" in md
    assert "## Core Content" in md
    assert "## Application" in md
    assert "## Self-Check" in md
    assert "## Summary" in md


@pytest.mark.skipif(
    not _real_archive_available(),
    reason="rdf-shacl-550 archive not available",
)
def test_real_archive_week_7_html_format_parses_cleanly():
    pack = render_study_pack(
        REAL_ARCHIVE,
        weeks=[7],
        include_exercises=True,
        include_self_check=True,
    )
    html = render_html(pack)
    parser = _CountingHTMLParser()
    parser.feed(html)
    parser.close()
    assert not parser.errors
    assert parser.tag_counts.get("h1", 0) == 1
    # Five distinct H2 buckets show up for week 7 (overview, content,
    # application, self-check, summary).
    assert parser.tag_counts.get("h2", 0) >= 5


@pytest.mark.skipif(
    not _real_archive_available(),
    reason="rdf-shacl-550 archive not available",
)
def test_real_archive_week_99_no_chunks_clear_error():
    with pytest.raises(StudyPackError) as excinfo:
        render_study_pack(REAL_ARCHIVE, weeks=[99])
    assert "no chunks found" in str(excinfo.value).lower()
