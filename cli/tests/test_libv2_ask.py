"""Tests for ``ed4all libv2 ask`` (Wave 78 Worker C).

The CLI is a thin wrapper around
:func:`LibV2.tools.intent_router.dispatch`; the engine itself has
exhaustive coverage in ``LibV2/tests/test_intent_router.py``. These
tests exercise the *Click surface*:

* required + optional flags wire up correctly,
* ``--show-routing`` emits the entity envelope,
* ``--format json`` produces a parseable canonical envelope,
* ``--format text`` emits an intent tag + result preview block,
* unknown slug doesn't crash the command (the dispatcher fails-soft).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest
from click.testing import CliRunner

from cli.commands.libv2_ask import ask_command
from lib.paths import LIBV2_PATH


LIVE_SLUG = os.environ.get("ED4ALL_INTENT_ROUTER_FIXTURE_SLUG")
LIVE_ARCHIVE = (
    LIBV2_PATH / "courses" / LIVE_SLUG if LIVE_SLUG else None
)


def _run(args: List[str]):
    return CliRunner().invoke(ask_command, args)


@pytest.fixture(scope="module")
def live_archive_present() -> bool:
    return (
        LIVE_ARCHIVE is not None
        and (LIVE_ARCHIVE / "corpus" / "chunks.jsonl").is_file()
    )


# ---------------------------------------------------------------------- #
# Synthetic archive (mirrors the engine fixture)                         #
# ---------------------------------------------------------------------- #


def _make_synthetic_archive(courses_root: Path, slug: str) -> Path:
    root = courses_root / slug
    (root / "corpus").mkdir(parents=True)
    chunks = [
        {
            "id": f"c{i:02d}",
            "chunk_type": ct,
            "difficulty": diff,
            "bloom_level": bl,
            "text": text,
            "word_count": wc,
            "learning_outcome_refs": refs,
            "source": {"module_id": mod},
        }
        for i, (ct, diff, bl, text, wc, refs, mod) in enumerate(
            [
                ("explanation", "foundational", "remember",
                 "RDF triples are subject-predicate-object statements.",
                 100, ["co-01"], "week_01_overview"),
                ("example", "intermediate", "apply",
                 "Example with sh:minCount usage in SHACL.",
                 150, ["co-02"], "week_03_content"),
                ("exercise", "intermediate", "apply",
                 "Exercise: write a SHACL shape.",
                 200, ["co-16"], "week_07_application"),
                ("assessment_item", "advanced", "evaluate",
                 "Question on SHACL features.",
                 50, ["to-04"], "week_08_assessment"),
            ],
            start=1,
        )
    ]
    with (root / "corpus" / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    objectives = {
        "terminal_outcomes": [{"id": "to-04"}],
        "component_objectives": [
            {"id": "co-16", "parent_terminal": "to-04"},
        ],
    }
    (root / "objectives.json").write_text(json.dumps(objectives), encoding="utf-8")
    return root


# ---------------------------------------------------------------------- #
# Smoke tests                                                             #
# ---------------------------------------------------------------------- #


def test_ask_help_lists_required_flags():
    result = _run(["--help"])
    assert result.exit_code == 0
    for flag in ("--slug", "--query", "--top-k", "--show-routing", "--format"):
        assert flag in result.output


def test_ask_missing_slug_errors():
    result = _run(["--query", "anything"])
    assert result.exit_code != 0
    assert "--slug" in result.output


def test_ask_missing_query_errors():
    result = _run(["--slug", "demo"])
    assert result.exit_code != 0
    assert "--query" in result.output


def test_ask_negative_top_k_rejected():
    result = _run(["--slug", "demo", "--query", "x", "--top-k", "-1"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------- #
# JSON format envelope                                                    #
# ---------------------------------------------------------------------- #


def test_ask_json_envelope_against_synthetic(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run([
        "--slug", "demo",
        "--query", "Which chunks assess to-04?",
        "--courses-root", str(courses_root),
        "--format", "json",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["intent_class"] == "objective_lookup"
    assert payload["slug"] == "demo"
    assert payload["query"] == "Which chunks assess to-04?"
    assert "results" in payload
    assert "entities" in payload
    assert payload["entities"]["objective_ids"] == ["to-04"]


def test_ask_json_default_omits_marker_flags(tmp_path: Path):
    """Without --show-routing, the JSON envelope omits the bulky cue
    flags / residual to keep default output compact."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run([
        "--slug", "demo",
        "--query", "Which chunks assess to-04?",
        "--courses-root", str(courses_root),
        "--format", "json",
    ])
    payload = json.loads(result.output)
    # Structural ID fields stay; cue flags + residual stripped.
    assert "objective_ids" in payload["entities"]
    assert "residual_text" not in payload["entities"]
    assert "has_misconception_marker" not in payload["entities"]


def test_ask_json_show_routing_includes_full_entities(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run([
        "--slug", "demo",
        "--query", "What misconceptions exist about RDF?",
        "--courses-root", str(courses_root),
        "--show-routing",
        "--format", "json",
    ])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    # With --show-routing, the full entity envelope is preserved.
    assert "residual_text" in payload["entities"]
    assert "has_misconception_marker" in payload["entities"]
    assert payload["entities"]["has_misconception_marker"] is True


# ---------------------------------------------------------------------- #
# Text format human-readable rendering                                    #
# ---------------------------------------------------------------------- #


def test_ask_text_format_includes_intent_tag(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run([
        "--slug", "demo",
        "--query", "Which chunks assess to-04?",
        "--courses-root", str(courses_root),
        "--format", "text",
    ])
    assert result.exit_code == 0, result.output
    assert "[OBJECTIVE]" in result.output
    assert "intent=objective_lookup" in result.output
    assert "route:" in result.output


def test_ask_text_show_routing_emits_entities(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run([
        "--slug", "demo",
        "--query", "Show me apply-level exercises for week 7",
        "--courses-root", str(courses_root),
        "--show-routing",
        "--format", "text",
    ])
    assert result.exit_code == 0, result.output
    assert "[FACETED]" in result.output
    # The entities block should appear when --show-routing is set.
    assert "entities:" in result.output
    assert "weeks:" in result.output
    assert "chunk_types:" in result.output


def test_ask_text_show_routing_off_omits_entities(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run([
        "--slug", "demo",
        "--query", "Show me apply-level exercises for week 7",
        "--courses-root", str(courses_root),
        "--format", "text",
    ])
    assert result.exit_code == 0
    # No --show-routing -> no entities: block.
    assert "entities:" not in result.output


def test_ask_text_misconception_renders_correction(tmp_path: Path, monkeypatch):
    """Text format for misconception_query should render
    Misconception: + Correction: lines for each result.
    Use monkeypatch so the test doesn't depend on the live archive's
    misconception inventory."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")

    # Patch tutoring_tools.match_misconception so the dispatcher
    # gets a deterministic result for the synthetic archive.
    import MCP.tools.tutoring_tools as tt_mod

    def fake_match(slug, text, top_k=5):
        return [{
            "misconception": "RDF triples are like SQL rows",
            "correction": "RDF is a graph data model, not relational",
            "chunk_id": "c01",
            "source_references": [],
            "concept_tags": [],
            "score": 0.9,
            "backend": "jaccard",
        }]

    monkeypatch.setattr(tt_mod, "match_misconception", fake_match)

    result = _run([
        "--slug", "demo",
        "--query", "What misconceptions exist about RDF?",
        "--courses-root", str(courses_root),
        "--format", "text",
    ])
    assert result.exit_code == 0, result.output
    assert "[MISCONCEPTION]" in result.output
    assert "Misconception:" in result.output
    assert "Correction:" in result.output


# ---------------------------------------------------------------------- #
# Error / edge cases                                                      #
# ---------------------------------------------------------------------- #


def test_ask_unknown_slug_does_not_crash(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    result = _run([
        "--slug", "no-such-slug",
        "--query", "anything?",
        "--courses-root", str(courses_root),
        "--format", "json",
    ])
    # Dispatcher fails-soft; CLI exits 0 with empty results.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["results"] == []


def test_ask_top_k_zero_returns_empty(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run([
        "--slug", "demo",
        "--query", "Which chunks assess to-04?",
        "--courses-root", str(courses_root),
        "--top-k", "0",
        "--format", "json",
    ])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["results"] == []


# ---------------------------------------------------------------------- #
# Live-archive smoke (skipped without the fixture)                       #
# ---------------------------------------------------------------------- #


def test_ask_live_concept_query_returns_results(live_archive_present):
    if not live_archive_present:
        pytest.skip("intent-router fixture archive not configured")
    result = _run([
        "--slug", LIVE_SLUG,
        "--query", "How does sh:minCount work?",
        "--top-k", "5",
        "--format", "json",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["intent_class"] == "concept_query"
    assert len(payload["results"]) >= 1


# ---------------------------------------------------------------------- #
# --library-wide (W4.2): opt-in grounded ask unioned across the catalog   #
# ---------------------------------------------------------------------- #


class _FakeGroundedAnswer:
    """Stand-in for a library-wide GroundedAnswer (only ``to_dict`` is used)."""

    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


def _fake_library_payload():
    return {
        "status": "answered",
        "query": "q?",
        "course_slug": "course-a",
        "engine": "lexical",
        "answer_text": "Unioned answer across courses.",
        "citations": [
            {"chunk_id": "course-a-c01", "course_slug": "course-a",
             "section_heading": "Intro A", "text_quote": "body from course-a"},
            {"chunk_id": "course-b-c01", "course_slug": "course-b",
             "section_heading": "Intro B", "text_quote": "body from course-b"},
        ],
    }


def test_ask_help_lists_library_wide_flag():
    result = _run(["--help"])
    assert result.exit_code == 0
    assert "--library-wide" in result.output


def test_ask_library_wide_routes_to_answer_library_question(tmp_path, monkeypatch):
    """--library-wide routes to answer_library_question with the resolved set."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "course-a")
    _make_synthetic_archive(courses_root, "course-b")

    seen = {}

    def _spy(libv2_root, slug, query, **kwargs):
        seen["libv2_root"] = libv2_root
        seen["slug"] = slug
        seen["query"] = query
        seen["kwargs"] = kwargs
        return _FakeGroundedAnswer(_fake_library_payload())

    monkeypatch.setattr("cli.commands.libv2_ask.answer_library_question", _spy)

    result = _run([
        "--slug", "course-a",
        "--query", "q?",
        "--courses-root", str(courses_root),
        "--library-wide",
        "--format", "json",
    ])
    assert result.exit_code == 0, result.output
    # Routed to the library-wide grounded-answer path (explicit flag wins).
    assert seen["slug"] == "course-a"
    assert seen["query"] == "q?"
    assert seen["kwargs"]["library_wide"] is True
    # The resolved catalog/filesystem course set is threaded through: home
    # first, both courses present (list_library_courses over the fs fallback).
    resolved = list(seen["kwargs"]["course_slugs"])
    assert resolved[0] == "course-a"
    assert set(resolved) == {"course-a", "course-b"}
    # Output is the grounded-answer envelope (NOT the intent-router envelope).
    out = json.loads(result.output)
    assert out["status"] == "answered"
    assert {c["course_slug"] for c in out["citations"]} == {"course-a", "course-b"}


def test_ask_library_wide_text_renders_per_course_provenance(tmp_path, monkeypatch):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "course-a")
    _make_synthetic_archive(courses_root, "course-b")

    monkeypatch.setattr(
        "cli.commands.libv2_ask.answer_library_question",
        lambda *a, **k: _FakeGroundedAnswer(_fake_library_payload()),
    )

    result = _run([
        "--slug", "course-a",
        "--query", "q?",
        "--courses-root", str(courses_root),
        "--library-wide",
        "--format", "text",
    ])
    assert result.exit_code == 0, result.output
    assert "[LIBRARY-WIDE]" in result.output
    assert "Unioned answer across courses." in result.output
    # Per-course provenance is surfaced for each citation.
    assert "course=course-a" in result.output
    assert "course=course-b" in result.output


def test_ask_without_library_wide_uses_single_course_dispatch(tmp_path, monkeypatch):
    """Default (flag off) is byte-identical: intent-routed dispatch, never lw."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")

    def _must_not_run(*a, **k):  # pragma: no cover - asserts non-invocation
        raise AssertionError("answer_library_question ran with the flag off")

    monkeypatch.setattr("cli.commands.libv2_ask.answer_library_question", _must_not_run)

    result = _run([
        "--slug", "demo",
        "--query", "Which chunks assess to-04?",
        "--courses-root", str(courses_root),
        "--format", "json",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # Byte-identical intent-router envelope (see test_ask_json_envelope_*).
    assert payload["intent_class"] == "objective_lookup"
    assert payload["slug"] == "demo"
    assert payload["entities"]["objective_ids"] == ["to-04"]
