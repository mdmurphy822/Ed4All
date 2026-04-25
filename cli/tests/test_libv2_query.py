"""Tests for ``ed4all libv2 query`` (Wave 77 Worker β).

Most assertions are made against the live ``rdf-shacl-550-rdf-shacl-550``
archive that ships in-repo: it's the canonical KG-quality fixture and
the counts referenced here were independently verified by ChatGPT's
Wave 76 review (``--week 7`` → 18; ``--outcome to-04`` → ≥69;
``--outcome co-18`` → ≥44).

A small handful of structural tests use synthetic fixtures so the suite
isn't tightly coupled to the live archive's exact byte content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
from click.testing import CliRunner

from cli.commands.libv2_query import query_command
from lib.paths import LIBV2_PATH


LIVE_SLUG = "rdf-shacl-550-rdf-shacl-550"
LIVE_ARCHIVE = LIBV2_PATH / "courses" / LIVE_SLUG


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def _run(args: List[str]) -> "click.testing.Result":
    return CliRunner().invoke(query_command, args)


@pytest.fixture(scope="module")
def live_archive_present() -> bool:
    """Skip live-archive tests if the fixture isn't checked in."""
    return (LIVE_ARCHIVE / "corpus" / "chunks.jsonl").is_file()


# ---------------------------------------------------------------------- #
# Synthetic-archive fixture (for structural tests)                        #
# ---------------------------------------------------------------------- #


def _make_synthetic_archive(courses_root: Path, slug: str) -> Path:
    """Tiny archive with predictable filter outcomes."""
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
                (
                    "explanation",
                    "foundational",
                    "remember",
                    "Intro chunk about RDF.",
                    100,
                    ["co-01"],
                    "week_01_overview",
                ),
                (
                    "example",
                    "intermediate",
                    "apply",
                    "Example with sh:minCount usage.",
                    150,
                    ["co-02"],
                    "week_03_content_01",
                ),
                (
                    "exercise",
                    "intermediate",
                    "apply",
                    "Exercise: write a SHACL shape.",
                    200,
                    ["co-16"],
                    "week_07_application",
                ),
                (
                    "exercise",
                    "advanced",
                    "analyze",
                    "Analyze the constraint violation.",
                    250,
                    ["co-17"],
                    "week_07_application",
                ),
                (
                    "assessment_item",
                    "advanced",
                    "evaluate",
                    "Question on SHACL features.",
                    50,
                    ["to-04"],
                    "week_08_assessment",
                ),
            ],
            start=1,
        )
    ]
    with (root / "corpus" / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    objectives = {
        "terminal_outcomes": [
            {"id": "to-01"},
            {"id": "to-04"},
        ],
        "component_objectives": [
            {"id": "co-01", "parent_terminal": "to-01"},
            {"id": "co-02", "parent_terminal": "to-01"},
            {"id": "co-16", "parent_terminal": "to-04"},
            {"id": "co-17", "parent_terminal": "to-04"},
        ],
    }
    (root / "objectives.json").write_text(json.dumps(objectives), encoding="utf-8")
    return root


# ---------------------------------------------------------------------- #
# Help / smoke tests                                                      #
# ---------------------------------------------------------------------- #


def test_query_help_lists_filters():
    result = _run(["--help"])
    assert result.exit_code == 0
    for flag in (
        "--slug",
        "--chunk-type",
        "--bloom",
        "--difficulty",
        "--week",
        "--module",
        "--outcome",
        "--text",
        "--limit",
        "--offset",
        "--sort",
        "--format",
    ):
        assert flag in result.output


# ---------------------------------------------------------------------- #
# Live-archive structural assertions                                      #
# ---------------------------------------------------------------------- #


def test_chunk_type_example_intermediate_returns_min_10(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    result = _run(
        [
            "--slug",
            LIVE_SLUG,
            "--chunk-type",
            "example",
            "--difficulty",
            "intermediate",
            "--format",
            "count",
        ]
    )
    assert result.exit_code == 0, result.output
    count = int(result.output.strip())
    assert count >= 10, f"expected >=10 example/intermediate chunks, got {count}"


def test_week_7_returns_18_chunks(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    result = _run(
        ["--slug", LIVE_SLUG, "--week", "7", "--format", "count"]
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "18"


def test_outcome_to_04_rolls_up_to_at_least_69(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    result = _run(
        ["--slug", LIVE_SLUG, "--outcome", "to-04", "--format", "count"]
    )
    assert result.exit_code == 0, result.output
    count = int(result.output.strip())
    assert count >= 69, f"expected >=69 chunks under to-04 rollup, got {count}"


def test_outcome_co_18_returns_at_least_44(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    result = _run(
        ["--slug", LIVE_SLUG, "--outcome", "co-18", "--format", "count"]
    )
    assert result.exit_code == 0, result.output
    count = int(result.output.strip())
    assert count >= 44, f"expected >=44 chunks tagged co-18, got {count}"


def test_text_filter_finds_sh_mincount(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    result = _run(
        [
            "--slug",
            LIVE_SLUG,
            "--text",
            "sh:minCount",
            "--format",
            "count",
        ]
    )
    assert result.exit_code == 0, result.output
    count = int(result.output.strip())
    assert count > 0, "expected at least one chunk containing 'sh:minCount'"


def test_text_filter_includes_match_in_json(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    result = _run(
        [
            "--slug",
            LIVE_SLUG,
            "--text",
            "sh:minCount",
            "--limit",
            "3",
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["returned"] >= 1
    for chunk in payload["chunks"]:
        assert "sh:mincount" in (chunk.get("text") or "").lower()


def test_bloom_apply_analyze_with_exercise_composes(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    # Composition: every returned chunk must satisfy ALL filters.
    result = _run(
        [
            "--slug",
            LIVE_SLUG,
            "--bloom",
            "apply,analyze",
            "--chunk-type",
            "exercise",
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total_matches"] >= 1
    for chunk in payload["chunks"]:
        assert chunk.get("chunk_type") == "exercise"
        assert chunk.get("bloom_level") in {"apply", "analyze"}


def test_week_range_with_limit_caps_results(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    result = _run(
        [
            "--slug",
            LIVE_SLUG,
            "--week",
            "1-3",
            "--limit",
            "5",
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["returned"] <= 5
    for chunk in payload["chunks"]:
        module_id = (chunk.get("source") or {}).get("module_id") or ""
        # week_01_… / week_02_… / week_03_…
        assert module_id.startswith(("week_01", "week_02", "week_03")), module_id


def test_format_count_returns_integer(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    result = _run(
        [
            "--slug",
            LIVE_SLUG,
            "--chunk-type",
            "exercise",
            "--format",
            "count",
        ]
    )
    assert result.exit_code == 0, result.output
    # Should be just an integer, no decoration.
    assert result.output.strip().isdigit()


def test_empty_filter_returns_all_219_chunks(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    result = _run(
        ["--slug", LIVE_SLUG, "--format", "count"]
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "219"


# ---------------------------------------------------------------------- #
# Error-path tests                                                        #
# ---------------------------------------------------------------------- #


def test_unknown_slug_raises_clear_error(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    runner = CliRunner(mix_stderr=True)
    result = runner.invoke(
        query_command,
        [
            "--slug",
            "no-such-course",
            "--courses-root",
            str(courses_root),
            "--format",
            "count",
        ],
    )
    assert result.exit_code != 0
    # Slug name must appear in the error so users know what failed.
    assert "no-such-course" in result.output


def test_invalid_chunk_type_value_rejected(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run(
        [
            "--slug",
            "demo",
            "--courses-root",
            str(courses_root),
            "--chunk-type",
            "bogus",
            "--format",
            "count",
        ]
    )
    assert result.exit_code != 0


def test_invalid_week_spec_rejected(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run(
        [
            "--slug",
            "demo",
            "--courses-root",
            str(courses_root),
            "--week",
            "10-3",
            "--format",
            "count",
        ]
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------- #
# Synthetic-archive composition tests                                     #
# ---------------------------------------------------------------------- #


def test_synthetic_to_rollup_includes_children(tmp_path: Path):
    """Querying ``to-04`` should match chunks tagged with co-16 / co-17 / to-04."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run(
        [
            "--slug",
            "demo",
            "--courses-root",
            str(courses_root),
            "--outcome",
            "to-04",
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # In the synthetic archive, to-04 + co-16 + co-17 = 3 chunks.
    assert payload["total_matches"] == 3
    assert "co-16" in payload["expanded_outcomes"]
    assert "co-17" in payload["expanded_outcomes"]
    assert "to-04" in payload["expanded_outcomes"]


def test_synthetic_table_format_renders_header(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run(
        [
            "--slug",
            "demo",
            "--courses-root",
            str(courses_root),
            "--format",
            "table",
        ]
    )
    assert result.exit_code == 0, result.output
    assert "ID" in result.output and "WK" in result.output
    assert "TYPE" in result.output
    assert "5 of 5 matches" in result.output


def test_synthetic_md_format_emits_numbered_list(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run(
        [
            "--slug",
            "demo",
            "--courses-root",
            str(courses_root),
            "--limit",
            "2",
            "--format",
            "md",
        ]
    )
    assert result.exit_code == 0, result.output
    assert "## 1." in result.output
    assert "## 2." in result.output
    assert "**bloom**" in result.output


def test_synthetic_offset_pagination(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run(
        [
            "--slug",
            "demo",
            "--courses-root",
            str(courses_root),
            "--offset",
            "2",
            "--limit",
            "2",
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["returned"] == 2
    assert payload["total_matches"] == 5  # All 5 still match.


def test_synthetic_module_filter_exact_match(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run(
        [
            "--slug",
            "demo",
            "--courses-root",
            str(courses_root),
            "--module",
            "week_07_application",
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total_matches"] == 2  # Two exercise chunks live there.
    for chunk in payload["chunks"]:
        assert (chunk.get("source") or {}).get("module_id") == "week_07_application"


def test_synthetic_sort_word_count_ascending(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    result = _run(
        [
            "--slug",
            "demo",
            "--courses-root",
            str(courses_root),
            "--sort",
            "word_count",
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    word_counts = [c.get("word_count") for c in payload["chunks"]]
    assert word_counts == sorted(word_counts)
