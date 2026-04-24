"""Wave 70 — ``hierarchy_level`` filter on ChunkFilter + CLI.

Unlike ``cognitive_domain`` (on the chunk directly), ``hierarchy_level``
lives on the LO — we look it up via chunk ``learning_outcome_refs[]``
against ``course.json``. Covers:

* ``_matches_filter`` correctly resolves hierarchy via outcomes lookup.
* Chunks whose LOs all have the wrong level → rejected.
* Chunks without LO refs → rejected (can't attest).
* Filter with no outcomes map → rejected (conservative).
* End-to-end via ``retrieve_chunks`` on a tiny fixture corpus.
* CLI wiring fires the filter through.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest import mock

import pytest
from click.testing import CliRunner

from LibV2.tools.libv2.cli import main
from LibV2.tools.libv2.retriever import (
    ChunkFilter,
    _matches_filter,
    retrieve_chunks,
    stream_chunks_from_course,
)


# -------------------------------------------------------------------- #
# Unit: _matches_filter with explicit outcomes map
# -------------------------------------------------------------------- #


def _outcomes_map() -> dict:
    return {
        "to-01": {"id": "TO-01", "hierarchy_level": "terminal"},
        "co-01": {"id": "CO-01", "hierarchy_level": "chapter"},
        "co-02": {"id": "CO-02", "hierarchy_level": "chapter"},
    }


class TestHierarchyLevelFilter:
    def test_terminal_match(self):
        chunk = {"learning_outcome_refs": ["TO-01"], "source": {}}
        assert _matches_filter(
            chunk,
            ChunkFilter(hierarchy_level="terminal"),
            outcomes_by_id=_outcomes_map(),
        )

    def test_chapter_match(self):
        chunk = {"learning_outcome_refs": ["CO-01"], "source": {}}
        assert _matches_filter(
            chunk,
            ChunkFilter(hierarchy_level="chapter"),
            outcomes_by_id=_outcomes_map(),
        )

    def test_mismatch(self):
        """A chunk whose LO is chapter but filter wants terminal is rejected."""
        chunk = {"learning_outcome_refs": ["CO-01"], "source": {}}
        assert not _matches_filter(
            chunk,
            ChunkFilter(hierarchy_level="terminal"),
            outcomes_by_id=_outcomes_map(),
        )

    def test_mixed_refs_match_on_any(self):
        """A chunk linking to both terminal + chapter LOs should match
        either filter (it contributes to both)."""
        chunk = {"learning_outcome_refs": ["TO-01", "CO-01"], "source": {}}
        assert _matches_filter(
            chunk,
            ChunkFilter(hierarchy_level="terminal"),
            outcomes_by_id=_outcomes_map(),
        )
        assert _matches_filter(
            chunk,
            ChunkFilter(hierarchy_level="chapter"),
            outcomes_by_id=_outcomes_map(),
        )

    def test_no_refs_rejected(self):
        """A chunk with no learning_outcome_refs can't be attested —
        reject rather than let it through."""
        chunk = {"source": {}}
        assert not _matches_filter(
            chunk,
            ChunkFilter(hierarchy_level="terminal"),
            outcomes_by_id=_outcomes_map(),
        )

    def test_no_outcomes_map_rejected(self):
        """No lookup table → conservative reject."""
        chunk = {"learning_outcome_refs": ["TO-01"], "source": {}}
        assert not _matches_filter(
            chunk,
            ChunkFilter(hierarchy_level="terminal"),
            outcomes_by_id=None,
        )
        # Empty dict is the same as None here.
        assert not _matches_filter(
            chunk,
            ChunkFilter(hierarchy_level="terminal"),
            outcomes_by_id={},
        )

    def test_case_insensitive_ref_resolution(self):
        """Refs may be emitted upper or lower case per TRAINFORGE_PRESERVE_LO_CASE
        behavior; the lookup must accept both."""
        # Outcomes are keyed lowercase; refs come through upper case.
        chunk = {"learning_outcome_refs": ["TO-01"], "source": {}}
        assert _matches_filter(
            chunk,
            ChunkFilter(hierarchy_level="terminal"),
            outcomes_by_id=_outcomes_map(),
        )


# -------------------------------------------------------------------- #
# End-to-end: stream_chunks_from_course applies hierarchy filter
# -------------------------------------------------------------------- #


def _make_course(tmp_path: Path, slug: str, chunks: List[dict], outcomes: List[dict]) -> Path:
    course_dir = tmp_path / "courses" / slug
    (course_dir / "corpus").mkdir(parents=True)
    with open(course_dir / "corpus" / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    with open(course_dir / "manifest.json", "w") as f:
        json.dump({"classification": {"primary_domain": "test-domain"}}, f)
    with open(course_dir / "course.json", "w") as f:
        json.dump({"learning_outcomes": outcomes}, f)
    return course_dir


def test_stream_chunks_from_course_applies_hierarchy_filter(tmp_path):
    chunks = [
        {
            "id": "c1",
            "text": "terminal level chunk",
            "learning_outcome_refs": ["TO-01"],
        },
        {
            "id": "c2",
            "text": "chapter level chunk",
            "learning_outcome_refs": ["CO-01"],
        },
    ]
    outcomes = [
        {"id": "TO-01", "hierarchy_level": "terminal"},
        {"id": "CO-01", "hierarchy_level": "chapter"},
    ]
    course_dir = _make_course(tmp_path, "demo", chunks, outcomes)

    # terminal filter → only c1
    got = list(
        stream_chunks_from_course(
            course_dir,
            "demo",
            "test-domain",
            ChunkFilter(hierarchy_level="terminal"),
        )
    )
    assert [c["id"] for c in got] == ["c1"]

    # chapter filter → only c2
    got = list(
        stream_chunks_from_course(
            course_dir,
            "demo",
            "test-domain",
            ChunkFilter(hierarchy_level="chapter"),
        )
    )
    assert [c["id"] for c in got] == ["c2"]


# -------------------------------------------------------------------- #
# CLI wiring
# -------------------------------------------------------------------- #


def _make_min_repo(tmp_path: Path) -> Path:
    (tmp_path / "courses").mkdir()
    (tmp_path / "catalog").mkdir()
    return tmp_path


def test_cli_hierarchy_level_flag_threads_to_retrieve():
    runner = CliRunner()
    captured: dict = {}

    def _fake_retrieve_chunks(**kwargs):
        captured.update(kwargs)
        return []

    with runner.isolated_filesystem() as fs:
        repo = _make_min_repo(Path(fs))
        with mock.patch(
            "LibV2.tools.libv2.retriever.retrieve_chunks",
            side_effect=_fake_retrieve_chunks,
        ):
            result = runner.invoke(
                main,
                [
                    "--repo",
                    str(repo),
                    "retrieve",
                    "query",
                    "--hierarchy-level",
                    "terminal",
                ],
            )

    assert result.exit_code == 0, result.output
    assert captured.get("hierarchy_level") == "terminal"


def test_cli_hierarchy_level_rejects_out_of_enum():
    """click.Choice guards the enum — invalid values exit != 0."""
    runner = CliRunner()
    with runner.isolated_filesystem() as fs:
        repo = _make_min_repo(Path(fs))
        result = runner.invoke(
            main,
            [
                "--repo",
                str(repo),
                "retrieve",
                "query",
                "--hierarchy-level",
                "bogus",
            ],
        )
    assert result.exit_code != 0
    assert "bogus" in result.output or "Invalid" in result.output
