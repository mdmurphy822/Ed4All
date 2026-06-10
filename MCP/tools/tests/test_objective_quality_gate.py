"""Tests for the TRAINFORGE_OBJECTIVE_QUALITY_GATE behaviour flag and the
``_is_low_quality_objective`` helper in
:mod:`MCP.tools._content_gen_helpers`.

Pre-diagnosed bug: Path A (echo ``extracted_lo_statements``) of
``synthesize_objectives_from_topics`` early-returns and suppresses Path B
(per-week heading rollup) entirely. So a few scraped junk "Learning
Objectives" fragments collapse a whole course onto those garbage
objectives. The fix gates LO quality and MERGES Path A + Path B behind the
default-OFF ``TRAINFORGE_OBJECTIVE_QUALITY_GATE`` flag.

These tests pin both the byte-stable flag-OFF behaviour and the merged
flag-ON behaviour.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import _content_gen_helpers as _cgh  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixture builder (mirrors parse_dart_html_files topic shape)
# ---------------------------------------------------------------------- #


def _mk_topic(
    heading: str,
    *,
    source_file: str = "nvidia_course",
    dart_block_ids: List[str] | None = None,
    chapter_id: str | None = None,
    extracted_lo_statements: List[str] | None = None,
    key_terms: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "heading": heading,
        "paragraphs": [
            (
                f"Body text for {heading} explaining the concept in "
                "sufficient depth to satisfy the grounding validator "
                "non-trivial paragraph floor of thirty words each."
            ),
        ],
        "key_terms": key_terms or [heading.split()[0].lower()],
        "source_file": source_file,
        "word_count": 60,
        "chapter_id": chapter_id,
        "dart_block_ids": (
            list(dart_block_ids) if dart_block_ids is not None else ["s1"]
        ),
        "extracted_lo_statements": list(extracted_lo_statements or []),
        "extracted_misconceptions": [],
        "extracted_questions": [],
    }


JUNK_LO = "submit the coding component of the course!"


def _eight_week_topics_with_one_junk_lo() -> List[Dict[str, Any]]:
    """8 distinct heading topics; the first carries a single junk extracted
    LO. Mirrors the NVIDIA corpus collapse that motivated the fix."""
    topics = [
        _mk_topic(
            "Introduction to Agentic Workflows",
            dart_block_ids=["s1"],
            extracted_lo_statements=[JUNK_LO],
        ),
    ]
    for i in range(2, 9):
        topics.append(
            _mk_topic(
                f"Module Topic {i} Heading",
                dart_block_ids=[f"s{i}"],
            )
        )
    return topics


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    monkeypatch.delenv("TRAINFORGE_OBJECTIVE_QUALITY_GATE", raising=False)
    yield


# ---------------------------------------------------------------------- #
# _is_low_quality_objective unit tests
# ---------------------------------------------------------------------- #


def test_is_low_quality_rejects_conversational_fragment() -> None:
    assert _cgh._is_low_quality_objective(JUNK_LO) is True


def test_is_low_quality_accepts_real_objective() -> None:
    assert (
        _cgh._is_low_quality_objective(
            "Construct a LangGraph StateGraph that routes between "
            "retrieval and generation nodes"
        )
        is False
    )


def test_is_low_quality_rejects_too_short() -> None:
    assert _cgh._is_low_quality_objective("Configure it") is True
    assert _cgh._is_low_quality_objective("Go") is True


def test_is_low_quality_rejects_too_long() -> None:
    fifty_words = " ".join(["analyze"] + ["concept"] * 49)
    assert _cgh._is_low_quality_objective(fifty_words) is True


def test_is_low_quality_rejects_question_prose() -> None:
    assert _cgh._is_low_quality_objective("What is an RDF triple?") is True


# ---------------------------------------------------------------------- #
# Flag OFF: byte-stable legacy behaviour (Path A suppresses Path B)
# ---------------------------------------------------------------------- #


def test_flag_off_path_a_suppresses_path_b(monkeypatch) -> None:
    """Flag OFF reproduces the historical collapse: the single junk
    extracted LO is the ONLY objective; Path B's 8-week heading rollup
    never runs. This pins byte-stability for the calibration corpus."""
    monkeypatch.delenv("TRAINFORGE_OBJECTIVE_QUALITY_GATE", raising=False)
    topics = _eight_week_topics_with_one_junk_lo()
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=8, max_terminal=2,
    )
    all_los = terminal + chapter
    assert len(all_los) == 1, "Path A must short-circuit Path B when flag OFF"
    assert all_los[0]["statement"] == JUNK_LO


# ---------------------------------------------------------------------- #
# Flag ON: junk filtered + Path A/B merged so every week is covered
# ---------------------------------------------------------------------- #


def test_flag_on_filters_junk_and_runs_path_b(monkeypatch) -> None:
    monkeypatch.setenv("TRAINFORGE_OBJECTIVE_QUALITY_GATE", "true")
    topics = _eight_week_topics_with_one_junk_lo()
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=8, max_terminal=2,
    )
    all_los = terminal + chapter
    statements = [lo["statement"] for lo in all_los]

    # Junk fragment is gone.
    assert JUNK_LO not in statements
    assert not any("submit" in s.lower() for s in statements)

    # Path B ran: at least one heading objective per week (8 weeks).
    assert len(all_los) >= 8, (
        f"expected >= 8 heading objectives, got {len(all_los)}: {statements}"
    )
    # Each distinct module heading surfaces as an objective.
    for i in range(2, 9):
        assert any(
            f"Module Topic {i} Heading" == s for s in statements
        ), f"week {i} heading missing from objectives"


def test_flag_on_keeps_good_drops_junk_and_fills_gaps(monkeypatch) -> None:
    """Mix of good + junk extracted LOs: good survive (Path A), junk are
    dropped by the gate, Path B fills weeks with no surviving LO. The
    week that carried a good LO is NOT double-covered."""
    monkeypatch.setenv("TRAINFORGE_OBJECTIVE_QUALITY_GATE", "true")
    good_lo = (
        "Evaluate retrieval-augmented generation pipelines for "
        "factual grounding accuracy"
    )
    topics = [
        # Week 1 chapter: one good LO + one junk LO.
        _mk_topic(
            "Retrieval Augmented Generation",
            chapter_id="ch1",
            dart_block_ids=["s1"],
            extracted_lo_statements=[good_lo, JUNK_LO],
        ),
        # Weeks 2-4 chapters: no extracted LOs → Path B headings.
        _mk_topic("Vector Index Tuning", chapter_id="ch2", dart_block_ids=["s2"]),
        _mk_topic("Agent Orchestration", chapter_id="ch3", dart_block_ids=["s3"]),
        _mk_topic("Guardrail Evaluation", chapter_id="ch4", dart_block_ids=["s4"]),
    ]
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=4, max_terminal=2,
    )
    all_los = terminal + chapter
    statements = [lo["statement"] for lo in all_los]

    # Good LO kept; junk dropped.
    assert good_lo in statements
    assert JUNK_LO not in statements

    # Path B fills the three gap weeks with their headings.
    for heading in (
        "Vector Index Tuning",
        "Agent Orchestration",
        "Guardrail Evaluation",
    ):
        assert heading in statements, f"gap week {heading!r} not filled"

    # The good-LO week (Retrieval Augmented Generation heading) is not
    # double-covered by a Path B heading rollup.
    assert statements.count("Retrieval Augmented Generation") == 0
    assert statements.count(good_lo) == 1


def test_flag_on_scales_terminals_with_corpus_length(monkeypatch) -> None:
    """Corpus-scaled terminal ceiling: a 12-week corpus earns more than the
    historical 2 terminals (one per ~2 weeks, capped 8)."""
    monkeypatch.setenv("TRAINFORGE_OBJECTIVE_QUALITY_GATE", "true")
    topics = [
        _mk_topic(f"Week {i} Heading", chapter_id=f"ch{i}", dart_block_ids=[f"s{i}"])
        for i in range(1, 13)
    ]
    terminal, _chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=12, max_terminal=2,
    )
    # round(12/2) = 6 terminals, well above the legacy default of 2.
    assert len(terminal) == 6, f"expected 6 scaled terminals, got {len(terminal)}"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
