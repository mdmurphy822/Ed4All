"""Behavioral contracts for objective quality filtering and synthesis.

The default-off ``TRAINFORGE_OBJECTIVE_QUALITY_GATE`` flag preserves the
extracted-objective path. When enabled, it filters low-quality statements and
combines the surviving objectives with heading-derived coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import _content_gen_helpers as _cgh  # noqa: E402

# ---------------------------------------------------------------------- #
# Topic fixtures matching the parsed-HTML input contract
# ---------------------------------------------------------------------- #


def _mk_topic(
    heading: str,
    *,
    source_file: str = "fixture_source",
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
    """Build eight weekly topics with one low-quality extracted objective."""
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
# Flag-off compatibility contract for the extracted-objective path
# ---------------------------------------------------------------------- #


def test_flag_off_path_a_suppresses_path_b(monkeypatch) -> None:
    """Flag-off synthesis returns extracted objectives without heading fill."""
    monkeypatch.delenv("TRAINFORGE_OBJECTIVE_QUALITY_GATE", raising=False)
    topics = _eight_week_topics_with_one_junk_lo()
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=8, max_terminal=2,
    )
    all_los = terminal + chapter
    assert len(all_los) == 1, "Path A must short-circuit Path B when flag OFF"
    assert all_los[0]["statement"] == JUNK_LO


# ---------------------------------------------------------------------- #
# Flag-on quality and weekly-coverage contract
# ---------------------------------------------------------------------- #


def test_flag_on_filters_junk_and_runs_path_b(monkeypatch) -> None:
    monkeypatch.setenv("TRAINFORGE_OBJECTIVE_QUALITY_GATE", "true")
    topics = _eight_week_topics_with_one_junk_lo()
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=8, max_terminal=2,
    )
    all_los = terminal + chapter
    statements = [lo["statement"] for lo in all_los]

    # Quality filtering excludes the low-quality extracted objective.
    assert JUNK_LO not in statements
    assert not any("submit" in s.lower() for s in statements)

    # Heading synthesis supplies at least one objective per week.
    assert len(all_los) >= 8, (
        f"expected >= 8 heading objectives, got {len(all_los)}: {statements}"
    )
    # Every distinct module heading contributes objective coverage.
    for i in range(2, 9):
        assert any(
            f"Module Topic {i} Heading" == s for s in statements
        ), f"week {i} heading missing from objectives"


def test_flag_on_keeps_good_drops_junk_and_fills_gaps(monkeypatch) -> None:
    """Keep valid objectives, filter junk, and fill uncovered weeks once."""
    monkeypatch.setenv("TRAINFORGE_OBJECTIVE_QUALITY_GATE", "true")
    good_lo = (
        "Evaluate retrieval-augmented generation pipelines for "
        "factual grounding accuracy"
    )
    topics = [
        # The first week exercises mixed valid and low-quality input.
        _mk_topic(
            "Retrieval Augmented Generation",
            chapter_id="ch1",
            dart_block_ids=["s1"],
            extracted_lo_statements=[good_lo, JUNK_LO],
        ),
        # Remaining weeks exercise heading-derived coverage.
        _mk_topic("Vector Index Tuning", chapter_id="ch2", dart_block_ids=["s2"]),
        _mk_topic("Agent Orchestration", chapter_id="ch3", dart_block_ids=["s3"]),
        _mk_topic("Guardrail Evaluation", chapter_id="ch4", dart_block_ids=["s4"]),
    ]
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=4, max_terminal=2,
    )
    all_los = terminal + chapter
    statements = [lo["statement"] for lo in all_los]

    # Filtering retains the valid objective and removes the low-quality one.
    assert good_lo in statements
    assert JUNK_LO not in statements

    # Heading synthesis fills each week without a retained objective.
    for heading in (
        "Vector Index Tuning",
        "Agent Orchestration",
        "Guardrail Evaluation",
    ):
        assert heading in statements, f"gap week {heading!r} not filled"

    # A week with a retained objective does not receive duplicate heading coverage.
    assert statements.count("Retrieval Augmented Generation") == 0
    assert statements.count(good_lo) == 1


def test_flag_on_scales_terminals_with_corpus_length(monkeypatch) -> None:
    """Scale the terminal-objective ceiling with the course duration."""
    monkeypatch.setenv("TRAINFORGE_OBJECTIVE_QUALITY_GATE", "true")
    topics = [
        _mk_topic(f"Week {i} Heading", chapter_id=f"ch{i}", dart_block_ids=[f"s{i}"])
        for i in range(1, 13)
    ]
    terminal, _chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=12, max_terminal=2,
    )
    # The duration rule yields one terminal objective per two weeks.
    assert len(terminal) == 6, f"expected 6 scaled terminals, got {len(terminal)}"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
